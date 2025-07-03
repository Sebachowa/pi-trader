
"""
Smart execution router for multi-asset trading.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from nautilus_trader.common.component import Component
from nautilus_trader.common.logging import Logger
from nautilus_trader.execution.client import ExecutionClient
from nautilus_trader.model.identifiers import ClientOrderId, Venue
from nautilus_trader.model.orders import Order

from multi_asset_system.core.asset_interface import Asset, AssetClass, MarketStatus


class RoutingStrategy(Enum):
    """Order routing strategies."""
    BEST_EXECUTION = "BEST_EXECUTION"
    LOWEST_COST = "LOWEST_COST"
    FASTEST_FILL = "FASTEST_FILL"
    SMART_ROUTE = "SMART_ROUTE"
    DIRECT = "DIRECT"
    DARK_POOL_SWEEP = "DARK_POOL_SWEEP"
    ICEBERG = "ICEBERG"


class VenueMetrics:
    """Metrics for venue performance tracking."""
    
    def __init__(self, venue: Venue):
        self.venue = venue
        self.fill_rate = Decimal("1.0")
        self.avg_fill_time_ms = 100.0
        self.rejection_rate = Decimal("0.0")
        self.avg_slippage = Decimal("0.0")
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.total_volume = Decimal("0")
        self.last_update = datetime.utcnow()
        
        # Rolling metrics
        self._fill_times = []
        self._slippages = []
        self._max_samples = 100
    
    def update_fill(self, fill_time_ms: float, slippage: Decimal) -> None:
        """Update metrics with a fill."""
        self.total_orders += 1
        self.filled_orders += 1
        
        self._fill_times.append(fill_time_ms)
        self._slippages.append(slippage)
        
        # Keep rolling window
        if len(self._fill_times) > self._max_samples:
            self._fill_times.pop(0)
        if len(self._slippages) > self._max_samples:
            self._slippages.pop(0)
        
        # Update averages
        self.avg_fill_time_ms = sum(self._fill_times) / len(self._fill_times)
        self.avg_slippage = sum(self._slippages) / len(self._slippages)
        self.fill_rate = Decimal(self.filled_orders) / Decimal(self.total_orders)
        
        self.last_update = datetime.utcnow()
    
    def update_rejection(self) -> None:
        """Update metrics with a rejection."""
        self.total_orders += 1
        self.rejected_orders += 1
        self.rejection_rate = Decimal(self.rejected_orders) / Decimal(self.total_orders)
        self.last_update = datetime.utcnow()


class ExecutionRouter(Component):
    """
    Smart order router for multi-asset execution.
    
    Features:
    - Intelligent venue selection
    - Asset-specific routing logic
    - Execution quality monitoring
    - Smart order slicing
    - Dark pool integration
    - Latency optimization
    """
    
    def __init__(
        self,
        logger: Logger,
        default_strategy: RoutingStrategy = RoutingStrategy.SMART_ROUTE,
        enable_dark_pools: bool = True,
        enable_smart_slicing: bool = True,
        msgbus=None,
    ):
        super().__init__(
            logger=logger,
            component_id="ExecutionRouter",
            msgbus=msgbus,
        )
        
        self.default_strategy = default_strategy
        self.enable_dark_pools = enable_dark_pools
        self.enable_smart_slicing = enable_smart_slicing
        
        # Venue tracking
        self._execution_clients: Dict[Venue, ExecutionClient] = {}
        self._venue_metrics: Dict[Venue, VenueMetrics] = {}
        self._venue_capabilities: Dict[Venue, Set[str]] = defaultdict(set)
        
        # Asset class routing preferences
        self._routing_preferences = self._default_routing_preferences()
        
        # Order tracking
        self._active_orders: Dict[ClientOrderId, Order] = {}
        self._order_venues: Dict[ClientOrderId, Venue] = {}
        self._parent_child_orders: Dict[ClientOrderId, List[ClientOrderId]] = defaultdict(list)
        
        # Performance tracking
        self._routing_decisions: List[Dict[str, Any]] = []
        self._max_routing_history = 1000
    
    def _default_routing_preferences(self) -> Dict[AssetClass, Dict[str, Any]]:
        """Default routing preferences by asset class."""
        return {
            AssetClass.CRYPTO: {
                "preferred_venues": ["BINANCE", "COINBASE", "KRAKEN"],
                "use_aggregation": True,
                "min_slice_size": Decimal("0.1"),  # BTC
                "max_slices": 5,
                "allow_market_orders": True,
                "prefer_maker": True,
            },
            AssetClass.EQUITY: {
                "preferred_venues": ["NYSE", "NASDAQ", "ARCA", "BATS"],
                "use_dark_pools": True,
                "min_slice_size": Decimal("100"),  # shares
                "max_slices": 10,
                "respect_market_hours": True,
                "use_midpoint": True,
            },
            AssetClass.FOREX: {
                "preferred_venues": ["LMAX", "CURRENEX", "FXALL"],
                "use_aggregation": True,
                "min_slice_size": Decimal("100000"),  # 1 standard lot
                "max_slices": 3,
                "prefer_ecn": True,
            },
            AssetClass.COMMODITY: {
                "preferred_venues": ["CME", "ICE", "NYMEX"],
                "use_aggregation": False,
                "min_slice_size": Decimal("1"),  # 1 contract
                "max_slices": 5,
                "avoid_close_to_expiry": True,
            },
        }
    
    def register_execution_client(
        self,
        venue: Venue,
        client: ExecutionClient,
        capabilities: Optional[Set[str]] = None,
    ) -> None:
        """Register an execution client for a venue."""
        self._execution_clients[venue] = client
        self._venue_metrics[venue] = VenueMetrics(venue)
        
        if capabilities:
            self._venue_capabilities[venue] = capabilities
        
        self._log.info(f"Registered execution client for {venue}")
    
    async def route_order(
        self,
        order: Order,
        asset: Asset,
        strategy: Optional[RoutingStrategy] = None,
    ) -> None:
        """
        Route order using intelligent routing logic.
        
        This method:
        1. Analyzes order characteristics
        2. Selects optimal venue(s)
        3. Potentially splits large orders
        4. Routes to execution
        """
        strategy = strategy or self.default_strategy
        
        # Check if direct routing requested
        if strategy == RoutingStrategy.DIRECT:
            await self._route_direct(order, asset)
            return
        
        # Get routing preferences for asset class
        prefs = self._routing_preferences.get(asset.asset_class, {})
        
        # Analyze order for routing decision
        routing_decision = await self._analyze_order(order, asset, strategy)
        
        # Record routing decision
        self._record_routing_decision(order, routing_decision)
        
        # Check if order should be split
        if self.enable_smart_slicing and self._should_split_order(order, asset, routing_decision):
            await self._route_split_order(order, asset, routing_decision)
        else:
            # Route as single order
            venue = routing_decision["selected_venue"]
            await self._execute_single_order(order, venue)
    
    async def _analyze_order(
        self,
        order: Order,
        asset: Asset,
        strategy: RoutingStrategy,
    ) -> Dict[str, Any]:
        """Analyze order and determine routing."""
        analysis = {
            "order_id": order.client_order_id,
            "asset_class": asset.asset_class,
            "order_size": order.quantity,
            "strategy": strategy,
            "timestamp": datetime.utcnow(),
        }
        
        # Get available venues for this asset
        available_venues = self._get_available_venues(asset)
        analysis["available_venues"] = available_venues
        
        if not available_venues:
            raise ValueError(f"No venues available for {asset.instrument_id}")
        
        # Score venues based on strategy
        venue_scores = {}
        
        for venue in available_venues:
            score = await self._score_venue(venue, order, asset, strategy)
            venue_scores[venue] = score
        
        # Select best venue
        best_venue = max(venue_scores, key=venue_scores.get)
        analysis["selected_venue"] = best_venue
        analysis["venue_scores"] = venue_scores
        
        # Determine if order should be split
        if self.enable_smart_slicing:
            split_analysis = self._analyze_split_potential(order, asset, best_venue)
            analysis.update(split_analysis)
        
        return analysis
    
    def _get_available_venues(self, asset: Asset) -> List[Venue]:
        """Get available venues for an asset."""
        # Get venues that have the instrument
        available = []
        
        # Check registered venues
        for venue, client in self._execution_clients.items():
            # In real implementation, would check if venue supports the instrument
            if venue == asset.venue:
                available.append(venue)
        
        # Add alternative venues based on asset class
        prefs = self._routing_preferences.get(asset.asset_class, {})
        preferred_venues = prefs.get("preferred_venues", [])
        
        for venue_name in preferred_venues:
            venue = Venue(venue_name)
            if venue in self._execution_clients and venue not in available:
                available.append(venue)
        
        return available
    
    async def _score_venue(
        self,
        venue: Venue,
        order: Order,
        asset: Asset,
        strategy: RoutingStrategy,
    ) -> float:
        """Score a venue based on routing strategy."""
        metrics = self._venue_metrics.get(venue)
        if not metrics:
            return 0.0
        
        score = 100.0  # Base score
        
        if strategy == RoutingStrategy.BEST_EXECUTION:
            # Balance fill rate, cost, and slippage
            score *= float(metrics.fill_rate)
            score *= (1 - float(metrics.avg_slippage))
            score *= (1 - float(metrics.rejection_rate))
            
        elif strategy == RoutingStrategy.LOWEST_COST:
            # Prioritize low fees and tight spreads
            # Would need fee schedule integration
            score *= (1 - float(metrics.avg_slippage))
            
        elif strategy == RoutingStrategy.FASTEST_FILL:
            # Prioritize speed
            if metrics.avg_fill_time_ms > 0:
                score *= (100 / metrics.avg_fill_time_ms)
            score *= float(metrics.fill_rate)
            
        elif strategy == RoutingStrategy.SMART_ROUTE:
            # Balanced approach with asset-specific logic
            score *= float(metrics.fill_rate)
            score *= (1 - float(metrics.avg_slippage))
            
            # Asset-specific adjustments
            if asset.asset_class == AssetClass.EQUITY:
                # Prefer primary exchange for equities
                if venue == asset.venue:
                    score *= 1.2
            elif asset.asset_class == AssetClass.CRYPTO:
                # Prefer high liquidity venues
                if metrics.total_volume > Decimal("1000000"):
                    score *= 1.3
        
        # Penalize stale metrics
        staleness = (datetime.utcnow() - metrics.last_update).seconds
        if staleness > 3600:  # 1 hour
            score *= 0.8
        
        return score
    
    def _should_split_order(
        self,
        order: Order,
        asset: Asset,
        routing_decision: Dict[str, Any],
    ) -> bool:
        """Determine if order should be split."""
        prefs = self._routing_preferences.get(asset.asset_class, {})
        min_slice = prefs.get("min_slice_size", Decimal("0"))
        max_slices = prefs.get("max_slices", 1)
        
        # Don't split if disabled or too small
        if max_slices <= 1 or order.quantity < min_slice * 2:
            return False
        
        # Check if order is large relative to typical size
        # This is simplified - would use market data in reality
        if asset.asset_class == AssetClass.EQUITY:
            return order.quantity > Decimal("10000")  # shares
        elif asset.asset_class == AssetClass.CRYPTO:
            return order.quantity > Decimal("10")  # BTC
        elif asset.asset_class == AssetClass.FOREX:
            return order.quantity > Decimal("10")  # lots
        
        return False
    
    def _analyze_split_potential(
        self,
        order: Order,
        asset: Asset,
        venue: Venue,
    ) -> Dict[str, Any]:
        """Analyze how to potentially split the order."""
        prefs = self._routing_preferences.get(asset.asset_class, {})
        
        min_slice = prefs.get("min_slice_size", Decimal("1"))
        max_slices = prefs.get("max_slices", 5)
        
        # Calculate optimal slice size
        # This is simplified - would use market depth in reality
        total_qty = order.quantity
        
        # Start with even distribution
        slice_count = min(max_slices, int(total_qty / min_slice))
        if slice_count > 1:
            slice_size = total_qty / slice_count
            
            # Round to tradeable size
            slice_size = self._round_to_lot_size(slice_size, asset)
            
            return {
                "should_split": True,
                "slice_count": slice_count,
                "slice_size": slice_size,
                "split_strategy": "EVEN_DISTRIBUTION",
            }
        
        return {"should_split": False}
    
    def _round_to_lot_size(self, quantity: Decimal, asset: Asset) -> Decimal:
        """Round quantity to valid lot size."""
        lot_size = asset.trading_rules.lot_size
        return (quantity // lot_size) * lot_size
    
    async def _route_split_order(
        self,
        parent_order: Order,
        asset: Asset,
        routing_decision: Dict[str, Any],
    ) -> None:
        """Route order as multiple slices."""
        slice_count = routing_decision["slice_count"]
        slice_size = routing_decision["slice_size"]
        
        self._log.info(
            f"Splitting order {parent_order.client_order_id} into {slice_count} slices of {slice_size}"
        )
        
        remaining_qty = parent_order.quantity
        child_orders = []
        
        for i in range(slice_count):
            # Calculate slice quantity
            if i == slice_count - 1:
                # Last slice gets remaining
                qty = remaining_qty
            else:
                qty = min(slice_size, remaining_qty)
            
            if qty <= 0:
                break
            
            # Create child order
            child_order = self._create_child_order(parent_order, qty, i)
            child_orders.append(child_order)
            
            # Track parent-child relationship
            self._parent_child_orders[parent_order.client_order_id].append(
                child_order.client_order_id
            )
            
            remaining_qty -= qty
        
        # Route child orders with delays to avoid market impact
        for i, child_order in enumerate(child_orders):
            if i > 0:
                # Add delay between slices
                await asyncio.sleep(1.0)  # Simplified - would be more sophisticated
            
            venue = routing_decision["selected_venue"]
            await self._execute_single_order(child_order, venue)
    
    def _create_child_order(self, parent_order: Order, quantity: Decimal, index: int) -> Order:
        """Create a child order from parent."""
        # This is simplified - would properly clone order with new quantity
        # In real implementation, would use proper order factory
        child_order = type(parent_order)(
            trader_id=parent_order.trader_id,
            strategy_id=parent_order.strategy_id,
            instrument_id=parent_order.instrument_id,
            client_order_id=ClientOrderId(f"{parent_order.client_order_id}_SLICE_{index}"),
            order_side=parent_order.side,
            quantity=quantity,
            init_id=parent_order.init_id,
            ts_init=self._clock.timestamp_ns(),
            time_in_force=parent_order.time_in_force,
            reduce_only=parent_order.reduce_only,
            tags=[f"parent:{parent_order.client_order_id}", f"slice:{index}"],
        )
        
        # Copy price for limit orders
        if hasattr(parent_order, 'price'):
            child_order.price = parent_order.price
        
        return child_order
    
    async def _route_direct(self, order: Order, asset: Asset) -> None:
        """Route directly to primary venue."""
        venue = asset.venue
        if venue not in self._execution_clients:
            raise ValueError(f"No execution client for venue {venue}")
        
        await self._execute_single_order(order, venue)
    
    async def _execute_single_order(self, order: Order, venue: Venue) -> None:
        """Execute a single order on a venue."""
        client = self._execution_clients.get(venue)
        if not client:
            raise ValueError(f"No execution client for venue {venue}")
        
        # Track order
        self._active_orders[order.client_order_id] = order
        self._order_venues[order.client_order_id] = venue
        
        # Record order submission time
        submit_time = datetime.utcnow()
        
        try:
            # Submit through execution client
            await client.submit_order(order)
            
            self._log.info(f"Routed order {order.client_order_id} to {venue}")
            
        except Exception as e:
            self._log.error(f"Failed to route order to {venue}: {e}")
            
            # Update venue metrics
            metrics = self._venue_metrics.get(venue)
            if metrics:
                metrics.update_rejection()
            
            # Clean up tracking
            del self._active_orders[order.client_order_id]
            del self._order_venues[order.client_order_id]
            
            raise
    
    async def cancel_order(self, order: Order) -> None:
        """Cancel an order."""
        # Check if it's a parent order with slices
        child_orders = self._parent_child_orders.get(order.client_order_id, [])
        
        if child_orders:
            # Cancel all child orders
            for child_id in child_orders:
                if child_id in self._active_orders:
                    child_order = self._active_orders[child_id]
                    venue = self._order_venues.get(child_id)
                    if venue:
                        client = self._execution_clients.get(venue)
                        if client:
                            await client.cancel_order(child_order)
        else:
            # Cancel single order
            venue = self._order_venues.get(order.client_order_id)
            if venue:
                client = self._execution_clients.get(venue)
                if client:
                    await client.cancel_order(order)
        
        # Clean up tracking
        self._cleanup_order_tracking(order.client_order_id)
    
    def _cleanup_order_tracking(self, order_id: ClientOrderId) -> None:
        """Clean up order tracking data."""
        # Remove from active orders
        self._active_orders.pop(order_id, None)
        self._order_venues.pop(order_id, None)
        
        # Clean up parent-child relationships
        self._parent_child_orders.pop(order_id, None)
        
        # Remove as child from any parent
        for parent_id, children in list(self._parent_child_orders.items()):
            if order_id in children:
                children.remove(order_id)
                if not children:
                    del self._parent_child_orders[parent_id]
    
    def _record_routing_decision(self, order: Order, decision: Dict[str, Any]) -> None:
        """Record routing decision for analysis."""
        self._routing_decisions.append(decision)
        
        # Keep bounded history
        if len(self._routing_decisions) > self._max_routing_history:
            self._routing_decisions.pop(0)
    
    def update_fill_metrics(
        self,
        order_id: ClientOrderId,
        fill_time_ms: float,
        slippage: Decimal,
    ) -> None:
        """Update venue metrics based on fill."""
        venue = self._order_venues.get(order_id)
        if venue:
            metrics = self._venue_metrics.get(venue)
            if metrics:
                metrics.update_fill(fill_time_ms, slippage)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = {
            "total_orders_routed": len(self._routing_decisions),
            "active_orders": len(self._active_orders),
            "venues": {}
        }
        
        # Venue statistics
        for venue, metrics in self._venue_metrics.items():
            stats["venues"][str(venue)] = {
                "fill_rate": float(metrics.fill_rate),
                "avg_fill_time_ms": metrics.avg_fill_time_ms,
                "rejection_rate": float(metrics.rejection_rate),
                "avg_slippage_bps": float(metrics.avg_slippage * 10000),  # basis points
                "total_orders": metrics.total_orders,
            }
        
        # Routing strategy distribution
        strategy_counts = defaultdict(int)
        for decision in self._routing_decisions:
            strategy = decision.get("strategy")
            if strategy:
                strategy_counts[strategy.value] += 1
        
        stats["routing_strategies"] = dict(strategy_counts)
        
        return stats