
"""
Unified trader for multi-asset trading.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from nautilus_trader.common.component import Component
from nautilus_trader.common.logging import Logger
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.messages import SubmitOrder, CancelOrder
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import (
    ClientOrderId,
    InstrumentId,
    PositionId,
    StrategyId,
    TraderId,
    Venue,
)
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders import MarketOrder, LimitOrder, StopMarketOrder
from nautilus_trader.model.position import Position

from multi_asset_system.core.asset_interface import Asset, AssetClass
from multi_asset_system.core.asset_manager import MultiAssetManager
from multi_asset_system.risk.portfolio_risk_manager import PortfolioRiskManager
from multi_asset_system.markets.execution_router import ExecutionRouter


class UnifiedTrader(Component):
    """
    Unified trader that handles multi-asset trading operations.
    
    This class provides:
    - Unified order management across asset classes
    - Smart order routing based on asset characteristics
    - Cross-asset position management
    - Integrated risk controls
    - Asset-specific execution logic
    """
    
    def __init__(
        self,
        trader_id: TraderId,
        strategy_id: StrategyId,
        asset_manager: MultiAssetManager,
        risk_manager: PortfolioRiskManager,
        execution_router: ExecutionRouter,
        logger: Logger,
        msgbus=None,
    ):
        super().__init__(
            logger=logger,
            component_id=f"{trader_id}-UnifiedTrader",
            msgbus=msgbus,
        )
        
        self.trader_id = trader_id
        self.strategy_id = strategy_id
        self.asset_manager = asset_manager
        self.risk_manager = risk_manager
        self.execution_router = execution_router
        
        # Position tracking
        self._positions: Dict[InstrumentId, Position] = {}
        self._open_orders: Dict[ClientOrderId, Any] = {}
        self._order_to_asset: Dict[ClientOrderId, Asset] = {}
        
        # Performance tracking
        self._trades_by_asset_class: Dict[AssetClass, int] = defaultdict(int)
        self._pnl_by_asset_class: Dict[AssetClass, Decimal] = defaultdict(Decimal)
        
        # Execution preferences by asset class
        self._execution_preferences = self._default_execution_preferences()
    
    def _default_execution_preferences(self) -> Dict[AssetClass, Dict[str, Any]]:
        """Default execution preferences by asset class."""
        return {
            AssetClass.CRYPTO: {
                "preferred_order_types": [OrderType.LIMIT, OrderType.MARKET],
                "use_iceberg": True,
                "max_slippage": Decimal("0.002"),  # 0.2%
                "retry_on_reject": True,
                "smart_routing": True,
            },
            AssetClass.EQUITY: {
                "preferred_order_types": [OrderType.LIMIT, OrderType.STOP],
                "use_iceberg": False,
                "max_slippage": Decimal("0.001"),  # 0.1%
                "retry_on_reject": False,
                "respect_market_hours": True,
                "use_dark_pools": True,
            },
            AssetClass.FOREX: {
                "preferred_order_types": [OrderType.MARKET, OrderType.LIMIT],
                "use_iceberg": False,
                "max_slippage": Decimal("0.0005"),  # 5 pips
                "retry_on_reject": True,
                "aggregate_liquidity": True,
            },
            AssetClass.COMMODITY: {
                "preferred_order_types": [OrderType.LIMIT, OrderType.STOP],
                "use_iceberg": True,
                "max_slippage": Decimal("0.0015"),
                "retry_on_reject": False,
                "avoid_delivery": True,
            },
        }
    
    async def submit_order(
        self,
        instrument_id: InstrumentId,
        order_side: OrderSide,
        quantity: Quantity,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Price] = None,
        stop_price: Optional[Price] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        display_qty: Optional[Quantity] = None,
        post_only: bool = False,
        tags: Optional[List[str]] = None,
    ) -> Optional[ClientOrderId]:
        """
        Submit an order with asset-specific handling.
        
        This method:
        1. Validates the order against asset rules
        2. Performs risk checks
        3. Routes to appropriate execution venue
        4. Handles asset-specific requirements
        """
        # Get asset
        asset = self.asset_manager.get_asset(instrument_id)
        if not asset:
            self._log.error(f"Unknown instrument: {instrument_id}")
            return None
        
        # Check if asset is tradable
        if not asset.is_tradable():
            self._log.warning(f"Asset {instrument_id} is not tradable (market status: {asset._market_status})")
            return None
        
        # Validate order against asset rules
        valid, error = asset.validate_order(quantity, price, order_type.value)
        if not valid:
            self._log.error(f"Order validation failed: {error}")
            return None
        
        # Get current portfolio value for risk checks
        portfolio_value = await self.risk_manager.get_portfolio_value()
        
        # Validate against portfolio risk limits
        valid, error = self.asset_manager.validate_cross_asset_order(
            instrument_id, quantity, price, portfolio_value
        )
        if not valid:
            self._log.error(f"Portfolio risk validation failed: {error}")
            return None
        
        # Additional risk checks
        risk_check = await self.risk_manager.check_order_risk(
            asset, order_side, quantity, price
        )
        if not risk_check.approved:
            self._log.error(f"Risk check failed: {risk_check.reason}")
            return None
        
        # Apply any risk-adjusted modifications
        if risk_check.adjusted_quantity:
            quantity = risk_check.adjusted_quantity
            self._log.info(f"Adjusted quantity to {quantity} based on risk limits")
        
        # Create order based on type and asset requirements
        order = self._create_order(
            asset=asset,
            order_side=order_side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            display_qty=display_qty,
            post_only=post_only,
            tags=tags,
        )
        
        if not order:
            return None
        
        # Store order-asset mapping
        self._order_to_asset[order.client_order_id] = asset
        self._open_orders[order.client_order_id] = order
        
        # Route order through execution router
        try:
            await self.execution_router.route_order(order, asset)
            self._log.info(f"Submitted {order_type.value} order {order.client_order_id} for {instrument_id}")
            return order.client_order_id
            
        except Exception as e:
            self._log.error(f"Failed to submit order: {e}")
            # Clean up
            del self._order_to_asset[order.client_order_id]
            del self._open_orders[order.client_order_id]
            return None
    
    def _create_order(
        self,
        asset: Asset,
        order_side: OrderSide,
        quantity: Quantity,
        order_type: OrderType,
        price: Optional[Price] = None,
        stop_price: Optional[Price] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        display_qty: Optional[Quantity] = None,
        post_only: bool = False,
        tags: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Create order with asset-specific adjustments."""
        # Generate order ID
        client_order_id = ClientOrderId(str(UUID4()))
        
        # Adjust for asset-specific requirements
        prefs = self._execution_preferences.get(asset.asset_class, {})
        
        # Check if order type is preferred for this asset
        if order_type not in prefs.get("preferred_order_types", []):
            self._log.warning(f"Order type {order_type} not preferred for {asset.asset_class}")
        
        # Apply asset-specific order modifications
        if asset.asset_class == AssetClass.EQUITY:
            # Ensure whole shares for equity orders
            if not asset.trading_rules.allow_fractional:
                quantity = Quantity(int(quantity))
            
            # Check market hours for market orders
            if order_type == OrderType.MARKET:
                session = asset.get_market_session(datetime.utcnow())
                if session in ["PRE_MARKET", "POST_MARKET"]:
                    # Convert to limit order for extended hours
                    order_type = OrderType.LIMIT
                    if not price:
                        price = asset._last_price  # Use last known price
                    self._log.info(f"Converted to LIMIT order for {session} trading")
        
        elif asset.asset_class == AssetClass.FOREX:
            # Adjust quantity to lot size
            lot_size = asset.standard_lot_size
            lots = quantity / lot_size
            adjusted_lots = round(lots / asset.trading_rules.lot_size) * asset.trading_rules.lot_size
            quantity = Quantity(adjusted_lots * lot_size)
        
        elif asset.asset_class == AssetClass.COMMODITY:
            # Check for delivery avoidance
            if prefs.get("avoid_delivery") and hasattr(asset, 'is_in_delivery_period'):
                if asset.is_in_delivery_period():
                    self._log.warning(f"Asset {asset.instrument_id} is in delivery period")
                    # Could auto-roll to next contract here
        
        # Create the actual order object based on type
        if order_type == OrderType.MARKET:
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.strategy_id,
                instrument_id=asset.instrument_id,
                client_order_id=client_order_id,
                order_side=order_side,
                quantity=quantity,
                init_id=UUID4(),
                ts_init=self._clock.timestamp_ns(),
                time_in_force=time_in_force,
                reduce_only=reduce_only,
                tags=tags,
            )
        
        elif order_type == OrderType.LIMIT:
            if not price:
                self._log.error("Price required for LIMIT order")
                return None
                
            order = LimitOrder(
                trader_id=self.trader_id,
                strategy_id=self.strategy_id,
                instrument_id=asset.instrument_id,
                client_order_id=client_order_id,
                order_side=order_side,
                quantity=quantity,
                price=price,
                init_id=UUID4(),
                ts_init=self._clock.timestamp_ns(),
                time_in_force=time_in_force,
                post_only=post_only,
                reduce_only=reduce_only,
                display_qty=display_qty,
                tags=tags,
            )
        
        elif order_type == OrderType.STOP_MARKET:
            if not stop_price:
                self._log.error("Stop price required for STOP order")
                return None
                
            order = StopMarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.strategy_id,
                instrument_id=asset.instrument_id,
                client_order_id=client_order_id,
                order_side=order_side,
                quantity=quantity,
                trigger_price=stop_price,
                init_id=UUID4(),
                ts_init=self._clock.timestamp_ns(),
                time_in_force=time_in_force,
                reduce_only=reduce_only,
                tags=tags,
            )
        
        else:
            self._log.error(f"Unsupported order type: {order_type}")
            return None
        
        return order
    
    async def cancel_order(self, client_order_id: ClientOrderId) -> bool:
        """Cancel an order."""
        if client_order_id not in self._open_orders:
            self._log.warning(f"Order {client_order_id} not found")
            return False
        
        try:
            order = self._open_orders[client_order_id]
            await self.execution_router.cancel_order(order)
            
            # Clean up
            del self._open_orders[client_order_id]
            del self._order_to_asset[client_order_id]
            
            self._log.info(f"Cancelled order {client_order_id}")
            return True
            
        except Exception as e:
            self._log.error(f"Failed to cancel order: {e}")
            return False
    
    async def cancel_all_orders(
        self,
        instrument_id: Optional[InstrumentId] = None,
        asset_class: Optional[AssetClass] = None,
    ) -> int:
        """Cancel all orders, optionally filtered by instrument or asset class."""
        cancelled_count = 0
        orders_to_cancel = []
        
        for order_id, order in self._open_orders.items():
            asset = self._order_to_asset.get(order_id)
            if not asset:
                continue
            
            # Apply filters
            if instrument_id and order.instrument_id != instrument_id:
                continue
            if asset_class and asset.asset_class != asset_class:
                continue
            
            orders_to_cancel.append(order_id)
        
        # Cancel orders
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        self._log.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    def get_position(self, instrument_id: InstrumentId) -> Optional[Position]:
        """Get position for an instrument."""
        return self._positions.get(instrument_id)
    
    def get_all_positions(
        self,
        asset_class: Optional[AssetClass] = None,
        only_open: bool = True,
    ) -> List[Position]:
        """Get all positions, optionally filtered."""
        positions = []
        
        for instrument_id, position in self._positions.items():
            if only_open and position.is_closed:
                continue
            
            if asset_class:
                asset = self.asset_manager.get_asset(instrument_id)
                if not asset or asset.asset_class != asset_class:
                    continue
            
            positions.append(position)
        
        return positions
    
    async def close_position(
        self,
        instrument_id: InstrumentId,
        reduce_only: bool = True,
    ) -> Optional[ClientOrderId]:
        """Close a position."""
        position = self.get_position(instrument_id)
        if not position or position.is_closed:
            self._log.warning(f"No open position for {instrument_id}")
            return None
        
        # Determine order side to close
        order_side = OrderSide.SELL if position.is_long else OrderSide.BUY
        
        # Submit closing order
        return await self.submit_order(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=abs(position.quantity),
            order_type=OrderType.MARKET,
            reduce_only=reduce_only,
            tags=["position_close"],
        )
    
    async def close_all_positions(
        self,
        asset_class: Optional[AssetClass] = None,
        reason: Optional[str] = None,
    ) -> int:
        """Close all positions."""
        positions = self.get_all_positions(asset_class=asset_class, only_open=True)
        closed_count = 0
        
        for position in positions:
            tags = ["close_all"]
            if reason:
                tags.append(f"reason:{reason}")
            
            if await self.close_position(position.instrument_id):
                closed_count += 1
        
        self._log.info(f"Closed {closed_count} positions")
        return closed_count
    
    def update_position(self, position: Position) -> None:
        """Update position tracking."""
        self._positions[position.instrument_id] = position
        
        # Update performance tracking
        if position.is_closed:
            asset = self.asset_manager.get_asset(position.instrument_id)
            if asset:
                self._trades_by_asset_class[asset.asset_class] += 1
                self._pnl_by_asset_class[asset.asset_class] += position.realized_pnl
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary."""
        total_trades = sum(self._trades_by_asset_class.values())
        total_pnl = sum(self._pnl_by_asset_class.values())
        
        summary = {
            "total_trades": total_trades,
            "total_pnl": float(total_pnl),
            "open_positions": len([p for p in self._positions.values() if not p.is_closed]),
            "open_orders": len(self._open_orders),
            "by_asset_class": {}
        }
        
        # Performance by asset class
        for asset_class in AssetClass:
            trades = self._trades_by_asset_class.get(asset_class, 0)
            if trades > 0:
                pnl = self._pnl_by_asset_class.get(asset_class, Decimal("0"))
                summary["by_asset_class"][asset_class.value] = {
                    "trades": trades,
                    "pnl": float(pnl),
                    "avg_pnl": float(pnl / trades),
                }
        
        return summary
    
    async def rebalance_portfolio(
        self,
        target_weights: Dict[AssetClass, Decimal],
        tolerance: Decimal = Decimal("0.05"),
    ) -> int:
        """Rebalance portfolio to target asset class weights."""
        # Get current portfolio value and positions
        portfolio_value = await self.risk_manager.get_portfolio_value()
        current_weights = await self._calculate_current_weights(portfolio_value)
        
        orders_submitted = 0
        
        for asset_class, target_weight in target_weights.items():
            current_weight = current_weights.get(asset_class, Decimal("0"))
            diff = target_weight - current_weight
            
            # Skip if within tolerance
            if abs(diff) < tolerance:
                continue
            
            # Calculate rebalancing trades
            target_value = portfolio_value * target_weight
            current_value = portfolio_value * current_weight
            value_diff = target_value - current_value
            
            # Get tradable assets in this class
            assets = self.asset_manager.get_tradable_assets(asset_class=asset_class)
            if not assets:
                continue
            
            # Simple allocation - split equally among assets
            # Real implementation would be more sophisticated
            per_asset_value = value_diff / len(assets)
            
            for asset in assets[:3]:  # Limit to top 3 assets
                if asset._last_price:
                    quantity = abs(per_asset_value / asset._last_price)
                    order_side = OrderSide.BUY if value_diff > 0 else OrderSide.SELL
                    
                    order_id = await self.submit_order(
                        instrument_id=asset.instrument_id,
                        order_side=order_side,
                        quantity=quantity,
                        order_type=OrderType.LIMIT,
                        price=asset._last_price,
                        tags=["rebalance"],
                    )
                    
                    if order_id:
                        orders_submitted += 1
        
        self._log.info(f"Rebalancing submitted {orders_submitted} orders")
        return orders_submitted
    
    async def _calculate_current_weights(
        self,
        portfolio_value: Decimal,
    ) -> Dict[AssetClass, Decimal]:
        """Calculate current portfolio weights by asset class."""
        weights = defaultdict(Decimal)
        
        for instrument_id, position in self._positions.items():
            if position.is_closed:
                continue
            
            asset = self.asset_manager.get_asset(instrument_id)
            if asset and asset._last_price:
                position_value = abs(position.quantity * asset._last_price)
                weights[asset.asset_class] += position_value / portfolio_value
        
        return dict(weights)