
"""
Order execution engine for realistic paper trading simulation.
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set

from nautilus_trader.common.clock import Clock
from nautilus_trader.common.logging import Logger
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, VenueOrderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders import Order

from paper_trading.engine.accounts import OrderFill


class FillProbability:
    """Calculate fill probability for different order types and market conditions."""
    
    @staticmethod
    def market_order() -> float:
        """Market orders always fill (with slippage)."""
        return 1.0
    
    @staticmethod
    def limit_order(
        order_price: float,
        market_price: float,
        side: OrderSide,
        volatility: float = 0.01,
        time_in_market_minutes: float = 0,
    ) -> float:
        """
        Calculate limit order fill probability.
        
        Considers:
        - Distance from market price
        - Order side vs price movement
        - Market volatility
        - Time order has been in market
        """
        # Calculate price distance as percentage
        price_distance = abs(order_price - market_price) / market_price
        
        # Base probability based on price distance
        if side == OrderSide.BUY:
            if order_price >= market_price:
                # Buy at or above market - high probability
                base_prob = 0.95
            else:
                # Buy below market - depends on distance
                base_prob = max(0.1, 1.0 - (price_distance / volatility))
        else:  # SELL
            if order_price <= market_price:
                # Sell at or below market - high probability
                base_prob = 0.95
            else:
                # Sell above market - depends on distance
                base_prob = max(0.1, 1.0 - (price_distance / volatility))
        
        # Time decay factor - longer in market, higher probability
        time_factor = min(1.0, 1.0 + (time_in_market_minutes / 60) * 0.2)
        
        # Volatility factor - higher volatility, higher fill probability
        vol_factor = 1.0 + volatility * 2
        
        return min(1.0, base_prob * time_factor * vol_factor)
    
    @staticmethod
    def stop_order(
        stop_price: float,
        market_price: float,
        side: OrderSide,
        volatility: float = 0.01,
    ) -> float:
        """Calculate stop order trigger probability."""
        # Calculate if stop should trigger
        if side == OrderSide.BUY:
            # Buy stop triggers when market >= stop price
            if market_price >= stop_price:
                return 1.0
        else:  # SELL
            # Sell stop triggers when market <= stop price
            if market_price <= stop_price:
                return 1.0
        
        # Calculate probability of hitting stop based on volatility
        price_distance = abs(stop_price - market_price) / market_price
        return min(0.5, volatility / price_distance) if price_distance > 0 else 0


class PartialFillGenerator:
    """Generate realistic partial fills for large orders."""
    
    @staticmethod
    def generate_fills(
        total_quantity: float,
        average_market_size: float,
        max_fills: int = 10,
        urgency: str = "normal",
    ) -> List[float]:
        """
        Generate partial fill sizes.
        
        Parameters
        ----------
        total_quantity : float
            Total order quantity.
        average_market_size : float
            Typical market trade size.
        max_fills : int
            Maximum number of partial fills.
        urgency : str
            Order urgency affecting fill pattern.
            
        Returns
        -------
        List[float]
            List of fill quantities.
        """
        if total_quantity <= average_market_size:
            # Small order - single fill
            return [total_quantity]
        
        # Determine fill pattern based on urgency
        if urgency == "aggressive":
            # Aggressive - larger chunks
            min_fill_pct = 0.2
            max_fill_pct = 0.5
        elif urgency == "passive":
            # Passive - smaller chunks
            min_fill_pct = 0.05
            max_fill_pct = 0.15
        else:
            # Normal
            min_fill_pct = 0.1
            max_fill_pct = 0.3
        
        fills = []
        remaining = total_quantity
        
        while remaining > 0 and len(fills) < max_fills:
            # Generate fill size
            fill_pct = random.uniform(min_fill_pct, max_fill_pct)
            fill_size = min(remaining, total_quantity * fill_pct)
            
            # Round to reasonable precision
            fill_size = round(fill_size, 2)
            
            if fill_size > 0:
                fills.append(fill_size)
                remaining -= fill_size
        
        # Add any remaining quantity to last fill
        if remaining > 0:
            if fills:
                fills[-1] += remaining
            else:
                fills.append(remaining)
        
        return fills


class OrderExecutor:
    """
    Handles realistic order execution for paper trading.
    
    Features:
    - Realistic execution delays
    - Partial fill simulation
    - Order type handling (market, limit, stop)
    - Time-based fill probability
    - Queue position modeling
    """
    
    def __init__(self, engine, clock: Clock, logger: Logger):
        self.engine = engine
        self.clock = clock
        self.logger = logger
        
        # Active orders
        self.pending_orders: Dict[ClientOrderId, Order] = {}
        self.order_queue: Dict[ClientOrderId, Dict] = {}  # Order queue position info
        
        # Execution tracking
        self.fill_generator = PartialFillGenerator()
        self.fill_probability = FillProbability()
        
        # Execution tasks
        self.execution_tasks: Dict[ClientOrderId, asyncio.Task] = {}
        self._running = False
    
    async def start(self):
        """Start the order executor."""
        self._running = True
        self.logger.info("Order Executor started")
    
    async def stop(self):
        """Stop the order executor."""
        self._running = False
        
        # Cancel all execution tasks
        for task in self.execution_tasks.values():
            task.cancel()
        
        if self.execution_tasks:
            await asyncio.gather(*self.execution_tasks.values(), return_exceptions=True)
        
        self.logger.info("Order Executor stopped")
    
    def submit_order(self, order: Order) -> None:
        """
        Submit an order for execution.
        
        Parameters
        ----------
        order : Order
            The order to execute.
        """
        self.pending_orders[order.client_order_id] = order
        
        # Initialize queue position
        self.order_queue[order.client_order_id] = {
            "position": self._calculate_queue_position(order),
            "submission_time": datetime.utcnow(),
            "fills": [],
        }
        
        # Start execution task
        task = asyncio.create_task(self._execute_order(order))
        self.execution_tasks[order.client_order_id] = task
    
    def cancel_order(self, client_order_id: ClientOrderId) -> None:
        """Cancel an order."""
        if client_order_id in self.execution_tasks:
            self.execution_tasks[client_order_id].cancel()
            self.pending_orders.pop(client_order_id, None)
            self.order_queue.pop(client_order_id, None)
            self.logger.info(f"Order {client_order_id} cancelled")
    
    async def _execute_order(self, order: Order) -> None:
        """Execute an order with realistic simulation."""
        try:
            # Apply processing delay
            delay_ms = self.engine.config.order_processing_delay_ms
            if order.order_type == OrderType.MARKET:
                delay_ms += self.engine.config.market_order_delay_ms
            else:
                delay_ms += self.engine.config.limit_order_delay_ms
            
            await asyncio.sleep(delay_ms / 1000)
            
            # Execute based on order type
            if order.order_type == OrderType.MARKET:
                await self._execute_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                await self._execute_limit_order(order)
            elif order.order_type == OrderType.STOP_MARKET:
                await self._execute_stop_order(order)
            else:
                self.logger.warning(f"Unsupported order type: {order.order_type}")
                
        except asyncio.CancelledError:
            self.logger.info(f"Order execution cancelled: {order.client_order_id}")
        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
        finally:
            self.execution_tasks.pop(order.client_order_id, None)
    
    async def _execute_market_order(self, order: Order) -> None:
        """Execute a market order."""
        # Market orders fill immediately (with slippage)
        execution_price = self.engine.get_execution_price(
            order.instrument_id,
            order.side,
            order.quantity,
            order.order_type,
        )
        
        if execution_price is None:
            self.logger.error(f"No market data for {order.instrument_id}")
            return
        
        # Check if partial fills are enabled
        if self.engine.config.enable_partial_fills and float(order.quantity) > 100:
            # Generate partial fills
            fills = self.fill_generator.generate_fills(
                float(order.quantity),
                average_market_size=100,  # Default market size
                urgency="normal",
            )
            
            for i, fill_qty in enumerate(fills):
                # Add small delay between fills
                if i > 0:
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                
                # Execute partial fill
                await self._execute_fill(
                    order,
                    Quantity(fill_qty, precision=order.quantity.precision),
                    execution_price,
                    is_partial=i < len(fills) - 1,
                )
        else:
            # Single fill
            await self._execute_fill(order, order.quantity, execution_price)
    
    async def _execute_limit_order(self, order: Order) -> None:
        """Execute a limit order."""
        queue_info = self.order_queue[order.client_order_id]
        
        while self._running and order.client_order_id in self.pending_orders:
            # Get current market price
            market_price = self._get_market_price(order.instrument_id, order.side)
            if market_price is None:
                await asyncio.sleep(1)
                continue
            
            # Calculate fill probability
            time_in_market = (datetime.utcnow() - queue_info["submission_time"]).total_seconds() / 60
            fill_prob = self.fill_probability.limit_order(
                float(order.price),
                float(market_price),
                order.side,
                volatility=0.01,  # Default volatility
                time_in_market_minutes=time_in_market,
            )
            
            # Check if order should fill
            if random.random() < fill_prob:
                # Determine execution price (might get price improvement)
                if random.random() < 0.1:  # 10% chance of price improvement
                    improvement = random.uniform(0, 0.0001)  # Up to 1 basis point
                    if order.side == OrderSide.BUY:
                        execution_price = Price(
                            float(order.price) * (1 - improvement),
                            precision=order.price.precision
                        )
                    else:
                        execution_price = Price(
                            float(order.price) * (1 + improvement),
                            precision=order.price.precision
                        )
                else:
                    execution_price = order.price
                
                # Execute fill
                await self._execute_fill(order, order.quantity, execution_price)
                break
            
            # Wait before next check
            await asyncio.sleep(1)
    
    async def _execute_stop_order(self, order: Order) -> None:
        """Execute a stop order."""
        while self._running and order.client_order_id in self.pending_orders:
            # Get current market price
            market_price = self._get_market_price(order.instrument_id, order.side)
            if market_price is None:
                await asyncio.sleep(1)
                continue
            
            # Check if stop should trigger
            should_trigger = False
            if order.side == OrderSide.BUY:
                should_trigger = float(market_price) >= float(order.stop_px)
            else:
                should_trigger = float(market_price) <= float(order.stop_px)
            
            if should_trigger:
                # Stop triggered - execute as market order
                execution_price = self.engine.get_execution_price(
                    order.instrument_id,
                    order.side,
                    order.quantity,
                    OrderType.MARKET,  # Stops execute as market orders
                )
                
                if execution_price:
                    await self._execute_fill(order, order.quantity, execution_price)
                break
            
            # Wait before next check
            await asyncio.sleep(0.5)
    
    async def _execute_fill(
        self,
        order: Order,
        quantity: Quantity,
        price: Price,
        is_partial: bool = False,
    ) -> None:
        """Execute a fill."""
        # Create venue order ID
        venue_order_id = f"PAPER-{UUID4()}"
        
        # Calculate commission
        commission = self.engine.account.calculate_commission(quantity, price)
        
        # Create fill
        fill = OrderFill(
            client_order_id=str(order.client_order_id),
            venue_order_id=venue_order_id,
            instrument_id=order.instrument_id,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            timestamp=datetime.utcnow(),
            is_partial=is_partial,
        )
        
        # Execute in engine
        self.engine.execute_fill(fill)
        
        # Update order tracking
        queue_info = self.order_queue.get(order.client_order_id, {})
        queue_info.setdefault("fills", []).append(fill)
        
        # Remove order if fully filled
        if not is_partial:
            self.pending_orders.pop(order.client_order_id, None)
            self.order_queue.pop(order.client_order_id, None)
    
    def _calculate_queue_position(self, order: Order) -> int:
        """Calculate queue position for limit orders."""
        # Simplified: random position based on order size
        if order.order_type != OrderType.LIMIT:
            return 0
        
        # Larger orders get worse queue position
        base_position = random.randint(1, 100)
        size_penalty = min(50, int(float(order.quantity) / 100))
        
        return base_position + size_penalty
    
    def _get_market_price(self, instrument_id, side: OrderSide) -> Optional[Price]:
        """Get current market price for order matching."""
        quote = self.engine.last_quotes.get(instrument_id)
        if quote:
            # Use opposite side for matching
            return quote.bid_price if side == OrderSide.BUY else quote.ask_price
        
        trade = self.engine.last_trades.get(instrument_id)
        if trade:
            return trade.price
        
        return None