#!/usr/bin/env python3
"""
Base Strategy for Nautilus Challenge
Professional trading strategy with risk management and monitoring
"""

from decimal import Decimal
from typing import Optional

from nautilus_trader.core.message import Event
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, PositionId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder, LimitOrder
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators.ema import ExponentialMovingAverage


class BaseStrategy(Strategy):
    """
    Base strategy with common functionality for all trading strategies.
    
    Includes:
    - Risk management
    - Position sizing
    - Performance tracking
    - Telegram notifications
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        bar_type: str,
        trade_size: Decimal,
        max_positions: int = 1,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0,
    ):
        super().__init__()
        
        # Configuration
        self.instrument_id = instrument_id
        self.bar_type = bar_type
        self.trade_size = trade_size
        self.max_positions = max_positions
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        
        # Indicators
        self.atr = AverageTrueRange(14)
        self.ema_20 = ExponentialMovingAverage(20)
        self.ema_50 = ExponentialMovingAverage(50)
        
        # State tracking
        self.position_count = 0
        self.daily_trades = 0
        self.daily_pnl = Decimal(0)
        self.total_pnl = Decimal(0)
        self.wins = 0
        self.losses = 0
        
        # Risk management
        self.max_daily_loss = Decimal("-0.02")  # 2% max daily loss
        self.max_position_risk = Decimal("0.01")  # 1% per position
        
    def on_start(self) -> None:
        """Initialize strategy when started."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return
            
        # Subscribe to data
        self.subscribe_bars(self.bar_type)
        self.subscribe_quote_ticks(self.instrument_id)
        
        self.log.info(f"Strategy started for {self.instrument_id}")
        
    def on_bar(self, bar: Bar) -> None:
        """Handle bar data - to be implemented by child strategies."""
        # Update indicators
        self.atr.update_raw(
            bar.high.as_double(),
            bar.low.as_double(),
            bar.close.as_double(),
        )
        self.ema_20.update_raw(bar.close.as_double())
        self.ema_50.update_raw(bar.close.as_double())
        
    def check_risk_limits(self) -> bool:
        """Check if we're within risk limits."""
        # Check daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            self.log.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return False
            
        # Check position limit
        if self.position_count >= self.max_positions:
            return False
            
        # Check daily trade limit
        if self.daily_trades >= 10:
            self.log.warning("Daily trade limit reached")
            return False
            
        return True
        
    def calculate_position_size(self, stop_distance: float) -> Decimal:
        """Calculate position size based on Kelly Criterion and risk management."""
        # Get account balance
        account_balance = self.portfolio.net_liquidation(self.instrument.quote_currency)
        
        # Calculate position size based on risk
        risk_amount = account_balance * self.max_position_risk
        position_size = risk_amount / Decimal(str(stop_distance))
        
        # Apply Kelly Criterion if we have enough history
        if self.wins + self.losses > 20:
            win_rate = self.wins / (self.wins + self.losses)
            avg_win = self.total_pnl / max(self.wins, 1)
            avg_loss = abs(self.total_pnl / max(self.losses, 1))
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                position_size *= Decimal(str(kelly_fraction))
                
        # Ensure minimum size
        return max(position_size, self.trade_size)
        
    def submit_market_order(
        self,
        order_side: OrderSide,
        quantity: Decimal,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> None:
        """Submit a market order with risk management."""
        if not self.check_risk_limits():
            return
            
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=quantity,
            time_in_force=TimeInForce.GTC,
        )
        
        self.submit_order(order)
        self.daily_trades += 1
        
        # Add stop loss and take profit if specified
        if stop_loss:
            self.submit_stop_loss(order, stop_loss)
        if take_profit:
            self.submit_take_profit(order, take_profit)
            
    def submit_stop_loss(self, parent_order, stop_price: Decimal) -> None:
        """Submit a stop loss order."""
        if parent_order.side == OrderSide.BUY:
            sl_side = OrderSide.SELL
        else:
            sl_side = OrderSide.BUY
            
        sl_order = self.order_factory.stop_market(
            instrument_id=self.instrument_id,
            order_side=sl_side,
            quantity=parent_order.quantity,
            trigger_price=stop_price,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        
        self.submit_order(sl_order)
        
    def submit_take_profit(self, parent_order, target_price: Decimal) -> None:
        """Submit a take profit order."""
        if parent_order.side == OrderSide.BUY:
            tp_side = OrderSide.SELL
        else:
            tp_side = OrderSide.BUY
            
        tp_order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=tp_side,
            quantity=parent_order.quantity,
            price=target_price,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        
        self.submit_order(tp_order)
        
    def on_position_opened(self, position: Position) -> None:
        """Handle position opened event."""
        self.position_count += 1
        self.log.info(f"Position opened: {position}")
        
    def on_position_closed(self, position: Position) -> None:
        """Handle position closed event."""
        self.position_count -= 1
        
        # Update P&L tracking
        pnl = position.realized_pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
            
        self.log.info(f"Position closed: {position}, PnL: {pnl}")
        
    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_trades = 0
        self.daily_pnl = Decimal(0)