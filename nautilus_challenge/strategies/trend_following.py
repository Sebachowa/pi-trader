#!/usr/bin/env python3
"""
Trend Following Strategy for Nautilus Challenge
"""

from decimal import Decimal
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.indicators.ema import ExponentialMovingAverage
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from .base_strategy import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using EMA crossovers and momentum confirmation.
    
    Entry signals:
    - Fast EMA crosses above slow EMA (bullish)
    - Fast EMA crosses below slow EMA (bearish)
    - RSI confirms trend direction
    - Volume confirmation
    
    Exit signals:
    - Stop loss at 2 ATR
    - Take profit at 3 ATR
    - Trend reversal
    """
    
    def __init__(
        self,
        instrument_id,
        bar_type,
        trade_size=Decimal("0.01"),
        fast_ema_period=20,
        slow_ema_period=50,
        rsi_period=14,
    ):
        super().__init__(
            instrument_id=instrument_id,
            bar_type=bar_type,
            trade_size=trade_size,
        )
        
        # Strategy specific indicators
        self.fast_ema = ExponentialMovingAverage(fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(slow_ema_period)
        self.rsi = RelativeStrengthIndex(rsi_period)
        
        # State tracking
        self.last_fast_value = None
        self.last_slow_value = None
        self.in_position = False
        self.position_side = None
        
    def on_bar(self, bar: Bar) -> None:
        """Process new bar data."""
        super().on_bar(bar)
        
        # Update indicators
        self.fast_ema.update_raw(bar.close.as_double())
        self.slow_ema.update_raw(bar.close.as_double())
        self.rsi.update_raw(bar.close.as_double())
        
        # Wait for indicators to be ready
        if not self.indicators_ready():
            return
            
        # Get current values
        fast_value = self.fast_ema.value
        slow_value = self.slow_ema.value
        rsi_value = self.rsi.value
        atr_value = self.atr.value
        
        # Check for crossovers
        if self.last_fast_value is not None and self.last_slow_value is not None:
            
            # Bullish crossover
            if (self.last_fast_value <= self.last_slow_value and 
                fast_value > slow_value and 
                rsi_value > 50 and rsi_value < 70):
                
                if not self.in_position or self.position_side == OrderSide.SELL:
                    self.close_all_positions()
                    self.enter_long(bar, atr_value)
                    
            # Bearish crossover
            elif (self.last_fast_value >= self.last_slow_value and 
                  fast_value < slow_value and 
                  rsi_value < 50 and rsi_value > 30):
                
                if not self.in_position or self.position_side == OrderSide.BUY:
                    self.close_all_positions()
                    self.enter_short(bar, atr_value)
                    
        # Update state
        self.last_fast_value = fast_value
        self.last_slow_value = slow_value
        
    def indicators_ready(self) -> bool:
        """Check if all indicators have enough data."""
        return (self.fast_ema.initialized and 
                self.slow_ema.initialized and 
                self.rsi.initialized and 
                self.atr.initialized)
                
    def enter_long(self, bar: Bar, atr_value: float) -> None:
        """Enter a long position."""
        if not self.check_risk_limits():
            return
            
        # Calculate position size and stops
        stop_loss = bar.close.as_double() - (atr_value * self.stop_loss_atr)
        position_size = self.calculate_position_size(atr_value * self.stop_loss_atr)
        
        # Submit order
        self.submit_market_order(
            order_side=OrderSide.BUY,
            quantity=position_size,
            stop_loss=Decimal(str(stop_loss)),
            take_profit=Decimal(str(bar.close.as_double() + (atr_value * self.take_profit_atr))),
        )
        
        self.in_position = True
        self.position_side = OrderSide.BUY
        
        self.log.info(
            f"LONG signal: Fast EMA > Slow EMA, RSI={self.rsi.value:.2f}, "
            f"Entry={bar.close}, SL={stop_loss:.2f}"
        )
        
    def enter_short(self, bar: Bar, atr_value: float) -> None:
        """Enter a short position."""
        if not self.check_risk_limits():
            return
            
        # Calculate position size and stops
        stop_loss = bar.close.as_double() + (atr_value * self.stop_loss_atr)
        position_size = self.calculate_position_size(atr_value * self.stop_loss_atr)
        
        # Submit order
        self.submit_market_order(
            order_side=OrderSide.SELL,
            quantity=position_size,
            stop_loss=Decimal(str(stop_loss)),
            take_profit=Decimal(str(bar.close.as_double() - (atr_value * self.take_profit_atr))),
        )
        
        self.in_position = True
        self.position_side = OrderSide.SELL
        
        self.log.info(
            f"SHORT signal: Fast EMA < Slow EMA, RSI={self.rsi.value:.2f}, "
            f"Entry={bar.close}, SL={stop_loss:.2f}"
        )
        
    def close_all_positions(self) -> None:
        """Close all open positions."""
        for position in self.cache.positions_open(venue=self.instrument_id.venue):
            if position.instrument_id == self.instrument_id:
                if position.side == OrderSide.BUY:
                    self.submit_market_order(OrderSide.SELL, position.quantity)
                else:
                    self.submit_market_order(OrderSide.BUY, position.quantity)
                    
        self.in_position = False
        self.position_side = None
        
    def on_position_closed(self, position) -> None:
        """Handle position closed."""
        super().on_position_closed(position)
        self.in_position = False
        self.position_side = None