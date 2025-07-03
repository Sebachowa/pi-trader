#!/usr/bin/env python3
"""
Basic risk management for Raspberry Pi trading bot
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional


class RiskManager:
    """Lightweight risk management system"""
    
    def __init__(self, risk_config: dict, trading_config: dict):
        self.risk_config = risk_config
        self.trading_config = trading_config
        self.daily_loss = 0.0
        self.last_trade_time = {}
        self.logger = logging.getLogger(__name__)
        
    def can_trade(self, symbol: str, positions: Dict) -> bool:
        """Check if trading is allowed based on risk rules"""
        # Check max positions
        if len(positions) >= self.trading_config['max_positions']:
            self.logger.warning("Max positions reached")
            return False
        
        # Check daily loss limit
        if self.daily_loss >= self.trading_config['max_daily_loss_pct']:
            self.logger.warning("Daily loss limit reached")
            return False
        
        # Check cooldown period
        if symbol in self.last_trade_time:
            cooldown = timedelta(minutes=self.risk_config['cooldown_minutes'])
            if datetime.now() - self.last_trade_time[symbol] < cooldown:
                return False
        
        return True
    
    def calculate_position_size(self, symbol: str, signal: Dict, balance: Dict) -> float:
        """Calculate safe position size"""
        try:
            # Get account balance
            free_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            
            # Apply position size percentage
            position_value = free_balance * self.trading_config['position_size_pct']
            
            # Apply max position size limit
            max_size = self.risk_config['max_position_size_usd']
            position_value = min(position_value, max_size)
            
            # Convert to asset amount (simplified, assumes USDT pairs)
            if 'price' in signal:
                position_size = position_value / signal['price']
            else:
                position_size = 0
            
            # Apply leverage
            position_size *= self.trading_config['leverage']
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def update_daily_loss(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_loss += pnl if pnl < 0 else 0
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_loss = 0.0
        self.last_trade_time.clear()
    
    def record_trade(self, symbol: str):
        """Record trade time for cooldown tracking"""
        self.last_trade_time[symbol] = datetime.now()
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        stop_loss_pct = self.trading_config['stop_loss_pct']
        
        if side == 'BUY':
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        take_profit_pct = self.trading_config['take_profit_pct']
        
        if side == 'BUY':
            return entry_price * (1 + take_profit_pct)
        else:
            return entry_price * (1 - take_profit_pct)
    
    def check_drawdown(self, current_equity: float, peak_equity: float) -> bool:
        """Check if drawdown exceeds limit"""
        if peak_equity <= 0:
            return False
        
        drawdown = (peak_equity - current_equity) / peak_equity
        max_drawdown = self.risk_config['max_drawdown_pct']
        
        if drawdown > max_drawdown:
            self.logger.error(f"Max drawdown exceeded: {drawdown:.2%}")
            return True
        
        return False