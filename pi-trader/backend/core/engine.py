"""
Trading Engine - Core component for Raspberry Pi Trader
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Trading position representation"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    @property
    def pnl(self) -> float:
        """Calculate current P&L"""
        if self.side == 'buy':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L percentage"""
        if self.entry_price == 0:
            return 0
        return (self.pnl / (self.entry_price * self.quantity)) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['pnl'] = self.pnl
        data['pnl_percentage'] = self.pnl_percentage
        data['opened_at'] = self.opened_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class TradingMetrics:
    """Real-time trading metrics"""
    total_balance: float
    available_balance: float
    equity: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    win_rate: float
    total_trades: int
    open_positions: int
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['updated_at'] = self.updated_at.isoformat()
        return data


class TradingEngine:
    """Main trading engine for Raspberry Pi"""
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_positions: int = 10,
        max_risk_per_trade: float = 0.02,
        max_daily_loss: float = 0.05,
    ):
        # Configuration
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        
        # State
        self.running = False
        self.positions: Dict[str, Position] = {}
        self.balance = initial_balance
        self.daily_start_balance = initial_balance
        self.last_reset = datetime.now().date()
        
        # Performance tracking
        self.trade_history = deque(maxlen=1000)
        self.equity_curve = deque(maxlen=10000)
        self.daily_returns = deque(maxlen=252)
        
        # Callbacks for real-time updates
        self._update_callbacks: List[Callable] = []
        self._position_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        # Risk management
        self.emergency_stop = False
        self.risk_limits = {
            'max_positions': max_positions,
            'max_risk_per_trade': max_risk_per_trade,
            'max_daily_loss': max_daily_loss,
            'max_drawdown': 0.20,  # 20% max drawdown
            'max_correlation': 0.7,
        }
        
        logger.info(f"Trading Engine initialized with balance: ${initial_balance}")
    
    async def start(self):
        """Start the trading engine"""
        if self.running:
            logger.warning("Trading engine already running")
            return
        
        self.running = True
        logger.info("Trading engine started")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._calculate_metrics())
        asyncio.create_task(self._check_risk_limits())
    
    async def stop(self):
        """Stop the trading engine"""
        self.running = False
        logger.info("Trading engine stopped")
    
    async def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Position]:
        """Open a new position"""
        # Check risk limits
        if not await self._can_open_position(symbol, side, quantity, price):
            return None
        
        # Create position
        position_id = f"{symbol}_{datetime.now().timestamp()}"
        position = Position(
            id=position_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        # Update state
        self.positions[position_id] = position
        self.balance -= quantity * price  # Simple calculation, adjust for leverage
        
        # Notify callbacks
        await self._notify_position_update('opened', position)
        
        logger.info(f"Opened position: {position_id} - {side} {quantity} {symbol} @ {price}")
        return position
    
    async def close_position(
        self,
        position_id: str,
        price: Optional[float] = None,
        reason: str = "manual"
    ) -> Optional[float]:
        """Close an existing position"""
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        
        # Update price if provided
        if price:
            position.current_price = price
        
        # Calculate P&L
        pnl = position.pnl
        
        # Update balance
        self.balance += (position.quantity * position.current_price) + pnl
        
        # Record trade
        self.trade_history.append({
            'position_id': position_id,
            'symbol': position.symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'pnl': pnl,
            'pnl_percentage': position.pnl_percentage,
            'duration': datetime.now() - position.opened_at,
            'closed_at': datetime.now(),
            'reason': reason,
        })
        
        # Remove position
        del self.positions[position_id]
        
        # Notify callbacks
        await self._notify_position_update('closed', position, reason)
        
        logger.info(f"Closed position: {position_id} - P&L: ${pnl:.2f} ({position.pnl_percentage:.2f}%)")
        return pnl
    
    async def update_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> bool:
        """Update position parameters"""
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return False
        
        position = self.positions[position_id]
        
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
        if current_price is not None:
            position.current_price = current_price
        
        position.updated_at = datetime.now()
        
        # Notify callbacks
        await self._notify_position_update('updated', position)
        
        return True
    
    async def get_metrics(self) -> TradingMetrics:
        """Get current trading metrics"""
        # Calculate metrics
        open_positions = len(self.positions)
        unrealized_pnl = sum(p.pnl for p in self.positions.values())
        equity = self.balance + unrealized_pnl
        
        # Daily P&L
        if datetime.now().date() > self.last_reset:
            self.daily_returns.append((equity - self.daily_start_balance) / self.daily_start_balance)
            self.daily_start_balance = equity
            self.last_reset = datetime.now().date()
        
        daily_pnl = equity - self.daily_start_balance
        
        # Win rate
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        
        # Drawdown
        if self.equity_curve:
            peak = max(self.equity_curve)
            current_drawdown = (peak - equity) / peak if peak > 0 else 0
            max_drawdown = max((peak - e) / peak for e in self.equity_curve) if peak > 0 else 0
        else:
            current_drawdown = 0
            max_drawdown = 0
        
        # Sharpe ratio (simplified)
        if len(self.daily_returns) > 20:
            returns = np.array(list(self.daily_returns))
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Margin used (simplified)
        margin_used = sum(p.quantity * p.entry_price * 0.1 for p in self.positions.values())  # 10:1 leverage
        
        return TradingMetrics(
            total_balance=self.balance,
            available_balance=self.balance - margin_used,
            equity=equity,
            margin_used=margin_used,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=equity - self.initial_balance - unrealized_pnl,
            daily_pnl=daily_pnl,
            win_rate=win_rate,
            total_trades=len(self.trade_history),
            open_positions=open_positions,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
        )
    
    async def emergency_stop_trading(self, reason: str):
        """Emergency stop - close all positions"""
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        self.emergency_stop = True
        
        # Close all positions
        position_ids = list(self.positions.keys())
        for position_id in position_ids:
            await self.close_position(position_id, reason=f"emergency_stop: {reason}")
        
        # Notify
        await self._notify_alert('emergency_stop', reason, 'critical')
    
    # Monitoring tasks
    async def _monitor_positions(self):
        """Monitor positions for stop loss and take profit"""
        while self.running:
            try:
                for position in list(self.positions.values()):
                    # Check stop loss
                    if position.stop_loss:
                        if position.side == 'buy' and position.current_price <= position.stop_loss:
                            await self.close_position(position.id, position.current_price, "stop_loss")
                        elif position.side == 'sell' and position.current_price >= position.stop_loss:
                            await self.close_position(position.id, position.current_price, "stop_loss")
                    
                    # Check take profit
                    if position.take_profit:
                        if position.side == 'buy' and position.current_price >= position.take_profit:
                            await self.close_position(position.id, position.current_price, "take_profit")
                        elif position.side == 'sell' and position.current_price <= position.take_profit:
                            await self.close_position(position.id, position.current_price, "take_profit")
                
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _calculate_metrics(self):
        """Periodically calculate and broadcast metrics"""
        while self.running:
            try:
                metrics = await self.get_metrics()
                
                # Update equity curve
                self.equity_curve.append(metrics.equity)
                
                # Notify callbacks
                for callback in self._update_callbacks:
                    await callback(metrics)
                
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
                await asyncio.sleep(10)
    
    async def _check_risk_limits(self):
        """Check risk limits and trigger alerts"""
        while self.running:
            try:
                metrics = await self.get_metrics()
                
                # Check daily loss limit
                daily_loss_pct = -metrics.daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0
                if daily_loss_pct > self.max_daily_loss:
                    await self.emergency_stop_trading(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                
                # Check max drawdown
                if metrics.max_drawdown > self.risk_limits['max_drawdown']:
                    await self._notify_alert('max_drawdown', f"Max drawdown exceeded: {metrics.max_drawdown:.2%}", 'critical')
                
                # Check position limits
                if metrics.open_positions >= self.max_positions:
                    await self._notify_alert('max_positions', f"Max positions reached: {metrics.open_positions}", 'warning')
                
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error checking risk limits: {e}")
                await asyncio.sleep(30)
    
    async def _can_open_position(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Check if position can be opened within risk limits"""
        if self.emergency_stop:
            logger.warning("Emergency stop active - cannot open positions")
            return False
        
        # Check position count
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Max positions reached: {len(self.positions)}")
            return False
        
        # Check risk per trade
        position_value = quantity * price
        risk_amount = position_value * self.max_risk_per_trade
        if risk_amount > self.balance * self.max_risk_per_trade:
            logger.warning(f"Position risk too high: ${risk_amount:.2f}")
            return False
        
        # Check daily loss
        metrics = await self.get_metrics()
        if -metrics.daily_pnl > self.daily_start_balance * self.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return False
        
        return True
    
    # Callback management
    def add_update_callback(self, callback: Callable):
        """Add callback for metric updates"""
        self._update_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable):
        """Add callback for position updates"""
        self._position_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alerts"""
        self._alert_callbacks.append(callback)
    
    async def _notify_position_update(self, action: str, position: Position, reason: str = ""):
        """Notify position update callbacks"""
        for callback in self._position_callbacks:
            await callback(action, position, reason)
    
    async def _notify_alert(self, alert_type: str, message: str, severity: str):
        """Notify alert callbacks"""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
        }
        for callback in self._alert_callbacks:
            await callback(alert)