"""
Trading metrics and position tracking for Raspberry Pi Trader
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


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