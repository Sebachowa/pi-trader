
"""
Paper trading account management with virtual balances and realistic accounting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Set

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import AccountId, InstrumentId, PositionId
from nautilus_trader.model.objects import Money, Price, Quantity


@dataclass
class Position:
    """Represents an open position in the paper trading account."""
    
    position_id: PositionId
    instrument_id: InstrumentId
    side: OrderSide
    quantity: Quantity
    entry_price: Price
    current_price: Optional[Price] = None
    realized_pnl: Decimal = Decimal("0")
    commission_paid: Decimal = Decimal("0")
    opened_time: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized PnL."""
        if not self.current_price:
            return Decimal("0")
        
        price_diff = float(self.current_price) - float(self.entry_price)
        if self.side == OrderSide.SELL:
            price_diff = -price_diff
        
        return Decimal(str(price_diff * float(self.quantity)))
    
    @property
    def total_pnl(self) -> Decimal:
        """Total PnL including realized and unrealized."""
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class OrderFill:
    """Represents a filled order in the paper trading system."""
    
    client_order_id: str
    venue_order_id: Optional[str]
    instrument_id: InstrumentId
    side: OrderSide
    quantity: Quantity
    price: Price
    commission: Decimal
    timestamp: datetime
    is_partial: bool = False
    realized_pnl: Optional[Decimal] = None


class PaperTradingAccount:
    """
    Paper trading account with virtual balance management.
    
    Features:
    - Multi-currency support
    - Margin and leverage handling
    - Position tracking
    - PnL calculation
    - Commission modeling
    - Risk metrics tracking
    """
    
    def __init__(
        self,
        account_id: AccountId,
        base_currency: str = "USD",
        initial_balance: Decimal = Decimal("100000"),
        leverage: int = 1,
        commission_rate: Decimal = Decimal("0.001"),  # 0.1%
    ):
        self.account_id = account_id
        self.base_currency = base_currency
        self.leverage = leverage
        self.commission_rate = commission_rate
        
        # Balances
        self.balances: Dict[str, Decimal] = {base_currency: initial_balance}
        self.initial_balance = initial_balance
        
        # Positions
        self.positions: Dict[PositionId, Position] = {}
        self.position_by_instrument: Dict[InstrumentId, List[PositionId]] = {}
        
        # Margin tracking
        self.margin_used: Decimal = Decimal("0")
        self.margin_reserved: Dict[str, Decimal] = {}  # Per order
        
        # PnL tracking
        self.realized_pnl: Decimal = Decimal("0")
        self.commission_paid: Decimal = Decimal("0")
        
        # Trade history
        self.fills: List[OrderFill] = []
        self.daily_pnl: Dict[str, Decimal] = {}  # Date -> PnL
        
        # Risk metrics
        self.max_balance = initial_balance
        self.min_balance = initial_balance
        self.max_drawdown = Decimal("0")
        self.current_drawdown = Decimal("0")
        
        # Account state
        self.created_time = datetime.utcnow()
        self.last_update = datetime.utcnow()
    
    def apply_fill(self, fill: OrderFill, instrument_id: InstrumentId) -> None:
        """
        Apply a fill to the account.
        
        Parameters
        ----------
        fill : OrderFill
            The fill to apply.
        instrument_id : InstrumentId
            The instrument being traded.
        """
        # Record fill
        self.fills.append(fill)
        
        # Deduct commission
        self._apply_commission(fill.commission)
        
        # Update or create position
        self._update_position(fill, instrument_id)
        
        # Update margin
        self._update_margin()
        
        # Update metrics
        self._update_metrics()
        
        self.last_update = datetime.utcnow()
    
    def update_position_price(self, position_id: PositionId, current_price: Price) -> None:
        """Update current price for a position."""
        if position_id in self.positions:
            self.positions[position_id].current_price = current_price
            self._update_metrics()
    
    def reserve_margin(self, order_id: str, amount: Decimal) -> bool:
        """
        Reserve margin for an order.
        
        Returns
        -------
        bool
            True if margin reserved successfully, False if insufficient.
        """
        available = self.get_available_balance()
        if amount > available:
            return False
        
        self.margin_reserved[order_id] = amount
        return True
    
    def release_margin(self, order_id: str) -> None:
        """Release reserved margin for an order."""
        self.margin_reserved.pop(order_id, None)
    
    def get_balance(self, currency: Optional[str] = None) -> Decimal:
        """Get balance for a currency."""
        if currency is None:
            currency = self.base_currency
        return self.balances.get(currency, Decimal("0"))
    
    def get_equity(self) -> Decimal:
        """Get total account equity including unrealized PnL."""
        balance = self.get_balance()
        unrealized = self.get_unrealized_pnl()
        return balance + unrealized
    
    def get_margin_used(self) -> Decimal:
        """Get total margin used by positions."""
        return self.margin_used
    
    def get_margin_reserved(self) -> Decimal:
        """Get total margin reserved for pending orders."""
        return sum(self.margin_reserved.values())
    
    def get_available_balance(self) -> Decimal:
        """Get available balance for new trades."""
        equity = self.get_equity()
        margin_total = self.get_margin_used() + self.get_margin_reserved()
        return max(Decimal("0"), equity - margin_total)
    
    def get_unrealized_pnl(self) -> Decimal:
        """Get total unrealized PnL across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> Decimal:
        """Get total realized PnL."""
        return self.realized_pnl
    
    def get_total_pnl(self) -> Decimal:
        """Get total PnL (realized + unrealized)."""
        return self.realized_pnl + self.get_unrealized_pnl()
    
    def get_max_drawdown(self) -> Decimal:
        """Get maximum drawdown experienced."""
        return self.max_drawdown
    
    def get_current_drawdown(self) -> Decimal:
        """Get current drawdown from peak."""
        return self.current_drawdown
    
    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)
    
    def get_positions_for_instrument(self, instrument_id: InstrumentId) -> List[Position]:
        """Get all positions for an instrument."""
        position_ids = self.position_by_instrument.get(instrument_id, [])
        return [self.positions[pid] for pid in position_ids if pid in self.positions]
    
    def _update_position(self, fill: OrderFill, instrument_id: InstrumentId) -> None:
        """Update or create position from fill."""
        # Check for existing positions
        existing_positions = self.get_positions_for_instrument(instrument_id)
        
        # Try to match against opposite side positions (for closing)
        remaining_qty = float(fill.quantity)
        
        for pos in existing_positions:
            if pos.side != fill.side and remaining_qty > 0:
                # Closing position
                close_qty = min(remaining_qty, float(pos.quantity))
                
                # Calculate realized PnL
                price_diff = float(fill.price) - float(pos.entry_price)
                if pos.side == OrderSide.SELL:
                    price_diff = -price_diff
                
                realized = Decimal(str(price_diff * close_qty))
                pos.realized_pnl += realized
                self.realized_pnl += realized
                
                # Record realized PnL in fill
                if fill.realized_pnl is None:
                    fill.realized_pnl = Decimal("0")
                fill.realized_pnl += realized
                
                # Update position quantity
                new_qty = float(pos.quantity) - close_qty
                if new_qty <= 0:
                    # Position fully closed
                    self._close_position(pos.position_id)
                else:
                    # Position partially closed
                    pos.quantity = Quantity(new_qty, precision=pos.quantity.precision)
                
                remaining_qty -= close_qty
        
        # If quantity remains, open new position
        if remaining_qty > 0:
            position_id = PositionId(f"{instrument_id}-{datetime.utcnow().timestamp()}")
            
            position = Position(
                position_id=position_id,
                instrument_id=instrument_id,
                side=fill.side,
                quantity=Quantity(remaining_qty, precision=fill.quantity.precision),
                entry_price=fill.price,
                commission_paid=fill.commission,
            )
            
            self.positions[position_id] = position
            
            if instrument_id not in self.position_by_instrument:
                self.position_by_instrument[instrument_id] = []
            self.position_by_instrument[instrument_id].append(position_id)
    
    def _close_position(self, position_id: PositionId) -> None:
        """Close a position."""
        if position_id not in self.positions:
            return
        
        position = self.positions.pop(position_id)
        
        # Remove from instrument mapping
        if position.instrument_id in self.position_by_instrument:
            self.position_by_instrument[position.instrument_id].remove(position_id)
            if not self.position_by_instrument[position.instrument_id]:
                del self.position_by_instrument[position.instrument_id]
    
    def _apply_commission(self, commission: Decimal) -> None:
        """Apply commission to account."""
        self.commission_paid += commission
        self.balances[self.base_currency] -= commission
    
    def _update_margin(self) -> None:
        """Update margin requirements."""
        total_margin = Decimal("0")
        
        for position in self.positions.values():
            if position.current_price:
                price = position.current_price
            else:
                price = position.entry_price
            
            notional = Decimal(str(float(position.quantity) * float(price)))
            margin = notional / Decimal(str(self.leverage))
            total_margin += margin
        
        self.margin_used = total_margin
    
    def _update_metrics(self) -> None:
        """Update account metrics."""
        current_equity = self.get_equity()
        
        # Update max/min balance
        if current_equity > self.max_balance:
            self.max_balance = current_equity
        if current_equity < self.min_balance:
            self.min_balance = current_equity
        
        # Update drawdown
        if self.max_balance > 0:
            self.current_drawdown = (self.max_balance - current_equity) / self.max_balance
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Update daily PnL
        today = datetime.utcnow().date().isoformat()
        self.daily_pnl[today] = self.get_total_pnl()
    
    def calculate_commission(self, quantity: Quantity, price: Price) -> Decimal:
        """Calculate commission for a trade."""
        notional = Decimal(str(float(quantity) * float(price)))
        return notional * self.commission_rate
    
    def to_dict(self) -> Dict:
        """Convert account state to dictionary."""
        return {
            "account_id": str(self.account_id),
            "base_currency": self.base_currency,
            "balances": {k: str(v) for k, v in self.balances.items()},
            "initial_balance": str(self.initial_balance),
            "leverage": self.leverage,
            "commission_rate": str(self.commission_rate),
            "realized_pnl": str(self.realized_pnl),
            "commission_paid": str(self.commission_paid),
            "margin_used": str(self.margin_used),
            "max_drawdown": str(self.max_drawdown),
            "current_drawdown": str(self.current_drawdown),
            "position_count": len(self.positions),
            "fills_count": len(self.fills),
            "created_time": self.created_time.isoformat(),
            "last_update": self.last_update.isoformat(),
        }
    
    def from_dict(self, data: Dict) -> None:
        """Restore account state from dictionary."""
        self.balances = {k: Decimal(v) for k, v in data.get("balances", {}).items()}
        self.initial_balance = Decimal(data.get("initial_balance", "100000"))
        self.realized_pnl = Decimal(data.get("realized_pnl", "0"))
        self.commission_paid = Decimal(data.get("commission_paid", "0"))
        self.margin_used = Decimal(data.get("margin_used", "0"))
        self.max_drawdown = Decimal(data.get("max_drawdown", "0"))
        self.current_drawdown = Decimal(data.get("current_drawdown", "0"))