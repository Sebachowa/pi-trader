
"""
Unified asset interface for multi-asset trading system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price, Quantity


class AssetClass(Enum):
    """Asset class enumeration."""
    CRYPTO = "CRYPTO"
    EQUITY = "EQUITY"
    FOREX = "FOREX"
    COMMODITY = "COMMODITY"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    BOND = "BOND"
    ETF = "ETF"


class MarketStatus(Enum):
    """Market status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PRE_MARKET = "PRE_MARKET"
    POST_MARKET = "POST_MARKET"
    AUCTION = "AUCTION"
    HALT = "HALT"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class MarketHours:
    """Market hours specification."""
    timezone: str
    open_time: Optional[time] = None  # None for 24/7 markets
    close_time: Optional[time] = None
    pre_market_open: Optional[time] = None
    post_market_close: Optional[time] = None
    trading_days: List[int] = None  # 0=Monday, 6=Sunday
    holidays: List[datetime] = None
    
    def __post_init__(self):
        if self.trading_days is None:
            # Default to weekdays for traditional markets
            self.trading_days = [0, 1, 2, 3, 4] if self.open_time else list(range(7))
        if self.holidays is None:
            self.holidays = []

    def is_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given timestamp."""
        if self.open_time is None:  # 24/7 market
            return True
            
        # Check if it's a trading day
        if timestamp.weekday() not in self.trading_days:
            return False
            
        # Check if it's a holiday
        if timestamp.date() in [h.date() for h in self.holidays]:
            return False
            
        # Check time
        current_time = timestamp.time()
        if self.open_time <= self.close_time:
            return self.open_time <= current_time <= self.close_time
        else:  # Overnight session
            return current_time >= self.open_time or current_time <= self.close_time


@dataclass
class TradingRules:
    """Asset-specific trading rules."""
    min_order_size: Decimal
    max_order_size: Decimal
    tick_size: Decimal
    lot_size: Decimal
    max_leverage: Decimal = Decimal("1.0")
    allow_short_selling: bool = True
    allow_fractional: bool = False
    maker_fee: Decimal = Decimal("0.0")
    taker_fee: Decimal = Decimal("0.0")
    settlement_period: int = 0  # T+0 for crypto, T+2 for stocks
    margin_requirement: Decimal = Decimal("1.0")
    maintenance_margin: Decimal = Decimal("0.8")
    
    # Order type support
    support_market_orders: bool = True
    support_limit_orders: bool = True
    support_stop_orders: bool = True
    support_stop_limit_orders: bool = True
    support_trailing_stop: bool = False
    support_iceberg_orders: bool = False
    support_contingent_orders: bool = False
    
    # Time in force support
    support_gtc: bool = True  # Good Till Canceled
    support_gtd: bool = True  # Good Till Date
    support_fok: bool = True  # Fill or Kill
    support_ioc: bool = True  # Immediate or Cancel


@dataclass
class RiskParameters:
    """Asset-specific risk parameters."""
    max_position_size: Decimal
    max_notional_value: Decimal
    position_limit: int = 1  # Number of concurrent positions
    daily_loss_limit: Decimal = Decimal("0.02")  # 2% default
    max_drawdown: Decimal = Decimal("0.10")  # 10% default
    concentration_limit: Decimal = Decimal("0.20")  # 20% of portfolio
    volatility_threshold: Decimal = Decimal("0.50")  # 50% annualized
    correlation_threshold: Decimal = Decimal("0.70")
    
    # Circuit breaker settings
    price_limit_up: Optional[Decimal] = None
    price_limit_down: Optional[Decimal] = None
    volatility_halt_threshold: Optional[Decimal] = None
    
    # Risk multipliers by market condition
    high_volatility_multiplier: Decimal = Decimal("0.5")
    low_liquidity_multiplier: Decimal = Decimal("0.7")
    news_event_multiplier: Decimal = Decimal("0.8")


class Asset(ABC):
    """
    Abstract base class for all assets.
    
    This interface provides a unified way to handle different asset classes
    while maintaining their specific characteristics and requirements.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        asset_class: AssetClass,
        base_currency: Currency,
        quote_currency: Optional[Currency] = None,
        market_hours: Optional[MarketHours] = None,
        trading_rules: Optional[TradingRules] = None,
        risk_parameters: Optional[RiskParameters] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.instrument_id = instrument_id
        self.asset_class = asset_class
        self.base_currency = base_currency
        self.quote_currency = quote_currency or Currency.from_str("USD")
        self.market_hours = market_hours or self._default_market_hours()
        self.trading_rules = trading_rules or self._default_trading_rules()
        self.risk_parameters = risk_parameters or self._default_risk_parameters()
        self.metadata = metadata or {}
        
        # Runtime state
        self._market_status = MarketStatus.CLOSED
        self._last_price = None
        self._last_update = None
        
    @abstractmethod
    def _default_market_hours(self) -> MarketHours:
        """Get default market hours for asset class."""
        pass
        
    @abstractmethod
    def _default_trading_rules(self) -> TradingRules:
        """Get default trading rules for asset class."""
        pass
        
    @abstractmethod
    def _default_risk_parameters(self) -> RiskParameters:
        """Get default risk parameters for asset class."""
        pass
    
    @abstractmethod
    def validate_order(
        self,
        quantity: Quantity,
        price: Optional[Price] = None,
        order_type: str = "MARKET",
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an order against asset-specific rules.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def calculate_fees(
        self,
        quantity: Quantity,
        price: Price,
        is_maker: bool = False,
    ) -> Decimal:
        """Calculate trading fees for the given order."""
        pass
    
    @abstractmethod
    def calculate_margin_requirement(
        self,
        quantity: Quantity,
        price: Price,
        is_short: bool = False,
    ) -> Decimal:
        """Calculate margin requirement for position."""
        pass
    
    def update_market_status(self, timestamp: datetime) -> MarketStatus:
        """Update and return current market status."""
        if self.market_hours.is_open(timestamp):
            self._market_status = MarketStatus.OPEN
        else:
            self._market_status = MarketStatus.CLOSED
        return self._market_status
    
    def update_price(self, price: Price, timestamp: datetime) -> None:
        """Update last known price."""
        self._last_price = price
        self._last_update = timestamp
    
    def is_tradable(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if asset is currently tradable."""
        if timestamp:
            self.update_market_status(timestamp)
        return self._market_status == MarketStatus.OPEN
    
    def get_position_sizing(
        self,
        account_balance: Decimal,
        risk_percent: Decimal,
        stop_loss_price: Optional[Price] = None,
    ) -> Quantity:
        """Calculate appropriate position size based on risk parameters."""
        # Kelly Criterion or fixed fractional position sizing
        max_position_value = account_balance * self.risk_parameters.concentration_limit
        
        if stop_loss_price and self._last_price:
            # Risk-based position sizing
            risk_amount = account_balance * risk_percent
            price_risk = abs(self._last_price - stop_loss_price)
            position_size = risk_amount / price_risk
        else:
            # Fixed fractional sizing
            position_size = max_position_value / self._last_price if self._last_price else Quantity.zero()
            
        # Apply constraints
        max_size = min(
            self.trading_rules.max_order_size,
            self.risk_parameters.max_position_size,
        )
        
        return min(position_size, max_size)
    
    @property
    def symbol(self) -> str:
        """Get asset symbol."""
        return self.instrument_id.symbol.value
    
    @property
    def venue(self) -> Venue:
        """Get trading venue."""
        return self.instrument_id.venue
    
    @property
    def is_crypto(self) -> bool:
        """Check if asset is cryptocurrency."""
        return self.asset_class == AssetClass.CRYPTO
    
    @property
    def is_equity(self) -> bool:
        """Check if asset is equity/stock."""
        return self.asset_class == AssetClass.EQUITY
    
    @property
    def is_forex(self) -> bool:
        """Check if asset is forex pair."""
        return self.asset_class == AssetClass.FOREX
    
    @property
    def is_derivative(self) -> bool:
        """Check if asset is derivative."""
        return self.asset_class in [AssetClass.FUTURE, AssetClass.OPTION]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert asset to dictionary representation."""
        return {
            "instrument_id": str(self.instrument_id),
            "asset_class": self.asset_class.value,
            "base_currency": self.base_currency.code,
            "quote_currency": self.quote_currency.code,
            "market_status": self._market_status.value,
            "last_price": float(self._last_price) if self._last_price else None,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "metadata": self.metadata,
        }