
"""
Equity/Stock asset implementation.
"""

from datetime import datetime, time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from multi_asset_system.core.asset_interface import (
    Asset,
    AssetClass,
    MarketHours,
    MarketStatus,
    RiskParameters,
    TradingRules,
)


class EquityAsset(Asset):
    """
    Equity/Stock asset implementation.
    
    Supports common stocks, preferred stocks, ADRs, and REITs.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        currency: Currency,
        exchange: str,  # NYSE, NASDAQ, LSE, etc.
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        market_cap: Optional[Decimal] = None,
        is_adr: bool = False,
        is_etf: bool = False,
        dividend_yield: Optional[Decimal] = None,
        trading_rules: Optional[TradingRules] = None,
        risk_parameters: Optional[RiskParameters] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.exchange = exchange
        self.sector = sector
        self.industry = industry
        self.market_cap = market_cap
        self.is_adr = is_adr
        self.is_etf = is_etf
        self.dividend_yield = dividend_yield
        
        # Determine asset class
        asset_class = AssetClass.ETF if is_etf else AssetClass.EQUITY
        
        super().__init__(
            instrument_id=instrument_id,
            asset_class=asset_class,
            base_currency=currency,
            quote_currency=currency,  # Equities trade in single currency
            market_hours=self._get_exchange_hours(exchange),
            trading_rules=trading_rules,
            risk_parameters=risk_parameters,
            metadata=metadata,
        )
        
        # Equity-specific attributes
        self.shares_outstanding = None
        self.float_shares = None
        self.short_interest = Decimal("0.0")
        self.beta = Decimal("1.0")
        self.earnings_date = None
        self.ex_dividend_date = None
        
    def _get_exchange_hours(self, exchange: str) -> MarketHours:
        """Get market hours for specific exchange."""
        # US Markets
        if exchange in ["NYSE", "NASDAQ", "AMEX"]:
            return MarketHours(
                timezone="America/New_York",
                open_time=time(9, 30),
                close_time=time(16, 0),
                pre_market_open=time(4, 0),
                post_market_close=time(20, 0),
                trading_days=[0, 1, 2, 3, 4],  # Monday to Friday
            )
        # European Markets
        elif exchange in ["LSE", "LSE-INTL"]:
            return MarketHours(
                timezone="Europe/London",
                open_time=time(8, 0),
                close_time=time(16, 30),
                pre_market_open=time(7, 0),
                post_market_close=time(17, 30),
                trading_days=[0, 1, 2, 3, 4],
            )
        elif exchange in ["XETRA", "FRA"]:
            return MarketHours(
                timezone="Europe/Berlin",
                open_time=time(9, 0),
                close_time=time(17, 30),
                pre_market_open=time(8, 0),
                post_market_close=time(20, 0),
                trading_days=[0, 1, 2, 3, 4],
            )
        # Asian Markets
        elif exchange in ["TSE", "TYO"]:
            return MarketHours(
                timezone="Asia/Tokyo",
                open_time=time(9, 0),
                close_time=time(15, 0),
                trading_days=[0, 1, 2, 3, 4],
            )
        elif exchange in ["HKEX", "HKG"]:
            return MarketHours(
                timezone="Asia/Hong_Kong",
                open_time=time(9, 30),
                close_time=time(16, 0),
                trading_days=[0, 1, 2, 3, 4],
            )
        else:
            # Default to US market hours
            return self._default_market_hours()
    
    def _default_market_hours(self) -> MarketHours:
        """Default equity market hours (US markets)."""
        return MarketHours(
            timezone="America/New_York",
            open_time=time(9, 30),
            close_time=time(16, 0),
            pre_market_open=time(4, 0),
            post_market_close=time(20, 0),
            trading_days=[0, 1, 2, 3, 4],  # Weekdays only
        )
    
    def _default_trading_rules(self) -> TradingRules:
        """Default equity trading rules."""
        # Pattern Day Trader rules for US markets
        is_us_market = self.exchange in ["NYSE", "NASDAQ", "AMEX"]
        
        return TradingRules(
            min_order_size=Decimal("1"),  # 1 share minimum
            max_order_size=Decimal("1000000"),  # 1M shares
            tick_size=Decimal("0.01"),  # Penny increments
            lot_size=Decimal("1"),  # Single shares
            max_leverage=Decimal("4.0") if is_us_market else Decimal("2.0"),  # PDT vs regular margin
            allow_short_selling=True,
            allow_fractional=self.is_etf,  # Some brokers allow fractional ETFs
            maker_fee=Decimal("0.0"),  # Maker rebates common
            taker_fee=Decimal("0.0005"),  # $0.005 per share
            settlement_period=2,  # T+2 settlement
            margin_requirement=Decimal("0.25") if is_us_market else Decimal("0.5"),
            maintenance_margin=Decimal("0.25"),
            support_market_orders=True,
            support_limit_orders=True,
            support_stop_orders=True,
            support_stop_limit_orders=True,
            support_trailing_stop=True,
            support_iceberg_orders=False,
            support_contingent_orders=True,
            support_gtc=True,
            support_gtd=True,
            support_fok=True,
            support_ioc=True,
        )
    
    def _default_risk_parameters(self) -> RiskParameters:
        """Default equity risk parameters."""
        # More conservative for equities
        return RiskParameters(
            max_position_size=Decimal("10000"),  # shares
            max_notional_value=Decimal("500000"),  # $500k per position
            position_limit=20,  # diversification
            daily_loss_limit=Decimal("0.02"),  # 2%
            max_drawdown=Decimal("0.15"),  # 15%
            concentration_limit=Decimal("0.10"),  # 10% per stock
            volatility_threshold=Decimal("0.40"),  # 40% annualized
            correlation_threshold=Decimal("0.60"),
            price_limit_up=Decimal("0.10"),  # 10% daily limit
            price_limit_down=Decimal("0.10"),
            volatility_halt_threshold=Decimal("0.05"),  # 5% in 5 minutes
        )
    
    def validate_order(
        self,
        quantity: Quantity,
        price: Optional[Price] = None,
        order_type: str = "MARKET",
    ) -> Tuple[bool, Optional[str]]:
        """Validate equity order."""
        # Check market hours
        if not self.is_tradable(datetime.utcnow()):
            if self._market_status == MarketStatus.PRE_MARKET:
                if order_type not in ["LIMIT"]:
                    return False, "Only limit orders allowed in pre-market"
            elif self._market_status == MarketStatus.POST_MARKET:
                if order_type not in ["LIMIT"]:
                    return False, "Only limit orders allowed in post-market"
            else:
                return False, f"Market is closed ({self._market_status.value})"
        
        # Standard validations
        valid, error = super().validate_order(quantity, price, order_type)
        if not valid:
            return valid, error
        
        # Check for whole shares (unless fractional allowed)
        if not self.trading_rules.allow_fractional and quantity % 1 != 0:
            return False, "Fractional shares not allowed"
        
        # Check short sale restrictions
        if order_type == "SELL" and self.short_interest > Decimal("0.20"):  # 20% short interest
            return False, "Stock on short sale restriction list"
        
        # Check for halts
        if self._market_status == MarketStatus.HALT:
            return False, "Trading halted"
        
        return True, None
    
    def calculate_fees(
        self,
        quantity: Quantity,
        price: Price,
        is_maker: bool = False,
    ) -> Decimal:
        """Calculate equity trading fees."""
        # Per-share pricing model
        fee_per_share = self.trading_rules.maker_fee if is_maker else self.trading_rules.taker_fee
        base_fee = quantity * fee_per_share
        
        # Minimum and maximum fees (common in equity markets)
        min_fee = Decimal("1.00")  # $1 minimum
        max_fee = quantity * price * Decimal("0.005")  # 0.5% maximum
        
        return max(min_fee, min(base_fee, max_fee))
    
    def calculate_margin_requirement(
        self,
        quantity: Quantity,
        price: Price,
        is_short: bool = False,
    ) -> Decimal:
        """Calculate margin requirement for equity position."""
        notional = quantity * price
        
        # Base margin requirement
        if is_short:
            # Short positions require more margin
            margin = notional * max(self.trading_rules.margin_requirement, Decimal("0.5"))
            
            # Hard-to-borrow stocks require additional margin
            if self.short_interest > Decimal("0.10"):  # 10% short interest
                margin *= Decimal("1.5")
        else:
            margin = notional * self.trading_rules.margin_requirement
        
        # Concentration rule (max 25% of buying power in one position)
        # This would be enforced at portfolio level
        
        # Add buffer for volatile stocks
        if self.beta > Decimal("1.5"):
            margin *= Decimal("1.2")  # 20% extra for high beta stocks
        
        return margin
    
    def calculate_dividend_payment(
        self,
        shares: Quantity,
        dividend_per_share: Decimal,
    ) -> Decimal:
        """Calculate dividend payment for position."""
        if not self.dividend_yield:
            return Decimal("0.0")
        
        return shares * dividend_per_share
    
    def is_earnings_blackout(self, days_before: int = 2) -> bool:
        """Check if in earnings blackout period."""
        if not self.earnings_date:
            return False
        
        days_until_earnings = (self.earnings_date - datetime.utcnow()).days
        return 0 <= days_until_earnings <= days_before
    
    def get_market_session(self, timestamp: datetime) -> str:
        """Get current market session."""
        if not self.market_hours.is_open(timestamp):
            return "CLOSED"
        
        current_time = timestamp.time()
        
        if self.market_hours.pre_market_open and current_time < self.market_hours.open_time:
            return "PRE_MARKET"
        elif self.market_hours.post_market_close and current_time > self.market_hours.close_time:
            return "POST_MARKET"
        else:
            return "REGULAR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with equity-specific fields."""
        data = super().to_dict()
        data.update({
            "exchange": self.exchange,
            "sector": self.sector,
            "industry": self.industry,
            "market_cap": float(self.market_cap) if self.market_cap else None,
            "is_adr": self.is_adr,
            "is_etf": self.is_etf,
            "dividend_yield": float(self.dividend_yield) if self.dividend_yield else None,
            "beta": float(self.beta),
            "short_interest": float(self.short_interest),
            "earnings_date": self.earnings_date.isoformat() if self.earnings_date else None,
            "ex_dividend_date": self.ex_dividend_date.isoformat() if self.ex_dividend_date else None,
        })
        return data