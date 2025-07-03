
"""
Foreign Exchange (Forex) asset implementation.
"""

from datetime import datetime, time
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from multi_asset_system.core.asset_interface import (
    Asset,
    AssetClass,
    MarketHours,
    RiskParameters,
    TradingRules,
)


class ForexAsset(Asset):
    """
    Foreign Exchange (Forex) asset implementation.
    
    Supports major, minor, and exotic currency pairs.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        base_currency: Currency,
        quote_currency: Currency,
        pair_type: str = "MAJOR",  # MAJOR, MINOR, EXOTIC
        pip_size: Decimal = Decimal("0.0001"),
        lot_size: Decimal = Decimal("100000"),  # Standard lot
        session: str = "GLOBAL",  # GLOBAL, LONDON, NY, TOKYO, SYDNEY
        trading_rules: Optional[TradingRules] = None,
        risk_parameters: Optional[RiskParameters] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.pair_type = pair_type
        self.pip_size = pip_size
        self.standard_lot_size = lot_size
        self.session = session
        
        # Determine if JPY pair (different pip calculation)
        self.is_jpy_pair = base_currency == Currency.JPY or quote_currency == Currency.JPY
        if self.is_jpy_pair:
            self.pip_size = Decimal("0.01")
        
        super().__init__(
            instrument_id=instrument_id,
            asset_class=AssetClass.FOREX,
            base_currency=base_currency,
            quote_currency=quote_currency,
            market_hours=self._get_session_hours(session),
            trading_rules=trading_rules,
            risk_parameters=risk_parameters,
            metadata=metadata,
        )
        
        # Forex-specific attributes
        self.swap_long = Decimal("0.0")  # Overnight swap rates
        self.swap_short = Decimal("0.0")
        self.interest_rate_diff = Decimal("0.0")
        self.liquidity_score = self._calculate_liquidity_score()
        
    def _get_session_hours(self, session: str) -> MarketHours:
        """Get market hours for specific FX session."""
        if session == "GLOBAL":
            # Forex is 24/5 (closed weekends)
            return MarketHours(
                timezone="UTC",
                open_time=time(22, 0),  # Sunday 10 PM UTC
                close_time=time(22, 0),  # Friday 10 PM UTC
                trading_days=[0, 1, 2, 3, 4],  # Monday to Friday
            )
        elif session == "LONDON":
            return MarketHours(
                timezone="Europe/London",
                open_time=time(8, 0),
                close_time=time(17, 0),
                trading_days=[0, 1, 2, 3, 4],
            )
        elif session == "NY":
            return MarketHours(
                timezone="America/New_York",
                open_time=time(8, 0),
                close_time=time(17, 0),
                trading_days=[0, 1, 2, 3, 4],
            )
        elif session == "TOKYO":
            return MarketHours(
                timezone="Asia/Tokyo",
                open_time=time(9, 0),
                close_time=time(18, 0),
                trading_days=[0, 1, 2, 3, 4],
            )
        elif session == "SYDNEY":
            return MarketHours(
                timezone="Australia/Sydney",
                open_time=time(7, 0),
                close_time=time(16, 0),
                trading_days=[0, 1, 2, 3, 4],
            )
        else:
            return self._default_market_hours()
    
    def _default_market_hours(self) -> MarketHours:
        """Default forex market hours (24/5)."""
        return MarketHours(
            timezone="UTC",
            open_time=None,  # 24 hour market during weekdays
            close_time=None,
            trading_days=[0, 1, 2, 3, 4],  # Closed weekends
        )
    
    def _default_trading_rules(self) -> TradingRules:
        """Default forex trading rules."""
        # Adjust based on pair type
        if self.pair_type == "MAJOR":
            spread_markup = Decimal("0.00001")  # 0.1 pip
            min_size = Decimal("0.01")  # Micro lot
            leverage = Decimal("100.0")
        elif self.pair_type == "MINOR":
            spread_markup = Decimal("0.00003")  # 0.3 pip
            min_size = Decimal("0.01")
            leverage = Decimal("50.0")
        else:  # EXOTIC
            spread_markup = Decimal("0.0001")  # 1 pip
            min_size = Decimal("0.1")  # Mini lot minimum
            leverage = Decimal("20.0")
        
        return TradingRules(
            min_order_size=min_size,  # In lots
            max_order_size=Decimal("100"),  # 100 standard lots
            tick_size=self.pip_size / 10,  # Fractional pips
            lot_size=Decimal("0.01"),  # Micro lot increments
            max_leverage=leverage,
            allow_short_selling=True,  # Always true for FX
            allow_fractional=True,
            maker_fee=Decimal("0.0"),  # Usually no commission
            taker_fee=spread_markup,  # Spread as "fee"
            settlement_period=2,  # T+2 for spot FX
            margin_requirement=Decimal(1) / leverage,
            maintenance_margin=Decimal(1) / leverage * Decimal("0.5"),
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
        """Default forex risk parameters."""
        if self.pair_type == "MAJOR":
            return RiskParameters(
                max_position_size=Decimal("10"),  # 10 standard lots
                max_notional_value=Decimal("10000000"),  # $10M
                position_limit=10,
                daily_loss_limit=Decimal("0.02"),  # 2%
                max_drawdown=Decimal("0.10"),  # 10%
                concentration_limit=Decimal("0.30"),  # 30% in one pair
                volatility_threshold=Decimal("0.15"),  # 15% annualized
                correlation_threshold=Decimal("0.80"),
            )
        elif self.pair_type == "MINOR":
            return RiskParameters(
                max_position_size=Decimal("5"),
                max_notional_value=Decimal("5000000"),
                position_limit=8,
                daily_loss_limit=Decimal("0.015"),
                max_drawdown=Decimal("0.08"),
                concentration_limit=Decimal("0.20"),
                volatility_threshold=Decimal("0.20"),
                correlation_threshold=Decimal("0.70"),
            )
        else:  # EXOTIC
            return RiskParameters(
                max_position_size=Decimal("2"),
                max_notional_value=Decimal("2000000"),
                position_limit=5,
                daily_loss_limit=Decimal("0.01"),
                max_drawdown=Decimal("0.05"),
                concentration_limit=Decimal("0.10"),
                volatility_threshold=Decimal("0.30"),
                correlation_threshold=Decimal("0.60"),
                high_volatility_multiplier=Decimal("0.3"),
            )
    
    def _calculate_liquidity_score(self) -> Decimal:
        """Calculate liquidity score based on pair type."""
        scores = {
            "MAJOR": Decimal("1.0"),
            "MINOR": Decimal("0.7"),
            "EXOTIC": Decimal("0.3"),
        }
        return scores.get(self.pair_type, Decimal("0.5"))
    
    def validate_order(
        self,
        quantity: Quantity,
        price: Optional[Price] = None,
        order_type: str = "MARKET",
    ) -> Tuple[bool, Optional[str]]:
        """Validate forex order."""
        # Convert quantity to lots for validation
        if quantity < self.trading_rules.min_order_size:
            return False, f"Minimum order size is {self.trading_rules.min_order_size} lots"
        
        if quantity > self.trading_rules.max_order_size:
            return False, f"Maximum order size is {self.trading_rules.max_order_size} lots"
        
        # Check if market is open (forex closed on weekends)
        if not self.is_tradable(datetime.utcnow()):
            return False, "Forex market is closed (weekend)"
        
        # Validate lot size increments
        if quantity % self.trading_rules.lot_size != 0:
            return False, f"Order size must be in increments of {self.trading_rules.lot_size} lots"
        
        return True, None
    
    def calculate_fees(
        self,
        quantity: Quantity,
        price: Price,
        is_maker: bool = False,
    ) -> Decimal:
        """Calculate forex trading costs (mainly spread)."""
        # Convert lots to units
        units = quantity * self.standard_lot_size
        
        # Forex typically charges through spread, not commission
        # But some ECN brokers charge commission
        if self.trading_rules.maker_fee > 0 or self.trading_rules.taker_fee > 0:
            # Commission-based pricing (per million traded)
            notional = units * price
            commission_per_million = Decimal("35")  # $35 per million
            fee = (notional / Decimal("1000000")) * commission_per_million
        else:
            # Spread-based pricing (included in price)
            fee = Decimal("0.0")
        
        return fee
    
    def calculate_margin_requirement(
        self,
        quantity: Quantity,
        price: Price,
        is_short: bool = False,
    ) -> Decimal:
        """Calculate margin requirement for forex position."""
        # Convert lots to units
        units = quantity * self.standard_lot_size
        notional = units * price
        
        # Base margin calculation
        margin = notional * self.trading_rules.margin_requirement
        
        # Add swap consideration for overnight positions
        if self.swap_long != 0 or self.swap_short != 0:
            daily_swap = self.swap_long if not is_short else self.swap_short
            # Add 3 days of negative swap as buffer
            if daily_swap < 0:
                margin += abs(daily_swap * units * 3)
        
        # Adjust for pair volatility
        if self.pair_type == "EXOTIC":
            margin *= Decimal("1.5")  # 50% extra for exotic pairs
        
        return margin
    
    def calculate_pip_value(
        self,
        quantity: Quantity,
        price: Price,
        account_currency: Currency,
    ) -> Decimal:
        """Calculate the value of one pip movement."""
        # Convert lots to units
        units = quantity * self.standard_lot_size
        
        # For XXX/USD pairs, pip value is straightforward
        if self.quote_currency == Currency.USD:
            if self.is_jpy_pair:
                pip_value = units * Decimal("0.01")
            else:
                pip_value = units * Decimal("0.0001")
        else:
            # Need to convert through exchange rate
            if self.is_jpy_pair:
                pip_value = (units * Decimal("0.01")) / price
            else:
                pip_value = (units * Decimal("0.0001")) / price
            
            # Would need additional conversion if account currency != USD
            # This is simplified - real implementation would use current rates
        
        return pip_value
    
    def calculate_swap(
        self,
        quantity: Quantity,
        days_held: int,
        is_long: bool = True,
    ) -> Decimal:
        """Calculate overnight swap/rollover charges."""
        # Convert lots to units
        units = quantity * self.standard_lot_size
        
        # Use appropriate swap rate
        daily_swap = self.swap_long if is_long else self.swap_short
        
        # Triple swap on Wednesday (for weekend)
        # This is simplified - actual calculation depends on settlement conventions
        total_swap = daily_swap * units * days_held
        
        return total_swap
    
    def get_session_volatility_multiplier(self, timestamp: datetime) -> Decimal:
        """Get volatility multiplier based on trading session overlap."""
        hour = timestamp.hour
        
        # London/NY overlap (most volatile)
        if 13 <= hour <= 17:  # UTC
            return Decimal("1.5")
        # London open
        elif 8 <= hour <= 12:
            return Decimal("1.3")
        # NY open
        elif 13 <= hour <= 20:
            return Decimal("1.2")
        # Asian session (generally calmer)
        elif 0 <= hour <= 8 or 22 <= hour <= 24:
            return Decimal("0.8")
        else:
            return Decimal("1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with forex-specific fields."""
        data = super().to_dict()
        data.update({
            "pair_type": self.pair_type,
            "pip_size": float(self.pip_size),
            "standard_lot_size": float(self.standard_lot_size),
            "session": self.session,
            "is_jpy_pair": self.is_jpy_pair,
            "swap_long": float(self.swap_long),
            "swap_short": float(self.swap_short),
            "interest_rate_diff": float(self.interest_rate_diff),
            "liquidity_score": float(self.liquidity_score),
        })
        return data