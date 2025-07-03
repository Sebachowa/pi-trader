
"""
Cryptocurrency asset implementation.
"""

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


class CryptoAsset(Asset):
    """
    Cryptocurrency asset implementation.
    
    Supports spot, futures, and perpetual contracts across multiple exchanges.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        base_currency: Currency,
        quote_currency: Currency,
        exchange_type: str = "SPOT",  # SPOT, FUTURES, PERP
        contract_size: Decimal = Decimal("1.0"),
        is_stablecoin: bool = False,
        network: Optional[str] = None,
        trading_rules: Optional[TradingRules] = None,
        risk_parameters: Optional[RiskParameters] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.exchange_type = exchange_type
        self.contract_size = contract_size
        self.is_stablecoin = is_stablecoin
        self.network = network
        
        super().__init__(
            instrument_id=instrument_id,
            asset_class=AssetClass.CRYPTO,
            base_currency=base_currency,
            quote_currency=quote_currency,
            trading_rules=trading_rules,
            risk_parameters=risk_parameters,
            metadata=metadata,
        )
        
        # Crypto-specific attributes
        self.funding_rate = Decimal("0.0")
        self.open_interest = Decimal("0.0")
        self.volume_24h = Decimal("0.0")
        
    def _default_market_hours(self) -> MarketHours:
        """Crypto markets are 24/7."""
        return MarketHours(
            timezone="UTC",
            open_time=None,  # 24/7
            close_time=None,
            trading_days=list(range(7)),  # All days
        )
    
    def _default_trading_rules(self) -> TradingRules:
        """Default crypto trading rules."""
        # Adjust based on exchange type
        if self.exchange_type == "SPOT":
            return TradingRules(
                min_order_size=Decimal("0.00001"),
                max_order_size=Decimal("10000.0"),
                tick_size=Decimal("0.01"),
                lot_size=Decimal("0.00001"),
                max_leverage=Decimal("3.0"),  # Spot margin
                allow_short_selling=False,
                allow_fractional=True,
                maker_fee=Decimal("0.001"),  # 0.1%
                taker_fee=Decimal("0.001"),
                settlement_period=0,  # Instant
                margin_requirement=Decimal("0.33"),  # 3x leverage
                maintenance_margin=Decimal("0.25"),
                support_trailing_stop=True,
                support_iceberg_orders=True,
            )
        else:  # Futures/Perp
            return TradingRules(
                min_order_size=Decimal("0.001"),
                max_order_size=Decimal("1000.0"),
                tick_size=Decimal("0.1"),
                lot_size=Decimal("0.001"),
                max_leverage=Decimal("125.0"),
                allow_short_selling=True,
                allow_fractional=True,
                maker_fee=Decimal("0.0002"),  # 0.02%
                taker_fee=Decimal("0.0005"),  # 0.05%
                settlement_period=0,
                margin_requirement=Decimal("0.008"),  # 125x leverage
                maintenance_margin=Decimal("0.004"),
                support_trailing_stop=True,
                support_iceberg_orders=True,
                support_contingent_orders=True,
            )
    
    def _default_risk_parameters(self) -> RiskParameters:
        """Default crypto risk parameters."""
        # More aggressive for crypto due to 24/7 nature
        if self.is_stablecoin:
            return RiskParameters(
                max_position_size=Decimal("1000000.0"),
                max_notional_value=Decimal("1000000.0"),
                position_limit=10,
                daily_loss_limit=Decimal("0.01"),  # 1% for stables
                max_drawdown=Decimal("0.05"),
                concentration_limit=Decimal("0.50"),  # Can hold more stables
                volatility_threshold=Decimal("0.10"),
                correlation_threshold=Decimal("0.90"),
            )
        else:
            return RiskParameters(
                max_position_size=Decimal("100.0"),
                max_notional_value=Decimal("1000000.0"),
                position_limit=5,
                daily_loss_limit=Decimal("0.05"),  # 5% for volatile crypto
                max_drawdown=Decimal("0.20"),  # 20% drawdown
                concentration_limit=Decimal("0.15"),  # 15% per crypto
                volatility_threshold=Decimal("1.00"),  # 100% annualized vol
                correlation_threshold=Decimal("0.70"),
                high_volatility_multiplier=Decimal("0.3"),  # More conservative
            )
    
    def validate_order(
        self,
        quantity: Quantity,
        price: Optional[Price] = None,
        order_type: str = "MARKET",
    ) -> Tuple[bool, Optional[str]]:
        """Validate crypto order."""
        # Check minimum order size
        if quantity < self.trading_rules.min_order_size:
            return False, f"Order size {quantity} below minimum {self.trading_rules.min_order_size}"
        
        # Check maximum order size
        if quantity > self.trading_rules.max_order_size:
            return False, f"Order size {quantity} exceeds maximum {self.trading_rules.max_order_size}"
        
        # Check lot size
        if quantity % self.trading_rules.lot_size != 0:
            return False, f"Order size must be multiple of {self.trading_rules.lot_size}"
        
        # Check price tick size for limit orders
        if price and order_type in ["LIMIT", "STOP_LIMIT"]:
            if price % self.trading_rules.tick_size != 0:
                return False, f"Price must be multiple of tick size {self.trading_rules.tick_size}"
        
        # Check if short selling is allowed
        if order_type == "SELL" and not self.trading_rules.allow_short_selling:
            if self.exchange_type == "SPOT":
                return False, "Short selling not allowed for spot markets"
        
        return True, None
    
    def calculate_fees(
        self,
        quantity: Quantity,
        price: Price,
        is_maker: bool = False,
    ) -> Decimal:
        """Calculate crypto trading fees."""
        notional = quantity * price
        fee_rate = self.trading_rules.maker_fee if is_maker else self.trading_rules.taker_fee
        
        # Apply volume discounts if available
        if self.volume_24h > Decimal("10000000"):  # $10M volume
            fee_rate *= Decimal("0.8")  # 20% discount
        elif self.volume_24h > Decimal("1000000"):  # $1M volume
            fee_rate *= Decimal("0.9")  # 10% discount
        
        return notional * fee_rate
    
    def calculate_margin_requirement(
        self,
        quantity: Quantity,
        price: Price,
        is_short: bool = False,
    ) -> Decimal:
        """Calculate margin requirement for crypto position."""
        notional = quantity * price * self.contract_size
        
        # Initial margin requirement
        margin = notional * self.trading_rules.margin_requirement
        
        # Add funding rate consideration for perpetuals
        if self.exchange_type == "PERP" and self.funding_rate != 0:
            # Estimate 8-hour funding cost
            funding_cost = abs(self.funding_rate) * notional / 3
            margin += funding_cost
        
        # Add volatility buffer
        if hasattr(self, "_volatility") and self._volatility > self.risk_parameters.volatility_threshold:
            margin *= Decimal("1.5")  # 50% extra margin for high volatility
        
        return margin
    
    def calculate_funding_payment(
        self,
        position_size: Quantity,
        hours_held: int = 8,
    ) -> Decimal:
        """Calculate funding payment for perpetual contracts."""
        if self.exchange_type != "PERP":
            return Decimal("0.0")
        
        # Most exchanges have 8-hour funding intervals
        funding_periods = hours_held // 8
        return position_size * self.funding_rate * funding_periods
    
    def get_liquidation_price(
        self,
        entry_price: Price,
        position_size: Quantity,
        margin: Decimal,
        is_long: bool = True,
    ) -> Price:
        """Calculate liquidation price for leveraged position."""
        if self.exchange_type == "SPOT":
            return Price(0)  # No liquidation for spot
        
        # Liquidation occurs when losses exceed available margin
        margin_ratio = margin / (position_size * entry_price)
        
        if is_long:
            # Long liquidation: entry_price * (1 - margin_ratio)
            liq_price = entry_price * (1 - margin_ratio + self.trading_rules.maintenance_margin)
        else:
            # Short liquidation: entry_price * (1 + margin_ratio)
            liq_price = entry_price * (1 + margin_ratio - self.trading_rules.maintenance_margin)
        
        return Price(max(0, liq_price))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with crypto-specific fields."""
        data = super().to_dict()
        data.update({
            "exchange_type": self.exchange_type,
            "contract_size": float(self.contract_size),
            "is_stablecoin": self.is_stablecoin,
            "network": self.network,
            "funding_rate": float(self.funding_rate),
            "open_interest": float(self.open_interest),
            "volume_24h": float(self.volume_24h),
        })
        return data