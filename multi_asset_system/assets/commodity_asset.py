
"""
Commodity and Futures asset implementation.
"""

from datetime import datetime, date, time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

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


class CommodityAsset(Asset):
    """
    Commodity and Futures asset implementation.
    
    Supports energy, metals, agriculture, and financial futures.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        commodity_type: str,  # ENERGY, METAL, AGRICULTURE, FINANCIAL
        contract_size: Decimal,
        contract_unit: str,  # barrels, ounces, bushels, etc.
        tick_value: Decimal,  # Dollar value per tick
        expiry_date: date,
        first_notice_date: Optional[date] = None,
        last_trading_date: Optional[date] = None,
        exchange: str = "CME",  # CME, ICE, NYMEX, COMEX, etc.
        trading_hours_type: str = "REGULAR",  # REGULAR, EXTENDED, 24H
        grade: Optional[str] = None,  # Delivery grade/specification
        trading_rules: Optional[TradingRules] = None,
        risk_parameters: Optional[RiskParameters] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.commodity_type = commodity_type
        self.contract_size = contract_size
        self.contract_unit = contract_unit
        self.tick_value = tick_value
        self.expiry_date = expiry_date
        self.first_notice_date = first_notice_date or expiry_date
        self.last_trading_date = last_trading_date or expiry_date
        self.exchange = exchange
        self.trading_hours_type = trading_hours_type
        self.grade = grade
        
        # Determine asset class
        asset_class = AssetClass.FUTURE if expiry_date else AssetClass.COMMODITY
        
        super().__init__(
            instrument_id=instrument_id,
            asset_class=asset_class,
            base_currency=Currency.USD,  # Most commodities priced in USD
            quote_currency=Currency.USD,
            market_hours=self._get_commodity_hours(exchange, trading_hours_type),
            trading_rules=trading_rules,
            risk_parameters=risk_parameters,
            metadata=metadata,
        )
        
        # Commodity-specific attributes
        self.basis = Decimal("0.0")  # Spot-futures basis
        self.storage_cost = Decimal("0.0")
        self.open_interest = 0
        self.volume_oi_ratio = Decimal("0.0")
        self.seasonality_factor = self._calculate_seasonality()
        self.roll_dates = self._calculate_roll_dates()
        
    def _get_commodity_hours(self, exchange: str, hours_type: str) -> MarketHours:
        """Get market hours for commodity exchanges."""
        if exchange in ["CME", "NYMEX", "COMEX"]:
            if hours_type == "24H":
                # Electronic trading (nearly 24h with maintenance break)
                return MarketHours(
                    timezone="America/Chicago",
                    open_time=time(17, 0),  # 5 PM CT Sunday
                    close_time=time(16, 0),  # 4 PM CT Friday
                    trading_days=[0, 1, 2, 3, 4],
                )
            else:
                # Regular pit trading hours
                return MarketHours(
                    timezone="America/Chicago",
                    open_time=time(8, 30),
                    close_time=time(13, 30),
                    trading_days=[0, 1, 2, 3, 4],
                )
        elif exchange == "ICE":
            return MarketHours(
                timezone="America/New_York",
                open_time=time(20, 0),  # 8 PM ET Sunday
                close_time=time(18, 0),  # 6 PM ET Friday
                trading_days=[0, 1, 2, 3, 4],
            )
        elif exchange == "LME":
            return MarketHours(
                timezone="Europe/London",
                open_time=time(1, 0),
                close_time=time(19, 0),
                trading_days=[0, 1, 2, 3, 4],
            )
        else:
            return self._default_market_hours()
    
    def _default_market_hours(self) -> MarketHours:
        """Default commodity market hours."""
        return MarketHours(
            timezone="America/Chicago",
            open_time=time(17, 0),  # Sunday evening
            close_time=time(16, 0),  # Friday afternoon
            trading_days=[0, 1, 2, 3, 4],
        )
    
    def _default_trading_rules(self) -> TradingRules:
        """Default commodity trading rules."""
        # Rules vary significantly by commodity type
        if self.commodity_type == "ENERGY":
            return TradingRules(
                min_order_size=Decimal("1"),  # 1 contract
                max_order_size=Decimal("1000"),
                tick_size=Decimal("0.01"),  # $0.01 for oil
                lot_size=Decimal("1"),
                max_leverage=Decimal("10.0"),
                allow_short_selling=True,
                allow_fractional=False,
                maker_fee=Decimal("2.50"),  # Per contract
                taker_fee=Decimal("2.50"),
                settlement_period=0,  # Daily mark-to-market
                margin_requirement=Decimal("0.10"),  # 10% initial margin
                maintenance_margin=Decimal("0.075"),  # 7.5%
                support_market_orders=True,
                support_limit_orders=True,
                support_stop_orders=True,
                support_stop_limit_orders=True,
                support_trailing_stop=False,
                support_iceberg_orders=True,
                support_contingent_orders=True,
            )
        elif self.commodity_type == "METAL":
            return TradingRules(
                min_order_size=Decimal("1"),
                max_order_size=Decimal("500"),
                tick_size=Decimal("0.10"),  # $0.10 for gold
                lot_size=Decimal("1"),
                max_leverage=Decimal("20.0"),
                allow_short_selling=True,
                allow_fractional=False,
                maker_fee=Decimal("2.00"),
                taker_fee=Decimal("2.00"),
                settlement_period=0,
                margin_requirement=Decimal("0.05"),  # 5% for precious metals
                maintenance_margin=Decimal("0.04"),
                support_market_orders=True,
                support_limit_orders=True,
                support_stop_orders=True,
                support_stop_limit_orders=True,
                support_trailing_stop=False,
                support_iceberg_orders=True,
                support_contingent_orders=True,
            )
        else:  # AGRICULTURE
            return TradingRules(
                min_order_size=Decimal("1"),
                max_order_size=Decimal("200"),
                tick_size=Decimal("0.25"),  # Quarter cent for grains
                lot_size=Decimal("1"),
                max_leverage=Decimal("8.0"),
                allow_short_selling=True,
                allow_fractional=False,
                maker_fee=Decimal("2.00"),
                taker_fee=Decimal("2.00"),
                settlement_period=0,
                margin_requirement=Decimal("0.125"),  # 12.5%
                maintenance_margin=Decimal("0.10"),
                support_market_orders=True,
                support_limit_orders=True,
                support_stop_orders=True,
                support_stop_limit_orders=True,
                support_trailing_stop=False,
                support_iceberg_orders=True,
                support_contingent_orders=True,
            )
    
    def _default_risk_parameters(self) -> RiskParameters:
        """Default commodity risk parameters."""
        if self.commodity_type == "ENERGY":
            return RiskParameters(
                max_position_size=Decimal("100"),  # contracts
                max_notional_value=Decimal("5000000"),  # $5M
                position_limit=10,
                daily_loss_limit=Decimal("0.03"),  # 3% - higher for volatility
                max_drawdown=Decimal("0.15"),
                concentration_limit=Decimal("0.25"),  # 25% in one commodity
                volatility_threshold=Decimal("0.50"),  # 50% annualized
                correlation_threshold=Decimal("0.60"),
                price_limit_up=Decimal("0.10"),  # Daily limit moves
                price_limit_down=Decimal("0.10"),
            )
        elif self.commodity_type == "METAL":
            return RiskParameters(
                max_position_size=Decimal("50"),
                max_notional_value=Decimal("10000000"),  # $10M for gold
                position_limit=8,
                daily_loss_limit=Decimal("0.02"),
                max_drawdown=Decimal("0.12"),
                concentration_limit=Decimal("0.30"),
                volatility_threshold=Decimal("0.30"),
                correlation_threshold=Decimal("0.70"),
                price_limit_up=Decimal("0.05"),
                price_limit_down=Decimal("0.05"),
            )
        else:  # AGRICULTURE
            return RiskParameters(
                max_position_size=Decimal("50"),
                max_notional_value=Decimal("2000000"),
                position_limit=15,  # More diversification needed
                daily_loss_limit=Decimal("0.025"),
                max_drawdown=Decimal("0.15"),
                concentration_limit=Decimal("0.15"),  # Lower concentration
                volatility_threshold=Decimal("0.40"),
                correlation_threshold=Decimal("0.50"),
                price_limit_up=Decimal("0.07"),  # Ag has limit moves
                price_limit_down=Decimal("0.07"),
            )
    
    def _calculate_seasonality(self) -> Decimal:
        """Calculate seasonality factor for agricultural commodities."""
        if self.commodity_type != "AGRICULTURE":
            return Decimal("1.0")
        
        # Simplified seasonality - real implementation would be more complex
        month = datetime.now().month
        
        # Harvest season (Sep-Nov) typically lower prices
        if 9 <= month <= 11:
            return Decimal("0.9")
        # Planting season (Apr-May) uncertainty
        elif 4 <= month <= 5:
            return Decimal("1.1")
        else:
            return Decimal("1.0")
    
    def _calculate_roll_dates(self) -> List[date]:
        """Calculate futures roll dates."""
        # Simplified - typically roll 5-10 days before expiry
        roll_dates = []
        
        # Would calculate based on contract specifications
        # This is a placeholder implementation
        if self.last_trading_date:
            roll_date = self.last_trading_date.replace(day=self.last_trading_date.day - 5)
            roll_dates.append(roll_date)
        
        return roll_dates
    
    def validate_order(
        self,
        quantity: Quantity,
        price: Optional[Price] = None,
        order_type: str = "MARKET",
    ) -> Tuple[bool, Optional[str]]:
        """Validate commodity/futures order."""
        # Check expiry
        days_to_expiry = (self.expiry_date - datetime.now().date()).days
        
        if days_to_expiry <= 0:
            return False, "Contract has expired"
        
        # Warn about first notice date for physical delivery
        if self.first_notice_date:
            days_to_notice = (self.first_notice_date - datetime.now().date()).days
            if days_to_notice <= 3 and days_to_notice > 0:
                # This would typically trigger a warning, not block the order
                pass
        
        # Check position limits (exchange-imposed)
        # This would need portfolio context
        
        # Standard validations
        return super().validate_order(quantity, price, order_type)
    
    def calculate_fees(
        self,
        quantity: Quantity,
        price: Price,
        is_maker: bool = False,
    ) -> Decimal:
        """Calculate commodity trading fees."""
        # Futures typically charge per contract
        per_contract_fee = self.trading_rules.maker_fee if is_maker else self.trading_rules.taker_fee
        
        # Total fees
        total_fee = quantity * per_contract_fee
        
        # Add exchange fees (simplified)
        exchange_fee = quantity * Decimal("1.00")  # $1 per contract
        
        return total_fee + exchange_fee
    
    def calculate_margin_requirement(
        self,
        quantity: Quantity,
        price: Price,
        is_short: bool = False,
    ) -> Decimal:
        """Calculate margin requirement for commodity position."""
        # Contract value
        notional = quantity * self.contract_size * price
        
        # Base margin
        margin = notional * self.trading_rules.margin_requirement
        
        # Adjust for volatility
        if self.commodity_type == "ENERGY":
            # Energy markets can be very volatile
            current_volatility = self._estimate_current_volatility()
            if current_volatility > self.risk_parameters.volatility_threshold:
                margin *= Decimal("1.5")
        
        # Adjust for time to expiry (higher margin near expiry)
        days_to_expiry = (self.expiry_date - datetime.now().date()).days
        if days_to_expiry < 10:
            margin *= Decimal("1.2")
        
        # Spread margin credits (would need portfolio context)
        # Calendar spreads, inter-commodity spreads get margin relief
        
        return margin
    
    def calculate_contract_value(self, price: Price) -> Decimal:
        """Calculate the full value of one contract."""
        return self.contract_size * price
    
    def calculate_tick_value_in_currency(self) -> Decimal:
        """Get the dollar value of one minimum price movement."""
        return self.tick_value
    
    def _estimate_current_volatility(self) -> Decimal:
        """Estimate current implied volatility (placeholder)."""
        # Would use options data or historical volatility
        base_vols = {
            "ENERGY": Decimal("0.40"),
            "METAL": Decimal("0.25"),
            "AGRICULTURE": Decimal("0.30"),
            "FINANCIAL": Decimal("0.15"),
        }
        return base_vols.get(self.commodity_type, Decimal("0.30"))
    
    def is_in_delivery_period(self) -> bool:
        """Check if contract is in delivery period."""
        if not self.first_notice_date:
            return False
        
        today = datetime.now().date()
        return today >= self.first_notice_date
    
    def get_nearby_contract_month(self) -> str:
        """Get the nearby contract month code."""
        # Futures month codes
        month_codes = {
            1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
            7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
        }
        return month_codes.get(self.expiry_date.month, "")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with commodity-specific fields."""
        data = super().to_dict()
        data.update({
            "commodity_type": self.commodity_type,
            "contract_size": float(self.contract_size),
            "contract_unit": self.contract_unit,
            "tick_value": float(self.tick_value),
            "expiry_date": self.expiry_date.isoformat(),
            "first_notice_date": self.first_notice_date.isoformat() if self.first_notice_date else None,
            "last_trading_date": self.last_trading_date.isoformat() if self.last_trading_date else None,
            "exchange": self.exchange,
            "trading_hours_type": self.trading_hours_type,
            "grade": self.grade,
            "basis": float(self.basis),
            "open_interest": self.open_interest,
            "seasonality_factor": float(self.seasonality_factor),
            "days_to_expiry": (self.expiry_date - datetime.now().date()).days,
        })
        return data