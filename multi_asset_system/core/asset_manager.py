
"""
Multi-asset manager for unified asset handling.
"""

import json
from collections import defaultdict
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Type

from nautilus_trader.common.component import Component
from nautilus_trader.common.logging import Logger
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price

from multi_asset_system.core.asset_interface import (
    Asset,
    AssetClass,
    MarketStatus,
    RiskParameters,
    TradingRules,
)
from multi_asset_system.assets.crypto_asset import CryptoAsset
from multi_asset_system.assets.equity_asset import EquityAsset
from multi_asset_system.assets.forex_asset import ForexAsset
from multi_asset_system.assets.commodity_asset import CommodityAsset


class AssetFactory:
    """Factory for creating asset instances."""
    
    @staticmethod
    def create_asset(
        instrument_id: InstrumentId,
        asset_class: AssetClass,
        config: Dict[str, Any],
    ) -> Asset:
        """Create an asset instance based on asset class."""
        if asset_class == AssetClass.CRYPTO:
            return CryptoAsset(
                instrument_id=instrument_id,
                base_currency=Currency.from_str(config.get("base_currency", "BTC")),
                quote_currency=Currency.from_str(config.get("quote_currency", "USDT")),
                exchange_type=config.get("exchange_type", "SPOT"),
                contract_size=Decimal(str(config.get("contract_size", "1.0"))),
                is_stablecoin=config.get("is_stablecoin", False),
                network=config.get("network"),
            )
        
        elif asset_class == AssetClass.EQUITY:
            return EquityAsset(
                instrument_id=instrument_id,
                currency=Currency.from_str(config.get("currency", "USD")),
                exchange=config.get("exchange", "NYSE"),
                sector=config.get("sector"),
                industry=config.get("industry"),
                market_cap=Decimal(str(config.get("market_cap", "0"))) if config.get("market_cap") else None,
                is_adr=config.get("is_adr", False),
                is_etf=config.get("is_etf", False),
                dividend_yield=Decimal(str(config.get("dividend_yield", "0"))) if config.get("dividend_yield") else None,
            )
        
        elif asset_class == AssetClass.FOREX:
            return ForexAsset(
                instrument_id=instrument_id,
                base_currency=Currency.from_str(config.get("base_currency", "EUR")),
                quote_currency=Currency.from_str(config.get("quote_currency", "USD")),
                pair_type=config.get("pair_type", "MAJOR"),
                pip_size=Decimal(str(config.get("pip_size", "0.0001"))),
                lot_size=Decimal(str(config.get("lot_size", "100000"))),
                session=config.get("session", "GLOBAL"),
            )
        
        elif asset_class in [AssetClass.COMMODITY, AssetClass.FUTURE]:
            expiry_str = config.get("expiry_date")
            expiry_date = datetime.fromisoformat(expiry_str).date() if expiry_str else date.today()
            
            return CommodityAsset(
                instrument_id=instrument_id,
                commodity_type=config.get("commodity_type", "ENERGY"),
                contract_size=Decimal(str(config.get("contract_size", "1000"))),
                contract_unit=config.get("contract_unit", "barrels"),
                tick_value=Decimal(str(config.get("tick_value", "10"))),
                expiry_date=expiry_date,
                exchange=config.get("exchange", "CME"),
                trading_hours_type=config.get("trading_hours_type", "REGULAR"),
                grade=config.get("grade"),
            )
        
        else:
            raise ValueError(f"Unsupported asset class: {asset_class}")


class MultiAssetManager(Component):
    """
    Centralized manager for all asset types.
    
    Provides unified interface for:
    - Asset registration and discovery
    - Market status monitoring
    - Risk parameter management
    - Cross-asset analytics
    - Trading rule enforcement
    """
    
    def __init__(
        self,
        logger: Logger,
        config_path: Optional[str] = None,
    ):
        super().__init__(
            logger=logger,
            component_id="MultiAssetManager",
        )
        
        # Asset storage
        self._assets: Dict[InstrumentId, Asset] = {}
        self._assets_by_class: Dict[AssetClass, Set[InstrumentId]] = defaultdict(set)
        self._assets_by_venue: Dict[Venue, Set[InstrumentId]] = defaultdict(set)
        
        # Market status tracking
        self._market_status: Dict[InstrumentId, MarketStatus] = {}
        self._last_status_update = datetime.utcnow()
        
        # Risk aggregation
        self._portfolio_risk_params = self._default_portfolio_risk_params()
        
        # Load configuration if provided
        if config_path:
            self.load_configuration(config_path)
    
    def _default_portfolio_risk_params(self) -> RiskParameters:
        """Default portfolio-wide risk parameters."""
        return RiskParameters(
            max_position_size=Decimal("1000000"),  # $1M USD equivalent
            max_notional_value=Decimal("10000000"),  # $10M total
            position_limit=50,  # Total positions
            daily_loss_limit=Decimal("0.03"),  # 3% portfolio
            max_drawdown=Decimal("0.20"),  # 20% portfolio
            concentration_limit=Decimal("1.0"),  # 100% (checked per asset)
            volatility_threshold=Decimal("0.50"),
            correlation_threshold=Decimal("0.60"),
        )
    
    def register_asset(
        self,
        asset: Asset,
        replace_existing: bool = False,
    ) -> None:
        """Register an asset with the manager."""
        instrument_id = asset.instrument_id
        
        if instrument_id in self._assets and not replace_existing:
            self._log.warning(f"Asset {instrument_id} already registered")
            return
        
        self._assets[instrument_id] = asset
        self._assets_by_class[asset.asset_class].add(instrument_id)
        self._assets_by_venue[asset.venue].add(instrument_id)
        
        self._log.info(f"Registered {asset.asset_class.value} asset: {instrument_id}")
    
    def create_and_register_asset(
        self,
        instrument_id: InstrumentId,
        asset_class: AssetClass,
        config: Dict[str, Any],
    ) -> Asset:
        """Create and register a new asset."""
        asset = AssetFactory.create_asset(instrument_id, asset_class, config)
        self.register_asset(asset)
        return asset
    
    def get_asset(self, instrument_id: InstrumentId) -> Optional[Asset]:
        """Get an asset by instrument ID."""
        return self._assets.get(instrument_id)
    
    def get_assets_by_class(self, asset_class: AssetClass) -> List[Asset]:
        """Get all assets of a specific class."""
        instrument_ids = self._assets_by_class.get(asset_class, set())
        return [self._assets[iid] for iid in instrument_ids]
    
    def get_assets_by_venue(self, venue: Venue) -> List[Asset]:
        """Get all assets from a specific venue."""
        instrument_ids = self._assets_by_venue.get(venue, set())
        return [self._assets[iid] for iid in instrument_ids]
    
    def get_tradable_assets(
        self,
        timestamp: Optional[datetime] = None,
        asset_class: Optional[AssetClass] = None,
    ) -> List[Asset]:
        """Get all currently tradable assets."""
        timestamp = timestamp or datetime.utcnow()
        
        # Update market status if needed
        if (timestamp - self._last_status_update).seconds > 60:
            self.update_all_market_status(timestamp)
        
        tradable = []
        assets = self.get_assets_by_class(asset_class) if asset_class else list(self._assets.values())
        
        for asset in assets:
            if asset.is_tradable(timestamp):
                tradable.append(asset)
        
        return tradable
    
    def update_all_market_status(self, timestamp: Optional[datetime] = None) -> None:
        """Update market status for all assets."""
        timestamp = timestamp or datetime.utcnow()
        
        for asset in self._assets.values():
            status = asset.update_market_status(timestamp)
            self._market_status[asset.instrument_id] = status
        
        self._last_status_update = timestamp
        self._log.debug(f"Updated market status for {len(self._assets)} assets")
    
    def validate_cross_asset_order(
        self,
        instrument_id: InstrumentId,
        quantity: Decimal,
        price: Optional[Price] = None,
        portfolio_value: Optional[Decimal] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate order against portfolio-wide risk limits.
        
        This checks concentration limits and cross-asset exposure.
        """
        asset = self.get_asset(instrument_id)
        if not asset:
            return False, f"Unknown asset: {instrument_id}"
        
        # First check asset-specific validation
        valid, error = asset.validate_order(quantity, price)
        if not valid:
            return valid, error
        
        # Check portfolio concentration if portfolio value provided
        if portfolio_value and price:
            position_value = quantity * price
            concentration = position_value / portfolio_value
            
            if concentration > asset.risk_parameters.concentration_limit:
                return False, f"Position would exceed concentration limit of {asset.risk_parameters.concentration_limit:.1%}"
        
        # Check cross-asset correlation limits
        # This would need portfolio context
        
        return True, None
    
    def get_correlated_assets(
        self,
        instrument_id: InstrumentId,
        threshold: Decimal = Decimal("0.70"),
    ) -> List[Tuple[Asset, Decimal]]:
        """Get assets correlated above threshold."""
        # This is a placeholder - real implementation would use historical data
        asset = self.get_asset(instrument_id)
        if not asset:
            return []
        
        correlated = []
        
        # Simplified correlation logic
        if asset.asset_class == AssetClass.CRYPTO:
            # Cryptos tend to be correlated
            for other in self.get_assets_by_class(AssetClass.CRYPTO):
                if other.instrument_id != instrument_id:
                    if asset.symbol[:3] == other.symbol[:3]:  # Same base currency
                        correlated.append((other, Decimal("0.90")))
                    else:
                        correlated.append((other, Decimal("0.75")))
        
        elif asset.asset_class == AssetClass.FOREX:
            # Currency pairs with shared currencies are correlated
            for other in self.get_assets_by_class(AssetClass.FOREX):
                if other.instrument_id != instrument_id:
                    if asset.base_currency == other.base_currency or asset.quote_currency == other.quote_currency:
                        correlated.append((other, Decimal("0.80")))
        
        return [(a, c) for a, c in correlated if c >= threshold]
    
    def calculate_portfolio_metrics(
        self,
        positions: Dict[InstrumentId, Decimal],
        prices: Dict[InstrumentId, Price],
    ) -> Dict[str, Any]:
        """Calculate portfolio-wide risk metrics."""
        total_value = Decimal("0")
        asset_class_exposure = defaultdict(Decimal)
        venue_exposure = defaultdict(Decimal)
        currency_exposure = defaultdict(Decimal)
        
        for instrument_id, position_size in positions.items():
            asset = self.get_asset(instrument_id)
            if not asset or instrument_id not in prices:
                continue
            
            position_value = position_size * prices[instrument_id]
            total_value += position_value
            
            # Aggregate exposures
            asset_class_exposure[asset.asset_class] += position_value
            venue_exposure[asset.venue] += position_value
            currency_exposure[asset.quote_currency] += position_value
        
        # Calculate concentration metrics
        concentrations = {}
        for instrument_id, position_size in positions.items():
            if instrument_id in prices:
                position_value = position_size * prices[instrument_id]
                concentrations[instrument_id] = position_value / total_value if total_value > 0 else Decimal("0")
        
        return {
            "total_value": total_value,
            "asset_class_exposure": dict(asset_class_exposure),
            "venue_exposure": dict(venue_exposure),
            "currency_exposure": dict(currency_exposure),
            "concentrations": concentrations,
            "position_count": len(positions),
            "max_concentration": max(concentrations.values()) if concentrations else Decimal("0"),
        }
    
    def get_market_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        asset_classes: Optional[List[AssetClass]] = None,
    ) -> Dict[date, List[Dict[str, Any]]]:
        """Get market events calendar."""
        calendar = defaultdict(list)
        
        assets = []
        if asset_classes:
            for asset_class in asset_classes:
                assets.extend(self.get_assets_by_class(asset_class))
        else:
            assets = list(self._assets.values())
        
        for asset in assets:
            # Add holidays
            if hasattr(asset.market_hours, 'holidays'):
                for holiday in asset.market_hours.holidays:
                    if start_date.date() <= holiday.date() <= end_date.date():
                        calendar[holiday.date()].append({
                            "type": "HOLIDAY",
                            "asset": str(asset.instrument_id),
                            "description": "Market closed",
                        })
            
            # Add futures expiries
            if isinstance(asset, CommodityAsset):
                if start_date.date() <= asset.expiry_date <= end_date.date():
                    calendar[asset.expiry_date].append({
                        "type": "EXPIRY",
                        "asset": str(asset.instrument_id),
                        "description": f"Contract expiry",
                    })
                
                if asset.first_notice_date and start_date.date() <= asset.first_notice_date <= end_date.date():
                    calendar[asset.first_notice_date].append({
                        "type": "FIRST_NOTICE",
                        "asset": str(asset.instrument_id),
                        "description": "First notice date",
                    })
            
            # Add earnings dates
            if isinstance(asset, EquityAsset) and asset.earnings_date:
                if start_date.date() <= asset.earnings_date.date() <= end_date.date():
                    calendar[asset.earnings_date.date()].append({
                        "type": "EARNINGS",
                        "asset": str(asset.instrument_id),
                        "description": "Earnings release",
                    })
        
        return dict(calendar)
    
    def save_configuration(self, filepath: str) -> None:
        """Save current asset configuration to file."""
        config = {
            "assets": {},
            "portfolio_risk_params": {
                "max_position_size": str(self._portfolio_risk_params.max_position_size),
                "max_notional_value": str(self._portfolio_risk_params.max_notional_value),
                "position_limit": self._portfolio_risk_params.position_limit,
                "daily_loss_limit": str(self._portfolio_risk_params.daily_loss_limit),
                "max_drawdown": str(self._portfolio_risk_params.max_drawdown),
            }
        }
        
        for instrument_id, asset in self._assets.items():
            config["assets"][str(instrument_id)] = asset.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self._log.info(f"Saved configuration to {filepath}")
    
    def load_configuration(self, filepath: str) -> None:
        """Load asset configuration from file."""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Load portfolio risk parameters
            if "portfolio_risk_params" in config:
                params = config["portfolio_risk_params"]
                self._portfolio_risk_params.max_position_size = Decimal(params.get("max_position_size", "1000000"))
                self._portfolio_risk_params.max_notional_value = Decimal(params.get("max_notional_value", "10000000"))
                self._portfolio_risk_params.position_limit = params.get("position_limit", 50)
                self._portfolio_risk_params.daily_loss_limit = Decimal(params.get("daily_loss_limit", "0.03"))
                self._portfolio_risk_params.max_drawdown = Decimal(params.get("max_drawdown", "0.20"))
            
            # Load assets
            for instrument_str, asset_config in config.get("assets", {}).items():
                parts = instrument_str.split(".")
                if len(parts) == 2:
                    symbol, venue = parts
                    instrument_id = InstrumentId(symbol=symbol, venue=Venue(venue))
                    asset_class = AssetClass(asset_config["asset_class"])
                    
                    self.create_and_register_asset(instrument_id, asset_class, asset_config)
            
            self._log.info(f"Loaded configuration from {filepath}")
            
        except Exception as e:
            self._log.error(f"Failed to load configuration: {e}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about registered assets."""
        stats = {
            "total_assets": len(self._assets),
            "by_class": {},
            "by_venue": {},
            "tradable_now": 0,
            "market_status": {},
        }
        
        # Count by class
        for asset_class in AssetClass:
            count = len(self._assets_by_class.get(asset_class, set()))
            if count > 0:
                stats["by_class"][asset_class.value] = count
        
        # Count by venue
        for venue, instruments in self._assets_by_venue.items():
            stats["by_venue"][str(venue)] = len(instruments)
        
        # Count tradable assets
        tradable = self.get_tradable_assets()
        stats["tradable_now"] = len(tradable)
        
        # Market status summary
        for status in MarketStatus:
            count = sum(1 for s in self._market_status.values() if s == status)
            if count > 0:
                stats["market_status"][status.value] = count
        
        return stats