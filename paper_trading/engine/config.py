
"""
Configuration presets for different paper trading scenarios.
"""

from decimal import Decimal
from pathlib import Path

from paper_trading.engine.core import PaperTradingConfig
from paper_trading.engine.market_impact import MarketImpactParams
from paper_trading.engine.slippage import SlippageParams
from paper_trading.engine.spread import SpreadParams


class PaperTradingConfigs:
    """Pre-configured paper trading setups for different scenarios."""
    
    @staticmethod
    def realistic_crypto() -> PaperTradingConfig:
        """Realistic configuration for crypto trading."""
        return PaperTradingConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDT",
            leverage=3,
            
            # Enable all realistic features
            enable_slippage=True,
            enable_market_impact=True,
            enable_spread_costs=True,
            enable_partial_fills=True,
            
            # Crypto-specific parameters
            slippage_params=SlippageParams(
                base_slippage_rate=0.0002,  # 2 basis points
                volatility_multiplier=3.0,   # High volatility impact
                asset_class_multipliers={"crypto": 1.5},
            ),
            
            market_impact_params=MarketImpactParams(
                linear_impact_coefficient=0.2,
                sqrt_impact_coefficient=0.8,
                temporary_impact_ratio=0.8,  # 80% temporary for crypto
            ),
            
            spread_params=SpreadParams(
                min_spreads={"crypto": 5.0},  # 5 basis points minimum
                volatility_spread_multiplier=2.5,
            ),
            
            # Fast execution for crypto
            order_processing_delay_ms=20,
            market_order_delay_ms=50,
            limit_order_delay_ms=100,
            
            # Risk limits
            max_position_size=Decimal("50000"),
            max_order_size=Decimal("10000"),
            max_daily_loss=Decimal("1000"),
            max_open_positions=20,
        )
    
    @staticmethod
    def realistic_forex() -> PaperTradingConfig:
        """Realistic configuration for forex trading."""
        return PaperTradingConfig(
            initial_balance=Decimal("100000"),
            base_currency="USD",
            leverage=50,  # Typical FX leverage
            
            enable_slippage=True,
            enable_market_impact=True,
            enable_spread_costs=True,
            enable_partial_fills=True,
            
            slippage_params=SlippageParams(
                base_slippage_rate=0.00005,  # 0.5 basis points
                volatility_multiplier=1.5,
                asset_class_multipliers={"forex": 0.5},
            ),
            
            market_impact_params=MarketImpactParams(
                linear_impact_coefficient=0.05,  # Low impact for FX
                sqrt_impact_coefficient=0.2,
                temporary_impact_ratio=0.9,  # 90% temporary
            ),
            
            spread_params=SpreadParams(
                min_spreads={"forex": 1.0},  # 1 basis point
                volatility_spread_multiplier=1.5,
            ),
            
            # Fast execution for FX
            order_processing_delay_ms=10,
            market_order_delay_ms=20,
            limit_order_delay_ms=50,
            
            # FX appropriate limits
            max_position_size=Decimal("5000000"),  # 5M units
            max_order_size=Decimal("1000000"),     # 1M units
            max_daily_loss=Decimal("5000"),
            max_open_positions=10,
        )
    
    @staticmethod
    def realistic_equities() -> PaperTradingConfig:
        """Realistic configuration for equity trading."""
        return PaperTradingConfig(
            initial_balance=Decimal("100000"),
            base_currency="USD",
            leverage=2,  # Typical margin account
            
            enable_slippage=True,
            enable_market_impact=True,
            enable_spread_costs=True,
            enable_partial_fills=True,
            
            slippage_params=SlippageParams(
                base_slippage_rate=0.0001,  # 1 basis point
                volatility_multiplier=2.0,
                size_impact_threshold=0.005,  # 0.5% of ADV
                market_open_multiplier=2.5,   # Higher at open
                market_close_multiplier=2.0,  # Higher at close
            ),
            
            market_impact_params=MarketImpactParams(
                linear_impact_coefficient=0.1,
                sqrt_impact_coefficient=0.5,
                power_law_exponent=0.6,
                temporary_impact_ratio=0.7,
            ),
            
            spread_params=SpreadParams(
                min_spreads={"equity": 2.0},  # 2 basis points
                volatility_spread_multiplier=2.0,
                time_spread_multiplier=2.0,
            ),
            
            # Realistic delays
            order_processing_delay_ms=50,
            market_order_delay_ms=100,
            limit_order_delay_ms=150,
            
            # Equity limits
            max_position_size=Decimal("50000"),
            max_order_size=Decimal("10000"),
            max_daily_loss=Decimal("2000"),
            max_open_positions=30,
        )
    
    @staticmethod
    def low_latency_hft() -> PaperTradingConfig:
        """Configuration for high-frequency trading simulation."""
        return PaperTradingConfig(
            initial_balance=Decimal("1000000"),
            base_currency="USD",
            leverage=10,
            
            enable_slippage=True,
            enable_market_impact=True,
            enable_spread_costs=True,
            enable_partial_fills=False,  # HFT typically gets full fills
            
            slippage_params=SlippageParams(
                base_slippage_rate=0.00001,  # 0.1 basis point
                volatility_multiplier=1.0,
                random_factor=0.1,  # Less randomness
            ),
            
            market_impact_params=MarketImpactParams(
                linear_impact_coefficient=0.01,  # Minimal impact
                sqrt_impact_coefficient=0.05,
                temporary_impact_ratio=0.95,  # Mostly temporary
                decay_time_minutes=5.0,  # Fast decay
            ),
            
            # Ultra-low latency
            order_processing_delay_ms=1,
            market_order_delay_ms=2,
            limit_order_delay_ms=5,
            
            # HFT limits
            max_position_size=Decimal("100000"),
            max_order_size=Decimal("10000"),
            max_daily_loss=Decimal("10000"),
            max_open_positions=100,
        )
    
    @staticmethod
    def conservative_testing() -> PaperTradingConfig:
        """Conservative configuration for strategy testing."""
        return PaperTradingConfig(
            initial_balance=Decimal("50000"),
            base_currency="USD",
            leverage=1,  # No leverage
            
            # Worst-case scenario settings
            enable_slippage=True,
            enable_market_impact=True,
            enable_spread_costs=True,
            enable_partial_fills=True,
            
            slippage_params=SlippageParams(
                base_slippage_rate=0.0003,  # 3 basis points
                volatility_multiplier=3.0,   # High impact
                size_impact_multiplier=2.0,  # Significant size impact
                random_factor=0.5,  # High randomness
            ),
            
            market_impact_params=MarketImpactParams(
                linear_impact_coefficient=0.3,
                sqrt_impact_coefficient=1.0,
                temporary_impact_ratio=0.5,  # 50/50 split
            ),
            
            spread_params=SpreadParams(
                min_spreads={"equity": 5.0},  # Wide spreads
                volatility_spread_multiplier=3.0,
                size_spread_multiplier=2.0,
            ),
            
            # Slower execution
            order_processing_delay_ms=100,
            market_order_delay_ms=200,
            limit_order_delay_ms=300,
            
            # Conservative limits
            max_position_size=Decimal("10000"),
            max_order_size=Decimal("5000"),
            max_daily_loss=Decimal("500"),
            max_open_positions=5,
        )
    
    @staticmethod
    def zero_cost_testing() -> PaperTradingConfig:
        """Zero-cost configuration for pure strategy logic testing."""
        return PaperTradingConfig(
            initial_balance=Decimal("100000"),
            base_currency="USD",
            leverage=1,
            
            # Disable all costs
            enable_slippage=False,
            enable_market_impact=False,
            enable_spread_costs=False,
            enable_partial_fills=False,
            
            # No delays
            order_processing_delay_ms=0,
            market_order_delay_ms=0,
            limit_order_delay_ms=0,
            
            # No limits
            max_position_size=None,
            max_order_size=None,
            max_daily_loss=None,
            max_open_positions=1000,
        )