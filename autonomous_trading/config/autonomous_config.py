# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Comprehensive configuration for the Autonomous Trading System.
"""

import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from nautilus_trader.config import LiveTradingNodeConfig
from nautilus_trader.model.identifiers import InstrumentId
from autonomous_trading.monitoring.notifications import NotificationConfig


class RiskLimitsConfig(BaseModel):
    """Risk management limits configuration."""
    max_daily_loss_percent: float = Field(default=2.0, ge=0.1, le=10.0)
    max_drawdown_percent: float = Field(default=10.0, ge=1.0, le=50.0)
    max_position_risk_percent: float = Field(default=1.0, ge=0.1, le=5.0)
    max_portfolio_risk_percent: float = Field(default=5.0, ge=1.0, le=20.0)
    max_correlation: float = Field(default=0.7, ge=0.0, le=1.0)
    var_confidence_level: float = Field(default=0.95, ge=0.9, le=0.99)
    emergency_stop_loss_percent: float = Field(default=5.0, ge=1.0, le=20.0)


class StrategyOrchestratorConfig(BaseModel):
    """Strategy orchestration configuration."""
    max_concurrent_strategies: int = Field(default=5, ge=1, le=20)
    min_strategy_allocation: float = Field(default=0.05, ge=0.01, le=0.2)
    max_strategy_allocation: float = Field(default=0.30, ge=0.1, le=0.5)
    performance_lookback_days: int = Field(default=30, ge=7, le=90)
    rebalance_interval_hours: int = Field(default=24, ge=1, le=168)
    enable_ai_strategies: bool = Field(default=True)
    enable_traditional_strategies: bool = Field(default=True)


class PerformanceOptimizerConfig(BaseModel):
    """Performance optimization configuration."""
    optimization_interval_hours: int = Field(default=24, ge=1, le=168)
    min_samples_for_optimization: int = Field(default=50, ge=10, le=1000)
    exploration_fraction: float = Field(default=0.2, ge=0.0, le=0.5)
    performance_window_days: int = Field(default=30, ge=7, le=90)
    enable_bayesian_optimization: bool = Field(default=True)
    enable_ml_optimization: bool = Field(default=True)


class MarketAnalyzerConfig(BaseModel):
    """Market analysis configuration."""
    lookback_periods: int = Field(default=100, ge=20, le=500)
    update_interval_seconds: int = Field(default=60, ge=10, le=600)
    anomaly_threshold: float = Field(default=3.0, ge=2.0, le=5.0)
    enable_regime_detection: bool = Field(default=True)
    enable_liquidity_analysis: bool = Field(default=True)
    enable_volatility_analysis: bool = Field(default=True)


class AutonomousSystemConfig(BaseModel):
    """Main configuration for the Autonomous Trading System."""
    
    # System identification
    system_name: str = Field(default="NATS-001", description="System identifier")
    trader_id: str = Field(default="AUTONOMOUS-001", description="Trader ID")
    
    # Core components
    enable_self_healing: bool = Field(default=True, description="Enable self-healing mechanisms")
    health_check_interval_seconds: int = Field(default=60, ge=10, le=600)
    max_recovery_attempts: int = Field(default=3, ge=1, le=10)
    recovery_delay_seconds: int = Field(default=30, ge=10, le=300)
    
    # Operational settings
    enable_auto_shutdown: bool = Field(default=True, description="Enable automatic shutdown on critical errors")
    daily_maintenance_time: Optional[str] = Field(default="03:00", description="Daily maintenance time in UTC (HH:MM)")
    enable_notifications: bool = Field(default=True, description="Enable notification system")
    state_persistence_path: str = Field(default="./autonomous_state.json", description="Path for state persistence")
    
    # Target performance
    target_annual_return_percent: float = Field(default=10.0, ge=1.0, le=50.0)
    min_acceptable_sharpe_ratio: float = Field(default=1.0, ge=0.0, le=5.0)
    
    # Risk configuration
    risk_limits: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    
    # Strategy configuration
    strategy_orchestrator: StrategyOrchestratorConfig = Field(default_factory=StrategyOrchestratorConfig)
    
    # Performance optimization
    performance_optimizer: PerformanceOptimizerConfig = Field(default_factory=PerformanceOptimizerConfig)
    
    # Market analysis
    market_analyzer: MarketAnalyzerConfig = Field(default_factory=MarketAnalyzerConfig)
    
    # Notification configuration
    notification_config: Optional[NotificationConfig] = Field(default=None)
    
    # Trading instruments
    instruments: List[str] = Field(
        default_factory=lambda: [
            "BTCUSDT.BINANCE",
            "ETHUSDT.BINANCE",
            "EURUSD.IDEALPRO",
            "GBPUSD.IDEALPRO",
        ],
        description="List of instruments to trade"
    )
    
    # Exchange/broker configuration
    exchange_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "BINANCE": {
                "api_key": os.getenv("BINANCE_API_KEY", ""),
                "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                "testnet": True,
            },
            "INTERACTIVE_BROKERS": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
            },
        },
        description="Exchange and broker configurations"
    )
    
    # Advanced features
    enable_paper_trading: bool = Field(default=False, description="Enable paper trading mode")
    enable_backtesting_validation: bool = Field(default=True, description="Validate strategies with backtesting")
    enable_walk_forward_analysis: bool = Field(default=True, description="Enable walk-forward optimization")
    
    # Resource management
    max_cpu_percent: float = Field(default=80.0, ge=10.0, le=100.0)
    max_memory_mb: int = Field(default=4096, ge=512, le=32768)
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True


def load_autonomous_config(config_path: Optional[str] = None) -> AutonomousSystemConfig:
    """Load autonomous system configuration from file or environment."""
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return AutonomousSystemConfig(**config_data)
    
    # Load from environment variables or use defaults
    return AutonomousSystemConfig()


def create_live_trading_config(autonomous_config: AutonomousSystemConfig) -> LiveTradingNodeConfig:
    """Create Nautilus LiveTradingNodeConfig from autonomous config."""
    # This would create the appropriate Nautilus configuration
    # based on the autonomous system configuration
    # Placeholder implementation
    
    config_dict = {
        "trader_id": autonomous_config.trader_id,
        "log_level": autonomous_config.log_level,
        # Additional configuration mapping...
    }
    
    return LiveTradingNodeConfig.parse_obj(config_dict)