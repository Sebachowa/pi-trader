
"""
Nautilus Autonomous Trading System (NATS)

A fully autonomous trading system designed for 24/7 operation with minimal intervention.
Targets 10% annual returns with comprehensive risk management and self-healing capabilities.
"""

from autonomous_trading.core.engine import AutonomousEngine
from autonomous_trading.core.risk_controller import RiskController
from autonomous_trading.core.market_analyzer import MarketAnalyzer
from autonomous_trading.strategies.orchestrator import StrategyOrchestrator
from autonomous_trading.monitoring.performance import PerformanceOptimizer
from autonomous_trading.monitoring.notifications import NotificationSystem

__all__ = [
    "AutonomousEngine",
    "RiskController",
    "MarketAnalyzer",
    "StrategyOrchestrator",
    "PerformanceOptimizer",
    "NotificationSystem",
]

__version__ = "1.0.0"