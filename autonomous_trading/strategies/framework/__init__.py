"""
Self-Improving Trading Strategy Framework

A comprehensive framework for creating, testing, and evolving trading strategies
with machine learning capabilities and automated optimization.
"""

from .interfaces import (
    BaseStrategyInterface,
    StrategyComponent,
    IndicatorSet,
    SignalGenerator,
    RiskManager,
    PositionSizer,
    OrderExecutor,
    PerformanceTracker,
)

from .core import (
    AdaptiveStrategy,
    StrategyLearner,
    ParameterOptimizer,
    MarketAdapter,
)

from .ensemble import (
    StrategyEnsemble,
    VotingMechanism,
    WeightOptimizer,
    StrategyMixer,
)

from .backtesting import (
    BacktestEngine,
    BacktestResult,
    PerformanceAnalyzer,
    WalkForwardValidator,
)

__all__ = [
    # Interfaces
    'BaseStrategyInterface',
    'StrategyComponent',
    'IndicatorSet',
    'SignalGenerator',
    'RiskManager',
    'PositionSizer',
    'OrderExecutor',
    'PerformanceTracker',
    
    # Core
    'AdaptiveStrategy',
    'StrategyLearner',
    'ParameterOptimizer',
    'MarketAdapter',
    
    # Ensemble
    'StrategyEnsemble',
    'VotingMechanism',
    'WeightOptimizer',
    'StrategyMixer',
    
    # Backtesting
    'BacktestEngine',
    'BacktestResult',
    'PerformanceAnalyzer',
    'WalkForwardValidator',
]