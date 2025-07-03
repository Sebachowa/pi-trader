# Self-Improving Trading Strategy Framework

A comprehensive framework for building adaptive, self-improving trading strategies with machine learning capabilities and automated optimization.

## Features

### Core Capabilities
- **Flexible Component Architecture**: Pluggable components for indicators, signals, risk management, position sizing, and order execution
- **Self-Learning**: Strategies that learn from their performance and adapt parameters automatically
- **Market Regime Adaptation**: Dynamic adjustment to changing market conditions
- **Ensemble Support**: Combine multiple strategies with various voting mechanisms
- **Comprehensive Backtesting**: High-performance backtesting with walk-forward analysis and Monte Carlo simulation

### Key Components

#### 1. Base Strategy Interface (`interfaces.py`)
- **StrategyComponent**: Base class for all pluggable components
- **IndicatorSet**: Manage technical indicators
- **SignalGenerator**: Generate and filter trading signals
- **RiskManager**: Adaptive risk management with position and daily limits
- **PositionSizer**: Multiple sizing methods (Kelly, volatility-based, risk parity)
- **OrderExecutor**: Smart order execution with multiple styles
- **PerformanceTracker**: Comprehensive performance metrics tracking

#### 2. Adaptive Strategy Core (`core.py`)
- **AdaptiveStrategy**: Self-improving strategy with automatic parameter optimization
- **StrategyLearner**: ML-based pattern discovery and performance prediction
- **ParameterOptimizer**: Bayesian and genetic algorithm optimization
- **MarketAdapter**: Market regime detection and adaptation

#### 3. Ensemble Capabilities (`ensemble.py`)
- **StrategyEnsemble**: Combine multiple strategies
- **VotingMechanism**: Various voting methods (majority, weighted, ML-based, adaptive)
- **WeightOptimizer**: Dynamic weight optimization (Sharpe, risk parity, minimum variance)
- **StrategyMixer**: Advanced strategy blending with smooth transitions

#### 4. Backtesting Engine (`backtesting.py`)
- **BacktestEngine**: High-performance backtesting with vectorized calculations
- **WalkForwardValidator**: Robust walk-forward analysis
- **PerformanceAnalyzer**: Comprehensive metrics calculation
- Monte Carlo simulation support
- Parallel parameter optimization

## Quick Start

### Basic Usage

```python
from autonomous_trading.strategies.framework import AdaptiveMomentumStrategy
from autonomous_trading.strategies.framework.backtesting import BacktestEngine, BacktestConfig
import pandas as pd
from datetime import datetime

# Create strategy
strategy = AdaptiveMomentumStrategy(
    strategy_id="momentum_001",
    config={
        'adaptation_enabled': True,
        'indicators': {
            'ema_fast': 12,
            'ema_slow': 26
        },
        'risk': {
            'max_position_risk': 0.02
        }
    }
)

# Initialize
await strategy.initialize()

# Create backtest engine
config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    initial_capital=100000
)

engine = BacktestEngine(config)

# Run backtest
data = pd.read_csv('historical_data.csv')  # Your data
result = await engine.run_backtest(strategy, data)

print(f"Sharpe Ratio: {result.performance_metrics['sharpe_ratio']}")
print(f"Total Return: {result.performance_metrics['total_return']:.2%}")
```

### Creating Custom Strategy

```python
from autonomous_trading.strategies.framework import AdaptiveStrategy

class MyCustomStrategy(AdaptiveStrategy):
    def __init__(self, strategy_id: str, config: Dict[str, Any] = None):
        super().__init__(strategy_id, config)
        self._setup_custom_indicators()
        self._setup_custom_rules()
    
    def _setup_custom_indicators(self):
        # Add your indicators
        self.indicators.add_indicator('custom', MyCustomIndicator())
    
    def _setup_custom_rules(self):
        # Define signal generation rules
        async def my_signal_rule(data):
            if condition_met:
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    confidence=0.7,
                    timestamp=datetime.utcnow(),
                    source='custom_rule'
                )
            return None
        
        self.signal_generator.add_rule(my_signal_rule, weight=1.0)
```

### Building Ensemble Strategy

```python
from autonomous_trading.strategies.framework import StrategyEnsemble, VotingMethod

# Create ensemble
ensemble = StrategyEnsemble(
    "multi_strategy",
    config={
        'voting_method': VotingMethod.WEIGHTED,
        'min_agreement': 0.6,
        'optimize_frequency': 100
    }
)

# Add strategies
ensemble.add_strategy(AdaptiveMomentumStrategy("momentum"), weight=0.4)
ensemble.add_strategy(MeanReversionStrategy("mean_rev"), weight=0.3)
ensemble.add_strategy(MarketMakingStrategy("mm"), weight=0.3)

# Initialize and run
await ensemble.initialize()
signal = await ensemble.on_data(market_data)
```

### Parameter Optimization

```python
# Define parameter grid
param_grid = {
    'indicators': {
        'ema_fast': [10, 12, 15, 20],
        'ema_slow': [20, 26, 30, 40]
    },
    'signals': {
        'min_confidence': [0.5, 0.6, 0.7, 0.8]
    }
}

# Run optimization
optimization_result = await engine.optimize_parameters(
    AdaptiveMomentumStrategy,
    data,
    param_grid,
    metric='sharpe_ratio'
)

print(f"Best parameters: {optimization_result['best_params']}")
print(f"Best Sharpe: {optimization_result['best_score']}")
```

### Walk-Forward Analysis

```python
from autonomous_trading.strategies.framework.backtesting import WalkForwardValidator

validator = WalkForwardValidator(engine)

# Run walk-forward validation
results = await validator.validate(
    AdaptiveMomentumStrategy,
    data,
    optimization_params=param_grid,
    train_periods=252,  # 1 year
    test_periods=63,    # 3 months
    optimization_metric='sharpe_ratio'
)

print(f"Average out-of-sample Sharpe: {results['summary']['avg_out_sample']}")
print(f"Performance degradation: {results['summary']['avg_degradation']:.2%}")
```

## Advanced Features

### Self-Learning Capabilities

The framework includes automatic learning features:

1. **Pattern Discovery**: Automatically discovers new trading patterns from historical data
2. **Parameter Adaptation**: Adjusts parameters based on recent performance
3. **Market Regime Detection**: Adapts behavior to different market conditions
4. **Performance Prediction**: ML models predict strategy performance

### Risk Management

Advanced risk management features:

- Position-level risk limits
- Daily loss limits
- Dynamic stop-loss adjustment
- Volatility-based position sizing
- Kelly Criterion implementation
- Risk parity allocation

### Execution Styles

Multiple order execution styles:

- **Aggressive**: Market orders with slippage modeling
- **Passive**: Limit orders for better fills
- **Smart**: Dynamic routing based on market conditions
- **Iceberg**: Split large orders into smaller chunks

## Configuration Options

### Strategy Configuration

```python
config = {
    # Indicators
    'indicators': {
        'ema_fast': 12,
        'ema_slow': 26,
        'rsi_period': 14
    },
    
    # Signals
    'signals': {
        'min_confidence': 0.6,
        'signal_threshold': 0.02
    },
    
    # Risk Management
    'risk': {
        'max_position_risk': 0.02,
        'max_daily_risk': 0.06,
        'max_positions': 5
    },
    
    # Position Sizing
    'sizing': {
        'method': 'kelly',  # 'fixed', 'volatility', 'kelly', 'risk_parity'
        'kelly_fraction': 0.25
    },
    
    # Order Execution
    'execution': {
        'style': 'smart',  # 'aggressive', 'passive', 'smart'
        'use_limit_orders': True
    },
    
    # Adaptation
    'adaptation_enabled': True,
    'adaptation_frequency': 100,
    'min_samples_to_adapt': 50
}
```

### Backtest Configuration

```python
backtest_config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    initial_capital=100000,
    commission=0.001,          # 0.1%
    slippage=0.0005,          # 0.05%
    data_frequency='1h',       # '1m', '5m', '1h', '1d'
    benchmark='SPY',           # Optional benchmark
    risk_free_rate=0.02,       # 2% annual
    max_positions=10,
    position_sizing='equal',   # 'equal', 'kelly', 'risk_parity'
    rebalance_frequency='daily',
    transaction_costs=True,
    use_stops=True,
    stop_loss=0.02,           # 2%
    take_profit=0.05,         # 5%
    enable_shorting=True,
    parallel_execution=True,
    n_workers=4
)
```

## Performance Metrics

The framework calculates comprehensive performance metrics:

- **Returns**: Total, annual, monthly, daily
- **Risk**: Volatility, max drawdown, VaR, CVaR
- **Risk-Adjusted**: Sharpe, Sortino, Calmar ratios
- **Trade Analysis**: Win rate, profit factor, expectancy
- **Consistency**: Rolling returns, recovery factor

## Best Practices

1. **Start Simple**: Begin with basic strategies and gradually add complexity
2. **Validate Thoroughly**: Use walk-forward analysis to avoid overfitting
3. **Monitor Adaptation**: Ensure learning improves performance
4. **Diversify**: Use ensemble strategies for robustness
5. **Risk First**: Always prioritize risk management
6. **Test Realistically**: Include transaction costs and slippage

## Integration with Nautilus Trader

The framework is designed to work seamlessly with Nautilus Trader:

```python
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId

# Process Nautilus data
async def on_bar(self, bar: Bar):
    signal = await self.strategy.on_data(bar)
    if signal:
        # Execute through Nautilus
        self.submit_order(...)
```

## Future Enhancements

Planned features:
- Deep learning signal generation
- Reinforcement learning optimization
- Multi-asset portfolio optimization
- Real-time performance monitoring
- Cloud-based backtesting
- Strategy marketplace integration