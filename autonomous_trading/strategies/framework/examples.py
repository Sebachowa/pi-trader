"""
Example strategy implementations using the self-improving framework.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional

from nautilus_trader.indicators.ema import ExponentialMovingAverage
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.bollinger_bands import BollingerBands

from .interfaces import (
    TradingSignal,
    SignalType,
    OrderExecutor,
)
from .core import AdaptiveStrategy


class AdaptiveMomentumStrategy(AdaptiveStrategy):
    """
    Example adaptive momentum strategy that learns and improves.
    
    Features:
    - Dynamic parameter adjustment
    - Market regime adaptation
    - Self-learning signal generation
    """
    
    def __init__(self, strategy_id: str = "adaptive_momentum", config: Dict[str, Any] = None):
        default_config = {
            'indicators': {
                'ema_fast': 12,
                'ema_slow': 26,
                'rsi_period': 14,
                'atr_period': 14,
            },
            'signals': {
                'min_confidence': 0.6,
                'momentum_threshold': 0.02,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            },
            'risk': {
                'max_position_risk': 0.02,
                'max_daily_risk': 0.06,
                'stop_loss_atr': 2.0,
            },
            'adaptation_enabled': True,
            'adaptation_frequency': 100,
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(strategy_id, default_config)
        
        # Initialize indicators
        self._setup_indicators()
        
        # Initialize signal rules
        self._setup_signal_rules()
    
    def _setup_indicators(self) -> None:
        """Setup technical indicators."""
        # Moving averages
        self.indicators.add_indicator(
            'ema_fast',
            ExponentialMovingAverage(self.config['indicators']['ema_fast'])
        )
        self.indicators.add_indicator(
            'ema_slow',
            ExponentialMovingAverage(self.config['indicators']['ema_slow'])
        )
        
        # RSI
        self.indicators.add_indicator(
            'rsi',
            RelativeStrengthIndex(self.config['indicators']['rsi_period'])
        )
        
        # ATR for volatility
        self.indicators.add_indicator(
            'atr',
            AverageTrueRange(self.config['indicators']['atr_period'])
        )
        
        # Bollinger Bands
        self.indicators.add_indicator(
            'bb',
            BollingerBands(20, 2.0)
        )
    
    def _setup_signal_rules(self) -> None:
        """Setup signal generation rules."""
        # Momentum crossover rule
        async def momentum_crossover(data: Dict[str, Any]) -> Optional[TradingSignal]:
            indicators = data.get('indicators', {})
            
            if 'ema_fast' not in indicators or 'ema_slow' not in indicators:
                return None
            
            fast = indicators['ema_fast']
            slow = indicators['ema_slow']
            
            if fast > slow * (1 + self.config['signals']['momentum_threshold']):
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=min(1.0, (fast - slow) / slow * 10),
                    confidence=0.7,
                    timestamp=datetime.utcnow(),
                    source='momentum_crossover',
                    metadata={'fast': fast, 'slow': slow}
                )
            elif fast < slow * (1 - self.config['signals']['momentum_threshold']):
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=min(1.0, (slow - fast) / fast * 10),
                    confidence=0.7,
                    timestamp=datetime.utcnow(),
                    source='momentum_crossover',
                    metadata={'fast': fast, 'slow': slow}
                )
            
            return None
        
        # RSI reversal rule
        async def rsi_reversal(data: Dict[str, Any]) -> Optional[TradingSignal]:
            indicators = data.get('indicators', {})
            rsi = indicators.get('rsi')
            
            if rsi is None:
                return None
            
            if rsi < self.config['signals']['rsi_oversold']:
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=(self.config['signals']['rsi_oversold'] - rsi) / self.config['signals']['rsi_oversold'],
                    confidence=0.6,
                    timestamp=datetime.utcnow(),
                    source='rsi_reversal',
                    metadata={'rsi': rsi}
                )
            elif rsi > self.config['signals']['rsi_overbought']:
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=(rsi - self.config['signals']['rsi_overbought']) / (100 - self.config['signals']['rsi_overbought']),
                    confidence=0.6,
                    timestamp=datetime.utcnow(),
                    source='rsi_reversal',
                    metadata={'rsi': rsi}
                )
            
            return None
        
        # Bollinger band bounce rule
        async def bb_bounce(data: Dict[str, Any]) -> Optional[TradingSignal]:
            indicators = data.get('indicators', {})
            bb = indicators.get('bb')
            price = data.get('market_data', {}).get('price')
            
            if not bb or not price:
                return None
            
            upper = bb.upper
            lower = bb.lower
            middle = bb.middle
            
            if price <= lower:
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=min(1.0, (lower - price) / lower),
                    confidence=0.65,
                    timestamp=datetime.utcnow(),
                    source='bb_bounce',
                    metadata={'price': price, 'lower_band': lower}
                )
            elif price >= upper:
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=min(1.0, (price - upper) / upper),
                    confidence=0.65,
                    timestamp=datetime.utcnow(),
                    source='bb_bounce',
                    metadata={'price': price, 'upper_band': upper}
                )
            
            return None
        
        # Add rules to signal generator
        self.signal_generator.add_rule(momentum_crossover, weight=1.0)
        self.signal_generator.add_rule(rsi_reversal, weight=0.8)
        self.signal_generator.add_rule(bb_bounce, weight=0.7)
        
        # Add filters
        async def trend_filter(signal: TradingSignal, data: Dict[str, Any]) -> bool:
            """Filter signals against the trend in volatile markets."""
            regime = data.get('regime')
            if regime == 'volatile':
                # In volatile markets, only take signals in trend direction
                indicators = data.get('indicators', {})
                fast = indicators.get('ema_fast', 0)
                slow = indicators.get('ema_slow', 0)
                
                if signal.signal_type == SignalType.BUY and fast < slow:
                    return False
                elif signal.signal_type == SignalType.SELL and fast > slow:
                    return False
            
            return True
        
        self.signal_generator.add_filter(trend_filter)


class MeanReversionStrategy(AdaptiveStrategy):
    """
    Example mean reversion strategy with adaptive features.
    """
    
    def __init__(self, strategy_id: str = "mean_reversion", config: Dict[str, Any] = None):
        default_config = {
            'indicators': {
                'lookback_period': 20,
                'zscore_threshold': 2.0,
                'half_life': 10,
            },
            'signals': {
                'min_confidence': 0.65,
                'entry_zscore': 2.0,
                'exit_zscore': 0.5,
            },
            'risk': {
                'max_position_risk': 0.015,
                'position_timeout': 48,  # hours
            }
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(strategy_id, default_config)
        
        self._setup_mean_reversion_indicators()
        self._setup_mean_reversion_rules()
    
    def _setup_mean_reversion_indicators(self) -> None:
        """Setup mean reversion specific indicators."""
        # Price rolling statistics
        class RollingZScore:
            def __init__(self, period: int):
                self.period = period
                self.prices = []
                self.value = 0
            
            def update_raw(self, high: float, low: float, close: float, volume: float):
                self.prices.append(close)
                if len(self.prices) > self.period:
                    self.prices.pop(0)
                
                if len(self.prices) >= 2:
                    mean = np.mean(self.prices)
                    std = np.std(self.prices)
                    if std > 0:
                        self.value = (self.prices[-1] - mean) / std
                    else:
                        self.value = 0
            
            def reset(self):
                self.prices = []
                self.value = 0
        
        self.indicators.add_indicator(
            'zscore',
            RollingZScore(self.config['indicators']['lookback_period'])
        )
    
    def _setup_mean_reversion_rules(self) -> None:
        """Setup mean reversion signal rules."""
        async def zscore_reversion(data: Dict[str, Any]) -> Optional[TradingSignal]:
            indicators = data.get('indicators', {})
            zscore = indicators.get('zscore', 0)
            
            entry_threshold = self.config['signals']['entry_zscore']
            
            if zscore < -entry_threshold:
                # Oversold - buy signal
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=min(1.0, abs(zscore) / 3),
                    confidence=0.7 + min(0.2, abs(zscore) / 10),
                    timestamp=datetime.utcnow(),
                    source='zscore_reversion',
                    metadata={'zscore': zscore}
                )
            elif zscore > entry_threshold:
                # Overbought - sell signal
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=min(1.0, abs(zscore) / 3),
                    confidence=0.7 + min(0.2, abs(zscore) / 10),
                    timestamp=datetime.utcnow(),
                    source='zscore_reversion',
                    metadata={'zscore': zscore}
                )
            elif abs(zscore) < self.config['signals']['exit_zscore']:
                # Close to mean - exit signal
                return TradingSignal(
                    signal_type=SignalType.CLOSE_LONG if self.state.position_count > 0 else SignalType.HOLD,
                    strength=0.8,
                    confidence=0.8,
                    timestamp=datetime.utcnow(),
                    source='zscore_exit',
                    metadata={'zscore': zscore}
                )
            
            return None
        
        self.signal_generator.add_rule(zscore_reversion, weight=1.0)


class MarketMakingStrategy(AdaptiveStrategy):
    """
    Example market making strategy with adaptive spread adjustment.
    """
    
    def __init__(self, strategy_id: str = "market_making", config: Dict[str, Any] = None):
        default_config = {
            'indicators': {
                'volatility_period': 20,
                'volume_period': 20,
            },
            'signals': {
                'base_spread': 0.002,  # 20 bps
                'min_spread': 0.001,   # 10 bps
                'max_spread': 0.005,   # 50 bps
                'inventory_limit': 10,
                'skew_factor': 0.5,
            },
            'execution': {
                'style': 'passive',
                'use_limit_orders': True,
            }
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(strategy_id, default_config)
        
        self.inventory = 0
        self.target_inventory = 0
        
        self._setup_market_making_rules()
    
    def _setup_market_making_rules(self) -> None:
        """Setup market making signal rules."""
        async def quote_generation(data: Dict[str, Any]) -> Optional[TradingSignal]:
            """Generate two-sided quotes."""
            market_data = data.get('market_data', {})
            indicators = data.get('indicators', {})
            
            mid_price = market_data.get('price', 0)
            volatility = indicators.get('atr', 0.01) / mid_price if mid_price > 0 else 0.01
            
            # Adjust spread based on volatility
            spread = self.config['signals']['base_spread']
            spread = max(self.config['signals']['min_spread'], 
                        min(self.config['signals']['max_spread'], 
                            spread * (1 + volatility * 10)))
            
            # Inventory skew
            inventory_ratio = self.inventory / self.config['signals']['inventory_limit']
            skew = inventory_ratio * self.config['signals']['skew_factor'] * spread
            
            # Generate signal based on inventory
            if self.inventory < self.config['signals']['inventory_limit']:
                # Can buy
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=0.5,
                    confidence=0.8,
                    timestamp=datetime.utcnow(),
                    source='market_making',
                    metadata={
                        'bid_price': mid_price * (1 - spread/2 - skew),
                        'ask_price': mid_price * (1 + spread/2 - skew),
                        'spread': spread,
                        'inventory': self.inventory
                    }
                )
            elif self.inventory > -self.config['signals']['inventory_limit']:
                # Can sell
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=0.5,
                    confidence=0.8,
                    timestamp=datetime.utcnow(),
                    source='market_making',
                    metadata={
                        'bid_price': mid_price * (1 - spread/2 - skew),
                        'ask_price': mid_price * (1 + spread/2 - skew),
                        'spread': spread,
                        'inventory': self.inventory
                    }
                )
            
            return None
        
        self.signal_generator.add_rule(quote_generation, weight=1.0)
    
    async def on_fill(self, order: Dict, fill: Any) -> None:
        """Update inventory on fills."""
        await super().on_fill(order, fill)
        
        # Update inventory
        if order['side'] == 'BUY':
            self.inventory += order['quantity']
        else:
            self.inventory -= order['quantity']


class EnsembleExampleStrategy(AdaptiveStrategy):
    """
    Example of how to create an ensemble strategy using the framework.
    """
    
    def __init__(self, strategy_id: str = "ensemble_example", config: Dict[str, Any] = None):
        super().__init__(strategy_id, config)
        
        # Create sub-strategies
        self.momentum_strategy = AdaptiveMomentumStrategy("ensemble_momentum")
        self.mean_reversion_strategy = MeanReversionStrategy("ensemble_mean_rev")
        
        # Create ensemble
        from .ensemble import StrategyEnsemble, VotingMethod
        
        self.ensemble = StrategyEnsemble(
            "main_ensemble",
            {
                'voting_method': VotingMethod.ADAPTIVE,
                'min_agreement': 0.6,
                'optimize_frequency': 50
            }
        )
        
        # Add strategies to ensemble
        self.ensemble.add_strategy(self.momentum_strategy, weight=0.6)
        self.ensemble.add_strategy(self.mean_reversion_strategy, weight=0.4)
    
    async def initialize(self) -> None:
        """Initialize ensemble."""
        await super().initialize()
        await self.ensemble.initialize()
    
    async def on_data(self, data: Any) -> Optional[TradingSignal]:
        """Process data through ensemble."""
        # Let ensemble handle signal generation
        signal = await self.ensemble.on_data(data)
        
        if signal:
            # Record signal for learning
            self._record_decision({
                'signal': signal,
                'ensemble_weights': self.ensemble.strategy_weights.copy(),
                'timestamp': datetime.utcnow()
            })
        
        return signal


class StrategyFactory:
    """
    Factory for creating pre-configured strategies.
    """
    
    STRATEGY_REGISTRY = {
        'adaptive_momentum': AdaptiveMomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'market_making': MarketMakingStrategy,
        'ensemble': EnsembleExampleStrategy,
    }
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any] = None) -> AdaptiveStrategy:
        """Create a strategy instance."""
        if strategy_type not in StrategyFactory.STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = StrategyFactory.STRATEGY_REGISTRY[strategy_type]
        return strategy_class(config=config)
    
    @staticmethod
    def create_optimized_strategy(strategy_type: str, 
                                 market_conditions: Dict[str, Any]) -> AdaptiveStrategy:
        """Create strategy with market-optimized parameters."""
        base_config = StrategyFactory._get_market_optimized_config(
            strategy_type, 
            market_conditions
        )
        
        return StrategyFactory.create_strategy(strategy_type, base_config)
    
    @staticmethod
    def _get_market_optimized_config(strategy_type: str, 
                                    market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized configuration based on market conditions."""
        volatility = market_conditions.get('volatility', 'medium')
        trend = market_conditions.get('trend', 'neutral')
        liquidity = market_conditions.get('liquidity', 'normal')
        
        config = {}
        
        if strategy_type == 'adaptive_momentum':
            if volatility == 'high':
                config['signals'] = {
                    'min_confidence': 0.7,  # Higher confidence in volatile markets
                    'momentum_threshold': 0.03,  # Wider threshold
                }
                config['risk'] = {
                    'max_position_risk': 0.015,  # Lower risk
                    'stop_loss_atr': 2.5,  # Wider stops
                }
            elif volatility == 'low':
                config['signals'] = {
                    'min_confidence': 0.55,
                    'momentum_threshold': 0.015,
                }
                config['risk'] = {
                    'max_position_risk': 0.025,
                    'stop_loss_atr': 1.5,
                }
        
        elif strategy_type == 'mean_reversion':
            if trend == 'strong':
                # Mean reversion works poorly in strong trends
                config['signals'] = {
                    'min_confidence': 0.75,  # Higher confidence required
                    'entry_zscore': 2.5,  # More extreme levels
                }
            else:
                config['signals'] = {
                    'min_confidence': 0.6,
                    'entry_zscore': 1.8,
                }
        
        elif strategy_type == 'market_making':
            if liquidity == 'low':
                config['signals'] = {
                    'base_spread': 0.003,  # Wider spreads
                    'inventory_limit': 5,  # Lower inventory
                }
            elif liquidity == 'high':
                config['signals'] = {
                    'base_spread': 0.0015,  # Tighter spreads
                    'inventory_limit': 15,  # Higher inventory
                }
        
        return config


# Usage example
async def example_usage():
    """Example of how to use the framework."""
    # Create an adaptive momentum strategy
    strategy = AdaptiveMomentumStrategy(
        strategy_id="momentum_001",
        config={
            'adaptation_enabled': True,
            'adaptation_frequency': 50,
            'risk': {
                'max_position_risk': 0.02,
                'max_daily_risk': 0.06
            }
        }
    )
    
    # Initialize
    await strategy.initialize()
    
    # Create market data (example)
    from nautilus_trader.model.data import Bar
    
    # Process data and get signals
    # signal = await strategy.on_data(bar_data)
    
    # Create ensemble
    from .ensemble import StrategyEnsemble, VotingMethod
    
    ensemble = StrategyEnsemble(
        "multi_strategy",
        config={
            'voting_method': VotingMethod.WEIGHTED,
            'min_agreement': 0.6
        }
    )
    
    # Add multiple strategies
    ensemble.add_strategy(
        AdaptiveMomentumStrategy("momentum_01"),
        weight=0.4
    )
    ensemble.add_strategy(
        MeanReversionStrategy("mean_rev_01"),
        weight=0.3
    )
    ensemble.add_strategy(
        MarketMakingStrategy("mm_01"),
        weight=0.3
    )
    
    # Initialize ensemble
    await ensemble.initialize()
    
    # Process data through ensemble
    # ensemble_signal = await ensemble.on_data(bar_data)
    
    # Run backtest
    from .backtesting import BacktestEngine, BacktestConfig
    
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    engine = BacktestEngine(config)
    
    # Load historical data
    # data = pd.DataFrame(...)  # Your historical data
    
    # Run backtest
    # result = await engine.run_backtest(strategy, data)
    
    # Run walk-forward analysis
    # wf_results = await engine.run_walk_forward_analysis(
    #     strategy, 
    #     data,
    #     train_period=252,
    #     test_period=63
    # )
    
    # Optimize parameters
    # param_grid = {
    #     'indicators': {
    #         'ema_fast': [10, 12, 15],
    #         'ema_slow': [20, 26, 30]
    #     }
    # }
    # 
    # optimization_result = await engine.optimize_parameters(
    #     AdaptiveMomentumStrategy,
    #     data,
    #     param_grid,
    #     metric='sharpe_ratio'
    # )