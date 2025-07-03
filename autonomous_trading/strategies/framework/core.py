"""
Core components for self-improving trading strategies.
"""

import asyncio
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna
from scipy.optimize import minimize

from .interfaces import (
    BaseStrategyInterface,
    StrategyComponent,
    TradingSignal,
    SignalType,
    StrategyState,
    IndicatorSet,
    SignalGenerator,
    RiskManager,
    PositionSizer,
    OrderExecutor,
    PerformanceTracker,
)


class AdaptiveStrategy(BaseStrategyInterface):
    """
    Self-adapting trading strategy that learns and improves over time.
    
    Features:
    - Dynamic parameter adjustment
    - Market regime adaptation
    - Performance-based learning
    - Component hot-swapping
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any] = None):
        super().__init__(strategy_id, config)
        
        # Adaptive components
        self.adaptation_enabled = config.get('adaptation_enabled', True) if config else True
        self.adaptation_frequency = config.get('adaptation_frequency', 100) if config else 100
        self.min_samples_to_adapt = config.get('min_samples_to_adapt', 50) if config else 50
        
        # Market regime tracking
        self.market_regimes = deque(maxlen=1000)
        self.regime_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'win_rate': 0})
        
        # Component versions
        self.component_versions = {}
        self.component_performance = defaultdict(list)
        
        # Adaptation history
        self.adaptation_history = deque(maxlen=100)
        self.data_buffer = deque(maxlen=10000)
        
        # Initialize learner
        self.learner = StrategyLearner(self)
        self.parameter_optimizer = ParameterOptimizer(self)
        self.market_adapter = MarketAdapter(self)
        
        # Counters
        self.tick_count = 0
        self.adaptation_count = 0
    
    async def initialize(self) -> None:
        """Initialize strategy and all adaptive components."""
        # Initialize base components
        await self.indicators.initialize()
        await self.signal_generator.initialize()
        await self.risk_manager.initialize()
        await self.position_sizer.initialize()
        await self.order_executor.initialize()
        await self.performance_tracker.initialize()
        
        # Initialize adaptive components
        await self.learner.initialize()
        await self.parameter_optimizer.initialize()
        await self.market_adapter.initialize()
        
        # Load any saved state
        await self._load_adaptive_state()
        
        self.state.is_active = True
    
    async def on_data(self, data: Any) -> Optional[TradingSignal]:
        """Process market data with adaptive behavior."""
        self.tick_count += 1
        
        # Buffer data for learning
        self.data_buffer.append({
            'timestamp': datetime.utcnow(),
            'data': data,
            'market_state': await self._extract_market_state(data)
        })
        
        # Update indicators
        indicator_values = await self.indicators.update(data)
        
        # Detect market regime
        current_regime = await self.market_adapter.detect_regime(data, indicator_values)
        self.market_regimes.append(current_regime)
        
        # Generate signal with regime context
        signal_data = {
            'market_data': data,
            'indicators': indicator_values,
            'regime': current_regime,
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }
        
        signal = await self.signal_generator.update(signal_data)
        
        # Adapt if needed
        if self.adaptation_enabled and self.tick_count % self.adaptation_frequency == 0:
            if len(self.data_buffer) >= self.min_samples_to_adapt:
                await self._adapt_components()
        
        return signal
    
    async def on_signal(self, signal: TradingSignal) -> Optional[Dict]:
        """Process signal with adaptive risk and position sizing."""
        # Check risk limits
        if not self.risk_manager.can_take_trade(signal.metadata.get('risk', 0.01)):
            return None
        
        # Calculate position size adaptively
        sizing_data = {
            'signal': signal,
            'performance': self.performance_tracker.get_performance_summary(),
            'volatility': signal.metadata.get('volatility', 0.01),
            'regime': self.market_regimes[-1] if self.market_regimes else None
        }
        
        position_size = await self.position_sizer.update(sizing_data)
        
        # Execute order
        market_data = {
            'mid_price': signal.metadata.get('price', 0),
            'bid': signal.metadata.get('bid', 0),
            'ask': signal.metadata.get('ask', 0),
            'spread': signal.metadata.get('spread', 0.001),
            'liquidity': signal.metadata.get('liquidity', 'medium'),
            'position_size': position_size
        }
        
        order = await self.order_executor.execute_signal(signal, market_data)
        
        # Record for learning
        self._record_decision({
            'signal': signal,
            'position_size': position_size,
            'order': order,
            'regime': self.market_regimes[-1] if self.market_regimes else None,
            'timestamp': datetime.utcnow()
        })
        
        return order
    
    async def on_fill(self, order: Dict, fill: Any) -> None:
        """Handle order fills and update learning data."""
        # Update state
        if order['side'] == 'BUY':
            self.state.position_count += 1
        else:
            self.state.position_count = max(0, self.state.position_count - 1)
        
        self.state.total_trades += 1
        
        # Update performance tracker
        trade_data = {
            'trade': {
                'order': order,
                'fill': fill,
                'return': 0,  # Will be calculated when position closes
                'duration_seconds': 0
            }
        }
        
        await self.performance_tracker.update(trade_data)
        
        # Learn from execution
        await self.learner.learn_from_execution(order, fill)
    
    async def _adapt_components(self) -> None:
        """Adapt strategy components based on performance."""
        self.adaptation_count += 1
        
        adaptation_results = {
            'timestamp': datetime.utcnow(),
            'changes': []
        }
        
        # Get current performance
        performance = self.performance_tracker.get_performance_summary()
        
        # Adapt signal generator
        if performance.get('win_rate', 0) < 0.45:
            # Poor win rate - adjust signal generation
            self.signal_generator.min_confidence *= 1.1  # Require higher confidence
            adaptation_results['changes'].append('increased_signal_confidence_threshold')
        
        # Adapt risk manager
        self.risk_manager.adapt_parameters(performance)
        
        # Optimize parameters
        if self.adaptation_count % 5 == 0:  # Every 5th adaptation
            optimized_params = await self.parameter_optimizer.optimize_parameters(
                self.data_buffer,
                performance
            )
            
            if optimized_params:
                await self._apply_optimized_parameters(optimized_params)
                adaptation_results['changes'].append('optimized_parameters')
        
        # Learn new patterns
        if len(self.data_buffer) >= 500:
            new_rules = await self.learner.discover_patterns(self.data_buffer)
            
            for rule in new_rules:
                self.signal_generator.add_rule(rule['function'], rule['weight'])
                adaptation_results['changes'].append(f"added_rule_{rule['name']}")
        
        self.adaptation_history.append(adaptation_results)
    
    async def _apply_optimized_parameters(self, params: Dict[str, Any]) -> None:
        """Apply optimized parameters to components."""
        # Update indicator parameters
        if 'indicators' in params:
            for name, value in params['indicators'].items():
                if name in self.indicators.indicators:
                    # Update indicator parameter (implementation specific)
                    pass
        
        # Update signal generator parameters
        if 'signals' in params:
            if 'min_confidence' in params['signals']:
                self.signal_generator.min_confidence = params['signals']['min_confidence']
        
        # Update risk parameters
        if 'risk' in params:
            if 'max_position_risk' in params['risk']:
                self.risk_manager.max_position_risk = params['risk']['max_position_risk']
            if 'max_daily_risk' in params['risk']:
                self.risk_manager.max_daily_risk = params['risk']['max_daily_risk']
    
    def _record_decision(self, decision: Dict[str, Any]) -> None:
        """Record trading decision for learning."""
        # Store decision with context for later analysis
        self.learner.record_decision(decision)
    
    async def _extract_market_state(self, data: Any) -> Dict[str, Any]:
        """Extract relevant market state features."""
        state = {}
        
        # Price features
        if hasattr(data, 'close'):
            state['price'] = data.close.as_double()
        
        # Volume features
        if hasattr(data, 'volume'):
            state['volume'] = data.volume.as_double()
        
        # Spread and liquidity
        if hasattr(data, 'bid') and hasattr(data, 'ask'):
            state['spread'] = (data.ask.as_double() - data.bid.as_double()) / data.bid.as_double()
        
        return state
    
    async def _load_adaptive_state(self) -> None:
        """Load saved adaptive state if available."""
        try:
            # Load from file or database
            # Placeholder implementation
            pass
        except:
            pass
    
    async def save_adaptive_state(self) -> None:
        """Save current adaptive state."""
        state = {
            'adaptation_count': self.adaptation_count,
            'regime_performance': dict(self.regime_performance),
            'component_versions': self.component_versions,
            'adaptation_history': list(self.adaptation_history)[-50:],
            'learner_state': await self.learner.get_state(),
            'optimizer_state': await self.parameter_optimizer.get_state()
        }
        
        # Save to file or database
        # Placeholder implementation
    
    def get_state(self) -> StrategyState:
        """Get current strategy state with adaptive metrics."""
        base_state = self.state
        
        # Add adaptive metrics
        base_state.metadata.update({
            'adaptation_count': self.adaptation_count,
            'tick_count': self.tick_count,
            'current_regime': self.market_regimes[-1] if self.market_regimes else None,
            'adaptation_enabled': self.adaptation_enabled,
            'last_adaptation': self.adaptation_history[-1] if self.adaptation_history else None
        })
        
        return base_state
    
    def adapt(self, market_conditions: Dict[str, Any]) -> None:
        """Manually trigger adaptation to market conditions."""
        # This is called externally when significant market changes are detected
        asyncio.create_task(self._adapt_components())


class StrategyLearner:
    """
    Machine learning component for strategy improvement.
    
    Features:
    - Pattern discovery
    - Performance prediction
    - Decision learning
    - Feature engineering
    """
    
    def __init__(self, strategy: AdaptiveStrategy):
        self.strategy = strategy
        self.decision_history = deque(maxlen=10000)
        self.pattern_library = []
        self.ml_models = {}
        self.feature_importance = {}
        self.learning_enabled = True
    
    async def initialize(self) -> None:
        """Initialize learning components."""
        # Initialize ML models
        self.ml_models['performance_predictor'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.ml_models['signal_classifier'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        self.feature_scaler = StandardScaler()
    
    def record_decision(self, decision: Dict[str, Any]) -> None:
        """Record trading decision for learning."""
        self.decision_history.append(decision)
    
    async def learn_from_execution(self, order: Dict, fill: Any) -> None:
        """Learn from order execution results."""
        # Find corresponding decision
        for decision in reversed(self.decision_history):
            if decision.get('order') == order:
                # Update with execution results
                decision['fill'] = fill
                decision['slippage'] = self._calculate_slippage(order, fill)
                break
    
    async def discover_patterns(self, data_buffer: deque) -> List[Dict[str, Any]]:
        """Discover new trading patterns from data."""
        if len(data_buffer) < 100:
            return []
        
        new_patterns = []
        
        # Extract features from buffer
        features, labels = self._prepare_learning_data(data_buffer)
        
        if len(features) < 50:
            return []
        
        # Train model to find patterns
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Train model
        self.ml_models['signal_classifier'].fit(X_train_scaled, y_train)
        
        # Evaluate
        score = self.ml_models['signal_classifier'].score(X_test_scaled, y_test)
        
        if score > 0.6:  # Decent predictive power
            # Extract feature importance
            feature_importance = self.ml_models['signal_classifier'].feature_importances_
            important_features = self._get_important_features(feature_importance)
            
            # Create new pattern rules based on important features
            for feature_set in self._generate_pattern_combinations(important_features):
                pattern_func = self._create_pattern_function(feature_set)
                
                new_patterns.append({
                    'name': f"learned_pattern_{len(self.pattern_library)}",
                    'function': pattern_func,
                    'weight': 0.5,  # Start with moderate weight
                    'score': score,
                    'features': feature_set
                })
        
        # Add to pattern library
        self.pattern_library.extend(new_patterns)
        
        return new_patterns
    
    def _prepare_learning_data(self, data_buffer: deque) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for machine learning."""
        features = []
        labels = []
        
        # Convert buffer to features and labels
        for i, item in enumerate(list(data_buffer)[:-1]):
            if i < 20:  # Need history for features
                continue
            
            # Extract features
            feature_vec = self._extract_features(data_buffer, i)
            features.append(feature_vec)
            
            # Create label (future return)
            future_return = self._calculate_future_return(data_buffer, i)
            labels.append(future_return)
        
        return np.array(features), np.array(labels)
    
    def _extract_features(self, data_buffer: deque, index: int) -> List[float]:
        """Extract features from data at given index."""
        features = []
        
        # Price features
        prices = [item['data'].close.as_double() for item in list(data_buffer)[max(0, index-20):index+1]
                 if hasattr(item['data'], 'close')]
        
        if prices:
            # Returns
            returns = np.diff(prices) / prices[:-1]
            features.extend([
                np.mean(returns) if len(returns) > 0 else 0,
                np.std(returns) if len(returns) > 1 else 0,
                returns[-1] if len(returns) > 0 else 0
            ])
            
            # Moving averages
            features.extend([
                np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1],
                np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
            ])
        else:
            features.extend([0] * 5)
        
        # Volume features
        volumes = [item['data'].volume.as_double() for item in list(data_buffer)[max(0, index-10):index+1]
                  if hasattr(item['data'], 'volume')]
        
        if volumes:
            features.extend([
                np.mean(volumes),
                volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1
            ])
        else:
            features.extend([0, 1])
        
        # Market state features
        market_state = data_buffer[index]['market_state']
        features.extend([
            market_state.get('spread', 0),
            1 if market_state.get('volume', 0) > np.mean(volumes) else 0 if volumes else 0
        ])
        
        return features
    
    def _calculate_future_return(self, data_buffer: deque, index: int, horizon: int = 10) -> float:
        """Calculate future return for labeling."""
        if index + horizon >= len(data_buffer):
            return 0.0
        
        current_price = data_buffer[index]['data'].close.as_double() if hasattr(data_buffer[index]['data'], 'close') else 0
        future_price = data_buffer[index + horizon]['data'].close.as_double() if hasattr(data_buffer[index + horizon]['data'], 'close') else 0
        
        if current_price > 0:
            return (future_price - current_price) / current_price
        
        return 0.0
    
    def _get_important_features(self, feature_importance: np.ndarray, 
                               threshold: float = 0.1) -> List[int]:
        """Get indices of important features."""
        return [i for i, importance in enumerate(feature_importance) if importance > threshold]
    
    def _generate_pattern_combinations(self, feature_indices: List[int]) -> List[List[int]]:
        """Generate combinations of features for pattern creation."""
        combinations = []
        
        # Single features
        for idx in feature_indices:
            combinations.append([idx])
        
        # Pairs of features
        for i, idx1 in enumerate(feature_indices):
            for idx2 in feature_indices[i+1:]:
                combinations.append([idx1, idx2])
        
        return combinations[:5]  # Limit to top 5 combinations
    
    def _create_pattern_function(self, feature_indices: List[int]) -> Callable:
        """Create a pattern detection function based on feature indices."""
        async def pattern_function(data: Dict[str, Any]) -> Optional[TradingSignal]:
            # Extract features from current data
            # This is a simplified implementation
            indicators = data.get('indicators', {})
            
            # Check if pattern conditions are met
            # Placeholder logic - would use trained model
            signal_strength = np.random.random()  # Would be calculated from features
            
            if signal_strength > 0.6:
                return TradingSignal(
                    signal_type=SignalType.BUY if signal_strength > 0.7 else SignalType.HOLD,
                    strength=signal_strength,
                    confidence=0.7,
                    timestamp=datetime.utcnow(),
                    source="learned_pattern",
                    metadata={'features': feature_indices}
                )
            
            return None
        
        return pattern_function
    
    def _calculate_slippage(self, order: Dict, fill: Any) -> float:
        """Calculate execution slippage."""
        expected_price = order['metadata'].get('expected_price', 0)
        fill_price = fill.price if hasattr(fill, 'price') else 0
        
        if expected_price > 0:
            return abs(fill_price - expected_price) / expected_price
        
        return 0.0
    
    async def get_state(self) -> Dict[str, Any]:
        """Get learner state."""
        return {
            'decision_count': len(self.decision_history),
            'pattern_count': len(self.pattern_library),
            'models_trained': list(self.ml_models.keys()),
            'learning_enabled': self.learning_enabled
        }


class ParameterOptimizer:
    """
    Optimizes strategy parameters using various techniques.
    
    Features:
    - Bayesian optimization
    - Grid search
    - Random search
    - Genetic algorithms
    """
    
    def __init__(self, strategy: AdaptiveStrategy):
        self.strategy = strategy
        self.optimization_history = deque(maxlen=1000)
        self.best_parameters = {}
        self.parameter_bounds = {}
        self.optimization_method = 'bayesian'
    
    async def initialize(self) -> None:
        """Initialize optimizer."""
        # Define parameter bounds
        self.parameter_bounds = {
            'indicators': {
                'ema_fast_period': (5, 50),
                'ema_slow_period': (20, 200),
                'rsi_period': (7, 21),
                'atr_period': (10, 30)
            },
            'signals': {
                'min_confidence': (0.5, 0.9),
                'signal_threshold': (0.3, 0.8)
            },
            'risk': {
                'max_position_risk': (0.005, 0.03),
                'max_daily_risk': (0.02, 0.10),
                'stop_loss_atr': (1.0, 4.0)
            },
            'sizing': {
                'kelly_fraction': (0.1, 0.5),
                'max_position_pct': (0.05, 0.25)
            }
        }
    
    async def optimize_parameters(self, data_buffer: deque, 
                                 performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy parameters based on data and performance."""
        if self.optimization_method == 'bayesian':
            return await self._bayesian_optimization(data_buffer, performance)
        elif self.optimization_method == 'genetic':
            return await self._genetic_optimization(data_buffer, performance)
        else:
            return await self._grid_search(data_buffer, performance)
    
    async def _bayesian_optimization(self, data_buffer: deque, 
                                    performance: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Bayesian optimization of parameters."""
        
        def objective(trial):
            # Sample parameters
            params = {}
            
            # Indicator parameters
            params['indicators'] = {
                'ema_fast_period': trial.suggest_int('ema_fast', 5, 50),
                'ema_slow_period': trial.suggest_int('ema_slow', 20, 200)
            }
            
            # Signal parameters
            params['signals'] = {
                'min_confidence': trial.suggest_float('min_confidence', 0.5, 0.9)
            }
            
            # Risk parameters
            params['risk'] = {
                'max_position_risk': trial.suggest_float('max_pos_risk', 0.005, 0.03),
                'stop_loss_atr': trial.suggest_float('stop_loss_atr', 1.0, 4.0)
            }
            
            # Evaluate parameters (simplified - would run backtest)
            score = self._evaluate_parameters(params, data_buffer, performance)
            
            return score
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Get best parameters
        best_params = study.best_params
        self.best_parameters = self._format_parameters(best_params)
        
        # Record optimization
        self.optimization_history.append({
            'timestamp': datetime.utcnow(),
            'method': 'bayesian',
            'best_value': study.best_value,
            'best_params': self.best_parameters,
            'n_trials': 50
        })
        
        return self.best_parameters
    
    async def _genetic_optimization(self, data_buffer: deque, 
                                   performance: Dict[str, Any]) -> Dict[str, Any]:
        """Genetic algorithm optimization."""
        population_size = 50
        generations = 20
        mutation_rate = 0.1
        
        # Initialize population
        population = [self._random_parameters() for _ in range(population_size)]
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = self._evaluate_parameters(individual, data_buffer, performance)
                fitness_scores.append(score)
            
            # Select best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = int(population_size * 0.2)
            elite = [population[i] for i in sorted_indices[:elite_size]]
            
            # Create new population
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Get best individual
        final_scores = [self._evaluate_parameters(ind, data_buffer, performance) 
                       for ind in population]
        best_idx = np.argmax(final_scores)
        self.best_parameters = population[best_idx]
        
        return self.best_parameters
    
    def _random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within bounds."""
        params = {}
        
        for category, bounds in self.parameter_bounds.items():
            params[category] = {}
            for param, (low, high) in bounds.items():
                if isinstance(low, int):
                    params[category][param] = np.random.randint(low, high + 1)
                else:
                    params[category][param] = np.random.uniform(low, high)
        
        return params
    
    def _evaluate_parameters(self, params: Dict[str, Any], data_buffer: deque,
                           performance: Dict[str, Any]) -> float:
        """Evaluate parameter set (simplified - would run full backtest)."""
        # This is a placeholder - in reality would run backtest
        # For now, use a combination of heuristics
        
        score = 0.0
        
        # Prefer reasonable risk levels
        if 'risk' in params:
            risk_level = params['risk'].get('max_position_risk', 0.02)
            if 0.01 <= risk_level <= 0.02:
                score += 0.2
        
        # Prefer moderate confidence thresholds
        if 'signals' in params:
            confidence = params['signals'].get('min_confidence', 0.7)
            if 0.6 <= confidence <= 0.8:
                score += 0.2
        
        # Consider current performance
        current_sharpe = performance.get('sharpe_ratio', 0)
        if current_sharpe > 1.5:
            score += 0.3
        elif current_sharpe > 1.0:
            score += 0.2
        
        # Add some randomness to encourage exploration
        score += np.random.random() * 0.3
        
        return score
    
    def _tournament_selection(self, population: List[Dict], 
                            fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm."""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_scores)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two parameter sets."""
        child = {}
        
        for category in parent1:
            child[category] = {}
            for param in parent1[category]:
                # Random choice between parents
                if np.random.random() < 0.5:
                    child[category][param] = parent1[category][param]
                else:
                    child[category][param] = parent2[category][param]
        
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutate parameter set."""
        mutated = individual.copy()
        
        # Select random parameter to mutate
        category = np.random.choice(list(mutated.keys()))
        param = np.random.choice(list(mutated[category].keys()))
        
        # Mutate within bounds
        bounds = self.parameter_bounds[category][param]
        if isinstance(bounds[0], int):
            mutated[category][param] = np.random.randint(bounds[0], bounds[1] + 1)
        else:
            # Gaussian mutation
            std = (bounds[1] - bounds[0]) * 0.1
            new_value = mutated[category][param] + np.random.normal(0, std)
            mutated[category][param] = np.clip(new_value, bounds[0], bounds[1])
        
        return mutated
    
    def _format_parameters(self, raw_params: Dict) -> Dict[str, Any]:
        """Format raw parameters into structured format."""
        formatted = {
            'indicators': {},
            'signals': {},
            'risk': {}
        }
        
        for key, value in raw_params.items():
            if 'ema' in key:
                formatted['indicators'][key] = value
            elif 'confidence' in key or 'threshold' in key:
                formatted['signals'][key] = value
            elif 'risk' in key or 'stop_loss' in key:
                formatted['risk'][key] = value
        
        return formatted
    
    async def get_state(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            'optimization_count': len(self.optimization_history),
            'best_parameters': self.best_parameters,
            'optimization_method': self.optimization_method,
            'last_optimization': self.optimization_history[-1] if self.optimization_history else None
        }


class MarketAdapter:
    """
    Adapts strategy behavior to different market conditions.
    
    Features:
    - Market regime detection
    - Dynamic strategy switching
    - Condition-based parameter adjustment
    """
    
    def __init__(self, strategy: AdaptiveStrategy):
        self.strategy = strategy
        self.regime_models = {}
        self.regime_history = deque(maxlen=1000)
        self.regime_parameters = {}
        self.current_regime = None
    
    async def initialize(self) -> None:
        """Initialize market adapter."""
        # Define regime-specific parameters
        self.regime_parameters = {
            'trending_up': {
                'signal_confidence': 0.6,
                'position_size_multiplier': 1.2,
                'stop_loss_tightness': 0.8
            },
            'trending_down': {
                'signal_confidence': 0.7,
                'position_size_multiplier': 0.8,
                'stop_loss_tightness': 1.2
            },
            'ranging': {
                'signal_confidence': 0.75,
                'position_size_multiplier': 1.0,
                'stop_loss_tightness': 1.0
            },
            'volatile': {
                'signal_confidence': 0.8,
                'position_size_multiplier': 0.6,
                'stop_loss_tightness': 1.5
            }
        }
        
        # Initialize regime detection models
        self.regime_models['volatility'] = self._create_volatility_detector()
        self.regime_models['trend'] = self._create_trend_detector()
    
    async def detect_regime(self, market_data: Any, 
                          indicators: Dict[str, float]) -> str:
        """Detect current market regime."""
        features = self._extract_regime_features(market_data, indicators)
        
        # Simple regime detection logic
        volatility = features.get('volatility', 0.01)
        trend_strength = features.get('trend_strength', 0)
        
        if volatility > 0.03:
            regime = 'volatile'
        elif abs(trend_strength) > 0.5:
            regime = 'trending_up' if trend_strength > 0 else 'trending_down'
        else:
            regime = 'ranging'
        
        self.current_regime = regime
        self.regime_history.append({
            'timestamp': datetime.utcnow(),
            'regime': regime,
            'features': features
        })
        
        # Adapt strategy to regime
        await self._adapt_to_regime(regime)
        
        return regime
    
    async def _adapt_to_regime(self, regime: str) -> None:
        """Adapt strategy parameters to market regime."""
        if regime not in self.regime_parameters:
            return
        
        params = self.regime_parameters[regime]
        
        # Adjust signal confidence threshold
        self.strategy.signal_generator.min_confidence = params['signal_confidence']
        
        # Adjust position sizing
        if hasattr(self.strategy.position_sizer, 'base_size'):
            self.strategy.position_sizer.base_size *= params['position_size_multiplier']
        
        # Adjust risk parameters
        if hasattr(self.strategy.risk_manager, 'stop_loss_multiplier'):
            self.strategy.risk_manager.stop_loss_multiplier = params['stop_loss_tightness']
    
    def _extract_regime_features(self, market_data: Any, 
                               indicators: Dict[str, float]) -> Dict[str, float]:
        """Extract features for regime detection."""
        features = {}
        
        # Volatility features
        if 'atr' in indicators:
            features['volatility'] = indicators['atr'] / market_data.close.as_double() if hasattr(market_data, 'close') else 0.01
        
        # Trend features
        if 'ema_fast' in indicators and 'ema_slow' in indicators:
            fast = indicators['ema_fast']
            slow = indicators['ema_slow']
            if slow > 0:
                features['trend_strength'] = (fast - slow) / slow
        
        # Volume features
        if hasattr(market_data, 'volume'):
            features['volume_intensity'] = market_data.volume.as_double()
        
        return features
    
    def _create_volatility_detector(self) -> Any:
        """Create volatility regime detector."""
        # Placeholder - would be actual model
        return None
    
    def _create_trend_detector(self) -> Any:
        """Create trend regime detector."""
        # Placeholder - would be actual model
        return None
    
    def get_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by regime."""
        regime_metrics = defaultdict(lambda: {
            'count': 0,
            'total_return': 0,
            'win_rate': 0,
            'avg_duration': 0
        })
        
        # Aggregate performance by regime
        # This would use actual trade data
        
        return dict(regime_metrics)