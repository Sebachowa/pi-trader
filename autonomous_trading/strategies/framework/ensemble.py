"""
Strategy ensemble and mixing capabilities for combining multiple strategies.
"""

import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

from .interfaces import (
    BaseStrategyInterface,
    TradingSignal,
    SignalType,
    SignalStrength,
    StrategyState,
)
from .core import AdaptiveStrategy


class VotingMethod(Enum):
    """Voting methods for ensemble decisions."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    THRESHOLD = "threshold"
    ML_BASED = "ml_based"
    ADAPTIVE = "adaptive"


class StrategyEnsemble:
    """
    Combines multiple trading strategies into an ensemble.
    
    Features:
    - Multiple voting mechanisms
    - Dynamic weight optimization
    - Performance-based adaptation
    - Risk-aware aggregation
    """
    
    def __init__(self, ensemble_id: str, config: Dict[str, Any] = None):
        self.ensemble_id = ensemble_id
        self.config = config or {}
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategyInterface] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Voting configuration
        self.voting_method = VotingMethod(config.get('voting_method', 'weighted'))
        self.min_agreement = config.get('min_agreement', 0.6)
        self.use_ml_voting = config.get('use_ml_voting', False)
        
        # Weight optimization
        self.weight_optimizer = WeightOptimizer(self)
        self.optimize_frequency = config.get('optimize_frequency', 100)
        self.optimization_counter = 0
        
        # Performance tracking
        self.ensemble_performance = deque(maxlen=1000)
        self.signal_history = deque(maxlen=1000)
        
        # ML voting model
        self.voting_model = None
        self.voting_features = []
        
        # State
        self.is_active = False
    
    async def initialize(self) -> None:
        """Initialize the ensemble and all strategies."""
        # Initialize all strategies
        init_tasks = [strategy.initialize() for strategy in self.strategies.values()]
        await asyncio.gather(*init_tasks)
        
        # Initialize equal weights if not set
        if not self.strategy_weights:
            n_strategies = len(self.strategies)
            for strategy_id in self.strategies:
                self.strategy_weights[strategy_id] = 1.0 / n_strategies
        
        # Initialize weight optimizer
        await self.weight_optimizer.initialize()
        
        # Initialize ML voting model if enabled
        if self.use_ml_voting:
            self._initialize_ml_voting()
        
        self.is_active = True
    
    def add_strategy(self, strategy: BaseStrategyInterface, weight: float = None) -> None:
        """Add a strategy to the ensemble."""
        self.strategies[strategy.strategy_id] = strategy
        
        if weight is not None:
            self.strategy_weights[strategy.strategy_id] = weight
        else:
            # Recalculate equal weights
            n_strategies = len(self.strategies)
            for sid in self.strategies:
                self.strategy_weights[sid] = 1.0 / n_strategies
    
    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the ensemble."""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            del self.strategy_weights[strategy_id]
            
            # Renormalize weights
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for sid in self.strategy_weights:
                    self.strategy_weights[sid] /= total_weight
    
    async def on_data(self, data: Any) -> Optional[TradingSignal]:
        """Process market data through all strategies and combine signals."""
        if not self.is_active or not self.strategies:
            return None
        
        # Collect signals from all strategies
        signal_tasks = [
            self._get_strategy_signal(strategy_id, strategy, data)
            for strategy_id, strategy in self.strategies.items()
        ]
        
        signals = await asyncio.gather(*signal_tasks)
        
        # Filter out None signals
        valid_signals = [(sid, sig) for (sid, sig) in signals if sig is not None]
        
        if not valid_signals:
            return None
        
        # Combine signals based on voting method
        combined_signal = await self._combine_signals(valid_signals, data)
        
        if combined_signal:
            # Record signal
            self.signal_history.append({
                'timestamp': datetime.utcnow(),
                'signal': combined_signal,
                'contributing_strategies': [sid for sid, _ in valid_signals]
            })
            
            # Update optimization counter
            self.optimization_counter += 1
            if self.optimization_counter >= self.optimize_frequency:
                await self._optimize_weights()
                self.optimization_counter = 0
        
        return combined_signal
    
    async def _get_strategy_signal(self, strategy_id: str, 
                                  strategy: BaseStrategyInterface, 
                                  data: Any) -> Tuple[str, Optional[TradingSignal]]:
        """Get signal from individual strategy."""
        try:
            signal = await strategy.on_data(data)
            return (strategy_id, signal)
        except Exception as e:
            # Log error and return None
            return (strategy_id, None)
    
    async def _combine_signals(self, valid_signals: List[Tuple[str, TradingSignal]], 
                              data: Any) -> Optional[TradingSignal]:
        """Combine multiple signals into ensemble signal."""
        if self.voting_method == VotingMethod.MAJORITY:
            return self._majority_voting(valid_signals)
        elif self.voting_method == VotingMethod.WEIGHTED:
            return self._weighted_voting(valid_signals)
        elif self.voting_method == VotingMethod.UNANIMOUS:
            return self._unanimous_voting(valid_signals)
        elif self.voting_method == VotingMethod.THRESHOLD:
            return self._threshold_voting(valid_signals)
        elif self.voting_method == VotingMethod.ML_BASED:
            return await self._ml_based_voting(valid_signals, data)
        elif self.voting_method == VotingMethod.ADAPTIVE:
            return await self._adaptive_voting(valid_signals, data)
        else:
            return self._weighted_voting(valid_signals)
    
    def _majority_voting(self, signals: List[Tuple[str, TradingSignal]]) -> Optional[TradingSignal]:
        """Simple majority voting."""
        signal_counts = defaultdict(int)
        
        for _, signal in signals:
            signal_counts[signal.signal_type] += 1
        
        # Find majority signal
        total_signals = len(signals)
        for signal_type, count in signal_counts.items():
            if count > total_signals / 2:
                # Calculate average strength and confidence
                matching_signals = [s for _, s in signals if s.signal_type == signal_type]
                avg_strength = np.mean([s.strength for s in matching_signals])
                avg_confidence = np.mean([s.confidence for s in matching_signals])
                
                return TradingSignal(
                    signal_type=signal_type,
                    strength=avg_strength,
                    confidence=avg_confidence,
                    timestamp=datetime.utcnow(),
                    source=f"ensemble_{self.ensemble_id}",
                    metadata={
                        'voting_method': 'majority',
                        'agreement_ratio': count / total_signals,
                        'contributing_strategies': len(signals)
                    }
                )
        
        return None
    
    def _weighted_voting(self, signals: List[Tuple[str, TradingSignal]]) -> Optional[TradingSignal]:
        """Weighted voting based on strategy weights."""
        weighted_scores = defaultdict(float)
        total_weight = 0
        
        for strategy_id, signal in signals:
            weight = self.strategy_weights.get(strategy_id, 1.0)
            weighted_scores[signal.signal_type] += weight * signal.strength
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Normalize scores
        for signal_type in weighted_scores:
            weighted_scores[signal_type] /= total_weight
        
        # Find best signal
        best_signal_type = max(weighted_scores, key=weighted_scores.get)
        best_score = weighted_scores[best_signal_type]
        
        if best_score >= self.min_agreement:
            # Calculate weighted averages
            matching_signals = [(sid, s) for sid, s in signals if s.signal_type == best_signal_type]
            
            weighted_strength = sum(
                self.strategy_weights.get(sid, 1.0) * s.strength 
                for sid, s in matching_signals
            ) / sum(self.strategy_weights.get(sid, 1.0) for sid, _ in matching_signals)
            
            weighted_confidence = sum(
                self.strategy_weights.get(sid, 1.0) * s.confidence 
                for sid, s in matching_signals
            ) / sum(self.strategy_weights.get(sid, 1.0) for sid, _ in matching_signals)
            
            return TradingSignal(
                signal_type=best_signal_type,
                strength=weighted_strength,
                confidence=weighted_confidence,
                timestamp=datetime.utcnow(),
                source=f"ensemble_{self.ensemble_id}",
                metadata={
                    'voting_method': 'weighted',
                    'weighted_score': best_score,
                    'contributing_strategies': len(signals)
                }
            )
        
        return None
    
    def _unanimous_voting(self, signals: List[Tuple[str, TradingSignal]]) -> Optional[TradingSignal]:
        """Require unanimous agreement."""
        if not signals:
            return None
        
        first_signal_type = signals[0][1].signal_type
        
        # Check if all signals agree
        if all(signal.signal_type == first_signal_type for _, signal in signals):
            # Average strength and confidence
            avg_strength = np.mean([s.strength for _, s in signals])
            avg_confidence = np.mean([s.confidence for _, s in signals])
            
            return TradingSignal(
                signal_type=first_signal_type,
                strength=avg_strength,
                confidence=avg_confidence,
                timestamp=datetime.utcnow(),
                source=f"ensemble_{self.ensemble_id}",
                metadata={
                    'voting_method': 'unanimous',
                    'contributing_strategies': len(signals)
                }
            )
        
        return None
    
    def _threshold_voting(self, signals: List[Tuple[str, TradingSignal]]) -> Optional[TradingSignal]:
        """Threshold-based voting."""
        signal_scores = defaultdict(list)
        
        for _, signal in signals:
            signal_scores[signal.signal_type].append(signal.strength * signal.confidence)
        
        # Calculate average scores
        avg_scores = {
            signal_type: np.mean(scores) 
            for signal_type, scores in signal_scores.items()
        }
        
        # Find signals above threshold
        threshold = 0.6  # Configurable
        valid_signals = {
            signal_type: score 
            for signal_type, score in avg_scores.items() 
            if score >= threshold
        }
        
        if valid_signals:
            best_signal_type = max(valid_signals, key=valid_signals.get)
            
            # Get matching signals
            matching_signals = [s for _, s in signals if s.signal_type == best_signal_type]
            
            return TradingSignal(
                signal_type=best_signal_type,
                strength=np.mean([s.strength for s in matching_signals]),
                confidence=np.mean([s.confidence for s in matching_signals]),
                timestamp=datetime.utcnow(),
                source=f"ensemble_{self.ensemble_id}",
                metadata={
                    'voting_method': 'threshold',
                    'score': valid_signals[best_signal_type],
                    'contributing_strategies': len(matching_signals)
                }
            )
        
        return None
    
    async def _ml_based_voting(self, signals: List[Tuple[str, TradingSignal]], 
                              data: Any) -> Optional[TradingSignal]:
        """ML-based signal combination."""
        if not self.voting_model:
            # Fall back to weighted voting
            return self._weighted_voting(signals)
        
        # Extract features
        features = self._extract_voting_features(signals, data)
        
        # Predict best action
        try:
            prediction = self.voting_model.predict([features])[0]
            confidence = self.voting_model.predict_proba([features])[0].max()
            
            # Map prediction to signal type
            signal_type = self._map_prediction_to_signal(prediction)
            
            # Calculate strength from contributing signals
            matching_signals = [s for _, s in signals if s.signal_type == signal_type]
            if matching_signals:
                strength = np.mean([s.strength for s in matching_signals])
            else:
                strength = 0.5
            
            return TradingSignal(
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.utcnow(),
                source=f"ensemble_{self.ensemble_id}",
                metadata={
                    'voting_method': 'ml_based',
                    'ml_confidence': confidence,
                    'contributing_strategies': len(signals)
                }
            )
            
        except Exception as e:
            # Fall back to weighted voting
            return self._weighted_voting(signals)
    
    async def _adaptive_voting(self, signals: List[Tuple[str, TradingSignal]], 
                              data: Any) -> Optional[TradingSignal]:
        """Adaptive voting that changes based on market conditions."""
        # Analyze recent performance of different voting methods
        recent_performance = self._analyze_voting_performance()
        
        # Choose best performing method
        if recent_performance:
            best_method = max(recent_performance, key=recent_performance.get)
            
            if best_method == 'majority':
                return self._majority_voting(signals)
            elif best_method == 'weighted':
                return self._weighted_voting(signals)
            elif best_method == 'threshold':
                return self._threshold_voting(signals)
        
        # Default to weighted voting
        return self._weighted_voting(signals)
    
    def _initialize_ml_voting(self) -> None:
        """Initialize ML voting model."""
        # Create ensemble of classifiers
        self.voting_model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=5)),
                ('lr', LogisticRegression(max_iter=1000))
            ],
            voting='soft'
        )
        
        # Would need training data here
        # Placeholder for now
    
    def _extract_voting_features(self, signals: List[Tuple[str, TradingSignal]], 
                                data: Any) -> List[float]:
        """Extract features for ML voting."""
        features = []
        
        # Signal agreement features
        signal_types = [s.signal_type for _, s in signals]
        for signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]:
            count = signal_types.count(signal_type)
            features.append(count / len(signals) if signals else 0)
        
        # Average strength and confidence
        features.append(np.mean([s.strength for _, s in signals]) if signals else 0)
        features.append(np.mean([s.confidence for _, s in signals]) if signals else 0)
        
        # Strategy weight features
        contributing_weights = sum(
            self.strategy_weights.get(sid, 1.0) 
            for sid, _ in signals
        )
        features.append(contributing_weights)
        
        # Market features (placeholder)
        features.extend([0.5] * 10)  # Would extract from data
        
        return features
    
    def _map_prediction_to_signal(self, prediction: int) -> SignalType:
        """Map ML prediction to signal type."""
        mapping = {
            0: SignalType.HOLD,
            1: SignalType.BUY,
            2: SignalType.SELL
        }
        return mapping.get(prediction, SignalType.HOLD)
    
    def _analyze_voting_performance(self) -> Dict[str, float]:
        """Analyze recent performance of voting methods."""
        # This would analyze actual performance data
        # Placeholder implementation
        return {
            'weighted': 0.7,
            'majority': 0.6,
            'threshold': 0.65
        }
    
    async def _optimize_weights(self) -> None:
        """Optimize strategy weights based on performance."""
        # Get recent performance data
        performance_data = self._get_strategy_performance()
        
        if not performance_data:
            return
        
        # Optimize weights
        optimized_weights = await self.weight_optimizer.optimize(performance_data)
        
        if optimized_weights:
            self.strategy_weights = optimized_weights
    
    def _get_strategy_performance(self) -> Dict[str, List[float]]:
        """Get recent performance for each strategy."""
        performance = {}
        
        for strategy_id in self.strategies:
            if strategy_id in self.strategy_performance:
                recent_returns = [p['return'] for p in self.strategy_performance[strategy_id]]
                performance[strategy_id] = recent_returns
        
        return performance
    
    def update_performance(self, strategy_id: str, performance: Dict[str, Any]) -> None:
        """Update performance tracking for a strategy."""
        self.strategy_performance[strategy_id].append({
            'timestamp': datetime.utcnow(),
            'return': performance.get('return', 0),
            'sharpe': performance.get('sharpe_ratio', 0),
            'drawdown': performance.get('drawdown', 0)
        })
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get ensemble performance metrics."""
        metrics = {
            'ensemble_id': self.ensemble_id,
            'strategy_count': len(self.strategies),
            'voting_method': self.voting_method.value,
            'strategy_weights': dict(self.strategy_weights),
            'signal_count': len(self.signal_history),
            'is_active': self.is_active
        }
        
        # Add performance metrics if available
        if self.ensemble_performance:
            recent_performance = list(self.ensemble_performance)[-100:]
            returns = [p['return'] for p in recent_performance]
            
            metrics.update({
                'total_return': sum(returns),
                'average_return': np.mean(returns),
                'volatility': np.std(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            })
        
        return metrics


class WeightOptimizer:
    """
    Optimizes weights for ensemble strategies.
    
    Methods:
    - Mean-Variance Optimization
    - Risk Parity
    - Maximum Sharpe
    - Minimum Correlation
    - Black-Litterman
    """
    
    def __init__(self, ensemble: StrategyEnsemble):
        self.ensemble = ensemble
        self.optimization_method = 'sharpe'
        self.min_weight = 0.05
        self.max_weight = 0.40
        self.risk_free_rate = 0.02
    
    async def initialize(self) -> None:
        """Initialize weight optimizer."""
        pass
    
    async def optimize(self, performance_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Optimize portfolio weights based on performance data."""
        if not performance_data or len(performance_data) < 2:
            return None
        
        # Convert to returns matrix
        returns_df = pd.DataFrame(performance_data)
        
        if len(returns_df) < 20:  # Need minimum data
            return None
        
        if self.optimization_method == 'sharpe':
            return self._maximize_sharpe(returns_df)
        elif self.optimization_method == 'risk_parity':
            return self._risk_parity(returns_df)
        elif self.optimization_method == 'min_variance':
            return self._minimum_variance(returns_df)
        elif self.optimization_method == 'equal':
            return self._equal_weight(returns_df)
        else:
            return self._maximize_sharpe(returns_df)
    
    def _maximize_sharpe(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Maximize Sharpe ratio optimization."""
        try:
            # Calculate expected returns and covariance
            expected_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            
            # Number of assets
            n_assets = len(expected_returns)
            
            # Optimization variables
            weights = cp.Variable(n_assets)
            
            # Expected portfolio return
            portfolio_return = expected_returns.values @ weights
            
            # Portfolio variance
            portfolio_variance = cp.quad_form(weights, cov_matrix.values)
            
            # Sharpe ratio (approximation for convex optimization)
            # We maximize return / sqrt(variance)
            objective = cp.Maximize(portfolio_return - 0.5 * self.risk_free_rate * portfolio_variance)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1.0,
                weights >= self.min_weight,
                weights <= self.max_weight
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if weights.value is not None:
                # Convert to dictionary
                return {
                    strategy_id: float(weight) 
                    for strategy_id, weight in zip(returns_df.columns, weights.value)
                }
            
        except Exception as e:
            pass
        
        return None
    
    def _risk_parity(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Risk parity optimization."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_df.cov()
            
            # Initial guess (equal weight)
            n_assets = len(returns_df.columns)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Objective: minimize sum of squared differences in risk contributions
            def risk_parity_objective(weights):
                portfolio_variance = weights @ cov_matrix.values @ weights
                marginal_contrib = cov_matrix.values @ weights
                contrib = weights * marginal_contrib
                average_contrib = np.mean(contrib)
                return np.sum((contrib - average_contrib) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return {
                    strategy_id: float(weight) 
                    for strategy_id, weight in zip(returns_df.columns, result.x)
                }
            
        except Exception as e:
            pass
        
        return None
    
    def _minimum_variance(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Minimum variance optimization."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_df.cov()
            
            # Number of assets
            n_assets = len(returns_df.columns)
            
            # Optimization variables
            weights = cp.Variable(n_assets)
            
            # Portfolio variance
            portfolio_variance = cp.quad_form(weights, cov_matrix.values)
            
            # Objective: minimize variance
            objective = cp.Minimize(portfolio_variance)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1.0,
                weights >= self.min_weight,
                weights <= self.max_weight
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if weights.value is not None:
                return {
                    strategy_id: float(weight) 
                    for strategy_id, weight in zip(returns_df.columns, weights.value)
                }
            
        except Exception as e:
            pass
        
        return None
    
    def _equal_weight(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Equal weight allocation."""
        n_strategies = len(returns_df.columns)
        weight = 1.0 / n_strategies
        
        return {
            strategy_id: weight 
            for strategy_id in returns_df.columns
        }


class StrategyMixer:
    """
    Advanced strategy mixing with dynamic blending.
    
    Features:
    - Conditional mixing based on market state
    - Time-based blending
    - Performance-based transitions
    - Smooth weight transitions
    """
    
    def __init__(self, mixer_id: str, config: Dict[str, Any] = None):
        self.mixer_id = mixer_id
        self.config = config or {}
        
        # Mixing configuration
        self.blend_mode = config.get('blend_mode', 'performance')
        self.transition_speed = config.get('transition_speed', 0.1)
        self.min_active_strategies = config.get('min_active', 2)
        self.max_active_strategies = config.get('max_active', 5)
        
        # Strategy pools
        self.strategy_pool: Dict[str, BaseStrategyInterface] = {}
        self.active_strategies: Set[str] = set()
        self.strategy_scores: Dict[str, float] = {}
        
        # Blending weights
        self.current_weights: Dict[str, float] = {}
        self.target_weights: Dict[str, float] = {}
        
        # Market state tracking
        self.market_states = deque(maxlen=100)
        self.state_strategy_map = self._initialize_state_map()
    
    def _initialize_state_map(self) -> Dict[str, List[str]]:
        """Initialize market state to strategy mapping."""
        return {
            'bull_market': ['trend_following', 'momentum', 'breakout'],
            'bear_market': ['mean_reversion', 'defensive', 'short_bias'],
            'high_volatility': ['volatility_trading', 'options', 'market_neutral'],
            'low_volatility': ['carry_trade', 'market_making', 'arbitrage'],
            'ranging': ['mean_reversion', 'pairs_trading', 'market_making']
        }
    
    async def update_mix(self, market_state: Dict[str, Any], 
                        performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Update strategy mix based on conditions."""
        # Detect market state
        current_state = self._detect_market_state(market_state)
        self.market_states.append(current_state)
        
        # Score all strategies
        await self._score_strategies(current_state, performance_data)
        
        # Select active strategies
        selected = self._select_strategies(current_state)
        
        # Calculate target weights
        self.target_weights = self._calculate_target_weights(selected, current_state)
        
        # Smooth transition
        self._smooth_weight_transition()
        
        return self.current_weights
    
    def _detect_market_state(self, market_data: Dict[str, Any]) -> str:
        """Detect current market state."""
        # Simplified state detection
        volatility = market_data.get('volatility', 0.02)
        trend = market_data.get('trend', 0)
        
        if volatility > 0.03:
            return 'high_volatility'
        elif volatility < 0.01:
            return 'low_volatility'
        elif trend > 0.001:
            return 'bull_market'
        elif trend < -0.001:
            return 'bear_market'
        else:
            return 'ranging'
    
    async def _score_strategies(self, market_state: str, 
                               performance_data: Dict[str, Any]) -> None:
        """Score strategies based on current conditions."""
        for strategy_id in self.strategy_pool:
            score = 0.0
            
            # Market state compatibility
            if strategy_id in self.state_strategy_map.get(market_state, []):
                score += 0.4
            
            # Recent performance
            if strategy_id in performance_data:
                perf = performance_data[strategy_id]
                sharpe = perf.get('sharpe_ratio', 0)
                score += min(0.3, sharpe / 10)  # Cap contribution
                
                # Drawdown penalty
                drawdown = perf.get('max_drawdown', 0)
                score -= drawdown * 0.2
            
            # Consistency bonus
            if strategy_id in self.active_strategies:
                score += 0.1
            
            self.strategy_scores[strategy_id] = max(0, score)
    
    def _select_strategies(self, market_state: str) -> Set[str]:
        """Select active strategies based on scores."""
        # Sort by score
        sorted_strategies = sorted(
            self.strategy_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top strategies
        n_select = min(
            self.max_active_strategies,
            max(self.min_active_strategies, len([s for s, score in sorted_strategies if score > 0.3]))
        )
        
        selected = set()
        for strategy_id, score in sorted_strategies[:n_select]:
            if score > 0.2:  # Minimum score threshold
                selected.add(strategy_id)
        
        self.active_strategies = selected
        return selected
    
    def _calculate_target_weights(self, selected: Set[str], 
                                 market_state: str) -> Dict[str, float]:
        """Calculate target weights for selected strategies."""
        if not selected:
            return {}
        
        weights = {}
        total_score = sum(self.strategy_scores[s] for s in selected)
        
        if total_score > 0:
            # Score-based weights
            for strategy_id in selected:
                weights[strategy_id] = self.strategy_scores[strategy_id] / total_score
        else:
            # Equal weights
            for strategy_id in selected:
                weights[strategy_id] = 1.0 / len(selected)
        
        return weights
    
    def _smooth_weight_transition(self) -> None:
        """Smoothly transition from current to target weights."""
        # Initialize current weights if empty
        if not self.current_weights:
            self.current_weights = self.target_weights.copy()
            return
        
        # Get all strategy IDs
        all_strategies = set(self.current_weights.keys()) | set(self.target_weights.keys())
        
        new_weights = {}
        for strategy_id in all_strategies:
            current = self.current_weights.get(strategy_id, 0)
            target = self.target_weights.get(strategy_id, 0)
            
            # Smooth transition
            new_weight = current + (target - current) * self.transition_speed
            
            # Only keep if above threshold
            if new_weight > 0.01:
                new_weights[strategy_id] = new_weight
        
        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            for strategy_id in new_weights:
                new_weights[strategy_id] /= total
        
        self.current_weights = new_weights
    
    def add_strategy_to_pool(self, strategy: BaseStrategyInterface) -> None:
        """Add strategy to the pool."""
        self.strategy_pool[strategy.strategy_id] = strategy
    
    def remove_strategy_from_pool(self, strategy_id: str) -> None:
        """Remove strategy from the pool."""
        if strategy_id in self.strategy_pool:
            del self.strategy_pool[strategy_id]
            
            # Remove from active and weights
            self.active_strategies.discard(strategy_id)
            self.current_weights.pop(strategy_id, None)
            self.target_weights.pop(strategy_id, None)


class VotingMechanism:
    """
    Advanced voting mechanism for ensemble decisions.
    
    Features:
    - Confidence-weighted voting
    - Time-decay voting
    - Performance-weighted voting
    - Contextual voting
    """
    
    def __init__(self, mechanism_type: str = 'confidence_weighted'):
        self.mechanism_type = mechanism_type
        self.performance_window = 100
        self.time_decay_factor = 0.95
        self.context_weights = {}
    
    def vote(self, signals: List[Tuple[str, TradingSignal]], 
             context: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Execute voting mechanism."""
        if self.mechanism_type == 'confidence_weighted':
            return self._confidence_weighted_vote(signals)
        elif self.mechanism_type == 'time_decay':
            return self._time_decay_vote(signals)
        elif self.mechanism_type == 'performance_weighted':
            return self._performance_weighted_vote(signals, context)
        elif self.mechanism_type == 'contextual':
            return self._contextual_vote(signals, context)
        else:
            return self._confidence_weighted_vote(signals)
    
    def _confidence_weighted_vote(self, signals: List[Tuple[str, TradingSignal]]) -> Optional[TradingSignal]:
        """Vote weighted by signal confidence."""
        if not signals:
            return None
        
        # Group by signal type
        signal_groups = defaultdict(list)
        for strategy_id, signal in signals:
            signal_groups[signal.signal_type].append((strategy_id, signal))
        
        # Calculate confidence-weighted scores
        scores = {}
        for signal_type, group in signal_groups.items():
            total_confidence = sum(s.confidence for _, s in group)
            weighted_strength = sum(s.confidence * s.strength for _, s in group) / total_confidence if total_confidence > 0 else 0
            scores[signal_type] = (total_confidence, weighted_strength, len(group))
        
        # Select best signal
        if scores:
            best_type = max(scores, key=lambda x: scores[x][0])  # By total confidence
            confidence, strength, count = scores[best_type]
            
            return TradingSignal(
                signal_type=best_type,
                strength=strength,
                confidence=confidence / count,  # Average confidence
                timestamp=datetime.utcnow(),
                source='ensemble_vote',
                metadata={'voting_type': 'confidence_weighted', 'signal_count': count}
            )
        
        return None
    
    def _time_decay_vote(self, signals: List[Tuple[str, TradingSignal]]) -> Optional[TradingSignal]:
        """Vote with time decay on older signals."""
        if not signals:
            return None
        
        current_time = datetime.utcnow()
        weighted_scores = defaultdict(float)
        
        for strategy_id, signal in signals:
            # Calculate time weight
            age_seconds = (current_time - signal.timestamp).total_seconds()
            time_weight = self.time_decay_factor ** (age_seconds / 60)  # Decay per minute
            
            # Add weighted score
            weighted_scores[signal.signal_type] += signal.strength * signal.confidence * time_weight
        
        if weighted_scores:
            best_type = max(weighted_scores, key=weighted_scores.get)
            
            # Get matching signals
            matching = [s for _, s in signals if s.signal_type == best_type]
            
            return TradingSignal(
                signal_type=best_type,
                strength=np.mean([s.strength for s in matching]),
                confidence=np.mean([s.confidence for s in matching]),
                timestamp=datetime.utcnow(),
                source='ensemble_vote',
                metadata={'voting_type': 'time_decay', 'score': weighted_scores[best_type]}
            )
        
        return None
    
    def _performance_weighted_vote(self, signals: List[Tuple[str, TradingSignal]], 
                                  context: Dict[str, Any]) -> Optional[TradingSignal]:
        """Vote weighted by strategy performance."""
        if not signals or not context or 'performance' not in context:
            return self._confidence_weighted_vote(signals)
        
        performance = context['performance']
        weighted_scores = defaultdict(float)
        
        for strategy_id, signal in signals:
            # Get strategy performance
            perf_weight = 1.0  # Default
            if strategy_id in performance:
                sharpe = performance[strategy_id].get('sharpe_ratio', 0)
                perf_weight = max(0.1, min(2.0, 0.5 + sharpe / 2))  # Map Sharpe to weight
            
            # Add weighted score
            weighted_scores[signal.signal_type] += signal.strength * signal.confidence * perf_weight
        
        if weighted_scores:
            best_type = max(weighted_scores, key=weighted_scores.get)
            
            return TradingSignal(
                signal_type=best_type,
                strength=np.mean([s.strength for _, s in signals if s.signal_type == best_type]),
                confidence=weighted_scores[best_type] / len(signals),
                timestamp=datetime.utcnow(),
                source='ensemble_vote',
                metadata={'voting_type': 'performance_weighted'}
            )
        
        return None
    
    def _contextual_vote(self, signals: List[Tuple[str, TradingSignal]], 
                        context: Dict[str, Any]) -> Optional[TradingSignal]:
        """Vote based on market context."""
        if not signals or not context:
            return self._confidence_weighted_vote(signals)
        
        # Extract context features
        volatility = context.get('volatility', 'medium')
        trend = context.get('trend', 'neutral')
        liquidity = context.get('liquidity', 'normal')
        
        # Apply context-specific weights
        weighted_scores = defaultdict(float)
        
        for strategy_id, signal in signals:
            context_weight = 1.0
            
            # Adjust weight based on context
            if volatility == 'high' and signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                context_weight *= 0.7  # Reduce aggression in high volatility
            elif volatility == 'low' and signal.signal_type == SignalType.HOLD:
                context_weight *= 0.8  # Less holding in low volatility
            
            if trend == 'strong_up' and signal.signal_type == SignalType.BUY:
                context_weight *= 1.2  # Favor trend following
            elif trend == 'strong_down' and signal.signal_type == SignalType.SELL:
                context_weight *= 1.2
            
            weighted_scores[signal.signal_type] += signal.strength * signal.confidence * context_weight
        
        if weighted_scores:
            best_type = max(weighted_scores, key=weighted_scores.get)
            
            return TradingSignal(
                signal_type=best_type,
                strength=np.mean([s.strength for _, s in signals if s.signal_type == best_type]),
                confidence=weighted_scores[best_type] / sum(weighted_scores.values()),
                timestamp=datetime.utcnow(),
                source='ensemble_vote',
                metadata={'voting_type': 'contextual', 'context': context}
            )
        
        return None