
"""
ML-Powered Strategy Selection and Optimization System

Intelligent strategy switching using reinforcement learning, genetic algorithms,
and Bayesian optimization for autonomous strategy evolution.
"""

import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import optuna

from autonomous_trading.core.market_analyzer import MarketRegime
from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
# from nautilus_trader.common.logging import Logger  # Not available in this version
from nautilus_trader.model.identifiers import InstrumentId, StrategyId
from nautilus_trader.trading.strategy import Strategy


class StrategyGene:
    """Represents genetic information for a strategy."""
    
    def __init__(self, strategy_type: str, parameters: Dict[str, Any]):
        self.strategy_type = strategy_type
        self.parameters = parameters
        self.fitness = 0.0
        self.generation = 0
        self.parent_ids = []
        self.mutation_rate = 0.1
        

class ReinforcementAgent:
    """RL agent for strategy selection using contextual bandits."""
    
    def __init__(self, n_strategies: int, n_features: int, epsilon: float = 0.1):
        self.n_strategies = n_strategies
        self.n_features = n_features
        self.epsilon = epsilon
        
        # Thompson Sampling parameters
        self.alpha = np.ones(n_strategies)  # Success counts
        self.beta = np.ones(n_strategies)   # Failure counts
        
        # Contextual bandit model
        self.models = [
            GradientBoostingRegressor(n_estimators=50, max_depth=3)
            for _ in range(n_strategies)
        ]
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.min_samples_to_train = 100
        
    def select_strategy(self, context: np.ndarray) -> int:
        """Select strategy using Thompson Sampling with context."""
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.randint(self.n_strategies)
        
        # Thompson Sampling
        samples = np.random.beta(self.alpha, self.beta)
        
        # Add contextual predictions if we have enough data
        if len(self.replay_buffer) > self.min_samples_to_train:
            predictions = []
            for i, model in enumerate(self.models):
                try:
                    pred = model.predict(context.reshape(1, -1))[0]
                    predictions.append(pred)
                except:
                    predictions.append(0)
            
            # Combine Thompson samples with predictions
            combined_scores = samples + 0.5 * np.array(predictions)
            return np.argmax(combined_scores)
        
        return np.argmax(samples)
    
    def update(self, strategy_idx: int, context: np.ndarray, reward: float):
        """Update agent with observed reward."""
        # Update Thompson Sampling parameters
        if reward > 0:
            self.alpha[strategy_idx] += reward
        else:
            self.beta[strategy_idx] += abs(reward)
        
        # Store experience
        self.replay_buffer.append((strategy_idx, context, reward))
        
        # Retrain models periodically
        if len(self.replay_buffer) > self.min_samples_to_train and len(self.replay_buffer) % 100 == 0:
            self._retrain_models()
    
    def _retrain_models(self):
        """Retrain contextual models from replay buffer."""
        # Organize data by strategy
        strategy_data = defaultdict(list)
        
        for strategy_idx, context, reward in self.replay_buffer:
            strategy_data[strategy_idx].append((context, reward))
        
        # Train each model
        for strategy_idx, data in strategy_data.items():
            if len(data) > 20:  # Minimum samples
                X = np.array([d[0] for d in data])
                y = np.array([d[1] for d in data])
                
                try:
                    self.models[strategy_idx].fit(X, y)
                except:
                    pass  # Skip if training fails


class MLStrategySelector(Component):
    """
    Machine Learning powered strategy selection and optimization system.
    
    Features:
    - Reinforcement learning for strategy selection
    - Genetic algorithms for strategy evolution
    - Bayesian optimization for parameter tuning
    - Market regime prediction
    - Strategy performance prediction
    - Ensemble strategy creation
    """
    
    def __init__(
        self,
        logger: Any,  # Logger type
        clock: LiveClock,
        msgbus: MessageBus,
        enable_evolution: bool = True,
        enable_rl_selection: bool = True,
        enable_bayesian_opt: bool = True,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        # Initialize component with minimal parameters
        try:
            super().__init__()
        except Exception:
            # If that fails, try with specific parameters
            pass
        
        self.clock = clock
        self.logger = logger
        self.msgbus = msgbus
        self._component_id = "ML-STRATEGY-SELECTOR"
        
        self.enable_evolution = enable_evolution
        self.enable_rl_selection = enable_rl_selection
        self.enable_bayesian_opt = enable_bayesian_opt
        
        # Genetic Algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Strategy population
        self._strategy_population: List[StrategyGene] = []
        self._strategy_registry: Dict[str, Type[Strategy]] = {}
        self._active_strategies: Dict[StrategyId, StrategyGene] = {}
        
        # Performance tracking
        self._strategy_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._regime_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # ML Models
        self._regime_predictor = RandomForestClassifier(n_estimators=100, max_depth=5)
        self._performance_predictor = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        self._feature_scaler = StandardScaler()
        
        # Reinforcement Learning
        self._rl_agent = None
        self._context_features = 50  # Number of market features
        
        # Bayesian Optimization
        self._bayesian_optimizer = None
        self._optimization_history = deque(maxlen=1000)
        
        # Feature engineering
        self._feature_cache = deque(maxlen=1000)
        self._regime_history = deque(maxlen=1000)
        
        # Tasks
        self._evolution_task = None
        self._optimization_task = None
        self._prediction_task = None
        
    async def initialize(self, strategy_registry: Dict[str, Type[Strategy]]) -> None:
        """Initialize the ML strategy selector."""
        self._strategy_registry = strategy_registry
        
        # Initialize RL agent
        if self.enable_rl_selection:
            self._rl_agent = ReinforcementAgent(
                n_strategies=len(strategy_registry),
                n_features=self._context_features,
            )
        
        # Initialize population
        if self.enable_evolution:
            self._initialize_population()
        
        # Start background tasks
        self._evolution_task = asyncio.create_task(self._evolution_loop())
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._prediction_task = asyncio.create_task(self._prediction_loop())
        
        if hasattr(self, "logger") and self.logger:

        
            self.logger.info("ML Strategy Selector initialized")

        
        else:

        
            print("INFO: " + str("ML Strategy Selector initialized"))
    
    def _initialize_population(self) -> None:
        """Initialize strategy population with diverse genes."""
        self._strategy_population = []
        
        for _ in range(self.population_size):
            # Random strategy type
            strategy_type = np.random.choice(list(self._strategy_registry.keys()))
            
            # Random parameters (would be strategy-specific)
            parameters = self._generate_random_parameters(strategy_type)
            
            gene = StrategyGene(strategy_type, parameters)
            self._strategy_population.append(gene)
    
    def _generate_random_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """Generate random parameters for a strategy type."""
        # This would be customized per strategy type
        # Example parameters for common strategies
        
        if "momentum" in strategy_type.lower():
            return {
                "lookback_period": np.random.randint(10, 100),
                "entry_threshold": np.random.uniform(0.5, 2.0),
                "exit_threshold": np.random.uniform(0.1, 0.5),
                "stop_loss": np.random.uniform(0.01, 0.05),
                "take_profit": np.random.uniform(0.02, 0.10),
            }
        elif "mean_reversion" in strategy_type.lower():
            return {
                "lookback_period": np.random.randint(20, 200),
                "entry_deviation": np.random.uniform(1.5, 3.0),
                "exit_deviation": np.random.uniform(0.0, 1.0),
                "position_sizing": np.random.uniform(0.1, 0.5),
            }
        elif "market_making" in strategy_type.lower():
            return {
                "spread_multiplier": np.random.uniform(1.0, 3.0),
                "order_levels": np.random.randint(1, 5),
                "order_spacing": np.random.uniform(0.001, 0.01),
                "inventory_limit": np.random.uniform(0.1, 0.5),
                "skew_factor": np.random.uniform(0.0, 0.5),
            }
        else:
            # Generic parameters
            return {
                "risk_per_trade": np.random.uniform(0.005, 0.02),
                "max_positions": np.random.randint(1, 10),
                "time_in_force": np.random.choice(["DAY", "GTC", "IOC"]),
            }
    
    async def select_strategies(
        self,
        market_state: Dict[str, Any],
        n_strategies: int = 5,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Select optimal strategies using ML."""
        
        # Extract market features
        features = await self._extract_market_features(market_state)
        
        selected_strategies = []
        
        if self.enable_rl_selection and self._rl_agent:
            # Use RL for selection
            for _ in range(n_strategies):
                strategy_idx = self._rl_agent.select_strategy(features)
                strategy_type = list(self._strategy_registry.keys())[strategy_idx]
                
                # Get best parameters for this strategy type
                parameters = await self._get_optimal_parameters(strategy_type, features)
                
                # Predict performance
                confidence = await self._predict_strategy_performance(
                    strategy_type, parameters, features
                )
                
                selected_strategies.append((strategy_type, parameters, confidence))
        
        else:
            # Use performance-based selection
            strategy_scores = await self._score_all_strategies(features)
            
            # Sort by score and select top N
            sorted_strategies = sorted(
                strategy_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_strategies]
            
            for strategy_type, score in sorted_strategies:
                parameters = await self._get_optimal_parameters(strategy_type, features)
                selected_strategies.append((strategy_type, parameters, score))
        
        return selected_strategies
    
    async def evolve_strategy(
        self,
        base_strategy: str,
        performance_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evolve a strategy using genetic algorithms."""
        if not self.enable_evolution:
            return None
        
        # Find genes for this strategy type
        strategy_genes = [
            g for g in self._strategy_population
            if g.strategy_type == base_strategy
        ]
        
        if len(strategy_genes) < 2:
            return None
        
        # Update fitness based on performance
        for gene in strategy_genes:
            gene.fitness = self._calculate_fitness(gene, performance_data)
        
        # Perform genetic operations
        new_gene = await self._genetic_evolution_step(strategy_genes)
        
        return new_gene.parameters if new_gene else None
    
    async def optimize_parameters(
        self,
        strategy_type: str,
        current_params: Dict[str, Any],
        market_features: np.ndarray,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using Bayesian optimization."""
        if not self.enable_bayesian_opt:
            return current_params
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
        )
        
        # Define objective function
        def objective(trial):
            # Sample parameters based on strategy type
            params = self._sample_parameters(trial, strategy_type)
            
            # Evaluate using surrogate model
            score = self._evaluate_parameters(
                strategy_type, params, market_features
            )
            
            return score
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        # Get best parameters
        best_params = study.best_params
        
        # Store optimization result
        self._optimization_history.append({
            "timestamp": datetime.utcnow(),
            "strategy_type": strategy_type,
            "best_params": best_params,
            "best_value": study.best_value,
            "n_trials": n_trials,
        })
        
        return best_params
    
    async def predict_regime_change(
        self,
        market_features: np.ndarray,
        horizon_hours: int = 24,
    ) -> Dict[str, float]:
        """Predict probability of regime change."""
        if len(self._feature_cache) < 100:
            # Not enough data for prediction
            return {"no_change": 1.0}
        
        try:
            # Scale features
            scaled_features = self._feature_scaler.transform(market_features.reshape(1, -1))
            
            # Predict regime probabilities
            probs = self._regime_predictor.predict_proba(scaled_features)[0]
            
            # Map to regime names
            regime_probs = {}
            regimes = ["trending_up", "trending_down", "ranging", "volatile"]
            
            for i, regime in enumerate(regimes):
                if i < len(probs):
                    regime_probs[regime] = probs[i]
            
            return regime_probs
            
        except Exception as e:
            if hasattr(self, "logger") and self.logger:

                self.logger.warning(f"Regime prediction failed: {e}")

            else:

                print("WARNING: " + str(f"Regime prediction failed: {e}"))
            return {"unknown": 1.0}
    
    async def create_ensemble_strategy(
        self,
        base_strategies: List[str],
        market_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create ensemble strategy from multiple base strategies."""
        features = await self._extract_market_features(market_state)
        
        ensemble_config = {
            "type": "ensemble",
            "strategies": [],
            "weights": [],
            "voting_method": "weighted_average",
            "rebalance_frequency": "daily",
        }
        
        total_score = 0.0
        
        for strategy_type in base_strategies:
            # Get optimal parameters
            params = await self._get_optimal_parameters(strategy_type, features)
            
            # Predict performance
            score = await self._predict_strategy_performance(
                strategy_type, params, features
            )
            
            ensemble_config["strategies"].append({
                "type": strategy_type,
                "parameters": params,
            })
            
            ensemble_config["weights"].append(score)
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            ensemble_config["weights"] = [
                w / total_score for w in ensemble_config["weights"]
            ]
        
        return ensemble_config
    
    async def _evolution_loop(self) -> None:
        """Background loop for genetic evolution."""
        while True:
            try:
                if not self.enable_evolution:
                    await asyncio.sleep(3600)
                    continue
                
                # Perform evolution every hour
                await asyncio.sleep(3600)
                
                # Evolve population
                await self._evolve_population()
                
                if hasattr(self, "logger") and self.logger:
                    self.logger.info(
                    f"Evolution completed - Generation {self._get_current_generation()}, "
                    f"Best fitness: {self._get_best_fitness():.3f}"
                )
                else:
                    print("INFO: " + str(
                    f"Evolution completed - Generation {self._get_current_generation()}, "
                    f"Best fitness: {self._get_best_fitness():.3f}"
                ))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if hasattr(self, "logger") and self.logger:

                    self.logger.error(f"Evolution error: {e}")

                else:

                    print("ERROR: " + str(f"Evolution error: {e}"))
    
    async def _optimization_loop(self) -> None:
        """Background loop for parameter optimization."""
        while True:
            try:
                if not self.enable_bayesian_opt:
                    await asyncio.sleep(3600)
                    continue
                
                # Optimize parameters every 6 hours
                await asyncio.sleep(21600)
                
                # Optimize each active strategy
                for strategy_id, gene in self._active_strategies.items():
                    market_features = await self._get_current_market_features()
                    
                    optimized_params = await self.optimize_parameters(
                        gene.strategy_type,
                        gene.parameters,
                        market_features,
                        n_trials=30,
                    )
                    
                    # Update gene with optimized parameters
                    gene.parameters = optimized_params
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if hasattr(self, "logger") and self.logger:

                    self.logger.error(f"Optimization error: {e}")

                else:

                    print("ERROR: " + str(f"Optimization error: {e}"))
    
    async def _prediction_loop(self) -> None:
        """Background loop for updating predictions."""
        while True:
            try:
                # Update predictions every 15 minutes
                await asyncio.sleep(900)
                
                # Collect recent market data
                market_features = await self._get_current_market_features()
                
                # Update feature cache
                self._feature_cache.append({
                    "timestamp": datetime.utcnow(),
                    "features": market_features,
                })
                
                # Retrain models if enough new data
                if len(self._feature_cache) > 100 and len(self._feature_cache) % 50 == 0:
                    await self._retrain_models()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if hasattr(self, "logger") and self.logger:

                    self.logger.error(f"Prediction loop error: {e}")

                else:

                    print("ERROR: " + str(f"Prediction loop error: {e}"))
    
    async def _evolve_population(self) -> None:
        """Perform one generation of evolution."""
        # Evaluate fitness of current population
        for gene in self._strategy_population:
            gene.fitness = await self._evaluate_gene_fitness(gene)
        
        # Sort by fitness
        self._strategy_population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Keep top performers (elitism)
        elite_size = int(self.population_size * 0.1)
        new_population = self._strategy_population[:elite_size]
        
        # Generate new offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = parent1  # Clone parent
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)
            
            # Update generation
            offspring.generation += 1
            offspring.parent_ids = [id(parent1), id(parent2)]
            
            new_population.append(offspring)
        
        self._strategy_population = new_population
    
    def _tournament_selection(self, tournament_size: int = 3) -> StrategyGene:
        """Select parent using tournament selection."""
        tournament = np.random.choice(
            self._strategy_population,
            size=tournament_size,
            replace=False
        )
        
        return max(tournament, key=lambda g: g.fitness)
    
    def _crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """Perform crossover between two parent genes."""
        # Use uniform crossover for parameters
        child_params = {}
        
        all_params = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        
        for param in all_params:
            if param in parent1.parameters and param in parent2.parameters:
                # Random choice between parents
                if np.random.random() < 0.5:
                    child_params[param] = parent1.parameters[param]
                else:
                    child_params[param] = parent2.parameters[param]
            elif param in parent1.parameters:
                child_params[param] = parent1.parameters[param]
            else:
                child_params[param] = parent2.parameters[param]
        
        # Inherit strategy type from better parent
        if parent1.fitness > parent2.fitness:
            strategy_type = parent1.strategy_type
        else:
            strategy_type = parent2.strategy_type
        
        return StrategyGene(strategy_type, child_params)
    
    def _mutate(self, gene: StrategyGene) -> StrategyGene:
        """Mutate a strategy gene."""
        mutated_params = gene.parameters.copy()
        
        # Mutate each parameter with probability
        for param, value in mutated_params.items():
            if np.random.random() < self.mutation_rate:
                if isinstance(value, (int, float)):
                    # Gaussian mutation for numeric parameters
                    std_dev = abs(value) * 0.1 if value != 0 else 0.1
                    mutated_value = value + np.random.normal(0, std_dev)
                    
                    # Keep same type
                    if isinstance(value, int):
                        mutated_value = int(mutated_value)
                    
                    mutated_params[param] = mutated_value
                    
                elif isinstance(value, str):
                    # For string parameters, occasionally try different values
                    # This would be customized based on valid options
                    pass
        
        return StrategyGene(gene.strategy_type, mutated_params)
    
    async def _evaluate_gene_fitness(self, gene: StrategyGene) -> float:
        """Evaluate fitness of a strategy gene."""
        # Look up historical performance for similar configurations
        perf_key = f"{gene.strategy_type}_{hash(str(sorted(gene.parameters.items())))}"
        
        if perf_key in self._strategy_performance:
            recent_performance = list(self._strategy_performance[perf_key])[-100:]
            
            if recent_performance:
                # Calculate fitness metrics
                returns = [p["return"] for p in recent_performance]
                
                # Sharpe ratio component
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe = 0
                
                # Win rate component
                win_rate = sum(1 for r in returns if r > 0) / len(returns)
                
                # Profit factor component
                profits = sum(r for r in returns if r > 0)
                losses = abs(sum(r for r in returns if r < 0))
                profit_factor = profits / losses if losses > 0 else 0
                
                # Combined fitness
                fitness = (
                    0.4 * max(0, sharpe) +
                    0.3 * win_rate +
                    0.3 * max(0, profit_factor - 1)
                )
                
                return fitness
        
        # No historical data - return neutral fitness
        return 0.5
    
    def _calculate_fitness(self, gene: StrategyGene, performance_data: Dict[str, Any]) -> float:
        """Calculate fitness from performance data."""
        # Extract key metrics
        sharpe_ratio = performance_data.get("sharpe_ratio", 0)
        total_return = performance_data.get("total_return", 0)
        max_drawdown = performance_data.get("max_drawdown", 0)
        win_rate = performance_data.get("win_rate", 0.5)
        
        # Fitness function emphasizing risk-adjusted returns
        fitness = (
            0.4 * max(0, sharpe_ratio) +
            0.2 * (1 + total_return) +
            0.2 * (1 - max_drawdown) +
            0.2 * win_rate
        )
        
        return fitness
    
    async def _genetic_evolution_step(
        self,
        strategy_genes: List[StrategyGene],
    ) -> Optional[StrategyGene]:
        """Perform one step of genetic evolution."""
        if len(strategy_genes) < 2:
            return None
        
        # Sort by fitness
        strategy_genes.sort(key=lambda g: g.fitness, reverse=True)
        
        # Select parents
        parent1 = strategy_genes[0]  # Best
        parent2 = self._tournament_selection()
        
        # Create offspring
        offspring = self._crossover(parent1, parent2)
        
        # Mutate
        if np.random.random() < self.mutation_rate * 1.5:  # Higher mutation rate
            offspring = self._mutate(offspring)
        
        return offspring
    
    def _sample_parameters(self, trial: optuna.Trial, strategy_type: str) -> Dict[str, Any]:
        """Sample parameters for Optuna trial."""
        if "momentum" in strategy_type.lower():
            return {
                "lookback_period": trial.suggest_int("lookback_period", 10, 100),
                "entry_threshold": trial.suggest_float("entry_threshold", 0.5, 2.0),
                "exit_threshold": trial.suggest_float("exit_threshold", 0.1, 0.5),
                "stop_loss": trial.suggest_float("stop_loss", 0.01, 0.05),
                "take_profit": trial.suggest_float("take_profit", 0.02, 0.10),
            }
        # Add more strategy types...
        
        return {}
    
    def _evaluate_parameters(
        self,
        strategy_type: str,
        params: Dict[str, Any],
        market_features: np.ndarray,
    ) -> float:
        """Evaluate parameters using surrogate model."""
        # This would use historical backtesting results or a trained model
        # For now, return a simulated score
        
        # Create feature vector from parameters and market features
        param_values = list(params.values())
        features = np.concatenate([param_values, market_features])
        
        try:
            # Use performance predictor if trained
            if hasattr(self._performance_predictor, 'n_features_in_'):
                score = self._performance_predictor.predict(features.reshape(1, -1))[0]
            else:
                # Random score for untrained model
                score = np.random.random()
            
            return score
            
        except:
            return np.random.random()
    
    async def _extract_market_features(self, market_state: Dict[str, Any]) -> np.ndarray:
        """Extract features from market state."""
        features = []
        
        # Price features
        features.extend([
            market_state.get("price_return_1h", 0),
            market_state.get("price_return_24h", 0),
            market_state.get("price_return_7d", 0),
        ])
        
        # Volatility features
        features.extend([
            market_state.get("volatility_1h", 0),
            market_state.get("volatility_24h", 0),
            market_state.get("volatility_ratio", 1),
        ])
        
        # Volume features
        features.extend([
            market_state.get("volume_ratio", 1),
            market_state.get("volume_trend", 0),
        ])
        
        # Technical indicators
        features.extend([
            market_state.get("rsi", 50),
            market_state.get("macd_signal", 0),
            market_state.get("bb_position", 0.5),
        ])
        
        # Market microstructure
        features.extend([
            market_state.get("bid_ask_spread", 0),
            market_state.get("order_book_imbalance", 0),
        ])
        
        # Pad with zeros if needed
        while len(features) < self._context_features:
            features.append(0)
        
        return np.array(features[:self._context_features])
    
    async def _get_optimal_parameters(
        self,
        strategy_type: str,
        market_features: np.ndarray,
    ) -> Dict[str, Any]:
        """Get optimal parameters for strategy type and market conditions."""
        # Look for best performing gene
        best_gene = None
        best_fitness = -np.inf
        
        for gene in self._strategy_population:
            if gene.strategy_type == strategy_type and gene.fitness > best_fitness:
                best_gene = gene
                best_fitness = gene.fitness
        
        if best_gene:
            # Further optimize if Bayesian optimization is enabled
            if self.enable_bayesian_opt:
                return await self.optimize_parameters(
                    strategy_type,
                    best_gene.parameters,
                    market_features,
                    n_trials=20,
                )
            return best_gene.parameters
        
        # Generate default parameters
        return self._generate_random_parameters(strategy_type)
    
    async def _predict_strategy_performance(
        self,
        strategy_type: str,
        parameters: Dict[str, Any],
        market_features: np.ndarray,
    ) -> float:
        """Predict strategy performance confidence score."""
        # Combine strategy parameters with market features
        param_vector = []
        
        # Encode strategy type
        strategy_encoding = hash(strategy_type) % 100 / 100
        param_vector.append(strategy_encoding)
        
        # Add parameter values
        for key in sorted(parameters.keys()):
            value = parameters[key]
            if isinstance(value, (int, float)):
                param_vector.append(value)
            else:
                # Hash non-numeric values
                param_vector.append(hash(str(value)) % 100 / 100)
        
        # Combine with market features
        full_features = np.concatenate([param_vector, market_features])
        
        try:
            if hasattr(self._performance_predictor, 'n_features_in_'):
                # Predict expected Sharpe ratio
                predicted_sharpe = self._performance_predictor.predict(
                    full_features.reshape(1, -1)
                )[0]
                
                # Convert to confidence score (0-1)
                confidence = 1 / (1 + np.exp(-predicted_sharpe))
            else:
                # No trained model yet
                confidence = 0.5
            
            return confidence
            
        except:
            return 0.5
    
    async def _score_all_strategies(self, market_features: np.ndarray) -> Dict[str, float]:
        """Score all available strategies for current market conditions."""
        scores = {}
        
        for strategy_type in self._strategy_registry:
            # Get best parameters
            params = await self._get_optimal_parameters(strategy_type, market_features)
            
            # Predict performance
            score = await self._predict_strategy_performance(
                strategy_type, params, market_features
            )
            
            # Adjust by historical regime performance
            regime_adjustment = self._get_regime_adjustment(strategy_type)
            
            scores[strategy_type] = score * regime_adjustment
        
        return scores
    
    def _get_regime_adjustment(self, strategy_type: str) -> float:
        """Get performance adjustment based on historical regime performance."""
        current_regime = self._get_current_regime()
        
        if current_regime in self._regime_performance:
            regime_perf = self._regime_performance[current_regime].get(strategy_type, 0)
            
            # Convert to multiplier (0.5 to 1.5)
            if regime_perf > 0:
                return min(1.5, 1.0 + regime_perf)
            else:
                return max(0.5, 1.0 + regime_perf)
        
        return 1.0
    
    def _get_current_regime(self) -> str:
        """Get current market regime."""
        if self._regime_history:
            return self._regime_history[-1]
        return "unknown"
    
    async def _get_current_market_features(self) -> np.ndarray:
        """Get current market feature vector."""
        # This would integrate with market data feeds
        # Placeholder implementation
        return np.random.randn(self._context_features)
    
    async def _retrain_models(self) -> None:
        """Retrain ML models with recent data."""
        if len(self._feature_cache) < 100:
            return
        
        # Prepare training data
        X = []
        y_regime = []
        y_performance = []
        
        for i, data in enumerate(list(self._feature_cache)[:-1]):
            X.append(data["features"])
            
            # Get next regime (if available)
            if i + 1 < len(self._regime_history):
                y_regime.append(self._regime_history[i + 1])
            
            # Get performance outcome (would come from strategy results)
            # Placeholder
            y_performance.append(np.random.random())
        
        X = np.array(X)
        
        # Train regime predictor
        if len(set(y_regime)) > 1:  # Need at least 2 classes
            try:
                self._feature_scaler.fit(X)
                X_scaled = self._feature_scaler.transform(X)
                self._regime_predictor.fit(X_scaled, y_regime)
                if hasattr(self, "logger") and self.logger:

                    self.logger.info("Retrained regime predictor")

                else:

                    print("INFO: " + str("Retrained regime predictor"))
            except Exception as e:
                if hasattr(self, "logger") and self.logger:

                    self.logger.error(f"Failed to train regime predictor: {e}")

                else:

                    print("ERROR: " + str(f"Failed to train regime predictor: {e}"))
        
        # Train performance predictor
        if len(y_performance) > 50:
            try:
                self._performance_predictor.fit(X, y_performance)
                if hasattr(self, "logger") and self.logger:

                    self.logger.info("Retrained performance predictor")

                else:

                    print("INFO: " + str("Retrained performance predictor"))
            except Exception as e:
                if hasattr(self, "logger") and self.logger:

                    self.logger.error(f"Failed to train performance predictor: {e}")

                else:

                    print("ERROR: " + str(f"Failed to train performance predictor: {e}"))
    
    def _get_current_generation(self) -> int:
        """Get current generation number."""
        if self._strategy_population:
            return max(g.generation for g in self._strategy_population)
        return 0
    
    def _get_best_fitness(self) -> float:
        """Get best fitness in population."""
        if self._strategy_population:
            return max(g.fitness for g in self._strategy_population)
        return 0.0
    
    def get_strategy_recommendations(self) -> Dict[str, Any]:
        """Get current strategy recommendations."""
        return {
            "timestamp": datetime.utcnow(),
            "current_generation": self._get_current_generation(),
            "best_fitness": self._get_best_fitness(),
            "active_strategies": len(self._active_strategies),
            "population_size": len(self._strategy_population),
            "top_strategies": [
                {
                    "type": g.strategy_type,
                    "fitness": g.fitness,
                    "generation": g.generation,
                }
                for g in sorted(self._strategy_population, key=lambda x: x.fitness, reverse=True)[:5]
            ],
            "optimization_count": len(self._optimization_history),
            "rl_performance": {
                "strategies": list(self._strategy_registry.keys()),
                "selection_rates": self._rl_agent.alpha / (self._rl_agent.alpha + self._rl_agent.beta)
                if self._rl_agent else [],
            },
        }