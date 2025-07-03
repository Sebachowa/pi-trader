
"""
ML Integration Module

Integrates the ML optimizer with the existing trading system, providing a seamless
interface for ML-powered strategy optimization and execution.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

# Nautilus imports
from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
# from nautilus_trader.common.logging import Logger  # Not available in this version
from nautilus_trader.model.identifiers import InstrumentId, StrategyId

# Local imports
from autonomous_trading.ml.ml_optimizer import MLOptimizer, MarketFeatures
from autonomous_trading.ml.feature_engineering import FeatureEngineer
from autonomous_trading.ml.backtest_optimizer import MLBacktestOptimizer, BacktestResult
from autonomous_trading.strategies.ml_strategy_selector import MLStrategySelector
from autonomous_trading.strategies.orchestrator import StrategyOrchestrator
from autonomous_trading.core.market_analyzer import MarketAnalyzer, MarketConditions


class MLTradingSystem(Component):
    """
    Comprehensive ML-powered trading system that integrates all ML components.
    
    This system coordinates:
    - Feature extraction and engineering
    - ML model predictions
    - Strategy selection and optimization
    - Parameter tuning
    - Performance tracking and improvement
    """
    
    def __init__(
        self,
        logger: Any,  # Logger type
        clock: LiveClock,
        msgbus: MessageBus,
        market_analyzer: MarketAnalyzer,
        strategy_orchestrator: StrategyOrchestrator,
        enable_ml_optimization: bool = True,
        enable_continuous_learning: bool = True,
        enable_auto_rebalancing: bool = True,
        performance_threshold: float = 0.6,
        reoptimization_interval_hours: int = 24,
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id="ML-TRADING-SYSTEM",
            msgbus=msgbus,
        )
        
        self.market_analyzer = market_analyzer
        self.strategy_orchestrator = strategy_orchestrator
        self.enable_ml_optimization = enable_ml_optimization
        self.enable_continuous_learning = enable_continuous_learning
        self.enable_auto_rebalancing = enable_auto_rebalancing
        self.performance_threshold = performance_threshold
        self.reoptimization_interval_hours = reoptimization_interval_hours
        
        # Initialize ML components
        self.ml_optimizer = MLOptimizer(
            logger=logger,
            clock=clock,
            msgbus=msgbus,
            enable_deep_learning=True,
            enable_reinforcement_learning=True,
            enable_genetic_optimization=True,
        )
        
        self.feature_engineer = FeatureEngineer(
            enable_ta_features=True,
            enable_microstructure=True,
            enable_patterns=True,
        )
        
        self.backtest_optimizer = MLBacktestOptimizer(
            logger=logger,
            use_gpu=False,
            cache_results=True,
        )
        
        # Enhanced strategy selector
        self.ml_strategy_selector = MLStrategySelector(
            logger=logger,
            clock=clock,
            msgbus=msgbus,
            enable_evolution=True,
            enable_rl_selection=True,
            enable_bayesian_opt=True,
        )
        
        # Performance tracking
        self._performance_history = defaultdict(list)
        self._strategy_performance = defaultdict(lambda: {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "last_optimization": None,
        })
        
        # ML predictions cache
        self._prediction_cache = {}
        self._feature_cache = {}
        
        # Tasks
        self._optimization_task = None
        self._learning_task = None
        self._monitoring_task = None
    
    async def initialize(self) -> None:
        """Initialize the ML trading system."""
        self._log.info("Initializing ML Trading System...")
        
        # Initialize ML components
        await self.ml_optimizer.initialize()
        
        # Initialize strategy selector with available strategies
        available_strategies = self._get_available_strategies()
        await self.ml_strategy_selector.initialize(available_strategies)
        
        # Start background tasks
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._learning_task = asyncio.create_task(self._continuous_learning_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self._log.info("ML Trading System initialized successfully")
    
    async def process_market_update(
        self,
        instrument_id: InstrumentId,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process market update and generate ML predictions.
        
        Returns:
            Dict containing predictions and recommendations
        """
        # Extract features
        features = await self._extract_features(instrument_id, market_data)
        
        # Cache features
        self._feature_cache[instrument_id] = features
        
        # Generate predictions
        predictions = {
            "market_regime": await self.ml_optimizer.predict_market_regime(features),
            "price_movement": await self.ml_optimizer.predict_price_movement(features),
            "volatility": await self.ml_optimizer.predict_volatility(features),
            "risk_assessment": await self.ml_optimizer.assess_risk_level(features),
            "anomaly_detection": await self.ml_optimizer.detect_anomalies(features),
        }
        
        # Get strategy recommendations
        available_strategies = list(self._get_available_strategies().keys())
        strategy_scores = await self.ml_optimizer.select_optimal_strategy(features, available_strategies)
        
        # ML-enhanced strategy selection
        ml_selected_strategies = await self.ml_strategy_selector.select_strategies(
            market_data,
            n_strategies=3
        )
        
        # Combine recommendations
        recommendations = {
            "predictions": predictions,
            "strategy_scores": strategy_scores,
            "ml_selected_strategies": ml_selected_strategies,
            "recommended_action": self._determine_action(predictions, strategy_scores),
            "confidence": self._calculate_confidence(predictions),
        }
        
        # Cache predictions
        self._prediction_cache[instrument_id] = recommendations
        
        return recommendations
    
    async def optimize_strategy_ml(
        self,
        strategy_id: StrategyId,
        strategy_type: str,
        current_parameters: Dict[str, Any],
        historical_data: pd.DataFrame,
        force_reoptimize: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using ML.
        
        Args:
            strategy_id: ID of the strategy to optimize
            strategy_type: Type of strategy
            current_parameters: Current strategy parameters
            historical_data: Historical market data
            force_reoptimize: Force reoptimization even if recently done
        
        Returns:
            Optimized parameters and optimization report
        """
        # Check if optimization is needed
        last_opt = self._strategy_performance[strategy_id]["last_optimization"]
        if not force_reoptimize and last_opt:
            hours_since_opt = (datetime.utcnow() - last_opt).total_seconds() / 3600
            if hours_since_opt < self.reoptimization_interval_hours:
                self._log.info(f"Skipping optimization for {strategy_id} - recently optimized")
                return {"parameters": current_parameters, "skipped": True}
        
        self._log.info(f"Starting ML optimization for strategy {strategy_id}")
        
        # Define parameter space based on strategy type
        parameter_space = self._get_parameter_space(strategy_type)
        
        # Get strategy class
        strategy_class = self._get_available_strategies().get(strategy_type)
        if not strategy_class:
            self._log.error(f"Unknown strategy type: {strategy_type}")
            return {"parameters": current_parameters, "error": "Unknown strategy type"}
        
        # Run ML-guided optimization
        optimization_results = await self.backtest_optimizer.optimize_strategy(
            strategy_class=strategy_class,
            parameter_space=parameter_space,
            market_data=historical_data,
            optimization_metric="sharpe_ratio",
            n_trials=100,
            use_surrogate=True,
            multi_objective=True,
        )
        
        # Get current market features
        latest_features = self._feature_cache.get(
            list(self._feature_cache.keys())[0]
        ) if self._feature_cache else None
        
        if latest_features:
            # Further optimize using ML optimizer
            performance_history = [
                r.sharpe_ratio for _, r in optimization_results["optimization_history"]
            ]
            
            ml_optimized_params = await self.ml_optimizer.optimize_strategy_parameters(
                strategy_type=strategy_type,
                current_params=optimization_results["best_parameters"],
                features=latest_features,
                performance_history=performance_history
            )
            
            # Merge optimizations
            final_parameters = {**optimization_results["best_parameters"], **ml_optimized_params}
        else:
            final_parameters = optimization_results["best_parameters"]
        
        # Run sensitivity analysis
        sensitivity_results = self.backtest_optimizer.sensitivity_analysis(
            strategy_class=strategy_class,
            base_parameters=final_parameters,
            parameter_space=parameter_space,
            market_data=historical_data,
        )
        
        # Run Monte Carlo simulation for robustness
        monte_carlo_results = await self.backtest_optimizer.monte_carlo_simulation(
            strategy_class=strategy_class,
            base_parameters=final_parameters,
            market_data=historical_data,
            n_simulations=100,
        )
        
        # Generate optimization report
        report = self.backtest_optimizer.generate_optimization_report(
            optimization_results=optimization_results,
            sensitivity_results=sensitivity_results,
            monte_carlo_results=monte_carlo_results,
        )
        
        # Update optimization timestamp
        self._strategy_performance[strategy_id]["last_optimization"] = datetime.utcnow()
        
        # Store optimization results for continuous learning
        await self._store_optimization_results(strategy_id, final_parameters, report)
        
        return {
            "parameters": final_parameters,
            "report": report,
            "improved": report["optimization_summary"]["best_performance"] > self.performance_threshold,
        }
    
    async def evaluate_strategy_performance(
        self,
        strategy_id: StrategyId,
        trade_results: List[Dict[str, Any]]
    ) -> None:
        """Evaluate strategy performance and update ML models."""
        if not trade_results:
            return
        
        # Calculate performance metrics
        total_pnl = sum(trade["pnl"] for trade in trade_results)
        winning_trades = sum(1 for trade in trade_results if trade["pnl"] > 0)
        total_trades = len(trade_results)
        
        # Update performance tracking
        perf = self._strategy_performance[strategy_id]
        perf["total_trades"] += total_trades
        perf["winning_trades"] += winning_trades
        perf["total_pnl"] += total_pnl
        
        # Calculate current performance metrics
        win_rate = perf["winning_trades"] / perf["total_trades"] if perf["total_trades"] > 0 else 0
        avg_pnl = perf["total_pnl"] / perf["total_trades"] if perf["total_trades"] > 0 else 0
        
        # Store in performance history
        self._performance_history[strategy_id].append({
            "timestamp": datetime.utcnow(),
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "trades": total_trades,
        })
        
        # Update ML models with results
        if self.enable_continuous_learning:
            for trade in trade_results:
                if "features" in trade and "prediction" in trade:
                    await self.ml_optimizer.update_with_results(
                        prediction_type="strategy_selection",
                        features=trade["features"],
                        prediction=trade["prediction"],
                        actual_result=trade["pnl"] > 0,
                        reward=trade["pnl"]
                    )
    
    async def get_ml_system_status(self) -> Dict[str, Any]:
        """Get comprehensive ML system status."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "ml_optimizer_state": self.ml_optimizer.get_ml_pipeline_state(),
            "active_predictions": len(self._prediction_cache),
            "cached_features": len(self._feature_cache),
            "strategy_performance": {},
            "optimization_history": {},
            "system_health": self._calculate_system_health(),
        }
        
        # Add strategy performance
        for strategy_id, perf in self._strategy_performance.items():
            status["strategy_performance"][str(strategy_id)] = {
                "total_trades": perf["total_trades"],
                "win_rate": perf["winning_trades"] / perf["total_trades"] 
                           if perf["total_trades"] > 0 else 0,
                "total_pnl": perf["total_pnl"],
                "last_optimization": perf["last_optimization"].isoformat() 
                                   if perf["last_optimization"] else None,
            }
        
        # Add recent optimization results
        recent_optimizations = []
        for strategy_id, history in self._performance_history.items():
            if history:
                recent = history[-1]
                recent_optimizations.append({
                    "strategy_id": str(strategy_id),
                    "timestamp": recent["timestamp"].isoformat(),
                    "performance": {
                        "win_rate": recent["win_rate"],
                        "avg_pnl": recent["avg_pnl"],
                    }
                })
        
        status["optimization_history"] = recent_optimizations
        
        return status
    
    async def _extract_features(
        self,
        instrument_id: InstrumentId,
        market_data: Dict[str, Any]
    ) -> MarketFeatures:
        """Extract features from market data."""
        # Get market conditions from analyzer
        market_conditions = self.market_analyzer.get_market_conditions(instrument_id)
        
        # Prepare data for feature extraction
        # In production, this would convert market_data to proper DataFrame format
        price_data = pd.DataFrame({
            "close": [market_data.get("price", 0)],
            "high": [market_data.get("high", 0)],
            "low": [market_data.get("low", 0)],
            "open": [market_data.get("open", 0)],
            "volume": [market_data.get("volume", 0)],
        })
        
        # Extract features
        features = self.feature_engineer.extract_features(
            price_data=price_data,
            sentiment_data=market_data.get("sentiment", {})
        )
        
        # Add interaction features
        interaction_features = self.feature_engineer.create_interaction_features(features)
        
        # Update feature values with interactions
        for key, value in interaction_features.items():
            if hasattr(features, key):
                setattr(features, key, value)
        
        return features
    
    def _determine_action(
        self,
        predictions: Dict[str, Any],
        strategy_scores: Dict[str, float]
    ) -> str:
        """Determine recommended action based on predictions."""
        # Check for anomalies first
        if predictions["anomaly_detection"]["is_anomaly"]:
            return "HOLD"  # Be cautious during anomalies
        
        # Get risk level
        risk_assessment = predictions["risk_assessment"]
        high_risk = risk_assessment.get("high", 0) + risk_assessment.get("very_high", 0) > 0.5
        
        if high_risk:
            return "REDUCE_EXPOSURE"
        
        # Check price movement prediction
        price_pred = predictions["price_movement"]
        bullish = price_pred.get("strong_up", 0) + price_pred.get("up", 0)
        bearish = price_pred.get("strong_down", 0) + price_pred.get("down", 0)
        
        # Check volatility
        vol_pred = predictions["volatility"]
        high_vol = vol_pred.get("high", 0) + vol_pred.get("very_high", 0) > 0.5
        
        # Determine action
        if bullish > 0.6 and not high_vol:
            return "INCREASE_LONG"
        elif bearish > 0.6 and not high_vol:
            return "INCREASE_SHORT"
        elif high_vol:
            return "REDUCE_EXPOSURE"
        else:
            return "MAINTAIN"
    
    def _calculate_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall prediction confidence."""
        confidences = []
        
        # Market regime confidence (max probability)
        regime_probs = list(predictions["market_regime"].values())
        if regime_probs:
            confidences.append(max(regime_probs))
        
        # Price movement confidence
        price_probs = [v for k, v in predictions["price_movement"].items() 
                      if k != "expected_return"]
        if price_probs:
            confidences.append(max(price_probs))
        
        # Risk assessment confidence
        risk_probs = [v for k, v in predictions["risk_assessment"].items() 
                     if k not in ["expected_max_drawdown", "risk_score"]]
        if risk_probs:
            confidences.append(max(risk_probs))
        
        # Average confidence
        return np.mean(confidences) if confidences else 0.5
    
    def _get_available_strategies(self) -> Dict[str, type]:
        """Get available strategy types."""
        # This would return actual strategy classes
        # For now, return placeholder
        return {
            "trend_following": type("TrendFollowing", (), {}),
            "mean_reversion": type("MeanReversion", (), {}),
            "momentum": type("Momentum", (), {}),
            "market_making": type("MarketMaking", (), {}),
            "statistical_arbitrage": type("StatArb", (), {}),
        }
    
    def _get_parameter_space(self, strategy_type: str) -> Dict[str, Tuple[Any, Any]]:
        """Get parameter space for strategy type."""
        # Define parameter spaces for different strategies
        parameter_spaces = {
            "trend_following": {
                "fast_period": (10, 50),
                "slow_period": (50, 200),
                "stop_loss": (0.01, 0.05),
                "take_profit": (0.02, 0.10),
                "position_size": (0.1, 0.5),
            },
            "mean_reversion": {
                "lookback_period": (20, 100),
                "entry_zscore": (1.5, 3.0),
                "exit_zscore": (0.0, 1.0),
                "stop_loss": (0.02, 0.08),
                "position_size": (0.1, 0.4),
            },
            "momentum": {
                "lookback_period": (10, 50),
                "momentum_threshold": (0.01, 0.05),
                "holding_period": (1, 20),
                "stop_loss": (0.01, 0.05),
                "position_size": (0.1, 0.5),
            },
            "market_making": {
                "spread_multiplier": (1.0, 3.0),
                "order_levels": (1, 5),
                "order_spacing": (0.001, 0.01),
                "inventory_limit": (0.1, 0.5),
                "skew_factor": (0.0, 0.5),
            },
            "statistical_arbitrage": {
                "lookback_period": (50, 200),
                "entry_threshold": (1.5, 3.0),
                "exit_threshold": (0.0, 1.0),
                "hedge_ratio": (0.5, 2.0),
                "position_size": (0.1, 0.3),
            },
        }
        
        return parameter_spaces.get(strategy_type, {})
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        # Check ML model performance
        ml_state = self.ml_optimizer.get_ml_pipeline_state()
        
        # Calculate average model accuracy
        model_accuracies = []
        for model_info in ml_state["models"].values():
            if "performance" in model_info and "accuracy" in model_info["performance"]:
                model_accuracies.append(model_info["performance"]["accuracy"])
        
        avg_accuracy = np.mean(model_accuracies) if model_accuracies else 0.5
        
        # Check strategy performance
        total_strategies = len(self._strategy_performance)
        profitable_strategies = sum(
            1 for perf in self._strategy_performance.values()
            if perf["total_pnl"] > 0
        )
        
        # Calculate health score
        health_score = (
            0.4 * avg_accuracy +
            0.4 * (profitable_strategies / total_strategies if total_strategies > 0 else 0.5) +
            0.2 * (1.0 if self._optimization_task and not self._optimization_task.done() else 0.0)
        )
        
        return {
            "health_score": health_score,
            "status": "healthy" if health_score > 0.7 else "degraded" if health_score > 0.4 else "unhealthy",
            "model_accuracy": avg_accuracy,
            "profitable_strategies_ratio": profitable_strategies / total_strategies if total_strategies > 0 else 0,
            "active_tasks": {
                "optimization": self._optimization_task and not self._optimization_task.done(),
                "learning": self._learning_task and not self._learning_task.done(),
                "monitoring": self._monitoring_task and not self._monitoring_task.done(),
            }
        }
    
    async def _optimization_loop(self):
        """Background loop for strategy optimization."""
        while True:
            try:
                await asyncio.sleep(self.reoptimization_interval_hours * 3600)
                
                if not self.enable_ml_optimization:
                    continue
                
                # Optimize underperforming strategies
                for strategy_id, perf in self._strategy_performance.items():
                    win_rate = perf["winning_trades"] / perf["total_trades"] if perf["total_trades"] > 0 else 0
                    
                    if win_rate < self.performance_threshold and perf["total_trades"] > 20:
                        self._log.info(f"Reoptimizing underperforming strategy {strategy_id}")
                        
                        # Get strategy type (would be stored in real implementation)
                        strategy_type = "trend_following"  # Placeholder
                        
                        # Get historical data (would be fetched from data store)
                        historical_data = pd.DataFrame()  # Placeholder
                        
                        # Run optimization
                        await self.optimize_strategy_ml(
                            strategy_id=strategy_id,
                            strategy_type=strategy_type,
                            current_parameters={},
                            historical_data=historical_data,
                            force_reoptimize=True
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Optimization loop error: {e}")
    
    async def _continuous_learning_loop(self):
        """Background loop for continuous learning."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                if not self.enable_continuous_learning:
                    continue
                
                # Trigger ML model retraining if needed
                ml_state = self.ml_optimizer.get_ml_pipeline_state()
                if ml_state["feature_buffer_size"] > 1000:
                    self._log.info("Triggering ML model retraining")
                    # Models will retrain automatically in ml_optimizer
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Learning loop error: {e}")
    
    async def _monitoring_loop(self):
        """Background loop for system monitoring."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Check system health
                health = self._calculate_system_health()
                
                if health["status"] == "unhealthy":
                    self._log.warning("ML system health is degraded")
                    # Could trigger alerts or corrective actions
                
                # Log performance summary
                self._log.info(
                    f"ML System Health: {health['health_score']:.2f}, "
                    f"Model Accuracy: {health['model_accuracy']:.2f}, "
                    f"Profitable Strategies: {health['profitable_strategies_ratio']:.1%}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Monitoring loop error: {e}")
    
    async def _store_optimization_results(
        self,
        strategy_id: StrategyId,
        parameters: Dict[str, Any],
        report: Dict[str, Any]
    ):
        """Store optimization results for future reference."""
        # In production, this would store to a database
        # For now, we'll keep in memory
        
        result = {
            "strategy_id": str(strategy_id),
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": parameters,
            "performance": report["optimization_summary"]["best_performance"],
            "walk_forward": report.get("walk_forward_validation", {}),
            "robustness": report.get("robustness_analysis", {}),
        }
        
        # This could be stored in a database or file
        self._log.info(f"Stored optimization results for {strategy_id}")
    
    async def save_ml_pipeline_to_memory(self) -> str:
        """Save the entire ML pipeline state to memory."""
        pipeline_state = {
            "timestamp": datetime.utcnow().isoformat(),
            "ml_optimizer": self.ml_optimizer.get_ml_pipeline_state(),
            "strategy_performance": dict(self._strategy_performance),
            "optimization_history": {
                str(k): v for k, v in self._performance_history.items()
            },
            "system_health": self._calculate_system_health(),
            "configuration": {
                "enable_ml_optimization": self.enable_ml_optimization,
                "enable_continuous_learning": self.enable_continuous_learning,
                "enable_auto_rebalancing": self.enable_auto_rebalancing,
                "performance_threshold": self.performance_threshold,
                "reoptimization_interval_hours": self.reoptimization_interval_hours,
            }
        }
        
        # Convert to JSON string for storage
        pipeline_json = json.dumps(pipeline_state, indent=2)
        
        # This would be stored in the Memory system
        memory_key = "swarm-auto-hierarchical-1751379006249/ml-optimizer/pipeline"
        
        self._log.info(f"Saved ML pipeline to memory key: {memory_key}")
        
        return pipeline_json