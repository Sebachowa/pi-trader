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
Performance Optimizer - Self-optimizing performance monitoring and parameter tuning.
"""

import asyncio
import json
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.logging import Logger
from nautilus_trader.model.identifiers import StrategyId


class ParameterSpace:
    """Define parameter search space for optimization."""
    
    def __init__(self, name: str, min_value: float, max_value: float, 
                 param_type: str = "float", step: Optional[float] = None):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.param_type = param_type
        self.step = step
        self.current_value = (min_value + max_value) / 2
        self.best_value = self.current_value
        self.optimization_history = []


class PerformanceOptimizer(Component):
    """
    Self-optimizing performance monitoring and parameter tuning system.
    
    Features:
    - Real-time performance tracking
    - Bayesian optimization for parameter tuning
    - Multi-objective optimization
    - Performance attribution analysis
    - Machine learning-based optimization
    """
    
    def __init__(
        self,
        logger: Logger,
        clock: LiveClock,
        msgbus: MessageBus,
        optimization_interval_hours: int = 24,
        min_samples_for_optimization: int = 50,
        exploration_fraction: float = 0.2,
        performance_window_days: int = 30,
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id="PERFORMANCE-OPTIMIZER",
            msgbus=msgbus,
        )
        
        self.optimization_interval_hours = optimization_interval_hours
        self.min_samples_for_optimization = min_samples_for_optimization
        self.exploration_fraction = exploration_fraction
        self.performance_window_days = performance_window_days
        
        # Performance tracking
        self._trade_history = deque(maxlen=10000)
        self._daily_returns = deque(maxlen=252)
        self._strategy_performance: Dict[StrategyId, Dict[str, Any]] = defaultdict(dict)
        
        # Parameter optimization
        self._parameter_spaces: Dict[str, ParameterSpace] = {}
        self._optimization_results: Dict[str, List[Tuple[Dict, float]]] = defaultdict(list)
        self._gaussian_processes: Dict[str, GaussianProcessRegressor] = {}
        
        # Performance metrics
        self._performance_metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_duration": timedelta(0),
            "trades_per_day": 0.0,
        }
        
        # Attribution analysis
        self._factor_contributions = defaultdict(float)
        self._strategy_contributions = defaultdict(float)
        
        # Optimization task
        self._optimization_task = None
        self._analysis_task = None

    async def start(self) -> None:
        """Start the performance optimizer."""
        self._log.info("Starting Performance Optimizer...")
        
        # Initialize parameter spaces
        self._initialize_parameter_spaces()
        
        # Start optimization and analysis tasks
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())

    async def stop(self) -> None:
        """Stop the performance optimizer."""
        self._log.info("Stopping Performance Optimizer...")
        
        if self._optimization_task:
            self._optimization_task.cancel()
        if self._analysis_task:
            self._analysis_task.cancel()

    def _initialize_parameter_spaces(self) -> None:
        """Initialize parameter search spaces for optimization."""
        # Risk parameters
        self._parameter_spaces["position_size_multiplier"] = ParameterSpace(
            "position_size_multiplier", 0.5, 2.0
        )
        self._parameter_spaces["stop_loss_multiplier"] = ParameterSpace(
            "stop_loss_multiplier", 0.5, 3.0
        )
        self._parameter_spaces["take_profit_multiplier"] = ParameterSpace(
            "take_profit_multiplier", 1.0, 5.0
        )
        
        # Strategy parameters
        self._parameter_spaces["fast_ema_period"] = ParameterSpace(
            "fast_ema_period", 5, 50, "int", 1
        )
        self._parameter_spaces["slow_ema_period"] = ParameterSpace(
            "slow_ema_period", 20, 200, "int", 5
        )
        self._parameter_spaces["volatility_lookback"] = ParameterSpace(
            "volatility_lookback", 10, 100, "int", 5
        )
        
        # ML parameters
        self._parameter_spaces["confidence_threshold"] = ParameterSpace(
            "confidence_threshold", 0.5, 0.95, "float", 0.05
        )
        self._parameter_spaces["ensemble_size"] = ParameterSpace(
            "ensemble_size", 3, 10, "int", 1
        )

    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record a completed trade for analysis."""
        trade_data["timestamp"] = datetime.utcnow()
        self._trade_history.append(trade_data)
        
        # Update strategy-specific performance
        strategy_id = trade_data.get("strategy_id")
        if strategy_id:
            self._update_strategy_performance(strategy_id, trade_data)

    def _update_strategy_performance(self, strategy_id: StrategyId, trade_data: Dict[str, Any]) -> None:
        """Update performance metrics for a specific strategy."""
        if strategy_id not in self._strategy_performance:
            self._strategy_performance[strategy_id] = {
                "trades": [],
                "total_pnl": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "total_return": 0.0,
                "peak_value": 1.0,
                "drawdown": 0.0,
            }
        
        perf = self._strategy_performance[strategy_id]
        perf["trades"].append(trade_data)
        
        pnl = trade_data.get("pnl", 0)
        perf["total_pnl"] += pnl
        
        if pnl > 0:
            perf["win_count"] += 1
        else:
            perf["loss_count"] += 1
        
        # Update return and drawdown
        perf["total_return"] = perf["total_pnl"] / 10000  # Assuming 10k starting capital
        current_value = 1 + perf["total_return"]
        
        if current_value > perf["peak_value"]:
            perf["peak_value"] = current_value
        
        perf["drawdown"] = (perf["peak_value"] - current_value) / perf["peak_value"]

    async def _optimization_loop(self) -> None:
        """Main optimization loop using Bayesian optimization."""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval_hours * 3600)
                
                # Check if we have enough data
                if len(self._trade_history) < self.min_samples_for_optimization:
                    self._log.info("Not enough trades for optimization yet")
                    continue
                
                # Run optimization for each parameter space
                for param_name, param_space in self._parameter_spaces.items():
                    await self._optimize_parameter(param_name, param_space)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Optimization error: {e}")

    async def _optimize_parameter(self, param_name: str, param_space: ParameterSpace) -> None:
        """Optimize a single parameter using Bayesian optimization."""
        self._log.info(f"Optimizing parameter: {param_name}")
        
        # Get historical results for this parameter
        historical_results = self._optimization_results.get(param_name, [])
        
        # Initialize or update Gaussian Process
        if param_name not in self._gaussian_processes or len(historical_results) < 5:
            # Use random exploration initially
            next_value = self._random_exploration(param_space)
        else:
            # Use Bayesian optimization
            next_value = await self._bayesian_optimization(param_name, param_space)
        
        # Set the parameter to the new value
        param_space.current_value = next_value
        param_space.optimization_history.append({
            "timestamp": datetime.utcnow(),
            "value": next_value,
        })
        
        self._log.info(f"Set {param_name} to {next_value}")

    def _random_exploration(self, param_space: ParameterSpace) -> float:
        """Random exploration within parameter bounds."""
        if param_space.param_type == "int":
            return np.random.randint(param_space.min_value, param_space.max_value + 1)
        else:
            return np.random.uniform(param_space.min_value, param_space.max_value)

    async def _bayesian_optimization(self, param_name: str, param_space: ParameterSpace) -> float:
        """Bayesian optimization using Gaussian Process."""
        # Get historical data
        historical_results = self._optimization_results[param_name]
        
        if not historical_results:
            return self._random_exploration(param_space)
        
        # Prepare data for GP
        X = np.array([[result[0][param_name]] for result in historical_results])
        y = np.array([result[1] for result in historical_results])  # Performance metric
        
        # Fit or update Gaussian Process
        if param_name not in self._gaussian_processes:
            kernel = Matern(length_scale=1.0, nu=2.5)
            self._gaussian_processes[param_name] = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )
        
        gp = self._gaussian_processes[param_name]
        gp.fit(X, y)
        
        # Generate candidate points
        if param_space.param_type == "int":
            candidates = np.arange(
                param_space.min_value, 
                param_space.max_value + 1, 
                param_space.step or 1
            ).reshape(-1, 1)
        else:
            candidates = np.linspace(
                param_space.min_value,
                param_space.max_value,
                100
            ).reshape(-1, 1)
        
        # Predict mean and variance
        mu, sigma = gp.predict(candidates, return_std=True)
        
        # Calculate acquisition function (Upper Confidence Bound)
        with np.errstate(divide="warn"):
            exploration_weight = 2.0  # Controls exploration vs exploitation
            ucb = mu + exploration_weight * sigma
        
        # Select best candidate
        best_idx = np.argmax(ucb)
        next_value = float(candidates[best_idx])
        
        # Apply epsilon-greedy exploration
        if np.random.random() < self.exploration_fraction:
            next_value = self._random_exploration(param_space)
        
        return next_value

    async def _analysis_loop(self) -> None:
        """Continuous performance analysis loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._analyze_performance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Analysis error: {e}")

    async def _analyze_performance(self) -> None:
        """Analyze overall system performance."""
        if len(self._trade_history) < 10:
            return
        
        trades = list(self._trade_history)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(trades)
        
        # Perform attribution analysis
        await self._perform_attribution_analysis(trades)
        
        # Update optimization objectives
        await self._update_optimization_objectives()

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> None:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return
        
        # Extract PnL series
        pnls = [trade.get("pnl", 0) for trade in trades]
        returns = [trade.get("return", 0) for trade in trades]
        
        # Win rate
        wins = sum(1 for pnl in pnls if pnl > 0)
        self._performance_metrics["win_rate"] = wins / len(pnls) if pnls else 0
        
        # Profit factor
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        self._performance_metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )
        
        # Total return
        self._performance_metrics["total_return"] = sum(returns)
        
        # Sharpe ratio (simplified)
        if len(returns) > 1:
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array) * 252  # Annualized
            std_return = np.std(returns_array) * np.sqrt(252)
            self._performance_metrics["sharpe_ratio"] = (
                mean_return / std_return if std_return > 0 else 0
            )
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / running_max
        self._performance_metrics["max_drawdown"] = np.max(drawdown)
        
        # Average trade duration
        durations = []
        for trade in trades:
            if "entry_time" in trade and "exit_time" in trade:
                duration = trade["exit_time"] - trade["entry_time"]
                durations.append(duration)
        
        if durations:
            avg_duration = sum(durations, timedelta(0)) / len(durations)
            self._performance_metrics["avg_trade_duration"] = avg_duration
        
        # Trades per day
        if trades:
            first_trade = min(trades, key=lambda x: x.get("timestamp", datetime.max))
            last_trade = max(trades, key=lambda x: x.get("timestamp", datetime.min))
            days_trading = (last_trade["timestamp"] - first_trade["timestamp"]).days
            self._performance_metrics["trades_per_day"] = (
                len(trades) / days_trading if days_trading > 0 else 0
            )

    async def _perform_attribution_analysis(self, trades: List[Dict[str, Any]]) -> None:
        """Perform performance attribution analysis."""
        # Reset contributions
        self._factor_contributions.clear()
        self._strategy_contributions.clear()
        
        total_pnl = sum(trade.get("pnl", 0) for trade in trades)
        
        if total_pnl == 0:
            return
        
        # Strategy attribution
        strategy_pnls = defaultdict(float)
        for trade in trades:
            strategy_id = trade.get("strategy_id")
            if strategy_id:
                strategy_pnls[strategy_id] += trade.get("pnl", 0)
        
        for strategy_id, pnl in strategy_pnls.items():
            self._strategy_contributions[strategy_id] = pnl / total_pnl
        
        # Factor attribution (simplified)
        # This would typically involve more sophisticated factor models
        factor_pnls = {
            "momentum": 0.0,
            "mean_reversion": 0.0,
            "volatility": 0.0,
            "liquidity": 0.0,
        }
        
        for trade in trades:
            # Simple heuristic attribution
            if trade.get("strategy_type") == "trend_following":
                factor_pnls["momentum"] += trade.get("pnl", 0)
            elif trade.get("strategy_type") == "mean_reversion":
                factor_pnls["mean_reversion"] += trade.get("pnl", 0)
            
            # Volatility attribution
            if trade.get("volatility_regime") == "high":
                factor_pnls["volatility"] += trade.get("pnl", 0) * 0.3
        
        for factor, pnl in factor_pnls.items():
            if pnl != 0:
                self._factor_contributions[factor] = pnl / total_pnl

    async def _update_optimization_objectives(self) -> None:
        """Update optimization objectives based on current performance."""
        # Calculate composite objective function value
        sharpe_weight = 0.4
        pf_weight = 0.3
        dd_weight = 0.3
        
        objective_value = (
            sharpe_weight * min(self._performance_metrics["sharpe_ratio"] / 2.0, 1.0) +
            pf_weight * min(self._performance_metrics["profit_factor"] / 2.0, 1.0) +
            dd_weight * (1 - min(self._performance_metrics["max_drawdown"] / 0.2, 1.0))
        )
        
        # Store current parameter values with objective
        current_params = {
            name: space.current_value 
            for name, space in self._parameter_spaces.items()
        }
        
        for param_name in self._parameter_spaces:
            self._optimization_results[param_name].append(
                (current_params, objective_value)
            )
            
            # Keep only recent results
            if len(self._optimization_results[param_name]) > 100:
                self._optimization_results[param_name].pop(0)

    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get current optimal parameter values."""
        return {
            name: space.current_value
            for name, space in self._parameter_spaces.items()
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance_metrics": {
                "total_return": round(self._performance_metrics["total_return"], 4),
                "sharpe_ratio": round(self._performance_metrics["sharpe_ratio"], 2),
                "max_drawdown": round(self._performance_metrics["max_drawdown"], 4),
                "win_rate": round(self._performance_metrics["win_rate"], 3),
                "profit_factor": round(self._performance_metrics["profit_factor"], 2),
                "avg_trade_duration": str(self._performance_metrics["avg_trade_duration"]),
                "trades_per_day": round(self._performance_metrics["trades_per_day"], 1),
            },
            "attribution": {
                "strategy_contributions": {
                    str(k): round(v, 3) 
                    for k, v in self._strategy_contributions.items()
                },
                "factor_contributions": {
                    k: round(v, 3) 
                    for k, v in self._factor_contributions.items()
                },
            },
            "optimal_parameters": self.get_optimal_parameters(),
            "total_trades_analyzed": len(self._trade_history),
        }

    def export_optimization_results(self, filepath: str) -> None:
        """Export optimization results to JSON file."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "parameter_spaces": {
                name: {
                    "min": space.min_value,
                    "max": space.max_value,
                    "current": space.current_value,
                    "best": space.best_value,
                    "history": space.optimization_history,
                }
                for name, space in self._parameter_spaces.items()
            },
            "optimization_results": {
                param: [
                    {"params": params, "objective": obj}
                    for params, obj in results
                ]
                for param, results in self._optimization_results.items()
            },
            "performance_metrics": self._performance_metrics,
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)