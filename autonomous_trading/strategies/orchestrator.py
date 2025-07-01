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
Strategy Orchestrator - AI-powered multi-strategy management for autonomous trading.
"""

import asyncio
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.logging import Logger
from nautilus_trader.model.identifiers import InstrumentId, StrategyId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.ai.strategies.ai_swarm_strategy import AISwarmStrategy
from autonomous_trading.core.market_analyzer import MarketRegime


class StrategyHealth(Enum):
    """Strategy health status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class StrategyMetrics:
    """Performance metrics for a strategy."""
    
    def __init__(self, strategy_id: StrategyId):
        self.strategy_id = strategy_id
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.last_trade_time = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.health = StrategyHealth.GOOD
        self.allocation_weight = 0.0
        self.returns_history = deque(maxlen=100)


class StrategyOrchestrator(Component):
    """
    AI-powered strategy orchestrator for autonomous multi-strategy management.
    
    Features:
    - Dynamic strategy allocation based on performance
    - Market condition-based strategy selection
    - Automatic strategy rotation
    - Performance-based weighting
    - Strategy health monitoring
    """
    
    def __init__(
        self,
        logger: Logger,
        clock: LiveClock,
        msgbus: MessageBus,
        max_concurrent_strategies: int = 5,
        min_strategy_allocation: float = 0.05,
        max_strategy_allocation: float = 0.30,
        performance_lookback_days: int = 30,
        rebalance_interval_hours: int = 24,
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id="STRATEGY-ORCHESTRATOR",
            msgbus=msgbus,
        )
        
        self.max_concurrent_strategies = max_concurrent_strategies
        self.min_strategy_allocation = min_strategy_allocation
        self.max_strategy_allocation = max_strategy_allocation
        self.performance_lookback_days = performance_lookback_days
        self.rebalance_interval_hours = rebalance_interval_hours
        
        # Strategy management
        self._available_strategies: Dict[str, Type[Strategy]] = {}
        self._active_strategies: Dict[StrategyId, Strategy] = {}
        self._strategy_metrics: Dict[StrategyId, StrategyMetrics] = {}
        self._strategy_allocations: Dict[StrategyId, float] = {}
        
        # Market condition mapping
        self._regime_strategy_map: Dict[MarketRegime, List[str]] = {
            MarketRegime.TRENDING_UP: ["trend_following", "momentum", "ai_swarm"],
            MarketRegime.TRENDING_DOWN: ["trend_following", "short_strategies", "ai_swarm"],
            MarketRegime.RANGING: ["mean_reversion", "market_making", "statistical_arbitrage"],
            MarketRegime.VOLATILE: ["volatility_strategies", "options_strategies", "ai_swarm"],
            MarketRegime.QUIET: ["market_making", "carry_strategies", "statistical_arbitrage"],
            MarketRegime.BREAKOUT: ["breakout_strategies", "momentum", "ai_swarm"],
            MarketRegime.BREAKDOWN: ["breakdown_strategies", "short_strategies", "ai_swarm"],
        }
        
        # Performance tracking
        self._portfolio_performance = {
            "total_pnl": 0.0,
            "daily_returns": deque(maxlen=252),
            "strategy_contributions": defaultdict(float),
        }
        
        # Tasks
        self._rebalance_task = None
        self._monitoring_task = None

    async def start(self) -> None:
        """Start the strategy orchestrator."""
        self._log.info("Starting Strategy Orchestrator...")
        
        # Register available strategies
        await self._register_strategies()
        
        # Start monitoring and rebalancing tasks
        self._rebalance_task = asyncio.create_task(self._rebalance_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop(self) -> None:
        """Stop the strategy orchestrator."""
        self._log.info("Stopping Strategy Orchestrator...")
        
        # Cancel tasks
        if self._rebalance_task:
            self._rebalance_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Stop all active strategies
        for strategy in self._active_strategies.values():
            await self._stop_strategy(strategy)

    async def _register_strategies(self) -> None:
        """Register available trading strategies."""
        # Import strategy classes dynamically
        try:
            from nautilus_trader.examples.strategies.ema_cross import EMACross
            from nautilus_trader.ai.strategies.ai_market_maker import AIMarketMaker
            from nautilus_trader.ai.strategies.ai_trend_follower import AITrendFollower
            from nautilus_trader.ai.strategies.ai_arbitrage_bot import AIArbitrageBot
            
            self._available_strategies = {
                "trend_following": AITrendFollower,
                "market_making": AIMarketMaker,
                "statistical_arbitrage": AIArbitrageBot,
                "ai_swarm": AISwarmStrategy,
                "momentum": EMACross,  # Can be replaced with momentum strategy
                "mean_reversion": EMACross,  # Can be replaced with mean reversion
            }
            
            self._log.info(f"Registered {len(self._available_strategies)} strategies")
            
        except ImportError as e:
            self._log.warning(f"Could not import all strategies: {e}")

    async def select_strategies(
        self,
        market_conditions: Dict[InstrumentId, Any],
        portfolio_balance: float,
    ) -> List[Tuple[str, float]]:
        """Select optimal strategies based on market conditions and performance."""
        selected_strategies = []
        
        # Aggregate market regimes
        regime_counts = defaultdict(int)
        for conditions in market_conditions.values():
            if hasattr(conditions, 'regime'):
                regime_counts[conditions.regime] += 1
        
        # Find dominant regime
        if regime_counts:
            dominant_regime = max(regime_counts, key=regime_counts.get)
            recommended_strategies = self._regime_strategy_map.get(dominant_regime, [])
        else:
            recommended_strategies = ["ai_swarm"]  # Default to AI swarm
        
        # Score strategies based on performance and suitability
        strategy_scores = {}
        
        for strategy_name in recommended_strategies:
            if strategy_name not in self._available_strategies:
                continue
            
            # Calculate performance score
            perf_score = self._calculate_strategy_performance_score(strategy_name)
            
            # Calculate suitability score
            suit_score = self._calculate_strategy_suitability_score(
                strategy_name, 
                market_conditions
            )
            
            # Combined score
            total_score = 0.6 * perf_score + 0.4 * suit_score
            strategy_scores[strategy_name] = total_score
        
        # Sort by score and select top strategies
        sorted_strategies = sorted(
            strategy_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Allocate capital using Kelly-like criterion
        total_allocation = 0.0
        
        for strategy_name, score in sorted_strategies[:self.max_concurrent_strategies]:
            if total_allocation >= 0.95:  # Leave 5% cash buffer
                break
            
            # Calculate allocation based on score and constraints
            base_allocation = score * 0.3  # Max 30% per strategy
            allocation = max(
                self.min_strategy_allocation,
                min(base_allocation, self.max_strategy_allocation)
            )
            
            # Ensure we don't over-allocate
            allocation = min(allocation, 0.95 - total_allocation)
            
            if allocation >= self.min_strategy_allocation:
                selected_strategies.append((strategy_name, allocation))
                total_allocation += allocation
        
        self._log.info(f"Selected strategies: {selected_strategies}")
        return selected_strategies

    def _calculate_strategy_performance_score(self, strategy_name: str) -> float:
        """Calculate performance score for a strategy."""
        # Find metrics for this strategy type
        strategy_metrics = []
        
        for strategy_id, metrics in self._strategy_metrics.items():
            if strategy_name in str(strategy_id):
                strategy_metrics.append(metrics)
        
        if not strategy_metrics:
            return 0.5  # Neutral score for new strategies
        
        # Average metrics across instances
        avg_win_rate = np.mean([m.win_rate for m in strategy_metrics])
        avg_sharpe = np.mean([m.sharpe_ratio for m in strategy_metrics])
        avg_profit_factor = np.mean([m.profit_factor for m in strategy_metrics])
        
        # Calculate composite score
        win_rate_score = min(avg_win_rate / 0.6, 1.0)  # Target 60% win rate
        sharpe_score = min(avg_sharpe / 2.0, 1.0)  # Target Sharpe of 2.0
        pf_score = min(avg_profit_factor / 1.5, 1.0)  # Target PF of 1.5
        
        # Weight the scores
        performance_score = (
            0.3 * win_rate_score +
            0.4 * sharpe_score +
            0.3 * pf_score
        )
        
        return performance_score

    def _calculate_strategy_suitability_score(
        self,
        strategy_name: str,
        market_conditions: Dict[InstrumentId, Any],
    ) -> float:
        """Calculate how suitable a strategy is for current market conditions."""
        suitability_scores = []
        
        for instrument_id, conditions in market_conditions.items():
            if not hasattr(conditions, 'regime'):
                continue
            
            # Check if strategy is recommended for this regime
            recommended = strategy_name in self._regime_strategy_map.get(
                conditions.regime, []
            )
            
            if recommended:
                # High suitability for recommended strategies
                base_score = 0.8
                
                # Adjust based on market metrics
                if strategy_name == "market_making" and hasattr(conditions, 'liquidity'):
                    if conditions.liquidity.value in ["high", "very_high"]:
                        base_score = 0.9
                    elif conditions.liquidity.value in ["low", "very_low"]:
                        base_score = 0.3
                
                elif strategy_name == "trend_following" and hasattr(conditions, 'trend_strength'):
                    base_score = min(0.5 + conditions.trend_strength * 2, 1.0)
                
                elif strategy_name == "mean_reversion" and conditions.regime == MarketRegime.RANGING:
                    base_score = 0.9
                
                suitability_scores.append(base_score)
            else:
                # Lower suitability for non-recommended strategies
                suitability_scores.append(0.3)
        
        if suitability_scores:
            return np.mean(suitability_scores)
        return 0.5

    async def deploy_strategy(
        self,
        strategy_name: str,
        allocation: float,
        instruments: List[InstrumentId],
    ) -> Optional[StrategyId]:
        """Deploy a new strategy instance."""
        if strategy_name not in self._available_strategies:
            self._log.error(f"Strategy {strategy_name} not available")
            return None
        
        strategy_class = self._available_strategies[strategy_name]
        
        # Generate unique strategy ID
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        strategy_id = StrategyId(f"{strategy_name}_{timestamp}")
        
        try:
            # Create strategy configuration
            # This would need proper configuration based on strategy type
            config = self._create_strategy_config(
                strategy_name,
                strategy_id,
                instruments,
                allocation,
            )
            
            # Instantiate strategy
            strategy = strategy_class(config)
            
            # Initialize metrics
            self._strategy_metrics[strategy_id] = StrategyMetrics(strategy_id)
            self._strategy_allocations[strategy_id] = allocation
            
            # Register and start strategy
            self._active_strategies[strategy_id] = strategy
            # await strategy.start()  # Would need proper integration
            
            self._log.info(f"Deployed strategy {strategy_id} with {allocation:.1%} allocation")
            return strategy_id
            
        except Exception as e:
            self._log.error(f"Failed to deploy strategy {strategy_name}: {e}")
            return None

    async def _stop_strategy(self, strategy: Strategy) -> None:
        """Stop and cleanup a strategy."""
        try:
            # await strategy.stop()  # Would need proper integration
            self._log.info(f"Stopped strategy {strategy.id}")
        except Exception as e:
            self._log.error(f"Error stopping strategy {strategy.id}: {e}")

    async def _rebalance_loop(self) -> None:
        """Periodic portfolio rebalancing."""
        while True:
            try:
                await asyncio.sleep(self.rebalance_interval_hours * 3600)
                await self._rebalance_portfolio()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Rebalancing error: {e}")

    async def _rebalance_portfolio(self) -> None:
        """Rebalance strategy allocations based on performance."""
        self._log.info("Starting portfolio rebalance...")
        
        # Calculate current strategy weights based on performance
        strategy_weights = {}
        total_score = 0.0
        
        for strategy_id, metrics in self._strategy_metrics.items():
            if strategy_id not in self._active_strategies:
                continue
            
            # Calculate weight based on risk-adjusted returns
            if metrics.sharpe_ratio > 0 and metrics.total_trades > 10:
                weight = metrics.sharpe_ratio * (1 - metrics.max_drawdown)
                weight *= (1 + metrics.win_rate - 0.5)  # Boost for high win rate
                
                # Penalize strategies with consecutive losses
                if metrics.consecutive_losses > 3:
                    weight *= 0.7
                elif metrics.consecutive_losses > 5:
                    weight *= 0.5
                
                strategy_weights[strategy_id] = max(0, weight)
                total_score += max(0, weight)
        
        # Normalize weights
        if total_score > 0:
            for strategy_id in strategy_weights:
                normalized_weight = strategy_weights[strategy_id] / total_score
                
                # Apply constraints
                new_allocation = max(
                    self.min_strategy_allocation,
                    min(normalized_weight, self.max_strategy_allocation)
                )
                
                old_allocation = self._strategy_allocations.get(strategy_id, 0)
                
                # Only rebalance if change is significant
                if abs(new_allocation - old_allocation) > 0.05:
                    self._strategy_allocations[strategy_id] = new_allocation
                    self._log.info(
                        f"Rebalanced {strategy_id}: {old_allocation:.1%} -> {new_allocation:.1%}"
                    )

    async def _monitoring_loop(self) -> None:
        """Monitor strategy health and performance."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._monitor_strategies()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Monitoring error: {e}")

    async def _monitor_strategies(self) -> None:
        """Monitor and update strategy health status."""
        for strategy_id, metrics in self._strategy_metrics.items():
            if strategy_id not in self._active_strategies:
                continue
            
            # Update health status
            health = self._calculate_strategy_health(metrics)
            metrics.health = health
            
            # Take action based on health
            if health == StrategyHealth.CRITICAL:
                self._log.warning(f"Strategy {strategy_id} in critical condition")
                # Consider stopping or reducing allocation
                if metrics.consecutive_losses > 10:
                    await self._emergency_stop_strategy(strategy_id)
            
            elif health == StrategyHealth.POOR:
                self._log.warning(f"Strategy {strategy_id} performing poorly")
                # Reduce allocation
                current_allocation = self._strategy_allocations.get(strategy_id, 0)
                self._strategy_allocations[strategy_id] = current_allocation * 0.7

    def _calculate_strategy_health(self, metrics: StrategyMetrics) -> StrategyHealth:
        """Calculate strategy health based on metrics."""
        if metrics.total_trades < 5:
            return StrategyHealth.GOOD  # Not enough data
        
        # Health factors
        factors = []
        
        # Win rate factor
        if metrics.win_rate > 0.55:
            factors.append(1.0)
        elif metrics.win_rate > 0.45:
            factors.append(0.7)
        elif metrics.win_rate > 0.35:
            factors.append(0.4)
        else:
            factors.append(0.1)
        
        # Profit factor
        if metrics.profit_factor > 1.5:
            factors.append(1.0)
        elif metrics.profit_factor > 1.2:
            factors.append(0.7)
        elif metrics.profit_factor > 1.0:
            factors.append(0.4)
        else:
            factors.append(0.1)
        
        # Drawdown factor
        if metrics.max_drawdown < 0.05:
            factors.append(1.0)
        elif metrics.max_drawdown < 0.1:
            factors.append(0.7)
        elif metrics.max_drawdown < 0.2:
            factors.append(0.4)
        else:
            factors.append(0.1)
        
        # Consecutive losses factor
        if metrics.consecutive_losses == 0:
            factors.append(1.0)
        elif metrics.consecutive_losses < 3:
            factors.append(0.7)
        elif metrics.consecutive_losses < 5:
            factors.append(0.4)
        else:
            factors.append(0.1)
        
        # Calculate average health score
        health_score = np.mean(factors)
        
        if health_score > 0.8:
            return StrategyHealth.EXCELLENT
        elif health_score > 0.6:
            return StrategyHealth.GOOD
        elif health_score > 0.4:
            return StrategyHealth.FAIR
        elif health_score > 0.2:
            return StrategyHealth.POOR
        else:
            return StrategyHealth.CRITICAL

    async def _emergency_stop_strategy(self, strategy_id: StrategyId) -> None:
        """Emergency stop a failing strategy."""
        self._log.warning(f"Emergency stopping strategy {strategy_id}")
        
        if strategy_id in self._active_strategies:
            strategy = self._active_strategies[strategy_id]
            await self._stop_strategy(strategy)
            
            # Remove from active strategies
            del self._active_strategies[strategy_id]
            del self._strategy_allocations[strategy_id]

    def _create_strategy_config(self, strategy_name: str, strategy_id: StrategyId, 
                               instruments: List[InstrumentId], allocation: float) -> Any:
        """Create strategy configuration based on type."""
        # This would create proper configuration for each strategy type
        # Placeholder implementation
        return {
            "strategy_id": strategy_id,
            "instruments": instruments,
            "allocation": allocation,
        }

    def get_orchestrator_report(self) -> Dict[str, Any]:
        """Generate comprehensive orchestrator report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_strategies": len(self._active_strategies),
            "total_allocation": sum(self._strategy_allocations.values()),
            "strategy_details": [
                {
                    "id": str(strategy_id),
                    "allocation": self._strategy_allocations.get(strategy_id, 0),
                    "health": metrics.health.value,
                    "win_rate": round(metrics.win_rate, 3),
                    "sharpe_ratio": round(metrics.sharpe_ratio, 2),
                    "profit_factor": round(metrics.profit_factor, 2),
                    "max_drawdown": round(metrics.max_drawdown, 3),
                    "consecutive_losses": metrics.consecutive_losses,
                    "total_trades": metrics.total_trades,
                }
                for strategy_id, metrics in self._strategy_metrics.items()
                if strategy_id in self._active_strategies
            ],
            "portfolio_performance": {
                "total_pnl": self._portfolio_performance["total_pnl"],
                "strategy_contributions": dict(self._portfolio_performance["strategy_contributions"]),
            },
        }