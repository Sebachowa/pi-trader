#!/usr/bin/env python3
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
Multi-strategy paper trading runner with automated selection.

This module provides a framework for running multiple strategies simultaneously
during paper trading evaluation and automatically selecting the best performers.
"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import StrategyId
from nautilus_trader.model.identifiers import TraderId

from paper_trading.performance.performance_tracker import PerformanceTracker
from paper_trading.performance.realtime_monitor import RealtimeMonitor


class StrategyPerformanceMetrics:
    """Track performance metrics for individual strategies."""
    
    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        self.start_time = datetime.utcnow()
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
        self.last_update = datetime.utcnow()
        self.is_active = True
        self.selection_score = 0.0
        
    def update_metrics(self, portfolio_data: Dict[str, Any]):
        """Update metrics from portfolio data."""
        self.total_pnl = portfolio_data.get("total_pnl", 0.0)
        self.realized_pnl = portfolio_data.get("realized_pnl", 0.0)
        self.unrealized_pnl = portfolio_data.get("unrealized_pnl", 0.0)
        self.total_trades = portfolio_data.get("total_trades", 0)
        self.winning_trades = portfolio_data.get("winning_trades", 0)
        self.losing_trades = portfolio_data.get("losing_trades", 0)
        self.last_update = datetime.utcnow()
        
        # Calculate derived metrics
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        # Update equity curve
        current_equity = portfolio_data.get("balance_total", 100000.0)
        self.equity_curve.append(current_equity)
        
        # Calculate max drawdown
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            drawdown = (current_equity - peak) / peak * 100
            self.max_drawdown = min(self.max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        if len(self.daily_returns) > 20:  # Need sufficient data
            returns_series = pd.Series(self.daily_returns)
            if returns_series.std() > 0:
                self.sharpe_ratio = (returns_series.mean() * 252) / (returns_series.std() * (252 ** 0.5))
    
    def calculate_selection_score(self) -> float:
        """
        Calculate a selection score for strategy ranking.
        
        Returns
        -------
        float
            Selection score (higher is better).
        """
        # Multi-factor scoring model
        score_components = {
            "sharpe_ratio": self.sharpe_ratio * 0.25,
            "win_rate": self.win_rate * 100 * 0.20,
            "profit_factor": self.profit_factor * 0.20,
            "drawdown_penalty": max(0, 100 + self.max_drawdown) * 0.15,
            "consistency": (1 - self._calculate_consistency_penalty()) * 100 * 0.10,
            "activity": min(self.total_trades / 10, 1) * 100 * 0.10,
        }
        
        # Calculate weighted score
        self.selection_score = sum(score_components.values())
        
        # Apply penalties
        if self.total_pnl < 0:
            self.selection_score *= 0.5  # Penalty for negative PnL
        
        if self.max_drawdown < -20:  # More than 20% drawdown
            self.selection_score *= 0.7  # Additional penalty
        
        if self.total_trades < 5:  # Too few trades
            self.selection_score *= 0.8
        
        return self.selection_score
    
    def _calculate_consistency_penalty(self) -> float:
        """Calculate consistency penalty based on return volatility."""
        if len(self.daily_returns) < 2:
            return 0.5  # Penalty for insufficient data
        
        returns_series = pd.Series(self.daily_returns)
        return min(returns_series.std() / abs(returns_series.mean() + 0.0001), 1.0)


class MultiStrategyRunner:
    """
    Manages multiple strategies during paper trading with automated selection.
    
    Parameters
    ----------
    config_type : str
        The configuration type (crypto, fx, equities, mixed).
    strategy_configs : List[Dict[str, Any]]
        List of strategy configurations to run.
    node_config : TradingNodeConfig
        The trading node configuration.
    selection_interval_hours : int
        Hours between strategy selection evaluations.
    max_concurrent_strategies : int
        Maximum number of strategies to run concurrently.
    output_dir : Path
        Directory for output files.
    """
    
    def __init__(
        self,
        config_type: str,
        strategy_configs: List[Dict[str, Any]],
        node_config: TradingNodeConfig,
        selection_interval_hours: int = 24,
        max_concurrent_strategies: int = 5,
        output_dir: Optional[Path] = None,
    ):
        self.config_type = config_type
        self.strategy_configs = strategy_configs
        self.node_config = node_config
        self.selection_interval_hours = selection_interval_hours
        self.max_concurrent_strategies = max_concurrent_strategies
        self.output_dir = output_dir or Path("paper_trading/results")
        
        # Session management
        self.session_id = f"multi_{config_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategy tracking
        self.strategy_metrics: Dict[str, StrategyPerformanceMetrics] = {}
        self.active_strategies: List[str] = []
        self.strategy_history: List[Dict[str, Any]] = []
        
        # Components
        self.node: Optional[TradingNode] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.monitor: Optional[RealtimeMonitor] = None
        
        # Selection state
        self.last_selection_time = datetime.utcnow()
        self.selection_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize the multi-strategy runner."""
        print(f"\n{'='*60}")
        print(f"Initializing Multi-Strategy Paper Trading")
        print(f"Session ID: {self.session_id}")
        print(f"Total Strategies: {len(self.strategy_configs)}")
        print(f"Max Concurrent: {self.max_concurrent_strategies}")
        print(f"{'='*60}\n")
        
        # Create trading node
        self.node = TradingNode(config=self.node_config)
        
        # Initialize all strategies (but don't activate all)
        for i, strategy_config in enumerate(self.strategy_configs):
            strategy_id = strategy_config.get("strategy_id", f"STRATEGY_{i+1}")
            self.strategy_metrics[strategy_id] = StrategyPerformanceMetrics(strategy_id)
            
            # Create strategy instance
            strategy = self._create_strategy(strategy_config, strategy_id)
            self.node.trader.add_strategy(strategy)
        
        # Register client factories
        self._register_client_factories()
        
        # Build the node
        self.node.build()
        
        # Initialize performance tracking
        self.performance_tracker = PerformanceTracker(
            trader_id=self.node_config.trader_id,
            output_dir=self.session_dir,
        )
        
        # Initialize monitor with custom callbacks
        self.monitor = RealtimeMonitor(
            trader_id=self.node_config.trader_id,
            msgbus=self.node.msgbus,
            clock=self.node.clock,
            portfolio=self.node.portfolio,
            update_interval_secs=60,
            alert_callbacks=self._get_alert_callbacks(),
        )
        
        self.node.trader.add_actor(self.monitor)
        
        # Perform initial strategy selection
        self._select_active_strategies()
        
        print("✓ Multi-strategy initialization complete")
    
    async def run(self, duration_days: int = 14):
        """
        Run the multi-strategy paper trading session.
        
        Parameters
        ----------
        duration_days : int
            Duration of the test in days.
        """
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(days=duration_days)
        
        print(f"\n{'='*60}")
        print(f"Starting Multi-Strategy Paper Trading")
        print(f"Duration: {duration_days} days")
        print(f"Active Strategies: {', '.join(self.active_strategies)}")
        print(f"{'='*60}\n")
        
        # Start monitoring
        self.monitor.start()
        
        # Save session info
        self._save_session_info(start_time, end_time)
        
        try:
            # Run the node
            await self.node.run_async()
            
            # Schedule periodic evaluations
            self.node.clock.set_timer(
                name="strategy_evaluation",
                interval=timedelta(hours=self.selection_interval_hours),
                callback=self._evaluate_and_select_strategies,
            )
            
            # Schedule performance updates
            self.node.clock.set_timer(
                name="performance_update",
                interval=timedelta(minutes=30),
                callback=self._update_strategy_performance,
            )
            
            # Run until end time
            while datetime.utcnow() < end_time:
                await asyncio.sleep(60)
            
            # Final evaluation
            await self._final_evaluation()
            
        except KeyboardInterrupt:
            print("\n\nMulti-strategy trading interrupted by user")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the multi-strategy runner."""
        print("\n\nShutting down multi-strategy session...")
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop()
            self.monitor.export_session_data(self.session_dir / "monitor_data.json")
        
        # Generate final reports
        self._generate_final_reports()
        
        # Stop the node
        if self.node:
            await self.node.stop_async()
            await asyncio.sleep(1)
            self.node.dispose()
        
        print("✓ Shutdown complete")
    
    def _create_strategy(self, config: Dict[str, Any], strategy_id: str):
        """Create a strategy instance from configuration."""
        strategy_type = config.get("type", "ema_cross")
        
        # Import strategy based on type
        if strategy_type == "ema_cross":
            from nautilus_trader.examples.strategies.ema_cross import EMACross, EMACrossConfig
            
            strategy_config = EMACrossConfig(
                strategy_id=StrategyId(strategy_id),
                instrument_id=config.get("instrument_id"),
                bar_type=config.get("bar_type"),
                fast_ema_period=config.get("fast_ema_period", 10),
                slow_ema_period=config.get("slow_ema_period", 20),
                trade_size=config.get("trade_size", "1.0"),
            )
            return EMACross(config=strategy_config)
        
        # Add more strategy types as needed
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def _register_client_factories(self):
        """Register client factories with the node."""
        venues = list(self.node.config.exec_clients.keys())
        for venue in venues:
            self.node.add_exec_client_factory(venue, SandboxLiveExecClientFactory)
    
    def _select_active_strategies(self):
        """Select which strategies should be active based on performance."""
        # For initial selection, activate top N strategies randomly
        if not self.strategy_history:
            import random
            all_strategies = list(self.strategy_metrics.keys())
            random.shuffle(all_strategies)
            self.active_strategies = all_strategies[:self.max_concurrent_strategies]
        else:
            # Score all strategies
            strategy_scores = []
            for strategy_id, metrics in self.strategy_metrics.items():
                score = metrics.calculate_selection_score()
                strategy_scores.append((strategy_id, score))
            
            # Sort by score and select top performers
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            self.active_strategies = [s[0] for s in strategy_scores[:self.max_concurrent_strategies]]
        
        # Update strategy states
        for strategy_id in self.strategy_metrics:
            self.strategy_metrics[strategy_id].is_active = strategy_id in self.active_strategies
        
        # Log selection
        self._log_strategy_selection()
    
    def _evaluate_and_select_strategies(self, event=None):
        """Evaluate current strategies and update selection."""
        print(f"\n{'='*40}")
        print("Strategy Evaluation and Selection")
        print(f"{'='*40}")
        
        # Update performance metrics
        self._update_strategy_performance()
        
        # Perform new selection
        previous_active = self.active_strategies.copy()
        self._select_active_strategies()
        
        # Determine changes
        added = set(self.active_strategies) - set(previous_active)
        removed = set(previous_active) - set(self.active_strategies)
        
        if added or removed:
            print(f"Strategy changes:")
            for strategy_id in added:
                print(f"  + Added: {strategy_id}")
                # TODO: Activate strategy in node
            
            for strategy_id in removed:
                print(f"  - Removed: {strategy_id}")
                # TODO: Deactivate strategy in node
        else:
            print("No strategy changes")
        
        # Save selection history
        self.selection_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "active_strategies": self.active_strategies,
            "strategy_scores": {
                sid: metrics.selection_score 
                for sid, metrics in self.strategy_metrics.items()
            },
            "changes": {
                "added": list(added),
                "removed": list(removed),
            }
        })
        
        self.last_selection_time = datetime.utcnow()
    
    def _update_strategy_performance(self, event=None):
        """Update performance metrics for all strategies."""
        if not self.node or not self.node.portfolio:
            return
        
        # Get portfolio state
        portfolio = self.node.portfolio
        
        # Update metrics for each strategy
        for strategy_id in self.active_strategies:
            if strategy_id not in self.strategy_metrics:
                continue
            
            # Get strategy-specific data
            strategy_positions = portfolio.positions_for_strategy(StrategyId(strategy_id))
            strategy_orders = portfolio.orders_for_strategy(StrategyId(strategy_id))
            
            # Calculate strategy PnL
            unrealized_pnl = sum(float(pos.unrealized_pnl()) for pos in strategy_positions)
            realized_pnl = sum(float(pos.realized_pnl) for pos in strategy_positions)
            
            # Count trades
            winning_trades = sum(1 for pos in strategy_positions if pos.realized_pnl > 0)
            losing_trades = sum(1 for pos in strategy_positions if pos.realized_pnl < 0)
            
            # Update metrics
            portfolio_data = {
                "total_pnl": unrealized_pnl + realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "total_trades": len(strategy_positions),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "balance_total": float(portfolio.balance_total()),
            }
            
            self.strategy_metrics[strategy_id].update_metrics(portfolio_data)
    
    def _log_strategy_selection(self):
        """Log current strategy selection."""
        print(f"\nActive Strategies ({len(self.active_strategies)}):")
        for i, strategy_id in enumerate(self.active_strategies, 1):
            metrics = self.strategy_metrics[strategy_id]
            print(f"  {i}. {strategy_id}: Score={metrics.selection_score:.2f}, "
                  f"PnL=${metrics.total_pnl:.2f}, Trades={metrics.total_trades}")
    
    def _get_alert_callbacks(self) -> Dict[str, Any]:
        """Get alert callback functions."""
        return {
            "RISK_CRITICAL": self._handle_critical_alert,
            "RISK_HIGH": self._handle_high_alert,
            "PERFORMANCE_INFO": self._handle_performance_alert,
        }
    
    def _handle_critical_alert(self, alert: Dict[str, Any]):
        """Handle critical risk alerts."""
        # Force strategy re-evaluation on critical alerts
        self._evaluate_and_select_strategies()
    
    def _handle_high_alert(self, alert: Dict[str, Any]):
        """Handle high risk alerts."""
        print(f"\n⚠️  HIGH RISK ALERT: {alert['message']}")
    
    def _handle_performance_alert(self, alert: Dict[str, Any]):
        """Handle performance alerts."""
        print(f"\n✅ PERFORMANCE ALERT: {alert['message']}")
    
    async def _final_evaluation(self):
        """Perform final evaluation of all strategies."""
        print(f"\n\n{'='*60}")
        print("FINAL MULTI-STRATEGY EVALUATION")
        print(f"{'='*60}\n")
        
        # Update all metrics one final time
        self._update_strategy_performance()
        
        # Rank all strategies
        final_rankings = []
        for strategy_id, metrics in self.strategy_metrics.items():
            final_rankings.append({
                "strategy_id": strategy_id,
                "final_score": metrics.calculate_selection_score(),
                "total_pnl": metrics.total_pnl,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "was_active": strategy_id in self.active_strategies,
            })
        
        # Sort by final score
        final_rankings.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Display top strategies
        print("Top Performing Strategies:")
        for i, strategy in enumerate(final_rankings[:5], 1):
            print(f"{i}. {strategy['strategy_id']}: "
                  f"Score={strategy['final_score']:.2f}, "
                  f"PnL=${strategy['total_pnl']:.2f}, "
                  f"Sharpe={strategy['sharpe_ratio']:.2f}")
        
        # Save final evaluation
        evaluation_data = {
            "session_id": self.session_id,
            "final_rankings": final_rankings,
            "selection_history": self.selection_history,
            "best_strategy": final_rankings[0]["strategy_id"] if final_rankings else None,
            "evaluation_timestamp": datetime.utcnow().isoformat(),
        }
        
        with open(self.session_dir / "final_evaluation.json", "w") as f:
            json.dump(evaluation_data, f, indent=2)
    
    def _generate_final_reports(self):
        """Generate comprehensive final reports."""
        if not self.strategy_metrics:
            return
        
        # Generate strategy comparison report
        comparison_data = []
        for strategy_id, metrics in self.strategy_metrics.items():
            comparison_data.append({
                "strategy_id": strategy_id,
                "total_pnl": metrics.total_pnl,
                "realized_pnl": metrics.realized_pnl,
                "unrealized_pnl": metrics.unrealized_pnl,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "final_score": metrics.selection_score,
                "times_active": sum(1 for h in self.selection_history 
                                  if strategy_id in h["active_strategies"]),
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(comparison_data)
        df.to_csv(self.session_dir / "strategy_comparison.csv", index=False)
        
        # Generate summary report
        self._generate_summary_report(df)
    
    def _generate_summary_report(self, comparison_df: pd.DataFrame):
        """Generate human-readable summary report."""
        summary_file = self.session_dir / "MULTI_STRATEGY_SUMMARY.txt"
        
        with open(summary_file, "w") as f:
            f.write(f"MULTI-STRATEGY PAPER TRADING SUMMARY\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Config Type: {self.config_type}\n")
            f.write(f"Total Strategies Tested: {len(self.strategy_metrics)}\n")
            f.write(f"Max Concurrent Strategies: {self.max_concurrent_strategies}\n")
            f.write(f"Selection Evaluations: {len(self.selection_history)}\n\n")
            
            # Best performing strategy
            if not comparison_df.empty:
                best_strategy = comparison_df.loc[comparison_df["final_score"].idxmax()]
                f.write(f"BEST PERFORMING STRATEGY\n")
                f.write(f"{'-'*30}\n")
                f.write(f"Strategy ID: {best_strategy['strategy_id']}\n")
                f.write(f"Final Score: {best_strategy['final_score']:.2f}\n")
                f.write(f"Total PnL: ${best_strategy['total_pnl']:.2f}\n")
                f.write(f"Win Rate: {best_strategy['win_rate']:.2%}\n")
                f.write(f"Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}\n")
                f.write(f"Max Drawdown: {best_strategy['max_drawdown']:.2f}%\n\n")
            
            # Strategy selection summary
            f.write(f"STRATEGY SELECTION SUMMARY\n")
            f.write(f"{'-'*30}\n")
            
            # Count how often each strategy was selected
            selection_counts = defaultdict(int)
            for history in self.selection_history:
                for strategy in history["active_strategies"]:
                    selection_counts[strategy] += 1
            
            # Sort by selection frequency
            sorted_selections = sorted(selection_counts.items(), 
                                     key=lambda x: x[1], reverse=True)
            
            f.write("Most Frequently Selected:\n")
            for strategy, count in sorted_selections[:5]:
                percentage = (count / len(self.selection_history)) * 100
                f.write(f"  - {strategy}: {count} times ({percentage:.1f}%)\n")
    
    def _save_session_info(self, start_time: datetime, end_time: datetime):
        """Save session information."""
        session_info = {
            "session_id": self.session_id,
            "config_type": self.config_type,
            "start_time": start_time.isoformat(),
            "planned_end_time": end_time.isoformat(),
            "total_strategies": len(self.strategy_configs),
            "max_concurrent_strategies": self.max_concurrent_strategies,
            "selection_interval_hours": self.selection_interval_hours,
            "strategy_configs": self.strategy_configs,
            "output_directory": str(self.session_dir),
        }
        
        with open(self.session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)