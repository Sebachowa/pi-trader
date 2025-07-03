"""
Automated strategy selection algorithms for paper trading.

This module provides sophisticated algorithms for automatically selecting
the best performing strategies based on multiple criteria.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StrategyScore:
    """Container for strategy scoring components."""
    
    strategy_id: str
    total_score: float
    performance_score: float
    risk_score: float
    consistency_score: float
    activity_score: float
    recency_score: float
    components: Dict[str, float]
    timestamp: datetime


class SelectionCriteria:
    """Criteria for strategy selection and evaluation."""
    
    def __init__(
        self,
        min_trades: int = 5,
        min_win_rate: float = 0.4,
        max_drawdown: float = -15.0,
        min_sharpe: float = 0.5,
        min_profit_factor: float = 1.0,
        lookback_days: int = 7,
        recency_weight: float = 0.8,
    ):
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.max_drawdown = max_drawdown
        self.min_sharpe = min_sharpe
        self.min_profit_factor = min_profit_factor
        self.lookback_days = lookback_days
        self.recency_weight = recency_weight


class StrategySelector(ABC):
    """Abstract base class for strategy selection algorithms."""
    
    @abstractmethod
    def select_strategies(
        self,
        strategy_metrics: Dict[str, Any],
        max_strategies: int,
        criteria: SelectionCriteria,
    ) -> List[str]:
        """Select strategies based on the algorithm."""
        pass
    
    @abstractmethod
    def score_strategy(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> StrategyScore:
        """Score an individual strategy."""
        pass


class MultiFactorSelector(StrategySelector):
    """
    Multi-factor strategy selector using weighted scoring.
    
    This selector uses multiple performance and risk factors to score
    and rank strategies for selection.
    """
    
    def __init__(
        self,
        performance_weight: float = 0.35,
        risk_weight: float = 0.25,
        consistency_weight: float = 0.20,
        activity_weight: float = 0.10,
        recency_weight: float = 0.10,
    ):
        self.weights = {
            "performance": performance_weight,
            "risk": risk_weight,
            "consistency": consistency_weight,
            "activity": activity_weight,
            "recency": recency_weight,
        }
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def select_strategies(
        self,
        strategy_metrics: Dict[str, Any],
        max_strategies: int,
        criteria: SelectionCriteria,
    ) -> List[str]:
        """Select top strategies based on multi-factor scoring."""
        scores = []
        
        for strategy_id, metrics in strategy_metrics.items():
            # Skip strategies that don't meet minimum criteria
            if not self._meets_minimum_criteria(metrics, criteria):
                continue
            
            score = self.score_strategy(metrics, criteria)
            scores.append((strategy_id, score))
        
        # Sort by total score
        scores.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Return top strategies
        return [s[0] for s in scores[:max_strategies]]
    
    def score_strategy(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> StrategyScore:
        """Score a strategy using multiple factors."""
        # Calculate individual scores
        performance_score = self._calculate_performance_score(metrics, criteria)
        risk_score = self._calculate_risk_score(metrics, criteria)
        consistency_score = self._calculate_consistency_score(metrics, criteria)
        activity_score = self._calculate_activity_score(metrics, criteria)
        recency_score = self._calculate_recency_score(metrics, criteria)
        
        # Calculate weighted total
        components = {
            "performance": performance_score,
            "risk": risk_score,
            "consistency": consistency_score,
            "activity": activity_score,
            "recency": recency_score,
        }
        
        total_score = sum(
            score * self.weights[component]
            for component, score in components.items()
        )
        
        return StrategyScore(
            strategy_id=metrics.get("strategy_id", "unknown"),
            total_score=total_score,
            performance_score=performance_score,
            risk_score=risk_score,
            consistency_score=consistency_score,
            activity_score=activity_score,
            recency_score=recency_score,
            components=components,
            timestamp=datetime.utcnow(),
        )
    
    def _meets_minimum_criteria(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> bool:
        """Check if strategy meets minimum selection criteria."""
        if metrics.get("total_trades", 0) < criteria.min_trades:
            return False
        
        if metrics.get("win_rate", 0) < criteria.min_win_rate:
            return False
        
        if metrics.get("max_drawdown", 0) < criteria.max_drawdown:
            return False
        
        return True
    
    def _calculate_performance_score(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> float:
        """Calculate performance score (0-100)."""
        score_components = []
        
        # Sharpe ratio component (normalized)
        sharpe = metrics.get("sharpe_ratio", 0)
        sharpe_score = min(max(sharpe / 3.0, 0), 1) * 100  # Cap at 3.0
        score_components.append(sharpe_score * 0.4)
        
        # Total return component
        total_return = metrics.get("total_pnl_pct", 0)
        return_score = min(max(total_return / 20.0, -1), 1) * 50 + 50  # -20% to +20%
        score_components.append(return_score * 0.3)
        
        # Profit factor component
        profit_factor = metrics.get("profit_factor", 0)
        pf_score = min(max((profit_factor - 1) / 2.0, 0), 1) * 100  # 1-3 range
        score_components.append(pf_score * 0.3)
        
        return sum(score_components)
    
    def _calculate_risk_score(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> float:
        """Calculate risk score (0-100, higher is better/lower risk)."""
        score_components = []
        
        # Drawdown component
        max_dd = metrics.get("max_drawdown", 0)
        dd_score = max(0, 100 + max_dd)  # 0% DD = 100, -100% DD = 0
        score_components.append(dd_score * 0.5)
        
        # Win rate component
        win_rate = metrics.get("win_rate", 0)
        wr_score = win_rate * 100
        score_components.append(wr_score * 0.3)
        
        # Risk-adjusted return (simplified Calmar ratio)
        if max_dd < -1:  # Avoid division by zero
            calmar = (metrics.get("total_pnl_pct", 0) * 12) / abs(max_dd)
            calmar_score = min(max(calmar / 3.0, 0), 1) * 100
        else:
            calmar_score = 100 if metrics.get("total_pnl_pct", 0) > 0 else 0
        score_components.append(calmar_score * 0.2)
        
        return sum(score_components)
    
    def _calculate_consistency_score(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> float:
        """Calculate consistency score (0-100)."""
        daily_returns = metrics.get("daily_returns", [])
        
        if len(daily_returns) < 5:
            return 50.0  # Neutral score for insufficient data
        
        returns_series = pd.Series(daily_returns)
        
        # Calculate consistency metrics
        score_components = []
        
        # Positive return days ratio
        positive_days = (returns_series > 0).sum() / len(returns_series)
        score_components.append(positive_days * 100 * 0.3)
        
        # Return stability (inverse of coefficient of variation)
        if returns_series.mean() != 0:
            cv = returns_series.std() / abs(returns_series.mean())
            stability_score = max(0, 100 - cv * 50)  # Lower CV is better
        else:
            stability_score = 50
        score_components.append(stability_score * 0.4)
        
        # Trend consistency (using linear regression)
        if len(returns_series) > 10:
            x = np.arange(len(returns_series))
            slope, _, r_value, _, _ = stats.linregress(x, returns_series.cumsum())
            trend_score = (r_value ** 2) * 100  # R-squared as consistency measure
            score_components.append(trend_score * 0.3)
        else:
            score_components.append(50 * 0.3)  # Neutral
        
        return sum(score_components)
    
    def _calculate_activity_score(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> float:
        """Calculate activity score (0-100)."""
        total_trades = metrics.get("total_trades", 0)
        days_active = metrics.get("days_active", 1)
        
        # Trades per day
        trades_per_day = total_trades / max(days_active, 1)
        
        # Score based on activity level (optimal range)
        if trades_per_day < 0.5:
            activity_score = trades_per_day * 100  # Too low
        elif trades_per_day <= 5:
            activity_score = 75 + (trades_per_day - 0.5) * 5.5  # Optimal range
        else:
            activity_score = max(0, 100 - (trades_per_day - 5) * 10)  # Too high
        
        return activity_score
    
    def _calculate_recency_score(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> float:
        """Calculate recency score based on recent performance."""
        recent_returns = metrics.get("recent_returns", [])
        all_returns = metrics.get("daily_returns", [])
        
        if not recent_returns or not all_returns:
            return 50.0  # Neutral
        
        # Compare recent performance to overall
        recent_avg = np.mean(recent_returns)
        overall_avg = np.mean(all_returns)
        
        if overall_avg != 0:
            improvement_ratio = recent_avg / overall_avg
            recency_score = min(max(improvement_ratio * 50, 0), 100)
        else:
            recency_score = 50
        
        # Apply recency weight
        return recency_score * criteria.recency_weight + 50 * (1 - criteria.recency_weight)


class AdaptiveSelector(StrategySelector):
    """
    Adaptive strategy selector that learns from historical performance.
    
    This selector adjusts its selection criteria based on what has worked
    well in recent market conditions.
    """
    
    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = history_file
        self.selection_history: List[Dict[str, Any]] = []
        self.performance_history: Dict[str, List[float]] = {}
        
        if history_file and history_file.exists():
            self._load_history()
    
    def select_strategies(
        self,
        strategy_metrics: Dict[str, Any],
        max_strategies: int,
        criteria: SelectionCriteria,
    ) -> List[str]:
        """Select strategies using adaptive criteria."""
        # Adjust criteria based on recent performance
        adapted_criteria = self._adapt_criteria(criteria)
        
        # Score all strategies
        scores = []
        for strategy_id, metrics in strategy_metrics.items():
            score = self.score_strategy(metrics, adapted_criteria)
            scores.append((strategy_id, score))
        
        # Apply portfolio construction rules
        selected = self._construct_portfolio(scores, max_strategies)
        
        # Record selection
        self._record_selection(selected, strategy_metrics)
        
        return selected
    
    def score_strategy(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> StrategyScore:
        """Score strategy with adaptive weighting."""
        # Base scoring similar to MultiFactorSelector
        base_selector = MultiFactorSelector()
        base_score = base_selector.score_strategy(metrics, criteria)
        
        # Adjust based on historical performance
        strategy_id = metrics.get("strategy_id", "unknown")
        if strategy_id in self.performance_history:
            historical_performance = np.mean(self.performance_history[strategy_id][-10:])
            adjustment = 1.0 + (historical_performance - 0.5) * 0.2  # ±10% adjustment
            base_score.total_score *= adjustment
        
        return base_score
    
    def _adapt_criteria(self, base_criteria: SelectionCriteria) -> SelectionCriteria:
        """Adapt selection criteria based on recent market conditions."""
        if len(self.selection_history) < 5:
            return base_criteria  # Not enough history
        
        # Analyze recent successful selections
        recent_winners = self._analyze_recent_winners()
        
        # Create adapted criteria
        adapted = SelectionCriteria(
            min_trades=base_criteria.min_trades,
            min_win_rate=max(
                base_criteria.min_win_rate * 0.8,
                recent_winners.get("avg_win_rate", 0.5) * 0.9
            ),
            max_drawdown=min(
                base_criteria.max_drawdown,
                recent_winners.get("avg_drawdown", -10) * 1.2
            ),
            min_sharpe=max(
                base_criteria.min_sharpe * 0.7,
                recent_winners.get("avg_sharpe", 1.0) * 0.8
            ),
            min_profit_factor=base_criteria.min_profit_factor,
            lookback_days=base_criteria.lookback_days,
            recency_weight=base_criteria.recency_weight,
        )
        
        return adapted
    
    def _construct_portfolio(
        self,
        scores: List[Tuple[str, StrategyScore]],
        max_strategies: int,
    ) -> List[str]:
        """Construct portfolio with diversification rules."""
        # Sort by score
        scores.sort(key=lambda x: x[1].total_score, reverse=True)
        
        selected = []
        strategy_types = {}
        
        for strategy_id, score in scores:
            # Apply diversification rules
            strategy_type = self._get_strategy_type(strategy_id)
            
            if strategy_type in strategy_types:
                if strategy_types[strategy_type] >= max_strategies // 3:
                    continue  # Limit same type strategies
            
            selected.append(strategy_id)
            strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
            
            if len(selected) >= max_strategies:
                break
        
        return selected
    
    def _get_strategy_type(self, strategy_id: str) -> str:
        """Extract strategy type from ID."""
        # Simple classification based on naming convention
        if "momentum" in strategy_id.lower():
            return "momentum"
        elif "mean_revert" in strategy_id.lower():
            return "mean_reversion"
        elif "arb" in strategy_id.lower():
            return "arbitrage"
        else:
            return "trend_following"
    
    def _analyze_recent_winners(self) -> Dict[str, float]:
        """Analyze characteristics of recent winning strategies."""
        if not self.selection_history:
            return {}
        
        # Get last 5 selections
        recent = self.selection_history[-5:]
        
        winning_metrics = {
            "win_rates": [],
            "sharpes": [],
            "drawdowns": [],
        }
        
        for selection in recent:
            for strategy_id in selection.get("selected", []):
                if strategy_id in self.performance_history:
                    recent_perf = np.mean(self.performance_history[strategy_id][-5:])
                    if recent_perf > 0.6:  # Winning strategy
                        metrics = selection.get("metrics", {}).get(strategy_id, {})
                        winning_metrics["win_rates"].append(metrics.get("win_rate", 0))
                        winning_metrics["sharpes"].append(metrics.get("sharpe_ratio", 0))
                        winning_metrics["drawdowns"].append(metrics.get("max_drawdown", 0))
        
        return {
            "avg_win_rate": np.mean(winning_metrics["win_rates"]) if winning_metrics["win_rates"] else 0.5,
            "avg_sharpe": np.mean(winning_metrics["sharpes"]) if winning_metrics["sharpes"] else 1.0,
            "avg_drawdown": np.mean(winning_metrics["drawdowns"]) if winning_metrics["drawdowns"] else -10,
        }
    
    def _record_selection(self, selected: List[str], metrics: Dict[str, Any]):
        """Record selection for future adaptation."""
        self.selection_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "selected": selected,
            "metrics": metrics,
        })
        
        # Keep only recent history
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-50:]
        
        # Save history if file specified
        if self.history_file:
            self._save_history()
    
    def update_performance(self, strategy_id: str, performance: float):
        """Update strategy performance history."""
        if strategy_id not in self.performance_history:
            self.performance_history[strategy_id] = []
        
        self.performance_history[strategy_id].append(performance)
        
        # Keep only recent performance
        if len(self.performance_history[strategy_id]) > 50:
            self.performance_history[strategy_id] = self.performance_history[strategy_id][-30:]
    
    def _load_history(self):
        """Load selection history from file."""
        try:
            with open(self.history_file) as f:
                data = json.load(f)
                self.selection_history = data.get("selection_history", [])
                self.performance_history = data.get("performance_history", {})
        except Exception as e:
            print(f"Error loading history: {e}")
    
    def _save_history(self):
        """Save selection history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump({
                    "selection_history": self.selection_history,
                    "performance_history": self.performance_history,
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")


class EnsembleSelector(StrategySelector):
    """
    Ensemble selector that combines multiple selection algorithms.
    
    This selector uses multiple selection algorithms and combines their
    results for more robust strategy selection.
    """
    
    def __init__(self):
        self.selectors = {
            "multi_factor": MultiFactorSelector(),
            "adaptive": AdaptiveSelector(),
        }
        self.selector_weights = {
            "multi_factor": 0.6,
            "adaptive": 0.4,
        }
    
    def select_strategies(
        self,
        strategy_metrics: Dict[str, Any],
        max_strategies: int,
        criteria: SelectionCriteria,
    ) -> List[str]:
        """Select strategies using ensemble of selectors."""
        # Get selections from each selector
        all_selections = {}
        for name, selector in self.selectors.items():
            selections = selector.select_strategies(
                strategy_metrics,
                max_strategies * 2,  # Get more candidates
                criteria,
            )
            all_selections[name] = selections
        
        # Combine selections with voting
        strategy_votes = {}
        for name, selections in all_selections.items():
            weight = self.selector_weights.get(name, 1.0)
            for i, strategy_id in enumerate(selections):
                if strategy_id not in strategy_votes:
                    strategy_votes[strategy_id] = 0
                # Higher rank gets more votes
                strategy_votes[strategy_id] += weight * (len(selections) - i)
        
        # Sort by votes and select top strategies
        sorted_strategies = sorted(
            strategy_votes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [s[0] for s in sorted_strategies[:max_strategies]]
    
    def score_strategy(
        self,
        metrics: Dict[str, Any],
        criteria: SelectionCriteria,
    ) -> StrategyScore:
        """Score strategy using ensemble of scorers."""
        scores = []
        
        for name, selector in self.selectors.items():
            score = selector.score_strategy(metrics, criteria)
            weight = self.selector_weights.get(name, 1.0)
            scores.append(score.total_score * weight)
        
        # Create ensemble score
        total_score = sum(scores) / sum(self.selector_weights.values())
        
        return StrategyScore(
            strategy_id=metrics.get("strategy_id", "unknown"),
            total_score=total_score,
            performance_score=0,  # Not calculated for ensemble
            risk_score=0,
            consistency_score=0,
            activity_score=0,
            recency_score=0,
            components={"ensemble": total_score},
            timestamp=datetime.utcnow(),
        )