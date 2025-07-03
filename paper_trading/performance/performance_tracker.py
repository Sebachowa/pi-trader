
"""
Performance tracking for paper trading evaluation.

This module provides comprehensive performance tracking and metrics collection
for paper trading sessions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nautilus_trader.analysis.analyzer import PortfolioAnalyzer
from nautilus_trader.model.identifiers import StrategyId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.persistence import ParquetDataCatalog
from nautilus_trader.portfolio.portfolio import Portfolio


class PerformanceTracker:
    """
    Tracks and analyzes paper trading performance.
    
    Parameters
    ----------
    trader_id : TraderId
        The trader ID for the session.
    output_dir : Path
        Directory to save performance reports.
    catalog_path : Path, optional
        Path to the data catalog for storing metrics.
        
    """
    
    def __init__(
        self,
        trader_id: TraderId,
        output_dir: Path,
        catalog_path: Path | None = None,
    ):
        self.trader_id = trader_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.catalog_path = catalog_path
        self.analyzer = PortfolioAnalyzer()
        
        # Performance metrics storage
        self.metrics_history: list[dict[str, Any]] = []
        self.daily_metrics: dict[str, list[dict[str, Any]]] = {}
        
    def track_portfolio_state(
        self,
        portfolio: Portfolio,
        timestamp: datetime,
        save_snapshot: bool = True,
    ) -> dict[str, Any]:
        """
        Track current portfolio state and calculate metrics.
        
        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to track.
        timestamp : datetime
            The timestamp for this snapshot.
        save_snapshot : bool
            Whether to save a snapshot to disk.
            
        Returns
        -------
        dict[str, Any]
            Current performance metrics.
            
        """
        # Calculate base metrics
        metrics = {
            "timestamp": timestamp.isoformat(),
            "trader_id": str(self.trader_id),
            "total_pnl": float(portfolio.net_pnl_total()),
            "unrealized_pnl": float(portfolio.unrealized_pnl_total()),
            "realized_pnl": float(portfolio.realized_pnl_total()),
            "balance_total": float(portfolio.balance_total()),
            "margin_used": float(portfolio.margin_total()),
            "margin_available": float(portfolio.margin_available_total()),
            "position_count": len(portfolio.positions()),
            "open_order_count": len(portfolio.orders()),
        }
        
        # Calculate per-strategy metrics
        strategy_metrics = {}
        for strategy_id in portfolio.strategy_ids():
            strategy_positions = portfolio.positions_for_strategy(strategy_id)
            strategy_orders = portfolio.orders_for_strategy(strategy_id)
            
            strategy_metrics[str(strategy_id)] = {
                "position_count": len(strategy_positions),
                "order_count": len(strategy_orders),
                "unrealized_pnl": sum(
                    float(pos.unrealized_pnl()) for pos in strategy_positions
                ),
                "realized_pnl": sum(
                    float(pos.realized_pnl) for pos in strategy_positions
                ),
            }
        
        metrics["strategy_metrics"] = strategy_metrics
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Save snapshot if requested
        if save_snapshot:
            self._save_snapshot(metrics, timestamp)
        
        return metrics
    
    def calculate_statistics(self, portfolio: Portfolio) -> dict[str, float]:
        """
        Calculate comprehensive performance statistics.
        
        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to analyze.
            
        Returns
        -------
        dict[str, float]
            Performance statistics.
            
        """
        # Use the portfolio analyzer
        self.analyzer.calculate_statistics(portfolio)
        
        # Get all statistics
        stats = {}
        for stat in self.analyzer.statistics:
            stats[stat.name] = float(stat.value) if stat.value is not None else 0.0
        
        # Add custom statistics
        if len(self.metrics_history) > 1:
            # Calculate max drawdown
            equity_curve = [m["balance_total"] for m in self.metrics_history]
            drawdowns = self._calculate_drawdowns(equity_curve)
            stats["max_drawdown"] = min(drawdowns) if drawdowns else 0.0
            stats["max_drawdown_duration_days"] = self._calculate_max_dd_duration()
            
            # Calculate daily returns
            returns = self._calculate_returns()
            if returns:
                stats["daily_return_mean"] = pd.Series(returns).mean()
                stats["daily_return_std"] = pd.Series(returns).std()
                stats["return_skewness"] = pd.Series(returns).skew()
                stats["return_kurtosis"] = pd.Series(returns).kurt()
        
        return stats
    
    def generate_report(
        self,
        portfolio: Portfolio,
        strategy_id: StrategyId | None = None,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to report on.
        strategy_id : StrategyId, optional
            Specific strategy to report on.
            
        Returns
        -------
        dict[str, Any]
            Performance report.
            
        """
        timestamp = datetime.utcnow()
        
        # Get current metrics
        current_metrics = self.track_portfolio_state(portfolio, timestamp, save_snapshot=False)
        
        # Calculate statistics
        statistics = self.calculate_statistics(portfolio)
        
        # Build report
        report = {
            "report_timestamp": timestamp.isoformat(),
            "trader_id": str(self.trader_id),
            "strategy_id": str(strategy_id) if strategy_id else None,
            "current_metrics": current_metrics,
            "statistics": statistics,
            "trading_summary": {
                "total_trades": len(portfolio.trade_ids()),
                "winning_trades": sum(1 for t in portfolio.trades() if t.realized_pnl > 0),
                "losing_trades": sum(1 for t in portfolio.trades() if t.realized_pnl < 0),
                "breakeven_trades": sum(1 for t in portfolio.trades() if t.realized_pnl == 0),
                "average_win": self._calculate_average_win(portfolio),
                "average_loss": self._calculate_average_loss(portfolio),
                "largest_win": self._calculate_largest_win(portfolio),
                "largest_loss": self._calculate_largest_loss(portfolio),
            },
            "position_summary": {
                "total_positions": len(portfolio.position_ids()),
                "open_positions": len(portfolio.positions_open()),
                "closed_positions": len(portfolio.positions_closed()),
                "position_side_breakdown": self._get_position_side_breakdown(portfolio),
            },
            "risk_metrics": {
                "value_at_risk_95": self._calculate_var(0.95),
                "value_at_risk_99": self._calculate_var(0.99),
                "expected_shortfall_95": self._calculate_expected_shortfall(0.95),
                "max_portfolio_heat": self._calculate_max_portfolio_heat(portfolio),
            },
        }
        
        # Save report
        report_path = self.output_dir / f"performance_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def save_metrics_to_csv(self, filename: str | None = None):
        """Save metrics history to CSV file."""
        if not self.metrics_history:
            return
        
        if filename is None:
            filename = f"metrics_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.output_dir / filename, index=False)
    
    def plot_performance(self, save_path: Path | None = None):
        """Generate performance plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            
            if not self.metrics_history:
                return
            
            df = pd.DataFrame(self.metrics_history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Equity curve
            ax1 = axes[0, 0]
            ax1.plot(df.index, df["balance_total"], label="Total Balance", linewidth=2)
            ax1.fill_between(df.index, df["balance_total"], alpha=0.3)
            ax1.set_title("Equity Curve")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Balance")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: PnL components
            ax2 = axes[0, 1]
            ax2.plot(df.index, df["unrealized_pnl"], label="Unrealized PnL", alpha=0.7)
            ax2.plot(df.index, df["realized_pnl"], label="Realized PnL", alpha=0.7)
            ax2.plot(df.index, df["total_pnl"], label="Total PnL", linewidth=2)
            ax2.set_title("PnL Breakdown")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("PnL")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Drawdown
            ax3 = axes[1, 0]
            equity = df["balance_total"].values
            drawdowns = self._calculate_drawdowns(equity)
            ax3.fill_between(df.index, 0, drawdowns, color='red', alpha=0.3)
            ax3.plot(df.index, drawdowns, color='darkred', linewidth=1)
            ax3.set_title("Drawdown")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Drawdown %")
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Returns distribution
            ax4 = axes[1, 1]
            returns = self._calculate_returns()
            if returns:
                ax4.hist(returns, bins=50, alpha=0.7, edgecolor='black')
                ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax4.set_title("Returns Distribution")
                ax4.set_xlabel("Daily Return %")
                ax4.set_ylabel("Frequency")
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.output_dir / f"performance_plots_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    # Helper methods
    def _save_snapshot(self, metrics: dict[str, Any], timestamp: datetime):
        """Save a performance snapshot."""
        snapshot_dir = self.output_dir / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        
        filename = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(snapshot_dir / filename, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def _calculate_drawdowns(self, equity_curve: list[float]) -> list[float]:
        """Calculate drawdown percentages."""
        if not equity_curve:
            return []
        
        peak = equity_curve[0]
        drawdowns = []
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = ((value - peak) / peak) * 100 if peak > 0 else 0
            drawdowns.append(drawdown)
        
        return drawdowns
    
    def _calculate_returns(self) -> list[float]:
        """Calculate daily returns."""
        if len(self.metrics_history) < 2:
            return []
        
        equity = [m["balance_total"] for m in self.metrics_history]
        returns = []
        
        for i in range(1, len(equity)):
            if equity[i-1] > 0:
                ret = ((equity[i] - equity[i-1]) / equity[i-1]) * 100
                returns.append(ret)
        
        return returns
    
    def _calculate_max_dd_duration(self) -> float:
        """Calculate maximum drawdown duration in days."""
        # Simplified implementation
        return 0.0
    
    def _calculate_average_win(self, portfolio: Portfolio) -> float:
        """Calculate average winning trade."""
        wins = [t.realized_pnl for t in portfolio.trades() if t.realized_pnl > 0]
        return float(sum(wins) / len(wins)) if wins else 0.0
    
    def _calculate_average_loss(self, portfolio: Portfolio) -> float:
        """Calculate average losing trade."""
        losses = [t.realized_pnl for t in portfolio.trades() if t.realized_pnl < 0]
        return float(sum(losses) / len(losses)) if losses else 0.0
    
    def _calculate_largest_win(self, portfolio: Portfolio) -> float:
        """Calculate largest winning trade."""
        wins = [t.realized_pnl for t in portfolio.trades() if t.realized_pnl > 0]
        return float(max(wins)) if wins else 0.0
    
    def _calculate_largest_loss(self, portfolio: Portfolio) -> float:
        """Calculate largest losing trade."""
        losses = [t.realized_pnl for t in portfolio.trades() if t.realized_pnl < 0]
        return float(min(losses)) if losses else 0.0
    
    def _get_position_side_breakdown(self, portfolio: Portfolio) -> dict[str, int]:
        """Get position side breakdown."""
        breakdown = {"LONG": 0, "SHORT": 0}
        for pos in portfolio.positions():
            if pos.is_long:
                breakdown["LONG"] += 1
            else:
                breakdown["SHORT"] += 1
        return breakdown
    
    def _calculate_var(self, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        returns = self._calculate_returns()
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return sorted_returns[index] if index < len(sorted_returns) else 0.0
    
    def _calculate_expected_shortfall(self, confidence_level: float) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        returns = self._calculate_returns()
        if not returns:
            return 0.0
        
        var = self._calculate_var(confidence_level)
        tail_losses = [r for r in returns if r <= var]
        return sum(tail_losses) / len(tail_losses) if tail_losses else 0.0
    
    def _calculate_max_portfolio_heat(self, portfolio: Portfolio) -> float:
        """Calculate maximum portfolio heat (risk exposure)."""
        # Simplified: ratio of margin used to total balance
        if portfolio.balance_total() > 0:
            return float(portfolio.margin_total() / portfolio.balance_total())
        return 0.0