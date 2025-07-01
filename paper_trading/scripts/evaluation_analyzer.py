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
Evaluation analyzer for comparing paper trading results with backtests.

This script provides comprehensive analysis of paper trading performance
and comparison with historical backtests to validate strategy performance.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class EvaluationAnalyzer:
    """
    Analyzes paper trading results and compares with backtest performance.
    
    Parameters
    ----------
    paper_results_dir : Path
        Directory containing paper trading results.
    backtest_results_dir : Path, optional
        Directory containing backtest results for comparison.
        
    """
    
    def __init__(
        self,
        paper_results_dir: Path,
        backtest_results_dir: Path | None = None,
    ):
        self.paper_results_dir = Path(paper_results_dir)
        self.backtest_results_dir = Path(backtest_results_dir) if backtest_results_dir else None
        
        # Load paper trading results
        self.paper_results = self._load_paper_results()
        self.backtest_results = self._load_backtest_results() if backtest_results_dir else None
    
    def _load_paper_results(self) -> dict[str, Any]:
        """Load paper trading results."""
        results = {}
        
        # Find all evaluation results
        for eval_file in self.paper_results_dir.glob("*/evaluation_results.json"):
            session_id = eval_file.parent.name
            with open(eval_file) as f:
                results[session_id] = json.load(f)
        
        return results
    
    def _load_backtest_results(self) -> dict[str, Any]:
        """Load backtest results for comparison."""
        if not self.backtest_results_dir:
            return {}
        
        results = {}
        
        # Load backtest results
        for result_file in self.backtest_results_dir.glob("*/backtest_results.json"):
            backtest_id = result_file.parent.name
            with open(result_file) as f:
                results[backtest_id] = json.load(f)
        
        return results
    
    def analyze_paper_trading_results(self) -> pd.DataFrame:
        """
        Analyze all paper trading results.
        
        Returns
        -------
        pd.DataFrame
            Summary of paper trading results.
            
        """
        if not self.paper_results:
            print("No paper trading results found")
            return pd.DataFrame()
        
        summary_data = []
        
        for session_id, results in self.paper_results.items():
            perf = results["final_performance"]["current_metrics"]
            stats = results["final_performance"]["statistics"]
            trading = results["final_performance"]["trading_summary"]
            
            summary_data.append({
                "session_id": session_id,
                "duration_days": results["test_duration_days"],
                "total_pnl": perf["total_pnl"],
                "pnl_pct": perf["pnl_pct"],
                "final_balance": perf["balance_total"],
                "total_trades": trading["total_trades"],
                "win_rate": trading["winning_trades"] / max(trading["total_trades"], 1),
                "sharpe_ratio": stats.get("sharpe_ratio", 0),
                "max_drawdown": stats.get("max_drawdown", 0),
                "profit_factor": stats.get("profit_factor", 0),
                "recommendation": results["recommendation"],
                "ready_for_live": results["ready_for_live"],
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values("pnl_pct", ascending=False)
    
    def compare_with_backtests(self) -> pd.DataFrame:
        """
        Compare paper trading results with backtest results.
        
        Returns
        -------
        pd.DataFrame
            Comparison of paper vs backtest performance.
            
        """
        if not self.backtest_results:
            print("No backtest results available for comparison")
            return pd.DataFrame()
        
        comparisons = []
        
        for session_id, paper_results in self.paper_results.items():
            # Find matching backtest
            strategy_type = session_id.split("_")[1]  # Extract strategy type
            matching_backtests = [
                (bt_id, bt_results)
                for bt_id, bt_results in self.backtest_results.items()
                if strategy_type in bt_id
            ]
            
            if not matching_backtests:
                continue
            
            # Compare with most recent backtest
            bt_id, bt_results = matching_backtests[-1]
            
            paper_stats = paper_results["final_performance"]["statistics"]
            bt_stats = bt_results.get("statistics", {})
            
            comparisons.append({
                "session_id": session_id,
                "backtest_id": bt_id,
                "metric": "total_return_pct",
                "paper_value": paper_results["final_performance"]["current_metrics"]["pnl_pct"],
                "backtest_value": bt_stats.get("total_return_pct", 0),
                "difference": paper_results["final_performance"]["current_metrics"]["pnl_pct"] - 
                             bt_stats.get("total_return_pct", 0),
            })
            
            # Compare other metrics
            metrics_to_compare = [
                "sharpe_ratio",
                "max_drawdown",
                "profit_factor",
                "win_rate",
            ]
            
            for metric in metrics_to_compare:
                paper_value = paper_stats.get(metric, 0)
                bt_value = bt_stats.get(metric, 0)
                
                comparisons.append({
                    "session_id": session_id,
                    "backtest_id": bt_id,
                    "metric": metric,
                    "paper_value": paper_value,
                    "backtest_value": bt_value,
                    "difference": paper_value - bt_value,
                })
        
        return pd.DataFrame(comparisons)
    
    def generate_evaluation_report(self, output_path: Path | None = None):
        """
        Generate comprehensive evaluation report.
        
        Parameters
        ----------
        output_path : Path, optional
            Path to save the report.
            
        """
        if output_path is None:
            output_path = Path("paper_trading_evaluation_report.html")
        
        # Analyze results
        paper_summary = self.analyze_paper_trading_results()
        comparison_df = self.compare_with_backtests()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Paper Trading Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                .metric-box {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Paper Trading Evaluation Report</h1>
            <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <div class="metric-box">
                    <h3>Total Sessions</h3>
                    <p>{len(self.paper_results)}</p>
                </div>
                <div class="metric-box">
                    <h3>Successful Sessions</h3>
                    <p>{sum(1 for r in self.paper_results.values() if r['recommendation'] == 'PASS')}</p>
                </div>
                <div class="metric-box">
                    <h3>Ready for Live</h3>
                    <p>{sum(1 for r in self.paper_results.values() if r['ready_for_live'])}</p>
                </div>
            </div>
            
            <h2>Paper Trading Results Summary</h2>
            {paper_summary.to_html(index=False, classes='table')}
            
        """
        
        # Add comparison section if available
        if not comparison_df.empty:
            # Pivot comparison data for better readability
            pivot_df = comparison_df.pivot_table(
                index=["session_id", "metric"],
                columns="backtest_id",
                values=["paper_value", "backtest_value", "difference"],
            )
            
            html_content += f"""
            <h2>Paper Trading vs Backtest Comparison</h2>
            {pivot_df.to_html(classes='table')}
            """
        
        # Add detailed session analysis
        html_content += """
            <h2>Detailed Session Analysis</h2>
        """
        
        for session_id, results in self.paper_results.items():
            criteria = results["evaluation_criteria"]
            recommendation = results["recommendation"]
            
            html_content += f"""
            <h3>Session: {session_id}</h3>
            <div class="summary">
                <p><strong>Duration:</strong> {results['test_duration_days']} days</p>
                <p><strong>Recommendation:</strong> 
                    <span class="{recommendation.lower()}">{recommendation}</span>
                </p>
                <p><strong>Evaluation Criteria:</strong></p>
                <ul>
            """
            
            for criterion, passed in criteria.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                html_content += f'<li>{criterion}: <span class="{("pass" if passed else "fail")}">{status}</span></li>'
            
            html_content += """
                </ul>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, "w") as f:
            f.write(html_content)
        
        print(f"Evaluation report saved to: {output_path}")
    
    def export_metrics_comparison(self, output_path: Path | None = None):
        """Export detailed metrics comparison to CSV."""
        if output_path is None:
            output_path = Path("paper_trading_metrics_comparison.csv")
        
        # Combine all metrics
        all_metrics = []
        
        for session_id, results in self.paper_results.items():
            metrics = {
                "session_id": session_id,
                "type": "paper_trading",
                **results["final_performance"]["statistics"],
                **results["final_performance"]["current_metrics"],
            }
            all_metrics.append(metrics)
        
        if self.backtest_results:
            for bt_id, results in self.backtest_results.items():
                metrics = {
                    "session_id": bt_id,
                    "type": "backtest",
                    **results.get("statistics", {}),
                }
                all_metrics.append(metrics)
        
        df = pd.DataFrame(all_metrics)
        df.to_csv(output_path, index=False)
        print(f"Metrics comparison exported to: {output_path}")
    
    def identify_best_performers(self, min_trades: int = 20) -> list[dict[str, Any]]:
        """
        Identify best performing strategies.
        
        Parameters
        ----------
        min_trades : int
            Minimum number of trades required.
            
        Returns
        -------
        list[dict[str, Any]]
            Best performing strategies.
            
        """
        best_performers = []
        
        for session_id, results in self.paper_results.items():
            trading = results["final_performance"]["trading_summary"]
            
            if trading["total_trades"] < min_trades:
                continue
            
            if results["ready_for_live"]:
                best_performers.append({
                    "session_id": session_id,
                    "total_pnl": results["final_performance"]["current_metrics"]["total_pnl"],
                    "pnl_pct": results["final_performance"]["current_metrics"]["pnl_pct"],
                    "sharpe_ratio": results["final_performance"]["statistics"].get("sharpe_ratio", 0),
                    "win_rate": trading["winning_trades"] / trading["total_trades"],
                    "total_trades": trading["total_trades"],
                })
        
        # Sort by PnL percentage
        best_performers.sort(key=lambda x: x["pnl_pct"], reverse=True)
        
        return best_performers


def main():
    """Main entry point for evaluation analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze paper trading evaluation results")
    parser.add_argument(
        "--paper-results",
        type=str,
        required=True,
        help="Directory containing paper trading results",
    )
    parser.add_argument(
        "--backtest-results",
        type=str,
        help="Directory containing backtest results for comparison",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--export-metrics",
        type=str,
        help="Export metrics comparison to CSV",
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = EvaluationAnalyzer(
        paper_results_dir=Path(args.paper_results),
        backtest_results_dir=Path(args.backtest_results) if args.backtest_results else None,
    )
    
    # Analyze results
    print("\n=== Paper Trading Results Summary ===")
    summary_df = analyzer.analyze_paper_trading_results()
    print(summary_df)
    
    # Compare with backtests if available
    if args.backtest_results:
        print("\n=== Paper vs Backtest Comparison ===")
        comparison_df = analyzer.compare_with_backtests()
        print(comparison_df)
    
    # Generate report
    report_path = Path(args.output_report) if args.output_report else None
    analyzer.generate_evaluation_report(report_path)
    
    # Export metrics if requested
    if args.export_metrics:
        analyzer.export_metrics_comparison(Path(args.export_metrics))
    
    # Show best performers
    print("\n=== Best Performing Strategies ===")
    best = analyzer.identify_best_performers()
    for i, strategy in enumerate(best[:5], 1):
        print(f"{i}. {strategy['session_id']}: "
              f"{strategy['pnl_pct']:.2f}% PnL, "
              f"{strategy['win_rate']:.2%} win rate, "
              f"{strategy['total_trades']} trades")


if __name__ == "__main__":
    main()