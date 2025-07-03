#!/usr/bin/env python3

"""
Paper trading runner for 2-week evaluation period.

This script manages the complete paper trading lifecycle including:
- Strategy deployment
- Performance monitoring
- Automated evaluation
- Report generation
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import StrategyId
from nautilus_trader.model.identifiers import TraderId

from paper_trading.configs.crypto_paper_config import get_crypto_paper_config
from paper_trading.configs.equities_paper_config import get_equities_paper_config
from paper_trading.configs.fx_paper_config import get_fx_paper_config
from paper_trading.data_feeds.data_feed_config import DataFeedConfigs
from paper_trading.performance.performance_tracker import PerformanceTracker
from paper_trading.performance.realtime_monitor import RealtimeMonitor


class PaperTradingRunner:
    """
    Manages paper trading sessions for strategy evaluation.
    
    Parameters
    ----------
    config_type : str
        The configuration type (crypto, fx, equities).
    strategy_configs : list[dict]
        List of strategy configurations to run.
    test_duration_days : int
        Duration of the paper trading test in days.
    output_dir : Path
        Directory for output files.
        
    """
    
    def __init__(
        self,
        config_type: str,
        strategy_configs: list[dict[str, Any]],
        test_duration_days: int = 14,
        output_dir: Path | None = None,
    ):
        self.config_type = config_type
        self.strategy_configs = strategy_configs
        self.test_duration_days = test_duration_days
        self.output_dir = output_dir or Path("paper_trading/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self.session_id = f"paper_{config_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # Components
        self.node: TradingNode | None = None
        self.performance_tracker: PerformanceTracker | None = None
        self.monitor: RealtimeMonitor | None = None
        
        # Session state
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.is_running = False
        self.evaluation_results: dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize the paper trading session."""
        print(f"\n{'='*60}")
        print(f"Initializing Paper Trading Session: {self.session_id}")
        print(f"{'='*60}\n")
        
        # Get trading node configuration
        node_config = self._get_node_config()
        
        # Add data client configurations
        data_configs = DataFeedConfigs.get_multi_venue_config(
            use_databento=self.config_type in ["equities", "futures"],
            use_binance=self.config_type == "crypto",
            use_ib=self.config_type in ["equities", "fx"],
        )
        
        # Merge data client configs
        node_config_dict = node_config.dict()
        node_config_dict["data_clients"] = data_configs
        node_config = TradingNodeConfig(**node_config_dict)
        
        # Create trading node
        self.node = TradingNode(config=node_config)
        
        # Initialize strategies
        for strategy_config in self.strategy_configs:
            strategy = self._create_strategy(strategy_config)
            self.node.trader.add_strategy(strategy)
        
        # Register client factories
        self._register_client_factories()
        
        # Build the node
        self.node.build()
        
        # Initialize performance tracking
        self.performance_tracker = PerformanceTracker(
            trader_id=node_config.trader_id,
            output_dir=self.session_dir,
        )
        
        # Initialize real-time monitor
        self.monitor = RealtimeMonitor(
            trader_id=node_config.trader_id,
            msgbus=self.node.msgbus,
            clock=self.node.clock,
            portfolio=self.node.portfolio,
            update_interval_secs=300,  # 5-minute updates
            alert_callbacks=self._get_alert_callbacks(),
        )
        
        # Add monitor as actor
        self.node.trader.add_actor(self.monitor)
        
        print("✓ Initialization complete")
    
    async def run(self):
        """Run the paper trading session."""
        if not self.node:
            raise RuntimeError("Runner not initialized")
        
        self.start_time = datetime.utcnow()
        self.end_time = self.start_time + timedelta(days=self.test_duration_days)
        self.is_running = True
        
        print(f"\n{'='*60}")
        print(f"Starting Paper Trading Session")
        print(f"Start Time: {self.start_time.isoformat()}")
        print(f"End Time: {self.end_time.isoformat()}")
        print(f"Duration: {self.test_duration_days} days")
        print(f"{'='*60}\n")
        
        # Start monitoring
        self.monitor.start()
        
        # Create session info file
        self._save_session_info()
        
        try:
            # Run the node
            await self.node.run_async()
            
            # Schedule periodic evaluations
            self.node.clock.set_timer(
                name="periodic_evaluation",
                interval=timedelta(hours=24),
                callback=self._periodic_evaluation,
            )
            
            # Schedule final evaluation
            self.node.clock.set_time_alert(
                name="final_evaluation",
                alert_time=unix_nanos_to_dt(int(self.end_time.timestamp() * 1e9)),
                callback=self._final_evaluation,
            )
            
            # Keep running until end time
            while self.is_running and datetime.utcnow() < self.end_time:
                await asyncio.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            print("\n\nPaper trading interrupted by user")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the paper trading session."""
        print("\n\nShutting down paper trading session...")
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop()
            self.monitor.export_session_data(self.session_dir / "monitor_data.json")
        
        # Generate final reports
        if self.performance_tracker and self.node:
            self._generate_final_reports()
        
        # Stop the node
        if self.node:
            await self.node.stop_async()
            await asyncio.sleep(1)
            self.node.dispose()
        
        self.is_running = False
        print("✓ Shutdown complete")
    
    def _get_node_config(self) -> TradingNodeConfig:
        """Get the appropriate node configuration."""
        trader_id = f"PAPER-{self.config_type.upper()}-001"
        
        if self.config_type == "crypto":
            return get_crypto_paper_config(
                trader_id=trader_id,
                log_level="INFO",
                use_binance=True,
                use_bybit=True,
            )
        elif self.config_type == "fx":
            return get_fx_paper_config(
                trader_id=trader_id,
                log_level="INFO",
                use_ecn=True,
            )
        elif self.config_type == "equities":
            return get_equities_paper_config(
                trader_id=trader_id,
                log_level="INFO",
                use_ib=True,
                use_nyse=True,
                use_nasdaq=True,
            )
        else:
            raise ValueError(f"Unknown config type: {self.config_type}")
    
    def _create_strategy(self, config: dict[str, Any]):
        """Create a strategy instance from configuration."""
        # Import and instantiate strategy based on config
        # This is a placeholder - actual implementation depends on strategy structure
        from nautilus_trader.examples.strategies.ema_cross import EMACross
        from nautilus_trader.examples.strategies.ema_cross import EMACrossConfig
        
        # Example strategy creation
        strategy_config = EMACrossConfig(
            instrument_id=config.get("instrument_id"),
            bar_type=config.get("bar_type"),
            fast_ema_period=config.get("fast_ema_period", 10),
            slow_ema_period=config.get("slow_ema_period", 20),
            trade_size=config.get("trade_size", "1.0"),
        )
        
        return EMACross(config=strategy_config)
    
    def _register_client_factories(self):
        """Register client factories with the node."""
        # Register sandbox execution client factory
        venues = list(self.node.config.exec_clients.keys())
        for venue in venues:
            self.node.add_exec_client_factory(venue, SandboxLiveExecClientFactory)
        
        # Register data client factories based on config
        if self.config_type == "crypto":
            from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
            if "BINANCE" in self.node.config.data_clients:
                self.node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
        
        elif self.config_type in ["equities", "fx"]:
            from nautilus_trader.adapters.interactive_brokers.factories import InteractiveBrokersLiveDataClientFactory
            if "IB" in self.node.config.data_clients:
                self.node.add_data_client_factory("IB", InteractiveBrokersLiveDataClientFactory)
    
    def _get_alert_callbacks(self) -> dict[str, Any]:
        """Get alert callback functions."""
        return {
            "RISK_CRITICAL": self._handle_critical_risk_alert,
            "RISK_HIGH": self._handle_high_risk_alert,
            "PERFORMANCE_INFO": self._handle_performance_alert,
        }
    
    def _handle_critical_risk_alert(self, alert: dict[str, Any]):
        """Handle critical risk alerts."""
        # Log to file
        alert_file = self.session_dir / "critical_alerts.json"
        alerts = []
        if alert_file.exists():
            with open(alert_file) as f:
                alerts = json.load(f)
        alerts.append(alert)
        with open(alert_file, "w") as f:
            json.dump(alerts, f, indent=2)
        
        # Could also send notifications, pause trading, etc.
    
    def _handle_high_risk_alert(self, alert: dict[str, Any]):
        """Handle high risk alerts."""
        print(f"\n⚠️  HIGH RISK ALERT: {alert['message']}")
    
    def _handle_performance_alert(self, alert: dict[str, Any]):
        """Handle performance alerts."""
        print(f"\n✅ PERFORMANCE ALERT: {alert['message']}")
    
    def _periodic_evaluation(self, event=None):
        """Perform periodic evaluation."""
        if not self.node or not self.performance_tracker:
            return
        
        # Generate performance report
        report = self.performance_tracker.generate_report(
            portfolio=self.node.portfolio,
        )
        
        # Save metrics history
        self.performance_tracker.save_metrics_to_csv(
            f"metrics_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        )
        
        # Generate plots
        self.performance_tracker.plot_performance()
        
        print(f"\n📊 Daily evaluation completed - PnL: ${report['current_metrics']['total_pnl']:.2f}")
    
    def _final_evaluation(self, event=None):
        """Perform final evaluation and generate reports."""
        print(f"\n\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}\n")
        
        self._generate_final_reports()
        self.is_running = False
    
    def _generate_final_reports(self):
        """Generate comprehensive final reports."""
        if not self.node or not self.performance_tracker:
            return
        
        # Generate final performance report
        final_report = self.performance_tracker.generate_report(
            portfolio=self.node.portfolio,
        )
        
        # Calculate evaluation metrics
        self.evaluation_results = self._calculate_evaluation_metrics(final_report)
        
        # Save evaluation results
        eval_file = self.session_dir / "evaluation_results.json"
        with open(eval_file, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
        
        # Generate plots
        self.performance_tracker.plot_performance(
            save_path=self.session_dir / "final_performance_plots.png"
        )
        
        print(f"\n✓ Final reports generated in {self.session_dir}")
    
    def _calculate_evaluation_metrics(self, report: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        stats = report["statistics"]
        trading = report["trading_summary"]
        
        # Determine if strategy passes evaluation criteria
        evaluation_criteria = {
            "positive_returns": report["current_metrics"]["total_pnl"] > 0,
            "acceptable_drawdown": stats.get("max_drawdown", 0) > -15.0,  # Max 15% drawdown
            "sufficient_trades": trading["total_trades"] >= 10,  # At least 10 trades
            "positive_win_rate": trading["winning_trades"] / max(trading["total_trades"], 1) > 0.4,
            "positive_sharpe": stats.get("sharpe_ratio", 0) > 0.5,
        }
        
        passed_criteria = sum(evaluation_criteria.values())
        total_criteria = len(evaluation_criteria)
        
        return {
            "session_id": self.session_id,
            "test_duration_days": self.test_duration_days,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": datetime.utcnow().isoformat(),
            "final_performance": report,
            "evaluation_criteria": evaluation_criteria,
            "criteria_passed": f"{passed_criteria}/{total_criteria}",
            "recommendation": "PASS" if passed_criteria >= 4 else "FAIL",
            "ready_for_live": passed_criteria == total_criteria,
        }
    
    def _generate_summary_report(self):
        """Generate a human-readable summary report."""
        if not self.evaluation_results:
            return
        
        summary_file = self.session_dir / "SUMMARY.txt"
        
        with open(summary_file, "w") as f:
            f.write(f"PAPER TRADING EVALUATION SUMMARY\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Duration: {self.test_duration_days} days\n")
            f.write(f"Config Type: {self.config_type}\n\n")
            
            # Performance Summary
            perf = self.evaluation_results["final_performance"]["current_metrics"]
            f.write(f"PERFORMANCE SUMMARY\n")
            f.write(f"{'-'*30}\n")
            f.write(f"Total PnL: ${perf['total_pnl']:.2f} ({perf['pnl_pct']:.2f}%)\n")
            f.write(f"Final Balance: ${perf['balance_total']:.2f}\n")
            f.write(f"Positions Held: {perf['position_count']}\n\n")
            
            # Trading Summary
            trading = self.evaluation_results["final_performance"]["trading_summary"]
            f.write(f"TRADING SUMMARY\n")
            f.write(f"{'-'*30}\n")
            f.write(f"Total Trades: {trading['total_trades']}\n")
            f.write(f"Winning Trades: {trading['winning_trades']}\n")
            f.write(f"Losing Trades: {trading['losing_trades']}\n")
            f.write(f"Win Rate: {trading['winning_trades']/max(trading['total_trades'],1)*100:.1f}%\n")
            f.write(f"Average Win: ${trading['average_win']:.2f}\n")
            f.write(f"Average Loss: ${trading['average_loss']:.2f}\n\n")
            
            # Evaluation Results
            f.write(f"EVALUATION RESULTS\n")
            f.write(f"{'-'*30}\n")
            criteria = self.evaluation_results["evaluation_criteria"]
            for criterion, passed in criteria.items():
                f.write(f"{criterion}: {'✓ PASS' if passed else '✗ FAIL'}\n")
            
            f.write(f"\nCriteria Passed: {self.evaluation_results['criteria_passed']}\n")
            f.write(f"Recommendation: {self.evaluation_results['recommendation']}\n")
            f.write(f"Ready for Live Trading: {'YES' if self.evaluation_results['ready_for_live'] else 'NO'}\n")
    
    def _save_session_info(self):
        """Save session information."""
        session_info = {
            "session_id": self.session_id,
            "config_type": self.config_type,
            "test_duration_days": self.test_duration_days,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "planned_end_time": self.end_time.isoformat() if self.end_time else None,
            "strategy_configs": self.strategy_configs,
            "output_directory": str(self.session_dir),
        }
        
        with open(self.session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)


async def main():
    """Main entry point for paper trading runner."""
    parser = argparse.ArgumentParser(description="Run paper trading evaluation")
    parser.add_argument(
        "--config-type",
        choices=["crypto", "fx", "equities"],
        required=True,
        help="Configuration type to use",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=14,
        help="Test duration in days (default: 14)",
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        help="Path to strategy configuration JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Load strategy configurations
    strategy_configs = []
    if args.strategy_config:
        with open(args.strategy_config) as f:
            strategy_configs = json.load(f)
    else:
        # Default strategy configs for each market
        if args.config_type == "crypto":
            strategy_configs = [
                {
                    "instrument_id": "BTCUSDT-PERP.BINANCE",
                    "bar_type": "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL",
                    "fast_ema_period": 10,
                    "slow_ema_period": 20,
                    "trade_size": "0.001",
                }
            ]
        elif args.config_type == "fx":
            strategy_configs = [
                {
                    "instrument_id": "EUR/USD.FX-ECN",
                    "bar_type": "EUR/USD.FX-ECN-1-MINUTE-MID-EXTERNAL",
                    "fast_ema_period": 12,
                    "slow_ema_period": 26,
                    "trade_size": "10000",
                }
            ]
        elif args.config_type == "equities":
            strategy_configs = [
                {
                    "instrument_id": "AAPL.NASDAQ",
                    "bar_type": "AAPL.NASDAQ-1-MINUTE-LAST-EXTERNAL",
                    "fast_ema_period": 9,
                    "slow_ema_period": 21,
                    "trade_size": "100",
                }
            ]
    
    # Create and run paper trading session
    runner = PaperTradingRunner(
        config_type=args.config_type,
        strategy_configs=strategy_configs,
        test_duration_days=args.duration,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    
    await runner.initialize()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())