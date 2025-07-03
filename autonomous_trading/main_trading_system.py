#!/usr/bin/env python3
"""
Main 24/7 Autonomous Trading System
Integrates all components into a cohesive, self-managing trading platform
"""

import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any

from nautilus_trader.adapters.binance.config import BinanceDataClientConfig, BinanceExecClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory, BinanceLiveExecClientFactory
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.common.enums import LogLevel
from nautilus_trader.config import InstrumentProviderConfig, ExecEngineConfig, TradingNodeConfig
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.data.engine import DataEngine
from nautilus_trader.execution.engine import ExecutionEngine
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId, TraderId, StrategyId
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.risk.engine import RiskEngine

# Import our components
from autonomous_trading.core.enhanced_engine import EnhancedAutonomousEngine
from autonomous_trading.core.adaptive_risk_controller import AdaptiveRiskController
from autonomous_trading.core.market_analyzer import MarketAnalyzer
from autonomous_trading.core.passive_income_optimizer import PassiveIncomeOptimizer
from autonomous_trading.monitoring.performance import PerformanceOptimizer
from autonomous_trading.monitoring.notifications import NotificationSystem
from autonomous_trading.strategies.ml_strategy_selector import MLStrategySelector
from autonomous_trading.strategies.orchestrator import StrategyOrchestrator

# Import Nautilus strategies
from nautilus_challenge.strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MarketMakingStrategy,
    MLStrategy,
)


class AutonomousTradingSystem:
    """
    Main 24/7 Autonomous Trading System
    
    Features:
    - Multi-exchange support (starting with Binance)
    - ML-powered strategy selection
    - Adaptive risk management
    - Self-healing capabilities
    - Performance monitoring
    - Passive income optimization
    - Telegram notifications
    """
    
    def __init__(self, config_path: str = "config_btc_start.json"):
        """Initialize the autonomous trading system."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Core components
        self.trading_node: Optional[TradingNode] = None
        self.autonomous_engine: Optional[EnhancedAutonomousEngine] = None
        self.risk_controller: Optional[AdaptiveRiskController] = None
        self.market_analyzer: Optional[MarketAnalyzer] = None
        self.strategy_selector: Optional[MLStrategySelector] = None
        self.strategy_orchestrator: Optional[StrategyOrchestrator] = None
        self.income_optimizer: Optional[PassiveIncomeOptimizer] = None
        self.performance_monitor: Optional[PerformanceOptimizer] = None
        self.notification_service: Optional[NotificationSystem] = None
        
        # System state
        self.is_running = False
        self.start_time = None
        self.total_uptime = timedelta()
        self.recovery_attempts = 0
        self.last_health_check = None
        
        # Trading state
        self.active_strategies: Dict[StrategyId, Any] = {}
        self.instruments: List[InstrumentId] = []
        self.mode = "paper"  # or "live"
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        # Validate required fields
        required_fields = ["trader_id", "initial_capital", "target_annual_return", "max_drawdown"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
                
        return config
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        asyncio.create_task(self.shutdown())
        
    async def initialize(self) -> None:
        """Initialize all system components."""
        print("🚀 Initializing Autonomous Trading System...")
        
        # Initialize notification service first for status updates
        await self._initialize_notifications()
        await self.notification_service.send_notification(
            "INFO",
            "🚀 Autonomous Trading System Starting",
            f"Mode: {self.config.get('mode', 'paper')}\n"
            f"Initial Capital: {self.config['initial_capital']} BTC\n"
            f"Target Return: {self.config['target_annual_return']*100}%"
        )
        
        # Initialize trading node
        await self._initialize_trading_node()
        
        # Initialize core components
        await self._initialize_autonomous_engine()
        await self._initialize_risk_controller()
        await self._initialize_market_analyzer()
        await self._initialize_strategy_components()
        await self._initialize_monitoring()
        
        # Load instruments
        await self._load_instruments()
        
        print("✅ System initialization complete")
        
    async def _initialize_trading_node(self) -> None:
        """Initialize Nautilus Trader trading node."""
        print("📊 Initializing trading node...")
        
        # Configure based on mode
        mode = self.config.get("mode", "paper")
        
        if mode == "live":
            # Live trading configuration
            data_config = BinanceDataClientConfig(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
                testnet=False,
            )
            exec_config = BinanceExecClientConfig(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
                testnet=False,
            )
        else:
            # Paper trading configuration
            data_config = BinanceDataClientConfig(
                api_key="paper_key",
                api_secret="paper_secret",
                testnet=True,
            )
            exec_config = BinanceExecClientConfig(
                api_key="paper_key",
                api_secret="paper_secret",
                testnet=True,
            )
            
        # Create trading node configuration with minimal parameters
        config = TradingNodeConfig(
            trader_id=TraderId(self.config["trader_id"]),
            data_clients={
                "BINANCE": data_config,
            },
            exec_clients={
                "BINANCE": exec_config,
            },
        )
        
        # Create trading node
        self.trading_node = TradingNode(config)
        
        # Register Binance client factories
        self.trading_node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
        self.trading_node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)
        
        # Build the node
        self.trading_node.build()
        
    async def _initialize_autonomous_engine(self) -> None:
        """Initialize the enhanced autonomous engine."""
        print("🤖 Initializing autonomous engine...")
        
        from autonomous_trading.config.minimal_intervention_architecture import MinimalInterventionConfig, InterventionLevel
        
        # Create minimal intervention config
        config = MinimalInterventionConfig(
            trader_id=self.config["trader_id"],
            target_annual_return=self.config["target_annual_return"] * 100,  # Convert to percentage
            max_drawdown_percent=self.config["max_drawdown"] * 100,  # Convert to percentage
            max_daily_loss_percent=self.config.get("max_daily_loss", 0.02) * 100,
            max_position_risk_percent=self.config.get("max_position_risk", 0.01) * 100,
            max_portfolio_risk_percent=self.config.get("max_portfolio_risk", 0.05) * 100,
            intervention_level=InterventionLevel.CRITICAL_ONLY,
        )
        
        # Create mock components for now
        class MockClock:
            def utc_now(self): return datetime.utcnow()
        
        class MockMessageBus:
            def publish(self, msg): pass
        
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
        
        self.autonomous_engine = EnhancedAutonomousEngine(
            config=config,
            trading_node=self.trading_node,
            logger=MockLogger(),
            clock=MockClock(),
            msgbus=MockMessageBus(),
        )
        
    async def _initialize_risk_controller(self) -> None:
        """Initialize adaptive risk controller."""
        print("🛡️ Initializing risk controller...")
        
        # Create mock components
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
        
        class MockClock:
            def utc_now(self): return datetime.utcnow()
        
        class MockMessageBus:
            def publish(self, msg): pass
        
        self.risk_controller = AdaptiveRiskController(
            logger=MockLogger(),
            clock=MockClock(),
            msgbus=MockMessageBus(),
            max_portfolio_risk_percent=self.config.get("max_portfolio_risk", 0.06) * 100,
            max_position_risk_percent=self.config.get("max_position_risk", 0.02) * 100,
            max_daily_loss_percent=self.config.get("max_daily_drawdown", 0.05) * 100,
        )
        
    async def _initialize_market_analyzer(self) -> None:
        """Initialize market analyzer."""
        print("📈 Initializing market analyzer...")
        
        # Create mock components
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
        
        class MockClock:
            def utc_now(self): return datetime.utcnow()
        
        class MockMessageBus:
            def publish(self, msg): pass
        
        self.market_analyzer = MarketAnalyzer(
            logger=MockLogger(),
            clock=MockClock(),
            msgbus=MockMessageBus(),
        )
        
    async def _initialize_strategy_components(self) -> None:
        """Initialize strategy selector and orchestrator."""
        print("🎯 Initializing strategy components...")
        
        # Create mock components
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
        
        class MockClock:
            def utc_now(self): return datetime.utcnow()
        
        class MockMessageBus:
            def publish(self, msg): pass
        
        # ML Strategy Selector
        self.strategy_selector = MLStrategySelector(
            logger=MockLogger(),
            clock=MockClock(),
            msgbus=MockMessageBus(),
            enable_evolution=True,
            enable_rl_selection=True,
            enable_bayesian_opt=True,
        )
        
        # Strategy Orchestrator
        self.strategy_orchestrator = StrategyOrchestrator(
            logger=MockLogger(),
            clock=MockClock(),
            msgbus=MockMessageBus(),
            max_concurrent_strategies=self.config.get("max_concurrent_strategies", 5),
            min_strategy_allocation=self.config.get("min_strategy_allocation", 0.05),
            max_strategy_allocation=self.config.get("max_strategy_allocation", 0.30),
        )
        
        # Register available strategies
        self._register_strategies()
        
    def _register_strategies(self) -> None:
        """Register all available trading strategies."""
        self.strategy_orchestrator._available_strategies = {
            "trend_following": TrendFollowingStrategy,
            "mean_reversion": MeanReversionStrategy,
            "momentum": MomentumStrategy,
            "market_making": MarketMakingStrategy,
            "ml_strategy": MLStrategy,
        }
        
    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring components."""
        print("📊 Initializing monitoring...")
        
        # Create mock components
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
        
        class MockClock:
            def utc_now(self): return datetime.utcnow()
        
        class MockMessageBus:
            def publish(self, msg): pass
        
        # Performance Monitor
        self.performance_monitor = PerformanceOptimizer(
            logger=MockLogger(),
            clock=MockClock(),
            msgbus=MockMessageBus(),
            optimization_interval_hours=self.config.get("reporting_interval", 24),
        )
        
        # Passive Income Optimizer
        self.income_optimizer = PassiveIncomeOptimizer(
            logger=MockLogger(),
            clock=MockClock(),
            msgbus=MockMessageBus(),
            target_monthly_income=self.config["initial_capital"] * self.config["target_annual_return"] / 12,
            initial_capital=self.config["initial_capital"],
        )
        
    async def _initialize_notifications(self) -> None:
        """Initialize notification service."""
        print("📱 Initializing notifications...")
        
        from autonomous_trading.monitoring.notifications import NotificationConfig
        
        # Create notification config
        notification_config = NotificationConfig()
        telegram_config = self.config.get("telegram", {})
        notification_config.telegram_config["bot_token"] = telegram_config.get("token", os.getenv("TELEGRAM_BOT_TOKEN", ""))
        notification_config.telegram_config["chat_ids"] = [telegram_config.get("chat_id", os.getenv("TELEGRAM_CHAT_ID", ""))]
        
        if self.config.get("email"):
            notification_config.email_config.update(self.config["email"])
        
        # Create mock logger and components for now
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def debug(self, msg): pass
        
        class MockClock:
            def utc_now(self): return datetime.utcnow()
        
        class MockMessageBus:
            def publish(self, msg): pass
        
        self.notification_service = NotificationSystem(
            logger=MockLogger(),
            clock=MockClock(),
            msgbus=MockMessageBus(),
            config=notification_config,
        )
        
    async def _load_instruments(self) -> None:
        """Load trading instruments."""
        print("💱 Loading instruments...")
        
        # Default instruments
        default_instruments = [
            "BTCUSDT.BINANCE",
            "ETHUSDT.BINANCE",
        ]
        
        configured_instruments = self.config.get("instruments", default_instruments)
        
        for instrument_str in configured_instruments:
            try:
                instrument_id = InstrumentId.from_str(instrument_str)
                self.instruments.append(instrument_id)
            except Exception as e:
                print(f"⚠️ Failed to load instrument {instrument_str}: {e}")
                
        print(f"✅ Loaded {len(self.instruments)} instruments")
        
    async def start(self) -> None:
        """Start the autonomous trading system."""
        print("\n🚀 Starting Autonomous Trading System...")
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        try:
            # Trading node is already running after initialization
            # Start all components
            await self.market_analyzer.start()
            await self.strategy_orchestrator.start()
            await self.performance_monitor.start()
            
            # Wait for instruments to load
            await asyncio.sleep(5)
            
            # Deploy initial strategies
            await self._deploy_initial_strategies()
            
            # Start main trading loop
            await self._run_trading_loop()
            
        except Exception as e:
            print(f"❌ Error during startup: {e}")
            if self.notification_service:
                await self.notification_service.send_notification(
                    "ERROR",
                    "Startup Error",
                    f"Failed to start trading system: {str(e)}"
                )
            await self.shutdown()
            
    async def _deploy_initial_strategies(self) -> None:
        """Deploy initial trading strategies based on market conditions."""
        print("🎯 Deploying initial strategies...")
        
        # Get market conditions
        market_conditions = await self._get_market_conditions()
        
        # Get portfolio balance
        # Get account balance
        # Get account using Venue identifier
        from nautilus_trader.model.identifiers import Venue
        account = self.trading_node.cache.account_for_venue(Venue("BINANCE"))
        if account:
            account_balance = account.balance_total_quoted(self.trading_node.cache.quote_currency("BTC"))
        else:
            # Use initial capital as fallback
            account_balance = self.config.get("initial_capital", 0.3)
        
        # Select strategies
        selected_strategies = await self.strategy_orchestrator.select_strategies(
            market_conditions,
            float(account_balance)
        )
        
        # Deploy each selected strategy
        for strategy_name, allocation in selected_strategies:
            strategy_id = await self.strategy_orchestrator.deploy_strategy(
                strategy_name,
                allocation,
                self.instruments,
            )
            
            if strategy_id:
                self.active_strategies[strategy_id] = {
                    "name": strategy_name,
                    "allocation": allocation,
                    "deployed_at": datetime.utcnow(),
                }
                
        print(f"✅ Deployed {len(self.active_strategies)} strategies")
        
    async def _get_market_conditions(self) -> Dict[InstrumentId, Any]:
        """Get current market conditions for all instruments."""
        conditions = {}
        
        for instrument_id in self.instruments:
            try:
                # Get market conditions (synchronous method)
                analysis = self.market_analyzer.get_market_conditions(instrument_id)
                if analysis:
                    conditions[instrument_id] = analysis
            except Exception as e:
                print(f"⚠️ Failed to analyze {instrument_id}: {e}")
                
        return conditions
        
    async def _run_trading_loop(self) -> None:
        """Main trading loop."""
        print("🔄 Starting main trading loop...")
        
        loop_counter = 0
        
        while self.is_running:
            try:
                loop_counter += 1
                
                # Health check
                if loop_counter % 12 == 0:  # Every hour (5 min intervals)
                    await self._perform_health_check()
                    
                # Risk check
                await self._check_risk_limits()
                
                # Market analysis update
                if loop_counter % 3 == 0:  # Every 15 minutes
                    await self._update_market_analysis()
                    
                # Performance tracking
                if loop_counter % 60 == 0:  # Every 5 hours
                    await self._track_performance()
                    
                # Sleep for interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                await self._handle_error(e)
                
    async def _perform_health_check(self) -> None:
        """Perform system health check."""
        self.last_health_check = datetime.utcnow()
        
        health_status = {
            "timestamp": self.last_health_check,
            "uptime": str(self.last_health_check - self.start_time),
            "active_strategies": len(self.active_strategies),
            "recovery_attempts": self.recovery_attempts,
            "components": {
                "trading_node": self.trading_node.is_running if self.trading_node else False,
                "autonomous_engine": self.autonomous_engine.is_healthy() if self.autonomous_engine else False,
                "risk_controller": True,  # Add actual health check
                "market_analyzer": True,  # Add actual health check
            }
        }
        
        # Check for issues
        unhealthy_components = [
            name for name, status in health_status["components"].items() 
            if not status
        ]
        
        if unhealthy_components:
            await self.notification_service.send_notification(
                "WARNING",
                "Health Check Warning",
                f"Unhealthy components: {', '.join(unhealthy_components)}"
            )
            
            # Attempt recovery
            await self._attempt_recovery(unhealthy_components)
            
    async def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        # Get portfolio metrics
        # Get account using Venue identifier
        from nautilus_trader.model.identifiers import Venue
        account = self.trading_node.cache.account_for_venue(Venue("BINANCE"))
        
        # Get current metrics (simplified for now)
        metrics = {
            "total_pnl": 0.0,  # TODO: Calculate from position history
            "unrealized_pnl": 0.0,  # TODO: Calculate from open positions
            "margin_used": 0.0,  # Not applicable for spot trading
            "margin_available": account.balance_free_quoted(self.trading_node.cache.quote_currency("BTC")) if account else 0.0,
        }
        
        # Check drawdown
        if self.performance_monitor:
            # Get drawdown from performance metrics
            perf_report = self.performance_monitor.get_performance_report()
            current_drawdown = perf_report.get("performance_metrics", {}).get("max_drawdown", 0.0)
            
            if current_drawdown > self.config["max_drawdown"]:
                await self.notification_service.send_notification(
                    "CRITICAL",
                    "Risk Limit Breach",
                    f"Current drawdown: {current_drawdown:.2%} exceeds limit: {self.config['max_drawdown']:.2%}"
                )
                
                # Reduce exposure
                await self._reduce_exposure(0.5)  # Reduce by 50%
                
    async def _update_market_analysis(self) -> None:
        """Update market analysis for all instruments."""
        market_conditions = await self._get_market_conditions()
        
        # Check if strategy rebalancing is needed
        regime_changes = 0
        for instrument_id, conditions in market_conditions.items():
            # Compare with previous regime
            # This is simplified - would need proper tracking
            regime_changes += 1
            
        if regime_changes > len(self.instruments) / 2:
            # Significant regime change - consider rebalancing
            await self._rebalance_strategies(market_conditions)
            
    async def _rebalance_strategies(self, market_conditions: Dict[InstrumentId, Any]) -> None:
        """Rebalance strategies based on new market conditions."""
        # Get account balance
        # Get account using Venue identifier
        from nautilus_trader.model.identifiers import Venue
        account = self.trading_node.cache.account_for_venue(Venue("BINANCE"))
        if account:
            account_balance = account.balance_total_quoted(self.trading_node.cache.quote_currency("BTC"))
        else:
            # Use initial capital as fallback
            account_balance = self.config.get("initial_capital", 0.3)
        
        # Get new strategy selection
        new_strategies = await self.strategy_orchestrator.select_strategies(
            market_conditions,
            float(account_balance)
        )
        
        # Compare with current strategies
        current_names = {s["name"] for s in self.active_strategies.values()}
        new_names = {name for name, _ in new_strategies}
        
        # Stop strategies no longer needed
        to_remove = current_names - new_names
        for strategy_id, details in list(self.active_strategies.items()):
            if details["name"] in to_remove:
                await self._stop_strategy(strategy_id)
                
        # Deploy new strategies
        to_add = new_names - current_names
        for strategy_name, allocation in new_strategies:
            if strategy_name in to_add:
                await self.strategy_orchestrator.deploy_strategy(
                    strategy_name,
                    allocation,
                    self.instruments,
                )
                
    async def _track_performance(self) -> None:
        """Track and report performance."""
        if not self.performance_monitor:
            return
            
        # Generate performance report
        report = self.performance_monitor.generate_report()
        
        # Send daily report
        current_hour = datetime.utcnow().hour
        if current_hour == 0:  # Midnight UTC
            await self.notification_service.send_notification(
                "📊 Daily Performance Report",
                self._format_performance_report(report)
            )
            
        # Check for income optimization opportunities
        if self.income_optimizer:
            optimization = self.income_optimizer.calculate_optimal_withdrawal()
            if optimization["recommended_withdrawal"] > 0:
                await self.notification_service.send_notification(
                    "💰 Income Optimization",
                    f"Recommended withdrawal: {optimization['recommended_withdrawal']:.4f} BTC\n"
                    f"Reason: {optimization['reason']}"
                )
                
    def _format_performance_report(self, report: Dict[str, Any]) -> str:
        """Format performance report for notification."""
        return f"""
📈 Performance Metrics:
• Total P&L: {report['total_pnl']:.4f} BTC ({report['total_return']:.2%})
• Today's P&L: {report['daily_pnl']:.4f} BTC ({report['daily_return']:.2%})
• Win Rate: {report['win_rate']:.2%}
• Sharpe Ratio: {report['sharpe_ratio']:.2f}
• Max Drawdown: {report['max_drawdown']:.2%}

🎯 Active Strategies: {report['active_strategies']}
📊 Total Trades: {report['total_trades']}
⏱️ Uptime: {report['uptime']}
"""
        
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors with recovery attempts."""
        self.recovery_attempts += 1
        
        if self.notification_service:
            await self.notification_service.send_notification(
                "ERROR",
                "Trading Loop Error",
                f"Error: {str(error)}\nRecovery attempt: {self.recovery_attempts}"
            )
        
        if self.recovery_attempts < 3:
            # Attempt recovery
            await asyncio.sleep(60)  # Wait 1 minute
            print(f"🔄 Attempting recovery #{self.recovery_attempts}...")
        else:
            # Too many failures - shutdown
            await self.notification_service.send_notification(
                "CRITICAL",
                "System Shutdown",
                "Too many recovery attempts. Shutting down for safety."
            )
            await self.shutdown()
            
    async def _attempt_recovery(self, unhealthy_components: List[str]) -> None:
        """Attempt to recover unhealthy components."""
        for component in unhealthy_components:
            try:
                if component == "trading_node":
                    # Restart trading node
                    await self.trading_node.stop()
                    await asyncio.sleep(5)
                    await self.trading_node.start()
                elif component == "autonomous_engine":
                    # Reinitialize autonomous engine
                    await self._initialize_autonomous_engine()
                # Add recovery logic for other components
                
                print(f"✅ Recovered component: {component}")
                
            except Exception as e:
                print(f"❌ Failed to recover {component}: {e}")
                
    async def _reduce_exposure(self, reduction_factor: float) -> None:
        """Reduce exposure by closing or reducing positions."""
        print(f"⚠️ Reducing exposure by {reduction_factor:.0%}")
        
        # Reduce allocations for all strategies
        for strategy_id in self.active_strategies:
            current_allocation = self.active_strategies[strategy_id]["allocation"]
            new_allocation = current_allocation * (1 - reduction_factor)
            
            # Update in orchestrator
            self.strategy_orchestrator._strategy_allocations[strategy_id] = new_allocation
            self.active_strategies[strategy_id]["allocation"] = new_allocation
            
    async def _stop_strategy(self, strategy_id: StrategyId) -> None:
        """Stop a specific strategy."""
        if strategy_id in self.active_strategies:
            # Stop in orchestrator
            if strategy_id in self.strategy_orchestrator._active_strategies:
                strategy = self.strategy_orchestrator._active_strategies[strategy_id]
                await self.strategy_orchestrator._stop_strategy(strategy)
                
            # Remove from tracking
            del self.active_strategies[strategy_id]
            
    async def shutdown(self) -> None:
        """Shutdown the trading system gracefully."""
        print("\n🛑 Shutting down Autonomous Trading System...")
        
        self.is_running = False
        
        # Send shutdown notification
        if self.notification_service:
            uptime_str = "N/A"
            if self.start_time:
                uptime = datetime.utcnow() - self.start_time
                uptime_str = str(uptime).split('.')[0]  # Remove microseconds
            
            await self.notification_service.send_notification(
                "INFO",
                "🛑 System Shutdown",
                f"Trading system shutting down\nUptime: {uptime_str}"
            )
            
        # Stop all strategies
        for strategy_id in list(self.active_strategies.keys()):
            await self._stop_strategy(strategy_id)
            
        # Stop components
        if self.strategy_orchestrator:
            await self.strategy_orchestrator.stop()
        if self.market_analyzer:
            await self.market_analyzer.stop()
        if self.performance_monitor:
            await self.performance_monitor.stop()
            
        # Stop trading node
        if self.trading_node:
            # Check if stop is a coroutine or regular method
            if hasattr(self.trading_node, 'stop'):
                result = self.trading_node.stop()
                if hasattr(result, '__await__'):
                    await result
            
        print("✅ Shutdown complete")
        

async def main():
    """Main entry point."""
    # Load configuration file path from arguments or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_btc_start.json"
    
    # Create and initialize system
    system = AutonomousTradingSystem(config_path)
    
    try:
        await system.initialize()
        await system.start()
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt received")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())