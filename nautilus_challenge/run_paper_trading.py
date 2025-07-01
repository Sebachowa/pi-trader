#!/usr/bin/env python3
"""
Nautilus Challenge - Real Paper Trading with Live Market Data
Professional trading system for 2-week evaluation before live trading
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from nautilus_trader.adapters.binance import BinanceDataClientConfig, BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance import BinanceExecClientConfig, BinanceLiveExecClientFactory
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.logging import Logger
from nautilus_trader.config import DataEngineConfig
from nautilus_trader.config import ExecEngineConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveRiskEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import MessageBusConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.data.engine import DataEngine
from nautilus_trader.execution.engine import ExecutionEngine
from nautilus_trader.live.data_engine import LiveDataEngine
from nautilus_trader.live.execution_engine import LiveExecutionEngine
from nautilus_trader.live.node import TradingNode
from nautilus_trader.live.risk_engine import LiveRiskEngine
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.msgbus.bus import MessageBus
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.risk.engine import RiskEngine

# Import strategies
sys.path.append(str(Path(__file__).parent.parent))
from nautilus_challenge.strategies.trend_following import TrendFollowingStrategy
from nautilus_challenge.monitoring.telegram_notifier import TelegramNotifier
from nautilus_challenge.monitoring.performance_tracker import PerformanceTracker


class NautilusChallenge:
    """Main class for running the Nautilus trading challenge."""
    
    def __init__(self):
        """Initialize the challenge."""
        self.config_path = Path(__file__).parent / "config" / "trading_config.json"
        self.load_config()
        self.setup_logging()
        self.notifier = TelegramNotifier()
        self.tracker = PerformanceTracker()
        
    def load_config(self):
        """Load trading configuration."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
    def setup_logging(self):
        """Set up logging configuration."""
        self.log_config = LoggingConfig(
            log_level="INFO",
            log_to_console=True,
            log_to_file=True,
            log_file_path=str(Path(__file__).parent / "logs" / f"nautilus_{datetime.now():%Y%m%d_%H%M%S}.log"),
        )
        
    async def setup_binance_connection(self):
        """Set up Binance data and execution clients."""
        # Data client configuration (public data, no API key needed)
        data_config = BinanceDataClientConfig(
            api_key=os.getenv("BINANCE_API_KEY", ""),  
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            testnet=False,  # Use real data
            base_url_http="https://api.binance.com",
            base_url_ws="wss://stream.binance.com:9443/ws",
            us=False,
            instrument_provider=InstrumentProviderConfig(
                load_all=False,
                filters={"symbols": self.config["instruments"]},
            ),
        )
        
        # Execution client for paper trading
        exec_config = BinanceExecClientConfig(
            api_key=os.getenv("BINANCE_TESTNET_API_KEY", "paper_trading"),
            api_secret=os.getenv("BINANCE_TESTNET_API_SECRET", "paper_trading"),
            testnet=True,  # Paper trading mode
            base_url_http="https://testnet.binance.vision",
            base_url_ws="wss://testnet.binance.vision/ws",
            us=False,
        )
        
        return data_config, exec_config
        
    async def run_challenge(self):
        """Run the paper trading challenge."""
        print("=" * 80)
        print(f"🚀 NAUTILUS TRADING CHALLENGE - REAL PAPER TRADING")
        print(f"📅 Start: {datetime.now()}")
        print(f"💰 Initial Capital: {self.config['challenge']['initial_capital_btc']} BTC")
        print(f"🎯 Target: {self.config['challenge']['target_annual_return']*100}% annual")
        print(f"📊 Duration: {self.config['challenge']['duration_days']} days")
        print("=" * 80)
        
        # Send start notification
        await self.notifier.send_challenge_start(self.config['challenge'])
        
        # Get Binance configuration
        data_config, exec_config = await self.setup_binance_connection()
        
        # Create trading node configuration
        node_config = TradingNodeConfig(
            trader_id="PAPER-001",
            logging=self.log_config,
            data_clients={
                "BINANCE": data_config,
            },
            exec_clients={
                "BINANCE": exec_config,
            },
            data_engine=DataEngineConfig(
                time_bars_timestamp_on_close=True,
                validate_data_sequence=True,
            ),
            risk_engine=LiveRiskEngineConfig(
                bypass=False,
                max_order_submit_rate="100/00:00:01",
                max_order_modify_rate="100/00:00:01",
                max_notional_per_order={"BTC": 100_000.0},
            ),
            exec_engine=ExecEngineConfig(
                load_cache=True,
                allow_cash_positions=True,
            ),
            streaming=None,  # Disable for paper trading
        )
        
        # Create and configure trading node
        self.node = TradingNode(config=node_config)
        
        # Initialize strategies
        strategies = []
        
        # Trend Following Strategy
        for instrument in self.config["instruments"]:
            if self.config["strategies"]["trend_following"]["enabled"]:
                strategy = TrendFollowingStrategy(
                    instrument_id=InstrumentId.from_str(f"{instrument}.BINANCE"),
                    bar_type=f"{instrument}.BINANCE-1-MINUTE-LAST-INTERNAL",
                    trade_size=Decimal(str(self.config["challenge"]["initial_capital_btc"] * 0.01)),
                )
                strategies.append(strategy)
                self.node.trader.add_strategy(strategy)
                
        # Subscribe to market data
        for instrument in self.config["instruments"]:
            # Subscribe to bars
            self.node.subscribe_bars(
                bar_type=f"{instrument}.BINANCE-1-MINUTE-LAST-INTERNAL",
                client_id="BINANCE",
            )
            # Subscribe to quotes
            self.node.subscribe_quote_ticks(
                instrument_id=InstrumentId.from_str(f"{instrument}.BINANCE"),
                client_id="BINANCE",
            )
            
        print(f"\n✅ Loaded {len(strategies)} strategies")
        print(f"📊 Monitoring {len(self.config['instruments'])} instruments")
        print(f"📱 Telegram notifications: ENABLED")
        print(f"💻 Dashboard: http://localhost:8080")
        print("\n🔄 Starting live market data feed...\n")
        
        try:
            # Run the trading node
            await self.node.run_async()
            
        except KeyboardInterrupt:
            print("\n⏹️ Stopping paper trading...")
            
        finally:
            # Generate final report
            await self.tracker.generate_final_report()
            await self.notifier.send_challenge_complete(self.tracker.get_summary())
            
            # Cleanup
            await self.node.stop_async()
            await self.node.dispose_async()
            
    async def monitor_performance(self):
        """Monitor performance and send periodic updates."""
        while self.node.is_running:
            await asyncio.sleep(3600)  # Check every hour
            
            # Get performance metrics
            metrics = self.tracker.get_current_metrics()
            
            # Send update if significant changes
            if metrics['total_trades'] > 0:
                await self.notifier.send_performance_update(metrics)
                
            # Check if we need to reset daily stats
            if datetime.now().hour == 0:
                for strategy in self.node.trader.strategies():
                    strategy.reset_daily_stats()


async def main():
    """Main entry point."""
    challenge = NautilusChallenge()
    
    # Run challenge and monitoring concurrently
    await asyncio.gather(
        challenge.run_challenge(),
        challenge.monitor_performance(),
    )


if __name__ == "__main__":
    asyncio.run(main())