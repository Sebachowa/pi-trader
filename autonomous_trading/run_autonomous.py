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
Main runner script for the Nautilus Autonomous Trading System.

This script demonstrates how to launch the fully autonomous trading system
with minimal user intervention.
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path

from nautilus_trader.common.logging import Logger
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId

from autonomous_trading.config.autonomous_config import (
    AutonomousSystemConfig,
    load_autonomous_config,
    create_live_trading_config,
)
from autonomous_trading.core.engine import AutonomousEngine, AutonomousEngineConfig
from autonomous_trading.monitoring.notifications import NotificationConfig


class AutonomousTradingSystem:
    """Main autonomous trading system runner."""
    
    def __init__(self, config: AutonomousSystemConfig):
        self.config = config
        self.engine = None
        self.trading_node = None
        self.logger = None
        self._running = False
        
    async def initialize(self) -> None:
        """Initialize the autonomous trading system."""
        print(f"Initializing {self.config.system_name}...")
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        
        # Create trader ID
        trader_id = TraderId(self.config.trader_id)
        
        # Create trading node configuration
        node_config = create_live_trading_config(self.config)
        
        # Initialize trading node
        self.trading_node = TradingNode(config=node_config)
        
        # Get core components
        self.logger = self.trading_node.logger
        clock = self.trading_node.clock
        msgbus = self.trading_node.msgbus
        
        # Create autonomous engine configuration
        engine_config = AutonomousEngineConfig(
            trader_id=self.config.trader_id,
            enable_self_healing=self.config.enable_self_healing,
            health_check_interval_seconds=self.config.health_check_interval_seconds,
            max_recovery_attempts=self.config.max_recovery_attempts,
            recovery_delay_seconds=self.config.recovery_delay_seconds,
            enable_auto_shutdown=self.config.enable_auto_shutdown,
            daily_maintenance_time=self.config.daily_maintenance_time,
            max_daily_loss_percent=self.config.risk_limits.max_daily_loss_percent,
            max_drawdown_percent=self.config.risk_limits.max_drawdown_percent,
            enable_notifications=self.config.enable_notifications,
            state_persistence_path=self.config.state_persistence_path,
        )
        
        # Initialize autonomous engine
        self.engine = AutonomousEngine(
            config=engine_config,
            trading_node=self.trading_node,
            logger=self.logger,
            clock=clock,
            msgbus=msgbus,
        )
        
        # Configure notification system if enabled
        if self.config.enable_notifications and self.config.notification_config:
            # Set up notification configuration
            # This would be loaded from config or environment
            pass
        
        print(f"{self.config.system_name} initialized successfully")
        
    async def start(self) -> None:
        """Start the autonomous trading system."""
        print(f"Starting {self.config.system_name}...")
        print(f"Target Annual Return: {self.config.target_annual_return_percent}%")
        print(f"Risk Limits - Max DD: {self.config.risk_limits.max_drawdown_percent}%, " 
              f"Max Daily Loss: {self.config.risk_limits.max_daily_loss_percent}%")
        print(f"Trading Instruments: {', '.join(self.config.instruments)}")
        
        self._running = True
        
        try:
            # Start the autonomous engine
            await self.engine.start()
            
            print(f"\n{self.config.system_name} is now running autonomously.")
            print("Press Ctrl+C to stop the system gracefully.")
            
            # Keep running until interrupted
            while self._running:
                await asyncio.sleep(1)
                
                # Check system health
                if self.engine.is_healthy:
                    # Optionally print status updates
                    pass
                else:
                    print("\nWARNING: System health check failed!")
                    
        except KeyboardInterrupt:
            print("\nShutdown signal received...")
        except Exception as e:
            print(f"\nERROR: {e}")
            self.logger.error(f"System error: {e}", exc_info=True)
        finally:
            await self.stop()
            
    async def stop(self) -> None:
        """Stop the autonomous trading system."""
        print(f"\nStopping {self.config.system_name}...")
        self._running = False
        
        if self.engine:
            await self.engine.stop()
            
        print(f"{self.config.system_name} stopped successfully")
        
    def get_system_report(self) -> None:
        """Print current system status report."""
        if not self.engine:
            print("System not initialized")
            return
            
        stats = self.engine.system_stats
        
        print(f"\n{'='*60}")
        print(f"{self.config.system_name} Status Report")
        print(f"{'='*60}")
        print(f"State: {stats['state']}")
        print(f"Health: {stats['health_status']}")
        print(f"Uptime: {stats['uptime_seconds']} seconds")
        print(f"Daily P&L: {stats['daily_pnl']:.2%}")
        print(f"Current Drawdown: {stats['current_drawdown']:.2%}")
        print(f"Error Count: {stats['error_count']}")
        print(f"Recovery Attempts: {stats['recovery_attempts']}")
        print(f"Circuit Breakers: {stats['circuit_breakers']}")
        print(f"Kill Switch Active: {stats['kill_switch_active']}")
        print(f"{'='*60}\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nautilus Autonomous Trading System - Minimal intervention trading"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON)",
        default=None,
    )
    
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable paper trading mode",
        default=False,
    )
    
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        help="List of instruments to trade (e.g., BTCUSDT.BINANCE)",
        default=None,
    )
    
    parser.add_argument(
        "--target-return",
        type=float,
        help="Target annual return percentage",
        default=10.0,
    )
    
    parser.add_argument(
        "--max-drawdown",
        type=float,
        help="Maximum drawdown percentage",
        default=10.0,
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
        default="INFO",
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_autonomous_config(args.config)
    
    # Override with command line arguments
    if args.paper:
        config.enable_paper_trading = True
        
    if args.instruments:
        config.instruments = args.instruments
        
    config.target_annual_return_percent = args.target_return
    config.risk_limits.max_drawdown_percent = args.max_drawdown
    config.log_level = args.log_level
    
    # Print startup banner
    print("\n" + "="*60)
    print("   NAUTILUS AUTONOMOUS TRADING SYSTEM   ")
    print("          Targeting 10% Annual Returns          ")
    print("          With Minimal Intervention             ")
    print("="*60 + "\n")
    
    # Create and run the autonomous system
    system = AutonomousTradingSystem(config)
    
    try:
        # Initialize the system
        await system.initialize()
        
        # Start autonomous trading
        await system.start()
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        logging.exception("Fatal error in autonomous system")
        sys.exit(1)


if __name__ == "__main__":
    # Set up signal handlers
    def signal_handler(sig, frame):
        print("\nShutdown signal received...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the autonomous trading system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem shutdown by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)