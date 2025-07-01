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
Run the monitoring dashboard for Nautilus Trader.

This script demonstrates how to integrate and run the monitoring dashboard
with a Nautilus Trader trading system.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.component import Logger
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId

from paper_trading.monitoring_dashboard import MonitoringDashboard, create_monitoring_dashboard


async def main():
    """
    Main function to run the monitoring dashboard.
    """
    # Configuration
    config = TradingNodeConfig(
        trader_id=TraderId("PAPER-001"),
        log_level="INFO",
        cache_database_config=None,  # In-memory cache
        data_engine_config={
            "type": "DataEngine",
            "config": {
                "time_bars_build_on_trade": True,
                "time_bars_timestamp_on_close": True,
                "validate_data_sequence": True,
            }
        },
        risk_engine_config={
            "type": "RiskEngine",
            "config": {
                "bypass": False,
                "max_order_submit_rate": "100/00:00:01",
                "max_order_modify_rate": "100/00:00:01",
                "max_notional_per_order": {},
            }
        },
        exec_engine_config={
            "type": "ExecutionEngine",
            "config": {
                "allow_cash_positions": True,
                "allow_pending_cancellation": True,
                "support_gtd_orders": True,
                "support_contingent_orders": True,
                "use_position_ids": True,
                "use_random_ids": False,
                "use_reduce_only": True,
            }
        },
    )
    
    # Create trading node
    node = TradingNode(config)
    
    # Build the node
    node.build()
    
    # Get components
    portfolio = node.portfolio
    msgbus = node.msgbus
    clock = node.clock
    
    # Create and add monitoring dashboard
    dashboard = create_monitoring_dashboard(
        trader_id=config.trader_id,
        msgbus=msgbus,
        clock=clock,
        portfolio=portfolio,
        host="127.0.0.1",
        port=5000,
    )
    
    # Add dashboard as an actor
    node.trader.add_actor(dashboard)
    
    # Start the node
    node.start()
    
    # Start the dashboard
    dashboard.start()
    
    print("\n" + "="*60)
    print("NAUTILUS TRADER MONITORING DASHBOARD")
    print("="*60)
    print(f"Dashboard URL: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        # Keep running until interrupted
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Stop components
        dashboard.stop()
        node.stop()
        node.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDashboard stopped.")