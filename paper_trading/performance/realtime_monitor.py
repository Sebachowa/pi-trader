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
Real-time performance monitoring for paper trading.

This module provides real-time monitoring capabilities for paper trading sessions
including live dashboards and alerts.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from nautilus_trader.common.actor import Actor
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.msgbus.bus import MessageBus
from nautilus_trader.portfolio.portfolio import Portfolio


class RealtimeMonitor(Actor):
    """
    Real-time performance monitor for paper trading.
    
    This actor monitors portfolio performance in real-time and provides:
    - Live metrics updates
    - Performance alerts
    - Risk warnings
    - Periodic snapshots
    
    Parameters
    ----------
    trader_id : TraderId
        The trader ID for the session.
    msgbus : MessageBus
        The message bus for the system.
    clock : LiveClock
        The clock for the system.
    portfolio : Portfolio
        The portfolio to monitor.
    update_interval_secs : float
        Update interval in seconds.
    alert_callbacks : dict[str, Callable], optional
        Alert callback functions.
        
    """
    
    def __init__(
        self,
        trader_id: TraderId,
        msgbus: MessageBus,
        clock: LiveClock,
        portfolio: Portfolio,
        update_interval_secs: float = 60.0,
        alert_callbacks: dict[str, Callable] | None = None,
    ):
        component_id = ComponentId(f"{trader_id}-MONITOR")
        super().__init__(component_id=component_id, msgbus=msgbus)
        
        self._clock = clock
        self._portfolio = portfolio
        self._update_interval_secs = update_interval_secs
        self._alert_callbacks = alert_callbacks or {}
        
        # Monitoring state
        self._metrics_buffer: list[dict[str, Any]] = []
        self._alert_history: list[dict[str, Any]] = []
        self._last_snapshot_time = None
        self._monitoring_active = False
        
        # Risk thresholds
        self.risk_thresholds = {
            "max_drawdown_pct": 10.0,
            "max_position_size_pct": 20.0,
            "max_daily_loss_pct": 5.0,
            "margin_usage_warning_pct": 80.0,
            "margin_usage_critical_pct": 95.0,
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "profit_target_daily_pct": 2.0,
            "profit_target_weekly_pct": 5.0,
            "win_rate_target": 0.6,
            "sharpe_target": 1.5,
        }
        
        # State tracking
        self._start_balance = None
        self._daily_high_balance = None
        self._session_high_balance = None
        self._last_daily_reset = None
    
    def start(self):
        """Start real-time monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._start_balance = float(self._portfolio.balance_total())
        self._daily_high_balance = self._start_balance
        self._session_high_balance = self._start_balance
        self._last_daily_reset = self._clock.utc_now()
        
        # Schedule periodic updates
        self._clock.set_timer(
            name="monitor_update",
            interval=timedelta(seconds=self._update_interval_secs),
            callback=self._update_metrics,
        )
        
        # Schedule daily reset
        self._clock.set_timer(
            name="daily_reset",
            interval=timedelta(hours=24),
            callback=self._daily_reset,
        )
        
        self.log.info(f"Started real-time monitoring with {self._update_interval_secs}s updates")
    
    def stop(self):
        """Stop real-time monitoring."""
        self._monitoring_active = False
        self._clock.cancel_timer("monitor_update")
        self._clock.cancel_timer("daily_reset")
        self.log.info("Stopped real-time monitoring")
    
    def _update_metrics(self, event=None):
        """Update performance metrics."""
        if not self._monitoring_active:
            return
        
        current_time = self._clock.utc_now()
        
        # Collect current metrics
        metrics = self._collect_metrics(current_time)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Update tracking variables
        self._update_tracking(metrics)
        
        # Buffer metrics
        self._metrics_buffer.append(metrics)
        
        # Trim buffer if too large
        if len(self._metrics_buffer) > 1000:
            self._metrics_buffer = self._metrics_buffer[-500:]
        
        # Log summary
        self.log.info(
            f"Monitor Update - "
            f"PnL: ${metrics['total_pnl']:.2f} "
            f"({metrics['pnl_pct']:.2f}%), "
            f"Positions: {metrics['position_count']}, "
            f"Margin: {metrics['margin_usage_pct']:.1f}%"
        )
    
    def _collect_metrics(self, timestamp: datetime) -> dict[str, Any]:
        """Collect current performance metrics."""
        balance_total = float(self._portfolio.balance_total())
        margin_total = float(self._portfolio.margin_total())
        margin_available = float(self._portfolio.margin_available_total())
        
        # Calculate PnL metrics
        total_pnl = balance_total - self._start_balance
        pnl_pct = (total_pnl / self._start_balance * 100) if self._start_balance > 0 else 0
        
        # Daily metrics
        daily_pnl = balance_total - self._daily_high_balance
        daily_pnl_pct = (daily_pnl / self._daily_high_balance * 100) if self._daily_high_balance > 0 else 0
        
        # Drawdown metrics
        session_drawdown = balance_total - self._session_high_balance
        session_drawdown_pct = (session_drawdown / self._session_high_balance * 100) if self._session_high_balance > 0 else 0
        
        # Position metrics
        positions = self._portfolio.positions()
        position_metrics = {
            "count": len(positions),
            "long_count": sum(1 for p in positions if p.is_long),
            "short_count": sum(1 for p in positions if p.is_short),
            "total_value": sum(abs(float(p.notional_value)) for p in positions),
            "unrealized_pnl": sum(float(p.unrealized_pnl()) for p in positions),
        }
        
        # Order metrics
        orders = self._portfolio.orders()
        order_metrics = {
            "count": len(orders),
            "buy_count": sum(1 for o in orders if o.is_buy),
            "sell_count": sum(1 for o in orders if o.is_sell),
        }
        
        # Risk metrics
        margin_usage_pct = (margin_total / balance_total * 100) if balance_total > 0 else 0
        
        return {
            "timestamp": timestamp.isoformat(),
            "balance_total": balance_total,
            "margin_total": margin_total,
            "margin_available": margin_available,
            "margin_usage_pct": margin_usage_pct,
            "total_pnl": total_pnl,
            "pnl_pct": pnl_pct,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "session_drawdown": session_drawdown,
            "session_drawdown_pct": session_drawdown_pct,
            "position_count": position_metrics["count"],
            "position_metrics": position_metrics,
            "order_metrics": order_metrics,
            "realized_pnl": float(self._portfolio.realized_pnl_total()),
            "unrealized_pnl": float(self._portfolio.unrealized_pnl_total()),
        }
    
    def _check_alerts(self, metrics: dict[str, Any]):
        """Check for alert conditions."""
        alerts = []
        
        # Risk alerts
        if abs(metrics["session_drawdown_pct"]) > self.risk_thresholds["max_drawdown_pct"]:
            alerts.append({
                "type": "RISK",
                "severity": "HIGH",
                "message": f"Maximum drawdown exceeded: {metrics['session_drawdown_pct']:.2f}%",
            })
        
        if abs(metrics["daily_pnl_pct"]) > self.risk_thresholds["max_daily_loss_pct"] and metrics["daily_pnl"] < 0:
            alerts.append({
                "type": "RISK",
                "severity": "HIGH",
                "message": f"Daily loss limit approached: {metrics['daily_pnl_pct']:.2f}%",
            })
        
        if metrics["margin_usage_pct"] > self.risk_thresholds["margin_usage_critical_pct"]:
            alerts.append({
                "type": "RISK",
                "severity": "CRITICAL",
                "message": f"Critical margin usage: {metrics['margin_usage_pct']:.1f}%",
            })
        elif metrics["margin_usage_pct"] > self.risk_thresholds["margin_usage_warning_pct"]:
            alerts.append({
                "type": "RISK",
                "severity": "WARNING",
                "message": f"High margin usage: {metrics['margin_usage_pct']:.1f}%",
            })
        
        # Performance alerts
        if metrics["daily_pnl_pct"] > self.performance_thresholds["profit_target_daily_pct"]:
            alerts.append({
                "type": "PERFORMANCE",
                "severity": "INFO",
                "message": f"Daily profit target reached: {metrics['daily_pnl_pct']:.2f}%",
            })
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert, metrics)
    
    def _process_alert(self, alert: dict[str, Any], metrics: dict[str, Any]):
        """Process and dispatch alerts."""
        alert["timestamp"] = self._clock.utc_now().isoformat()
        alert["metrics_snapshot"] = metrics
        
        # Add to history
        self._alert_history.append(alert)
        
        # Log alert
        log_method = self.log.error if alert["severity"] == "CRITICAL" else (
            self.log.warning if alert["severity"] in ["HIGH", "WARNING"] else self.log.info
        )
        log_method(f"ALERT [{alert['severity']}] {alert['type']}: {alert['message']}")
        
        # Call registered callbacks
        callback_key = f"{alert['type']}_{alert['severity']}"
        if callback_key in self._alert_callbacks:
            try:
                self._alert_callbacks[callback_key](alert)
            except Exception as e:
                self.log.error(f"Alert callback error: {e}")
    
    def _update_tracking(self, metrics: dict[str, Any]):
        """Update tracking variables."""
        balance = metrics["balance_total"]
        
        # Update session high
        if balance > self._session_high_balance:
            self._session_high_balance = balance
        
        # Update daily high
        if balance > self._daily_high_balance:
            self._daily_high_balance = balance
    
    def _daily_reset(self, event=None):
        """Reset daily tracking variables."""
        self._daily_high_balance = float(self._portfolio.balance_total())
        self._last_daily_reset = self._clock.utc_now()
        self.log.info("Daily tracking variables reset")
    
    def get_current_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return self._collect_metrics(self._clock.utc_now())
    
    def get_metrics_history(self, lookback_mins: int = 60) -> list[dict[str, Any]]:
        """Get metrics history for specified lookback period."""
        cutoff_time = self._clock.utc_now() - timedelta(minutes=lookback_mins)
        cutoff_str = cutoff_time.isoformat()
        
        return [
            m for m in self._metrics_buffer
            if m["timestamp"] >= cutoff_str
        ]
    
    def get_alert_history(self, severity: str | None = None) -> list[dict[str, Any]]:
        """Get alert history, optionally filtered by severity."""
        if severity:
            return [a for a in self._alert_history if a["severity"] == severity]
        return self._alert_history.copy()
    
    def export_session_data(self, output_path: Path):
        """Export session monitoring data."""
        session_data = {
            "session_info": {
                "trader_id": str(self.trader_id),
                "start_time": self._last_daily_reset.isoformat() if self._last_daily_reset else None,
                "export_time": self._clock.utc_now().isoformat(),
                "monitoring_active": self._monitoring_active,
            },
            "current_metrics": self.get_current_metrics(),
            "performance_summary": {
                "start_balance": self._start_balance,
                "current_balance": float(self._portfolio.balance_total()),
                "session_high_balance": self._session_high_balance,
                "total_pnl": float(self._portfolio.balance_total()) - self._start_balance,
                "total_pnl_pct": ((float(self._portfolio.balance_total()) - self._start_balance) / 
                                 self._start_balance * 100) if self._start_balance > 0 else 0,
            },
            "risk_thresholds": self.risk_thresholds,
            "performance_thresholds": self.performance_thresholds,
            "metrics_history": self._metrics_buffer,
            "alert_history": self._alert_history,
        }
        
        with open(output_path, "w") as f:
            json.dump(session_data, f, indent=2)
        
        self.log.info(f"Exported session data to {output_path}")