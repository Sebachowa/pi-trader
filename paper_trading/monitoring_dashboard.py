
"""
Comprehensive monitoring dashboard for the Nautilus Trader Challenge.

This module provides a web-based monitoring dashboard that tracks:
- Real-time P&L and performance metrics
- Strategy health and execution status
- Risk metrics and exposure
- System health and connectivity
- Progress toward 10% annual return goal
"""

import asyncio
import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

from nautilus_trader.common.actor import Actor
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.identifiers import ComponentId, TraderId
from nautilus_trader.msgbus.bus import MessageBus
from nautilus_trader.portfolio.portfolio import Portfolio


class MonitoringDashboard(Actor):
    """
    Comprehensive monitoring dashboard for Nautilus Trader.
    
    Provides real-time monitoring, alerts, and analytics through a web interface.
    """
    
    def __init__(
        self,
        trader_id: TraderId,
        msgbus: MessageBus,
        clock: LiveClock,
        portfolio: Portfolio,
        host: str = "127.0.0.1",
        port: int = 5000,
        update_interval_secs: float = 1.0,
    ):
        component_id = ComponentId(f"{trader_id}-DASHBOARD")
        super().__init__(component_id=component_id, msgbus=msgbus)
        
        self._clock = clock
        self._portfolio = portfolio
        self._host = host
        self._port = port
        self._update_interval_secs = update_interval_secs
        
        # Flask app
        self._app = Flask(__name__)
        CORS(self._app)
        
        # Dashboard state
        self._metrics_cache = {}
        self._performance_history = []
        self._alerts = []
        self._strategy_status = {}
        self._system_health = {}
        
        # Challenge tracking
        self._challenge_start_balance = 100000.0  # $100k starting capital
        self._challenge_target_return = 0.10  # 10% annual return
        self._challenge_start_time = None
        
        # Risk limits
        self._risk_limits = {
            "max_drawdown": 0.20,  # 20% max drawdown
            "max_position_concentration": 0.25,  # 25% max position size
            "max_daily_loss": 0.05,  # 5% daily loss limit
            "max_leverage": 2.0,  # 2x max leverage
        }
        
        # Setup routes
        self._setup_routes()
        
        # Web server thread
        self._server_thread = None
        self._running = False
    
    def start(self):
        """Start the monitoring dashboard."""
        if self._running:
            return
        
        self._running = True
        self._challenge_start_time = self._clock.utc_now()
        
        # Start metrics update timer
        self._clock.set_timer(
            name="dashboard_update",
            interval=timedelta(seconds=self._update_interval_secs),
            callback=self._update_metrics,
        )
        
        # Start web server in separate thread
        self._server_thread = threading.Thread(target=self._run_server)
        self._server_thread.daemon = True
        self._server_thread.start()
        
        self.log.info(f"Monitoring dashboard started at http://{self._host}:{self._port}")
    
    def stop(self):
        """Stop the monitoring dashboard."""
        self._running = False
        self._clock.cancel_timer("dashboard_update")
        self.log.info("Monitoring dashboard stopped")
    
    def _run_server(self):
        """Run the Flask server."""
        self._app.run(host=self._host, port=self._port, debug=False)
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self._app.route("/")
        def index():
            """Serve the main dashboard page."""
            return render_template_string(self._get_dashboard_html())
        
        @self._app.route("/api/metrics")
        def get_metrics():
            """Get current metrics."""
            return jsonify(self._metrics_cache)
        
        @self._app.route("/api/performance_history")
        def get_performance_history():
            """Get performance history."""
            lookback = int(request.args.get("lookback_mins", 60))
            cutoff_time = self._clock.utc_now() - timedelta(minutes=lookback)
            
            history = [
                m for m in self._performance_history
                if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
            ]
            
            return jsonify(history)
        
        @self._app.route("/api/alerts")
        def get_alerts():
            """Get recent alerts."""
            return jsonify(self._alerts[-50:])  # Last 50 alerts
        
        @self._app.route("/api/strategy_status")
        def get_strategy_status():
            """Get strategy status."""
            return jsonify(self._strategy_status)
        
        @self._app.route("/api/system_health")
        def get_system_health():
            """Get system health."""
            return jsonify(self._system_health)
        
        @self._app.route("/api/challenge_progress")
        def get_challenge_progress():
            """Get challenge progress."""
            return jsonify(self._calculate_challenge_progress())
    
    def _update_metrics(self, event=None):
        """Update all metrics."""
        if not self._running:
            return
        
        try:
            # Collect current metrics
            metrics = self._collect_comprehensive_metrics()
            
            # Update cache
            self._metrics_cache = metrics
            
            # Add to history
            self._performance_history.append(metrics)
            
            # Trim history (keep last 24 hours)
            if len(self._performance_history) > 86400 / self._update_interval_secs:
                self._performance_history = self._performance_history[-int(86400 / self._update_interval_secs):]
            
            # Check alerts
            self._check_alerts(metrics)
            
            # Update system health
            self._update_system_health()
            
        except Exception as e:
            self.log.error(f"Error updating metrics: {e}")
    
    def _collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics."""
        timestamp = self._clock.utc_now()
        
        # Portfolio metrics
        balance_total = float(self._portfolio.balance_total())
        margin_total = float(self._portfolio.margin_total())
        margin_available = float(self._portfolio.margin_available_total())
        realized_pnl = float(self._portfolio.realized_pnl_total())
        unrealized_pnl = float(self._portfolio.unrealized_pnl_total())
        
        # Calculate returns
        total_pnl = realized_pnl + unrealized_pnl
        pnl_pct = (total_pnl / self._challenge_start_balance * 100) if self._challenge_start_balance > 0 else 0
        
        # Position analysis
        positions = self._portfolio.positions()
        position_metrics = self._analyze_positions(positions, balance_total)
        
        # Order analysis
        orders = self._portfolio.orders()
        order_metrics = self._analyze_orders(orders)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(
            balance_total, 
            margin_total, 
            position_metrics,
            self._performance_history
        )
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(
            self._performance_history,
            total_pnl
        )
        
        return {
            "timestamp": timestamp.isoformat(),
            "portfolio": {
                "balance_total": balance_total,
                "margin_total": margin_total,
                "margin_available": margin_available,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "pnl_pct": pnl_pct,
            },
            "positions": position_metrics,
            "orders": order_metrics,
            "risk": risk_metrics,
            "performance": performance_metrics,
        }
    
    def _analyze_positions(self, positions: List, balance_total: float) -> Dict[str, Any]:
        """Analyze current positions."""
        if not positions:
            return {
                "count": 0,
                "long_count": 0,
                "short_count": 0,
                "total_value": 0.0,
                "total_exposure": 0.0,
                "largest_position_pct": 0.0,
                "concentration_risk": "LOW",
                "positions_detail": []
            }
        
        position_details = []
        total_value = 0.0
        
        for pos in positions:
            value = abs(float(pos.notional_value))
            total_value += value
            
            position_details.append({
                "symbol": str(pos.instrument_id),
                "side": "LONG" if pos.is_long else "SHORT",
                "quantity": float(pos.quantity),
                "entry_price": float(pos.avg_px_open) if pos.avg_px_open else 0.0,
                "current_price": float(pos.last_px) if pos.last_px else 0.0,
                "value": value,
                "unrealized_pnl": float(pos.unrealized_pnl()),
                "pnl_pct": float(pos.pnl_pct()) if hasattr(pos, 'pnl_pct') else 0.0,
                "position_pct": (value / balance_total * 100) if balance_total > 0 else 0.0,
            })
        
        # Sort by value
        position_details.sort(key=lambda x: x["value"], reverse=True)
        
        # Calculate concentration
        largest_position_pct = (position_details[0]["value"] / balance_total * 100) if balance_total > 0 else 0.0
        
        # Determine concentration risk level
        if largest_position_pct > 20:
            concentration_risk = "HIGH"
        elif largest_position_pct > 15:
            concentration_risk = "MEDIUM"
        else:
            concentration_risk = "LOW"
        
        return {
            "count": len(positions),
            "long_count": sum(1 for p in positions if p.is_long),
            "short_count": sum(1 for p in positions if p.is_short),
            "total_value": total_value,
            "total_exposure": total_value / balance_total if balance_total > 0 else 0.0,
            "largest_position_pct": largest_position_pct,
            "concentration_risk": concentration_risk,
            "positions_detail": position_details[:10],  # Top 10 positions
        }
    
    def _analyze_orders(self, orders: List) -> Dict[str, Any]:
        """Analyze current orders."""
        if not orders:
            return {
                "count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "pending_count": 0,
                "orders_detail": []
            }
        
        order_details = []
        for order in orders[:20]:  # Last 20 orders
            order_details.append({
                "order_id": str(order.client_order_id),
                "symbol": str(order.instrument_id),
                "side": str(order.side),
                "type": str(order.order_type),
                "quantity": float(order.quantity),
                "price": float(order.price) if order.price else 0.0,
                "status": str(order.status),
                "created": unix_nanos_to_dt(order.ts_init).isoformat(),
            })
        
        return {
            "count": len(orders),
            "buy_count": sum(1 for o in orders if o.is_buy),
            "sell_count": sum(1 for o in orders if o.is_sell),
            "pending_count": sum(1 for o in orders if o.is_pending),
            "orders_detail": order_details,
        }
    
    def _calculate_risk_metrics(
        self, 
        balance: float, 
        margin: float, 
        position_metrics: Dict,
        history: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate risk metrics."""
        # Leverage
        leverage = (position_metrics["total_value"] / balance) if balance > 0 else 0.0
        
        # Margin usage
        margin_usage_pct = (margin / balance * 100) if balance > 0 else 0.0
        
        # Calculate drawdown
        if history:
            balances = [h["portfolio"]["balance_total"] for h in history]
            peak_balance = max(balances) if balances else balance
            current_drawdown = ((peak_balance - balance) / peak_balance * 100) if peak_balance > 0 else 0.0
        else:
            current_drawdown = 0.0
        
        # Risk score (0-100, higher is riskier)
        risk_score = min(100, (
            (leverage / self._risk_limits["max_leverage"]) * 25 +
            (position_metrics["largest_position_pct"] / (self._risk_limits["max_position_concentration"] * 100)) * 25 +
            (current_drawdown / (self._risk_limits["max_drawdown"] * 100)) * 25 +
            (margin_usage_pct / 100) * 25
        ))
        
        # Risk level
        if risk_score > 75:
            risk_level = "CRITICAL"
        elif risk_score > 50:
            risk_level = "HIGH"
        elif risk_score > 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "leverage": leverage,
            "margin_usage_pct": margin_usage_pct,
            "current_drawdown_pct": current_drawdown,
            "position_concentration": position_metrics["largest_position_pct"],
            "risk_score": risk_score,
            "risk_level": risk_level,
            "limits": {
                "max_leverage": self._risk_limits["max_leverage"],
                "max_drawdown_pct": self._risk_limits["max_drawdown"] * 100,
                "max_position_pct": self._risk_limits["max_position_concentration"] * 100,
                "max_daily_loss_pct": self._risk_limits["max_daily_loss"] * 100,
            }
        }
    
    def _calculate_performance_metrics(
        self, 
        history: List[Dict], 
        total_pnl: float
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not history or len(history) < 2:
            return {
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "trades_today": 0,
                "trades_total": 0,
            }
        
        # Calculate returns series
        returns = []
        for i in range(1, len(history)):
            prev_balance = history[i-1]["portfolio"]["balance_total"]
            curr_balance = history[i]["portfolio"]["balance_total"]
            if prev_balance > 0:
                ret = (curr_balance - prev_balance) / prev_balance
                returns.append(ret)
        
        # Calculate metrics
        total_return_pct = (total_pnl / self._challenge_start_balance * 100) if self._challenge_start_balance > 0 else 0.0
        
        # Simplified Sharpe ratio (annualized)
        if returns:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            periods_per_year = 252 * 24 * 3600 / self._update_interval_secs  # Trading seconds per year
            sharpe_ratio = (avg_return * periods_per_year) / (std_return * (periods_per_year ** 0.5)) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Win rate and profit factor would need trade history
        # For now, we'll estimate based on P&L changes
        positive_changes = sum(1 for r in returns if r > 0)
        win_rate = (positive_changes / len(returns) * 100) if returns else 0.0
        
        # Profit factor
        gains = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))
        profit_factor = gains / losses if losses > 0 else gains if gains > 0 else 0.0
        
        return {
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trades_today": 0,  # Would need trade history
            "trades_total": 0,  # Would need trade history
        }
    
    def _calculate_challenge_progress(self) -> Dict[str, Any]:
        """Calculate progress toward challenge goal."""
        if not self._challenge_start_time:
            return {}
        
        current_balance = float(self._portfolio.balance_total())
        total_pnl = current_balance - self._challenge_start_balance
        current_return = (total_pnl / self._challenge_start_balance) if self._challenge_start_balance > 0 else 0.0
        
        # Time elapsed
        elapsed = self._clock.utc_now() - self._challenge_start_time
        days_elapsed = elapsed.total_seconds() / 86400
        
        # Annualized return
        if days_elapsed > 0:
            annualized_return = current_return * (365 / days_elapsed)
        else:
            annualized_return = 0.0
        
        # Progress toward goal
        progress_pct = (annualized_return / self._challenge_target_return * 100) if self._challenge_target_return > 0 else 0.0
        
        # Days remaining (assume 30-day challenge)
        days_remaining = max(0, 30 - days_elapsed)
        
        # Required daily return to meet goal
        if days_remaining > 0 and current_return < self._challenge_target_return:
            remaining_return_needed = self._challenge_target_return - current_return
            required_daily_return = remaining_return_needed / days_remaining * 100
        else:
            required_daily_return = 0.0
        
        return {
            "start_balance": self._challenge_start_balance,
            "current_balance": current_balance,
            "total_pnl": total_pnl,
            "current_return_pct": current_return * 100,
            "target_return_pct": self._challenge_target_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "progress_pct": progress_pct,
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "required_daily_return_pct": required_daily_return,
            "on_track": annualized_return >= self._challenge_target_return,
        }
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions."""
        alerts = []
        timestamp = self._clock.utc_now().isoformat()
        
        # Risk alerts
        risk = metrics["risk"]
        
        if risk["leverage"] > self._risk_limits["max_leverage"]:
            alerts.append({
                "timestamp": timestamp,
                "type": "RISK",
                "severity": "CRITICAL",
                "message": f"Leverage exceeds limit: {risk['leverage']:.2f}x (max: {self._risk_limits['max_leverage']}x)",
            })
        
        if risk["current_drawdown_pct"] > self._risk_limits["max_drawdown"] * 100:
            alerts.append({
                "timestamp": timestamp,
                "type": "RISK",
                "severity": "HIGH",
                "message": f"Drawdown exceeds limit: {risk['current_drawdown_pct']:.2f}% (max: {self._risk_limits['max_drawdown'] * 100}%)",
            })
        
        if risk["position_concentration"] > self._risk_limits["max_position_concentration"] * 100:
            alerts.append({
                "timestamp": timestamp,
                "type": "RISK",
                "severity": "HIGH",
                "message": f"Position concentration too high: {risk['position_concentration']:.2f}% (max: {self._risk_limits['max_position_concentration'] * 100}%)",
            })
        
        if risk["margin_usage_pct"] > 90:
            alerts.append({
                "timestamp": timestamp,
                "type": "RISK",
                "severity": "CRITICAL",
                "message": f"Critical margin usage: {risk['margin_usage_pct']:.1f}%",
            })
        elif risk["margin_usage_pct"] > 75:
            alerts.append({
                "timestamp": timestamp,
                "type": "RISK",
                "severity": "WARNING",
                "message": f"High margin usage: {risk['margin_usage_pct']:.1f}%",
            })
        
        # Performance alerts
        challenge_progress = self._calculate_challenge_progress()
        if challenge_progress.get("on_track", False):
            if challenge_progress["progress_pct"] >= 100:
                alerts.append({
                    "timestamp": timestamp,
                    "type": "PERFORMANCE",
                    "severity": "SUCCESS",
                    "message": f"Challenge goal achieved! Current annualized return: {challenge_progress['annualized_return_pct']:.2f}%",
                })
        
        # Add alerts to history
        for alert in alerts:
            self._alerts.append(alert)
            
            # Log based on severity
            if alert["severity"] == "CRITICAL":
                self.log.error(f"CRITICAL ALERT: {alert['message']}")
            elif alert["severity"] == "HIGH":
                self.log.warning(f"HIGH ALERT: {alert['message']}")
            elif alert["severity"] == "SUCCESS":
                self.log.info(f"SUCCESS: {alert['message']}")
    
    def _update_system_health(self):
        """Update system health metrics."""
        self._system_health = {
            "dashboard_status": "RUNNING" if self._running else "STOPPED",
            "last_update": self._clock.utc_now().isoformat(),
            "metrics_buffer_size": len(self._performance_history),
            "alerts_count": len(self._alerts),
            "update_interval_secs": self._update_interval_secs,
        }
    
    def _get_dashboard_html(self) -> str:
        """Get the dashboard HTML template."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nautilus Trader - Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #0f0f0f;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 28px;
            font-weight: 600;
            background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: #1a1a1a;
            border: 1px solid #2d2d2d;
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        
        .metric-card.positive {
            border-color: #4caf50;
        }
        
        .metric-card.negative {
            border-color: #f44336;
        }
        
        .metric-card.warning {
            border-color: #ff9800;
        }
        
        .metric-label {
            font-size: 14px;
            color: #888;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .metric-change {
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .positive { color: #4caf50; }
        .negative { color: #f44336; }
        .warning { color: #ff9800; }
        
        .section {
            background: #1a1a1a;
            border: 1px solid #2d2d2d;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .section-title {
            font-size: 20px;
            font-weight: 600;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-bottom: 20px;
        }
        
        .positions-table, .alerts-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .positions-table th, .alerts-table th {
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid #2d2d2d;
            font-weight: 600;
            color: #888;
        }
        
        .positions-table td, .alerts-table td {
            padding: 12px;
            border-bottom: 1px solid #2d2d2d;
        }
        
        .positions-table tr:hover, .alerts-table tr:hover {
            background: #252525;
        }
        
        .alert-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .alert-critical { background: #f44336; }
        .alert-high { background: #ff5722; }
        .alert-warning { background: #ff9800; }
        .alert-info { background: #2196f3; }
        .alert-success { background: #4caf50; }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #2d2d2d;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4fc3f7 0%, #29b6f6 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: 600;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-running { background: #4caf50; }
        .status-stopped { background: #f44336; }
        
        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>Nautilus Trader - Challenge Monitoring Dashboard</h1>
            <p style="color: #888; margin-top: 5px;">Real-time performance monitoring and analytics</p>
        </div>
    </div>
    
    <div class="container">
        <!-- Challenge Progress -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Challenge Progress</h2>
                <span id="system-status">
                    <span class="status-indicator status-running"></span>
                    <span>System Running</span>
                </span>
            </div>
            <div id="challenge-progress">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%">0%</div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <span>Start: $100,000</span>
                    <span id="challenge-status">Current: $0</span>
                    <span>Target: 10% Annual Return</span>
                </div>
            </div>
        </div>
        
        <!-- Key Metrics -->
        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be populated by JavaScript -->
        </div>
        
        <!-- Performance Chart -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Performance Chart</h2>
                <select id="chart-timeframe">
                    <option value="60">1 Hour</option>
                    <option value="240">4 Hours</option>
                    <option value="1440">24 Hours</option>
                </select>
            </div>
            <div class="chart-container">
                <canvas id="performance-chart"></canvas>
            </div>
        </div>
        
        <!-- Active Positions -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Active Positions</h2>
                <span id="positions-count">0 positions</span>
            </div>
            <table class="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Size %</th>
                    </tr>
                </thead>
                <tbody id="positions-tbody">
                    <!-- Positions will be populated by JavaScript -->
                </tbody>
            </table>
        </div>
        
        <!-- Recent Alerts -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Recent Alerts</h2>
                <button onclick="clearAlerts()" style="background: #f44336; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Clear</button>
            </div>
            <table class="alerts-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Type</th>
                        <th>Severity</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody id="alerts-tbody">
                    <!-- Alerts will be populated by JavaScript -->
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Global variables
        let performanceChart = null;
        const updateInterval = 1000; // 1 second
        
        // Initialize chart
        const ctx = document.getElementById('performance-chart').getContext('2d');
        performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total P&L ($)',
                    data: [],
                    borderColor: '#4fc3f7',
                    backgroundColor: 'rgba(79, 195, 247, 0.1)',
                    tension: 0.1
                }, {
                    label: 'Balance ($)',
                    data: [],
                    borderColor: '#29b6f6',
                    backgroundColor: 'rgba(41, 182, 246, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#e0e0e0'
                        }
                    }
                },
                scales: {
                    y: {
                        ticks: {
                            color: '#888'
                        },
                        grid: {
                            color: '#2d2d2d'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#888'
                        },
                        grid: {
                            color: '#2d2d2d'
                        }
                    }
                }
            }
        });
        
        // Update functions
        async function updateDashboard() {
            try {
                // Fetch current metrics
                const metricsResponse = await fetch('/api/metrics');
                const metrics = await metricsResponse.json();
                
                // Update key metrics
                updateKeyMetrics(metrics);
                
                // Update positions
                updatePositions(metrics.positions);
                
                // Fetch and update challenge progress
                const challengeResponse = await fetch('/api/challenge_progress');
                const challenge = await challengeResponse.json();
                updateChallengeProgress(challenge);
                
                // Fetch and update alerts
                const alertsResponse = await fetch('/api/alerts');
                const alerts = await alertsResponse.json();
                updateAlerts(alerts);
                
                // Update chart
                await updateChart();
                
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }
        
        function updateKeyMetrics(metrics) {
            const gridHtml = `
                <div class="metric-card ${metrics.portfolio.total_pnl >= 0 ? 'positive' : 'negative'}">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value ${metrics.portfolio.total_pnl >= 0 ? 'positive' : 'negative'}">
                        $${formatNumber(metrics.portfolio.total_pnl)}
                    </div>
                    <div class="metric-change ${metrics.portfolio.pnl_pct >= 0 ? 'positive' : 'negative'}">
                        ${metrics.portfolio.pnl_pct >= 0 ? '+' : ''}${metrics.portfolio.pnl_pct.toFixed(2)}%
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Balance</div>
                    <div class="metric-value">$${formatNumber(metrics.portfolio.balance_total)}</div>
                    <div class="metric-change">Available: $${formatNumber(metrics.portfolio.margin_available)}</div>
                </div>
                
                <div class="metric-card ${metrics.risk.risk_level === 'LOW' ? 'positive' : metrics.risk.risk_level === 'HIGH' || metrics.risk.risk_level === 'CRITICAL' ? 'negative' : 'warning'}">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value ${metrics.risk.risk_level === 'LOW' ? 'positive' : metrics.risk.risk_level === 'HIGH' || metrics.risk.risk_level === 'CRITICAL' ? 'negative' : 'warning'}">
                        ${metrics.risk.risk_level}
                    </div>
                    <div class="metric-change">Score: ${metrics.risk.risk_score.toFixed(0)}/100</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Positions</div>
                    <div class="metric-value">${metrics.positions.count}</div>
                    <div class="metric-change">
                        Long: ${metrics.positions.long_count} | Short: ${metrics.positions.short_count}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Leverage</div>
                    <div class="metric-value ${metrics.risk.leverage > 1.5 ? 'warning' : ''}">${metrics.risk.leverage.toFixed(2)}x</div>
                    <div class="metric-change">Max: ${metrics.risk.limits.max_leverage}x</div>
                </div>
                
                <div class="metric-card ${metrics.risk.current_drawdown_pct <= -10 ? 'negative' : metrics.risk.current_drawdown_pct <= -5 ? 'warning' : ''}">
                    <div class="metric-label">Drawdown</div>
                    <div class="metric-value ${metrics.risk.current_drawdown_pct <= -10 ? 'negative' : metrics.risk.current_drawdown_pct <= -5 ? 'warning' : ''}">
                        ${metrics.risk.current_drawdown_pct.toFixed(2)}%
                    </div>
                    <div class="metric-change">Max: -${metrics.risk.limits.max_drawdown_pct}%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value ${metrics.performance.sharpe_ratio >= 1.5 ? 'positive' : metrics.performance.sharpe_ratio < 0.5 ? 'negative' : ''}">
                        ${metrics.performance.sharpe_ratio.toFixed(2)}
                    </div>
                    <div class="metric-change">Target: 1.5+</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value ${metrics.performance.win_rate >= 60 ? 'positive' : metrics.performance.win_rate < 40 ? 'negative' : ''}">
                        ${metrics.performance.win_rate.toFixed(1)}%
                    </div>
                    <div class="metric-change">Profit Factor: ${metrics.performance.profit_factor.toFixed(2)}</div>
                </div>
            `;
            
            document.getElementById('metrics-grid').innerHTML = gridHtml;
        }
        
        function updateChallengeProgress(challenge) {
            if (!challenge.progress_pct) return;
            
            const progressFill = document.getElementById('progress-fill');
            progressFill.style.width = `${Math.min(100, challenge.progress_pct)}%`;
            progressFill.textContent = `${challenge.progress_pct.toFixed(1)}%`;
            
            document.getElementById('challenge-status').textContent = 
                `Current: $${formatNumber(challenge.current_balance)} | ` +
                `Return: ${challenge.current_return_pct.toFixed(2)}% | ` +
                `Annualized: ${challenge.annualized_return_pct.toFixed(2)}%`;
            
            if (challenge.on_track) {
                progressFill.style.background = 'linear-gradient(90deg, #4caf50 0%, #66bb6a 100%)';
            } else {
                progressFill.style.background = 'linear-gradient(90deg, #ff9800 0%, #ffa726 100%)';
            }
        }
        
        function updatePositions(positions) {
            if (!positions.positions_detail) return;
            
            document.getElementById('positions-count').textContent = `${positions.count} positions`;
            
            const tbody = document.getElementById('positions-tbody');
            const rows = positions.positions_detail.map(pos => `
                <tr>
                    <td>${pos.symbol}</td>
                    <td class="${pos.side === 'LONG' ? 'positive' : 'negative'}">${pos.side}</td>
                    <td>${pos.quantity.toFixed(4)}</td>
                    <td>$${pos.entry_price.toFixed(2)}</td>
                    <td>$${pos.current_price.toFixed(2)}</td>
                    <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                        $${formatNumber(pos.unrealized_pnl)}
                    </td>
                    <td class="${pos.pnl_pct >= 0 ? 'positive' : 'negative'}">
                        ${pos.pnl_pct >= 0 ? '+' : ''}${pos.pnl_pct.toFixed(2)}%
                    </td>
                    <td class="${pos.position_pct > 15 ? 'warning' : ''}">
                        ${pos.position_pct.toFixed(1)}%
                    </td>
                </tr>
            `).join('');
            
            tbody.innerHTML = rows || '<tr><td colspan="8" style="text-align: center; color: #888;">No active positions</td></tr>';
        }
        
        function updateAlerts(alerts) {
            const tbody = document.getElementById('alerts-tbody');
            const recentAlerts = alerts.slice(-10).reverse();
            
            const rows = recentAlerts.map(alert => `
                <tr>
                    <td>${new Date(alert.timestamp).toLocaleTimeString()}</td>
                    <td>${alert.type}</td>
                    <td><span class="alert-badge alert-${alert.severity.toLowerCase()}">${alert.severity}</span></td>
                    <td>${alert.message}</td>
                </tr>
            `).join('');
            
            tbody.innerHTML = rows || '<tr><td colspan="4" style="text-align: center; color: #888;">No alerts</td></tr>';
        }
        
        async function updateChart() {
            const timeframe = document.getElementById('chart-timeframe').value;
            const response = await fetch(`/api/performance_history?lookback_mins=${timeframe}`);
            const history = await response.json();
            
            if (history.length === 0) return;
            
            const labels = history.map(h => new Date(h.timestamp).toLocaleTimeString());
            const pnlData = history.map(h => h.portfolio.total_pnl);
            const balanceData = history.map(h => h.portfolio.balance_total);
            
            performanceChart.data.labels = labels;
            performanceChart.data.datasets[0].data = pnlData;
            performanceChart.data.datasets[1].data = balanceData;
            performanceChart.update();
        }
        
        function formatNumber(num) {
            return new Intl.NumberFormat('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }).format(num);
        }
        
        function clearAlerts() {
            // This would need an API endpoint to clear alerts
            console.log('Clear alerts functionality would be implemented here');
        }
        
        // Start updates
        updateDashboard();
        setInterval(updateDashboard, updateInterval);
        
        // Chart timeframe change
        document.getElementById('chart-timeframe').addEventListener('change', updateChart);
    </script>
</body>
</html>
        '''


# Convenience function to create and start the dashboard
def create_monitoring_dashboard(
    trader_id: TraderId,
    msgbus: MessageBus,
    clock: LiveClock,
    portfolio: Portfolio,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> MonitoringDashboard:
    """
    Create and return a monitoring dashboard instance.
    
    Parameters
    ----------
    trader_id : TraderId
        The trader ID.
    msgbus : MessageBus
        The message bus.
    clock : LiveClock
        The system clock.
    portfolio : Portfolio
        The portfolio to monitor.
    host : str
        The host to bind to.
    port : int
        The port to bind to.
        
    Returns
    -------
    MonitoringDashboard
        The dashboard instance.
        
    """
    dashboard = MonitoringDashboard(
        trader_id=trader_id,
        msgbus=msgbus,
        clock=clock,
        portfolio=portfolio,
        host=host,
        port=port,
    )
    
    return dashboard