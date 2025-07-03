#!/usr/bin/env python3
"""
System Monitor for Autonomous Trading
Provides real-time monitoring, logging, and alerting
"""

import asyncio
import json
import logging
import os
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import aiofiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


class SystemMonitor:
    """
    Comprehensive monitoring system for the autonomous trader.
    
    Features:
    - Real-time performance metrics
    - System health monitoring
    - Trade activity logging
    - Alert management
    - Web dashboard
    - Historical data storage
    """
    
    def __init__(
        self,
        db_path: str = "./data/monitoring.db",
        log_dir: str = "./logs",
        alert_thresholds: Optional[Dict[str, float]] = None,
        web_enabled: bool = True,
        web_port: int = 8080,
    ):
        self.db_path = Path(db_path)
        self.log_dir = Path(log_dir)
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.web_enabled = web_enabled
        self.web_port = web_port
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize database
        self._init_database()
        
        # Metrics storage
        self.metrics = {
            "system": defaultdict(lambda: deque(maxlen=1000)),
            "performance": defaultdict(lambda: deque(maxlen=1000)),
            "trades": deque(maxlen=1000),
            "alerts": deque(maxlen=100),
        }
        
        # Current state
        self.current_state = {
            "system_health": "healthy",
            "active_strategies": 0,
            "open_positions": 0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "uptime": timedelta(),
            "last_update": datetime.utcnow(),
        }
        
        # Web application
        if self.web_enabled:
            self.app = self._create_web_app()
            self.websocket_clients: List[WebSocket] = []
            
    def _default_thresholds(self) -> Dict[str, float]:
        """Default alert thresholds."""
        return {
            "max_drawdown": 0.05,
            "daily_loss": 0.03,
            "memory_usage_mb": 4096,
            "cpu_usage_percent": 80,
            "error_rate_per_hour": 10,
            "latency_ms": 1000,
            "min_win_rate": 0.40,
        }
        
    def _setup_logging(self) -> None:
        """Setup structured logging."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Main logger
        self.logger = logging.getLogger("SystemMonitor")
        self.logger.setLevel(logging.INFO)
        
        # File handler with rotation
        log_file = self.log_dir / f"monitor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)
        
    def _init_database(self) -> None:
        """Initialize SQLite database for historical data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp DATETIME PRIMARY KEY,
                cpu_usage REAL,
                memory_usage REAL,
                active_strategies INTEGER,
                open_positions INTEGER,
                error_count INTEGER,
                avg_latency REAL
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp DATETIME PRIMARY KEY,
                total_pnl REAL,
                daily_pnl REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_trades INTEGER
            )
        """)
        
        # Trade history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                strategy_id TEXT,
                instrument TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                pnl REAL,
                commission REAL
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                level TEXT,
                category TEXT,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        conn.commit()
        conn.close()
        
    def log_system_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        active_strategies: int,
        open_positions: int,
        error_count: int = 0,
        avg_latency: float = 0.0,
    ) -> None:
        """Log system performance metrics."""
        timestamp = datetime.utcnow()
        
        # Store in memory
        self.metrics["system"]["cpu"].append((timestamp, cpu_usage))
        self.metrics["system"]["memory"].append((timestamp, memory_usage))
        self.metrics["system"]["strategies"].append((timestamp, active_strategies))
        self.metrics["system"]["positions"].append((timestamp, open_positions))
        
        # Update current state
        self.current_state.update({
            "active_strategies": active_strategies,
            "open_positions": open_positions,
            "last_update": timestamp,
        })
        
        # Check thresholds
        if cpu_usage > self.alert_thresholds["cpu_usage_percent"]:
            self._create_alert("warning", "system", f"High CPU usage: {cpu_usage:.1f}%")
            
        if memory_usage > self.alert_thresholds["memory_usage_mb"]:
            self._create_alert("warning", "system", f"High memory usage: {memory_usage:.0f} MB")
            
        # Store in database
        self._store_system_metrics(
            timestamp, cpu_usage, memory_usage, active_strategies,
            open_positions, error_count, avg_latency
        )
        
    def log_performance_metrics(
        self,
        total_pnl: float,
        daily_pnl: float,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float,
        total_trades: int,
    ) -> None:
        """Log trading performance metrics."""
        timestamp = datetime.utcnow()
        
        # Store in memory
        self.metrics["performance"]["pnl"].append((timestamp, total_pnl))
        self.metrics["performance"]["daily_pnl"].append((timestamp, daily_pnl))
        self.metrics["performance"]["win_rate"].append((timestamp, win_rate))
        self.metrics["performance"]["sharpe"].append((timestamp, sharpe_ratio))
        
        # Update current state
        self.current_state.update({
            "total_pnl": total_pnl,
            "daily_pnl": daily_pnl,
        })
        
        # Check thresholds
        if daily_pnl < -self.alert_thresholds["daily_loss"]:
            self._create_alert("critical", "performance", 
                             f"Daily loss exceeds threshold: {daily_pnl:.4f} BTC")
            
        if max_drawdown > self.alert_thresholds["max_drawdown"]:
            self._create_alert("critical", "performance",
                             f"Max drawdown exceeds threshold: {max_drawdown:.2%}")
            
        if win_rate < self.alert_thresholds["min_win_rate"]:
            self._create_alert("warning", "performance",
                             f"Low win rate: {win_rate:.2%}")
            
        # Store in database
        self._store_performance_metrics(
            timestamp, total_pnl, daily_pnl, win_rate,
            sharpe_ratio, max_drawdown, total_trades
        )
        
    def log_trade(
        self,
        strategy_id: str,
        instrument: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0.0,
        commission: float = 0.0,
    ) -> None:
        """Log individual trade."""
        timestamp = datetime.utcnow()
        
        trade = {
            "timestamp": timestamp,
            "strategy_id": strategy_id,
            "instrument": instrument,
            "side": side,
            "quantity": quantity,
            "price": price,
            "pnl": pnl,
            "commission": commission,
        }
        
        # Store in memory
        self.metrics["trades"].append(trade)
        
        # Store in database
        self._store_trade(trade)
        
        # Log
        self.logger.info(
            f"Trade executed: {side} {quantity} {instrument} @ {price} "
            f"(PnL: {pnl:.4f}, Strategy: {strategy_id})"
        )
        
    def log_error(self, category: str, message: str, exception: Optional[Exception] = None) -> None:
        """Log error with optional exception details."""
        timestamp = datetime.utcnow()
        
        # Create alert
        self._create_alert("error", category, message)
        
        # Log with exception details
        if exception:
            self.logger.error(f"{category}: {message}", exc_info=exception)
        else:
            self.logger.error(f"{category}: {message}")
            
    def _create_alert(self, level: str, category: str, message: str) -> None:
        """Create and store alert."""
        alert = {
            "timestamp": datetime.utcnow(),
            "level": level,
            "category": category,
            "message": message,
            "resolved": False,
        }
        
        # Store in memory
        self.metrics["alerts"].append(alert)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO alerts (timestamp, level, category, message) VALUES (?, ?, ?, ?)",
            (alert["timestamp"], level, category, message)
        )
        conn.commit()
        conn.close()
        
        # Log
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(f"Alert: [{category}] {message}")
        
        # Notify websocket clients
        if self.web_enabled:
            asyncio.create_task(self._broadcast_alert(alert))
            
    def _store_system_metrics(self, timestamp, cpu, memory, strategies, positions, errors, latency) -> None:
        """Store system metrics in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO system_metrics 
            (timestamp, cpu_usage, memory_usage, active_strategies, open_positions, error_count, avg_latency)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, cpu, memory, strategies, positions, errors, latency))
        conn.commit()
        conn.close()
        
    def _store_performance_metrics(self, timestamp, total_pnl, daily_pnl, win_rate, sharpe, drawdown, trades) -> None:
        """Store performance metrics in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO performance_metrics 
            (timestamp, total_pnl, daily_pnl, win_rate, sharpe_ratio, max_drawdown, total_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, total_pnl, daily_pnl, win_rate, sharpe, drawdown, trades))
        conn.commit()
        conn.close()
        
    def _store_trade(self, trade: Dict[str, Any]) -> None:
        """Store trade in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades 
            (timestamp, strategy_id, instrument, side, quantity, price, pnl, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["timestamp"], trade["strategy_id"], trade["instrument"],
            trade["side"], trade["quantity"], trade["price"],
            trade["pnl"], trade["commission"]
        ))
        conn.commit()
        conn.close()
        
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get performance metrics
        cursor.execute("""
            SELECT 
                AVG(total_pnl) as avg_pnl,
                MAX(total_pnl) as max_pnl,
                MIN(total_pnl) as min_pnl,
                AVG(win_rate) as avg_win_rate,
                AVG(sharpe_ratio) as avg_sharpe,
                MAX(max_drawdown) as max_drawdown,
                SUM(total_trades) as total_trades
            FROM performance_metrics
            WHERE timestamp > ?
        """, (since,))
        
        perf_data = cursor.fetchone()
        
        # Get trade statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as trade_count,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(pnl) as avg_pnl_per_trade,
                SUM(pnl) as total_trade_pnl,
                SUM(commission) as total_commission
            FROM trades
            WHERE timestamp > ?
        """, (since,))
        
        trade_data = cursor.fetchone()
        
        conn.close()
        
        return {
            "period_hours": hours,
            "performance": {
                "avg_pnl": perf_data[0] or 0,
                "max_pnl": perf_data[1] or 0,
                "min_pnl": perf_data[2] or 0,
                "avg_win_rate": perf_data[3] or 0,
                "avg_sharpe": perf_data[4] or 0,
                "max_drawdown": perf_data[5] or 0,
            },
            "trades": {
                "total": trade_data[0] or 0,
                "winning": trade_data[1] or 0,
                "losing": trade_data[2] or 0,
                "avg_pnl": trade_data[3] or 0,
                "total_pnl": trade_data[4] or 0,
                "total_commission": trade_data[5] or 0,
            }
        }
        
    def _create_web_app(self) -> FastAPI:
        """Create FastAPI web application for monitoring dashboard."""
        app = FastAPI(title="Autonomous Trading Monitor")
        
        @app.get("/")
        async def dashboard():
            """Serve monitoring dashboard."""
            return HTMLResponse(content=self._get_dashboard_html())
            
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.websocket_clients.append(websocket)
            
            try:
                # Send initial state
                await websocket.send_json({
                    "type": "state",
                    "data": self.current_state,
                })
                
                # Keep connection alive
                while True:
                    await asyncio.sleep(1)
                    
            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)
                
        @app.get("/api/metrics/{category}")
        async def get_metrics(category: str, limit: int = 100):
            """Get metrics for specific category."""
            if category in self.metrics:
                data = list(self.metrics[category])[-limit:]
                return {"category": category, "data": data}
            return {"error": "Category not found"}
            
        @app.get("/api/summary/{hours}")
        async def get_summary(hours: int):
            """Get performance summary."""
            return self.get_performance_summary(hours)
            
        @app.get("/api/alerts")
        async def get_alerts(limit: int = 50):
            """Get recent alerts."""
            return {"alerts": list(self.metrics["alerts"])[-limit:]}
            
        return app
        
    async def _broadcast_alert(self, alert: Dict[str, Any]) -> None:
        """Broadcast alert to all connected websocket clients."""
        message = {
            "type": "alert",
            "data": alert,
        }
        
        for client in self.websocket_clients[:]:
            try:
                await client.send_json(message)
            except:
                self.websocket_clients.remove(client)
                
    def _get_dashboard_html(self) -> str:
        """Get HTML for monitoring dashboard."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Autonomous Trading Monitor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .status-card h3 {
            margin: 0 0 10px 0;
            color: #4a9eff;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        .metric-value {
            font-weight: bold;
        }
        .positive { color: #4ade80; }
        .negative { color: #f87171; }
        .neutral { color: #e0e0e0; }
        .alerts {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .alert-error { background: #7f1d1d; }
        .alert-warning { background: #78350f; }
        .alert-info { background: #1e3a8a; }
        #chart-container {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            height: 400px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Autonomous Trading System Monitor</h1>
            <p id="last-update">Last Update: -</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>System Health</h3>
                <div class="metric">
                    <span>Status:</span>
                    <span class="metric-value" id="system-health">-</span>
                </div>
                <div class="metric">
                    <span>Uptime:</span>
                    <span class="metric-value" id="uptime">-</span>
                </div>
                <div class="metric">
                    <span>Active Strategies:</span>
                    <span class="metric-value" id="active-strategies">-</span>
                </div>
                <div class="metric">
                    <span>Open Positions:</span>
                    <span class="metric-value" id="open-positions">-</span>
                </div>
            </div>
            
            <div class="status-card">
                <h3>Performance</h3>
                <div class="metric">
                    <span>Total P&L:</span>
                    <span class="metric-value" id="total-pnl">-</span>
                </div>
                <div class="metric">
                    <span>Daily P&L:</span>
                    <span class="metric-value" id="daily-pnl">-</span>
                </div>
                <div class="metric">
                    <span>Win Rate:</span>
                    <span class="metric-value" id="win-rate">-</span>
                </div>
                <div class="metric">
                    <span>Sharpe Ratio:</span>
                    <span class="metric-value" id="sharpe-ratio">-</span>
                </div>
            </div>
            
            <div class="status-card">
                <h3>Resources</h3>
                <div class="metric">
                    <span>CPU Usage:</span>
                    <span class="metric-value" id="cpu-usage">-</span>
                </div>
                <div class="metric">
                    <span>Memory Usage:</span>
                    <span class="metric-value" id="memory-usage">-</span>
                </div>
                <div class="metric">
                    <span>Disk Usage:</span>
                    <span class="metric-value" id="disk-usage">-</span>
                </div>
                <div class="metric">
                    <span>Network Latency:</span>
                    <span class="metric-value" id="latency">-</span>
                </div>
            </div>
        </div>
        
        <div class="alerts">
            <h3>Recent Alerts</h3>
            <div id="alerts-container"></div>
        </div>
        
        <div id="chart-container">
            <canvas id="pnl-chart"></canvas>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        // Chart setup
        const ctx = document.getElementById('pnl-chart').getContext('2d');
        const pnlChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total P&L',
                    data: [],
                    borderColor: '#4a9eff',
                    backgroundColor: 'rgba(74, 158, 255, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#e0e0e0' },
                        grid: { color: '#444' }
                    },
                    y: {
                        ticks: { color: '#e0e0e0' },
                        grid: { color: '#444' }
                    }
                }
            }
        });
        
        // Update functions
        function updateState(data) {
            document.getElementById('last-update').textContent = 
                `Last Update: ${new Date().toLocaleString()}`;
            
            // System health
            document.getElementById('system-health').textContent = data.system_health;
            document.getElementById('uptime').textContent = data.uptime || '-';
            document.getElementById('active-strategies').textContent = data.active_strategies;
            document.getElementById('open-positions').textContent = data.open_positions;
            
            // Performance
            const totalPnl = document.getElementById('total-pnl');
            totalPnl.textContent = `${data.total_pnl.toFixed(4)} BTC`;
            totalPnl.className = data.total_pnl >= 0 ? 'metric-value positive' : 'metric-value negative';
            
            const dailyPnl = document.getElementById('daily-pnl');
            dailyPnl.textContent = `${data.daily_pnl.toFixed(4)} BTC`;
            dailyPnl.className = data.daily_pnl >= 0 ? 'metric-value positive' : 'metric-value negative';
        }
        
        function addAlert(alert) {
            const container = document.getElementById('alerts-container');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${alert.level}`;
            alertDiv.innerHTML = `
                <strong>${alert.timestamp}:</strong> [${alert.category}] ${alert.message}
            `;
            container.insertBefore(alertDiv, container.firstChild);
            
            // Keep only last 10 alerts
            while (container.children.length > 10) {
                container.removeChild(container.lastChild);
            }
        }
        
        // WebSocket handlers
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'state') {
                updateState(message.data);
            } else if (message.type === 'alert') {
                addAlert(message.data);
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
        
        ws.onclose = function() {
            console.log('WebSocket connection closed');
            // Attempt reconnection after 5 seconds
            setTimeout(() => window.location.reload(), 5000);
        };
        
        // Fetch initial data
        fetch('/api/summary/24')
            .then(response => response.json())
            .then(data => {
                console.log('Summary data:', data);
            });
    </script>
</body>
</html>
        """
        
    async def start_web_server(self) -> None:
        """Start the web monitoring server."""
        if self.web_enabled:
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=self.web_port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            await server.serve()
            
    def cleanup(self) -> None:
        """Cleanup resources."""
        # Close database connections
        # Save any pending metrics
        self.logger.info("Monitor cleanup completed")