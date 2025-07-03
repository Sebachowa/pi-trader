
"""
Monitoring server integration for Nautilus Trader.

This module provides Flask server setup and WebSocket support for real-time updates.
"""

import asyncio
import json
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from nautilus_trader.common.clock import LiveClock
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.portfolio.portfolio import Portfolio


class MonitoringServer:
    """
    WebSocket-enabled monitoring server for real-time updates.
    """
    
    def __init__(
        self,
        app: Flask,
        host: str = "127.0.0.1",
        port: int = 5000,
        cors_origins: str = "*",
    ):
        """
        Initialize the monitoring server.
        
        Parameters
        ----------
        app : Flask
            The Flask application instance.
        host : str
            The host to bind to.
        port : int
            The port to bind to.
        cors_origins : str
            CORS origins configuration.
            
        """
        self._app = app
        self._host = host
        self._port = port
        
        # Enable CORS
        CORS(app, origins=cors_origins)
        
        # Initialize SocketIO
        self._socketio = SocketIO(app, cors_allowed_origins=cors_origins)
        
        # Connected clients
        self._clients = set()
        
        # Setup WebSocket handlers
        self._setup_websocket_handlers()
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers."""
        
        @self._socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            client_id = request.sid
            self._clients.add(client_id)
            emit('connected', {'client_id': client_id})
            print(f"Client connected: {client_id}")
        
        @self._socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            self._clients.discard(client_id)
            print(f"Client disconnected: {client_id}")
        
        @self._socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription requests."""
            channel = data.get('channel')
            client_id = request.sid
            
            # Join the channel room
            if channel:
                self._socketio.server.enter_room(client_id, channel)
                emit('subscribed', {'channel': channel})
                print(f"Client {client_id} subscribed to {channel}")
        
        @self._socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Handle unsubscription requests."""
            channel = data.get('channel')
            client_id = request.sid
            
            # Leave the channel room
            if channel:
                self._socketio.server.leave_room(client_id, channel)
                emit('unsubscribed', {'channel': channel})
                print(f"Client {client_id} unsubscribed from {channel}")
    
    def broadcast_update(self, channel: str, data: Dict[str, Any]):
        """
        Broadcast update to all clients in a channel.
        
        Parameters
        ----------
        channel : str
            The channel to broadcast to.
        data : dict
            The data to broadcast.
            
        """
        self._socketio.emit(channel, data, room=channel)
    
    def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast metrics update."""
        self.broadcast_update('metrics', metrics)
    
    def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert."""
        self.broadcast_update('alerts', alert)
    
    def broadcast_positions(self, positions: Dict[str, Any]):
        """Broadcast positions update."""
        self.broadcast_update('positions', positions)
    
    def run(self):
        """Run the monitoring server."""
        self._socketio.run(
            self._app,
            host=self._host,
            port=self._port,
            debug=False,
            use_reloader=False,
        )


class AlertManager:
    """
    Manages alerts and notifications for the monitoring system.
    """
    
    def __init__(self, max_alerts: int = 1000):
        """
        Initialize the alert manager.
        
        Parameters
        ----------
        max_alerts : int
            Maximum number of alerts to keep in history.
            
        """
        self._max_alerts = max_alerts
        self._alerts = []
        self._alert_callbacks = {}
        self._alert_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "WARNING": 0,
            "INFO": 0,
            "SUCCESS": 0,
        }
    
    def add_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a new alert.
        
        Parameters
        ----------
        alert_type : str
            The type of alert (RISK, PERFORMANCE, SYSTEM, etc.).
        severity : str
            The severity level (CRITICAL, HIGH, WARNING, INFO, SUCCESS).
        message : str
            The alert message.
        metadata : dict, optional
            Additional metadata for the alert.
            
        Returns
        -------
        dict
            The created alert.
            
        """
        alert = {
            "id": str(UUID4()),
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
            "acknowledged": False,
        }
        
        # Add to history
        self._alerts.append(alert)
        
        # Trim history if needed
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]
        
        # Update counts
        if severity in self._alert_counts:
            self._alert_counts[severity] += 1
        
        # Trigger callbacks
        self._trigger_callbacks(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Parameters
        ----------
        alert_id : str
            The alert ID to acknowledge.
            
        Returns
        -------
        bool
            True if alert was found and acknowledged.
            
        """
        for alert in self._alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                return True
        return False
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        """
        Get alerts with optional filtering.
        
        Parameters
        ----------
        severity : str, optional
            Filter by severity.
        alert_type : str, optional
            Filter by alert type.
        unacknowledged_only : bool
            Only return unacknowledged alerts.
        limit : int
            Maximum number of alerts to return.
            
        Returns
        -------
        list[dict]
            The filtered alerts.
            
        """
        filtered = self._alerts
        
        if severity:
            filtered = [a for a in filtered if a["severity"] == severity]
        
        if alert_type:
            filtered = [a for a in filtered if a["type"] == alert_type]
        
        if unacknowledged_only:
            filtered = [a for a in filtered if not a["acknowledged"]]
        
        return filtered[-limit:]
    
    def get_alert_counts(self) -> Dict[str, int]:
        """Get alert counts by severity."""
        return self._alert_counts.copy()
    
    def register_callback(self, severity: str, callback):
        """
        Register a callback for alerts of a specific severity.
        
        Parameters
        ----------
        severity : str
            The severity level to trigger on.
        callback : callable
            The callback function to call.
            
        """
        if severity not in self._alert_callbacks:
            self._alert_callbacks[severity] = []
        self._alert_callbacks[severity].append(callback)
    
    def _trigger_callbacks(self, alert: Dict[str, Any]):
        """Trigger registered callbacks for an alert."""
        severity = alert["severity"]
        if severity in self._alert_callbacks:
            for callback in self._alert_callbacks[severity]:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Error in alert callback: {e}")


class PerformanceAnalyzer:
    """
    Analyzes trading performance and generates insights.
    """
    
    def __init__(self, target_annual_return: float = 0.10):
        """
        Initialize the performance analyzer.
        
        Parameters
        ----------
        target_annual_return : float
            The target annual return (default 10%).
            
        """
        self._target_annual_return = target_annual_return
        self._performance_history = []
    
    def analyze_performance(
        self,
        current_balance: float,
        starting_balance: float,
        days_elapsed: float,
        positions: list,
        trades_history: list = None,
    ) -> Dict[str, Any]:
        """
        Analyze current performance and generate insights.
        
        Parameters
        ----------
        current_balance : float
            Current account balance.
        starting_balance : float
            Starting account balance.
        days_elapsed : float
            Number of days elapsed.
        positions : list
            Current positions.
        trades_history : list, optional
            Historical trades.
            
        Returns
        -------
        dict
            Performance analysis results.
            
        """
        # Calculate returns
        total_return = (current_balance - starting_balance) / starting_balance
        
        # Annualized return
        if days_elapsed > 0:
            annualized_return = total_return * (365 / days_elapsed)
        else:
            annualized_return = 0.0
        
        # Progress toward goal
        progress_pct = (annualized_return / self._target_annual_return * 100) if self._target_annual_return > 0 else 0
        
        # Required daily return to meet goal
        days_remaining = max(0, 30 - days_elapsed)  # Assuming 30-day challenge
        if days_remaining > 0 and total_return < self._target_annual_return:
            remaining_return_needed = self._target_annual_return - total_return
            required_daily_return = remaining_return_needed / days_remaining
        else:
            required_daily_return = 0.0
        
        # Position analysis
        position_concentration = self._analyze_position_concentration(positions, current_balance)
        
        # Generate insights
        insights = self._generate_insights(
            annualized_return,
            progress_pct,
            required_daily_return,
            position_concentration,
            days_elapsed,
            days_remaining,
        )
        
        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "target_return_pct": self._target_annual_return * 100,
            "progress_pct": progress_pct,
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "required_daily_return_pct": required_daily_return * 100,
            "on_track": annualized_return >= self._target_annual_return,
            "position_concentration": position_concentration,
            "insights": insights,
        }
    
    def _analyze_position_concentration(
        self,
        positions: list,
        balance: float,
    ) -> Dict[str, Any]:
        """Analyze position concentration risk."""
        if not positions or balance <= 0:
            return {
                "largest_position_pct": 0.0,
                "concentration_score": 0,
                "diversification_rating": "N/A",
            }
        
        # Calculate position sizes
        position_values = [abs(float(p.notional_value)) for p in positions]
        total_exposure = sum(position_values)
        
        # Largest position
        largest_position = max(position_values) if position_values else 0
        largest_position_pct = (largest_position / balance * 100) if balance > 0 else 0
        
        # Concentration score (0-100, higher is more concentrated)
        concentration_score = min(100, largest_position_pct * 2)
        
        # Diversification rating
        if len(positions) >= 5 and largest_position_pct < 20:
            diversification_rating = "GOOD"
        elif len(positions) >= 3 and largest_position_pct < 30:
            diversification_rating = "MODERATE"
        else:
            diversification_rating = "POOR"
        
        return {
            "largest_position_pct": largest_position_pct,
            "concentration_score": concentration_score,
            "diversification_rating": diversification_rating,
            "position_count": len(positions),
            "total_exposure": total_exposure,
            "exposure_pct": (total_exposure / balance * 100) if balance > 0 else 0,
        }
    
    def _generate_insights(
        self,
        annualized_return: float,
        progress_pct: float,
        required_daily_return: float,
        position_concentration: Dict[str, Any],
        days_elapsed: float,
        days_remaining: float,
    ) -> list[str]:
        """Generate performance insights."""
        insights = []
        
        # Progress insights
        if progress_pct >= 100:
            insights.append("Congratulations! You've achieved your target annual return.")
        elif progress_pct >= 75:
            insights.append("Excellent progress! You're on track to meet your target.")
        elif progress_pct >= 50:
            insights.append("Good progress. Maintain consistency to reach your target.")
        elif progress_pct >= 25:
            insights.append("Making progress but need to increase returns to meet target.")
        else:
            insights.append("Behind target. Consider adjusting strategy to improve returns.")
        
        # Required return insights
        if required_daily_return > 0.02:  # > 2% daily
            insights.append(f"Warning: Required {required_daily_return:.1%} daily return is very aggressive.")
        elif required_daily_return > 0.01:  # > 1% daily
            insights.append(f"Need {required_daily_return:.1%} daily return to meet target - challenging but achievable.")
        
        # Position concentration insights
        if position_concentration["concentration_score"] > 70:
            insights.append("High position concentration detected. Consider diversifying.")
        elif position_concentration["diversification_rating"] == "GOOD":
            insights.append("Good portfolio diversification maintained.")
        
        # Time-based insights
        if days_remaining < 7 and progress_pct < 50:
            insights.append("Limited time remaining. May need to take calculated risks.")
        elif days_elapsed < 3:
            insights.append("Early in the challenge. Focus on establishing consistent strategy.")
        
        return insights


# Helper function to setup monitoring with all components
def setup_complete_monitoring(
    app: Flask,
    portfolio: Portfolio,
    clock: LiveClock,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> tuple[MonitoringServer, AlertManager, PerformanceAnalyzer]:
    """
    Setup complete monitoring with server, alerts, and analytics.
    
    Parameters
    ----------
    app : Flask
        The Flask application.
    portfolio : Portfolio
        The portfolio to monitor.
    clock : LiveClock
        The system clock.
    host : str
        The host to bind to.
    port : int
        The port to bind to.
        
    Returns
    -------
    tuple
        (monitoring_server, alert_manager, performance_analyzer)
        
    """
    # Create components
    monitoring_server = MonitoringServer(app, host, port)
    alert_manager = AlertManager()
    performance_analyzer = PerformanceAnalyzer(target_annual_return=0.10)
    
    # Setup alert callbacks to broadcast via WebSocket
    def broadcast_alert(alert):
        monitoring_server.broadcast_alert(alert)
    
    # Register callbacks for critical alerts
    alert_manager.register_callback("CRITICAL", broadcast_alert)
    alert_manager.register_callback("HIGH", broadcast_alert)
    
    return monitoring_server, alert_manager, performance_analyzer