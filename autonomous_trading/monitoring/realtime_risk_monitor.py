
"""
Real-time Risk Monitoring System

Continuous monitoring of portfolio risk with alerts, notifications, and dashboards.
"""

import asyncio
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass
import json
import websockets
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
# from nautilus_trader.common.logging import Logger  # Not available in this version


@dataclass
class RiskAlert:
    """Risk alert with severity and actions."""
    id: str
    timestamp: datetime
    severity: str  # info, warning, critical, emergency
    category: str  # drawdown, correlation, exposure, volatility, etc.
    message: str
    value: float
    threshold: float
    recommended_action: str
    auto_action_taken: bool = False
    acknowledged: bool = False


@dataclass
class RiskMetricThreshold:
    """Configurable threshold for risk metrics."""
    metric_name: str
    warning_level: float
    critical_level: float
    emergency_level: float
    check_interval: int  # seconds
    enabled: bool = True


class RealtimeRiskMonitor(Component):
    """
    Real-time risk monitoring with alerts and dashboards.
    
    Features:
    - Continuous risk metric monitoring
    - Multi-level alerts (info, warning, critical, emergency)
    - WebSocket real-time updates
    - Prometheus metrics export
    - Risk dashboard generation
    - Alert history and analytics
    """
    
    def __init__(
        self,
        logger: Any,  # Logger type
        clock: LiveClock,
        msgbus: MessageBus,
        risk_manager,  # ComprehensiveRiskManager instance
        websocket_port: int = 8765,
        prometheus_port: int = 9090,
        update_interval: float = 1.0,  # seconds
        alert_cooldown: int = 300,  # seconds
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id="REALTIME-RISK-MONITOR",
            msgbus=msgbus,
        )
        
        self.risk_manager = risk_manager
        self.websocket_port = websocket_port
        self.prometheus_port = prometheus_port
        self.update_interval = update_interval
        self.alert_cooldown = alert_cooldown
        
        # Monitoring state
        self._monitoring_active = False
        self._websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        self._monitoring_task = None
        self._websocket_task = None
        
        # Risk metrics tracking
        self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3600))
        self._alert_history: deque = deque(maxlen=1000)
        self._active_alerts: Dict[str, RiskAlert] = {}
        self._alert_cooldowns: Dict[str, datetime] = {}
        
        # Thresholds configuration
        self._thresholds = self._initialize_thresholds()
        
        # Metric aggregations
        self._minute_aggregations = defaultdict(list)
        self._hourly_aggregations = defaultdict(list)
        self._daily_aggregations = defaultdict(list)
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Alert callbacks
        self._alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
    def _initialize_thresholds(self) -> Dict[str, RiskMetricThreshold]:
        """Initialize risk metric thresholds."""
        return {
            "drawdown": RiskMetricThreshold(
                metric_name="drawdown",
                warning_level=0.03,  # 3%
                critical_level=0.05,  # 5%
                emergency_level=0.08,  # 8%
                check_interval=5,
            ),
            "daily_loss": RiskMetricThreshold(
                metric_name="daily_loss",
                warning_level=0.01,  # 1%
                critical_level=0.015,  # 1.5%
                emergency_level=0.02,  # 2%
                check_interval=10,
            ),
            "portfolio_var": RiskMetricThreshold(
                metric_name="portfolio_var",
                warning_level=-0.03,  # -3%
                critical_level=-0.05,  # -5%
                emergency_level=-0.08,  # -8%
                check_interval=30,
            ),
            "correlation_risk": RiskMetricThreshold(
                metric_name="correlation_risk",
                warning_level=0.6,
                critical_level=0.7,
                emergency_level=0.8,
                check_interval=60,
            ),
            "concentration_risk": RiskMetricThreshold(
                metric_name="concentration_risk",
                warning_level=0.25,
                critical_level=0.35,
                emergency_level=0.45,
                check_interval=60,
            ),
            "exposure_ratio": RiskMetricThreshold(
                metric_name="exposure_ratio",
                warning_level=0.8,
                critical_level=1.0,
                emergency_level=1.2,
                check_interval=10,
            ),
            "volatility_spike": RiskMetricThreshold(
                metric_name="volatility_spike",
                warning_level=2.0,  # 2x normal
                critical_level=3.0,  # 3x normal
                emergency_level=4.0,  # 4x normal
                check_interval=5,
            ),
            "sharpe_deterioration": RiskMetricThreshold(
                metric_name="sharpe_deterioration",
                warning_level=0.5,
                critical_level=0.3,
                emergency_level=0.0,
                check_interval=300,
            ),
        }
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics for monitoring."""
        # Gauges for current values
        self.prom_drawdown = Gauge('risk_drawdown', 'Current portfolio drawdown')
        self.prom_daily_loss = Gauge('risk_daily_loss', 'Current daily loss')
        self.prom_portfolio_var = Gauge('risk_portfolio_var', 'Portfolio Value at Risk')
        self.prom_correlation = Gauge('risk_correlation', 'Portfolio correlation risk')
        self.prom_exposure = Gauge('risk_exposure', 'Total portfolio exposure')
        self.prom_position_count = Gauge('risk_position_count', 'Number of open positions')
        
        # Counters for events
        self.prom_alerts_total = Counter(
            'risk_alerts_total', 
            'Total risk alerts generated',
            ['severity', 'category']
        )
        self.prom_emergency_actions = Counter(
            'risk_emergency_actions_total',
            'Total emergency actions taken',
            ['action_type']
        )
        
        # Histograms for distributions
        self.prom_position_pnl = Histogram(
            'risk_position_pnl',
            'Position P&L distribution'
        )
        self.prom_risk_score = Histogram(
            'risk_score',
            'Overall risk score distribution'
        )
    
    async def start_monitoring(self) -> None:
        """Start real-time risk monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start Prometheus metrics server
        start_http_server(self.prometheus_port)
        self._log.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        
        # Start monitoring loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start WebSocket server
        self._websocket_task = asyncio.create_task(self._start_websocket_server())
        
        self._log.info("Real-time risk monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        if self._websocket_task:
            self._websocket_task.cancel()
        
        # Close WebSocket connections
        for client in self._websocket_clients:
            await client.close()
        
        self._log.info("Real-time risk monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        last_check_times = {metric: datetime.utcnow() for metric in self._thresholds}
        
        while self._monitoring_active:
            try:
                # Get current risk metrics
                risk_metrics = await self.risk_manager.monitor_portfolio_risk()
                
                # Update metric history
                self._update_metric_history(risk_metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(risk_metrics)
                
                # Check thresholds
                current_time = datetime.utcnow()
                for metric_name, threshold in self._thresholds.items():
                    if not threshold.enabled:
                        continue
                    
                    # Check if it's time to check this metric
                    time_since_last = (current_time - last_check_times[metric_name]).seconds
                    if time_since_last >= threshold.check_interval:
                        await self._check_threshold(metric_name, threshold, risk_metrics)
                        last_check_times[metric_name] = current_time
                
                # Generate and broadcast updates
                update_data = self._generate_update_data(risk_metrics)
                await self._broadcast_update(update_data)
                
                # Update aggregations
                self._update_aggregations(risk_metrics)
                
                # Sleep until next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self._log.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _update_metric_history(self, risk_metrics: Dict[str, Any]) -> None:
        """Update metric history for tracking."""
        timestamp = datetime.utcnow()
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
        
        # Store key metrics
        self._metric_history["drawdown"].append(
            (timestamp, portfolio_metrics.get("current_drawdown", 0))
        )
        self._metric_history["daily_pnl"].append(
            (timestamp, self.risk_manager._daily_pnl)
        )
        self._metric_history["portfolio_var"].append(
            (timestamp, portfolio_metrics.get("portfolio_var", 0))
        )
        self._metric_history["correlation_risk"].append(
            (timestamp, portfolio_metrics.get("correlation_risk", 0))
        )
        self._metric_history["total_exposure"].append(
            (timestamp, portfolio_metrics.get("total_exposure", 0))
        )
        self._metric_history["sharpe_ratio"].append(
            (timestamp, portfolio_metrics.get("sharpe_ratio", 0))
        )
    
    def _update_prometheus_metrics(self, risk_metrics: Dict[str, Any]) -> None:
        """Update Prometheus metrics."""
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
        
        # Update gauges
        self.prom_drawdown.set(portfolio_metrics.get("current_drawdown", 0))
        self.prom_daily_loss.set(abs(self.risk_manager._daily_pnl))
        self.prom_portfolio_var.set(portfolio_metrics.get("portfolio_var", 0))
        self.prom_correlation.set(portfolio_metrics.get("correlation_risk", 0))
        self.prom_exposure.set(portfolio_metrics.get("total_exposure", 0))
        self.prom_position_count.set(portfolio_metrics.get("effective_positions", 0))
        
        # Calculate and update risk score
        risk_score = self._calculate_risk_score(risk_metrics)
        self.prom_risk_score.observe(risk_score)
    
    async def _check_threshold(
        self,
        metric_name: str,
        threshold: RiskMetricThreshold,
        risk_metrics: Dict[str, Any],
    ) -> None:
        """Check if a metric exceeds its threshold."""
        # Get current value
        current_value = self._get_metric_value(metric_name, risk_metrics)
        if current_value is None:
            return
        
        # Check cooldown
        if metric_name in self._alert_cooldowns:
            if datetime.utcnow() < self._alert_cooldowns[metric_name]:
                return
        
        # Determine severity
        severity = None
        if metric_name in ["portfolio_var", "daily_loss"]:  # Negative metrics
            if current_value <= threshold.emergency_level:
                severity = "emergency"
            elif current_value <= threshold.critical_level:
                severity = "critical"
            elif current_value <= threshold.warning_level:
                severity = "warning"
        else:  # Positive metrics
            if current_value >= threshold.emergency_level:
                severity = "emergency"
            elif current_value >= threshold.critical_level:
                severity = "critical"
            elif current_value >= threshold.warning_level:
                severity = "warning"
        
        if severity:
            await self._create_alert(
                metric_name,
                severity,
                current_value,
                threshold,
                risk_metrics
            )
    
    def _get_metric_value(
        self,
        metric_name: str,
        risk_metrics: Dict[str, Any],
    ) -> Optional[float]:
        """Extract metric value from risk metrics."""
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
        
        if metric_name == "drawdown":
            return portfolio_metrics.get("current_drawdown", 0)
        elif metric_name == "daily_loss":
            return abs(self.risk_manager._daily_pnl)
        elif metric_name == "portfolio_var":
            return portfolio_metrics.get("portfolio_var", 0)
        elif metric_name == "correlation_risk":
            return portfolio_metrics.get("correlation_risk", 0)
        elif metric_name == "concentration_risk":
            return portfolio_metrics.get("concentration_risk", 0)
        elif metric_name == "exposure_ratio":
            account_balance = float(self.risk_manager.portfolio.account_balance_total())
            if account_balance > 0:
                return portfolio_metrics.get("total_exposure", 0) / account_balance
            return 0
        elif metric_name == "volatility_spike":
            # Check for volatility spike across positions
            if self.risk_manager._market_conditions:
                extreme_count = sum(
                    1 for c in self.risk_manager._market_conditions.values()
                    if c.volatility_regime == "extreme"
                )
                return extreme_count / len(self.risk_manager._market_conditions)
            return 0
        elif metric_name == "sharpe_deterioration":
            return portfolio_metrics.get("sharpe_ratio", 1.0)
        
        return None
    
    async def _create_alert(
        self,
        metric_name: str,
        severity: str,
        current_value: float,
        threshold: RiskMetricThreshold,
        risk_metrics: Dict[str, Any],
    ) -> None:
        """Create and process a risk alert."""
        alert_id = f"{metric_name}_{datetime.utcnow().timestamp()}"
        
        # Determine recommended action
        recommended_action = self._get_recommended_action(
            metric_name, severity, current_value
        )
        
        alert = RiskAlert(
            id=alert_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            category=metric_name,
            message=f"{metric_name} threshold exceeded: {current_value:.4f}",
            value=current_value,
            threshold=getattr(threshold, f"{severity}_level"),
            recommended_action=recommended_action,
        )
        
        # Store alert
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        # Update Prometheus counter
        self.prom_alerts_total.labels(severity=severity, category=metric_name).inc()
        
        # Set cooldown
        self._alert_cooldowns[metric_name] = datetime.utcnow() + timedelta(
            seconds=self.alert_cooldown
        )
        
        # Execute callbacks
        await self._execute_alert_callbacks(alert)
        
        # Log alert
        log_method = getattr(self._log, severity if severity != "emergency" else "critical")
        log_method(
            f"Risk Alert [{severity.upper()}] {metric_name}: {current_value:.4f} "
            f"(threshold: {alert.threshold:.4f})"
        )
        
        # Take automatic action if emergency
        if severity == "emergency" and self.risk_manager.enable_ml_predictions:
            await self._take_automatic_action(alert, risk_metrics)
    
    def _get_recommended_action(
        self,
        metric_name: str,
        severity: str,
        current_value: float,
    ) -> str:
        """Get recommended action for an alert."""
        actions = {
            "drawdown": {
                "warning": "Consider reducing position sizes",
                "critical": "Reduce all positions by 50%",
                "emergency": "Close all positions immediately",
            },
            "daily_loss": {
                "warning": "Stop opening new positions",
                "critical": "Close losing positions",
                "emergency": "Emergency stop - close all positions",
            },
            "portfolio_var": {
                "warning": "Review portfolio risk",
                "critical": "Reduce portfolio leverage",
                "emergency": "Implement emergency hedges",
            },
            "correlation_risk": {
                "warning": "Avoid correlated trades",
                "critical": "Close highly correlated positions",
                "emergency": "Reduce to uncorrelated core positions",
            },
            "concentration_risk": {
                "warning": "Diversify portfolio",
                "critical": "Reduce largest positions",
                "emergency": "Force diversification",
            },
            "exposure_ratio": {
                "warning": "Monitor leverage carefully",
                "critical": "Reduce leverage to 1:1",
                "emergency": "Deleverage immediately",
            },
            "volatility_spike": {
                "warning": "Widen stop losses",
                "critical": "Reduce position sizes",
                "emergency": "Move to cash",
            },
            "sharpe_deterioration": {
                "warning": "Review strategy performance",
                "critical": "Pause underperforming strategies",
                "emergency": "Switch to capital preservation mode",
            },
        }
        
        return actions.get(metric_name, {}).get(severity, "Monitor closely")
    
    async def _take_automatic_action(
        self,
        alert: RiskAlert,
        risk_metrics: Dict[str, Any],
    ) -> None:
        """Take automatic action for emergency alerts."""
        self._log.critical(f"Taking automatic action for emergency alert: {alert.category}")
        
        if alert.category == "drawdown":
            await self.risk_manager.close_all_positions("Emergency: Maximum drawdown exceeded")
        elif alert.category == "daily_loss":
            self.risk_manager._emergency_stop_active = True
        elif alert.category == "correlation_risk":
            await self.risk_manager._reduce_correlated_positions()
        elif alert.category == "volatility_spike":
            await self.risk_manager._reduce_all_positions(0.5)
        
        alert.auto_action_taken = True
        
        # Update Prometheus counter
        self.prom_emergency_actions.labels(action_type=alert.category).inc()
    
    def _generate_update_data(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate update data for WebSocket broadcast."""
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
        
        # Calculate additional metrics
        risk_score = self._calculate_risk_score(risk_metrics)
        health_status = self._calculate_health_status(risk_metrics)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "portfolio_metrics": portfolio_metrics,
            "risk_score": risk_score,
            "health_status": health_status,
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity,
                    "category": alert.category,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self._active_alerts.values()
                if not alert.acknowledged
            ],
            "position_count": len(risk_metrics.get("position_metrics", [])),
            "capital_preservation_mode": risk_metrics.get("capital_preservation_active", False),
            "emergency_stop": self.risk_manager._emergency_stop_active,
            "metric_trends": self._calculate_metric_trends(),
        }
    
    def _calculate_risk_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-100)."""
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
        risk_utilization = risk_metrics.get("risk_utilization", {})
        
        # Weight different risk factors
        factors = {
            "drawdown": min(100, portfolio_metrics.get("current_drawdown", 0) * 1000),
            "daily_loss": min(100, abs(self.risk_manager._daily_pnl) * 5000),
            "var": min(100, abs(portfolio_metrics.get("portfolio_var", 0)) * 2000),
            "correlation": min(100, portfolio_metrics.get("correlation_risk", 0) * 100),
            "exposure": min(100, risk_utilization.get("exposure_utilization", 0) * 100),
        }
        
        # Calculate weighted score
        weights = {
            "drawdown": 0.3,
            "daily_loss": 0.2,
            "var": 0.2,
            "correlation": 0.15,
            "exposure": 0.15,
        }
        
        risk_score = sum(
            factors[key] * weights[key] 
            for key in factors
        )
        
        return min(100, risk_score)
    
    def _calculate_health_status(self, risk_metrics: Dict[str, Any]) -> str:
        """Calculate portfolio health status."""
        risk_score = self._calculate_risk_score(risk_metrics)
        
        if risk_score < 20:
            return "excellent"
        elif risk_score < 40:
            return "good"
        elif risk_score < 60:
            return "fair"
        elif risk_score < 80:
            return "poor"
        else:
            return "critical"
    
    def _calculate_metric_trends(self) -> Dict[str, str]:
        """Calculate trend direction for key metrics."""
        trends = {}
        
        for metric_name, history in self._metric_history.items():
            if len(history) < 10:
                trends[metric_name] = "stable"
                continue
            
            # Get recent values
            recent_values = [value for _, value in list(history)[-10:]]
            
            # Calculate trend
            if len(recent_values) >= 2:
                change = recent_values[-1] - recent_values[0]
                avg = np.mean(recent_values)
                
                if avg != 0:
                    change_pct = change / avg
                    
                    if change_pct > 0.1:
                        trends[metric_name] = "increasing"
                    elif change_pct < -0.1:
                        trends[metric_name] = "decreasing"
                    else:
                        trends[metric_name] = "stable"
                else:
                    trends[metric_name] = "stable"
        
        return trends
    
    async def _broadcast_update(self, update_data: Dict[str, Any]) -> None:
        """Broadcast update to all WebSocket clients."""
        if not self._websocket_clients:
            return
        
        message = json.dumps(update_data)
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self._websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self._websocket_clients -= disconnected_clients
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time updates."""
        async def handle_client(websocket, path):
            """Handle WebSocket client connection."""
            self._websocket_clients.add(websocket)
            self._log.info(f"WebSocket client connected from {websocket.remote_address}")
            
            try:
                # Send initial data
                risk_metrics = await self.risk_manager.monitor_portfolio_risk()
                initial_data = self._generate_update_data(risk_metrics)
                await websocket.send(json.dumps(initial_data))
                
                # Keep connection alive
                async for message in websocket:
                    # Handle client messages (e.g., acknowledge alerts)
                    try:
                        data = json.loads(message)
                        await self._handle_client_message(data, websocket)
                    except json.JSONDecodeError:
                        pass
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self._websocket_clients.remove(websocket)
                self._log.info(f"WebSocket client disconnected from {websocket.remote_address}")
        
        # Start server
        await websockets.serve(handle_client, "localhost", self.websocket_port)
        self._log.info(f"WebSocket server started on port {self.websocket_port}")
    
    async def _handle_client_message(
        self,
        data: Dict[str, Any],
        websocket: websockets.WebSocketServerProtocol,
    ) -> None:
        """Handle messages from WebSocket clients."""
        action = data.get("action")
        
        if action == "acknowledge_alert":
            alert_id = data.get("alert_id")
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].acknowledged = True
                await websocket.send(json.dumps({
                    "type": "alert_acknowledged",
                    "alert_id": alert_id,
                }))
        
        elif action == "get_history":
            metric_name = data.get("metric")
            period = data.get("period", 3600)  # Default 1 hour
            
            if metric_name in self._metric_history:
                history = list(self._metric_history[metric_name])[-period:]
                await websocket.send(json.dumps({
                    "type": "metric_history",
                    "metric": metric_name,
                    "data": [
                        {"timestamp": ts.isoformat(), "value": val}
                        for ts, val in history
                    ],
                }))
    
    def _update_aggregations(self, risk_metrics: Dict[str, Any]) -> None:
        """Update metric aggregations for different time periods."""
        current_time = datetime.utcnow()
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
        
        # Minute aggregation
        minute_key = current_time.replace(second=0, microsecond=0)
        self._minute_aggregations[minute_key].append(portfolio_metrics)
        
        # Hourly aggregation
        hour_key = current_time.replace(minute=0, second=0, microsecond=0)
        self._hourly_aggregations[hour_key].append(portfolio_metrics)
        
        # Daily aggregation
        day_key = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        self._daily_aggregations[day_key].append(portfolio_metrics)
    
    def register_alert_callback(
        self,
        severity: str,
        callback: Callable[[RiskAlert], None],
    ) -> None:
        """Register callback for specific alert severity."""
        self._alert_callbacks[severity].append(callback)
    
    async def _execute_alert_callbacks(self, alert: RiskAlert) -> None:
        """Execute registered callbacks for an alert."""
        callbacks = self._alert_callbacks.get(alert.severity, [])
        callbacks.extend(self._alert_callbacks.get("all", []))
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self._log.error(f"Error executing alert callback: {e}")
    
    def get_risk_summary(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get risk summary for specified period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=period_hours)
        
        # Filter alerts by period
        period_alerts = [
            alert for alert in self._alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        # Count by severity
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for alert in period_alerts:
            severity_counts[alert.severity] += 1
            category_counts[alert.category] += 1
        
        # Calculate metric statistics
        metric_stats = {}
        for metric_name, history in self._metric_history.items():
            period_values = [
                value for timestamp, value in history
                if timestamp >= cutoff_time
            ]
            
            if period_values:
                metric_stats[metric_name] = {
                    "current": period_values[-1] if period_values else 0,
                    "min": min(period_values),
                    "max": max(period_values),
                    "mean": np.mean(period_values),
                    "std": np.std(period_values),
                }
        
        return {
            "period_hours": period_hours,
            "total_alerts": len(period_alerts),
            "alerts_by_severity": dict(severity_counts),
            "alerts_by_category": dict(category_counts),
            "metric_statistics": metric_stats,
            "current_active_alerts": len(self._active_alerts),
            "health_status": self._calculate_health_status(
                {"portfolio_metrics": {}}  # Use current metrics
            ),
        }
    
    async def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk monitoring report."""
        # Get current risk state
        risk_metrics = await self.risk_manager.monitor_portfolio_risk()
        
        # Generate report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_status": {
                "active": self._monitoring_active,
                "websocket_clients": len(self._websocket_clients),
                "update_interval": self.update_interval,
            },
            "current_metrics": risk_metrics,
            "risk_score": self._calculate_risk_score(risk_metrics),
            "health_status": self._calculate_health_status(risk_metrics),
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity,
                    "category": alert.category,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat(),
                    "auto_action_taken": alert.auto_action_taken,
                }
                for alert in self._active_alerts.values()
            ],
            "summary_24h": self.get_risk_summary(24),
            "summary_7d": self.get_risk_summary(168),
            "metric_trends": self._calculate_metric_trends(),
            "threshold_configuration": {
                name: {
                    "warning": threshold.warning_level,
                    "critical": threshold.critical_level,
                    "emergency": threshold.emergency_level,
                    "check_interval": threshold.check_interval,
                    "enabled": threshold.enabled,
                }
                for name, threshold in self._thresholds.items()
            },
        }
        
        return report