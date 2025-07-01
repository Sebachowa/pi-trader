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
Autonomous Engine - Central coordination for 24/7 trading operations.
"""

import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.logging import Logger
from nautilus_trader.config import LiveTradingNodeConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class AutonomousEngineConfig:
    """Configuration for autonomous engine."""
    
    def __init__(
        self,
        trader_id: str = "AUTONOMOUS-001",
        enable_self_healing: bool = True,
        health_check_interval_seconds: int = 60,
        max_recovery_attempts: int = 3,
        recovery_delay_seconds: int = 30,
        enable_auto_shutdown: bool = True,
        daily_maintenance_time: Optional[str] = "03:00",  # UTC
        max_daily_loss_percent: float = 2.0,
        max_drawdown_percent: float = 10.0,
        enable_notifications: bool = True,
        state_persistence_path: str = "./autonomous_state.json",
    ):
        self.trader_id = trader_id
        self.enable_self_healing = enable_self_healing
        self.health_check_interval_seconds = health_check_interval_seconds
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_delay_seconds = recovery_delay_seconds
        self.enable_auto_shutdown = enable_auto_shutdown
        self.daily_maintenance_time = daily_maintenance_time
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_drawdown_percent = max_drawdown_percent
        self.enable_notifications = enable_notifications
        self.state_persistence_path = state_persistence_path


class AutonomousEngine(Component):
    """
    Central autonomous engine for 24/7 trading operations.
    
    Features:
    - Self-healing and error recovery
    - Automatic health monitoring
    - State persistence and recovery
    - Dynamic resource management
    - Scheduled maintenance windows
    """
    
    def __init__(
        self,
        config: AutonomousEngineConfig,
        trading_node: TradingNode,
        logger: Logger,
        clock: LiveClock,
        msgbus: MessageBus,
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id=f"{config.trader_id}-ENGINE",
            msgbus=msgbus,
        )
        
        self.config = config
        self.trading_node = trading_node
        self._state = SystemState.INITIALIZING
        self._health_status = HealthStatus.HEALTHY
        
        # Recovery tracking
        self._recovery_attempts = 0
        self._last_recovery_time = None
        self._error_count = 0
        self._last_error_time = None
        
        # Performance tracking
        self._start_time = None
        self._uptime_seconds = 0
        self._daily_pnl = 0.0
        self._peak_balance = 0.0
        self._current_drawdown = 0.0
        
        # Components
        self._risk_controller = None
        self._market_analyzer = None
        self._strategy_orchestrator = None
        self._performance_optimizer = None
        self._notification_system = None
        
        # Tasks
        self._health_check_task = None
        self._maintenance_task = None
        self._state_persistence_task = None
        
        # Circuit breakers
        self._kill_switch_active = False
        self._circuit_breaker_triggers = {
            "max_daily_loss": False,
            "max_drawdown": False,
            "system_error": False,
            "connection_loss": False,
        }
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    async def start(self) -> None:
        """Start the autonomous engine."""
        self._log.info("Starting Autonomous Engine...")
        
        try:
            # Load persisted state if exists
            await self._load_persisted_state()
            
            # Initialize components
            await self._initialize_components()
            
            # Start trading node
            await self.trading_node.start()
            
            # Start monitoring tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            self._state_persistence_task = asyncio.create_task(self._state_persistence_loop())
            
            self._state = SystemState.RUNNING
            self._start_time = datetime.utcnow()
            
            self._log.info("Autonomous Engine started successfully")
            
            # Send startup notification
            if self._notification_system:
                await self._notification_system.send_notification(
                    level="INFO",
                    title="Autonomous Trading System Started",
                    message=f"System started at {self._start_time} UTC",
                )
            
        except Exception as e:
            self._log.error(f"Failed to start Autonomous Engine: {e}")
            self._state = SystemState.ERROR
            await self._handle_critical_error(e)

    async def stop(self) -> None:
        """Stop the autonomous engine gracefully."""
        self._log.info("Stopping Autonomous Engine...")
        
        self._state = SystemState.STOPPING
        
        try:
            # Cancel monitoring tasks
            if self._health_check_task:
                self._health_check_task.cancel()
            if self._maintenance_task:
                self._maintenance_task.cancel()
            if self._state_persistence_task:
                self._state_persistence_task.cancel()
            
            # Stop trading node
            await self.trading_node.stop()
            
            # Save final state
            await self._persist_state()
            
            self._state = SystemState.STOPPED
            self._log.info("Autonomous Engine stopped successfully")
            
            # Send shutdown notification
            if self._notification_system:
                await self._notification_system.send_notification(
                    level="INFO",
                    title="Autonomous Trading System Stopped",
                    message=f"System stopped at {datetime.utcnow()} UTC. Uptime: {self._uptime_seconds}s",
                )
            
        except Exception as e:
            self._log.error(f"Error during shutdown: {e}")

    async def _initialize_components(self) -> None:
        """Initialize all autonomous components."""
        from autonomous_trading.core.risk_controller import RiskController
        from autonomous_trading.core.market_analyzer import MarketAnalyzer
        from autonomous_trading.strategies.orchestrator import StrategyOrchestrator
        from autonomous_trading.monitoring.performance import PerformanceOptimizer
        from autonomous_trading.monitoring.notifications import NotificationSystem
        
        # Initialize risk controller
        self._risk_controller = RiskController(
            logger=self._log,
            clock=self._clock,
            msgbus=self._msgbus,
            max_daily_loss_percent=self.config.max_daily_loss_percent,
            max_drawdown_percent=self.config.max_drawdown_percent,
        )
        
        # Initialize market analyzer
        self._market_analyzer = MarketAnalyzer(
            logger=self._log,
            clock=self._clock,
            msgbus=self._msgbus,
        )
        
        # Initialize strategy orchestrator
        self._strategy_orchestrator = StrategyOrchestrator(
            logger=self._log,
            clock=self._clock,
            msgbus=self._msgbus,
        )
        
        # Initialize performance optimizer
        self._performance_optimizer = PerformanceOptimizer(
            logger=self._log,
            clock=self._clock,
            msgbus=self._msgbus,
        )
        
        # Initialize notification system
        if self.config.enable_notifications:
            self._notification_system = NotificationSystem(
                logger=self._log,
                clock=self._clock,
                msgbus=self._msgbus,
            )

    async def _health_check_loop(self) -> None:
        """Continuous health monitoring loop."""
        while self._state == SystemState.RUNNING:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Health check error: {e}")

    async def _perform_health_check(self) -> None:
        """Perform comprehensive system health check."""
        health_metrics = {
            "uptime": (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
            "error_count": self._error_count,
            "recovery_attempts": self._recovery_attempts,
            "daily_pnl": self._daily_pnl,
            "current_drawdown": self._current_drawdown,
            "memory_usage_mb": self._get_memory_usage(),
            "cpu_usage_percent": self._get_cpu_usage(),
        }
        
        # Check circuit breakers
        if self._daily_pnl < -(self.config.max_daily_loss_percent / 100):
            self._circuit_breaker_triggers["max_daily_loss"] = True
            self._health_status = HealthStatus.CRITICAL
            await self._trigger_circuit_breaker("Maximum daily loss exceeded")
        
        if self._current_drawdown > (self.config.max_drawdown_percent / 100):
            self._circuit_breaker_triggers["max_drawdown"] = True
            self._health_status = HealthStatus.CRITICAL
            await self._trigger_circuit_breaker("Maximum drawdown exceeded")
        
        # Check component health
        if self._risk_controller and not await self._risk_controller.is_healthy():
            self._health_status = HealthStatus.WARNING
        
        self._log.debug(f"Health check completed: {health_metrics}")

    async def _maintenance_loop(self) -> None:
        """Scheduled maintenance operations."""
        if not self.config.daily_maintenance_time:
            return
        
        while self._state == SystemState.RUNNING:
            try:
                # Calculate next maintenance window
                next_maintenance = self._calculate_next_maintenance_time()
                sleep_seconds = (next_maintenance - datetime.utcnow()).total_seconds()
                
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
                
                # Perform maintenance
                await self._perform_maintenance()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Maintenance error: {e}")

    async def _perform_maintenance(self) -> None:
        """Perform scheduled maintenance tasks."""
        self._log.info("Starting scheduled maintenance...")
        
        # Pause trading during maintenance
        old_state = self._state
        self._state = SystemState.PAUSED
        
        try:
            # Clean up old logs
            await self._cleanup_old_logs()
            
            # Optimize memory usage
            await self._optimize_memory()
            
            # Reset daily metrics
            self._daily_pnl = 0.0
            self._error_count = 0
            
            # Persist current state
            await self._persist_state()
            
            self._log.info("Maintenance completed successfully")
            
        finally:
            # Resume trading
            self._state = old_state

    async def _state_persistence_loop(self) -> None:
        """Periodically persist system state."""
        while self._state == SystemState.RUNNING:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._persist_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"State persistence error: {e}")

    async def _persist_state(self) -> None:
        """Save current system state to disk."""
        state_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "state": self._state.value,
            "health_status": self._health_status.value,
            "uptime_seconds": self._uptime_seconds,
            "daily_pnl": self._daily_pnl,
            "current_drawdown": self._current_drawdown,
            "error_count": self._error_count,
            "recovery_attempts": self._recovery_attempts,
            "circuit_breakers": self._circuit_breaker_triggers,
        }
        
        try:
            with open(self.config.state_persistence_path, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            self._log.error(f"Failed to persist state: {e}")

    async def _load_persisted_state(self) -> None:
        """Load previously persisted state."""
        if not os.path.exists(self.config.state_persistence_path):
            return
        
        try:
            with open(self.config.state_persistence_path, 'r') as f:
                state_data = json.load(f)
            
            self._error_count = state_data.get("error_count", 0)
            self._recovery_attempts = state_data.get("recovery_attempts", 0)
            
            self._log.info(f"Loaded persisted state from {state_data['timestamp']}")
            
        except Exception as e:
            self._log.error(f"Failed to load persisted state: {e}")

    async def _handle_critical_error(self, error: Exception) -> None:
        """Handle critical system errors with recovery attempts."""
        self._error_count += 1
        self._last_error_time = datetime.utcnow()
        
        if not self.config.enable_self_healing:
            self._log.error("Self-healing disabled, shutting down...")
            await self.stop()
            return
        
        if self._recovery_attempts >= self.config.max_recovery_attempts:
            self._log.error("Maximum recovery attempts exceeded, shutting down...")
            self._kill_switch_active = True
            await self.stop()
            return
        
        self._log.info(f"Attempting recovery (attempt {self._recovery_attempts + 1}/{self.config.max_recovery_attempts})...")
        self._state = SystemState.RECOVERING
        self._recovery_attempts += 1
        
        # Wait before recovery
        await asyncio.sleep(self.config.recovery_delay_seconds)
        
        try:
            # Attempt to restart components
            await self._restart_failed_components()
            
            self._state = SystemState.RUNNING
            self._log.info("Recovery successful")
            
            # Send recovery notification
            if self._notification_system:
                await self._notification_system.send_notification(
                    level="WARNING",
                    title="System Recovered from Error",
                    message=f"Recovery successful after {self._recovery_attempts} attempts",
                )
            
        except Exception as recovery_error:
            self._log.error(f"Recovery failed: {recovery_error}")
            await self._handle_critical_error(recovery_error)

    async def _restart_failed_components(self) -> None:
        """Restart any failed components."""
        # Restart trading node if needed
        if not self.trading_node.is_running:
            await self.trading_node.start()
        
        # Re-initialize components
        await self._initialize_components()

    async def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker to stop all trading."""
        self._log.warning(f"Circuit breaker triggered: {reason}")
        
        # Pause all trading
        self._state = SystemState.PAUSED
        
        # Close all positions
        if self._risk_controller:
            await self._risk_controller.close_all_positions("Circuit breaker triggered")
        
        # Send critical notification
        if self._notification_system:
            await self._notification_system.send_notification(
                level="CRITICAL",
                title="Circuit Breaker Triggered",
                message=f"Trading halted: {reason}",
                priority="high",
            )

    def _calculate_next_maintenance_time(self) -> datetime:
        """Calculate next maintenance window time."""
        now = datetime.utcnow()
        
        # Parse maintenance time
        hour, minute = map(int, self.config.daily_maintenance_time.split(':'))
        
        # Calculate next occurrence
        next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        if next_time <= now:
            # Already passed today, schedule for tomorrow
            next_time += timedelta(days=1)
        
        return next_time

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.cpu_percent(interval=1)
        except ImportError:
            return 0.0

    async def _cleanup_old_logs(self) -> None:
        """Clean up old log files during maintenance."""
        # Implementation depends on logging configuration
        pass

    async def _optimize_memory(self) -> None:
        """Optimize memory usage during maintenance."""
        import gc
        gc.collect()

    def _handle_shutdown_signal(self, signum, frame) -> None:
        """Handle system shutdown signals."""
        self._log.info(f"Received shutdown signal {signum}")
        asyncio.create_task(self.stop())

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return (
            self._state == SystemState.RUNNING and
            self._health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING] and
            not self._kill_switch_active
        )

    @property
    def system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return {
            "state": self._state.value,
            "health_status": self._health_status.value,
            "uptime_seconds": self._uptime_seconds,
            "daily_pnl": self._daily_pnl,
            "current_drawdown": self._current_drawdown,
            "error_count": self._error_count,
            "recovery_attempts": self._recovery_attempts,
            "circuit_breakers": self._circuit_breaker_triggers,
            "kill_switch_active": self._kill_switch_active,
        }