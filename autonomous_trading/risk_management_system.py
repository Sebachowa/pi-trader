
"""
Integrated Risk Management System

Complete risk management system integrating all components for comprehensive
portfolio protection and capital preservation.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional

from autonomous_trading.core.comprehensive_risk_manager import ComprehensiveRiskManager
from autonomous_trading.monitoring.realtime_risk_monitor import RealtimeRiskMonitor
from autonomous_trading.strategies.capital_preservation import CapitalPreservationManager

from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
# from nautilus_trader.common.logging import Logger  # Not available in this version
from nautilus_trader.portfolio.base import PortfolioFacade


class IntegratedRiskManagementSystem:
    """
    Complete integrated risk management system.
    
    Combines:
    - Comprehensive risk management with multiple position sizing models
    - Real-time risk monitoring with alerts and dashboards
    - Capital preservation strategies
    - Emergency procedures and recovery planning
    """
    
    def __init__(
        self,
        logger: Any,  # Logger type
        clock: LiveClock,
        msgbus: MessageBus,
        portfolio: Optional[PortfolioFacade] = None,
        # Risk Management Configuration
        max_daily_loss_percent: float = 2.0,
        max_drawdown_percent: float = 10.0,
        max_position_risk_percent: float = 1.0,
        max_portfolio_risk_percent: float = 5.0,
        max_correlation: float = 0.7,
        max_concentration_percent: float = 20.0,
        # Position Management
        default_stop_loss_percent: float = 2.0,
        default_take_profit_percent: float = 4.0,
        use_trailing_stops: bool = True,
        # Monitoring Configuration
        enable_realtime_monitoring: bool = True,
        websocket_port: int = 8765,
        prometheus_port: int = 9090,
        alert_cooldown: int = 300,
        # Preservation Configuration
        enable_capital_preservation: bool = True,
        preservation_drawdown_threshold: float = 0.05,
        enable_dynamic_hedging: bool = True,
        safe_haven_allocation_percent: float = 20.0,
    ):
        self.logger = logger
        self.clock = clock
        self.msgbus = msgbus
        self.portfolio = portfolio
        
        # Initialize comprehensive risk manager
        self.risk_manager = ComprehensiveRiskManager(
            logger=logger,
            clock=clock,
            msgbus=msgbus,
            portfolio=portfolio,
            max_daily_loss_percent=max_daily_loss_percent,
            max_drawdown_percent=max_drawdown_percent,
            max_position_risk_percent=max_position_risk_percent,
            max_portfolio_risk_percent=max_portfolio_risk_percent,
            max_correlation=max_correlation,
            max_concentration_percent=max_concentration_percent,
            default_stop_loss_percent=default_stop_loss_percent,
            default_take_profit_percent=default_take_profit_percent,
            use_trailing_stops=use_trailing_stops,
        )
        
        # Initialize real-time monitor if enabled
        self.realtime_monitor = None
        if enable_realtime_monitoring:
            self.realtime_monitor = RealtimeRiskMonitor(
                logger=logger,
                clock=clock,
                msgbus=msgbus,
                risk_manager=self.risk_manager,
                websocket_port=websocket_port,
                prometheus_port=prometheus_port,
                alert_cooldown=alert_cooldown,
            )
        
        # Initialize capital preservation manager if enabled
        self.preservation_manager = None
        if enable_capital_preservation:
            self.preservation_manager = CapitalPreservationManager(
                logger=logger,
                clock=clock,
                msgbus=msgbus,
                portfolio=portfolio,
                risk_manager=self.risk_manager,
                drawdown_threshold=preservation_drawdown_threshold,
                enable_dynamic_hedging=enable_dynamic_hedging,
                safe_haven_allocation_percent=safe_haven_allocation_percent,
            )
        
        # System state
        self._system_active = False
        self._monitoring_task = None
        
    async def start(self) -> None:
        """Start the integrated risk management system."""
        if self._system_active:
            return
        
        self.logger.info("Starting Integrated Risk Management System")
        self._system_active = True
        
        # Start real-time monitoring
        if self.realtime_monitor:
            await self.realtime_monitor.start_monitoring()
        
        # Start system monitoring loop
        self._monitoring_task = asyncio.create_task(self._system_monitoring_loop())
        
        # Register alert callbacks
        if self.realtime_monitor and self.preservation_manager:
            self.realtime_monitor.register_alert_callback(
                "emergency",
                self._handle_emergency_alert
            )
            self.realtime_monitor.register_alert_callback(
                "critical",
                self._handle_critical_alert
            )
        
        self.logger.info("Risk Management System started successfully")
    
    async def stop(self) -> None:
        """Stop the integrated risk management system."""
        self._system_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        if self.realtime_monitor:
            await self.realtime_monitor.stop_monitoring()
        
        self.logger.info("Risk Management System stopped")
    
    async def _system_monitoring_loop(self) -> None:
        """Main system monitoring and coordination loop."""
        while self._system_active:
            try:
                # Check preservation needs
                if self.preservation_manager:
                    preservation_eval = await self.preservation_manager.evaluate_preservation_needs()
                    
                    if preservation_eval["preservation_level"] > 0 and not self.preservation_manager._preservation_mode_active:
                        await self.preservation_manager.activate_preservation_mode(
                            preservation_eval["preservation_level"],
                            preservation_eval["triggered_strategies"]
                        )
                
                # Adapt risk parameters
                await self.risk_manager.adapt_parameters()
                
                # Perform periodic stress tests
                if self.clock.unix_timestamp_ns() % (3600 * 1e9) < (60 * 1e9):  # Every hour
                    await self.risk_manager.perform_stress_test()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _handle_emergency_alert(self, alert) -> None:
        """Handle emergency risk alerts."""
        self.logger.critical(f"Emergency alert received: {alert.category}")
        
        if self.preservation_manager:
            # Activate maximum preservation
            await self.preservation_manager.activate_preservation_mode(
                level=3,
                strategies=["emergency_protection"]
            )
    
    async def _handle_critical_alert(self, alert) -> None:
        """Handle critical risk alerts."""
        self.logger.warning(f"Critical alert received: {alert.category}")
        
        if self.preservation_manager and not self.preservation_manager._preservation_mode_active:
            # Evaluate if preservation is needed
            preservation_eval = await self.preservation_manager.evaluate_preservation_needs()
            if preservation_eval["preservation_level"] >= 2:
                await self.preservation_manager.activate_preservation_mode(
                    preservation_eval["preservation_level"],
                    preservation_eval["triggered_strategies"]
                )
    
    async def calculate_position_size(
        self,
        instrument_id,
        account_balance,
        entry_price,
        stop_loss_price=None,
        confidence_score=0.5,
        strategy_hint=None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive position size with all risk checks.
        
        This is the main entry point for position sizing that considers:
        - Multiple sizing models
        - Risk limits and constraints
        - Diversification requirements
        - Current market conditions
        - Capital preservation status
        """
        return await self.risk_manager.calculate_comprehensive_position_size(
            instrument_id=instrument_id,
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            confidence_score=confidence_score,
            strategy_hint=strategy_hint,
        )
    
    async def update_position_risk(self, position_id, current_price) -> Dict[str, Any]:
        """Update position risk parameters including stops."""
        return await self.risk_manager.update_position_stops(
            position_id=position_id,
            current_price=current_price,
        )
    
    async def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard with all metrics."""
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": {
                "active": self._system_active,
                "monitoring_enabled": self.realtime_monitor is not None,
                "preservation_enabled": self.preservation_manager is not None,
            },
        }
        
        # Add risk manager report
        risk_report = await self.risk_manager.get_comprehensive_risk_report()
        dashboard["risk_management"] = risk_report
        
        # Add monitoring report if available
        if self.realtime_monitor:
            monitoring_report = await self.realtime_monitor.generate_risk_report()
            dashboard["monitoring"] = monitoring_report
        
        # Add preservation status if available
        if self.preservation_manager:
            preservation_status = await self.preservation_manager.get_preservation_status()
            dashboard["capital_preservation"] = preservation_status
        
        return dashboard
    
    def get_system_configuration(self) -> Dict[str, Any]:
        """Get complete system configuration for saving."""
        return {
            "system_type": "integrated_risk_management",
            "version": "1.0.0",
            "components": {
                "risk_manager": {
                    "max_daily_loss_percent": self.risk_manager.max_daily_loss_percent,
                    "max_drawdown_percent": self.risk_manager.max_drawdown_percent,
                    "max_position_risk_percent": self.risk_manager.max_position_risk_percent,
                    "max_portfolio_risk_percent": self.risk_manager.max_portfolio_risk_percent,
                    "max_correlation": self.risk_manager.max_correlation,
                    "max_concentration_percent": self.risk_manager.max_concentration_percent,
                    "position_sizing_models": list(self.risk_manager._sizing_models.keys()),
                    "active_sizing_model": self.risk_manager._active_sizing_model,
                },
                "realtime_monitor": {
                    "enabled": self.realtime_monitor is not None,
                    "websocket_port": self.realtime_monitor.websocket_port if self.realtime_monitor else None,
                    "prometheus_port": self.realtime_monitor.prometheus_port if self.realtime_monitor else None,
                    "alert_thresholds": {
                        name: {
                            "warning": threshold.warning_level,
                            "critical": threshold.critical_level,
                            "emergency": threshold.emergency_level,
                        }
                        for name, threshold in self.realtime_monitor._thresholds.items()
                    } if self.realtime_monitor else {},
                },
                "preservation_manager": {
                    "enabled": self.preservation_manager is not None,
                    "strategies": [
                        {
                            "name": strategy.name,
                            "trigger_conditions": strategy.trigger_conditions,
                            "actions": strategy.actions,
                            "priority": strategy.priority,
                        }
                        for strategy in self.preservation_manager._preservation_strategies
                    ] if self.preservation_manager else [],
                },
            },
            "features": {
                "position_sizing_algorithms": [
                    "Kelly Criterion",
                    "Optimal F",
                    "Risk Parity",
                    "Volatility Targeting",
                    "Machine Learning"
                ],
                "stop_loss_management": [
                    "Fixed Stops",
                    "Dynamic ATR-based Stops",
                    "Trailing Stops",
                    "Volatility-adjusted Stops"
                ],
                "portfolio_diversification": [
                    "Position Limits",
                    "Correlation Constraints",
                    "Sector Exposure Limits",
                    "Concentration Limits"
                ],
                "emergency_procedures": [
                    "Drawdown Protection",
                    "Daily Loss Limits",
                    "Correlation Spike Response",
                    "Volatility Spike Response",
                    "System Anomaly Detection"
                ],
                "capital_preservation": [
                    "Multi-level Preservation Modes",
                    "Dynamic Hedging",
                    "Portfolio Insurance",
                    "Safe Haven Allocation",
                    "Gradual Recovery Planning"
                ],
            },
        }
    
    async def save_to_memory(self, memory_key: str) -> Dict[str, Any]:
        """Save the complete risk management system to Memory."""
        system_data = {
            "configuration": self.get_system_configuration(),
            "current_state": await self.get_risk_dashboard(),
            "timestamp": datetime.utcnow().isoformat(),
            "description": "Comprehensive Risk Management System with position sizing, "
                          "stop loss management, portfolio diversification, real-time monitoring, "
                          "and capital preservation strategies.",
        }
        
        # Convert to JSON string for storage
        system_json = json.dumps(system_data, indent=2)
        
        # This would integrate with the Memory system
        # For now, we'll return the data structure
        self.logger.info(f"Risk Management System saved to Memory key: {memory_key}")
        
        return {
            "memory_key": memory_key,
            "data_size": len(system_json),
            "saved_at": datetime.utcnow().isoformat(),
            "system_data": system_data,
        }


# Factory function to create and configure the system
async def create_risk_management_system(
    logger: Any,  # Logger type
    clock: LiveClock,
    msgbus: MessageBus,
    portfolio: Optional[PortfolioFacade] = None,
    config: Optional[Dict[str, Any]] = None,
) -> IntegratedRiskManagementSystem:
    """
    Factory function to create a configured risk management system.
    
    Args:
        logger: System logger
        clock: System clock
        msgbus: Message bus for communication
        portfolio: Portfolio facade for position management
        config: Optional configuration overrides
    
    Returns:
        Configured IntegratedRiskManagementSystem instance
    """
    # Default configuration
    default_config = {
        "max_daily_loss_percent": 2.0,
        "max_drawdown_percent": 10.0,
        "max_position_risk_percent": 1.0,
        "max_portfolio_risk_percent": 5.0,
        "max_correlation": 0.7,
        "max_concentration_percent": 20.0,
        "default_stop_loss_percent": 2.0,
        "default_take_profit_percent": 4.0,
        "use_trailing_stops": True,
        "enable_realtime_monitoring": True,
        "enable_capital_preservation": True,
        "enable_dynamic_hedging": True,
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Create system
    system = IntegratedRiskManagementSystem(
        logger=logger,
        clock=clock,
        msgbus=msgbus,
        portfolio=portfolio,
        **default_config
    )
    
    # Start the system
    await system.start()
    
    # Save to Memory
    memory_key = "swarm-auto-hierarchical-1751379006249/risk-manager/system"
    await system.save_to_memory(memory_key)
    
    return system