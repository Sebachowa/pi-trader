
"""
Enhanced Autonomous Engine with Minimal Intervention capabilities.

This engine extends the base autonomous engine with advanced features for
truly hands-off operation while maintaining safety and profitability.
"""

import asyncio
import json
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from autonomous_trading.core.engine import AutonomousEngine, SystemState, HealthStatus
from autonomous_trading.config.minimal_intervention_architecture import (
    MinimalInterventionConfig,
    InterventionLevel,
    AutomationFeature,
)
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
# from nautilus_trader.common.logging import Logger  # Not available in this version
from nautilus_trader.live.node import TradingNode


class DecisionType(Enum):
    """Types of autonomous decisions."""
    RISK_ADJUSTMENT = "risk_adjustment"
    STRATEGY_SELECTION = "strategy_selection"
    CAPITAL_ALLOCATION = "capital_allocation"
    POSITION_SIZING = "position_sizing"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    EMERGENCY_ACTION = "emergency_action"


class DecisionConfidence(Enum):
    """Confidence levels for autonomous decisions."""
    VERY_HIGH = "very_high"  # >90% confidence
    HIGH = "high"  # 75-90% confidence
    MEDIUM = "medium"  # 50-75% confidence
    LOW = "low"  # <50% confidence


class AutonomousDecision:
    """Represents an autonomous decision made by the system."""
    
    def __init__(
        self,
        decision_type: DecisionType,
        action: str,
        confidence: DecisionConfidence,
        reasoning: str,
        parameters: Dict[str, Any],
        requires_confirmation: bool = False,
    ):
        self.decision_type = decision_type
        self.action = action
        self.confidence = confidence
        self.reasoning = reasoning
        self.parameters = parameters
        self.requires_confirmation = requires_confirmation
        self.timestamp = datetime.utcnow()
        self.executed = False
        self.result = None


class EnhancedAutonomousEngine(AutonomousEngine):
    """
    Enhanced autonomous engine with minimal intervention capabilities.
    
    Features:
    - Advanced decision-making with confidence scoring
    - Self-optimizing parameters
    - Intelligent strategy evolution
    - Automated capital management
    - Predictive maintenance
    - Machine learning integration
    """
    
    def __init__(
        self,
        config: MinimalInterventionConfig,
        trading_node: TradingNode,
        logger: Any,  # Logger type
        clock: LiveClock,
        msgbus: MessageBus,
    ):
        # Initialize base engine with adapted config
        base_config = self._create_base_config(config)
        super().__init__(
            config=base_config,
            trading_node=trading_node,
            logger=logger,
            clock=clock,
            msgbus=msgbus,
        )
        
        self.enhanced_config = config
        
        # Decision tracking
        self._pending_decisions: List[AutonomousDecision] = []
        self._decision_history: deque = deque(maxlen=1000)
        self._decision_performance: Dict[DecisionType, Dict[str, float]] = defaultdict(
            lambda: {"success_rate": 0.0, "avg_impact": 0.0, "count": 0}
        )
        
        # Learning components
        self._market_memory: deque = deque(maxlen=10000)
        self._strategy_performance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._optimization_results: deque = deque(maxlen=50)
        
        # Enhanced state tracking
        self._autonomous_metrics = {
            "decisions_made": 0,
            "decisions_pending": 0,
            "interventions_requested": 0,
            "self_heals_performed": 0,
            "optimizations_completed": 0,
            "strategies_evolved": 0,
        }
        
        # Advanced components
        self._ml_predictor = None
        self._strategy_evolver = None
        self._capital_optimizer = None
        
        # Tasks
        self._decision_engine_task = None
        self._learning_task = None
        self._evolution_task = None
        
    def _create_base_config(self, config: MinimalInterventionConfig):
        """Create base engine config from enhanced config."""
        from autonomous_trading.core.engine import AutonomousEngineConfig
        
        return AutonomousEngineConfig(
            trader_id=config.trader_id,
            enable_self_healing=AutomationFeature.SELF_HEALING in config.enabled_features,
            health_check_interval_seconds=30,
            max_recovery_attempts=5,
            recovery_delay_seconds=30,
            enable_auto_shutdown=True,
            daily_maintenance_time=config.maintenance_time_utc,
            max_daily_loss_percent=config.max_daily_loss_percent,
            max_drawdown_percent=config.max_drawdown_percent,
            enable_notifications=config.critical_notification_only,
            state_persistence_path="./enhanced_autonomous_state.json",
        )
    
    async def start(self) -> None:
        """Start the enhanced autonomous engine."""
        await super().start()
        
        # Initialize enhanced components
        await self._initialize_enhanced_components()
        
        # Start enhanced tasks
        self._decision_engine_task = asyncio.create_task(self._decision_engine_loop())
        self._learning_task = asyncio.create_task(self._learning_loop())
        self._evolution_task = asyncio.create_task(self._evolution_loop())
        
        self._log.info("Enhanced Autonomous Engine started with minimal intervention mode")
    
    async def stop(self) -> None:
        """Stop the enhanced autonomous engine."""
        # Cancel enhanced tasks
        for task in [self._decision_engine_task, self._learning_task, self._evolution_task]:
            if task:
                task.cancel()
        
        await super().stop()
    
    async def _initialize_enhanced_components(self) -> None:
        """Initialize enhanced autonomous components."""
        # Initialize ML predictor
        if self.enhanced_config.ml_enabled:
            self._ml_predictor = await self._create_ml_predictor()
        
        # Initialize strategy evolver
        if AutomationFeature.STRATEGY_EVOLUTION in self.enhanced_config.enabled_features:
            self._strategy_evolver = await self._create_strategy_evolver()
        
        # Initialize capital optimizer
        if AutomationFeature.CAPITAL_MANAGEMENT in self.enhanced_config.enabled_features:
            self._capital_optimizer = await self._create_capital_optimizer()
    
    async def _decision_engine_loop(self) -> None:
        """Main decision-making loop for autonomous operations."""
        while self._state == SystemState.RUNNING:
            try:
                # Analyze current state
                system_state = await self._analyze_system_state()
                
                # Generate decisions based on state
                decisions = await self._generate_decisions(system_state)
                
                # Filter decisions by confidence and intervention level
                executable_decisions = self._filter_decisions(decisions)
                
                # Execute autonomous decisions
                for decision in executable_decisions:
                    await self._execute_decision(decision)
                
                # Handle pending decisions requiring confirmation
                await self._handle_pending_decisions()
                
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Decision engine error: {e}")
    
    async def _analyze_system_state(self) -> Dict[str, Any]:
        """Comprehensive system state analysis."""
        state = {
            "timestamp": datetime.utcnow(),
            "health": self._health_status,
            "performance": await self._get_performance_metrics(),
            "risk": await self._get_risk_metrics(),
            "market": await self._get_market_conditions(),
            "strategies": await self._get_strategy_status(),
            "capital": await self._get_capital_status(),
        }
        
        # Add ML predictions if available
        if self._ml_predictor:
            state["predictions"] = await self._ml_predictor.get_predictions()
        
        return state
    
    async def _generate_decisions(self, system_state: Dict[str, Any]) -> List[AutonomousDecision]:
        """Generate autonomous decisions based on system state."""
        decisions = []
        
        # Risk adjustment decisions
        if AutomationFeature.AUTO_RISK_ADJUSTMENT in self.enhanced_config.enabled_features:
            risk_decisions = await self._generate_risk_decisions(system_state)
            decisions.extend(risk_decisions)
        
        # Strategy selection decisions
        if AutomationFeature.STRATEGY_EVOLUTION in self.enhanced_config.enabled_features:
            strategy_decisions = await self._generate_strategy_decisions(system_state)
            decisions.extend(strategy_decisions)
        
        # Capital allocation decisions
        if AutomationFeature.CAPITAL_MANAGEMENT in self.enhanced_config.enabled_features:
            capital_decisions = await self._generate_capital_decisions(system_state)
            decisions.extend(capital_decisions)
        
        # Portfolio rebalancing decisions
        if AutomationFeature.PORTFOLIO_REBALANCING in self.enhanced_config.enabled_features:
            rebalance_decisions = await self._generate_rebalance_decisions(system_state)
            decisions.extend(rebalance_decisions)
        
        return decisions
    
    async def _generate_risk_decisions(self, state: Dict[str, Any]) -> List[AutonomousDecision]:
        """Generate risk adjustment decisions."""
        decisions = []
        risk_metrics = state.get("risk", {})
        performance = state.get("performance", {})
        
        # Check if risk adjustment is needed
        current_drawdown = risk_metrics.get("current_drawdown", 0)
        volatility_ratio = risk_metrics.get("volatility_ratio", 1.0)
        win_rate = performance.get("win_rate", 0.5)
        
        # Drawdown-based risk reduction
        if current_drawdown > 0.05:  # 5% drawdown
            confidence = DecisionConfidence.HIGH if current_drawdown > 0.08 else DecisionConfidence.MEDIUM
            reduction_factor = min(0.5, current_drawdown * 10)  # Up to 50% reduction
            
            decision = AutonomousDecision(
                decision_type=DecisionType.RISK_ADJUSTMENT,
                action="reduce_position_sizes",
                confidence=confidence,
                reasoning=f"Drawdown of {current_drawdown:.1%} detected, reducing risk",
                parameters={
                    "reduction_factor": reduction_factor,
                    "apply_to": "all_strategies",
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        # Volatility-based adjustment
        if volatility_ratio > 1.5:  # 50% higher than average
            decision = AutonomousDecision(
                decision_type=DecisionType.RISK_ADJUSTMENT,
                action="adjust_volatility_scaling",
                confidence=DecisionConfidence.HIGH,
                reasoning=f"High volatility detected (ratio: {volatility_ratio:.2f})",
                parameters={
                    "volatility_multiplier": 1.0 / volatility_ratio,
                    "duration_hours": 24,
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        # Performance-based risk increase (careful)
        if win_rate > 0.65 and current_drawdown < 0.02:  # High win rate, low drawdown
            decision = AutonomousDecision(
                decision_type=DecisionType.RISK_ADJUSTMENT,
                action="increase_position_sizes",
                confidence=DecisionConfidence.MEDIUM,
                reasoning=f"Strong performance (win rate: {win_rate:.1%}) with low risk",
                parameters={
                    "increase_factor": 1.2,  # 20% increase
                    "max_increase": 1.5,  # Cap at 50% increase
                },
                requires_confirmation=self.enhanced_config.intervention_level != InterventionLevel.NONE,
            )
            decisions.append(decision)
        
        return decisions
    
    async def _generate_strategy_decisions(self, state: Dict[str, Any]) -> List[AutonomousDecision]:
        """Generate strategy selection and evolution decisions."""
        decisions = []
        strategies = state.get("strategies", {})
        market = state.get("market", {})
        
        # Analyze strategy performance
        underperforming = []
        outperforming = []
        
        for strategy_id, metrics in strategies.items():
            if metrics.get("sharpe_ratio", 0) < 0.5:
                underperforming.append(strategy_id)
            elif metrics.get("sharpe_ratio", 0) > 2.0:
                outperforming.append(strategy_id)
        
        # Remove underperforming strategies
        if underperforming:
            decision = AutonomousDecision(
                decision_type=DecisionType.STRATEGY_SELECTION,
                action="remove_strategies",
                confidence=DecisionConfidence.HIGH,
                reasoning=f"Removing {len(underperforming)} underperforming strategies",
                parameters={
                    "strategy_ids": underperforming,
                    "replacement_strategy": "ai_swarm",  # Default replacement
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        # Clone and evolve outperforming strategies
        if outperforming and self._strategy_evolver:
            decision = AutonomousDecision(
                decision_type=DecisionType.STRATEGY_SELECTION,
                action="evolve_strategies",
                confidence=DecisionConfidence.MEDIUM,
                reasoning=f"Evolving {len(outperforming)} high-performing strategies",
                parameters={
                    "base_strategies": outperforming,
                    "evolution_method": "parameter_mutation",
                    "mutation_rate": 0.1,
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        # Market regime-based strategy activation
        regime = market.get("regime", "unknown")
        recommended_strategies = self._get_regime_strategies(regime)
        active_strategies = set(strategies.keys())
        
        missing_strategies = set(recommended_strategies) - active_strategies
        if missing_strategies:
            decision = AutonomousDecision(
                decision_type=DecisionType.STRATEGY_SELECTION,
                action="deploy_strategies",
                confidence=DecisionConfidence.HIGH,
                reasoning=f"Market regime '{regime}' suggests additional strategies",
                parameters={
                    "strategies": list(missing_strategies),
                    "allocation_each": 0.1,  # 10% each
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        return decisions
    
    async def _generate_capital_decisions(self, state: Dict[str, Any]) -> List[AutonomousDecision]:
        """Generate capital allocation and management decisions."""
        decisions = []
        capital = state.get("capital", {})
        performance = state.get("performance", {})
        
        total_capital = capital.get("total", 0)
        free_capital = capital.get("free", 0)
        monthly_return = performance.get("monthly_return", 0)
        
        # Compound profits decision
        if self.enhanced_config.compound_profits and monthly_return > 0.01:  # 1% monthly profit
            reinvest_amount = monthly_return * total_capital * \
                self.enhanced_config.profit_reinvestment_percent / 100
            
            decision = AutonomousDecision(
                decision_type=DecisionType.CAPITAL_ALLOCATION,
                action="reinvest_profits",
                confidence=DecisionConfidence.VERY_HIGH,
                reasoning=f"Reinvesting {self.enhanced_config.profit_reinvestment_percent}% of profits",
                parameters={
                    "amount": reinvest_amount,
                    "allocation_method": self.enhanced_config.capital_allocation_method,
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        # Kelly criterion position sizing
        if self.enhanced_config.capital_allocation_method == "kelly_criterion":
            kelly_fractions = await self._calculate_kelly_fractions(state)
            
            decision = AutonomousDecision(
                decision_type=DecisionType.CAPITAL_ALLOCATION,
                action="adjust_position_sizes",
                confidence=DecisionConfidence.HIGH,
                reasoning="Applying Kelly criterion for optimal position sizing",
                parameters={
                    "kelly_fractions": kelly_fractions,
                    "safety_multiplier": 0.25,  # Use 25% of Kelly for safety
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        # Reserve fund management
        reserve_target = total_capital * self.enhanced_config.reserve_fund_percent / 100
        current_reserve = capital.get("reserve", 0)
        
        if current_reserve < reserve_target * 0.8:  # Below 80% of target
            decision = AutonomousDecision(
                decision_type=DecisionType.CAPITAL_ALLOCATION,
                action="replenish_reserves",
                confidence=DecisionConfidence.HIGH,
                reasoning=f"Reserve fund below target ({current_reserve:.0f} < {reserve_target:.0f})",
                parameters={
                    "amount": reserve_target - current_reserve,
                    "source": "trading_profits",
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        return decisions
    
    async def _generate_rebalance_decisions(self, state: Dict[str, Any]) -> List[AutonomousDecision]:
        """Generate portfolio rebalancing decisions."""
        decisions = []
        
        # Get current allocations
        current_allocations = await self._get_current_allocations()
        target_allocations = self.enhanced_config.diversification_targets
        
        # Calculate deviations
        rebalance_needed = False
        rebalance_trades = []
        
        for asset_class, target_weight in target_allocations.items():
            current_weight = current_allocations.get(asset_class, 0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > self.enhanced_config.rebalancing_threshold_percent / 100:
                rebalance_needed = True
                rebalance_trades.append({
                    "asset_class": asset_class,
                    "current": current_weight,
                    "target": target_weight,
                    "adjustment": target_weight - current_weight,
                })
        
        if rebalance_needed:
            decision = AutonomousDecision(
                decision_type=DecisionType.PORTFOLIO_REBALANCE,
                action="rebalance_portfolio",
                confidence=DecisionConfidence.HIGH,
                reasoning="Portfolio allocation deviates from targets",
                parameters={
                    "trades": rebalance_trades,
                    "execution_method": "gradual",  # Gradual to minimize impact
                    "time_window_hours": 24,
                },
                requires_confirmation=False,
            )
            decisions.append(decision)
        
        return decisions
    
    def _filter_decisions(self, decisions: List[AutonomousDecision]) -> List[AutonomousDecision]:
        """Filter decisions based on confidence and intervention settings."""
        executable = []
        
        for decision in decisions:
            # Check intervention level
            if self.enhanced_config.intervention_level == InterventionLevel.NONE:
                # Execute all decisions automatically
                executable.append(decision)
            
            elif self.enhanced_config.intervention_level == InterventionLevel.CRITICAL_ONLY:
                # Only require confirmation for critical decisions
                if decision.decision_type == DecisionType.EMERGENCY_ACTION:
                    decision.requires_confirmation = True
                    self._pending_decisions.append(decision)
                else:
                    executable.append(decision)
            
            elif self.enhanced_config.intervention_level == InterventionLevel.STRATEGIC:
                # Require confirmation for strategic decisions
                if decision.decision_type in [
                    DecisionType.STRATEGY_SELECTION,
                    DecisionType.CAPITAL_ALLOCATION,
                ]:
                    decision.requires_confirmation = True
                    self._pending_decisions.append(decision)
                else:
                    executable.append(decision)
            
            else:  # OPERATIONAL level
                # Require confirmation for most decisions
                if decision.confidence == DecisionConfidence.VERY_HIGH:
                    executable.append(decision)
                else:
                    decision.requires_confirmation = True
                    self._pending_decisions.append(decision)
        
        return executable
    
    async def _execute_decision(self, decision: AutonomousDecision) -> None:
        """Execute an autonomous decision."""
        self._log.info(
            f"Executing {decision.decision_type.value}: {decision.action} "
            f"(confidence: {decision.confidence.value})"
        )
        
        try:
            # Route to appropriate handler
            if decision.decision_type == DecisionType.RISK_ADJUSTMENT:
                await self._execute_risk_adjustment(decision)
            
            elif decision.decision_type == DecisionType.STRATEGY_SELECTION:
                await self._execute_strategy_selection(decision)
            
            elif decision.decision_type == DecisionType.CAPITAL_ALLOCATION:
                await self._execute_capital_allocation(decision)
            
            elif decision.decision_type == DecisionType.PORTFOLIO_REBALANCE:
                await self._execute_portfolio_rebalance(decision)
            
            elif decision.decision_type == DecisionType.EMERGENCY_ACTION:
                await self._execute_emergency_action(decision)
            
            # Record execution
            decision.executed = True
            decision.result = "success"
            self._decision_history.append(decision)
            self._autonomous_metrics["decisions_made"] += 1
            
            # Update decision performance
            self._update_decision_performance(decision, success=True)
            
        except Exception as e:
            self._log.error(f"Failed to execute decision: {e}")
            decision.result = f"failed: {e}"
            self._decision_history.append(decision)
            self._update_decision_performance(decision, success=False)
    
    async def _execute_risk_adjustment(self, decision: AutonomousDecision) -> None:
        """Execute risk adjustment decision."""
        params = decision.parameters
        
        if decision.action == "reduce_position_sizes":
            # Update risk controller with new limits
            reduction_factor = params["reduction_factor"]
            if self._risk_controller:
                current_limit = self._risk_controller.max_position_risk_percent
                new_limit = current_limit * (1 - reduction_factor)
                self._risk_controller.max_position_risk_percent = max(0.5, new_limit)  # Min 0.5%
                
        elif decision.action == "adjust_volatility_scaling":
            # Apply volatility scaling to all strategies
            if self._strategy_orchestrator:
                await self._strategy_orchestrator.apply_volatility_scaling(
                    params["volatility_multiplier"]
                )
    
    async def _execute_strategy_selection(self, decision: AutonomousDecision) -> None:
        """Execute strategy selection decision."""
        params = decision.parameters
        
        if decision.action == "remove_strategies":
            # Stop underperforming strategies
            for strategy_id in params["strategy_ids"]:
                if self._strategy_orchestrator:
                    await self._strategy_orchestrator._emergency_stop_strategy(strategy_id)
        
        elif decision.action == "deploy_strategies":
            # Deploy new strategies
            if self._strategy_orchestrator:
                for strategy_name in params["strategies"]:
                    await self._strategy_orchestrator.deploy_strategy(
                        strategy_name,
                        params["allocation_each"],
                        [],  # Instruments would be determined by strategy
                    )
    
    async def _execute_capital_allocation(self, decision: AutonomousDecision) -> None:
        """Execute capital allocation decision."""
        params = decision.parameters
        
        if decision.action == "reinvest_profits":
            # This would integrate with portfolio management
            self._log.info(f"Reinvesting {params['amount']:.2f} in profits")
        
        elif decision.action == "adjust_position_sizes":
            # Apply Kelly criterion sizing
            if self._risk_controller:
                # Update position sizing logic with Kelly fractions
                pass
    
    async def _execute_portfolio_rebalance(self, decision: AutonomousDecision) -> None:
        """Execute portfolio rebalancing decision."""
        params = decision.parameters
        
        # This would integrate with execution system
        for trade in params["trades"]:
            self._log.info(
                f"Rebalancing {trade['asset_class']}: "
                f"{trade['current']:.1%} -> {trade['target']:.1%}"
            )
    
    async def _execute_emergency_action(self, decision: AutonomousDecision) -> None:
        """Execute emergency action decision."""
        self._log.warning(f"Executing emergency action: {decision.action}")
        
        # Emergency actions are always executed immediately
        if decision.action == "stop_all_trading":
            self._state = SystemState.PAUSED
            if self._risk_controller:
                await self._risk_controller.close_all_positions("Emergency stop")
    
    def _update_decision_performance(self, decision: AutonomousDecision, success: bool) -> None:
        """Update performance metrics for decision types."""
        perf = self._decision_performance[decision.decision_type]
        perf["count"] += 1
        
        # Update success rate
        old_rate = perf["success_rate"]
        perf["success_rate"] = (old_rate * (perf["count"] - 1) + (1 if success else 0)) / perf["count"]
    
    async def _learning_loop(self) -> None:
        """Continuous learning and adaptation loop."""
        while self._state == SystemState.RUNNING:
            try:
                # Collect market data
                market_data = await self._collect_market_data()
                self._market_memory.append(market_data)
                
                # Update ML models if enabled
                if self._ml_predictor and len(self._market_memory) > 100:
                    await self._ml_predictor.update(list(self._market_memory))
                
                # Learn from decision outcomes
                await self._learn_from_decisions()
                
                # Update strategy performance history
                await self._update_strategy_performance()
                
                await asyncio.sleep(3600)  # Run hourly
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Learning loop error: {e}")
    
    async def _evolution_loop(self) -> None:
        """Strategy evolution and optimization loop."""
        while self._state == SystemState.RUNNING:
            try:
                if not self._strategy_evolver:
                    await asyncio.sleep(86400)  # Daily check
                    continue
                
                # Evaluate strategies for evolution
                evolution_candidates = await self._identify_evolution_candidates()
                
                # Evolve promising strategies
                for candidate in evolution_candidates:
                    evolved = await self._strategy_evolver.evolve(candidate)
                    if evolved:
                        self._autonomous_metrics["strategies_evolved"] += 1
                
                # Optimize parameters using Bayesian optimization
                if self.enhanced_config.optimization_enabled:
                    optimization_result = await self._run_parameter_optimization()
                    self._optimization_results.append(optimization_result)
                    self._autonomous_metrics["optimizations_completed"] += 1
                
                await asyncio.sleep(86400)  # Run daily
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Evolution loop error: {e}")
    
    async def _handle_pending_decisions(self) -> None:
        """Handle decisions requiring human confirmation."""
        if not self._pending_decisions:
            return
        
        # Check for automated approval conditions
        for decision in self._pending_decisions[:]:  # Copy to allow modification
            # Auto-approve high confidence decisions after timeout
            age = (datetime.utcnow() - decision.timestamp).total_seconds()
            
            if decision.confidence == DecisionConfidence.VERY_HIGH and age > 300:  # 5 minutes
                self._log.info(f"Auto-approving high confidence decision: {decision.action}")
                await self._execute_decision(decision)
                self._pending_decisions.remove(decision)
            
            elif age > 3600:  # 1 hour timeout
                self._log.warning(f"Decision timed out: {decision.action}")
                self._pending_decisions.remove(decision)
        
        # Update metrics
        self._autonomous_metrics["decisions_pending"] = len(self._pending_decisions)
    
    def approve_decision(self, decision_index: int) -> bool:
        """Approve a pending decision for execution."""
        if 0 <= decision_index < len(self._pending_decisions):
            decision = self._pending_decisions.pop(decision_index)
            asyncio.create_task(self._execute_decision(decision))
            return True
        return False
    
    def reject_decision(self, decision_index: int) -> bool:
        """Reject a pending decision."""
        if 0 <= decision_index < len(self._pending_decisions):
            decision = self._pending_decisions.pop(decision_index)
            decision.result = "rejected"
            self._decision_history.append(decision)
            return True
        return False
    
    @property
    def intervention_requests(self) -> List[Dict[str, Any]]:
        """Get current intervention requests."""
        return [
            {
                "index": i,
                "type": d.decision_type.value,
                "action": d.action,
                "confidence": d.confidence.value,
                "reasoning": d.reasoning,
                "parameters": d.parameters,
                "age_minutes": (datetime.utcnow() - d.timestamp).total_seconds() / 60,
            }
            for i, d in enumerate(self._pending_decisions)
        ]
    
    @property
    def autonomous_report(self) -> Dict[str, Any]:
        """Generate comprehensive autonomous operation report."""
        base_stats = self.system_stats
        
        return {
            **base_stats,
            "autonomous_metrics": self._autonomous_metrics,
            "intervention_level": self.enhanced_config.intervention_level.value,
            "enabled_features": [f.value for f in self.enhanced_config.enabled_features],
            "pending_decisions": len(self._pending_decisions),
            "decision_performance": {
                k.value: v for k, v in self._decision_performance.items()
            },
            "optimization_count": len(self._optimization_results),
            "learning_samples": len(self._market_memory),
        }
    
    # Helper methods
    
    async def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        # This would integrate with portfolio tracking
        return {
            "daily_return": 0.01,
            "monthly_return": 0.05,
            "win_rate": 0.55,
            "sharpe_ratio": 1.5,
            "profit_factor": 1.3,
        }
    
    async def _get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        return {
            "current_drawdown": self._current_drawdown,
            "volatility_ratio": 1.2,
            "var_95": 0.02,
            "portfolio_heat": 0.7,
        }
    
    async def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions."""
        if self._market_analyzer:
            return await self._market_analyzer.get_market_analysis()
        return {"regime": "unknown"}
    
    async def _get_strategy_status(self) -> Dict[str, Dict[str, float]]:
        """Get status of all active strategies."""
        if self._strategy_orchestrator:
            report = self._strategy_orchestrator.get_orchestrator_report()
            return {
                detail["id"]: {
                    "sharpe_ratio": detail["sharpe_ratio"],
                    "win_rate": detail["win_rate"],
                    "allocation": detail["allocation"],
                }
                for detail in report.get("strategy_details", [])
            }
        return {}
    
    async def _get_capital_status(self) -> Dict[str, float]:
        """Get current capital status."""
        # This would integrate with portfolio
        return {
            "total": 100000,
            "free": 20000,
            "reserve": 10000,
            "at_risk": 70000,
        }
    
    async def _get_current_allocations(self) -> Dict[str, float]:
        """Get current portfolio allocations by asset class."""
        # This would calculate from actual positions
        return {
            "crypto": 0.45,
            "forex": 0.25,
            "commodities": 0.15,
            "cash": 0.15,
        }
    
    def _get_regime_strategies(self, regime: str) -> List[str]:
        """Get recommended strategies for market regime."""
        regime_map = {
            "trending_up": ["trend_following", "momentum", "ai_swarm"],
            "trending_down": ["trend_following", "short_strategies"],
            "ranging": ["mean_reversion", "market_making"],
            "volatile": ["volatility_strategies", "ai_swarm"],
        }
        return regime_map.get(regime, ["ai_swarm"])
    
    async def _calculate_kelly_fractions(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Kelly criterion fractions for position sizing."""
        # Simplified Kelly calculation
        strategies = state.get("strategies", {})
        kelly_fractions = {}
        
        for strategy_id, metrics in strategies.items():
            win_rate = metrics.get("win_rate", 0.5)
            avg_win = 1.5  # Placeholder
            avg_loss = 1.0  # Placeholder
            
            if win_rate > 0 and avg_loss > 0:
                # Kelly formula: f = (p * b - q) / b
                # where p = win rate, q = loss rate, b = win/loss ratio
                b = avg_win / avg_loss
                q = 1 - win_rate
                f = (win_rate * b - q) / b
                
                # Constrain to reasonable values
                kelly_fractions[strategy_id] = max(0.0, min(0.25, f))
        
        return kelly_fractions
    
    async def _collect_market_data(self) -> Dict[str, Any]:
        """Collect current market data for learning."""
        return {
            "timestamp": datetime.utcnow(),
            "prices": {},  # Would get from data feeds
            "volumes": {},
            "volatility": {},
            "regime": "unknown",
        }
    
    async def _learn_from_decisions(self) -> None:
        """Learn from past decision outcomes."""
        # Analyze recent decisions and their outcomes
        recent_decisions = list(self._decision_history)[-50:]
        
        for decision_type in DecisionType:
            type_decisions = [d for d in recent_decisions if d.decision_type == decision_type]
            if len(type_decisions) > 5:
                # Update confidence thresholds based on outcomes
                success_by_confidence = defaultdict(list)
                for d in type_decisions:
                    success = d.result == "success"
                    success_by_confidence[d.confidence].append(success)
                
                # Adjust future confidence requirements
                for confidence, outcomes in success_by_confidence.items():
                    success_rate = sum(outcomes) / len(outcomes)
                    if success_rate < 0.5 and confidence != DecisionConfidence.VERY_HIGH:
                        # Increase confidence requirement for this decision type
                        self._log.info(
                            f"Increasing confidence requirement for {decision_type.value} "
                            f"due to low success rate ({success_rate:.1%})"
                        )
    
    async def _update_strategy_performance(self) -> None:
        """Update strategy performance history."""
        if self._strategy_orchestrator:
            report = self._strategy_orchestrator.get_orchestrator_report()
            for detail in report.get("strategy_details", []):
                strategy_id = detail["id"]
                performance = {
                    "timestamp": datetime.utcnow(),
                    "sharpe_ratio": detail["sharpe_ratio"],
                    "win_rate": detail["win_rate"],
                    "allocation": detail["allocation"],
                }
                self._strategy_performance_history[strategy_id].append(performance)
    
    async def _identify_evolution_candidates(self) -> List[str]:
        """Identify strategies suitable for evolution."""
        candidates = []
        
        for strategy_id, history in self._strategy_performance_history.items():
            if len(history) < 10:
                continue
            
            # Check for consistent performance
            recent_sharpe = np.mean([h["sharpe_ratio"] for h in list(history)[-10:]])
            recent_win_rate = np.mean([h["win_rate"] for h in list(history)[-10:]])
            
            if recent_sharpe > 1.5 and recent_win_rate > 0.55:
                candidates.append(strategy_id)
        
        return candidates
    
    async def _run_parameter_optimization(self) -> Dict[str, Any]:
        """Run parameter optimization using Bayesian methods."""
        # Placeholder for Bayesian optimization
        return {
            "timestamp": datetime.utcnow(),
            "method": "bayesian",
            "parameters_tested": 50,
            "improvement": 0.05,
            "best_params": {},
        }
    
    async def _create_ml_predictor(self):
        """Create ML predictor component."""
        # Placeholder - would integrate with actual ML models
        return None
    
    async def _create_strategy_evolver(self):
        """Create strategy evolution component."""
        # Placeholder - would implement genetic algorithms or similar
        return None
    
    async def _create_capital_optimizer(self):
        """Create capital optimization component."""
        # Placeholder - would implement portfolio optimization
        return None