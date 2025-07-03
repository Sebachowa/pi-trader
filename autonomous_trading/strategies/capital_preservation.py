
"""
Capital Preservation Strategies

Advanced strategies for protecting capital during adverse market conditions.
"""

import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from scipy import stats
import asyncio

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
# from nautilus_trader.common.logging import Logger  # Not available in this version
from nautilus_trader.model.identifiers import InstrumentId, PositionId
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.portfolio.base import PortfolioFacade


@dataclass
class PreservationStrategy:
    """Capital preservation strategy configuration."""
    name: str
    trigger_conditions: Dict[str, float]
    actions: List[str]
    priority: int
    enabled: bool = True
    cooldown_hours: int = 24
    last_activated: Optional[datetime] = None


@dataclass
class HedgePosition:
    """Hedge position for portfolio protection."""
    instrument_id: InstrumentId
    size: float
    hedge_type: str  # beta, volatility, tail, correlation
    target_exposure: float
    effectiveness: float = 0.0
    cost: float = 0.0
    expiry: Optional[datetime] = None


class CapitalPreservationManager(Component):
    """
    Advanced capital preservation strategies manager.
    
    Features:
    - Multiple preservation strategies
    - Dynamic hedging mechanisms
    - Portfolio insurance implementation
    - Safe haven allocation
    - Risk reduction protocols
    - Recovery planning
    """
    
    def __init__(
        self,
        logger: Any,  # Logger type
        clock: LiveClock,
        msgbus: MessageBus,
        portfolio: Optional[PortfolioFacade] = None,
        risk_manager=None,  # ComprehensiveRiskManager instance
        # Preservation thresholds
        drawdown_threshold: float = 0.05,  # 5%
        volatility_threshold: float = 0.25,  # 25% annualized
        correlation_threshold: float = 0.8,
        loss_streak_threshold: int = 5,
        # Hedging configuration
        enable_dynamic_hedging: bool = True,
        max_hedge_cost_percent: float = 1.0,  # 1% of portfolio
        hedge_rebalance_interval: int = 3600,  # seconds
        # Safe haven configuration
        safe_haven_allocation_percent: float = 20.0,
        safe_haven_instruments: List[str] = ["USDT", "USDC", "DAI"],
        # Recovery configuration
        recovery_confidence_threshold: float = 0.7,
        gradual_reentry_periods: int = 5,
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id="CAPITAL-PRESERVATION",
            msgbus=msgbus,
        )
        
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        
        # Preservation thresholds
        self.drawdown_threshold = drawdown_threshold
        self.volatility_threshold = volatility_threshold
        self.correlation_threshold = correlation_threshold
        self.loss_streak_threshold = loss_streak_threshold
        
        # Hedging configuration
        self.enable_dynamic_hedging = enable_dynamic_hedging
        self.max_hedge_cost_percent = max_hedge_cost_percent
        self.hedge_rebalance_interval = hedge_rebalance_interval
        
        # Safe haven configuration
        self.safe_haven_allocation_percent = safe_haven_allocation_percent
        self.safe_haven_instruments = safe_haven_instruments
        
        # Recovery configuration
        self.recovery_confidence_threshold = recovery_confidence_threshold
        self.gradual_reentry_periods = gradual_reentry_periods
        
        # State tracking
        self._preservation_mode_active = False
        self._preservation_level = 0  # 0-3 (none, light, moderate, maximum)
        self._hedge_positions: Dict[str, HedgePosition] = {}
        self._safe_haven_allocation = 0.0
        
        # Strategy configurations
        self._preservation_strategies = self._initialize_strategies()
        self._active_strategies: Set[str] = set()
        
        # Performance tracking
        self._preservation_events = deque(maxlen=100)
        self._hedge_effectiveness = deque(maxlen=50)
        self._recovery_metrics = defaultdict(list)
        
        # Market regime tracking
        self._regime_history = deque(maxlen=100)
        self._regime_transitions = deque(maxlen=20)
        
        # Monitoring tasks
        self._monitoring_task = None
        self._hedge_rebalance_task = None
        
    def _initialize_strategies(self) -> List[PreservationStrategy]:
        """Initialize capital preservation strategies."""
        return [
            PreservationStrategy(
                name="drawdown_protection",
                trigger_conditions={"drawdown": self.drawdown_threshold},
                actions=["reduce_positions", "increase_cash", "activate_hedges"],
                priority=1,
            ),
            PreservationStrategy(
                name="volatility_protection",
                trigger_conditions={"volatility": self.volatility_threshold},
                actions=["reduce_leverage", "widen_stops", "volatility_hedge"],
                priority=2,
            ),
            PreservationStrategy(
                name="correlation_breakdown",
                trigger_conditions={"correlation": self.correlation_threshold},
                actions=["diversify_positions", "correlation_hedge"],
                priority=3,
            ),
            PreservationStrategy(
                name="loss_streak_protection",
                trigger_conditions={"consecutive_losses": self.loss_streak_threshold},
                actions=["pause_trading", "position_review", "strategy_rotation"],
                priority=2,
            ),
            PreservationStrategy(
                name="tail_risk_protection",
                trigger_conditions={"tail_risk_score": 0.9},
                actions=["tail_hedge", "reduce_all_positions"],
                priority=1,
            ),
            PreservationStrategy(
                name="liquidity_crisis",
                trigger_conditions={"liquidity_score": 0.3},
                actions=["close_illiquid", "increase_cash", "emergency_exits"],
                priority=1,
            ),
            PreservationStrategy(
                name="systematic_risk",
                trigger_conditions={"systematic_risk_score": 0.8},
                actions=["market_neutral", "sector_rotation", "defensive_assets"],
                priority=1,
            ),
        ]
    
    async def evaluate_preservation_needs(self) -> Dict[str, Any]:
        """Evaluate current need for capital preservation."""
        if not self.portfolio or not self.risk_manager:
            return {"error": "Missing portfolio or risk manager"}
        
        # Get current risk metrics
        risk_metrics = await self.risk_manager.monitor_portfolio_risk()
        
        # Evaluate each strategy
        triggered_strategies = []
        preservation_score = 0.0
        
        for strategy in self._preservation_strategies:
            if not strategy.enabled:
                continue
            
            # Check cooldown
            if strategy.last_activated:
                if datetime.utcnow() - strategy.last_activated < timedelta(hours=strategy.cooldown_hours):
                    continue
            
            # Check trigger conditions
            triggered = self._check_strategy_triggers(strategy, risk_metrics)
            
            if triggered:
                triggered_strategies.append(strategy)
                preservation_score += 1.0 / strategy.priority
        
        # Determine preservation level
        if preservation_score >= 2.0:
            preservation_level = 3  # Maximum
        elif preservation_score >= 1.0:
            preservation_level = 2  # Moderate
        elif preservation_score >= 0.5:
            preservation_level = 1  # Light
        else:
            preservation_level = 0  # None
        
        return {
            "preservation_score": preservation_score,
            "preservation_level": preservation_level,
            "triggered_strategies": [s.name for s in triggered_strategies],
            "current_mode": self._preservation_mode_active,
            "recommendations": self._generate_recommendations(
                preservation_level, triggered_strategies
            ),
        }
    
    def _check_strategy_triggers(
        self,
        strategy: PreservationStrategy,
        risk_metrics: Dict[str, Any],
    ) -> bool:
        """Check if strategy trigger conditions are met."""
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
        
        for condition, threshold in strategy.trigger_conditions.items():
            current_value = None
            
            if condition == "drawdown":
                current_value = portfolio_metrics.get("current_drawdown", 0)
            elif condition == "volatility":
                current_value = self._calculate_portfolio_volatility()
            elif condition == "correlation":
                current_value = portfolio_metrics.get("correlation_risk", 0)
            elif condition == "consecutive_losses":
                current_value = self.risk_manager._consecutive_losses
            elif condition == "tail_risk_score":
                current_value = self._calculate_tail_risk_score(risk_metrics)
            elif condition == "liquidity_score":
                current_value = self._calculate_liquidity_score()
            elif condition == "systematic_risk_score":
                current_value = self._calculate_systematic_risk_score()
            
            if current_value is not None and current_value >= threshold:
                return True
        
        return False
    
    async def activate_preservation_mode(
        self,
        level: int,
        strategies: List[str],
    ) -> Dict[str, Any]:
        """Activate capital preservation mode."""
        self._preservation_mode_active = True
        self._preservation_level = level
        
        self._log.warning(
            f"Activating capital preservation mode - Level: {level}, "
            f"Strategies: {strategies}"
        )
        
        results = {
            "timestamp": datetime.utcnow(),
            "level": level,
            "strategies": strategies,
            "actions_taken": [],
        }
        
        # Execute preservation actions based on level
        if level >= 1:
            # Light preservation
            action_result = await self._execute_light_preservation()
            results["actions_taken"].extend(action_result)
        
        if level >= 2:
            # Moderate preservation
            action_result = await self._execute_moderate_preservation()
            results["actions_taken"].extend(action_result)
        
        if level >= 3:
            # Maximum preservation
            action_result = await self._execute_maximum_preservation()
            results["actions_taken"].extend(action_result)
        
        # Execute strategy-specific actions
        for strategy_name in strategies:
            strategy = next(
                (s for s in self._preservation_strategies if s.name == strategy_name),
                None
            )
            if strategy:
                action_result = await self._execute_strategy_actions(strategy)
                results["actions_taken"].extend(action_result)
                
                # Update strategy state
                strategy.last_activated = datetime.utcnow()
                self._active_strategies.add(strategy_name)
        
        # Record preservation event
        self._preservation_events.append({
            "timestamp": datetime.utcnow(),
            "level": level,
            "strategies": strategies,
            "portfolio_value": float(self.portfolio.account_balance_total()),
        })
        
        # Start monitoring
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._preservation_monitoring_loop())
        
        return results
    
    async def _execute_light_preservation(self) -> List[Dict[str, Any]]:
        """Execute light capital preservation actions."""
        actions = []
        
        # Reduce position sizes by 25%
        if self.risk_manager:
            self.risk_manager._risk_reduction_mode = True
            actions.append({
                "action": "risk_reduction_mode",
                "details": "Enabled risk reduction mode",
            })
        
        # Increase cash allocation
        cash_action = await self._increase_cash_allocation(0.1)  # 10%
        if cash_action:
            actions.append(cash_action)
        
        # Tighten risk parameters
        if self.risk_manager:
            self.risk_manager.max_position_risk_percent *= 0.75
            actions.append({
                "action": "tighten_risk_parameters",
                "details": "Reduced position risk by 25%",
            })
        
        return actions
    
    async def _execute_moderate_preservation(self) -> List[Dict[str, Any]]:
        """Execute moderate capital preservation actions."""
        actions = []
        
        # Reduce positions by 50%
        reduction_result = await self._reduce_all_positions(0.5)
        actions.append({
            "action": "reduce_positions",
            "details": f"Reduced {reduction_result} positions by 50%",
        })
        
        # Activate hedges
        if self.enable_dynamic_hedging:
            hedge_result = await self._activate_portfolio_hedges()
            actions.append({
                "action": "activate_hedges",
                "details": f"Activated {len(hedge_result)} hedges",
                "hedges": hedge_result,
            })
        
        # Move to defensive assets
        defensive_result = await self._rotate_to_defensive_assets()
        if defensive_result:
            actions.append(defensive_result)
        
        return actions
    
    async def _execute_maximum_preservation(self) -> List[Dict[str, Any]]:
        """Execute maximum capital preservation actions."""
        actions = []
        
        # Close all speculative positions
        close_result = await self._close_speculative_positions()
        actions.append({
            "action": "close_speculative",
            "details": f"Closed {close_result} speculative positions",
        })
        
        # Maximum cash allocation
        cash_action = await self._increase_cash_allocation(0.5)  # 50%
        if cash_action:
            actions.append(cash_action)
        
        # Implement portfolio insurance
        insurance_result = await self._implement_portfolio_insurance()
        if insurance_result:
            actions.append(insurance_result)
        
        # Emergency stop
        if self.risk_manager:
            self.risk_manager._emergency_stop_active = True
            actions.append({
                "action": "emergency_stop",
                "details": "Activated emergency trading stop",
            })
        
        return actions
    
    async def _execute_strategy_actions(
        self,
        strategy: PreservationStrategy,
    ) -> List[Dict[str, Any]]:
        """Execute specific strategy actions."""
        actions = []
        
        for action in strategy.actions:
            if action == "reduce_positions":
                result = await self._reduce_all_positions(0.3)
                actions.append({
                    "strategy": strategy.name,
                    "action": action,
                    "result": f"Reduced {result} positions",
                })
            
            elif action == "increase_cash":
                result = await self._increase_cash_allocation(0.2)
                if result:
                    actions.append({
                        "strategy": strategy.name,
                        "action": action,
                        "result": result,
                    })
            
            elif action == "activate_hedges":
                hedges = await self._activate_portfolio_hedges()
                actions.append({
                    "strategy": strategy.name,
                    "action": action,
                    "result": f"Activated {len(hedges)} hedges",
                })
            
            elif action == "volatility_hedge":
                hedge = await self._create_volatility_hedge()
                if hedge:
                    actions.append({
                        "strategy": strategy.name,
                        "action": action,
                        "result": "Created volatility hedge",
                    })
            
            elif action == "tail_hedge":
                hedge = await self._create_tail_hedge()
                if hedge:
                    actions.append({
                        "strategy": strategy.name,
                        "action": action,
                        "result": "Created tail risk hedge",
                    })
            
            elif action == "correlation_hedge":
                hedge = await self._create_correlation_hedge()
                if hedge:
                    actions.append({
                        "strategy": strategy.name,
                        "action": action,
                        "result": "Created correlation hedge",
                    })
        
        return actions
    
    async def _activate_portfolio_hedges(self) -> List[HedgePosition]:
        """Activate portfolio hedges based on current exposures."""
        hedges = []
        
        if not self.portfolio:
            return hedges
        
        # Calculate portfolio exposures
        portfolio_beta = await self._calculate_portfolio_beta()
        portfolio_volatility = self._calculate_portfolio_volatility()
        
        # Beta hedge
        if abs(portfolio_beta) > 0.3:
            beta_hedge = await self._create_beta_hedge(portfolio_beta)
            if beta_hedge:
                hedges.append(beta_hedge)
                self._hedge_positions["beta"] = beta_hedge
        
        # Volatility hedge
        if portfolio_volatility > self.volatility_threshold:
            vol_hedge = await self._create_volatility_hedge()
            if vol_hedge:
                hedges.append(vol_hedge)
                self._hedge_positions["volatility"] = vol_hedge
        
        # Tail risk hedge
        tail_risk = self._calculate_tail_risk_score({})
        if tail_risk > 0.7:
            tail_hedge = await self._create_tail_hedge()
            if tail_hedge:
                hedges.append(tail_hedge)
                self._hedge_positions["tail"] = tail_hedge
        
        return hedges
    
    async def _create_beta_hedge(self, portfolio_beta: float) -> Optional[HedgePosition]:
        """Create beta hedge for market exposure."""
        if not self.portfolio:
            return None
        
        portfolio_value = float(self.portfolio.account_balance_total())
        hedge_size = abs(portfolio_beta) * portfolio_value
        
        # Check hedge cost constraint
        hedge_cost = hedge_size * 0.001  # Estimate 0.1% cost
        if hedge_cost > portfolio_value * (self.max_hedge_cost_percent / 100):
            hedge_size *= (self.max_hedge_cost_percent / 100) / (hedge_cost / portfolio_value)
        
        return HedgePosition(
            instrument_id=InstrumentId.from_str("SPY_INVERSE"),  # Example
            size=-hedge_size,  # Negative for short
            hedge_type="beta",
            target_exposure=-portfolio_beta,
            cost=hedge_cost,
        )
    
    async def _create_volatility_hedge(self) -> Optional[HedgePosition]:
        """Create volatility hedge."""
        if not self.portfolio:
            return None
        
        portfolio_value = float(self.portfolio.account_balance_total())
        current_vol = self._calculate_portfolio_volatility()
        
        # Size based on volatility level
        hedge_size = portfolio_value * min(0.1, (current_vol - 0.15) / 2)
        
        return HedgePosition(
            instrument_id=InstrumentId.from_str("VIX_CALL"),  # Example
            size=hedge_size,
            hedge_type="volatility",
            target_exposure=0.15,  # Target 15% volatility
            cost=hedge_size * 0.02,  # Estimate 2% premium
        )
    
    async def _create_tail_hedge(self) -> Optional[HedgePosition]:
        """Create tail risk hedge."""
        if not self.portfolio:
            return None
        
        portfolio_value = float(self.portfolio.account_balance_total())
        
        # Out-of-the-money put options
        hedge_size = portfolio_value * 0.05  # 5% allocation
        
        return HedgePosition(
            instrument_id=InstrumentId.from_str("SPY_PUT_OTM"),  # Example
            size=hedge_size,
            hedge_type="tail",
            target_exposure=0.0,
            cost=hedge_size * 0.03,  # 3% premium for OTM options
            expiry=datetime.utcnow() + timedelta(days=30),
        )
    
    async def _create_correlation_hedge(self) -> Optional[HedgePosition]:
        """Create correlation hedge."""
        if not self.portfolio:
            return None
        
        # This would analyze portfolio correlations and create offsetting positions
        # Simplified implementation
        return None
    
    async def _increase_cash_allocation(self, target_percent: float) -> Optional[Dict[str, Any]]:
        """Increase cash allocation by closing positions."""
        if not self.portfolio:
            return None
        
        current_cash = self._calculate_cash_percentage()
        if current_cash >= target_percent:
            return None
        
        # Calculate how much to liquidate
        portfolio_value = float(self.portfolio.account_balance_total())
        target_cash_value = portfolio_value * target_percent
        current_cash_value = portfolio_value * current_cash
        to_liquidate = target_cash_value - current_cash_value
        
        # Close positions to raise cash
        positions_closed = await self._close_positions_for_cash(to_liquidate)
        
        return {
            "action": "increase_cash",
            "target_percent": target_percent * 100,
            "positions_closed": positions_closed,
            "amount_raised": to_liquidate,
        }
    
    async def _rotate_to_defensive_assets(self) -> Dict[str, Any]:
        """Rotate portfolio to defensive assets."""
        # This would close risky positions and open defensive ones
        # Simplified implementation
        return {
            "action": "defensive_rotation",
            "details": "Rotated to defensive assets",
        }
    
    async def _implement_portfolio_insurance(self) -> Dict[str, Any]:
        """Implement portfolio insurance strategy."""
        if not self.portfolio:
            return {}
        
        portfolio_value = float(self.portfolio.account_balance_total())
        
        # Calculate insurance parameters
        floor_value = portfolio_value * 0.85  # 85% floor
        cushion = portfolio_value - floor_value
        
        # CPPI (Constant Proportion Portfolio Insurance) strategy
        multiplier = 3  # Risk multiplier
        risky_allocation = cushion * multiplier
        safe_allocation = portfolio_value - risky_allocation
        
        return {
            "action": "portfolio_insurance",
            "type": "CPPI",
            "floor_value": floor_value,
            "cushion": cushion,
            "risky_allocation": risky_allocation,
            "safe_allocation": safe_allocation,
        }
    
    async def _preservation_monitoring_loop(self) -> None:
        """Monitor preservation effectiveness and adjust as needed."""
        while self._preservation_mode_active:
            try:
                # Evaluate current conditions
                evaluation = await self.evaluate_preservation_needs()
                
                # Check if we can reduce preservation level
                if evaluation["preservation_level"] < self._preservation_level:
                    await self._reduce_preservation_level()
                
                # Rebalance hedges if needed
                if self.enable_dynamic_hedging:
                    await self._rebalance_hedges()
                
                # Check recovery conditions
                if await self._check_recovery_conditions():
                    await self._initiate_recovery_phase()
                
                # Sleep
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self._log.error(f"Error in preservation monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _rebalance_hedges(self) -> None:
        """Rebalance hedge positions."""
        for hedge_type, hedge in self._hedge_positions.items():
            # Calculate hedge effectiveness
            effectiveness = await self._calculate_hedge_effectiveness(hedge)
            hedge.effectiveness = effectiveness
            
            # Rebalance if effectiveness is low
            if effectiveness < 0.7:
                self._log.info(f"Rebalancing {hedge_type} hedge (effectiveness: {effectiveness:.2f})")
                # Implement rebalancing logic
    
    async def _check_recovery_conditions(self) -> bool:
        """Check if conditions are suitable for recovery."""
        if not self.risk_manager:
            return False
        
        # Check drawdown recovery
        if self.risk_manager._current_drawdown > self.drawdown_threshold * 0.5:
            return False
        
        # Check volatility normalization
        if self._calculate_portfolio_volatility() > self.volatility_threshold * 0.8:
            return False
        
        # Check win rate improvement
        if self.risk_manager._win_rate < 0.5:
            return False
        
        # Check market conditions
        market_score = await self._calculate_market_recovery_score()
        if market_score < self.recovery_confidence_threshold:
            return False
        
        return True
    
    async def _initiate_recovery_phase(self) -> Dict[str, Any]:
        """Initiate gradual recovery from preservation mode."""
        self._log.info("Initiating recovery phase from capital preservation")
        
        recovery_plan = {
            "start_time": datetime.utcnow(),
            "phases": self.gradual_reentry_periods,
            "current_phase": 1,
            "target_exposure": [],
        }
        
        # Calculate gradual exposure increase
        current_exposure = self._calculate_current_exposure_percent()
        target_exposure = 1.0  # 100% normal exposure
        
        for phase in range(self.gradual_reentry_periods):
            phase_exposure = current_exposure + (
                (target_exposure - current_exposure) * 
                (phase + 1) / self.gradual_reentry_periods
            )
            recovery_plan["target_exposure"].append(phase_exposure)
        
        # Start recovery
        self._recovery_metrics["current_plan"] = recovery_plan
        
        return recovery_plan
    
    async def _reduce_preservation_level(self) -> None:
        """Reduce preservation level gradually."""
        if self._preservation_level > 0:
            self._preservation_level -= 1
            self._log.info(f"Reducing preservation level to {self._preservation_level}")
            
            # Adjust risk parameters
            if self.risk_manager:
                if self._preservation_level == 0:
                    self.risk_manager._risk_reduction_mode = False
                    self.risk_manager._emergency_stop_active = False
    
    # Helper methods
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate current portfolio volatility."""
        if not self.risk_manager or not self.risk_manager._returns_history:
            return 0.0
        
        returns = np.array(list(self.risk_manager._returns_history)[-20:])
        if len(returns) < 5:
            return 0.0
        
        # Annualized volatility
        return np.std(returns) * np.sqrt(252)
    
    def _calculate_tail_risk_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate tail risk score (0-1)."""
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
        
        # Use CVaR and other tail metrics
        cvar = abs(portfolio_metrics.get("portfolio_cvar", 0))
        var = abs(portfolio_metrics.get("portfolio_var", 0))
        
        if var > 0:
            tail_ratio = cvar / var
        else:
            tail_ratio = 1.0
        
        # Higher ratio indicates fatter tails
        tail_score = min(1.0, (tail_ratio - 1.0) / 0.5)
        
        return tail_score
    
    def _calculate_liquidity_score(self) -> float:
        """Calculate portfolio liquidity score (0-1)."""
        # Simplified implementation
        return 0.8
    
    def _calculate_systematic_risk_score(self) -> float:
        """Calculate systematic risk score (0-1)."""
        # This would analyze market-wide risks
        return 0.3
    
    def _generate_recommendations(
        self,
        preservation_level: int,
        triggered_strategies: List[PreservationStrategy],
    ) -> List[str]:
        """Generate preservation recommendations."""
        recommendations = []
        
        if preservation_level >= 1:
            recommendations.append("Reduce position sizes by 25-50%")
            recommendations.append("Increase stop loss distances")
            recommendations.append("Avoid new speculative positions")
        
        if preservation_level >= 2:
            recommendations.append("Move 30-50% to cash or safe assets")
            recommendations.append("Implement portfolio hedges")
            recommendations.append("Consider defensive sector rotation")
        
        if preservation_level >= 3:
            recommendations.append("Emergency capital preservation required")
            recommendations.append("Close all speculative positions")
            recommendations.append("Maximum cash allocation (50%+)")
            recommendations.append("Implement portfolio insurance")
        
        # Strategy-specific recommendations
        for strategy in triggered_strategies:
            if strategy.name == "drawdown_protection":
                recommendations.append("Focus on capital recovery, not returns")
            elif strategy.name == "volatility_protection":
                recommendations.append("Wait for volatility to normalize")
            elif strategy.name == "loss_streak_protection":
                recommendations.append("Review and adjust trading strategies")
        
        return recommendations
    
    async def _reduce_all_positions(self, reduction_factor: float) -> int:
        """Reduce all positions by a factor."""
        if not self.portfolio:
            return 0
        
        positions = self.portfolio.positions_open()
        
        # This would integrate with execution system
        # Placeholder implementation
        self._log.info(f"Reducing {len(positions)} positions by {reduction_factor*100:.0f}%")
        
        return len(positions)
    
    async def _close_speculative_positions(self) -> int:
        """Close positions deemed speculative."""
        if not self.portfolio:
            return 0
        
        positions = self.portfolio.positions_open()
        speculative_count = 0
        
        for position in positions:
            # Determine if position is speculative
            # This would use various criteria
            if self._is_speculative_position(position):
                speculative_count += 1
        
        return speculative_count
    
    def _is_speculative_position(self, position: Position) -> bool:
        """Determine if a position is speculative."""
        # Simplified criteria
        # Would check leverage, volatility, sector, etc.
        return False
    
    def _calculate_cash_percentage(self) -> float:
        """Calculate current cash percentage of portfolio."""
        if not self.portfolio:
            return 0.0
        
        # Simplified calculation
        return 0.1  # 10% cash
    
    async def _close_positions_for_cash(self, target_amount: float) -> int:
        """Close positions to raise specified cash amount."""
        # This would close least profitable or most risky positions first
        return 0
    
    async def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta."""
        # Simplified calculation
        return 0.8
    
    async def _calculate_hedge_effectiveness(self, hedge: HedgePosition) -> float:
        """Calculate effectiveness of a hedge position."""
        # This would analyze hedge performance vs portfolio
        return 0.85
    
    async def _calculate_market_recovery_score(self) -> float:
        """Calculate market recovery score for re-entry."""
        # This would analyze market conditions
        return 0.6
    
    def _calculate_current_exposure_percent(self) -> float:
        """Calculate current exposure as percentage of normal."""
        if self._preservation_level == 3:
            return 0.3
        elif self._preservation_level == 2:
            return 0.5
        elif self._preservation_level == 1:
            return 0.75
        else:
            return 1.0
    
    async def get_preservation_status(self) -> Dict[str, Any]:
        """Get current capital preservation status."""
        return {
            "preservation_mode_active": self._preservation_mode_active,
            "preservation_level": self._preservation_level,
            "active_strategies": list(self._active_strategies),
            "hedge_positions": {
                hedge_type: {
                    "instrument": str(hedge.instrument_id),
                    "size": hedge.size,
                    "effectiveness": hedge.effectiveness,
                    "cost": hedge.cost,
                }
                for hedge_type, hedge in self._hedge_positions.items()
            },
            "safe_haven_allocation": self._safe_haven_allocation,
            "recovery_status": self._recovery_metrics.get("current_plan"),
            "preservation_events": len(self._preservation_events),
        }