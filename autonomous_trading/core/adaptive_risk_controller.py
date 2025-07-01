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
Adaptive Risk Controller with Self-Adjusting Parameters

Advanced risk management system that autonomously adjusts risk parameters
based on market conditions, performance metrics, and machine learning predictions.
"""

import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats
from sklearn.ensemble import IsolationForest

from autonomous_trading.core.risk_controller import RiskController, RiskLevel
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.logging import Logger
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.portfolio.base import PortfolioFacade


class MarketCondition:
    """Represents current market conditions for risk adjustment."""
    
    def __init__(self):
        self.volatility_regime = "normal"  # low, normal, high, extreme
        self.trend_strength = 0.0  # -1 to 1
        self.liquidity_score = 1.0  # 0 to 1
        self.correlation_level = 0.0  # Average correlation
        self.event_risk = False  # Major news/events
        self.anomaly_score = 0.0  # Market anomaly detection
        

class AdaptiveRiskController(RiskController):
    """
    Advanced risk controller with autonomous parameter adjustment.
    
    Features:
    - Machine learning-based risk prediction
    - Dynamic volatility regime detection
    - Adaptive position sizing with multiple models
    - Correlation-based portfolio risk management
    - Stress testing and scenario analysis
    - Real-time anomaly detection
    - Self-optimizing risk parameters
    """
    
    def __init__(
        self,
        logger: Logger,
        clock: LiveClock,
        msgbus: MessageBus,
        portfolio: Optional[PortfolioFacade] = None,
        max_daily_loss_percent: float = 2.0,
        max_drawdown_percent: float = 10.0,
        max_position_risk_percent: float = 1.0,
        max_portfolio_risk_percent: float = 5.0,
        max_correlation: float = 0.7,
        var_confidence_level: float = 0.95,
        cvar_confidence_level: float = 0.95,
        stress_test_scenarios: int = 1000,
        enable_ml_predictions: bool = True,
        adaptation_speed: float = 0.1,  # How quickly to adjust parameters
    ):
        super().__init__(
            logger=logger,
            clock=clock,
            msgbus=msgbus,
            portfolio=portfolio,
            max_daily_loss_percent=max_daily_loss_percent,
            max_drawdown_percent=max_drawdown_percent,
            max_position_risk_percent=max_position_risk_percent,
            max_portfolio_risk_percent=max_portfolio_risk_percent,
            max_correlation=max_correlation,
            var_confidence_level=var_confidence_level,
        )
        
        # Enhanced configuration
        self.cvar_confidence_level = cvar_confidence_level
        self.stress_test_scenarios = stress_test_scenarios
        self.enable_ml_predictions = enable_ml_predictions
        self.adaptation_speed = adaptation_speed
        
        # Market condition tracking
        self._market_conditions: Dict[InstrumentId, MarketCondition] = {}
        self._volatility_regimes = deque(maxlen=100)
        self._regime_transitions = deque(maxlen=50)
        
        # Advanced risk metrics
        self._cvar_history = deque(maxlen=252)
        self._stress_test_results = deque(maxlen=20)
        self._tail_risk_metrics = {}
        self._regime_specific_limits = self._initialize_regime_limits()
        
        # ML components
        self._anomaly_detector = IsolationForest(contamination=0.05)
        self._risk_predictor = None  # Would be proper ML model
        self._parameter_optimizer = None
        
        # Adaptive parameters
        self._adaptive_risk_multipliers = defaultdict(lambda: 1.0)
        self._parameter_history = deque(maxlen=1000)
        self._optimization_scores = deque(maxlen=100)
        
        # Position sizing models
        self._sizing_models = {
            "kelly": self._kelly_position_size,
            "optimal_f": self._optimal_f_position_size,
            "risk_parity": self._risk_parity_position_size,
            "volatility_targeting": self._volatility_target_position_size,
            "machine_learning": self._ml_position_size,
        }
        self._active_sizing_model = "volatility_targeting"
        
        # Enhanced tracking
        self._position_outcomes = defaultdict(list)  # Track outcomes by instrument
        self._regime_performance = defaultdict(lambda: {"pnl": 0, "trades": 0})
        self._adaptation_log = deque(maxlen=500)
        
    def _initialize_regime_limits(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk limits for different market regimes."""
        return {
            "low": {
                "position_risk_multiplier": 1.2,
                "portfolio_risk_multiplier": 1.1,
                "max_positions": 15,
                "concentration_limit": 0.15,
            },
            "normal": {
                "position_risk_multiplier": 1.0,
                "portfolio_risk_multiplier": 1.0,
                "max_positions": 12,
                "concentration_limit": 0.12,
            },
            "high": {
                "position_risk_multiplier": 0.7,
                "portfolio_risk_multiplier": 0.8,
                "max_positions": 8,
                "concentration_limit": 0.08,
            },
            "extreme": {
                "position_risk_multiplier": 0.3,
                "portfolio_risk_multiplier": 0.5,
                "max_positions": 5,
                "concentration_limit": 0.05,
            },
        }
    
    async def calculate_adaptive_position_size(
        self,
        instrument_id: InstrumentId,
        account_balance: Money,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal] = None,
        confidence_score: float = 0.5,
    ) -> Quantity:
        """Calculate position size using adaptive multi-model approach."""
        
        # Update market conditions
        await self._update_market_conditions(instrument_id)
        
        # Get base position size from parent
        base_size = super().calculate_position_size(
            instrument_id, account_balance, entry_price, stop_loss_price
        )
        
        # Apply selected sizing model
        model_func = self._sizing_models[self._active_sizing_model]
        model_size = await model_func(
            instrument_id, account_balance, entry_price, stop_loss_price, confidence_score
        )
        
        # Blend sizes based on model confidence
        model_confidence = self._get_model_confidence()
        final_size = (base_size.as_double() * (1 - model_confidence) + 
                     model_size * model_confidence)
        
        # Apply regime-specific adjustments
        regime_adjusted = await self._apply_regime_adjustments(
            instrument_id, final_size
        )
        
        # Apply portfolio-level constraints
        constrained_size = await self._apply_portfolio_constraints(
            instrument_id, regime_adjusted
        )
        
        # Apply ML predictions if enabled
        if self.enable_ml_predictions:
            ml_adjusted = await self._apply_ml_adjustments(
                instrument_id, constrained_size, confidence_score
            )
        else:
            ml_adjusted = constrained_size
        
        # Final safety checks
        final_quantity = self._apply_final_safety_checks(
            instrument_id, ml_adjusted, account_balance
        )
        
        # Log adaptation
        self._log_adaptation(instrument_id, base_size.as_double(), final_quantity)
        
        return Quantity.from_int(int(final_quantity))
    
    async def _update_market_conditions(self, instrument_id: InstrumentId) -> None:
        """Update current market conditions for the instrument."""
        if instrument_id not in self._market_conditions:
            self._market_conditions[instrument_id] = MarketCondition()
        
        condition = self._market_conditions[instrument_id]
        
        # Update volatility regime
        current_vol = self._get_current_volatility(instrument_id)
        historical_vol = self._get_historical_volatility(instrument_id)
        vol_percentile = self._calculate_volatility_percentile(current_vol)
        
        if vol_percentile < 0.25:
            condition.volatility_regime = "low"
        elif vol_percentile < 0.75:
            condition.volatility_regime = "normal"
        elif vol_percentile < 0.95:
            condition.volatility_regime = "high"
        else:
            condition.volatility_regime = "extreme"
        
        # Update trend strength
        condition.trend_strength = await self._calculate_trend_strength(instrument_id)
        
        # Update liquidity score
        condition.liquidity_score = await self._calculate_liquidity_score(instrument_id)
        
        # Update correlation level
        condition.correlation_level = await self._calculate_market_correlation()
        
        # Check for event risk
        condition.event_risk = await self._check_event_risk(instrument_id)
        
        # Anomaly detection
        if self._anomaly_detector and len(self._returns_history) > 50:
            features = self._extract_market_features(instrument_id)
            condition.anomaly_score = self._detect_anomalies(features)
    
    async def _kelly_position_size(
        self,
        instrument_id: InstrumentId,
        account_balance: Money,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        confidence_score: float,
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        # Get historical performance for this instrument
        outcomes = self._position_outcomes.get(instrument_id, [])
        
        if len(outcomes) < 20:  # Not enough data
            return 0.0
        
        # Calculate win rate and average win/loss
        wins = [o for o in outcomes if o > 0]
        losses = [abs(o) for o in outcomes if o < 0]
        
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / len(outcomes)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        # Kelly formula with confidence adjustment
        b = avg_win / avg_loss
        p = win_rate * confidence_score  # Adjust win rate by confidence
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        
        # Apply Kelly fraction with safety factor
        safety_factor = 0.25  # Use 25% of Kelly
        position_value = float(account_balance.as_decimal()) * kelly_fraction * safety_factor
        
        if stop_loss_price:
            risk_per_unit = abs(float(entry_price - stop_loss_price))
            if risk_per_unit > 0:
                return position_value / risk_per_unit
        
        # Fallback to value-based sizing
        return position_value / float(entry_price)
    
    async def _optimal_f_position_size(
        self,
        instrument_id: InstrumentId,
        account_balance: Money,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        confidence_score: float,
    ) -> float:
        """Calculate position size using Optimal f method."""
        outcomes = self._position_outcomes.get(instrument_id, [])
        
        if len(outcomes) < 30:
            return 0.0
        
        # Convert outcomes to returns
        returns = np.array(outcomes) / 100  # Assuming percentage returns
        
        # Find optimal f through optimization
        best_f = 0.0
        best_twr = 0.0  # Terminal Wealth Relative
        
        for f in np.linspace(0.01, 0.5, 50):
            twr = 1.0
            for ret in returns:
                holding_period_return = 1 + f * ret
                if holding_period_return <= 0:
                    twr = 0
                    break
                twr *= holding_period_return
            
            if twr > best_twr:
                best_twr = twr
                best_f = f
        
        # Adjust by confidence
        adjusted_f = best_f * confidence_score * 0.5  # Conservative application
        
        position_value = float(account_balance.as_decimal()) * adjusted_f
        
        if stop_loss_price:
            risk_per_unit = abs(float(entry_price - stop_loss_price))
            if risk_per_unit > 0:
                return position_value / risk_per_unit
        
        return position_value / float(entry_price)
    
    async def _risk_parity_position_size(
        self,
        instrument_id: InstrumentId,
        account_balance: Money,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        confidence_score: float,
    ) -> float:
        """Calculate position size using risk parity approach."""
        if not self.portfolio:
            return 0.0
        
        # Get all positions and their volatilities
        positions = self.portfolio.positions_open()
        position_vols = {}
        
        for pos in positions:
            vol = self._get_position_volatility(pos.instrument_id)
            position_vols[pos.instrument_id] = vol
        
        # Add potential new position
        new_vol = self._get_current_volatility(instrument_id)
        position_vols[instrument_id] = new_vol
        
        # Calculate risk parity weights
        total_inv_vol = sum(1/v for v in position_vols.values() if v > 0)
        
        if total_inv_vol == 0:
            return 0.0
        
        # Weight for new position
        if new_vol > 0:
            weight = (1 / new_vol) / total_inv_vol
        else:
            weight = 1 / len(position_vols)
        
        # Adjust by confidence
        weight *= confidence_score
        
        # Convert to position size
        position_value = float(account_balance.as_decimal()) * weight
        
        if stop_loss_price:
            risk_per_unit = abs(float(entry_price - stop_loss_price))
            if risk_per_unit > 0:
                return position_value / risk_per_unit
        
        return position_value / float(entry_price)
    
    async def _volatility_target_position_size(
        self,
        instrument_id: InstrumentId,
        account_balance: Money,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        confidence_score: float,
    ) -> float:
        """Calculate position size targeting specific volatility."""
        # Target portfolio volatility (annualized)
        target_vol = 0.12  # 12% target volatility
        
        # Get instrument volatility
        inst_vol = self._get_current_volatility(instrument_id)
        
        if inst_vol <= 0:
            return 0.0
        
        # Calculate base position size for target volatility
        position_fraction = target_vol / inst_vol
        
        # Adjust by confidence and market conditions
        condition = self._market_conditions.get(instrument_id)
        if condition:
            if condition.volatility_regime == "extreme":
                position_fraction *= 0.3
            elif condition.volatility_regime == "high":
                position_fraction *= 0.7
            elif condition.volatility_regime == "low":
                position_fraction *= 1.2
        
        # Apply confidence adjustment
        position_fraction *= confidence_score
        
        # Ensure reasonable bounds
        position_fraction = min(position_fraction, 0.2)  # Max 20% per position
        
        position_value = float(account_balance.as_decimal()) * position_fraction
        
        if stop_loss_price:
            risk_per_unit = abs(float(entry_price - stop_loss_price))
            if risk_per_unit > 0:
                return position_value / risk_per_unit
        
        return position_value / float(entry_price)
    
    async def _ml_position_size(
        self,
        instrument_id: InstrumentId,
        account_balance: Money,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        confidence_score: float,
    ) -> float:
        """Calculate position size using machine learning predictions."""
        if not self._risk_predictor:
            # Fallback to volatility targeting
            return await self._volatility_target_position_size(
                instrument_id, account_balance, entry_price, stop_loss_price, confidence_score
            )
        
        # Extract features for ML model
        features = await self._extract_ml_features(instrument_id)
        
        # Get risk prediction
        risk_prediction = self._risk_predictor.predict_risk(features)
        
        # Convert risk prediction to position size
        # Lower predicted risk = larger position
        risk_multiplier = 1.0 / (1.0 + risk_prediction)
        
        base_fraction = self.max_position_risk_percent / 100
        ml_fraction = base_fraction * risk_multiplier * confidence_score
        
        position_value = float(account_balance.as_decimal()) * ml_fraction
        
        if stop_loss_price:
            risk_per_unit = abs(float(entry_price - stop_loss_price))
            if risk_per_unit > 0:
                return position_value / risk_per_unit
        
        return position_value / float(entry_price)
    
    async def _apply_regime_adjustments(
        self,
        instrument_id: InstrumentId,
        base_size: float,
    ) -> float:
        """Apply market regime-specific adjustments to position size."""
        condition = self._market_conditions.get(instrument_id)
        if not condition:
            return base_size
        
        regime_limits = self._regime_specific_limits[condition.volatility_regime]
        
        # Apply regime multiplier
        adjusted_size = base_size * regime_limits["position_risk_multiplier"]
        
        # Further adjustments based on other conditions
        if condition.event_risk:
            adjusted_size *= 0.5  # Reduce size during events
        
        if condition.anomaly_score > 0.8:
            adjusted_size *= 0.3  # Significant reduction for anomalies
        elif condition.anomaly_score > 0.5:
            adjusted_size *= 0.7
        
        # Trend adjustments
        if abs(condition.trend_strength) > 0.7:
            # Strong trend - can increase size slightly
            adjusted_size *= 1.1
        
        # Liquidity adjustments
        adjusted_size *= condition.liquidity_score
        
        return adjusted_size
    
    async def _apply_portfolio_constraints(
        self,
        instrument_id: InstrumentId,
        size: float,
    ) -> float:
        """Apply portfolio-level constraints to position size."""
        if not self.portfolio:
            return size
        
        # Check concentration limits
        condition = self._market_conditions.get(instrument_id)
        if condition:
            regime_limits = self._regime_specific_limits[condition.volatility_regime]
            max_concentration = regime_limits["concentration_limit"]
        else:
            max_concentration = 0.12  # Default 12%
        
        # Calculate what percentage this position would be
        account_balance = self.portfolio.account_balance_total()
        position_value = size * 100  # Rough estimate
        position_percent = position_value / float(account_balance)
        
        if position_percent > max_concentration:
            size *= max_concentration / position_percent
        
        # Check number of positions
        open_positions = len(self.portfolio.positions_open())
        max_positions = regime_limits.get("max_positions", 12)
        
        if open_positions >= max_positions:
            return 0.0  # Don't open new positions
        
        # Check correlation constraints
        if await self._would_exceed_correlation_limit(instrument_id, size):
            size *= 0.5  # Reduce size for high correlation
        
        return size
    
    async def _apply_ml_adjustments(
        self,
        instrument_id: InstrumentId,
        size: float,
        confidence_score: float,
    ) -> float:
        """Apply machine learning-based adjustments."""
        if not self._risk_predictor:
            return size
        
        # Get ML risk score
        features = await self._extract_ml_features(instrument_id)
        risk_score = self._risk_predictor.predict_risk(features)
        
        # High risk score reduces position size
        if risk_score > 0.8:
            size *= 0.3
        elif risk_score > 0.6:
            size *= 0.7
        elif risk_score < 0.2:
            size *= 1.2  # Low risk, can increase slightly
        
        # Adjust by prediction confidence
        ml_confidence = self._risk_predictor.get_confidence()
        final_multiplier = 1.0 + (ml_confidence - 0.5) * 0.2
        
        return size * final_multiplier
    
    def _apply_final_safety_checks(
        self,
        instrument_id: InstrumentId,
        size: float,
        account_balance: Money,
    ) -> float:
        """Apply final safety checks to position size."""
        # Absolute maximum position size
        max_position_value = float(account_balance.as_decimal()) * 0.2  # 20% max
        max_units = max_position_value / 100  # Rough estimate
        
        size = min(size, max_units)
        
        # Minimum position size (avoid tiny positions)
        min_position_value = float(account_balance.as_decimal()) * 0.001  # 0.1% min
        min_units = min_position_value / 100  # Rough estimate
        
        if size < min_units:
            return 0.0
        
        # Check daily loss limit impact
        potential_loss = size * 0.02  # Assume 2% position loss
        if self._daily_pnl - potential_loss < -(self.max_daily_loss_percent / 100):
            # Would exceed daily loss limit
            return 0.0
        
        return size
    
    async def adapt_parameters(self) -> None:
        """Autonomously adapt risk parameters based on performance."""
        # Analyze recent performance
        recent_performance = self._analyze_recent_performance()
        
        # Adapt position risk limits
        if recent_performance["sharpe_ratio"] > 2.0:
            # Excellent performance - can increase risk slightly
            self._adapt_parameter(
                "max_position_risk_percent",
                self.max_position_risk_percent * 1.05,
                min_value=0.5,
                max_value=2.0,
            )
        elif recent_performance["sharpe_ratio"] < 0.5:
            # Poor performance - reduce risk
            self._adapt_parameter(
                "max_position_risk_percent",
                self.max_position_risk_percent * 0.95,
                min_value=0.5,
                max_value=2.0,
            )
        
        # Adapt portfolio risk limits based on drawdown
        if self._current_drawdown > 0.05:
            # In drawdown - reduce portfolio risk
            self._adapt_parameter(
                "max_portfolio_risk_percent",
                self.max_portfolio_risk_percent * 0.9,
                min_value=3.0,
                max_value=10.0,
            )
        elif self._current_drawdown < 0.01 and recent_performance["win_rate"] > 0.6:
            # Low drawdown, high win rate - can increase portfolio risk
            self._adapt_parameter(
                "max_portfolio_risk_percent",
                self.max_portfolio_risk_percent * 1.05,
                min_value=3.0,
                max_value=10.0,
            )
        
        # Adapt correlation limits based on market conditions
        avg_correlation = np.mean([c.correlation_level for c in self._market_conditions.values()])
        if avg_correlation > 0.7:
            # High market correlation - tighten limits
            self._adapt_parameter(
                "max_correlation",
                self.max_correlation * 0.95,
                min_value=0.5,
                max_value=0.8,
            )
        
        # Switch position sizing models based on performance
        await self._select_optimal_sizing_model()
        
        # Log adaptations
        self._log.info(
            f"Adapted parameters - Position risk: {self.max_position_risk_percent:.2f}%, "
            f"Portfolio risk: {self.max_portfolio_risk_percent:.2f}%, "
            f"Correlation limit: {self.max_correlation:.2f}, "
            f"Sizing model: {self._active_sizing_model}"
        )
    
    def _adapt_parameter(
        self,
        param_name: str,
        new_value: float,
        min_value: float,
        max_value: float,
    ) -> None:
        """Adapt a single parameter with constraints."""
        old_value = getattr(self, param_name)
        
        # Apply adaptation speed
        adapted_value = old_value + (new_value - old_value) * self.adaptation_speed
        
        # Apply constraints
        constrained_value = max(min_value, min(adapted_value, max_value))
        
        # Set new value
        setattr(self, param_name, constrained_value)
        
        # Log adaptation
        self._adaptation_log.append({
            "timestamp": datetime.utcnow(),
            "parameter": param_name,
            "old_value": old_value,
            "new_value": constrained_value,
            "reason": "performance_adaptation",
        })
    
    async def _select_optimal_sizing_model(self) -> None:
        """Select the best performing position sizing model."""
        if len(self._position_outcomes) < 100:
            return  # Not enough data
        
        # Test each model on recent data
        model_scores = {}
        
        for model_name in self._sizing_models:
            score = await self._evaluate_sizing_model(model_name)
            model_scores[model_name] = score
        
        # Select best model
        best_model = max(model_scores, key=model_scores.get)
        
        if best_model != self._active_sizing_model:
            self._log.info(
                f"Switching sizing model from {self._active_sizing_model} to {best_model} "
                f"(score: {model_scores[best_model]:.3f})"
            )
            self._active_sizing_model = best_model
    
    async def perform_stress_test(self) -> Dict[str, Any]:
        """Perform comprehensive stress testing on current portfolio."""
        if not self.portfolio:
            return {}
        
        positions = self.portfolio.positions_open()
        if not positions:
            return {"status": "no_positions"}
        
        # Generate stress scenarios
        scenarios = self._generate_stress_scenarios()
        
        results = {
            "timestamp": datetime.utcnow(),
            "scenario_results": [],
            "worst_case_loss": 0.0,
            "var_stress": 0.0,
            "cvar_stress": 0.0,
        }
        
        scenario_losses = []
        
        for scenario in scenarios:
            scenario_loss = 0.0
            
            for position in positions:
                # Apply scenario to position
                position_loss = self._calculate_position_stress_loss(
                    position, scenario
                )
                scenario_loss += position_loss
            
            scenario_losses.append(scenario_loss)
            
            results["scenario_results"].append({
                "scenario": scenario["name"],
                "loss": scenario_loss,
                "probability": scenario.get("probability", 1/len(scenarios)),
            })
        
        # Calculate stress VaR and CVaR
        sorted_losses = sorted(scenario_losses)
        var_index = int(self.var_confidence_level * len(sorted_losses))
        
        results["var_stress"] = sorted_losses[var_index] if var_index < len(sorted_losses) else 0
        results["cvar_stress"] = np.mean(sorted_losses[var_index:]) if var_index < len(sorted_losses) else 0
        results["worst_case_loss"] = max(scenario_losses) if scenario_losses else 0
        
        # Store results
        self._stress_test_results.append(results)
        
        # Take action if stress test shows excessive risk
        account_balance = self.portfolio.account_balance_total()
        max_acceptable_loss = float(account_balance) * (self.max_drawdown_percent / 100)
        
        if results["worst_case_loss"] > max_acceptable_loss:
            self._log.warning(
                f"Stress test shows excessive risk: worst case loss {results['worst_case_loss']:.2f} "
                f"exceeds limit {max_acceptable_loss:.2f}"
            )
            self._emergency_stop_active = True
        
        return results
    
    def _generate_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Generate stress test scenarios."""
        scenarios = []
        
        # Historical scenarios
        scenarios.extend([
            {
                "name": "2008 Financial Crisis",
                "market_drop": -0.40,
                "volatility_spike": 3.0,
                "correlation_increase": 0.9,
                "probability": 0.05,
            },
            {
                "name": "2020 COVID Crash",
                "market_drop": -0.35,
                "volatility_spike": 4.0,
                "correlation_increase": 0.95,
                "probability": 0.05,
            },
            {
                "name": "Flash Crash",
                "market_drop": -0.10,
                "volatility_spike": 5.0,
                "correlation_increase": 1.0,
                "probability": 0.02,
            },
        ])
        
        # Statistical scenarios
        for i in range(self.stress_test_scenarios - len(scenarios)):
            scenarios.append({
                "name": f"Statistical Scenario {i+1}",
                "market_drop": np.random.normal(-0.05, 0.02),
                "volatility_spike": np.random.lognormal(0.5, 0.5),
                "correlation_increase": np.random.beta(5, 2),
                "probability": 1/self.stress_test_scenarios,
            })
        
        return scenarios
    
    def _calculate_position_stress_loss(
        self,
        position: Any,
        scenario: Dict[str, Any],
    ) -> float:
        """Calculate position loss under stress scenario."""
        # Simplified stress loss calculation
        position_value = abs(float(position.quantity.as_decimal()) * float(position.avg_px_open))
        
        # Apply market drop
        base_loss = position_value * abs(scenario["market_drop"])
        
        # Amplify by volatility spike
        volatility_adjusted_loss = base_loss * scenario["volatility_spike"]
        
        # Further amplify for leveraged positions
        if hasattr(position, "leverage") and position.leverage > 1:
            volatility_adjusted_loss *= position.leverage
        
        return volatility_adjusted_loss
    
    async def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk analytics dashboard."""
        base_report = self.get_risk_report()
        
        # Add adaptive risk metrics
        adaptive_metrics = {
            "current_regime": self._get_dominant_regime(),
            "adaptation_count": len(self._adaptation_log),
            "active_sizing_model": self._active_sizing_model,
            "model_confidence": self._get_model_confidence(),
            "recent_adaptations": list(self._adaptation_log)[-5:],
        }
        
        # Add stress test results
        stress_metrics = {}
        if self._stress_test_results:
            latest_stress = self._stress_test_results[-1]
            stress_metrics = {
                "last_stress_test": latest_stress["timestamp"],
                "worst_case_loss": latest_stress["worst_case_loss"],
                "stress_var": latest_stress["var_stress"],
                "stress_cvar": latest_stress["cvar_stress"],
            }
        
        # Add ML predictions if available
        ml_metrics = {}
        if self._risk_predictor:
            ml_metrics = {
                "risk_forecast_24h": await self._get_risk_forecast(24),
                "risk_forecast_7d": await self._get_risk_forecast(168),
                "anomaly_detections": len([c for c in self._market_conditions.values() 
                                         if c.anomaly_score > 0.5]),
            }
        
        # Combine all metrics
        return {
            **base_report,
            "adaptive_metrics": adaptive_metrics,
            "stress_metrics": stress_metrics,
            "ml_metrics": ml_metrics,
            "regime_performance": dict(self._regime_performance),
        }
    
    # Helper methods
    
    def _get_historical_volatility(self, instrument_id: InstrumentId, periods: int = 20) -> float:
        """Calculate historical volatility."""
        # This would use actual price data
        # Placeholder implementation
        return 0.02
    
    def _calculate_volatility_percentile(self, current_vol: float) -> float:
        """Calculate volatility percentile."""
        if not self._volatility_history:
            return 0.5
        
        return stats.percentileofscore(self._volatility_history, current_vol) / 100
    
    async def _calculate_trend_strength(self, instrument_id: InstrumentId) -> float:
        """Calculate trend strength from -1 to 1."""
        # This would use price data and technical indicators
        # Placeholder implementation
        return np.random.uniform(-1, 1)
    
    async def _calculate_liquidity_score(self, instrument_id: InstrumentId) -> float:
        """Calculate liquidity score from 0 to 1."""
        # This would use order book data, spreads, volume
        # Placeholder implementation
        return 0.8
    
    async def _calculate_market_correlation(self) -> float:
        """Calculate average market correlation."""
        # This would use correlation matrix of all instruments
        # Placeholder implementation
        return 0.3
    
    async def _check_event_risk(self, instrument_id: InstrumentId) -> bool:
        """Check for upcoming events that could impact risk."""
        # This would check economic calendar, earnings, etc.
        # Placeholder implementation
        return False
    
    def _extract_market_features(self, instrument_id: InstrumentId) -> np.ndarray:
        """Extract features for anomaly detection."""
        # This would extract various market microstructure features
        # Placeholder implementation
        return np.random.randn(10)
    
    def _detect_anomalies(self, features: np.ndarray) -> float:
        """Detect market anomalies using ML."""
        if self._anomaly_detector and features is not None:
            # Return anomaly score
            score = self._anomaly_detector.decision_function([features])[0]
            # Normalize to 0-1 range
            return 1 / (1 + np.exp(-score))
        return 0.0
    
    def _get_position_volatility(self, instrument_id: InstrumentId) -> float:
        """Get position-specific volatility."""
        return self._get_current_volatility(instrument_id)
    
    async def _would_exceed_correlation_limit(
        self,
        instrument_id: InstrumentId,
        size: float,
    ) -> bool:
        """Check if adding position would exceed correlation limits."""
        # Simplified check
        current_correlation = self._get_average_correlation(
            instrument_id,
            self.portfolio.positions_open() if self.portfolio else []
        )
        return current_correlation > self.max_correlation
    
    async def _extract_ml_features(self, instrument_id: InstrumentId) -> np.ndarray:
        """Extract features for ML risk prediction."""
        # This would extract comprehensive features
        # Placeholder implementation
        return np.random.randn(50)
    
    def _get_model_confidence(self) -> float:
        """Get confidence in current sizing model."""
        # Based on recent performance
        if len(self._optimization_scores) < 10:
            return 0.5
        
        recent_scores = list(self._optimization_scores)[-10:]
        return np.mean(recent_scores)
    
    def _analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent trading performance."""
        if len(self._returns_history) < 20:
            return {
                "sharpe_ratio": 0.0,
                "win_rate": 0.5,
                "avg_return": 0.0,
            }
        
        returns = np.array(list(self._returns_history)[-100:])
        
        return {
            "sharpe_ratio": self._sharpe_ratio,
            "win_rate": self._win_rate,
            "avg_return": np.mean(returns),
        }
    
    async def _evaluate_sizing_model(self, model_name: str) -> float:
        """Evaluate performance of a sizing model."""
        # This would backtest the model on recent data
        # Placeholder implementation
        scores = {
            "kelly": 0.7,
            "optimal_f": 0.65,
            "risk_parity": 0.75,
            "volatility_targeting": 0.8,
            "machine_learning": 0.85,
        }
        return scores.get(model_name, 0.5)
    
    def _get_dominant_regime(self) -> str:
        """Get the dominant market regime."""
        if not self._market_conditions:
            return "unknown"
        
        regimes = [c.volatility_regime for c in self._market_conditions.values()]
        if regimes:
            return max(set(regimes), key=regimes.count)
        return "normal"
    
    async def _get_risk_forecast(self, hours: int) -> float:
        """Get ML risk forecast for specified hours ahead."""
        # This would use the ML predictor
        # Placeholder implementation
        return np.random.uniform(0.1, 0.9)
    
    def _log_adaptation(
        self,
        instrument_id: InstrumentId,
        base_size: float,
        final_size: float,
    ) -> None:
        """Log position size adaptation."""
        self._adaptation_log.append({
            "timestamp": datetime.utcnow(),
            "instrument": str(instrument_id),
            "base_size": base_size,
            "final_size": final_size,
            "adjustment_factor": final_size / base_size if base_size > 0 else 0,
            "regime": self._get_dominant_regime(),
            "model": self._active_sizing_model,
        })