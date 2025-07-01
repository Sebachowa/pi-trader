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
Autonomous Risk Controller - Dynamic risk management with self-adjusting parameters.
"""

import asyncio
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.logging import Logger
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.model.identifiers import InstrumentId, PositionId
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.portfolio.base import PortfolioFacade
from nautilus_trader.risk.sizing import FixedRiskSizer


class RiskLevel:
    """Dynamic risk level based on market conditions and performance."""
    
    def __init__(self, base_level: float = 0.02):
        self.base_level = base_level
        self.current_level = base_level
        self.min_level = base_level * 0.25  # 25% of base
        self.max_level = base_level * 2.0   # 200% of base
        self.adjustment_factor = 1.0


class RiskController(Component):
    """
    Autonomous risk management system with dynamic parameter adjustment.
    
    Features:
    - Dynamic position sizing based on volatility and performance
    - Correlation risk management
    - Drawdown protection with auto-reduction
    - VaR calculations and limits
    - Emergency stop mechanisms
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
        lookback_periods: int = 252,
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id="RISK-CONTROLLER",
            msgbus=msgbus,
        )
        
        self.portfolio = portfolio
        
        # Risk limits
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_drawdown_percent = max_drawdown_percent
        self.max_position_risk_percent = max_position_risk_percent
        self.max_portfolio_risk_percent = max_portfolio_risk_percent
        self.max_correlation = max_correlation
        self.var_confidence_level = var_confidence_level
        
        # Risk tracking
        self._daily_pnl = 0.0
        self._peak_balance = 0.0
        self._current_drawdown = 0.0
        self._position_risks: Dict[PositionId, float] = {}
        self._correlations: Dict[str, Dict[str, float]] = {}
        
        # Historical data for calculations
        self._returns_history = deque(maxlen=lookback_periods)
        self._volatility_history = deque(maxlen=lookback_periods)
        self._correlation_matrix = None
        
        # Dynamic risk levels
        self._risk_levels: Dict[InstrumentId, RiskLevel] = {}
        self._portfolio_risk_level = RiskLevel(base_level=max_portfolio_risk_percent / 100)
        
        # Performance metrics for adjustment
        self._win_rate = 0.5
        self._profit_factor = 1.0
        self._sharpe_ratio = 0.0
        self._consecutive_losses = 0
        self._consecutive_wins = 0
        
        # Emergency controls
        self._emergency_stop_active = False
        self._risk_reduction_mode = False
        self._blocked_instruments: Set[InstrumentId] = set()

    async def is_healthy(self) -> bool:
        """Check if risk controller is healthy."""
        return not self._emergency_stop_active and self._current_drawdown < self.max_drawdown_percent / 100

    def calculate_position_size(
        self,
        instrument_id: InstrumentId,
        account_balance: Money,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal] = None,
    ) -> Quantity:
        """Calculate dynamic position size based on current risk parameters."""
        
        # Check if trading is allowed
        if self._emergency_stop_active or instrument_id in self._blocked_instruments:
            return Quantity.zero()
        
        # Get dynamic risk level for instrument
        risk_level = self._get_dynamic_risk_level(instrument_id)
        
        # Apply portfolio-wide risk adjustments
        if self._risk_reduction_mode:
            risk_level.current_level *= 0.5  # Reduce risk by 50%
        
        # Calculate risk amount
        risk_amount = float(account_balance.as_decimal()) * risk_level.current_level
        
        # Calculate position size based on stop loss
        if stop_loss_price:
            risk_per_unit = abs(float(entry_price - stop_loss_price))
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                position_size = 0
        else:
            # Use ATR-based sizing if no stop loss provided
            atr = self._calculate_atr(instrument_id)
            if atr > 0:
                position_size = risk_amount / (atr * 2)  # 2 ATR stop
            else:
                # Fallback to fixed percentage
                position_size = risk_amount / (float(entry_price) * 0.02)
        
        # Apply position limits
        position_size = self._apply_position_limits(instrument_id, position_size)
        
        return Quantity.from_int(int(position_size))

    def _get_dynamic_risk_level(self, instrument_id: InstrumentId) -> RiskLevel:
        """Get dynamic risk level for an instrument."""
        if instrument_id not in self._risk_levels:
            self._risk_levels[instrument_id] = RiskLevel(
                base_level=self.max_position_risk_percent / 100
            )
        
        risk_level = self._risk_levels[instrument_id]
        
        # Adjust based on performance metrics
        performance_multiplier = self._calculate_performance_multiplier()
        
        # Adjust based on market volatility
        volatility_multiplier = self._calculate_volatility_multiplier(instrument_id)
        
        # Adjust based on correlation risk
        correlation_multiplier = self._calculate_correlation_multiplier(instrument_id)
        
        # Combine adjustments
        total_multiplier = performance_multiplier * volatility_multiplier * correlation_multiplier
        
        # Apply to risk level
        risk_level.current_level = risk_level.base_level * total_multiplier
        risk_level.current_level = max(risk_level.min_level, 
                                      min(risk_level.current_level, risk_level.max_level))
        
        return risk_level

    def _calculate_performance_multiplier(self) -> float:
        """Calculate risk adjustment based on recent performance."""
        multiplier = 1.0
        
        # Reduce risk after consecutive losses
        if self._consecutive_losses >= 3:
            multiplier *= 0.7
        elif self._consecutive_losses >= 2:
            multiplier *= 0.85
        
        # Increase risk after consecutive wins (carefully)
        if self._consecutive_wins >= 5 and self._profit_factor > 1.5:
            multiplier *= 1.2
        elif self._consecutive_wins >= 3 and self._profit_factor > 1.2:
            multiplier *= 1.1
        
        # Adjust based on Sharpe ratio
        if self._sharpe_ratio > 2.0:
            multiplier *= 1.15
        elif self._sharpe_ratio < 0.5:
            multiplier *= 0.8
        
        # Adjust based on win rate
        if self._win_rate > 0.6:
            multiplier *= 1.1
        elif self._win_rate < 0.4:
            multiplier *= 0.9
        
        return multiplier

    def _calculate_volatility_multiplier(self, instrument_id: InstrumentId) -> float:
        """Calculate risk adjustment based on market volatility."""
        current_volatility = self._get_current_volatility(instrument_id)
        average_volatility = self._get_average_volatility(instrument_id)
        
        if average_volatility == 0:
            return 1.0
        
        vol_ratio = current_volatility / average_volatility
        
        # Reduce risk in high volatility
        if vol_ratio > 1.5:
            return 0.7
        elif vol_ratio > 1.2:
            return 0.85
        # Increase risk in low volatility (carefully)
        elif vol_ratio < 0.8:
            return 1.1
        else:
            return 1.0

    def _calculate_correlation_multiplier(self, instrument_id: InstrumentId) -> float:
        """Calculate risk adjustment based on portfolio correlation."""
        if not self.portfolio:
            return 1.0
        
        # Get current positions
        positions = self.portfolio.positions_open()
        if not positions:
            return 1.0
        
        # Calculate average correlation with existing positions
        avg_correlation = self._get_average_correlation(instrument_id, positions)
        
        # Reduce risk for high correlation
        if avg_correlation > self.max_correlation:
            return 0.5
        elif avg_correlation > 0.5:
            return 0.8
        else:
            return 1.0

    def _apply_position_limits(self, instrument_id: InstrumentId, position_size: float) -> float:
        """Apply various position limits."""
        # Check daily loss limit
        if self._daily_pnl <= -(self.max_daily_loss_percent / 100):
            self._log.warning("Daily loss limit reached, blocking new positions")
            return 0
        
        # Check drawdown limit
        if self._current_drawdown >= (self.max_drawdown_percent / 100) * 0.8:
            self._log.warning("Approaching drawdown limit, reducing position size")
            position_size *= 0.5
        
        # Check portfolio exposure
        if self.portfolio:
            current_exposure = self._calculate_portfolio_exposure()
            if current_exposure > self.max_portfolio_risk_percent / 100:
                self._log.warning("Portfolio exposure limit reached")
                return 0
        
        return max(0, position_size)

    async def update_performance_metrics(self, trade_result: Dict[str, Any]) -> None:
        """Update performance metrics after a trade."""
        # Update win/loss tracking
        if trade_result.get("pnl", 0) > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
        
        # Update returns history
        self._returns_history.append(trade_result.get("return", 0))
        
        # Recalculate metrics
        await self._recalculate_metrics()

    async def _recalculate_metrics(self) -> None:
        """Recalculate performance and risk metrics."""
        if len(self._returns_history) < 10:
            return
        
        returns = np.array(self._returns_history)
        
        # Calculate win rate
        wins = np.sum(returns > 0)
        self._win_rate = wins / len(returns)
        
        # Calculate profit factor
        profits = returns[returns > 0]
        losses = abs(returns[returns < 0])
        if len(losses) > 0 and np.sum(losses) > 0:
            self._profit_factor = np.sum(profits) / np.sum(losses)
        
        # Calculate Sharpe ratio
        if np.std(returns) > 0:
            self._sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))

    def calculate_var(self, confidence_level: Optional[float] = None) -> float:
        """Calculate Value at Risk (VaR) for the portfolio."""
        if not self._returns_history:
            return 0.0
        
        confidence = confidence_level or self.var_confidence_level
        returns = np.array(self._returns_history)
        
        # Calculate historical VaR
        var_index = int((1 - confidence) * len(returns))
        sorted_returns = np.sort(returns)
        
        if var_index < len(sorted_returns):
            return -sorted_returns[var_index]
        else:
            return 0.0

    async def check_risk_limits(self) -> Dict[str, bool]:
        """Check all risk limits and return status."""
        limits_status = {
            "daily_loss": self._daily_pnl > -(self.max_daily_loss_percent / 100),
            "drawdown": self._current_drawdown < (self.max_drawdown_percent / 100),
            "portfolio_risk": True,
            "var_limit": True,
        }
        
        # Check portfolio risk
        if self.portfolio:
            portfolio_risk = self._calculate_portfolio_exposure()
            limits_status["portfolio_risk"] = portfolio_risk < (self.max_portfolio_risk_percent / 100)
        
        # Check VaR limit
        current_var = self.calculate_var()
        limits_status["var_limit"] = current_var < (self.max_portfolio_risk_percent / 100)
        
        # Activate emergency controls if needed
        if not all(limits_status.values()):
            await self._activate_emergency_controls(limits_status)
        
        return limits_status

    async def _activate_emergency_controls(self, limits_status: Dict[str, bool]) -> None:
        """Activate emergency risk controls."""
        self._log.warning(f"Risk limits breached: {limits_status}")
        
        if not limits_status["daily_loss"] or not limits_status["drawdown"]:
            self._emergency_stop_active = True
            await self.close_all_positions("Risk limits breached")
        else:
            self._risk_reduction_mode = True

    async def close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        if not self.portfolio:
            return
        
        self._log.warning(f"Closing all positions: {reason}")
        
        positions = self.portfolio.positions_open()
        for position in positions:
            # Emit close position command
            # This would integrate with the execution system
            pass

    def _calculate_portfolio_exposure(self) -> float:
        """Calculate current portfolio exposure as percentage."""
        if not self.portfolio:
            return 0.0
        
        total_exposure = 0.0
        account_balance = self.portfolio.account_balance_total()
        
        for position in self.portfolio.positions_open():
            position_value = abs(float(position.quantity.as_decimal()) * float(position.avg_px_open))
            total_exposure += position_value
        
        if account_balance > 0:
            return total_exposure / float(account_balance)
        return 0.0

    def _calculate_atr(self, instrument_id: InstrumentId, periods: int = 14) -> float:
        """Calculate Average True Range for position sizing."""
        # This would integrate with market data
        # Placeholder implementation
        return 0.01

    def _get_current_volatility(self, instrument_id: InstrumentId) -> float:
        """Get current volatility for an instrument."""
        # This would integrate with market data
        # Placeholder implementation
        return 0.02

    def _get_average_volatility(self, instrument_id: InstrumentId) -> float:
        """Get average historical volatility."""
        if self._volatility_history:
            return np.mean(self._volatility_history)
        return 0.02

    def _get_average_correlation(self, instrument_id: InstrumentId, positions: List[Position]) -> float:
        """Calculate average correlation with existing positions."""
        # This would use historical price data to calculate correlations
        # Placeholder implementation
        return 0.3

    async def update_drawdown(self, current_balance: float) -> None:
        """Update drawdown calculations."""
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
        
        if self._peak_balance > 0:
            self._current_drawdown = (self._peak_balance - current_balance) / self._peak_balance

    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "risk_status": {
                "emergency_stop": self._emergency_stop_active,
                "risk_reduction_mode": self._risk_reduction_mode,
                "current_drawdown": self._current_drawdown,
                "daily_pnl": self._daily_pnl,
            },
            "performance_metrics": {
                "win_rate": self._win_rate,
                "profit_factor": self._profit_factor,
                "sharpe_ratio": self._sharpe_ratio,
                "consecutive_wins": self._consecutive_wins,
                "consecutive_losses": self._consecutive_losses,
            },
            "risk_metrics": {
                "portfolio_exposure": self._calculate_portfolio_exposure(),
                "var_95": self.calculate_var(0.95),
                "var_99": self.calculate_var(0.99),
            },
            "dynamic_risk_levels": {
                str(k): v.current_level for k, v in self._risk_levels.items()
            },
            "blocked_instruments": list(self._blocked_instruments),
        }