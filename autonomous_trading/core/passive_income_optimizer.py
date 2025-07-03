
"""
Passive Income Optimization System

Algorithms and strategies for maximizing sustainable passive income through
autonomous trading with focus on compounding, risk management, and long-term wealth building.
"""

import numpy as np
import pandas as pd
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
# from nautilus_trader.common.logging import Logger  # Not available in this version


@dataclass
class IncomeStream:
    """Represents a passive income stream."""
    source: str  # trading_profits, market_making, arbitrage, etc.
    amount: float
    frequency: str  # daily, weekly, monthly
    reliability: float  # 0-1 score
    risk_level: float  # 0-1 score
    capital_required: float
    current_yield: float  # Annual percentage yield
    

@dataclass 
class CompoundingPlan:
    """Plan for compounding returns."""
    initial_capital: float
    target_capital: float
    monthly_contribution: float
    reinvestment_rate: float  # Percentage of profits to reinvest
    withdrawal_rate: float  # Percentage of profits to withdraw
    time_horizon_months: int
    expected_monthly_return: float
    risk_tolerance: float
    

@dataclass
class WithdrawalSchedule:
    """Schedule for systematic withdrawals."""
    frequency: str  # monthly, quarterly, annually
    amount_type: str  # fixed, percentage, dynamic
    base_amount: float
    min_balance_required: float
    emergency_fund_months: int
    inflation_adjustment: bool
    

class PassiveIncomeOptimizer(Component):
    """
    Optimizer for maximizing passive income through autonomous trading.
    
    Features:
    - Multiple income stream optimization
    - Compound interest calculations
    - Sustainable withdrawal strategies
    - Risk-adjusted portfolio construction
    - Tax-efficient planning
    - Long-term wealth projection
    """
    
    def __init__(
        self,
        logger: Any,  # Logger type
        clock: LiveClock,
        msgbus: MessageBus,
        target_monthly_income: float = 1000.0,
        initial_capital: float = 10000.0,
        risk_tolerance: float = 0.5,  # 0=conservative, 1=aggressive
        compounding_enabled: bool = True,
        withdrawal_enabled: bool = True,
        tax_rate: float = 0.25,
        inflation_rate: float = 0.03,
    ):
        # Initialize component with minimal parameters
        try:
            super().__init__()
        except Exception:
            # If that fails, try with specific parameters
            pass
        
        self.clock = clock
        self.logger = logger
        self.msgbus = msgbus
        self._component_id = "PASSIVE-INCOME-OPTIMIZER"
        
        self.target_monthly_income = target_monthly_income
        self.initial_capital = initial_capital
        self.risk_tolerance = risk_tolerance
        self.compounding_enabled = compounding_enabled
        self.withdrawal_enabled = withdrawal_enabled
        self.tax_rate = tax_rate
        self.inflation_rate = inflation_rate
        
        # Income tracking
        self._income_streams: List[IncomeStream] = []
        self._income_history = deque(maxlen=365)  # Daily history
        self._monthly_income = deque(maxlen=60)  # 5 years monthly
        
        # Capital tracking
        self._capital_history = deque(maxlen=365)
        self._current_capital = initial_capital
        self._peak_capital = initial_capital
        self._total_withdrawn = 0.0
        self._total_reinvested = 0.0
        
        # Performance metrics
        self._monthly_returns = deque(maxlen=60)
        self._annual_returns = deque(maxlen=10)
        self._sharpe_ratios = deque(maxlen=12)
        self._max_drawdowns = deque(maxlen=12)
        
        # Optimization models
        self._portfolio_optimizer = None
        self._return_predictor = LinearRegression()
        self._risk_models = {}
        
        # Plans
        self._compounding_plan: Optional[CompoundingPlan] = None
        self._withdrawal_schedule: Optional[WithdrawalSchedule] = None
        self._optimization_history = deque(maxlen=100)
        
    def initialize_income_streams(self) -> None:
        """Initialize available income streams based on capital and risk tolerance."""
        self._income_streams = []
        
        # Trading profits stream
        self._income_streams.append(IncomeStream(
            source="trading_profits",
            amount=0.0,  # Variable
            frequency="daily",
            reliability=0.7 + (self.risk_tolerance * 0.2),
            risk_level=0.3 + (self.risk_tolerance * 0.4),
            capital_required=self.initial_capital * 0.4,
            current_yield=0.10 + (self.risk_tolerance * 0.05),  # 10-15% base
        ))
        
        # Market making income
        if self.initial_capital > 20000:
            self._income_streams.append(IncomeStream(
                source="market_making",
                amount=0.0,
                frequency="daily",
                reliability=0.8,
                risk_level=0.2,
                capital_required=self.initial_capital * 0.3,
                current_yield=0.08,  # 8% annual
            ))
        
        # Arbitrage opportunities
        if self.initial_capital > 50000:
            self._income_streams.append(IncomeStream(
                source="arbitrage",
                amount=0.0,
                frequency="weekly",
                reliability=0.6,
                risk_level=0.1,
                capital_required=self.initial_capital * 0.2,
                current_yield=0.06,  # 6% annual
            ))
        
        # Funding rate harvesting (crypto)
        if self.risk_tolerance > 0.3:
            self._income_streams.append(IncomeStream(
                source="funding_rates",
                amount=0.0,
                frequency="daily",
                reliability=0.5,
                risk_level=0.4,
                capital_required=self.initial_capital * 0.1,
                current_yield=0.12,  # 12% annual (variable)
            ))
        
        # Liquidity provision
        if self.initial_capital > 30000 and self.risk_tolerance > 0.4:
            self._income_streams.append(IncomeStream(
                source="liquidity_provision",
                amount=0.0,
                frequency="daily",
                reliability=0.7,
                risk_level=0.3,
                capital_required=self.initial_capital * 0.2,
                current_yield=0.15,  # 15% annual
            ))
        
        if hasattr(self, "logger") and self.logger:
            self.logger.info(f"Initialized {len(self._income_streams)} income streams")
        else:
            print("INFO: " + str(f"Initialized {len(self._income_streams)} income streams"))
    
    async def optimize_income_allocation(self) -> Dict[str, float]:
        """Optimize capital allocation across income streams."""
        if not self._income_streams:
            self.initialize_income_streams()
        
        # Prepare optimization problem
        n_streams = len(self._income_streams)
        
        # Objective: Maximize risk-adjusted income
        def objective(weights):
            expected_income = 0.0
            portfolio_risk = 0.0
            
            for i, stream in enumerate(self._income_streams):
                if weights[i] > 0:
                    # Expected income from this stream
                    stream_income = weights[i] * self._current_capital * stream.current_yield / 12
                    expected_income += stream_income * stream.reliability
                    
                    # Risk contribution
                    portfolio_risk += weights[i] ** 2 * stream.risk_level ** 2
            
            portfolio_risk = np.sqrt(portfolio_risk)
            
            # Risk-adjusted return (simplified Sharpe-like ratio)
            if portfolio_risk > 0:
                risk_adjusted_income = expected_income / portfolio_risk
            else:
                risk_adjusted_income = expected_income
            
            # Penalty for not meeting income target
            income_shortfall = max(0, self.target_monthly_income - expected_income)
            penalty = income_shortfall * 10
            
            return -(risk_adjusted_income - penalty)
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds (0 to 100% per stream)
        bounds = [(0, 1) for _ in range(n_streams)]
        
        # Additional constraints based on capital requirements
        for i, stream in enumerate(self._income_streams):
            max_weight = min(1.0, self._current_capital / stream.capital_required)
            bounds[i] = (0, max_weight)
        
        # Initial guess (equal weights)
        x0 = np.ones(n_streams) / n_streams
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )
        
        # Extract optimal allocation
        optimal_allocation = {}
        
        if result.success:
            for i, stream in enumerate(self._income_streams):
                optimal_allocation[stream.source] = result.x[i]
            
            # Calculate expected income
            expected_monthly = sum(
                self._current_capital * alloc * stream.current_yield / 12 * stream.reliability
                for stream, alloc in zip(self._income_streams, result.x)
            )
            
            if hasattr(self, "logger") and self.logger:

            
                self.logger.info(
                f"Optimized allocation - Expected monthly income: ${expected_monthly:.2f}"
            )

            
            else:

            
                print("INFO: " + str(
                f"Optimized allocation - Expected monthly income: ${expected_monthly:.2f}"
            ))
        else:
            # Fallback to equal allocation
            for stream in self._income_streams:
                optimal_allocation[stream.source] = 1.0 / n_streams
            
            if hasattr(self, "logger") and self.logger:

            
                self.logger.warning("Optimization failed, using equal allocation")

            
            else:

            
                print("WARNING: " + str("Optimization failed, using equal allocation"))
        
        # Store optimization result
        self._optimization_history.append({
            "timestamp": datetime.utcnow(),
            "allocation": optimal_allocation,
            "expected_income": expected_monthly if result.success else 0,
            "capital": self._current_capital,
        })
        
        return optimal_allocation
    
    def create_compounding_plan(
        self,
        target_capital: Optional[float] = None,
        time_horizon_months: int = 60,  # 5 years default
    ) -> CompoundingPlan:
        """Create optimal compounding plan to reach target capital."""
        if target_capital is None:
            # Default target: 10x initial capital
            target_capital = self.initial_capital * 10
        
        # Estimate monthly returns based on history
        if len(self._monthly_returns) > 6:
            avg_return = np.mean(self._monthly_returns)
            std_return = np.std(self._monthly_returns)
            # Conservative estimate (mean - 0.5 * std)
            expected_return = avg_return - 0.5 * std_return
        else:
            # Use income stream yields
            weighted_yield = sum(
                stream.current_yield * stream.reliability * 
                (stream.capital_required / self._current_capital)
                for stream in self._income_streams
                if stream.capital_required <= self._current_capital
            )
            expected_return = weighted_yield / 12  # Monthly
        
        # Calculate optimal reinvestment rate
        if self.withdrawal_enabled and self.target_monthly_income > 0:
            # Need to balance withdrawals with growth
            min_withdrawal = self.target_monthly_income
            expected_profit = self._current_capital * expected_return
            
            if expected_profit > min_withdrawal:
                reinvestment_rate = (expected_profit - min_withdrawal) / expected_profit
            else:
                reinvestment_rate = 0.0  # Can't meet withdrawal target
                if hasattr(self, "logger") and self.logger:

                    self.logger.warning(
                    f"Expected profit ${expected_profit:.2f} below target withdrawal ${min_withdrawal:.2f}"
                )

                else:

                    print("WARNING: " + str(
                    f"Expected profit ${expected_profit:.2f} below target withdrawal ${min_withdrawal:.2f}"
                ))
        else:
            # Full reinvestment for maximum growth
            reinvestment_rate = 1.0
        
        # Create plan
        plan = CompoundingPlan(
            initial_capital=self._current_capital,
            target_capital=target_capital,
            monthly_contribution=0.0,  # Could add external contributions
            reinvestment_rate=reinvestment_rate,
            withdrawal_rate=1.0 - reinvestment_rate,
            time_horizon_months=time_horizon_months,
            expected_monthly_return=expected_return,
            risk_tolerance=self.risk_tolerance,
        )
        
        self._compounding_plan = plan
        
        # Calculate projections
        projections = self._calculate_compound_projections(plan)
        
        if hasattr(self, "logger") and self.logger:

        
            self.logger.info(
            f"Compounding plan created - Target: ${target_capital:,.0f} in {time_horizon_months} months, "
            f"Reinvestment rate: {reinvestment_rate:.1%}"
        )

        
        else:

        
            print("INFO: " + str(
            f"Compounding plan created - Target: ${target_capital:,.0f} in {time_horizon_months} months, "
            f"Reinvestment rate: {reinvestment_rate:.1%}"
        ))
        
        return plan
    
    def _calculate_compound_projections(
        self,
        plan: CompoundingPlan,
    ) -> Dict[str, List[float]]:
        """Calculate compound growth projections."""
        projections = {
            "capital": [plan.initial_capital],
            "income": [0],
            "withdrawn": [0],
            "reinvested": [0],
        }
        
        capital = plan.initial_capital
        
        for month in range(plan.time_horizon_months):
            # Monthly return
            monthly_profit = capital * plan.expected_monthly_return
            
            # Split between reinvestment and withdrawal
            reinvested = monthly_profit * plan.reinvestment_rate
            withdrawn = monthly_profit * plan.withdrawal_rate
            
            # Update capital
            capital += reinvested + plan.monthly_contribution
            
            # Store projections
            projections["capital"].append(capital)
            projections["income"].append(monthly_profit)
            projections["withdrawn"].append(withdrawn)
            projections["reinvested"].append(reinvested)
        
        return projections
    
    def create_withdrawal_schedule(
        self,
        withdrawal_amount: Optional[float] = None,
        frequency: str = "monthly",
        preserve_capital: bool = True,
    ) -> WithdrawalSchedule:
        """Create sustainable withdrawal schedule."""
        if withdrawal_amount is None:
            withdrawal_amount = self.target_monthly_income
        
        # Calculate minimum balance to preserve
        if preserve_capital:
            # Need enough capital to generate target income
            avg_yield = np.mean([s.current_yield for s in self._income_streams])
            min_balance = withdrawal_amount * 12 / avg_yield
            
            # Add safety margin
            min_balance *= 1.5
        else:
            min_balance = self.initial_capital * 0.25  # Keep 25% minimum
        
        # Determine withdrawal type
        if self._current_capital * avg_yield / 12 > withdrawal_amount * 1.2:
            # Can use fixed withdrawals
            amount_type = "fixed"
        else:
            # Need percentage-based to preserve capital
            amount_type = "percentage"
            withdrawal_amount = withdrawal_amount / self._current_capital
        
        schedule = WithdrawalSchedule(
            frequency=frequency,
            amount_type=amount_type,
            base_amount=withdrawal_amount,
            min_balance_required=min_balance,
            emergency_fund_months=6,
            inflation_adjustment=True,
        )
        
        self._withdrawal_schedule = schedule
        
        if hasattr(self, "logger") and self.logger:

        
            self.logger.info(
            f"Withdrawal schedule created - {frequency} {amount_type} withdrawals, "
            f"Min balance: ${min_balance:,.0f}"
        )

        
        else:

        
            print("INFO: " + str(
            f"Withdrawal schedule created - {frequency} {amount_type} withdrawals, "
            f"Min balance: ${min_balance:,.0f}"
        ))
        
        return schedule
    
    async def calculate_sustainable_income(
        self,
        time_horizon_years: int = 30,
        inflation_adjusted: bool = True,
    ) -> Dict[str, float]:
        """Calculate sustainable passive income levels."""
        # Get current income potential
        allocation = await self.optimize_income_allocation()
        
        # Calculate base income
        base_monthly_income = sum(
            self._current_capital * alloc * stream.current_yield / 12 * stream.reliability
            for stream, alloc in zip(self._income_streams, allocation.values())
        )
        
        # Account for taxes
        after_tax_income = base_monthly_income * (1 - self.tax_rate)
        
        # Calculate sustainable withdrawal rate (SWR)
        # Using modified 4% rule adjusted for risk
        annual_swr = 0.04 - (self.risk_tolerance * 0.01)  # 3-4% based on risk
        monthly_swr = annual_swr / 12
        
        sustainable_withdrawal = self._current_capital * monthly_swr
        
        # Project future income with compounding
        future_income = {}
        capital = self._current_capital
        
        for year in [1, 5, 10, 20, 30]:
            if year <= time_horizon_years:
                # Compound capital
                for _ in range(year * 12):
                    monthly_return = capital * base_monthly_income / self._current_capital
                    reinvested = monthly_return * 0.7  # Reinvest 70%
                    capital += reinvested
                
                # Calculate income at this point
                projected_income = capital * monthly_swr
                
                # Adjust for inflation if requested
                if inflation_adjusted:
                    projected_income /= (1 + self.inflation_rate) ** year
                
                future_income[f"year_{year}"] = projected_income
        
        return {
            "current_monthly_potential": base_monthly_income,
            "after_tax_monthly": after_tax_income,
            "sustainable_withdrawal": sustainable_withdrawal,
            "required_capital_for_target": self.target_monthly_income / monthly_swr,
            "future_income": future_income,
            "years_to_target": self._calculate_years_to_target(
                base_monthly_income, self.target_monthly_income
            ),
        }
    
    def _calculate_years_to_target(
        self,
        current_income: float,
        target_income: float,
    ) -> float:
        """Calculate years needed to reach target income."""
        if current_income >= target_income:
            return 0.0
        
        if self._compounding_plan:
            # Use compound growth formula
            r = self._compounding_plan.expected_monthly_return
            reinvest_rate = self._compounding_plan.reinvestment_rate
            
            if r > 0 and reinvest_rate > 0:
                # Solve for time: target = current * (1 + r*reinvest)^n
                ratio = target_income / current_income
                months = np.log(ratio) / np.log(1 + r * reinvest_rate)
                return months / 12
        
        # Fallback calculation
        growth_rate = 0.08  # Assume 8% annual growth
        ratio = target_income / current_income
        return np.log(ratio) / np.log(1 + growth_rate)
    
    async def execute_withdrawal(self, override_amount: Optional[float] = None) -> float:
        """Execute withdrawal according to schedule."""
        if not self._withdrawal_schedule:
            if hasattr(self, "logger") and self.logger:

                self.logger.warning("No withdrawal schedule set")

            else:

                print("WARNING: " + str("No withdrawal schedule set"))
            return 0.0
        
        # Check if withdrawal is due
        # (This would integrate with actual scheduling)
        
        # Determine withdrawal amount
        if override_amount:
            amount = override_amount
        else:
            if self._withdrawal_schedule.amount_type == "fixed":
                amount = self._withdrawal_schedule.base_amount
            else:
                amount = self._current_capital * self._withdrawal_schedule.base_amount
        
        # Apply inflation adjustment
        if self._withdrawal_schedule.inflation_adjustment:
            # Calculate months since schedule created
            months_elapsed = 12  # Placeholder
            inflation_factor = (1 + self.inflation_rate / 12) ** months_elapsed
            amount *= inflation_factor
        
        # Check minimum balance
        if self._current_capital - amount < self._withdrawal_schedule.min_balance_required:
            if hasattr(self, "logger") and self.logger:

                self.logger.warning(
                f"Withdrawal would breach minimum balance "
                f"(${self._current_capital - amount:.2f} < ${self._withdrawal_schedule.min_balance_required:.2f})"
            )
            else:
                print("WARNING: " + str(
                f"Withdrawal would breach minimum balance "
                f"(${self._current_capital - amount:.2f} < ${self._withdrawal_schedule.min_balance_required:.2f})"
            ))
            # Reduce withdrawal to maintain minimum
            amount = max(0, self._current_capital - self._withdrawal_schedule.min_balance_required)
        
        # Execute withdrawal
        if amount > 0:
            self._current_capital -= amount
            self._total_withdrawn += amount
            
            if hasattr(self, "logger") and self.logger:

            
                self.logger.info(f"Withdrew ${amount:.2f}, New balance: ${self._current_capital:.2f}")

            
            else:

            
                print("INFO: " + str(f"Withdrew ${amount:.2f}, New balance: ${self._current_capital:.2f}"))
        
        return amount
    
    async def rebalance_income_streams(self) -> None:
        """Rebalance allocations based on performance."""
        # Update stream performance metrics
        for stream in self._income_streams:
            # This would use actual performance data
            # Update reliability and yield based on recent results
            pass
        
        # Re-optimize allocation
        new_allocation = await self.optimize_income_allocation()
        
        # Calculate rebalancing trades
        # (This would integrate with execution system)
        
        if hasattr(self, "logger") and self.logger:

        
            self.logger.info("Income streams rebalanced")

        
        else:

        
            print("INFO: " + str("Income streams rebalanced"))
    
    def calculate_tax_efficiency(self) -> Dict[str, float]:
        """Calculate tax-efficient withdrawal strategies."""
        gross_income = sum(self._monthly_income) if self._monthly_income else 0
        
        # Tax strategies
        strategies = {
            "standard": gross_income * self.tax_rate,
            "tax_loss_harvesting": gross_income * self.tax_rate * 0.8,  # 20% reduction
            "long_term_gains": gross_income * (self.tax_rate * 0.6),  # Lower LTCG rate
            "tax_deferred": 0,  # Defer until withdrawal
        }
        
        # Calculate effective rates
        effective_rates = {
            strategy: tax / gross_income if gross_income > 0 else 0
            for strategy, tax in strategies.items()
        }
        
        return {
            "gross_income": gross_income,
            "tax_strategies": strategies,
            "effective_rates": effective_rates,
            "recommended_strategy": min(strategies, key=strategies.get),
            "potential_savings": strategies["standard"] - min(strategies.values()),
        }
    
    def project_wealth_growth(
        self,
        years: int = 20,
        scenarios: List[str] = ["conservative", "moderate", "optimistic"],
    ) -> Dict[str, pd.DataFrame]:
        """Project long-term wealth growth under different scenarios."""
        projections = {}
        
        scenario_params = {
            "conservative": {
                "annual_return": 0.06,
                "volatility": 0.10,
                "withdrawal_rate": 0.03,
            },
            "moderate": {
                "annual_return": 0.10,
                "volatility": 0.15,
                "withdrawal_rate": 0.04,
            },
            "optimistic": {
                "annual_return": 0.15,
                "volatility": 0.25,
                "withdrawal_rate": 0.05,
            },
        }
        
        for scenario in scenarios:
            params = scenario_params[scenario]
            
            # Monte Carlo simulation
            n_simulations = 1000
            months = years * 12
            
            wealth_paths = np.zeros((n_simulations, months + 1))
            wealth_paths[:, 0] = self._current_capital
            
            for sim in range(n_simulations):
                capital = self._current_capital
                
                for month in range(months):
                    # Random return
                    monthly_return = np.random.normal(
                        params["annual_return"] / 12,
                        params["volatility"] / np.sqrt(12)
                    )
                    
                    # Growth
                    capital *= (1 + monthly_return)
                    
                    # Withdrawal
                    withdrawal = capital * params["withdrawal_rate"] / 12
                    capital -= withdrawal
                    
                    wealth_paths[sim, month + 1] = capital
            
            # Calculate statistics
            percentiles = [10, 25, 50, 75, 90]
            stats = pd.DataFrame({
                "month": range(months + 1),
                "mean": wealth_paths.mean(axis=0),
                **{
                    f"p{p}": np.percentile(wealth_paths, p, axis=0)
                    for p in percentiles
                },
            })
            
            projections[scenario] = stats
        
        return projections
    
    async def update_income_tracking(self, income_data: Dict[str, float]) -> None:
        """Update income tracking with actual results."""
        daily_total = sum(income_data.values())
        
        self._income_history.append({
            "date": datetime.utcnow(),
            "total": daily_total,
            "breakdown": income_data,
        })
        
        # Update capital
        if self.compounding_enabled and self._compounding_plan:
            reinvest_amount = daily_total * self._compounding_plan.reinvestment_rate
            self._current_capital += reinvest_amount
            self._total_reinvested += reinvest_amount
        
        # Update monthly tracking
        if len(self._income_history) > 0 and self._income_history[-1]["date"].day == 1:
            # New month - calculate previous month's total
            monthly_total = sum(
                h["total"] for h in self._income_history
                if h["date"].month == self._income_history[-2]["date"].month
            )
            self._monthly_income.append(monthly_total)
            
            # Calculate monthly return
            if self._capital_history:
                prev_capital = self._capital_history[-30] if len(self._capital_history) > 30 else self.initial_capital
                monthly_return = (self._current_capital - prev_capital) / prev_capital
                self._monthly_returns.append(monthly_return)
        
        # Update capital history
        self._capital_history.append(self._current_capital)
        
        # Update peak
        if self._current_capital > self._peak_capital:
            self._peak_capital = self._current_capital
    
    def get_income_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive income analytics dashboard."""
        # Current metrics
        current_metrics = {
            "current_capital": self._current_capital,
            "peak_capital": self._peak_capital,
            "total_withdrawn": self._total_withdrawn,
            "total_reinvested": self._total_reinvested,
        }
        
        # Income statistics
        if self._income_history:
            recent_income = [h["total"] for h in list(self._income_history)[-30:]]
            income_stats = {
                "daily_average": np.mean(recent_income),
                "daily_std": np.std(recent_income),
                "monthly_projection": np.mean(recent_income) * 30,
                "annual_projection": np.mean(recent_income) * 365,
            }
        else:
            income_stats = {
                "daily_average": 0,
                "daily_std": 0,
                "monthly_projection": 0,
                "annual_projection": 0,
            }
        
        # Performance metrics
        performance = {
            "total_return": (self._current_capital - self.initial_capital) / self.initial_capital,
            "monthly_returns": list(self._monthly_returns)[-12:] if self._monthly_returns else [],
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "max_drawdown": self._calculate_max_drawdown(),
        }
        
        # Passive income score (0-100)
        income_score = self._calculate_income_score()
        
        # Recommendations
        recommendations = self._generate_income_recommendations()
        
        return {
            "timestamp": datetime.utcnow(),
            "current_metrics": current_metrics,
            "income_statistics": income_stats,
            "performance": performance,
            "passive_income_score": income_score,
            "active_streams": len([s for s in self._income_streams if s.amount > 0]),
            "compounding_active": self.compounding_enabled,
            "withdrawal_active": self.withdrawal_enabled,
            "recommendations": recommendations,
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(self._monthly_returns) < 12:
            return 0.0
        
        returns = np.array(self._monthly_returns)
        excess_returns = returns - 0.002  # Assume 2.4% annual risk-free rate
        
        if np.std(excess_returns) > 0:
            return np.mean(excess_returns) * np.sqrt(12) / np.std(excess_returns)
        return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self._capital_history) < 2:
            return 0.0
        
        capital_array = np.array(self._capital_history)
        peak = np.maximum.accumulate(capital_array)
        drawdown = (peak - capital_array) / peak
        
        return np.max(drawdown)
    
    def _calculate_income_score(self) -> float:
        """Calculate passive income score (0-100)."""
        score = 0.0
        
        # Income reliability (25 points)
        if self._monthly_income:
            income_cv = np.std(self._monthly_income) / np.mean(self._monthly_income)
            reliability_score = max(0, 25 * (1 - income_cv))
            score += reliability_score
        
        # Income sufficiency (25 points)
        if income_stats := self.get_income_dashboard()["income_statistics"]:
            monthly_projection = income_stats["monthly_projection"]
            sufficiency_score = min(25, 25 * monthly_projection / self.target_monthly_income)
            score += sufficiency_score
        
        # Capital preservation (25 points)
        preservation_score = 25 * min(1, self._current_capital / self.initial_capital)
        score += preservation_score
        
        # Growth rate (25 points)
        if len(self._capital_history) > 30:
            monthly_growth = (self._current_capital / self._capital_history[-30]) ** (1/30) - 1
            annual_growth = (1 + monthly_growth) ** 12 - 1
            growth_score = min(25, 25 * annual_growth / 0.12)  # 12% annual target
            score += growth_score
        
        return min(100, max(0, score))
    
    def _generate_income_recommendations(self) -> List[str]:
        """Generate actionable recommendations for improving passive income."""
        recommendations = []
        
        # Check if meeting income target
        current_monthly = np.mean([h["total"] for h in list(self._income_history)[-30:]]
                                 ) * 30 if self._income_history else 0
        
        if current_monthly < self.target_monthly_income * 0.8:
            recommendations.append(
                f"Current income ${current_monthly:.0f} below target ${self.target_monthly_income:.0f}. "
                "Consider increasing capital or accepting higher risk strategies."
            )
        
        # Check diversification
        active_streams = sum(1 for s in self._income_streams if s.amount > 0)
        if active_streams < 3 and len(self._income_streams) > 3:
            recommendations.append(
                "Low diversification detected. Consider activating more income streams."
            )
        
        # Check compounding
        if not self.compounding_enabled and self._current_capital < self.initial_capital * 5:
            recommendations.append(
                "Enable compounding to accelerate wealth growth. Current capital only "
                f"{self._current_capital / self.initial_capital:.1f}x initial."
            )
        
        # Check withdrawal sustainability
        if self._withdrawal_schedule:
            withdrawal_rate = self._withdrawal_schedule.base_amount * 12 / self._current_capital
            if withdrawal_rate > 0.05:  # Above 5% annual
                recommendations.append(
                    f"High withdrawal rate ({withdrawal_rate:.1%} annual) may not be sustainable. "
                    "Consider reducing withdrawals or increasing capital."
                )
        
        # Performance improvement
        if hasattr(self, "_sharpe_ratio") and self._calculate_sharpe_ratio() < 1.0:
            recommendations.append(
                "Risk-adjusted returns below target (Sharpe < 1.0). "
                "Review strategy allocation and risk management."
            )
        
        return recommendations