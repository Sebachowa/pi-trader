
"""
Portfolio risk manager for multi-asset trading.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

import numpy as np
from nautilus_trader.common.component import Component
from nautilus_trader.common.logging import Logger
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.position import Position

from multi_asset_system.core.asset_interface import Asset, AssetClass
from multi_asset_system.core.asset_manager import MultiAssetManager


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    approved: bool
    reason: Optional[str] = None
    adjusted_quantity: Optional[Quantity] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-wide risk metrics."""
    total_value: Decimal
    total_exposure: Decimal
    cash_available: Decimal
    margin_used: Decimal
    margin_available: Decimal
    daily_pnl: Decimal
    daily_return: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    sharpe_ratio: Decimal
    var_95: Decimal  # Value at Risk
    concentration_ratio: Decimal
    correlation_risk: Decimal
    asset_class_exposure: Dict[AssetClass, Decimal]
    currency_exposure: Dict[Currency, Decimal]
    
    @property
    def leverage(self) -> Decimal:
        """Calculate current leverage."""
        if self.total_value > 0:
            return self.total_exposure / self.total_value
        return Decimal("0")
    
    @property
    def margin_utilization(self) -> Decimal:
        """Calculate margin utilization percentage."""
        total_margin = self.margin_used + self.margin_available
        if total_margin > 0:
            return self.margin_used / total_margin
        return Decimal("0")


class PortfolioRiskManager(Component):
    """
    Comprehensive risk manager for multi-asset portfolios.
    
    Features:
    - Cross-asset risk aggregation
    - Dynamic position limits
    - Correlation-based risk adjustment
    - Real-time exposure monitoring
    - Asset class diversification rules
    - Currency exposure management
    """
    
    def __init__(
        self,
        asset_manager: MultiAssetManager,
        logger: Logger,
        initial_capital: Decimal = Decimal("100000"),
        max_portfolio_leverage: Decimal = Decimal("3.0"),
        max_daily_loss_percent: Decimal = Decimal("0.03"),  # 3%
        max_drawdown_percent: Decimal = Decimal("0.15"),  # 15%
        max_concentration_percent: Decimal = Decimal("0.20"),  # 20% per position
        max_correlation_exposure: Decimal = Decimal("0.50"),  # 50% in correlated assets
        msgbus=None,
    ):
        super().__init__(
            logger=logger,
            component_id="PortfolioRiskManager",
            msgbus=msgbus,
        )
        
        self.asset_manager = asset_manager
        self.initial_capital = initial_capital
        self.max_portfolio_leverage = max_portfolio_leverage
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_drawdown_percent = max_drawdown_percent
        self.max_concentration_percent = max_concentration_percent
        self.max_correlation_exposure = max_correlation_exposure
        
        # Portfolio state
        self._cash_balances: Dict[Currency, Decimal] = {
            Currency.USD: initial_capital
        }
        self._positions: Dict[InstrumentId, Position] = {}
        self._daily_pnl = Decimal("0")
        self._peak_value = initial_capital
        self._daily_start_value = initial_capital
        self._last_reset = datetime.utcnow()
        
        # Risk tracking
        self._exposure_by_asset: Dict[InstrumentId, Decimal] = defaultdict(Decimal)
        self._exposure_by_class: Dict[AssetClass, Decimal] = defaultdict(Decimal)
        self._exposure_by_currency: Dict[Currency, Decimal] = defaultdict(Decimal)
        self._margin_requirements: Dict[InstrumentId, Decimal] = defaultdict(Decimal)
        
        # Historical data for risk calculations
        self._returns_history: List[Decimal] = []
        self._correlation_matrix: Optional[np.ndarray] = None
        self._volatility_estimates: Dict[InstrumentId, Decimal] = {}
        
        # Risk limits by asset class (can be customized)
        self._asset_class_limits = self._default_asset_class_limits()
    
    def _default_asset_class_limits(self) -> Dict[AssetClass, Dict[str, Decimal]]:
        """Default risk limits by asset class."""
        return {
            AssetClass.CRYPTO: {
                "max_allocation": Decimal("0.30"),  # 30% of portfolio
                "max_leverage": Decimal("2.0"),
                "position_limit": 10,
                "max_single_position": Decimal("0.10"),  # 10% per crypto
            },
            AssetClass.EQUITY: {
                "max_allocation": Decimal("0.60"),  # 60% of portfolio
                "max_leverage": Decimal("2.0"),  # Reg T margin
                "position_limit": 30,
                "max_single_position": Decimal("0.05"),  # 5% per stock
            },
            AssetClass.FOREX: {
                "max_allocation": Decimal("0.40"),
                "max_leverage": Decimal("10.0"),
                "position_limit": 10,
                "max_single_position": Decimal("0.15"),
            },
            AssetClass.COMMODITY: {
                "max_allocation": Decimal("0.25"),
                "max_leverage": Decimal("5.0"),
                "position_limit": 8,
                "max_single_position": Decimal("0.08"),
            },
        }
    
    async def check_order_risk(
        self,
        asset: Asset,
        order_side: OrderSide,
        quantity: Quantity,
        price: Optional[Price] = None,
    ) -> RiskCheckResult:
        """
        Comprehensive risk check for new orders.
        
        Checks:
        1. Position limits
        2. Concentration limits
        3. Leverage limits
        4. Daily loss limits
        5. Correlation exposure
        6. Asset class allocation
        """
        result = RiskCheckResult(approved=True)
        
        # Use last price if not provided
        if not price:
            price = asset._last_price
            if not price:
                return RiskCheckResult(approved=False, reason="No price available")
        
        # Calculate order value and required margin
        order_value = quantity * price
        margin_required = asset.calculate_margin_requirement(quantity, price, order_side == OrderSide.SELL)
        
        # Get current portfolio metrics
        metrics = await self.calculate_portfolio_metrics()
        
        # Check 1: Margin availability
        if margin_required > metrics.margin_available:
            return RiskCheckResult(
                approved=False,
                reason=f"Insufficient margin: required {margin_required}, available {metrics.margin_available}"
            )
        
        # Check 2: Daily loss limit
        if metrics.daily_return < -self.max_daily_loss_percent:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit exceeded: {metrics.daily_return:.2%}"
            )
        
        # Check 3: Portfolio leverage
        new_exposure = metrics.total_exposure + order_value
        new_leverage = new_exposure / metrics.total_value
        
        if new_leverage > self.max_portfolio_leverage:
            # Try to adjust quantity
            max_value = (self.max_portfolio_leverage * metrics.total_value) - metrics.total_exposure
            adjusted_qty = max_value / price
            
            if adjusted_qty < asset.trading_rules.min_order_size:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Would exceed max leverage of {self.max_portfolio_leverage}x"
                )
            
            result.adjusted_quantity = Quantity(adjusted_qty)
            result.warnings.append(f"Quantity adjusted from {quantity} to {adjusted_qty} for leverage limit")
        
        # Check 4: Position concentration
        existing_exposure = self._exposure_by_asset.get(asset.instrument_id, Decimal("0"))
        new_position_exposure = existing_exposure + order_value
        concentration = new_position_exposure / metrics.total_value
        
        if concentration > self.max_concentration_percent:
            max_position_value = metrics.total_value * self.max_concentration_percent
            remaining_capacity = max_position_value - existing_exposure
            
            if remaining_capacity <= 0:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Position would exceed concentration limit of {self.max_concentration_percent:.1%}"
                )
            
            adjusted_qty = remaining_capacity / price
            if adjusted_qty < asset.trading_rules.min_order_size:
                return RiskCheckResult(
                    approved=False,
                    reason="Position at concentration limit"
                )
            
            if not result.adjusted_quantity or adjusted_qty < result.adjusted_quantity:
                result.adjusted_quantity = Quantity(adjusted_qty)
                result.warnings.append("Quantity adjusted for concentration limit")
        
        # Check 5: Asset class limits
        class_limits = self._asset_class_limits.get(asset.asset_class, {})
        current_class_exposure = metrics.asset_class_exposure.get(asset.asset_class, Decimal("0"))
        new_class_exposure = current_class_exposure + order_value
        class_allocation = new_class_exposure / metrics.total_value
        
        max_class_allocation = class_limits.get("max_allocation", Decimal("1.0"))
        if class_allocation > max_class_allocation:
            result.warnings.append(
                f"{asset.asset_class.value} allocation would be {class_allocation:.1%} "
                f"(limit: {max_class_allocation:.1%})"
            )
        
        # Check 6: Correlation risk
        correlated_exposure = await self._calculate_correlation_exposure(asset, order_value)
        if correlated_exposure > self.max_correlation_exposure:
            result.warnings.append(
                f"High correlation exposure: {correlated_exposure:.1%} of portfolio"
            )
        
        # Check 7: Asset-specific risk parameters
        if hasattr(asset, 'risk_parameters'):
            # Check against asset's own risk limits
            if order_value > asset.risk_parameters.max_notional_value:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Exceeds asset max notional of {asset.risk_parameters.max_notional_value}"
                )
        
        # Final approval with any adjustments
        return result
    
    async def calculate_portfolio_metrics(self) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        total_cash = sum(self._cash_balances.values())
        total_exposure = Decimal("0")
        total_margin_used = Decimal("0")
        
        # Aggregate exposures
        for instrument_id, position in self._positions.items():
            if position.is_closed:
                continue
            
            asset = self.asset_manager.get_asset(instrument_id)
            if not asset or not asset._last_price:
                continue
            
            position_value = abs(position.quantity * asset._last_price)
            total_exposure += position_value
            
            # Update exposure tracking
            self._exposure_by_asset[instrument_id] = position_value
            self._exposure_by_class[asset.asset_class] += position_value
            self._exposure_by_currency[asset.quote_currency] += position_value
            
            # Calculate margin
            margin = asset.calculate_margin_requirement(
                abs(position.quantity),
                asset._last_price,
                position.is_short
            )
            total_margin_used += margin
            self._margin_requirements[instrument_id] = margin
        
        # Portfolio value
        total_value = total_cash + sum(p.unrealized_pnl for p in self._positions.values() if not p.is_closed)
        
        # Calculate returns
        daily_return = (total_value - self._daily_start_value) / self._daily_start_value if self._daily_start_value > 0 else Decimal("0")
        
        # Drawdown calculation
        if total_value > self._peak_value:
            self._peak_value = total_value
            current_drawdown = Decimal("0")
        else:
            current_drawdown = (self._peak_value - total_value) / self._peak_value
        
        # Calculate Sharpe ratio (simplified - would use proper returns history)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Value at Risk
        var_95 = self._calculate_var(confidence=0.95)
        
        # Concentration ratio (Herfindahl index)
        concentration = self._calculate_concentration_ratio()
        
        # Correlation risk
        correlation_risk = await self._calculate_portfolio_correlation_risk()
        
        return PortfolioRiskMetrics(
            total_value=total_value,
            total_exposure=total_exposure,
            cash_available=total_cash,
            margin_used=total_margin_used,
            margin_available=max(total_value - total_margin_used, Decimal("0")),
            daily_pnl=self._daily_pnl,
            daily_return=daily_return,
            max_drawdown=self.max_drawdown_percent,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            concentration_ratio=concentration,
            correlation_risk=correlation_risk,
            asset_class_exposure=dict(self._exposure_by_class),
            currency_exposure=dict(self._exposure_by_currency),
        )
    
    def _calculate_sharpe_ratio(self, risk_free_rate: Decimal = Decimal("0.02")) -> Decimal:
        """Calculate portfolio Sharpe ratio."""
        if len(self._returns_history) < 20:
            return Decimal("0")
        
        returns = np.array([float(r) for r in self._returns_history[-252:]])  # Last year
        
        if len(returns) == 0:
            return Decimal("0")
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return Decimal("0")
        
        # Annualized Sharpe
        sharpe = (avg_return - float(risk_free_rate) / 252) / std_return * np.sqrt(252)
        return Decimal(str(sharpe))
    
    def _calculate_var(self, confidence: float = 0.95) -> Decimal:
        """Calculate Value at Risk."""
        if len(self._returns_history) < 20:
            return Decimal("0")
        
        returns = np.array([float(r) for r in self._returns_history[-100:]])
        
        if len(returns) == 0:
            return Decimal("0")
        
        # Parametric VaR (assumes normal distribution)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        
        var = mean_return + z_score * std_return
        return Decimal(str(abs(var)))
    
    def _calculate_concentration_ratio(self) -> Decimal:
        """Calculate portfolio concentration (Herfindahl index)."""
        if not self._exposure_by_asset:
            return Decimal("0")
        
        total_exposure = sum(self._exposure_by_asset.values())
        if total_exposure == 0:
            return Decimal("0")
        
        # Sum of squared weights
        herfindahl = sum(
            (exposure / total_exposure) ** 2
            for exposure in self._exposure_by_asset.values()
        )
        
        return herfindahl
    
    async def _calculate_correlation_exposure(
        self,
        asset: Asset,
        additional_exposure: Decimal,
    ) -> Decimal:
        """Calculate exposure to correlated assets."""
        correlated_assets = self.asset_manager.get_correlated_assets(
            asset.instrument_id,
            threshold=Decimal("0.70")
        )
        
        correlated_exposure = additional_exposure
        
        for corr_asset, correlation in correlated_assets:
            if corr_asset.instrument_id in self._exposure_by_asset:
                # Weight by correlation strength
                correlated_exposure += self._exposure_by_asset[corr_asset.instrument_id] * correlation
        
        metrics = await self.calculate_portfolio_metrics()
        if metrics.total_value > 0:
            return correlated_exposure / metrics.total_value
        
        return Decimal("0")
    
    async def _calculate_portfolio_correlation_risk(self) -> Decimal:
        """Calculate overall portfolio correlation risk."""
        # Simplified - would use actual correlation matrix
        # High concentration in one asset class increases correlation risk
        
        if not self._exposure_by_class:
            return Decimal("0")
        
        total_exposure = sum(self._exposure_by_class.values())
        if total_exposure == 0:
            return Decimal("0")
        
        # Calculate entropy (diversity measure)
        entropy = Decimal("0")
        for exposure in self._exposure_by_class.values():
            if exposure > 0:
                weight = exposure / total_exposure
                entropy -= weight * weight.ln()  # Using natural log
        
        # Convert to risk score (0-1, where 1 is highest risk)
        # Low entropy = high concentration = high risk
        max_entropy = Decimal(str(len(AssetClass))).ln()
        if max_entropy > 0:
            diversity_score = entropy / max_entropy
            correlation_risk = 1 - diversity_score
        else:
            correlation_risk = Decimal("1")
        
        return correlation_risk
    
    def update_position(self, position: Position) -> None:
        """Update position in risk tracking."""
        self._positions[position.instrument_id] = position
        
        # Update daily P&L
        if position.realized_pnl:
            self._daily_pnl += position.realized_pnl
    
    def reset_daily_metrics(self) -> None:
        """Reset daily risk metrics (call at start of trading day)."""
        self._daily_pnl = Decimal("0")
        self._daily_start_value = sum(self._cash_balances.values())
        
        for position in self._positions.values():
            if not position.is_closed:
                self._daily_start_value += position.unrealized_pnl
        
        self._last_reset = datetime.utcnow()
        self._log.info(f"Reset daily metrics. Starting value: {self._daily_start_value}")
    
    async def get_portfolio_value(self) -> Decimal:
        """Get current portfolio value."""
        metrics = await self.calculate_portfolio_metrics()
        return metrics.total_value
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary for monitoring."""
        metrics = asyncio.run(self.calculate_portfolio_metrics())
        
        return {
            "portfolio_value": float(metrics.total_value),
            "total_exposure": float(metrics.total_exposure),
            "leverage": float(metrics.leverage),
            "margin_utilization": float(metrics.margin_utilization),
            "daily_pnl": float(metrics.daily_pnl),
            "daily_return": float(metrics.daily_return),
            "current_drawdown": float(metrics.current_drawdown),
            "sharpe_ratio": float(metrics.sharpe_ratio),
            "var_95": float(metrics.var_95),
            "concentration_ratio": float(metrics.concentration_ratio),
            "correlation_risk": float(metrics.correlation_risk),
            "asset_class_exposure": {
                k.value: float(v) for k, v in metrics.asset_class_exposure.items()
            },
            "warnings": self._generate_risk_warnings(metrics),
        }
    
    def _generate_risk_warnings(self, metrics: PortfolioRiskMetrics) -> List[str]:
        """Generate risk warnings based on current metrics."""
        warnings = []
        
        # Leverage warning
        if metrics.leverage > self.max_portfolio_leverage * Decimal("0.8"):
            warnings.append(f"High leverage: {metrics.leverage:.1f}x")
        
        # Drawdown warning
        if metrics.current_drawdown > self.max_drawdown_percent * Decimal("0.7"):
            warnings.append(f"Approaching max drawdown: {metrics.current_drawdown:.1%}")
        
        # Daily loss warning
        if metrics.daily_return < -self.max_daily_loss_percent * Decimal("0.7"):
            warnings.append(f"Approaching daily loss limit: {metrics.daily_return:.1%}")
        
        # Concentration warning
        if metrics.concentration_ratio > Decimal("0.15"):
            warnings.append(f"High concentration: {metrics.concentration_ratio:.2f}")
        
        # Margin warning
        if metrics.margin_utilization > Decimal("0.8"):
            warnings.append(f"High margin utilization: {metrics.margin_utilization:.1%}")
        
        return warnings