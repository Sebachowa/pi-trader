"""
Base Strategy Interfaces - Flexible, pluggable components for trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import numpy as np
import pandas as pd

from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId, PositionId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import Order
from nautilus_trader.model.position import Position


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


class SignalStrength(Enum):
    """Signal strength levels."""
    VERY_STRONG = 1.0
    STRONG = 0.75
    MEDIUM = 0.5
    WEAK = 0.25
    VERY_WEAK = 0.1


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal_type: SignalType
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal should trigger action."""
        return self.strength >= 0.5 and self.confidence >= 0.6


@dataclass
class StrategyState:
    """Current state of a strategy."""
    is_active: bool
    position_count: int
    open_pnl: float
    realized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    current_drawdown: float
    max_drawdown: float
    last_update: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyComponent(ABC):
    """Base class for all strategy components."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self._performance_metrics = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def update(self, data: Any) -> Any:
        """Update component with new data."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset component to initial state."""
        pass
    
    def adapt_parameters(self, performance_data: Dict[str, Any]) -> None:
        """Adapt component parameters based on performance."""
        pass


class IndicatorSet(StrategyComponent):
    """Manages technical indicators for strategy."""
    
    def __init__(self, name: str = "indicator_set", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.indicators = {}
        self.indicator_values = {}
        self.update_callbacks = []
    
    def add_indicator(self, name: str, indicator: Any) -> None:
        """Add a technical indicator."""
        self.indicators[name] = indicator
        self.indicator_values[name] = None
    
    def remove_indicator(self, name: str) -> None:
        """Remove an indicator."""
        if name in self.indicators:
            del self.indicators[name]
            del self.indicator_values[name]
    
    async def initialize(self) -> None:
        """Initialize all indicators."""
        for name, indicator in self.indicators.items():
            if hasattr(indicator, 'reset'):
                indicator.reset()
        self.is_initialized = True
    
    async def update(self, data: Bar) -> Dict[str, float]:
        """Update all indicators with new bar data."""
        for name, indicator in self.indicators.items():
            if hasattr(indicator, 'update_raw'):
                # Nautilus Trader style
                indicator.update_raw(
                    data.high.as_double(),
                    data.low.as_double(),
                    data.close.as_double(),
                    data.volume.as_double()
                )
                if hasattr(indicator, 'value'):
                    self.indicator_values[name] = indicator.value
            elif hasattr(indicator, 'update'):
                # Generic indicator
                value = indicator.update(data)
                self.indicator_values[name] = value
        
        # Trigger callbacks
        for callback in self.update_callbacks:
            await callback(self.indicator_values)
        
        return self.indicator_values
    
    def get_state(self) -> Dict[str, Any]:
        """Get current indicator values."""
        return {
            "values": self.indicator_values.copy(),
            "indicator_count": len(self.indicators),
            "is_initialized": self.is_initialized
        }
    
    def reset(self) -> None:
        """Reset all indicators."""
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
        self.indicator_values.clear()


class SignalGenerator(StrategyComponent):
    """Generates trading signals from market data and indicators."""
    
    def __init__(self, name: str = "signal_generator", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.signal_rules = []
        self.signal_history = []
        self.signal_filters = []
        self.min_confidence = config.get('min_confidence', 0.6) if config else 0.6
    
    def add_rule(self, rule_func: Callable, weight: float = 1.0) -> None:
        """Add a signal generation rule."""
        self.signal_rules.append((rule_func, weight))
    
    def add_filter(self, filter_func: Callable) -> None:
        """Add a signal filter."""
        self.signal_filters.append(filter_func)
    
    async def initialize(self) -> None:
        """Initialize signal generator."""
        self.signal_history = []
        self.is_initialized = True
    
    async def update(self, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal from data."""
        signals = []
        total_weight = 0
        
        # Evaluate all rules
        for rule_func, weight in self.signal_rules:
            try:
                signal = await rule_func(data)
                if signal:
                    signals.append((signal, weight))
                    total_weight += weight
            except Exception as e:
                # Log error and continue
                pass
        
        if not signals:
            return None
        
        # Combine signals
        combined_signal = self._combine_signals(signals, total_weight)
        
        # Apply filters
        for filter_func in self.signal_filters:
            if not await filter_func(combined_signal, data):
                return None
        
        # Check confidence threshold
        if combined_signal.confidence >= self.min_confidence:
            self.signal_history.append(combined_signal)
            return combined_signal
        
        return None
    
    def _combine_signals(self, signals: List[Tuple[TradingSignal, float]], 
                        total_weight: float) -> TradingSignal:
        """Combine multiple signals into one."""
        if len(signals) == 1:
            return signals[0][0]
        
        # Weighted average of strength and confidence
        strength = sum(s[0].strength * s[1] for s in signals) / total_weight
        confidence = sum(s[0].confidence * s[1] for s in signals) / total_weight
        
        # Most common signal type (weighted)
        signal_types = {}
        for signal, weight in signals:
            signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + weight
        
        dominant_signal = max(signal_types, key=signal_types.get)
        
        # Combine metadata
        metadata = {}
        for signal, _ in signals:
            metadata.update(signal.metadata)
        metadata['combined_from'] = len(signals)
        
        return TradingSignal(
            signal_type=dominant_signal,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            source="combined",
            metadata=metadata
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get signal generator state."""
        return {
            "rule_count": len(self.signal_rules),
            "filter_count": len(self.signal_filters),
            "signal_count": len(self.signal_history),
            "recent_signals": self.signal_history[-10:] if self.signal_history else []
        }
    
    def reset(self) -> None:
        """Reset signal generator."""
        self.signal_history = []


class RiskManager(StrategyComponent):
    """Manages risk for trading strategy."""
    
    def __init__(self, name: str = "risk_manager", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.max_position_risk = config.get('max_position_risk', 0.02) if config else 0.02
        self.max_daily_risk = config.get('max_daily_risk', 0.06) if config else 0.06
        self.max_positions = config.get('max_positions', 5) if config else 5
        self.daily_loss = 0.0
        self.position_risks = {}
    
    async def initialize(self) -> None:
        """Initialize risk manager."""
        self.daily_loss = 0.0
        self.position_risks = {}
        self.is_initialized = True
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update risk metrics."""
        # Update position risks
        if 'positions' in data:
            for position in data['positions']:
                self._update_position_risk(position)
        
        # Update daily loss
        if 'realized_pnl' in data:
            self.daily_loss = min(0, data['realized_pnl'])
        
        return self.get_risk_metrics()
    
    def can_take_trade(self, proposed_risk: float) -> bool:
        """Check if new trade is within risk limits."""
        total_risk = sum(self.position_risks.values()) + proposed_risk
        
        # Check position risk limit
        if proposed_risk > self.max_position_risk:
            return False
        
        # Check daily risk limit
        if abs(self.daily_loss) + total_risk > self.max_daily_risk:
            return False
        
        # Check position count limit
        if len(self.position_risks) >= self.max_positions:
            return False
        
        return True
    
    def calculate_stop_loss(self, entry_price: float, position_size: float, 
                           account_balance: float) -> float:
        """Calculate stop loss based on risk parameters."""
        risk_amount = account_balance * self.max_position_risk
        price_risk = risk_amount / position_size
        
        return price_risk
    
    def _update_position_risk(self, position: Position) -> None:
        """Update risk for a position."""
        if position.is_open:
            # Calculate current risk
            entry_price = position.avg_px_open
            current_price = position.last_px
            position_value = abs(position.quantity * current_price)
            
            if position.side == OrderSide.BUY:
                risk = (entry_price - current_price) / entry_price
            else:
                risk = (current_price - entry_price) / entry_price
            
            self.position_risks[position.id] = max(0, risk)
        else:
            # Remove closed position
            self.position_risks.pop(position.id, None)
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        return {
            "total_risk": sum(self.position_risks.values()),
            "daily_loss": self.daily_loss,
            "position_count": len(self.position_risks),
            "can_trade": self.can_take_trade(0),
            "risk_utilization": sum(self.position_risks.values()) / self.max_daily_risk
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get risk manager state."""
        return {
            "risk_metrics": self.get_risk_metrics(),
            "position_risks": dict(self.position_risks),
            "config": {
                "max_position_risk": self.max_position_risk,
                "max_daily_risk": self.max_daily_risk,
                "max_positions": self.max_positions
            }
        }
    
    def reset(self) -> None:
        """Reset risk manager."""
        self.daily_loss = 0.0
        self.position_risks = {}
    
    def adapt_parameters(self, performance_data: Dict[str, Any]) -> None:
        """Adapt risk parameters based on performance."""
        # Reduce risk after losses
        if 'consecutive_losses' in performance_data:
            losses = performance_data['consecutive_losses']
            if losses > 3:
                self.max_position_risk *= 0.8
            elif losses > 5:
                self.max_position_risk *= 0.6
        
        # Increase risk after wins (carefully)
        if 'consecutive_wins' in performance_data:
            wins = performance_data['consecutive_wins']
            if wins > 5 and 'sharpe_ratio' in performance_data:
                if performance_data['sharpe_ratio'] > 2.0:
                    self.max_position_risk = min(
                        self.max_position_risk * 1.1,
                        self.config.get('max_position_risk', 0.02) * 1.5
                    )


class PositionSizer(StrategyComponent):
    """Calculates optimal position sizes."""
    
    def __init__(self, name: str = "position_sizer", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.sizing_method = config.get('method', 'fixed') if config else 'fixed'
        self.base_size = config.get('base_size', 1.0) if config else 1.0
        self.use_kelly = config.get('use_kelly', True) if config else True
        self.kelly_fraction = 0.25  # Conservative Kelly
        self.performance_history = []
    
    async def initialize(self) -> None:
        """Initialize position sizer."""
        self.performance_history = []
        self.is_initialized = True
    
    async def update(self, data: Dict[str, Any]) -> float:
        """Calculate position size."""
        if 'performance' in data:
            self.performance_history.append(data['performance'])
        
        if self.sizing_method == 'fixed':
            return self.base_size
        elif self.sizing_method == 'volatility':
            return self._volatility_sizing(data)
        elif self.sizing_method == 'kelly':
            return self._kelly_sizing(data)
        elif self.sizing_method == 'risk_parity':
            return self._risk_parity_sizing(data)
        else:
            return self.base_size
    
    def _volatility_sizing(self, data: Dict[str, Any]) -> float:
        """Size based on volatility."""
        if 'volatility' not in data:
            return self.base_size
        
        # Inverse volatility sizing
        target_vol = 0.15  # 15% annual volatility target
        current_vol = data['volatility']
        
        if current_vol > 0:
            size = self.base_size * (target_vol / current_vol)
            return max(0.1, min(size, 2.0))  # Cap between 0.1x and 2x
        
        return self.base_size
    
    def _kelly_sizing(self, data: Dict[str, Any]) -> float:
        """Kelly Criterion sizing."""
        if len(self.performance_history) < 20:
            return self.base_size * 0.5  # Start conservative
        
        # Calculate win rate and win/loss ratio
        wins = [p for p in self.performance_history if p > 0]
        losses = [p for p in self.performance_history if p < 0]
        
        if not wins or not losses:
            return self.base_size * 0.5
        
        win_rate = len(wins) / len(self.performance_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        if avg_loss > 0:
            b = avg_win / avg_loss
            q = 1 - win_rate
            kelly = (win_rate * b - q) / b
            
            # Apply fraction and constraints
            kelly = max(0, min(kelly, 1.0)) * self.kelly_fraction
            
            return self.base_size * (1 + kelly)
        
        return self.base_size
    
    def _risk_parity_sizing(self, data: Dict[str, Any]) -> float:
        """Risk parity sizing."""
        if 'portfolio_assets' not in data:
            return self.base_size
        
        # Calculate equal risk contribution
        assets = data['portfolio_assets']
        volatilities = [asset.get('volatility', 0.15) for asset in assets]
        
        if volatilities and volatilities[0] > 0:
            # Size inversely proportional to volatility
            total_inv_vol = sum(1/v for v in volatilities if v > 0)
            weight = (1/volatilities[0]) / total_inv_vol
            
            return self.base_size * weight
        
        return self.base_size
    
    def get_state(self) -> Dict[str, Any]:
        """Get position sizer state."""
        recent_performance = self.performance_history[-20:] if self.performance_history else []
        
        return {
            "sizing_method": self.sizing_method,
            "current_kelly": self._calculate_current_kelly() if self.use_kelly else None,
            "performance_count": len(self.performance_history),
            "recent_performance": recent_performance
        }
    
    def _calculate_current_kelly(self) -> float:
        """Calculate current Kelly fraction."""
        if len(self.performance_history) < 20:
            return 0.0
        
        wins = [p for p in self.performance_history if p > 0]
        losses = [p for p in self.performance_history if p < 0]
        
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / len(self.performance_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss > 0:
            b = avg_win / avg_loss
            q = 1 - win_rate
            kelly = (win_rate * b - q) / b
            return max(0, min(kelly, 1.0))
        
        return 0.0
    
    def reset(self) -> None:
        """Reset position sizer."""
        self.performance_history = []


class OrderExecutor(StrategyComponent):
    """Handles order execution logic."""
    
    def __init__(self, name: str = "order_executor", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.execution_style = config.get('style', 'aggressive') if config else 'aggressive'
        self.use_limit_orders = config.get('use_limit_orders', False) if config else False
        self.slippage_model = config.get('slippage_model', 'linear') if config else 'linear'
        self.pending_orders = {}
        self.executed_orders = []
    
    async def initialize(self) -> None:
        """Initialize order executor."""
        self.pending_orders = {}
        self.executed_orders = []
        self.is_initialized = True
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update order execution state."""
        # Check pending orders
        if 'market_data' in data:
            await self._check_pending_orders(data['market_data'])
        
        return {
            "pending_count": len(self.pending_orders),
            "executed_count": len(self.executed_orders)
        }
    
    async def execute_signal(self, signal: TradingSignal, market_data: Dict[str, Any]) -> Order:
        """Execute a trading signal."""
        if self.execution_style == 'aggressive':
            return await self._aggressive_execution(signal, market_data)
        elif self.execution_style == 'passive':
            return await self._passive_execution(signal, market_data)
        elif self.execution_style == 'smart':
            return await self._smart_execution(signal, market_data)
        else:
            return await self._aggressive_execution(signal, market_data)
    
    async def _aggressive_execution(self, signal: TradingSignal, 
                                   market_data: Dict[str, Any]) -> Order:
        """Aggressive market order execution."""
        # Create market order
        order = self._create_market_order(signal, market_data)
        
        # Add expected slippage
        expected_slippage = self._calculate_slippage(
            order.quantity, 
            market_data.get('liquidity', 'medium')
        )
        
        order.metadata['expected_slippage'] = expected_slippage
        
        return order
    
    async def _passive_execution(self, signal: TradingSignal, 
                                market_data: Dict[str, Any]) -> Order:
        """Passive limit order execution."""
        # Create limit order at favorable price
        if signal.signal_type in [SignalType.BUY, SignalType.SCALE_IN]:
            # Place below current price
            limit_price = market_data['bid'] * (1 - 0.0005)  # 5 bps below bid
        else:
            # Place above current price
            limit_price = market_data['ask'] * (1 + 0.0005)  # 5 bps above ask
        
        order = self._create_limit_order(signal, market_data, limit_price)
        self.pending_orders[order.client_order_id] = order
        
        return order
    
    async def _smart_execution(self, signal: TradingSignal, 
                              market_data: Dict[str, Any]) -> Order:
        """Smart routing with dynamic execution."""
        spread = market_data.get('spread', 0.001)
        liquidity = market_data.get('liquidity', 'medium')
        urgency = signal.strength
        
        # Decide execution method based on conditions
        if urgency > 0.8 or spread < 0.0005:
            # High urgency or tight spread - use market order
            return await self._aggressive_execution(signal, market_data)
        elif liquidity == 'high' and spread < 0.001:
            # Good liquidity and reasonable spread - use limit order
            return await self._passive_execution(signal, market_data)
        else:
            # Split order or use iceberg
            return await self._split_execution(signal, market_data)
    
    async def _split_execution(self, signal: TradingSignal, 
                              market_data: Dict[str, Any]) -> Order:
        """Split large orders into smaller chunks."""
        # Implementation for order splitting
        # This is simplified - real implementation would be more complex
        return await self._aggressive_execution(signal, market_data)
    
    def _create_market_order(self, signal: TradingSignal, 
                            market_data: Dict[str, Any]) -> Order:
        """Create a market order."""
        # This is a placeholder - actual implementation would use Nautilus order factory
        order = {
            'type': OrderType.MARKET,
            'side': OrderSide.BUY if signal.signal_type in [SignalType.BUY, SignalType.SCALE_IN] else OrderSide.SELL,
            'quantity': market_data.get('position_size', 1.0),
            'metadata': {
                'signal': signal,
                'expected_price': market_data.get('mid_price', 0)
            }
        }
        return order
    
    def _create_limit_order(self, signal: TradingSignal, market_data: Dict[str, Any], 
                           limit_price: float) -> Order:
        """Create a limit order."""
        # This is a placeholder - actual implementation would use Nautilus order factory
        order = {
            'type': OrderType.LIMIT,
            'side': OrderSide.BUY if signal.signal_type in [SignalType.BUY, SignalType.SCALE_IN] else OrderSide.SELL,
            'quantity': market_data.get('position_size', 1.0),
            'price': limit_price,
            'metadata': {
                'signal': signal,
                'created_at': datetime.utcnow()
            }
        }
        return order
    
    def _calculate_slippage(self, quantity: float, liquidity: str) -> float:
        """Calculate expected slippage."""
        if self.slippage_model == 'linear':
            base_slippage = {
                'high': 0.0001,
                'medium': 0.0005,
                'low': 0.002
            }.get(liquidity, 0.001)
            
            # Scale with quantity
            return base_slippage * (1 + quantity / 1000)
        
        return 0.0005  # Default 5 bps
    
    async def _check_pending_orders(self, market_data: Dict[str, Any]) -> None:
        """Check and manage pending orders."""
        current_time = datetime.utcnow()
        
        for order_id, order in list(self.pending_orders.items()):
            # Check if order should be cancelled (timeout)
            order_age = (current_time - order['metadata']['created_at']).seconds
            
            if order_age > 300:  # 5 minute timeout
                del self.pending_orders[order_id]
                # Would cancel order here
    
    def get_state(self) -> Dict[str, Any]:
        """Get order executor state."""
        return {
            "execution_style": self.execution_style,
            "pending_orders": len(self.pending_orders),
            "executed_orders": len(self.executed_orders[-100:]),
            "use_limit_orders": self.use_limit_orders
        }
    
    def reset(self) -> None:
        """Reset order executor."""
        self.pending_orders = {}
        self.executed_orders = []


class PerformanceTracker(StrategyComponent):
    """Tracks and analyzes strategy performance."""
    
    def __init__(self, name: str = "performance_tracker", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.metrics_window = config.get('metrics_window', 100) if config else 100
        self.trades = []
        self.equity_curve = []
        self.metrics_cache = {}
        self.benchmark_returns = []
    
    async def initialize(self) -> None:
        """Initialize performance tracker."""
        self.trades = []
        self.equity_curve = []
        self.metrics_cache = {}
        self.is_initialized = True
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Update performance metrics."""
        if 'trade' in data:
            self.trades.append(data['trade'])
        
        if 'equity' in data:
            self.equity_curve.append({
                'timestamp': datetime.utcnow(),
                'equity': data['equity'],
                'drawdown': self._calculate_drawdown(data['equity'])
            })
        
        # Recalculate metrics
        self.metrics_cache = self._calculate_metrics()
        
        return self.metrics_cache
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(self.trades) < 2:
            return {}
        
        metrics = {}
        
        # Basic metrics
        returns = [t.get('return', 0) for t in self.trades[-self.metrics_window:]]
        metrics['total_trades'] = len(self.trades)
        metrics['win_rate'] = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        
        # Return metrics
        if returns:
            metrics['avg_return'] = np.mean(returns)
            metrics['total_return'] = np.sum(returns)
            metrics['volatility'] = np.std(returns) if len(returns) > 1 else 0
            
            # Sharpe ratio (assuming daily returns)
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['volatility'] * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
        
        # Risk metrics
        if self.equity_curve:
            equities = [e['equity'] for e in self.equity_curve]
            metrics['max_drawdown'] = self._calculate_max_drawdown(equities)
            metrics['current_drawdown'] = self.equity_curve[-1]['drawdown']
            
            # Calmar ratio
            if metrics['max_drawdown'] > 0 and len(self.equity_curve) > 252:
                annual_return = (equities[-1] / equities[-252] - 1) if equities[-252] > 0 else 0
                metrics['calmar_ratio'] = annual_return / metrics['max_drawdown']
        
        # Win/Loss analysis
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if wins:
            metrics['avg_win'] = np.mean(wins)
            metrics['max_win'] = np.max(wins)
        
        if losses:
            metrics['avg_loss'] = np.mean(losses)
            metrics['max_loss'] = np.min(losses)
            
            # Profit factor
            if wins:
                metrics['profit_factor'] = sum(wins) / abs(sum(losses))
        
        # Consecutive wins/losses
        metrics['consecutive_wins'] = self._count_consecutive(returns, positive=True)
        metrics['consecutive_losses'] = self._count_consecutive(returns, positive=False)
        
        # Recovery metrics
        if 'max_drawdown' in metrics and metrics['max_drawdown'] > 0:
            metrics['recovery_factor'] = metrics.get('total_return', 0) / metrics['max_drawdown']
        
        # Information ratio (if benchmark available)
        if self.benchmark_returns:
            excess_returns = [
                r - b for r, b in zip(returns[-len(self.benchmark_returns):], self.benchmark_returns)
            ]
            if len(excess_returns) > 1:
                tracking_error = np.std(excess_returns)
                if tracking_error > 0:
                    metrics['information_ratio'] = np.mean(excess_returns) / tracking_error * np.sqrt(252)
        
        return metrics
    
    def _calculate_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown."""
        if not self.equity_curve:
            return 0.0
        
        peak = max(e['equity'] for e in self.equity_curve)
        return (peak - current_equity) / peak if peak > 0 else 0.0
    
    def _calculate_max_drawdown(self, equities: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equities:
            return 0.0
        
        peak = equities[0]
        max_dd = 0.0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            else:
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _count_consecutive(self, returns: List[float], positive: bool) -> int:
        """Count consecutive wins or losses."""
        if not returns:
            return 0
        
        count = 0
        for r in reversed(returns):
            if (positive and r > 0) or (not positive and r < 0):
                count += 1
            else:
                break
        
        return count
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = self.metrics_cache.copy()
        
        # Add additional analysis
        if self.trades:
            # Trade duration analysis
            durations = [t.get('duration_seconds', 0) for t in self.trades if 'duration_seconds' in t]
            if durations:
                metrics['avg_trade_duration'] = np.mean(durations)
                metrics['median_trade_duration'] = np.median(durations)
        
        # Monthly returns
        if self.equity_curve and len(self.equity_curve) > 30:
            monthly_returns = self._calculate_monthly_returns()
            metrics['monthly_returns'] = monthly_returns
            metrics['best_month'] = max(monthly_returns.values()) if monthly_returns else 0
            metrics['worst_month'] = min(monthly_returns.values()) if monthly_returns else 0
        
        return metrics
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate returns by month."""
        monthly_returns = {}
        
        # Group equity curve by month
        current_month = None
        month_start_equity = None
        
        for point in self.equity_curve:
            month_key = point['timestamp'].strftime('%Y-%m')
            
            if month_key != current_month:
                if current_month and month_start_equity:
                    # Calculate return for previous month
                    month_end_equity = prev_equity
                    monthly_return = (month_end_equity / month_start_equity - 1) if month_start_equity > 0 else 0
                    monthly_returns[current_month] = monthly_return
                
                current_month = month_key
                month_start_equity = point['equity']
            
            prev_equity = point['equity']
        
        # Don't forget the last month
        if current_month and month_start_equity:
            month_end_equity = self.equity_curve[-1]['equity']
            monthly_return = (month_end_equity / month_start_equity - 1) if month_start_equity > 0 else 0
            monthly_returns[current_month] = monthly_return
        
        return monthly_returns
    
    def get_state(self) -> Dict[str, Any]:
        """Get performance tracker state."""
        return {
            "metrics": self.metrics_cache,
            "trade_count": len(self.trades),
            "equity_points": len(self.equity_curve),
            "has_benchmark": len(self.benchmark_returns) > 0
        }
    
    def reset(self) -> None:
        """Reset performance tracker."""
        self.trades = []
        self.equity_curve = []
        self.metrics_cache = {}
        self.benchmark_returns = []
    
    def set_benchmark(self, returns: List[float]) -> None:
        """Set benchmark returns for comparison."""
        self.benchmark_returns = returns


class BaseStrategyInterface(ABC):
    """Base interface for all trading strategies."""
    
    def __init__(self, strategy_id: str, config: Dict[str, Any] = None):
        self.strategy_id = strategy_id
        self.config = config or {}
        
        # Initialize components
        self.indicators = IndicatorSet(f"{strategy_id}_indicators", config.get('indicators'))
        self.signal_generator = SignalGenerator(f"{strategy_id}_signals", config.get('signals'))
        self.risk_manager = RiskManager(f"{strategy_id}_risk", config.get('risk'))
        self.position_sizer = PositionSizer(f"{strategy_id}_sizing", config.get('sizing'))
        self.order_executor = OrderExecutor(f"{strategy_id}_execution", config.get('execution'))
        self.performance_tracker = PerformanceTracker(f"{strategy_id}_performance", config.get('performance'))
        
        # Strategy state
        self.state = StrategyState(
            is_active=False,
            position_count=0,
            open_pnl=0.0,
            realized_pnl=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            last_update=datetime.utcnow()
        )
        
        # Learning components (optional)
        self.learner = None
        self.parameter_optimizer = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the strategy and all components."""
        pass
    
    @abstractmethod
    async def on_data(self, data: Any) -> Optional[TradingSignal]:
        """Process new market data and generate signals."""
        pass
    
    @abstractmethod
    async def on_signal(self, signal: TradingSignal) -> Optional[Order]:
        """Process trading signal and create orders."""
        pass
    
    @abstractmethod
    async def on_fill(self, order: Order, fill: Any) -> None:
        """Handle order fill events."""
        pass
    
    @abstractmethod
    def get_state(self) -> StrategyState:
        """Get current strategy state."""
        pass
    
    @abstractmethod
    def adapt(self, market_conditions: Dict[str, Any]) -> None:
        """Adapt strategy to changing market conditions."""
        pass
    
    async def start(self) -> None:
        """Start the strategy."""
        await self.initialize()
        self.state.is_active = True
        self.state.last_update = datetime.utcnow()
    
    async def stop(self) -> None:
        """Stop the strategy."""
        self.state.is_active = False
        self.state.last_update = datetime.utcnow()
    
    def reset(self) -> None:
        """Reset strategy to initial state."""
        self.indicators.reset()
        self.signal_generator.reset()
        self.risk_manager.reset()
        self.position_sizer.reset()
        self.order_executor.reset()
        self.performance_tracker.reset()
        
        self.state = StrategyState(
            is_active=False,
            position_count=0,
            open_pnl=0.0,
            realized_pnl=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            last_update=datetime.utcnow()
        )