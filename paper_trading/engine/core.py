
"""
Core paper trading engine for realistic market simulation.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from nautilus_trader.common.clock import Clock
from nautilus_trader.common.logging import Logger
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.data import OrderBookDelta, QuoteTick, TradeTick
from nautilus_trader.model.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from nautilus_trader.model.identifiers import AccountId, ClientOrderId, InstrumentId, PositionId, StrategyId, TraderId, VenueOrderId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.orders import Order
from nautilus_trader.msgbus.bus import MessageBus

from paper_trading.engine.accounts import PaperTradingAccount
from paper_trading.engine.execution import OrderExecutor, OrderFill
from paper_trading.engine.market_impact import MarketImpactModel, MarketImpactParams
from paper_trading.engine.slippage import SlippageModel, SlippageParams
from paper_trading.engine.spread import SpreadModel, SpreadParams


@dataclass
class PaperTradingConfig:
    """Configuration for the paper trading engine."""
    
    # Account settings
    initial_balance: Decimal = Decimal("100000")
    base_currency: str = "USD"
    leverage: int = 1
    
    # Execution settings
    enable_slippage: bool = True
    enable_market_impact: bool = True
    enable_spread_costs: bool = True
    enable_partial_fills: bool = True
    
    # Model parameters
    slippage_params: SlippageParams = field(default_factory=lambda: SlippageParams())
    market_impact_params: MarketImpactParams = field(default_factory=lambda: MarketImpactParams())
    spread_params: SpreadParams = field(default_factory=lambda: SpreadParams())
    
    # Execution delays (milliseconds)
    order_processing_delay_ms: int = 50
    market_order_delay_ms: int = 100
    limit_order_delay_ms: int = 150
    
    # Risk limits
    max_position_size: Optional[Decimal] = None
    max_order_size: Optional[Decimal] = None
    max_daily_loss: Optional[Decimal] = None
    max_open_positions: int = 10
    
    # Persistence
    save_state_interval_seconds: int = 300
    state_file_path: Optional[Path] = None


class PaperTradingEngine:
    """
    Comprehensive paper trading engine with realistic market simulation.
    
    Features:
    - Realistic order execution with delays
    - Slippage modeling based on market conditions
    - Market impact simulation
    - Bid/ask spread handling
    - Partial fill simulation
    - Position and account management
    - Performance tracking
    """
    
    def __init__(
        self,
        trader_id: TraderId,
        msgbus: MessageBus,
        clock: Clock,
        logger: Logger,
        config: Optional[PaperTradingConfig] = None,
    ):
        self.trader_id = trader_id
        self.msgbus = msgbus
        self.clock = clock
        self.logger = logger
        self.config = config or PaperTradingConfig()
        
        # Initialize components
        self.slippage_model = SlippageModel(self.config.slippage_params)
        self.market_impact_model = MarketImpactModel(self.config.market_impact_params)
        self.spread_model = SpreadModel(self.config.spread_params)
        
        # Create paper trading account
        self.account = PaperTradingAccount(
            account_id=AccountId(f"{trader_id}-PAPER-001"),
            base_currency=self.config.base_currency,
            initial_balance=self.config.initial_balance,
            leverage=self.config.leverage,
        )
        
        # Order executor
        self.order_executor = OrderExecutor(
            engine=self,
            clock=self.clock,
            logger=self.logger,
        )
        
        # State tracking
        self.orders: Dict[ClientOrderId, Order] = {}
        self.active_orders: Set[ClientOrderId] = set()
        self.positions: Dict[PositionId, Dict[str, Any]] = {}
        self.fills: List[OrderFill] = []
        
        # Market data cache
        self.last_quotes: Dict[InstrumentId, QuoteTick] = {}
        self.last_trades: Dict[InstrumentId, TradeTick] = {}
        self.order_books: Dict[InstrumentId, List[OrderBookDelta]] = {}
        
        # Performance tracking
        self.metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "partial_fills": 0,
            "rejected_orders": 0,
            "total_slippage": Decimal("0"),
            "total_spread_cost": Decimal("0"),
            "total_market_impact": Decimal("0"),
        }
        
        # Risk tracking
        self.daily_loss = Decimal("0")
        self.daily_loss_reset_time = datetime.utcnow()
        
        # State persistence
        self._last_state_save = datetime.utcnow()
        if self.config.state_file_path:
            self._load_state()
        
        self.logger.info(f"Paper Trading Engine initialized for {trader_id}")
    
    async def start(self):
        """Start the paper trading engine."""
        self.logger.info("Starting Paper Trading Engine")
        
        # Start order executor
        await self.order_executor.start()
        
        # Schedule state persistence
        if self.config.save_state_interval_seconds > 0:
            self.clock.set_timer(
                name="save_state",
                interval=self.config.save_state_interval_seconds,
                callback=self._save_state,
            )
        
        self.logger.info("Paper Trading Engine started")
    
    async def stop(self):
        """Stop the paper trading engine."""
        self.logger.info("Stopping Paper Trading Engine")
        
        # Stop order executor
        await self.order_executor.stop()
        
        # Save final state
        if self.config.state_file_path:
            self._save_state()
        
        self.logger.info("Paper Trading Engine stopped")
    
    def submit_order(self, order: Order) -> None:
        """
        Submit an order to the paper trading engine.
        
        Parameters
        ----------
        order : Order
            The order to submit.
        """
        self.logger.info(f"Submitting order: {order}")
        
        # Validate order
        validation_error = self._validate_order(order)
        if validation_error:
            self.logger.error(f"Order validation failed: {validation_error}")
            self._reject_order(order, validation_error)
            return
        
        # Check risk limits
        risk_error = self._check_risk_limits(order)
        if risk_error:
            self.logger.error(f"Risk limit exceeded: {risk_error}")
            self._reject_order(order, risk_error)
            return
        
        # Add to tracking
        self.orders[order.client_order_id] = order
        self.active_orders.add(order.client_order_id)
        self.metrics["total_orders"] += 1
        
        # Submit to executor
        self.order_executor.submit_order(order)
    
    def cancel_order(self, client_order_id: ClientOrderId) -> None:
        """Cancel an order."""
        if client_order_id not in self.active_orders:
            self.logger.warning(f"Cannot cancel order {client_order_id}: not found or already filled")
            return
        
        self.order_executor.cancel_order(client_order_id)
    
    def update_market_data(self, data: Any) -> None:
        """Update market data used for order execution."""
        if isinstance(data, QuoteTick):
            self.last_quotes[data.instrument_id] = data
        elif isinstance(data, TradeTick):
            self.last_trades[data.instrument_id] = data
        elif isinstance(data, OrderBookDelta):
            if data.instrument_id not in self.order_books:
                self.order_books[data.instrument_id] = []
            self.order_books[data.instrument_id].append(data)
            # Keep only recent deltas
            if len(self.order_books[data.instrument_id]) > 100:
                self.order_books[data.instrument_id] = self.order_books[data.instrument_id][-100:]
    
    def get_execution_price(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        order_type: OrderType,
    ) -> Optional[Price]:
        """
        Calculate the execution price including slippage, spread, and market impact.
        
        Returns
        -------
        Price or None
            The execution price, or None if no market data available.
        """
        # Get base price from market data
        base_price = self._get_base_price(instrument_id, side)
        if base_price is None:
            return None
        
        price_value = float(base_price)
        
        # Apply spread cost
        if self.config.enable_spread_costs:
            spread_cost = self.spread_model.calculate_spread_cost(
                instrument_id, side, quantity, self.last_quotes.get(instrument_id)
            )
            if side == OrderSide.BUY:
                price_value += float(spread_cost)
            else:
                price_value -= float(spread_cost)
        
        # Apply slippage
        if self.config.enable_slippage and order_type == OrderType.MARKET:
            slippage = self.slippage_model.calculate_slippage(
                instrument_id,
                side,
                quantity,
                self._get_market_volatility(instrument_id),
                self._get_market_liquidity(instrument_id),
            )
            if side == OrderSide.BUY:
                price_value *= (1 + float(slippage))
            else:
                price_value *= (1 - float(slippage))
        
        # Apply market impact
        if self.config.enable_market_impact:
            impact = self.market_impact_model.calculate_impact(
                instrument_id,
                side,
                quantity,
                self._get_market_depth(instrument_id),
                self._get_average_daily_volume(instrument_id),
            )
            if side == OrderSide.BUY:
                price_value *= (1 + float(impact))
            else:
                price_value *= (1 - float(impact))
        
        return Price(price_value, precision=base_price.precision)
    
    def execute_fill(self, fill: OrderFill) -> None:
        """
        Execute a fill in the paper trading account.
        
        Parameters
        ----------
        fill : OrderFill
            The fill to execute.
        """
        order = self.orders.get(fill.client_order_id)
        if not order:
            self.logger.error(f"Cannot execute fill: order {fill.client_order_id} not found")
            return
        
        # Update account
        self.account.apply_fill(fill, order.instrument_id)
        
        # Track fill
        self.fills.append(fill)
        
        # Update metrics
        if fill.is_partial:
            self.metrics["partial_fills"] += 1
        else:
            self.metrics["filled_orders"] += 1
            self.active_orders.discard(fill.client_order_id)
        
        # Calculate costs
        base_price = self._get_base_price(order.instrument_id, order.side)
        if base_price:
            price_diff = abs(float(fill.price) - float(base_price))
            self.metrics["total_slippage"] += Decimal(str(price_diff * float(fill.quantity)))
    
    def _validate_order(self, order: Order) -> Optional[str]:
        """Validate order parameters."""
        # Check instrument is known
        if order.instrument_id not in self.last_quotes and order.instrument_id not in self.last_trades:
            return f"No market data available for {order.instrument_id}"
        
        # Check order size limits
        if self.config.max_order_size and order.quantity > self.config.max_order_size:
            return f"Order size {order.quantity} exceeds maximum {self.config.max_order_size}"
        
        # Check account balance for margin
        required_margin = self._calculate_required_margin(order)
        if required_margin > self.account.get_available_balance():
            return f"Insufficient margin: required {required_margin}, available {self.account.get_available_balance()}"
        
        return None
    
    def _check_risk_limits(self, order: Order) -> Optional[str]:
        """Check risk limits."""
        # Check max open positions
        if len(self.positions) >= self.config.max_open_positions:
            return f"Maximum open positions ({self.config.max_open_positions}) reached"
        
        # Check daily loss limit
        if self.config.max_daily_loss:
            if self.daily_loss >= self.config.max_daily_loss:
                return f"Daily loss limit ({self.config.max_daily_loss}) exceeded"
        
        # Check position size limit
        if self.config.max_position_size:
            current_position = self._get_position_size(order.instrument_id)
            new_position = current_position + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
            if abs(new_position) > self.config.max_position_size:
                return f"Position size would exceed maximum {self.config.max_position_size}"
        
        return None
    
    def _reject_order(self, order: Order, reason: str) -> None:
        """Reject an order."""
        self.metrics["rejected_orders"] += 1
        # Send rejection event through message bus
        # This would integrate with NautilusTrader's event system
    
    def _get_base_price(self, instrument_id: InstrumentId, side: OrderSide) -> Optional[Price]:
        """Get base price from market data."""
        quote = self.last_quotes.get(instrument_id)
        if quote:
            return quote.ask_price if side == OrderSide.BUY else quote.bid_price
        
        trade = self.last_trades.get(instrument_id)
        if trade:
            return trade.price
        
        return None
    
    def _get_market_volatility(self, instrument_id: InstrumentId) -> float:
        """Calculate market volatility from recent price data."""
        # Simplified: use recent price range
        trades = [t for t in self.last_trades.values() if t.instrument_id == instrument_id]
        if len(trades) < 2:
            return 0.01  # Default 1% volatility
        
        prices = [float(t.price) for t in trades[-20:]]  # Last 20 trades
        if len(prices) < 2:
            return 0.01
        
        return (max(prices) - min(prices)) / sum(prices) * len(prices)
    
    def _get_market_liquidity(self, instrument_id: InstrumentId) -> float:
        """Estimate market liquidity from order book depth."""
        # Simplified: use quote size as proxy
        quote = self.last_quotes.get(instrument_id)
        if quote:
            return float(quote.bid_size + quote.ask_size)
        return 100.0  # Default liquidity
    
    def _get_market_depth(self, instrument_id: InstrumentId) -> Dict[str, List[tuple]]:
        """Get order book depth."""
        # Simplified: use last quote
        quote = self.last_quotes.get(instrument_id)
        if quote:
            return {
                "bids": [(float(quote.bid_price), float(quote.bid_size))],
                "asks": [(float(quote.ask_price), float(quote.ask_size))],
            }
        return {"bids": [], "asks": []}
    
    def _get_average_daily_volume(self, instrument_id: InstrumentId) -> float:
        """Estimate average daily volume."""
        # Simplified: return default value
        # In production, this would query historical data
        return 1000000.0
    
    def _calculate_required_margin(self, order: Order) -> Decimal:
        """Calculate required margin for order."""
        base_price = self._get_base_price(order.instrument_id, order.side)
        if not base_price:
            return Decimal("0")
        
        notional = Decimal(str(float(order.quantity) * float(base_price)))
        return notional / Decimal(str(self.config.leverage))
    
    def _get_position_size(self, instrument_id: InstrumentId) -> Decimal:
        """Get current position size for instrument."""
        # Sum all positions for this instrument
        total = Decimal("0")
        for pos_data in self.positions.values():
            if pos_data["instrument_id"] == instrument_id:
                total += pos_data["quantity"] * (1 if pos_data["side"] == OrderSide.BUY else -1)
        return total
    
    def _save_state(self, _=None) -> None:
        """Save engine state to file."""
        if not self.config.state_file_path:
            return
        
        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "account": self.account.to_dict(),
            "metrics": self.metrics,
            "orders": len(self.orders),
            "active_orders": len(self.active_orders),
            "positions": len(self.positions),
            "fills": len(self.fills),
        }
        
        self.config.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.state_file_path, "w") as f:
            json.dump(state, f, indent=2)
        
        self._last_state_save = datetime.utcnow()
    
    def _load_state(self) -> None:
        """Load engine state from file."""
        if not self.config.state_file_path or not self.config.state_file_path.exists():
            return
        
        try:
            with open(self.config.state_file_path) as f:
                state = json.load(f)
            
            # Restore account state
            self.account.from_dict(state.get("account", {}))
            
            # Restore metrics
            self.metrics.update(state.get("metrics", {}))
            
            self.logger.info(f"Loaded state from {self.config.state_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "account": {
                "balance": float(self.account.get_balance()),
                "equity": float(self.account.get_equity()),
                "margin_used": float(self.account.get_margin_used()),
                "margin_available": float(self.account.get_available_balance()),
                "unrealized_pnl": float(self.account.get_unrealized_pnl()),
                "realized_pnl": float(self.account.get_realized_pnl()),
            },
            "trading": {
                "total_orders": self.metrics["total_orders"],
                "filled_orders": self.metrics["filled_orders"],
                "partial_fills": self.metrics["partial_fills"],
                "rejected_orders": self.metrics["rejected_orders"],
                "win_rate": self._calculate_win_rate(),
                "avg_win": float(self._calculate_avg_win()),
                "avg_loss": float(self._calculate_avg_loss()),
                "profit_factor": float(self._calculate_profit_factor()),
            },
            "costs": {
                "total_slippage": float(self.metrics["total_slippage"]),
                "total_spread_cost": float(self.metrics["total_spread_cost"]),
                "total_market_impact": float(self.metrics["total_market_impact"]),
                "avg_slippage_per_trade": float(self._calculate_avg_slippage()),
            },
            "risk": {
                "max_drawdown": float(self.account.get_max_drawdown()),
                "current_drawdown": float(self.account.get_current_drawdown()),
                "sharpe_ratio": float(self._calculate_sharpe_ratio()),
                "sortino_ratio": float(self._calculate_sortino_ratio()),
            },
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from fills."""
        if not self.fills:
            return 0.0
        
        wins = sum(1 for fill in self.fills if fill.realized_pnl and fill.realized_pnl > 0)
        return wins / len(self.fills) if self.fills else 0.0
    
    def _calculate_avg_win(self) -> Decimal:
        """Calculate average winning trade."""
        wins = [fill.realized_pnl for fill in self.fills if fill.realized_pnl and fill.realized_pnl > 0]
        return sum(wins) / len(wins) if wins else Decimal("0")
    
    def _calculate_avg_loss(self) -> Decimal:
        """Calculate average losing trade."""
        losses = [fill.realized_pnl for fill in self.fills if fill.realized_pnl and fill.realized_pnl < 0]
        return sum(losses) / len(losses) if losses else Decimal("0")
    
    def _calculate_profit_factor(self) -> Decimal:
        """Calculate profit factor."""
        gross_profit = sum(fill.realized_pnl for fill in self.fills if fill.realized_pnl and fill.realized_pnl > 0)
        gross_loss = abs(sum(fill.realized_pnl for fill in self.fills if fill.realized_pnl and fill.realized_pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else Decimal("0")
    
    def _calculate_avg_slippage(self) -> Decimal:
        """Calculate average slippage per trade."""
        if self.metrics["filled_orders"] == 0:
            return Decimal("0")
        return self.metrics["total_slippage"] / Decimal(str(self.metrics["filled_orders"]))
    
    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio from returns."""
        # Simplified calculation
        if len(self.fills) < 2:
            return Decimal("0")
        
        returns = []
        for i in range(1, len(self.fills)):
            if self.fills[i].realized_pnl and self.fills[i-1].realized_pnl:
                returns.append(float(self.fills[i].realized_pnl))
        
        if not returns:
            return Decimal("0")
        
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        return Decimal(str(avg_return / std_return)) if std_return > 0 else Decimal("0")
    
    def _calculate_sortino_ratio(self) -> Decimal:
        """Calculate Sortino ratio from returns."""
        # Simplified calculation focusing on downside deviation
        if len(self.fills) < 2:
            return Decimal("0")
        
        returns = []
        for fill in self.fills:
            if fill.realized_pnl:
                returns.append(float(fill.realized_pnl))
        
        if not returns:
            return Decimal("0")
        
        avg_return = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return Decimal("999")  # No downside
        
        downside_deviation = (sum(r ** 2 for r in downside_returns) / len(downside_returns)) ** 0.5
        
        return Decimal(str(avg_return / downside_deviation)) if downside_deviation > 0 else Decimal("0")