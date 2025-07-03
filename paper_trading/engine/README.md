# Comprehensive Paper Trading Engine

A sophisticated paper trading simulator that provides realistic market conditions for strategy testing and validation.

## Features

### 🎯 Realistic Order Execution
- **Slippage Modeling**: Dynamic slippage based on volatility, order size, and market conditions
- **Market Impact Simulation**: Implements linear, square-root, and power-law impact models
- **Bid/Ask Spread Handling**: Realistic spread costs with dynamic adjustments
- **Partial Fills**: Large orders are filled in multiple parts, simulating real market behavior
- **Execution Delays**: Configurable delays for different order types

### 📊 Market Models

#### Slippage Model
- Base slippage rates by asset class
- Volatility-based adjustments
- Order size impact
- Time of day factors
- Random microstructure noise

#### Market Impact Model
- Linear impact for small orders
- Square-root impact (Almgren-Chriss) for medium orders
- Power law impact for large orders
- Temporary vs permanent impact decomposition
- Order book consumption modeling

#### Spread Model
- Dynamic spread calculation
- Time-based adjustments (wider spreads outside market hours)
- Size-based spread widening
- Cross-currency/synthetic spread adjustments

### 💰 Account Management
- Multi-currency support
- Margin and leverage handling
- Real-time position tracking
- Commission modeling
- Risk metrics (drawdown, exposure)

### 📈 Performance Tracking
- Comprehensive metrics tracking
- Real-time PnL calculation
- Cost analysis (slippage, spread, impact)
- Risk metrics (Sharpe, Sortino, drawdown)
- State persistence for recovery

## Quick Start

```python
import asyncio
from paper_trading.engine.core import PaperTradingEngine
from paper_trading.engine.config import PaperTradingConfigs

# Create engine with realistic crypto configuration
config = PaperTradingConfigs.realistic_crypto()
engine = PaperTradingEngine(trader_id, msgbus, clock, logger, config)

# Start engine
await engine.start()

# Submit orders
engine.submit_order(market_order)

# Get performance summary
summary = engine.get_performance_summary()
```

## Configuration Presets

### 1. Realistic Crypto
- 3x leverage
- 2 basis points base slippage
- 5 basis points minimum spread
- High volatility impact

### 2. Realistic Forex
- 50x leverage
- 0.5 basis points base slippage
- 1 basis point minimum spread
- Fast execution

### 3. Realistic Equities
- 2x leverage
- 1 basis point base slippage
- 2 basis points minimum spread
- Time-of-day effects

### 4. Low Latency HFT
- 10x leverage
- 0.1 basis point base slippage
- 1-5ms execution delays
- Minimal market impact

### 5. Conservative Testing
- No leverage
- 3 basis points base slippage
- Worst-case scenario settings
- Maximum realistic costs

## Advanced Usage

### Custom Configuration
```python
from paper_trading.engine.core import PaperTradingConfig
from paper_trading.engine.slippage import SlippageParams

config = PaperTradingConfig(
    initial_balance=Decimal("100000"),
    enable_slippage=True,
    enable_market_impact=True,
    slippage_params=SlippageParams(
        base_slippage_rate=0.0002,
        volatility_multiplier=2.0,
    ),
)
```

### Market Data Updates
```python
# Update with quote data
engine.update_market_data(quote_tick)

# Update with trade data
engine.update_market_data(trade_tick)

# Update with order book deltas
engine.update_market_data(order_book_delta)
```

### Performance Analysis
```python
# Get comprehensive summary
summary = engine.get_performance_summary()

# Access specific metrics
account_equity = engine.account.get_equity()
unrealized_pnl = engine.account.get_unrealized_pnl()
max_drawdown = engine.account.get_max_drawdown()

# Cost breakdown
total_slippage = engine.metrics["total_slippage"]
avg_slippage = engine._calculate_avg_slippage()
```

## Integration with NautilusTrader

The paper trading engine is designed to integrate seamlessly with NautilusTrader:

```python
from nautilus_trader.live.node import TradingNode
from paper_trading.engine.core import PaperTradingEngine

# Use within trading node
node = TradingNode(config)
paper_engine = PaperTradingEngine(
    node.trader_id,
    node.msgbus,
    node.clock,
    node.logger,
)

# Route orders through paper engine
# instead of live execution
```

## Files

- `core.py` - Main paper trading engine
- `slippage.py` - Slippage modeling
- `market_impact.py` - Market impact calculations
- `spread.py` - Bid/ask spread handling
- `accounts.py` - Virtual account management
- `execution.py` - Order execution simulation
- `config.py` - Configuration presets
- `example_usage.py` - Usage examples