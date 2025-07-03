# Multi-Asset Trading System

A comprehensive, unified trading system for managing multiple asset classes including cryptocurrencies, stocks, forex, and commodities/futures. Built on top of Nautilus Trader, this system provides seamless asset management with intelligent risk controls and execution routing.

## Features

### 🌐 Unified Asset Interface
- **Single API** for all asset classes
- **Asset-specific characteristics** preserved (market hours, tick sizes, margin requirements)
- **Extensible design** for adding new asset types
- **Real-time market status** tracking

### 💎 Supported Asset Classes

#### Cryptocurrencies
- Spot, futures, and perpetual contracts
- 24/7 market support
- Exchange-specific configurations (Binance, Coinbase, etc.)
- Funding rate calculations for perpetuals
- Stablecoin detection and handling

#### Equities/Stocks
- Common stocks, ETFs, ADRs
- Market hours enforcement (regular, pre-market, post-market)
- Exchange-specific rules (NYSE, NASDAQ, etc.)
- Dividend and earnings tracking
- Pattern Day Trader (PDT) rule compliance

#### Foreign Exchange (Forex)
- Major, minor, and exotic currency pairs
- 24/5 market (weekday trading)
- Pip calculations and lot size management
- Swap/rollover calculations
- Session-based volatility adjustments

#### Commodities/Futures
- Energy, metals, agriculture contracts
- Contract expiry management
- First notice date warnings
- Seasonality factors
- Roll date calculations

### 🛡️ Risk Management

#### Portfolio-Level Controls
- **Maximum leverage limits** across all positions
- **Daily loss limits** with automatic circuit breakers
- **Drawdown protection** with configurable thresholds
- **Concentration limits** per position and asset class
- **Correlation-based exposure** management

#### Asset-Specific Risk Parameters
- Custom position limits by asset class
- Dynamic margin requirements
- Volatility-adjusted position sizing
- Asset class allocation limits

### 🚀 Smart Order Execution

#### Execution Router Features
- **Intelligent venue selection** based on historical performance
- **Order splitting** for large orders
- **Dark pool integration** for equities
- **Asset-specific routing logic**
- **Real-time execution quality monitoring**

#### Routing Strategies
- Best Execution
- Lowest Cost
- Fastest Fill
- Smart Route (adaptive)
- Direct routing

### 📊 Portfolio Analytics
- Real-time P&L tracking by asset class
- Sharpe ratio calculation
- Value at Risk (VaR) metrics
- Concentration analysis (Herfindahl index)
- Cross-asset correlation monitoring

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd trader/multi_asset_system

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from multi_asset_system.core.asset_manager import MultiAssetManager
from multi_asset_system.core.unified_trader import UnifiedTrader
from multi_asset_system.risk.portfolio_risk_manager import PortfolioRiskManager

# Initialize the system
asset_manager = MultiAssetManager(logger=logger)
risk_manager = PortfolioRiskManager(
    asset_manager=asset_manager,
    initial_capital=Decimal("100000")
)
trader = UnifiedTrader(
    trader_id=TraderId("TRADER-001"),
    asset_manager=asset_manager,
    risk_manager=risk_manager,
    execution_router=execution_router,
    logger=logger
)

# Register assets
btc = asset_manager.create_and_register_asset(
    instrument_id=InstrumentId("BTCUSDT", "BINANCE"),
    asset_class=AssetClass.CRYPTO,
    config={"base_currency": "BTC", "quote_currency": "USDT"}
)

# Submit orders with automatic risk checks
order_id = await trader.submit_order(
    instrument_id=btc.instrument_id,
    order_side=OrderSide.BUY,
    quantity=Decimal("0.1"),
    order_type=OrderType.MARKET
)
```

## Configuration

### Portfolio Risk Parameters
```json
{
  "portfolio_risk_params": {
    "initial_capital": "100000",
    "max_portfolio_leverage": "3.0",
    "max_daily_loss_percent": "0.03",
    "max_drawdown_percent": "0.15",
    "max_concentration_percent": "0.20"
  }
}
```

### Asset Class Limits
```json
{
  "CRYPTO": {
    "max_allocation": "0.30",
    "max_leverage": "2.0",
    "position_limit": 10
  },
  "EQUITY": {
    "max_allocation": "0.60",
    "max_leverage": "2.0",
    "position_limit": 30
  }
}
```

## Architecture

### Core Components

1. **Asset Interface** (`asset_interface.py`)
   - Base abstract class for all assets
   - Common methods for validation, fees, margin
   - Market hours and trading rules

2. **Asset Manager** (`asset_manager.py`)
   - Central registry for all assets
   - Cross-asset analytics
   - Market calendar management

3. **Unified Trader** (`unified_trader.py`)
   - Single interface for trading all assets
   - Automatic risk validation
   - Position management

4. **Portfolio Risk Manager** (`portfolio_risk_manager.py`)
   - Real-time risk monitoring
   - Portfolio-wide limits enforcement
   - Risk metrics calculation

5. **Execution Router** (`execution_router.py`)
   - Smart order routing
   - Venue selection
   - Order splitting logic

### Asset Implementations

Each asset class has its own implementation with specific logic:

- `crypto_asset.py` - Cryptocurrency-specific features
- `equity_asset.py` - Stock market rules and regulations
- `forex_asset.py` - FX pair handling and pip calculations
- `commodity_asset.py` - Futures contract management

## Usage Examples

### Multi-Asset Portfolio Setup
```python
# Load configuration
asset_manager.load_configuration("config/multi_asset_config.json")

# Get all tradable assets
tradable = asset_manager.get_tradable_assets()
print(f"Currently tradable: {len(tradable)} assets")

# Get assets by class
cryptos = asset_manager.get_assets_by_class(AssetClass.CRYPTO)
equities = asset_manager.get_assets_by_class(AssetClass.EQUITY)
```

### Risk-Aware Order Submission
```python
# Order automatically validated against:
# - Asset-specific rules (min/max size, tick size)
# - Portfolio risk limits (concentration, leverage)
# - Market hours restrictions

order_id = await trader.submit_order(
    instrument_id=InstrumentId("AAPL", "NASDAQ"),
    order_side=OrderSide.BUY,
    quantity=Decimal("100"),
    order_type=OrderType.LIMIT,
    price=Price(180),
    tags=["earnings_play"]
)
```

### Portfolio Rebalancing
```python
# Define target allocation
target_weights = {
    AssetClass.CRYPTO: Decimal("0.20"),
    AssetClass.EQUITY: Decimal("0.50"),
    AssetClass.FOREX: Decimal("0.20"),
    AssetClass.COMMODITY: Decimal("0.10")
}

# Automatic rebalancing with tolerance
orders = await trader.rebalance_portfolio(
    target_weights=target_weights,
    tolerance=Decimal("0.05")
)
```

### Risk Monitoring
```python
# Get comprehensive risk metrics
metrics = await risk_manager.calculate_portfolio_metrics()

print(f"Portfolio Value: ${metrics.total_value}")
print(f"Leverage: {metrics.leverage}x")
print(f"Daily P&L: ${metrics.daily_pnl}")
print(f"VaR (95%): {metrics.var_95:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

# Check for risk warnings
warnings = risk_manager.get_risk_summary()["warnings"]
for warning in warnings:
    print(f"⚠️  {warning}")
```

## Advanced Features

### Custom Asset Implementation
```python
from multi_asset_system.core.asset_interface import Asset

class CryptoOption(Asset):
    """Custom implementation for crypto options."""
    
    def __init__(self, strike_price, expiry_date, **kwargs):
        self.strike_price = strike_price
        self.expiry_date = expiry_date
        super().__init__(**kwargs)
    
    def calculate_margin_requirement(self, quantity, price, is_short=False):
        # Custom margin logic for options
        pass
```

### Execution Strategy Customization
```python
# Register custom routing logic
execution_router.register_routing_strategy(
    "VWAP",
    lambda order, asset: vwap_routing_logic(order, asset)
)

# Use custom strategy
await trader.submit_order(
    instrument_id=instrument_id,
    quantity=quantity,
    routing_strategy="VWAP"
)
```

## Performance Considerations

- **Efficient asset lookup** using dictionaries
- **Lazy loading** of market data
- **Caching** of frequently accessed calculations
- **Async/await** for non-blocking operations
- **Batch operations** for multiple orders

## Testing

Run the demonstration script to see all features:

```bash
python examples/multi_asset_demo.py
```

This will demonstrate:
- Asset registration
- Risk validation
- Order submission
- Portfolio metrics
- Market hours handling
- Portfolio rebalancing

## Best Practices

1. **Always update prices** before trading
2. **Use risk checks** for all orders
3. **Monitor correlation** between positions
4. **Respect market hours** for traditional assets
5. **Handle contract rollovers** for futures
6. **Set appropriate position limits** per asset class
7. **Use smart routing** for better execution

## Integration with Nautilus Trader

This system is designed to work seamlessly with Nautilus Trader:

```python
# Use with Nautilus strategies
class MultiAssetStrategy(Strategy):
    def __init__(self, unified_trader):
        super().__init__()
        self.trader = unified_trader
    
    def on_data(self, data):
        # Strategy logic
        if signal:
            self.trader.submit_order(...)
```

## Future Enhancements

- [ ] Options support (equity and crypto)
- [ ] Fixed income/bonds
- [ ] Real-time correlation matrix updates
- [ ] Machine learning for venue selection
- [ ] Advanced order types (TWAP, VWAP, Iceberg)
- [ ] Multi-currency portfolio management
- [ ] Tax optimization features
- [ ] Regulatory reporting

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- All tests pass
- New features include tests
- Documentation is updated

## License

Licensed under the GNU Lesser General Public License Version 3.0