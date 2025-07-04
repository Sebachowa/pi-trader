# 🏗️ System Architecture

Comprehensive overview of the trading bot's architecture and design principles.

## 🎯 Design Philosophy

### Autonomous Operation
```
MARKET → ANALYSIS → DECISION → EXECUTION → MONITORING
   ↑                                            ↓
   └────────────── LEARNING ←──────────────────┘
```

The bot operates **completely autonomously**, making all trading decisions without human intervention based on:
- Real-time market analysis
- Risk management rules
- Predefined strategies
- Continuous performance monitoring

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────┐
│            RASPBERRY PI TRADING BOT         │
├─────────────────────────────────────────────┤
│  🔍 MARKET SCANNER                          │
│     • Monitors 100+ trading pairs          │
│     • Detects opportunities in real-time   │
│     • Calculates confidence scores         │
├─────────────────────────────────────────────┤
│  🧠 STRATEGY ENGINE                         │
│     • Trend following detection            │
│     • Mean reversion identification        │
│     • Momentum analysis                    │
│     • Multi-timeframe confirmation         │
├─────────────────────────────────────────────┤
│  ⚡ EXECUTION ENGINE                        │
│     • Risk-managed position sizing         │
│     • Dynamic stop loss/take profit        │
│     • Order execution and management       │
├─────────────────────────────────────────────┤
│  🛡️ RISK MANAGER                            │
│     • Portfolio protection                 │
│     • Drawdown limits                      │
│     • Position size limits                 │
│     • Exposure management                  │
├─────────────────────────────────────────────┤
│  📊 MONITORING & LOGGING                    │
│     • Real-time performance tracking       │
│     • Beautiful colored logs               │
│     • Tax calculation                      │
│     • System health monitoring             │
└─────────────────────────────────────────────┘
```

## 🔧 Core Components

### 1. Market Scanner
**File**: `core/market_scanner.py`

```python
class MarketScanner:
    """
    High-performance scanner that continuously monitors
    multiple symbols for trading opportunities
    """
    
    Features:
    • Async/concurrent scanning of 100+ pairs
    • Technical indicator calculation
    • Opportunity scoring (0-100)
    • Configurable scan intervals
    • Memory-efficient operation
```

**Key Capabilities**:
- Scans 100 symbols in ~8-10 seconds
- Uses minimal CPU/RAM on Raspberry Pi
- Finds opportunities across multiple strategies
- Adaptive scoring for different market conditions

### 2. Strategy Engine
**Files**: `strategies/trend_following.py`, `strategies/mean_reversion.py`

```python
Supported Strategies:
┌──────────────────────────────────────┐
│  📈 Trend Following                  │
│     • MACD convergence/divergence    │
│     • EMA crossovers                 │
│     • Price above/below moving avg   │
│     • RSI momentum confirmation      │
├──────────────────────────────────────┤
│  📊 Mean Reversion                   │
│     • Bollinger Bands oversold      │
│     • RSI extreme levels            │
│     • Volume spike confirmation      │
│     • Support/resistance levels     │
├──────────────────────────────────────┤
│  🚀 Momentum Trading                 │
│     • Strong price movements        │
│     • Volume breakouts              │
│     • Multi-timeframe confirmation  │
│     • Trend acceleration            │
└──────────────────────────────────────┘
```

### 3. Risk Management
**File**: `core/risk_manager.py`

```python
Risk Controls:
• Position sizing: 5-15% of portfolio per trade
• Stop loss: 2% maximum loss per position
• Take profit: 3-5% target per position
• Daily loss limit: 2% of total portfolio
• Maximum positions: 3 concurrent trades
• Cooldown periods: 15 minutes between trades on same pair
```

### 4. Execution Engine
**File**: `core/engine.py`

```python
class TradingEngine:
    """
    Main orchestrator that coordinates all components
    """
    
    Responsibilities:
    • Receives opportunities from scanner
    • Validates trades with risk manager
    • Executes orders via exchange API
    • Manages open positions
    • Handles position exits (SL/TP)
```

## 🔄 Data Flow

### 1. Market Data Ingestion
```
Exchange API → Market Scanner → Technical Indicators → Opportunity Scoring
```

### 2. Decision Making
```
Opportunities → Strategy Validation → Risk Check → Position Sizing → Order Creation
```

### 3. Position Management
```
Open Positions → Price Monitoring → Exit Conditions → Order Execution → P&L Calculation
```

### 4. Monitoring & Logging
```
All Events → Structured Logging → Tax Calculation → Performance Metrics → Notifications
```

## 🎛️ Configuration System

### Environment-Based Configuration
```bash
# .env file (sensitive data)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=true

# config.json (trading parameters)
{
  "trading": { "position_size_pct": 0.1 },
  "risk": { "max_daily_loss_pct": 0.02 },
  "scanner": { "min_opportunity_score": 40 }
}
```

### Adaptive Configuration
The bot automatically adapts certain parameters:
- **Testnet mode**: Lower scoring thresholds
- **Market conditions**: Adjusted risk parameters
- **System resources**: Optimized scan frequencies

## 🚀 Performance Characteristics

### Resource Usage (Raspberry Pi 4)
```
CPU Usage:    15-30% average
RAM Usage:    150-250MB
Disk I/O:     Minimal (logs only)
Network:      ~1MB/hour (API calls)
Startup Time: <10 seconds
```

### Trading Performance
```
Scan Time:      8-10 seconds for 100 symbols
Response Time:  <1 second from signal to order
Opportunity Rate: 5-20 per hour (depending on volatility)
Win Rate Target: 55-65% (strategy dependent)
```

## 🔧 Extensibility

### Adding New Strategies
```python
# 1. Create strategy file
class MyStrategy(BaseStrategy):
    def analyze(self, data):
        # Your strategy logic
        return opportunity_score

# 2. Register in engine
strategies = {
    'my_strategy': MyStrategy(),
    'trend_following': TrendFollowing(),
    'mean_reversion': MeanReversion()
}
```

### Custom Indicators
```python
# Add to market_scanner.py
def _custom_indicator(self, closes, highs, lows):
    # Calculate your indicator
    return indicator_value
```

### External Integrations
```python
# Webhook notifications
def send_webhook(self, message):
    requests.post(webhook_url, json=message)

# Database logging
def log_to_database(self, trade_data):
    database.insert(trade_data)
```

## 🛡️ Security Architecture

### API Security
- Read-only API keys where possible
- Spot trading permissions only (no withdrawals)
- Testnet for development and testing
- Key rotation capabilities

### System Security
```bash
# Systemd service with restricted permissions
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/home/pi/trading-bot/logs
```

### Data Protection
- Sensitive data in environment variables
- Logs exclude API keys and secrets
- Configuration files with proper permissions
- Backup and recovery procedures

## 📊 Monitoring Architecture

### Multi-Level Logging
```python
Levels:
🐛 DEBUG   - Detailed execution flow
📝 INFO    - Normal operations
⚠️ WARNING - Potential issues
❌ ERROR   - Problems requiring attention
🔥 CRITICAL- System failures
```

### Health Monitoring
```python
System Metrics:
• CPU/RAM/Disk usage
• Network connectivity
• Exchange API health
• Position count and P&L
• Error rates and frequencies
```

### Alerting System
```python
Alert Triggers:
• High CPU/memory usage
• Exchange connectivity issues
• Large losses or drawdowns
• System errors or crashes
• Unusual trading patterns
```

## 🔮 Future Architecture Considerations

### Scalability
- Multi-exchange support
- Distributed scanning
- Load balancing
- Database backends

### Machine Learning Integration
- Pattern recognition
- Adaptive parameters
- Sentiment analysis
- Market regime detection

### Advanced Features
- Portfolio optimization
- Cross-pair arbitrage
- Social trading integration
- Advanced order types

---

**Related Documentation:**
- [Scanner Flow](scanner-flow.md) - Detailed scanner operation
- [Logging Guide](logging-guide.md) - Understanding the logs
- [Configuration Guide](../getting-started/configuration.md) - Setup parameters