# 🤖 Nautilus AutoBot - Autonomous Trading System

An intelligent, self-managing trading bot built on top of [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader).

## 🎯 Features

- **10% Annual Return Target** - Optimized for consistent passive income
- **Minimal Human Intervention** - Runs autonomously for weeks
- **Self-Healing System** - Automatic error recovery and adaptation
- **ML-Powered Strategy Selection** - Chooses best strategies for market conditions
- **Multi-Strategy Orchestration** - Runs multiple strategies simultaneously
- **Adaptive Risk Management** - Dynamic position sizing based on market regime
- **Passive Income Optimization** - Compound growth with withdrawal planning

## 🏗️ Architecture

```
nautilus_autobot/
├── autonomous_trading/         # Core autonomous system
│   ├── core/                  # Engine, risk controller, optimizers
│   ├── strategies/            # Trading strategies
│   ├── monitoring/            # Performance tracking
│   └── config/               # Configuration files
├── docs/                      # Documentation
├── tests/                     # Test suite
└── scripts/                   # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Nautilus Trader
- Exchange API keys (for live trading)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nautilus_autobot.git
cd nautilus_autobot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example config:
```bash
cp autonomous_trading/config/autonomous_config_complete.json config.json
```

2. Set up exchange credentials (for live trading):
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_secret"
```

3. Configure Telegram notifications (optional):
```bash
cp telegram_config.example.json telegram_config.json
# Edit with your bot token and chat ID
```

## 📊 Running the Challenge

### 14-Day Paper Trading Evaluation

```bash
# Run paper trading for 14 days
python run_challenge.py --mode paper --days 14 --capital 0.3
```

### Monitor Performance

```bash
# Check status
python scripts/check_status.py

# View live logs
tail -f logs/autobot.log

# Generate performance report
python scripts/generate_report.py
```

## 🎯 Performance Targets

- **Annual Return**: 10% (0.83% monthly)
- **Max Drawdown**: 10%
- **Sharpe Ratio**: >1.5
- **Win Rate**: >55%
- **Daily Loss Limit**: 2%

## 🧠 Strategies

The bot includes multiple strategies that adapt to market conditions:

1. **Trend Following** - EMA crossovers with momentum confirmation
2. **Mean Reversion** - Bollinger Bands with RSI divergence
3. **Market Making** - Spread capture in liquid markets
4. **Arbitrage** - Cross-exchange opportunities
5. **ML Predictions** - LSTM-based price forecasting

## 🛡️ Risk Management

- **Adaptive Position Sizing** - Based on volatility and regime
- **Portfolio Limits** - Max exposure per asset and total
- **Correlation Control** - Avoid concentrated risks
- **Circuit Breakers** - Emergency stops on extreme moves

## 📱 Monitoring

- **Telegram Notifications** - Real-time trade alerts
- **Web Dashboard** - Performance metrics and charts
- **Daily Reports** - Automated performance summaries
- **Health Checks** - System status monitoring

## 🔧 Customization

### Add a New Strategy

```python
# autonomous_trading/strategies/my_strategy.py
from autonomous_trading.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def on_data(self, data):
        # Your trading logic here
        pass
```

### Modify Risk Parameters

Edit `config.json`:
```json
{
  "risk_management": {
    "max_position_size": 0.05,
    "max_daily_loss_percent": 2.0
  }
}
```

## 📈 Live Trading

After successful paper trading:

```bash
# Switch to live mode with small capital
python run_challenge.py --mode live --capital 0.1

# Monitor carefully for first week
python scripts/monitor_live.py
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [Nautilus Trader](https://nautilustrader.io)
- Inspired by quantitative trading research
- Community contributions welcome

## ⚠️ Disclaimer

Trading involves significant risk. Past performance is not indicative of future results. Only trade with capital you can afford to lose. This software is provided as-is without any guarantees.

---

**Remember**: The goal is passive income through intelligent, autonomous trading with minimal human intervention.