# 🤖 Raspberry Pi Trading Bot

A lightweight, autonomous cryptocurrency trading bot optimized for Raspberry Pi with beautiful colored logging and tax tracking.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-red.svg)

## ✨ Features

- 🚀 **Lightweight** - Uses only ~200MB vs 2GB+ for typical bots
- 💡 **Smart Scanner** - Finds opportunities across 100+ trading pairs
- 🎯 **Multiple Strategies** - Trend following, mean reversion, momentum
- 💰 **Tax Tracking** - Built-in capital gains calculation
- 📱 **Telegram Alerts** - Real-time notifications
- 🌈 **Beautiful Logs** - Color-coded with emojis for easy monitoring
- 🧪 **Testnet Support** - Safe testing with Binance testnet

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Sebachowa/pi-trader.git
cd pi-trader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # Or use scripts/deployment/setup_pi.sh
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your API keys (or use testnet keys)
```

### 3. Run
```bash
# Demo mode (no API keys needed)
python run.py --demo

# Testnet trading
python run.py

# Check options
python run.py --help
```

## 📖 Documentation

### Getting Started
- [Quick Start Guide](docs/getting-started/quick-start.md) - Local setup and first run
- [Configuration Guide](docs/getting-started/configuration.md) - All settings explained
- [Deployment Guide](docs/getting-started/deployment.md) - Production deployment

### Technical Details
- [Architecture Overview](docs/technical/architecture.md) - System design
- [Scanner Flow](docs/technical/scanner-flow.md) - How opportunities are found
- [Logging Guide](docs/technical/logging-guide.md) - Understanding the logs

### Features
- [Tax Tracking](docs/features/tax-features.md) - Capital gains calculation
- [Strategies](docs/features/strategies.md) - Trading strategies explained

### Analysis
- [Alternative Bots](docs/analysis/alternatives.md) - Comparison with other bots
- [Why This Bot?](docs/analysis/why-this-bot.md) - Unique advantages

## 🎯 Example Output

```
12:38:32 📝  INFO     [engine      ] 🚀 Raspberry Pi Trading Bot Starting
12:38:33 📝 ⚙️  INFO     [monitor     ] ⚙️  System: CPU 15.5%, RAM 45.2% | Positions: 0 | Equity: $10,000.00
12:38:35 📝 🔍  INFO     [scanner     ] 🔍 Scan completed in 8.45s, found 3 opportunities 🎯
12:38:35 📝 💡  INFO     [scanner     ] 💡 OPPORTUNITY FOUND! BTC/USDT - trend_following (score: 85.5)
12:38:36 📝 🎯  INFO     [engine      ] 🎯 Trend Following signal: 🟢 BUY (confidence: 85.5%)
12:38:37 📝 💰  INFO     [engine      ] 💰 TRADE OPENED: BTC/USDT - Size: 0.0125 @ $108,950.50
```

## 🛠️ System Requirements

- Raspberry Pi 4 (2GB+ RAM recommended)
- Python 3.9+
- ~200MB free space
- Internet connection

## 📊 Performance

- **CPU Usage**: ~15-30% on Pi 4
- **RAM Usage**: ~150-250MB
- **Scan Time**: ~8-10s for 100 symbols
- **Startup Time**: <10 seconds

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

## 📄 License

MIT License - see LICENSE file for details.

## ⚠️ Disclaimer

This bot is for educational purposes. Cryptocurrency trading carries significant risk. Never trade with money you can't afford to lose.

---

Made with ❤️ for the Raspberry Pi community