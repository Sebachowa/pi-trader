# 🤖 Raspberry Pi Crypto Trader

A lightweight, 24/7 autonomous cryptocurrency trading system optimized for Raspberry Pi deployment.

## 🎯 Features

- **Lightweight**: Optimized for ARM architecture and resource-constrained environments
- **24/7 Autonomous Operation**: Runs continuously without manual intervention
- **Multiple Strategies**: Trend following, mean reversion, and more
- **Real-time Monitoring**: Web dashboard for monitoring positions and performance
- **Risk Management**: Built-in position sizing and risk controls
- **Easy Deployment**: Automated deployment with GitHub Actions
- **Paper Trading**: Test strategies safely before going live
- **Tax Calculation**: Automatic tax tracking and reporting for 15+ jurisdictions
- **Smart Market Scanner**: Finds best opportunities across 100+ pairs every 30 seconds

## 📊 Architecture

```
┌─────────────────────────────────────────────┐
│           AUTONOMOUS TRADER 24/7            │
├─────────────────────────────────────────────┤
│  1. MARKET SCANNER                          │
│     • Monitors multiple pairs               │
│     • Detects trading opportunities         │
├─────────────────────────────────────────────┤
│  2. STRATEGY ENGINE                         │
│     • Trend Following                       │
│     • Mean Reversion                        │
│     • Custom Strategies                     │
├─────────────────────────────────────────────┤
│  3. RISK MANAGER                            │
│     • Position Sizing                       │
│     • Stop Loss/Take Profit                 │
│     • Portfolio Limits                      │
├─────────────────────────────────────────────┤
│  4. MONITORING                              │
│     • Real-time Dashboard                   │
│     • Performance Metrics                   │
│     • Alert System                          │
└─────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Raspberry Pi 5 (8GB+ RAM recommended)
- Python 3.11+
- Exchange API keys (Binance, Bybit, etc.)
- Stable internet connection

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trader.git
cd trader
```

2. Install dependencies:
```bash
pip install -r requirements-pi.txt
```

3. Configure your settings:
```bash
cp config/config.example.json config/config.json
# Edit config/config.json with your API keys and preferences
```

4. Run the trader:
```bash
python run.py
```

## 📱 Monitoring Dashboard

Access the web dashboard at `http://your-pi-ip:8080` to monitor:

- Current positions
- P&L in real-time
- Trading history
- System health
- Risk metrics

## 🔧 Configuration

### Exchange Configuration
```json
{
  "exchange": {
    "name": "binance",
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "testnet": true
  }
}
```

### Risk Management
```json
{
  "risk": {
    "max_position_size": 0.1,
    "max_positions": 3,
    "stop_loss_percent": 0.02,
    "take_profit_percent": 0.03
  }
}
```

### Strategy Settings
```json
{
  "strategies": {
    "enabled": ["trend_following", "mean_reversion"],
    "timeframes": ["5m", "15m", "1h"]
  }
}
```

## 🚢 Deployment

### Option 1: Direct Deployment

```bash
./deploy_to_pi.sh
```

### Option 2: GitHub Actions (Recommended)

1. Fork this repository
2. Add secrets to your GitHub repository:
   - `PI_HOST`: Your Raspberry Pi IP
   - `PI_USER`: SSH username
   - `PI_SSH_KEY`: SSH private key
3. Push to main branch to trigger deployment

### Option 3: Docker

```bash
docker-compose up -d
```

## 📈 Performance Expectations

With $1,000 initial capital:
- **Monthly**: 3-8% ($30-80)
- **Annual**: 40-100% ($400-1000)
- **Max Drawdown**: 10% ($100)

## 🛡️ Security

- API keys stored securely in environment variables
- No keys in code or logs
- IP whitelist on exchange
- 2FA recommended on exchange account

## 📊 Monitoring and Alerts

Configure Telegram notifications:
```bash
cp telegram_config.example.json telegram_config.json
# Add your bot token and chat ID
```

Receive alerts for:
- New positions opened/closed
- Daily P&L summary
- System errors
- Risk limit warnings

## 💰 Tax Features

Automatic tax calculation and optimization:

```bash
# Real-time tax monitoring
python scripts/tax_dashboard.py

# Generate annual tax report
python scripts/generate_tax_report.py --year 2024

# Estimate quarterly payments
python scripts/generate_tax_report.py --estimate
```

Features:
- **Multi-jurisdiction support** (USA, Spain, Germany, UK, etc.)
- **Real-time tax impact** before closing positions
- **Tax loss harvesting** suggestions
- **Form 8949** export for US taxes
- **TurboTax** compatible exports

[See full tax documentation](docs/TAX_FEATURES.md)

## 🔍 Troubleshooting

Check logs:
```bash
tail -f logs/trader.log
```

System status:
```bash
systemctl status trader
```

Health check:
```bash
python scripts/health_check.py
```

## 📝 License

MIT License - see LICENSE file for details

## ⚠️ Disclaimer

Trading cryptocurrencies carries risk. This software is provided as-is without warranty. Always test with small amounts first and never invest more than you can afford to lose.