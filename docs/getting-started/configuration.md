# 🔧 Configuration Guide

Complete guide to configuring your trading bot.

## 📁 Configuration Files

The bot uses two configuration files:

1. **`config/config.json`** - Trading parameters, strategies, risk settings
2. **`.env`** - Sensitive data like API keys (never committed to git)

## 🔑 Environment Variables (.env)

### Quick Setup
```bash
# Copy template
cp .env.example .env

# Edit with your details
nano .env
```

### Required Variables
```bash
# Exchange API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true  # true = testnet, false = live trading
```

### Optional Variables
```bash
# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=123456789:ABCDEF...
TELEGRAM_CHAT_ID=123456789

# Tax Configuration
TAX_JURISDICTION=USA  # USA, EU, UK, etc.
TAX_YEAR=2024

# System
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
MAX_POSITIONS=3
RISK_LEVEL=moderate  # conservative, moderate, aggressive
```

## ⚙️ Trading Configuration (config.json)

### Exchange Settings
```json
{
  "exchange": {
    "name": "binance",
    "testnet": true  // Overridden by BINANCE_TESTNET in .env
  }
}
```

### Trading Parameters
```json
{
  "trading": {
    "max_positions": 3,           // Maximum concurrent trades
    "position_size_pct": 0.1,     // 10% of balance per trade
    "max_daily_loss_pct": 0.02,   // Stop trading if 2% daily loss
    "stop_loss_pct": 0.02,        // 2% stop loss
    "take_profit_pct": 0.05,      // 5% take profit
    "leverage": 1                 // 1x leverage (spot trading)
  }
}
```

### Risk Management
```json
{
  "risk": {
    "max_drawdown_pct": 0.1,      // Max 10% portfolio drawdown
    "max_position_size_usd": 1000, // Max $1000 per position
    "min_volume_24h": 100000,     // Min $100k daily volume
    "cooldown_minutes": 15        // Wait 15min between trades on same pair
  }
}
```

### Scanner Settings
```json
{
  "scanner": {
    "interval_seconds": 30,        // Scan every 30 seconds
    "min_volume_24h": 100000,      // Filter low-volume pairs
    "min_opportunity_score": 40,   // Minimum score to trade (0-100)
    "max_concurrent_scans": 50,    // Max symbols scanned at once
    "top_volume_count": 100,       // Focus on top 100 by volume
    "blacklist": ["BUSD/USDT"]     // Pairs to avoid
  }
}
```

### Strategies
```json
{
  "strategies": {
    "enabled": ["trend_following", "mean_reversion"],
    "timeframes": ["15m", "1h"],   // Analyze multiple timeframes
    "default_lookback": 100        // Bars of historical data
  }
}
```

### Monitoring
```json
{
  "monitoring": {
    "update_interval_seconds": 30, // System stats frequency
    "log_level": "INFO",           // Overridden by LOG_LEVEL in .env
    "enable_notifications": false,
    "webhook_url": ""              // For custom alerts
  }
}
```

## 🎯 Configuration Presets

### Conservative (Low Risk)
```json
{
  "trading": {
    "position_size_pct": 0.05,     // 5% per trade
    "max_daily_loss_pct": 0.01     // 1% daily limit
  },
  "scanner": {
    "min_opportunity_score": 70    // Higher quality trades only
  }
}
```

### Aggressive (Higher Risk)
```json
{
  "trading": {
    "position_size_pct": 0.15,     // 15% per trade
    "max_daily_loss_pct": 0.05     // 5% daily limit
  },
  "scanner": {
    "min_opportunity_score": 40    // More trades
  }
}
```

### Testnet Optimized
```json
{
  "scanner": {
    "min_opportunity_score": 30,   // Lower threshold for testnet
    "min_volume_24h": 10000        // Less volume requirement
  }
}
```

## 🛡️ Security Best Practices

### API Key Setup
1. **Use dedicated keys** - Create separate keys for the bot
2. **Minimal permissions** - Enable only Read + Spot Trading
3. **No withdrawals** - Never enable withdrawal permissions
4. **IP restrictions** - Restrict to your IP if possible

### Environment File Security
```bash
# ✅ Good practices:
chmod 600 .env          # Restrict file permissions
git status              # Verify .env not tracked
ls -la .env             # Check ownership

# ❌ Never do:
git add .env            # Don't commit to git
cat .env > backup.txt   # Don't create plaintext copies
chmod 777 .env          # Don't make world-readable
```

### Production Security
```bash
# Use environment variables instead of .env file
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
python run.py

# Or use systemd with EnvironmentFile
# See deployment guide for details
```

## 🔄 Configuration Updates

### Live Reload
Some settings can be updated without restart:
- Scanner frequency
- Risk limits
- Log levels

### Restart Required
Changes to these require bot restart:
- API keys
- Exchange settings
- Strategy selection

### Update Process
```bash
# 1. Stop bot
Ctrl+C

# 2. Edit configuration
nano config/config.json

# 3. Restart
python run.py
```

## 🧪 Testing Configurations

### Validate Settings
```bash
# Test configuration loading
python -c "from core.config_loader import ConfigLoader; print('✓ Config valid')"

# Test API connection
python test_binance_testnet.py
```

### Monitor Changes
```bash
# Watch performance with new settings
tail -f logs/trader_*.log | grep "opportunities\|opened\|closed"
```

## 🚨 Common Issues

### Bot doesn't find opportunities
- Lower `min_opportunity_score` (try 30-50)
- Increase `max_concurrent_scans` (try 100)
- Check `min_volume_24h` isn't too high

### Trades aren't executing
- Check API key permissions
- Verify sufficient balance
- Review risk limits
- Check cooldown periods

### High CPU usage
- Reduce `max_concurrent_scans`
- Increase `interval_seconds`
- Limit `top_volume_count`

### Memory issues
- Reduce `default_lookback` (try 50)
- Limit concurrent positions
- Restart bot daily (systemd timer)

## 📊 Performance Tuning

### For Raspberry Pi
```json
{
  "scanner": {
    "max_concurrent_scans": 25,    // Reduce CPU load
    "interval_seconds": 45         // Less frequent scans
  },
  "strategies": {
    "default_lookback": 50         // Less memory usage
  }
}
```

### For High-End Systems
```json
{
  "scanner": {
    "max_concurrent_scans": 100,   // More parallelism
    "interval_seconds": 15,        // Faster scanning
    "top_volume_count": 200        // More symbols
  }
}
```

---

**Next:** [Deployment Guide](deployment.md) - Deploy to production