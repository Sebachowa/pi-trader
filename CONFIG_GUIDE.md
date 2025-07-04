# 🔧 Configuration Guide

## Overview

This trading bot uses a two-tier configuration system:
1. **config.json** - Non-sensitive default settings
2. **.env file** - Sensitive data like API keys (never committed to git)

## Quick Start

### 1. Create your .env file
```bash
cp .env.example .env
nano .env
```

### 2. Add your API keys
```bash
# Exchange API Configuration
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_API_SECRET=your_actual_secret_here
BINANCE_TESTNET=true  # Use false for real trading
```

### 3. Run the bot
```bash
python run.py
```

## Configuration Priority

Environment variables (.env) **always override** config.json values for sensitive data.

```python
# Example flow:
1. Load config.json
2. Load .env file
3. .env values override config.json for:
   - API keys
   - Secrets
   - Passwords
```

## Available Environment Variables

### Required
- `BINANCE_API_KEY` - Your exchange API key
- `BINANCE_API_SECRET` - Your exchange API secret

### Optional
- `BINANCE_TESTNET` - Use testnet (default: true)
- `TELEGRAM_BOT_TOKEN` - For notifications
- `TELEGRAM_CHAT_ID` - Your Telegram chat ID
- `TAX_JURISDICTION` - Tax jurisdiction (default: USA)
- `TAX_YEAR` - Tax year (default: current year)
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

## Security Best Practices

### ✅ DO:
- Keep .env file locally only
- Use different .env for development/production
- Rotate API keys regularly
- Use read-only API keys when possible

### ❌ DON'T:
- Commit .env to git
- Share your .env file
- Use same API keys for multiple bots
- Store .env in cloud without encryption

## Deployment

### For Raspberry Pi:
```bash
# Copy .env to your Pi (one time)
scp .env pi@raspberry-ip:~/trader/

# Or create it directly on Pi
ssh pi@raspberry-ip
cd ~/trader
nano .env
# Add your keys
```

### For Docker:
```bash
# Pass env file to Docker
docker run --env-file .env trader

# Or use docker-compose
docker-compose up
```

## Troubleshooting

### "Invalid configuration" error
- Check .env file exists
- Verify API keys are set
- No quotes needed in .env file

### "API key invalid" error
- Verify keys are correct
- Check testnet vs mainnet
- Ensure no extra spaces

## Example Configurations

### Development (.env.dev)
```bash
BINANCE_API_KEY=testnet_key_here
BINANCE_API_SECRET=testnet_secret_here
BINANCE_TESTNET=true
LOG_LEVEL=DEBUG
```

### Production (.env.prod)
```bash
BINANCE_API_KEY=real_api_key_here
BINANCE_API_SECRET=real_secret_here
BINANCE_TESTNET=false
LOG_LEVEL=INFO
```

### Tax Tracking for Different Countries
```bash
# USA
TAX_JURISDICTION=USA
TAX_YEAR=2024

# Spain
TAX_JURISDICTION=SPAIN
TAX_YEAR=2024

# Germany (tax-free after 1 year!)
TAX_JURISDICTION=GERMANY
TAX_YEAR=2024
```