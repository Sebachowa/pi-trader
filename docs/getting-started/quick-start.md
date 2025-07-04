# 🚀 Quick Start Guide

Get your trading bot running in 5 minutes!

## 📋 Prerequisites

- Python 3.9+ installed
- ~200MB free space
- Internet connection

## 🎮 Option 1: Demo Mode (No API Keys)

Perfect for trying out the bot without any setup:

```bash
# Clone and setup
git clone https://github.com/Sebachowa/pi-trader.git
cd pi-trader
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
./setup_pi.sh  # Or: pip install -r requirements-pi.txt

# Run demo (no API keys needed)
python run.py --demo
```

You'll see simulated trading with fake data!

## 🧪 Option 2: Testnet Mode (Safe Testing)

For realistic testing with Binance testnet:

### 1. Get Testnet API Keys
1. Go to [testnet.binance.vision](https://testnet.binance.vision/)
2. Create a free test account
3. Generate API keys

### 2. Configure
```bash
cp .env.example .env
nano .env  # Or use any text editor
```

Add your testnet keys:
```bash
BINANCE_API_KEY=your_testnet_key_here
BINANCE_API_SECRET=your_testnet_secret_here
BINANCE_TESTNET=true
```

### 3. Run
```bash
python run.py
```

## 💰 Option 3: Live Trading (Real Money)

⚠️ **Only after testing thoroughly with testnet!**

1. Get real Binance API keys from [binance.com](https://www.binance.com/en/my/settings/api-management)
2. Enable only: ✅ Read, ✅ Spot Trading (NO withdrawals)
3. Update `.env`:
   ```bash
   BINANCE_TESTNET=false
   ```

## 📊 What You'll See

Beautiful colored logs with emojis:

```
12:38:32 📝  INFO     [engine      ] 🚀 Raspberry Pi Trading Bot Starting
12:38:33 📝 ⚙️  INFO     [monitor     ] ⚙️  System: CPU 15.5%, RAM 45.2% | Positions: 0 | Equity: $10,000.00
12:38:35 📝 🔍  INFO     [scanner     ] 🔍 Scan completed in 8.45s, found 3 opportunities 🎯
12:38:35 📝 💡  INFO     [scanner     ] 💡 OPPORTUNITY FOUND! BTC/USDT - trend_following (score: 85.5)
12:38:36 📝 🎯  INFO     [engine      ] 🎯 Trend Following signal: 🟢 BUY (confidence: 85.5%)
12:38:37 📝 💰  INFO     [engine      ] 💰 TRADE OPENED: BTC/USDT - Size: 0.0125 @ $108,950.50
```

## 🔧 Command Options

```bash
# Demo mode (no API keys)
python run.py --demo

# Testnet/live trading
python run.py

# Paper trading (market data, simulated trades)
python run.py --paper

# Different log levels
python run.py --log-level DEBUG  # More details
python run.py --log-level WARNING  # Less verbose

# Help
python run.py --help
```

## 🔍 Testing Your Setup

### 1. Check Dependencies
```bash
python -c "import ccxt; print('✓ ccxt installed')"
python -c "import colorama; print('✓ colorama installed')"
```

### 2. Test Connection
```bash
python test_binance_testnet.py
```

Should show:
```
✅ Testnet connection successful!
💰 USDT Balance: 10,000.00
```

## 🚨 Troubleshooting

### "No module named X"
```bash
pip install -r requirements-pi.txt
```

### "API key invalid"
- Check you copied keys correctly
- For testnet: make sure `BINANCE_TESTNET=true`
- No quotes needed in .env file

### "Scanner finds no opportunities"
Normal for testnet! Edit `config/config.json`:
```json
{
  "scanner": {
    "min_opportunity_score": 30
  }
}
```

### Connection issues
- Check internet connection
- Firewall might block connections
- Try from different network

## 🎯 Next Steps

Once running successfully:

1. **Watch for 30 minutes** - See how it behaves
2. **Check the logs** - Understand what it's doing
3. **Review [Configuration Guide](configuration.md)** - Customize settings
4. **Try [Deployment Guide](deployment.md)** - Set up on Raspberry Pi

## 💡 Pro Tips

- Start with demo mode to learn the interface
- Use testnet for realistic testing
- Monitor CPU/RAM usage with `htop`
- Check logs in `logs/` directory
- Stop with `Ctrl+C`

---

**Need help?** Check the other guides or create an issue on GitHub!