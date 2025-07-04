# 🚀 Getting Started - No API Keys Required!

## Quick Start (3 minutes)

### 1. Test Locally in Demo Mode

```bash
# Option A: Use the interactive launcher
./start.sh
# Select option 1 (Demo Mode)

# Option B: Direct command
python run.py --demo
```

You'll see:
```
🤖 Raspberry Pi Trading Bot Starting
📊 Mode: DEMO (No API keys required)
🔍 Scanning markets...
📈 Opportunity found: BTC/USDT @ $45,234.00
✅ Opened position: BTC/USDT
```

### 2. Deploy to Raspberry Pi

```bash
# Commit and push to GitHub
git add .
git commit -m "Initial setup with placeholders"
git push origin main

# GitHub Actions will deploy automatically!
```

### 3. Run on Raspberry Pi

SSH to your Pi:
```bash
ssh pi@your-pi-ip
cd ~/trader
./start.sh
# Select option 1 (Demo Mode)
```

## 📝 When You Get API Keys

### Step 1: Get Binance API Keys

1. **For Testing (Recommended First)**:
   - Go to https://testnet.binance.vision/
   - Register (no KYC needed)
   - Generate API keys

2. **For Real Trading**:
   - Go to https://www.binance.com/en/my/settings/api-management
   - Create API with:
     - ✅ Enable Reading
     - ✅ Enable Spot Trading
     - ❌ Disable Withdrawals

### Step 2: Update Your .env

```bash
# Edit .env file
nano .env

# Replace placeholders:
BINANCE_API_KEY=your_real_api_key_here
BINANCE_API_SECRET=your_real_secret_here
BINANCE_TESTNET=true  # or false for real trading
```

### Step 3: Test with Real Data

```bash
# Paper trading with real market data
python run.py --paper

# When ready for real trading
python run.py
```

## 🎯 Development Workflow

### Local Testing
```bash
# 1. Demo mode (no API needed)
python run.py --demo

# 2. Test specific components
python scripts/test_scanner.py

# 3. Check configuration
python scripts/check_config.py
```

### Deployment
```bash
# Automatic deployment on push
git push origin main

# Manual deployment
./deploy_to_pi.sh
```

## 📊 Monitoring

### View Logs
```bash
# Local
tail -f logs/trader_*.log

# On Raspberry Pi
ssh pi@your-pi-ip
sudo journalctl -u trader -f
```

### Tax Dashboard (works in demo too!)
```bash
python scripts/tax_dashboard.py
```

## ⚠️ Important Notes

1. **Demo Mode Limitations**:
   - Uses simulated prices
   - No real market data
   - Perfect for testing deployment

2. **Placeholder Values**:
   - `PLACEHOLDER_API_KEY_12345` - Not a real key
   - Safe to commit to git
   - Won't work for real trading

3. **Security**:
   - Never commit real API keys
   - Always use .env for secrets
   - Check before pushing: `python scripts/check_config.py`

## 🆘 Troubleshooting

### "Invalid configuration" Error
```bash
# Run in demo mode
DEMO_MODE=true python run.py

# Or use the flag
python run.py --demo
```

### Can't Install Dependencies
```bash
# Use system packages on Pi
sudo apt install python3-numpy python3-pandas
pip install --no-deps ccxt
```

### Service Won't Start on Pi
```bash
# Check logs
sudo journalctl -u trader -n 50

# Run manually first
cd ~/trader
python run.py --demo
```

## 🎉 Next Steps

1. ✅ Run demo mode locally
2. ✅ Deploy to Raspberry Pi
3. ✅ Get Binance testnet API keys
4. ✅ Test paper trading
5. 🚀 Go live when ready!

Happy Trading! 🤖💰