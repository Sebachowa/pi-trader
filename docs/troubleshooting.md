# 🔧 Troubleshooting Guide

Complete guide to diagnose and fix common issues with the trading bot.

## 🚨 Bot Not Finding Opportunities

This is the most common issue, especially on testnet.

### Quick Check
```bash
# Run from your Raspberry Pi
cd ~/trading-bot

# Quick status check
./scripts/utils/quick_check.sh

# Check complete history
./scripts/utils/check_history.sh

# Monitor live activity
python scripts/utils/monitor_live.py

# Analyze all logs
python scripts/utils/analyze_logs.py --all
```

### Common Causes & Solutions

#### 1. Score Threshold Too High
**Problem**: Default score of 70 is too high for testnet
```bash
# Check current setting
grep min_opportunity_score config/config.json

# Fix: Edit config.json
nano config/config.json
# Change to: "min_opportunity_score": 30
```

#### 2. Wrong Scanner Mode
**Problem**: Using mainnet scanner on testnet
```bash
# Check logs for scanner type
grep "scanner initialized" logs/trader_*.log | tail -1

# Should see: "Testnet scanner initialized"
# If not, check testnet setting:
grep testnet config/config.json
# Should be: "testnet": true
```

#### 3. Low Market Activity
**Problem**: Testnet has less volatility
```bash
# Check recent market activity
python scripts/utils/debug_scanner.py

# Shows actual price movements and scores
```

#### 4. Volume Requirements Too Strict
**Problem**: Testnet has lower volumes
```bash
# Check volume settings
grep min_volume_24h config/config.json

# Fix: Lower to 10000 or 100000
```

## 📊 Viewing Historical Data

### View All Logs
```bash
# See all opportunities ever found
grep "OPPORTUNITY FOUND" logs/*.log

# Count opportunities by day
for f in logs/*.log; do 
    echo "$f: $(grep -c "OPPORTUNITY FOUND" $f)"
done

# See all trades executed
grep "TRADE OPENED" logs/*.log
```

### Analyze Specific Time Period
```bash
# Last hour of activity
python scripts/utils/analyze_logs.py --latest

# Today's activity
grep "$(date +%Y-%m-%d)" logs/trader_*.log | grep "OPPORTUNITY"
```

### Real-time Monitoring
```bash
# Watch live logs
tail -f logs/trader_*.log | grep --color -E "OPPORTUNITY|TRADE|ERROR|WARNING"

# Monitor with analysis
python scripts/utils/monitor_live.py
```

## 🐛 Debugging Tools

### 1. Test Scanner Directly
```bash
# Test scanner with lower thresholds
python scripts/utils/debug_scanner.py

# Test specific symbol
python -c "
from scripts.utils.debug_scanner import debug_scanner
debug_scanner()
"
```

### 2. Check API Connection
```bash
# Test Binance connection
python tests/test_binance_testnet.py

# Should show balance and market data
```

### 3. Force Opportunity Detection
```bash
# Temporarily lower all thresholds
cp config/config.json config/config.backup.json

# Edit config with very low thresholds
nano config/config.json
# Set: 
# "min_opportunity_score": 20
# "min_volume_24h": 1000

# Restart bot and watch
```

## 📈 Understanding Scores

### Why Scores Are Low on Testnet

1. **Low Volatility**: Moves are often <0.5% (need 2%+)
2. **Low Volume**: Volume ratios rarely exceed 1.5x (need 3x+)
3. **Few Traders**: Less trend development

### Score Calculation
```
Trend Following: Base 50 + trend_strength
Mean Reversion: Base 40 + deviation
Momentum: Base 40 + move_percentage * 500
Volume: Base 30 + volume_ratio * 10
```

## 🔍 Step-by-Step Diagnosis

### 1. Check Bot Is Running
```bash
ps aux | grep run.py
# Should see python process
```

### 2. Check Logs Exist
```bash
ls -la logs/
# Should see trader_*.log files
```

### 3. Check Recent Scans
```bash
tail -50 logs/trader_*.log | grep "Scan completed"
# Should see scans every 30-60 seconds
```

### 4. Check for Errors
```bash
grep -i error logs/trader_*.log | tail -20
# Look for API errors, connection issues
```

### 5. Verify Configuration
```bash
# Check all settings
python -m json.tool config/config.json

# Verify API keys loaded
grep -E "API key|testnet" logs/trader_*.log | tail -5
```

## 💡 Quick Fixes

### Force More Opportunities
```json
// config/config.json - Aggressive testnet settings
{
  "scanner": {
    "min_opportunity_score": 25,
    "min_volume_24h": 10000,
    "interval_seconds": 30
  },
  "trading": {
    "position_size_pct": 0.05  // Smaller positions
  }
}
```

### Reset and Restart
```bash
# Stop bot
pkill -f run.py

# Clear old logs (optional)
mkdir -p logs/archive
mv logs/*.log logs/archive/

# Start fresh
python run.py
```

### Monitor for 1 Hour
```bash
# Let it run and collect data
sleep 3600

# Then analyze
python scripts/utils/analyze_logs.py --latest
```

## 🆘 Still No Opportunities?

### Last Resort Checklist
1. ✅ Verify testnet has balance (10,000 USDT)
2. ✅ Try mainnet with tiny amounts ($10)
3. ✅ Run during high volatility (US market hours)
4. ✅ Compare with Freqtrade on same settings
5. ✅ Post logs to GitHub issues for help

### Get Help
```bash
# Create diagnostic report
{
    echo "=== CONFIG ==="
    cat config/config.json
    echo -e "\n=== LAST 100 LOGS ==="
    tail -100 logs/trader_*.log
    echo -e "\n=== ANALYSIS ==="
    python scripts/utils/analyze_logs.py --latest
} > diagnostic_report.txt

# Share diagnostic_report.txt in GitHub issue
```

---

**Remember**: Testnet is much quieter than mainnet. It's normal to see fewer opportunities. The bot IS working if you see regular scans in the logs!