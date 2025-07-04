# Beautiful Logging Guide 🌈

## Emoji Reference

### Log Levels
- 🐛 **DEBUG** - Detailed debugging information
- 📝 **INFO** - General information
- ⚠️ **WARNING** - Warning messages
- ❌ **ERROR** - Error messages
- 🔥 **CRITICAL** - Critical errors

### Trading Events
- 💡 **Opportunity Found** - Scanner found a trading opportunity
- 🎯 **Strategy Signal** - Strategy generated a buy/sell signal
- 🟢 **BUY Signal** - Buy action recommended
- 🔴 **SELL Signal** - Sell action recommended
- ⚡ **Executing Order** - Order being placed
- 💰 **Trade Opened** - New position opened
- 💸 **Profit** - Trade closed with profit
- 📉 **Loss** - Trade closed with loss
- 📊 **Position Update** - Current position status

### System Events
- 🚀 **Startup** - Bot starting
- 🛑 **Shutdown** - Bot stopping
- ⚙️ **System Status** - CPU/RAM/Disk usage
- 🔍 **Scanner** - Market scanning activity
- 👁️ **Monitoring** - System monitoring
- 🌐 **Network** - Network operations
- 💳 **Balance** - Balance updates
- 🧪 **Testnet** - Testnet mode active

### Special Highlights
- **Percentages** appear in yellow: `15.5%`
- **Dollar amounts** appear in green: `$10,000.50`
- **Scores** appear in magenta: `score: 85.5`

## Example Log Messages

```
12:38:32 📝  INFO     [engine      ] 🚀 Raspberry Pi Trading Bot Starting
12:38:33 📝 ⚙️  INFO     [monitor     ] ⚙️  System: CPU 15.5%, RAM 45.2% | Positions: 0 | Equity: $10,000.00
12:38:35 📝 🔍  INFO     [scanner     ] 🔍 Scan completed in 8.45s, found 3 opportunities 🎯
12:38:35 📝 💡  INFO     [scanner     ] 💡 OPPORTUNITY FOUND! BTC/USDT - trend_following (score: 85.5)
12:38:36 📝 🎯  INFO     [engine      ] 🎯 Trend Following signal: 🟢 BUY (confidence: 85.5%)
12:38:36 📝 ⚡  INFO     [engine      ] ⚡ Executing BUY order: 0.0125 BTC/USDT
12:38:37 📝 💰  INFO     [engine      ] 💰 TRADE OPENED: BTC/USDT - Size: 0.0125 @ $108,950.50
```

## Quick Tips

1. **Follow the Flow**: 
   - 💡 Opportunity → 🎯 Signal → ⚡ Execution → 💰 Trade Opened

2. **Monitor Health**:
   - Watch for ⚠️ warnings
   - Address ❌ errors immediately
   - ⚙️ system status shows resource usage

3. **Track Performance**:
   - 💸 = Profitable trades
   - 📉 = Losing trades
   - 📊 = Position updates

4. **Scanner Activity**:
   - 🔍 with "found 0" = No opportunities
   - 🔍 with "found X" + 🎯 = Opportunities detected

## Configuration

To adjust log level, use:
```bash
python run.py --log-level DEBUG  # More details
python run.py --log-level INFO   # Normal (default)
python run.py --log-level WARNING # Less verbose
```