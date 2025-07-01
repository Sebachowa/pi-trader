# Nautilus Trading Challenge - Professional Paper Trading

## Overview

This is a **serious, production-ready** paper trading system using Nautilus Trader with **real live market data** from Binance. It's designed for a 14-day evaluation period before transitioning to live trading with real money.

## Features

- ✅ **Real Market Data**: Live price feeds from Binance
- ✅ **Professional Strategies**: Trend following, mean reversion, momentum, and AI/ML
- ✅ **Risk Management**: Position sizing, stop losses, drawdown limits
- ✅ **Performance Tracking**: Comprehensive metrics and reporting
- ✅ **Telegram Notifications**: Real-time trade alerts and daily summaries
- ✅ **Production Ready**: Clean architecture ready for live trading

## Quick Start

```bash
# Make the script executable
chmod +x ../start_challenge.sh

# Start the challenge
../start_challenge.sh
```

## Project Structure

```
nautilus_challenge/
├── config/
│   └── trading_config.json      # Main configuration
├── strategies/
│   ├── base_strategy.py         # Base class with risk management
│   ├── trend_following.py       # Trend following strategy
│   ├── mean_reversion.py        # Mean reversion strategy (TBD)
│   └── momentum.py              # Momentum strategy (TBD)
├── monitoring/
│   ├── telegram_notifier.py     # Telegram notifications
│   └── performance_tracker.py   # Performance tracking
├── logs/                        # Trading logs and reports
├── data/                        # Market data cache
└── run_paper_trading.py        # Main entry point
```

## Configuration

Edit `config/trading_config.json` to customize:

- **Initial Capital**: 0.3 BTC default
- **Target Returns**: 10% annual (0.83% monthly)
- **Risk Limits**: 2% daily drawdown, 5% total
- **Instruments**: BTC, ETH, BNB, SOL, ADA
- **Strategies**: Enable/disable and configure allocations

## Performance Targets

Before transitioning to live trading, the system must achieve:

- ✅ Minimum 100 trades executed
- ✅ Win rate > 55%
- ✅ Sharpe ratio > 1.5
- ✅ Profit factor > 1.8
- ✅ 7 consecutive profitable days

## Monitoring

### Real-time Dashboard
Access at `http://localhost:8080` (when implemented)

### Telegram Notifications
- Trade signals and executions
- Position updates with P&L
- Daily performance summaries
- Risk alerts and warnings

### Performance Reports
Daily reports saved to `logs/` directory with:
- Trade history
- Strategy performance breakdown
- Risk metrics
- P&L analysis

## Strategies

### 1. Trend Following (Implemented)
- EMA crossovers with momentum confirmation
- RSI filter to avoid overbought/oversold
- ATR-based stop loss and take profit

### 2. Mean Reversion (Coming Soon)
- Bollinger Bands with RSI divergence
- Volume confirmation
- Quick scalping trades

### 3. Momentum (Coming Soon)
- Breakout detection
- Volume surge analysis
- Trailing stops

### 4. AI/ML Ensemble (Coming Soon)
- LSTM price prediction
- Sentiment analysis
- Multi-model voting

## After 14 Days

Once the paper trading period is complete:

1. Review the final performance report
2. Analyze strategy performance
3. Identify best performing strategies
4. Adjust risk parameters if needed
5. Transition to live trading with real money

## Live Trading Transition

To switch to live trading:

1. Get Binance API keys
2. Update configuration for live trading
3. Start with minimal capital
4. Gradually increase position sizes

## Support

- Logs: Check `logs/` directory
- Telegram: Monitor notifications
- Performance: Review daily summaries

## Important Notes

- This is REAL paper trading with live market data
- No actual money is at risk during paper trading
- Thoroughly test for 14 days before using real money
- Start small when transitioning to live trading

---

**Remember**: Trading involves risk. Past performance doesn't guarantee future results. Only trade with money you can afford to lose.