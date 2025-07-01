# Nautilus Autonomous Trading System (NATS)

A fully autonomous trading system built on Nautilus Trader, designed for 24/7 operation with minimal user intervention while targeting 10% annual returns.

## Overview

The Nautilus Autonomous Trading System (NATS) is a comprehensive autonomous trading solution that combines:

- **Self-Healing Mechanisms**: Automatic error recovery and connection management
- **Dynamic Risk Management**: Auto-adjusting position sizes based on market conditions
- **Multi-Strategy Orchestration**: AI-powered strategy selection and allocation
- **Performance Optimization**: Machine learning-based parameter tuning
- **Real-Time Monitoring**: Comprehensive alerts and notifications

## Key Features

### 1. Autonomous Engine
- 24/7 operation with automatic startup/shutdown
- Self-healing error recovery (up to 3 attempts)
- Health monitoring and circuit breakers
- State persistence and recovery
- Scheduled maintenance windows

### 2. Risk Controller
- Dynamic position sizing based on:
  - Market volatility
  - Recent performance
  - Portfolio correlation
- Multiple risk limits:
  - Max daily loss: 2%
  - Max drawdown: 10%
  - Max position risk: 1%
  - Max portfolio risk: 5%
- VaR calculations and stress testing
- Emergency stop mechanisms

### 3. Market Analyzer
- Real-time market regime detection:
  - Trending (up/down)
  - Ranging
  - Volatile
  - Breakout/Breakdown
- Liquidity analysis
- Volatility monitoring
- Anomaly detection
- Multi-timeframe analysis

### 4. Strategy Orchestrator
- Dynamic strategy allocation
- Performance-based weighting
- Market condition-based selection
- Strategy health monitoring
- Automatic rotation and rebalancing

### 5. Performance Optimizer
- Bayesian optimization for parameters
- Real-time performance tracking
- Attribution analysis
- Machine learning optimization
- Walk-forward analysis

### 6. Notification System
- Multi-channel support:
  - Email
  - SMS
  - Webhooks
  - Telegram
  - Slack
  - Discord
- Configurable alert rules
- Daily summaries
- Weekly reports
- Rate limiting

## Installation

```bash
# Ensure Nautilus Trader is installed
pip install nautilus-trader

# Install additional dependencies
pip install scikit-learn scipy aiohttp
```

## Quick Start

### 1. Basic Usage

```bash
# Run with default configuration (paper trading)
python autonomous_trading/run_autonomous.py --paper

# Run with specific instruments
python autonomous_trading/run_autonomous.py --instruments BTCUSDT.BINANCE ETHUSDT.BINANCE

# Run with custom risk parameters
python autonomous_trading/run_autonomous.py --target-return 15 --max-drawdown 15
```

### 2. Configuration File

Create a configuration file `config.json`:

```json
{
  "system_name": "NATS-001",
  "trader_id": "AUTONOMOUS-001",
  "target_annual_return_percent": 10.0,
  "risk_limits": {
    "max_daily_loss_percent": 2.0,
    "max_drawdown_percent": 10.0,
    "max_position_risk_percent": 1.0
  },
  "instruments": [
    "BTCUSDT.BINANCE",
    "ETHUSDT.BINANCE",
    "EURUSD.IDEALPRO"
  ],
  "notification_config": {
    "email_config": {
      "sender_email": "trading@example.com",
      "recipient_emails": ["alerts@example.com"]
    }
  }
}
```

Run with configuration:
```bash
python autonomous_trading/run_autonomous.py --config config.json
```

### 3. Environment Variables

Set exchange credentials:
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Autonomous Engine                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Health    │  │  Self-Heal   │  │  State Mgmt  │  │
│  │  Monitor    │  │  Recovery    │  │  Persistence │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌───────▼────────┐ ┌──────▼─────────┐
│     Risk       │ │    Market      │ │   Strategy     │
│  Controller    │ │   Analyzer     │ │ Orchestrator   │
│                │ │                │ │                │
│ • Dynamic Size │ │ • Regime Detect│ │ • Multi-Strat  │
│ • Correlation  │ │ • Volatility   │ │ • AI Selection │
│ • VaR/Limits   │ │ • Liquidity    │ │ • Rebalancing  │
└────────────────┘ └────────────────┘ └────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐                    ┌──────▼─────────┐
│  Performance   │                    │  Notification  │
│   Optimizer    │                    │    System      │
│                │                    │                │
│ • Bayesian Opt │                    │ • Multi-Channel│
│ • ML Tuning    │                    │ • Alerts       │
│ • Attribution  │                    │ • Reports      │
└────────────────┘                    └────────────────┘
```

## Strategies

The system supports multiple strategy types:

1. **AI Swarm Strategy**: Multi-agent consensus trading
2. **Trend Following**: Momentum-based strategies
3. **Market Making**: Liquidity provision
4. **Statistical Arbitrage**: Cross-exchange opportunities
5. **Mean Reversion**: Range-bound trading

## Risk Management

Multiple layers of risk control:

1. **Position Level**: Dynamic sizing based on volatility and performance
2. **Portfolio Level**: Correlation limits and exposure management
3. **System Level**: Circuit breakers and emergency stops
4. **Operational Level**: Health monitoring and self-healing

## Performance Targets

- **Annual Return**: 10% (configurable)
- **Maximum Drawdown**: 10%
- **Sharpe Ratio**: > 1.0
- **Win Rate**: > 50%
- **Daily Loss Limit**: 2%

## Monitoring

### Real-Time Metrics
- System health status
- Position exposure
- P&L tracking
- Risk metrics (VaR, drawdown)
- Strategy performance

### Notifications
- Critical errors
- Risk limit breaches
- Daily summaries
- Weekly reports
- Performance alerts

## Advanced Features

### Self-Optimization
- Bayesian parameter tuning
- Performance-based strategy weighting
- Dynamic risk adjustment
- Market regime adaptation

### Failsafe Mechanisms
- Automatic position closing on errors
- Connection recovery
- State restoration
- Graceful degradation

## Production Deployment

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- Stable internet connection
- Unix-based OS recommended

### Deployment Checklist
1. ✅ Configure exchange API credentials
2. ✅ Set up notification channels
3. ✅ Configure risk parameters
4. ✅ Test in paper trading mode
5. ✅ Set up monitoring alerts
6. ✅ Configure automatic startup
7. ✅ Enable state persistence
8. ✅ Set up backup procedures

### Running as a Service

Create systemd service file `/etc/systemd/system/nautilus-autonomous.service`:

```ini
[Unit]
Description=Nautilus Autonomous Trading System
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/path/to/nautilus_trader
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 /path/to/autonomous_trading/run_autonomous.py --config /path/to/config.json
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable nautilus-autonomous
sudo systemctl start nautilus-autonomous
```

## Safety Guidelines

⚠️ **Important**: Always test thoroughly in paper trading before live deployment

1. Start with minimal capital allocation
2. Monitor closely during first week
3. Set conservative risk limits initially
4. Enable all notification channels
5. Have manual override procedures ready
6. Regular backup of state files
7. Monitor system resources

## Troubleshooting

### Common Issues

1. **Connection Lost**
   - System will auto-reconnect up to 3 times
   - Check exchange API status
   - Verify network connectivity

2. **Risk Limits Hit**
   - System will pause trading
   - Review recent performance
   - Adjust parameters if needed

3. **High CPU/Memory Usage**
   - Check number of active strategies
   - Review data processing load
   - Consider scaling resources

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review system status report
3. Consult Nautilus Trader documentation
4. Submit issues on GitHub

## License

Licensed under GNU Lesser General Public License v3.0