# Nautilus Trader Challenge - Monitoring Dashboard

## Overview

The Monitoring Dashboard is a comprehensive web-based interface for real-time monitoring of your Nautilus Trader strategies during the 30-day challenge. It provides all essential metrics, alerts, and analytics to track your progress toward the 10% annual return goal without constant manual checking.

## Features

### Real-Time Metrics
- **P&L Tracking**: Real-time profit/loss in dollars and percentage
- **Balance Monitoring**: Current balance, margin usage, and available funds
- **Position Analysis**: Active positions with entry/exit prices, P&L, and concentration risk
- **Performance Metrics**: Sharpe ratio, win rate, profit factor
- **Risk Monitoring**: Leverage, drawdown, position concentration, risk score

### Challenge Progress Tracking
- Visual progress bar showing advancement toward 10% annual return goal
- Current vs. annualized returns
- Days elapsed and remaining
- Required daily return to meet target
- On-track indicator

### Intelligent Alerts System
- **Risk Alerts**: High leverage, excessive drawdown, position concentration
- **Performance Alerts**: Goal achievement, daily targets met
- **System Alerts**: Connection issues, data feed problems
- Severity levels: CRITICAL, HIGH, WARNING, INFO, SUCCESS
- Alert history with acknowledgment capability

### Interactive Charts
- Performance history (1H, 4H, 24H views)
- P&L and balance trends
- Real-time updates every second

### Position Management View
- All active positions with detailed metrics
- Position sizing as percentage of portfolio
- Individual P&L tracking
- Concentration risk indicators

## Installation

1. Install required dependencies:
```bash
pip install flask flask-cors flask-socketio
```

2. The monitoring dashboard files are located in:
```
paper_trading/
├── monitoring_dashboard.py     # Main dashboard implementation
├── monitoring_server.py        # WebSocket server and utilities
└── run_monitoring_dashboard.py # Example runner script
```

## Usage

### Quick Start

1. **Integrate with your trading system:**
```python
from paper_trading.monitoring_dashboard import create_monitoring_dashboard

# In your trading node setup
dashboard = create_monitoring_dashboard(
    trader_id=trader_id,
    msgbus=node.msgbus,
    clock=node.clock,
    portfolio=node.portfolio,
    host="127.0.0.1",
    port=5000,
)

# Add as actor and start
node.trader.add_actor(dashboard)
dashboard.start()
```

2. **Access the dashboard:**
   - Open your web browser
   - Navigate to: `http://127.0.0.1:5000`
   - Dashboard will automatically start updating

### Standalone Usage

Run the dashboard independently:
```bash
python paper_trading/run_monitoring_dashboard.py
```

### Integration with Existing Strategies

Add monitoring to your existing strategy:

```python
# In your strategy file
from paper_trading.monitoring_dashboard import MonitoringDashboard

class YourStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        
        # Create dashboard
        self.dashboard = MonitoringDashboard(
            trader_id=self.trader_id,
            msgbus=self.msgbus,
            clock=self.clock,
            portfolio=self.portfolio,
        )
    
    def on_start(self):
        # Start dashboard when strategy starts
        self.dashboard.start()
        self.log.info(f"Monitoring dashboard: http://127.0.0.1:5000")
    
    def on_stop(self):
        # Stop dashboard
        self.dashboard.stop()
```

## Dashboard Sections

### 1. Challenge Progress
- **Progress Bar**: Visual representation of progress toward 10% annual return
- **Key Metrics**: Starting balance, current balance, target return
- **Time Tracking**: Days elapsed and remaining in challenge
- **Required Performance**: Daily return needed to meet goal

### 2. Key Metrics Grid
Real-time display of 8 critical metrics:
- Total P&L ($ and %)
- Account Balance
- Risk Level & Score
- Active Positions Count
- Current Leverage
- Maximum Drawdown
- Sharpe Ratio
- Win Rate

### 3. Performance Chart
- Configurable timeframes (1H, 4H, 24H)
- Dual axis showing P&L and balance
- Auto-updating every second
- Interactive tooltips

### 4. Active Positions Table
- Symbol, side (long/short), quantity
- Entry and current prices
- Individual P&L in $ and %
- Position size as % of portfolio
- Sortable columns

### 5. Recent Alerts
- Timestamp, type, severity level
- Alert messages with context
- Color-coded by severity
- Clear functionality

## Risk Management Features

### Risk Limits (Configurable)
- Maximum drawdown: 20%
- Maximum position concentration: 25%
- Maximum daily loss: 5%
- Maximum leverage: 2x

### Risk Monitoring
- **Risk Score**: 0-100 composite score
- **Risk Level**: LOW, MEDIUM, HIGH, CRITICAL
- **Real-time Alerts**: Immediate notification when limits approached
- **Position Concentration**: Warns on overexposure to single positions

## Alert Types

### Risk Alerts
- Leverage exceeds limits
- Drawdown breaches threshold
- Position concentration too high
- Margin usage critical

### Performance Alerts
- Daily profit targets reached
- Challenge goal achieved
- Win rate improvements

### System Alerts
- Connection issues
- Data feed problems
- Strategy errors

## Advanced Features

### WebSocket Support
Real-time updates via WebSocket channels:
- `metrics`: Portfolio and performance updates
- `positions`: Position changes
- `alerts`: New alerts
- `challenge`: Progress updates

### Alert Callbacks
Register custom callbacks for specific alert types:
```python
def my_alert_handler(alert):
    if alert['severity'] == 'CRITICAL':
        # Send email, SMS, etc.
        send_notification(alert['message'])

dashboard._alert_manager.register_callback('CRITICAL', my_alert_handler)
```

### Performance Analysis
Built-in analyzer provides:
- Annualized return calculations
- Progress tracking
- Required performance calculations
- Position concentration analysis
- Trading insights and recommendations

## Customization

### Modify Risk Limits
```python
dashboard._risk_limits = {
    "max_drawdown": 0.15,          # 15% max drawdown
    "max_position_concentration": 0.30,  # 30% max position
    "max_daily_loss": 0.03,        # 3% daily loss
    "max_leverage": 3.0,           # 3x leverage
}
```

### Update Interval
```python
# Create with custom update interval
dashboard = MonitoringDashboard(
    # ... other params ...
    update_interval_secs=0.5,  # Update every 500ms
)
```

### Custom Alerts
```python
# In your strategy
if condition_met:
    dashboard._alerts.append({
        "timestamp": datetime.utcnow().isoformat(),
        "type": "CUSTOM",
        "severity": "INFO",
        "message": "Custom condition triggered",
    })
```

## Troubleshooting

### Dashboard Not Loading
1. Check Flask is installed: `pip install flask flask-cors`
2. Verify port 5000 is available
3. Check browser console for errors

### No Data Showing
1. Ensure dashboard.start() was called
2. Verify portfolio object is correctly passed
3. Check trading node is running

### WebSocket Connection Failed
1. Install flask-socketio: `pip install flask-socketio`
2. Check firewall settings
3. Verify CORS settings if accessing remotely

## Tips for Challenge Success

1. **Monitor Risk Score**: Keep below 50 for sustainable trading
2. **Track Required Daily Return**: Adjust strategy if falling behind
3. **Watch Position Concentration**: Diversify to reduce risk
4. **Review Alerts Daily**: Address issues promptly
5. **Use Performance Insights**: Follow dashboard recommendations

## Security Considerations

- Dashboard binds to localhost by default
- No authentication required (add if exposing externally)
- Sensitive data stays within local network
- Consider VPN for remote access

## Performance Impact

- Minimal CPU usage (<1%)
- Low memory footprint (~50MB)
- Efficient updates (only changed data)
- No impact on trading performance

---

The Monitoring Dashboard is your command center for the Nautilus challenge. Use it to stay informed, manage risk, and track your progress toward the 10% annual return goal!