# Comprehensive Risk Management System

## Overview

The Risk Management System is a complete, integrated solution for protecting trading capital while maximizing returns. It combines advanced position sizing algorithms, dynamic stop loss management, portfolio diversification rules, real-time monitoring, and capital preservation strategies.

## System Components

### 1. Comprehensive Risk Manager (`comprehensive_risk_manager.py`)

The core risk management engine that provides:

- **Multi-Model Position Sizing**
  - Kelly Criterion (with 25% safety factor)
  - Optimal F (with Monte Carlo simulation)
  - Risk Parity (equal risk contribution)
  - Volatility Targeting (regime-adjusted)
  - Machine Learning based sizing

- **Dynamic Stop Loss & Take Profit**
  - ATR-based dynamic stops
  - Volatility-adjusted stop distances
  - Trailing stop activation and management
  - Risk/reward ratio optimization

- **Portfolio Constraints**
  - Maximum position limits (3-20 positions)
  - Correlation constraints (max 0.7)
  - Concentration limits (max 20% per position)
  - Sector exposure limits

### 2. Real-time Risk Monitor (`realtime_risk_monitor.py`)

Continuous monitoring system with:

- **Real-time Metrics Tracking**
  - Portfolio drawdown
  - Daily P&L
  - Value at Risk (VaR)
  - Correlation risk
  - Exposure levels

- **Multi-Level Alerts**
  - Info: Minor deviations
  - Warning: Approaching limits
  - Critical: Immediate attention needed
  - Emergency: Automatic actions triggered

- **Distribution Channels**
  - WebSocket server (port 8765) for real-time updates
  - Prometheus metrics (port 9090) for monitoring
  - Alert callbacks for custom integrations

### 3. Capital Preservation Manager (`capital_preservation.py`)

Advanced protection strategies:

- **Preservation Levels**
  1. Light (25% reduction)
  2. Moderate (50% reduction + hedges)
  3. Maximum (emergency mode)

- **Protection Strategies**
  - Drawdown protection
  - Volatility protection
  - Correlation breakdown response
  - Loss streak protection
  - Tail risk hedging
  - Liquidity crisis management

- **Hedging Mechanisms**
  - Beta hedging (market exposure)
  - Volatility hedging (VIX-based)
  - Tail risk hedging (OTM options)
  - Portfolio insurance (CPPI)

### 4. Integrated System (`risk_management_system.py`)

Complete system integration providing:

- Unified API for all risk operations
- Coordinated monitoring and responses
- Automatic preservation activation
- System state management
- Memory persistence

## Risk Limits Configuration

```python
{
    "max_daily_loss_percent": 2.0,      # Daily loss limit
    "max_drawdown_percent": 10.0,       # Maximum drawdown
    "max_position_risk_percent": 1.0,   # Per position risk
    "max_portfolio_risk_percent": 5.0,  # Total portfolio risk
    "max_correlation": 0.7,             # Position correlation limit
    "max_concentration_percent": 20.0   # Position concentration
}
```

## Usage Examples

### Initialize System

```python
from autonomous_trading.risk_management_system import create_risk_management_system

# Create and start the system
risk_system = await create_risk_management_system(
    logger=logger,
    clock=clock,
    msgbus=msgbus,
    portfolio=portfolio,
    config={
        "max_daily_loss_percent": 2.0,
        "enable_realtime_monitoring": True,
        "enable_capital_preservation": True
    }
)
```

### Calculate Position Size

```python
# Calculate position size with all risk checks
result = await risk_system.calculate_position_size(
    instrument_id=InstrumentId.from_str("BTC/USDT"),
    account_balance=Money(10000, Currency.USD),
    entry_price=Decimal("50000"),
    stop_loss_price=Decimal("49000"),
    confidence_score=0.7,
    strategy_hint="trend_following"
)

# Result includes:
# - Optimal position size
# - Selected sizing model
# - Stop loss and take profit levels
# - Risk/reward ratio
# - Diversification score
```

### Monitor Portfolio Risk

```python
# Get comprehensive risk dashboard
dashboard = await risk_system.get_risk_dashboard()

# Dashboard includes:
# - Current risk metrics
# - Active alerts
# - Position analysis
# - Preservation status
# - Emergency procedures status
```

### Update Position Stops

```python
# Update trailing stops and risk parameters
updates = await risk_system.update_position_risk(
    position_id=position.id,
    current_price=51000
)
```

## Emergency Procedures

The system automatically triggers emergency procedures when:

1. **Maximum Drawdown (10%)**: Close all positions
2. **Daily Loss (2%)**: Stop trading for 24 hours
3. **Correlation Spike (>0.9)**: Reduce correlated positions
4. **Volatility Spike (3x normal)**: Reduce all positions by 50%
5. **System Anomaly (>95%)**: Activate safe mode

## Monitoring Dashboard

Access the real-time monitoring dashboard:

- **WebSocket**: `ws://localhost:8765`
- **Prometheus**: `http://localhost:9090/metrics`

Dashboard provides:
- Live risk metrics
- Position heatmap
- Alert notifications
- Historical charts
- System health status

## Capital Preservation Modes

### Light Mode (Level 1)
- Reduce positions by 25%
- Increase cash to 10%
- Tighten risk parameters

### Moderate Mode (Level 2)
- Reduce positions by 50%
- Activate portfolio hedges
- Rotate to defensive assets

### Maximum Mode (Level 3)
- Close speculative positions
- Increase cash to 50%
- Implement portfolio insurance
- Emergency trading stop

## Best Practices

1. **Position Sizing**
   - Always use confidence scores
   - Provide strategy hints for optimal model selection
   - Include stop loss prices when available

2. **Risk Monitoring**
   - Review dashboard at least hourly
   - Respond to critical alerts immediately
   - Monitor correlation changes

3. **Capital Preservation**
   - Don't override automatic preservation
   - Allow gradual recovery phases
   - Monitor hedge effectiveness

4. **System Maintenance**
   - Review risk parameters weekly
   - Adjust thresholds based on market conditions
   - Analyze preservation event outcomes

## Integration with Trading System

The risk management system integrates seamlessly with the autonomous trading system:

```python
# In your trading strategy
async def on_signal(self, signal):
    # Get position size from risk manager
    sizing_result = await self.risk_system.calculate_position_size(
        instrument_id=signal.instrument,
        account_balance=self.portfolio.account_balance_total(),
        entry_price=signal.entry_price,
        stop_loss_price=signal.stop_loss,
        confidence_score=signal.confidence
    )
    
    # Place order with calculated size
    if sizing_result["quantity"].as_double() > 0:
        await self.place_order(
            instrument_id=signal.instrument,
            quantity=sizing_result["quantity"],
            stop_loss=sizing_result["stop_loss"],
            take_profit=sizing_result["take_profit"]
        )
```

## Performance Metrics

The system tracks:
- Win rate and profit factor
- Sharpe, Sortino, and Calmar ratios
- Maximum favorable/adverse excursions
- Stop loss effectiveness
- Hedge performance
- Capital preservation outcomes

## Troubleshooting

### Common Issues

1. **No position size returned**
   - Check if emergency stop is active
   - Verify daily loss limits
   - Ensure correlation constraints

2. **Alerts not triggering**
   - Verify monitoring is active
   - Check alert cooldown periods
   - Review threshold configurations

3. **Preservation not activating**
   - Check strategy cooldown periods
   - Verify trigger thresholds
   - Review market condition detection

## Future Enhancements

Planned improvements:
- Machine learning risk prediction models
- Advanced correlation analysis
- Options-based hedging strategies
- Multi-asset portfolio optimization
- Sentiment-based risk adjustment

---

**System Status**: The complete risk management system has been saved to Memory at:
`swarm-auto-hierarchical-1751379006249/risk-manager/system`