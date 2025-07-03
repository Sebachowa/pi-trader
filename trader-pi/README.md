# Raspberry Pi Trading Bot

A lightweight cryptocurrency trading bot optimized for running on Raspberry Pi devices.

## Features

- **Resource Efficient**: Designed specifically for resource-constrained environments
- **Multiple Strategies**: Trend following and mean reversion strategies included
- **Risk Management**: Built-in position sizing and stop-loss management
- **Real-time Monitoring**: Lightweight system and performance monitoring
- **Auto-recovery**: Handles connection issues and restarts automatically

## Requirements

- Raspberry Pi 4 (recommended) or Raspberry Pi 3B+
- Raspbian OS (Buster or newer)
- Python 3.9+
- At least 2GB RAM (4GB recommended)
- Stable internet connection

## Quick Setup

1. **Clone to your Pi:**
   ```bash
   git clone <repository-url>
   cd trader-pi
   ```

2. **Install dependencies:**
   ```bash
   python3 -m pip install -r requirements-pi.txt
   ```

3. **Configure:**
   - Edit `config/config.json` with your exchange API keys
   - Adjust trading parameters as needed
   - Configure instruments in `config/instruments.json`

4. **Test locally:**
   ```bash
   python3 run.py --dry-run
   ```

5. **Deploy as service:**
   ```bash
   ./deploy_to_pi.sh
   ```

## Configuration

### config/config.json

Main configuration file containing:
- Exchange settings (API keys, testnet mode)
- Trading parameters (position sizes, risk limits)
- Risk management settings
- Monitoring configuration
- Strategy selection

### config/instruments.json

Define which trading pairs to monitor and trade:
- Symbol configuration
- Min/max order sizes
- Strategy assignments per instrument

## Deployment

### Manual Running

```bash
python3 run.py [options]

Options:
  --config PATH       Path to configuration file
  --log-level LEVEL   Logging level (DEBUG, INFO, WARNING, ERROR)
  --dry-run          Run in simulation mode
```

### As a System Service

1. Deploy using the script:
   ```bash
   ./deploy_to_pi.sh [PI_HOST] [PI_USER]
   ```

2. Control the service:
   ```bash
   # Start
   sudo systemctl start trader
   
   # Stop
   sudo systemctl stop trader
   
   # Check status
   sudo systemctl status trader
   
   # View logs
   sudo journalctl -u trader -f
   ```

## Strategies

### Trend Following
- Uses EMA crossover signals
- Confirms with RSI
- Suitable for trending markets

### Mean Reversion
- Uses Bollinger Bands
- Trades price extremes
- Works well in ranging markets

## Resource Optimization

The bot is optimized for Pi hardware:
- Minimal memory footprint
- Efficient numpy operations
- Lightweight monitoring
- Configurable update intervals

## Monitoring

- CPU and memory usage tracking
- Trading performance metrics
- Optional webhook notifications
- Log rotation to prevent disk fill

## Security

- Store API keys securely
- Use read-only API keys when possible
- Enable 2FA on exchange accounts
- Regular security updates

## Troubleshooting

### High CPU Usage
- Increase update intervals in config
- Reduce number of instruments
- Disable unused strategies

### Memory Issues
- Check log file sizes
- Reduce lookback periods
- Enable log rotation

### Connection Issues
- Check internet stability
- Increase timeout values
- Enable auto-reconnect

## Performance Tips

1. Use SSD instead of SD card for better I/O
2. Ensure proper cooling for stable operation
3. Use wired ethernet when possible
4. Monitor SD card health regularly

## Disclaimer

This bot is for educational purposes. Trading cryptocurrencies carries significant risk. Always test thoroughly with small amounts before increasing position sizes.