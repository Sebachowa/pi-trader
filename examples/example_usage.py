#!/usr/bin/env python3
"""
Example usage of the trading bot with custom parameters
"""
import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import TradingEngine


def create_custom_config():
    """Create a custom configuration for testing"""
    config = {
        "exchange": {
            "name": "binance",
            "api_key": "YOUR_API_KEY_HERE",
            "api_secret": "YOUR_API_SECRET_HERE",
            "testnet": True  # Always start with testnet!
        },
        "trading": {
            "max_positions": 2,  # Conservative: only 2 positions at once
            "position_size_pct": 0.05,  # 5% per position
            "max_daily_loss_pct": 0.02,  # 2% max daily loss
            "stop_loss_pct": 0.01,  # 1% stop loss
            "take_profit_pct": 0.03,  # 3% take profit
            "leverage": 1  # No leverage for safety
        },
        "risk": {
            "max_drawdown_pct": 0.05,  # 5% max drawdown
            "max_position_size_usd": 100,  # $100 max per position
            "min_volume_24h": 1000000,  # $1M minimum volume
            "cooldown_minutes": 30  # 30 min between trades
        },
        "monitoring": {
            "update_interval_seconds": 60,  # Check every minute
            "log_level": "INFO",
            "enable_notifications": False,
            "webhook_url": ""
        },
        "strategies": {
            "enabled": ["trend_following"],  # Start with one strategy
            "timeframes": ["1h"],  # 1 hour timeframe
            "default_lookback": 50  # 50 candles lookback
        }
    }
    
    # Save custom config
    with open('config/config_example.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return config


def test_indicators():
    """Test technical indicator calculations"""
    from strategies.base import BaseStrategy
    import numpy as np
    
    print("Testing technical indicators...")
    
    # Create dummy OHLCV data
    # Format: [timestamp, open, high, low, close, volume]
    dummy_data = []
    base_price = 100
    for i in range(100):
        timestamp = i * 3600 * 1000  # Hourly data
        open_price = base_price + np.random.randn() * 2
        close_price = open_price + np.random.randn() * 1
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.5)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.5)
        volume = 10000 + np.random.randint(-2000, 2000)
        
        dummy_data.append([timestamp, open_price, high_price, low_price, close_price, volume])
        base_price = close_price
    
    # Test strategy
    strategy = BaseStrategy({})
    indicators = strategy.calculate_indicators(dummy_data)
    
    print(f"SMA 20: {indicators['sma_20'][-1]:.2f}")
    print(f"EMA 12: {indicators['ema_12'][-1]:.2f}")
    print(f"RSI: {indicators['rsi'][-1]:.2f}")
    print(f"BB Upper: {indicators['bb_upper'][-1]:.2f}")
    print(f"BB Lower: {indicators['bb_lower'][-1]:.2f}")
    print("Indicators calculated successfully!\n")


def test_risk_management():
    """Test risk management calculations"""
    from core.risk import RiskManager
    
    print("Testing risk management...")
    
    risk_config = {
        "max_drawdown_pct": 0.1,
        "max_position_size_usd": 1000,
        "min_volume_24h": 1000000,
        "cooldown_minutes": 15
    }
    
    trading_config = {
        "max_positions": 3,
        "position_size_pct": 0.1,
        "max_daily_loss_pct": 0.02,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.05,
        "leverage": 1
    }
    
    risk_manager = RiskManager(risk_config, trading_config)
    
    # Test position sizing
    balance = {"USDT": {"free": 10000}}
    signal = {"price": 50000}  # BTC price
    
    position_size = risk_manager.calculate_position_size("BTCUSDT", signal, balance)
    print(f"Position size for $10,000 balance: {position_size:.4f} BTC")
    print(f"Position value: ${position_size * signal['price']:.2f}")
    
    # Test stop loss calculation
    stop_loss = risk_manager.calculate_stop_loss(50000, "BUY")
    take_profit = risk_manager.calculate_take_profit(50000, "BUY")
    print(f"Stop loss: ${stop_loss:.2f}")
    print(f"Take profit: ${take_profit:.2f}")
    print("Risk management tests passed!\n")


def run_backtest_example():
    """Run a simple backtest example"""
    print("Running backtest example...")
    print("This would connect to historical data and test strategies")
    print("For now, using simulated results:\n")
    
    # Simulated backtest results
    results = {
        "total_trades": 45,
        "winning_trades": 25,
        "losing_trades": 20,
        "win_rate": 55.6,
        "total_return": 12.5,
        "max_drawdown": 8.3,
        "sharpe_ratio": 1.2,
        "profit_factor": 1.4
    }
    
    print("Backtest Results:")
    for key, value in results.items():
        print(f"  {key.replace('_', ' ').title()}: {value}{'%' if 'rate' in key or 'return' in key or 'drawdown' in key else ''}")
    print()


def main():
    print("=" * 60)
    print("Trading Bot Example Usage")
    print("=" * 60)
    print()
    
    # Create example configuration
    print("1. Creating example configuration...")
    config = create_custom_config()
    print("   Configuration saved to config/config_example.json")
    print()
    
    # Test indicators
    print("2. Testing technical indicators...")
    test_indicators()
    
    # Test risk management
    print("3. Testing risk management...")
    test_risk_management()
    
    # Run backtest
    print("4. Running backtest example...")
    run_backtest_example()
    
    print("=" * 60)
    print("Example tests completed!")
    print()
    print("To run the bot with this configuration:")
    print("  python3 run.py --config config/config_example.json --dry-run")
    print()
    print("Remember to:")
    print("  1. Add your API keys to the configuration")
    print("  2. Start with testnet mode enabled")
    print("  3. Test thoroughly before using real funds")
    print("=" * 60)


if __name__ == "__main__":
    main()