#!/usr/bin/env python3
"""
Test the testnet scanner with realistic testnet data
"""
import numpy as np
from core.testnet_scanner import TestnetScanner

def test_testnet_thresholds():
    """Test that testnet scanner finds opportunities with small moves"""
    scanner = TestnetScanner('binance', max_concurrent=1)
    
    # Simulate testnet-like data
    closes = np.array([100.0] * 90 + [100.1, 100.15, 100.2, 100.25, 100.3, 100.35, 100.4, 100.45, 100.5, 100.55])
    volumes = np.array([1000] * 95 + [1200, 1300, 1400, 1500, 1600])  # 1.6x volume at end
    
    # Calculate indicators
    indicators = {
        'sma_20': np.mean(closes[-20:]),
        'sma_50': np.mean(closes[-50:]),
        'ema_12': np.mean(closes[-12:]),
        'ema_26': np.mean(closes[-26:]),
        'macd': np.mean(closes[-12:]) - np.mean(closes[-26:]),
        'macd_signal': 0,
        'rsi': 65,  # Bullish but not overbought
        'bb_upper': 100.8,
        'bb_lower': 99.8,
        'atr': 0.5,
        'volume_sma': np.mean(volumes[-20:]),
        'volume_ratio': volumes[-1] / np.mean(volumes[-20:]),
        'volatility': 0.1
    }
    
    print("🧪 Testing Testnet Scanner Thresholds")
    print("=" * 50)
    print(f"Price move (10 bars): {(closes[-1] - closes[-10])/closes[-10]*100:.2f}%")
    print(f"Price move (5 bars): {(closes[-1] - closes[-5])/closes[-5]*100:.2f}%")
    print(f"Volume ratio: {indicators['volume_ratio']:.2f}x")
    print(f"RSI: {indicators['rsi']}")
    
    # Test momentum strategy
    momentum_opp = scanner._check_momentum('TEST/USDT', closes, indicators)
    if momentum_opp:
        print(f"\n✅ Momentum opportunity found! Score: {momentum_opp.score:.1f}")
    else:
        print("\n❌ No momentum opportunity (threshold not met)")
    
    # Test volume breakout
    volume_opp = scanner._check_volume_breakout('TEST/USDT', closes, indicators)
    if volume_opp:
        print(f"✅ Volume breakout found! Score: {volume_opp.score:.1f}")
    else:
        print("❌ No volume breakout (threshold not met)")
    
    # Show what mainnet scanner would find
    from core.market_scanner import MarketScanner
    mainnet_scanner = MarketScanner('binance', max_concurrent=1)
    
    mainnet_momentum = mainnet_scanner._check_momentum('TEST/USDT', closes, indicators)
    mainnet_volume = mainnet_scanner._check_volume_breakout('TEST/USDT', closes, indicators)
    
    print("\n📊 Mainnet scanner comparison:")
    print(f"   Momentum: {'Found' if mainnet_momentum else 'Not found'}")
    print(f"   Volume: {'Found' if mainnet_volume else 'Not found'}")

if __name__ == "__main__":
    test_testnet_thresholds()