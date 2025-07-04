#!/usr/bin/env python3
"""
Debug scanner to see what's happening
"""
import asyncio
import json
import os
from dotenv import load_dotenv
import ccxt

load_dotenv()

def debug_scanner():
    """Debug why scanner finds no opportunities"""
    print("🔍 DEBUG: Scanner Investigation")
    print("=" * 60)
    
    # Check config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    print(f"✅ Config loaded:")
    print(f"   Min score: {config['scanner']['min_opportunity_score']}")
    print(f"   Min volume: ${config['scanner']['min_volume_24h']:,}")
    
    # Initialize exchange
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True
        }
    })
    exchange.set_sandbox_mode(True)
    
    # Get markets
    markets = exchange.load_markets()
    print(f"\n📊 Total markets: {len(markets)}")
    
    # Get top volume symbols
    tickers = exchange.fetch_tickers()
    usdt_pairs = [(symbol, ticker) for symbol, ticker in tickers.items() 
                  if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > 0]
    
    # Sort by volume
    usdt_pairs.sort(key=lambda x: x[1].get('quoteVolume', 0), reverse=True)
    
    print(f"\n🔝 Top 10 USDT pairs by volume:")
    for i, (symbol, ticker) in enumerate(usdt_pairs[:10]):
        volume = ticker.get('quoteVolume', 0)
        change = ticker.get('percentage', 0)
        print(f"   {i+1}. {symbol}: Vol=${volume:,.0f}, Change={change:+.2f}%")
    
    # Check a specific symbol
    print(f"\n🎯 Checking BTC/USDT in detail:")
    symbol = 'BTC/USDT'
    
    # Get OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=100)
    if ohlcv:
        closes = [x[4] for x in ohlcv]
        volumes = [x[5] for x in ohlcv]
        
        # Calculate simple indicators
        current_price = closes[-1]
        sma20 = sum(closes[-20:]) / 20
        price_change_5bars = (closes[-1] - closes[-6]) / closes[-6] * 100
        volume_avg = sum(volumes[-20:]) / 20
        volume_ratio = volumes[-1] / volume_avg if volume_avg > 0 else 0
        
        print(f"   Current price: ${current_price:,.2f}")
        print(f"   SMA 20: ${sma20:,.2f}")
        print(f"   Price vs SMA: {(current_price - sma20) / sma20 * 100:+.2f}%")
        print(f"   5-bar change: {price_change_5bars:+.2f}%")
        print(f"   Volume ratio: {volume_ratio:.2f}x")
        
        # Calculate what scores would be
        print(f"\n📈 Estimated scores for different strategies:")
        
        # Trend following (needs positive MACD, price > SMA)
        if current_price > sma20:
            trend_score = 50  # Base score when conditions met
            print(f"   Trend Following: ~{trend_score} (price > SMA)")
        else:
            print(f"   Trend Following: 0 (price < SMA)")
        
        # Momentum (needs 2% move)
        if abs(price_change_5bars) > 2:
            momentum_score = 40 + abs(price_change_5bars) * 5
            print(f"   Momentum: ~{momentum_score:.0f} (>{price_change_5bars:+.1f}% move)")
        else:
            print(f"   Momentum: 0 (needs >2% move, got {price_change_5bars:+.1f}%)")
        
        # Volume breakout (needs 3x volume)
        if volume_ratio > 3:
            volume_score = 30 + volume_ratio * 10
            print(f"   Volume Breakout: ~{volume_score:.0f} ({volume_ratio:.1f}x volume)")
        else:
            print(f"   Volume Breakout: 0 (needs >3x volume, got {volume_ratio:.1f}x)")
    
    exchange.close()
    
    print(f"\n⚠️  Issues preventing opportunities:")
    print(f"   1. Testnet has low volatility (most moves <2%)")
    print(f"   2. Volume ratios rarely exceed 3x")
    print(f"   3. Strategies are calibrated for mainnet conditions")
    print(f"\n💡 Recommendation: Lower strategy thresholds for testnet")

if __name__ == "__main__":
    debug_scanner()