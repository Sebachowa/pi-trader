#!/usr/bin/env python3
"""
Quick test to check scanner opportunities with different thresholds
"""
import asyncio
import json
from core.market_scanner import MarketScanner
from dotenv import load_dotenv
import os

load_dotenv()

async def test_scanner():
    """Test scanner with different score thresholds"""
    # Load config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize scanner
    scanner = MarketScanner('binance', max_concurrent=50)
    await scanner.initialize(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_API_SECRET'),
        testnet=True
    )
    
    print("🔍 Testing scanner with different score thresholds...")
    print("=" * 60)
    
    # Run scan
    symbols = scanner._select_symbols()[:20]  # Test with 20 symbols
    opportunities = await scanner.scan_market(symbols)
    
    # Show all opportunities regardless of score
    if opportunities:
        print(f"Found {len(opportunities)} raw opportunities:")
        for opp in sorted(opportunities, key=lambda x: x['score'], reverse=True):
            print(f"  {opp['symbol']}: Score={opp['score']:.1f}, Strategy={opp['strategy']}")
    else:
        print("No opportunities found (even with score >= 0)")
    
    # Show distribution
    print("\n📊 Score distribution:")
    thresholds = [0, 30, 40, 50, 60, 70]
    for threshold in thresholds:
        count = len([o for o in opportunities if o['score'] >= threshold])
        print(f"  Score >= {threshold}: {count} opportunities")
    
    await scanner.close()

if __name__ == "__main__":
    asyncio.run(test_scanner())