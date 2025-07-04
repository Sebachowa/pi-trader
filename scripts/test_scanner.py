#!/usr/bin/env python3
"""
Test and demonstrate the market scanner capabilities
"""
import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.market_scanner import MarketScanner


async def test_scanner():
    """Test the market scanner with different profiles"""
    
    # Load config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    # Load scanner profiles
    with open('config/scanner_profiles.json', 'r') as f:
        profiles = json.load(f)
    
    # Initialize scanner
    scanner = MarketScanner()
    
    print("🚀 Initializing Market Scanner...")
    await scanner.initialize(
        api_key=config['exchange']['api_key'],
        api_secret=config['exchange']['api_secret'],
        testnet=config['exchange']['testnet']
    )
    
    # Test 1: Get top volume symbols
    print("\n📊 Test 1: Top Volume Symbols")
    print("-" * 50)
    top_symbols = await scanner._get_top_volume_symbols(1000000)
    print(f"Found {len(top_symbols)} symbols with >$1M volume")
    print(f"Top 5: {top_symbols[:5]}")
    
    # Test 2: Quick scan with conservative profile
    print("\n🔍 Test 2: Conservative Scan")
    print("-" * 50)
    conservative = profiles['conservative']
    opportunities = await scanner.scan_markets(
        symbols=conservative['whitelist'],
        min_volume_24h=conservative['min_volume_24h']
    )
    
    print(f"Found {len(opportunities)} opportunities")
    for opp in opportunities[:3]:
        print(f"  {opp.symbol} - {opp.strategy} (score: {opp.score:.1f})")
        print(f"    Entry: ${opp.entry_price:.4f}")
        print(f"    SL: ${opp.stop_loss:.4f} | TP: ${opp.take_profit:.4f}")
    
    # Test 3: Performance test
    print("\n⚡ Test 3: Performance Test")
    print("-" * 50)
    import time
    
    start_time = time.time()
    opportunities = await scanner.scan_markets(
        min_volume_24h=1000000,
        timeframes=['5m', '15m']
    )
    scan_time = time.time() - start_time
    
    stats = scanner.get_scan_stats()
    print(f"Scan completed in {scan_time:.2f} seconds")
    print(f"Scanned symbols: ~{len(opportunities) * 2} (symbols × timeframes)")
    print(f"Opportunities found: {len(opportunities)}")
    print(f"Average scan time: {stats['avg_scan_time']:.2f}s")
    
    # Test 4: Strategy distribution
    print("\n📈 Test 4: Strategy Distribution")
    print("-" * 50)
    strategy_counts = {}
    for opp in opportunities:
        strategy_counts[opp.strategy] = strategy_counts.get(opp.strategy, 0) + 1
    
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count} opportunities")
    
    # Test 5: Continuous scan simulation
    print("\n🔄 Test 5: Continuous Scan (5 iterations)")
    print("-" * 50)
    
    for i in range(5):
        print(f"\nScan {i+1}:")
        opportunities = await scanner.scan_markets(
            symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            timeframes=['5m']
        )
        
        if opportunities:
            best = opportunities[0]
            print(f"  Best: {best.symbol} - {best.strategy} (score: {best.score:.1f})")
        else:
            print("  No opportunities found")
        
        await asyncio.sleep(5)
    
    # Cleanup
    await scanner.close()
    print("\n✅ Scanner test completed!")


async def compare_scan_methods():
    """Compare different scanning methods"""
    print("\n🔬 Comparing Scan Methods")
    print("=" * 60)
    
    # Load config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    scanner = MarketScanner()
    await scanner.initialize(
        api_key=config['exchange']['api_key'],
        api_secret=config['exchange']['api_secret'],
        testnet=config['exchange']['testnet']
    )
    
    # Method 1: Auto selection (top volume)
    print("\n1️⃣ AUTO SELECTION (Top Volume)")
    start = asyncio.get_event_loop().time()
    opps1 = await scanner.scan_markets(min_volume_24h=1000000)
    time1 = asyncio.get_event_loop().time() - start
    print(f"   Time: {time1:.2f}s | Opportunities: {len(opps1)}")
    
    # Method 2: Whitelist only
    print("\n2️⃣ WHITELIST ONLY")
    whitelist = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    start = asyncio.get_event_loop().time()
    opps2 = await scanner.scan_markets(symbols=whitelist)
    time2 = asyncio.get_event_loop().time() - start
    print(f"   Time: {time2:.2f}s | Opportunities: {len(opps2)}")
    
    # Method 3: High concurrency
    print("\n3️⃣ HIGH CONCURRENCY (100)")
    scanner.max_concurrent = 100
    start = asyncio.get_event_loop().time()
    opps3 = await scanner.scan_markets(min_volume_24h=500000)
    time3 = asyncio.get_event_loop().time() - start
    print(f"   Time: {time3:.2f}s | Opportunities: {len(opps3)}")
    
    print("\n📊 Summary:")
    print(f"   Fastest: Method 2 (Whitelist) - {time2:.2f}s")
    print(f"   Most opportunities: Method 3 - {len(opps3)} opportunities")
    print(f"   Best balance: Method 1 - {time1:.2f}s, {len(opps1)} opportunities")
    
    await scanner.close()


if __name__ == "__main__":
    print("🤖 Market Scanner Test Suite")
    print("=" * 60)
    
    # Check if API keys are configured
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    if not config['exchange']['api_key']:
        print("⚠️  Warning: No API keys configured!")
        print("   Results will be limited or may fail.")
        print("   Add your keys to config/config.json")
        print()
    
    # Run tests
    asyncio.run(test_scanner())
    
    # Run comparison
    asyncio.run(compare_scan_methods())