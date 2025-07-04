#!/usr/bin/env python3
"""
Debug why opportunities aren't being traded
"""
import os
import json
from dotenv import load_dotenv
import ccxt

load_dotenv()

def debug_opportunity():
    """Debug USDT/ARS opportunity processing"""
    print("🔍 DEBUG: Why USDT/ARS isn't being traded")
    print("=" * 60)
    
    # Load config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
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
    
    # Check market info
    markets = exchange.load_markets()
    
    if 'USDT/ARS' in markets:
        market = markets['USDT/ARS']
        print(f"✅ USDT/ARS market found")
        print(f"   Active: {market['active']}")
        print(f"   Min order: {market['limits']['amount']['min']} {market['base']}")
        print(f"   Min cost: {market['limits']['cost']['min']} {market['quote']}")
        
        # Get current price
        ticker = exchange.fetch_ticker('USDT/ARS')
        print(f"\n💹 Current market:")
        print(f"   Price: {ticker['last']} ARS")
        print(f"   24h Volume: ${ticker['quoteVolume']:,.0f}")
        
        # Calculate position size
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"\n💰 Balance:")
        print(f"   USDT: {usdt_balance}")
        
        # Config settings
        position_size_pct = config['trading']['position_size_pct']
        max_position_usd = config['risk']['max_position_size_usd']
        
        # Calculate intended position
        intended_size = usdt_balance * position_size_pct
        position_size = min(intended_size, max_position_usd)
        
        print(f"\n📊 Position sizing:")
        print(f"   10% of balance: {intended_size} USDT")
        print(f"   Max allowed: {max_position_usd} USDT")
        print(f"   Final size: {position_size} USDT")
        
        # Check if meets minimum
        if position_size < market['limits']['amount']['min']:
            print(f"\n❌ Position size ({position_size}) < minimum ({market['limits']['amount']['min']})")
        else:
            print(f"\n✅ Position size meets minimum order requirement")
            
        # Check trading status
        print(f"\n🔍 Why it might not trade:")
        print(f"   1. Check if USDT/ARS is in your config whitelist/blacklist")
        print(f"   2. Risk manager cooldown period: {config['risk']['cooldown_minutes']} minutes")
        print(f"   3. Max positions: {config['trading']['max_positions']}")
        
    else:
        print("❌ USDT/ARS market not found!")
        print("\nAvailable ARS pairs:")
        ars_pairs = [s for s in markets.keys() if '/ARS' in s]
        for pair in sorted(ars_pairs)[:10]:
            print(f"   {pair}")

if __name__ == "__main__":
    debug_opportunity()