#!/usr/bin/env python3
"""
Test Binance Testnet connection with proper configuration
"""
import os
import asyncio
import ccxt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_sync_connection():
    """Test synchronous connection to Binance Testnet"""
    print("🔧 Testing Binance Testnet Connection (Sync)")
    print("=" * 50)
    
    try:
        # Create exchange instance with proper testnet config
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # SPOT not FUTURE!
                'adjustForTimeDifference': True,  # Important for testnet
            }
        })
        
        # Set testnet URLs
        exchange.set_sandbox_mode(True)
        
        print("✅ Exchange created with testnet URLs:")
        print(f"   Base URL: {exchange.urls['api']['public']}")
        
        # Test 1: Fetch markets
        print("\n📊 Fetching markets...")
        markets = exchange.load_markets()
        print(f"✅ Loaded {len(markets)} markets")
        print(f"   Sample markets: {list(markets.keys())[:5]}")
        
        # Test 2: Fetch ticker
        print("\n💹 Fetching BTC/USDT ticker...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ BTC/USDT Price: ${ticker['last']:,.2f}")
        
        # Test 3: Fetch balance (requires valid API key)
        print("\n💰 Fetching account balance...")
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('total', 0)
            print(f"✅ USDT Balance: {usdt_balance:,.2f}")
            
            # Show all non-zero balances
            for currency, info in balance['total'].items():
                if info > 0:
                    print(f"   {currency}: {info}")
                    
        except Exception as e:
            print(f"❌ Balance fetch failed: {str(e)}")
            print("   (This is normal if API key permissions are not set)")
        
        # Test 4: Fetch open orders
        print("\n📝 Fetching open orders...")
        try:
            orders = exchange.fetch_open_orders()
            print(f"✅ Open orders: {len(orders)}")
        except Exception as e:
            print(f"❌ Orders fetch failed: {str(e)}")
        
        print("\n✅ Testnet connection successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        # Common error solutions
        if "Invalid Api-Key" in str(e):
            print("\n💡 Solution: Check that your API key is correct in .env file")
        elif "Timestamp" in str(e):
            print("\n💡 Solution: Your system time might be off. Sync your clock.")
        elif "does not have" in str(e) and "testnet" in str(e):
            print("\n💡 This endpoint is not available on testnet")
            
        return False


async def test_async_connection():
    """Test async connection (used by market scanner)"""
    print("\n\n🔧 Testing Binance Testnet Connection (Async)")
    print("=" * 50)
    
    try:
        # Create async exchange instance
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # SPOT not FUTURE!
                'adjustForTimeDifference': True,
            }
        })
        
        # Set testnet mode
        exchange.set_sandbox_mode(True)
        
        # Load markets
        print("📊 Loading markets...")
        await exchange.load_markets()
        print(f"✅ Loaded {len(exchange.markets)} markets")
        
        # Test fetching multiple tickers
        print("\n💹 Fetching multiple tickers concurrently...")
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        tasks = [exchange.fetch_ticker(symbol) for symbol in symbols]
        tickers = await asyncio.gather(*tasks)
        
        for symbol, ticker in zip(symbols, tickers):
            print(f"   {symbol}: ${ticker['last']:,.2f}")
        
        # Close connection
        await exchange.close()
        
        print("\n✅ Async testnet connection successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Async connection failed: {str(e)}")
        return False


def check_env_vars():
    """Check if environment variables are set"""
    print("🔍 Checking environment variables...")
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    testnet = os.getenv('BINANCE_TESTNET', 'true')
    
    if not api_key or api_key.startswith('PLACEHOLDER'):
        print("❌ BINANCE_API_KEY not set or is placeholder")
        return False
    else:
        print(f"✅ BINANCE_API_KEY: {api_key[:10]}...")
    
    if not api_secret or api_secret.startswith('PLACEHOLDER'):
        print("❌ BINANCE_API_SECRET not set or is placeholder")
        return False
    else:
        print(f"✅ BINANCE_API_SECRET: {api_secret[:10]}...")
    
    print(f"✅ BINANCE_TESTNET: {testnet}")
    
    return True


if __name__ == "__main__":
    print("🚀 Binance Testnet Connection Test")
    print("=" * 50)
    
    if not check_env_vars():
        print("\n⚠️  Please set your API keys in the .env file")
        exit(1)
    
    # Run sync test
    if test_sync_connection():
        # Run async test
        asyncio.run(test_async_connection())
    
    print("\n📌 Note: Binance Testnet only supports SPOT trading, not futures!")
    print("📌 Make sure to use 'spot' as defaultType in your configuration")