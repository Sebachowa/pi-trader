#!/usr/bin/env python3
"""
Test paper trading with Binance Testnet
This uses only supported endpoints for testnet
"""
import os
import sys
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.market_scanner import MarketScanner

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_paper_trading():
    """Test paper trading with real market data from testnet"""
    logger.info("🚀 Starting Paper Trading Test with Binance Testnet")
    logger.info("=" * 50)
    
    # Check environment variables
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or api_key.startswith('PLACEHOLDER'):
        logger.error("❌ Please set real Binance Testnet API keys in .env file")
        return
    
    # Initialize scanner
    scanner = MarketScanner(exchange_name='binance', max_concurrent=10)
    
    try:
        # Initialize with testnet
        logger.info("📊 Connecting to Binance Testnet...")
        await scanner.initialize(api_key, api_secret, testnet=True)
        logger.info("✅ Connected to Binance Testnet!")
        
        # Paper trading simulation
        paper_balance = 10000  # Start with $10,000 USDT
        paper_positions = {}
        
        logger.info(f"💰 Starting paper balance: ${paper_balance:,.2f} USDT")
        
        # Run for a few iterations
        for i in range(3):
            logger.info(f"\n🔍 Scan #{i+1} - Looking for opportunities...")
            
            # Scan only a few major pairs to avoid rate limits
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
            opportunities = await scanner.scan_markets(
                symbols=symbols,
                min_volume_24h=0,  # No volume filter for test
                timeframes=['5m']
            )
            
            if opportunities:
                logger.info(f"📈 Found {len(opportunities)} opportunities!")
                
                # Take the best opportunity
                best = opportunities[0]
                logger.info(f"   Best: {best.symbol} - {best.strategy} (score: {best.score:.1f})")
                logger.info(f"   Signal: {best.signal} @ ${best.entry_price:,.2f}")
                
                # Simulate paper trade
                if best.signal == 'BUY' and best.symbol not in paper_positions:
                    # Allocate 20% of balance
                    position_size = paper_balance * 0.2
                    quantity = position_size / best.entry_price
                    
                    paper_positions[best.symbol] = {
                        'quantity': quantity,
                        'entry_price': best.entry_price,
                        'strategy': best.strategy,
                        'stop_loss': best.stop_loss,
                        'take_profit': best.take_profit
                    }
                    
                    paper_balance -= position_size
                    logger.info(f"✅ PAPER TRADE: Bought {quantity:.4f} {best.symbol} @ ${best.entry_price:,.2f}")
                    logger.info(f"   Stop Loss: ${best.stop_loss:,.2f}, Take Profit: ${best.take_profit:,.2f}")
            else:
                logger.info("   No opportunities found")
            
            # Check existing positions
            if paper_positions:
                logger.info("\n📊 Current Paper Positions:")
                for symbol, pos in paper_positions.items():
                    try:
                        # Get current price
                        ticker = await scanner.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        pnl = (current_price - pos['entry_price']) * pos['quantity']
                        pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                        
                        logger.info(f"   {symbol}: ${current_price:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                        
                        # Check stop loss or take profit
                        if current_price <= pos['stop_loss']:
                            logger.info(f"   ⛔ Stop loss hit! Closing position...")
                            paper_balance += pos['quantity'] * current_price
                            del paper_positions[symbol]
                        elif current_price >= pos['take_profit']:
                            logger.info(f"   🎯 Take profit hit! Closing position...")
                            paper_balance += pos['quantity'] * current_price
                            del paper_positions[symbol]
                            
                    except Exception as e:
                        logger.error(f"   Error checking {symbol}: {e}")
            
            logger.info(f"\n💰 Current paper balance: ${paper_balance:,.2f} USDT")
            logger.info(f"📈 Positions open: {len(paper_positions)}")
            
            # Wait before next scan
            if i < 2:  # Don't wait after last iteration
                logger.info("\n⏳ Waiting 30 seconds before next scan...")
                await asyncio.sleep(30)
        
        # Final summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 PAPER TRADING SUMMARY")
        logger.info(f"Final balance: ${paper_balance:,.2f} USDT")
        logger.info(f"P&L: ${paper_balance - 10000:+,.2f} ({((paper_balance - 10000) / 10000) * 100:+.2f}%)")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.error(f"   Type: {type(e).__name__}")
        
        # Specific error handling
        if "Invalid Api-Key" in str(e):
            logger.info("\n💡 Make sure you're using Binance TESTNET API keys, not mainnet keys")
        elif "does not have" in str(e) and "sandbox" in str(e):
            logger.info("\n💡 This might be a CCXT version issue. Try: pip install --upgrade ccxt")
            
    finally:
        # Clean up
        await scanner.close()
        logger.info("\n✅ Test completed!")


if __name__ == "__main__":
    print("\n📝 Binance Testnet Paper Trading Test")
    print("This will simulate trading using real market data from testnet")
    print("No real money will be used!\n")
    
    asyncio.run(test_paper_trading())