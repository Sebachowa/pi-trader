#!/usr/bin/env python3
"""
Run the trading bot in demo mode without real API keys
Perfect for testing deployment and configuration
"""
import asyncio
import logging
from datetime import datetime
import random
import os

# Override config to not require real API keys
os.environ['DEMO_MODE'] = 'true'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoTradingEngine:
    """Demo version that doesn't require API keys"""
    
    def __init__(self):
        logger.info("🚀 Starting Trading Bot in DEMO MODE")
        logger.info("📌 No API keys required - using simulated data")
        self.balance = 10000
        self.positions = {}
        
    async def run(self):
        """Run demo trading simulation"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
        
        while True:
            # Simulate market scan
            logger.info("🔍 Scanning markets...")
            await asyncio.sleep(2)
            
            # Random opportunity
            if random.random() > 0.5:
                symbol = random.choice(symbols)
                price = {
                    'BTC/USDT': 45000 + random.randint(-500, 500),
                    'ETH/USDT': 2500 + random.randint(-50, 50),
                    'BNB/USDT': 300 + random.randint(-10, 10),
                    'SOL/USDT': 100 + random.randint(-5, 5)
                }[symbol]
                
                logger.info(f"📈 Opportunity found: {symbol} @ ${price:,.2f}")
                
                # Simulate position
                if symbol not in self.positions and len(self.positions) < 3:
                    self.positions[symbol] = {
                        'entry': price,
                        'size': self.balance * 0.1
                    }
                    logger.info(f"✅ Opened position: {symbol}")
            
            # Update positions
            for symbol in list(self.positions.keys()):
                if random.random() > 0.8:
                    pnl = random.uniform(-2, 5)
                    logger.info(f"💰 Closed {symbol}: {pnl:+.2f}%")
                    del self.positions[symbol]
            
            # Status
            logger.info(f"📊 Balance: ${self.balance:,.2f} | Positions: {len(self.positions)}")
            
            await asyncio.sleep(30)


async def main():
    """Main entry point"""
    logger.info("="*50)
    logger.info("🤖 RASPBERRY PI CRYPTO TRADER - DEMO MODE")
    logger.info("="*50)
    logger.info("This is a demo without real API connections")
    logger.info("Perfect for testing deployment and monitoring")
    logger.info("="*50)
    
    engine = DemoTradingEngine()
    
    try:
        await engine.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Demo stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())