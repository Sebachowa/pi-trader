#!/usr/bin/env python3
"""
Main entry point for Raspberry Pi trading bot
"""
import os
import sys
import signal
import argparse
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import TradingEngine


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logging.info("Shutdown signal received")
    sys.exit(0)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration with beautiful output"""
    from core.logger import TradingLogger
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Setup root logger with our custom formatter
    root_logger = TradingLogger.setup_logger('', log_level, log_file)
    
    # Set third-party loggers to WARNING
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return root_logger


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Raspberry Pi Trading Bot')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode without API keys'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Paper trading mode (requires API keys for market data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in simulation mode without placing real orders'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Import enhanced logger functions
    from core.logger import TradingLogger
    logger = TradingLogger.setup_logger(__name__)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Beautiful startup logging
    TradingLogger.log_startup(logger)
    logger.info(f"📁 Config: {args.config}")
    
    # Determine mode
    if args.demo:
        mode = "🎮 DEMO (No API keys required)"
        os.environ['DEMO_MODE'] = 'true'
    elif args.paper or args.dry_run:
        mode = "📝 PAPER TRADING"
    else:
        mode = "💰 LIVE TRADING"
    
    logger.info(f"Mode: {mode}")
    
    try:
        # Run demo mode if requested
        if args.demo:
            logger.info("🎮 Starting DEMO mode...")
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))
            from run_demo import main as demo_main
            import asyncio
            asyncio.run(demo_main())
            return
        
        # Create and start trading engine
        engine = TradingEngine(args.config)
        
        if args.dry_run or args.paper:
            logger.info("📝 Running in PAPER TRADING mode - no real orders will be placed")
            # TODO: Implement paper trading mode
        
        # Start trading
        engine.start()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Trading bot stopped")


if __name__ == "__main__":
    main()