#!/usr/bin/env python3
"""
Test the beautiful logging system
"""
import logging
from core.logger import TradingLogger, log_strategy_signal, log_scanner_results

# Setup logger
logger = TradingLogger.setup_logger('test_logging')

# Show startup
TradingLogger.log_startup(logger)

# Simulate system status
logger.info("⚙️  Initializing trading engine...")
TradingLogger.log_system_status(logger, 15.5, 45.2, 2, 10000.50)

# Simulate scanner results
logger.info("🔍 Starting market scan...")
log_scanner_results(logger, 0, 8.45)
log_scanner_results(logger, 3, 9.12)

# Simulate opportunities
TradingLogger.log_opportunity(logger, "BTC/USDT", "trend_following", 85.5)
TradingLogger.log_opportunity(logger, "ETH/USDT", "mean_reversion", 72.3)
TradingLogger.log_opportunity(logger, "USDT/ARS", "mean_reversion", 40.1)

# Simulate signals
log_strategy_signal(logger, "Trend Following", "BUY", 0.855)
log_strategy_signal(logger, "Mean Reversion", "SELL", 0.623)

# Simulate trade execution
logger.info("⚡ Executing BUY order: 0.0125 BTC/USDT")
TradingLogger.log_trade_opened(logger, "BTC/USDT", 0.0125, 108950.50)

# Simulate position updates
logger.info("📊 Position update: BTC/USDT - Current P&L: $125.50 (+1.15%)")
logger.warning("⚠️  Stop loss approaching for ETH/USDT position")

# Simulate trade closing
TradingLogger.log_trade_closed(logger, "ETH/USDT", -45.20, "STOP_LOSS")
TradingLogger.log_trade_closed(logger, "BTC/USDT", 250.75, "TAKE_PROFIT")

# Test error messages
logger.error("❌ Failed to connect to exchange API")
logger.warning("⚠️  Low balance warning: USDT balance < $100")

# Show different log levels
logger.debug("🐛 Debug: Order book depth = 50")
logger.info("📝 Info: Normal operation")
logger.warning("⚠️  Warning: High volatility detected")
logger.error("❌ Error: Order rejected by exchange")

print("\n✨ Beautiful logging demonstration complete!")