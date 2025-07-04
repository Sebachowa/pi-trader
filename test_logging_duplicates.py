#!/usr/bin/env python3
"""
Test script to verify duplicate logging is fixed
"""
import logging
from core.logger import TradingLogger

# Setup root logger
root_logger = TradingLogger.setup_logger('', 'INFO')

# Create child loggers like in the actual app
engine_logger = TradingLogger.setup_logger('core.engine')
scanner_logger = TradingLogger.setup_logger('core.market_scanner')
monitor_logger = TradingLogger.setup_logger('core.monitor')

print("🧪 Testing for duplicate log messages...")
print("=" * 60)

# Test each logger
print("\n1. Testing root logger:")
root_logger.info("Message from ROOT logger")

print("\n2. Testing engine logger:")
engine_logger.info("Message from ENGINE logger")

print("\n3. Testing scanner logger:")
scanner_logger.info("Message from SCANNER logger")

print("\n4. Testing monitor logger:")
monitor_logger.info("Message from MONITOR logger")

print("\n5. Testing multiple messages:")
for i in range(3):
    engine_logger.info(f"Engine message #{i+1}")

print("\n✅ Test complete. Each message should appear ONLY ONCE.")
print("If you see duplicate messages, the issue persists.")
print("If each message appears once, the issue is fixed!")