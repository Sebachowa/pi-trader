#!/usr/bin/env python3
"""
Beautiful logging system with colors and emojis
"""
import logging
import sys
from datetime import datetime
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform support
init(autoreset=True)

class EmojiColorFormatter(logging.Formatter):
    """Custom formatter with colors and emojis"""
    
    # Emoji mappings for different log types
    EMOJIS = {
        'DEBUG': '🐛',
        'INFO': '📝',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '🔥',
        # Custom emojis for specific messages
        'scanner': '🔍',
        'opportunity': '💡',
        'position': '📊',
        'trade': '💰',
        'profit': '💸',
        'loss': '📉',
        'system': '⚙️ ',
        'network': '🌐',
        'balance': '💳',
        'strategy': '🎯',
        'risk': '🛡️ ',
        'tax': '📋',
        'testnet': '🧪',
        'startup': '🚀',
        'shutdown': '🛑',
        'success': '✅',
        'failed': '❌',
        'waiting': '⏳',
        'processing': '⚡',
        'monitoring': '👁️ ',
    }
    
    # Color mappings
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE,
    }
    
    # Module colors
    MODULE_COLORS = {
        'engine': Fore.BLUE,
        'scanner': Fore.MAGENTA,
        'monitor': Fore.CYAN,
        'risk': Fore.YELLOW,
        'tax': Fore.GREEN,
    }
    
    def format(self, record):
        # Get base emoji and color
        emoji = self.EMOJIS.get(record.levelname, '📌')
        color = self.COLORS.get(record.levelname, '')
        
        # Check message for specific keywords to add context emojis
        msg_lower = record.getMessage().lower()
        for keyword, context_emoji in self.EMOJIS.items():
            if keyword in msg_lower:
                emoji = f"{emoji} {context_emoji}"
                break
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Get module color
        module_name = record.name.split('.')[-1] if '.' in record.name else record.name
        module_color = self.MODULE_COLORS.get(module_name, Fore.WHITE)
        
        # Build the formatted message
        level_text = f"{color}{record.levelname:8}{Style.RESET_ALL}"
        module_text = f"{module_color}{module_name:12}{Style.RESET_ALL}"
        
        # Special formatting for specific message types
        message = record.getMessage()
        
        # Highlight numbers and percentages
        import re
        # Highlight percentages
        message = re.sub(r'(\d+\.?\d*%)', f'{Fore.YELLOW}\\1{Style.RESET_ALL}', message)
        # Highlight dollar amounts
        message = re.sub(r'(\$[\d,]+\.?\d*)', f'{Fore.GREEN}\\1{Style.RESET_ALL}', message)
        # Highlight scores
        message = re.sub(r'(score:\s*\d+\.?\d*)', f'{Fore.MAGENTA}\\1{Style.RESET_ALL}', message, flags=re.IGNORECASE)
        
        # Format the final log line
        return f"{Fore.BLUE}{timestamp}{Style.RESET_ALL} {emoji}  {level_text} [{module_text}] {message}"


class TradingLogger:
    """Enhanced trading logger with beautiful output"""
    
    @staticmethod
    def setup_logger(name: str, log_level: str = "INFO", log_file: str = None):
        """Setup a logger with our custom formatter"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(EmojiColorFormatter())
        logger.addHandler(console_handler)
        
        # File handler (without colors)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_startup(logger):
        """Log beautiful startup message"""
        logger.info("=" * 60)
        logger.info("🤖 Raspberry Pi Trading Bot Starting")
        logger.info("=" * 60)
    
    @staticmethod
    def log_opportunity(logger, symbol: str, strategy: str, score: float):
        """Log opportunity with special formatting"""
        logger.info(
            f"💡 OPPORTUNITY FOUND! {Fore.YELLOW}{symbol}{Style.RESET_ALL} "
            f"- {strategy} (score: {score:.1f})"
        )
    
    @staticmethod
    def log_trade_opened(logger, symbol: str, size: float, price: float):
        """Log trade opening with special formatting"""
        logger.info(
            f"💰 TRADE OPENED: {Fore.GREEN}{symbol}{Style.RESET_ALL} "
            f"- Size: {size:.4f} @ ${price:,.2f}"
        )
    
    @staticmethod
    def log_trade_closed(logger, symbol: str, pnl: float, reason: str):
        """Log trade closing with special formatting"""
        emoji = "💸" if pnl > 0 else "📉"
        color = Fore.GREEN if pnl > 0 else Fore.RED
        logger.info(
            f"{emoji} TRADE CLOSED: {symbol} - "
            f"P&L: {color}${pnl:,.2f}{Style.RESET_ALL} ({reason})"
        )
    
    @staticmethod
    def log_system_status(logger, cpu: float, ram: float, positions: int, equity: float):
        """Log system status with special formatting"""
        logger.info(
            f"⚙️  System: CPU {cpu}%, RAM {ram}% | "
            f"Positions: {positions} | Equity: ${equity:,.2f}"
        )


# Example custom log functions
def log_scanner_results(logger, found: int, time_taken: float):
    """Log scanner results with custom formatting"""
    if found > 0:
        logger.info(
            f"🔍 Scan completed in {time_taken:.2f}s, "
            f"found {Fore.YELLOW}{found} opportunities{Style.RESET_ALL} 🎯"
        )
    else:
        logger.info(f"🔍 Scan completed in {time_taken:.2f}s, no opportunities found")


def log_balance_update(logger, currency: str, amount: float):
    """Log balance updates"""
    logger.info(f"💳 Balance update: {currency} = {amount:,.2f}")


def log_strategy_signal(logger, strategy: str, action: str, confidence: float):
    """Log strategy signals"""
    emoji = "🟢" if action == "BUY" else "🔴"
    logger.info(
        f"🎯 {strategy} signal: {emoji} {action} "
        f"(confidence: {confidence:.1%})"
    )