"""
Telegram notification system for trading bot
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict
from telegram import Bot
from telegram.error import TelegramError
import logging


class TelegramNotifier:
    """Send trading notifications via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """Initialize Telegram bot"""
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        self.logger = logging.getLogger(__name__)
        self.enabled = bool(bot_token and chat_id)
        
        if not self.enabled:
            self.logger.warning("Telegram notifications disabled - missing token or chat_id")
    
    async def send_message(self, message: str, parse_mode: str = 'Markdown'):
        """Send a message to Telegram"""
        if not self.enabled:
            return
            
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
        except TelegramError as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
    
    def send_sync(self, message: str):
        """Synchronous wrapper for sending messages"""
        if not self.enabled:
            return
            
        try:
            asyncio.run(self.send_message(message))
        except RuntimeError:
            # If already in async context, create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_message(message))
            loop.close()
    
    def format_opportunity(self, opportunity: Dict) -> str:
        """Format opportunity notification"""
        return f"""
🎯 *OPPORTUNITY FOUND!*

*Symbol:* `{opportunity['symbol']}`
*Strategy:* {opportunity['strategy']}
*Score:* {opportunity['score']:.2f}
*Entry Price:* ${opportunity['entry_price']:.4f}
*Expected Move:* {opportunity['expected_return']:.2f}%
*Volume:* ${opportunity['volume_24h']:,.0f}

_Time: {datetime.now().strftime('%H:%M:%S')}_
"""
    
    def format_trade_opened(self, trade: Dict) -> str:
        """Format trade opened notification"""
        return f"""
💰 *TRADE OPENED*

*Symbol:* `{trade['symbol']}`
*Side:* {trade['side'].upper()}
*Entry:* ${trade['entry_price']:.4f}
*Amount:* {trade['amount']:.4f}
*Stop Loss:* ${trade['stop_loss']:.4f} ({trade['stop_loss_pct']:.1f}%)
*Take Profit:* ${trade['take_profit']:.4f} ({trade['take_profit_pct']:.1f}%)

_Time: {datetime.now().strftime('%H:%M:%S')}_
"""
    
    def format_trade_closed(self, trade: Dict) -> str:
        """Format trade closed notification"""
        pnl = trade['pnl']
        pnl_pct = trade['pnl_pct']
        emoji = "✅" if pnl > 0 else "❌"
        
        return f"""
{emoji} *TRADE CLOSED*

*Symbol:* `{trade['symbol']}`
*Entry:* ${trade['entry_price']:.4f}
*Exit:* ${trade['exit_price']:.4f}
*PnL:* ${pnl:.2f} ({pnl_pct:+.2f}%)
*Duration:* {trade['duration']}

_Time: {datetime.now().strftime('%H:%M:%S')}_
"""
    
    def format_daily_summary(self, summary: Dict) -> str:
        """Format daily summary"""
        return f"""
📊 *DAILY SUMMARY*

*Total Trades:* {summary['total_trades']}
*Win Rate:* {summary['win_rate']:.1f}%
*Total PnL:* ${summary['total_pnl']:.2f}
*Best Trade:* ${summary['best_trade']:.2f}
*Worst Trade:* ${summary['worst_trade']:.2f}
*Current Balance:* ${summary['balance']:.2f}

_Date: {datetime.now().strftime('%Y-%m-%d')}_
"""
    
    def format_error(self, error: str) -> str:
        """Format error notification"""
        return f"""
⚠️ *ERROR ALERT*

{error}

_Time: {datetime.now().strftime('%H:%M:%S')}_
"""
    
    def format_system_status(self, status: Dict) -> str:
        """Format system status"""
        return f"""
🤖 *SYSTEM STATUS*

*CPU:* {status['cpu']:.1f}%
*Memory:* {status['memory']:.1f}%
*Active Positions:* {status['positions']}
*Current Equity:* ${status['equity']:.2f}
*Uptime:* {status['uptime']}

_Time: {datetime.now().strftime('%H:%M:%S')}_
"""