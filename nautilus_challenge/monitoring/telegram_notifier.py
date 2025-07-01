#!/usr/bin/env python3
"""
Telegram Notifier for Trading Challenge
Sends real-time notifications for trades, performance, and alerts
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class TelegramNotifier:
    """Handle Telegram notifications for trading events."""
    
    def __init__(self):
        """Initialize notifier with existing telegram config."""
        self.config_path = Path(__file__).parent.parent.parent / "telegram_config.json"
        self.load_config()
        
    def load_config(self):
        """Load telegram configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.telegram_config = config.get("notifications", {}).get("telegram", {})
                self.enabled = self.telegram_config.get("enabled", False)
        else:
            self.enabled = False
            
    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram."""
        if not self.enabled:
            return False
            
        try:
            bot_token = self.telegram_config["bot_token"]
            chat_id = self.telegram_config["chat_id"]
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            params = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            data = json.dumps(params).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                return result.get("ok", False)
                
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
            
    async def send_challenge_start(self, config: Dict[str, Any]):
        """Send challenge start notification."""
        message = f"""🚀 <b>NAUTILUS CHALLENGE STARTED</b>
━━━━━━━━━━━━━━━━━━━━━━━
💰 Initial Capital: {config['initial_capital_btc']} BTC
🎯 Target: {config['target_annual_return']*100}% annual ({config['target_monthly_return']*100}% monthly)
📊 Duration: {config['duration_days']} days
📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━━━━
<b>Risk Parameters:</b>
• Max Daily Drawdown: {config['max_daily_drawdown']*100}%
• Max Total Drawdown: {config['max_total_drawdown']*100}%

<b>Live Market Data:</b> ✅ Connected
<b>Paper Trading Mode:</b> ✅ Active

Good luck! 🍀"""
        
        await self.send_message(message)
        
    async def send_trade_signal(self, trade: Dict[str, Any]):
        """Send trade execution notification."""
        emoji = "🟢" if trade['side'] == "BUY" else "🔴"
        pnl_emoji = "✅" if trade.get('pnl', 0) > 0 else "❌"
        
        message = f"""{emoji} <b>{trade['side']} {trade['instrument']}</b>
━━━━━━━━━━━━━━━━━━━━━━━
📈 Strategy: {trade['strategy']}
💵 Entry: ${trade['entry_price']:,.2f}
📊 Size: {trade['size']} ({trade['position_pct']:.1f}% of capital)
🎯 Target: ${trade['take_profit']:,.2f}
🛡️ Stop: ${trade['stop_loss']:,.2f}
━━━━━━━━━━━━━━━━━━━━━━━
⚡ Confidence: {trade['confidence']}%
📊 Risk/Reward: 1:{trade['risk_reward']:.1f}"""
        
        await self.send_message(message)
        
    async def send_position_closed(self, position: Dict[str, Any]):
        """Send position closed notification."""
        pnl = position['pnl']
        pnl_pct = position['pnl_percent']
        emoji = "🟢" if pnl > 0 else "🔴"
        
        message = f"""{emoji} <b>POSITION CLOSED</b>
━━━━━━━━━━━━━━━━━━━━━━━
📊 {position['instrument']}
💰 P&L: {pnl:+.4f} BTC ({pnl_pct:+.2f}%)
⏱️ Duration: {position['duration']}
━━━━━━━━━━━━━━━━━━━━━━━
💼 Balance: {position['new_balance']:.4f} BTC
📈 Total P&L: {position['total_pnl']:+.4f} BTC
🎯 Win Rate: {position['win_rate']:.1f}%"""
        
        await self.send_message(message)
        
    async def send_daily_summary(self, summary: Dict[str, Any]):
        """Send daily performance summary."""
        message = f"""📊 <b>DAILY SUMMARY</b>
━━━━━━━━━━━━━━━━━━━━━━━
📅 {summary['date']}
━━━━━━━━━━━━━━━━━━━━━━━
<b>Performance:</b>
• P&L: {summary['daily_pnl']:+.4f} BTC ({summary['daily_pnl_pct']:+.2f}%)
• Trades: {summary['trades']}
• Win Rate: {summary['win_rate']:.1f}%
• Best Trade: {summary['best_trade']:+.4f} BTC
• Worst Trade: {summary['worst_trade']:+.4f} BTC

<b>Portfolio:</b>
• Balance: {summary['balance']:.4f} BTC
• Total P&L: {summary['total_pnl']:+.4f} BTC
• Days Running: {summary['days_running']}
• Monthly Target Progress: {summary['monthly_progress']:.1f}%

<b>Risk Metrics:</b>
• Daily Drawdown: {summary['daily_drawdown']:.2f}%
• Max Drawdown: {summary['max_drawdown']:.2f}%
• Sharpe Ratio: {summary['sharpe_ratio']:.2f}"""
        
        await self.send_message(message)
        
    async def send_alert(self, alert: Dict[str, Any]):
        """Send risk or system alert."""
        emoji_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "CRITICAL": "🚨",
            "SUCCESS": "✅"
        }
        
        emoji = emoji_map.get(alert['level'], "📢")
        
        message = f"""{emoji} <b>{alert['title']}</b>
━━━━━━━━━━━━━━━━━━━━━━━
{alert['message']}
━━━━━━━━━━━━━━━━━━━━━━━
Time: {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(message)
        
    async def send_performance_update(self, metrics: Dict[str, Any]):
        """Send periodic performance update."""
        message = f"""📈 <b>PERFORMANCE UPDATE</b>
━━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━━━━
• Balance: {metrics['balance']:.4f} BTC
• P&L Today: {metrics['daily_pnl']:+.4f} BTC
• Total P&L: {metrics['total_pnl']:+.4f} BTC ({metrics['total_pnl_pct']:+.2f}%)
• Active Positions: {metrics['active_positions']}
• Today's Trades: {metrics['daily_trades']}
• Win Rate: {metrics['win_rate']:.1f}%"""
        
        await self.send_message(message)
        
    async def send_challenge_complete(self, summary: Dict[str, Any]):
        """Send challenge completion summary."""
        success = summary['total_pnl'] > 0 and summary['sharpe_ratio'] > 1.5
        emoji = "🏆" if success else "📊"
        
        message = f"""{emoji} <b>CHALLENGE COMPLETE!</b>
━━━━━━━━━━━━━━━━━━━━━━━
<b>Final Results:</b>
• Duration: {summary['duration_days']} days
• Total Trades: {summary['total_trades']}
• Final Balance: {summary['final_balance']:.4f} BTC
• Total P&L: {summary['total_pnl']:+.4f} BTC ({summary['total_pnl_pct']:+.2f}%)

<b>Performance Metrics:</b>
• Win Rate: {summary['win_rate']:.1f}%
• Profit Factor: {summary['profit_factor']:.2f}
• Sharpe Ratio: {summary['sharpe_ratio']:.2f}
• Max Drawdown: {summary['max_drawdown']:.2f}%
• Best Day: {summary['best_day']:+.4f} BTC
• Worst Day: {summary['worst_day']:+.4f} BTC

<b>Strategy Performance:</b>
{summary['strategy_breakdown']}

<b>Ready for Live Trading:</b> {"✅ YES" if success else "❌ NO"}
━━━━━━━━━━━━━━━━━━━━━━━
{"🎉 Congratulations! Ready for live trading!" if success else "📚 More practice needed. Review the results."}"""
        
        await self.send_message(message)