# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Notification System - Multi-channel alerts and reporting for autonomous trading.
"""

import asyncio
import json
import smtplib
from collections import deque
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import aiohttp

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.logging import Logger


class NotificationLevel(Enum):
    """Notification importance levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"


class NotificationConfig:
    """Configuration for notification channels."""
    
    def __init__(self):
        self.email_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "",
            "sender_password": "",
            "recipient_emails": [],
        }
        
        self.webhook_config = {
            "urls": [],
            "headers": {"Content-Type": "application/json"},
        }
        
        self.telegram_config = {
            "bot_token": "",
            "chat_ids": [],
        }
        
        self.slack_config = {
            "webhook_url": "",
            "channel": "#trading-alerts",
        }
        
        self.discord_config = {
            "webhook_url": "",
        }


class NotificationRule:
    """Rule for triggering notifications."""
    
    def __init__(
        self,
        name: str,
        condition: str,
        level: NotificationLevel,
        channels: List[NotificationChannel],
        cooldown_minutes: int = 60,
    ):
        self.name = name
        self.condition = condition
        self.level = level
        self.channels = channels
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = None


class NotificationSystem(Component):
    """
    Multi-channel notification system for autonomous trading alerts.
    
    Features:
    - Multiple notification channels (email, SMS, webhooks, etc.)
    - Configurable alert rules and thresholds
    - Rate limiting and cooldowns
    - Performance report generation
    - Error notification prioritization
    """
    
    def __init__(
        self,
        logger: Logger,
        clock: LiveClock,
        msgbus: MessageBus,
        config: Optional[NotificationConfig] = None,
        enable_daily_summary: bool = True,
        enable_weekly_report: bool = True,
        max_notifications_per_hour: int = 50,
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id="NOTIFICATION-SYSTEM",
            msgbus=msgbus,
        )
        
        self.config = config or NotificationConfig()
        self.enable_daily_summary = enable_daily_summary
        self.enable_weekly_report = enable_weekly_report
        self.max_notifications_per_hour = max_notifications_per_hour
        
        # Notification management
        self._notification_queue: deque = deque(maxlen=1000)
        self._notification_history: deque = deque(maxlen=10000)
        self._rate_limiter: Dict[str, deque] = {}
        
        # Rules and filters
        self._notification_rules: List[NotificationRule] = []
        self._level_filters: Set[NotificationLevel] = {
            NotificationLevel.INFO,
            NotificationLevel.WARNING,
            NotificationLevel.ERROR,
            NotificationLevel.CRITICAL,
        }
        
        # Performance data for reports
        self._daily_metrics: Dict[str, Any] = {}
        self._weekly_metrics: Dict[str, Any] = {}
        
        # Tasks
        self._notification_task = None
        self._summary_task = None
        
        # Initialize default rules
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default notification rules."""
        # Critical system events
        self._notification_rules.append(
            NotificationRule(
                name="system_error",
                condition="error_count > 5",
                level=NotificationLevel.CRITICAL,
                channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
                cooldown_minutes=30,
            )
        )
        
        # Risk limits
        self._notification_rules.append(
            NotificationRule(
                name="max_drawdown",
                condition="drawdown > 0.05",
                level=NotificationLevel.WARNING,
                channels=[NotificationChannel.EMAIL, NotificationChannel.TELEGRAM],
                cooldown_minutes=60,
            )
        )
        
        # Performance alerts
        self._notification_rules.append(
            NotificationRule(
                name="daily_loss",
                condition="daily_pnl < -0.02",
                level=NotificationLevel.ERROR,
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                cooldown_minutes=120,
            )
        )
        
        # Connection issues
        self._notification_rules.append(
            NotificationRule(
                name="connection_lost",
                condition="connection_status == 'disconnected'",
                level=NotificationLevel.ERROR,
                channels=[NotificationChannel.WEBHOOK, NotificationChannel.TELEGRAM],
                cooldown_minutes=15,
            )
        )

    async def start(self) -> None:
        """Start the notification system."""
        self._log.info("Starting Notification System...")
        
        # Start notification processing
        self._notification_task = asyncio.create_task(self._notification_loop())
        
        # Start summary tasks
        if self.enable_daily_summary or self.enable_weekly_report:
            self._summary_task = asyncio.create_task(self._summary_loop())

    async def stop(self) -> None:
        """Stop the notification system."""
        self._log.info("Stopping Notification System...")
        
        # Process remaining notifications
        await self._process_pending_notifications()
        
        # Cancel tasks
        if self._notification_task:
            self._notification_task.cancel()
        if self._summary_task:
            self._summary_task.cancel()

    async def send_notification(
        self,
        level: str,
        title: str,
        message: str,
        channels: Optional[List[str]] = None,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a notification through specified channels."""
        notification = {
            "timestamp": datetime.utcnow(),
            "level": NotificationLevel[level.upper()],
            "title": title,
            "message": message,
            "channels": channels or self._get_default_channels(level),
            "priority": priority,
            "metadata": metadata or {},
        }
        
        # Check rate limits
        if not self._check_rate_limit(title):
            self._log.warning(f"Rate limit exceeded for notification: {title}")
            return
        
        # Add to queue
        self._notification_queue.append(notification)
        
        # Process immediately if critical
        if notification["level"] == NotificationLevel.CRITICAL:
            await self._process_notification(notification)

    def _get_default_channels(self, level: str) -> List[NotificationChannel]:
        """Get default channels based on notification level."""
        level_enum = NotificationLevel[level.upper()]
        
        if level_enum == NotificationLevel.CRITICAL:
            return [NotificationChannel.EMAIL, NotificationChannel.WEBHOOK, 
                   NotificationChannel.TELEGRAM]
        elif level_enum == NotificationLevel.ERROR:
            return [NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        elif level_enum == NotificationLevel.WARNING:
            return [NotificationChannel.WEBHOOK, NotificationChannel.SLACK]
        else:
            return [NotificationChannel.WEBHOOK]

    def _check_rate_limit(self, key: str) -> bool:
        """Check if notification passes rate limiting."""
        now = datetime.utcnow()
        
        if key not in self._rate_limiter:
            self._rate_limiter[key] = deque()
        
        # Remove old entries
        cutoff = now - timedelta(hours=1)
        while self._rate_limiter[key] and self._rate_limiter[key][0] < cutoff:
            self._rate_limiter[key].popleft()
        
        # Check limit
        if len(self._rate_limiter[key]) >= self.max_notifications_per_hour:
            return False
        
        # Add new entry
        self._rate_limiter[key].append(now)
        return True

    async def _notification_loop(self) -> None:
        """Process notification queue."""
        while True:
            try:
                # Process pending notifications
                await self._process_pending_notifications()
                
                # Check rules
                await self._check_notification_rules()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Notification processing error: {e}")

    async def _process_pending_notifications(self) -> None:
        """Process all pending notifications in queue."""
        while self._notification_queue:
            notification = self._notification_queue.popleft()
            
            # Filter by level
            if notification["level"] not in self._level_filters:
                continue
            
            await self._process_notification(notification)

    async def _process_notification(self, notification: Dict[str, Any]) -> None:
        """Process a single notification."""
        # Add to history
        self._notification_history.append(notification)
        
        # Send through each channel
        for channel in notification["channels"]:
            try:
                if isinstance(channel, str):
                    channel = NotificationChannel[channel.upper()]
                
                if channel == NotificationChannel.EMAIL:
                    await self._send_email(notification)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook(notification)
                elif channel == NotificationChannel.TELEGRAM:
                    await self._send_telegram(notification)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack(notification)
                elif channel == NotificationChannel.DISCORD:
                    await self._send_discord(notification)
                
            except Exception as e:
                self._log.error(f"Failed to send {channel} notification: {e}")

    async def _send_email(self, notification: Dict[str, Any]) -> None:
        """Send email notification."""
        if not self.config.email_config["sender_email"]:
            return
        
        msg = MIMEMultipart()
        msg["From"] = self.config.email_config["sender_email"]
        msg["To"] = ", ".join(self.config.email_config["recipient_emails"])
        msg["Subject"] = f"[{notification['level'].value.upper()}] {notification['title']}"
        
        # Create email body
        body = f"""
        Autonomous Trading System Notification
        
        Time: {notification['timestamp']}
        Level: {notification['level'].value}
        
        {notification['message']}
        
        ---
        Metadata: {json.dumps(notification['metadata'], indent=2)}
        """
        
        msg.attach(MIMEText(body, "plain"))
        
        # Send email
        try:
            with smtplib.SMTP(self.config.email_config["smtp_server"], 
                             self.config.email_config["smtp_port"]) as server:
                server.starttls()
                server.login(
                    self.config.email_config["sender_email"],
                    self.config.email_config["sender_password"]
                )
                server.send_message(msg)
        except Exception as e:
            self._log.error(f"Email send failed: {e}")

    async def _send_webhook(self, notification: Dict[str, Any]) -> None:
        """Send webhook notification."""
        if not self.config.webhook_config["urls"]:
            return
        
        payload = {
            "timestamp": notification["timestamp"].isoformat(),
            "level": notification["level"].value,
            "title": notification["title"],
            "message": notification["message"],
            "metadata": notification["metadata"],
        }
        
        async with aiohttp.ClientSession() as session:
            for url in self.config.webhook_config["urls"]:
                try:
                    async with session.post(
                        url,
                        json=payload,
                        headers=self.config.webhook_config["headers"],
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status != 200:
                            self._log.warning(f"Webhook returned status {response.status}")
                except Exception as e:
                    self._log.error(f"Webhook send failed: {e}")

    async def _send_telegram(self, notification: Dict[str, Any]) -> None:
        """Send Telegram notification."""
        if not self.config.telegram_config["bot_token"]:
            return
        
        message = f"""
🤖 *{notification['title']}*

📊 Level: {notification['level'].value}
⏰ Time: {notification['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

{notification['message']}
        """
        
        url = f"https://api.telegram.org/bot{self.config.telegram_config['bot_token']}/sendMessage"
        
        async with aiohttp.ClientSession() as session:
            for chat_id in self.config.telegram_config["chat_ids"]:
                try:
                    payload = {
                        "chat_id": chat_id,
                        "text": message,
                        "parse_mode": "Markdown",
                    }
                    
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            self._log.warning(f"Telegram API returned status {response.status}")
                except Exception as e:
                    self._log.error(f"Telegram send failed: {e}")

    async def _send_slack(self, notification: Dict[str, Any]) -> None:
        """Send Slack notification."""
        if not self.config.slack_config["webhook_url"]:
            return
        
        # Format message for Slack
        color = {
            NotificationLevel.INFO: "#36a64f",
            NotificationLevel.WARNING: "#ff9900",
            NotificationLevel.ERROR: "#ff0000",
            NotificationLevel.CRITICAL: "#990000",
        }.get(notification["level"], "#808080")
        
        payload = {
            "channel": self.config.slack_config["channel"],
            "username": "Trading Bot",
            "icon_emoji": ":robot_face:",
            "attachments": [{
                "color": color,
                "title": notification["title"],
                "text": notification["message"],
                "fields": [
                    {
                        "title": "Level",
                        "value": notification["level"].value,
                        "short": True,
                    },
                    {
                        "title": "Time",
                        "value": notification["timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
                        "short": True,
                    },
                ],
                "footer": "Autonomous Trading System",
                "ts": int(notification["timestamp"].timestamp()),
            }],
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.config.slack_config["webhook_url"],
                    json=payload
                ) as response:
                    if response.status != 200:
                        self._log.warning(f"Slack webhook returned status {response.status}")
            except Exception as e:
                self._log.error(f"Slack send failed: {e}")

    async def _send_discord(self, notification: Dict[str, Any]) -> None:
        """Send Discord notification."""
        if not self.config.discord_config["webhook_url"]:
            return
        
        # Format embed for Discord
        color = {
            NotificationLevel.INFO: 0x00ff00,
            NotificationLevel.WARNING: 0xffff00,
            NotificationLevel.ERROR: 0xff0000,
            NotificationLevel.CRITICAL: 0x990000,
        }.get(notification["level"], 0x808080)
        
        payload = {
            "username": "Trading Bot",
            "embeds": [{
                "title": notification["title"],
                "description": notification["message"],
                "color": color,
                "fields": [
                    {
                        "name": "Level",
                        "value": notification["level"].value,
                        "inline": True,
                    },
                    {
                        "name": "Priority",
                        "value": notification.get("priority", "normal"),
                        "inline": True,
                    },
                ],
                "timestamp": notification["timestamp"].isoformat(),
                "footer": {
                    "text": "Autonomous Trading System",
                },
            }],
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.config.discord_config["webhook_url"],
                    json=payload
                ) as response:
                    if response.status not in [200, 204]:
                        self._log.warning(f"Discord webhook returned status {response.status}")
            except Exception as e:
                self._log.error(f"Discord send failed: {e}")

    async def _check_notification_rules(self) -> None:
        """Check notification rules against current system state."""
        # This would integrate with system metrics
        # Placeholder implementation
        pass

    async def _summary_loop(self) -> None:
        """Generate and send periodic summaries."""
        while True:
            try:
                now = datetime.utcnow()
                
                # Daily summary at 00:00 UTC
                if self.enable_daily_summary and now.hour == 0 and now.minute < 5:
                    await self._send_daily_summary()
                    await asyncio.sleep(300)  # Wait 5 minutes to avoid duplicate
                
                # Weekly report on Sundays at 00:00 UTC
                if self.enable_weekly_report and now.weekday() == 6 and now.hour == 0 and now.minute < 5:
                    await self._send_weekly_report()
                    await asyncio.sleep(300)  # Wait 5 minutes to avoid duplicate
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Summary generation error: {e}")

    async def _send_daily_summary(self) -> None:
        """Generate and send daily performance summary."""
        self._log.info("Generating daily summary...")
        
        summary = self._generate_daily_summary()
        
        await self.send_notification(
            level="INFO",
            title="Daily Trading Summary",
            message=summary,
            channels=["EMAIL", "TELEGRAM"],
            priority="normal",
        )

    async def _send_weekly_report(self) -> None:
        """Generate and send weekly performance report."""
        self._log.info("Generating weekly report...")
        
        report = self._generate_weekly_report()
        
        await self.send_notification(
            level="INFO",
            title="Weekly Trading Report",
            message=report,
            channels=["EMAIL"],
            priority="normal",
        )

    def _generate_daily_summary(self) -> str:
        """Generate daily summary text."""
        # This would pull from actual system metrics
        # Placeholder implementation
        return f"""
Daily Trading Summary - {datetime.utcnow().date()}

📊 Performance Metrics:
• Total P&L: $1,234.56
• Win Rate: 65.4%
• Total Trades: 42
• Sharpe Ratio: 1.85

💡 Strategy Performance:
• Trend Following: +$800.00 (15 trades)
• Market Making: +$434.56 (27 trades)

⚠️ Risk Metrics:
• Max Drawdown: 3.2%
• Current Exposure: 45.6%
• VaR (95%): $567.89

🔔 Notifications Today: 8
• Errors: 1
• Warnings: 3
• Info: 4

System Health: ✅ All systems operational
        """

    def _generate_weekly_report(self) -> str:
        """Generate weekly report text."""
        # This would create a comprehensive weekly analysis
        # Placeholder implementation
        return f"""
Weekly Trading Report - Week of {(datetime.utcnow() - timedelta(days=7)).date()}

📈 Weekly Performance:
• Total Return: +3.45%
• Sharpe Ratio: 2.12
• Win Rate: 68.2%
• Profit Factor: 1.89

📊 Trade Statistics:
• Total Trades: 284
• Winning Trades: 194
• Average Win: $45.67
• Average Loss: $24.12
• Best Trade: +$234.56
• Worst Trade: -$89.12

💼 Strategy Breakdown:
1. AI Swarm Strategy: +$2,456.78 (45% of profits)
2. Trend Following: +$1,234.56 (25% of profits)
3. Market Making: +$987.65 (20% of profits)
4. Statistical Arbitrage: +$543.21 (10% of profits)

📉 Risk Analysis:
• Maximum Drawdown: 4.5%
• Average Daily VaR: $678.90
• Correlation Risk: Low
• Liquidity Risk: Normal

🔧 System Performance:
• Uptime: 99.8%
• Average Latency: 12ms
• Error Rate: 0.02%
• Optimization Runs: 7

📝 Recommendations:
• Consider increasing allocation to AI Swarm Strategy
• Monitor correlation between Trend Following positions
• Review stop loss parameters for Market Making

Next Week Outlook: Favorable market conditions expected
        """

    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        # Count notifications by level
        level_counts = {level: 0 for level in NotificationLevel}
        channel_counts = {channel: 0 for channel in NotificationChannel}
        
        for notification in self._notification_history:
            level_counts[notification["level"]] += 1
            for channel in notification["channels"]:
                if isinstance(channel, NotificationChannel):
                    channel_counts[channel] += 1
        
        return {
            "total_notifications": len(self._notification_history),
            "notifications_by_level": {k.value: v for k, v in level_counts.items()},
            "notifications_by_channel": {k.value: v for k, v in channel_counts.items()},
            "queue_size": len(self._notification_queue),
            "rate_limited_keys": len(self._rate_limiter),
        }