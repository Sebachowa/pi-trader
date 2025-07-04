#!/usr/bin/env python3
"""
Lightweight monitoring system for Raspberry Pi
"""
import json
import logging
import psutil
from datetime import datetime
from typing import Dict, Optional
import requests


class Monitor:
    """System and trading performance monitor"""
    
    def __init__(self, config: dict):
        self.config = config
        from core.logger import TradingLogger
        self.logger = TradingLogger.setup_logger(__name__)
        self.stats = {
            'start_time': datetime.now(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'peak_equity': 0.0,
            'current_equity': 0.0
        }
        
    def update(self, positions: Dict, exchange):
        """Update monitoring statistics"""
        try:
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Get account balance
            balance = exchange.fetch_balance()
            total_balance = balance['USDT']['total'] if 'USDT' in balance else 0
            
            # Update equity tracking
            self.stats['current_equity'] = total_balance
            if total_balance > self.stats['peak_equity']:
                self.stats['peak_equity'] = total_balance
            
            # Log status with enhanced formatting
            from core.logger import TradingLogger
            TradingLogger.log_system_status(
                self.logger,
                cpu_percent,
                memory_percent,
                len(positions),
                total_balance
            )
            
            # Send notification if enabled
            if self.config['enable_notifications']:
                self._send_notification({
                    'cpu': cpu_percent,
                    'memory': memory_percent,
                    'positions': len(positions),
                    'equity': total_balance
                })
                
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
    
    def record_trade(self, symbol: str, pnl: float):
        """Record trade statistics"""
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        
        if pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 3600
        win_rate = (self.stats['winning_trades'] / self.stats['total_trades'] * 100) \
                   if self.stats['total_trades'] > 0 else 0
        
        return {
            'runtime_hours': round(runtime, 2),
            'total_trades': self.stats['total_trades'],
            'win_rate': round(win_rate, 2),
            'total_pnl': round(self.stats['total_pnl'], 2),
            'current_equity': round(self.stats['current_equity'], 2),
            'peak_equity': round(self.stats['peak_equity'], 2)
        }
    
    def _send_notification(self, data: Dict):
        """Send webhook notification"""
        if not self.config['webhook_url']:
            return
        
        try:
            payload = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': data['cpu'],
                    'memory_percent': data['memory']
                },
                'trading': {
                    'positions': data['positions'],
                    'equity': data['equity']
                }
            }
            
            requests.post(
                self.config['webhook_url'],
                json=payload,
                timeout=5
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    def save_stats(self, filepath: str = "stats.json"):
        """Save statistics to file"""
        try:
            stats_data = {
                'timestamp': datetime.now().isoformat(),
                'performance': self.get_performance_summary(),
                'detailed_stats': self.stats
            }
            
            with open(filepath, 'w') as f:
                json.dump(stats_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save stats: {e}")