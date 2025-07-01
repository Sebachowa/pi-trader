#!/usr/bin/env python3
"""
Performance Tracker for Nautilus Challenge
Track and analyze trading performance metrics
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


class PerformanceTracker:
    """Track and analyze trading performance."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.trades = []
        self.daily_pnl = defaultdict(float)
        self.positions = []
        self.start_time = datetime.now()
        self.initial_balance = 0.3  # BTC
        self.current_balance = self.initial_balance
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        # Strategy performance
        self.strategy_performance = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'win_rate': 0.0
        })
        
    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade."""
        self.trades.append({
            **trade,
            'timestamp': datetime.now()
        })
        
        # Update metrics
        self.total_trades += 1
        pnl = trade['pnl']
        self.total_pnl += pnl
        self.current_balance += pnl
        
        # Update daily P&L
        today = datetime.now().date()
        self.daily_pnl[today] += pnl
        
        # Win/loss tracking
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        # Strategy tracking
        strategy = trade.get('strategy', 'unknown')
        self.strategy_performance[strategy]['trades'] += 1
        self.strategy_performance[strategy]['pnl'] += pnl
        if pnl > 0:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1
            
        # Update win rate
        strategy_data = self.strategy_performance[strategy]
        if strategy_data['trades'] > 0:
            strategy_data['win_rate'] = strategy_data['wins'] / strategy_data['trades']
            
        # Drawdown calculation
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        else:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # Calculate Sharpe ratio
        daily_returns = list(self.daily_pnl.values())
        if len(daily_returns) > 1:
            returns_array = np.array(daily_returns) / self.initial_balance
            sharpe = np.sqrt(252) * (np.mean(returns_array) / max(np.std(returns_array), 0.001))
        else:
            sharpe = 0.0
            
        # Calculate profit factor
        total_wins = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = total_wins / max(total_losses, 0.001)
        
        return {
            'balance': self.current_balance,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': (self.total_pnl / self.initial_balance) * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate * 100,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown * 100,
            'daily_pnl': self.daily_pnl[datetime.now().date()],
            'daily_trades': len([t for t in self.trades if t['timestamp'].date() == datetime.now().date()]),
            'active_positions': 0,  # Will be updated from live data
            'days_running': (datetime.now() - self.start_time).days
        }
        
    def get_daily_summary(self, date=None) -> Dict[str, Any]:
        """Get summary for a specific day."""
        if date is None:
            date = datetime.now().date()
            
        daily_trades = [t for t in self.trades if t['timestamp'].date() == date]
        daily_pnl = self.daily_pnl[date]
        
        if daily_trades:
            best_trade = max(t['pnl'] for t in daily_trades)
            worst_trade = min(t['pnl'] for t in daily_trades)
            daily_wins = sum(1 for t in daily_trades if t['pnl'] > 0)
            win_rate = daily_wins / len(daily_trades) * 100
        else:
            best_trade = worst_trade = 0
            win_rate = 0
            
        # Calculate monthly progress
        days_in_month = 30
        days_elapsed = min((datetime.now() - self.start_time).days, days_in_month)
        target_monthly = self.initial_balance * 0.0083  # 0.83% monthly target
        monthly_progress = (self.total_pnl / target_monthly) * 100 if days_elapsed > 0 else 0
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': (daily_pnl / self.initial_balance) * 100,
            'trades': len(daily_trades),
            'win_rate': win_rate,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'balance': self.current_balance,
            'total_pnl': self.total_pnl,
            'days_running': days_elapsed,
            'monthly_progress': monthly_progress,
            'daily_drawdown': 0,  # Will be calculated from intraday data
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': self.get_current_metrics()['sharpe_ratio']
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """Get complete challenge summary."""
        metrics = self.get_current_metrics()
        
        # Calculate best/worst days
        if self.daily_pnl:
            best_day = max(self.daily_pnl.values())
            worst_day = min(self.daily_pnl.values())
        else:
            best_day = worst_day = 0
            
        # Strategy breakdown
        strategy_lines = []
        for strategy, data in self.strategy_performance.items():
            if data['trades'] > 0:
                line = f"• {strategy}: {data['trades']} trades, " \
                       f"{data['win_rate']*100:.1f}% win rate, " \
                       f"{data['pnl']:+.4f} BTC"
                strategy_lines.append(line)
                
        return {
            'duration_days': (datetime.now() - self.start_time).days,
            'total_trades': self.total_trades,
            'final_balance': self.current_balance,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': metrics['total_pnl_pct'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'best_day': best_day,
            'worst_day': worst_day,
            'strategy_breakdown': '\n'.join(strategy_lines) if strategy_lines else 'No trades executed'
        }
        
    async def generate_final_report(self):
        """Generate final challenge report."""
        summary = self.get_summary()
        
        # Save to file
        report_path = Path(__file__).parent.parent / "logs" / f"challenge_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                'summary': summary,
                'trades': self.trades,
                'daily_pnl': {str(k): v for k, v in self.daily_pnl.items()},
                'strategy_performance': dict(self.strategy_performance),
                'config': {
                    'initial_balance': self.initial_balance,
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat()
                }
            }, f, indent=2, default=str)
            
        print(f"\n📊 Final report saved: {report_path}")
        
        return summary