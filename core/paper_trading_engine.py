"""
Paper Trading Engine - Uses real market data but simulates trades
"""
import logging
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict
import json
import os


class PaperTradingEngine:
    """Simulates trades using real market data without placing actual orders"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.order_id_counter = 1000
        self.logger = logging.getLogger(__name__)
        
        # Load existing paper trading state if exists
        self.state_file = "data/paper_trading_state.json"
        self._load_state()
    
    def create_market_buy_order(self, symbol: str, amount: float, price: float) -> Dict:
        """Simulate a market buy order"""
        # Calculate cost
        cost = amount * price
        fee = cost * 0.001  # 0.1% trading fee
        total_cost = cost + fee
        
        if total_cost > self.balance:
            raise Exception(f"Insufficient balance. Need {total_cost:.2f}, have {self.balance:.2f}")
        
        # Deduct from balance
        self.balance -= total_cost
        
        # Create position
        order_id = f"PAPER_{self.order_id_counter}"
        self.order_id_counter += 1
        
        self.positions[symbol] = {
            'amount': amount,
            'entry_price': price,
            'entry_time': datetime.now(),
            'current_price': price,
            'pnl': 0,
            'pnl_percent': 0
        }
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': 'buy',
            'amount': amount,
            'price': price,
            'cost': cost,
            'fee': fee,
            'status': 'filled',
            'timestamp': datetime.now().isoformat()
        }
        
        self.trade_history.append(order)
        self._save_state()
        
        self.logger.info(f"📝 PAPER BUY: {amount:.4f} {symbol} @ ${price:.4f} (${total_cost:.2f})")
        
        return order
    
    def create_market_sell_order(self, symbol: str, amount: float, price: float) -> Dict:
        """Simulate a market sell order"""
        if symbol not in self.positions:
            raise Exception(f"No position in {symbol}")
        
        position = self.positions[symbol]
        
        # Calculate proceeds
        proceeds = amount * price
        fee = proceeds * 0.001
        net_proceeds = proceeds - fee
        
        # Calculate PnL
        cost_basis = position['entry_price'] * amount
        pnl = net_proceeds - cost_basis
        pnl_percent = (pnl / cost_basis) * 100
        
        # Update balance
        self.balance += net_proceeds
        
        # Create order
        order_id = f"PAPER_{self.order_id_counter}"
        self.order_id_counter += 1
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': 'sell',
            'amount': amount,
            'price': price,
            'proceeds': proceeds,
            'fee': fee,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'status': 'filled',
            'timestamp': datetime.now().isoformat()
        }
        
        self.trade_history.append(order)
        
        # Remove position
        del self.positions[symbol]
        self._save_state()
        
        emoji = "💚" if pnl > 0 else "💔"
        self.logger.info(
            f"📝 PAPER SELL: {amount:.4f} {symbol} @ ${price:.4f} "
            f"{emoji} PnL: ${pnl:.2f} ({pnl_percent:+.2f}%)"
        )
        
        return order
    
    def update_position_prices(self, current_prices: Dict[str, float]):
        """Update current prices and PnL for all positions"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position['current_price'] = current_price
                
                # Calculate unrealized PnL
                value = position['amount'] * current_price
                cost = position['amount'] * position['entry_price']
                position['pnl'] = value - cost
                position['pnl_percent'] = (position['pnl'] / cost) * 100
    
    def get_balance_info(self) -> Dict:
        """Get account balance info"""
        # Calculate total position value
        position_value = sum(
            pos['amount'] * pos['current_price'] 
            for pos in self.positions.values()
        )
        
        total_equity = self.balance + position_value
        
        return {
            'USDT': {
                'free': self.balance,
                'used': position_value,
                'total': total_equity
            },
            'total_pnl': total_equity - self.initial_balance,
            'total_pnl_percent': ((total_equity - self.initial_balance) / self.initial_balance) * 100
        }
    
    def get_performance_stats(self) -> Dict:
        """Get trading performance statistics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        sells = [t for t in self.trade_history if t['side'] == 'sell']
        winning_trades = [t for t in sells if t['pnl'] > 0]
        losing_trades = [t for t in sells if t['pnl'] < 0]
        
        return {
            'total_trades': len(sells),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(sells) * 100) if sells else 0,
            'total_pnl': sum(t['pnl'] for t in sells),
            'avg_win': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'best_trade': max((t['pnl'] for t in sells), default=0),
            'worst_trade': min((t['pnl'] for t in sells), default=0)
        }
    
    def _save_state(self):
        """Save paper trading state to file"""
        os.makedirs('data', exist_ok=True)
        
        state = {
            'balance': self.balance,
            'positions': self.positions,
            'trade_history': self.trade_history[-100:],  # Keep last 100 trades
            'order_id_counter': self.order_id_counter,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_state(self):
        """Load paper trading state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.balance = state.get('balance', self.initial_balance)
                    self.positions = state.get('positions', {})
                    self.trade_history = state.get('trade_history', [])
                    self.order_id_counter = state.get('order_id_counter', 1000)
                    self.logger.info(f"📝 Loaded paper trading state: ${self.balance:.2f} balance")
            except Exception as e:
                self.logger.warning(f"Could not load paper trading state: {e}")