"""
Tax calculation module for cryptocurrency trading
Supports multiple jurisdictions and calculation methods
"""
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from decimal import Decimal
import pandas as pd
from collections import defaultdict


@dataclass
class TaxableEvent:
    """Represents a taxable trading event"""
    timestamp: datetime
    event_type: str  # 'trade', 'deposit', 'withdrawal', 'fee'
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    fee_currency: str
    proceeds: float  # For sells
    cost_basis: float  # For sells
    gain_loss: float  # Realized P&L
    holding_period_days: int
    tax_lot_id: str  # For tracking specific lots


class TaxCalculator:
    """
    Calculate taxes for cryptocurrency trading
    Supports FIFO, LIFO, and specific lot identification
    """
    
    def __init__(self, tax_config: Dict):
        self.config = tax_config
        self.method = tax_config.get('method', 'FIFO')  # FIFO, LIFO, HIFO
        self.tax_year = tax_config.get('tax_year', datetime.now().year)
        
        # Tax rates by jurisdiction
        self.short_term_rate = tax_config.get('short_term_rate', 0.35)
        self.long_term_rate = tax_config.get('long_term_rate', 0.15)
        self.long_term_threshold_days = tax_config.get('long_term_days', 365)
        
        # Transaction tracking
        self.transactions = []
        self.tax_lots = defaultdict(list)  # symbol -> list of lots
        self.taxable_events = []
        
        # Summary data
        self.realized_gains = Decimal('0')
        self.realized_losses = Decimal('0')
        self.fees_paid = Decimal('0')
        
    def add_transaction(self, transaction: Dict):
        """Add a trading transaction"""
        self.transactions.append(transaction)
        
        if transaction['side'] == 'buy':
            self._add_tax_lot(transaction)
        elif transaction['side'] == 'sell':
            self._process_sale(transaction)
    
    def _add_tax_lot(self, buy_transaction: Dict):
        """Add a tax lot for a buy transaction"""
        lot = {
            'id': f"{buy_transaction['symbol']}_{buy_transaction['timestamp']}",
            'timestamp': buy_transaction['timestamp'],
            'symbol': buy_transaction['symbol'],
            'quantity': buy_transaction['quantity'],
            'remaining_quantity': buy_transaction['quantity'],
            'price': buy_transaction['price'],
            'fee': buy_transaction.get('fee', 0),
            'cost_basis': (buy_transaction['quantity'] * buy_transaction['price']) + 
                         buy_transaction.get('fee', 0)
        }
        self.tax_lots[buy_transaction['symbol']].append(lot)
    
    def _process_sale(self, sell_transaction: Dict):
        """Process a sale and calculate gains/losses"""
        symbol = sell_transaction['symbol']
        quantity_to_sell = sell_transaction['quantity']
        sell_price = sell_transaction['price']
        sell_timestamp = datetime.fromisoformat(sell_transaction['timestamp'])
        
        if symbol not in self.tax_lots or not self.tax_lots[symbol]:
            # No cost basis - might be from before tracking started
            self._create_unknown_cost_basis_event(sell_transaction)
            return
        
        # Sort lots based on method
        lots = self._sort_lots_by_method(self.tax_lots[symbol])
        
        remaining_quantity = quantity_to_sell
        total_cost_basis = Decimal('0')
        taxable_events = []
        
        for lot in lots:
            if remaining_quantity <= 0:
                break
            
            if lot['remaining_quantity'] <= 0:
                continue
            
            # Calculate how much to take from this lot
            quantity_from_lot = min(remaining_quantity, lot['remaining_quantity'])
            
            # Calculate cost basis for this portion
            lot_cost_basis = (quantity_from_lot / lot['quantity']) * lot['cost_basis']
            total_cost_basis += Decimal(str(lot_cost_basis))
            
            # Calculate proceeds and gain/loss
            proceeds = quantity_from_lot * sell_price
            gain_loss = proceeds - float(lot_cost_basis)
            
            # Calculate holding period
            buy_timestamp = datetime.fromisoformat(lot['timestamp'])
            holding_days = (sell_timestamp - buy_timestamp).days
            
            # Create taxable event
            event = TaxableEvent(
                timestamp=sell_timestamp,
                event_type='trade',
                symbol=symbol,
                side='sell',
                quantity=quantity_from_lot,
                price=sell_price,
                fee=sell_transaction.get('fee', 0) * (quantity_from_lot / quantity_to_sell),
                fee_currency='USDT',
                proceeds=proceeds,
                cost_basis=float(lot_cost_basis),
                gain_loss=gain_loss,
                holding_period_days=holding_days,
                tax_lot_id=lot['id']
            )
            
            taxable_events.append(event)
            self.taxable_events.append(event)
            
            # Update lot
            lot['remaining_quantity'] -= quantity_from_lot
            remaining_quantity -= quantity_from_lot
            
            # Update summary
            if gain_loss > 0:
                self.realized_gains += Decimal(str(gain_loss))
            else:
                self.realized_losses += Decimal(str(gain_loss))
        
        # Clean up empty lots
        self.tax_lots[symbol] = [lot for lot in self.tax_lots[symbol] 
                                if lot['remaining_quantity'] > 0]
    
    def _sort_lots_by_method(self, lots: List[Dict]) -> List[Dict]:
        """Sort tax lots based on the selected method"""
        if self.method == 'FIFO':
            return sorted(lots, key=lambda x: x['timestamp'])
        elif self.method == 'LIFO':
            return sorted(lots, key=lambda x: x['timestamp'], reverse=True)
        elif self.method == 'HIFO':
            return sorted(lots, key=lambda x: x['price'], reverse=True)
        else:
            return lots
    
    def _create_unknown_cost_basis_event(self, sell_transaction: Dict):
        """Create event for sales with unknown cost basis"""
        event = TaxableEvent(
            timestamp=datetime.fromisoformat(sell_transaction['timestamp']),
            event_type='trade',
            symbol=sell_transaction['symbol'],
            side='sell',
            quantity=sell_transaction['quantity'],
            price=sell_transaction['price'],
            fee=sell_transaction.get('fee', 0),
            fee_currency='USDT',
            proceeds=sell_transaction['quantity'] * sell_transaction['price'],
            cost_basis=0,  # Unknown
            gain_loss=sell_transaction['quantity'] * sell_transaction['price'],  # Assume all gain
            holding_period_days=366,  # Assume long-term
            tax_lot_id='UNKNOWN'
        )
        self.taxable_events.append(event)
    
    def calculate_tax_summary(self) -> Dict:
        """Calculate tax summary for the year"""
        year_events = [e for e in self.taxable_events 
                      if e.timestamp.year == self.tax_year]
        
        short_term_gains = Decimal('0')
        short_term_losses = Decimal('0')
        long_term_gains = Decimal('0')
        long_term_losses = Decimal('0')
        
        for event in year_events:
            if event.holding_period_days < self.long_term_threshold_days:
                if event.gain_loss > 0:
                    short_term_gains += Decimal(str(event.gain_loss))
                else:
                    short_term_losses += Decimal(str(event.gain_loss))
            else:
                if event.gain_loss > 0:
                    long_term_gains += Decimal(str(event.gain_loss))
                else:
                    long_term_losses += Decimal(str(event.gain_loss))
        
        # Calculate net gains/losses
        net_short_term = short_term_gains + short_term_losses
        net_long_term = long_term_gains + long_term_losses
        
        # Calculate taxes
        short_term_tax = float(net_short_term) * self.short_term_rate if net_short_term > 0 else 0
        long_term_tax = float(net_long_term) * self.long_term_rate if net_long_term > 0 else 0
        
        return {
            'tax_year': self.tax_year,
            'method': self.method,
            'summary': {
                'short_term_gains': float(short_term_gains),
                'short_term_losses': float(short_term_losses),
                'net_short_term': float(net_short_term),
                'long_term_gains': float(long_term_gains),
                'long_term_losses': float(long_term_losses),
                'net_long_term': float(net_long_term),
                'total_proceeds': sum(e.proceeds for e in year_events),
                'total_cost_basis': sum(e.cost_basis for e in year_events),
                'total_fees': float(self.fees_paid),
                'number_of_trades': len(year_events)
            },
            'tax_liability': {
                'short_term_tax': short_term_tax,
                'long_term_tax': long_term_tax,
                'total_tax': short_term_tax + long_term_tax,
                'effective_rate': (short_term_tax + long_term_tax) / 
                                float(net_short_term + net_long_term) 
                                if (net_short_term + net_long_term) > 0 else 0
            },
            'quarterly_estimates': self._calculate_quarterly_estimates(
                short_term_tax + long_term_tax
            )
        }
    
    def _calculate_quarterly_estimates(self, annual_tax: float) -> Dict:
        """Calculate quarterly estimated tax payments"""
        # IRS safe harbor: 90% of current year or 100% of prior year
        safe_harbor = annual_tax * 0.9
        quarterly = safe_harbor / 4
        
        current_date = datetime.now()
        year = self.tax_year
        
        return {
            'Q1': {
                'due_date': f"{year}-04-15",
                'amount': quarterly,
                'paid': current_date > datetime(year, 4, 15)
            },
            'Q2': {
                'due_date': f"{year}-06-15",
                'amount': quarterly,
                'paid': current_date > datetime(year, 6, 15)
            },
            'Q3': {
                'due_date': f"{year}-09-15",
                'amount': quarterly,
                'paid': current_date > datetime(year, 9, 15)
            },
            'Q4': {
                'due_date': f"{year + 1}-01-15",
                'amount': quarterly,
                'paid': current_date > datetime(year + 1, 1, 15)
            }
        }
    
    def export_form_8949(self, filename: str = None):
        """Export IRS Form 8949 format (USA)"""
        if filename is None:
            filename = f"form_8949_{self.tax_year}.csv"
        
        year_events = [e for e in self.taxable_events 
                      if e.timestamp.year == self.tax_year]
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Headers as per Form 8949
            writer.writerow([
                'Description of Property',
                'Date Acquired',
                'Date Sold',
                'Proceeds',
                'Cost Basis',
                'Gain or Loss',
                'Short/Long Term'
            ])
            
            for event in year_events:
                term = 'Short' if event.holding_period_days < self.long_term_threshold_days else 'Long'
                
                # Find acquisition date
                buy_date = (event.timestamp - timedelta(days=event.holding_period_days)).strftime('%m/%d/%Y')
                
                writer.writerow([
                    f"{event.quantity:.8f} {event.symbol}",
                    buy_date,
                    event.timestamp.strftime('%m/%d/%Y'),
                    f"${event.proceeds:.2f}",
                    f"${event.cost_basis:.2f}",
                    f"${event.gain_loss:.2f}",
                    term
                ])
        
        return filename
    
    def export_summary_report(self, filename: str = None):
        """Export a comprehensive tax summary report"""
        if filename is None:
            filename = f"tax_summary_{self.tax_year}.json"
        
        summary = self.calculate_tax_summary()
        
        # Add detailed transaction list
        summary['transactions'] = [
            {
                'date': e.timestamp.isoformat(),
                'symbol': e.symbol,
                'quantity': e.quantity,
                'proceeds': e.proceeds,
                'cost_basis': e.cost_basis,
                'gain_loss': e.gain_loss,
                'holding_days': e.holding_period_days,
                'tax_rate': self.short_term_rate if e.holding_period_days < self.long_term_threshold_days 
                           else self.long_term_rate
            }
            for e in self.taxable_events
            if e.timestamp.year == self.tax_year
        ]
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return filename
    
    def generate_tax_loss_harvesting_suggestions(self) -> List[Dict]:
        """Suggest positions to sell for tax loss harvesting"""
        suggestions = []
        
        for symbol, lots in self.tax_lots.items():
            for lot in lots:
                if lot['remaining_quantity'] <= 0:
                    continue
                
                # Assume current price (would need to fetch real price)
                current_price = lot['price'] * 0.9  # Mock 10% loss
                current_value = lot['remaining_quantity'] * current_price
                cost_basis = (lot['remaining_quantity'] / lot['quantity']) * lot['cost_basis']
                
                potential_loss = current_value - cost_basis
                
                if potential_loss < 0:  # Only losses
                    suggestions.append({
                        'symbol': symbol,
                        'quantity': lot['remaining_quantity'],
                        'current_price': current_price,
                        'cost_basis': cost_basis,
                        'potential_loss': potential_loss,
                        'tax_savings': abs(potential_loss) * self.short_term_rate,
                        'lot_date': lot['timestamp']
                    })
        
        # Sort by tax savings
        suggestions.sort(key=lambda x: x['tax_savings'], reverse=True)
        
        return suggestions