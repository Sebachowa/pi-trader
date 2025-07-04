"""
Real-time tax monitoring for trading positions
Shows tax implications before closing positions
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class TaxMonitor:
    """
    Monitor tax implications in real-time
    Helps make tax-efficient trading decisions
    """
    
    def __init__(self, tax_config: Dict):
        self.config = tax_config
        self.short_term_rate = tax_config.get('short_term_rate', 0.35)
        self.long_term_rate = tax_config.get('long_term_rate', 0.15)
        self.long_term_days = tax_config.get('long_term_days', 365)
        
        # Tracking
        self.current_year_gains = Decimal('0')
        self.current_year_losses = Decimal('0')
        self.current_quarter_tax = Decimal('0')
        
    def analyze_position_tax_impact(self, position, current_price: float) -> Dict:
        """
        Analyze tax impact of closing a position
        """
        # Calculate potential P&L
        if position.side == 'buy':
            potential_pnl = (current_price - position.entry_price) * position.quantity
        else:
            potential_pnl = (position.entry_price - current_price) * position.quantity
        
        # Calculate holding period
        holding_days = (datetime.now() - position.opened_at).days
        is_long_term = holding_days >= self.long_term_days
        
        # Determine tax rate
        if is_long_term:
            tax_rate = self.long_term_rate
            term_type = "long-term"
        else:
            tax_rate = self.short_term_rate
            term_type = "short-term"
        
        # Calculate tax impact
        if potential_pnl > 0:
            tax_liability = potential_pnl * tax_rate
        else:
            # Losses can offset gains
            tax_liability = 0
            tax_benefit = abs(potential_pnl) * tax_rate
        
        # Days until long-term
        days_to_long_term = max(0, self.long_term_days - holding_days)
        
        # Tax if we wait for long-term
        if not is_long_term and potential_pnl > 0:
            tax_if_wait = potential_pnl * self.long_term_rate
            tax_savings = tax_liability - tax_if_wait
        else:
            tax_if_wait = tax_liability
            tax_savings = 0
        
        return {
            'symbol': position.symbol,
            'holding_days': holding_days,
            'is_long_term': is_long_term,
            'term_type': term_type,
            'potential_pnl': potential_pnl,
            'tax_rate': tax_rate,
            'tax_liability': tax_liability,
            'tax_benefit': tax_benefit if potential_pnl < 0 else 0,
            'after_tax_profit': potential_pnl - tax_liability,
            'days_to_long_term': days_to_long_term,
            'tax_if_wait': tax_if_wait,
            'potential_tax_savings': tax_savings,
            'recommendation': self._get_tax_recommendation(
                potential_pnl, holding_days, days_to_long_term, tax_savings
            )
        }
    
    def _get_tax_recommendation(
        self, 
        pnl: float, 
        holding_days: int, 
        days_to_long_term: int,
        tax_savings: float
    ) -> str:
        """Generate tax-based recommendation"""
        
        # Loss harvesting
        if pnl < 0:
            if self.current_year_gains > abs(pnl):
                return "HARVEST_LOSS: Sell to offset gains"
            else:
                return "HOLD: Save loss for future gains"
        
        # Profitable position
        if days_to_long_term > 0 and days_to_long_term <= 30:
            return f"WAIT: {days_to_long_term} days to save ${tax_savings:.2f} in taxes"
        elif days_to_long_term > 30 and tax_savings > pnl * 0.1:
            return "CONSIDER_WAITING: Significant tax savings possible"
        else:
            return "TAX_NEUTRAL: Proceed based on market conditions"
    
    def get_portfolio_tax_summary(self, positions: Dict, current_prices: Dict) -> Dict:
        """Get tax summary for entire portfolio"""
        
        short_term_unrealized = Decimal('0')
        long_term_unrealized = Decimal('0')
        harvest_opportunities = []
        wait_opportunities = []
        
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position.current_price)
            analysis = self.analyze_position_tax_impact(position, current_price)
            
            if analysis['is_long_term']:
                long_term_unrealized += Decimal(str(analysis['potential_pnl']))
            else:
                short_term_unrealized += Decimal(str(analysis['potential_pnl']))
            
            # Identify opportunities
            if analysis['potential_pnl'] < 0 and abs(analysis['potential_pnl']) > 100:
                harvest_opportunities.append(analysis)
            
            if analysis['days_to_long_term'] > 0 and analysis['days_to_long_term'] <= 30:
                wait_opportunities.append(analysis)
        
        # Calculate current tax liability
        estimated_tax = 0
        if short_term_unrealized > 0:
            estimated_tax += float(short_term_unrealized) * self.short_term_rate
        if long_term_unrealized > 0:
            estimated_tax += float(long_term_unrealized) * self.long_term_rate
        
        return {
            'unrealized_gains': {
                'short_term': float(short_term_unrealized),
                'long_term': float(long_term_unrealized),
                'total': float(short_term_unrealized + long_term_unrealized)
            },
            'estimated_tax': estimated_tax,
            'effective_rate': estimated_tax / float(short_term_unrealized + long_term_unrealized) 
                            if (short_term_unrealized + long_term_unrealized) > 0 else 0,
            'harvest_opportunities': sorted(
                harvest_opportunities, 
                key=lambda x: x['tax_benefit'], 
                reverse=True
            )[:5],
            'wait_opportunities': sorted(
                wait_opportunities,
                key=lambda x: x['potential_tax_savings'],
                reverse=True
            )[:5],
            'ytd_realized_gains': float(self.current_year_gains),
            'ytd_realized_losses': float(self.current_year_losses),
            'quarterly_estimate': float(self.current_quarter_tax)
        }
    
    def update_realized_gains(self, pnl: float, is_long_term: bool):
        """Update realized gains for the year"""
        if pnl > 0:
            self.current_year_gains += Decimal(str(pnl))
        else:
            self.current_year_losses += Decimal(str(pnl))
        
        # Update quarterly estimate
        net_gains = self.current_year_gains + self.current_year_losses
        if net_gains > 0:
            # Estimate based on YTD
            days_passed = datetime.now().timetuple().tm_yday
            projected_annual = float(net_gains) * (365 / days_passed)
            
            # Calculate tax
            # Simplified - assumes all short-term for conservative estimate
            annual_tax = projected_annual * self.short_term_rate
            self.current_quarter_tax = Decimal(str(annual_tax / 4))
    
    def should_close_position(self, position, current_price: float, market_signal: str) -> Dict:
        """
        Decision helper combining market signals and tax considerations
        """
        tax_analysis = self.analyze_position_tax_impact(position, current_price)
        
        # Strong market signal overrides tax considerations
        if market_signal in ['STOP_LOSS', 'EMERGENCY_EXIT']:
            return {
                'decision': 'CLOSE',
                'reason': market_signal,
                'tax_impact': tax_analysis['tax_liability']
            }
        
        # Weak market signal - consider taxes
        if market_signal == 'TAKE_PROFIT':
            if tax_analysis['days_to_long_term'] <= 7:
                return {
                    'decision': 'WAIT',
                    'reason': f"Wait {tax_analysis['days_to_long_term']} days for long-term gains",
                    'tax_savings': tax_analysis['potential_tax_savings']
                }
            else:
                return {
                    'decision': 'CLOSE',
                    'reason': 'Take profit - tax impact acceptable',
                    'tax_impact': tax_analysis['tax_liability']
                }
        
        # Loss harvesting opportunity
        if tax_analysis['potential_pnl'] < -100 and self.current_year_gains > 0:
            return {
                'decision': 'CLOSE',
                'reason': 'Harvest loss to offset gains',
                'tax_benefit': tax_analysis['tax_benefit']
            }
        
        return {
            'decision': 'HOLD',
            'reason': tax_analysis['recommendation'],
            'tax_analysis': tax_analysis
        }
    
    def get_tax_efficient_exit_plan(self, positions: Dict, target_cash: float) -> List[Dict]:
        """
        Generate tax-efficient plan to raise cash
        """
        exit_candidates = []
        
        for symbol, position in positions.items():
            analysis = self.analyze_position_tax_impact(position, position.current_price)
            
            # Score each position for tax-efficient exit
            score = 0
            
            # Prefer losses (tax benefit)
            if analysis['potential_pnl'] < 0:
                score += 100
            
            # Prefer long-term gains over short-term
            if analysis['is_long_term']:
                score += 50
            
            # Penalize positions close to long-term
            if 0 < analysis['days_to_long_term'] <= 30:
                score -= 75
            
            exit_candidates.append({
                'symbol': symbol,
                'position': position,
                'tax_analysis': analysis,
                'score': score,
                'value': position.quantity * position.current_price
            })
        
        # Sort by tax efficiency
        exit_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Build exit plan
        exit_plan = []
        total_value = 0
        
        for candidate in exit_candidates:
            if total_value >= target_cash:
                break
            
            exit_plan.append({
                'symbol': candidate['symbol'],
                'quantity': candidate['position'].quantity,
                'value': candidate['value'],
                'tax_impact': candidate['tax_analysis']['tax_liability'],
                'reason': candidate['tax_analysis']['recommendation']
            })
            
            total_value += candidate['value']
        
        return exit_plan