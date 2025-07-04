"""
Testnet-optimized scanner with adjusted thresholds
"""
from core.market_scanner import MarketScanner, MarketOpportunity
from typing import Optional, Dict
import numpy as np
from datetime import datetime

class TestnetScanner(MarketScanner):
    """Scanner with adjusted thresholds for testnet conditions"""
    
    def _check_momentum(
        self, 
        symbol: str, 
        closes: np.ndarray, 
        indicators: Dict
    ) -> Optional[MarketOpportunity]:
        """Check for momentum opportunities (testnet adjusted)"""
        current_price = closes[-1]
        
        # Testnet: Lower threshold from 2% to 0.2%
        momentum = (current_price - closes[-10]) / closes[-10]
        
        if (momentum > 0.002 and  # 0.2% move (was 2%)
            indicators['rsi'] > 55 and indicators['rsi'] < 80 and
            indicators['volume_ratio'] > 1.2):  # 1.2x (was 2.0x)
            
            # Calculate entry and exits
            atr = indicators['atr']
            stop_loss = current_price - (1.5 * atr)
            take_profit = current_price + (2.5 * atr)
            
            # Score based on momentum strength
            score = min(80, 40 + momentum * 5000)  # 10x multiplier
            
            return MarketOpportunity(
                symbol=symbol,
                strategy='momentum',
                score=score,
                signal='BUY',
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume_24h=indicators['volume_sma'] * current_price * 288,
                volatility=indicators['volatility'],
                timestamp=datetime.now(),
                metadata={
                    'momentum': momentum,
                    'volume_spike': indicators['volume_ratio']
                }
            )
        
        return None
    
    def _check_volume_breakout(
        self, 
        symbol: str, 
        closes: np.ndarray, 
        indicators: Dict
    ) -> Optional[MarketOpportunity]:
        """Check for volume breakout opportunities (testnet adjusted)"""
        current_price = closes[-1]
        
        # Volume spike with price breakout
        price_change = (current_price - closes[-5]) / closes[-5]
        
        if (indicators['volume_ratio'] > 1.5 and  # 1.5x (was 3.0x)
            price_change > 0.001 and  # 0.1% (was 1%)
            current_price > indicators['sma_20']):
            
            # Calculate entry and exits
            atr = indicators['atr']
            stop_loss = current_price - (1.5 * atr)
            take_profit = current_price + (3 * atr)
            
            # Score based on volume intensity
            score = min(75, 30 + indicators['volume_ratio'] * 20)  # 2x multiplier
            
            return MarketOpportunity(
                symbol=symbol,
                strategy='volume_breakout',
                score=score,
                signal='BUY',
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume_24h=indicators['volume_sma'] * current_price * 288,
                volatility=indicators['volatility'],
                timestamp=datetime.now(),
                metadata={
                    'volume_ratio': indicators['volume_ratio'],
                    'price_change': price_change
                }
            )
        
        return None
    
    def _check_mean_reversion(
        self, 
        symbol: str, 
        closes: np.ndarray, 
        indicators: Dict
    ) -> Optional[MarketOpportunity]:
        """Check for mean reversion opportunities (testnet adjusted)"""
        current_price = closes[-1]
        
        # Testnet: More lenient oversold conditions
        if (current_price < indicators['bb_lower'] and
            indicators['rsi'] < 40 and  # Was 30
            indicators['volume_ratio'] > 1.2):  # Was 1.5
            
            # Calculate entry and exits
            stop_loss = current_price * 0.98
            take_profit = indicators['sma_20']  # Target mean
            
            # Score calculation
            deviation = (indicators['sma_20'] - current_price) / current_price
            score = min(85, 40 + deviation * 200)  # 2x multiplier
            
            return MarketOpportunity(
                symbol=symbol,
                strategy='mean_reversion',
                score=score,
                signal='BUY',
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume_24h=indicators['volume_sma'] * current_price * 288,
                volatility=indicators['volatility'],
                timestamp=datetime.now(),
                metadata={
                    'rsi': indicators['rsi'],
                    'bb_position': 'below_lower',
                    'deviation': deviation
                }
            )
        
        return None