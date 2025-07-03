#!/usr/bin/env python3
"""
Mean reversion strategy using Bollinger Bands
"""
from typing import Dict, List, Optional
import numpy as np

from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands
    - Buy when price touches lower band and RSI oversold
    - Sell when price touches upper band and RSI overbought
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.min_bb_width_pct = config.get('min_bb_width_pct', 0.01)
        
    def analyze(self, symbol: str, ohlcv_data: Dict[str, List]) -> Optional[Dict]:
        """Analyze market data for mean reversion signals"""
        try:
            # Use primary timeframe
            primary_tf = self.config['timeframes'][0]
            if primary_tf not in ohlcv_data:
                return None
            
            ohlcv = ohlcv_data[primary_tf]
            if len(ohlcv) < self.bb_period + 10:
                return None
            
            # Calculate indicators
            indicators = self.calculate_indicators(ohlcv)
            
            # Extract values
            current_price = ohlcv[-1][4]
            bb_upper = indicators['bb_upper'][-1]
            bb_middle = indicators['bb_middle'][-1]
            bb_lower = indicators['bb_lower'][-1]
            rsi = indicators['rsi'][-1]
            
            # Skip if any indicator is NaN
            if np.isnan([bb_upper, bb_middle, bb_lower, rsi]).any():
                return None
            
            # Calculate Bollinger Band width
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < self.min_bb_width_pct:
                return None  # Bands too narrow, low volatility
            
            # Check volume
            volumes = np.array([candle[5] for candle in ohlcv])
            if not self.check_volume(volumes):
                return None
            
            # Price history for pattern detection
            closes = np.array([candle[4] for candle in ohlcv])
            
            # Buy signal: Price at lower band + RSI oversold
            if (current_price <= bb_lower * 1.001 and  # Small buffer
                rsi < self.rsi_oversold and
                self._check_bounce_pattern(closes, bb_lower)):
                
                # Calculate confidence
                band_penetration = (bb_lower - current_price) / bb_lower
                rsi_oversold_depth = (self.rsi_oversold - rsi) / self.rsi_oversold
                confidence = min((band_penetration + rsi_oversold_depth) * 50, 100)
                
                signal = self.generate_signal('BUY', current_price)
                signal['confidence'] = confidence
                signal['reason'] = f"Price at lower BB, RSI: {rsi:.1f}"
                signal['take_profit'] = bb_middle  # Target middle band
                
                self.logger.info(f"{symbol}: {signal['reason']}")
                return signal
            
            # Sell signal: Price at upper band + RSI overbought
            elif (current_price >= bb_upper * 0.999 and  # Small buffer
                  rsi > self.rsi_overbought and
                  self._check_reversal_pattern(closes, bb_upper)):
                
                # Calculate confidence
                band_penetration = (current_price - bb_upper) / bb_upper
                rsi_overbought_depth = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                confidence = min((band_penetration + rsi_overbought_depth) * 50, 100)
                
                signal = self.generate_signal('SELL', current_price)
                signal['confidence'] = confidence
                signal['reason'] = f"Price at upper BB, RSI: {rsi:.1f}"
                signal['take_profit'] = bb_middle  # Target middle band
                
                self.logger.info(f"{symbol}: {signal['reason']}")
                return signal
            
            # Exit at middle band for mean reversion
            elif self._should_exit_at_middle(current_price, bb_middle, closes):
                signal = self.generate_signal('SELL', current_price)
                signal['confidence'] = 70
                signal['reason'] = "Price returned to middle band"
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _check_bounce_pattern(self, closes: np.ndarray, lower_band: float) -> bool:
        """Check if price shows bounce pattern at lower band"""
        if len(closes) < 3:
            return False
        
        # Look for price touching band and starting to reverse
        recent_closes = closes[-3:]
        touched_band = any(price <= lower_band * 1.001 for price in recent_closes[:-1])
        reversing = recent_closes[-1] > recent_closes[-2]
        
        return touched_band and reversing
    
    def _check_reversal_pattern(self, closes: np.ndarray, upper_band: float) -> bool:
        """Check if price shows reversal pattern at upper band"""
        if len(closes) < 3:
            return False
        
        # Look for price touching band and starting to reverse
        recent_closes = closes[-3:]
        touched_band = any(price >= upper_band * 0.999 for price in recent_closes[:-1])
        reversing = recent_closes[-1] < recent_closes[-2]
        
        return touched_band and reversing
    
    def _should_exit_at_middle(self, current_price: float, middle_band: float, 
                              closes: np.ndarray) -> bool:
        """Check if position should be exited at middle band"""
        if len(closes) < 5:
            return False
        
        # Check if price crossed middle band
        price_at_middle = abs(current_price - middle_band) / middle_band < 0.002
        
        # Check if we came from extremes
        recent_min = np.min(closes[-10:])
        recent_max = np.max(closes[-10:])
        from_extreme = (recent_max - recent_min) / middle_band > 0.02
        
        return price_at_middle and from_extreme