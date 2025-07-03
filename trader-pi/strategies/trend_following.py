#!/usr/bin/env python3
"""
Simple trend following strategy using moving average crossover
"""
from typing import Dict, List, Optional
import numpy as np

from strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using EMA crossover
    - Buy when fast EMA crosses above slow EMA
    - Sell when fast EMA crosses below slow EMA
    - Confirm with RSI and volume
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.fast_period = config.get('fast_ema', 12)
        self.slow_period = config.get('slow_ema', 26)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
    def analyze(self, symbol: str, ohlcv_data: Dict[str, List]) -> Optional[Dict]:
        """Analyze market data for trend following signals"""
        try:
            # Use primary timeframe
            primary_tf = self.config['timeframes'][0]
            if primary_tf not in ohlcv_data:
                return None
            
            ohlcv = ohlcv_data[primary_tf]
            if len(ohlcv) < self.slow_period + 10:  # Need enough data
                return None
            
            # Calculate indicators
            indicators = self.calculate_indicators(ohlcv)
            
            # Extract latest values
            current_price = ohlcv[-1][4]  # Close price
            fast_ema = indicators['ema_12'][-1]
            slow_ema = indicators['ema_26'][-1]
            prev_fast_ema = indicators['ema_12'][-2]
            prev_slow_ema = indicators['ema_26'][-2]
            rsi = indicators['rsi'][-1]
            
            # Check volume
            volumes = np.array([candle[5] for candle in ohlcv])
            if not self.check_volume(volumes):
                return None
            
            # Check for bullish crossover
            if (prev_fast_ema <= prev_slow_ema and 
                fast_ema > slow_ema and 
                rsi < self.rsi_overbought):
                
                # Calculate confidence based on crossover strength and RSI
                crossover_strength = abs(fast_ema - slow_ema) / slow_ema
                rsi_room = (self.rsi_overbought - rsi) / self.rsi_overbought
                confidence = min(crossover_strength * 100 + rsi_room * 50, 100)
                
                signal = self.generate_signal('BUY', current_price)
                signal['confidence'] = confidence
                signal['reason'] = f"Bullish EMA crossover, RSI: {rsi:.1f}"
                
                self.logger.info(f"{symbol}: {signal['reason']}")
                return signal
            
            # Check for bearish crossover
            elif (prev_fast_ema >= prev_slow_ema and 
                  fast_ema < slow_ema and 
                  rsi > self.rsi_oversold):
                
                # Calculate confidence
                crossover_strength = abs(fast_ema - slow_ema) / slow_ema
                rsi_room = (rsi - self.rsi_oversold) / (100 - self.rsi_oversold)
                confidence = min(crossover_strength * 100 + rsi_room * 50, 100)
                
                signal = self.generate_signal('SELL', current_price)
                signal['confidence'] = confidence
                signal['reason'] = f"Bearish EMA crossover, RSI: {rsi:.1f}"
                
                self.logger.info(f"{symbol}: {signal['reason']}")
                return signal
            
            # Additional exit conditions for open positions
            # Strong trend reversal indicated by RSI extremes
            if rsi > self.rsi_overbought + 10:  # Very overbought
                signal = self.generate_signal('SELL', current_price)
                signal['confidence'] = 80
                signal['reason'] = f"RSI extremely overbought: {rsi:.1f}"
                return signal
            
            elif rsi < self.rsi_oversold - 10:  # Very oversold
                signal = self.generate_signal('BUY', current_price)
                signal['confidence'] = 80
                signal['reason'] = f"RSI extremely oversold: {rsi:.1f}"
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def validate_trend(self, ohlcv: List, timeframe: str = '1h') -> bool:
        """Validate trend direction using higher timeframe"""
        try:
            if len(ohlcv) < 50:
                return False
            
            # Calculate trend on higher timeframe
            closes = np.array([candle[4] for candle in ohlcv])
            sma_50 = self._sma(closes, 50)
            
            # Check if price is above SMA for uptrend
            current_price = closes[-1]
            current_sma = sma_50[-1]
            
            return not np.isnan(current_sma) and current_price > current_sma
            
        except Exception:
            return False