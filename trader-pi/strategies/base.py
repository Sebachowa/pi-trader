#!/usr/bin/env python3
"""
Base strategy class for trading strategies
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def analyze(self, symbol: str, ohlcv_data: Dict[str, List]) -> Optional[Dict]:
        """
        Analyze market data and generate trading signal
        
        Args:
            symbol: Trading pair symbol
            ohlcv_data: Dict of timeframe -> OHLCV data
            
        Returns:
            Signal dict with action, price, stop_loss, take_profit
            or None if no signal
        """
        pass
    
    def calculate_indicators(self, ohlcv: List) -> Dict:
        """Calculate common technical indicators"""
        # Convert to numpy arrays for easier calculation
        closes = np.array([candle[4] for candle in ohlcv])
        highs = np.array([candle[2] for candle in ohlcv])
        lows = np.array([candle[3] for candle in ohlcv])
        volumes = np.array([candle[5] for candle in ohlcv])
        
        indicators = {}
        
        # Simple Moving Averages
        indicators['sma_20'] = self._sma(closes, 20)
        indicators['sma_50'] = self._sma(closes, 50)
        
        # Exponential Moving Averages
        indicators['ema_12'] = self._ema(closes, 12)
        indicators['ema_26'] = self._ema(closes, 26)
        
        # RSI
        indicators['rsi'] = self._rsi(closes, 14)
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = \
            self._bollinger_bands(closes, 20, 2)
        
        # Volume metrics
        indicators['volume_sma'] = self._sma(volumes, 20)
        
        return indicators
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        if len(data) < period:
            return np.array([])
        
        sma = np.convolve(data, np.ones(period)/period, mode='valid')
        return np.concatenate([np.full(period-1, np.nan), sma])
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        if len(data) < period:
            return np.array([])
        
        ema = np.zeros_like(data)
        ema[:period] = np.nan
        ema[period-1] = np.mean(data[:period])
        
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return np.array([])
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = self._sma(gains, period)[period-1:]
        avg_losses = self._sma(losses, period)[period-1:]
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([np.full(period, np.nan), rsi])
    
    def _bollinger_bands(self, data: np.ndarray, period: int = 20, 
                        num_std: float = 2) -> tuple:
        """Bollinger Bands"""
        middle = self._sma(data, period)
        
        # Calculate rolling standard deviation
        std = np.zeros_like(data)
        std[:period] = np.nan
        
        for i in range(period, len(data)):
            std[i] = np.std(data[i-period+1:i+1])
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        return upper, middle, lower
    
    def check_volume(self, volumes: np.ndarray, min_volume: float = None) -> bool:
        """Check if volume is sufficient for trading"""
        if min_volume is None:
            min_volume = self.config.get('min_volume_24h', 1000000)
        
        # Check average volume over last 24 candles
        if len(volumes) >= 24:
            avg_volume = np.mean(volumes[-24:])
            return avg_volume >= min_volume
        
        return False
    
    def generate_signal(self, action: str, price: float, 
                       stop_loss_pct: float = 0.02, 
                       take_profit_pct: float = 0.05) -> Dict:
        """Generate trading signal"""
        signal = {
            'action': action,
            'price': price,
            'timestamp': np.datetime64('now'),
            'confidence': 0.0  # To be set by strategy
        }
        
        if action == 'BUY':
            signal['stop_loss'] = price * (1 - stop_loss_pct)
            signal['take_profit'] = price * (1 + take_profit_pct)
        elif action == 'SELL':
            signal['stop_loss'] = price * (1 + stop_loss_pct)
            signal['take_profit'] = price * (1 - take_profit_pct)
        
        return signal