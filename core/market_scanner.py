"""
High-performance market scanner for cryptocurrency trading
"""
import asyncio
import aiohttp
import ccxt.async_support as ccxt
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketOpportunity:
    """Represents a trading opportunity found by scanner"""
    symbol: str
    strategy: str
    score: float  # 0-100
    signal: str  # BUY/SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    volume_24h: float
    volatility: float
    timestamp: datetime
    metadata: Dict[str, Any]


class MarketScanner:
    """
    High-performance market scanner that continuously monitors multiple symbols
    for trading opportunities across different strategies.
    """
    
    def __init__(self, exchange_name: str = 'binance', max_concurrent: int = 50):
        self.exchange_name = exchange_name
        self.max_concurrent = max_concurrent
        self.exchange = None
        self.running = False
        
        # Market state
        self.market_data: Dict[str, Dict] = {}
        self.opportunities: List[MarketOpportunity] = []
        self.symbol_whitelist: Set[str] = set()
        self.symbol_blacklist: Set[str] = set()
        
        # Performance metrics
        self.scan_times = []
        self.last_scan = None
        
    async def initialize(self, api_key: str, api_secret: str, testnet: bool = True):
        """Initialize the async exchange connection"""
        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'testnet': testnet
            }
        })
        await self.exchange.load_markets()
        logger.info(f"Scanner initialized with {len(self.exchange.markets)} markets")
        
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
    
    async def scan_markets(
        self, 
        symbols: Optional[List[str]] = None,
        min_volume_24h: float = 1000000,  # $1M minimum volume
        timeframes: List[str] = ['5m', '15m', '1h']
    ) -> List[MarketOpportunity]:
        """
        Scan multiple markets concurrently for opportunities
        """
        scan_start = datetime.now()
        
        # Get symbols to scan
        if symbols is None:
            symbols = await self._get_top_volume_symbols(min_volume_24h)
        
        # Filter symbols
        symbols = [s for s in symbols if s not in self.symbol_blacklist]
        if self.symbol_whitelist:
            symbols = [s for s in symbols if s in self.symbol_whitelist]
        
        logger.info(f"Scanning {len(symbols)} symbols...")
        
        # Create scanning tasks
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                task = self._scan_symbol_timeframe(symbol, timeframe, semaphore)
                tasks.append(task)
        
        # Execute all scans concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        opportunities = []
        for result in results:
            if isinstance(result, MarketOpportunity):
                opportunities.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scan error: {result}")
        
        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        # Update metrics
        scan_time = (datetime.now() - scan_start).total_seconds()
        self.scan_times.append(scan_time)
        self.last_scan = datetime.now()
        
        logger.info(f"Scan completed in {scan_time:.2f}s, found {len(opportunities)} opportunities")
        
        self.opportunities = opportunities
        return opportunities
    
    async def _get_top_volume_symbols(self, min_volume: float) -> List[str]:
        """Get symbols with highest 24h volume"""
        try:
            tickers = await self.exchange.fetch_tickers()
            
            # Filter and sort by volume
            volume_pairs = []
            for symbol, ticker in tickers.items():
                if ticker.get('quoteVolume', 0) > min_volume:
                    volume_pairs.append((symbol, ticker['quoteVolume']))
            
            volume_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 100 symbols
            return [pair[0] for pair in volume_pairs[:100]]
            
        except Exception as e:
            logger.error(f"Failed to fetch top volume symbols: {e}")
            return []
    
    async def _scan_symbol_timeframe(
        self, 
        symbol: str, 
        timeframe: str,
        semaphore: asyncio.Semaphore
    ) -> Optional[MarketOpportunity]:
        """Scan a single symbol/timeframe combination"""
        async with semaphore:
            try:
                # Fetch OHLCV data
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                if len(ohlcv) < 50:
                    return None
                
                # Convert to numpy arrays for faster computation
                closes = np.array([x[4] for x in ohlcv])
                highs = np.array([x[2] for x in ohlcv])
                lows = np.array([x[3] for x in ohlcv])
                volumes = np.array([x[5] for x in ohlcv])
                
                # Calculate indicators concurrently
                indicators = self._calculate_indicators(closes, highs, lows, volumes)
                
                # Check multiple strategies
                opportunities = []
                
                # Trend Following
                trend_opp = self._check_trend_following(symbol, closes, indicators)
                if trend_opp:
                    opportunities.append(trend_opp)
                
                # Mean Reversion
                mr_opp = self._check_mean_reversion(symbol, closes, indicators)
                if mr_opp:
                    opportunities.append(mr_opp)
                
                # Momentum
                mom_opp = self._check_momentum(symbol, closes, indicators)
                if mom_opp:
                    opportunities.append(mom_opp)
                
                # Volume Breakout
                vol_opp = self._check_volume_breakout(symbol, closes, volumes, indicators)
                if vol_opp:
                    opportunities.append(vol_opp)
                
                # Return best opportunity
                if opportunities:
                    return max(opportunities, key=lambda x: x.score)
                
                return None
                
            except Exception as e:
                logger.debug(f"Error scanning {symbol}/{timeframe}: {e}")
                return None
    
    def _calculate_indicators(
        self, 
        closes: np.ndarray, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        volumes: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate technical indicators efficiently"""
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = np.mean(closes[-20:])
        indicators['sma_50'] = np.mean(closes[-50:])
        indicators['ema_12'] = self._ema(closes, 12)
        indicators['ema_26'] = self._ema(closes, 26)
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = self._ema(np.array([indicators['macd']]), 9)
        
        # RSI
        indicators['rsi'] = self._rsi(closes, 14)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = np.std(closes[-bb_period:])
        indicators['bb_upper'] = indicators['sma_20'] + (2 * bb_std)
        indicators['bb_lower'] = indicators['sma_20'] - (2 * bb_std)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['sma_20']
        
        # ATR (Average True Range)
        indicators['atr'] = self._atr(highs, lows, closes, 14)
        
        # Volume indicators
        indicators['volume_sma'] = np.mean(volumes[-20:])
        indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma']
        
        # Volatility
        returns = np.diff(closes) / closes[:-1]
        indicators['volatility'] = np.std(returns[-20:]) * np.sqrt(365*24*12)  # Annualized for 5min
        
        return indicators
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.nan
        multiplier = 2 / (period + 1)
        ema = data[-period:].mean()
        for price in data[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        deltas = np.diff(closes)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        tr1 = highs - lows
        tr2 = abs(highs - np.roll(closes, 1))
        tr3 = abs(lows - np.roll(closes, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(tr[-period:])
    
    def _check_trend_following(
        self, 
        symbol: str, 
        closes: np.ndarray, 
        indicators: Dict
    ) -> Optional[MarketOpportunity]:
        """Check for trend following opportunities"""
        current_price = closes[-1]
        
        # Bullish trend conditions
        if (indicators['ema_12'] > indicators['ema_26'] and
            indicators['macd'] > indicators['macd_signal'] and
            current_price > indicators['sma_50'] and
            indicators['rsi'] > 40 and indicators['rsi'] < 70):
            
            # Calculate entry and exits
            atr = indicators['atr']
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
            
            # Score based on trend strength
            trend_strength = abs(indicators['macd']) / current_price * 100
            score = min(90, 50 + trend_strength * 10)
            
            return MarketOpportunity(
                symbol=symbol,
                strategy='trend_following',
                score=score,
                signal='BUY',
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume_24h=indicators['volume_sma'] * current_price * 288,  # 5min bars in 24h
                volatility=indicators['volatility'],
                timestamp=datetime.now(),
                metadata={
                    'rsi': indicators['rsi'],
                    'macd': indicators['macd'],
                    'trend_strength': trend_strength
                }
            )
        
        return None
    
    def _check_mean_reversion(
        self, 
        symbol: str, 
        closes: np.ndarray, 
        indicators: Dict
    ) -> Optional[MarketOpportunity]:
        """Check for mean reversion opportunities"""
        current_price = closes[-1]
        
        # Oversold conditions
        if (current_price < indicators['bb_lower'] and
            indicators['rsi'] < 30 and
            indicators['volume_ratio'] > 1.5):
            
            # Calculate entry and exits
            stop_loss = current_price * 0.98
            take_profit = indicators['sma_20']  # Target mean
            
            # Score based on deviation from mean
            deviation = (indicators['sma_20'] - current_price) / current_price
            score = min(85, 40 + deviation * 100)
            
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
    
    def _check_momentum(
        self, 
        symbol: str, 
        closes: np.ndarray, 
        indicators: Dict
    ) -> Optional[MarketOpportunity]:
        """Check for momentum opportunities"""
        current_price = closes[-1]
        
        # Strong momentum conditions
        momentum = (current_price - closes[-10]) / closes[-10]
        
        if (momentum > 0.02 and  # 2% move in last 10 bars
            indicators['rsi'] > 60 and indicators['rsi'] < 80 and
            indicators['volume_ratio'] > 2.0):
            
            # Calculate entry and exits
            atr = indicators['atr']
            stop_loss = current_price - (1.5 * atr)
            take_profit = current_price + (2.5 * atr)
            
            # Score based on momentum strength
            score = min(80, 40 + momentum * 500)
            
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
        volumes: np.ndarray,
        indicators: Dict
    ) -> Optional[MarketOpportunity]:
        """Check for volume breakout opportunities"""
        current_price = closes[-1]
        
        # Volume spike with price breakout
        price_change = (current_price - closes[-5]) / closes[-5]
        
        if (indicators['volume_ratio'] > 3.0 and
            price_change > 0.01 and
            current_price > indicators['sma_20']):
            
            # Calculate entry and exits
            atr = indicators['atr']
            stop_loss = current_price - (1.5 * atr)
            take_profit = current_price + (3 * atr)
            
            # Score based on volume intensity
            score = min(75, 30 + indicators['volume_ratio'] * 10)
            
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
    
    async def continuous_scan(self, scan_interval: int = 30):
        """Run continuous market scanning"""
        self.running = True
        
        while self.running:
            try:
                # Scan markets
                opportunities = await self.scan_markets()
                
                # Log top opportunities
                if opportunities:
                    logger.info(f"Top opportunities:")
                    for opp in opportunities[:5]:
                        logger.info(
                            f"  {opp.symbol} - {opp.strategy} "
                            f"(score: {opp.score:.1f}, signal: {opp.signal})"
                        )
                
                # Wait before next scan
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def stop(self):
        """Stop continuous scanning"""
        self.running = False
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scanner performance statistics"""
        if not self.scan_times:
            return {}
        
        return {
            'avg_scan_time': np.mean(self.scan_times),
            'min_scan_time': np.min(self.scan_times),
            'max_scan_time': np.max(self.scan_times),
            'total_scans': len(self.scan_times),
            'last_scan': self.last_scan,
            'opportunities_found': len(self.opportunities)
        }