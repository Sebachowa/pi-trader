"""
Aggressive Testnet Scanner - Finds opportunities with minimal movements
"""
from core.market_scanner import MarketScanner, MarketOpportunity
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime
import logging


class AggressiveTestnetScanner(MarketScanner):
    """Ultra-sensitive scanner for testnet's low volatility"""
    
    def __init__(self, exchange_name: str, max_concurrent: int = 50):
        super().__init__(exchange_name, max_concurrent)
        self.logger = logging.getLogger(__name__)
        
    async def scan_markets(self, min_volume_24h: float = 100) -> List[MarketOpportunity]:
        """Scan with very low thresholds for testnet"""
        # Override parent min volume
        opportunities = []
        
        try:
            # Get all active symbols
            markets = await self.exchange.load_markets()
            usdt_pairs = [
                symbol for symbol in markets 
                if symbol.endswith('/USDT') and 
                markets[symbol]['active'] and
                symbol not in ['BUSD/USDT', 'USDC/USDT', 'DAI/USDT', 'USDT/USDT']
            ]
            
            self.logger.info(f"🔍 Scanning {len(usdt_pairs)} USDT pairs on testnet")
            
            # Scan each symbol
            for symbol in usdt_pairs:
                try:
                    opp = await self._scan_symbol_aggressive(symbol)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    self.logger.debug(f"Error scanning {symbol}: {e}")
                    continue
            
            # Sort by score
            opportunities.sort(key=lambda x: x.score, reverse=True)
            
            # Log results
            if opportunities:
                self.logger.info(f"✨ Found {len(opportunities)} opportunities!")
                for opp in opportunities[:5]:
                    self.logger.info(
                        f"  - {opp.symbol}: {opp.strategy} "
                        f"(score: {opp.score:.1f}, expected: {opp.expected_return:.2f}%)"
                    )
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Scanner error: {e}")
            return []
    
    async def _scan_symbol_aggressive(self, symbol: str) -> Optional[MarketOpportunity]:
        """Aggressively scan a single symbol"""
        try:
            # Fetch minimal data (5m candles, last 50)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '5m', limit=50)
            if len(ohlcv) < 20:
                return None
            
            # Extract data
            closes = np.array([x[4] for x in ohlcv])
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])
            
            current_price = closes[-1]
            
            # Ultra-simple indicators
            sma_10 = np.mean(closes[-10:])
            sma_20 = np.mean(closes[-20:])
            
            # Price position (0-100)
            recent_high = np.max(highs[-20:])
            recent_low = np.min(lows[-20:])
            price_range = recent_high - recent_low
            price_position = ((current_price - recent_low) / price_range * 100) if price_range > 0 else 50
            
            # Volume check (any volume is good for testnet)
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Micro-movements detection
            micro_momentum_5 = (current_price - closes[-5]) / closes[-5] * 100
            micro_momentum_10 = (current_price - closes[-10]) / closes[-10] * 100
            
            # Generate opportunities based on multiple strategies
            opportunity = None
            
            # 1. Micro Breakout (0.1% move)
            if micro_momentum_5 > 0.1 and current_price > sma_10 and volume_ratio > 1.1:
                opportunity = self._create_opportunity(
                    symbol, 'micro_breakout', 
                    score=30 + min(micro_momentum_5 * 50, 30),
                    current_price=current_price,
                    expected_return=0.5
                )
            
            # 2. Oversold Bounce (price near bottom)
            elif price_position < 30 and micro_momentum_5 > -0.1:
                opportunity = self._create_opportunity(
                    symbol, 'oversold_bounce',
                    score=35 + (30 - price_position),
                    current_price=current_price,
                    expected_return=0.7
                )
            
            # 3. Moving Average Cross
            elif current_price > sma_10 > sma_20 and micro_momentum_10 > 0:
                opportunity = self._create_opportunity(
                    symbol, 'ma_cross',
                    score=40 + min(micro_momentum_10 * 20, 20),
                    current_price=current_price,
                    expected_return=0.8
                )
            
            # 4. Volume Spike
            elif volume_ratio > 1.5 and micro_momentum_5 > 0:
                opportunity = self._create_opportunity(
                    symbol, 'volume_spike',
                    score=25 + min(volume_ratio * 10, 25),
                    current_price=current_price,
                    expected_return=0.6
                )
            
            # 5. Range Breakout (even tiny ranges)
            elif price_position > 80 and micro_momentum_5 > 0.05:
                opportunity = self._create_opportunity(
                    symbol, 'range_breakout',
                    score=30 + price_position - 80,
                    current_price=current_price,
                    expected_return=0.4
                )
            
            return opportunity
            
        except Exception as e:
            self.logger.debug(f"Error scanning {symbol}: {e}")
            return None
    
    def _create_opportunity(
        self, 
        symbol: str, 
        strategy: str, 
        score: float,
        current_price: float,
        expected_return: float
    ) -> MarketOpportunity:
        """Create opportunity with tight stops for testnet"""
        # Very tight stops for testnet (0.3-0.5%)
        stop_loss = current_price * 0.995  # 0.5% stop
        take_profit = current_price * (1 + expected_return / 100)  # expected return
        
        return MarketOpportunity(
            symbol=symbol,
            strategy=f'testnet_{strategy}',
            score=min(score, 80),  # Cap at 80
            signal='BUY',
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume_24h=0,  # Not important for testnet
            volatility=expected_return,
            timestamp=datetime.now(),
            expected_return=expected_return,
            metadata={
                'testnet_mode': True,
                'strategy_details': strategy
            }
        )