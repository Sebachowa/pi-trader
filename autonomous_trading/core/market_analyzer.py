
"""
Market Analyzer - Real-time market condition analysis for autonomous trading.
"""

import asyncio
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from nautilus_trader.common.component import Component
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
# from nautilus_trader.common.logging import Logger  # Not available in this version
from nautilus_trader.model.data import Bar, OrderBookDeltas, QuoteTick, TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


class LiquidityLevel(Enum):
    """Market liquidity classifications."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    VERY_LOW = "very_low"


class MarketConditions:
    """Current market conditions for an instrument."""
    
    def __init__(self, instrument_id: InstrumentId):
        self.instrument_id = instrument_id
        self.regime = MarketRegime.RANGING
        self.liquidity = LiquidityLevel.NORMAL
        self.volatility = 0.0
        self.trend_strength = 0.0
        self.momentum = 0.0
        self.spread_ratio = 0.0
        self.volume_ratio = 1.0
        self.anomaly_score = 0.0
        self.last_update = datetime.utcnow()


class MarketAnalyzer(Component):
    """
    Real-time market analysis system for autonomous trading decisions.
    
    Features:
    - Market regime detection
    - Volatility analysis
    - Liquidity monitoring
    - Anomaly detection
    - Multi-timeframe analysis
    """
    
    def __init__(
        self,
        logger: Any,  # Logger type
        clock: LiveClock,
        msgbus: MessageBus,
        lookback_periods: int = 100,
        update_interval_seconds: int = 60,
        anomaly_threshold: float = 3.0,
    ):
        # Initialize component with minimal parameters
        try:
            super().__init__()
        except Exception:
            # If that fails, try with specific parameters
            pass
        
        self.clock = clock
        self.logger = logger
        self.msgbus = msgbus
        self._component_id = "MARKET-ANALYZER"
        
        self.lookback_periods = lookback_periods
        self.update_interval_seconds = update_interval_seconds
        self.anomaly_threshold = anomaly_threshold
        
        # Market data storage
        self._bars: Dict[InstrumentId, Dict[str, deque]] = {}
        self._quotes: Dict[InstrumentId, deque] = {}
        self._trades: Dict[InstrumentId, deque] = {}
        self._orderbook_snapshots: Dict[InstrumentId, deque] = {}
        
        # Analysis results
        self._market_conditions: Dict[InstrumentId, MarketConditions] = {}
        self._regime_history: Dict[InstrumentId, deque] = {}
        self._volatility_history: Dict[InstrumentId, deque] = {}
        
        # Analysis tasks
        self._analysis_task = None
        self._instruments_to_analyze: Set[InstrumentId] = set()

    async def start(self) -> None:
        """Start the market analyzer."""
        if hasattr(self, 'logger') and self.logger:
            self.logger.info("Starting Market Analyzer...")
        else:
            print("INFO: Starting Market Analyzer...")
        self._analysis_task = asyncio.create_task(self._analysis_loop())

    async def stop(self) -> None:
        """Stop the market analyzer."""
        if hasattr(self, 'logger') and self.logger:
            self.logger.info("Stopping Market Analyzer...")
        else:
            print("INFO: Stopping Market Analyzer...")
        if self._analysis_task:
            self._analysis_task.cancel()

    def register_instrument(self, instrument_id: InstrumentId) -> None:
        """Register an instrument for analysis."""
        self._instruments_to_analyze.add(instrument_id)
        
        # Initialize storage
        if instrument_id not in self._bars:
            self._bars[instrument_id] = {
                "1m": deque(maxlen=self.lookback_periods * 60),
                "5m": deque(maxlen=self.lookback_periods * 12),
                "15m": deque(maxlen=self.lookback_periods * 4),
                "1h": deque(maxlen=self.lookback_periods),
            }
        
        if instrument_id not in self._quotes:
            self._quotes[instrument_id] = deque(maxlen=self.lookback_periods * 100)
        
        if instrument_id not in self._trades:
            self._trades[instrument_id] = deque(maxlen=self.lookback_periods * 100)
        
        if instrument_id not in self._market_conditions:
            self._market_conditions[instrument_id] = MarketConditions(instrument_id)

    def on_bar(self, bar: Bar) -> None:
        """Process bar data."""
        instrument_id = bar.bar_type.instrument_id
        timeframe = self._get_timeframe_key(bar.bar_type)
        
        if instrument_id in self._bars and timeframe in self._bars[instrument_id]:
            self._bars[instrument_id][timeframe].append(bar)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        """Process quote tick data."""
        if tick.instrument_id in self._quotes:
            self._quotes[tick.instrument_id].append(tick)

    def on_trade_tick(self, tick: TradeTick) -> None:
        """Process trade tick data."""
        if tick.instrument_id in self._trades:
            self._trades[tick.instrument_id].append(tick)

    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        """Process order book updates."""
        if deltas.instrument_id in self._orderbook_snapshots:
            # Store orderbook snapshot for liquidity analysis
            self._orderbook_snapshots[deltas.instrument_id].append(deltas)

    async def _analysis_loop(self) -> None:
        """Main analysis loop."""
        while True:
            try:
                # Analyze all registered instruments
                for instrument_id in self._instruments_to_analyze:
                    await self._analyze_instrument(instrument_id)
                
                await asyncio.sleep(self.update_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Market analysis error: {e}")

    async def _analyze_instrument(self, instrument_id: InstrumentId) -> None:
        """Perform comprehensive analysis for an instrument."""
        if instrument_id not in self._market_conditions:
            return
        
        conditions = self._market_conditions[instrument_id]
        
        # Detect market regime
        conditions.regime = await self._detect_market_regime(instrument_id)
        
        # Analyze volatility
        conditions.volatility = await self._calculate_volatility(instrument_id)
        
        # Analyze liquidity
        conditions.liquidity = await self._analyze_liquidity(instrument_id)
        
        # Calculate trend strength
        conditions.trend_strength = await self._calculate_trend_strength(instrument_id)
        
        # Calculate momentum
        conditions.momentum = await self._calculate_momentum(instrument_id)
        
        # Analyze spread
        conditions.spread_ratio = await self._calculate_spread_ratio(instrument_id)
        
        # Analyze volume
        conditions.volume_ratio = await self._calculate_volume_ratio(instrument_id)
        
        # Detect anomalies
        conditions.anomaly_score = await self._detect_anomalies(instrument_id)
        
        conditions.last_update = datetime.utcnow()
        
        # Store regime history
        if instrument_id not in self._regime_history:
            self._regime_history[instrument_id] = deque(maxlen=100)
        self._regime_history[instrument_id].append((conditions.regime, conditions.last_update))

    async def _detect_market_regime(self, instrument_id: InstrumentId) -> MarketRegime:
        """Detect current market regime using multiple indicators."""
        if instrument_id not in self._bars:
            return MarketRegime.RANGING
        
        # Get multiple timeframe bars
        bars_15m = list(self._bars[instrument_id].get("15m", []))
        bars_1h = list(self._bars[instrument_id].get("1h", []))
        
        if len(bars_15m) < 20 or len(bars_1h) < 10:
            return MarketRegime.RANGING
        
        # Calculate indicators
        prices_15m = np.array([float(bar.close) for bar in bars_15m[-20:]])
        prices_1h = np.array([float(bar.close) for bar in bars_1h[-10:]])
        
        # Moving averages
        ma_fast = np.mean(prices_15m[-8:])
        ma_slow = np.mean(prices_15m)
        
        # Price position relative to MAs
        current_price = prices_15m[-1]
        price_vs_ma = (current_price - ma_slow) / ma_slow
        
        # Trend direction
        trend_direction = ma_fast - ma_slow
        trend_strength = abs(trend_direction) / ma_slow
        
        # Volatility check
        volatility = np.std(prices_15m) / np.mean(prices_15m)
        
        # ADX calculation (simplified)
        high_low_ranges = np.array([float(bar.high - bar.low) for bar in bars_15m[-14:]])
        atr = np.mean(high_low_ranges)
        price_range = np.max(prices_15m) - np.min(prices_15m)
        adx_proxy = (price_range / atr) if atr > 0 else 0
        
        # Breakout detection
        recent_high = np.max(prices_1h[-5:])
        recent_low = np.min(prices_1h[-5:])
        prev_high = np.max(prices_1h[-10:-5])
        prev_low = np.min(prices_1h[-10:-5])
        
        # Determine regime
        if current_price > recent_high and recent_high > prev_high:
            return MarketRegime.BREAKOUT
        elif current_price < recent_low and recent_low < prev_low:
            return MarketRegime.BREAKDOWN
        elif trend_strength > 0.02 and adx_proxy > 2:
            if trend_direction > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        elif volatility > 0.03:
            return MarketRegime.VOLATILE
        elif volatility < 0.01:
            return MarketRegime.QUIET
        else:
            return MarketRegime.RANGING

    async def _calculate_volatility(self, instrument_id: InstrumentId) -> float:
        """Calculate current volatility using multiple methods."""
        if instrument_id not in self._bars:
            return 0.0
        
        bars_1h = list(self._bars[instrument_id].get("1h", []))
        if len(bars_1h) < 24:
            return 0.0
        
        # Calculate different volatility measures
        close_prices = np.array([float(bar.close) for bar in bars_1h[-24:]])
        returns = np.diff(np.log(close_prices))
        
        # Standard deviation of returns
        std_volatility = np.std(returns) * np.sqrt(24)  # Annualized
        
        # Parkinson volatility (using high-low)
        high_low_ratios = np.array([
            np.log(float(bar.high) / float(bar.low)) 
            for bar in bars_1h[-24:]
        ])
        parkinson_volatility = np.sqrt(np.mean(high_low_ratios**2) / (4 * np.log(2))) * np.sqrt(252*24)
        
        # Garman-Klass volatility
        gk_values = []
        for bar in bars_1h[-24:]:
            hl = np.log(float(bar.high) / float(bar.low))
            co = np.log(float(bar.close) / float(bar.open))
            gk_values.append(0.5 * hl**2 - (2*np.log(2) - 1) * co**2)
        
        gk_volatility = np.sqrt(np.mean(gk_values)) * np.sqrt(252*24)
        
        # Combine volatility measures
        combined_volatility = (std_volatility + parkinson_volatility + gk_volatility) / 3
        
        # Store in history
        if instrument_id not in self._volatility_history:
            self._volatility_history[instrument_id] = deque(maxlen=100)
        self._volatility_history[instrument_id].append(combined_volatility)
        
        return combined_volatility

    async def _analyze_liquidity(self, instrument_id: InstrumentId) -> LiquidityLevel:
        """Analyze market liquidity from quotes and trades."""
        if instrument_id not in self._quotes or instrument_id not in self._trades:
            return LiquidityLevel.NORMAL
        
        quotes = list(self._quotes[instrument_id])
        trades = list(self._trades[instrument_id])
        
        if len(quotes) < 100 or len(trades) < 50:
            return LiquidityLevel.NORMAL
        
        # Calculate spread metrics
        spreads = []
        for quote in quotes[-100:]:
            spread = float(quote.ask_price - quote.bid_price)
            mid_price = float(quote.ask_price + quote.bid_price) / 2
            spread_ratio = spread / mid_price if mid_price > 0 else 0
            spreads.append(spread_ratio)
        
        avg_spread = np.mean(spreads)
        
        # Calculate volume metrics
        trade_volumes = [float(trade.size) for trade in trades[-50:]]
        avg_volume = np.mean(trade_volumes)
        volume_std = np.std(trade_volumes)
        
        # Calculate quote depth (if orderbook data available)
        # This would use orderbook snapshots if available
        
        # Determine liquidity level
        if avg_spread < 0.0001 and avg_volume > 1000:
            return LiquidityLevel.VERY_HIGH
        elif avg_spread < 0.0005 and avg_volume > 500:
            return LiquidityLevel.HIGH
        elif avg_spread < 0.001 and avg_volume > 100:
            return LiquidityLevel.NORMAL
        elif avg_spread < 0.005 and avg_volume > 10:
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW

    async def _calculate_trend_strength(self, instrument_id: InstrumentId) -> float:
        """Calculate trend strength using ADX-like calculation."""
        if instrument_id not in self._bars:
            return 0.0
        
        bars_1h = list(self._bars[instrument_id].get("1h", []))
        if len(bars_1h) < 14:
            return 0.0
        
        # Calculate directional movement
        dm_plus = []
        dm_minus = []
        true_ranges = []
        
        for i in range(1, len(bars_1h[-14:])):
            high_diff = float(bars_1h[i].high - bars_1h[i-1].high)
            low_diff = float(bars_1h[i-1].low - bars_1h[i].low)
            
            dm_plus.append(max(high_diff, 0) if high_diff > low_diff else 0)
            dm_minus.append(max(low_diff, 0) if low_diff > high_diff else 0)
            
            true_range = max(
                float(bars_1h[i].high - bars_1h[i].low),
                abs(float(bars_1h[i].high - bars_1h[i-1].close)),
                abs(float(bars_1h[i].low - bars_1h[i-1].close))
            )
            true_ranges.append(true_range)
        
        # Calculate smoothed values
        if true_ranges and np.sum(true_ranges) > 0:
            di_plus = np.sum(dm_plus) / np.sum(true_ranges)
            di_minus = np.sum(dm_minus) / np.sum(true_ranges)
            
            if (di_plus + di_minus) > 0:
                dx = abs(di_plus - di_minus) / (di_plus + di_minus)
                return dx
        
        return 0.0

    async def _calculate_momentum(self, instrument_id: InstrumentId) -> float:
        """Calculate price momentum."""
        if instrument_id not in self._bars:
            return 0.0
        
        bars_1h = list(self._bars[instrument_id].get("1h", []))
        if len(bars_1h) < 10:
            return 0.0
        
        # Rate of change
        current_price = float(bars_1h[-1].close)
        past_price = float(bars_1h[-10].close)
        
        if past_price > 0:
            roc = (current_price - past_price) / past_price
            return roc
        
        return 0.0

    async def _calculate_spread_ratio(self, instrument_id: InstrumentId) -> float:
        """Calculate current spread as ratio of price."""
        if instrument_id not in self._quotes:
            return 0.0
        
        quotes = list(self._quotes[instrument_id])
        if not quotes:
            return 0.0
        
        latest_quote = quotes[-1]
        spread = float(latest_quote.ask_price - latest_quote.bid_price)
        mid_price = float(latest_quote.ask_price + latest_quote.bid_price) / 2
        
        if mid_price > 0:
            return spread / mid_price
        return 0.0

    async def _calculate_volume_ratio(self, instrument_id: InstrumentId) -> float:
        """Calculate current volume relative to average."""
        if instrument_id not in self._bars:
            return 1.0
        
        bars_1h = list(self._bars[instrument_id].get("1h", []))
        if len(bars_1h) < 24:
            return 1.0
        
        volumes = [float(bar.volume) for bar in bars_1h[-24:]]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])
        
        if avg_volume > 0:
            return current_volume / avg_volume
        return 1.0

    async def _detect_anomalies(self, instrument_id: InstrumentId) -> float:
        """Detect market anomalies using statistical methods."""
        anomaly_scores = []
        
        # Price anomaly
        if instrument_id in self._bars:
            bars_1h = list(self._bars[instrument_id].get("1h", []))
            if len(bars_1h) >= 24:
                prices = np.array([float(bar.close) for bar in bars_1h[-24:]])
                returns = np.diff(np.log(prices))
                
                if len(returns) > 0:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    
                    if std_return > 0:
                        latest_return = returns[-1]
                        z_score = abs((latest_return - mean_return) / std_return)
                        anomaly_scores.append(z_score)
        
        # Volume anomaly
        if instrument_id in self._bars:
            bars_1h = list(self._bars[instrument_id].get("1h", []))
            if len(bars_1h) >= 24:
                volumes = np.array([float(bar.volume) for bar in bars_1h[-24:]])
                mean_volume = np.mean(volumes[:-1])
                std_volume = np.std(volumes[:-1])
                
                if std_volume > 0:
                    latest_volume = volumes[-1]
                    z_score = abs((latest_volume - mean_volume) / std_volume)
                    anomaly_scores.append(z_score)
        
        # Spread anomaly
        if instrument_id in self._quotes:
            quotes = list(self._quotes[instrument_id])
            if len(quotes) >= 100:
                spreads = []
                for quote in quotes[-100:]:
                    spread = float(quote.ask_price - quote.bid_price)
                    mid_price = float(quote.ask_price + quote.bid_price) / 2
                    spread_ratio = spread / mid_price if mid_price > 0 else 0
                    spreads.append(spread_ratio)
                
                mean_spread = np.mean(spreads[:-1])
                std_spread = np.std(spreads[:-1])
                
                if std_spread > 0:
                    latest_spread = spreads[-1]
                    z_score = abs((latest_spread - mean_spread) / std_spread)
                    anomaly_scores.append(z_score)
        
        # Return maximum anomaly score
        if anomaly_scores:
            return max(anomaly_scores)
        return 0.0

    def get_market_conditions(self, instrument_id: InstrumentId) -> Optional[MarketConditions]:
        """Get current market conditions for an instrument."""
        return self._market_conditions.get(instrument_id)

    def get_recommended_strategies(self, instrument_id: InstrumentId) -> List[str]:
        """Get recommended strategies based on market conditions."""
        conditions = self.get_market_conditions(instrument_id)
        if not conditions:
            return []
        
        recommendations = []
        
        # Based on market regime
        if conditions.regime == MarketRegime.TRENDING_UP:
            recommendations.extend(["trend_following", "momentum"])
        elif conditions.regime == MarketRegime.TRENDING_DOWN:
            recommendations.extend(["trend_following", "short_momentum"])
        elif conditions.regime == MarketRegime.RANGING:
            recommendations.extend(["mean_reversion", "market_making"])
        elif conditions.regime == MarketRegime.VOLATILE:
            recommendations.extend(["volatility_arbitrage", "options_strategies"])
        elif conditions.regime == MarketRegime.BREAKOUT:
            recommendations.extend(["breakout_momentum", "trend_following"])
        
        # Based on liquidity
        if conditions.liquidity in [LiquidityLevel.HIGH, LiquidityLevel.VERY_HIGH]:
            recommendations.append("market_making")
        elif conditions.liquidity in [LiquidityLevel.LOW, LiquidityLevel.VERY_LOW]:
            # Remove market making in low liquidity
            recommendations = [s for s in recommendations if s != "market_making"]
        
        # Based on volatility
        if conditions.volatility > 0.3:  # High volatility
            recommendations.append("volatility_strategies")
        elif conditions.volatility < 0.1:  # Low volatility
            recommendations.append("carry_strategies")
        
        # Remove duplicates and return
        return list(set(recommendations))

    def _get_timeframe_key(self, bar_type) -> str:
        """Extract timeframe key from bar type."""
        # This would parse the bar_type to get timeframe
        # Placeholder implementation
        return "1h"

    def get_market_report(self) -> Dict[str, Any]:
        """Generate comprehensive market analysis report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "instruments_analyzed": len(self._instruments_to_analyze),
            "market_conditions": {},
            "regime_distribution": {},
            "liquidity_distribution": {},
            "high_volatility_instruments": [],
            "anomalies_detected": [],
        }
        
        # Collect regime distribution
        regime_counts = {regime: 0 for regime in MarketRegime}
        liquidity_counts = {level: 0 for level in LiquidityLevel}
        
        for instrument_id, conditions in self._market_conditions.items():
            # Add to report
            report["market_conditions"][str(instrument_id)] = {
                "regime": conditions.regime.value,
                "liquidity": conditions.liquidity.value,
                "volatility": round(conditions.volatility, 4),
                "trend_strength": round(conditions.trend_strength, 4),
                "momentum": round(conditions.momentum, 4),
                "anomaly_score": round(conditions.anomaly_score, 2),
                "recommended_strategies": self.get_recommended_strategies(instrument_id),
            }
            
            # Count distributions
            regime_counts[conditions.regime] += 1
            liquidity_counts[conditions.liquidity] += 1
            
            # Flag high volatility
            if conditions.volatility > 0.3:
                report["high_volatility_instruments"].append(str(instrument_id))
            
            # Flag anomalies
            if conditions.anomaly_score > self.anomaly_threshold:
                report["anomalies_detected"].append({
                    "instrument": str(instrument_id),
                    "score": round(conditions.anomaly_score, 2),
                    "type": "statistical_anomaly"
                })
        
        # Add distributions to report
        report["regime_distribution"] = {k.value: v for k, v in regime_counts.items()}
        report["liquidity_distribution"] = {k.value: v for k, v in liquidity_counts.items()}
        
        return report