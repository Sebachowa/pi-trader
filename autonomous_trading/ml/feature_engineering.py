
"""
Advanced Feature Engineering for ML Trading

Comprehensive feature extraction and engineering for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
from scipy import stats
from scipy.signal import find_peaks
import talib
from arch import arch_model

from autonomous_trading.ml.ml_optimizer import MarketFeatures


class FeatureEngineer:
    """
    Advanced feature engineering for trading ML models.
    
    Features:
    - Technical indicators at multiple timeframes
    - Market microstructure features
    - Statistical features
    - Pattern recognition features
    - Sentiment and alternative data integration
    """
    
    def __init__(
        self,
        lookback_periods: Dict[str, int] = None,
        enable_ta_features: bool = True,
        enable_microstructure: bool = True,
        enable_patterns: bool = True,
        enable_fourier: bool = True,
        enable_wavelets: bool = True,
    ):
        self.lookback_periods = lookback_periods or {
            "short": 20,
            "medium": 50,
            "long": 200,
        }
        self.enable_ta_features = enable_ta_features
        self.enable_microstructure = enable_microstructure
        self.enable_patterns = enable_patterns
        self.enable_fourier = enable_fourier
        self.enable_wavelets = enable_wavelets
        
        # Feature caches
        self._price_cache = deque(maxlen=max(lookback_periods.values()))
        self._volume_cache = deque(maxlen=max(lookback_periods.values()))
        self._spread_cache = deque(maxlen=100)
        self._trade_cache = deque(maxlen=1000)
    
    def extract_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        orderbook_data: Optional[pd.DataFrame] = None,
        trade_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[Dict[str, float]] = None,
    ) -> MarketFeatures:
        """Extract comprehensive features from market data."""
        
        # Price returns at multiple timeframes
        price_features = self._calculate_price_features(price_data)
        
        # Volatility features
        volatility_features = self._calculate_volatility_features(price_data)
        
        # Volume features
        volume_features = self._calculate_volume_features(volume_data) if volume_data is not None else {}
        
        # Technical indicators
        ta_features = self._calculate_technical_indicators(price_data) if self.enable_ta_features else {}
        
        # Market microstructure
        micro_features = self._calculate_microstructure_features(
            orderbook_data, trade_data
        ) if self.enable_microstructure and orderbook_data is not None else {}
        
        # Pattern features
        pattern_features = self._calculate_pattern_features(price_data) if self.enable_patterns else {}
        
        # Regime features
        regime_features = self._calculate_regime_features(price_data, volatility_features)
        
        # Correlation features
        correlation_features = self._calculate_correlation_features(price_data)
        
        # Sentiment features
        sentiment_features = sentiment_data or {}
        
        # Combine all features
        features = MarketFeatures(
            # Price features
            price_return_1m=price_features.get("return_1m", 0.0),
            price_return_5m=price_features.get("return_5m", 0.0),
            price_return_15m=price_features.get("return_15m", 0.0),
            price_return_1h=price_features.get("return_1h", 0.0),
            price_return_4h=price_features.get("return_4h", 0.0),
            price_return_1d=price_features.get("return_1d", 0.0),
            price_return_1w=price_features.get("return_1w", 0.0),
            
            # Volatility features
            volatility_1h=volatility_features.get("volatility_1h", 0.0),
            volatility_4h=volatility_features.get("volatility_4h", 0.0),
            volatility_1d=volatility_features.get("volatility_1d", 0.0),
            volatility_ratio_short_long=volatility_features.get("vol_ratio", 1.0),
            garch_volatility=volatility_features.get("garch_vol", 0.0),
            realized_volatility=volatility_features.get("realized_vol", 0.0),
            implied_volatility=volatility_features.get("implied_vol", 0.0),
            
            # Volume features
            volume_ratio_1h=volume_features.get("volume_ratio_1h", 1.0),
            volume_ratio_4h=volume_features.get("volume_ratio_4h", 1.0),
            volume_ratio_1d=volume_features.get("volume_ratio_1d", 1.0),
            volume_trend=volume_features.get("volume_trend", 0.0),
            volume_volatility=volume_features.get("volume_volatility", 0.0),
            buy_sell_ratio=volume_features.get("buy_sell_ratio", 1.0),
            
            # Technical indicators
            rsi_14=ta_features.get("rsi_14", 50.0),
            rsi_30=ta_features.get("rsi_30", 50.0),
            macd_signal=ta_features.get("macd_signal", 0.0),
            macd_histogram=ta_features.get("macd_histogram", 0.0),
            bb_position=ta_features.get("bb_position", 0.5),
            bb_width=ta_features.get("bb_width", 0.0),
            atr_14=ta_features.get("atr_14", 0.0),
            adx_14=ta_features.get("adx_14", 0.0),
            cci_20=ta_features.get("cci_20", 0.0),
            
            # Market microstructure
            bid_ask_spread=micro_features.get("spread", 0.0),
            spread_volatility=micro_features.get("spread_volatility", 0.0),
            order_book_imbalance=micro_features.get("book_imbalance", 0.0),
            trade_intensity=micro_features.get("trade_intensity", 0.0),
            quote_intensity=micro_features.get("quote_intensity", 0.0),
            
            # Trend features
            trend_strength=ta_features.get("trend_strength", 0.0),
            trend_consistency=ta_features.get("trend_consistency", 0.0),
            support_distance=pattern_features.get("support_distance", 0.0),
            resistance_distance=pattern_features.get("resistance_distance", 0.0),
            
            # Regime features
            regime_stability=regime_features.get("regime_stability", 0.0),
            regime_transition_prob=regime_features.get("transition_prob", 0.0),
            
            # Correlation features
            correlation_btc=correlation_features.get("btc_correlation", 0.0),
            correlation_market_index=correlation_features.get("market_correlation", 0.0),
            beta=correlation_features.get("beta", 1.0),
            
            # Sentiment features
            sentiment_score=sentiment_features.get("sentiment_score", 0.0),
            social_volume=sentiment_features.get("social_volume", 0.0),
            news_impact=sentiment_features.get("news_impact", 0.0),
        )
        
        return features
    
    def _calculate_price_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price-based features."""
        features = {}
        
        if len(price_data) < 2:
            return features
        
        # Calculate returns at different timeframes
        close_prices = price_data["close"].values
        current_price = close_prices[-1]
        
        # Define periods (assuming 1-minute base timeframe)
        periods = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
        }
        
        for period_name, period_minutes in periods.items():
            if len(close_prices) > period_minutes:
                past_price = close_prices[-period_minutes - 1]
                if past_price > 0:
                    features[f"return_{period_name}"] = (current_price - past_price) / past_price
                else:
                    features[f"return_{period_name}"] = 0.0
            else:
                features[f"return_{period_name}"] = 0.0
        
        # Log returns
        if len(close_prices) > 1:
            log_returns = np.diff(np.log(close_prices))
            features["mean_log_return"] = np.mean(log_returns)
            features["log_return_skew"] = stats.skew(log_returns)
            features["log_return_kurtosis"] = stats.kurtosis(log_returns)
        
        # Price momentum
        if len(close_prices) > 20:
            features["momentum_10"] = (close_prices[-1] - close_prices[-11]) / close_prices[-11]
            features["momentum_20"] = (close_prices[-1] - close_prices[-21]) / close_prices[-21]
        
        return features
    
    def _calculate_volatility_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility features using multiple methods."""
        features = {}
        
        if len(price_data) < 20:
            return features
        
        close_prices = price_data["close"].values
        high_prices = price_data["high"].values if "high" in price_data else close_prices
        low_prices = price_data["low"].values if "low" in price_data else close_prices
        
        # Standard deviation based volatility
        returns = np.diff(np.log(close_prices))
        
        features["volatility_1h"] = np.std(returns[-60:]) * np.sqrt(60) if len(returns) > 60 else 0.0
        features["volatility_4h"] = np.std(returns[-240:]) * np.sqrt(240) if len(returns) > 240 else 0.0
        features["volatility_1d"] = np.std(returns) * np.sqrt(1440) if len(returns) > 1440 else 0.0
        
        # Volatility ratio
        if features["volatility_1d"] > 0:
            features["vol_ratio"] = features["volatility_1h"] / features["volatility_1d"]
        else:
            features["vol_ratio"] = 1.0
        
        # Parkinson volatility (high-low estimator)
        if len(high_prices) > 20 and len(low_prices) > 20:
            hl_ratio = np.log(high_prices[-20:] / low_prices[-20:])
            features["parkinson_vol"] = np.sqrt(np.mean(hl_ratio**2) / (4 * np.log(2))) * np.sqrt(252)
        
        # Garman-Klass volatility
        if "open" in price_data and len(price_data) > 20:
            open_prices = price_data["open"].values[-20:]
            gk_values = 0.5 * np.log(high_prices[-20:] / low_prices[-20:])**2 - \
                       (2 * np.log(2) - 1) * np.log(close_prices[-20:] / open_prices)**2
            features["gk_volatility"] = np.sqrt(np.mean(gk_values)) * np.sqrt(252)
        
        # GARCH volatility
        try:
            if len(returns) > 100:
                model = arch_model(returns * 100, vol='Garch', p=1, q=1)
                model_fit = model.fit(disp='off')
                features["garch_vol"] = model_fit.conditional_volatility[-1] / 100
            else:
                features["garch_vol"] = features.get("volatility_1h", 0.0)
        except:
            features["garch_vol"] = features.get("volatility_1h", 0.0)
        
        # Realized volatility
        if len(returns) > 20:
            features["realized_vol"] = np.sqrt(np.sum(returns[-20:]**2)) * np.sqrt(252)
        
        # Implied volatility proxy (using ATR)
        if len(price_data) > 14:
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            if len(atr) > 0 and not np.isnan(atr[-1]):
                features["implied_vol"] = atr[-1] / close_prices[-1] * np.sqrt(252)
            else:
                features["implied_vol"] = features.get("volatility_1h", 0.0)
        
        return features
    
    def _calculate_volume_features(self, volume_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based features."""
        features = {}
        
        if volume_data is None or len(volume_data) < 20:
            return features
        
        volumes = volume_data["volume"].values
        
        # Volume ratios
        current_volume = volumes[-1]
        features["volume_ratio_1h"] = current_volume / np.mean(volumes[-60:]) if len(volumes) > 60 else 1.0
        features["volume_ratio_4h"] = current_volume / np.mean(volumes[-240:]) if len(volumes) > 240 else 1.0
        features["volume_ratio_1d"] = current_volume / np.mean(volumes) if len(volumes) > 1440 else 1.0
        
        # Volume trend
        if len(volumes) > 20:
            volume_ma_short = np.mean(volumes[-10:])
            volume_ma_long = np.mean(volumes[-20:])
            features["volume_trend"] = (volume_ma_short - volume_ma_long) / volume_ma_long if volume_ma_long > 0 else 0.0
        
        # Volume volatility
        if len(volumes) > 20:
            features["volume_volatility"] = np.std(volumes[-20:]) / np.mean(volumes[-20:])
        
        # Buy/Sell volume ratio (if available)
        if "buy_volume" in volume_data and "sell_volume" in volume_data:
            buy_vol = volume_data["buy_volume"].values[-20:]
            sell_vol = volume_data["sell_volume"].values[-20:]
            total_buy = np.sum(buy_vol)
            total_sell = np.sum(sell_vol)
            features["buy_sell_ratio"] = total_buy / total_sell if total_sell > 0 else 1.0
        else:
            features["buy_sell_ratio"] = 1.0
        
        # Volume-weighted average price (VWAP) deviation
        if "close" in volume_data:
            prices = volume_data["close"].values[-20:]
            vwap = np.sum(prices * volumes[-20:]) / np.sum(volumes[-20:])
            features["vwap_deviation"] = (prices[-1] - vwap) / vwap if vwap > 0 else 0.0
        
        return features
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical analysis indicators."""
        features = {}
        
        if len(price_data) < 50:
            return features
        
        close = price_data["close"].values
        high = price_data["high"].values if "high" in price_data else close
        low = price_data["low"].values if "low" in price_data else close
        
        # RSI
        rsi_14 = talib.RSI(close, timeperiod=14)
        rsi_30 = talib.RSI(close, timeperiod=30)
        features["rsi_14"] = rsi_14[-1] if len(rsi_14) > 0 and not np.isnan(rsi_14[-1]) else 50.0
        features["rsi_30"] = rsi_30[-1] if len(rsi_30) > 0 and not np.isnan(rsi_30[-1]) else 50.0
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        if len(macd) > 0:
            features["macd_signal"] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0
            features["macd_histogram"] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
            features["macd_divergence"] = macd[-1] - macd_signal[-1] if not np.isnan(macd[-1]) else 0.0
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        if len(bb_upper) > 0:
            bb_width = bb_upper[-1] - bb_lower[-1]
            features["bb_width"] = bb_width / bb_middle[-1] if bb_middle[-1] > 0 else 0.0
            features["bb_position"] = (close[-1] - bb_lower[-1]) / bb_width if bb_width > 0 else 0.5
        
        # ATR
        atr_14 = talib.ATR(high, low, close, timeperiod=14)
        features["atr_14"] = atr_14[-1] / close[-1] if len(atr_14) > 0 and not np.isnan(atr_14[-1]) else 0.0
        
        # ADX
        adx_14 = talib.ADX(high, low, close, timeperiod=14)
        features["adx_14"] = adx_14[-1] if len(adx_14) > 0 and not np.isnan(adx_14[-1]) else 0.0
        
        # CCI
        cci_20 = talib.CCI(high, low, close, timeperiod=20)
        features["cci_20"] = cci_20[-1] if len(cci_20) > 0 and not np.isnan(cci_20[-1]) else 0.0
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        if len(slowk) > 0:
            features["stoch_k"] = slowk[-1] if not np.isnan(slowk[-1]) else 50.0
            features["stoch_d"] = slowd[-1] if not np.isnan(slowd[-1]) else 50.0
        
        # Williams %R
        willr = talib.WILLR(high, low, close, timeperiod=14)
        features["williams_r"] = willr[-1] if len(willr) > 0 and not np.isnan(willr[-1]) else -50.0
        
        # OBV trend
        if "volume" in price_data:
            obv = talib.OBV(close, price_data["volume"].values)
            if len(obv) > 20:
                obv_ma = talib.SMA(obv, timeperiod=20)
                features["obv_trend"] = (obv[-1] - obv_ma[-1]) / obv_ma[-1] if obv_ma[-1] != 0 else 0.0
        
        # Trend strength
        if len(close) > 50:
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            if len(sma_20) > 0 and len(sma_50) > 0:
                features["trend_strength"] = (sma_20[-1] - sma_50[-1]) / sma_50[-1] if sma_50[-1] > 0 else 0.0
        
        # Trend consistency (using linear regression)
        if len(close) > 20:
            x = np.arange(20)
            y = close[-20:]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            features["trend_consistency"] = r_value**2  # R-squared
        
        return features
    
    def _calculate_microstructure_features(
        self,
        orderbook_data: Optional[pd.DataFrame],
        trade_data: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate market microstructure features."""
        features = {}
        
        if orderbook_data is not None and len(orderbook_data) > 0:
            # Bid-ask spread
            bid = orderbook_data["best_bid"].values[-1]
            ask = orderbook_data["best_ask"].values[-1]
            mid = (bid + ask) / 2
            features["spread"] = (ask - bid) / mid if mid > 0 else 0.0
            
            # Spread volatility
            if len(orderbook_data) > 20:
                spreads = (orderbook_data["best_ask"] - orderbook_data["best_bid"]) / \
                         ((orderbook_data["best_ask"] + orderbook_data["best_bid"]) / 2)
                features["spread_volatility"] = spreads.iloc[-20:].std()
            
            # Order book imbalance
            if "bid_volume" in orderbook_data and "ask_volume" in orderbook_data:
                bid_vol = orderbook_data["bid_volume"].values[-1]
                ask_vol = orderbook_data["ask_volume"].values[-1]
                total_vol = bid_vol + ask_vol
                features["book_imbalance"] = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0
            
            # Order book depth
            if "bid_depth" in orderbook_data and "ask_depth" in orderbook_data:
                features["book_depth"] = orderbook_data["bid_depth"].values[-1] + \
                                       orderbook_data["ask_depth"].values[-1]
        
        if trade_data is not None and len(trade_data) > 0:
            # Trade intensity
            if "timestamp" in trade_data:
                time_diff = (trade_data["timestamp"].iloc[-1] - trade_data["timestamp"].iloc[0]).total_seconds()
                features["trade_intensity"] = len(trade_data) / time_diff if time_diff > 0 else 0.0
            
            # Average trade size
            if "size" in trade_data:
                features["avg_trade_size"] = trade_data["size"].mean()
                features["trade_size_volatility"] = trade_data["size"].std() / features["avg_trade_size"]
            
            # Price impact
            if "price" in trade_data and "size" in trade_data:
                # Simplified price impact calculation
                prices = trade_data["price"].values
                sizes = trade_data["size"].values
                if len(prices) > 10:
                    price_changes = np.diff(prices)
                    features["price_impact"] = np.corrcoef(sizes[:-1], np.abs(price_changes))[0, 1]
        
        # Quote intensity (updates per second)
        features["quote_intensity"] = len(orderbook_data) / 60 if orderbook_data is not None else 0.0
        
        return features
    
    def _calculate_pattern_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pattern recognition features."""
        features = {}
        
        if len(price_data) < 50:
            return features
        
        close = price_data["close"].values
        high = price_data["high"].values if "high" in price_data else close
        low = price_data["low"].values if "low" in price_data else close
        
        # Support and resistance levels
        # Find local minima and maxima
        peaks, _ = find_peaks(high[-50:], distance=5)
        troughs, _ = find_peaks(-low[-50:], distance=5)
        
        current_price = close[-1]
        
        # Nearest resistance
        if len(peaks) > 0:
            resistance_levels = high[-50:][peaks]
            above_current = resistance_levels[resistance_levels > current_price]
            if len(above_current) > 0:
                nearest_resistance = np.min(above_current)
                features["resistance_distance"] = (nearest_resistance - current_price) / current_price
            else:
                features["resistance_distance"] = 0.05  # No resistance found
        
        # Nearest support
        if len(troughs) > 0:
            support_levels = low[-50:][troughs]
            below_current = support_levels[support_levels < current_price]
            if len(below_current) > 0:
                nearest_support = np.max(below_current)
                features["support_distance"] = (current_price - nearest_support) / current_price
            else:
                features["support_distance"] = 0.05  # No support found
        
        # Japanese candlestick patterns
        if "open" in price_data and len(price_data) > 3:
            # Doji detection
            body_size = np.abs(close[-1] - price_data["open"].values[-1])
            shadow_size = high[-1] - low[-1]
            features["is_doji"] = 1.0 if body_size < 0.1 * shadow_size else 0.0
            
            # Hammer/Hanging man
            lower_shadow = min(close[-1], price_data["open"].values[-1]) - low[-1]
            upper_shadow = high[-1] - max(close[-1], price_data["open"].values[-1])
            features["is_hammer"] = 1.0 if lower_shadow > 2 * body_size and upper_shadow < body_size else 0.0
            
            # Engulfing pattern
            prev_body = price_data["close"].values[-2] - price_data["open"].values[-2]
            curr_body = close[-1] - price_data["open"].values[-1]
            features["is_engulfing"] = 1.0 if np.abs(curr_body) > np.abs(prev_body) and \
                                            np.sign(curr_body) != np.sign(prev_body) else 0.0
        
        # Trend patterns
        if len(close) > 20:
            # Higher highs and higher lows (uptrend)
            recent_highs = high[-20:]
            recent_lows = low[-20:]
            
            hh_count = sum(1 for i in range(1, len(recent_highs)) 
                          if recent_highs[i] > recent_highs[i-1])
            hl_count = sum(1 for i in range(1, len(recent_lows)) 
                          if recent_lows[i] > recent_lows[i-1])
            
            features["uptrend_strength"] = (hh_count + hl_count) / (2 * (len(recent_highs) - 1))
            
            # Lower highs and lower lows (downtrend)
            lh_count = sum(1 for i in range(1, len(recent_highs)) 
                          if recent_highs[i] < recent_highs[i-1])
            ll_count = sum(1 for i in range(1, len(recent_lows)) 
                          if recent_lows[i] < recent_lows[i-1])
            
            features["downtrend_strength"] = (lh_count + ll_count) / (2 * (len(recent_highs) - 1))
        
        # Fibonacci retracement levels
        if len(close) > 50:
            recent_high = np.max(high[-50:])
            recent_low = np.min(low[-50:])
            price_range = recent_high - recent_low
            
            if price_range > 0:
                retracement = (recent_high - current_price) / price_range
                
                # Check proximity to Fibonacci levels
                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                min_distance = min(abs(retracement - level) for level in fib_levels)
                features["fib_proximity"] = 1.0 - min_distance  # Closer to Fib level = higher value
        
        return features
    
    def _calculate_regime_features(
        self,
        price_data: pd.DataFrame,
        volatility_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate market regime features."""
        features = {}
        
        if len(price_data) < 100:
            return features
        
        close = price_data["close"].values
        
        # Regime stability (using rolling correlation of returns)
        returns = np.diff(np.log(close))
        if len(returns) > 40:
            # Compare recent returns pattern with previous period
            recent_returns = returns[-20:]
            previous_returns = returns[-40:-20]
            
            # Rolling correlation
            correlations = []
            for i in range(10):
                corr = np.corrcoef(recent_returns[i:i+10], previous_returns[i:i+10])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            features["regime_stability"] = np.mean(correlations)
        
        # Regime transition probability
        # Using volatility clustering and trend changes
        vol_1h = volatility_features.get("volatility_1h", 0.0)
        vol_1d = volatility_features.get("volatility_1d", 0.0)
        
        # High short-term volatility relative to long-term suggests regime change
        vol_ratio = vol_1h / vol_1d if vol_1d > 0 else 1.0
        
        # Trend reversal detection
        if len(close) > 50:
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            
            if len(sma_20) > 1 and len(sma_50) > 1:
                # Check for MA crossover
                current_cross = sma_20[-1] - sma_50[-1]
                prev_cross = sma_20[-2] - sma_50[-2]
                is_crossover = np.sign(current_cross) != np.sign(prev_cross)
                
                # Combine factors
                features["transition_prob"] = min(1.0, 
                    0.3 * (vol_ratio - 1.0) + 
                    0.5 * (1.0 if is_crossover else 0.0) +
                    0.2 * (1.0 - features.get("regime_stability", 0.5))
                )
        
        # Regime classification features
        # These would be used by the ML model to classify regime
        features["mean_return_20"] = np.mean(returns[-20:]) if len(returns) > 20 else 0.0
        features["vol_change_rate"] = (vol_1h - vol_1d) / vol_1d if vol_1d > 0 else 0.0
        features["trend_strength_regime"] = (close[-1] - close[-20]) / close[-20] if len(close) > 20 else 0.0
        
        return features
    
    def _calculate_correlation_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation with major assets."""
        features = {
            "btc_correlation": 0.0,  # Would need BTC price data
            "market_correlation": 0.0,  # Would need market index data
            "beta": 1.0,
        }
        
        # In production, you would:
        # 1. Load BTC and market index data
        # 2. Calculate rolling correlations
        # 3. Calculate beta using CAPM model
        
        return features
    
    def calculate_fourier_features(self, price_data: pd.DataFrame, n_components: int = 10) -> Dict[str, float]:
        """Extract frequency domain features using Fourier transform."""
        features = {}
        
        if len(price_data) < 100:
            return features
        
        close = price_data["close"].values[-100:]
        
        # Detrend the data
        detrended = close - np.linspace(close[0], close[-1], len(close))
        
        # Apply FFT
        fft_result = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended))
        
        # Get dominant frequencies
        power_spectrum = np.abs(fft_result)**2
        dominant_indices = np.argsort(power_spectrum)[-n_components:]
        
        for i, idx in enumerate(dominant_indices):
            features[f"fourier_freq_{i}"] = frequencies[idx]
            features[f"fourier_power_{i}"] = power_spectrum[idx]
        
        # Spectral entropy (measure of complexity)
        normalized_spectrum = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(normalized_spectrum * np.log(normalized_spectrum + 1e-10))
        features["spectral_entropy"] = spectral_entropy
        
        return features
    
    def calculate_wavelet_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Extract features using wavelet transform."""
        features = {}
        
        try:
            import pywt
        except ImportError:
            return features
        
        if len(price_data) < 100:
            return features
        
        close = price_data["close"].values[-100:]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(close, 'db4', level=4)
        
        # Extract features from each level
        for i, coeff in enumerate(coeffs):
            features[f"wavelet_energy_{i}"] = np.sum(coeff**2)
            features[f"wavelet_std_{i}"] = np.std(coeff)
            features[f"wavelet_mean_{i}"] = np.mean(np.abs(coeff))
        
        return features
    
    def create_lag_features(
        self,
        features: MarketFeatures,
        lag_periods: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """Create lagged features for time series modeling."""
        lag_features = {}
        
        # Store current features
        feature_dict = asdict(features)
        self._feature_cache.append(feature_dict)
        
        # Create lag features
        for lag in lag_periods:
            if len(self._feature_cache) > lag:
                past_features = self._feature_cache[-lag-1]
                for feature_name, current_value in feature_dict.items():
                    if isinstance(current_value, (int, float)):
                        past_value = past_features.get(feature_name, 0.0)
                        lag_features[f"{feature_name}_lag_{lag}"] = past_value
                        lag_features[f"{feature_name}_change_{lag}"] = current_value - past_value
        
        return lag_features
    
    def create_interaction_features(self, features: MarketFeatures) -> Dict[str, float]:
        """Create interaction features between different indicators."""
        interaction_features = {}
        feature_dict = asdict(features)
        
        # Price-Volume interactions
        interaction_features["price_volume_correlation"] = \
            feature_dict["price_return_1h"] * feature_dict["volume_ratio_1h"]
        
        # Volatility-Volume interactions
        interaction_features["volatility_volume_product"] = \
            feature_dict["volatility_1h"] * feature_dict["volume_volatility"]
        
        # RSI-Trend interactions
        interaction_features["rsi_trend_signal"] = \
            (feature_dict["rsi_14"] - 50) * feature_dict["trend_strength"]
        
        # Spread-Volatility interactions
        interaction_features["spread_volatility_ratio"] = \
            feature_dict["bid_ask_spread"] / (feature_dict["volatility_1h"] + 0.001)
        
        # Support/Resistance strength
        interaction_features["sr_strength"] = \
            1.0 / (feature_dict["support_distance"] + feature_dict["resistance_distance"] + 0.01)
        
        return interaction_features