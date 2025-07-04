# 🎯 Trading Strategies

Comprehensive guide to the bot's trading strategies and how they work.

## 📊 Strategy Overview

The bot implements **4 core strategies** that work together to identify profitable trading opportunities:

| Strategy | Focus | Time Horizon | Risk Level | Win Rate Target |
|----------|-------|--------------|------------|-----------------|
| **Trend Following** | Momentum continuation | Medium-term | Medium | 60-70% |
| **Mean Reversion** | Oversold bounces | Short-term | Low | 65-75% |
| **Momentum** | Strong breakouts | Short-term | High | 55-65% |
| **Volume Breakout** | Unusual activity | Very short | Medium | 50-60% |

## 📈 1. Trend Following Strategy

### Concept
Identifies and follows established trends by detecting when price momentum is likely to continue.

### Technical Indicators
```python
signals = {
    'ema_12_above_ema_26': True,    # Short EMA above long EMA
    'macd_above_signal': True,       # MACD line above signal line
    'price_above_sma_50': True,      # Price above 50-period SMA
    'rsi_range': '40-70'             # RSI in momentum range
}
```

### Entry Conditions
- **EMA Crossover**: 12-period EMA above 26-period EMA
- **MACD Bullish**: MACD line above signal line
- **Price Position**: Current price above 50-period SMA
- **RSI Confirmation**: RSI between 40-70 (avoiding extremes)

### Scoring Formula
```python
def calculate_trend_score(indicators):
    base_score = 50  # Base score when conditions met
    trend_strength = abs(indicators['macd']) / current_price * 100
    final_score = min(90, base_score + trend_strength * 10)
    return final_score
```

### Example Trade
```
Symbol: BTC/USDT
Entry: $45,234.56
Stop Loss: $44,329.12 (2% below entry)
Take Profit: $46,140.01 (2x risk)
Score: 82.5
Reason: Strong MACD divergence with EMA support
```

## 📊 2. Mean Reversion Strategy

### Concept
Capitalizes on oversold conditions where price is likely to bounce back to the mean.

### Technical Indicators
```python
signals = {
    'price_below_bb_lower': True,    # Price below Bollinger lower band
    'rsi_oversold': 'RSI < 30',      # Extremely oversold
    'volume_spike': '> 1.5x average' # Increased selling pressure
}
```

### Entry Conditions
- **Bollinger Bands**: Price below lower Bollinger Band
- **RSI Oversold**: RSI below 30 (extreme oversold)
- **Volume Confirmation**: Volume 1.5x above average
- **Support Level**: Near historical support if available

### Scoring Formula
```python
def calculate_reversion_score(indicators):
    base_score = 40  # Lower base score (contrarian)
    sma_20 = indicators['sma_20']
    deviation = (sma_20 - current_price) / current_price
    final_score = min(85, base_score + deviation * 100)
    return final_score
```

### Example Trade
```
Symbol: ETH/USDT
Entry: $2,234.56 (oversold at BB lower)
Stop Loss: $2,188.27 (2% below)
Take Profit: $2,345.67 (back to SMA 20)
Score: 72.3
Reason: RSI 28, price 3% below 20-day average
```

## 🚀 3. Momentum Strategy

### Concept
Catches strong price movements early by detecting acceleration in trending moves.

### Technical Indicators
```python
signals = {
    'price_change_10_bars': '> 2%',   # Strong recent movement
    'rsi_momentum': '60-80',          # Bullish but not overbought
    'volume_surge': '> 2x average'    # Volume confirmation
}
```

### Entry Conditions
- **Price Movement**: >2% move in last 10 bars
- **RSI Range**: Between 60-80 (strong but sustainable)
- **Volume Confirmation**: Volume >2x average
- **Trend Alignment**: Movement aligned with overall trend

### Scoring Formula
```python
def calculate_momentum_score(indicators):
    base_score = 40
    momentum = (current_price - price_10_bars_ago) / price_10_bars_ago
    final_score = min(80, base_score + momentum * 500)
    return final_score
```

### Example Trade
```
Symbol: SOL/USDT
Entry: $98.45 (3.2% move in 10 bars)
Stop Loss: $95.87 (1.5x ATR below)
Take Profit: $104.23 (2.5x risk)
Score: 76.8
Reason: Strong breakout with volume confirmation
```

## 📊 4. Volume Breakout Strategy

### Concept
Identifies significant price movements supported by unusual trading volume.

### Technical Indicators
```python
signals = {
    'volume_ratio': '> 3x average',   # Exceptional volume
    'price_movement': '> 1% in 5 bars', # Price breakout
    'price_above_sma_20': True        # Bullish structure
}
```

### Entry Conditions
- **Volume Spike**: Volume >3x recent average
- **Price Breakout**: >1% move in last 5 bars
- **Technical Structure**: Price above 20-period SMA
- **Breakout Confirmation**: Clean break of resistance

### Scoring Formula
```python
def calculate_volume_score(indicators):
    base_score = 30  # Lower base (volume can be misleading)
    volume_ratio = indicators['volume_ratio']
    final_score = min(75, base_score + volume_ratio * 10)
    return final_score
```

### Example Trade
```
Symbol: ADA/USDT
Entry: $0.4567 (5x volume spike)
Stop Loss: $0.4452 (1.5x ATR)
Take Profit: $0.4798 (3x risk)
Score: 68.5
Reason: Massive volume with price acceleration
```

## 🎛️ Strategy Configuration

### Risk Parameters by Strategy
```json
{
  "trend_following": {
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_atr_multiplier": 3.0,
    "min_rsi": 40,
    "max_rsi": 70
  },
  "mean_reversion": {
    "stop_loss_percentage": 0.02,
    "take_profit_sma_target": "sma_20",
    "max_rsi": 30,
    "min_volume_ratio": 1.5
  },
  "momentum": {
    "min_move_percentage": 0.02,
    "stop_loss_atr_multiplier": 1.5,
    "take_profit_atr_multiplier": 2.5,
    "min_volume_ratio": 2.0
  },
  "volume_breakout": {
    "min_volume_ratio": 3.0,
    "min_price_move": 0.01,
    "stop_loss_atr_multiplier": 1.5,
    "take_profit_atr_multiplier": 3.0
  }
}
```

### Strategy Selection Logic
```python
def select_best_strategy(opportunities):
    """
    Choose the highest scoring opportunity from all strategies
    """
    if not opportunities:
        return None
    
    # Sort by score (highest first)
    sorted_opportunities = sorted(opportunities, 
                                key=lambda x: x.score, 
                                reverse=True)
    
    # Return best opportunity above threshold
    best = sorted_opportunities[0]
    if best.score >= config['min_opportunity_score']:
        return best
    
    return None
```

## 📊 Performance Characteristics

### Expected Performance by Strategy

| Strategy | Avg Score | Win Rate | Avg Gain | Avg Loss | Risk/Reward |
|----------|-----------|----------|----------|----------|-------------|
| **Trend Following** | 65-85 | 65% | 4.2% | -2.1% | 1:2 |
| **Mean Reversion** | 55-75 | 70% | 3.8% | -1.9% | 1:2 |
| **Momentum** | 60-80 | 58% | 5.1% | -2.3% | 1:2.2 |
| **Volume Breakout** | 50-70 | 55% | 4.7% | -2.2% | 1:2.1 |

### Market Condition Adaptation

```python
market_conditions = {
    'trending_up': {
        'preferred': ['trend_following', 'momentum'],
        'avoid': ['mean_reversion']
    },
    'trending_down': {
        'preferred': ['mean_reversion'],
        'avoid': ['trend_following', 'momentum']
    },
    'sideways': {
        'preferred': ['mean_reversion', 'volume_breakout'],
        'avoid': ['momentum']
    },
    'high_volatility': {
        'preferred': ['momentum', 'volume_breakout'],
        'avoid': ['trend_following']
    }
}
```

## 🔧 Customizing Strategies

### Adjusting for Testnet
```json
{
  "testnet_adjustments": {
    "trend_following": {
      "min_move_threshold": 0.001,
      "reduced_volume_requirements": true
    },
    "mean_reversion": {
      "rsi_threshold": 40,
      "volume_multiplier": 1.2
    },
    "momentum": {
      "min_move_percentage": 0.002,
      "volume_multiplier": 1.2
    },
    "volume_breakout": {
      "volume_ratio": 1.5,
      "min_price_move": 0.001
    }
  }
}
```

### Adding Custom Strategies
```python
class CustomStrategy(BaseStrategy):
    def analyze(self, symbol, data):
        """
        Implement your custom strategy logic
        """
        # Your analysis here
        opportunity_score = calculate_score(data)
        
        if opportunity_score > threshold:
            return MarketOpportunity(
                symbol=symbol,
                strategy='custom_strategy',
                score=opportunity_score,
                signal='BUY',
                entry_price=data['close'][-1],
                stop_loss=calculate_stop_loss(data),
                take_profit=calculate_take_profit(data)
            )
        
        return None
```

## 📈 Strategy Optimization

### Backtesting Results
```python
# Example backtesting results (6 months testnet data)
results = {
    'trend_following': {
        'total_trades': 245,
        'win_rate': 0.67,
        'avg_gain': 0.042,
        'max_drawdown': 0.08,
        'sharpe_ratio': 1.34
    },
    'mean_reversion': {
        'total_trades': 312,
        'win_rate': 0.71,
        'avg_gain': 0.038,
        'max_drawdown': 0.06,
        'sharpe_ratio': 1.52
    }
}
```

### Performance Monitoring
```python
def monitor_strategy_performance():
    """
    Track strategy performance in real-time
    """
    metrics = {
        'trades_per_strategy': count_by_strategy(),
        'win_rate_by_strategy': calculate_win_rates(),
        'profit_by_strategy': calculate_profits(),
        'recent_performance': get_last_30_days()
    }
    
    # Alert if strategy underperforming
    for strategy, performance in metrics.items():
        if performance['win_rate'] < 0.45:  # Below 45%
            send_alert(f"Strategy {strategy} underperforming")
```

---

**Next:** [Tax Features](tax-features.md) - Understanding the built-in tax tracking