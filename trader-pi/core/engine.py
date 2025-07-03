#!/usr/bin/env python3
"""
Lightweight trading engine optimized for Raspberry Pi
"""
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import ccxt

from core.risk import RiskManager
from core.monitor import Monitor


class TradingEngine:
    """Simplified trading engine for resource-constrained environments"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = self._load_config(config_path)
        self.exchange = None
        self.risk_manager = None
        self.monitor = None
        self.strategies = {}
        self.positions = {}
        self.running = False
        
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def initialize(self):
        """Initialize exchange connection and components"""
        try:
            # Initialize exchange
            exchange_config = self.config['exchange']
            self.exchange = getattr(ccxt, exchange_config['name'])({
                'apiKey': exchange_config['api_key'],
                'secret': exchange_config['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'testnet': exchange_config['testnet']
                }
            })
            
            # Initialize components
            self.risk_manager = RiskManager(self.config['risk'], self.config['trading'])
            self.monitor = Monitor(self.config['monitoring'])
            
            # Load strategies
            self._load_strategies()
            
            self.logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    def _load_strategies(self):
        """Load enabled strategies"""
        enabled_strategies = self.config['strategies']['enabled']
        
        if 'trend_following' in enabled_strategies:
            from strategies.trend_following import TrendFollowingStrategy
            self.strategies['trend_following'] = TrendFollowingStrategy(
                self.config['strategies']
            )
        
        if 'mean_reversion' in enabled_strategies:
            from strategies.mean_reversion import MeanReversionStrategy
            self.strategies['mean_reversion'] = MeanReversionStrategy(
                self.config['strategies']
            )
        
        self.logger.info(f"Loaded strategies: {list(self.strategies.keys())}")
    
    def start(self):
        """Start the trading engine"""
        if not self.initialize():
            return False
        
        self.running = True
        self.logger.info("Starting trading engine...")
        
        try:
            while self.running:
                self._trading_loop()
                time.sleep(self.config['monitoring']['update_interval_seconds'])
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            self.stop()
        except Exception as e:
            self.logger.error(f"Critical error: {e}")
            self.stop()
    
    def _trading_loop(self):
        """Main trading loop"""
        try:
            # Update market data
            instruments = self._load_instruments()
            
            for instrument in instruments:
                if not instrument['enabled']:
                    continue
                
                symbol = instrument['symbol']
                
                # Fetch market data
                ohlcv = self._fetch_ohlcv(symbol)
                if not ohlcv:
                    continue
                
                # Check risk limits
                if not self.risk_manager.can_trade(symbol, self.positions):
                    continue
                
                # Run strategies
                for strategy_name in instrument['strategies']:
                    if strategy_name not in self.strategies:
                        continue
                    
                    strategy = self.strategies[strategy_name]
                    signal = strategy.analyze(symbol, ohlcv)
                    
                    if signal:
                        self._process_signal(symbol, signal, instrument)
            
            # Update positions
            self._update_positions()
            
            # Monitor system
            self.monitor.update(self.positions, self.exchange)
            
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
    
    def _load_instruments(self) -> List[Dict]:
        """Load instrument configurations"""
        with open('config/instruments.json', 'r') as f:
            return json.load(f)['instruments']
    
    def _fetch_ohlcv(self, symbol: str) -> Optional[List]:
        """Fetch OHLCV data for a symbol"""
        try:
            timeframes = self.config['strategies']['timeframes']
            ohlcv_data = {}
            
            for tf in timeframes:
                data = self.exchange.fetch_ohlcv(
                    symbol, 
                    tf, 
                    limit=self.config['strategies']['default_lookback']
                )
                ohlcv_data[tf] = data
            
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def _process_signal(self, symbol: str, signal: Dict, instrument: Dict):
        """Process trading signal"""
        try:
            # Check if we already have a position
            if symbol in self.positions and signal['action'] == 'BUY':
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, 
                signal, 
                self.exchange.fetch_balance()
            )
            
            if position_size < instrument['min_order_size']:
                return
            
            # Place order
            if signal['action'] == 'BUY':
                order = self.exchange.create_market_buy_order(symbol, position_size)
                
                # Record position
                self.positions[symbol] = {
                    'entry_price': order['price'],
                    'size': position_size,
                    'entry_time': datetime.now(),
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit')
                }
                
                self.logger.info(f"Opened position: {symbol} @ {order['price']}")
                
            elif signal['action'] == 'SELL' and symbol in self.positions:
                order = self.exchange.create_market_sell_order(
                    symbol, 
                    self.positions[symbol]['size']
                )
                
                # Calculate P&L
                pnl = (order['price'] - self.positions[symbol]['entry_price']) * \
                      self.positions[symbol]['size']
                
                self.logger.info(f"Closed position: {symbol} @ {order['price']}, PnL: {pnl}")
                
                # Remove position
                del self.positions[symbol]
                
        except Exception as e:
            self.logger.error(f"Failed to process signal for {symbol}: {e}")
    
    def _update_positions(self):
        """Update open positions and check for exits"""
        for symbol, position in list(self.positions.items()):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Check stop loss
                if position['stop_loss'] and current_price <= position['stop_loss']:
                    self._close_position(symbol, 'STOP_LOSS')
                
                # Check take profit
                elif position['take_profit'] and current_price >= position['take_profit']:
                    self._close_position(symbol, 'TAKE_PROFIT')
                
            except Exception as e:
                self.logger.error(f"Failed to update position {symbol}: {e}")
    
    def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.positions[symbol]
            order = self.exchange.create_market_sell_order(symbol, position['size'])
            
            self.logger.info(f"Position closed ({reason}): {symbol} @ {order['price']}")
            del self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {e}")
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, 'SHUTDOWN')
        
        self.logger.info("Trading engine stopped")


if __name__ == "__main__":
    engine = TradingEngine()
    engine.start()