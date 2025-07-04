#!/usr/bin/env python3
"""
Lightweight trading engine optimized for Raspberry Pi
"""
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque
import numpy as np
import ccxt
import ccxt.async_support as ccxt_async

from core.config_loader import ConfigLoader
from core.risk import RiskManager
from core.monitor import Monitor
from core.trading_metrics import Position, TradingMetrics
from core.market_scanner import MarketScanner, MarketOpportunity
from core.testnet_scanner import TestnetScanner
from core.tax_calculator import TaxCalculator


class TradingEngine:
    """Simplified trading engine for resource-constrained environments"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = ConfigLoader.load(config_path)
        
        # Validate configuration
        if not ConfigLoader.validate_config(self.config):
            raise ValueError("Invalid configuration. Please check your .env file")
        
        self.exchange = None
        self.risk_manager = None
        self.monitor = None
        self.strategies = {}
        self.positions: Dict[str, Position] = {}
        self.running = False
        
        # Performance tracking
        self.trade_history = deque(maxlen=1000)
        self.equity_curve = deque(maxlen=10000)
        self.initial_balance = float(self.config.get('trading', {}).get('initial_balance', 10000))
        self.balance = self.initial_balance
        
        # Market scanner
        self.scanner = None
        self.scanner_task = None
        self.opportunities_queue = asyncio.Queue(maxsize=100)
        
        # Tax calculator
        tax_config = self.config.get('tax', {
            'method': 'FIFO',
            'short_term_rate': 0.35,
            'long_term_rate': 0.15,
            'long_term_days': 365
        })
        self.tax_calculator = TaxCalculator(tax_config)
        
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize exchange connection and components"""
        try:
            # Initialize exchange
            exchange_config = self.config['exchange']
            # Create exchange instance
            self.exchange = getattr(ccxt, exchange_config['name'])({
                'apiKey': exchange_config['api_key'],
                'secret': exchange_config['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Use SPOT trading for testnet compatibility
                    'adjustForTimeDifference': True  # Important for testnet
                }
            })
            
            # Set testnet mode if enabled
            if exchange_config.get('testnet', False):
                self.exchange.set_sandbox_mode(True)
            
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
        
        # Run async event loop
        asyncio.run(self._async_main())
    
    async def _async_main(self):
        """Main async entry point"""
        try:
            # Initialize scanner
            await self._initialize_scanner()
            
            # Start scanner task
            self.scanner_task = asyncio.create_task(self._scanner_loop())
            
            # Run main trading loop
            while self.running:
                await self._async_trading_loop()
                await asyncio.sleep(self.config['monitoring']['update_interval_seconds'])
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            await self.async_stop()
        except Exception as e:
            self.logger.error(f"Critical error: {e}")
            await self.async_stop()
    
    async def _initialize_scanner(self):
        """Initialize the market scanner"""
        exchange_config = self.config['exchange']
        
        # Use TestnetScanner for testnet mode
        scanner_class = TestnetScanner if exchange_config.get('testnet', False) else MarketScanner
        
        self.scanner = scanner_class(
            exchange_name=exchange_config['name'],
            max_concurrent=50
        )
        await self.scanner.initialize(
            api_key=exchange_config['api_key'],
            api_secret=exchange_config['api_secret'],
            testnet=exchange_config['testnet']
        )
        scanner_type = "Testnet scanner" if exchange_config.get('testnet', False) else "Market scanner"
        self.logger.info(f"{scanner_type} initialized")
    
    async def _scanner_loop(self):
        """Run continuous market scanning"""
        scan_interval = self.config.get('scanner', {}).get('interval_seconds', 30)
        
        while self.running:
            try:
                # Scan markets
                opportunities = await self.scanner.scan_markets(
                    min_volume_24h=self.config.get('scanner', {}).get('min_volume_24h', 1000000)
                )
                
                # Queue top opportunities
                for opp in opportunities[:10]:  # Top 10
                    if self.opportunities_queue.full():
                        try:
                            self.opportunities_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self.opportunities_queue.put(opp)
                
                # Log scanner stats
                stats = self.scanner.get_scan_stats()
                self.logger.info(
                    f"Scanner stats - Avg time: {stats.get('avg_scan_time', 0):.2f}s, "
                    f"Opportunities: {stats.get('opportunities_found', 0)}"
                )
                
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                self.logger.error(f"Scanner error: {e}")
                await asyncio.sleep(60)
    
    async def _async_trading_loop(self):
        """Async trading loop that processes scanner results"""
        try:
            # Process opportunities from scanner
            while not self.opportunities_queue.empty():
                try:
                    opp = await asyncio.wait_for(
                        self.opportunities_queue.get(), 
                        timeout=0.1
                    )
                    await self._process_opportunity(opp)
                except asyncio.TimeoutError:
                    break
            
            # Original trading loop logic (converted to async)
            await self._update_positions_async()
            
            # Monitor system
            self.monitor.update(self.positions, self.exchange)
            
        except Exception as e:
            self.logger.error(f"Error in async trading loop: {e}")
    
    async def _process_opportunity(self, opp: MarketOpportunity):
        """Process a market opportunity from scanner"""
        try:
            # Check if we already have a position
            if opp.symbol in self.positions:
                return
            
            # Check risk limits
            if not self.risk_manager.can_trade(opp.symbol, self.positions):
                return
            
            # Check opportunity score threshold
            min_score = self.config.get('scanner', {}).get('min_opportunity_score', 70)
            if opp.score < min_score:
                return
            
            # Create signal from opportunity
            signal = {
                'action': opp.signal,
                'stop_loss': opp.stop_loss,
                'take_profit': opp.take_profit,
                'strategy': opp.strategy,
                'score': opp.score,
                'price': opp.entry_price  # Add price for position sizing
            }
            
            # Process the signal
            instrument = {
                'symbol': opp.symbol,
                'min_order_size': 0.001,  # TODO: Get from exchange
                'strategies': [opp.strategy]
            }
            
            self._process_signal(opp.symbol, signal, instrument)
            
            self.logger.info(
                f"Processed opportunity: {opp.symbol} - {opp.strategy} "
                f"(score: {opp.score:.1f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing opportunity: {e}")
    
    async def _update_positions_async(self):
        """Async version of position updates"""
        # Convert to sync for now (can be optimized later)
        self._update_positions()
    
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
            balance = self.exchange.fetch_balance()
            position_size = self.risk_manager.calculate_position_size(
                symbol, 
                signal, 
                balance
            )
            
            self.logger.info(f"Calculated position size for {symbol}: {position_size}")
            
            if position_size < instrument['min_order_size']:
                self.logger.warning(f"Position size {position_size} < min order size {instrument['min_order_size']} for {symbol}")
                return
            
            # Place order
            if signal['action'] == 'BUY':
                self.logger.info(f"Attempting to buy {position_size} {symbol}")
                order = self.exchange.create_market_buy_order(symbol, position_size)
                
                # Record position
                position = Position(
                    id=f"{symbol}_{datetime.now().timestamp()}",
                    symbol=symbol,
                    side='buy',
                    quantity=position_size,
                    entry_price=order['price'],
                    current_price=order['price'],
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit')
                )
                self.positions[symbol] = position
                
                self.logger.info(f"✅ Opened position: {symbol} @ {order['price']}")
                
            elif signal['action'] == 'SELL' and symbol in self.positions:
                order = self.exchange.create_market_sell_order(
                    symbol, 
                    self.positions[symbol]['size']
                )
                
                # Calculate P&L
                position = self.positions[symbol]
                position.current_price = order['price']
                pnl = position.pnl
                
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
                position.current_price = current_price
                position.updated_at = datetime.now()
                
                # Check stop loss
                if position.stop_loss and current_price <= position.stop_loss:
                    self._close_position(symbol, 'STOP_LOSS')
                
                # Check take profit
                elif position.take_profit and current_price >= position.take_profit:
                    self._close_position(symbol, 'TAKE_PROFIT')
                
            except Exception as e:
                self.logger.error(f"Failed to update position {symbol}: {e}")
    
    def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.positions[symbol]
            order = self.exchange.create_market_sell_order(symbol, position.quantity)
            
            # Update final price and record trade
            position.current_price = order['price']
            trade_record = {
                'symbol': symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': position.current_price,
                'quantity': position.quantity,
                'pnl': position.pnl,
                'pnl_percentage': position.pnl_percentage,
                'duration': datetime.now() - position.opened_at,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            self.trade_history.append(trade_record)
            
            # Record for tax purposes
            # First record the buy
            self.tax_calculator.add_transaction({
                'timestamp': position.opened_at.isoformat(),
                'symbol': symbol,
                'side': 'buy',
                'quantity': position.quantity,
                'price': position.entry_price,
                'fee': position.quantity * position.entry_price * 0.001  # 0.1% fee estimate
            })
            
            # Then record the sell
            self.tax_calculator.add_transaction({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': 'sell',
                'quantity': position.quantity,
                'price': position.current_price,
                'fee': position.quantity * position.current_price * 0.001
            })
            
            self.logger.info(f"Position closed ({reason}): {symbol} @ {order['price']}, PnL: ${position.pnl:.2f} ({position.pnl_percentage:.2f}%)")
            del self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {e}")
    
    def get_metrics(self) -> TradingMetrics:
        """Get current trading metrics"""
        try:
            # Get account balance
            balance_info = self.exchange.fetch_balance()
            total_balance = float(balance_info['USDT']['total']) if 'USDT' in balance_info else self.balance
            free_balance = float(balance_info['USDT']['free']) if 'USDT' in balance_info else self.balance
            
            # Calculate metrics
            open_positions = len(self.positions)
            unrealized_pnl = sum(p.pnl for p in self.positions.values())
            equity = total_balance + unrealized_pnl
            
            # Win rate from trade history
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
            
            # Daily P&L
            today_trades = [t for t in self.trade_history 
                          if isinstance(t.get('duration'), datetime) and t['duration'].date() == datetime.now().date()]
            daily_pnl = sum(t['pnl'] for t in today_trades) + unrealized_pnl
            
            # Drawdown calculation
            if self.equity_curve:
                peak = max(self.equity_curve)
                current_drawdown = (peak - equity) / peak if peak > 0 else 0
                max_drawdown = max((peak - e) / peak for e in self.equity_curve) if peak > 0 else 0
            else:
                current_drawdown = 0
                max_drawdown = 0
            
            # Update equity curve
            self.equity_curve.append(equity)
            
            return TradingMetrics(
                total_balance=total_balance,
                available_balance=free_balance,
                equity=equity,
                margin_used=total_balance - free_balance,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=sum(t['pnl'] for t in self.trade_history),
                daily_pnl=daily_pnl,
                win_rate=win_rate,
                total_trades=len(self.trade_history),
                open_positions=open_positions,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                sharpe_ratio=0  # TODO: Implement Sharpe ratio calculation
            )
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {e}")
            # Return default metrics on error
            return TradingMetrics(
                total_balance=self.balance,
                available_balance=self.balance,
                equity=self.balance,
                margin_used=0,
                unrealized_pnl=0,
                realized_pnl=0,
                daily_pnl=0,
                win_rate=0,
                total_trades=0,
                open_positions=0,
                current_drawdown=0,
                max_drawdown=0,
                sharpe_ratio=0
            )
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, 'SHUTDOWN')
        
        self.logger.info("Trading engine stopped")
    
    async def async_stop(self):
        """Async stop for proper cleanup"""
        self.running = False
        
        # Stop scanner
        if self.scanner:
            self.scanner.stop()
            await self.scanner.close()
        
        # Cancel scanner task
        if self.scanner_task:
            self.scanner_task.cancel()
            try:
                await self.scanner_task
            except asyncio.CancelledError:
                pass
        
        # Close positions
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, 'SHUTDOWN')
        
        self.logger.info("Trading engine stopped")


if __name__ == "__main__":
    engine = TradingEngine()
    engine.start()