"""
Comprehensive backtesting harness for strategy validation and optimization.
"""

import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import pickle

from .interfaces import (
    BaseStrategyInterface,
    TradingSignal,
    SignalType,
    StrategyState,
)
from .core import AdaptiveStrategy
from .ensemble import StrategyEnsemble


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    data_frequency: str = '1h'
    benchmark: Optional[str] = None
    risk_free_rate: float = 0.02
    max_positions: int = 10
    position_sizing: str = 'equal'  # equal, kelly, risk_parity
    rebalance_frequency: str = 'daily'
    transaction_costs: bool = True
    use_stops: bool = True
    stop_loss: float = 0.02  # 2%
    take_profit: float = 0.05  # 5%
    enable_shorting: bool = True
    margin_requirement: float = 0.5  # 50% for shorts
    parallel_execution: bool = True
    n_workers: int = 4


@dataclass
class Trade:
    """Represents a single trade."""
    trade_id: str
    instrument: str
    side: str  # buy/sell
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0
    commission: float = 0
    slippage: float = 0
    pnl: float = 0
    return_pct: float = 0
    duration: Optional[timedelta] = None
    exit_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_id: str
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: pd.DataFrame
    performance_metrics: Dict[str, float]
    monthly_returns: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    signals: List[Dict[str, Any]]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_id': self.strategy_id,
            'config': self.config.__dict__,
            'n_trades': len(self.trades),
            'performance_metrics': self.performance_metrics,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


class BacktestEngine:
    """
    High-performance backtesting engine with parallel execution.
    
    Features:
    - Vectorized calculations for speed
    - Multi-asset support
    - Walk-forward analysis
    - Monte Carlo simulation
    - Parameter optimization
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_cache = {}
        self.results_cache = {}
        self.current_positions = {}
        self.open_trades = {}
        self.closed_trades = []
        self.equity_curve = []
        self.signals = []
        self.executor = None
        
        if config.parallel_execution:
            self.executor = ProcessPoolExecutor(max_workers=config.n_workers)
    
    async def run_backtest(self, strategy: BaseStrategyInterface, 
                          data: pd.DataFrame) -> BacktestResult:
        """Run backtest for a single strategy."""
        start_time = datetime.utcnow()
        
        # Initialize strategy
        await strategy.initialize()
        
        # Reset state
        self._reset_state()
        
        # Initialize equity
        current_equity = self.config.initial_capital
        self.equity_curve.append({
            'timestamp': self.config.start_date,
            'equity': current_equity,
            'cash': current_equity,
            'positions_value': 0
        })
        
        # Process data chronologically
        for idx, (timestamp, row) in enumerate(data.iterrows()):
            # Update market data
            market_data = self._prepare_market_data(row)
            
            # Get signal from strategy
            signal = await strategy.on_data(market_data)
            
            if signal:
                self.signals.append({
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': row.get('close', 0)
                })
                
                # Process signal
                await self._process_signal(signal, timestamp, row)
            
            # Update open positions
            self._update_positions(timestamp, row)
            
            # Check stops
            if self.config.use_stops:
                self._check_stops(timestamp, row)
            
            # Update equity
            current_equity = self._calculate_equity(row)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'cash': self._calculate_cash(),
                'positions_value': self._calculate_positions_value(row)
            })
            
            # Rebalance if needed
            if self._should_rebalance(timestamp):
                await self._rebalance_portfolio(strategy, timestamp, row)
        
        # Close all remaining positions
        self._close_all_positions(data.index[-1], data.iloc[-1])
        
        # Calculate results
        result = self._calculate_results(strategy.strategy_id, start_time)
        
        # Cache results
        self.results_cache[strategy.strategy_id] = result
        
        return result
    
    async def run_ensemble_backtest(self, ensemble: StrategyEnsemble, 
                                   data: pd.DataFrame) -> BacktestResult:
        """Run backtest for an ensemble of strategies."""
        start_time = datetime.utcnow()
        
        # Initialize ensemble
        await ensemble.initialize()
        
        # Reset state
        self._reset_state()
        
        # Initialize equity
        current_equity = self.config.initial_capital
        
        # Process data
        for timestamp, row in data.iterrows():
            market_data = self._prepare_market_data(row)
            
            # Get ensemble signal
            signal = await ensemble.on_data(market_data)
            
            if signal:
                self.signals.append({
                    'timestamp': timestamp,
                    'signal': signal,
                    'ensemble_weights': ensemble.strategy_weights.copy()
                })
                
                await self._process_signal(signal, timestamp, row)
            
            # Update positions and equity
            self._update_positions(timestamp, row)
            current_equity = self._calculate_equity(row)
            
            # Update individual strategy performance
            for strategy_id, strategy in ensemble.strategies.items():
                # Track individual performance for weight optimization
                pass
        
        # Calculate results
        result = self._calculate_results(f"ensemble_{ensemble.ensemble_id}", start_time)
        
        return result
    
    async def run_walk_forward_analysis(self, strategy: BaseStrategyInterface,
                                       data: pd.DataFrame,
                                       train_period: int = 252,
                                       test_period: int = 63,
                                       overlap: int = 0) -> List[BacktestResult]:
        """Run walk-forward analysis."""
        results = []
        
        total_periods = len(data)
        start_idx = 0
        
        while start_idx + train_period + test_period <= total_periods:
            # Training data
            train_end = start_idx + train_period
            train_data = data.iloc[start_idx:train_end]
            
            # Test data
            test_start = train_end
            test_end = test_start + test_period
            test_data = data.iloc[test_start:test_end]
            
            # Train strategy
            if hasattr(strategy, 'train'):
                await strategy.train(train_data)
            
            # Test strategy
            result = await self.run_backtest(strategy, test_data)
            result.metadata['walk_forward'] = {
                'train_start': start_idx,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            }
            
            results.append(result)
            
            # Move to next period
            start_idx += test_period - overlap
        
        return results
    
    async def run_monte_carlo_simulation(self, strategy: BaseStrategyInterface,
                                        data: pd.DataFrame,
                                        n_simulations: int = 1000,
                                        confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for robustness testing."""
        base_result = await self.run_backtest(strategy, data)
        base_returns = self._calculate_returns_from_trades(base_result.trades)
        
        simulation_results = []
        
        for i in range(n_simulations):
            # Resample returns with replacement
            resampled_returns = np.random.choice(base_returns, size=len(base_returns), replace=True)
            
            # Calculate equity curve
            equity = self.config.initial_capital
            equity_curve = [equity]
            
            for ret in resampled_returns:
                equity *= (1 + ret)
                equity_curve.append(equity)
            
            # Calculate metrics
            total_return = (equity / self.config.initial_capital - 1)
            sharpe = np.mean(resampled_returns) / np.std(resampled_returns) * np.sqrt(252) if np.std(resampled_returns) > 0 else 0
            max_dd = self._calculate_max_drawdown_from_equity(equity_curve)
            
            simulation_results.append({
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'final_equity': equity
            })
        
        # Calculate statistics
        df_results = pd.DataFrame(simulation_results)
        
        monte_carlo_stats = {
            'n_simulations': n_simulations,
            'base_sharpe': base_result.performance_metrics['sharpe_ratio'],
            'percentiles': {}
        }
        
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'final_equity']:
            monte_carlo_stats['percentiles'][metric] = {}
            for level in confidence_levels:
                monte_carlo_stats['percentiles'][metric][f'p{int(level*100)}'] = df_results[metric].quantile(level)
        
        monte_carlo_stats['mean_metrics'] = df_results.mean().to_dict()
        monte_carlo_stats['std_metrics'] = df_results.std().to_dict()
        
        return monte_carlo_stats
    
    async def optimize_parameters(self, strategy_class: type,
                                 data: pd.DataFrame,
                                 param_grid: Dict[str, List[Any]],
                                 metric: str = 'sharpe_ratio',
                                 n_jobs: int = -1) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        # Run backtests in parallel
        if self.config.parallel_execution and len(param_combinations) > 1:
            # Parallel execution
            tasks = []
            for params in param_combinations:
                strategy = strategy_class(f"opt_{hash(str(params))}", params)
                tasks.append(self.run_backtest(strategy, data))
            
            results = await asyncio.gather(*tasks)
            
            for params, result in zip(param_combinations, results):
                score = result.performance_metrics.get(metric, -np.inf)
                all_results.append({
                    'params': params,
                    'score': score,
                    'metrics': result.performance_metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
        else:
            # Sequential execution
            for params in param_combinations:
                strategy = strategy_class(f"opt_{hash(str(params))}", params)
                result = await self.run_backtest(strategy, data)
                
                score = result.performance_metrics.get(metric, -np.inf)
                all_results.append({
                    'params': params,
                    'score': score,
                    'metrics': result.performance_metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': sorted(all_results, key=lambda x: x['score'], reverse=True),
            'optimization_metric': metric
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        return combinations
    
    async def _process_signal(self, signal: TradingSignal, timestamp: datetime, row: pd.Series) -> None:
        """Process trading signal and create orders."""
        instrument = row.get('instrument', 'default')
        
        if signal.signal_type in [SignalType.BUY, SignalType.SCALE_IN]:
            await self._open_long_position(instrument, timestamp, row, signal)
        elif signal.signal_type in [SignalType.SELL, SignalType.SCALE_OUT]:
            if self.config.enable_shorting:
                await self._open_short_position(instrument, timestamp, row, signal)
            else:
                await self._close_position(instrument, timestamp, row, 'signal')
        elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            await self._close_position(instrument, timestamp, row, 'signal')
    
    async def _open_long_position(self, instrument: str, timestamp: datetime, 
                                 row: pd.Series, signal: TradingSignal) -> None:
        """Open a long position."""
        if instrument in self.current_positions:
            # Already have position - check if scaling in
            if signal.signal_type != SignalType.SCALE_IN:
                return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, row)
        
        # Calculate costs
        entry_price = row['close'] * (1 + self.config.slippage)
        commission = position_size * entry_price * self.config.commission
        
        # Create trade
        trade = Trade(
            trade_id=f"{instrument}_{timestamp}",
            instrument=instrument,
            side='buy',
            entry_time=timestamp,
            entry_price=entry_price,
            quantity=position_size,
            commission=commission,
            slippage=self.config.slippage * entry_price,
            metadata={'signal': signal}
        )
        
        # Add to open trades
        self.open_trades[trade.trade_id] = trade
        self.current_positions[instrument] = trade
    
    async def _open_short_position(self, instrument: str, timestamp: datetime,
                                  row: pd.Series, signal: TradingSignal) -> None:
        """Open a short position."""
        if not self.config.enable_shorting:
            return
        
        if instrument in self.current_positions:
            if signal.signal_type != SignalType.SCALE_IN:
                return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, row)
        
        # Calculate costs
        entry_price = row['close'] * (1 - self.config.slippage)
        commission = position_size * entry_price * self.config.commission
        
        # Create trade
        trade = Trade(
            trade_id=f"{instrument}_{timestamp}_short",
            instrument=instrument,
            side='sell',
            entry_time=timestamp,
            entry_price=entry_price,
            quantity=-position_size,  # Negative for short
            commission=commission,
            slippage=self.config.slippage * entry_price,
            metadata={'signal': signal}
        )
        
        self.open_trades[trade.trade_id] = trade
        self.current_positions[instrument] = trade
    
    async def _close_position(self, instrument: str, timestamp: datetime,
                             row: pd.Series, reason: str) -> None:
        """Close an existing position."""
        if instrument not in self.current_positions:
            return
        
        trade = self.current_positions[instrument]
        
        # Calculate exit price
        if trade.side == 'buy':
            exit_price = row['close'] * (1 - self.config.slippage)
        else:
            exit_price = row['close'] * (1 + self.config.slippage)
        
        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.duration = timestamp - trade.entry_time
        
        # Calculate P&L
        if trade.side == 'buy':
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity - trade.commission * 2
            trade.return_pct = (exit_price / trade.entry_price - 1) - (trade.commission * 2 / (trade.entry_price * trade.quantity))
        else:
            trade.pnl = (trade.entry_price - exit_price) * abs(trade.quantity) - trade.commission * 2
            trade.return_pct = (trade.entry_price / exit_price - 1) - (trade.commission * 2 / (trade.entry_price * abs(trade.quantity)))
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_trades[trade.trade_id]
        del self.current_positions[instrument]
    
    def _update_positions(self, timestamp: datetime, row: pd.Series) -> None:
        """Update open positions with current prices."""
        for instrument, trade in self.current_positions.items():
            current_price = row.get('close', trade.entry_price)
            
            # Calculate unrealized P&L
            if trade.side == 'buy':
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl = (trade.entry_price - current_price) * abs(trade.quantity)
            
            trade.metadata['unrealized_pnl'] = unrealized_pnl
            trade.metadata['current_price'] = current_price
    
    def _check_stops(self, timestamp: datetime, row: pd.Series) -> None:
        """Check stop loss and take profit levels."""
        positions_to_close = []
        
        for instrument, trade in self.current_positions.items():
            current_price = row.get('close', trade.entry_price)
            
            if trade.side == 'buy':
                # Check stop loss
                if current_price <= trade.entry_price * (1 - self.config.stop_loss):
                    positions_to_close.append((instrument, 'stop_loss'))
                # Check take profit
                elif current_price >= trade.entry_price * (1 + self.config.take_profit):
                    positions_to_close.append((instrument, 'take_profit'))
            else:
                # Short position stops
                if current_price >= trade.entry_price * (1 + self.config.stop_loss):
                    positions_to_close.append((instrument, 'stop_loss'))
                elif current_price <= trade.entry_price * (1 - self.config.take_profit):
                    positions_to_close.append((instrument, 'take_profit'))
        
        # Close positions that hit stops
        for instrument, reason in positions_to_close:
            asyncio.create_task(self._close_position(instrument, timestamp, row, reason))
    
    def _calculate_position_size(self, signal: TradingSignal, row: pd.Series) -> float:
        """Calculate position size based on config."""
        available_capital = self._calculate_cash()
        
        if self.config.position_sizing == 'equal':
            # Equal weight across max positions
            position_value = available_capital / self.config.max_positions
        elif self.config.position_sizing == 'kelly':
            # Kelly criterion (simplified)
            kelly_fraction = signal.metadata.get('kelly_fraction', 0.25)
            position_value = available_capital * kelly_fraction
        elif self.config.position_sizing == 'risk_parity':
            # Risk parity (simplified)
            volatility = signal.metadata.get('volatility', 0.02)
            target_risk = 0.01  # 1% risk per position
            position_value = available_capital * (target_risk / volatility)
        else:
            position_value = available_capital / self.config.max_positions
        
        # Convert to shares
        price = row.get('close', 1)
        position_size = position_value / price
        
        return max(0, position_size)
    
    def _calculate_equity(self, row: pd.Series) -> float:
        """Calculate total equity."""
        cash = self._calculate_cash()
        positions_value = self._calculate_positions_value(row)
        return cash + positions_value
    
    def _calculate_cash(self) -> float:
        """Calculate available cash."""
        cash = self.config.initial_capital
        
        # Subtract invested capital
        for trade in self.open_trades.values():
            if trade.side == 'buy':
                cash -= trade.entry_price * trade.quantity + trade.commission
            else:
                # Short position - add proceeds
                cash += trade.entry_price * abs(trade.quantity) - trade.commission
        
        # Add/subtract closed trade P&L
        for trade in self.closed_trades:
            cash += trade.pnl
        
        return cash
    
    def _calculate_positions_value(self, row: pd.Series) -> float:
        """Calculate current value of all positions."""
        total_value = 0
        
        for instrument, trade in self.current_positions.items():
            current_price = row.get('close', trade.entry_price)
            
            if trade.side == 'buy':
                total_value += current_price * trade.quantity
            else:
                # Short position value (negative)
                total_value += (2 * trade.entry_price - current_price) * abs(trade.quantity)
        
        return total_value
    
    def _should_rebalance(self, timestamp: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        if not self.equity_curve:
            return False
        
        last_rebalance = self.equity_curve[0]['timestamp']
        
        if self.config.rebalance_frequency == 'daily':
            return (timestamp - last_rebalance).days >= 1
        elif self.config.rebalance_frequency == 'weekly':
            return (timestamp - last_rebalance).days >= 7
        elif self.config.rebalance_frequency == 'monthly':
            return (timestamp - last_rebalance).days >= 30
        
        return False
    
    async def _rebalance_portfolio(self, strategy: BaseStrategyInterface,
                                  timestamp: datetime, row: pd.Series) -> None:
        """Rebalance portfolio positions."""
        # This would implement portfolio rebalancing logic
        pass
    
    def _close_all_positions(self, timestamp: datetime, row: pd.Series) -> None:
        """Close all remaining open positions."""
        instruments_to_close = list(self.current_positions.keys())
        
        for instrument in instruments_to_close:
            asyncio.create_task(self._close_position(instrument, timestamp, row, 'end_of_backtest'))
    
    def _calculate_results(self, strategy_id: str, start_time: datetime) -> BacktestResult:
        """Calculate backtest results and metrics."""
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['returns'].dropna()
        
        # Calculate performance metrics
        metrics = PerformanceAnalyzer.calculate_metrics(
            equity_curve=equity_df['equity'].values,
            trades=self.closed_trades,
            daily_returns=daily_returns.values,
            config=self.config
        )
        
        # Monthly returns
        monthly_returns = equity_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create positions DataFrame
        positions_data = []
        for trade in self.closed_trades:
            positions_data.append({
                'instrument': trade.instrument,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'side': trade.side,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct
            })
        
        positions_df = pd.DataFrame(positions_data) if positions_data else pd.DataFrame()
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return BacktestResult(
            strategy_id=strategy_id,
            config=self.config,
            trades=self.closed_trades,
            equity_curve=equity_df,
            performance_metrics=metrics,
            monthly_returns=monthly_returns,
            daily_returns=daily_returns,
            positions=positions_df,
            signals=self.signals,
            execution_time=execution_time,
            metadata={
                'n_signals': len(self.signals),
                'n_trades': len(self.closed_trades)
            }
        )
    
    def _prepare_market_data(self, row: pd.Series) -> Any:
        """Prepare market data for strategy."""
        # This would convert pandas row to appropriate market data format
        # Placeholder for now
        return row
    
    def _reset_state(self) -> None:
        """Reset engine state for new backtest."""
        self.current_positions = {}
        self.open_trades = {}
        self.closed_trades = []
        self.equity_curve = []
        self.signals = []
    
    def _calculate_returns_from_trades(self, trades: List[Trade]) -> np.ndarray:
        """Calculate returns array from trades."""
        returns = []
        for trade in trades:
            if trade.return_pct is not None:
                returns.append(trade.return_pct)
        return np.array(returns)
    
    def _calculate_max_drawdown_from_equity(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if self.executor:
            self.executor.shutdown(wait=False)


class PerformanceAnalyzer:
    """
    Analyzes backtest performance and calculates comprehensive metrics.
    """
    
    @staticmethod
    def calculate_metrics(equity_curve: np.ndarray, trades: List[Trade],
                         daily_returns: np.ndarray, config: BacktestConfig) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 0 else 0
        metrics['total_trades'] = len(trades)
        
        # Win/Loss metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        metrics['winning_trades'] = len(winning_trades)
        metrics['losing_trades'] = len(losing_trades)
        metrics['win_rate'] = len(winning_trades) / len(trades) if trades else 0
        
        # Return metrics
        if len(daily_returns) > 0:
            metrics['annual_return'] = np.mean(daily_returns) * 252
            metrics['volatility'] = np.std(daily_returns) * np.sqrt(252)
            metrics['sharpe_ratio'] = (metrics['annual_return'] - config.risk_free_rate) / metrics['volatility'] if metrics['volatility'] > 0 else 0
            metrics['sortino_ratio'] = PerformanceAnalyzer._calculate_sortino(daily_returns, config.risk_free_rate)
            metrics['calmar_ratio'] = PerformanceAnalyzer._calculate_calmar(equity_curve, metrics['annual_return'])
        
        # Risk metrics
        metrics['max_drawdown'] = PerformanceAnalyzer._calculate_max_drawdown(equity_curve)
        metrics['max_drawdown_duration'] = PerformanceAnalyzer._calculate_max_dd_duration(equity_curve)
        metrics['var_95'] = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        metrics['cvar_95'] = np.mean(daily_returns[daily_returns <= metrics['var_95']]) if len(daily_returns) > 0 else 0
        
        # Trade analysis
        if trades:
            trade_returns = [t.return_pct for t in trades if t.return_pct is not None]
            metrics['avg_trade_return'] = np.mean(trade_returns) if trade_returns else 0
            metrics['trade_return_std'] = np.std(trade_returns) if trade_returns else 0
            
            if winning_trades:
                metrics['avg_win'] = np.mean([t.pnl for t in winning_trades])
                metrics['avg_win_return'] = np.mean([t.return_pct for t in winning_trades if t.return_pct is not None])
                metrics['largest_win'] = max(t.pnl for t in winning_trades)
            
            if losing_trades:
                metrics['avg_loss'] = np.mean([t.pnl for t in losing_trades])
                metrics['avg_loss_return'] = np.mean([t.return_pct for t in losing_trades if t.return_pct is not None])
                metrics['largest_loss'] = min(t.pnl for t in losing_trades)
            
            # Profit factor
            if losing_trades and winning_trades:
                gross_profit = sum(t.pnl for t in winning_trades)
                gross_loss = abs(sum(t.pnl for t in losing_trades))
                metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf
            else:
                metrics['profit_factor'] = 0
            
            # Expectancy
            metrics['expectancy'] = metrics['win_rate'] * metrics.get('avg_win', 0) - (1 - metrics['win_rate']) * abs(metrics.get('avg_loss', 0))
            
            # Trade duration
            durations = [t.duration.total_seconds() / 3600 for t in trades if t.duration]  # Hours
            if durations:
                metrics['avg_trade_duration_hours'] = np.mean(durations)
                metrics['median_trade_duration_hours'] = np.median(durations)
        
        # Recovery metrics
        metrics['recovery_factor'] = metrics['total_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
        metrics['profit_to_max_dd'] = (equity_curve[-1] - equity_curve[0]) / (metrics['max_drawdown'] * equity_curve[0]) if metrics['max_drawdown'] > 0 else 0
        
        # Consistency metrics
        if len(daily_returns) > 20:
            metrics['downside_deviation'] = np.std(daily_returns[daily_returns < 0]) * np.sqrt(252) if len(daily_returns[daily_returns < 0]) > 0 else 0
            metrics['upside_deviation'] = np.std(daily_returns[daily_returns > 0]) * np.sqrt(252) if len(daily_returns[daily_returns > 0]) > 0 else 0
            
            # Rolling metrics
            rolling_returns = pd.Series(daily_returns).rolling(20).mean()
            metrics['return_consistency'] = len(rolling_returns[rolling_returns > 0]) / len(rolling_returns.dropna()) if len(rolling_returns.dropna()) > 0 else 0
        
        return metrics
    
    @staticmethod
    def _calculate_sortino(returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0
        
        downside_std = np.std(downside_returns) * np.sqrt(252)
        excess_return = np.mean(returns) * 252 - risk_free_rate
        
        return excess_return / downside_std if downside_std > 0 else 0
    
    @staticmethod
    def _calculate_calmar(equity_curve: np.ndarray, annual_return: float) -> float:
        """Calculate Calmar ratio."""
        max_dd = PerformanceAnalyzer._calculate_max_drawdown(equity_curve)
        return annual_return / max_dd if max_dd > 0 else 0
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) == 0:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    @staticmethod
    def _calculate_max_dd_duration(equity_curve: np.ndarray) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(equity_curve) == 0:
            return 0
        
        peak = equity_curve[0]
        max_duration = 0
        current_duration = 0
        
        for value in equity_curve:
            if value >= peak:
                peak = value
                current_duration = 0
            else:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
        
        return max_duration


class WalkForwardValidator:
    """
    Performs walk-forward validation for strategy robustness testing.
    """
    
    def __init__(self, engine: BacktestEngine):
        self.engine = engine
        self.validation_results = []
    
    async def validate(self, strategy_class: type, data: pd.DataFrame,
                      optimization_params: Dict[str, List[Any]],
                      train_periods: int = 252,
                      test_periods: int = 63,
                      optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Perform walk-forward validation with parameter optimization."""
        results = {
            'periods': [],
            'in_sample_performance': [],
            'out_sample_performance': [],
            'parameter_stability': {},
            'performance_degradation': []
        }
        
        # Track parameter choices across periods
        parameter_history = defaultdict(list)
        
        # Walk through data
        periods = self._generate_walk_forward_periods(len(data), train_periods, test_periods)
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            # Training data
            train_data = data.iloc[train_start:train_end]
            
            # Optimize parameters on training data
            optimization_result = await self.engine.optimize_parameters(
                strategy_class,
                train_data,
                optimization_params,
                metric=optimization_metric
            )
            
            best_params = optimization_result['best_params']
            in_sample_score = optimization_result['best_score']
            
            # Track parameters
            for param, value in best_params.items():
                parameter_history[param].append(value)
            
            # Test on out-of-sample data
            test_data = data.iloc[test_start:test_end]
            strategy = strategy_class(f"wf_test_{i}", best_params)
            test_result = await self.engine.run_backtest(strategy, test_data)
            
            out_sample_score = test_result.performance_metrics.get(optimization_metric, 0)
            
            # Record results
            results['periods'].append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            results['in_sample_performance'].append(in_sample_score)
            results['out_sample_performance'].append(out_sample_score)
            results['performance_degradation'].append(
                (in_sample_score - out_sample_score) / in_sample_score if in_sample_score != 0 else 0
            )
        
        # Calculate parameter stability
        for param, values in parameter_history.items():
            results['parameter_stability'][param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                'changes': sum(1 for i in range(1, len(values)) if values[i] != values[i-1])
            }
        
        # Summary statistics
        results['summary'] = {
            'avg_in_sample': np.mean(results['in_sample_performance']),
            'avg_out_sample': np.mean(results['out_sample_performance']),
            'avg_degradation': np.mean(results['performance_degradation']),
            'consistency': np.corrcoef(
                results['in_sample_performance'],
                results['out_sample_performance']
            )[0, 1] if len(results['in_sample_performance']) > 1 else 0
        }
        
        return results
    
    def _generate_walk_forward_periods(self, data_length: int, train_size: int,
                                      test_size: int) -> List[Tuple[int, int, int, int]]:
        """Generate train/test period indices."""
        periods = []
        current_pos = 0
        
        while current_pos + train_size + test_size <= data_length:
            train_start = current_pos
            train_end = current_pos + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            periods.append((train_start, train_end, test_start, test_end))
            
            # Move forward by test period
            current_pos += test_size
        
        return periods