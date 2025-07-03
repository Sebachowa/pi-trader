#!/usr/bin/env python3

"""
Multi-asset trading system demonstration.

This script shows how to:
1. Set up the multi-asset manager
2. Register different asset types
3. Perform risk-aware trading across asset classes
4. Monitor portfolio metrics
"""

import asyncio
import json
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

from nautilus_trader.common.logging import LoggerAdapter, LogLevel
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId, StrategyId, TraderId, Venue
from nautilus_trader.model.objects import Price, Quantity

# Import multi-asset system components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from multi_asset_system.core.asset_interface import AssetClass
from multi_asset_system.core.asset_manager import MultiAssetManager
from multi_asset_system.core.unified_trader import UnifiedTrader
from multi_asset_system.risk.portfolio_risk_manager import PortfolioRiskManager
from multi_asset_system.markets.execution_router import ExecutionRouter
from multi_asset_system.assets.crypto_asset import CryptoAsset
from multi_asset_system.assets.equity_asset import EquityAsset
from multi_asset_system.assets.forex_asset import ForexAsset
from multi_asset_system.assets.commodity_asset import CommodityAsset


class MultiAssetTradingDemo:
    """Demonstration of multi-asset trading capabilities."""
    
    def __init__(self):
        # Setup logging
        self.logger = LoggerAdapter("MultiAssetDemo", LogLevel.INFO)
        
        # Initialize components
        self.asset_manager = MultiAssetManager(logger=self.logger)
        self.risk_manager = PortfolioRiskManager(
            asset_manager=self.asset_manager,
            logger=self.logger,
            initial_capital=Decimal("100000"),
        )
        self.execution_router = ExecutionRouter(logger=self.logger)
        
        # Initialize trader
        self.trader = UnifiedTrader(
            trader_id=TraderId("DEMO-001"),
            strategy_id=StrategyId("MULTI-ASSET"),
            asset_manager=self.asset_manager,
            risk_manager=self.risk_manager,
            execution_router=self.execution_router,
            logger=self.logger,
        )
    
    def setup_assets(self):
        """Register various asset types."""
        print("\n=== Setting Up Multi-Asset Universe ===")
        
        # Register cryptocurrencies
        btc = self.asset_manager.create_and_register_asset(
            instrument_id=InstrumentId(symbol="BTCUSDT", venue=Venue("BINANCE")),
            asset_class=AssetClass.CRYPTO,
            config={
                "base_currency": "BTC",
                "quote_currency": "USDT",
                "exchange_type": "SPOT",
            }
        )
        print(f"✓ Registered: {btc.instrument_id} ({btc.asset_class.value})")
        
        eth = self.asset_manager.create_and_register_asset(
            instrument_id=InstrumentId(symbol="ETHUSDT", venue=Venue("BINANCE")),
            asset_class=AssetClass.CRYPTO,
            config={
                "base_currency": "ETH",
                "quote_currency": "USDT",
                "exchange_type": "SPOT",
            }
        )
        print(f"✓ Registered: {eth.instrument_id} ({eth.asset_class.value})")
        
        # Register equities
        aapl = self.asset_manager.create_and_register_asset(
            instrument_id=InstrumentId(symbol="AAPL", venue=Venue("NASDAQ")),
            asset_class=AssetClass.EQUITY,
            config={
                "currency": "USD",
                "exchange": "NASDAQ",
                "sector": "Technology",
                "market_cap": "3000000000000",
            }
        )
        print(f"✓ Registered: {aapl.instrument_id} ({aapl.asset_class.value})")
        
        spy = self.asset_manager.create_and_register_asset(
            instrument_id=InstrumentId(symbol="SPY", venue=Venue("NYSE")),
            asset_class=AssetClass.ETF,
            config={
                "currency": "USD",
                "exchange": "NYSE",
                "is_etf": True,
            }
        )
        print(f"✓ Registered: {spy.instrument_id} ({spy.asset_class.value})")
        
        # Register forex pairs
        eurusd = self.asset_manager.create_and_register_asset(
            instrument_id=InstrumentId(symbol="EUR/USD", venue=Venue("LMAX")),
            asset_class=AssetClass.FOREX,
            config={
                "base_currency": "EUR",
                "quote_currency": "USD",
                "pair_type": "MAJOR",
            }
        )
        print(f"✓ Registered: {eurusd.instrument_id} ({eurusd.asset_class.value})")
        
        # Register commodities
        gold = self.asset_manager.create_and_register_asset(
            instrument_id=InstrumentId(symbol="GC", venue=Venue("COMEX")),
            asset_class=AssetClass.COMMODITY,
            config={
                "commodity_type": "METAL",
                "contract_size": "100",
                "contract_unit": "troy ounces",
                "tick_value": "10",
                "expiry_date": "2025-08-28",
                "exchange": "COMEX",
            }
        )
        print(f"✓ Registered: {gold.instrument_id} ({gold.asset_class.value})")
        
        # Update prices (in real system would come from market data)
        btc.update_price(Price(50000), datetime.utcnow())
        eth.update_price(Price(3000), datetime.utcnow())
        aapl.update_price(Price(180), datetime.utcnow())
        spy.update_price(Price(450), datetime.utcnow())
        eurusd.update_price(Price(1.0850), datetime.utcnow())
        gold.update_price(Price(2050), datetime.utcnow())
    
    def display_asset_info(self):
        """Display information about registered assets."""
        print("\n=== Asset Information ===")
        
        stats = self.asset_manager.get_summary_statistics()
        print(f"\nTotal Assets: {stats['total_assets']}")
        print("\nBy Asset Class:")
        for asset_class, count in stats['by_class'].items():
            print(f"  {asset_class}: {count}")
        
        print("\nAsset Details:")
        for asset_class in AssetClass:
            assets = self.asset_manager.get_assets_by_class(asset_class)
            if assets:
                print(f"\n{asset_class.value}:")
                for asset in assets:
                    print(f"  • {asset.instrument_id}")
                    print(f"    - Trading Rules: min={asset.trading_rules.min_order_size}, "
                          f"max={asset.trading_rules.max_order_size}, "
                          f"leverage={asset.trading_rules.max_leverage}")
                    print(f"    - Risk Params: max_position={asset.risk_parameters.max_position_size}, "
                          f"concentration={asset.risk_parameters.concentration_limit:.1%}")
    
    async def demonstrate_risk_checks(self):
        """Demonstrate risk management across asset classes."""
        print("\n=== Risk Management Demonstration ===")
        
        # Test orders for different assets
        test_orders = [
            # Crypto order
            {
                "instrument": "BTCUSDT.BINANCE",
                "side": OrderSide.BUY,
                "quantity": Decimal("0.5"),  # 0.5 BTC
                "description": "Buy 0.5 BTC"
            },
            # Equity order
            {
                "instrument": "AAPL.NASDAQ",
                "side": OrderSide.BUY,
                "quantity": Decimal("100"),  # 100 shares
                "description": "Buy 100 AAPL shares"
            },
            # Large equity order (should trigger concentration limit)
            {
                "instrument": "SPY.NYSE",
                "side": OrderSide.BUY,
                "quantity": Decimal("500"),  # 500 shares
                "description": "Buy 500 SPY shares (large order)"
            },
            # Forex order
            {
                "instrument": "EUR/USD.LMAX",
                "side": OrderSide.BUY,
                "quantity": Decimal("2"),  # 2 standard lots
                "description": "Buy 2 lots EUR/USD"
            },
        ]
        
        for order_params in test_orders:
            print(f"\n→ Testing: {order_params['description']}")
            
            instrument_id = InstrumentId.from_str(order_params['instrument'])
            asset = self.asset_manager.get_asset(instrument_id)
            
            if not asset:
                print("  ✗ Asset not found")
                continue
            
            # Perform risk check
            risk_result = await self.risk_manager.check_order_risk(
                asset=asset,
                order_side=order_params['side'],
                quantity=order_params['quantity'],
                price=asset._last_price,
            )
            
            if risk_result.approved:
                print(f"  ✓ Risk check PASSED")
                if risk_result.adjusted_quantity:
                    print(f"    - Quantity adjusted: {order_params['quantity']} → {risk_result.adjusted_quantity}")
            else:
                print(f"  ✗ Risk check FAILED: {risk_result.reason}")
            
            if risk_result.warnings:
                for warning in risk_result.warnings:
                    print(f"    ⚠ {warning}")
    
    async def demonstrate_trading(self):
        """Demonstrate multi-asset trading."""
        print("\n=== Trading Demonstration ===")
        
        # Submit some orders
        orders = [
            # Crypto trade
            {
                "instrument": "BTCUSDT.BINANCE",
                "side": OrderSide.BUY,
                "quantity": Decimal("0.1"),
                "order_type": OrderType.MARKET,
                "description": "Market buy 0.1 BTC"
            },
            # Equity trade
            {
                "instrument": "AAPL.NASDAQ",
                "side": OrderSide.BUY,
                "quantity": Decimal("50"),
                "order_type": OrderType.LIMIT,
                "price": Price(179),
                "description": "Limit buy 50 AAPL @ $179"
            },
            # Forex trade
            {
                "instrument": "EUR/USD.LMAX",
                "side": OrderSide.SELL,
                "quantity": Decimal("1"),
                "order_type": OrderType.MARKET,
                "description": "Market sell 1 lot EUR/USD"
            },
        ]
        
        for order in orders:
            print(f"\n→ Submitting: {order['description']}")
            
            try:
                order_id = await self.trader.submit_order(
                    instrument_id=InstrumentId.from_str(order['instrument']),
                    order_side=order['side'],
                    quantity=order['quantity'],
                    order_type=order['order_type'],
                    price=order.get('price'),
                )
                
                if order_id:
                    print(f"  ✓ Order submitted: {order_id}")
                else:
                    print(f"  ✗ Order submission failed")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    def display_portfolio_metrics(self):
        """Display portfolio risk metrics."""
        print("\n=== Portfolio Risk Metrics ===")
        
        risk_summary = self.risk_manager.get_risk_summary()
        
        print(f"\nPortfolio Value: ${risk_summary['portfolio_value']:,.2f}")
        print(f"Total Exposure: ${risk_summary['total_exposure']:,.2f}")
        print(f"Leverage: {risk_summary['leverage']:.2f}x")
        print(f"Margin Utilization: {risk_summary['margin_utilization']:.1%}")
        
        print(f"\nRisk Metrics:")
        print(f"  Daily P&L: ${risk_summary['daily_pnl']:,.2f}")
        print(f"  Daily Return: {risk_summary['daily_return']:.2%}")
        print(f"  Current Drawdown: {risk_summary['current_drawdown']:.2%}")
        print(f"  VaR (95%): {risk_summary['var_95']:.2%}")
        print(f"  Sharpe Ratio: {risk_summary['sharpe_ratio']:.2f}")
        
        print(f"\nAsset Class Exposure:")
        for asset_class, exposure in risk_summary['asset_class_exposure'].items():
            print(f"  {asset_class}: ${exposure:,.2f}")
        
        if risk_summary['warnings']:
            print(f"\nRisk Warnings:")
            for warning in risk_summary['warnings']:
                print(f"  ⚠ {warning}")
    
    def demonstrate_market_hours(self):
        """Demonstrate market hours handling."""
        print("\n=== Market Hours Demonstration ===")
        
        current_time = datetime.utcnow()
        print(f"\nCurrent Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update market status
        self.asset_manager.update_all_market_status(current_time)
        
        # Check tradable assets by class
        for asset_class in [AssetClass.CRYPTO, AssetClass.EQUITY, AssetClass.FOREX]:
            tradable = self.asset_manager.get_tradable_assets(
                timestamp=current_time,
                asset_class=asset_class
            )
            
            print(f"\n{asset_class.value}:")
            if tradable:
                for asset in tradable:
                    print(f"  ✓ {asset.instrument_id} - OPEN")
            else:
                # Show closed markets
                all_assets = self.asset_manager.get_assets_by_class(asset_class)
                for asset in all_assets:
                    print(f"  ✗ {asset.instrument_id} - {asset._market_status.value}")
                    
    async def demonstrate_portfolio_rebalancing(self):
        """Demonstrate portfolio rebalancing."""
        print("\n=== Portfolio Rebalancing Demonstration ===")
        
        # Define target allocation
        target_weights = {
            AssetClass.CRYPTO: Decimal("0.20"),    # 20%
            AssetClass.EQUITY: Decimal("0.50"),    # 50%
            AssetClass.FOREX: Decimal("0.20"),     # 20%
            AssetClass.COMMODITY: Decimal("0.10"), # 10%
        }
        
        print("\nTarget Allocation:")
        for asset_class, weight in target_weights.items():
            print(f"  {asset_class.value}: {weight:.0%}")
        
        # Perform rebalancing
        orders_submitted = await self.trader.rebalance_portfolio(
            target_weights=target_weights,
            tolerance=Decimal("0.05")  # 5% tolerance
        )
        
        print(f"\nRebalancing orders submitted: {orders_submitted}")
    
    def save_configuration(self):
        """Save current configuration."""
        config_path = "multi_asset_config_demo.json"
        self.asset_manager.save_configuration(config_path)
        print(f"\n✓ Configuration saved to {config_path}")
    
    async def run_demo(self):
        """Run the complete demonstration."""
        print("=" * 60)
        print("MULTI-ASSET TRADING SYSTEM DEMONSTRATION")
        print("=" * 60)
        
        # Setup assets
        self.setup_assets()
        
        # Display asset information
        self.display_asset_info()
        
        # Demonstrate risk checks
        await self.demonstrate_risk_checks()
        
        # Demonstrate trading
        await self.demonstrate_trading()
        
        # Display portfolio metrics
        self.display_portfolio_metrics()
        
        # Demonstrate market hours
        self.demonstrate_market_hours()
        
        # Demonstrate rebalancing
        await self.demonstrate_portfolio_rebalancing()
        
        # Save configuration
        self.save_configuration()
        
        # Display performance summary
        print("\n=== Performance Summary ===")
        perf = self.trader.get_performance_summary()
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Open Positions: {perf['open_positions']}")
        print(f"Open Orders: {perf['open_orders']}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")


async def main():
    """Main entry point."""
    demo = MultiAssetTradingDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())