#!/usr/bin/env python3

"""
Example usage of the comprehensive paper trading engine.
"""

import asyncio
from decimal import Decimal
from datetime import datetime

from nautilus_trader.common.clock import TestClock
from nautilus_trader.common.logging import Logger, LogLevel
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, TraderId, ClientOrderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders import MarketOrder, LimitOrder
from nautilus_trader.msgbus.bus import MessageBus

from paper_trading.engine.core import PaperTradingEngine
from paper_trading.engine.config import PaperTradingConfigs


async def example_crypto_trading():
    """Example of paper trading crypto with realistic simulation."""
    print("=== Crypto Paper Trading Example ===\n")
    
    # Setup components
    trader_id = TraderId("PAPER-TRADER-001")
    clock = TestClock()
    logger = Logger(name="PaperTradingExample", level=LogLevel.INFO)
    msgbus = MessageBus(trader_id, clock, logger)
    
    # Create paper trading engine with crypto config
    config = PaperTradingConfigs.realistic_crypto()
    engine = PaperTradingEngine(trader_id, msgbus, clock, logger, config)
    
    # Start engine
    await engine.start()
    
    # Create instrument
    btc_instrument = InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))
    
    # Simulate market data
    quote = QuoteTick(
        instrument_id=btc_instrument,
        bid_price=Price.from_str("50000.00"),
        ask_price=Price.from_str("50010.00"),
        bid_size=Quantity.from_str("2.5"),
        ask_size=Quantity.from_str("2.5"),
        ts_event=clock.timestamp_ns(),
        ts_init=clock.timestamp_ns(),
    )
    engine.update_market_data(quote)
    
    print(f"Market: BTC/USDT")
    print(f"Bid: ${quote.bid_price} x {quote.bid_size}")
    print(f"Ask: ${quote.ask_price} x {quote.ask_size}")
    print(f"Spread: {float(quote.ask_price) - float(quote.bid_price):.2f} ({((float(quote.ask_price) - float(quote.bid_price)) / float(quote.bid_price) * 10000):.1f} bps)\n")
    
    # Create and submit market order
    print("1. Submitting market buy order...")
    market_order = MarketOrder(
        trader_id=trader_id,
        strategy_id=trader_id.get_strategy_id("TEST"),
        instrument_id=btc_instrument,
        client_order_id=ClientOrderId(f"MKT-{UUID4()}"),
        order_side=OrderSide.BUY,
        quantity=Quantity.from_str("0.1"),
        init_id=UUID4(),
        ts_init=clock.timestamp_ns(),
    )
    
    engine.submit_order(market_order)
    await asyncio.sleep(0.2)  # Wait for execution
    
    # Check execution
    fills = engine.fills
    if fills:
        fill = fills[-1]
        print(f"✓ Market order filled:")
        print(f"  - Quantity: {fill.quantity}")
        print(f"  - Price: ${fill.price}")
        print(f"  - Commission: ${fill.commission:.2f}")
        print(f"  - Slippage: {(float(fill.price) - float(quote.ask_price)) / float(quote.ask_price) * 10000:.1f} bps\n")
    
    # Submit limit order
    print("2. Submitting limit buy order below market...")
    limit_order = LimitOrder(
        trader_id=trader_id,
        strategy_id=trader_id.get_strategy_id("TEST"),
        instrument_id=btc_instrument,
        client_order_id=ClientOrderId(f"LMT-{UUID4()}"),
        order_side=OrderSide.BUY,
        quantity=Quantity.from_str("0.05"),
        price=Price.from_str("49900.00"),  # Below market
        init_id=UUID4(),
        ts_init=clock.timestamp_ns(),
    )
    
    engine.submit_order(limit_order)
    print(f"✓ Limit order submitted at ${limit_order.price}\n")
    
    # Simulate price movement
    print("3. Simulating market movement...")
    await asyncio.sleep(0.1)
    
    # Price drops - limit order should fill
    new_quote = QuoteTick(
        instrument_id=btc_instrument,
        bid_price=Price.from_str("49895.00"),
        ask_price=Price.from_str("49905.00"),
        bid_size=Quantity.from_str("3.0"),
        ask_size=Quantity.from_str("3.0"),
        ts_event=clock.timestamp_ns(),
        ts_init=clock.timestamp_ns(),
    )
    engine.update_market_data(new_quote)
    print(f"Market moved: Bid ${new_quote.bid_price} / Ask ${new_quote.ask_price}")
    
    await asyncio.sleep(2)  # Wait for limit order to fill
    
    # Check if limit order filled
    if len(engine.fills) > 1:
        fill = engine.fills[-1]
        print(f"✓ Limit order filled:")
        print(f"  - Quantity: {fill.quantity}")
        print(f"  - Price: ${fill.price}")
        print(f"  - Price improvement: ${float(limit_order.price) - float(fill.price):.2f}\n")
    
    # Show account summary
    print("4. Account Summary:")
    summary = engine.get_performance_summary()
    account = summary["account"]
    trading = summary["trading"]
    costs = summary["costs"]
    
    print(f"Balance: ${account['balance']:,.2f}")
    print(f"Equity: ${account['equity']:,.2f}")
    print(f"Unrealized PnL: ${account['unrealized_pnl']:,.2f}")
    print(f"Realized PnL: ${account['realized_pnl']:,.2f}")
    print(f"\nTrading Stats:")
    print(f"Total Orders: {trading['total_orders']}")
    print(f"Filled Orders: {trading['filled_orders']}")
    print(f"\nExecution Costs:")
    print(f"Total Slippage: ${costs['total_slippage']:.2f}")
    print(f"Total Spread Cost: ${costs['total_spread_cost']:.2f}")
    print(f"Avg Slippage/Trade: ${costs['avg_slippage_per_trade']:.2f}")
    
    # Stop engine
    await engine.stop()


async def example_partial_fills():
    """Example demonstrating partial fill simulation."""
    print("\n=== Partial Fill Example ===\n")
    
    # Setup
    trader_id = TraderId("PAPER-TRADER-002")
    clock = TestClock()
    logger = Logger(name="PartialFillExample", level=LogLevel.INFO)
    msgbus = MessageBus(trader_id, clock, logger)
    
    # Create engine with partial fills enabled
    config = PaperTradingConfigs.realistic_equities()
    config.enable_partial_fills = True
    engine = PaperTradingEngine(trader_id, msgbus, clock, logger, config)
    
    await engine.start()
    
    # Create instrument
    aapl_instrument = InstrumentId(Symbol("AAPL"), Venue("NASDAQ"))
    
    # Market data
    quote = QuoteTick(
        instrument_id=aapl_instrument,
        bid_price=Price.from_str("175.00"),
        ask_price=Price.from_str("175.05"),
        bid_size=Quantity.from_str("100"),
        ask_size=Quantity.from_str("100"),
        ts_event=clock.timestamp_ns(),
        ts_init=clock.timestamp_ns(),
    )
    engine.update_market_data(quote)
    
    # Submit large order that will be partially filled
    print("Submitting large market order (1000 shares)...")
    large_order = MarketOrder(
        trader_id=trader_id,
        strategy_id=trader_id.get_strategy_id("TEST"),
        instrument_id=aapl_instrument,
        client_order_id=ClientOrderId(f"LARGE-{UUID4()}"),
        order_side=OrderSide.BUY,
        quantity=Quantity.from_str("1000"),  # Large order
        init_id=UUID4(),
        ts_init=clock.timestamp_ns(),
    )
    
    engine.submit_order(large_order)
    
    # Wait for partial fills
    await asyncio.sleep(3)
    
    # Show fills
    print(f"\nOrder filled in {len([f for f in engine.fills if f.client_order_id == str(large_order.client_order_id)])} parts:")
    total_qty = 0
    total_cost = 0
    for i, fill in enumerate(engine.fills):
        if fill.client_order_id == str(large_order.client_order_id):
            print(f"  Fill {i+1}: {fill.quantity} @ ${fill.price}")
            total_qty += float(fill.quantity)
            total_cost += float(fill.quantity) * float(fill.price)
    
    if total_qty > 0:
        avg_price = total_cost / total_qty
        print(f"\nTotal: {total_qty} shares @ ${avg_price:.2f} average")
    
    await engine.stop()


async def example_market_impact():
    """Example showing market impact on large orders."""
    print("\n=== Market Impact Example ===\n")
    
    # Setup
    trader_id = TraderId("PAPER-TRADER-003")
    clock = TestClock()
    logger = Logger(name="MarketImpactExample", level=LogLevel.INFO)
    msgbus = MessageBus(trader_id, clock, logger)
    
    config = PaperTradingConfigs.realistic_equities()
    engine = PaperTradingEngine(trader_id, msgbus, clock, logger, config)
    
    await engine.start()
    
    # Create instrument
    spy_instrument = InstrumentId(Symbol("SPY"), Venue("NYSE"))
    
    # Market data with depth
    quote = QuoteTick(
        instrument_id=spy_instrument,
        bid_price=Price.from_str("450.00"),
        ask_price=Price.from_str("450.02"),
        bid_size=Quantity.from_str("500"),
        ask_size=Quantity.from_str("500"),
        ts_event=clock.timestamp_ns(),
        ts_init=clock.timestamp_ns(),
    )
    engine.update_market_data(quote)
    
    print("Market Impact Analysis for SPY:")
    print(f"Current market: ${quote.bid_price} / ${quote.ask_price}\n")
    
    # Test different order sizes
    sizes = [100, 500, 1000, 5000, 10000]
    
    for size in sizes:
        # Calculate execution price with impact
        exec_price = engine.get_execution_price(
            spy_instrument,
            OrderSide.BUY,
            Quantity.from_str(str(size)),
            OrderType.MARKET,
        )
        
        if exec_price:
            impact_bps = (float(exec_price) - float(quote.ask_price)) / float(quote.ask_price) * 10000
            total_cost = float(exec_price) * size
            print(f"Order size: {size:,} shares")
            print(f"  Expected price: ${exec_price}")
            print(f"  Market impact: {impact_bps:.1f} bps")
            print(f"  Total cost: ${total_cost:,.2f}\n")
    
    await engine.stop()


async def main():
    """Run all examples."""
    await example_crypto_trading()
    await example_partial_fills()
    await example_market_impact()


if __name__ == "__main__":
    asyncio.run(main())