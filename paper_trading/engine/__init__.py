
"""
Paper trading engine with realistic market simulation.
"""

from paper_trading.engine.core import PaperTradingEngine
from paper_trading.engine.execution import OrderExecutor
from paper_trading.engine.market_impact import MarketImpactModel
from paper_trading.engine.slippage import SlippageModel
from paper_trading.engine.accounts import PaperTradingAccount

__all__ = [
    "PaperTradingEngine",
    "OrderExecutor",
    "MarketImpactModel",
    "SlippageModel",
    "PaperTradingAccount",
]