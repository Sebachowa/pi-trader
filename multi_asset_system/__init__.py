
"""
Multi-Asset Trading System

A unified framework for trading across multiple asset classes including:
- Cryptocurrencies
- Stocks/Equities
- Foreign Exchange (Forex)
- Commodities and Futures

This system provides a consistent interface for asset management, risk control,
and market operations across different asset classes.
"""

from multi_asset_system.core.asset_interface import Asset, AssetClass
from multi_asset_system.core.asset_manager import MultiAssetManager
from multi_asset_system.core.unified_trader import UnifiedTrader

__version__ = "1.0.0"

__all__ = [
    "Asset",
    "AssetClass",
    "MultiAssetManager",
    "UnifiedTrader",
]