
"""
Bid/ask spread modeling for realistic paper trading execution.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity


@dataclass
class SpreadParams:
    """Parameters for spread modeling."""
    
    # Minimum spreads by asset class (in basis points)
    min_spreads: Dict[str, float] = None
    
    # Spread widening factors
    volatility_spread_multiplier: float = 2.0  # How much volatility widens spread
    size_spread_multiplier: float = 1.5  # How much large orders widen spread
    time_spread_multiplier: float = 1.8  # Multiplier for off-hours
    
    # Time-based spread adjustments
    market_hours_start: int = 9  # 9 AM
    market_hours_end: int = 16  # 4 PM
    off_hours_multiplier: float = 2.5
    
    # Cross-spread impact (for crosses vs direct quotes)
    cross_spread_multiplier: float = 1.3
    
    def __post_init__(self):
        if self.min_spreads is None:
            self.min_spreads = {
                "crypto": 5.0,    # 5 basis points minimum
                "forex": 1.0,     # 1 basis point for major pairs
                "equity": 2.0,    # 2 basis points
                "futures": 1.5,   # 1.5 basis points
                "options": 10.0,  # 10 basis points
            }


class SpreadModel:
    """
    Model for calculating realistic bid/ask spreads and associated costs.
    
    Considers:
    - Asset class characteristics
    - Current market conditions
    - Order size impact on spread
    - Time of day effects
    - Cross vs direct quotes
    """
    
    def __init__(self, params: Optional[SpreadParams] = None):
        self.params = params or SpreadParams()
        
        # Cache for spread statistics
        self._spread_cache: Dict[InstrumentId, Dict[str, float]] = {}
    
    def calculate_spread_cost(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        quote: Optional[QuoteTick] = None,
        volatility: Optional[float] = None,
        typical_spread_bps: Optional[float] = None,
    ) -> Decimal:
        """
        Calculate the spread cost for an order.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument being traded.
        side : OrderSide
            Buy or sell side.
        quantity : Quantity
            Order quantity.
        quote : QuoteTick, optional
            Current quote data.
        volatility : float, optional
            Current volatility for spread adjustment.
        typical_spread_bps : float, optional
            Typical spread in basis points if known.
            
        Returns
        -------
        Decimal
            Spread cost per unit (half-spread for crossing).
        """
        # Get base spread
        if quote and quote.bid_price and quote.ask_price:
            # Use actual quote spread
            spread = float(quote.ask_price) - float(quote.bid_price)
            mid_price = (float(quote.ask_price) + float(quote.bid_price)) / 2
            spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        elif typical_spread_bps is not None:
            # Use provided typical spread
            spread_bps = typical_spread_bps
        else:
            # Use default based on asset class
            asset_class = self._get_asset_class(instrument_id)
            spread_bps = self.params.min_spreads.get(asset_class, 5.0)
        
        # Adjust for market conditions
        spread_bps = self._adjust_spread_for_conditions(
            spread_bps, instrument_id, quantity, volatility
        )
        
        # Calculate cost (half-spread for crossing)
        if quote:
            mid_price = (float(quote.ask_price) + float(quote.bid_price)) / 2
        else:
            # Estimate from cached data or use 100 as default
            mid_price = self._get_cached_price(instrument_id) or 100.0
        
        # Half spread cost (you pay half when crossing the spread)
        spread_cost = (spread_bps / 10000) * mid_price * 0.5
        
        return Decimal(str(spread_cost))
    
    def calculate_effective_spread(
        self,
        instrument_id: InstrumentId,
        quantity: Quantity,
        quote: Optional[QuoteTick] = None,
        include_market_impact: bool = True,
    ) -> Decimal:
        """
        Calculate effective spread including size impact.
        
        Returns
        -------
        Decimal
            Effective spread in price units.
        """
        if not quote:
            return Decimal("0")
        
        base_spread = float(quote.ask_price) - float(quote.bid_price)
        
        if not include_market_impact:
            return Decimal(str(base_spread))
        
        # Calculate size impact on spread
        size_impact = self._calculate_size_impact(quantity, quote)
        effective_spread = base_spread * (1 + size_impact)
        
        return Decimal(str(effective_spread))
    
    def get_execution_prices(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        quote: QuoteTick,
        include_spread_cost: bool = True,
    ) -> Dict[str, Price]:
        """
        Get expected execution prices including spread.
        
        Returns
        -------
        dict
            Contains 'worst_price', 'expected_price', 'best_price'.
        """
        if side == OrderSide.BUY:
            # Buying - start from ask
            base_price = quote.ask_price
            spread_direction = 1  # Pay more
        else:
            # Selling - start from bid
            base_price = quote.bid_price
            spread_direction = -1  # Receive less
        
        if not include_spread_cost:
            return {
                "worst_price": base_price,
                "expected_price": base_price,
                "best_price": base_price,
            }
        
        # Calculate spread cost
        spread_cost = self.calculate_spread_cost(
            instrument_id, side, quantity, quote
        )
        
        # Calculate prices
        spread_adjustment = float(spread_cost) * spread_direction
        
        expected_price = Price(
            float(base_price) + spread_adjustment,
            precision=base_price.precision
        )
        
        # Best case - might get price improvement
        best_adjustment = spread_adjustment * 0.5  # 50% price improvement
        best_price = Price(
            float(base_price) + best_adjustment,
            precision=base_price.precision
        )
        
        # Worst case - wider spread
        worst_adjustment = spread_adjustment * 1.5  # 50% worse
        worst_price = Price(
            float(base_price) + worst_adjustment,
            precision=base_price.precision
        )
        
        return {
            "worst_price": worst_price,
            "expected_price": expected_price,
            "best_price": best_price,
        }
    
    def _adjust_spread_for_conditions(
        self,
        base_spread_bps: float,
        instrument_id: InstrumentId,
        quantity: Quantity,
        volatility: Optional[float] = None,
    ) -> float:
        """Adjust spread based on market conditions."""
        adjusted_spread = base_spread_bps
        
        # Time of day adjustment
        current_hour = datetime.utcnow().hour
        if (current_hour < self.params.market_hours_start or 
            current_hour >= self.params.market_hours_end):
            adjusted_spread *= self.params.off_hours_multiplier
        
        # Volatility adjustment
        if volatility and volatility > 0.02:  # High volatility threshold
            vol_multiplier = 1 + (volatility - 0.02) * self.params.volatility_spread_multiplier
            adjusted_spread *= vol_multiplier
        
        # Size adjustment (simplified)
        # In reality, would check against typical quote sizes
        size_factor = 1.0
        if float(quantity) > 1000:  # Large order threshold
            size_factor = self.params.size_spread_multiplier
        adjusted_spread *= size_factor
        
        # Check if cross-currency/synthetic
        if self._is_cross(instrument_id):
            adjusted_spread *= self.params.cross_spread_multiplier
        
        return adjusted_spread
    
    def _calculate_size_impact(self, quantity: Quantity, quote: QuoteTick) -> float:
        """Calculate how order size impacts effective spread."""
        if not quote.bid_size or not quote.ask_size:
            return 0.0
        
        # Compare order size to quote size
        quote_size = min(float(quote.bid_size), float(quote.ask_size))
        if quote_size <= 0:
            return 0.0
        
        size_ratio = float(quantity) / quote_size
        
        # Linear impact up to 2x quote size, then sqrt
        if size_ratio <= 1:
            return 0.0
        elif size_ratio <= 2:
            return (size_ratio - 1) * 0.1  # 10% per quote size
        else:
            return 0.1 + (size_ratio - 2) ** 0.5 * 0.05
    
    def _get_asset_class(self, instrument_id: InstrumentId) -> str:
        """Determine asset class from instrument ID."""
        symbol = str(instrument_id.symbol)
        
        if any(crypto in symbol for crypto in ["BTC", "ETH", "USDT", "USDC"]):
            return "crypto"
        elif "/" in symbol and len(symbol.split("/")[0]) == 3:
            return "forex"
        elif any(fut in symbol for fut in ["-", "_PERP", "FUT"]):
            return "futures"
        elif any(opt in symbol for opt in ["C", "P"]) and symbol[-1].isdigit():
            return "options"
        else:
            return "equity"
    
    def _is_cross(self, instrument_id: InstrumentId) -> bool:
        """Check if instrument is a cross/synthetic."""
        symbol = str(instrument_id.symbol)
        
        # Forex crosses (not vs USD)
        if "/" in symbol:
            currencies = symbol.split("/")
            if len(currencies) == 2 and "USD" not in currencies:
                return True
        
        # Crypto crosses (not vs USDT/USDC)
        if any(crypto in symbol for crypto in ["BTC", "ETH"]):
            if not any(stable in symbol for stable in ["USDT", "USDC", "USD"]):
                return True
        
        return False
    
    def _get_cached_price(self, instrument_id: InstrumentId) -> Optional[float]:
        """Get cached price for spread calculation."""
        if instrument_id in self._spread_cache:
            return self._spread_cache[instrument_id].get("last_price")
        return None
    
    def update_cache(self, instrument_id: InstrumentId, quote: QuoteTick) -> None:
        """Update spread cache with latest data."""
        if instrument_id not in self._spread_cache:
            self._spread_cache[instrument_id] = {}
        
        mid_price = (float(quote.ask_price) + float(quote.bid_price)) / 2
        spread = float(quote.ask_price) - float(quote.bid_price)
        
        self._spread_cache[instrument_id].update({
            "last_price": mid_price,
            "last_spread": spread,
            "last_update": datetime.utcnow(),
        })