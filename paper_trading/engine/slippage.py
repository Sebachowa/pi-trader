
"""
Slippage modeling for realistic paper trading execution.
"""

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


@dataclass
class SlippageParams:
    """Parameters for slippage modeling."""
    
    # Base slippage rates (as percentage)
    base_slippage_rate: float = 0.0001  # 0.01% base slippage
    
    # Volatility impact
    volatility_multiplier: float = 2.0  # How much volatility increases slippage
    high_volatility_threshold: float = 0.02  # 2% volatility is considered high
    
    # Size impact
    size_impact_threshold: float = 0.01  # 1% of average volume causes impact
    size_impact_multiplier: float = 1.5  # How much large orders increase slippage
    
    # Liquidity impact
    low_liquidity_multiplier: float = 3.0  # Multiplier for low liquidity conditions
    liquidity_threshold: float = 100  # Below this is considered low liquidity
    
    # Time of day factors
    market_open_multiplier: float = 2.0  # First 30 mins
    market_close_multiplier: float = 1.5  # Last 30 mins
    lunch_hour_multiplier: float = 1.2  # 12-1 PM
    
    # Random component
    random_factor: float = 0.3  # Up to 30% random variation
    
    # Asset class specific adjustments
    asset_class_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.asset_class_multipliers is None:
            self.asset_class_multipliers = {
                "crypto": 1.5,    # Higher slippage for crypto
                "forex": 0.5,     # Lower slippage for major FX pairs
                "equity": 1.0,    # Standard for equities
                "futures": 0.8,   # Lower for liquid futures
                "options": 2.0,   # Higher for options
            }


class SlippageModel:
    """
    Advanced slippage model for realistic order execution simulation.
    
    Models slippage based on:
    - Market volatility
    - Order size relative to typical volume
    - Current liquidity conditions
    - Time of day
    - Asset class characteristics
    - Random market microstructure noise
    """
    
    def __init__(self, params: Optional[SlippageParams] = None):
        self.params = params or SlippageParams()
        
        # Cache for instrument-specific parameters
        self._instrument_cache: Dict[InstrumentId, Dict[str, float]] = {}
        
        # Historical slippage tracking for calibration
        self._slippage_history: Dict[InstrumentId, list] = {}
    
    def calculate_slippage(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        current_volatility: float,
        current_liquidity: float,
        average_volume: Optional[float] = None,
        time_of_day_hours: Optional[float] = None,
    ) -> Decimal:
        """
        Calculate slippage for an order.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument being traded.
        side : OrderSide
            Buy or sell side.
        quantity : Quantity
            Order quantity.
        current_volatility : float
            Current market volatility (as decimal, e.g., 0.02 for 2%).
        current_liquidity : float
            Current liquidity measure (e.g., bid/ask depth).
        average_volume : float, optional
            Average daily volume for size impact calculation.
        time_of_day_hours : float, optional
            Hour of day (0-24) for time-based adjustments.
            
        Returns
        -------
        Decimal
            Slippage as a decimal (e.g., 0.0005 for 0.05%).
        """
        # Start with base slippage
        slippage = self.params.base_slippage_rate
        
        # Apply volatility impact
        volatility_factor = self._calculate_volatility_impact(current_volatility)
        slippage *= volatility_factor
        
        # Apply size impact
        if average_volume:
            size_factor = self._calculate_size_impact(quantity, average_volume)
            slippage *= size_factor
        
        # Apply liquidity impact
        liquidity_factor = self._calculate_liquidity_impact(current_liquidity)
        slippage *= liquidity_factor
        
        # Apply time of day impact
        if time_of_day_hours is not None:
            time_factor = self._calculate_time_impact(time_of_day_hours)
            slippage *= time_factor
        
        # Apply asset class multiplier
        asset_class = self._get_asset_class(instrument_id)
        if asset_class in self.params.asset_class_multipliers:
            slippage *= self.params.asset_class_multipliers[asset_class]
        
        # Apply side-specific adjustments (buying typically has more slippage)
        if side == OrderSide.BUY:
            slippage *= 1.1  # 10% more slippage for buys
        
        # Add random component
        random_adjustment = 1 + (random.random() - 0.5) * self.params.random_factor
        slippage *= random_adjustment
        
        # Ensure slippage is non-negative
        slippage = max(0, slippage)
        
        # Track for calibration
        self._track_slippage(instrument_id, slippage)
        
        return Decimal(str(slippage))
    
    def _calculate_volatility_impact(self, volatility: float) -> float:
        """Calculate impact of volatility on slippage."""
        if volatility <= 0:
            return 1.0
        
        if volatility > self.params.high_volatility_threshold:
            # High volatility - significant impact
            excess_vol = volatility - self.params.high_volatility_threshold
            return 1 + (excess_vol / self.params.high_volatility_threshold) * self.params.volatility_multiplier
        else:
            # Normal volatility - proportional impact
            return 1 + (volatility / self.params.high_volatility_threshold) * 0.5
    
    def _calculate_size_impact(self, quantity: Quantity, average_volume: float) -> float:
        """Calculate impact of order size on slippage."""
        if average_volume <= 0:
            return 1.0
        
        size_ratio = float(quantity) / average_volume
        
        if size_ratio > self.params.size_impact_threshold:
            # Large order relative to volume
            excess_size = size_ratio - self.params.size_impact_threshold
            return 1 + (excess_size / self.params.size_impact_threshold) * self.params.size_impact_multiplier
        else:
            # Normal size - minimal impact
            return 1 + (size_ratio / self.params.size_impact_threshold) * 0.2
    
    def _calculate_liquidity_impact(self, liquidity: float) -> float:
        """Calculate impact of liquidity on slippage."""
        if liquidity <= 0:
            return self.params.low_liquidity_multiplier
        
        if liquidity < self.params.liquidity_threshold:
            # Low liquidity - significant impact
            liquidity_ratio = liquidity / self.params.liquidity_threshold
            return 1 + (1 - liquidity_ratio) * (self.params.low_liquidity_multiplier - 1)
        else:
            # Good liquidity - minimal impact
            return 1.0
    
    def _calculate_time_impact(self, hour: float) -> float:
        """Calculate time of day impact on slippage."""
        # Market open (first 30 minutes)
        if 9.5 <= hour < 10:
            return self.params.market_open_multiplier
        
        # Lunch hour
        elif 12 <= hour < 13:
            return self.params.lunch_hour_multiplier
        
        # Market close (last 30 minutes)
        elif 15.5 <= hour < 16:
            return self.params.market_close_multiplier
        
        # Normal trading hours
        else:
            return 1.0
    
    def _get_asset_class(self, instrument_id: InstrumentId) -> str:
        """Determine asset class from instrument ID."""
        symbol = str(instrument_id.symbol)
        
        # Crypto detection
        if any(crypto in symbol for crypto in ["BTC", "ETH", "USDT", "USDC"]):
            return "crypto"
        
        # Forex detection
        if "/" in symbol and len(symbol.split("/")[0]) == 3:
            return "forex"
        
        # Futures detection
        if any(fut in symbol for fut in ["-", "_PERP", "FUT"]):
            return "futures"
        
        # Options detection
        if any(opt in symbol for opt in ["C", "P"]) and symbol[-1].isdigit():
            return "options"
        
        # Default to equity
        return "equity"
    
    def _track_slippage(self, instrument_id: InstrumentId, slippage: float) -> None:
        """Track slippage for calibration and analysis."""
        if instrument_id not in self._slippage_history:
            self._slippage_history[instrument_id] = []
        
        self._slippage_history[instrument_id].append({
            "timestamp": Decimal(str(self._get_timestamp())),
            "slippage": slippage,
        })
        
        # Keep only recent history (last 1000 entries)
        if len(self._slippage_history[instrument_id]) > 1000:
            self._slippage_history[instrument_id] = self._slippage_history[instrument_id][-1000:]
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().timestamp()
    
    def get_average_slippage(self, instrument_id: InstrumentId) -> Optional[Decimal]:
        """Get average historical slippage for an instrument."""
        if instrument_id not in self._slippage_history or not self._slippage_history[instrument_id]:
            return None
        
        slippages = [entry["slippage"] for entry in self._slippage_history[instrument_id]]
        return Decimal(str(sum(slippages) / len(slippages)))
    
    def calibrate_from_history(self, instrument_id: InstrumentId, target_slippage: Decimal) -> None:
        """
        Calibrate model parameters based on observed vs target slippage.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to calibrate.
        target_slippage : Decimal
            The target average slippage to achieve.
        """
        current_avg = self.get_average_slippage(instrument_id)
        if current_avg is None:
            return
        
        # Calculate adjustment factor
        adjustment = float(target_slippage) / float(current_avg)
        
        # Store instrument-specific calibration
        if instrument_id not in self._instrument_cache:
            self._instrument_cache[instrument_id] = {}
        
        self._instrument_cache[instrument_id]["calibration_factor"] = adjustment