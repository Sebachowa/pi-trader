
"""
Market impact modeling for realistic order execution simulation.
"""

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


@dataclass
class MarketImpactParams:
    """Parameters for market impact modeling."""
    
    # Linear impact parameters
    linear_impact_coefficient: float = 0.1  # basis points per percent of ADV
    
    # Square root impact (Almgren-Chriss model)
    sqrt_impact_coefficient: float = 0.5  # basis points
    
    # Power law parameters
    power_law_exponent: float = 0.6  # Between 0.5 and 1.0
    power_law_coefficient: float = 0.2
    
    # Temporary vs permanent impact
    temporary_impact_ratio: float = 0.7  # 70% temporary, 30% permanent
    decay_time_minutes: float = 30.0  # Time for temporary impact to decay
    
    # Order book shape impact
    book_depth_coefficient: float = 0.05  # Impact per level consumed
    book_imbalance_multiplier: float = 1.5  # Multiplier for order book imbalance
    
    # Urgency and aggression factors
    aggressive_multiplier: float = 2.0  # For aggressive orders
    passive_multiplier: float = 0.5  # For passive orders
    
    # Asset-specific parameters
    asset_volatility_adjustment: bool = True  # Adjust for asset volatility
    volatility_coefficient: float = 0.8  # How much volatility affects impact


class MarketImpactModel:
    """
    Sophisticated market impact model for paper trading.
    
    Implements multiple impact models:
    - Linear impact for small orders
    - Square-root impact (Almgren-Chriss) for medium orders
    - Power law impact for large orders
    - Order book shape-based impact
    - Temporary vs permanent impact decomposition
    """
    
    def __init__(self, params: Optional[MarketImpactParams] = None):
        self.params = params or MarketImpactParams()
        
        # Cache for instrument-specific data
        self._instrument_cache: Dict[InstrumentId, Dict[str, float]] = {}
        
        # Impact history for analysis
        self._impact_history: Dict[InstrumentId, List[Dict]] = {}
    
    def calculate_impact(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        market_depth: Dict[str, List[Tuple[float, float]]],
        average_daily_volume: float,
        volatility: Optional[float] = None,
        urgency: str = "normal",  # "passive", "normal", "aggressive"
    ) -> Decimal:
        """
        Calculate total market impact for an order.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument being traded.
        side : OrderSide
            Buy or sell side.
        quantity : Quantity
            Order quantity.
        market_depth : dict
            Order book with 'bids' and 'asks' as [(price, size), ...].
        average_daily_volume : float
            Average daily volume for the instrument.
        volatility : float, optional
            Current volatility for adjustment.
        urgency : str
            Order urgency level affecting impact.
            
        Returns
        -------
        Decimal
            Market impact as a decimal (e.g., 0.001 for 0.1%).
        """
        if average_daily_volume <= 0:
            return Decimal("0")
        
        # Calculate participation rate
        participation_rate = float(quantity) / average_daily_volume
        
        # Choose impact model based on order size
        if participation_rate < 0.001:  # Less than 0.1% of ADV
            impact = self._linear_impact(participation_rate)
        elif participation_rate < 0.01:  # 0.1% to 1% of ADV
            impact = self._sqrt_impact(participation_rate)
        else:  # Large orders
            impact = self._power_law_impact(participation_rate)
        
        # Add order book impact
        book_impact = self._order_book_impact(quantity, side, market_depth)
        impact += book_impact
        
        # Apply volatility adjustment
        if volatility and self.params.asset_volatility_adjustment:
            vol_adjustment = 1 + (volatility * self.params.volatility_coefficient)
            impact *= vol_adjustment
        
        # Apply urgency multiplier
        if urgency == "aggressive":
            impact *= self.params.aggressive_multiplier
        elif urgency == "passive":
            impact *= self.params.passive_multiplier
        
        # Apply side bias (buying typically has more impact)
        if side == OrderSide.BUY:
            impact *= 1.05
        
        # Track impact
        self._track_impact(instrument_id, impact, participation_rate)
        
        return Decimal(str(impact))
    
    def calculate_temporary_permanent_split(
        self,
        total_impact: Decimal,
        time_since_execution_minutes: float,
    ) -> Tuple[Decimal, Decimal]:
        """
        Split impact into temporary and permanent components.
        
        Parameters
        ----------
        total_impact : Decimal
            Total market impact.
        time_since_execution_minutes : float
            Time since order execution.
            
        Returns
        -------
        Tuple[Decimal, Decimal]
            (temporary_impact, permanent_impact)
        """
        permanent = total_impact * Decimal(str(1 - self.params.temporary_impact_ratio))
        
        # Temporary impact decays exponentially
        decay_factor = math.exp(-time_since_execution_minutes / self.params.decay_time_minutes)
        temporary = total_impact * Decimal(str(self.params.temporary_impact_ratio * decay_factor))
        
        return temporary, permanent
    
    def _linear_impact(self, participation_rate: float) -> float:
        """Linear impact model for small orders."""
        return participation_rate * self.params.linear_impact_coefficient * 0.0001  # Convert to decimal
    
    def _sqrt_impact(self, participation_rate: float) -> float:
        """Square-root impact model (Almgren-Chriss)."""
        return self.params.sqrt_impact_coefficient * math.sqrt(participation_rate) * 0.0001
    
    def _power_law_impact(self, participation_rate: float) -> float:
        """Power law impact model for large orders."""
        return (self.params.power_law_coefficient * 
                (participation_rate ** self.params.power_law_exponent) * 0.0001)
    
    def _order_book_impact(
        self,
        quantity: Quantity,
        side: OrderSide,
        market_depth: Dict[str, List[Tuple[float, float]]],
    ) -> float:
        """Calculate impact based on order book consumption."""
        if not market_depth:
            return 0.0
        
        # Get relevant side of book
        book_side = market_depth.get("asks" if side == OrderSide.BUY else "bids", [])
        if not book_side:
            return 0.0
        
        # Calculate how many levels would be consumed
        remaining_qty = float(quantity)
        levels_consumed = 0
        total_cost = 0.0
        base_price = book_side[0][0] if book_side else 0.0
        
        for price, size in book_side:
            if remaining_qty <= 0:
                break
            
            consumed = min(remaining_qty, size)
            total_cost += consumed * price
            remaining_qty -= consumed
            
            if consumed > 0:
                levels_consumed += consumed / size  # Fractional level consumption
        
        # Calculate average execution price
        if float(quantity) - remaining_qty > 0:
            avg_price = total_cost / (float(quantity) - remaining_qty)
            price_impact = abs(avg_price - base_price) / base_price if base_price > 0 else 0
        else:
            price_impact = 0
        
        # Add impact for order book imbalance
        imbalance = self._calculate_book_imbalance(market_depth)
        if (side == OrderSide.BUY and imbalance > 0) or (side == OrderSide.SELL and imbalance < 0):
            # Trading against the imbalance
            price_impact *= self.params.book_imbalance_multiplier
        
        # Apply book depth coefficient
        return price_impact + (levels_consumed * self.params.book_depth_coefficient * 0.0001)
    
    def _calculate_book_imbalance(self, market_depth: Dict[str, List[Tuple[float, float]]]) -> float:
        """Calculate order book imbalance (-1 to 1, positive means more asks)."""
        bids = market_depth.get("bids", [])
        asks = market_depth.get("asks", [])
        
        if not bids and not asks:
            return 0.0
        
        bid_volume = sum(size for _, size in bids[:5])  # Top 5 levels
        ask_volume = sum(size for _, size in asks[:5])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (ask_volume - bid_volume) / total_volume
    
    def _track_impact(
        self,
        instrument_id: InstrumentId,
        impact: float,
        participation_rate: float,
    ) -> None:
        """Track impact for analysis and calibration."""
        if instrument_id not in self._impact_history:
            self._impact_history[instrument_id] = []
        
        self._impact_history[instrument_id].append({
            "timestamp": self._get_timestamp(),
            "impact": impact,
            "participation_rate": participation_rate,
        })
        
        # Keep only recent history
        if len(self._impact_history[instrument_id]) > 1000:
            self._impact_history[instrument_id] = self._impact_history[instrument_id][-1000:]
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().timestamp()
    
    def estimate_total_cost(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        market_depth: Dict[str, List[Tuple[float, float]]],
        average_daily_volume: float,
        base_price: float,
    ) -> Dict[str, Decimal]:
        """
        Estimate total execution cost including impact.
        
        Returns
        -------
        dict
            Contains 'impact_cost', 'total_cost', 'average_price'.
        """
        impact = self.calculate_impact(
            instrument_id, side, quantity, market_depth, average_daily_volume
        )
        
        # Calculate impact cost
        impact_cost = Decimal(str(base_price)) * Decimal(str(float(quantity))) * impact
        
        # Calculate average execution price
        if side == OrderSide.BUY:
            avg_price = Decimal(str(base_price)) * (Decimal("1") + impact)
        else:
            avg_price = Decimal(str(base_price)) * (Decimal("1") - impact)
        
        # Total cost
        total_cost = avg_price * Decimal(str(float(quantity)))
        
        return {
            "impact_cost": impact_cost,
            "total_cost": total_cost,
            "average_price": avg_price,
            "impact_percentage": impact * Decimal("100"),  # As percentage
        }
    
    def get_impact_curve(
        self,
        instrument_id: InstrumentId,
        side: OrderSide,
        max_quantity: Quantity,
        market_depth: Dict[str, List[Tuple[float, float]]],
        average_daily_volume: float,
        num_points: int = 20,
    ) -> List[Tuple[float, float]]:
        """
        Generate impact curve for different order sizes.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (quantity, impact) pairs.
        """
        curve = []
        max_qty = float(max_quantity)
        
        for i in range(1, num_points + 1):
            qty = max_qty * (i / num_points)
            impact = self.calculate_impact(
                instrument_id,
                side,
                Quantity(qty, precision=max_quantity.precision),
                market_depth,
                average_daily_volume,
            )
            curve.append((qty, float(impact)))
        
        return curve