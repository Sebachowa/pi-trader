
"""
Paper trading configuration for foreign exchange (FX) markets.

This configuration supports FX trading simulation with realistic market conditions.
"""

from decimal import Decimal

from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import DataEngineConfig
from nautilus_trader.config import ExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import RiskEngineConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue


# Define FX venues
FX_VENUE = Venue("FX-ECN")
FX_VENUE_PRIME = Venue("FX-PRIME")


def get_fx_paper_config(
    trader_id: str = "PAPER-FX-001",
    log_level: str = "INFO",
    use_ecn: bool = True,
    use_prime: bool = False,
    starting_balance_usd: float = 1_000_000.0,
    starting_balance_eur: float = 850_000.0,
    starting_balance_gbp: float = 750_000.0,
    starting_balance_jpy: float = 100_000_000.0,
    default_leverage: float = 50.0,
) -> TradingNodeConfig:
    """
    Generate a paper trading configuration for FX markets.

    Parameters
    ----------
    trader_id : str
        The trader ID for the paper trading session.
    log_level : str
        The logging level (DEBUG, INFO, WARNING, ERROR).
    use_ecn : bool
        Whether to include ECN venue configuration.
    use_prime : bool
        Whether to include Prime Brokerage configuration.
    starting_balance_usd : float
        Starting USD balance.
    starting_balance_eur : float
        Starting EUR balance.
    starting_balance_gbp : float
        Starting GBP balance.
    starting_balance_jpy : float
        Starting JPY balance.
    default_leverage : float
        Default leverage for FX trading.

    Returns
    -------
    TradingNodeConfig
        The configured trading node for FX paper trading.

    """
    exec_clients = {}
    
    # Common starting balances for FX
    starting_balances = [
        f"{starting_balance_usd} USD",
        f"{starting_balance_eur} EUR",
        f"{starting_balance_gbp} GBP",
        f"{starting_balance_jpy} JPY",
    ]
    
    if use_ecn:
        exec_clients[FX_VENUE.value] = SandboxExecutionClientConfig(
            venue=FX_VENUE.value,
            starting_balances=starting_balances,
            oms_type="NETTING",  # FX typically uses netting
            account_type="MARGIN",
            default_leverage=Decimal(default_leverage),
            leverages={
                "EUR/USD": 100.0,  # Major pairs higher leverage
                "GBP/USD": 100.0,
                "USD/JPY": 100.0,
                "EUR/GBP": 50.0,   # Cross pairs lower leverage
                "EUR/JPY": 50.0,
                "GBP/JPY": 50.0,
            },
            book_type="L2_MBP",
            bar_execution=True,
            trade_execution=True,  # Enable trade tick execution
            reject_stop_orders=False,  # FX accepts stop orders
            support_gtd_orders=True,
            support_contingent_orders=True,
            use_position_ids=False,  # FX uses netting
            use_random_ids=False,
            use_reduce_only=False,
        )
    
    if use_prime:
        exec_clients[FX_VENUE_PRIME.value] = SandboxExecutionClientConfig(
            venue=FX_VENUE_PRIME.value,
            starting_balances=starting_balances,
            oms_type="NETTING",
            account_type="MARGIN",
            default_leverage=Decimal(default_leverage * 2),  # Prime typically offers higher leverage
            leverages={
                "EUR/USD": 200.0,
                "GBP/USD": 200.0,
                "USD/JPY": 200.0,
                "EUR/GBP": 100.0,
                "EUR/JPY": 100.0,
                "GBP/JPY": 100.0,
            },
            book_type="L1_MBP",  # Prime typically provides L1 quotes
            bar_execution=True,
            trade_execution=True,
            reject_stop_orders=False,
            support_gtd_orders=True,
            support_contingent_orders=True,
            use_position_ids=False,
            use_random_ids=False,
            use_reduce_only=False,
        )
    
    return TradingNodeConfig(
        trader_id=TraderId(trader_id),
        logging=LoggingConfig(
            log_level=log_level,
            log_colors=True,
            use_pyo3=True,
            log_directory="paper_trading/logs",
        ),
        data_engine=DataEngineConfig(
            time_bars_build_with_no_updates=True,
            time_bars_timestamp_on_close=True,
            time_bars_interval_type="MID_POINT",
            validate_data_sequence=True,
            external_data=True,
        ),
        risk_engine=RiskEngineConfig(
            bypass=False,
            max_order_submit_rate="200/00:00:01",  # Higher rate for FX
            max_order_modify_rate="100/00:00:01",
            max_notional_per_order={"USD": 10_000_000},  # Higher notional for FX
        ),
        exec_engine=ExecEngineConfig(
            load_cache=False,
            allow_cash_positions=True,
        ),
        cache=CacheConfig(
            timestamps_as_iso8601=True,
            flush_on_start=False,
            save_market_data=True,
        ),
        exec_clients=exec_clients,
        timeout_connection=10.0,  # Faster timeouts for FX
        timeout_reconciliation=5.0,
        timeout_portfolio=5.0,
        timeout_disconnection=5.0,
        timeout_post_stop=2.0,
    )