
"""
Paper trading configuration for equity markets.

This configuration supports equity trading simulation with support for:
- US equities (NYSE, NASDAQ)
- Interactive Brokers integration
- Realistic market hours and trading rules
"""

from decimal import Decimal

from nautilus_trader.adapters.interactive_brokers import IB, IB_VENUE
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import DataEngineConfig
from nautilus_trader.config import ExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import RiskEngineConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue


# Define equity venues
NYSE_VENUE = Venue("NYSE")
NASDAQ_VENUE = Venue("NASDAQ")


def get_equities_paper_config(
    trader_id: str = "PAPER-EQUITIES-001",
    log_level: str = "INFO",
    use_ib: bool = True,
    use_nyse: bool = True,
    use_nasdaq: bool = True,
    starting_balance_usd: float = 100_000.0,
    enable_margin: bool = True,
    margin_multiplier: float = 2.0,  # Reg T margin
) -> TradingNodeConfig:
    """
    Generate a paper trading configuration for equity markets.

    Parameters
    ----------
    trader_id : str
        The trader ID for the paper trading session.
    log_level : str
        The logging level (DEBUG, INFO, WARNING, ERROR).
    use_ib : bool
        Whether to include Interactive Brokers configuration.
    use_nyse : bool
        Whether to include NYSE configuration.
    use_nasdaq : bool
        Whether to include NASDAQ configuration.
    starting_balance_usd : float
        Starting USD balance.
    enable_margin : bool
        Whether to enable margin trading.
    margin_multiplier : float
        Margin multiplier (2.0 for Reg T, 4.0 for pattern day trading).

    Returns
    -------
    TradingNodeConfig
        The configured trading node for equity paper trading.

    """
    exec_clients = {}
    
    # Starting balance
    starting_balances = [f"{starting_balance_usd} USD"]
    
    # Account type based on margin settings
    account_type = "MARGIN" if enable_margin else "CASH"
    default_leverage = Decimal(margin_multiplier if enable_margin else 1.0)
    
    if use_ib:
        exec_clients[IB] = SandboxExecutionClientConfig(
            venue=IB_VENUE,
            starting_balances=starting_balances,
            oms_type="NETTING",
            account_type=account_type,
            default_leverage=default_leverage,
            book_type="L1_MBP",  # IB provides L1 quotes
            bar_execution=True,
            trade_execution=True,
            reject_stop_orders=False,
            support_gtd_orders=True,
            support_contingent_orders=True,
            use_position_ids=False,
            use_random_ids=False,
            use_reduce_only=False,
        )
    
    if use_nyse:
        exec_clients[NYSE_VENUE.value] = SandboxExecutionClientConfig(
            venue=NYSE_VENUE.value,
            starting_balances=starting_balances,
            oms_type="NETTING",
            account_type=account_type,
            default_leverage=default_leverage,
            leverages={
                "AAPL": margin_multiplier,
                "MSFT": margin_multiplier,
                "GOOGL": margin_multiplier,
                "AMZN": margin_multiplier,
                "JPM": margin_multiplier,
                "BAC": margin_multiplier,
                "WMT": margin_multiplier,
                "JNJ": margin_multiplier,
                "PG": margin_multiplier,
                "XOM": margin_multiplier,
            },
            book_type="L2_MBP",
            bar_execution=True,
            trade_execution=True,
            reject_stop_orders=False,
            support_gtd_orders=True,
            support_contingent_orders=True,
            use_position_ids=False,
            use_random_ids=False,
            use_reduce_only=False,
        )
    
    if use_nasdaq:
        exec_clients[NASDAQ_VENUE.value] = SandboxExecutionClientConfig(
            venue=NASDAQ_VENUE.value,
            starting_balances=starting_balances,
            oms_type="NETTING",
            account_type=account_type,
            default_leverage=default_leverage,
            leverages={
                "TSLA": margin_multiplier,
                "NVDA": margin_multiplier,
                "META": margin_multiplier,
                "NFLX": margin_multiplier,
                "AMD": margin_multiplier,
                "INTC": margin_multiplier,
                "CSCO": margin_multiplier,
                "ADBE": margin_multiplier,
                "CRM": margin_multiplier,
                "PYPL": margin_multiplier,
            },
            book_type="L2_MBP",
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
            time_bars_interval_type="LAST",  # Use last price for equities
            validate_data_sequence=True,
            external_data=True,
        ),
        risk_engine=RiskEngineConfig(
            bypass=False,
            max_order_submit_rate="60/00:00:01",  # Lower rate for equities
            max_order_modify_rate="30/00:00:01",
            max_notional_per_order={"USD": 1_000_000},
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
        timeout_connection=30.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )