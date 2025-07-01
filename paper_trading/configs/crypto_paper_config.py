# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Paper trading configuration for cryptocurrency markets.

This configuration supports multiple cryptocurrency exchanges including:
- Binance (spot and futures)
- Bybit
- Coinbase International Exchange
- OKX
"""

from decimal import Decimal

from nautilus_trader.adapters.binance import BINANCE, BINANCE_VENUE
from nautilus_trader.adapters.bybit import BYBIT, BYBIT_VENUE
from nautilus_trader.adapters.coinbase_intx import COINBASE_INTX, COINBASE_INTX_VENUE
from nautilus_trader.adapters.okx import OKX, OKX_VENUE
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import DataEngineConfig
from nautilus_trader.config import ExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import RiskEngineConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.model.identifiers import TraderId


def get_crypto_paper_config(
    trader_id: str = "PAPER-CRYPTO-001",
    log_level: str = "INFO",
    use_binance: bool = True,
    use_bybit: bool = True,
    use_coinbase: bool = False,
    use_okx: bool = False,
    starting_balance_usdt: float = 100_000.0,
    starting_balance_btc: float = 2.0,
    starting_balance_eth: float = 30.0,
) -> TradingNodeConfig:
    """
    Generate a paper trading configuration for cryptocurrency markets.

    Parameters
    ----------
    trader_id : str
        The trader ID for the paper trading session.
    log_level : str
        The logging level (DEBUG, INFO, WARNING, ERROR).
    use_binance : bool
        Whether to include Binance configuration.
    use_bybit : bool
        Whether to include Bybit configuration.
    use_coinbase : bool
        Whether to include Coinbase INTX configuration.
    use_okx : bool
        Whether to include OKX configuration.
    starting_balance_usdt : float
        Starting USDT balance for each exchange.
    starting_balance_btc : float
        Starting BTC balance for each exchange.
    starting_balance_eth : float
        Starting ETH balance for each exchange.

    Returns
    -------
    TradingNodeConfig
        The configured trading node for paper trading.

    """
    exec_clients = {}
    
    # Common starting balances for all exchanges
    starting_balances = [
        f"{starting_balance_usdt} USDT",
        f"{starting_balance_btc} BTC",
        f"{starting_balance_eth} ETH",
    ]
    
    if use_binance:
        exec_clients[BINANCE] = SandboxExecutionClientConfig(
            venue=BINANCE_VENUE,
            starting_balances=starting_balances,
            oms_type="HEDGING",  # Binance futures support hedging mode
            account_type="MARGIN",
            default_leverage=Decimal(10),
            leverages={
                "BTCUSDT-PERP": 20.0,
                "ETHUSDT-PERP": 20.0,
            },
            book_type="L2_MBP",  # Level 2 Market By Price
            bar_execution=True,
            reject_stop_orders=True,
            support_gtd_orders=False,
            support_contingent_orders=True,
            use_position_ids=True,
            use_random_ids=False,
            use_reduce_only=True,
        )
    
    if use_bybit:
        exec_clients[BYBIT] = SandboxExecutionClientConfig(
            venue=BYBIT_VENUE,
            starting_balances=starting_balances,
            oms_type="HEDGING",
            account_type="MARGIN",
            default_leverage=Decimal(10),
            leverages={
                "BTCUSDT": 25.0,
                "ETHUSDT": 25.0,
            },
            book_type="L2_MBP",
            bar_execution=True,
            reject_stop_orders=True,
            support_gtd_orders=False,
            support_contingent_orders=True,
            use_position_ids=True,
            use_random_ids=False,
            use_reduce_only=True,
        )
    
    if use_coinbase:
        exec_clients[COINBASE_INTX] = SandboxExecutionClientConfig(
            venue=COINBASE_INTX_VENUE,
            starting_balances=starting_balances,
            oms_type="NETTING",
            account_type="MARGIN",
            default_leverage=Decimal(5),
            book_type="L2_MBP",
            bar_execution=True,
            reject_stop_orders=True,
            support_gtd_orders=True,
            support_contingent_orders=True,
            use_position_ids=False,
            use_random_ids=False,
            use_reduce_only=True,
        )
    
    if use_okx:
        exec_clients[OKX] = SandboxExecutionClientConfig(
            venue=OKX_VENUE,
            starting_balances=starting_balances,
            oms_type="HEDGING",
            account_type="MARGIN",
            default_leverage=Decimal(10),
            leverages={
                "BTC-USDT-SWAP": 20.0,
                "ETH-USDT-SWAP": 20.0,
            },
            book_type="L2_MBP",
            bar_execution=True,
            reject_stop_orders=True,
            support_gtd_orders=False,
            support_contingent_orders=True,
            use_position_ids=True,
            use_random_ids=False,
            use_reduce_only=True,
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
            max_order_submit_rate="100/00:00:01",
            max_order_modify_rate="50/00:00:01",
            max_notional_per_order={"USD": 100_000},
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