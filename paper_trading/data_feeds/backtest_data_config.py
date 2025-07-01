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
Backtest data configurations for paper trading evaluation.

This module provides configurations for backtesting with historical data
to evaluate strategies before paper trading.
"""

from datetime import datetime
from pathlib import Path

from nautilus_trader.backtest.config import BacktestDataConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import OrderBookDelta
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick


class BacktestDataConfigs:
    """Provides backtest data configurations for strategy evaluation."""
    
    @staticmethod
    def get_bar_data_config(
        catalog_path: str | Path,
        instrument_id: str,
        bar_spec: str = "1-MINUTE",
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        client_id: str | None = None,
    ) -> BacktestDataConfig:
        """
        Get configuration for bar data backtesting.
        
        Parameters
        ----------
        catalog_path : str or Path
            Path to the data catalog.
        instrument_id : str
            The instrument ID to backtest.
        bar_spec : str
            The bar specification (e.g., "1-MINUTE", "5-MINUTE", "1-HOUR").
        start_time : datetime or str, optional
            Start time for the backtest.
        end_time : datetime or str, optional
            End time for the backtest.
        client_id : str, optional
            The client ID for the data.
            
        Returns
        -------
        BacktestDataConfig
            The configured backtest data.
            
        """
        return BacktestDataConfig(
            catalog_path=str(catalog_path),
            data_cls=Bar,
            instrument_id=instrument_id,
            bar_spec=bar_spec,
            start_time=start_time,
            end_time=end_time,
            client_id=client_id,
        )
    
    @staticmethod
    def get_tick_data_config(
        catalog_path: str | Path,
        instrument_id: str,
        data_type: str = "quote",
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        client_id: str | None = None,
    ) -> BacktestDataConfig:
        """
        Get configuration for tick data backtesting.
        
        Parameters
        ----------
        catalog_path : str or Path
            Path to the data catalog.
        instrument_id : str
            The instrument ID to backtest.
        data_type : str
            The tick data type ("quote" or "trade").
        start_time : datetime or str, optional
            Start time for the backtest.
        end_time : datetime or str, optional
            End time for the backtest.
        client_id : str, optional
            The client ID for the data.
            
        Returns
        -------
        BacktestDataConfig
            The configured backtest data.
            
        """
        data_cls = QuoteTick if data_type == "quote" else TradeTick
        
        return BacktestDataConfig(
            catalog_path=str(catalog_path),
            data_cls=data_cls,
            instrument_id=instrument_id,
            start_time=start_time,
            end_time=end_time,
            client_id=client_id,
        )
    
    @staticmethod
    def get_orderbook_data_config(
        catalog_path: str | Path,
        instrument_id: str,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        client_id: str | None = None,
    ) -> BacktestDataConfig:
        """
        Get configuration for order book data backtesting.
        
        Parameters
        ----------
        catalog_path : str or Path
            Path to the data catalog.
        instrument_id : str
            The instrument ID to backtest.
        start_time : datetime or str, optional
            Start time for the backtest.
        end_time : datetime or str, optional
            End time for the backtest.
        client_id : str, optional
            The client ID for the data.
            
        Returns
        -------
        BacktestDataConfig
            The configured backtest data.
            
        """
        return BacktestDataConfig(
            catalog_path=str(catalog_path),
            data_cls=OrderBookDelta,
            instrument_id=instrument_id,
            start_time=start_time,
            end_time=end_time,
            client_id=client_id,
        )
    
    @staticmethod
    def get_multi_instrument_config(
        catalog_path: str | Path,
        instrument_configs: list[dict],
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
    ) -> list[BacktestDataConfig]:
        """
        Get configuration for multi-instrument backtesting.
        
        Parameters
        ----------
        catalog_path : str or Path
            Path to the data catalog.
        instrument_configs : list[dict]
            List of instrument configurations with keys:
            - instrument_id: str
            - data_type: str ("bar", "quote", "trade", "orderbook")
            - bar_spec: str (optional, for bar data)
        start_time : datetime or str, optional
            Start time for the backtest.
        end_time : datetime or str, optional
            End time for the backtest.
            
        Returns
        -------
        list[BacktestDataConfig]
            List of configured backtest data.
            
        """
        configs = []
        
        for config in instrument_configs:
            instrument_id = config["instrument_id"]
            data_type = config["data_type"]
            
            if data_type == "bar":
                bar_spec = config.get("bar_spec", "1-MINUTE")
                configs.append(
                    BacktestDataConfigs.get_bar_data_config(
                        catalog_path=catalog_path,
                        instrument_id=instrument_id,
                        bar_spec=bar_spec,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
            elif data_type in ["quote", "trade"]:
                configs.append(
                    BacktestDataConfigs.get_tick_data_config(
                        catalog_path=catalog_path,
                        instrument_id=instrument_id,
                        data_type=data_type,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
            elif data_type == "orderbook":
                configs.append(
                    BacktestDataConfigs.get_orderbook_data_config(
                        catalog_path=catalog_path,
                        instrument_id=instrument_id,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
        
        return configs