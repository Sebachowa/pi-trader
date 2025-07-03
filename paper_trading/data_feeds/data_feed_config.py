
"""
Data feed configurations for paper trading.

This module provides configurations for various data sources including:
- Databento for US equities and futures
- Tardis for cryptocurrency data
- Binance live data for real-time crypto feeds
- Interactive Brokers for multi-asset data
"""

from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.databento import DatabentoDataClientConfig
from nautilus_trader.adapters.interactive_brokers import InteractiveBrokersDataClientConfig
from nautilus_trader.adapters.tardis import TardisDataClientConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.model.identifiers import Venue


class DataFeedConfigs:
    """Provides data feed configurations for paper trading."""
    
    @staticmethod
    def get_databento_config(
        api_key: str | None = None,
        dataset: str = "GLBX.MDP3",
        symbols: list[str] | None = None,
        load_all_instruments: bool = False,
    ) -> DatabentoDataClientConfig:
        """
        Get Databento data client configuration.
        
        Parameters
        ----------
        api_key : str, optional
            The Databento API key (uses env var if None).
        dataset : str
            The Databento dataset to use.
        symbols : list[str], optional
            Specific symbols to subscribe to.
        load_all_instruments : bool
            Whether to load all available instruments.
            
        Returns
        -------
        DatabentoDataClientConfig
            The configured data client.
            
        """
        return DatabentoDataClientConfig(
            api_key=api_key,  # Uses DATABENTO_API_KEY env var if None
            http_gateway="https://hist.databento.com",
            instrument_provider=InstrumentProviderConfig(
                load_all=load_all_instruments,
                filters={"symbols": symbols} if symbols else None,
            ),
            parent_symbols={
                "ES.FUT": ["ES.c.0", "ES.c.1"],  # E-mini S&P 500
                "NQ.FUT": ["NQ.c.0", "NQ.c.1"],  # E-mini Nasdaq
                "CL.FUT": ["CL.c.0", "CL.c.1"],  # Crude Oil
                "GC.FUT": ["GC.c.0", "GC.c.1"],  # Gold
            },
        )
    
    @staticmethod
    def get_tardis_config(
        api_key: str | None = None,
        exchange: str = "binance",
        symbols: list[str] | None = None,
    ) -> TardisDataClientConfig:
        """
        Get Tardis data client configuration for historical crypto data.
        
        Parameters
        ----------
        api_key : str, optional
            The Tardis API key (uses env var if None).
        exchange : str
            The exchange to get data from.
        symbols : list[str], optional
            Specific symbols to get data for.
            
        Returns
        -------
        TardisDataClientConfig
            The configured data client.
            
        """
        return TardisDataClientConfig(
            api_key=api_key,  # Uses TARDIS_API_KEY env var if None
            base_url_http="https://api.tardis.dev/v1",
            base_url_ws="wss://api.tardis.dev/v1/stream",
            instrument_provider=InstrumentProviderConfig(
                load_all=False,
                filters={"symbols": symbols} if symbols else None,
            ),
        )
    
    @staticmethod
    def get_binance_live_config(
        testnet: bool = True,
        account_type: str = "spot",
        symbols: list[str] | None = None,
    ) -> BinanceDataClientConfig:
        """
        Get Binance live data client configuration.
        
        Parameters
        ----------
        testnet : bool
            Whether to use testnet.
        account_type : str
            The account type (spot, usdt_future, coin_future).
        symbols : list[str], optional
            Specific symbols to subscribe to.
            
        Returns
        -------
        BinanceDataClientConfig
            The configured data client.
            
        """
        from nautilus_trader.adapters.binance import BinanceAccountType
        
        account_types = {
            "spot": BinanceAccountType.SPOT,
            "usdt_future": BinanceAccountType.USDT_FUTURE,
            "coin_future": BinanceAccountType.COIN_FUTURE,
        }
        
        return BinanceDataClientConfig(
            api_key=None,  # Uses BINANCE_API_KEY env var
            api_secret=None,  # Uses BINANCE_API_SECRET env var
            account_type=account_types.get(account_type, BinanceAccountType.SPOT),
            testnet=testnet,
            instrument_provider=InstrumentProviderConfig(
                load_all=len(symbols) == 0 if symbols else False,
                filters={"symbols": symbols} if symbols else None,
            ),
        )
    
    @staticmethod
    def get_ib_config(
        gateway_host: str = "127.0.0.1",
        gateway_port: int = 4002,  # Paper trading port
        client_id: int = 1,
        load_all_instruments: bool = False,
    ) -> InteractiveBrokersDataClientConfig:
        """
        Get Interactive Brokers data client configuration.
        
        Parameters
        ----------
        gateway_host : str
            The IB Gateway host.
        gateway_port : int
            The IB Gateway port (4002 for paper, 4001 for live).
        client_id : int
            The client ID for the connection.
        load_all_instruments : bool
            Whether to load all available instruments.
            
        Returns
        -------
        InteractiveBrokersDataClientConfig
            The configured data client.
            
        """
        return InteractiveBrokersDataClientConfig(
            username=None,  # Uses IB_USERNAME env var
            password=None,  # Uses IB_PASSWORD env var
            host=gateway_host,
            port=gateway_port,
            client_id=client_id,
            gateway_start=False,  # Don't auto-start gateway
            timeout=120,
            instrument_provider=InstrumentProviderConfig(
                load_all=load_all_instruments,
            ),
        )
    
    @staticmethod
    def get_multi_venue_config(
        use_databento: bool = True,
        use_binance: bool = True,
        use_ib: bool = False,
        databento_symbols: list[str] | None = None,
        binance_symbols: list[str] | None = None,
    ) -> dict:
        """
        Get a multi-venue data configuration for paper trading.
        
        Parameters
        ----------
        use_databento : bool
            Whether to include Databento configuration.
        use_binance : bool
            Whether to include Binance configuration.
        use_ib : bool
            Whether to include Interactive Brokers configuration.
        databento_symbols : list[str], optional
            Databento symbols to subscribe to.
        binance_symbols : list[str], optional
            Binance symbols to subscribe to.
            
        Returns
        -------
        dict
            Dictionary of data client configurations.
            
        """
        configs = {}
        
        if use_databento:
            configs["DATABENTO"] = DataFeedConfigs.get_databento_config(
                symbols=databento_symbols,
            )
        
        if use_binance:
            configs["BINANCE"] = DataFeedConfigs.get_binance_live_config(
                testnet=True,
                symbols=binance_symbols,
            )
        
        if use_ib:
            configs["IB"] = DataFeedConfigs.get_ib_config(
                gateway_port=4002,  # Paper trading port
            )
        
        return configs