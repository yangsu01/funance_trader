from abc import ABC, abstractmethod
import pandas as pd


class TradingRule(ABC):
    @abstractmethod
    def generate_next_forecast(self, data: pd.Series) -> float:
        """Generates a forecast for next period based on input data
            Meant to be used one step at a time for backtesting

        Args:
            data (pd.Series): time series data

        Returns:
            float: forecast for next period
        """
        raise NotImplementedError

    @abstractmethod
    def generate_forecasts(self, data: pd.DataFrame) -> pd.Series:
        """Generates a series of forecasts for historical price data.
            Forecasts are generated after sufficient data is available.

        Args:
            data (pd.DataFrame): historical price data with 'Close' column

        Returns:
            pd.Series: series of forecasts
        """
        raise NotImplementedError

    

