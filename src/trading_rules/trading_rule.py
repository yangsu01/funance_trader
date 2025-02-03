from abc import ABC, abstractmethod
import pandas as pd


class TradingRule(ABC):
    @abstractmethod
    def generate_forecast(self, data: pd.DataFrame) -> float:
        """Generates a forecast for next period based on input data
            Forecast is a value between -20 and 20 with mean(abs(forecast)) = 10
            With -20 representing a strong sell signal, 20 a strong buy signal, and 0 being neutral

        Args:
            data (pd.DataFrame): time series data 

        Returns:
            float: forecast for next period
        """
        raise NotImplementedError

    @abstractmethod
    def generate_positions(self, forecast: int, cash: float, price: float) -> dict:
        """Determines the size of the position to take based on forecast, cash, and price

        Args:
            forecast (int): forecast for next period
            cash (float): amount of cash available
            price (float): price of asset

        Returns:
            int: number of shares to buy (positive integer) or sell (negative integer)
        """
        raise NotImplementedError