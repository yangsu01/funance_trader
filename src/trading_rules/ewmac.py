import pandas as pd
from typing import Dict

from .trading_rule import TradingRule
from .indicators import EWMA

class EWMACRule(TradingRule):
    def __init__(
        self,
        short_window: int,
        long_window: int,
        initial_position: int=0,
        scale_factor: float=1.0,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.position = initial_position
        self.scale_factor = scale_factor
        self.forecasts = pd.Series(dtype=float)
        self.short_ewma = EWMA(short_window)
        self.long_ewma = EWMA(long_window)
    
    def generate_positions(self, data: pd.Series) -> float:
        """Generates sized positions based on current given price data
        
        Args:
            data (pd.Series): multi-index time series of price and ticker. Contains at least 'Close'

        Returns:
            float: _description_
        """
        pass

    def generate_forecast(self, data: pd.Series) -> float:
        price = data['Close'].iloc[0]
        timestamp = data.name
        
        # update EWMA with new data
        short_ewma = self.short_ewma.update(price, timestamp)
        long_ewma = self.long_ewma.update(price, timestamp)
        
        # only generate forecast if enough data
        if self.long_ewma.check_enough_data():
            forecast = short_ewma - long_ewma
        else:
            forecast = 0.0
        
        scaled_forecast = forecast*self.scale_factor
        
        self.forecasts.loc[timestamp] = scaled_forecast
        return scaled_forecast
        
    
    def get_plot_data(self) -> Dict[str, pd.Series]:
        return {
            f'{self.short_window} Day EWMA': self.short_ewma.get_history(),
            f'{self.long_window} Day EWMA': self.long_ewma.get_history(),
        }
    
    def get_forecasts(self) -> pd.Series:
        return self.forecasts