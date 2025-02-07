import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

from ..trading_rules import TradingRule
from ..trading_rules.indicators import EWMA

class BacktestTrader:
    def __init__(
        self,
        rule: TradingRule,
        data: pd.DataFrame,
        cash: float=10000.0,
        commission: float=0.0,
        fee_type: str='percent',
        volatility_target: Optional[float]=None,
        vol_window: int=36
    ):  
        """Initializes a BacktestTrader object

        Args:
            rule (TradingRule): trading rule to backtest
            data (pd.DataFrame): multi-index time series data of price and ticker. Contains at least open, close.
            cash (float, optional): starting trading capital. Defaults to 10000.0.
            commission (float, optional): commission fee charged per trade. Defaults to 0.0.
            fee_type (str, optional): type of fee. 'percent' or 'flat'. Defaults to 'percent'.
            volatility_target (float, optional): Annualized target volatility of portfolio. Defaults to 0.20.
            
        Raises:
            ValueError: fee_type must be either 'percent' or 'flat'
        """
        if fee_type not in ['percent', 'flat']:
            raise ValueError("fee_type must be either 'percent' or 'flat'")
        
        self.rule = rule
        self.data = data
        self.cash = cash
        self.commission = commission
        self.commission_type = fee_type
        self.vol_target = volatility_target
        self.cash_history = pd.Series(dtype=float)
        self.value_history = pd.Series(dtype=float)
        self.trade_history = pd.DataFrame(
            index=pd.to_datetime([]),
            columns=['ticker', 'action', 'price', 'size']
        )
        self.instrument_returns = pd.DataFrame(
            index=pd.to_datetime([]),
            columns=self.data.columns.get_level_values('Ticker').unique()
        )
        self.order = None
        self.positions = {
            x: 0 for x in self.data.columns.get_level_values('Ticker').unique()
        }
        self.timestamp = None
        self.vol_ewma = EWMA(vol_window)
        self.last_price = None
        self.enough_data = False
    
    
    def _log(self, message: str, timestamp: datetime=None):
        """Logs a message with a timestamp

        Args:
            timestamp (datetime): timestamp of message
            message (str): message to log
        """
        ts = timestamp or self.timestamp
        print(f'{ts}: {message}')
    
    
    def _submit_order(self, positions: Dict[str, int], log: bool):
        """Records an order to be executed

        Args:
            positions (Dict[str, int]): dictionary of ticker and how many units to buy/sell
            log (bool): whether to log the order
        """
        self.order = positions

        if log:
            self._log(
                f'Order submitted.'
            )
            
    
    def _calculate_fee(self, price: float, size: int) -> float:
        """Calculates the fee for a trade

        Args:
            price (float): price of the trade
            size (int): number of units traded

        Returns:
            float: fee for the trade
        """
        if self.commission_type == 'percent':
            return np.abs(self.commission*price*size)
        else:
            return self.commission
    
    
    def _record_trade(self, ticker: str, action: str, price: float, size: int):
        """Records a trade

        Args:
            ticker (str): ticker of the trade
            action (str): action of the trade
            price (float): price of the trade
            size (int): number of units traded
        """
        self.trade_history.loc[self.timestamp] = [ticker, action, price, size]
    
    
    def _execute_trade(self, order: dict, prices: pd.Series, log: bool):
        """Executes a trade

        Args:
            order (dict): dictionary of ticker and how many units to buy/sell
            prices (pd.Series): prices of the trade
            log (bool): whether to log the trade

        """
        for ticker, size in order.items():
            price = prices[ticker]
            
            # buy transaction
            if size > 0:
                    
                # calculate affordable amount
                if self.commission_type == 'percent':
                    max_size = int(self.cash/(price * (1+self.commission)))
                else:
                    max_size = int(self.cash/(price + self.commission))
                
                fill_size = min(max_size, size)
                
                # execute trade
                if fill_size > 0:
                    fee = self._calculate_fee(price, fill_size)
                    
                    # log trade
                    if log:
                        self._log(
                            f'Bought {fill_size}/{size} of {ticker} at ${price}, '
                            f'commission: ${fee}'
                        )
                    
                    # update update records
                    self.cash -= fill_size * price + fee
                    self.positions[ticker] += fill_size
                    self._record_trade(ticker, 'buy', price, fill_size)
                
            # sell transaction
            elif size < 0:
                # calculate sellable size (assumes no shorting)
                fill_size = min(np.abs(self.positions[ticker]), np.abs(size))
                
                if fill_size > 0:
                    fee = self._calculate_fee(price, fill_size)
                    
                    # log trade
                    if log:
                        self._log(
                            f'Sold {fill_size}/{self.positions[ticker]} of held {ticker} at ${price}, '
                            f'commission: ${fee}'
                        )
                    
                    # update records
                    self.cash += fill_size * price - fee
                    self.positions[ticker] -= fill_size
                    self._record_trade(ticker, 'sell', price, fill_size)

    
    def _update_history(self, prices: pd.Series):
        """Updates the history of the trader

        Args:
            prices (pd.Series): prices of the trade
        """
        # cash
        self.cash_history[self.timestamp] = self.cash

        # portfolio value
        value = self.cash + sum(
            prices.loc[x] * self.positions[x] for x in self.positions.keys()
        )
        self.value_history[self.timestamp] = value
        
    
    def _forecasts_to_positions(self, forecasts: Dict[str, float], prices: pd.Series) -> Dict[str, int]:
        """Generates a position based on scaled forecasts and position size based on volatility targeting
        
        Args:
            forecasts (Dict[str, float]): dictionary of ticker and scaled forecasts
            prices (pd.Series): current prices for each ticker
            
        Returns:
            Dict[str, int]: dictionary of ticker and number of shares to buy/sell to reach target positions
        """
        target_positions = self.positions.copy()

        if self.last_price is not None:
            # calculate current portfolio value
            current_value = self.cash + sum(
                self.data.loc[self.timestamp, ('Close', ticker)] * self.positions[ticker] 
                for ticker in self.positions.keys()
            )
            
            for ticker in forecasts.keys():
                # log returns for current day
                returns = np.log(prices[ticker] / self.last_price[ticker])
                self.instrument_returns.loc[self.timestamp, ticker] = returns
                # update rolling variance estimate
                price_var = self.vol_ewma.update(returns**2, self.timestamp) 
                # convert to volatility and set minimum
                price_vol = max(np.sqrt(price_var), 0.001)

                # if volatility target is provided, calculate position size to achieve target volatility
                if self.vol_target is not None:
                    position_size = (self.vol_target / np.sqrt(252)) * (current_value / (price_vol * prices[ticker]))
                    target_positions[ticker] = int(position_size * forecasts[ticker])
                else:
                    if forecasts[ticker] < 0:
                        target_positions[ticker] = -self.positions[ticker]  # sell all
                    elif forecasts[ticker] > 0:
                        target_positions[ticker] = int(self.cash / prices[ticker])  # buy all

        shares_to_trade = {
            ticker: target_positions[ticker] - self.positions[ticker]
            for ticker in target_positions.keys()
        }
        # update last price
        self.last_price = prices
        return shares_to_trade
    
    
    def run_backtest(self, log: bool=False):
        """Runs a backtest

        Args:
            log (bool, optional): whether to log the backtest. Defaults to False.
        """
        if log:
            self._log(
                f'Backtesting... \n'
                f'Starting Value: ${self.cash}'
            )
            
        for i in range(len(self.data)):
            self.timestamp = self.data.index[i]
            bar = self.data.iloc[i]
            
            # check to resolve any open orders
            if self.order is not None:      
                self._execute_trade(
                    self.order,
                    bar['Open'],
                    log
                )
            
            # get position from trading rule
            forecasts = self.rule.generate_next_forecast(bar)
            positions = self._forecasts_to_positions(forecasts, bar['Close'])
            
            # submit order if non zero positions
            if any(position !=0 for position in positions.values()):
                self._submit_order(
                    positions,
                    log
                )
                # set enough data to true after first non zero forecast
                self.enough_data = True

            # update history
            if self.enough_data:
                self._update_history(bar['Close'])
        
        if log:
            self._log(
                f'Ending Value: ${self.value_history.iloc[-1]} \n' if len(self.value_history) > 0 else 'No trades were made. \n'
                f'Backtest complete. \n'
            )
    
    
    def plot_backtest(self):
        pass
    
    
    def plot_forecasts(self):
        pass
    
    
    def get_analysis(self) -> Dict:
        return {
            'cash_history': self.cash_history,
            'value_history': self.value_history,
            'trade_history': self.trade_history,
            'instrument_returns': self.instrument_returns
        }