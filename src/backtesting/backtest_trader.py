import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import gmean
from typing import Tuple, List, Dict

from ..trading_rules import TradingRule

class BacktestTrader:
    def __init__(
        self,
        rule: TradingRule,
        data: pd.DataFrame,
        cash: float=10000.0,
        commission: float=0.0,
        fee_type: str='percent',
    ):  
        """Initializes a BacktestTrader object

        Args:
            rule (TradingRule): trading rule to backtest
            data (pd.DataFrame): multi-index time series data of price and ticker. Contains at least open, close.
            cash (float, optional): starting trading capital. Defaults to 10000.0.
            commission (float, optional): commission fee charged per trade. Defaults to 0.0.
            fee_type (str, optional): type of fee. 'percent' or 'flat'. Defaults to 'percent'.

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
        self.cash_history = pd.Series(dtype=float)
        self.value_history = pd.Series(dtype=float)
        self.trade_history = pd.DataFrame(
            index=pd.to_datetime([]),
            columns=['ticker', 'action', 'price', 'size']
        )
        self.order = None
        self.positions = {
            x: 0 for x in self.data.columns.get_level_values('ticker').unique()
        }
        self.timestamp = None
    
    def _log(self, message: str, timestamp: datetime=None):
        """Logs a message with a timestamp

        Args:
            timestamp (datetime): timestamp of message
            message (str): message to log
        """
        ts = timestamp or self.timestamp
        print(f'{ts}: {message}')
    
    
    def _submit_order(self, positions: Dict[str, int], log: bool):
        self.order = positions
        
        if log:
            self._log(
                f'Order submitted.'
            )
            
    
    def _calculate_fee(self, price: float, size: int) -> float:
        if self.commission_type == 'percent':
            return np.abs(self.commission*price*size)
        else:
            return self.commission
    
    
    def _record_trade(self, ticker: str, action: str, price: float, size: int):
        self.trade_history = self.trade_history.loc[self.timestamp] = [
            {
                'ticker': ticker,
                'action': action,
                'price': price,
                'size': size
            },
        ]
    
    
    def _execute_trade(self, order: dict, prices: pd.Series, log: bool):
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
        # cash
        self.cash_history[self.timestamp] = self.cash
        
        # portfolio value
        value = self.cash + sum(
            prices.loc[x] * self.positions[x] for x in self.positions.keys()
        )
        self.value_history[self.timestamp] = value
    
    
    def run_backtest(self, log: bool=False):
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
            positions = self.rule.generate_positions(bar)
            
            # submit order if non zero positions
            if any(position !=0 for position in positions.values()):
                self._submit_order(
                    positions,
                    log
                )
            
            # update history
            self._update_history(bar['Close'])
        
        if log:
            self._log(
                f'Ending Value: ${self.value_history.iloc[-1]} \n'
                f'Backtest complete. \n'
            )
            
    
    def run_bootstrap(self):
        pass
    
    
    def plot_backtest(self):
        pass
    
    
    def plot_forecasts(self):
        pass
    
    
    def plot_bootstrap(self):
        pass
    
    
    def get_analysis(self) -> Dict:
        pass