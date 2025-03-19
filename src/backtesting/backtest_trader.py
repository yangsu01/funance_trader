import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from matplotlib import pyplot as plt

from ..trading_rules import TradingRule
from ..trading_rules.indicators import EWMA
from ..utils import calculate_returns

class BacktestTrader:
    def __init__(
        self,
        rule: TradingRule,
        data: pd.DataFrame,
        cash: Optional[float]=10000.0,
        commission: Optional[float]=0.0,
        fee_type: str='percent',
        volatility_target: Optional[float]=None,
        vol_window: int=35
    ):  
        """Initializes a BacktestTrader object

        Args:
            rule (TradingRule): trading rule to backtest
            data (pd.DataFrame): multi-index time series data of price and ticker. Contains at least open, close.
            cash (float, optional): starting trading capital. Defaults to 10000.0.
            commission (float, optional): commission fee charged per trade. Defaults to 0.0.
            fee_type (str, optional): type of fee. 'percent' or 'flat'. Defaults to 'percent'.
            volatility_target (float, optional): Annualized target volatility of portfolio. Defaults to 0.20.
            vol_window (int, optional): Window size for volatility calculation. Defaults to 35.
            
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
        # Convert annual volatility target to daily
        self.vol_target = None if volatility_target is None else volatility_target/np.sqrt(252)
        self.cash_history = pd.Series(dtype=float)
        self.value_history = pd.Series(dtype=float)
        self.trade_history = pd.DataFrame(
            index=pd.to_datetime([]),
            columns=['ticker', 'action', 'price', 'size', 'fee', 'profit_loss']
        )
        self.instrument_returns = pd.DataFrame(
            index=pd.to_datetime([]),
            columns=self.data.columns.get_level_values('Ticker').unique()
        )
        self.order = None
        self.positions = {
            x: 0 for x in self.data.columns.get_level_values('Ticker').unique()
        }
        self.average_prices = {
            x: 0 for x in self.data.columns.get_level_values('Ticker').unique()
        }
        self.timestamp = None
        self.vol_ewma = {
            ticker: EWMA(vol_window) for ticker in self.data.columns.get_level_values('Ticker').unique()
        }
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
            message = []
            for ticker, size in self.order.items():
                action = "buy" if size > 0 else "sell"
                message.append(f"{action} {abs(size)} shares of {ticker}")
            self._log(
                f'Order submitted: {", ".join(message)}'
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
    
    
    def _record_trade(
        self,
        ticker: str,
        action: str,
        price: float,
        size: int,
        fee: float,
        profit_loss: float=0
    ):
        """Records a trade

        Args:
            ticker (str): ticker of the trade
            action (str): action of the trade
            price (float): price of the trade
            size (int): number of units traded
            fee (float): fee of the trade
            profit_loss (float, optional): profit or loss of the trade. Defaults to None for buys.
        """
        self.trade_history.loc[self.timestamp] = [
            ticker,
            action,
            np.round(price, 2),
            size,
            np.round(fee, 2),
            np.round(profit_loss, 2)
        ]
    
    
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
                    max_size = int((self.cash - self.commission)/price)
                
                fill_size = min(max_size, size)
                
                # execute trade
                if fill_size > 0:
                    fee = self._calculate_fee(price, fill_size)
                    
                    # log trade
                    if log:
                        self._log(
                            f'Bought {fill_size}/{size} of {ticker} at ${np.round(price, 2)}, '
                            f'commission: ${np.round(fee, 2)}'
                        )
                    
                    # update cash, average price, and positions
                    self.cash -= fill_size * price + fee
                    self.average_prices[ticker] = (
                        (self.average_prices[ticker]*self.positions[ticker] + fill_size*price) / (self.positions[ticker] + fill_size)
                    )
                    self.positions[ticker] += fill_size
                    
                    self._record_trade(ticker, 'buy', price, fill_size, fee)
                
            # sell transaction
            elif size < 0:
                # calculate sellable size (assumes no shorting)
                fill_size = min(np.abs(self.positions[ticker]), np.abs(size))
                
                if fill_size > 0:
                    fee = self._calculate_fee(price, fill_size)
                    
                    # log trade
                    if log:
                        self._log(
                            f'Sold {fill_size}/{self.positions[ticker]} of held {ticker} at ${np.round(price, 2)}, '
                            f'commission: ${np.round(fee, 2)}'
                        )
                    
                    # update records
                    self.cash += fill_size * price - fee
                    self.positions[ticker] -= fill_size
                    profit_loss = fill_size * (price - self.average_prices[ticker])
                    self._record_trade(ticker, 'sell', price, fill_size, fee, profit_loss)

    
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
        
    
    def _forecasts_to_positions(self, forecasts: Dict[str, float], prices: pd.Series, position_margin: float) -> Dict[str, int]:
        """Generates a position based on scaled forecasts and position size based on volatility targeting
        
        Args:
            forecasts (Dict[str, float]): dictionary of ticker and scaled forecasts (-20 to +20)
            prices (pd.Series): current prices for each ticker
            position_margin (float): minimum position change threshold as fraction of current position
            
        Returns:
            Dict[str, int]: dictionary of ticker and number of shares to buy/sell to reach target positions
        """
        positions_to_trade = {ticker: 0 for ticker in forecasts.keys()}
        portfolio_value = self.cash + sum(
            prices.loc[x] * self.positions[x] for x in self.positions.keys()
        )

        if self.last_price is not None:
            for ticker in forecasts.keys():
                # Calculate % price volatility using log returns
                returns = np.log(prices[ticker] / self.last_price[ticker])
                self.instrument_returns.loc[self.timestamp, ticker] = returns
                price_var = self.vol_ewma[ticker].update(returns**2, self.timestamp)
                price_vol = max(np.sqrt(price_var), 0.001)  # Minimum vol floor

                # Position sizing based on volatility targeting
                if self.vol_target is not None:
                    # Calculate instrument currency volatility 
                    instrument_cash_vol = price_vol * prices[ticker]
                    
                    # Calculate position multiplier based on target risk
                    # Position multiplier = target risk / instrument risk
                    # Where target risk = portfolio value * target vol
                    # And instrument risk = price * price vol
                    position_multiplier = (portfolio_value * self.vol_target) / instrument_cash_vol
                    
                    # Scale forecast from -20 to +20 range to actual position
                    # Forecast of 10 = 100% of base position
                    target_position = int(position_multiplier * (forecasts[ticker] / 10))

                else:
                    # Simple long/short without volatility targeting
                    if forecasts[ticker] < 0:
                        target_position = 0  # sell all
                    elif forecasts[ticker] > 0:
                        target_position = int(self.cash / prices[ticker]) + self.positions[ticker]  # buy all
                    else:
                        target_position = self.positions[ticker]  # hold
                
                # Calculate position change needed
                position_adjustment = target_position - self.positions[ticker]
                adjustment_size = np.abs(position_margin * self.positions[ticker])

                # Only trade if adjustment exceeds minimum size
                if np.abs(position_adjustment) > adjustment_size:
                    positions_to_trade[ticker] = position_adjustment

        # Store prices for next volatility calculation
        self.last_price = prices
        return positions_to_trade
    
    
    def run(self, position_margin: float=0.0, log: bool=False):
        """Runs a backtest

        Args:
            position_margin (float, optional): current position will not be adjusted 
                if the trade size is less than position_margin * current position. Defaults to 0.0.
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
                self.order = None
            
            # get position from trading rule
            forecasts = self.rule.generate_next_forecast(bar)
            positions = self._forecasts_to_positions(forecasts, bar['Close'], position_margin)
            
            # submit order if non zero positions
            if any(position != 0 for position in positions.values()):
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
    
    
    def plot(self):
        _, axs = plt.subplots(
            4,
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1, 3, 1]},
            figsize=(16, 16)
        )
        plt.subplots_adjust(hspace=0.05) # adjust vertical gap
        start_date = self.value_history.index[0]
        asset_prices = self.data['Close'].loc[start_date:]
        
        # instrument price and indicators
        indicators = self.rule.get_plot_data()
        axs[0].plot(asset_prices, label='Asset Price')
        for key, val in indicators.items():
            axs[0].plot(val.loc[start_date:], label=key)
        
        axs[0].set_ylabel('Price')
        axs[0].legend(loc='upper left')
        
        # trades
        trades = self.trade_history[self.trade_history['action'] == 'sell']
        colors = np.where(
            trades['profit_loss'] >= 0,
            'g',
            'r'
        )
        axs[1].scatter(
            trades.index,
            trades['profit_loss'],
            s=10,
            color=colors
        )
        
        axs[1].set_ylabel('Profit / Loss')
        
        # cumulative returns
        cum_asset_ret = asset_prices/asset_prices.iloc[0] - 1
        cum_strategy_ret = self.value_history/self.value_history.iloc[0] - 1
        axs[2].plot(cum_asset_ret, label='Asset Returns')
        axs[2].plot(cum_strategy_ret, label='Strategy Returns')
        
        axs[2].set_ylabel('Cumulative Returns')
        axs[2].legend(loc='upper left')

        # cash and value
        axs[3].plot(self.cash_history, label='Cash')
        axs[3].plot(self.value_history, label='Portfolio Value')
        
        axs[3].set_ylabel('Value')
        axs[3].legend(loc='upper left')
        
        # title
        asset_returns = calculate_returns(asset_prices)['annual'][0] # TODO fix this for multiple assets
        strategy_returns = calculate_returns(self.value_history)['annual']
        plt.suptitle(
            'Backtest Results \n' + 
            f'Annual Asset Returns: {asset_returns*100: .2f}% \n' +
            f'Annual Backtest Returns: {strategy_returns*100: .2f}%',
            fontsize=24
        )

        plt.show()
    
    
    def analysis(self) -> Dict:
        return {
            'cash_history': self.cash_history,
            'value_history': self.value_history,
            'trade_history': self.trade_history
        }