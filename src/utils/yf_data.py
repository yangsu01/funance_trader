from typing import List
import yfinance as yf
from pandas import DataFrame
        

def get_prices(tickers: List[str], start: str, end: str) -> DataFrame:
    """Gets historical prices for a list of tickers

    Args:
        tickers (List[str]): list of stock tickers
        start (str): start date in format 'YYYY-MM-DD'
        end (str): end date in format 'YYYY-MM-DD'

    Returns:
        DataFrame: multi-index DataFrame with historical prices
    """
    return yf.Tickers(tickers).history(start=start, end=end)