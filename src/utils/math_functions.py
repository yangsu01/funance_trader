from scipy.stats import gmean
from typing import Dict
from pandas import Series
import numpy as np

def calculate_returns(price_data: Series) -> Dict[str, float]:
    """Calculates returns from daily price data

    Args:
        price_data (pd.Series): daily prices
    Returns:
        Dict[str, float]: dictionary of daily and annualized returns
    """
    daily_returns = price_data.pct_change().dropna()
    daily_mean = gmean(daily_returns + 1) - 1
    annual_mean = (daily_mean + 1) ** 252 - 1
    
    return {
        'daily': np.round(daily_mean, 4),
        'annual': np.round(annual_mean, 4)
    }
    