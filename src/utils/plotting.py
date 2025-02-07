from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def plot_time_series(
    data: pd.Series,
    title: str,
    ylabel: str='Price ($)',
    xlabel: str='Date',
) -> None:
    """ Plot time series data

    Args:
        data (pd.Series): series of dates and values
        title (str): title of the plot
        ylabel (str, optional): y axis label. Defaults to 'Price ($)'.
        xlabel (str, optional): x axis label. Defaults to 'Date'.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    plt.show()


def plot_distribution(
    data: pd.Series,
    title: str,
    dist: str='norm',
    hist_bins: int=50,
    hist_title: str='Returns Distribution',
    hist_xlabel: str='Returns',
) -> None:
    """ Plot histogram/KDE and normal QQ plot of data

    Args:
        data (pd.Series): data to plot
        title (str): title of the plot
        distribution (str, optional): distribution to compare to (norm, t). Defaults to 'norm'.
        hist_bins (int, optional): number of bins for the histogram. Defaults to 50.
    """
    kde = stats.gaussian_kde(data)
    x_values = np.linspace(min(data), max(data), 1000)
    kde_values = kde(x_values)
    
    if dist == 'norm':
        mean = np.mean(data)
        std = np.std(data)
        values = stats.norm.pdf(x_values, mean, std)
        title_str = title + '\n' + rf' $\mu={mean:.4f}$, $\sigma={std:.4f}$'
    
    elif dist == 't':
        df, loc, scale = stats.t.fit(data)
        values = stats.t.pdf(x_values, df, loc, scale)
        title_str = title + '\n' + rf'$\nu={df:.4f}$, $\mu={loc:.4f}$, $\lambda={scale:.4f}$'
        
    else:
        raise ValueError('Distribution must be "norm" or "t"')

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # histogram and KDE
    axs[0].hist(data, bins=hist_bins, density=True, alpha=0.8)
    axs[0].plot(x_values, values, color='blue', linestyle='--', label=dist)
    axs[0].plot(x_values, kde_values, color='red', linestyle='-.', label='KDE')
    axs[0].set_xlabel(hist_xlabel)
    axs[0].set_ylabel('Density')
    axs[0].set_title(hist_title)
    axs[0].legend()
    
    # QQ plot
    if dist == 'norm':
        stats.probplot(data, plot=axs[1], dist='norm', sparams=(mean, std))
        axs[1].set_title('Normal Distribution QQ Plot')
    elif dist == 't':
        stats.probplot(data, plot=axs[1], dist='t', sparams=(df, loc, scale)) 
        axs[1].set_title('t-Distribution QQ Plot')

    fig.suptitle(title_str)
    fig.tight_layout()