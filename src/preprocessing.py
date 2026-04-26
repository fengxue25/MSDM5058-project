"""
Part 1: Data Preprocessing
Download stock data and compute daily returns

Can be used indenpendently to only download the data
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os


def download_data(tickers, start_date, end_date, data_dir='data'):
    """
    Download stock data from Yahoo Finance

    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    data_dir : str
        Directory to save data

    Returns:
    --------
    dict : Dictionary of DataFrames with closing prices
    """
    data = {}

    for ticker in tickers:
        print(f"Downloading {ticker} data...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if len(df) < 4000:
            print(f"Warning: {ticker} has only {len(df)} days of data")

        # Save to CSV
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(f"{data_dir}/{ticker}.csv")

        data[ticker] = df['Close'].squeeze()
        print(f"{ticker}: {len(df)} days of data downloaded")

    return data


def compute_returns(prices):
    
    """
    Compute daily log returns

    X(t) = ln(S(t)/S(t-1))

    Parameters:
    -----------
    prices : pd.Series
        Time series of prices

    Returns:
    --------
    pd.Series : Daily log returns
    """
    print("Computing the log_return...")
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()


def demean(returns):
    
    """
    Subtract mean from returns

    Parameters:
    -----------
    returns : pd.Series
        Time series of returns

    Returns:
    --------
    pd.Series : Demeaned returns
    """
    mean = returns.mean()
    print(f"Mean of returns: {float(mean):.6f}")
    return returns - mean


def preprocess_data(prices,ticker,output_dir):
    
    """
    Full preprocessing pipeline

    Parameters:
    -----------
    prices : pd.Series
        Time series of prices

    Returns:
    --------
    tuple : (prices, returns, demeaned_returns)
    """
    returns = compute_returns(prices)
    demeaned_returns = demean(returns)
    returns_centered = pd.Series(demeaned_returns, index=returns.index)

    # Create aligned data for saving
    save_data = pd.DataFrame({
        'Date': prices.index,
        'Price': prices.values,
    })
    save_data['Returns'] = [np.nan] + list(returns.values)
    save_data['Returns_Centered'] = [np.nan] + list(returns_centered.values)

    # Drop NaN rows and save
    save_data.dropna().to_csv(f"{output_dir}/data/{ticker}_processed.csv", index=False)
    print(f"Saved data to csv: {output_dir}/data/{ticker}_processed.csv")

    return prices, returns, returns_centered


def plot_prices_and_returns(prices, returns, ticker_name, figsize):
    """
    Plot price and return time series

    Parameters:
    -----------
    prices : pd.Series
        Time series of prices
    returns : pd.Series
        Time series of returns
    ticker : str
        Stock ticker symbol
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Plot prices
    axes[0].plot(prices.index, prices.values, linewidth=0.8)
    axes[0].set_title(f'{ticker_name}', fontsize=14)
    axes[0].set_ylabel('Daily Closing Price S (USD)')
    axes[0].grid(True, alpha=0.3)

    # Plot returns
    axes[1].plot(returns.index, returns.values, linewidth=0.5, color='orange')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Daily Log Returns (Centered) X')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    tickers = []
    start_date = ''
    end_date = ''

    # Download or load data
    data = download_data(tickers, start_date, end_date)
