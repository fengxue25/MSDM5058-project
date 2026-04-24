"""
Part 2: Stationarity and Autocorrelation Analysis
ADF test, ACF/PACF plots, ARMA model fitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
import warnings
warnings.filterwarnings('ignore')


def adf_test(series, name='Series'):
    """
    Perform Augmented Dickey-Fuller test for stationarity

    H0: The series has a unit root (non-stationary)
    H1: The series is stationary

    Parameters:
    -----------
    series : pd.Series
        Time series to test
    name : str
        Name of the series for display

    Returns:
    --------
    dict : Test results
    """
    result = adfuller(series, autolag='AIC')

    output = {
        'name': name,
        'adf_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }

    print(f"\n{'='*50}")
    print(f"Augmented Dickey-Fuller Test: {name}")
    print(f"{'='*50}")
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Used Lags: {result[2]}")
    print(f"Number of Observations: {result[3]}")
    print("\nCritical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")

    if result[1] < 0.05:
        print(f"\nResult: Series is STATIONARY (p < 0.05)")
    else:
        print(f"\nResult: Series is NON-STATIONARY (p >= 0.05)")

    return output


def plot_acf_pacf(series, lags=50, title='', figsize=(14, 5)):
    """
    Plot ACF and PACF of a time series

    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    lags : int
        Number of lags to plot
    title : str
        Title prefix for plots
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ACF plot
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f'{title} Autocorrelation Function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')

    # PACF plot
    plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title(f'{title} Partial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')

    plt.tight_layout()
    return fig


def plot_acf_absolute(series, lags=50, title='', figsize=(14, 5)):
    """
    Plot ACF of absolute values (nonlinear autocorrelation)

    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    lags : int
        Number of lags to plot
    title : str
        Title prefix for plots
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    abs_series = np.abs(series)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ACF of original series (linear autocorrelation)
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f'{title} Linear Autocorrelation (ACF of X)')
    axes[0].set_xlabel('Lag')

    # ACF of absolute values (nonlinear autocorrelation)
    plot_acf(abs_series, lags=lags, ax=axes[1])
    axes[1].set_title(f'{title} Nonlinear Autocorrelation (ACF of |X|)')
    axes[1].set_xlabel('Lag')

    plt.tight_layout()
    return fig


def compare_linear_nonlinear_acf(series, lags=50):
    """
    Compare linear and nonlinear autocorrelation

    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    lags : int
        Number of lags

    Returns:
    --------
    dict : Comparison results
    """
    linear_acf = acf(series, nlags=lags)[1:]  # Exclude lag 0
    nonlinear_acf = acf(np.abs(series), nlags=lags)[1:]

    # Find significant lags
    conf_int = 1.96 / np.sqrt(len(series))

    results = {
        'linear_acf': linear_acf,
        'nonlinear_acf': nonlinear_acf,
        'conf_int': conf_int,
        'linear_significant': np.sum(np.abs(linear_acf) > conf_int),
        'nonlinear_significant': np.sum(np.abs(nonlinear_acf) > conf_int)
    }

    print(f"\nComparison of Linear vs Nonlinear Autocorrelation:")
    print(f"Significant lags (linear): {results['linear_significant']}")
    print(f"Significant lags (nonlinear): {results['nonlinear_significant']}")

    if results['nonlinear_significant'] > results['linear_significant']:
        print("\nInterpretation: Nonlinear autocorrelation is more persistent.")
        print("This suggests volatility clustering (common in financial returns).")

    return results


def select_arma_order(series, max_ar=5, max_ma=5):
    """
    Select optimal ARMA order using information criteria

    Parameters:
    -----------
    series : pd.Series
        Time series to fit
    max_ar : int
        Maximum AR order
    max_ma : int
        Maximum MA order

    Returns:
    --------
    tuple : (best_aic_order, best_bic_order)
    """
    print("\nSelecting optimal ARMA order...")

    try:
        result = arma_order_select_ic(series, max_ar=max_ar, max_ma=max_ma, ic=['aic', 'bic'])

        print(f"Best AIC order: {result.aic_min_order}")
        print(f"Best BIC order: {result.bic_min_order}")

        return result.aic_min_order, result.bic_min_order
    except Exception as e:
        print(f"Error in order selection: {e}")
        return (1, 1), (1, 1)


def fit_arma(series, order):
    """
    Fit ARMA model to the series

    Parameters:
    -----------
    series : pd.Series
        Time series to fit
    order : tuple
        (AR order, MA order)

    Returns:
    --------
    ARIMA result object
    """
    model = ARIMA(series, order=(order[0], 0, order[1]))
    result = model.fit()

    print(f"\nARMA{order} Model Summary:")
    print(result.summary().tables[1])

    return result


def compare_arma(series, guess_order, figsize=(12, 8)):
    """
    Compare guessed ARMA order with optimal order

    Parameters:
    -----------
    series : pd.Series
        Time series to fit
    guess_order : tuple
        Guessed (AR, MA) order based on ACF/PACF
    figsize : tuple
        Figure size for residual plots

    Returns:
    --------
    dict : Results dictionary
    """
    # Find optimal order
    aic_order, bic_order = select_arma_order(series)

    # Fit models
    print(f"\nFitting guessed order: {guess_order}")
    guess_model = fit_arma(series, guess_order)

    print(f"\nFitting AIC optimal order: {aic_order}")
    aic_model = fit_arma(series, aic_order)

    results = {
        'guess_order': guess_order,
        'guess_model': guess_model,
        'aic_order': aic_order,
        'aic_model': aic_model,
        'bic_order': bic_order
    }

    return results


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    n = 1000
    returns = np.random.randn(n)

    # Test ADF
    adf_test(returns, 'Random Returns')

    # Plot ACF/PACF
    plot_acf_pacf(returns, title='Random Returns')
    plt.show()
