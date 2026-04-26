"""
Part 2: Stationarity and Autocorrelation Analysis
ADF test, ACF/PACF plots, ARMA model fitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

def compare_arma_guess_with_best(guess_order, results_df):
    """
    Compare manually guessed ARMA order with best fitted order.

    Parameters
    ----------
    guess_order : tuple
        (p, q) guessed from ACF/PACF
    results_df : pd.DataFrame
        Output from arma_order_heatmap()

    Returns
    -------
    dict
    """
    best_row = results_df.iloc[0]
    best_order = (int(best_row['p']), int(best_row['q']))

    match = guess_order == best_order

    guess_row = results_df[
        (results_df['p'] == guess_order[0]) &
        (results_df['q'] == guess_order[1])
    ]

    if not guess_row.empty:
        guess_rank = int(guess_row.iloc[0]['rank'])
        guess_value = float(guess_row.iloc[0]['criterion'])
    else:
        guess_rank = None
        guess_value = None

    return {
        'guess_order': guess_order,
        'best_order': best_order,
        'match': match,
        'guess_rank': guess_rank,
        'guess_criterion': guess_value,
        'best_criterion': float(best_row['criterion'])
    }
    
def arma_order_heatmap(series, max_ar=9, max_ma=9, criterion='aic', title=''):
    """
    Fit ARMA(p,q) models on a grid and plot a heatmap of AIC/BIC.
    Each cell is annotated by rank: 1 = best fit, 2 = second best, etc.

    Parameters
    ----------
    series : array-like
        Input time series
    max_ar : int
        Maximum AR order p
    max_ma : int
        Maximum MA order q
    criterion : str
        'aic' or 'bic'
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
    results_df : pd.DataFrame
        Table of all fitted orders and criterion values
    """
    x = pd.Series(series).dropna().astype(float)

    value_matrix = np.full((max_ar + 1, max_ma + 1), np.nan)
    records = []

    for p in range(max_ar + 1):
        for q in range(max_ma + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(x, order=(p, 0, q))
                fitted = model.fit()

                value = fitted.aic if criterion.lower() == 'aic' else fitted.bic
                value_matrix[p, q] = value

                records.append({
                    'p': p,
                    'q': q,
                    'criterion': value
                })
            except Exception:
                continue

    results_df = pd.DataFrame(records)

    if results_df.empty:
        raise ValueError("No ARMA model could be fitted on the specified grid.")

    results_df = results_df.sort_values('criterion', ascending=True).reset_index(drop=True)
    results_df['rank'] = np.arange(1, len(results_df) + 1)

    rank_matrix = np.full((max_ar + 1, max_ma + 1), '', dtype=object)
    top_n = 10
    top_results = results_df[results_df['rank'] <= top_n]
    for _, row in top_results.iterrows():
        rank_matrix[int(row['p']), int(row['q'])] = str(int(row['rank']))

    best_row = results_df.iloc[0]
    best_order = (int(best_row['p']), int(best_row['q']))

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        value_matrix,
        annot=rank_matrix,
        fmt='',
        cmap='turbo',
        cbar_kws={'label': criterion.upper()},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    ax.set_xlabel('MA(q)')
    ax.set_ylabel('AR(p)')
    if title:
        ax.set_title(f'{title} ARMA Order Selection ({criterion.upper()})')
    else:
        ax.set_title(f'ARMA Order Selection ({criterion.upper()})')

    # 高亮最优模型
    best_p, best_q = best_order
    rect = plt.Rectangle((best_q, best_p), 1, 1, fill=False, edgecolor='black', lw=2.5)
    ax.add_patch(rect)

    plt.tight_layout()
    return fig, results_df

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
    axes[0].set_title(f'{title} \nAutocorrelation Function (ACF)')
    axes[0].set_xlabel('Lag (day)')
    axes[0].set_ylabel('ACF')

    # PACF plot
    plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title(f'{title} \nPartial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')

    for ax in axes:
        # Get all bar patches (ACF/PACF are usually bar plots)
        if len(ax.patches) == 0:
            continue
        heights = [patch.get_height() for patch in ax.patches]
        if heights:
            max_val = max(heights)
            min_val = min(heights)
            # Add 10% margin
            margin = max(0.05, (max_val - min_val) * 0.1)
            y_min = min_val - margin
            y_max = max_val + margin
            # Ensure at least a small range to avoid flat lines
            if y_max - y_min < 0.05:
                y_min = -0.1
                y_max = 0.1
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig


def plot_acf_absolute(series, lags=50, title='', figsize=(10, 6)):
    """
    Plot ACF of X and |X| on the same figure (linear vs nonlinear autocorrelation).

    Parameters:
    -----------
    series : pd.Series or array-like
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

    # Convert to numpy array if needed
    x = np.asarray(series).flatten()
    abs_x = np.abs(x)
    n = len(x)

    # Compute ACF (including lag 0)
    acf_x = acf(x, nlags=lags, fft=True)
    acf_abs = acf(abs_x, nlags=lags, fft=True)

    lags_range = np.arange(lags + 1)
    
    # Confidence interval: ±1.96/√N
    conf_band = 1.96 / np.sqrt(n)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot linear ACF (X)
    ax.plot(lags_range, acf_x, 'o-', color='steelblue', marker='o',
            markersize=6, linewidth=1.5, label='Linear (ACF of X)')

    # Plot nonlinear ACF (|X|)
    ax.plot(lags_range, acf_abs, 's-', color='darkorange', marker='s',
            markersize=6, linewidth=1.5, label='Nonlinear (ACF of |X|)')

    # Add confidence bands (dashed lines)
    ax.axhline(y=conf_band, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=f'±1.96/√N = ±{conf_band:.3f}')
    ax.axhline(y=-conf_band, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Auto‑adjust y‑axis limits based on data, with a symmetric 10% margin
    all_vals = np.concatenate([acf_x, acf_abs])
    y_min, y_max = all_vals.min(), all_vals.max()
    margin = max(0.05, (y_max - y_min) * 0.1)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Labels and title
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title(f'{title} \nLinear vs Nonlinear Autocorrelation', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

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
    print(f"\n{'='*60}")
    print(f"ARMA Order Comparison")
    print(f"{'='*60}")
    print(f"Guessed order based on ACF/PACF plots: {guess_order}")
    print(f"AIC optimal order: {aic_order}")
    print(f"BIC optimal order: {bic_order}")

    # Fit models
    print(f"\nFitting guessed order: {guess_order}")
    guess_model = fit_arma(series, guess_order)

    print(f"\nFitting AIC optimal order: {aic_order}")
    aic_model = fit_arma(series, aic_order)
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    if guess_order == aic_order:
        print(f"✓ Guessed order matches AIC optimal order.")
    else:
        print(f"✗ Guessed order ({guess_order}) differs from AIC optimal ({aic_order}).")
        print(f"  Model selection criteria should be prioritized over visual guess.")
        print(f"  However, examine parameter significance: often lower-order models are preferred.")
    
    if guess_order == bic_order:
        print(f"✓ Guessed order matches BIC optimal order.")
    else:
        print(f"✗ Guessed order ({guess_order}) differs from BIC optimal ({bic_order}).")

    results = {
        'guess_order': guess_order,
        'guess_model': guess_model,
        'aic_order': aic_order,
        'aic_model': aic_model,
        'bic_order': bic_order
    }

    return results

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
