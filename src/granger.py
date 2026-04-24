"""
Part 4: Granger Causality Analysis
VARMA model fitting and F-tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')


def prepare_bivariate_series(x1, x2, name1='X1', name2='X2'):
    """
    Prepare bivariate time series for VARMA model

    Parameters:
    -----------
    x1, x2 : array-like
        Two time series
    name1, name2 : str
        Names for the series

    Returns:
    --------
    pd.DataFrame : Combined DataFrame
    """
    # Ensure same length
    min_len = min(len(x1), len(x2))
    x1 = x1[:min_len]
    x2 = x2[:min_len]

    df = pd.DataFrame({
        name1: x1,
        name2: x2
    })

    return df


def select_varma_order(data, max_ar=3, max_ma=3):
    """
    Select optimal VARMA order using information criteria

    Parameters:
    -----------
    data : pd.DataFrame
        Bivariate time series
    max_ar : int
        Maximum AR order
    max_ma : int
        Maximum MA order

    Returns:
    --------
    tuple : (best_aic_order, best_bic_order)
    """
    print("Selecting VARMA order...")

    best_aic = np.inf
    best_bic = np.inf
    best_aic_order = (1, 0)
    best_bic_order = (1, 0)

    results = []

    for ar in range(max_ar + 1):
        for ma in range(max_ma + 1):
            if ar == 0 and ma == 0:
                continue

            try:
                model = VARMAX(data, order=(ar, ma))
                result = model.fit(disp=False)

                results.append({
                    'order': (ar, ma),
                    'aic': result.aic,
                    'bic': result.bic
                })

                if result.aic < best_aic:
                    best_aic = result.aic
                    best_aic_order = (ar, ma)

                if result.bic < best_bic:
                    best_bic = result.bic
                    best_bic_order = (ar, ma)

            except Exception as e:
                continue

    print(f"Best AIC order: {best_aic_order} (AIC={best_aic:.2f})")
    print(f"Best BIC order: {best_bic_order} (BIC={best_bic:.2f})")

    return best_aic_order, best_bic_order


def varma_model(data, order=(1, 0)):
    """
    Fit VARMA model to bivariate data

    Parameters:
    -----------
    data : pd.DataFrame
        Bivariate time series
    order : tuple
        (AR order, MA order)

    Returns:
    --------
    VARMAX result object
    """
    print(f"\nFitting VARMA{order} model...")

    model = VARMAX(data, order=order)
    result = model.fit(disp=False)

    print("\n" + "="*60)
    print("VARMA Model Summary")
    print("="*60)
    print(result.summary())

    return result


def get_coefficients(result):
    """
    Extract and display regression coefficients

    Parameters:
    -----------
    result : VARMAX result
        Fitted VARMA model

    Returns:
    --------
    dict : Coefficients and their significance
    """
    params = result.params
    pvalues = result.pvalues
    stderr = result.bse

    print("\n" + "="*60)
    print("Regression Coefficients and Significance")
    print("="*60)

    print(f"\n{'Parameter':<20} {'Value':>12} {'Std Err':>12} {'p-value':>12} {'Signif':>8}")
    print("-"*64)

    for param in params.index:
        val = params[param]
        se = stderr[param]
        pval = pvalues[param]

        if pval < 0.01:
            signif = '***'
        elif pval < 0.05:
            signif = '**'
        elif pval < 0.1:
            signif = '*'
        else:
            signif = ''

        print(f"{param:<20} {val:>12.4f} {se:>12.4f} {pval:>12.4f} {signif:>8}")

    print("\nSignificance: *** p<0.01, ** p<0.05, * p<0.1")

    return {
        'params': params,
        'pvalues': pvalues,
        'stderr': stderr
    }


def granger_causality_test(data, maxlag=10, name1='X1', name2='X2'):
    """
    Perform Granger causality tests

    Parameters:
    -----------
    data : pd.DataFrame
        Bivariate time series with two columns
    maxlag : int
        Maximum lag to test
    name1, name2 : str
        Names of the series

    Returns:
    --------
    dict : Test results
    """
    print("\n" + "="*60)
    print("Granger Causality Tests")
    print("="*60)

    results = {}

    # Test if X2 Granger causes X1
    print(f"\n{name2} → {name1} (Does {name2} Granger cause {name1}?)")
    print("-"*60)
    test_1 = grangercausalitytests(data[[name1, name2]], maxlag=maxlag, verbose=True)

    # Test if X1 Granger causes X2
    print(f"\n{name1} → {name2} (Does {name1} Granger cause {name2}?)")
    print("-"*60)
    test_2 = grangercausalitytests(data[[name2, name1]], maxlag=maxlag, verbose=True)

    # Summarize results
    print("\n" + "="*60)
    print("Summary of Granger Causality Tests")
    print("="*60)

    # Extract p-values for F-tests at each lag
    pvalues_1 = []
    pvalues_2 = []

    for lag in range(1, maxlag + 1):
        pval_1 = test_1[lag][0]['ssr_ftest'][1]
        pval_2 = test_2[lag][0]['ssr_ftest'][1]
        pvalues_1.append(pval_1)
        pvalues_2.append(pval_2)

    print(f"\n{name2} → {name1}:")
    sig_lags_1 = [i+1 for i, p in enumerate(pvalues_1) if p < 0.05]
    if sig_lags_1:
        print(f"  Significant at lags: {sig_lags_1}")
        print(f"  Result: {name2} DOES Granger cause {name1} at significance level 0.05")
    else:
        print(f"  No significant causality found")

    print(f"\n{name1} → {name2}:")
    sig_lags_2 = [i+1 for i, p in enumerate(pvalues_2) if p < 0.05]
    if sig_lags_2:
        print(f"  Significant at lags: {sig_lags_2}")
        print(f"  Result: {name1} DOES Granger cause {name2} at significance level 0.05")
    else:
        print(f"  No significant causality found")

    results = {
        f'{name2}_causes_{name1}': {
            'pvalues': pvalues_1,
            'significant_lags': sig_lags_1
        },
        f'{name1}_causes_{name2}': {
            'pvalues': pvalues_2,
            'significant_lags': sig_lags_2
        }
    }

    return results


def plot_granger_results(results, name1='X1', name2='X2', figsize=(12, 5)):
    """
    Plot Granger causality test results

    Parameters:
    -----------
    results : dict
        Results from granger_causality_test()
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: X2 -> X1
    key1 = f'{name2}_causes_{name1}'
    pvalues1 = results[key1]['pvalues']
    lags = range(1, len(pvalues1) + 1)

    axes[0].bar(lags, pvalues1, color='steelblue', alpha=0.7)
    axes[0].axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('p-value')
    axes[0].set_title(f'{name2} → {name1}')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Plot 2: X1 -> X2
    key2 = f'{name1}_causes_{name2}'
    pvalues2 = results[key2]['pvalues']

    axes[1].bar(lags, pvalues2, color='darkorange', alpha=0.7)
    axes[1].axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('p-value')
    axes[1].set_title(f'{name1} → {name2}')
    axes[1].legend()
    axes[1].set_yscale('log')

    plt.tight_layout()
    return fig


def plot_irf(result, steps=20, figsize=(12, 8)):
    """
    Plot Impulse Response Functions

    Parameters:
    -----------
    result : VARMAX result
        Fitted VARMA model
    steps : int
        Number of steps for IRF
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    irf = result.impulse_responses(steps=steps)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Get variable names
    var_names = result.data.ynames

    # IRF plots
    for i, var1 in enumerate(var_names):
        for j, var2 in enumerate(var_names):
            ax = axes[i, j]
            response = irf.iloc[:, i*len(var_names) + j]
            ax.bar(range(len(response)), response, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title(f'Response of {var1} to {var2} shock')
            ax.set_xlabel('Period')
            ax.set_ylabel('Response')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    n = 1000

    # Generate correlated series
    e1 = np.random.randn(n)
    e2 = np.random.randn(n)

    x1 = np.zeros(n)
    x2 = np.zeros(n)

    for t in range(1, n):
        x1[t] = 0.5 * x1[t-1] + e1[t]
        x2[t] = 0.3 * x2[t-1] + 0.2 * x1[t-1] + e2[t]  # X1 Granger causes X2

    data = prepare_bivariate_series(x1, x2, 'X1', 'X2')

    # Select order
    best_aic, best_bic = select_varma_order(data, max_ar=3, max_ma=3)

    # Fit model
    result = varma_model(data, order=best_bic)

    # Get coefficients
    coeffs = get_coefficients(result)

    # Granger causality
    gc_results = granger_causality_test(data, maxlag=10)
