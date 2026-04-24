"""
Part 3: Fractal Behavior Analysis
Hurst Exponent, DFA, Multifractal Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

try:
    import nolds
    NOLDS_AVAILABLE = True
except Exception:
    NOLDS_AVAILABLE = False
    print("Warning: nolds not available. Using built-in Hurst calculation.")

try:
    from MFDFA import MFDFA
    MFdfa_AVAILABLE = True
except ImportError:
    MFdfa_AVAILABLE = False
    print("Warning: MFDFA not installed. Some functions may not work.")


def hurst_exponent(series, max_lag=None, method='rs'):
    """
    Compute Hurst exponent using R/S analysis

    Parameters:
    -----------
    series : array-like
        Time series
    max_lag : int
        Maximum lag/window size
    method : str
        'rs' for R/S analysis, 'nolds' for nolds library

    Returns:
    --------
    dict : Hurst exponent and related results
    """
    series = np.array(series).flatten()
    n = len(series)

    if max_lag is None:
        max_lag = n // 10

    if method == 'nolds' and NOLDS_AVAILABLE:
        H = nolds.hurst_rs(series)
        print(f"Hurst Exponent (nolds): H = {H:.4f}")
        return {'H': H, 'method': 'nolds'}

    # R/S Analysis implementation
    lags = np.unique(np.logspace(np.log10(10), np.log10(max_lag), 30).astype(int))
    rs_values = []

    for lag in lags:
        # Split series into chunks
        n_chunks = n // lag
        if n_chunks < 1:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = series[i*lag:(i+1)*lag]

            # Calculate R/S
            mean_chunk = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_chunk)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)

            if S > 0:
                rs_chunk.append(R / S)

        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))

    lags = lags[:len(rs_values)]
    rs_values = np.array(rs_values)

    # Fit log(R/S) = H * log(n) + c
    log_lags = np.log(lags)
    log_rs = np.log(rs_values)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_rs)

    results = {
        'H': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'lags': lags,
        'rs_values': rs_values,
        'method': 'rs'
    }

    print(f"\nHurst Exponent (R/S Analysis): H = {slope:.4f}")
    print(f"R-squared: {r_value**2:.4f}")

    if slope < 0.5:
        print("Interpretation: Mean-reverting series (H < 0.5)")
    elif slope > 0.5:
        print("Interpretation: Trending/persistent series (H > 0.5)")
    else:
        print("Interpretation: Random walk (H ≈ 0.5)")

    return results


def plot_hurst(results, figsize=(10, 6)):
    """
    Plot R/S analysis result

    Parameters:
    -----------
    results : dict
        Results from hurst_exponent()
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    lags = results['lags']
    rs_values = results['rs_values']
    H = results['H']

    log_lags = np.log(lags)
    log_rs = np.log(rs_values)

    # Scatter plot
    ax.scatter(log_lags, log_rs, alpha=0.7, label='R/S values')

    # Fit line
    x_fit = np.linspace(min(log_lags), max(log_lags), 100)
    y_fit = results['intercept'] + H * x_fit
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: H = {H:.4f}')

    ax.set_xlabel('log(n)', fontsize=12)
    ax.set_ylabel('log(R/S)', fontsize=12)
    ax.set_title(f'Rescaled Range Analysis: R(n) ∝ n^H', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def dfa_analysis(series, scales=None, order=1):
    """
    Detrended Fluctuation Analysis

    Parameters:
    -----------
    series : array-like
        Time series
    scales : array-like
        Window sizes for DFA
    order : int
        Detrending order (1=linear, 2=quadratic)

    Returns:
    --------
    dict : DFA results including scaling exponent alpha
    """
    series = np.array(series).flatten()
    n = len(series)

    if scales is None:
        scales = np.unique(np.logspace(np.log10(10), np.log10(n//10), 30).astype(int))

    # Compute cumulative sum
    y = np.cumsum(series - np.mean(series))

    fluctuations = []

    for scale in scales:
        n_segments = n // scale
        if n_segments < 1:
            continue

        rms = []
        for i in range(n_segments):
            segment = y[i*scale:(i+1)*scale]
            x = np.arange(scale)

            # Fit polynomial
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)

            # Detrend and compute RMS
            detrended = segment - trend
            rms.append(np.sqrt(np.mean(detrended**2)))

        fluctuations.append(np.mean(rms))

    scales = scales[:len(fluctuations)]
    fluctuations = np.array(fluctuations)

    # Fit log(F) = alpha * log(n) + c
    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_fluct)

    results = {
        'alpha': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'scales': scales,
        'fluctuations': fluctuations,
        'order': order
    }

    print(f"\nDFA Scaling Exponent: α = {slope:.4f}")
    print(f"R-squared: {r_value**2:.4f}")

    # Relationship between H and alpha
    H_estimated = slope - 0.5 if slope < 1 else slope
    print(f"Estimated Hurst exponent from DFA: H ≈ {H_estimated:.4f}")

    return results


def plot_dfa(results, figsize=(10, 6)):
    """
    Plot DFA results

    Parameters:
    -----------
    results : dict
        Results from dfa_analysis()
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    scales = results['scales']
    fluctuations = results['fluctuations']
    alpha = results['alpha']

    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)

    # Scatter plot
    ax.scatter(log_scales, log_fluct, alpha=0.7, label='F(n) values')

    # Fit line
    x_fit = np.linspace(min(log_scales), max(log_scales), 100)
    y_fit = results['intercept'] + alpha * x_fit
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: α = {alpha:.4f}')

    ax.set_xlabel('log(n)', fontsize=12)
    ax.set_ylabel('log(F(n))', fontsize=12)
    ax.set_title(f'Detrended Fluctuation Analysis: F(n) ∝ n^α', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def multifractal_analysis(prices, q_values=None, tau_values=None):
    """
    Multifractal analysis using absolute moments

    M(q,τ) = <|Y(t+τ) - Y(t)|^q> ∝ τ^{f(q)/q}

    Parameters:
    -----------
    prices : array-like
        Price series (will use log(S))
    q_values : array-like
        Order values
    tau_values : array-like
        Time lag values

    Returns:
    --------
    dict : Multifractal analysis results
    """
    Y = np.log(np.array(prices).flatten())
    n = len(Y)

    if q_values is None:
        q_values = np.arange(1, 11)

    if tau_values is None:
        tau_values = np.unique(np.logspace(np.log10(1), np.log10(n//10), 30).astype(int))

    # Calculate M(q, τ)
    M = np.zeros((len(q_values), len(tau_values)))

    for i, q in enumerate(q_values):
        for j, tau in enumerate(tau_values):
            increments = np.abs(Y[tau:] - Y[:-tau])
            M[i, j] = np.mean(increments ** q)

    # Calculate f(q)/q from scaling
    f_over_q = []
    for i, q in enumerate(q_values):
        log_tau = np.log(tau_values)
        log_M = np.log(M[i, :])

        # Linear fit
        slope, intercept, r_value, _, _ = stats.linregress(log_tau, log_M)
        f_over_q.append(slope)

    f_values = np.array(f_over_q) * np.array(q_values)

    results = {
        'q_values': q_values,
        'tau_values': tau_values,
        'M': M,
        'f_over_q': np.array(f_over_q),
        'f_values': f_values
    }

    print("\nMultifractal Analysis Results:")
    print(f"f(1) ≈ {f_values[0]:.4f} (should be close to H)")
    print(f"f(q) range: [{min(f_values):.4f}, {max(f_values):.4f}]")

    if max(f_values) - min(f_values) > 0.1:
        print("Series exhibits multifractality")
    else:
        print("Series is approximately monofractal")

    return results


def plot_multifractal(results, figsize=(14, 5)):
    """
    Plot multifractal analysis results

    Parameters:
    -----------
    results : dict
        Results from multifractal_analysis()
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot M(q,τ)^{1/q} vs τ
    ax1 = axes[0]
    q_values = results['q_values']
    tau_values = results['tau_values']
    M = results['M']

    for i, q in enumerate(q_values[::2]):  # Plot every other q for clarity
        ax1.loglog(tau_values, M[i, :]**(1/q), 'o-', markersize=3, label=f'q={q}')

    ax1.set_xlabel('τ (time lag)', fontsize=12)
    ax1.set_ylabel('M(q,τ)^{1/q}', fontsize=12)
    ax1.set_title('Scaling of Absolute Moments', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot f(q)/q vs q
    ax2 = axes[1]
    ax2.plot(q_values, results['f_over_q'], 'bo-', markersize=8)
    ax2.axhline(y=results['f_over_q'][0], color='r', linestyle='--', label=f'H ≈ {results["f_over_q"][0]:.4f}')
    ax2.set_xlabel('q', fontsize=12)
    ax2.set_ylabel('f(q)/q', fontsize=12)
    ax2.set_title('Multifractal Spectrum', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def mfdfa_analysis(series, q_values=None, scales=None):
    """
    Multifractal Detrended Fluctuation Analysis (MF-DFA)

    Parameters:
    -----------
    series : array-like
        Time series
    q_values : array-like
        Order values for MF-DFA
    scales : array-like
        Window sizes

    Returns:
    --------
    dict : MF-DFA results
    """
    if not MFdfa_AVAILABLE:
        print("MFDFA library not available. Using simplified implementation.")
        return _simple_mfdfa(series, q_values, scales)

    series = np.array(series).flatten()

    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
        q_values = q_values[q_values != 0]  # Remove q=0

    if scales is None:
        n = len(series)
        scales = np.unique(np.logspace(np.log10(10), np.log10(n//10), 30).astype(int))

    # Run MF-DFA
    lag, dfa = MFDFA(series, scales, q=q_values, order=1)

    # Extract scaling exponents
    alphas = []
    for i, q in enumerate(q_values):
        log_lag = np.log(lag)
        log_dfa = np.log(dfa[:, i])

        valid = np.isfinite(log_dfa)
        slope, _, _, _, _ = stats.linregress(log_lag[valid], log_dfa[valid])
        alphas.append(slope)

    results = {
        'q_values': q_values,
        'scales': lag,
        'fluctuations': dfa,
        'alphas': np.array(alphas)
    }

    print("\nMF-DFA Results:")
    print(f"α(2) = {alphas[np.argmin(np.abs(q_values - 2))]:.4f}")
    print(f"α range: [{min(alphas):.4f}, {max(alphas):.4f}]")

    return results


def _simple_mfdfa(series, q_values=None, scales=None):
    """
    Simplified MF-DFA implementation
    """
    series = np.array(series).flatten()
    n = len(series)

    if q_values is None:
        q_values = np.array([-4, -2, -1, 1, 2, 4])

    if scales is None:
        scales = np.unique(np.logspace(np.log10(10), np.log10(n//10), 20).astype(int))

    # Cumulative sum
    y = np.cumsum(series - np.mean(series))

    alphas = []
    F_q = []

    for q in q_values:
        fluctuations = []
        for scale in scales:
            n_segments = n // scale
            if n_segments < 1:
                continue

            rms = []
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                x = np.arange(scale)

                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend

                rms.append(np.sqrt(np.mean(detrended**2)))

            rms = np.array(rms)
            if q == 0:
                f_q = np.exp(np.mean(np.log(rms)))
            else:
                f_q = (np.mean(rms**q))**(1/q)

            fluctuations.append(f_q)

        scales_valid = scales[:len(fluctuations)]
        log_scales = np.log(scales_valid)
        log_fluct = np.log(fluctuations)

        slope, _, _, _, _ = stats.linregress(log_scales, log_fluct)
        alphas.append(slope)
        F_q.append(fluctuations)

    return {
        'q_values': q_values,
        'scales': scales_valid,
        'F_q': F_q,
        'alphas': np.array(alphas)
    }


def plot_mfdfa(results, figsize=(10, 6)):
    """
    Plot MF-DFA results

    Parameters:
    -----------
    results : dict
        Results from mfdfa_analysis()
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(results['q_values'], results['alphas'], 'bo-', markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random walk (α=0.5)')
    ax.set_xlabel('q', fontsize=12)
    ax.set_ylabel('α(q)', fontsize=12)
    ax.set_title('MF-DFA: Scaling Exponents vs Order', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    n = 5000

    # Generate fractional Gaussian noise with H=0.7
    H_true = 0.7
    noise = np.random.randn(n)

    # Simple method to create persistent series
    series = np.zeros(n)
    series[0] = noise[0]
    for i in range(1, n):
        series[i] = 0.7 * series[i-1] + 0.3 * noise[i]

    # Test Hurst
    H_results = hurst_exponent(series)
    plot_hurst(H_results)
    plt.show()

    # Test DFA
    dfa_results = dfa_analysis(series)
    plot_dfa(dfa_results)
    plt.show()
