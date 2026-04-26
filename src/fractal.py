"""
Part 3: Fractal Behavior Analysis
Hurst Exponent, DFA, Multifractal Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from MFDFA import MFDFA
    MFDFA_AVAILABLE = True
except Exception:
    MFDFA_AVAILABLE = False
    print("Warning: MFDFA not installed. Some functions may not work.")


def hurst_exponent(series, max_lag=None, method='rs'):
    series = np.asarray(series, dtype=float).flatten()
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 50:
        raise ValueError('series too short for Hurst estimation')

    if max_lag is None:
        max_lag = max(20, n // 10)

    lags = np.unique(np.logspace(np.log10(10), np.log10(max_lag), 30).astype(int))
    rs_vals = []
    valid_lags = []

    for lag in lags:
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        chunk_rs = []
        for i in range(n_chunks):
            chunk = series[i * lag:(i + 1) * lag]
            mean_chunk = np.mean(chunk)
            dev = np.cumsum(chunk - mean_chunk)
            R = np.max(dev) - np.min(dev)
            S = np.std(chunk, ddof=1)
            if np.isfinite(R) and np.isfinite(S) and S > 0:
                chunk_rs.append(R / S)
        if chunk_rs:
            rs_vals.append(np.mean(chunk_rs))
            valid_lags.append(lag)

    valid_lags = np.asarray(valid_lags)
    rs_vals = np.asarray(rs_vals)
    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_vals)
    slope, intercept, r_value, _, _ = stats.linregress(log_lags, log_rs)
    return {
        'H': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'lags': valid_lags,
        'rs_values': rs_vals,
        'method': 'rs'
    }


def plot_hurst(results, title="",figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    log_lags = np.log(results['lags'])
    log_rs = np.log(results['rs_values'])
    ax.scatter(log_lags, log_rs, alpha=0.7, label='R/S values')
    x_fit = np.linspace(log_lags.min(), log_lags.max(), 100)
    y_fit = results['intercept'] + results['H'] * x_fit
    ax.plot(x_fit, y_fit, 'r-', lw=2, label=f"Fit: H = {results['H']:.4f}")
    ax.set_xlabel('log(n)')
    ax.set_ylabel('log(R/S)')
    ax.set_title(f'{title} \nRescaled Range Analysis: R/S ∝ n^H')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def dfa_analysis(series, scales=None, order=1):
    series = np.asarray(series, dtype=float).flatten()
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 50:
        raise ValueError('series too short for DFA')

    if scales is None:
        scales = np.unique(np.logspace(np.log10(10), np.log10(n // 10), 30).astype(int))

    y = np.cumsum(series - np.mean(series))
    fluctuations = []
    valid_scales = []

    for scale in scales:
        n_segments = n // scale
        if n_segments < 2:
            continue
        rms = []
        x = np.arange(scale)
        for i in range(n_segments):
            segment = y[i * scale:(i + 1) * scale]
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            val = np.sqrt(np.mean(detrended ** 2))
            if np.isfinite(val) and val > 0:
                rms.append(val)
        if rms:
            fluctuations.append(np.mean(rms))
            valid_scales.append(scale)

    valid_scales = np.asarray(valid_scales)
    fluctuations = np.asarray(fluctuations)
    log_scales = np.log(valid_scales)
    log_fluct = np.log(fluctuations)
    slope, intercept, r_value, _, _ = stats.linregress(log_scales, log_fluct)
    return {
        'alpha': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'scales': valid_scales,
        'fluctuations': fluctuations,
        'order': order
    }


def plot_dfa(results,title ="", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    log_scales = np.log(results['scales'])
    log_fluct = np.log(results['fluctuations'])
    ax.scatter(log_scales, log_fluct, alpha=0.7, label='F(n) values')
    x_fit = np.linspace(log_scales.min(), log_scales.max(), 100)
    y_fit = results['intercept'] + results['alpha'] * x_fit
    ax.plot(x_fit, y_fit, 'r-', lw=2, label=f"Fit: α = {results['alpha']:.4f}")
    ax.set_xlabel('log(n)')
    ax.set_ylabel('log(F(n))')
    ax.set_title(f'{title} \nDetrended Fluctuation Analysis: F(n) ∝ n^α')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def multifractal_analysis(prices_or_log_prices, q_values=None, tau_values=None, input_is_log_price=False, eps=1e-12):
    y = np.asarray(prices_or_log_prices, dtype=float).flatten()
    y = y[np.isfinite(y)]
    if not input_is_log_price:
        if np.any(y <= 0):
            raise ValueError('price series must be positive when input_is_log_price=False')
        y = np.log(y)

    n = len(y)
    if n < 100:
        raise ValueError('series too short for multifractal structure-function analysis')

    if q_values is None:
        q_values = np.array([1.0, 2.0, 3.0, 4.0])
    q_values = np.asarray(q_values, dtype=float)
    if np.any(np.isclose(q_values, 0.0)):
        raise ValueError('q=0 is not supported in this structure-function implementation')

    if tau_values is None:
        tau_values = np.unique(np.logspace(np.log10(5), np.log10(max(10, n // 20)), 20).astype(int))
    tau_values = np.asarray(tau_values, dtype=int)
    tau_values = tau_values[tau_values > 0]

    M = np.full((len(q_values), len(tau_values)), np.nan)
    zeta = np.full(len(q_values), np.nan)
    r2 = np.full(len(q_values), np.nan)

    for i, q in enumerate(q_values):
        vals = []
        valid_tau = []
        for tau in tau_values:
            if tau >= n:
                continue
            increments = np.abs(y[tau:] - y[:-tau])
            increments = np.maximum(increments, eps)
            moment = np.mean(increments ** q)
            if np.isfinite(moment) and moment > 0:
                vals.append(moment)
                valid_tau.append(tau)
        if len(vals) >= 3:
            valid_tau = np.asarray(valid_tau)
            vals = np.asarray(vals)
            cols = np.searchsorted(tau_values, valid_tau)
            M[i, cols] = vals
            slope, intercept, r_value, _, _ = stats.linregress(np.log(valid_tau), np.log(vals))
            zeta[i] = slope
            r2[i] = r_value ** 2

    return {
        'q_values': q_values,
        'tau_values': tau_values,
        'M': M,
        'zeta_q': zeta,
        'f_over_q': zeta / q_values,
        'r_squared': r2
    }


def plot_multifractal(results, title ="", figsize=(14, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1, ax2 = axes
    q_values = results['q_values']
    tau_values = results['tau_values']
    M = results['M']

    plot_idx = np.arange(len(q_values))[::max(1, len(q_values) // 6 or 1)]
    for i in plot_idx:
        q = q_values[i]
        row = M[i, :]
        valid = np.isfinite(row) & (row > 0)
        if valid.sum() < 3:
            continue
        ax1.loglog(tau_values[valid], row[valid] ** (1.0 / q), 'o-', ms=3, label=f'q={q:g}')

    ax1.set_xlabel('τ (time lag)')
    ax1.set_ylabel('M(q,τ)^(1/q)')
    ax1.set_title(f'{title} Scaling of Absolute Moments')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    valid2 = np.isfinite(results['f_over_q'])
    ax2.plot(q_values[valid2], results['f_over_q'][valid2], 'bo-', ms=7)
    idx_q1 = np.argmin(np.abs(q_values - 1.0))
    if np.isfinite(results['f_over_q'][idx_q1]) and np.isclose(q_values[idx_q1], 1.0):
        h1 = results['f_over_q'][idx_q1]
        ax2.axhline(h1, color='r', ls='--', label=f'q=1 slope ≈ {h1:.4f}')
        ax2.legend()
    ax2.set_xlabel('q')
    ax2.set_ylabel('ζ(q)/q')
    ax2.set_title(f'{title} Structure Function Scaling Exponent')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def mfdfa_analysis(series, q_values=None, scales=None, order=1):
    x = np.asarray(series, dtype=float).flatten()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 100:
        raise ValueError('series too short for MF-DFA')

    if q_values is None:
        q_values = np.array([1.0, 2.0, 3.0, 4.0])
    q_values = np.asarray(q_values, dtype=float)
    q_values = q_values[~np.isclose(q_values, 0.0)]

    if scales is None:
        scales = np.unique(np.logspace(np.log10(10), np.log10(max(20, n // 10)), 24).astype(int))
    scales = np.asarray(scales, dtype=int)
    scales = scales[scales > order + 2]

    if MFDFA_AVAILABLE:
        lag, dfa = MFDFA(x, scales, q=q_values, order=order)
        alphas = []
        for i in range(len(q_values)):
            yy = dfa[:, i]
            valid = np.isfinite(yy) & (yy > 0)
            slope, _, _, _, _ = stats.linregress(np.log(lag[valid]), np.log(yy[valid]))
            alphas.append(slope)
        return {
            'q_values': q_values,
            'scales': np.asarray(lag),
            'fluctuations': np.asarray(dfa),
            'alphas': np.asarray(alphas)
        }

    y = np.cumsum(x - np.mean(x))
    F_q_all = []
    alphas = []
    valid_scales_ref = None

    for q in q_values:
        f_q_scales = []
        valid_scales = []
        for s in scales:
            ns = n // s
            if ns < 2:
                continue
            rms = []
            idx = np.arange(s)
            for v in range(ns):
                seg = y[v * s:(v + 1) * s]
                coeffs = np.polyfit(idx, seg, order)
                trend = np.polyval(coeffs, idx)
                val = np.sqrt(np.mean((seg - trend) ** 2))
                if np.isfinite(val) and val > 0:
                    rms.append(val)
            rms = np.asarray(rms)
            if len(rms) < 2:
                continue
            if np.isclose(q, 0.0):
                fq = np.exp(0.5 * np.mean(np.log(rms ** 2)))
            else:
                fq = (np.mean(rms ** q)) ** (1.0 / q)
            if np.isfinite(fq) and fq > 0:
                f_q_scales.append(fq)
                valid_scales.append(s)
        valid_scales = np.asarray(valid_scales)
        f_q_scales = np.asarray(f_q_scales)
        if len(f_q_scales) < 3:
            alphas.append(np.nan)
            F_q_all.append(f_q_scales)
            continue
        slope, _, _, _, _ = stats.linregress(np.log(valid_scales), np.log(f_q_scales))
        alphas.append(slope)
        F_q_all.append(f_q_scales)
        if valid_scales_ref is None or len(valid_scales) < len(valid_scales_ref):
            valid_scales_ref = valid_scales

    return {
        'q_values': q_values,
        'scales': valid_scales_ref,
        'F_q': F_q_all,
        'alphas': np.asarray(alphas)
    }


def plot_mfdfa(results, title="", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    valid = np.isfinite(results['alphas'])
    ax.plot(results['q_values'][valid], results['alphas'][valid], 'bo-', ms=8)
    ax.axhline(0.5, color='r', ls='--', alpha=0.5, label='Random walk (α=0.5)')
    ax.set_xlabel('q')
    ax.set_ylabel('α(q)')
    ax.set_title(f'{title} \nMF-DFA: Scaling Exponents vs Order')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
