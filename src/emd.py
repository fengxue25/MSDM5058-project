"""
Part 6: Empirical Mode Decomposition (EMD)
Patched version based on user's emd.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class EMD:
    def __init__(self, n_max_imfs=10, max_iterations=1000, spline_method='cubic'):
        self.n_max_imfs = n_max_imfs
        self.max_iterations = max_iterations
        self.spline_method = spline_method

    def _find_extrema(self, x):
        maxima_idx = []
        minima_idx = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1]:
                maxima_idx.append(i)
            if x[i] < x[i - 1] and x[i] < x[i + 1]:
                minima_idx.append(i)
        return np.array(maxima_idx), np.array(minima_idx)

    def _interpolate_envelope(self, x, extrema_idx, kind='cubic'):
        if len(extrema_idx) < 2:
            return None
        xp = extrema_idx
        yp = x[extrema_idx]
        xp = np.concatenate([[0], xp, [len(x) - 1]])
        yp = np.concatenate([[x[0]], yp, [x[-1]]])
        try:
            if kind == 'cubic' and len(xp) >= 4:
                f = interp1d(xp, yp, kind='cubic', fill_value='extrapolate')
            else:
                f = interp1d(xp, yp, kind='linear', fill_value='extrapolate')
            return f(np.arange(len(x)))
        except Exception:
            return None

    def _is_imf(self, x):
        zero_crossings = np.sum(np.diff(np.sign(x)) != 0)
        maxima_idx, minima_idx = self._find_extrema(x)
        n_extrema = len(maxima_idx) + len(minima_idx)
        return abs(zero_crossings - n_extrema) <= 1

    def emd(self, series):
        series = np.asarray(series, dtype=float).flatten()
        imfs = []
        residue = series.copy()

        for _ in range(self.n_max_imfs):
            h = residue.copy()
            extracted = False
            for _ in range(self.max_iterations):
                maxima_idx, minima_idx = self._find_extrema(h)
                if len(maxima_idx) < 2 or len(minima_idx) < 2:
                    break
                upper = self._interpolate_envelope(h, maxima_idx, self.spline_method)
                lower = self._interpolate_envelope(h, minima_idx, self.spline_method)
                if upper is None or lower is None:
                    break
                mean_env = (upper + lower) / 2.0
                h_new = h - mean_env
                if self._is_imf(h_new):
                    imfs.append(h_new)
                    residue = residue - h_new
                    extracted = True
                    break
                h = h_new
            maxima_idx, minima_idx = self._find_extrema(residue)
            if not extracted or (len(maxima_idx) < 2 and len(minima_idx) < 2):
                break

        imfs = np.array(imfs) if len(imfs) > 0 else np.empty((0, len(series)))
        self.imfs = imfs
        self.residue = residue
        return imfs, residue


def emd_decomposition(series, max_imf=None):
    series = np.asarray(series, dtype=float).flatten()
    series = series[np.isfinite(series)]
    n_max = max_imf if max_imf is not None else 10
    emd = EMD(n_max_imfs=n_max)
    imfs, residue = emd.emd(series)
    n_imfs = imfs.shape[0]

    print('\nEmpirical Mode Decomposition Results:')
    print(f'Number of IMFs extracted: {n_imfs}')
    print(f'Original series length: {len(series)}')

    return {
        'imfs': imfs,
        'residue': residue,
        'n_imfs': n_imfs,
        'original': series,
    }

# def plot_imfs(results, figsize=(14, 10)):
#     imfs = results['imfs']
#     residue = results['residue']
#     n_imfs = results['n_imfs']
#     original = results['original']
#     if n_imfs == 0:
#         return None
#     n_plots = n_imfs + 2
#     fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
#     axes[0].plot(original, linewidth=0.5)
#     axes[0].set_ylabel('Original')
#     axes[0].set_title('Original Time Series')
#     axes[0].grid(True, alpha=0.3)
#     for i in range(n_imfs):
#         axes[i + 1].plot(imfs[i], linewidth=0.5)
#         axes[i + 1].set_ylabel(f'IMF {i + 1}')
#         axes[i + 1].grid(True, alpha=0.3)
#     axes[-1].plot(residue, linewidth=0.5, color='red')
#     axes[-1].set_ylabel('Residue')
#     axes[-1].set_xlabel('Time')
#     axes[-1].grid(True, alpha=0.3)
#     plt.tight_layout()
#     return fig

def plot_selected_imfs(results, indices=None, figsize=(10, 8), title=''):
    imfs = results['imfs']
    n_imfs = results['n_imfs']
    if n_imfs == 0:
        return None
    if indices is None:
        k = n_imfs
        indices = [1, max(1, k // 4), max(1, k // 2), max(1, (3 * k) // 4), k]
    indices = sorted(set([i for i in indices if 1 <= i <= n_imfs]))
    fig, axes = plt.subplots(len(indices), 1, figsize=figsize, sharex=True)
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.plot(imfs[idx - 1], linewidth=0.8)
        ax.set_ylabel(f'c{idx}')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('t')
    
    if title:
        fig.suptitle(f'{title} Selected IMFs', y=1.02)
                     
    plt.tight_layout()
    return fig

def _simple_hurst(series, max_lag=None):
    series = np.asarray(series, dtype=float).flatten()
    n = len(series)
    if n < 40:
        return np.nan
    if max_lag is None:
        max_lag = max(20, n // 4)
    lags = np.unique(np.logspace(np.log10(10), np.log10(max_lag), 20).astype(int))
    rs_values = []
    valid_lags = []
    for lag in lags:
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = series[i * lag:(i + 1) * lag]
            mean_chunk = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_chunk)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            valid_lags.append(lag)
            rs_values.append(np.mean(rs_chunk))
    if len(valid_lags) < 3:
        return np.nan
    coeffs = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
    return float(coeffs[0])

def compute_imf_hurst(imfs, method='rs'):
    n_imfs = imfs.shape[0] if len(imfs) > 0 else 0
    if n_imfs == 0:
        return []
    hurst_values = []
    print('\nHurst Exponents for IMFs:')
    print('-' * 40)
    for i in range(n_imfs):
        imf = imfs[i]
        H = _simple_hurst(imf)
        hurst_values.append(H)
        print(f'IMF {i + 1}: H = {H:.4f}')
    return hurst_values

def plot_hurst_imf(hurst_values,title="", figsize=(10, 6)):
    if not hurst_values:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    orders = range(1, len(hurst_values) + 1)
    ax.plot(orders, hurst_values, 'bo-', markersize=7, linewidth=1.8)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random walk (H=0.5)')
    ax.set_xlabel('IMF Order')
    ax.set_ylabel('Hurst Exponent')
    ax.set_title(f'{title} Hurst Exponents of IMFs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def analyze_imf_psd(imfs,title="", indices=[0, 1], figsize=(10, 4)):
    n_imfs = imfs.shape[0] if len(imfs) > 0 else 0
    indices = [i for i in indices if i < n_imfs]
    if not indices:
        return None
    fig, axes = plt.subplots(1, len(indices), figsize=figsize)
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        imf = imfs[idx]
        frequencies, psd = signal.periodogram(imf, fs=1.0, detrend='constant', scaling='density')
        ax.plot(frequencies, np.maximum(psd, 1e-12), linewidth=0.8, color='#9c755f', alpha=0.75)
        ax.set_xlim(0, 0.5)
        ax.set_yscale('log')
        ax.set_xlabel('frequency (cycles per day)')
        ax.set_ylabel('power')
        ax.set_title(f'{title} \nIMF {idx + 1} PSD')
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def analyze_reduced_series(results, title="", figsize=(8, 5)):
    original = results['original']
    imfs = results['imfs']
    n_imfs = results['n_imfs']
    if n_imfs == 0:
        return None
    c1 = imfs[0]
    c2 = imfs[1] if n_imfs > 1 else np.zeros_like(c1)
    x_minus_c1 = original - c1
    x_minus_c1_c2 = original - c1 - c2

    fig, ax = plt.subplots(figsize=figsize)
    for series, label, color in [
        (original, 'X', '#555555'),
        (x_minus_c1, 'X - c1', '#1f77b4'),
        (x_minus_c1_c2, 'X - c1 - c2', '#d62728')
    ]:
        frequencies, psd = signal.periodogram(series, fs=1.0, detrend='constant', scaling='density')
        ax.plot(frequencies, np.maximum(psd, 1e-12), linewidth=0.9, label=label, color=color)
    ax.set_xlim(0, 0.5)
    ax.set_yscale('log')
    ax.set_xlabel('frequency (cycles per day)')
    ax.set_ylabel('power')
    ax.set_title(f'{title} \nPSD of original and reduced series')
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def analyze_imfs(results):
    imfs = results['imfs']
    n_imfs = results['n_imfs']
    if n_imfs == 0:
        return {'hurst_values': [], 'dominant_frequencies': []}
    hurst_values = compute_imf_hurst(imfs)
    print('\nFrequency analysis of IMFs:')
    print('-' * 40)
    dominant_freqs = []
    for i in range(n_imfs):
        imf = imfs[i]
        frequencies, psd = signal.periodogram(imf, fs=1.0, detrend='constant', scaling='density')
        if len(psd) > 1:
            peak_idx = np.argmax(psd[1:]) + 1
            peak_freq = frequencies[peak_idx]
        else:
            peak_freq = np.nan
        dominant_freqs.append(peak_freq)
        peak_period = 1 / peak_freq if peak_freq > 0 else np.inf
        print(f'IMF {i + 1}: Peak freq = {peak_freq:.4f} cycles/day, Period = {peak_period:.1f} days')
    return {
        'hurst_values': hurst_values,
        'dominant_frequencies': dominant_freqs,
    }
