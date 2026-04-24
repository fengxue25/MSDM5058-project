"""
Part 6: Empirical Mode Decomposition (EMD)
Pure Python implementation without external dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

try:
    import nolds
    NOLDS_AVAILABLE = True
except Exception:
    NOLDS_AVAILABLE = False


class EMD:
    """
    Empirical Mode Decomposition implementation
    Pure Python/NumPy implementation
    """

    def __init__(self, n_max_imfs=10, max_iterations=1000, spline_method='cubic'):
        """
        Initialize EMD

        Parameters:
        -----------
        n_max_imfs : int
            Maximum number of IMFs to extract
        max_iterations : int
            Maximum iterations per IMF
        spline_method : str
            Interpolation method for envelope
        """
        self.n_max_imfs = n_max_imfs
        self.max_iterations = max_iterations
        self.spline_method = spline_method

    def _find_extrema(self, signal):
        """Find local maxima and minima"""
        # Find local maxima
        maxima_idx = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                maxima_idx.append(i)

        # Find local minima
        minima_idx = []
        for i in range(1, len(signal) - 1):
            if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                minima_idx.append(i)

        return np.array(maxima_idx), np.array(minima_idx)

    def _interpolate_envelope(self, signal, extrema_idx, kind='cubic'):
        """Interpolate envelope through extrema"""
        if len(extrema_idx) < 2:
            return None

        x = extrema_idx
        y = signal[extrema_idx]

        # Add boundary conditions
        x = np.concatenate([[0], x, [len(signal) - 1]])
        y = np.concatenate([[signal[0]], y, [signal[-1]]])

        try:
            if kind == 'cubic' and len(x) >= 4:
                f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
            else:
                f = interp1d(x, y, kind='linear', fill_value='extrapolate')
            return f(np.arange(len(signal)))
        except:
            return None

    def _is_imf(self, signal):
        """Check if signal is an IMF"""
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)

        # Count extrema
        maxima_idx, minima_idx = self._find_extrema(signal)
        n_extrema = len(maxima_idx) + len(minima_idx)

        # IMF condition: number of zero crossings equals number of extrema (±1)
        return abs(zero_crossings - n_extrema) <= 1

    def emd(self, signal):
        """
        Perform EMD decomposition

        Parameters:
        -----------
        signal : array-like
            Input signal

        Returns:
        --------
        imfs : ndarray
            Intrinsic Mode Functions
        residue : ndarray
            Residual signal
        """
        signal = np.array(signal).flatten()
        imfs = []
        residue = signal.copy()

        for imf_count in range(self.n_max_imfs):
            h = residue.copy()

            for iteration in range(self.max_iterations):
                # Find extrema
                maxima_idx, minima_idx = self._find_extrema(h)

                # Check if we have enough extrema
                if len(maxima_idx) < 2 or len(minima_idx) < 2:
                    break

                # Compute upper and lower envelopes
                upper_envelope = self._interpolate_envelope(h, maxima_idx, self.spline_method)
                lower_envelope = self._interpolate_envelope(h, minima_idx, self.spline_method)

                if upper_envelope is None or lower_envelope is None:
                    break

                # Compute mean envelope
                mean_envelope = (upper_envelope + lower_envelope) / 2

                # Update h
                h_new = h - mean_envelope

                # Check if IMF condition is met
                if self._is_imf(h_new):
                    imfs.append(h_new)
                    residue = residue - h_new
                    break

                h = h_new

            # Stop if residue is monotonic
            maxima_idx, minima_idx = self._find_extrema(residue)
            if len(maxima_idx) < 2 and len(minima_idx) < 2:
                break

        imfs = np.array(imfs) if imfs else np.array([]).reshape(0, len(signal))

        self.imfs = imfs
        self.residue = residue

        return imfs, residue


def emd_decomposition(series, max_imf=None):
    """
    Perform Empirical Mode Decomposition

    Parameters:
    -----------
    series : array-like
        Time series to decompose
    max_imf : int
        Maximum number of IMFs to extract

    Returns:
    --------
    dict : EMD results including IMFs and residual
    """
    series = np.array(series).flatten()

    # Create EMD object
    n_max = max_imf if max_imf else 10
    emd = EMD(n_max_imfs=n_max)

    imfs, residue = emd.emd(series)

    n_imfs = imfs.shape[0] if len(imfs) > 0 else 0

    print(f"\nEmpirical Mode Decomposition Results:")
    print(f"Number of IMFs extracted: {n_imfs}")
    print(f"Original series length: {len(series)}")

    results = {
        'imfs': imfs,
        'residue': residue,
        'n_imfs': n_imfs,
        'original': series
    }

    return results


def plot_imfs(results, figsize=(14, 10)):
    """
    Plot all IMFs and residual
    """
    imfs = results['imfs']
    residue = results['residue']
    n_imfs = results['n_imfs']
    original = results['original']

    if n_imfs == 0:
        print("No IMFs to plot")
        return None

    # Total plots: original + n_imfs + residue
    n_plots = n_imfs + 2

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    # Plot original
    axes[0].plot(original, linewidth=0.5)
    axes[0].set_ylabel('Original')
    axes[0].set_title('Original Time Series')
    axes[0].grid(True, alpha=0.3)

    # Plot IMFs
    for i in range(n_imfs):
        axes[i+1].plot(imfs[i], linewidth=0.5)
        axes[i+1].set_ylabel(f'IMF {i+1}')
        axes[i+1].grid(True, alpha=0.3)

    # Plot residue
    axes[-1].plot(residue, linewidth=0.5, color='red')
    axes[-1].set_ylabel('Residue')
    axes[-1].set_xlabel('Time')
    axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_selected_imfs(results, indices=None, figsize=(14, 8)):
    """
    Plot selected IMFs
    """
    imfs = results['imfs']
    n_imfs = results['n_imfs']

    if n_imfs == 0:
        print("No IMFs to plot")
        return None

    if indices is None:
        # Default: plot c1, c_k/2, c_3k/4, c_k
        indices = [
            1,                          # c_1
            n_imfs // 2,                # c_k/2
            3 * n_imfs // 4,            # c_3k/4
            n_imfs                      # c_k
        ]
        indices = [i for i in indices if 1 <= i <= n_imfs]
        indices = sorted(set(indices))

    n_plots = len(indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        if 1 <= idx <= n_imfs:
            ax.plot(results['imfs'][idx-1], linewidth=0.5)
            ax.set_ylabel(f'IMF {idx}')
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    return fig


def compute_imf_hurst(imfs, method='rs'):
    """
    Compute Hurst exponent for each IMF
    """
    n_imfs = imfs.shape[0] if len(imfs) > 0 else 0

    if n_imfs == 0:
        return []

    hurst_values = []

    print("\nHurst Exponents for IMFs:")
    print("-" * 40)

    for i in range(n_imfs):
        imf = imfs[i]

        if NOLDS_AVAILABLE:
            try:
                H = nolds.hurst_rs(imf)
            except:
                H = _simple_hurst(imf)
        else:
            H = _simple_hurst(imf)

        hurst_values.append(H)
        print(f"IMF {i+1}: H = {H:.4f}")

    return hurst_values


def _simple_hurst(series, max_lag=None):
    """
    Simple Hurst exponent calculation using R/S analysis
    """
    series = np.array(series).flatten()
    n = len(series)

    if max_lag is None:
        max_lag = n // 4

    lags = np.unique(np.logspace(np.log10(10), np.log10(max_lag), 20).astype(int))

    rs_values = []
    valid_lags = []

    for lag in lags:
        n_chunks = n // lag
        if n_chunks < 1:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = series[i*lag:(i+1)*lag]
            mean_chunk = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_chunk)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)

            if S > 0:
                rs_chunk.append(R / S)

        if rs_chunk:
            valid_lags.append(lag)
            rs_values.append(np.mean(rs_chunk))

    if len(valid_lags) < 5:
        return 0.5

    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)

    coeffs = np.polyfit(log_lags, log_rs, 1)
    return coeffs[0]


def plot_hurst_imf(hurst_values, figsize=(10, 6)):
    """
    Plot Hurst exponents of IMFs against their orders
    """
    if not hurst_values:
        print("No Hurst values to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    orders = range(1, len(hurst_values) + 1)

    ax.plot(orders, hurst_values, 'bo-', markersize=8, linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random walk (H=0.5)')
    ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('IMF Order', fontsize=12)
    ax.set_ylabel('Hurst Exponent', fontsize=12)
    ax.set_title('Hurst Exponents of IMFs', fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_imf_psd(imfs, indices=[0, 1], figsize=(14, 5)):
    """
    Plot PSD of specified IMFs
    """
    n_imfs = imfs.shape[0] if len(imfs) > 0 else 0

    if n_imfs == 0:
        print("No IMFs to analyze")
        return None

    indices = [i for i in indices if i < n_imfs]

    if not indices:
        print("No valid IMF indices")
        return None

    fig, axes = plt.subplots(1, len(indices), figsize=figsize)

    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        imf = imfs[idx]

        # Compute PSD
        frequencies, psd = signal.welch(imf, fs=1.0, nperseg=min(256, len(imf)//2))

        ax.loglog(frequencies[1:], psd[1:], linewidth=0.8)
        ax.set_xlabel('Frequency (cycles/day)')
        ax.set_ylabel('PSD')
        ax.set_title(f'IMF {idx+1} Power Spectrum')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_reduced_series(results, figsize=(14, 8)):
    """
    Analyze X - c1 and X - c1 - c2
    """
    original = results['original']
    imfs = results['imfs']
    n_imfs = results['n_imfs']

    if n_imfs == 0:
        print("No IMFs available")
        return None

    c1 = imfs[0]
    c2 = imfs[1] if n_imfs > 1 else np.zeros_like(c1)

    X_minus_c1 = original - c1
    X_minus_c1_c2 = original - c1 - c2

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Time series
    axes[0, 0].plot(X_minus_c1, linewidth=0.5)
    axes[0, 0].set_title('X - IMF1')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(X_minus_c1_c2, linewidth=0.5)
    axes[0, 1].set_title('X - IMF1 - IMF2')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].grid(True, alpha=0.3)

    # PSD comparison
    freq_orig, psd_orig = signal.welch(original, fs=1.0, nperseg=min(256, len(original)//2))
    freq_c1, psd_c1 = signal.welch(X_minus_c1, fs=1.0, nperseg=min(256, len(X_minus_c1)//2))
    freq_c1c2, psd_c1c2 = signal.welch(X_minus_c1_c2, fs=1.0, nperseg=min(256, len(X_minus_c1_c2)//2))

    axes[1, 0].loglog(freq_orig[1:], psd_orig[1:], linewidth=0.8, alpha=0.7, label='Original X')
    axes[1, 0].loglog(freq_c1[1:], psd_c1[1:], linewidth=0.8, alpha=0.7, label='X - IMF1')
    axes[1, 0].set_title('PSD Comparison: Original vs X-IMF1')
    axes[1, 0].set_xlabel('Frequency (cycles/day)')
    axes[1, 0].set_ylabel('PSD')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].loglog(freq_orig[1:], psd_orig[1:], linewidth=0.8, alpha=0.7, label='Original X')
    axes[1, 1].loglog(freq_c1c2[1:], psd_c1c2[1:], linewidth=0.8, alpha=0.7, label='X - IMF1 - IMF2')
    axes[1, 1].set_title('PSD Comparison: Original vs X-IMF1-IMF2')
    axes[1, 1].set_xlabel('Frequency (cycles/day)')
    axes[1, 1].set_ylabel('PSD')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_imfs(results):
    """
    Complete IMF analysis
    """
    imfs = results['imfs']
    n_imfs = results['n_imfs']

    if n_imfs == 0:
        return {'hurst_values': [], 'dominant_frequencies': []}

    # Compute Hurst for each IMF
    hurst_values = compute_imf_hurst(imfs)

    # Frequency analysis
    print("\nFrequency analysis of IMFs:")
    print("-" * 40)

    dominant_freqs = []
    for i in range(n_imfs):
        imf = imfs[i]
        frequencies, psd = signal.welch(imf, fs=1.0, nperseg=min(256, len(imf)//2))

        peak_idx = np.argmax(psd[1:]) + 1
        peak_freq = frequencies[peak_idx]
        peak_period = 1/peak_freq if peak_freq > 0 else np.inf

        dominant_freqs.append(peak_freq)

        print(f"IMF {i+1}: Peak freq = {peak_freq:.4f} cycles/day, Period = {peak_period:.1f} days")

    return {
        'hurst_values': hurst_values,
        'dominant_frequencies': dominant_freqs
    }


if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    n = 1000
    t = np.arange(n)

    # Generate signal with multiple components
    signal_data = (np.sin(2*np.pi*0.01*t) +
                   0.5*np.sin(2*np.pi*0.05*t) +
                   0.3*np.sin(2*np.pi*0.1*t) +
                   0.2*t/n +  # Trend
                   0.5*np.random.randn(n))  # Noise

    # EMD decomposition
    results = emd_decomposition(signal_data)

    if results is not None and results['n_imfs'] > 0:
        # Plot all IMFs
        plot_imfs(results)
        plt.show()

        # Plot selected IMFs
        plot_selected_imfs(results)
        plt.show()

        # Analyze IMFs
        analysis = analyze_imfs(results)

        # Plot Hurst
        plot_hurst_imf(analysis['hurst_values'])
        plt.show()
