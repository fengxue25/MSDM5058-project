"""
Part 5: Fourier Transform and Power Spectrum
FFT and Power Spectral Density analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
import warnings
warnings.filterwarnings('ignore')


def fourier_transform(series, sampling_rate=1.0):
    """
    Compute Fourier Transform of a time series

    Parameters:
    -----------
    series : array-like
        Time series data
    sampling_rate : float
        Sampling rate (1 for daily data)

    Returns:
    --------
    dict : Fourier transform results
    """
    series = np.array(series).flatten()
    n = len(series)

    # Compute FFT
    fft_values = fft(series)
    fft_magnitude = np.abs(fft_values)

    # Compute frequencies
    frequencies = fftfreq(n, d=1.0/sampling_rate)

    # Only positive frequencies
    positive_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_mask]
    magnitude_pos = fft_magnitude[positive_mask]

    # Normalize magnitude
    magnitude_normalized = magnitude_pos / n

    results = {
        'frequencies': frequencies_pos,
        'magnitude': magnitude_normalized,
        'fft_values': fft_values,
        'n': n
    }

    print(f"\nFourier Transform computed for {n} data points")
    print(f"Frequency range: 0 to {max(frequencies_pos):.4f} cycles/day")
    print(f"Nyquist frequency: {sampling_rate/2:.4f} cycles/day")

    return results


def power_spectrum(series, sampling_rate=1.0, method='welch'):
    """
    Compute Power Spectral Density (PSD)

    Parameters:
    -----------
    series : array-like
        Time series data
    sampling_rate : float
        Sampling rate (1 for daily data)
    method : str
        'welch' for Welch's method, 'periodogram' for simple periodogram

    Returns:
    --------
    dict : Power spectrum results
    """
    series = np.array(series).flatten()

    if method == 'welch':
        frequencies, psd = signal.welch(series, fs=sampling_rate, nperseg=min(1024, len(series)//4))
    else:
        frequencies, psd = signal.periodogram(series, fs=sampling_rate)

    results = {
        'frequencies': frequencies,
        'psd': psd,
        'method': method,
        'sampling_rate': sampling_rate
    }

    # Compute total power
    total_power = np.trapz(psd, frequencies)
    print(f"\nPower Spectrum ({method} method)")
    print(f"Total power: {total_power:.6f}")
    print(f"Peak frequency: {frequencies[np.argmax(psd)]:.6f} cycles/day")

    return results


def plot_fourier(results, title='', figsize=(14, 5)):
    """
    Plot Fourier transform magnitude

    Parameters:
    -----------
    results : dict
        Results from fourier_transform()
    title : str
        Title for the plot
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    frequencies = results['frequencies']
    magnitude = results['magnitude']

    # Plot 1: Full spectrum
    axes[0].plot(frequencies, magnitude, linewidth=0.5)
    axes[0].set_xlabel('Frequency (cycles/day)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title(f'{title} Fourier Coefficients Magnitude')
    axes[0].grid(True, alpha=0.3)

    # Nyquist limit line
    nyquist = 0.5
    axes[0].axvline(x=nyquist, color='red', linestyle='--', alpha=0.5, label=f'Nyquist ({nyquist})')
    axes[0].legend()

    # Plot 2: Low frequency region
    low_freq_mask = frequencies <= 0.1
    axes[1].plot(frequencies[low_freq_mask], magnitude[low_freq_mask], linewidth=0.8)
    axes[1].set_xlabel('Frequency (cycles/day)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title(f'{title} Low Frequency Region (≤ 0.1 cycles/day)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_power_spectrum(results, title='', figsize=(14, 5)):
    """
    Plot Power Spectral Density

    Parameters:
    -----------
    results : dict
        Results from power_spectrum()
    title : str
        Title for the plot
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    frequencies = results['frequencies']
    psd = results['psd']

    # Plot 1: Linear scale
    axes[0].plot(frequencies, psd, linewidth=0.8)
    axes[0].set_xlabel('Frequency (cycles/day)')
    axes[0].set_ylabel('Power Spectral Density')
    axes[0].set_title(f'{title} Power Spectrum (Linear Scale)')
    axes[0].grid(True, alpha=0.3)

    # Nyquist limit line
    nyquist = 0.5
    axes[0].axvline(x=nyquist, color='red', linestyle='--', alpha=0.5, label=f'Nyquist ({nyquist})')
    axes[0].legend()

    # Plot 2: Log-log scale
    axes[1].loglog(frequencies[1:], psd[1:], linewidth=0.8)  # Skip zero frequency
    axes[1].set_xlabel('Frequency (cycles/day)')
    axes[1].set_ylabel('Power Spectral Density')
    axes[1].set_title(f'{title} Power Spectrum (Log-Log Scale)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compare_spectra(series_list, labels, title='', figsize=(12, 6)):
    """
    Compare power spectra of multiple series

    Parameters:
    -----------
    series_list : list
        List of time series
    labels : list
        Labels for each series
    title : str
        Title for the plot
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(series_list)))

    for series, label, color in zip(series_list, labels, colors):
        results = power_spectrum(series)
        ax.loglog(results['frequencies'][1:], results['psd'][1:],
                  linewidth=0.8, label=label, color=color, alpha=0.8)

    ax.set_xlabel('Frequency (cycles/day)', fontsize=12)
    ax.set_ylabel('Power Spectral Density', fontsize=12)
    ax.set_title(f'{title} Power Spectrum Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_spectral_properties(results, title=''):
    """
    Analyze and comment on spectral properties

    Parameters:
    -----------
    results : dict
        Results from power_spectrum()

    Returns:
    --------
    dict : Analysis results
    """
    frequencies = results['frequencies']
    psd = results['psd']

    # Find peaks
    peaks, properties = signal.find_peaks(psd, height=np.max(psd)*0.01)

    print(f"\n{title} Spectral Analysis:")
    print("="*50)

    if len(peaks) > 0:
        print(f"Number of significant peaks: {len(peaks)}")
        print("\nTop peaks:")
        sorted_peaks = sorted(peaks, key=lambda x: psd[x], reverse=True)[:5]
        for peak in sorted_peaks:
            freq = frequencies[peak]
            period = 1/freq if freq > 0 else np.inf
            print(f"  Frequency: {freq:.6f} cycles/day, Period: {period:.1f} days")

    # Check for 1/f noise (pink noise)
    # Fit log(PSD) = -beta * log(f) + c
    valid_mask = (frequencies > 0) & (psd > 0)
    if np.sum(valid_mask) > 10:
        log_f = np.log(frequencies[valid_mask])
        log_psd = np.log(psd[valid_mask])

        slope, intercept = np.polyfit(log_f, log_psd, 1)

        print(f"\nSpectral slope (β): {-slope:.4f}")
        if -slope > 1.5:
            print("Series exhibits pink/red noise characteristics")
        elif -slope > 0.5:
            print("Series exhibits intermediate noise characteristics")
        else:
            print("Series is closer to white noise")

    return {
        'peaks': peaks,
        'peak_frequencies': frequencies[peaks] if len(peaks) > 0 else [],
        'spectral_slope': -slope if 'slope' in dir() else None
    }


def plot_spectrogram(series, sampling_rate=1.0, window_size=256, figsize=(12, 6)):
    """
    Plot time-frequency spectrogram

    Parameters:
    -----------
    series : array-like
        Time series data
    sampling_rate : float
        Sampling rate
    window_size : int
        Window size for STFT
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
    """
    series = np.array(series).flatten()

    f, t, Sxx = signal.spectrogram(series, fs=sampling_rate, nperseg=window_size)

    fig, ax = plt.subplots(figsize=figsize)

    pcm = ax.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (cycles/day)')
    ax.set_xlabel('Time (days)')
    ax.set_title('Spectrogram')
    plt.colorbar(pcm, ax=ax, label='Power (dB)')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    n = 1000
    t = np.arange(n)

    # Generate signal with multiple frequencies
    signal_data = (np.sin(2*np.pi*0.01*t) +
                   0.5*np.sin(2*np.pi*0.05*t) +
                   0.3*np.sin(2*np.pi*0.1*t) +
                   0.5*np.random.randn(n))

    # Fourier transform
    fft_results = fourier_transform(signal_data)
    plot_fourier(fft_results, title='Synthetic Signal')
    plt.show()

    # Power spectrum
    psd_results = power_spectrum(signal_data)
    plot_power_spectrum(psd_results, title='Synthetic Signal')
    plt.show()

    # Spectral analysis
    analyze_spectral_properties(psd_results, title='Synthetic Signal')
