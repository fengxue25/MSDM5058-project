"""
Part 5: Fourier Transform and Power Spectrum
Patched version based on user's fourier-2.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq
import warnings
warnings.filterwarnings('ignore')


def fourier_transform(series, sampling_rate=1.0):
    series = np.asarray(series, dtype=float).flatten()
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 2:
        raise ValueError('Series is too short for Fourier transform')

    fft_values = rfft(series)
    frequencies = rfftfreq(n, d=1.0 / sampling_rate)
    magnitude = np.abs(fft_values) / n

    results = {
        'frequencies': frequencies,
        'magnitude': magnitude,
        'fft_values': fft_values,
        'n': n,
        'sampling_rate': sampling_rate,
        'nyquist': sampling_rate / 2.0,
    }

    print(f"\nFourier Transform computed for {n} data points")
    print(f"Frequency range: 0 to {frequencies.max():.4f} cycles/day")
    print(f"Nyquist frequency: {sampling_rate / 2:.4f} cycles/day")
    return results


def power_spectrum(series, sampling_rate=1.0, method='periodogram'):
    series = np.asarray(series, dtype=float).flatten()
    series = series[np.isfinite(series)]
    if len(series) < 2:
        raise ValueError('Series is too short for PSD')

    if method == 'welch':
        nperseg = min(1024, max(32, len(series) // 4))
        frequencies, psd = signal.welch(series, fs=sampling_rate, nperseg=nperseg, detrend='constant')
    else:
        frequencies, psd = signal.periodogram(series, fs=sampling_rate, detrend='constant', scaling='density')

    results = {
        'frequencies': frequencies,
        'psd': psd,
        'method': method,
        'sampling_rate': sampling_rate,
        'nyquist': sampling_rate / 2.0,
    }

    total_power = np.trapz(psd, frequencies)
    peak_idx = int(np.argmax(psd))
    print(f"\nPower Spectrum ({method} method)")
    print(f"Total power: {total_power:.6f}")
    print(f"Peak frequency: {frequencies[peak_idx]:.6f} cycles/day")
    return results


def plot_fourier(results, title='', figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    frequencies = results['frequencies']
    magnitude = results['magnitude']
    mask = frequencies <= results.get('nyquist', 0.5)

    ax.plot(frequencies[mask], magnitude[mask], linewidth=0.8, color='#9c755f', alpha=0.75)
    ax.set_xlim(0, results.get('nyquist', 0.5))
    ax.set_xlabel('frequency (cycles per day)')
    ax.set_ylabel('magnitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def plot_power_spectrum(results, title='', figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)

    frequencies = np.array(results['frequencies'])
    psd = np.maximum(np.array(results['psd']), 1e-12)
    nyquist = results.get('nyquist', 0.5)

    mask = frequencies <= nyquist
    freq_plot = frequencies[mask]
    psd_plot = psd[mask]

    ymin = 1e-10
    ymax = max(psd_plot.max() * 1.1, ymin * 10)

    ax.plot(freq_plot, psd_plot, linewidth=0.8, color='#9c755f', alpha=0.75)
    ax.set_xlim(0, nyquist)
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('frequency (cycles per day)')
    ax.set_ylabel('power')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


# def compare_spectra(series_list, labels, title='', figsize=(10, 6)):
#     fig, ax = plt.subplots(figsize=figsize)
#     colors = plt.cm.tab10(np.linspace(0, 1, len(series_list)))
#     for series, label, color in zip(series_list, labels, colors):
#         results = power_spectrum(series, method='periodogram')
#         freqs = results['frequencies']
#         psd = np.maximum(results['psd'], 1e-12)
#         mask = freqs > 0
#         ax.plot(freqs[mask], psd[mask], linewidth=1.0, label=label, color=color, alpha=0.85)
#     ax.set_xlim(0, 0.5)
#     ax.set_yscale('log')
#     ax.set_xlabel('frequency (cycles per day)')
#     ax.set_ylabel('power')
#     ax.set_title(title if title else 'Power Spectrum Comparison')
#     ax.legend()
#     ax.grid(True, alpha=0.25)
#     plt.tight_layout()
#     return fig


def analyze_spectral_properties(results, title=''):
    frequencies = results['frequencies']
    psd = results['psd']
    peaks, _ = signal.find_peaks(psd, height=np.max(psd) * 0.01 if len(psd) else None)

    print(f"\n{title} Spectral Analysis:")
    print('=' * 50)
    if len(peaks) > 0:
        print(f"Number of significant peaks: {len(peaks)}")
        sorted_peaks = sorted(peaks, key=lambda x: psd[x], reverse=True)[:5]
        for peak in sorted_peaks:
            freq = frequencies[peak]
            period = 1 / freq if freq > 0 else np.inf
            print(f" Frequency: {freq:.6f} cycles/day, Period: {period:.1f} days")

    valid_mask = (frequencies > 0) & (psd > 0)
    slope_value = None
    if np.sum(valid_mask) > 10:
        log_f = np.log(frequencies[valid_mask])
        log_psd = np.log(psd[valid_mask])
        slope, intercept = np.polyfit(log_f, log_psd, 1)
        slope_value = -slope
        print(f"\nSpectral slope (β): {slope_value:.4f}")
        if slope_value > 1.5:
            print('Series exhibits pink/red noise characteristics')
        elif slope_value > 0.5:
            print('Series exhibits intermediate noise characteristics')
        else:
            print('Series is closer to white noise')

    return {
        'peaks': peaks,
        'peak_frequencies': frequencies[peaks] if len(peaks) > 0 else np.array([]),
        'spectral_slope': slope_value,
    }


# def plot_spectrogram(series, sampling_rate=1.0, window_size=256, figsize=(12, 6)):
#     series = np.asarray(series, dtype=float).flatten()
#     series = series[np.isfinite(series)]
#     f, t, Sxx = signal.spectrogram(series, fs=sampling_rate, nperseg=min(window_size, len(series)))
#     fig, ax = plt.subplots(figsize=figsize)
#     pcm = ax.pcolormesh(t, f, 10 * np.log10(np.maximum(Sxx, 1e-12)), shading='gouraud', cmap='viridis')
#     ax.set_ylim(0, sampling_rate / 2.0)
#     ax.set_ylabel('Frequency (cycles/day)')
#     ax.set_xlabel('Time (days)')
#     ax.set_title('Spectrogram')
#     plt.colorbar(pcm, ax=ax, label='Power (dB)')
#     plt.tight_layout()
#     return fig
