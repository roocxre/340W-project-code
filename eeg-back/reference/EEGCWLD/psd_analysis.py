import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Optional, Dict


# Define EEG frequency bands with colors
DEFAULT_BANDS = {
    'Delta': (1, 4, '#FF6B6B'),
    'Theta': (4, 8, '#4ECDC4'),
    'Alpha': (8, 13, '#45B7D1'),
    'Beta': (13, 30, '#96CEB4'),
    'Gamma': (30, 40, '#FFEAA7')
}


def compute_psd_welch(
    data: np.ndarray,
    sfreq: float,
    nperseg: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data array.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 2048.
    
    Returns
    -------
    tuple
        (frequencies, power spectral density).
    
    Examples
    --------
    >>> raw = load_edf('data/subject1.edf')
    >>> data = raw.get_data()[0]
    >>> freqs, psd = compute_psd_welch(data, raw.info['sfreq'])
    """
    frequencies, psd = signal.welch(data, sfreq, nperseg=nperseg)
    return frequencies, psd


def compute_psd_mne(
    raw: mne.io.Raw,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_fft: int = 2048,
    picks: Optional[str] = None,
    verbose: bool = False
) -> mne.time_frequency.Spectrum:
    """
    Compute PSD using MNE's built-in method.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object.
    fmin : float, optional
        Low frequency bound. Default is 1.0.
    fmax : float, optional
        High frequency bound. Default is 40.0.
    n_fft : int, optional
        Number of FFT points. Default is 2048.
    picks : str or None, optional
        Channel selection. Default is None (all EEG channels).
    verbose : bool, optional
        If True, print information. Default is False.
    
    Returns
    -------
    mne.time_frequency.Spectrum
        Spectrum object containing PSD data.
    
    Examples
    --------
    >>> raw = load_edf('data/subject1.edf')
    >>> psd = compute_psd_mne(raw, fmin=1.0, fmax=40.0)
    """
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, n_fft=n_fft, 
                         picks=picks, verbose=verbose)
    return psd


def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float]
) -> float:
    """
    Compute absolute power in a frequency band.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    psd : np.ndarray
        Power spectral density.
    band : tuple
        (low_freq, high_freq) for the band.
    
    Returns
    -------
    float
        Integrated power in the band.
    
    Examples
    --------
    >>> freqs, psd = compute_psd_welch(data, sfreq)
    >>> alpha_power = compute_band_power(freqs, psd, (8, 13))
    """
    low, high = band
    band_mask = (freqs >= low) & (freqs <= high)
    band_power = np.trapz(psd[band_mask], freqs[band_mask])
    return band_power


def analyze_bands(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Analyze power in all frequency bands.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    psd : np.ndarray
        Power spectral density.
    bands : dict or None, optional
        Dictionary of band definitions. Default uses DEFAULT_BANDS.
    
    Returns
    -------
    dict
        Dictionary with band analysis results.
    
    Examples
    --------
    >>> freqs, psd = compute_psd_welch(data, sfreq)
    >>> results = analyze_bands(freqs, psd)
    >>> print(results['Alpha']['power'])
    """
    if bands is None:
        bands = {k: v[:2] for k, v in DEFAULT_BANDS.items()}
    
    results = {}
    total_power = 0
    
    # Calculate total power in the frequency range
    fmin, fmax = freqs[0], freqs[-1]
    total_mask = (freqs >= fmin) & (freqs <= fmax)
    total_power = np.trapz(psd[total_mask], freqs[total_mask])
    
    for band_name, (low, high) in bands.items():
        # Find frequency indices for this band
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = np.trapz(psd[band_mask], freqs[band_mask])
        
        # Find dominant frequency in this band
        band_psd = psd[band_mask]
        band_freqs = freqs[band_mask]
        if len(band_freqs) > 0:
            dominant_freq = band_freqs[np.argmax(band_psd)]
        else:
            dominant_freq = 0
        
        # Calculate relative power
        relative_power = (band_power / total_power) * 100 if total_power > 0 else 0
        
        results[band_name] = {
            'power': band_power,
            'dominant_freq': dominant_freq,
            'relative_power': relative_power,
            'frequency_range': (low, high)
        }
    
    return results


def compute_psd_comparison(
    raw_clean: mne.io.Raw,
    raw_noisy: mne.io.Raw,
    channel: Optional[str] = None,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_fft: int = 2048
) -> Dict:
    """
    Compare PSD between clean and noisy EEG data.
    
    Parameters
    ----------
    raw_clean : mne.io.Raw
        Clean raw EEG data.
    raw_noisy : mne.io.Raw
        Noisy raw EEG data.
    channel : str or None, optional
        Channel to analyze. Default is first channel.
    fmin : float, optional
        Low frequency bound. Default is 1.0.
    fmax : float, optional
        High frequency bound. Default is 40.0.
    n_fft : int, optional
        Number of FFT points. Default is 2048.
    
    Returns
    -------
    dict
        Dictionary containing comparison results.
    
    Examples
    --------
    >>> clean = load_edf('data/Clean_276.EDF')
    >>> noisy = load_edf('data/Noisy_256.EDF')
    >>> results = compute_psd_comparison(clean, noisy)
    """
    # Select channel
    if channel is None:
        channel = raw_clean.ch_names[0]
    
    # Get data for specified channel
    data_clean = raw_clean.get_data(picks=[channel])[0]
    data_noisy = raw_noisy.get_data(picks=[channel])[0]
    
    # Compute PSD
    freqs, psd_clean = compute_psd_welch(data_clean, raw_clean.info['sfreq'], nperseg=n_fft)
    _, psd_noisy = compute_psd_welch(data_noisy, raw_noisy.info['sfreq'], nperseg=n_fft)
    
    # Filter to specified frequency range
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    psd_clean = psd_clean[freq_mask]
    psd_noisy = psd_noisy[freq_mask]
    
    # Analyze bands
    bands_clean = analyze_bands(freqs, psd_clean)
    bands_noisy = analyze_bands(freqs, psd_noisy)
    
    # Calculate statistics
    mean_power_clean = np.mean(psd_clean)
    mean_power_noisy = np.mean(psd_noisy)
    power_ratio = mean_power_noisy / mean_power_clean if mean_power_clean > 0 else 0
    
    return {
        'frequencies': freqs,
        'psd_clean': psd_clean,
        'psd_noisy': psd_noisy,
        'channel': channel,
        'bands_clean': bands_clean,
        'bands_noisy': bands_noisy,
        'mean_power_clean': mean_power_clean,
        'mean_power_noisy': mean_power_noisy,
        'power_ratio': power_ratio,
        'sfreq': raw_clean.info['sfreq']
    }


def plot_psd_comparison(
    results: Dict,
    save_path: Optional[str] = None,
    dpi: int = 300,
    style: str = 'publication'
) -> plt.Figure:
    """
    Create publication-quality PSD comparison plots.
    
    Parameters
    ----------
    results : dict
        Results from compute_psd_comparison.
    save_path : str or None, optional
        Path to save figure. Default is None (no saving).
    dpi : int, optional
        DPI for saved figure. Default is 300.
    style : str, optional
        Plot style ('publication' or 'default'). Default is 'publication'.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Examples
    --------
    >>> results = compute_psd_comparison(clean, noisy)
    >>> fig = plot_psd_comparison(results, save_path='psd_comparison.png')
    """
    if style == 'publication':
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'axes.linewidth': 1.2,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
        })
    
    freqs = results['frequencies']
    psd_clean = results['psd_clean']
    psd_noisy = results['psd_noisy']
    channel = results['channel']
    
    # Create figure with side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define bands
    bands = {k: v[:2] for k, v in DEFAULT_BANDS.items()}
    band_colors = {k: v[2] for k, v in DEFAULT_BANDS.items()}
    
    # Plot Clean EEG PSD
    axes[0].semilogy(freqs, psd_clean, color='#2E86C1', linewidth=2, label='Clean EEG')
    axes[0].set_xlabel('Frequency (Hz)', fontweight='bold')
    axes[0].set_ylabel('Power Spectral Density (V²/Hz)', fontweight='bold')
    axes[0].set_title(f'Clean EEG PSD\nChannel: {channel}', fontweight='bold', pad=20)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(freqs[0], freqs[-1])
    
    # Add frequency band annotations
    for band, (low, high) in bands.items():
        axes[0].axvspan(low, high, alpha=0.1, color=band_colors[band], 
                       label=f'{band} ({low}-{high} Hz)')
    axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Plot Noisy EEG PSD
    axes[1].semilogy(freqs, psd_noisy, color='#E74C3C', linewidth=2, label='Noisy EEG')
    axes[1].set_xlabel('Frequency (Hz)', fontweight='bold')
    axes[1].set_ylabel('Power Spectral Density (V²/Hz)', fontweight='bold')
    axes[1].set_title(f'Noisy EEG PSD\nChannel: {channel}', fontweight='bold', pad=20)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim(freqs[0], freqs[-1])
    
    # Add frequency band annotations
    for band, (low, high) in bands.items():
        axes[1].axvspan(low, high, alpha=0.1, color=band_colors[band],
                       label=f'{band} ({low}-{high} Hz)')
    axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_psd_overlay(
    results: Dict,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create overlay comparison plot of clean vs noisy PSD.
    
    Parameters
    ----------
    results : dict
        Results from compute_psd_comparison.
    save_path : str or None, optional
        Path to save figure. Default is None.
    dpi : int, optional
        DPI for saved figure. Default is 300.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Examples
    --------
    >>> results = compute_psd_comparison(clean, noisy)
    >>> fig = plot_psd_overlay(results, save_path='psd_overlay.png')
    """
    freqs = results['frequencies']
    psd_clean = results['psd_clean']
    psd_noisy = results['psd_noisy']
    channel = results['channel']
    power_ratio = results['power_ratio']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot both PSDs
    ax.semilogy(freqs, psd_clean, color='#2E86C1', linewidth=2.5, 
               label='Clean EEG', alpha=0.8)
    ax.semilogy(freqs, psd_noisy, color='#E74C3C', linewidth=2.5, 
               label='Noisy EEG', alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Power Spectral Density (V²/Hz)', fontsize=14, fontweight='bold')
    ax.set_title(f'Power Spectral Density Comparison\nChannel: {channel}',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(freqs[0], freqs[-1])
    
    # Add frequency band background colors
    bands = {k: v[:2] for k, v in DEFAULT_BANDS.items()}
    band_colors = {k: v[2] for k, v in DEFAULT_BANDS.items()}
    
    for band, (low, high) in bands.items():
        ax.axvspan(low, high, alpha=0.05, color=band_colors[band])
        ax.text((low + high) / 2, ax.get_ylim()[1] * 0.8, band,
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=band_colors[band], alpha=0.3))
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9, shadow=True, fancybox=True)
    
    # Add statistical information
    ax.text(0.02, 0.98, f'Mean Power Ratio (Noisy/Clean): {power_ratio:.2f}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved: {save_path}")
    
    return fig


def print_psd_summary(results: Dict) -> None:
    """
    Print summary statistics for PSD comparison.
    
    Parameters
    ----------
    results : dict
        Results from compute_psd_comparison.
    
    Examples
    --------
    >>> results = compute_psd_comparison(clean, noisy)
    >>> print_psd_summary(results)
    """
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Channel analyzed: {results['channel']}")
    print(f"Frequency range: {results['frequencies'][0]}-{results['frequencies'][-1]} Hz")
    print(f"Mean power (Clean): {results['mean_power_clean']:.2e} V²/Hz")
    print(f"Mean power (Noisy): {results['mean_power_noisy']:.2e} V²/Hz")
    print(f"Power ratio (Noisy/Clean): {results['power_ratio']:.2f}")
    
    print(f"\nBand-specific power analysis:")
    print("-" * 50)
    print(f"{'Band':<10} {'Clean Power':>15} {'Noisy Power':>15} {'Ratio':>10}")
    print("-" * 50)
    
    for band_name in results['bands_clean']:
        clean_power = results['bands_clean'][band_name]['power']
        noisy_power = results['bands_noisy'][band_name]['power']
        ratio = noisy_power / clean_power if clean_power > 0 else 0
        freq_range = results['bands_clean'][band_name]['frequency_range']
        print(f"{band_name} ({freq_range[0]}-{freq_range[1]} Hz): {clean_power:.2e} {noisy_power:.2e} {ratio:.2f}")
    
    print("="*50)
