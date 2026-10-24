"""
BEAM Topographic Mapping
=======================

Functions for Brain Electrical Activity Mapping (BEAM) and topographic 
visualization of EEG power distribution.

Author: Research Team
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


# Default frequency bands for BEAM analysis
BEAM_BANDS = {
    "Theta": (4, 8),
    "Alpha": (8, 13)
}

# Default colormap
DEFAULT_CMAP = 'jet'


def compute_topomap(
    raw: mne.io.Raw,
    fmin: float,
    fmax: float,
    n_fft: int = 256
) -> Tuple[np.ndarray, mne.Info]:
    """
    Compute topographic power map for specified frequency band.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object.
    fmin : float
        Low frequency bound.
    fmax : float
        High frequency bound.
    n_fft : int, optional
        Number of FFT points. Default is 256.
    
    Returns
    -------
    tuple
        (mean_power, info) where mean_power is the average power per channel
        and info is the MNE info object.
    
    Examples
    --------
    >>> raw = load_edf('data/subject1.edf')
    >>> power, info = compute_topomap(raw, 8, 13)  # Alpha band
    """
    # Filter data
    raw_filtered = raw.copy().filter(fmin, fmax, fir_design='firwin')
    
    # Create epochs
    events = mne.make_fixed_length_events(raw_filtered, duration=2.0)
    picks = mne.pick_types(raw_filtered.info, eeg=True, exclude='bads')
    epochs = mne.Epochs(
        raw_filtered, events, tmin=0, tmax=2.0,
        baseline=None, picks=picks, preload=True
    )
    
    # Compute PSD
    n_fft = min(n_fft, len(epochs.times))
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft)
    psds = psd.get_data()
    
    # Calculate mean power across time
    mean_power = psds.mean(axis=(0, 2))
    
    return mean_power, epochs.info


def compute_beam_comparison(
    raw_data: Dict[str, mne.io.Raw],
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict:
    """
    Compute BEAM maps for multiple conditions and frequency bands.
    
    Parameters
    ----------
    raw_data : dict
        Dictionary with condition names as keys and Raw objects as values.
    bands : dict or None, optional
        Dictionary with band names as keys and (fmin, fmax) tuples as values.
        Default is Theta and Alpha bands.
    
    Returns
    -------
    dict
        Dictionary containing BEAM data for each condition and band.
    
    Examples
    --------
    >>> files = {'No Workload': 'data/nw.edf', 'Workload': 'data/hw.edf'}
    >>> raw_data = {cond: load_edf(path) for cond, path in files.items()}
    >>> results = compute_beam_comparison(raw_data)
    """
    if bands is None:
        bands = BEAM_BANDS
    
    results = {
        'bands': bands,
        'conditions': {}
    }
    
    for condition, raw in raw_data.items():
        results['conditions'][condition] = {}
        
        for band_name, (fmin, fmax) in bands.items():
            try:
                power, info = compute_topomap(raw, fmin, fmax)
                results['conditions'][condition][band_name] = {
                    'power': power,
                    'info': info
                }
            except Exception as e:
                print(f"Warning: Error computing {band_name} for {condition}: {e}")
    
    return results


def plot_beam_grid(
    results: Dict,
    save_path: Optional[str] = None,
    dpi: int = 300,
    figsize: Optional[Tuple] = None,
    cmap: str = DEFAULT_CMAP
) -> plt.Figure:
    """
    Create grid of BEAM topographic maps.
    
    Parameters
    ----------
    results : dict
        Results from compute_beam_comparison.
    save_path : str or None, optional
        Path to save figure. Default is None.
    dpi : int, optional
        DPI for saved figure. Default is 300.
    figsize : tuple or None, optional
        Figure size. Default is auto-calculated based on grid.
    cmap : str, optional
        Colormap name. Default is 'jet'.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Examples
    --------
    >>> results = compute_beam_comparison(raw_data)
    >>> fig = plot_beam_grid(results, save_path='beam_maps.png')
    """
    bands = results['bands']
    conditions = results['conditions']
    
    n_bands = len(bands)
    n_conditions = len(conditions)
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (3 * n_conditions, 3 * n_bands / 2)
    
    fig, axes = plt.subplots(n_bands, n_conditions, figsize=figsize)
    
    # Ensure axes is 2D
    if n_bands == 1:
        axes = axes.reshape(1, -1)
    if n_conditions == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each condition and band
    for row, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        for col, (condition, band_data) in enumerate(conditions.items()):
            ax = axes[row, col]
            
            if band_name in band_data:
                power = band_data[band_name]['power']
                info = band_data[band_name]['info']
                
                mne.viz.plot_topomap(
                    power, info, 
                    axes=ax, 
                    show=False, 
                    contours=0, 
                    cmap=cmap
                )
                
                title = f"{band_name}\n{condition}"
                ax.set_title(title, fontsize=9)
            else:
                ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_single_beam(
    raw: mne.io.Raw,
    fmin: float,
    fmax: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cmap: str = DEFAULT_CMAP
) -> plt.Figure:
    """
    Plot single BEAM topographic map.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object.
    fmin : float
        Low frequency bound.
    fmax : float
        High frequency bound.
    title : str or None, optional
        Plot title. Default is auto-generated.
    save_path : str or None, optional
        Path to save figure. Default is None.
    cmap : str, optional
        Colormap name. Default is 'jet'.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Examples
    --------
    >>> raw = load_edf('data/subject1.edf')
    >>> fig = plot_single_beam(raw, 8, 13, save_path='alpha_beam.png')
    """
    power, info = compute_topomap(raw, fmin, fmax)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    mne.viz.plot_topomap(
        power, info,
        axes=ax,
        show=False,
        contours=0,
        cmap=cmap
    )
    
    if title is None:
        title = f"BEAM Map: {fmin}-{fmax} Hz"
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_beam_difference(
    results: Dict,
    reference_condition: str = 'No Workload',
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create difference maps showing changes from reference condition.
    
    Parameters
    ----------
    results : dict
        Results from compute_beam_comparison.
    reference_condition : str, optional
        Reference condition for difference calculation.
        Default is 'No Workload'.
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
    >>> results = compute_beam_comparison(raw_data)
    >>> fig = plot_beam_difference(results, save_path='beam_diff.png')
    """
    bands = results['bands']
    conditions = results['conditions']
    
    # Get reference data
    if reference_condition not in conditions:
        print(f"Warning: Reference condition '{reference_condition}' not found")
        return None
    
    ref_conditions = [c for c in conditions.keys() if c != reference_condition]
    
    n_bands = len(bands)
    n_diff = len(ref_conditions)
    
    if n_diff == 0:
        print("Warning: No conditions to compare")
        return None
    
    figsize = (3 * n_diff, 3 * n_bands / 2)
    fig, axes = plt.subplots(n_bands, n_diff, figsize=figsize)
    
    # Ensure axes is 2D
    if n_bands == 1:
        axes = axes.reshape(1, -1)
    if n_diff == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot difference maps
    for row, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        for col, condition in enumerate(ref_conditions):
            ax = axes[row, col]
            
            ref_power = conditions[reference_condition][band_name]['power']
            comp_power = conditions[condition][band_name]['power']
            info = conditions[condition][band_name]['info']
            
            diff_power = comp_power - ref_power
            
            mne.viz.plot_topomap(
                diff_power, info,
                axes=ax,
                show=False,
                contours=0,
                cmap='bwr'
            )
            
            title = f"{band_name}\n{condition} - {reference_condition}"
            ax.set_title(title, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig
