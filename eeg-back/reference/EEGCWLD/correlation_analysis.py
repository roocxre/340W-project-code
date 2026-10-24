import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


def compute_correlation(
    raw: mne.io.Raw,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation matrix for EEG channels.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data object.
    method : str, optional
        Correlation method ('pearson', 'spearman', or 'kendall'). 
        Default is 'pearson'.
    
    Returns
    -------
    pandas.DataFrame
        Correlation matrix with channel names as index and columns.
    
    Examples
    --------
    >>> raw = load_edf('data/subject1.edf')
    >>> corr = compute_correlation(raw)
    """
    # Get data
    data, _ = raw.get_data(return_times=True)
    
    # Create DataFrame
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    
    # Compute correlation
    correlation = df.corr(method=method)
    
    return correlation


def compute_correlation_comparison(
    raw_data: Dict[str, mne.io.Raw],
    method: str = 'pearson'
) -> Dict:
    """
    Compute correlation matrices for multiple conditions.
    
    Parameters
    ----------
    raw_data : dict
        Dictionary with condition names as keys and Raw objects as values.
    method : str, optional
        Correlation method. Default is 'pearson'.
    
    Returns
    -------
    dict
        Dictionary with correlation matrices for each condition.
    
    Examples
    --------
    >>> files = {
    ...     'No Workload': 'data/sub02_nw.edf',
    ...     'High Workload': 'data/sub02_hw.edf'
    ... }
    >>> raw_data = {cond: load_edf(path) for cond, path in files.items()}
    >>> results = compute_correlation_comparison(raw_data)
    """
    correlations = {}
    
    for condition, raw in raw_data.items():
        try:
            correlations[condition] = compute_correlation(raw, method=method)
        except Exception as e:
            print(f"Warning: Error computing correlation for {condition}: {e}")
    
    return correlations


def compute_difference_maps(
    correlations: Dict[str, pd.DataFrame],
    reference_condition: str = 'No Workload'
) -> Dict:
    """
    Compute difference maps between conditions.
    
    Parameters
    ----------
    correlations : dict
        Dictionary with condition names as keys and DataFrames as values.
    reference_condition : str, optional
        Reference condition for difference calculation.
        Default is 'No Workload'.
    
    Returns
    -------
    dict
        Dictionary with difference maps for each condition vs reference.
    
    Examples
    --------
    >>> corr = compute_correlation_comparison(raw_data)
    >>> diff = compute_difference_maps(corr)
    """
    if reference_condition not in correlations:
        raise ValueError(f"Reference condition '{reference_condition}' not found")
    
    ref_corr = correlations[reference_condition]
    diff_maps = {}
    
    for condition, corr in correlations.items():
        if condition != reference_condition:
            diff_maps[f"{condition} - {reference_condition}"] = corr - ref_corr
    
    return diff_maps


def plot_correlation_matrix(
    correlation: pd.DataFrame,
    title: str = "EEG Channel Correlation",
    save_path: Optional[str] = None,
    dpi: int = 300,
    figsize: Optional[Tuple] = None,
    cmap: str = 'coolwarm',
    center: float = 0
) -> plt.Figure:
    """
    Plot single correlation matrix as heatmap.
    
    Parameters
    ----------
    correlation : pandas.DataFrame
        Correlation matrix.
    title : str, optional
        Plot title. Default is "EEG Channel Correlation".
    save_path : str or None, optional
        Path to save figure. Default is None.
    dpi : int, optional
        DPI for saved figure. Default is 300.
    figsize : tuple or None, optional
        Figure size. Default is auto-calculated.
    cmap : str, optional
        Colormap name. Default is 'coolwarm'.
    center : float, optional
        Center value for colormap. Default is 0.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Examples
    --------
    >>> corr = compute_correlation(raw)
    >>> fig = plot_correlation_matrix(corr, save_path='correlation.png')
    """
    if figsize is None:
        n = len(correlation)
        figsize = (n/2, n/2)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sns.heatmap(
        correlation,
        ax=ax,
        cmap=cmap,
        center=center,
        square=True,
        cbar=False,
        linewidths=0.5
    )
    
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_correlation_comparison(
    correlations: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    dpi: int = 1500,
    figsize: Optional[Tuple] = None
) -> plt.Figure:
    """
    Create comparison plot of correlation matrices for multiple conditions.
    
    Parameters
    ----------
    correlations : dict
        Dictionary with condition names as keys and DataFrames as values.
    save_path : str or None, optional
        Path to save figure. Default is None.
    dpi : int, optional
        DPI for saved figure. Default is 1500.
    figsize : tuple or None, optional
        Figure size. Default is auto-calculated.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Examples
    --------
    >>> corr = compute_correlation_comparison(raw_data)
    >>> fig = plot_correlation_comparison(corr, save_path='corr_comparison.png')
    """
    n = len(correlations)
    
    if figsize is None:
        n_ch = len(list(correlations.values())[0])
        figsize = (n_ch/2 * n, n_ch/2)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    # Ensure axes is a list
    if n == 1:
        axes = [axes]
    
    for ax, (condition, corr) in zip(axes, correlations.items()):
        sns.heatmap(
            corr,
            ax=ax,
            cmap='coolwarm',
            center=0,
            square=True,
            cbar=False,
            linewidths=0.5
        )
        ax.set_title(condition, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_correlation_with_difference(
    correlations: Dict[str, pd.DataFrame],
    reference_condition: str = 'No Workload',
    save_path: Optional[str] = None,
    dpi: int = 1500,
    figsize: Optional[Tuple] = None
) -> plt.Figure:
    """
    Create comparison plot with correlation matrices and difference maps.
    
    Parameters
    ----------
    correlations : dict
        Dictionary with condition names as keys and DataFrames as values.
    reference_condition : str, optional
        Reference condition. Default is 'No Workload'.
    save_path : str or None, optional
        Path to save figure. Default is None.
    dpi : int, optional
        DPI for saved figure. Default is 1500.
    figsize : tuple or None, optional
        Figure size. Default is auto-calculated.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Examples
    --------
    >>> corr = compute_correlation_comparison(raw_data)
    >>> fig = plot_correlation_with_difference(corr, save_path='corr_with_diff.png')
    """
    # Compute difference maps
    diff_maps = compute_difference_maps(correlations, reference_condition)
    
    # Create lists for plotting
    all_plots = list(correlations.items()) + list(diff_maps.items())
    n = len(all_plots)
    
    if figsize is None:
        n_ch = len(list(correlations.values())[0])
        figsize = (n_ch/2 * n, n_ch/2)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, (title, corr) in zip(axes, all_plots):
        # Use 'bwr' colormap for difference maps
        cmap = 'coolwarm' if 'No' in title else 'bwr'
        
        sns.heatmap(
            corr,
            ax=ax,
            cmap=cmap,
            center=0,
            square=True,
            cbar=True if ax == axes[-1] else False,
            linewidths=0.4
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_workload_correlation_grid(
    correlations: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    dpi: int = 1200,
    figsize: Tuple = (16, 16)
) -> plt.Figure:
    """
    Create grid layout for workload correlation analysis.
    
    Parameters
    ----------
    correlations : dict
        Dictionary with condition names as keys and DataFrames as values.
    save_path : str or None, optional
        Path to save figure. Default is None.
    dpi : int, optional
        DPI for saved figure. Default is 1200.
    figsize : tuple, optional
        Figure size. Default is (16, 16).
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    
    Examples
    --------
    >>> corr = compute_correlation_comparison(raw_data)
    >>> fig = plot_workload_correlation_grid(corr, save_path='corr_grid.png')
    """
    n_conditions = len(correlations)
    
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    
    conditions = list(correlations.keys())
    
    # Plot correlation matrices in first column
    for i, condition in enumerate(conditions):
        if i < 4:
            sns.heatmap(
                correlations[condition],
                ax=axes[i, 0],
                cmap='coolwarm',
                center=0,
                square=True,
                cbar=False,
                linewidths=0.4
            )
            axes[i, 0].set_title(condition, fontsize=12)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
    
    # Plot difference maps in second column (if enough conditions)
    diff_maps = compute_difference_maps(correlations, conditions[0])
    diff_titles = list(diff_maps.keys())
    
    for i, diff_title in enumerate(diff_titles):
        if i < 4:
            sns.heatmap(
                diff_maps[diff_title],
                ax=axes[i, 1],
                cmap='bwr',
                center=0,
                square=True,
                cbar=False,
                linewidths=0.4
            )
            axes[i, 1].set_title(diff_title, fontsize=12)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
    
    # Hide unused axes
    for i in range(len(conditions), 4):
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
    
    fig.suptitle("EEG Channel Correlation and Differences Across Cognitive Workload Levels",
                 fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def get_channel_correlation_stats(
    correlation: pd.DataFrame
) -> Dict:
    """
    Get summary statistics for correlation matrix.
    
    Parameters
    ----------
    correlation : pandas.DataFrame
        Correlation matrix.
    
    Returns
    -------
    dict
        Dictionary with correlation statistics.
    
    Examples
    --------
    >>> corr = compute_correlation(raw)
    >>> stats = get_channel_correlation_stats(corr)
    """
    # Get upper triangle values (excluding diagonal)
    n = len(correlation)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    values = correlation.values[mask]
    
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'n_channels': n
    }
