"""Visualization functions for CKA results.

This module provides publication-quality visualization functions that
always return (Figure, Axes) tuples for further customization.
"""

from typing import List, Literal, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _normalize_series(
    values: torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray],
) -> List[np.ndarray]:
    if isinstance(values, (torch.Tensor, np.ndarray)):
        array = _to_numpy(values)
        if array.ndim == 1:
            return [array]
        if array.ndim == 2:
            return [array[i] for i in range(array.shape[0])]
        raise ValueError("values must be 1D or 2D.")
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            raise ValueError("values must contain at least one series.")
        if all(np.isscalar(item) for item in values):
            return [np.asarray(values)]
        series = []
        for item in values:
            array = _to_numpy(item)
            if array.ndim != 1:
                raise ValueError("Each series must be 1D.")
            series.append(array)
        return series
    raise TypeError("Unsupported values type.")


def _normalize_x_values(
    x_values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None,
    n_lines: int,
    n_points: int,
) -> List[np.ndarray]:
    if x_values is None:
        return [np.arange(n_points) for _ in range(n_lines)]

    if isinstance(x_values, np.ndarray) and x_values.ndim == 2:
        if x_values.shape != (n_lines, n_points):
            raise ValueError("x_values shape must match (n_lines, n_points).")
        return [x_values[i] for i in range(n_lines)]

    if isinstance(x_values, (list, tuple)) and len(x_values) > 0:
        first = x_values[0]
        if isinstance(first, (list, tuple, np.ndarray)) and not np.isscalar(first):
            if len(x_values) != n_lines:
                raise ValueError("x_values must match number of lines.")
            series = []
            for item in x_values:
                arr = np.asarray(item)
                if arr.ndim != 1 or len(arr) != n_points:
                    raise ValueError(
                        "Each x_values series must be 1D and match length."
                    )
                series.append(arr)
            return series

    arr = np.asarray(x_values)
    if arr.ndim != 1 or len(arr) != n_points:
        raise ValueError("x_values must be 1D and match series length.")
    return [arr for _ in range(n_lines)]


def _normalize_per_line(values: Sequence | None, n_lines: int, name: str) -> List:
    if values is None:
        return [None] * n_lines
    if isinstance(values, (list, tuple)):
        if len(values) == n_lines:
            return list(values)
        if len(values) == 1:
            return list(values) * n_lines
        raise ValueError(f"{name} length must match number of lines.")
    return [values for _ in range(n_lines)]


def _blend_color(
    color: Sequence[float], target: Sequence[float], alpha: float
) -> tuple:
    base = np.array(color[:3], dtype=float)
    tgt = np.array(target[:3], dtype=float)
    blended = base * (1.0 - alpha) + tgt * alpha
    return tuple(blended.tolist())


def _generate_tab10_colors(
    n_lines: int, overflow: Literal["variant", "repeat", "tab20"]
) -> List:
    base = list(plt.get_cmap("tab10").colors)
    if n_lines <= len(base):
        return base[:n_lines]

    if overflow == "repeat":
        return [base[i % len(base)] for i in range(n_lines)]

    if overflow == "tab20":
        tab20 = list(plt.get_cmap("tab20").colors)
        if n_lines <= len(tab20):
            return tab20[:n_lines]
        return [tab20[i % len(tab20)] for i in range(n_lines)]

    variants = [
        ((1.0, 1.0, 1.0), 0.35),
        ((0.0, 0.0, 0.0), 0.35),
        ((1.0, 1.0, 1.0), 0.6),
        ((0.0, 0.0, 0.0), 0.6),
        ((1.0, 1.0, 1.0), 0.8),
        ((0.0, 0.0, 0.0), 0.8),
    ]
    colors = []
    for idx in range(n_lines):
        base_color = base[idx % len(base)]
        cycle = idx // len(base)
        if cycle == 0:
            colors.append(base_color)
            continue
        target, alpha = variants[min(cycle - 1, len(variants) - 1)]
        colors.append(_blend_color(base_color, target, alpha))
    return colors


def plot_cka_heatmap(
    cka_matrix: torch.Tensor | np.ndarray,
    layers1: List[str] | None = None,
    layers2: List[str] | None = None,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    title: str | None = None,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = False,
    annot_fmt: str = ".2f",
    figsize: Tuple[float, float] | None = None,
    ax: Axes | None = None,
    colorbar: bool = True,
    tick_fontsize: int = 8,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    annot_fontsize: int = 6,
    layer_name_depth: int | None = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot CKA similarity matrix as a heatmap.

    Args:
        cka_matrix: CKA similarity matrix of shape (n_layers1, n_layers2).
        layers1: Layer names for y-axis (model1).
        layers2: Layer names for x-axis (model2).
        model1_name: Display name for model1.
        model2_name: Display name for model2.
        title: Plot title. If None, auto-generated.
        cmap: Matplotlib colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        annot: Show values in cells.
        annot_fmt: Format string for annotations.
        figsize: Figure size (width, height).
        ax: Existing axes to plot on.
        colorbar: Show colorbar.
        tick_fontsize: Font size for tick labels.
        label_fontsize: Font size for axis labels.
        title_fontsize: Font size for title.
        annot_fontsize: Font size for cell annotations.
        layer_name_depth: Number of name parts to show from end.
            E.g., 2 for "module.layer" from "encoder.module.layer".
        show: Whether to call plt.show().

    Returns:
        Tuple of (Figure, Axes).
    """
    # Convert to numpy
    if isinstance(cka_matrix, torch.Tensor):
        matrix = cka_matrix.detach().cpu().numpy()
    else:
        matrix = np.asarray(cka_matrix)

    n_layers1, n_layers2 = matrix.shape

    # Create figure if needed
    if ax is None:
        if figsize is None:
            figsize = (max(6, n_layers2 * 0.4 + 2), max(5, n_layers1 * 0.4 + 1))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Set colormap bounds
    if vmin is None:
        vmin = float(np.nanmin(matrix))
    if vmax is None:
        vmax = float(np.nanmax(matrix))

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Add annotations
    if annot:
        for i in range(n_layers1):
            for j in range(n_layers2):
                val = matrix[i, j]
                if not np.ma.is_masked(val) and not np.isnan(val):
                    text_color = "white" if val < (vmin + vmax) / 2 else "black"
                    ax.text(
                        j,
                        i,
                        format(val, annot_fmt),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=annot_fontsize,
                    )

    # Set axis labels
    ax.set_xlabel(f"{model2_name} Layers", fontsize=label_fontsize)
    ax.set_ylabel(f"{model1_name} Layers", fontsize=label_fontsize)

    # Helper function to shorten layer names
    def shorten_name(name: str, depth: int | None) -> str:
        if depth is None:
            return name
        parts = name.split(".")
        return ".".join(parts[-depth:])

    # Set tick labels
    if layers1 is not None:
        shortened = [shorten_name(layer, layer_name_depth) for layer in layers1]
        ax.set_yticks(range(n_layers1))
        ax.set_yticklabels(shortened, fontsize=tick_fontsize)

    if layers2 is not None:
        shortened = [shorten_name(layer, layer_name_depth) for layer in layers2]
        ax.set_xticks(range(n_layers2))
        ax.set_xticklabels(shortened, fontsize=tick_fontsize, rotation=45, ha="right")

    # Title
    if title is None:
        title = f"CKA: {model1_name} vs {model2_name}"
    ax.set_title(title, fontsize=title_fontsize)

    # Colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("CKA Similarity", fontsize=label_fontsize - 2)

    # Invert y-axis so layer 0 is at top
    ax.invert_yaxis()

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_cka_comparison(
    matrices: List[torch.Tensor | np.ndarray],
    titles: List[str],
    layers: List[str] | None = None,
    ncols: int = 2,
    figsize: Tuple[float, float] | None = None,
    share_colorbar: bool = True,
    cmap: str = "magma",
    show: bool = False,
    **heatmap_kwargs,
) -> Tuple[Figure, np.ndarray]:
    """Plot multiple CKA matrices side by side for comparison.

    Args:
        matrices: List of CKA matrices.
        titles: Titles for each subplot.
        layers: Layer names (shared across all plots).
        ncols: Number of columns in subplot grid.
        figsize: Figure size. If None, auto-calculated.
        share_colorbar: Use shared colorbar with same scale.
        cmap: Colormap name.
        show: Whether to call plt.show().
        **heatmap_kwargs: Additional arguments for plot_cka_heatmap.

    Returns:
        Tuple of (Figure, array of Axes).
    """
    n_plots = len(matrices)
    nrows = (n_plots + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=share_colorbar
    )
    axes = np.atleast_2d(axes)

    # Find global min/max for shared colorbar
    if share_colorbar:
        all_values = []
        for m in matrices:
            if isinstance(m, torch.Tensor):
                all_values.append(m.detach().cpu().numpy().flatten())
            else:
                all_values.append(np.asarray(m).flatten())
        all_values = np.concatenate(all_values)
        vmin = float(np.nanmin(all_values))
        vmax = float(np.nanmax(all_values))
        heatmap_kwargs["vmin"] = vmin
        heatmap_kwargs["vmax"] = vmax

    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        plot_cka_heatmap(
            matrix,
            layers1=layers,
            layers2=layers,
            title=title,
            ax=ax,
            cmap=cmap,
            colorbar=not share_colorbar,
            **heatmap_kwargs,
        )

    # Hide empty subplots
    for idx in range(n_plots, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    # Add shared colorbar
    if share_colorbar:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="CKA Similarity")
    else:
        fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_cka_trend(
    values: torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray],
    x_values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence | None = None,
    linestyles: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    xlabel: str = "Epoch",
    ylabel: str = "CKA Score",
    title: str | None = None,
    figsize: Tuple[float, float] | None = None,
    ax: Axes | None = None,
    legend: bool = False,
    grid: bool = True,
    show: bool = False,
    ylim: Tuple[float, float] | None = (0.0, 1.05),
    show_range: bool = False,
    range_values: torch.Tensor | np.ndarray | Sequence | Tuple | None = None,
    range_alpha: float = 0.2,
    color_overflow: Literal["variant", "repeat", "tab20"] = "variant",
) -> Tuple[Figure, Axes]:
    """Plot CKA trend lines over epochs, steps, or layers.

    Args:
        values: 1D array for a single line, 2D array (n_lines, n_points),
            or a list of 1D arrays.
        x_values: Optional x-axis values (shared) or per-line x-axis values.
        labels: Legend labels for lines.
        colors: Line colors. Defaults to Tableau10 (tab10) with overflow handling.
        linestyles: Line styles per line.
        markers: Marker styles per line.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        figsize: Figure size (width, height).
        ax: Existing axes to plot on.
        legend: Show legend (only if multiple lines).
        grid: Show grid.
        show: Whether to call plt.show().
        ylim: Y-axis limits. Defaults to (0, 1.05).
        show_range: Whether to show value ranges as shaded bands.
        range_values: Standard deviation values for shading, or (lower, upper) tuple.
        range_alpha: Alpha for shaded range.
        color_overflow: Strategy for >10 lines: "variant", "repeat", or "tab20".

    Returns:
        Tuple of (Figure, Axes).
    """
    series = _normalize_series(values)
    n_lines = len(series)
    n_points = len(series[0])
    if any(len(line) != n_points for line in series):
        raise ValueError("All series must have the same length.")

    x_series = _normalize_x_values(x_values, n_lines, n_points)

    if colors is None:
        colors = _generate_tab10_colors(n_lines, color_overflow)
    colors = _normalize_per_line(colors, n_lines, "colors")
    linestyles = _normalize_per_line(linestyles, n_lines, "linestyles")
    markers = _normalize_per_line(markers, n_lines, "markers")
    labels = _normalize_per_line(labels, n_lines, "labels")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    lower_series = upper_series = None
    if show_range:
        if range_values is None:
            raise ValueError(
                "show_range=True requires range_values or use plot_cka_trend_with_range."
            )
        if isinstance(range_values, tuple) and len(range_values) == 2:
            lower_series = _normalize_series(range_values[0])
            upper_series = _normalize_series(range_values[1])
            if len(lower_series) == 1 and n_lines > 1:
                lower_series *= n_lines
            if len(upper_series) == 1 and n_lines > 1:
                upper_series *= n_lines
        else:
            std_series = _normalize_series(range_values)
            if len(std_series) == 1 and n_lines > 1:
                std_series *= n_lines
            if len(std_series) != n_lines:
                raise ValueError("range_values must match number of lines.")
            lower_series = [s - r for s, r in zip(series, std_series)]
            upper_series = [s + r for s, r in zip(series, std_series)]

        if len(lower_series) != n_lines or len(upper_series) != n_lines:
            raise ValueError("range_values must match number of lines.")
        for lower, upper in zip(lower_series, upper_series):
            if len(lower) != n_points or len(upper) != n_points:
                raise ValueError("range_values length must match series length.")

    for idx, (line, x_line) in enumerate(zip(series, x_series)):
        ax.plot(
            x_line,
            line,
            color=colors[idx],
            linestyle=linestyles[idx],
            marker=markers[idx],
            label=labels[idx],
        )
        if show_range:
            ax.fill_between(
                x_line,
                lower_series[idx],
                upper_series[idx],
                color=colors[idx],
                alpha=range_alpha,
                linewidth=0,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(grid, linestyle="--", alpha=0.3)

    if legend and n_lines > 1:
        if all(label is None for label in labels):
            ax.legend([f"Series {i + 1}" for i in range(n_lines)])
        else:
            ax.legend()

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_cka_trend_with_range(
    values: torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray],
    x_values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence | None = None,
    linestyles: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    xlabel: str = "Epoch",
    ylabel: str = "CKA Similarity",
    title: str | None = None,
    figsize: Tuple[float, float] | None = None,
    ax: Axes | None = None,
    legend: bool = False,
    grid: bool = True,
    show: bool = False,
    ylim: Tuple[float, float] | None = (0.0, 1.05),
    range_alpha: float = 0.2,
    color_overflow: Literal["variant", "repeat", "tab20"] = "variant",
) -> Tuple[Figure, Axes]:
    """Plot mean ± std trends from repeated CKA measurements.

    Args:
        values: 2D array (n_runs, n_points) for a single line, 3D array
            (n_lines, n_runs, n_points), or a list of 2D arrays.
    """
    if isinstance(values, (torch.Tensor, np.ndarray)):
        array = _to_numpy(values)
        if array.ndim == 2:
            grouped = [array]
        elif array.ndim == 3:
            grouped = [array[i] for i in range(array.shape[0])]
        else:
            raise ValueError("values must be 2D or 3D for range plotting.")
    elif isinstance(values, (list, tuple)):
        if len(values) == 0:
            raise ValueError("values must contain at least one group.")
        grouped = []
        for item in values:
            array = _to_numpy(item)
            if array.ndim != 2:
                raise ValueError("Each group must be 2D (n_runs, n_points).")
            grouped.append(array)
    else:
        raise TypeError("Unsupported values type.")

    means = [np.nanmean(group, axis=0) for group in grouped]
    stds = [np.nanstd(group, axis=0) for group in grouped]

    return plot_cka_trend(
        means,
        x_values=x_values,
        labels=labels,
        colors=colors,
        linestyles=linestyles,
        markers=markers,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figsize=figsize,
        ax=ax,
        legend=legend,
        grid=grid,
        show=show,
        ylim=ylim,
        show_range=True,
        range_values=stds,
        range_alpha=range_alpha,
        color_overflow=color_overflow,
    )


def plot_cka_layer_trend(
    cka_matrices: torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray],
    layers: Sequence[str] | None = None,
    layer_name_depth: int | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence | None = None,
    linestyles: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    xlabel: str = "Layer",
    ylabel: str = "CKA Similarity",
    title: str | None = None,
    figsize: Tuple[float, float] | None = None,
    ax: Axes | None = None,
    legend: bool = False,
    grid: bool = True,
    show: bool = False,
    ylim: Tuple[float, float] | None = (0.0, 1.05),
    color_overflow: Literal["variant", "repeat", "tab20"] = "variant",
) -> Tuple[Figure, Axes]:
    """Plot diagonal CKA values across layers for one or more matrices."""
    if isinstance(cka_matrices, (torch.Tensor, np.ndarray)):
        matrices = [_to_numpy(cka_matrices)]
    elif isinstance(cka_matrices, (list, tuple)):
        if len(cka_matrices) == 0:
            raise ValueError("cka_matrices must contain at least one matrix.")
        matrices = [_to_numpy(m) for m in cka_matrices]
    else:
        raise TypeError("Unsupported cka_matrices type.")

    diagonals = [np.diag(matrix) for matrix in matrices]
    n_points = len(diagonals[0])
    if layers is not None and len(layers) != n_points:
        raise ValueError("layers length must match diagonal length.")

    # Use "o" marker if none specified
    plot_markers = markers if markers is not None else ["o"]

    fig, ax = plot_cka_trend(
        diagonals,
        labels=labels,
        colors=colors,
        linestyles=linestyles,
        markers=plot_markers,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figsize=figsize,
        ax=ax,
        legend=legend,
        grid=grid,
        show=show,
        ylim=ylim,
        color_overflow=color_overflow,
    )

    # Set tick labels
    tick_indices = list(range(n_points))
    if layers is not None:
        if layer_name_depth is not None:
            shortened = [
                ".".join(layer.split(".")[-layer_name_depth:]) for layer in layers
            ]
        else:
            shortened = list(layers)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(shortened, rotation=45, ha="right")
    else:
        ax.set_xticks(tick_indices)

    return fig, ax


def save_figure(
    fig: Figure,
    path: str,
    dpi: int = 150,
    bbox_inches: str = "tight",
    transparent: bool = False,
) -> None:
    """Save figure to file with sensible defaults.

    Args:
        fig: Matplotlib figure.
        path: Output path.
        dpi: Resolution in dots per inch.
        bbox_inches: Bounding box setting.
        transparent: Whether background should be transparent.
    """
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    plt.close(fig)  # Close to free memory
