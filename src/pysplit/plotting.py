"""Pie-chart visualisation of doublet-mode deconvolution.

Each cell is drawn at its spatial position as a pie whose slices are its
primary and secondary cell types, sized by their weights. The pie *edge*
colour marks the primary cell type; a central marker encodes the spot class:

* coloured dot - ``doublet_certain`` (colour = secondary cell type),
* black dot - ``doublet_uncertain``,
* cross - ``reject``,
* nothing - confident ``singlet``.

This makes local contamination obvious: a T cell sitting inside a tumour
region shows up as a pie half-filled with the tumour colour.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import get_logger
from .rctd import RCTDResult

logger = get_logger()

__all__ = [
    "pie_dataframe",
    "crop_pie_dataframe",
    "plot_pie",
    "plot_pie_by_coordinates",
    "plot_pie_around_cell",
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib (pip install 'pysplit-st[plot]')."
        ) from exc
    return plt


def pie_dataframe(rctd: RCTDResult, coords: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-cell frame with one weight column per cell type, ready for plotting.

    Parameters
    ----------
    rctd
        A post-processed deconvolution result.
    coords
        ``x``/``y`` coordinates per cell. Optional if ``results_df`` already
        carries ``x``/``y``.

    Returns
    -------
    pandas.DataFrame
        Columns ``x``, ``y``, ``spot_class``, ``first_type``, ``second_type``,
        the doublet weights, plus one column per cell type holding that type's
        share of the cell (primary weight, secondary weight, or zero).
    """
    rctd.check_post_processed()
    df = rctd.results_df
    keep = [
        c
        for c in (
            "spot_class",
            "first_type",
            "second_type",
            "weight_first_type",
            "weight_second_type",
            "w1_larger_w2",
            "x",
            "y",
        )
        if c in df.columns
    ]
    out = df[keep].copy()
    if coords is not None:
        aligned = coords.reindex(out.index)
        out["x"] = aligned.iloc[:, 0].to_numpy()
        out["y"] = aligned.iloc[:, 1].to_numpy()
    if "x" not in out.columns or "y" not in out.columns:
        raise ValueError(
            "No coordinates available: pass `coords` or set `rctd.coords` before "
            "post-processing."
        )

    cell_types = rctd.cell_types
    shares = pd.DataFrame(0.0, index=out.index, columns=cell_types)
    first = out["first_type"].astype(object).to_numpy()
    second = out["second_type"].astype(object).to_numpy()
    w1 = pd.to_numeric(out["weight_first_type"], errors="coerce").to_numpy()
    type_pos = {t: i for i, t in enumerate(cell_types)}
    values = shares.to_numpy()
    for i in range(len(out)):
        ft = first[i]
        if ft is None or pd.isna(ft) or ft not in type_pos:
            continue
        st = second[i]
        if st is None or pd.isna(st) or st not in type_pos:
            values[i, type_pos[ft]] = 1.0
        else:
            weight = 1.0 if np.isnan(w1[i]) else float(w1[i])
            values[i, type_pos[ft]] = weight
            values[i, type_pos[st]] = 1.0 - weight
    shares = pd.DataFrame(values, index=out.index, columns=cell_types)

    out["cell_id"] = out.index
    return pd.concat([out, shares], axis=1)


def crop_pie_dataframe(
    pie_df: pd.DataFrame, cell_id: str, radius: float = 100.0
) -> pd.DataFrame:
    """Square crop of ``radius`` around ``cell_id``."""
    if cell_id not in pie_df.index:
        raise KeyError(f"{cell_id!r} is not in the pie DataFrame.")
    x_center = float(pie_df.loc[cell_id, "x"])
    y_center = float(pie_df.loc[cell_id, "y"])
    return pie_df[
        pie_df["x"].between(x_center - radius, x_center + radius)
        & pie_df["y"].between(y_center - radius, y_center + radius)
    ]


def _cell_types_in(pie_df: pd.DataFrame) -> list[str]:
    labels = pd.unique(
        np.concatenate(
            [
                pie_df["first_type"].astype(object).to_numpy(),
                pie_df["second_type"].astype(object).to_numpy(),
            ]
        )
    )
    return [str(x) for x in labels if not pd.isna(x)]


def plot_pie(
    pie_df: pd.DataFrame,
    palette: Mapping[str, str] | None = None,
    pie_radius: float | None = None,
    ax=None,
    legend: bool = True,
    edge_width: float = 1.2,
):
    """Draw each cell as a weight-proportional pie at its spatial position.

    Parameters
    ----------
    pie_df
        Output of :func:`pie_dataframe`, usually cropped first - one pie per
        cell gets slow and illegible beyond a few thousand cells.
    palette
        Cell type -> colour. Missing types fall back to a tab20 cycle.
    pie_radius
        Pie radius in data units. Defaults to 1/40th of the plotted extent.
    ax
        Existing axes to draw on.
    legend
        Draw a cell-type legend.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _require_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Wedge

    if len(pie_df) == 0:
        raise ValueError("`pie_df` is empty - nothing to plot.")
    if len(pie_df) > 5000:
        logger.warning(
            "Drawing %d pies; consider cropping with crop_pie_dataframe() first.",
            len(pie_df),
        )

    cell_types = _cell_types_in(pie_df)
    colors = dict(palette or {})
    cycle = plt.get_cmap("tab20").colors
    for i, cell_type in enumerate(sorted(cell_types)):
        colors.setdefault(cell_type, cycle[i % len(cycle)])

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    x = pie_df["x"].to_numpy(dtype=float)
    # y is flipped so the plot matches the image orientation
    y = -pie_df["y"].to_numpy(dtype=float)
    if pie_radius is None:
        extent = max(np.ptp(x), np.ptp(y))
        pie_radius = (extent / 40.0) if extent > 0 else 1.0

    spot_class = pie_df["spot_class"].astype(object).to_numpy()
    first = pie_df["first_type"].astype(object).to_numpy()
    second = pie_df["second_type"].astype(object).to_numpy()

    for i in range(len(pie_df)):
        center = (x[i], y[i])
        edge = colors.get(first[i], "lightgrey")
        weights = [
            (t, float(pie_df.iloc[i][t]))
            for t in cell_types
            if t in pie_df.columns and float(pie_df.iloc[i][t]) > 0
        ]
        total = sum(w for _, w in weights)
        if total <= 0:
            ax.add_patch(
                Circle(center, pie_radius, facecolor="lightgrey", edgecolor=edge,
                       linewidth=edge_width)
            )
            continue
        start = 90.0
        for cell_type, weight in weights:
            span = 360.0 * weight / total
            ax.add_patch(
                Wedge(
                    center,
                    pie_radius,
                    start - span,
                    start,
                    facecolor=colors.get(cell_type, "lightgrey"),
                    edgecolor="none",
                )
            )
            start -= span
        ax.add_patch(
            Circle(center, pie_radius, facecolor="none", edgecolor=edge,
                   linewidth=edge_width)
        )

        # spot-class marker in the middle of the pie
        sc = spot_class[i]
        if sc == "doublet_certain":
            ax.plot(*center, marker="o", markersize=4,
                    color=colors.get(second[i], "black"), zorder=5)
        elif sc == "doublet_uncertain":
            ax.plot(*center, marker="o", markersize=4, color="black", zorder=5)
        elif sc == "reject":
            ax.plot(*center, marker="x", markersize=5, color="black", zorder=5)

    pad = 2 * pie_radius
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if legend:
        handles = [
            Line2D([], [], marker="o", linestyle="", markerfacecolor=colors[t],
                   markeredgecolor="none", markersize=8, label=t)
            for t in sorted(cell_types)
        ]
        handles += [
            Line2D([], [], marker="o", linestyle="", color="black", markersize=4,
                   label="doublet_uncertain"),
            Line2D([], [], marker="x", linestyle="", color="black", markersize=5,
                   label="reject"),
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=8)
    return ax


def plot_pie_by_coordinates(
    pie_df: pd.DataFrame,
    x_lims: Sequence[float],
    y_lims: Sequence[float],
    **kwargs,
):
    """:func:`plot_pie` restricted to a coordinate window."""
    cropped = pie_df[
        pie_df["x"].between(min(x_lims), max(x_lims))
        & pie_df["y"].between(min(y_lims), max(y_lims))
    ]
    return plot_pie(cropped, **kwargs)


def plot_pie_around_cell(
    pie_df: pd.DataFrame,
    cell_id: str,
    radius: float = 100.0,
    highlight: bool = True,
    **kwargs,
):
    """:func:`plot_pie` around one cell, which is marked with a star."""
    cropped = crop_pie_dataframe(pie_df, cell_id=cell_id, radius=radius)
    ax = plot_pie(cropped, **kwargs)
    if highlight:
        ax.plot(
            float(cropped.loc[cell_id, "x"]),
            -float(cropped.loc[cell_id, "y"]),
            marker="*",
            markersize=9,
            color="black",
            zorder=6,
        )
    return ax
