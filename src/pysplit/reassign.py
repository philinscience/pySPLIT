"""Reassign removed transcripts to their likely neighbouring cell of origin.

Purification deletes contaminating counts. Physically, though, those
transcripts belong to *some* nearby cell - typically a spatial neighbour whose
primary cell type matches the contaminated cell's secondary type. Rather than
discarding them, they can be redistributed to those neighbours, which recovers
sensitivity without reintroducing contamination.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ._utils import as_csr, get_logger
from .neighbors import Neighborhood

logger = get_logger()

__all__ = ["build_reassignment_operator", "reassign_residual_counts"]


def build_reassignment_operator(
    neighborhood: Neighborhood,
    cells_with_residual: Sequence[str],
    ncount: Mapping[str, float] | pd.Series | None = None,
    self_keep: float = 0.0,
    weight_power: float = 1.0,
    eps: float = 0.0,
) -> sparse.csr_matrix:
    """Build the ``cells x cells`` operator that moves residual counts around.

    Row ``i`` distributes cell ``i``'s residual over the neighbours whose
    primary cell type is ``i``'s secondary type (the
    ``second_type_neighbors_no_reject`` field), so the transcripts land on the
    cells they most plausibly came from. Rows sum to one (plus ``self_keep``
    retained on the diagonal).

    Parameters
    ----------
    neighborhood
        A spatial neighbourhood with ``second_type_neighbors_no_reject``
        computed (i.e. after :func:`~pysplit.spatial_metrics`).
    cells_with_residual
        Cells whose residual should be redistributed - usually the cells with
        ``purification_status == 'purified'``.
    ncount
        Per-cell total counts. When given, a recipient's share is proportional
        to ``(nCount + eps) ** weight_power`` instead of uniform.
    self_keep
        Fraction of the residual the donor cell keeps for itself.
    weight_power, eps
        Shape the count-proportional weighting.

    Returns
    -------
    scipy.sparse.csr_matrix
        Operator indexed by ``neighborhood.cell_id`` on both axes.
    """
    neighborhood.require(
        "second_type_neighbors_no_reject",
        hint="Run spatial_metrics() on a spatial neighborhood first.",
    )
    if not 0.0 <= self_keep <= 1.0:
        raise ValueError(f"`self_keep` must lie in [0, 1], got {self_keep}.")

    cell_ids = list(neighborhood.cell_id)
    position = {c: i for i, c in enumerate(cell_ids)}
    n = len(cell_ids)

    weights_vector = None
    if ncount is not None:
        series = pd.Series(ncount)
        if series.index.equals(pd.RangeIndex(len(series))) and len(series) == n:
            series.index = cell_ids
        series = series.reindex(cell_ids).astype(float)
        weights_vector = (np.clip(series.to_numpy(), 0, None) + eps) ** weight_power

    nn_idx = neighborhood.nn_idx
    neighbour_slots = neighborhood.fields["second_type_neighbors_no_reject"]

    rows, cols, values = [], [], []
    missing = 0
    for cell in cells_with_residual:
        i = position.get(str(cell))
        if i is None:
            missing += 1
            continue
        recipients = nn_idx[i, neighbour_slots[i]]
        recipients = recipients[recipients >= 0]
        k = recipients.size
        if k == 0:
            continue  # nowhere to send it: the residual is simply dropped
        if self_keep > 0:
            rows.append(i)
            cols.append(i)
            values.append(self_keep)
        if weights_vector is not None:
            share = weights_vector[recipients]
            share = np.where(np.isfinite(share), share, 0.0)
            total = share.sum()
            share = share / total if total > 0 else np.full(k, 1.0 / k)
        else:
            share = np.full(k, 1.0 / k)
        rows.extend([i] * k)
        cols.extend(recipients.tolist())
        values.extend((share * (1.0 - self_keep)).tolist())

    if missing:
        logger.warning(
            "%d cell(s) in `cells_with_residual` are absent from the "
            "neighborhood and were skipped.",
            missing,
        )
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(n, n), dtype=float
    )


def reassign_residual_counts(
    raw_counts,
    purified_counts,
    neighborhood: Neighborhood,
    purification_status: pd.Series,
    cell_names: Sequence[str] | None = None,
    mode: str = "uniform",
    return_operator: bool = False,
    **kwargs,
):
    """Add each cell's removed counts back onto its likely neighbours of origin.

    Parameters
    ----------
    raw_counts, purified_counts
        ``cells x genes`` matrices on the same axes; the difference is the
        residual that gets redistributed.
    neighborhood
        Spatial neighbourhood after :func:`~pysplit.spatial_metrics`.
    purification_status
        Per-cell Series; only cells marked ``'purified'`` donate their residual.
    cell_names
        Row labels of the matrices. Required unless they are DataFrames.
    mode
        ``'uniform'`` splits a residual equally between recipients;
        ``'count_proportional'`` weights recipients by their total counts.
    return_operator
        Also return the reassignment operator.
    **kwargs
        Forwarded to :func:`build_reassignment_operator` (``self_keep``,
        ``weight_power``, ``eps``).

    Returns
    -------
    scipy.sparse.csr_matrix
        Corrected counts, or ``(counts, operator)`` when ``return_operator``.
    """
    if isinstance(raw_counts, pd.DataFrame):
        cell_names = list(raw_counts.index) if cell_names is None else cell_names
        raw_counts = raw_counts.to_numpy()
    if isinstance(purified_counts, pd.DataFrame):
        purified_counts = purified_counts.to_numpy()
    if cell_names is None:
        raise ValueError("`cell_names` is required when the matrices are not DataFrames.")
    cell_names = list(map(str, cell_names))

    raw = as_csr(raw_counts)
    purified = as_csr(purified_counts)
    if raw.shape != purified.shape:
        raise ValueError(
            f"raw_counts {raw.shape} and purified_counts {purified.shape} must have "
            "the same shape (cells x genes)."
        )

    if mode not in ("uniform", "count_proportional"):
        raise ValueError(f"`mode` must be 'uniform' or 'count_proportional', got {mode!r}.")
    ncount = None
    if mode == "count_proportional":
        ncount = pd.Series(np.asarray(raw.sum(axis=1)).ravel(), index=cell_names)

    status = pd.Series(purification_status)
    status.index = status.index.map(str)
    donors = status.index[status.astype(object) == "purified"].tolist()

    operator = build_reassignment_operator(
        neighborhood, cells_with_residual=donors, ncount=ncount, **kwargs
    )

    # align the operator to the matrix rows (both axes are cells)
    positions = pd.Index(neighborhood.cell_id).get_indexer(cell_names)
    if (positions < 0).any():
        raise ValueError(
            f"{int((positions < 0).sum())} cell(s) of the count matrix are absent "
            "from the neighborhood."
        )
    operator = operator[positions][:, positions]

    residual = (raw - purified).tocsr()
    negative = int((residual.data < 0).sum())
    if negative:
        logger.warning(
            "%d negative residual value(s) were clipped to 0 before reassignment.", negative
        )
        residual.data[residual.data < 0] = 0.0
        residual.eliminate_zeros()

    # residual^T moved along the operator: recipients receive column-wise
    reassigned = operator.T @ residual
    corrected = (purified + reassigned).tocsr()

    logger.info(
        "reassigned %.3g of %.3g residual counts (%.1f%%) from %d donor cell(s).",
        float(reassigned.sum()),
        float(residual.sum()),
        100 * float(reassigned.sum()) / max(float(residual.sum()), 1e-12),
        len(donors),
    )
    if return_operator:
        return corrected, operator
    return corrected
