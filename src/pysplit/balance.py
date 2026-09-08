"""Combining raw and purified data, and SPLIT-shift label swapping.

Purifying every cell is rarely what you want: cells whose secondary signal is
not actually present in their surroundings are more likely to hold a genuine
(possibly reference-absent) phenotype than contamination, and purifying them
erases it. The functions here decide, per cell, whether to take the raw or the
purified profile:

* :func:`balance_by_spot_class` - purify anything the deconvolution did not
  call a confident singlet;
* :func:`balance_by_score` - purify only cells whose local diffusion score
  exceeds a threshold (*spatially-aware SPLIT*), optionally swapping primary
  and secondary labels where the transcriptomic neighbourhood disagrees
  (*SPLIT-shift*);
* :func:`split_cells` / :func:`balance_split` - keep *both* halves of a cell as
  separate observations instead of discarding the residual.
"""

from __future__ import annotations

from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ._utils import as_csr, get_logger

logger = get_logger()

__all__ = [
    "balance_by_score",
    "balance_by_spot_class",
    "split_cells",
    "balance_split",
    "shift_labels",
]

_SWAP_BACKUP_SUFFIX = "_before_swap"


# --------------------------------------------------------------------------- #
def _require_columns(obs: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    missing = [c for c in columns if c not in obs.columns]
    if missing:
        raise ValueError(
            f"{missing} not found in {where}. Compute them first "
            "(see run_post_process_rctd / spatial_metrics / transcriptomics_metrics)."
        )


def _counts(adata, layer: str | None) -> sparse.csr_matrix:
    return as_csr(adata.X if layer is None else adata.layers[layer])


def _subset_matrix(adata, cells: Sequence[str], genes: Sequence[str], layer) -> sparse.csr_matrix:
    matrix = _counts(adata, layer)
    rows = adata.obs_names.get_indexer(list(cells))
    cols = adata.var_names.get_indexer(list(genes))
    return matrix[rows][:, cols]


def _clip_negative(matrix: sparse.csr_matrix, context: str) -> sparse.csr_matrix:
    """Zero out negative residuals, which floating-point noise can produce."""
    negative = matrix.data < 0
    n_negative = int(negative.sum())
    if n_negative:
        logger.warning(
            "%s: %d negative value(s) in the residual profile were set to 0 "
            "(most negative: %.3g).",
            context,
            n_negative,
            float(matrix.data.min()),
        )
        matrix = matrix.copy()
        matrix.data[matrix.data < 0] = 0.0
        matrix.eliminate_zeros()
    return matrix


# --------------------------------------------------------------------------- #
# SPLIT-shift
# --------------------------------------------------------------------------- #
def shift_labels(
    obs: pd.DataFrame,
    cells_to_swap: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Swap primary and secondary labels (and weights) for selected cells.

    When ``cells_to_swap`` is ``None`` the SPLIT-shift criterion is applied:
    swap a cell whose transcriptomic neighbourhood agrees with neither its
    primary cell type nor its primary class, while the neighbourhood's class
    *does* match the cell's secondary class.

    The pre-swap values are preserved in ``*_before_swap`` columns and a
    boolean ``swap`` column records what happened.

    Returns
    -------
    (obs, swap_mask)
        A copy of ``obs`` with the swap applied, and the boolean mask.
    """
    obs = obs.copy()
    if cells_to_swap is None:
        needed = [
            "first_type_neighborhood_agreement",
            "first_type_class_neighborhood_agreement",
            "second_type_class",
            "first_type_class_neighborhood",
        ]
        _require_columns(obs, needed, "obs")
        def _disagrees(column: str) -> np.ndarray:
            # only an explicit False counts; a missing agreement is no evidence
            values = obs[column].to_numpy(dtype=object)
            return np.array([v is False or v == False for v in values], dtype=bool)

        disagrees = _disagrees("first_type_neighborhood_agreement") & _disagrees(
            "first_type_class_neighborhood_agreement"
        )
        class_matches = obs["second_type_class"].astype(object) == obs[
            "first_type_class_neighborhood"
        ].astype(object)
        mask = (disagrees & class_matches & obs["second_type"].notna()).to_numpy()
    else:
        mask = obs.index.isin(list(cells_to_swap))

    pairs = [
        ("first_type", "second_type"),
        ("weight_first_type", "weight_second_type"),
        ("first_type_class", "second_type_class"),
    ]
    for left, right in pairs:
        if left not in obs.columns or right not in obs.columns:
            continue
        left_values = obs[left].astype(object).to_numpy(copy=True)
        right_values = obs[right].astype(object).to_numpy(copy=True)
        obs[left + _SWAP_BACKUP_SUFFIX] = left_values
        obs[right + _SWAP_BACKUP_SUFFIX] = right_values
        new_left = left_values.copy()
        new_right = right_values.copy()
        new_left[mask] = right_values[mask]
        new_right[mask] = left_values[mask]
        obs[left] = new_left
        obs[right] = new_right

    obs["swap"] = mask
    logger.info("SPLIT-shift: swapped labels for %d cell(s).", int(mask.sum()))
    return obs, mask


# --------------------------------------------------------------------------- #
# balancing
# --------------------------------------------------------------------------- #
def _balance(
    adata_raw,
    adata_purified,
    cells_purified: Sequence[str],
    cells_raw: Sequence[str],
    cells_removed: Sequence[str],
    layer_raw,
    layer_purified,
    do_swap: bool,
) -> ad.AnnData:
    """Assemble one AnnData from raw cells, purified cells and optional swaps."""
    genes = [g for g in adata_raw.var_names if g in set(adata_purified.var_names)]
    cells_purified = [c for c in cells_purified if c in set(adata_purified.obs_names)]
    cells_raw = list(cells_raw)

    order = cells_raw + cells_purified
    if not order:
        raise ValueError("Nothing left after balancing: no raw and no purified cells.")

    obs = adata_raw.obs.loc[order].copy()
    obs["purification_status"] = "raw"
    if cells_purified:
        status = adata_purified.obs.reindex(cells_purified).get("purification_status")
        obs.loc[cells_purified, "purification_status"] = (
            "purified" if status is None else status.fillna("purified").to_numpy()
        )
    if len(cells_removed):
        logger.info("%d rejected cell(s) were removed.", len(cells_removed))

    if do_swap:
        obs, swap_mask = shift_labels(obs)
    else:
        obs["swap"] = False
        swap_mask = np.zeros(len(order), dtype=bool)

    raw_block = _subset_matrix(adata_raw, cells_raw, genes, layer_raw)
    purified_block = _subset_matrix(adata_purified, cells_purified, genes, layer_purified)

    # A swapped cell's real profile is the *residual* the purification removed,
    # not the profile it kept - so those rows are replaced.
    swapped = [c for c in np.asarray(order, dtype=object)[swap_mask] if c in set(cells_purified)]
    if swapped:
        positions = pd.Index(cells_purified).get_indexer(swapped)
        raw_swapped = _subset_matrix(adata_raw, swapped, genes, layer_raw)
        pure_swapped = _subset_matrix(adata_purified, swapped, genes, layer_purified)
        residual = _clip_negative((raw_swapped - pure_swapped).tocsr(), "SPLIT-shift")
        keep = np.ones(len(cells_purified))
        keep[positions] = 0.0
        scatter = sparse.csr_matrix(
            (np.ones(len(swapped)), (positions, np.arange(len(swapped)))),
            shape=(len(cells_purified), len(swapped)),
        )
        purified_block = sparse.diags(keep) @ purified_block + scatter @ residual

    matrix = sparse.vstack([raw_block, purified_block], format="csr")

    out = ad.AnnData(X=matrix, obs=obs, var=adata_raw.var.loc[genes].copy())
    positions = adata_raw.obs_names.get_indexer(order)
    for key, value in adata_raw.obsm.items():
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[0] == adata_raw.n_obs:
            out.obsm[key] = array[positions]
    return out


def balance_by_score(
    adata_raw,
    adata_purified,
    threshold: float = 0.05,
    score_name: str = "neighborhood_weights_second_type",
    spot_class_key: str = "spot_class",
    swap_labels: bool = False,
    layer_raw: str | None = None,
    layer_purified: str | None = None,
) -> ad.AnnData:
    """Spatially-aware SPLIT: purify only cells with local secondary signal.

    A cell takes its purified profile when its diffusion score exceeds
    ``threshold`` (or when the deconvolution called it a ``doublet_uncertain``,
    whose two-type assignment cannot be trusted); otherwise it keeps its raw
    profile. ``reject`` cells are dropped.

    Parameters
    ----------
    adata_raw
        Raw counts with the deconvolution results and the spatial
        neighbourhood metrics in ``obs``.
    adata_purified
        Output of :func:`~pysplit.purify` on the same cells.
    threshold
        Lower values purify more cells. ``0.05`` is a reasonable default for
        Xenium.
    score_name
        Column of ``adata_raw.obs`` to threshold. Other sensible choices are
        ``second_type_neighbors_N`` and ``second_type_neighbors_no_reject_N``.
    spot_class_key
        Column holding the deconvolution spot class.
    swap_labels
        Enable **SPLIT-shift** (see :func:`shift_labels`).

    Returns
    -------
    anndata.AnnData
        Balanced counts with ``obs['purification_status']`` and ``obs['swap']``.
    """
    _require_columns(adata_raw.obs, [score_name, spot_class_key, "first_type"], "adata_raw.obs")
    shared = [c for c in adata_raw.obs_names if c in set(adata_purified.obs_names)]
    obs = adata_raw.obs.loc[shared]

    spot_class = obs[spot_class_key].astype(object)
    score = pd.to_numeric(obs[score_name], errors="coerce")

    removed = obs.index[spot_class == "reject"]
    purify_mask = ((spot_class != "reject") & (score > threshold)) | (
        spot_class == "doublet_uncertain"
    )
    cells_purified = obs.index[purify_mask].difference(removed, sort=False)
    cells_raw = obs.index.difference(cells_purified.union(removed, sort=False), sort=False)

    logger.info(
        "balance_by_score: %d purified, %d raw, %d removed (threshold %s > %g).",
        len(cells_purified),
        len(cells_raw),
        len(removed),
        score_name,
        threshold,
    )
    return _balance(
        adata_raw,
        adata_purified,
        cells_purified=list(cells_purified),
        cells_raw=list(cells_raw),
        cells_removed=list(removed),
        layer_raw=layer_raw,
        layer_purified=layer_purified,
        do_swap=swap_labels,
    )


def balance_by_spot_class(
    adata_raw,
    adata_purified,
    spot_class_key: str = "spot_class",
    swap_labels: bool = False,
    layer_raw: str | None = None,
    layer_purified: str | None = None,
) -> ad.AnnData:
    """Purify everything the deconvolution did not call a confident singlet.

    Confident singlets keep their raw profile, ``reject`` cells are dropped and
    everything else takes the purified profile.
    """
    _require_columns(adata_raw.obs, [spot_class_key, "first_type"], "adata_raw.obs")
    shared = [c for c in adata_raw.obs_names if c in set(adata_purified.obs_names)]
    obs = adata_raw.obs.loc[shared]
    spot_class = obs[spot_class_key].astype(object)

    removed = obs.index[spot_class == "reject"]
    cells_raw = obs.index[spot_class == "singlet"]
    cells_purified = obs.index.difference(cells_raw.union(removed, sort=False), sort=False)

    logger.info(
        "balance_by_spot_class: %d purified, %d raw, %d removed.",
        len(cells_purified),
        len(cells_raw),
        len(removed),
    )
    return _balance(
        adata_raw,
        adata_purified,
        cells_purified=list(cells_purified),
        cells_raw=list(cells_raw),
        cells_removed=list(removed),
        layer_raw=layer_raw,
        layer_purified=layer_purified,
        do_swap=swap_labels,
    )


# --------------------------------------------------------------------------- #
# splitting cells into two observations
# --------------------------------------------------------------------------- #
def split_cells(
    adata_raw,
    adata_purified,
    layer_raw: str | None = None,
    layer_purified: str | None = None,
    suffixes: tuple[str, str] = ("_1", "_2"),
) -> ad.AnnData:
    """Keep both halves of every purified cell as separate observations.

    Instead of throwing the removed signal away, each cell becomes two
    observations: ``<cell>_1`` carrying the purified profile labelled with the
    primary cell type, and ``<cell>_2`` carrying the residual labelled with the
    secondary cell type. Useful when the contamination itself is of interest.

    Returns
    -------
    anndata.AnnData
        Twice as many observations, with ``obs['decomposition_order']`` set to
        ``'first'``/``'second'`` and ``obs['cell_type']`` to the corresponding
        label.
    """
    genes = [g for g in adata_purified.var_names if g in set(adata_raw.var_names)]
    cells = [c for c in adata_purified.obs_names if c in set(adata_raw.obs_names)]

    purified = _subset_matrix(adata_purified, cells, genes, layer_purified)
    raw = _subset_matrix(adata_raw, cells, genes, layer_raw)
    residual = _clip_negative((raw - purified).tocsr(), "split_cells")

    obs = adata_purified.obs.loc[cells].copy()
    obs["cell_id"] = list(cells)

    first = obs.copy()
    first["cell_type"] = first.get("first_type", pd.Series(index=first.index, dtype=object))
    first["decomposition_order"] = "first"
    first.index = pd.Index([f"{c}{suffixes[0]}" for c in cells])

    second = obs.copy()
    second["cell_type"] = second.get(
        "second_type", pd.Series(index=second.index, dtype=object)
    )
    second["decomposition_order"] = "second"
    if "purification_status" in second.columns:
        status = second["purification_status"].astype(object)
        second["purification_status"] = np.where(status == "raw", "null", status)
    second.index = pd.Index([f"{c}{suffixes[1]}" for c in cells])

    return ad.AnnData(
        X=sparse.vstack([purified, residual], format="csr"),
        obs=pd.concat([first, second]),
        var=adata_purified.var.loc[genes].copy(),
    )


def balance_split(
    adata_raw,
    adata_purified,
    spot_class_key: str = "spot_class",
    purify_singlets: bool = True,
    split_singlets: bool = True,
    split_doublets_uncertain: bool = False,
    layer_raw: str | None = None,
    layer_purified: str | None = None,
) -> ad.AnnData:
    """Balanced dataset that keeps raw cells whole and split cells in two halves.

    Cells that should stay raw are carried over unchanged; the rest are split
    into a purified half and a residual half (see :func:`split_cells`).

    Parameters
    ----------
    purify_singlets
        If ``True`` (default), a cell is kept raw only when it has no secondary
        cell type at all; otherwise every ``singlet`` is kept raw.
    split_singlets
        Keep the residual half of purified singlets.
    split_doublets_uncertain
        Keep the residual half of ``doublet_uncertain`` cells. ``False`` by
        default, since their secondary label is unreliable.
    """
    _require_columns(adata_raw.obs, [spot_class_key, "first_type"], "adata_raw.obs")
    obs_raw = adata_raw.obs
    spot_class = obs_raw[spot_class_key].astype(object)

    removed = obs_raw.index[spot_class == "reject"]
    if purify_singlets:
        keep_raw = obs_raw.index[obs_raw["second_type"].isna()]
    else:
        keep_raw = obs_raw.index[spot_class == "singlet"]
    keep_raw = keep_raw.difference(removed, sort=False)
    to_split = obs_raw.index.difference(keep_raw.union(removed, sort=False), sort=False)
    to_split = [c for c in to_split if c in set(adata_purified.obs_names)]

    genes = [g for g in adata_raw.var_names if g in set(adata_purified.var_names)]

    raw_part = ad.AnnData(
        X=_subset_matrix(adata_raw, keep_raw, genes, layer_raw),
        obs=obs_raw.loc[keep_raw].copy(),
        var=adata_raw.var.loc[genes].copy(),
    )
    raw_part.obs["decomposition_order"] = "raw"
    raw_part.obs["purification_status"] = "raw"
    raw_part.obs["cell_id"] = list(keep_raw)
    raw_part.obs["cell_type"] = raw_part.obs["first_type"]

    split_part = split_cells(
        adata_raw[:, genes],
        adata_purified[to_split, genes],
        layer_raw=layer_raw,
        layer_purified=layer_purified,
    )

    # drop residual halves we were asked not to keep
    drop = pd.Series(False, index=split_part.obs_names)
    is_second = split_part.obs["decomposition_order"] == "second"
    second_spot_class = (
        adata_raw.obs.loc[split_part.obs["cell_id"], spot_class_key].astype(object).to_numpy()
    )
    if not split_singlets:
        if purify_singlets:
            drop |= is_second & (second_spot_class == "singlet")
        else:
            logger.warning("Cannot split singlets: with purify_singlets=False they stay raw.")
    if not split_doublets_uncertain:
        drop |= is_second & (second_spot_class == "doublet_uncertain")
    split_part = split_part[~drop.to_numpy()].copy()

    common = [c for c in raw_part.obs.columns if c in split_part.obs.columns]
    combined = ad.AnnData(
        X=sparse.vstack([as_csr(raw_part.X), as_csr(split_part.X)], format="csr"),
        obs=pd.concat([raw_part.obs[common], split_part.obs[common]]),
        var=raw_part.var.copy(),
    )
    logger.info(
        "balance_split: %d raw + %d split observations (%d cells removed).",
        raw_part.n_obs,
        split_part.n_obs,
        len(removed),
    )
    return combined
