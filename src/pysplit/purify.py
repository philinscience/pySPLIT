"""Core SPLIT purification.

The algorithm rescales every observed count by the fraction of the expected
expression at that gene that is attributable to the cell's *primary* cell type:

.. math::

    \\text{purified}_{c,g}
      = \\text{counts}_{c,g}
        \\cdot \\frac{w_{c,t_1(c)} \\, R_{t_1(c),g}}
                     {\\sum_{t} w_{c,t} \\, R_{t,g}}

where ``w`` are per-cell deconvolution weights and ``R`` a reference profile
matrix (cell types x genes). Genes with no reference support in the primary
cell type get a zero numerator and are therefore removed entirely.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ._utils import (
    as_csr,
    as_object_labels,
    chunk_slices,
    get_logger,
    index_of,
    to_dense_array,
)

logger = get_logger()

__all__ = ["purify", "purify_counts", "clean_reference", "auto_belonging_threshold"]


# --------------------------------------------------------------------------- #
# input coercion
# --------------------------------------------------------------------------- #
def _weights_frame(weights) -> pd.DataFrame:
    """Coerce deconvolution weights to a dense ``cells x cell_types`` frame."""
    if isinstance(weights, pd.DataFrame):
        return weights.astype(float)
    raise TypeError(
        "`weights` must be a pandas DataFrame (index = cell ids, "
        "columns = cell types). Got %r." % type(weights).__name__
    )


def _reference_frame(reference) -> pd.DataFrame:
    """Coerce the reference to a ``cell_types x genes`` frame."""
    if not isinstance(reference, pd.DataFrame):
        raise TypeError(
            "`reference` must be a pandas DataFrame with cell types and genes "
            "on its axes. Got %r." % type(reference).__name__
        )
    return reference.astype(float)


def _orient_reference(reference: pd.DataFrame, cell_types: Sequence[str]) -> pd.DataFrame:
    """Return the reference as ``cell_types x genes``, transposing if needed.

    Both orientations are accepted because the R package is inconsistent about
    it: whichever axis matches the deconvolution cell types becomes the rows.
    """
    cell_types = list(cell_types)
    rows_match = len(set(cell_types) & set(map(str, reference.index)))
    cols_match = len(set(cell_types) & set(map(str, reference.columns)))
    if cols_match > rows_match:
        return reference.T
    return reference


# --------------------------------------------------------------------------- #
# residual contamination removal
# --------------------------------------------------------------------------- #
def auto_belonging_threshold(reference: pd.DataFrame, quantile: float = 0.75) -> float:
    """Derive a specificity threshold from the reference itself.

    For every gene the ratio of its smallest non-zero to its largest reference
    value is computed; the ``quantile``-th percentile of those ratios is
    returned. Genes that sit below ``threshold * max`` in a cell type are
    considered not to belong to it.

    Parameters
    ----------
    reference
        ``cell_types x genes`` reference profiles.
    quantile
        Quantile of the min/max ratio distribution to use. Default ``0.75``.
    """
    ref = reference.to_numpy(dtype=float)
    gene_max = ref.max(axis=0)
    ratios = np.zeros(ref.shape[1], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(ref.shape[1]):
            col = ref[:, j]
            mx = gene_max[j]
            if mx <= 0:
                continue
            pos = col[col > 0]
            if pos.size:
                ratios[j] = pos.min() / mx
    ratios = ratios[ratios > 0]
    if ratios.size == 0:
        return 0.0
    return float(np.quantile(ratios, quantile))


def clean_reference(
    reference: pd.DataFrame,
    belonging_threshold: float | None = None,
    quantile: float = 0.75,
    verbose: bool = True,
) -> tuple[pd.DataFrame, float, pd.Series]:
    """Zero out reference entries that are not specific to a cell type.

    Returns the cleaned reference, the threshold used and the fraction of each
    cell type's profile that was removed.
    """
    ref = _reference_frame(reference)
    if belonging_threshold is None:
        belonging_threshold = auto_belonging_threshold(ref, quantile=quantile)
        if verbose:
            logger.info(
                "auto belonging_threshold: %.4f (%.0fth pct of min/max ratio)",
                belonging_threshold,
                quantile * 100,
            )
    values = ref.to_numpy(dtype=float, copy=True)
    thresholds = belonging_threshold * values.max(axis=0)
    original_sums = values.sum(axis=1)
    values[values < thresholds[None, :]] = 0.0
    cleaned = pd.DataFrame(values, index=ref.index, columns=ref.columns)

    with np.errstate(invalid="ignore", divide="ignore"):
        removed = 1.0 - np.where(original_sums > 0, values.sum(axis=1) / original_sums, 0.0)
    removed = pd.Series(removed, index=ref.index, name="fraction_profile_removed")
    if verbose:
        logger.info(
            "fraction of reference profile removed per cell type:\n%s",
            removed.round(3).to_string(),
        )
    return cleaned, float(belonging_threshold), removed


# --------------------------------------------------------------------------- #
# low-level purification
# --------------------------------------------------------------------------- #
def purify_counts(
    counts,
    weights: pd.DataFrame,
    reference: pd.DataFrame,
    primary_cell_type=None,
    cell_names: Sequence[str] | None = None,
    gene_names: Sequence[str] | None = None,
    cells_to_purify: Sequence[str] | None = None,
    remove_residual_contamination: bool = False,
    belonging_threshold: float | None = None,
    require_sum_to_one: bool = True,
    chunk_size: int = 100_000,
    verbose: bool = True,
) -> dict:
    """Purify a ``cells x genes`` count matrix.

    Parameters
    ----------
    counts
        ``cells x genes`` matrix of raw counts (sparse or dense).
    weights
        ``cells x cell_types`` DataFrame of deconvolution weights.
    reference
        Reference profiles; either ``cell_types x genes`` or ``genes x
        cell_types`` (the orientation is detected from the cell-type labels).
    primary_cell_type
        Per-cell dominant cell type. Falls back to the arg-max of ``weights``.
    cell_names, gene_names
        Axis labels of ``counts``. Required unless ``counts`` is a DataFrame.
    cells_to_purify
        Restrict purification to these cells; all others are returned raw.
    remove_residual_contamination
        Zero out genes with no specific reference support in the primary cell
        type (see :func:`clean_reference`).
    belonging_threshold
        Specificity threshold for the above. ``None`` derives it from the
        reference.
    require_sum_to_one
        Rescale weight rows that do not sum to one.
    chunk_size
        Number of cells processed at a time. Only affects peak memory.

    Returns
    -------
    dict
        ``purified_counts`` (``cells x genes`` CSR), ``counts`` (the aligned
        raw counts), ``cell_meta`` (per-cell DataFrame), ``reference``
        (possibly cleaned) and ``params``.
    """
    # --- axis labels --------------------------------------------------------
    if isinstance(counts, pd.DataFrame):
        cell_names = list(counts.index) if cell_names is None else list(cell_names)
        gene_names = list(counts.columns) if gene_names is None else list(gene_names)
        counts = counts.to_numpy()
    if cell_names is None or gene_names is None:
        raise ValueError(
            "`cell_names` and `gene_names` are required when `counts` is not a DataFrame."
        )
    cell_names = list(map(str, cell_names))
    gene_names = list(map(str, gene_names))
    counts = as_csr(counts)
    if counts.shape != (len(cell_names), len(gene_names)):
        raise ValueError(
            f"counts has shape {counts.shape} but {len(cell_names)} cell names and "
            f"{len(gene_names)} gene names were given (expected cells x genes)."
        )

    weights = _weights_frame(weights)
    weights.index = weights.index.map(str)
    reference = _orient_reference(_reference_frame(reference), weights.columns)
    reference.index = reference.index.map(str)
    reference.columns = reference.columns.map(str)

    # --- shared cells / genes ----------------------------------------------
    counts_pos = pd.Series(np.arange(len(cell_names)), index=cell_names)
    shared_cells = [c for c in cell_names if c in set(weights.index)]
    if len(shared_cells) < len(cell_names):
        logger.warning(
            "%d cell(s) have no deconvolution result and are dropped from the output.",
            len(cell_names) - len(shared_cells),
        )
    if not shared_cells:
        raise ValueError("No cells shared between `counts` and `weights`.")

    ref_genes = set(reference.columns)
    shared_genes = [g for g in gene_names if g in ref_genes]
    if len(shared_genes) < len(gene_names):
        logger.warning(
            "%d gene(s) are absent from the reference and are dropped from the output.",
            len(gene_names) - len(shared_genes),
        )
    if not shared_genes:
        raise ValueError("No genes shared between `counts` and `reference`.")

    counts = counts[counts_pos.loc[shared_cells].to_numpy(), :]
    gene_pos = pd.Series(np.arange(len(gene_names)), index=gene_names)
    counts = counts[:, gene_pos.loc[shared_genes].to_numpy()]
    weights = weights.loc[shared_cells]

    # --- cell types ---------------------------------------------------------
    cell_types = sorted(map(str, weights.columns))
    if sorted(reference.index.tolist()) != cell_types:
        only_w = sorted(set(cell_types) - set(reference.index))
        only_r = sorted(set(reference.index) - set(cell_types))
        raise ValueError(
            "Reference and deconvolution cell types do not match.\n"
            f"  only in weights:   {only_w}\n"
            f"  only in reference: {only_r}"
        )
    weights = weights[cell_types]
    reference = reference.loc[cell_types, shared_genes]

    # --- primary cell type --------------------------------------------------
    if primary_cell_type is None:
        logger.warning(
            "`primary_cell_type` not provided; using the arg-max of the "
            "deconvolution weights."
        )
        primary = pd.Series(
            np.asarray(cell_types, dtype=object)[weights.to_numpy().argmax(axis=1)],
            index=weights.index,
        )
    else:
        primary = pd.Series(primary_cell_type)
        primary.index = primary.index.map(str)
        missing = [c for c in shared_cells if c not in primary.index]
        if missing:
            raise ValueError(
                f"`primary_cell_type` is missing {len(missing)} of the shared cells "
                f"(e.g. {missing[:3]})."
            )
        primary = primary.loc[shared_cells]
    primary_labels = as_object_labels(primary.to_numpy())

    # --- residual contamination removal ------------------------------------
    profile_removed = None
    if remove_residual_contamination:
        if verbose:
            logger.info("cleaning reference profiles ...")
        reference, belonging_threshold, profile_removed = clean_reference(
            reference, belonging_threshold=belonging_threshold, verbose=verbose
        )

    # --- weights ------------------------------------------------------------
    weight_values = weights.to_numpy(dtype=float, copy=True)
    row_sums = weight_values.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-8):
        logger.warning("Some deconvolution weights do not sum to 1.")
        if require_sum_to_one:
            if verbose:
                logger.info("rescaling deconvolution weights per cell.")
            nz = row_sums > 0
            weight_values[nz] = weight_values[nz] / row_sums[nz, None]

    # --- optionally keep a subset of cells raw ------------------------------
    if cells_to_purify is not None:
        keep_raw = sorted(set(shared_cells) - set(map(str, cells_to_purify)))
        if keep_raw:
            if verbose:
                logger.info(
                    "collapsing weights onto the primary cell type for %d "
                    "non-purified cell(s).",
                    len(keep_raw),
                )
            pos = pd.Series(np.arange(len(shared_cells)), index=shared_cells)
            raw_rows = pos.loc[keep_raw].to_numpy()
            raw_types = index_of(primary_labels[raw_rows], cell_types)
            invalid = raw_types < 0
            if invalid.any():
                logger.warning(
                    "%d non-purified cell(s) have a primary cell type absent from the "
                    "deconvolution matrix and were left unchanged.",
                    int(invalid.sum()),
                )
            valid_rows = raw_rows[~invalid]
            weight_values[valid_rows, :] = 0.0
            weight_values[valid_rows, raw_types[~invalid]] = 1.0

    primary_idx = index_of(primary_labels, cell_types)
    unknown_primary = primary_idx < 0
    if unknown_primary.any():
        logger.warning(
            "%d cell(s) have a missing or unknown primary cell type; their counts "
            "are set to zero.",
            int(unknown_primary.sum()),
        )
    w1 = np.zeros(len(shared_cells), dtype=float)
    known = ~unknown_primary
    w1[known] = weight_values[np.flatnonzero(known), primary_idx[known]]

    # --- purification -------------------------------------------------------
    ref_values = reference.to_numpy(dtype=float)
    n_cells, n_genes = counts.shape

    slices = chunk_slices(n_cells, chunk_size)
    blocks = []
    for block, sl in enumerate(slices, start=1):
        if verbose and len(slices) > 1:
            logger.info("processing block %d/%d", block, len(slices))
        block_counts = counts[sl]
        coo = block_counts.tocoo()
        rows = coo.row  # block-local cell index
        cols = coo.col  # gene index
        offset = sl.start

        # Denominator sum_t w[cell, t] * ref[t, gene], evaluated only where a
        # count was actually observed - everywhere else the purified value is
        # zero regardless, so the full dense cells x genes product is never
        # materialised.
        denom = np.zeros(rows.shape[0], dtype=float)
        block_weights = weight_values[sl]
        for t in range(len(cell_types)):
            wt = block_weights[:, t]
            if not wt.any():
                continue
            denom += wt[rows] * ref_values[t, cols]

        p_idx = primary_idx[offset + rows]
        numer = np.where(p_idx >= 0, w1[offset + rows] * ref_values[p_idx, cols], 0.0)

        denom[denom == 0] = 1.0  # the numerator is zero there, so avoid 0/0
        values = coo.data * (numer / denom)

        keep = values != 0
        blocks.append(
            sparse.csr_matrix(
                (values[keep], (rows[keep], cols[keep])),
                shape=block_counts.shape,
            )
        )
        del coo, denom, numer, values

    purified = blocks[0] if len(blocks) == 1 else sparse.vstack(blocks, format="csr")
    purified.eliminate_zeros()

    # --- per-cell metadata --------------------------------------------------
    n_cell_types = (weight_values > 0).sum(axis=1)
    cell_meta = pd.DataFrame(
        {
            "cell_id": shared_cells,
            "primary_cell_type": primary_labels,
            "first_type": primary_labels,
            "w1_primary": w1,
            "n_cell_types": n_cell_types,
            "purification_status": np.where(n_cell_types > 1, "purified", "raw"),
            "nCount_raw": np.asarray(counts.sum(axis=1)).ravel(),
            "nCount_purified": np.asarray(purified.sum(axis=1)).ravel(),
        },
        index=pd.Index(shared_cells, name="cell_id"),
    )

    return {
        "purified_counts": purified,
        "counts": counts,
        "cell_meta": cell_meta,
        "cell_names": shared_cells,
        "gene_names": shared_genes,
        "reference": reference,
        "weights": pd.DataFrame(weight_values, index=shared_cells, columns=cell_types),
        "params": {
            "remove_residual_contamination": remove_residual_contamination,
            "belonging_threshold": belonging_threshold,
            "require_sum_to_one": require_sum_to_one,
            "chunk_size": chunk_size,
            "n_cells": len(shared_cells),
            "n_genes": len(shared_genes),
            "fraction_profile_removed": (
                None if profile_removed is None else profile_removed.to_dict()
            ),
        },
    }
