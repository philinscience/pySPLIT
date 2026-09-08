"""User-facing entry point: :func:`purify`."""

from __future__ import annotations

from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ._utils import get_logger
from .purify import purify_counts
from .rctd import RCTDResult, rctd_to_purify_input

logger = get_logger()

__all__ = ["purify"]


def purify(
    data,
    rctd: RCTDResult | None = None,
    weights: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
    primary_cell_type=None,
    layer: str | None = None,
    cell_names: Sequence[str] | None = None,
    gene_names: Sequence[str] | None = None,
    cells_to_purify: Sequence[str] | None = None,
    remove_residual_contamination: bool = False,
    belonging_threshold: float | None = None,
    require_sum_to_one: bool = True,
    chunk_size: int = 100_000,
    keep_raw_layer: bool = True,
    verbose: bool = True,
):
    """Purify spatial counts using deconvolution results.

    Two ways to call this:

    **From a deconvolution result** (RCTD or anything wrapped in
    :class:`~pysplit.RCTDResult`)::

        rctd = pysplit.run_post_process_rctd(rctd)
        purified = pysplit.purify(adata, rctd=rctd, remove_residual_contamination=True)

    **Annotation-method agnostic**, from the three pieces directly::

        purified = pysplit.purify(
            adata,
            weights=weights,             # cells x cell types
            reference=reference,         # cell types x genes
            primary_cell_type=labels,    # optional; defaults to arg-max of weights
        )

    Parameters
    ----------
    data
        An :class:`~anndata.AnnData` of raw counts (``cells x genes``), or a
        raw ``cells x genes`` matrix / DataFrame.
    rctd
        A post-processed deconvolution result. Mutually exclusive with
        ``weights``/``reference``.
    weights
        ``cells x cell_types`` DataFrame of deconvolution weights.
    reference
        Reference profiles as a DataFrame; either orientation is accepted.
    primary_cell_type
        Per-cell dominant cell type (Series or dict). Defaults to the arg-max
        of ``weights``, with a warning.
    layer
        Layer of ``data`` holding raw counts. ``None`` uses ``.X``.
    cell_names, gene_names
        Axis labels, required only when ``data`` is a bare matrix (an AnnData
        or DataFrame carries its own).
    cells_to_purify
        Purify only these cells; the rest are returned untouched. Prefer
        :func:`~pysplit.balance_by_score` for the spatially-aware workflow.
    remove_residual_contamination
        Also remove non-specific signal - genes with no specific reference
        support in the cell's primary cell type. Typically removes a further
        2-5% of counts and sharpens marker specificity. Default ``False``
        (backward compatible).
    belonging_threshold
        Specificity threshold for the above; ``None`` derives it from the
        reference (the 75th percentile of the per-gene min/max ratio).
    require_sum_to_one
        Rescale weight rows that do not sum to one.
    chunk_size
        Cells processed per block. Only affects peak memory.
    keep_raw_layer
        Store the aligned raw counts in ``layers['counts_raw']``.

    Returns
    -------
    anndata.AnnData or dict
        When ``data`` is an AnnData, a new AnnData whose ``X`` holds the
        purified counts, with per-cell metadata in ``obs`` and the run
        parameters in ``uns['pysplit']``. Otherwise the raw dict from
        :func:`~pysplit.purify_counts`.

    Notes
    -----
    Cells without a deconvolution result (including ``reject`` cells, which
    :func:`~pysplit.rctd_to_purify_input` drops) are absent from the output.
    Purified counts are fractional, not integers.
    """
    if rctd is not None:
        if weights is not None or reference is not None:
            raise ValueError(
                "Pass either `rctd` or `weights`/`reference`, not both."
            )
        rctd.check_post_processed()
        converted = rctd_to_purify_input(rctd)
        weights = converted["weights"]
        reference = converted["reference"]
        if primary_cell_type is None:
            primary_cell_type = converted["primary_cell_type"]
    else:
        missing = [
            name
            for name, value in (("weights", weights), ("reference", reference))
            if value is None
        ]
        if missing:
            raise ValueError(
                "Without `rctd`, both `weights` and `reference` are required "
                f"(missing: {missing})."
            )

    is_anndata = isinstance(data, ad.AnnData)
    if is_anndata:
        counts = data.X if layer is None else data.layers[layer]
        cell_names = list(map(str, data.obs_names))
        gene_names = list(map(str, data.var_names))
    else:
        counts = data

    result = purify_counts(
        counts,
        weights=weights,
        reference=reference,
        primary_cell_type=primary_cell_type,
        cell_names=cell_names,
        gene_names=gene_names,
        cells_to_purify=cells_to_purify,
        remove_residual_contamination=remove_residual_contamination,
        belonging_threshold=belonging_threshold,
        require_sum_to_one=require_sum_to_one,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    if verbose:
        raw_total = float(result["counts"].sum())
        kept = float(result["purified_counts"].sum())
        logger.info(
            "purified %d cells x %d genes; kept %.1f%% of counts.",
            result["params"]["n_cells"],
            result["params"]["n_genes"],
            100 * kept / max(raw_total, 1e-12),
        )

    if not is_anndata:
        return result

    obs = result["cell_meta"]
    if rctd is not None:
        extra = rctd.results_df.reindex(obs.index)
        extra = extra.drop(columns=[c for c in obs.columns if c in extra.columns])
        obs = pd.concat([obs, extra], axis=1)
    else:
        carry_over = data.obs.reindex(obs.index)
        carry_over = carry_over.drop(
            columns=[c for c in obs.columns if c in carry_over.columns]
        )
        obs = pd.concat([obs, carry_over], axis=1)

    out = ad.AnnData(
        X=result["purified_counts"],
        obs=obs,
        var=data.var.loc[result["gene_names"]].copy(),
    )
    if keep_raw_layer:
        out.layers["counts_raw"] = result["counts"]

    positions = data.obs_names.get_indexer(result["cell_names"])
    for key, value in data.obsm.items():
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[0] == data.n_obs:
            out.obsm[key] = array[positions]

    out.uns["pysplit"] = dict(result["params"])
    out.uns["pysplit"]["cell_types"] = list(result["reference"].index)
    return out
