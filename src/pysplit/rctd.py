"""Deconvolution results: container, post-processing and conversion.

SPLIT was originally built around RCTD doublet-mode output, but the
purification step itself only needs three things:

* a ``cells x cell_types`` weight matrix,
* a ``cell_types x genes`` reference,
* (optionally) a primary cell type per cell.

:class:`RCTDResult` holds a doublet-mode result and knows how to turn it into
those three pieces. Any other deconvolution tool can skip it entirely and call
:func:`pysplit.purify` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ._utils import (
    SPOT_CLASS_LEVELS,
    as_object_labels,
    entropy,
    get_logger,
    row_normalize,
)

logger = get_logger()

__all__ = [
    "RCTDResult",
    "from_rctd_py",
    "run_post_process_rctd",
    "rctd_to_purify_input",
    "read_rctdpy",
    "SPOT_CLASS_LEVELS",
]

#: Integer -> label mapping used by ``rctd-py`` for its ``spot_class`` column.
RCTDPY_SPOT_CLASS_MAP = {
    0: "reject",
    1: "singlet",
    2: "doublet_certain",
    3: "doublet_uncertain",
}


@dataclass
class RCTDResult:
    """A doublet-mode deconvolution result.

    Attributes
    ----------
    results_df
        Per-cell results, indexed by cell id. Must contain ``spot_class``,
        ``first_type`` and ``second_type``; post-processing adds
        ``weight_first_type``, ``weight_second_type`` and friends.
    weights
        ``cells x cell_types`` full decomposition weights.
    weights_doublet
        ``cells x 2`` doublet-mode weights with columns
        ``first_type``/``second_type``.
    reference
        ``cell_types x genes`` mean reference profiles.
    class_df
        Optional cell type -> higher-level class mapping (a Series indexed by
        cell type). Enables the ``*_class`` metrics.
    coords
        Optional ``x``/``y`` spatial coordinates per cell.
    singlet_scores
        Optional per-cell mapping of cell type -> singlet score. Only needed
        for the non-lite post-processing extras.
    post_processed
        Whether :func:`run_post_process_rctd` has been applied.
    """

    results_df: pd.DataFrame
    weights: pd.DataFrame
    weights_doublet: pd.DataFrame | None = None
    reference: pd.DataFrame | None = None
    class_df: pd.Series | None = None
    coords: pd.DataFrame | None = None
    singlet_scores: Sequence[Mapping[str, float]] | None = None
    post_processed: bool = False
    results_df_raw: pd.DataFrame | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.results_df = self.results_df.copy()
        self.results_df.index = self.results_df.index.map(str)
        self.weights = self.weights.copy()
        self.weights.index = self.weights.index.map(str)
        self.weights.columns = self.weights.columns.map(str)
        if self.weights_doublet is not None:
            self.weights_doublet = self.weights_doublet.copy()
            self.weights_doublet.index = self.weights_doublet.index.map(str)
        if self.class_df is not None and isinstance(self.class_df, pd.DataFrame):
            col = "class" if "class" in self.class_df.columns else self.class_df.columns[0]
            self.class_df = self.class_df[col]
        if self.class_df is not None:
            self.class_df = pd.Series(self.class_df)
            self.class_df.index = self.class_df.index.map(str)

    @property
    def cell_types(self) -> list[str]:
        return list(self.weights.columns)

    @property
    def cell_ids(self) -> list[str]:
        return list(self.results_df.index)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RCTDResult({len(self.cell_ids)} cells x {len(self.cell_types)} cell types, "
            f"post_processed={self.post_processed})"
        )

    # ------------------------------------------------------------------ #
    def to_purify_input(self) -> dict:
        """Shorthand for :func:`rctd_to_purify_input`."""
        return rctd_to_purify_input(self)

    def check_post_processed(self) -> None:
        """Raise if post-processing has not been run."""
        if not self.post_processed:
            raise ValueError(
                "This deconvolution result has not been post-processed by SPLIT.\n"
                "Run:  rctd = pysplit.run_post_process_rctd(rctd)"
            )


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
def _correct_singlets(rctd: RCTDResult, min_weight: float) -> pd.DataFrame:
    """Fix labels of cells with only one plausible cell type.

    RCTD reports a ``second_type`` even for cells where a single cell type
    carries essentially all the weight; in that case the secondary label is
    arbitrary. Such cells are relabelled as confident singlets with
    ``second_type = NA`` so that purification leaves them untouched.
    """
    df = rctd.results_df.copy()
    weights = rctd.weights.loc[df.index]
    weight_values = weights.to_numpy(dtype=float)

    n_candidates = (weight_values > min_weight).sum(axis=1)
    confident_singlet = n_candidates == 1
    no_cell_type = n_candidates == 0
    is_reject = (df["spot_class"].astype(object) == "reject").to_numpy()

    argmax_type = np.asarray(rctd.cell_types, dtype=object)[weight_values.argmax(axis=1)]

    first_type = as_object_labels(df["first_type"].to_numpy())
    second_type = as_object_labels(df["second_type"].to_numpy())
    first_type[confident_singlet] = argmax_type[confident_singlet]
    first_type[no_cell_type] = None
    second_type[confident_singlet] = None
    second_type[no_cell_type] = None

    spot_class = as_object_labels(df["spot_class"].to_numpy())
    spot_class[confident_singlet & ~is_reject] = "singlet"
    spot_class[no_cell_type] = "reject"

    df["first_type"] = first_type
    df["second_type"] = second_type
    df["spot_class"] = pd.Categorical(
        spot_class, categories=list(SPOT_CLASS_LEVELS), ordered=True
    )
    df["n_candidates"] = n_candidates

    if rctd.weights_doublet is not None:
        wd = rctd.weights_doublet.loc[df.index].to_numpy(dtype=float)
        df["max_doublet_weight"] = wd.max(axis=1)

    normalised = row_normalize(weight_values)
    df["rctd_weights_entropy"] = [entropy(row) for row in normalised]
    return df


def _update_scores(rctd: RCTDResult, df: pd.DataFrame, lite: bool) -> pd.DataFrame:
    """Attach doublet weights, class labels and (optionally) singlet scores."""
    if rctd.weights_doublet is None:
        raise ValueError(
            "`weights_doublet` is required for post-processing: a cells x 2 frame "
            "with the weights of the primary and secondary cell type."
        )
    wd = rctd.weights_doublet.loc[df.index]
    cols = list(wd.columns)
    first_col = "first_type" if "first_type" in cols else cols[0]
    second_col = "second_type" if "second_type" in cols else cols[1]
    df["weight_first_type"] = wd[first_col].to_numpy(dtype=float)
    df["weight_second_type"] = wd[second_col].to_numpy(dtype=float)

    # NOTE: the raw doublet weights are kept as reported by the deconvolution,
    # even for cells relabelled as confident singlets. Their secondary label is
    # dropped, so `rctd_to_purify_input` emits a single weight for them which
    # `purify` then renormalises to 1.

    if not lite and rctd.singlet_scores is not None:
        scores = list(rctd.singlet_scores)
        first = as_object_labels(df["first_type"].to_numpy())
        second = as_object_labels(df["second_type"].to_numpy())
        df["singlet_score_first"] = [
            scores[i].get(first[i], np.nan) if first[i] is not None else np.nan
            for i in range(len(df))
        ]
        df["singlet_score_second"] = [
            scores[i].get(second[i], np.nan) if second[i] is not None else np.nan
            for i in range(len(df))
        ]
        df["delta_singlet_score_first_second"] = (
            df["singlet_score_second"] - df["singlet_score_first"]
        )
        if "min_score" in df:
            df["score_diff"] = df["singlet_score_first"] - df["min_score"]
            if "singlet_score" in df:
                df["score_diff_old"] = df["singlet_score"] - df["min_score"]
        deltas = []
        for score_map in scores:
            vals = np.sort(np.asarray(list(score_map.values()), dtype=float))
            deltas.append(np.inf if vals.size < 2 else vals[1] - vals[0])
        df["delta_singlet_score"] = deltas

    if rctd.class_df is not None:
        class_map = rctd.class_df
        classes = list(pd.unique(class_map.to_numpy()))
        df["first_type_class"] = pd.Categorical(
            [class_map.get(t, None) for t in df["first_type"]], categories=classes
        )
        df["second_type_class"] = pd.Categorical(
            [class_map.get(t, None) for t in df["second_type"]], categories=classes
        )
        df["same_class"] = (
            df["first_type_class"].astype(object) == df["second_type_class"].astype(object)
        ).where(df["second_type_class"].notna())
    return df


def _alternative_annotations(rctd: RCTDResult, df: pd.DataFrame) -> pd.DataFrame:
    """Alternative label assignments, used to gauge annotation stability."""
    weights = rctd.weights.loc[df.index]
    weight_values = weights.to_numpy(dtype=float)
    df["annot_max_weight"] = np.asarray(rctd.cell_types, dtype=object)[
        weight_values.argmax(axis=1)
    ]

    if rctd.singlet_scores is not None:
        df["annot_min_singlet_score"] = [
            min(m, key=m.get) if len(m) else None for m in rctd.singlet_scores
        ]
    else:
        df["annot_min_singlet_score"] = None

    w1 = df["weight_first_type"].to_numpy(dtype=float)
    w2 = df["weight_second_type"].to_numpy(dtype=float)
    first = as_object_labels(df["first_type"].to_numpy())
    second = as_object_labels(df["second_type"].to_numpy())
    take_first = (w1 > w2) | np.isnan(w2)
    df["annot_max_doublet_weight"] = np.where(take_first, first, second)
    df["w1_larger_w2"] = first == df["annot_max_doublet_weight"].to_numpy()
    return df


def run_post_process_rctd(
    rctd: RCTDResult,
    min_weight: float = 0.05,
    lite: bool = True,
    verbose: bool = True,
) -> RCTDResult:
    """Make a doublet-mode deconvolution result SPLIT-compatible.

    This is the Python equivalent of ``SPLIT::run_post_process_RCTD()`` and
    must be run before purification. It

    1. relabels cells with a single plausible cell type as confident singlets
       (``second_type`` becomes missing, so they are not over-purified),
    2. marks cells with no cell type above ``min_weight`` as ``reject``,
    3. attaches the doublet weights, higher-level classes and alternative
       annotations used by the neighbourhood metrics.

    Parameters
    ----------
    rctd
        The result to post-process. It is not modified in place.
    min_weight
        A cell type counts as a candidate for a cell if its full-decomposition
        weight exceeds this. Default ``0.05``.
    lite
        Skip the singlet-score based extras, which need ``singlet_scores`` and
        are not used by purification. Default ``True``.

    Returns
    -------
    RCTDResult
        A new post-processed result.
    """
    if verbose:
        logger.info("correcting singlets ...")
    df = _correct_singlets(rctd, min_weight=min_weight)

    if verbose:
        logger.info("updating scores ...")
    df = _update_scores(rctd, df, lite=lite)

    if rctd.coords is not None:
        if verbose:
            logger.info("adding coordinates ...")
        coords = rctd.coords.reindex(df.index)
        df["x"] = coords.iloc[:, 0].to_numpy()
        df["y"] = coords.iloc[:, 1].to_numpy()

    if verbose:
        logger.info("computing alternative annotations ...")
    df = _alternative_annotations(rctd, df)

    out = RCTDResult(
        results_df=df,
        weights=rctd.weights,
        weights_doublet=rctd.weights_doublet,
        reference=rctd.reference,
        class_df=rctd.class_df,
        coords=rctd.coords,
        singlet_scores=rctd.singlet_scores,
        post_processed=True,
        results_df_raw=rctd.results_df,
    )
    return out


# --------------------------------------------------------------------------- #
# conversion to purification input
# --------------------------------------------------------------------------- #
def rctd_to_purify_input(rctd: RCTDResult) -> dict:
    """Extract weights, primary labels and reference from a deconvolution result.

    The weight matrix is assembled the way SPLIT expects it:

    * ``singlet`` and ``doublet_certain`` cells keep only their primary and
      secondary cell type (secondary dropped when missing),
    * ``doublet_uncertain`` cells keep their full decomposition, because their
      two-type assignment is not trustworthy,
    * ``reject`` cells are dropped.

    Returns
    -------
    dict
        ``primary_cell_type`` (Series), ``weights`` (``cells x cell_types``
        DataFrame) and ``reference`` (``cell_types x genes`` DataFrame).
    """
    if not rctd.post_processed:
        logger.warning(
            "This result has not been post-processed by SPLIT; calling "
            "run_post_process_rctd() automatically. Run it explicitly to "
            "silence this warning."
        )
        rctd = run_post_process_rctd(rctd)

    df = rctd.results_df
    required = [
        "first_type",
        "second_type",
        "weight_first_type",
        "weight_second_type",
        "spot_class",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from `results_df`: "
            f"{missing}. Make sure run_post_process_rctd() completed successfully."
        )
    if rctd.reference is None:
        raise ValueError("`reference` is required to build the purification input.")

    cell_types = rctd.cell_types
    spot_class = df["spot_class"].astype(object)

    doublet_uncertain = df.index[spot_class == "doublet_uncertain"]
    two_type = df.loc[spot_class.isin(["singlet", "doublet_certain"])]

    rows, cols, vals = [], [], []
    # primary + secondary for confidently assigned cells
    for type_col, weight_col in (
        ("first_type", "weight_first_type"),
        ("second_type", "weight_second_type"),
    ):
        labels = as_object_labels(two_type[type_col].to_numpy())
        weights = two_type[weight_col].to_numpy(dtype=float)
        keep = np.array([lab is not None for lab in labels]) & ~np.isnan(weights)
        rows.extend(np.asarray(two_type.index, dtype=object)[keep].tolist())
        cols.extend(labels[keep].tolist())
        vals.extend(weights[keep].tolist())

    order = list(dict.fromkeys(list(two_type.index) + list(doublet_uncertain)))
    weight_frame = pd.DataFrame(
        0.0, index=pd.Index(order, name="cell_id"), columns=cell_types
    )
    if rows:
        weight_frame.values[
            weight_frame.index.get_indexer(rows),
            weight_frame.columns.get_indexer(cols),
        ] = vals
    # full decomposition for uncertain doublets
    if len(doublet_uncertain):
        weight_frame.loc[doublet_uncertain, :] = (
            rctd.weights.loc[doublet_uncertain, cell_types].to_numpy(dtype=float)
        )

    primary = pd.Series(
        as_object_labels(df.loc[order, "first_type"].to_numpy()),
        index=pd.Index(order, name="cell_id"),
        name="primary_cell_type",
    )

    reference = rctd.reference
    if len(set(cell_types) & set(map(str, reference.columns))) > len(
        set(cell_types) & set(map(str, reference.index))
    ):
        reference = reference.T

    return {
        "primary_cell_type": primary,
        "weights": weight_frame,
        "reference": reference,
    }


# --------------------------------------------------------------------------- #
# rctd-py interoperability
# --------------------------------------------------------------------------- #
def read_rctdpy(
    save_dir: str | os.PathLike,
    class_df=None,
    min_weight: float = 0.05,
    spot_class_map: Mapping[int, str] | None = None,
    post_process: bool = True,
    coords: pd.DataFrame | None = None,
) -> RCTDResult:
    """Load a `rctd-py <https://github.com/p-gueguen/rctd-py>`_ output directory.

    Expects the layout ``rctd-py`` writes:

    ``cell_ids.parquet``, ``weights.parquet``, ``weights_doublet.parquet``,
    ``spot_results.parquet``, ``metadata.parquet`` and
    ``reference_profiles.h5`` (datasets ``profiles`` as cell types x genes,
    ``gene_names``, ``cell_type_names``).

    Parameters
    ----------
    save_dir
        Directory containing the files above.
    class_df
        Optional cell type -> class mapping. Strongly recommended: it makes
        the ``*_class`` neighbourhood metrics available.
    min_weight
        Passed to :func:`run_post_process_rctd`.
    spot_class_map
        Integer -> label mapping for ``spot_class``. Defaults to
        :data:`RCTDPY_SPOT_CLASS_MAP`.
    post_process
        Run :func:`run_post_process_rctd` on the loaded result (default).
    coords
        Optional ``x``/``y`` coordinates per cell, since ``rctd-py`` does not
        store them.

    Returns
    -------
    RCTDResult
    """
    save_dir = os.fspath(save_dir)
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"{save_dir} is not a directory.")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("read_rctdpy() requires h5py (pip install h5py).") from exc

    spot_class_map = dict(RCTDPY_SPOT_CLASS_MAP if spot_class_map is None else spot_class_map)

    def _read(name: str) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(save_dir, name))

    cell_ids = _read("cell_ids.parquet")["cell_id"].astype(str).tolist()
    weights_df = _read("weights.parquet")
    weights_doublet_df = _read("weights_doublet.parquet")
    spot_results = _read("spot_results.parquet")
    metadata = _read("metadata.parquet")
    cell_type_names = metadata["cell_type_names"].astype(str).tolist()

    mapped = spot_results["spot_class"].map(spot_class_map)
    if mapped.isna().any():
        bad = sorted(spot_results.loc[mapped.isna(), "spot_class"].unique().tolist())
        raise ValueError(
            f"Unmapped spot_class value(s): {bad}. Pass a matching `spot_class_map`."
        )

    results_df = pd.DataFrame(
        {
            "spot_class": pd.Categorical(
                mapped.to_numpy(), categories=list(SPOT_CLASS_LEVELS), ordered=True
            ),
            "first_type": spot_results["first_type_name"].astype(object).to_numpy(),
            "second_type": spot_results["second_type_name"].astype(object).to_numpy(),
            "min_score": spot_results.get("min_score"),
            "singlet_score": spot_results.get("singlet_score"),
        },
        index=pd.Index(cell_ids, name="cell_id"),
    )

    weights = pd.DataFrame(
        weights_df.to_numpy(dtype=float),
        index=pd.Index(cell_ids, name="cell_id"),
        columns=cell_type_names,
    )
    doublet_cols = (
        ["w_1", "w_2"]
        if {"w_1", "w_2"}.issubset(weights_doublet_df.columns)
        else list(weights_doublet_df.columns[:2])
    )
    weights_doublet = pd.DataFrame(
        weights_doublet_df[doublet_cols].to_numpy(dtype=float),
        index=pd.Index(cell_ids, name="cell_id"),
        columns=["first_type", "second_type"],
    )

    with h5py.File(os.path.join(save_dir, "reference_profiles.h5"), "r") as handle:
        profiles = np.asarray(handle["profiles"])  # cell types x genes
        ref_genes = [_decode(g) for g in handle["gene_names"][:]]
        ref_types = [_decode(t) for t in handle["cell_type_names"][:]]
    reference = pd.DataFrame(profiles, index=ref_types, columns=ref_genes)

    rctd = RCTDResult(
        results_df=results_df,
        weights=weights,
        weights_doublet=weights_doublet,
        reference=reference,
        class_df=class_df,
        coords=coords,
    )
    if post_process:
        rctd = run_post_process_rctd(rctd, min_weight=min_weight)
    return rctd


def from_rctd_py(
    result,
    reference=None,
    cell_ids: Sequence[str] | None = None,
    class_df=None,
    coords: pd.DataFrame | None = None,
    min_weight: float = 0.05,
    post_process: bool = True,
) -> RCTDResult:
    """Wrap an in-memory `rctd-py <https://github.com/p-gueguen/rctd-py>`_ result.

    The Python-native counterpart to :func:`read_rctdpy`: instead of writing
    parquet/HDF5 and reading it back, hand over the ``DoubletResult`` that
    ``rctd.run_rctd(..., mode="doublet")`` just returned.

    Parameters
    ----------
    result
        A ``rctd._types.DoubletResult``.
    reference
        Either a ``rctd.Reference``, or a ``cell_types x genes`` /
        ``genes x cell_types`` DataFrame of mean profiles. Required for
        purification; may be omitted if you only want the neighbourhood metrics.
    cell_ids
        Cell ids of the spatial object handed to RCTD, in its original order.
        ``result.pixel_mask`` is applied to them, since RCTD drops cells that
        fail its UMI filters.
    class_df
        Optional cell type -> class mapping (Series, or DataFrame with a
        ``class`` column). Enables the ``*_class`` neighbourhood metrics.
    coords
        Optional ``x``/``y`` coordinates per cell.
    min_weight, post_process
        Passed to / control :func:`run_post_process_rctd`.

    Returns
    -------
    RCTDResult

    Examples
    --------
    >>> import rctd, pysplit                                  # doctest: +SKIP
    >>> ref = rctd.Reference(chromium, cell_type_col="cell_type")
    >>> res = rctd.run_rctd(xenium, ref, mode="doublet")
    >>> rctd_result = pysplit.from_rctd_py(
    ...     res, reference=ref, cell_ids=list(xenium.obs_names),
    ...     coords=xenium.obs[["x", "y"]],
    ... )
    >>> purified = pysplit.purify(xenium, rctd=rctd_result)
    """
    cell_type_names = [_decode(t) for t in result.cell_type_names]

    weights = np.asarray(result.weights, dtype=float)
    n_rows = weights.shape[0]

    # RCTD silently drops cells failing its UMI filters; pixel_mask says which
    # of the input cells survived, so the ids must be subset the same way.
    if cell_ids is None:
        ids = [str(i) for i in range(n_rows)]
    else:
        ids = [str(c) for c in cell_ids]
        mask = getattr(result, "pixel_mask", None)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype == bool and mask.size == len(ids):
                ids = [c for c, keep in zip(ids, mask) if keep]
            elif mask.dtype != bool and mask.size == n_rows:
                ids = [ids[i] for i in mask.astype(int)]
        if len(ids) != n_rows:
            raise ValueError(
                f"Got {len(ids)} cell id(s) after applying pixel_mask but the "
                f"result has {n_rows} row(s). Pass `cell_ids` in the same order "
                "as the AnnData handed to rctd.run_rctd()."
            )

    def _labels(index_attr: str, name_attr: str) -> np.ndarray:
        """Cell type labels, from explicit names if given else via indices."""
        names = getattr(result, name_attr, None)
        if names is not None:
            values = np.asarray(names, dtype=object)
            if values.size == n_rows and not np.issubdtype(
                np.asarray(names).dtype, np.number
            ):
                return as_object_labels([_decode(v) for v in values])
        indices = np.asarray(getattr(result, index_attr))
        lookup = np.asarray(cell_type_names, dtype=object)
        out = np.full(n_rows, None, dtype=object)
        valid = (indices >= 0) & (indices < len(cell_type_names))
        out[valid] = lookup[indices[valid].astype(int)]
        return out

    spot_class = np.asarray(result.spot_class)
    if np.issubdtype(spot_class.dtype, np.number):
        # RCTDPY_SPOT_CLASS_MAP matches rctd-py's own SPOT_CLASS_NAMES ordering;
        # using it avoids importing rctd (and therefore torch) just for a lookup.
        spot_labels = np.asarray(
            [RCTDPY_SPOT_CLASS_MAP.get(int(v)) for v in spot_class], dtype=object
        )
    else:
        spot_labels = np.asarray([_decode(v) for v in spot_class], dtype=object)
    unknown = {v for v in spot_labels if v not in SPOT_CLASS_LEVELS}
    if unknown:
        raise ValueError(f"Unrecognised spot_class value(s): {sorted(unknown)}.")

    index = pd.Index(ids, name="cell_id")
    results_df = pd.DataFrame(
        {
            "spot_class": pd.Categorical(
                spot_labels, categories=list(SPOT_CLASS_LEVELS), ordered=True
            ),
            "first_type": _labels("first_type", "first_type_name"),
            "second_type": _labels("second_type", "second_type_name"),
        },
        index=index,
    )
    for attr in ("min_score", "singlet_score"):
        values = getattr(result, attr, None)
        if values is not None:
            results_df[attr] = np.asarray(values, dtype=float)

    doublet = np.asarray(result.weights_doublet, dtype=float)
    rctd_result = RCTDResult(
        results_df=results_df,
        weights=pd.DataFrame(weights, index=index, columns=cell_type_names),
        weights_doublet=pd.DataFrame(
            doublet[:, :2], index=index, columns=["first_type", "second_type"]
        ),
        reference=_reference_from_rctd_py(reference, cell_type_names),
        class_df=class_df,
        coords=None if coords is None else coords.reindex(index),
    )
    if post_process:
        rctd_result = run_post_process_rctd(rctd_result, min_weight=min_weight)
    return rctd_result


def _reference_from_rctd_py(reference, cell_type_names: Sequence[str]):
    """Coerce a rctd-py Reference (or a plain frame) to cell_types x genes."""
    if reference is None:
        return None
    if isinstance(reference, pd.DataFrame):
        frame = reference
    elif hasattr(reference, "profiles") and hasattr(reference, "gene_names"):
        # A rctd.Reference: `profiles` is an unlabelled genes x cell_types
        # array, so the axes have to be named from the object itself - the
        # orientation heuristic below has nothing to work with otherwise.
        frame = pd.DataFrame(
            np.asarray(reference.profiles, dtype=float),
            index=[_decode(g) for g in reference.gene_names],
            columns=[_decode(t) for t in reference.cell_type_names],
        ).T
    else:
        raise TypeError(
            "Could not read mean profiles from the given reference. Pass a "
            "cell_types x genes (or genes x cell_types) DataFrame instead."
        )
    frame = frame.copy()
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    wanted = set(map(str, cell_type_names))
    if len(wanted & set(frame.columns)) > len(wanted & set(frame.index)):
        frame = frame.T
    return frame


def _decode(value) -> str:
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
