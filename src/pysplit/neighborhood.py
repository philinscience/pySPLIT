"""Neighbourhood metrics on top of a kNN graph.

Everything here consumes a :class:`~pysplit.neighbors.Neighborhood` and adds

* **per-neighbour fields** - ``cells x k`` matrices holding, say, the
  ``first_type`` of every neighbour, and
* **per-cell metrics** - 1-D arrays that end up in ``adata.obs``.

The two headline metrics are ``neighborhood_weights_second_type`` (how much of
a cell's secondary cell type is present in its *spatial* surroundings - the
local diffusion score that drives spatially-aware SPLIT) and
``first_type_neighborhood`` / ``*_agreement`` (which phenotype a cell's
*transcriptomic* neighbourhood agrees on - the basis of SPLIT-shift).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ._utils import get_logger, most_frequent, normalized_certainty
from .neighbors import Neighborhood
from .rctd import RCTDResult

logger = get_logger()

__all__ = [
    "SPATIAL_COLUMNS",
    "TRANSCRIPTOMIC_COLUMNS",
    "add_deconvolution_to_neighborhood",
    "add_infiltration_metrics",
    "add_neighborhood_weight_composition",
    "add_cell_type_neighborhood_weights",
    "add_spilling_type_weights",
    "add_spilling_type_ncount",
    "add_annotation_from_neighbors",
    "add_neighborhood_annotation_certainty",
    "spatial_metrics",
    "transcriptomics_metrics",
    "neighborhood_to_dataframe",
    "compute_swapping_score",
]

#: Columns :func:`spatial_metrics` needs projected onto the neighbourhood.
SPATIAL_COLUMNS = (
    "spot_class",
    "first_type",
    "second_type",
    "weight_first_type",
    "weight_second_type",
    "first_type_class",
    "second_type_class",
)

#: Columns :func:`transcriptomics_metrics` needs projected onto the neighbourhood.
TRANSCRIPTOMIC_COLUMNS = (
    "first_type",
    "second_type",
    "first_type_class",
    "second_type_class",
)

_LABEL_COLUMNS = (
    "spot_class",
    "first_type",
    "second_type",
    "first_type_class",
    "second_type_class",
    "annot_max_weight",
    "annot_min_singlet_score",
    "annot_max_doublet_weight",
)


# --------------------------------------------------------------------------- #
# gathering per-neighbour values
# --------------------------------------------------------------------------- #
def _gather_labels(values, nn_idx: np.ndarray) -> np.ndarray:
    """``cells x k`` object array of ``values`` at each neighbour; ``None`` if pruned."""
    series = pd.Series(values)
    # Normalise missing values once on the 1-D input rather than on the much
    # larger gathered matrix - at k=100 that is the difference between one
    # pass over n cells and one over n*k entries.
    flat = series.astype(object).to_numpy(copy=True)
    flat[pd.isna(series).to_numpy()] = None
    out = np.full(nn_idx.shape, None, dtype=object)
    valid = nn_idx >= 0
    out[valid] = flat[nn_idx[valid]]
    return out


def _gather_numeric(values, nn_idx: np.ndarray) -> np.ndarray:
    """``cells x k`` float array of ``values`` at each neighbour; ``nan`` if pruned."""
    values = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    out = np.full(nn_idx.shape, np.nan, dtype=float)
    valid = nn_idx >= 0
    out[valid] = values[nn_idx[valid]]
    return out


def add_deconvolution_to_neighborhood(
    neighborhood: Neighborhood,
    rctd: RCTDResult,
    columns: Sequence[str] | None = None,
) -> Neighborhood:
    """Project per-cell deconvolution results onto every neighbour slot.

    For each requested column of ``rctd.results_df`` a ``cells x k`` matrix is
    added to ``neighborhood.fields``, so that ``fields['first_type'][i, j]`` is
    the primary cell type of the ``j``-th neighbour of cell ``i``.

    Parameters
    ----------
    neighborhood
        Graph from :func:`~pysplit.build_spatial_network` or
        :func:`~pysplit.build_transcriptomics_network`.
    rctd
        A post-processed deconvolution result.
    columns
        Columns to project. Defaults to every column of ``results_df``.

    Returns
    -------
    Neighborhood
        The same object, mutated in place and returned for chaining.
    """
    rctd.check_post_processed()
    df = rctd.results_df

    missing = [c for c in neighborhood.cell_id if c not in df.index]
    if missing:
        logger.warning(
            "%d cell(s) of the graph are absent from the deconvolution results; "
            "their neighbour entries will be missing.",
            len(missing),
        )
    aligned = df.reindex(neighborhood.cell_id)

    columns = list(aligned.columns) if columns is None else list(columns)
    for col in columns:
        if col not in aligned.columns:
            continue
        series = aligned[col]
        if col in _LABEL_COLUMNS or series.dtype == object or isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            neighborhood.fields[col] = _gather_labels(series, neighborhood.nn_idx)
        else:
            neighborhood.fields[col] = _gather_numeric(series, neighborhood.nn_idx)
    return neighborhood


# --------------------------------------------------------------------------- #
# helpers on cells x k label matrices
# --------------------------------------------------------------------------- #
def _is_missing(mat: np.ndarray) -> np.ndarray:
    return pd.isna(mat.astype(object))


def _match_focal(mat: np.ndarray, focal: np.ndarray | None = None) -> np.ndarray:
    """Boolean ``cells x (k-1)`` mask: neighbour label equals the focal label.

    Missing values never match, mirroring R's ``NA == NA -> NA`` semantics.
    """
    focal_values = mat[:, [0]] if focal is None else np.asarray(focal, dtype=object)[:, None]
    neighbours = mat[:, 1:]
    valid = ~_is_missing(neighbours) & ~_is_missing(focal_values)
    return (neighbours == focal_values) & valid


def _match_indices(mask: np.ndarray) -> list[np.ndarray]:
    """Per-cell neighbour column indices (into the ``cells x k`` layout) where ``mask``."""
    rows, cols = np.nonzero(mask)
    cols = cols + 1  # mask starts at neighbour column 1
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    splits = np.searchsorted(rows, np.arange(mask.shape[0] + 1))
    return [cols[splits[i] : splits[i + 1]] for i in range(mask.shape[0])]


# --------------------------------------------------------------------------- #
# 1. infiltration metrics
# --------------------------------------------------------------------------- #
def add_infiltration_metrics(neighborhood: Neighborhood) -> Neighborhood:
    """Count how many neighbours carry the focal cell's primary/secondary type.

    A cell whose *secondary* cell type is the *primary* cell type of several
    neighbours is very likely contaminated by them rather than expressing a
    genuine mixed phenotype - that is what ``second_type_neighbors_N`` captures.

    Adds the metrics ``total_neighbors_N``, ``annotated_neighbors_N``,
    ``total_singlets_neighbors_N``, and ``{first,second,same_second}_type*
    _neighbors_N`` (plus the underlying index lists as fields).
    """
    neighborhood.require(
        "first_type",
        "second_type",
        "spot_class",
        hint="Run add_deconvolution_to_neighborhood() first.",
    )
    f = neighborhood.fields
    m = neighborhood.metrics
    k = neighborhood.k

    first_type = f["first_type"]
    second_type = f["second_type"]
    spot_class = f["spot_class"]

    m["total_neighbors_N"] = neighborhood.valid.sum(axis=1) - 1
    m["annotated_neighbors_N"] = (~_is_missing(first_type)).sum(axis=1) - 1
    m["total_singlets_neighbors_N"] = (spot_class == "singlet").sum(axis=1)

    # neighbours whose annotation was rejected carry no usable label
    first_type_no_reject = first_type.copy()
    first_type_no_reject[spot_class == "reject"] = None
    f["first_type_no_reject"] = first_type_no_reject

    is_singlet_neighbour = spot_class[:, 1:] == "singlet"

    def _register(name: str, mask: np.ndarray) -> None:
        f[f"{name}"] = _match_indices(mask)
        m[f"{name}_N"] = mask.sum(axis=1)

    # --- neighbours whose PRIMARY type is the focal cell's SECONDARY type ----
    focal_second = second_type[:, 0]
    mask_second = _match_focal(first_type, focal=focal_second)
    _register("second_type_neighbors", mask_second)

    mask_second_no_reject = _match_focal(first_type_no_reject, focal=focal_second)
    _register("second_type_neighbors_no_reject", mask_second_no_reject)
    _register("second_type_singlets_neighbors", mask_second & is_singlet_neighbour)

    if "first_type_class" in f and "second_type_class" in f:
        mask_second_class = _match_focal(
            f["first_type_class"], focal=f["second_type_class"][:, 0]
        )
        _register("second_type_class_neighbors", mask_second_class)
    else:
        logger.warning(
            "'first_type_class'/'second_type_class' not available; class-level "
            "infiltration metrics were skipped (supply `class_df` to your "
            "RCTDResult to enable them)."
        )

    # --- neighbours sharing the focal cell's PRIMARY type -------------------
    mask_first = _match_focal(first_type)
    _register("first_type_neighbors", mask_first)
    _register("first_type_singlets_neighbors", mask_first & is_singlet_neighbour)

    if "first_type_class" in f:
        _register("first_type_class_neighbors", _match_focal(f["first_type_class"]))

    # --- neighbours sharing the focal cell's SECONDARY type -----------------
    _register("same_second_type_neighbors", _match_focal(second_type))
    return neighborhood


# --------------------------------------------------------------------------- #
# 2. weight composition of the neighbourhood
# --------------------------------------------------------------------------- #
def add_neighborhood_weight_composition(
    neighborhood: Neighborhood,
    cell_types: Sequence[str] | None = None,
) -> Neighborhood:
    """Average cell-type composition of every cell's neighbourhood.

    Each annotated neighbour contributes its doublet weights (``w1`` to its
    primary type, ``w2`` to its secondary type; a confident singlet contributes
    ``1`` to its primary type). The focal cell is excluded, and the result is
    normalised to sum to one per cell.

    The composition matrix is stored in ``extras['neighborhood_weight_composition']``
    as a ``cells x cell_types`` DataFrame.
    """
    neighborhood.require(
        "first_type",
        "second_type",
        "weight_first_type",
        "weight_second_type",
        hint="Run add_deconvolution_to_neighborhood() first.",
    )
    f = neighborhood.fields
    first_type = f["first_type"][:, 0]
    second_type = f["second_type"][:, 0]
    w1 = f["weight_first_type"][:, 0]
    w2 = f["weight_second_type"][:, 0]

    if cell_types is None:
        labels = pd.unique(
            np.concatenate([f["first_type"].ravel(), f["second_type"].ravel()])
        )
        cell_types = sorted(str(x) for x in labels if not pd.isna(x))
    cell_types = list(cell_types)
    type_pos = {t: i for i, t in enumerate(cell_types)}

    n_cells = neighborhood.n_cells
    # per-cell contribution profile
    profile = np.zeros((n_cells, len(cell_types)), dtype=float)
    for i in range(n_cells):
        ft = first_type[i]
        if ft is None or pd.isna(ft) or ft not in type_pos:
            continue
        st = second_type[i]
        if st is not None and not pd.isna(st) and st in type_pos:
            profile[i, type_pos[ft]] += 0.0 if np.isnan(w1[i]) else w1[i]
            profile[i, type_pos[st]] += 0.0 if np.isnan(w2[i]) else w2[i]
        else:
            profile[i, type_pos[ft]] += 1.0

    # adjacency over annotated neighbours, focal cell excluded
    nn_idx = neighborhood.nn_idx[:, 1:]
    annotated = ~_is_missing(f["first_type"][:, 1:])
    keep = (nn_idx >= 0) & annotated
    rows = np.repeat(np.arange(n_cells), keep.sum(axis=1))
    cols = nn_idx[keep]
    adjacency = sparse.csr_matrix(
        (np.ones(rows.shape[0]), (rows, cols)), shape=(n_cells, n_cells)
    )

    composition = adjacency @ profile
    totals = composition.sum(axis=1)
    nz = totals > 0
    composition[nz] = composition[nz] / totals[nz, None]

    neighborhood.extras["neighborhood_weight_composition"] = pd.DataFrame(
        composition, index=pd.Index(neighborhood.cell_id, name="cell_id"), columns=cell_types
    )
    return neighborhood


def add_cell_type_neighborhood_weights(neighborhood: Neighborhood) -> Neighborhood:
    """Pull each cell's own primary/secondary weight out of its neighbourhood composition.

    ``neighborhood_weights_second_type`` is the **local diffusion score**: the
    share of the neighbourhood made up of the cell's own secondary cell type. A
    high value means the secondary signal really is present around the cell, so
    purifying it is warranted; a low value suggests a genuine phenotype instead.
    """
    if "neighborhood_weight_composition" not in neighborhood.extras:
        raise ValueError(
            "Run add_neighborhood_weight_composition() before "
            "add_cell_type_neighborhood_weights()."
        )
    composition = neighborhood.extras["neighborhood_weight_composition"]
    values = composition.to_numpy()
    type_pos = {t: i for i, t in enumerate(composition.columns)}

    first_type = neighborhood.fields["first_type"][:, 0]
    second_type = neighborhood.fields["second_type"][:, 0]

    w_first = np.full(neighborhood.n_cells, np.nan)
    w_second = np.full(neighborhood.n_cells, np.nan)
    for i in range(neighborhood.n_cells):
        ft, st = first_type[i], second_type[i]
        if ft is None or pd.isna(ft) or ft not in type_pos:
            continue
        w_first[i] = values[i, type_pos[ft]]
        if st is not None and not pd.isna(st) and st in type_pos:
            w_second[i] = values[i, type_pos[st]]
        else:
            w_second[i] = 0.0

    neighborhood.metrics["neighborhood_weights_first_type"] = w_first
    neighborhood.metrics["neighborhood_weights_second_type"] = w_second
    return neighborhood


def _sum_over_indices(matrix: np.ndarray, index_lists: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(index_lists), dtype=float)
    for i, idx in enumerate(index_lists):
        if len(idx):
            out[i] = np.nansum(matrix[i, idx])
    return out


def add_spilling_type_weights(neighborhood: Neighborhood) -> Neighborhood:
    """Weights carried by neighbours of the focal cell's secondary ("spilling") type.

    Adds ``sum_w1_second_type_in_neighborhood`` (weight those neighbours put on
    it as their *primary* type), ``sum_w2_second_type_in_neighborhood`` (as
    their *secondary* type) and ``max_weight_of_spilling_type_in_neighborhood``.
    """
    neighborhood.require(
        "second_type_neighbors",
        "same_second_type_neighbors",
        "weight_first_type",
        "weight_second_type",
        hint="Run add_infiltration_metrics() first.",
    )
    f = neighborhood.fields
    w1_lists = [
        f["weight_first_type"][i, idx] for i, idx in enumerate(f["second_type_neighbors"])
    ]
    w2_lists = [
        f["weight_second_type"][i, idx]
        for i, idx in enumerate(f["same_second_type_neighbors"])
    ]
    f["w1_second_type_in_neighborhood"] = w1_lists
    f["w2_second_type_in_neighborhood"] = w2_lists

    neighborhood.metrics["sum_w1_second_type_in_neighborhood"] = _sum_over_indices(
        f["weight_first_type"], f["second_type_neighbors"]
    )
    neighborhood.metrics["sum_w2_second_type_in_neighborhood"] = _sum_over_indices(
        f["weight_second_type"], f["same_second_type_neighbors"]
    )
    neighborhood.metrics["max_weight_of_spilling_type_in_neighborhood"] = np.array(
        [
            max(
                [0.0]
                + [v for v in np.concatenate([w1_lists[i], w2_lists[i]]) if not np.isnan(v)]
            )
            for i in range(neighborhood.n_cells)
        ]
    )
    return neighborhood


def add_spilling_type_ncount(neighborhood: Neighborhood) -> Neighborhood:
    """Total counts held by the neighbourhood and by its spilling-type members.

    Requires an ``nCount`` field (project it with
    ``add_deconvolution_to_neighborhood`` after putting ``nCount`` into
    ``results_df``, or set ``neighborhood.fields['nCount']`` yourself).
    """
    neighborhood.require("nCount", "second_type_neighbors")
    ncount = neighborhood.fields["nCount"]
    neighborhood.metrics["sum_nCount_neighborhood"] = np.nansum(ncount[:, 1:], axis=1)
    neighborhood.metrics["sum_nCount_neighborhood_spilling_type"] = _sum_over_indices(
        ncount, neighborhood.fields["second_type_neighbors"]
    )
    return neighborhood


# --------------------------------------------------------------------------- #
# 3. annotation transfer from the neighbourhood
# --------------------------------------------------------------------------- #
def _label_codes(mat: np.ndarray) -> tuple[np.ndarray, list]:
    """Factorise a label matrix to integer codes plus the sorted categories.

    Missing values get code ``-1``. Categories are sorted so that ties in the
    mode resolve on the label, matching R's ``which.max(table(...))`` over a
    sorted factor.
    """
    flat = pd.Series(mat.reshape(-1), dtype=object)
    categories = sorted({v for v in flat.dropna().unique()}, key=str)
    lookup = {v: i for i, v in enumerate(categories)}
    codes = np.fromiter(
        (lookup.get(v, -1) for v in flat), dtype=np.int64, count=flat.size
    ).reshape(mat.shape)
    return codes, categories


def _neighbour_counts(codes: np.ndarray, n_categories: int) -> np.ndarray:
    """``cells x (n_categories + 1)`` label counts over neighbours.

    The final column counts missing labels, so it loses mode ties against any
    real label - which is where R's ``useNA = "ifany"`` puts it too.
    """
    n_rows = codes.shape[0]
    counts = np.zeros((n_rows, n_categories + 1), dtype=np.int64)
    rows = np.repeat(np.arange(n_rows), codes.shape[1])
    columns = np.where(codes < 0, n_categories, codes).reshape(-1)
    np.add.at(counts, (rows, columns), 1)
    return counts


def _neighbour_mode(mat: np.ndarray) -> np.ndarray:
    """Modal label among each cell's neighbours (column 0, the cell, excluded).

    Vectorised over cells: a per-row pandas ``value_counts`` costs minutes at
    Xenium scale, while this is one ``np.add.at`` plus an ``argmax``.
    """
    neighbours = mat[:, 1:]
    if neighbours.shape[1] == 0:
        return np.full(mat.shape[0], None, dtype=object)
    codes, categories = _label_codes(neighbours)
    if not categories:
        return np.full(mat.shape[0], None, dtype=object)
    counts = _neighbour_counts(codes, len(categories))
    labels = np.asarray(categories + [None], dtype=object)
    out = labels[counts.argmax(axis=1)]
    out[counts.sum(axis=1) == 0] = None  # no neighbours at all -> no mode
    return out


def _neighbour_certainty(mat: np.ndarray) -> np.ndarray:
    """``1 - H(labels) / log(k)`` per cell, vectorised.

    Missing labels are dropped from the entropy (as R's ``table()`` does) but
    still counted in ``k``, so a neighbourhood padded with pruned slots scores
    lower than a full one.
    """
    neighbours = mat[:, 1:]
    n_rows, k = neighbours.shape
    if k <= 1:
        return np.full(n_rows, np.nan)
    codes, categories = _label_codes(neighbours)
    if not categories:
        return np.full(n_rows, np.nan)
    counts = _neighbour_counts(codes, len(categories))[:, : len(categories)]
    totals = counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        proportions = np.where(totals[:, None] > 0, counts / totals[:, None], 0.0)
        terms = np.where(proportions > 0, proportions * np.log(proportions), 0.0)
    out = 1.0 - (-terms.sum(axis=1)) / np.log(k)
    out[totals == 0] = np.nan
    return out


def add_annotation_from_neighbors(neighborhood: Neighborhood) -> Neighborhood:
    """Majority label of each cell's neighbourhood, and whether the cell agrees.

    ``first_type_neighborhood_agreement == False`` flags a cell whose own
    primary label disagrees with its transcriptomic neighbourhood - a candidate
    for SPLIT-shift label swapping.
    """
    neighborhood.require(
        "first_type", "second_type", hint="Run add_deconvolution_to_neighborhood() first."
    )
    f, m = neighborhood.fields, neighborhood.metrics

    for key in ("first_type", "second_type", "first_type_class", "second_type_class"):
        if key not in f:
            if key.endswith("_class"):
                logger.warning(
                    "'%s' not available; '%s_neighborhood' was not computed.", key, key
                )
            continue
        mode = _neighbour_mode(f[key])
        m[f"{key}_neighborhood"] = mode
        focal = f[key][:, 0]
        agreement = np.array(
            [
                (
                    np.nan
                    if (focal[i] is None or pd.isna(focal[i]) or mode[i] is None)
                    else focal[i] == mode[i]
                )
                for i in range(neighborhood.n_cells)
            ],
            dtype=object,
        )
        m[f"{key}_neighborhood_agreement"] = agreement
    return neighborhood


def add_neighborhood_annotation_certainty(neighborhood: Neighborhood) -> Neighborhood:
    """Label homogeneity of each neighbourhood, in ``[0, 1]``.

    ``1`` means every neighbour shares one label; ``0`` means labels are spread
    evenly. Computed as ``1 - H(labels) / log(k)``.
    """
    neighborhood.require(
        "first_type", "second_type", hint="Run add_deconvolution_to_neighborhood() first."
    )
    f, m = neighborhood.fields, neighborhood.metrics
    for key in ("first_type", "second_type", "first_type_class", "second_type_class"):
        if key not in f:
            if key.endswith("_class"):
                logger.warning(
                    "'%s' not available; '%s_neighborhood_certainty' was not computed.",
                    key,
                    key,
                )
            continue
        m[f"{key}_neighborhood_certainty"] = _neighbour_certainty(f[key])
    return neighborhood


# --------------------------------------------------------------------------- #
# 4. pipelines
# --------------------------------------------------------------------------- #
def spatial_metrics(neighborhood: Neighborhood, rctd: RCTDResult) -> Neighborhood:
    """Full spatial-neighbourhood pipeline for spatially-aware SPLIT.

    Runs :func:`add_deconvolution_to_neighborhood`,
    :func:`add_infiltration_metrics`, :func:`add_neighborhood_weight_composition`,
    :func:`add_cell_type_neighborhood_weights` and
    :func:`add_spilling_type_weights`.

    The metric to threshold afterwards is ``neighborhood_weights_second_type``.

    Only :data:`SPATIAL_COLUMNS` are projected onto the neighbourhood; call
    :func:`add_deconvolution_to_neighborhood` yourself first if you need more.
    """
    rctd.check_post_processed()
    # Only project the columns the metrics below read: projecting all of
    # results_df would build a cells x k object matrix per column.
    add_deconvolution_to_neighborhood(neighborhood, rctd, columns=SPATIAL_COLUMNS)
    add_infiltration_metrics(neighborhood)
    add_neighborhood_weight_composition(neighborhood, cell_types=rctd.cell_types)
    add_cell_type_neighborhood_weights(neighborhood)
    add_spilling_type_weights(neighborhood)
    return neighborhood


def transcriptomics_metrics(neighborhood: Neighborhood, rctd: RCTDResult) -> Neighborhood:
    """Full transcriptomic-neighbourhood pipeline for SPLIT-shift.

    Runs :func:`add_deconvolution_to_neighborhood`,
    :func:`add_annotation_from_neighbors` and
    :func:`add_neighborhood_annotation_certainty`.

    Only :data:`TRANSCRIPTOMIC_COLUMNS` are projected onto the neighbourhood,
    which matters at the large ``k`` this mode wants.
    """
    rctd.check_post_processed()
    add_deconvolution_to_neighborhood(
        neighborhood, rctd, columns=TRANSCRIPTOMIC_COLUMNS
    )
    add_annotation_from_neighbors(neighborhood)
    add_neighborhood_annotation_certainty(neighborhood)
    return neighborhood


def neighborhood_to_dataframe(neighborhood: Neighborhood) -> pd.DataFrame:
    """Per-cell metrics as a DataFrame ready for ``adata.obs``."""
    return neighborhood.to_dataframe()


# --------------------------------------------------------------------------- #
# 5. swapping score
# --------------------------------------------------------------------------- #
def compute_swapping_score(obs: pd.DataFrame) -> pd.DataFrame:
    """Heuristic score for how much a cell's labels want to be swapped.

    Expects a per-cell frame carrying both the deconvolution results and the
    spatial *and* transcriptomic neighbourhood metrics (i.e. the ``adata.obs``
    you get after running both pipelines and merging them in).

    A positive ``total_swapping_score`` means the evidence leans towards the
    secondary cell type being the cell's real phenotype. This is a diagnostic:
    :func:`~pysplit.balance_by_score` with ``swap_labels=True`` performs the
    actual SPLIT-shift using neighbourhood agreement, not this score.
    """
    df = obs
    positive_terms = [
        (2.0, "second_type", "first_type_neighborhood"),
        (1.0, "second_type_class", "first_type_class_neighborhood"),
        (1.0, "first_type", "second_type_neighborhood"),
        (1.0, "first_type_class", "second_type_class_neighborhood"),
        (1.0, "second_type", "annot_min_singlet_score"),
        (1.0, "second_type", "annot_max_weight"),
        (1.0, "second_type", "annot_max_doublet_weight"),
    ]
    negative_terms = [
        (2.0, "first_type", "first_type_neighborhood"),
        (1.0, "second_type_class", "first_type_class_neighborhood"),
        (1.0, "second_type", "second_type_neighborhood"),
        (1.0, "second_type_class", "second_type_class_neighborhood"),
        (1.0, "first_type", "annot_min_singlet_score"),
        (1.0, "first_type", "annot_max_weight"),
        (1.0, "first_type", "annot_max_doublet_weight"),
    ]

    def _accumulate(terms, extra=0.0):
        total = np.full(len(df), float(extra))
        for weight, left, right in terms:
            if left not in df.columns or right not in df.columns:
                continue
            match = (
                df[left].astype(object) == df[right].astype(object)
            ).to_numpy() & df[left].notna().to_numpy() & df[right].notna().to_numpy()
            total = total + weight * match
        return total

    positive = _accumulate(positive_terms) / 8.0
    # "the neighbourhood does not agree with the secondary type" counts against a swap
    mismatch = np.zeros(len(df))
    if {"second_type", "first_type_neighborhood"}.issubset(df.columns):
        mismatch = 2.0 * (
            (df["second_type"].astype(object) != df["first_type_neighborhood"].astype(object))
            & df["second_type"].notna()
            & df["first_type_neighborhood"].notna()
        ).to_numpy()
    negative = (_accumulate(negative_terms) + mismatch) / 10.0

    out = pd.DataFrame(index=df.index)
    out["transcriptomic_swapping_score"] = positive - negative
    if {"neighborhood_weights_first_type", "neighborhood_weights_second_type"}.issubset(
        df.columns
    ):
        # A neighbourhood dominated by the cell's *primary* type suggests that
        # primary signal is spillover from those neighbours, so the secondary
        # type is the cell's real phenotype - hence first minus second.
        out["spatial_swapping_score"] = (
            df["neighborhood_weights_first_type"].to_numpy(dtype=float)
            - df["neighborhood_weights_second_type"].to_numpy(dtype=float)
        )
    else:
        out["spatial_swapping_score"] = np.nan
    out["total_swapping_score"] = (
        out["transcriptomic_swapping_score"] + out["spatial_swapping_score"]
    )
    out["max_swapping_score"] = out[
        ["transcriptomic_swapping_score", "spatial_swapping_score"]
    ].max(axis=1)
    out["min_swapping_score"] = out[
        ["transcriptomic_swapping_score", "spatial_swapping_score"]
    ].min(axis=1)
    # a swap is only meaningful when there is a secondary cell type to swap in
    if "second_type" in df.columns:
        out.loc[df["second_type"].isna(), :] = np.nan
    return out
