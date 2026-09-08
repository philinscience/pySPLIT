"""Small shared helpers for pySPLIT."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger("pysplit")

NA = None  # sentinel used for "missing cell type" in object arrays

#: Ordering of RCTD spot classes, from least to most confident (matches the
#: ``ordered()`` factor used by the R package).
SPOT_CLASS_LEVELS = ("reject", "doublet_uncertain", "doublet_certain", "singlet")


def get_logger() -> logging.Logger:
    return logger


# --------------------------------------------------------------------------- #
# matrices
# --------------------------------------------------------------------------- #
def as_csr(x) -> sparse.csr_matrix:
    """Return ``x`` as a CSR matrix without copying when possible."""
    if sparse.isspmatrix_csr(x):
        return x
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


def to_dense_array(x) -> np.ndarray:
    """Return a dense 2-D :class:`numpy.ndarray` view of ``x``."""
    if sparse.issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def row_normalize(weights: np.ndarray, clip_negative: bool = True) -> np.ndarray:
    """Normalise rows to sum to one.

    Mirrors ``spacexr::normalize_weights``: negative weights are clipped to
    zero first, then each row is divided by its sum. All-zero rows are left
    untouched rather than producing ``NaN``.
    """
    w = np.asarray(weights, dtype=float).copy()
    if clip_negative:
        w[w < 0] = 0.0
    row_sums = w.sum(axis=1)
    nz = row_sums > 0
    w[nz] = w[nz] / row_sums[nz, None]
    return w


# --------------------------------------------------------------------------- #
# information theory
# --------------------------------------------------------------------------- #
def entropy(counts: Iterable[float], base: float | None = None) -> float:
    """Shannon entropy of a vector of (unnormalised) frequencies.

    Equivalent to ``entropy::entropy()`` in R: the input is normalised to a
    probability vector and zero-probability entries contribute nothing.
    ``base=None`` means natural log, matching the R default.
    """
    p = np.asarray(list(counts), dtype=float)
    total = p.sum()
    if total <= 0:
        return np.nan
    p = p[p > 0] / total
    h = float(-(p * np.log(p)).sum())
    if base is not None:
        h /= np.log(base)
    return h


def normalized_certainty(values: Sequence, drop_na: bool = True) -> float:
    """``1 - H(values) / log(n)`` - neighbourhood label homogeneity in ``[0, 1]``.

    ``1`` means every neighbour carries the same label, ``0`` means labels are
    uniformly spread over the neighbourhood.
    """
    vals = pd.Series(list(values), dtype=object)
    n = len(vals)
    if n <= 1:
        return np.nan
    if drop_na:
        vals = vals[vals.notna()]
    if len(vals) == 0:
        return np.nan
    freqs = vals.value_counts().to_numpy()
    return float(1.0 - entropy(freqs) / np.log(n))


def most_frequent(values: Sequence, na_as_category: bool = True):
    """Modal value of ``values``.

    Categories are compared in sorted order with missing values last, so ties
    resolve deterministically (this is what R's ``which.max(table(...))``
    does for a sorted factor).
    """
    vals = pd.Series(list(values), dtype=object)
    n_na = int(vals.isna().sum())
    counts = vals.value_counts(dropna=True)
    if len(counts):
        # sort by (-count, key) so ties break on the label
        order = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        best_label, best_count = order[0]
    else:
        best_label, best_count = None, 0
    if na_as_category and n_na > best_count:
        return None
    if best_count == 0:
        return None
    return best_label


# --------------------------------------------------------------------------- #
# label handling
# --------------------------------------------------------------------------- #
def as_object_labels(x) -> np.ndarray:
    """Coerce a label vector to a 1-D object array with ``None`` for missing."""
    arr = pd.Series(np.asarray(x, dtype=object))
    arr = arr.where(arr.notna(), None)
    return arr.to_numpy(dtype=object)


def index_of(labels: Sequence, universe: Sequence) -> np.ndarray:
    """Positions of ``labels`` within ``universe``; ``-1`` for missing labels."""
    lookup = {v: i for i, v in enumerate(universe)}
    return np.fromiter((lookup.get(v, -1) for v in labels), dtype=np.int64, count=len(labels))


def chunk_slices(n: int, chunk_size: int) -> list[slice]:
    """Split ``range(n)`` into consecutive slices of at most ``chunk_size``."""
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        return [slice(0, n)]
    return [slice(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]
