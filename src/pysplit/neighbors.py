"""k-nearest-neighbour graphs over space or transcriptome.

Two graphs drive the optional SPLIT modes:

* a **spatial** graph (pruned by a physical radius) tells us how much of a
  cell's secondary signal is actually present around it - the local diffusion
  potential used by spatially-aware SPLIT;
* a **transcriptomic** graph (over a PCA embedding) tells us which phenotype a
  cell's expression neighbourhood agrees on - the basis of SPLIT-shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ._utils import get_logger

logger = get_logger()

__all__ = ["Neighborhood", "compute_neighborhood", "build_spatial_network",
           "build_transcriptomics_network"]


@dataclass
class Neighborhood:
    """A kNN neighbourhood plus whatever per-neighbour metrics were computed.

    Attributes
    ----------
    nn_idx
        ``cells x k`` integer array of neighbour row indices. Column 0 is the
        cell itself. Pruned/missing entries are ``-1``.
    nn_dist
        Matching distances; ``nan`` where pruned.
    cell_id
        Cell ids in row order.
    fields
        Per-neighbour matrices (``cells x k``, e.g. the ``first_type`` of every
        neighbour) added by :func:`pysplit.add_deconvolution_to_neighborhood`.
    metrics
        Per-cell 1-D metrics, the things that end up in ``adata.obs``.
    """

    nn_idx: np.ndarray
    nn_dist: np.ndarray
    cell_id: list[str]
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, np.ndarray] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_cells(self) -> int:
        return self.nn_idx.shape[0]

    @property
    def k(self) -> int:
        return self.nn_idx.shape[1]

    @property
    def valid(self) -> np.ndarray:
        """Boolean ``cells x k`` mask of usable neighbour slots."""
        return self.nn_idx >= 0

    def __contains__(self, key: str) -> bool:
        return key in self.fields or key in self.metrics

    def require(self, *keys: str, hint: str = "") -> None:
        """Raise a helpful error if any of ``keys`` has not been computed."""
        missing = [k for k in keys if k not in self]
        if missing:
            message = f"The following fields are missing from the neighborhood: {missing}."
            if hint:
                message += f"\n{hint}"
            raise ValueError(message)

    def to_dataframe(self) -> pd.DataFrame:
        """Per-cell metrics as a DataFrame indexed by cell id (for ``adata.obs``)."""
        return pd.DataFrame(
            {k: np.asarray(v) for k, v in self.metrics.items()},
            index=pd.Index(self.cell_id, name="cell_id"),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Neighborhood({self.n_cells} cells, k={self.k}, "
            f"{len(self.fields)} fields, {len(self.metrics)} metrics)"
        )


# --------------------------------------------------------------------------- #
def _get_embedding(adata, basis: str, dims) -> np.ndarray:
    """Fetch an embedding from ``obsm``, tolerating scanpy's ``X_`` prefix."""
    for key in (basis, f"X_{basis}", basis.removeprefix("X_")):
        if key in adata.obsm:
            embedding = np.asarray(adata.obsm[key])
            break
    else:
        raise KeyError(
            f"Neither {basis!r} nor 'X_{basis}' is present in adata.obsm "
            f"(available: {list(adata.obsm)})."
        )
    if dims is not None:
        dims = list(dims)
        if max(dims) >= embedding.shape[1]:
            raise ValueError(
                f"Requested dimension {max(dims)} but {basis!r} has only "
                f"{embedding.shape[1]} column(s)."
            )
        embedding = embedding[:, dims]
    return np.ascontiguousarray(embedding, dtype=float)


def compute_neighborhood(
    adata,
    basis: str = "X_pca",
    dims=None,
    k_knn: int = 20,
    prune: bool = False,
    radius: float = np.inf,
    metric: str = "euclidean",
    n_jobs: int | None = -1,
) -> Neighborhood:
    """Build a kNN neighbourhood over an embedding stored in ``adata.obsm``.

    Parameters
    ----------
    adata
        AnnData whose ``obs_names`` label the cells.
    basis
        Key in ``obsm``. ``'spatial'`` and ``'X_pca'`` both work, with or
        without the ``X_`` prefix.
    dims
        Columns of the embedding to use (e.g. ``range(50)`` for 50 PCs).
    k_knn
        Neighbourhood size *including* the cell itself.
    prune
        Drop neighbours further away than ``radius``.
    radius
        Pruning distance, in the units of the embedding (µm for Xenium
        coordinates).
    metric, n_jobs
        Passed to :class:`sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    Neighborhood
    """
    embedding = _get_embedding(adata, basis, dims)
    n_cells = embedding.shape[0]
    k = int(min(k_knn, n_cells))
    if k < 2:
        raise ValueError("`k_knn` must be at least 2 (the cell plus one neighbour).")

    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=n_jobs).fit(embedding)
    nn_dist, nn_idx = nn.kneighbors(embedding, return_distance=True)
    nn_idx = nn_idx.astype(np.int64)
    nn_dist = nn_dist.astype(float)

    # sklearn does not guarantee the cell itself comes first for duplicate
    # coordinates; make column 0 the cell itself unconditionally.
    self_first = nn_idx[:, 0] == np.arange(n_cells)
    for i in np.flatnonzero(~self_first):
        pos = np.flatnonzero(nn_idx[i] == i)
        if pos.size:
            j = pos[0]
            nn_idx[i, [0, j]] = nn_idx[i, [j, 0]]
            nn_dist[i, [0, j]] = nn_dist[i, [j, 0]]
        else:
            nn_idx[i, 0] = i
            nn_dist[i, 0] = 0.0

    if prune:
        max_dist = float(np.nanmax(nn_dist))
        if radius > max_dist:
            logger.warning(
                "No pruning applied: radius (%.3g) exceeds the largest distance "
                "in the graph (%.3g).",
                radius,
                max_dist,
            )
        else:
            drop = nn_dist > radius
            drop[:, 0] = False  # never prune the cell itself
            n_before = int(nn_idx.shape[0] * (nn_idx.shape[1] - 1))
            nn_idx[drop] = -1
            nn_dist[drop] = np.nan
            n_pruned = int(drop.sum())
            logger.info(
                "pruned %d/%d edges (%.0f%%) beyond %.3g",
                n_pruned,
                n_before,
                100 * n_pruned / max(n_before, 1),
                radius,
            )

    return Neighborhood(
        nn_idx=nn_idx,
        nn_dist=nn_dist,
        cell_id=list(map(str, adata.obs_names)),
    )


def build_spatial_network(
    adata,
    basis: str = "spatial",
    dims=(0, 1),
    k_knn: int = 20,
    prune: bool = True,
    radius: float = 30.0,
    **kwargs,
) -> Neighborhood:
    """Spatial kNN graph, pruned at ``radius`` (µm for Xenium).

    A radius of 15-30 µm keeps the graph to physically adjacent cells, which is
    what makes the diffusion score meaningful.
    """
    return compute_neighborhood(
        adata, basis=basis, dims=dims, k_knn=k_knn, prune=prune, radius=radius, **kwargs
    )


def build_transcriptomics_network(
    adata,
    basis: str = "X_pca",
    dims=range(50),
    k_knn: int = 100,
    prune: bool = False,
    **kwargs,
) -> Neighborhood:
    """Transcriptomic kNN graph over a PCA embedding.

    ``k_knn`` is deliberately large (100 by default): SPLIT-shift asks what
    phenotype a cell's expression neighbourhood agrees on, which needs a broad
    neighbourhood to be stable.
    """
    return compute_neighborhood(
        adata, basis=basis, dims=dims, k_knn=k_knn, prune=prune, **kwargs
    )
