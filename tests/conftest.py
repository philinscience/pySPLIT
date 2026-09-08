"""Synthetic Xenium-like fixtures shared by the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None

CELL_TYPES = ["Tumor", "Tcell", "Myeloid", "Stromal"]
CLASS_MAP = {"Tumor": "Epithelial", "Tcell": "Immune", "Myeloid": "Immune",
             "Stromal": "Stromal"}


def make_reference(n_genes: int = 30, seed: int = 0) -> pd.DataFrame:
    """Reference with a block of markers exclusive to each cell type."""
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame(
        rng.uniform(0.001, 0.01, size=(len(CELL_TYPES), n_genes)),
        index=CELL_TYPES,
        columns=[f"gene{i}" for i in range(n_genes)],
    )
    block = n_genes // len(CELL_TYPES)
    for i, cell_type in enumerate(CELL_TYPES):
        marker_genes = reference.columns[i * block : (i + 1) * block]
        reference.loc[cell_type, marker_genes] = rng.uniform(0.05, 0.2, size=block)
    # one gene expressed by nobody, to exercise the all-zero-denominator path
    reference.iloc[:, -1] = 0.0
    return reference


def make_dataset(n_cells: int = 60, n_genes: int = 30, seed: int = 1):
    """Counts, weights, labels and coordinates for a small synthetic sample."""
    rng = np.random.default_rng(seed)
    reference = make_reference(n_genes=n_genes, seed=seed)
    cell_ids = [f"cell{i}" for i in range(n_cells)]

    first_type = rng.choice(CELL_TYPES, size=n_cells)
    second_type = np.array(
        [rng.choice([t for t in CELL_TYPES if t != ft]) for ft in first_type], dtype=object
    )
    w1 = rng.uniform(0.55, 1.0, size=n_cells)

    # spot classes: a mix of confident singlets, doublets and a few rejects
    spot_class = np.array(["doublet_certain"] * n_cells, dtype=object)
    spot_class[rng.random(n_cells) < 0.35] = "singlet"
    spot_class[rng.random(n_cells) < 0.15] = "doublet_uncertain"
    spot_class[:2] = "reject"

    # full decomposition weights, dominated by the two assigned types
    weights = rng.uniform(0, 0.02, size=(n_cells, len(CELL_TYPES)))
    type_pos = {t: i for i, t in enumerate(CELL_TYPES)}
    for i in range(n_cells):
        weights[i, type_pos[first_type[i]]] += w1[i]
        weights[i, type_pos[second_type[i]]] += 1.0 - w1[i]
    weights = weights / weights.sum(axis=1, keepdims=True)

    # counts drawn from the true mixed profile, so purification has work to do
    profile = weights @ reference.to_numpy()
    depth = rng.integers(80, 400, size=n_cells)
    counts = rng.poisson(profile / profile.sum(axis=1, keepdims=True) * depth[:, None])

    coords = pd.DataFrame(
        rng.uniform(0, 100, size=(n_cells, 2)), index=cell_ids, columns=["x", "y"]
    )

    results_df = pd.DataFrame(
        {
            "spot_class": spot_class,
            "first_type": first_type,
            "second_type": second_type,
            "min_score": rng.uniform(0, 1, n_cells),
            "singlet_score": rng.uniform(0, 1, n_cells),
        },
        index=pd.Index(cell_ids, name="cell_id"),
    )
    weights_df = pd.DataFrame(weights, index=cell_ids, columns=CELL_TYPES)
    weights_doublet = pd.DataFrame(
        np.column_stack([w1, 1 - w1]),
        index=cell_ids,
        columns=["first_type", "second_type"],
    )
    return dict(
        counts=sparse.csr_matrix(counts),
        cell_ids=cell_ids,
        gene_ids=list(reference.columns),
        reference=reference,
        results_df=results_df,
        weights=weights_df,
        weights_doublet=weights_doublet,
        coords=coords,
        class_df=pd.Series(CLASS_MAP),
    )


@pytest.fixture
def dataset():
    return make_dataset()


@pytest.fixture
def rctd(dataset):
    import pysplit

    result = pysplit.RCTDResult(
        results_df=dataset["results_df"],
        weights=dataset["weights"],
        weights_doublet=dataset["weights_doublet"],
        reference=dataset["reference"],
        class_df=dataset["class_df"],
        coords=dataset["coords"],
    )
    return pysplit.run_post_process_rctd(result, verbose=False)


@pytest.fixture
def adata(dataset):
    obj = ad.AnnData(
        X=dataset["counts"].astype(float),
        obs=pd.DataFrame(index=pd.Index(dataset["cell_ids"], name="cell_id")),
        var=pd.DataFrame(index=pd.Index(dataset["gene_ids"], name="gene")),
    )
    obj.obsm["spatial"] = dataset["coords"].to_numpy()
    return obj
