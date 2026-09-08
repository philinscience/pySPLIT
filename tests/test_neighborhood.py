"""kNN graphs and neighbourhood metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysplit
from conftest import CELL_TYPES


def test_graph_puts_the_cell_itself_first(adata):
    nbhd = pysplit.build_spatial_network(adata, k_knn=6, prune=False)
    assert nbhd.nn_idx.shape == (adata.n_obs, 6)
    assert np.array_equal(nbhd.nn_idx[:, 0], np.arange(adata.n_obs))
    assert np.allclose(nbhd.nn_dist[:, 0], 0.0)


def test_pruning_drops_distant_neighbours(adata):
    loose = pysplit.build_spatial_network(adata, k_knn=10, prune=False)
    tight = pysplit.build_spatial_network(adata, k_knn=10, prune=True, radius=8.0)
    assert tight.valid.sum() < loose.valid.sum()
    # nothing beyond the radius survives, and the cell itself always does
    assert np.all(np.isnan(tight.nn_dist[~tight.valid]))
    assert np.nanmax(tight.nn_dist) <= 8.0
    assert tight.valid[:, 0].all()


def test_infiltration_counts_match_a_hand_built_graph():
    """A T cell ringed by three tumour cells has three spilling neighbours."""
    cell_ids = ["focal", "n1", "n2", "n3"]
    results_df = pd.DataFrame(
        {
            "spot_class": ["doublet_certain", "singlet", "singlet", "reject"],
            "first_type": ["Tcell", "Tumor", "Tumor", "Tumor"],
            "second_type": ["Tumor", None, None, None],
            "weight_first_type": [0.7, 1.0, 1.0, 1.0],
            "weight_second_type": [0.3, np.nan, np.nan, np.nan],
        },
        index=pd.Index(cell_ids, name="cell_id"),
    )
    weights = pd.DataFrame(
        np.eye(4, len(CELL_TYPES)), index=cell_ids, columns=CELL_TYPES
    )
    rctd = pysplit.RCTDResult(
        results_df=results_df, weights=weights, post_processed=True
    )
    nbhd = pysplit.Neighborhood(
        nn_idx=np.array([[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]]),
        nn_dist=np.zeros((4, 4)),
        cell_id=cell_ids,
    )
    pysplit.add_deconvolution_to_neighborhood(nbhd, rctd)
    pysplit.add_infiltration_metrics(nbhd)

    # all three neighbours are Tumor == the focal cell's second type
    assert nbhd.metrics["second_type_neighbors_N"][0] == 3
    # one of them was rejected, so only two count when rejects are excluded
    assert nbhd.metrics["second_type_neighbors_no_reject_N"][0] == 2
    # two of the three are confident singlets
    assert nbhd.metrics["second_type_singlets_neighbors_N"][0] == 2
    # no neighbour shares the focal cell's own primary type
    assert nbhd.metrics["first_type_neighbors_N"][0] == 0
    assert nbhd.metrics["total_neighbors_N"][0] == 3


def test_diffusion_score_is_high_inside_a_matching_neighbourhood():
    """neighborhood_weights_second_type ~ 1 when every neighbour is that type."""
    cell_ids = ["focal", "n1", "n2"]
    results_df = pd.DataFrame(
        {
            "spot_class": ["doublet_certain", "singlet", "singlet"],
            "first_type": ["Tcell", "Tumor", "Tumor"],
            "second_type": ["Tumor", None, None],
            "weight_first_type": [0.7, 1.0, 1.0],
            "weight_second_type": [0.3, np.nan, np.nan],
        },
        index=pd.Index(cell_ids, name="cell_id"),
    )
    weights = pd.DataFrame(np.eye(3, len(CELL_TYPES)), index=cell_ids, columns=CELL_TYPES)
    rctd = pysplit.RCTDResult(
        results_df=results_df, weights=weights, post_processed=True
    )
    nbhd = pysplit.Neighborhood(
        nn_idx=np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]]),
        nn_dist=np.zeros((3, 3)),
        cell_id=cell_ids,
    )
    pysplit.add_deconvolution_to_neighborhood(nbhd, rctd)
    pysplit.add_infiltration_metrics(nbhd)
    pysplit.add_neighborhood_weight_composition(nbhd, cell_types=CELL_TYPES)
    pysplit.add_cell_type_neighborhood_weights(nbhd)

    assert nbhd.metrics["neighborhood_weights_second_type"][0] == pytest.approx(1.0)
    assert nbhd.metrics["neighborhood_weights_first_type"][0] == pytest.approx(0.0)
    # composition rows are normalised
    composition = nbhd.extras["neighborhood_weight_composition"]
    assert np.allclose(composition.sum(axis=1), 1.0)


def test_singlets_have_zero_diffusion_score():
    """A cell with no secondary type gets 0, not NaN - nothing can spill in."""
    cell_ids = ["focal", "n1"]
    results_df = pd.DataFrame(
        {
            "spot_class": ["singlet", "singlet"],
            "first_type": ["Tcell", "Tumor"],
            "second_type": [None, None],
            "weight_first_type": [1.0, 1.0],
            "weight_second_type": [np.nan, np.nan],
        },
        index=pd.Index(cell_ids, name="cell_id"),
    )
    weights = pd.DataFrame(np.eye(2, len(CELL_TYPES)), index=cell_ids, columns=CELL_TYPES)
    rctd = pysplit.RCTDResult(results_df=results_df, weights=weights, post_processed=True)
    nbhd = pysplit.Neighborhood(
        nn_idx=np.array([[0, 1], [1, 0]]), nn_dist=np.zeros((2, 2)), cell_id=cell_ids
    )
    pysplit.add_deconvolution_to_neighborhood(nbhd, rctd)
    pysplit.add_infiltration_metrics(nbhd)
    pysplit.add_neighborhood_weight_composition(nbhd, cell_types=CELL_TYPES)
    pysplit.add_cell_type_neighborhood_weights(nbhd)
    assert nbhd.metrics["neighborhood_weights_second_type"][0] == pytest.approx(0.0)


def test_annotation_transfer_and_certainty():
    """A cell surrounded by one label takes it, with certainty 1."""
    cell_ids = ["focal", "n1", "n2", "n3"]
    results_df = pd.DataFrame(
        {
            "spot_class": ["doublet_certain"] * 4,
            "first_type": ["Tcell", "Tumor", "Tumor", "Tumor"],
            "second_type": ["Tumor", "Tcell", "Tcell", "Myeloid"],
            "weight_first_type": [0.6] * 4,
            "weight_second_type": [0.4] * 4,
        },
        index=pd.Index(cell_ids, name="cell_id"),
    )
    weights = pd.DataFrame(np.eye(4, len(CELL_TYPES)), index=cell_ids, columns=CELL_TYPES)
    rctd = pysplit.RCTDResult(results_df=results_df, weights=weights, post_processed=True)
    nbhd = pysplit.Neighborhood(
        nn_idx=np.array([[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]]),
        nn_dist=np.zeros((4, 4)),
        cell_id=cell_ids,
    )
    pysplit.add_deconvolution_to_neighborhood(nbhd, rctd)
    pysplit.add_annotation_from_neighbors(nbhd)
    pysplit.add_neighborhood_annotation_certainty(nbhd)

    assert nbhd.metrics["first_type_neighborhood"][0] == "Tumor"
    assert nbhd.metrics["first_type_neighborhood_agreement"][0] is False
    assert nbhd.metrics["first_type_neighborhood_certainty"][0] == pytest.approx(1.0)
    # the focal cell's neighbours are split 2 Tcell / 1 Myeloid on second_type
    assert nbhd.metrics["second_type_neighborhood"][0] == "Tcell"
    assert nbhd.metrics["second_type_neighborhood_certainty"][0] < 1.0


def test_full_pipelines_run_on_a_synthetic_sample(adata, rctd):
    sp_nw = pysplit.build_spatial_network(adata, k_knn=8, prune=True, radius=25.0)
    pysplit.spatial_metrics(sp_nw, rctd)
    spatial = sp_nw.to_dataframe()
    assert "neighborhood_weights_second_type" in spatial.columns
    assert len(spatial) == adata.n_obs
    scores = spatial["neighborhood_weights_second_type"].dropna()
    assert scores.between(0, 1).all()

    adata.obsm["X_pca"] = np.asarray(adata.X.todense())[:, :5]
    tr_nw = pysplit.build_transcriptomics_network(adata, dims=range(5), k_knn=10)
    pysplit.transcriptomics_metrics(tr_nw, rctd)
    transcriptomic = tr_nw.to_dataframe()
    assert "first_type_neighborhood_agreement" in transcriptomic.columns
    assert "first_type_class_neighborhood" in transcriptomic.columns

    merged = spatial.join(transcriptomic, rsuffix="_tr").join(
        rctd.results_df, rsuffix="_rctd"
    )
    swapping = pysplit.compute_swapping_score(merged)
    assert "total_swapping_score" in swapping.columns
    assert len(swapping) == adata.n_obs


def test_missing_prerequisites_raise_a_helpful_error(adata, rctd):
    nbhd = pysplit.build_spatial_network(adata, k_knn=5, prune=False)
    with pytest.raises(ValueError, match="add_deconvolution_to_neighborhood"):
        pysplit.add_infiltration_metrics(nbhd)


# --------------------------------------------------------------------------- #
# the vectorised helpers must stay equivalent to the readable reference
# implementations in _utils (which is what pins their tie/NA semantics)
# --------------------------------------------------------------------------- #
def _label_cases(rng, n=200, k=13):
    labels = [f"T{i}" for i in range(9)]
    with_na = np.asarray(labels + [None], dtype=object)
    cases = {"random with missing": rng.choice(with_na, size=(n, k))}

    unanimous = rng.choice(np.asarray(labels, dtype=object), size=(n, k))
    unanimous[:, 1:] = labels[0]
    cases["unanimous"] = unanimous

    all_missing = rng.choice(np.asarray(labels, dtype=object), size=(n, k))
    all_missing[:, 1:] = None
    cases["all neighbours missing"] = all_missing

    tie = rng.choice(np.asarray(labels, dtype=object), size=(n, k))
    tie[:, 1:] = np.tile(np.asarray([labels[3], labels[1]], dtype=object),
                         (n, (k - 1) // 2))
    cases["exact two-way tie"] = tie

    cases["real labels tie with missing"] = rng.choice(
        np.asarray(labels[:2] + [None], dtype=object), size=(n, k)
    )
    return cases


@pytest.mark.parametrize("case", list(_label_cases(np.random.default_rng(0))))
def test_vectorised_mode_matches_the_reference(case):
    from pysplit._utils import most_frequent
    from pysplit.neighborhood import _neighbour_mode

    matrix = _label_cases(np.random.default_rng(0))[case]
    expected = [most_frequent(row) for row in matrix[:, 1:]]
    assert list(_neighbour_mode(matrix)) == expected


@pytest.mark.parametrize("case", list(_label_cases(np.random.default_rng(0))))
def test_vectorised_certainty_matches_the_reference(case):
    from pysplit._utils import normalized_certainty
    from pysplit.neighborhood import _neighbour_certainty

    matrix = _label_cases(np.random.default_rng(0))[case]
    expected = np.array(
        [normalized_certainty(row) for row in matrix[:, 1:]], dtype=float
    )
    assert np.allclose(_neighbour_certainty(matrix), expected, equal_nan=True)
