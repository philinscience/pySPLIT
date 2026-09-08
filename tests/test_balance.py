"""Balancing raw and purified data, SPLIT-shift, splitting and reassignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import pysplit


@pytest.fixture
def workflow(adata, rctd):
    """Raw AnnData carrying spatial metrics, plus its purified counterpart."""
    purified = pysplit.purify(adata, rctd=rctd, verbose=False)
    adata.obs = adata.obs.join(rctd.results_df)
    sp_nw = pysplit.build_spatial_network(adata, k_knn=8, prune=True, radius=30.0)
    pysplit.spatial_metrics(sp_nw, rctd)
    adata.obs = adata.obs.join(sp_nw.to_dataframe())
    return adata, purified, rctd, sp_nw


def test_purify_returns_anndata_aligned_to_the_input(adata, rctd):
    purified = pysplit.purify(adata, rctd=rctd, verbose=False)
    assert purified.n_obs <= adata.n_obs  # rejects are dropped
    assert list(purified.var_names) == list(adata.var_names)
    assert "purification_status" in purified.obs
    assert "first_type" in purified.obs
    assert "counts_raw" in purified.layers
    assert "spatial" in purified.obsm
    assert purified.uns["pysplit"]["n_cells"] == purified.n_obs
    # rejects really are gone
    rejected = rctd.results_df.index[
        rctd.results_df["spot_class"].astype(object) == "reject"
    ]
    assert not set(rejected) & set(purified.obs_names)


def test_purify_agnostic_interface_matches_the_rctd_interface(adata, rctd):
    from_rctd = pysplit.purify(adata, rctd=rctd, verbose=False)
    converted = pysplit.rctd_to_purify_input(rctd)
    agnostic = pysplit.purify(
        adata,
        weights=converted["weights"],
        reference=converted["reference"],
        primary_cell_type=converted["primary_cell_type"],
        verbose=False,
    )
    assert list(agnostic.obs_names) == list(from_rctd.obs_names)
    assert abs(agnostic.X - from_rctd.X).max() == pytest.approx(0.0, abs=1e-12)


def test_balance_by_score_mixes_raw_and_purified(workflow):
    raw, purified, _, _ = workflow
    balanced = pysplit.balance_by_score(raw, purified, threshold=0.05)
    statuses = set(balanced.obs["purification_status"])
    assert statuses <= {"raw", "purified"}
    assert not balanced.obs["swap"].any()
    # rejects are dropped
    assert not (balanced.obs["spot_class"].astype(object) == "reject").any()

    # a stricter threshold purifies fewer cells
    strict = pysplit.balance_by_score(raw, purified, threshold=0.9)
    n_purified = lambda obj: int((obj.obs["purification_status"] == "purified").sum())
    assert n_purified(strict) <= n_purified(balanced)


def test_balance_by_score_takes_raw_counts_for_unpurified_cells(workflow):
    raw, purified, _, _ = workflow
    balanced = pysplit.balance_by_score(raw, purified, threshold=0.05)
    kept_raw = balanced.obs_names[balanced.obs["purification_status"] == "raw"]
    if len(kept_raw):
        from_balanced = balanced[kept_raw, raw.var_names].X
        from_raw = raw[kept_raw, raw.var_names].X
        assert abs(from_balanced - from_raw).max() == pytest.approx(0.0, abs=1e-9)


def test_balance_by_spot_class_keeps_singlets_raw(workflow):
    raw, purified, _, _ = workflow
    balanced = pysplit.balance_by_spot_class(raw, purified)
    singlets = raw.obs_names[raw.obs["spot_class"].astype(object) == "singlet"]
    singlets = [c for c in singlets if c in set(balanced.obs_names)]
    assert (balanced.obs.loc[singlets, "purification_status"] == "raw").all()


def test_split_shift_swaps_labels_and_uses_the_residual(workflow):
    raw, purified, rctd, _ = workflow
    raw.obsm["X_pca"] = np.asarray(raw.X.todense())[:, :5]
    tr_nw = pysplit.build_transcriptomics_network(raw, dims=range(5), k_knn=10)
    pysplit.transcriptomics_metrics(tr_nw, rctd)
    raw.obs = raw.obs.join(tr_nw.to_dataframe())

    shifted = pysplit.balance_by_score(raw, purified, threshold=0.05, swap_labels=True)
    assert "swap" in shifted.obs
    assert "first_type_before_swap" in shifted.obs

    swapped = shifted.obs_names[shifted.obs["swap"].to_numpy(dtype=bool)]
    for cell in swapped:
        row = shifted.obs.loc[cell]
        assert row["first_type"] == row["second_type_before_swap"]
        assert row["second_type"] == row["first_type_before_swap"]

    # a swapped, purified cell carries the residual, not the purified profile
    swapped_purified = [
        c for c in swapped if shifted.obs.loc[c, "purification_status"] == "purified"
    ]
    for cell in swapped_purified[:3]:
        expected = (
            raw[cell, shifted.var_names].X - purified[cell, shifted.var_names].X
        ).todense()
        expected = np.clip(np.asarray(expected), 0, None)
        actual = np.asarray(shifted[cell].X.todense())
        assert np.allclose(actual, expected, atol=1e-9)


def test_shift_labels_on_an_explicit_cell_list():
    obs = pd.DataFrame(
        {
            "first_type": ["Tcell", "Tumor"],
            "second_type": ["Tumor", "Tcell"],
            "weight_first_type": [0.7, 0.8],
            "weight_second_type": [0.3, 0.2],
        },
        index=["a", "b"],
    )
    shifted, mask = pysplit.shift_labels(obs, cells_to_swap=["a"])
    assert mask.tolist() == [True, False]
    assert shifted.loc["a", "first_type"] == "Tumor"
    assert shifted.loc["a", "weight_first_type"] == 0.3
    assert shifted.loc["b", "first_type"] == "Tumor"  # untouched


def test_split_cells_conserves_total_counts(adata, rctd):
    purified = pysplit.purify(adata, rctd=rctd, verbose=False)
    split = pysplit.split_cells(adata, purified)
    assert split.n_obs == 2 * purified.n_obs
    assert set(split.obs["decomposition_order"]) == {"first", "second"}

    raw_total = float(adata[purified.obs_names, purified.var_names].X.sum())
    assert float(split.X.sum()) == pytest.approx(raw_total, rel=1e-9)

    first = split.obs_names[split.obs["decomposition_order"] == "first"]
    assert (split.obs.loc[first, "cell_type"] == split.obs.loc[first, "first_type"]).all()


def test_balance_split_keeps_raw_cells_whole(workflow):
    raw, purified, _, _ = workflow
    combined = pysplit.balance_split(raw, purified)
    assert set(combined.obs["decomposition_order"]) <= {"raw", "first", "second"}
    # uncertain doublets contribute only their first half by default
    second = combined.obs[combined.obs["decomposition_order"] == "second"]
    origin = raw.obs.loc[second["cell_id"], "spot_class"].astype(object)
    assert not (origin == "doublet_uncertain").any()


# --------------------------------------------------------------------------- #
# residual reassignment
# --------------------------------------------------------------------------- #
def test_reassignment_operator_rows_sum_to_one(workflow):
    raw, purified, _, sp_nw = workflow
    donors = purified.obs_names[purified.obs["purification_status"] == "purified"]
    operator = pysplit.build_reassignment_operator(sp_nw, cells_with_residual=donors)
    row_sums = np.asarray(operator.sum(axis=1)).ravel()
    # a donor either distributes its whole residual or has no recipient at all
    assert np.all(np.isclose(row_sums, 1.0) | np.isclose(row_sums, 0.0))


def test_reassignment_moves_counts_without_creating_them(workflow):
    raw, purified, _, sp_nw = workflow
    cells = list(purified.obs_names)
    genes = list(purified.var_names)
    raw_matrix = raw[cells, genes].X
    corrected = pysplit.reassign_residual_counts(
        raw_matrix,
        purified.X,
        neighborhood=sp_nw,
        purification_status=purified.obs["purification_status"],
        cell_names=cells,
        mode="uniform",
    )
    # counts are only moved around, never invented
    assert float(corrected.sum()) >= float(purified.X.sum()) - 1e-9
    assert float(corrected.sum()) <= float(raw_matrix.sum()) + 1e-9
    # per gene, nothing is created either
    assert np.all(
        np.asarray(corrected.sum(axis=0)) <= np.asarray(raw_matrix.sum(axis=0)) + 1e-9
    )


def test_reassignment_self_keep_retains_a_share(workflow):
    raw, purified, _, sp_nw = workflow
    donors = purified.obs_names[purified.obs["purification_status"] == "purified"]
    operator = pysplit.build_reassignment_operator(
        sp_nw, cells_with_residual=donors, self_keep=0.25
    )
    diagonal = operator.diagonal()
    has_recipients = np.asarray(operator.sum(axis=1)).ravel() > 0
    assert np.all(diagonal[has_recipients] == pytest.approx(0.25))


def test_count_proportional_mode_favours_deeper_neighbours(workflow):
    raw, purified, _, sp_nw = workflow
    donors = purified.obs_names[purified.obs["purification_status"] == "purified"]
    ncount = pd.Series(
        np.asarray(raw.X.sum(axis=1)).ravel(), index=list(raw.obs_names)
    )
    operator = pysplit.build_reassignment_operator(
        sp_nw, cells_with_residual=donors, ncount=ncount
    )
    row_sums = np.asarray(operator.sum(axis=1)).ravel()
    assert np.all(np.isclose(row_sums, 1.0) | np.isclose(row_sums, 0.0))

    # within one donor row, a deeper recipient receives at least as much
    for i in np.flatnonzero(np.isclose(row_sums, 1.0))[:5]:
        row = operator.getrow(i).tocoo()
        if row.nnz < 2:
            continue
        depths = ncount.iloc[row.col].to_numpy()
        order = np.argsort(depths)
        assert np.all(np.diff(row.data[order]) >= -1e-12)


def test_bad_mode_is_rejected(workflow):
    raw, purified, _, sp_nw = workflow
    with pytest.raises(ValueError, match="uniform"):
        pysplit.reassign_residual_counts(
            raw[purified.obs_names, purified.var_names].X,
            purified.X,
            neighborhood=sp_nw,
            purification_status=purified.obs["purification_status"],
            cell_names=list(purified.obs_names),
            mode="nonsense",
        )


def test_purify_accepts_bare_matrices_and_dataframes(dataset, rctd):
    """Without AnnData you get the raw dict back; labels come from either source."""
    converted = pysplit.rctd_to_purify_input(rctd)
    dense = np.asarray(dataset["counts"].todense())

    from_matrix = pysplit.purify(
        dense,
        weights=converted["weights"],
        reference=converted["reference"],
        primary_cell_type=converted["primary_cell_type"],
        cell_names=dataset["cell_ids"],
        gene_names=dataset["gene_ids"],
        verbose=False,
    )
    assert isinstance(from_matrix, dict)

    frame = pd.DataFrame(dense, index=dataset["cell_ids"], columns=dataset["gene_ids"])
    from_frame = pysplit.purify(frame, rctd=rctd, verbose=False)
    assert abs(
        from_matrix["purified_counts"] - from_frame["purified_counts"]
    ).max() == pytest.approx(0.0, abs=1e-12)


def test_purify_rejects_conflicting_inputs(adata, rctd):
    converted = pysplit.rctd_to_purify_input(rctd)
    with pytest.raises(ValueError, match="not both"):
        pysplit.purify(adata, rctd=rctd, weights=converted["weights"], verbose=False)
    with pytest.raises(ValueError, match="both `weights` and `reference`"):
        pysplit.purify(adata, weights=converted["weights"], verbose=False)
