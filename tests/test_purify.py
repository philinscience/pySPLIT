"""Core purification behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import pysplit
from conftest import CELL_TYPES, make_dataset


def test_purification_formula_matches_by_hand():
    """Two cells, three genes: check the exact rescaling factor."""
    reference = pd.DataFrame(
        [[1.0, 0.0, 0.5], [0.0, 2.0, 0.5]],
        index=["A", "B"],
        columns=["g1", "g2", "g3"],
    )
    weights = pd.DataFrame([[0.75, 0.25], [1.0, 0.0]], index=["c1", "c2"], columns=["A", "B"])
    counts = pd.DataFrame(
        [[10.0, 8.0, 4.0], [6.0, 6.0, 6.0]], index=["c1", "c2"], columns=["g1", "g2", "g3"]
    )
    result = pysplit.purify_counts(
        counts,
        weights=weights,
        reference=reference,
        primary_cell_type=pd.Series({"c1": "A", "c2": "A"}),
        verbose=False,
    )
    purified = np.asarray(result["purified_counts"].todense())

    # cell 1, primary A: g1 -> 0.75*1 / (0.75*1) = 1 (kept in full)
    #                    g2 -> A has no g2 at all, so the count is contamination
    #                    g3 -> 0.75*0.5 / (0.75*0.5 + 0.25*0.5) = 0.75
    assert purified[0, 0] == pytest.approx(10.0)
    assert purified[0, 1] == pytest.approx(0.0)
    assert purified[0, 2] == pytest.approx(4.0 * 0.75)
    # cell 2 is a pure singlet: nothing is removed except the orphan gene
    assert purified[1, 0] == pytest.approx(6.0)
    assert purified[1, 1] == pytest.approx(0.0)
    assert purified[1, 2] == pytest.approx(6.0)


def test_purified_counts_never_exceed_raw(dataset):
    result = pysplit.purify_counts(
        dataset["counts"],
        weights=dataset["weights"],
        reference=dataset["reference"],
        primary_cell_type=dataset["results_df"]["first_type"],
        cell_names=dataset["cell_ids"],
        gene_names=dataset["gene_ids"],
        verbose=False,
    )
    difference = (result["counts"] - result["purified_counts"]).todense()
    assert np.all(np.asarray(difference) >= -1e-9)
    assert result["purified_counts"].sum() < result["counts"].sum()


def test_chunking_is_exact(dataset):
    kwargs = dict(
        weights=dataset["weights"],
        reference=dataset["reference"],
        primary_cell_type=dataset["results_df"]["first_type"],
        cell_names=dataset["cell_ids"],
        gene_names=dataset["gene_ids"],
        verbose=False,
    )
    whole = pysplit.purify_counts(dataset["counts"], chunk_size=10_000, **kwargs)
    chunked = pysplit.purify_counts(dataset["counts"], chunk_size=7, **kwargs)
    assert whole["purified_counts"].shape == chunked["purified_counts"].shape
    difference = abs(whole["purified_counts"] - chunked["purified_counts"])
    assert difference.max() == pytest.approx(0.0, abs=1e-12)


def test_singlet_with_full_weight_is_unchanged_except_orphans():
    """A cell whose weight sits entirely on one type keeps all its counts."""
    reference = pd.DataFrame(
        [[1.0, 1.0], [0.0, 1.0]], index=["A", "B"], columns=["g1", "g2"]
    )
    result = pysplit.purify_counts(
        pd.DataFrame([[5.0, 7.0]], index=["c1"], columns=["g1", "g2"]),
        weights=pd.DataFrame([[1.0, 0.0]], index=["c1"], columns=["A", "B"]),
        reference=reference,
        primary_cell_type=pd.Series({"c1": "A"}),
        verbose=False,
    )
    assert np.allclose(np.asarray(result["purified_counts"].todense()), [[5.0, 7.0]])
    assert result["cell_meta"].loc["c1", "purification_status"] == "raw"


def test_reference_orientation_is_detected(dataset):
    kwargs = dict(
        weights=dataset["weights"],
        primary_cell_type=dataset["results_df"]["first_type"],
        cell_names=dataset["cell_ids"],
        gene_names=dataset["gene_ids"],
        verbose=False,
    )
    as_given = pysplit.purify_counts(dataset["counts"], reference=dataset["reference"], **kwargs)
    transposed = pysplit.purify_counts(
        dataset["counts"], reference=dataset["reference"].T, **kwargs
    )
    assert abs(as_given["purified_counts"] - transposed["purified_counts"]).max() < 1e-12


def test_mismatched_cell_types_raise(dataset):
    reference = dataset["reference"].rename(index={"Tumor": "Neoplastic"})
    with pytest.raises(ValueError, match="cell types do not match"):
        pysplit.purify_counts(
            dataset["counts"],
            weights=dataset["weights"],
            reference=reference,
            cell_names=dataset["cell_ids"],
            gene_names=dataset["gene_ids"],
            verbose=False,
        )


def test_cells_to_purify_leaves_others_raw(dataset):
    subset = dataset["cell_ids"][:10]
    result = pysplit.purify_counts(
        dataset["counts"],
        weights=dataset["weights"],
        reference=dataset["reference"],
        primary_cell_type=dataset["results_df"]["first_type"],
        cell_names=dataset["cell_ids"],
        gene_names=dataset["gene_ids"],
        cells_to_purify=subset,
        verbose=False,
    )
    meta = result["cell_meta"]
    # collapsed weights mean a single cell type, i.e. "raw"
    assert (meta.loc[dataset["cell_ids"][10:], "n_cell_types"] == 1).all()
    assert (meta.loc[dataset["cell_ids"][10:], "purification_status"] == "raw").all()


# --------------------------------------------------------------------------- #
# residual contamination removal
# --------------------------------------------------------------------------- #
def test_auto_threshold_is_a_quantile_of_min_over_max():
    reference = pd.DataFrame(
        [[1.0, 1.0, 10.0], [2.0, 5.0, 1.0]], index=["A", "B"], columns=["g1", "g2", "g3"]
    )
    # per-gene min/max ratios: 0.5, 0.2, 0.1
    expected = np.quantile([0.5, 0.2, 0.1], 0.75)
    assert pysplit.auto_belonging_threshold(reference) == pytest.approx(expected)


def test_clean_reference_zeroes_unspecific_entries():
    reference = pd.DataFrame(
        [[10.0, 0.1], [0.2, 8.0]], index=["A", "B"], columns=["g1", "g2"]
    )
    cleaned, threshold, removed = pysplit.clean_reference(
        reference, belonging_threshold=0.5, verbose=False
    )
    assert cleaned.loc["A", "g2"] == 0.0
    assert cleaned.loc["B", "g1"] == 0.0
    assert cleaned.loc["A", "g1"] == 10.0
    assert threshold == 0.5
    assert removed.loc["A"] == pytest.approx(0.1 / 10.1)


def test_residual_removal_removes_more_counts(dataset):
    kwargs = dict(
        weights=dataset["weights"],
        reference=dataset["reference"],
        primary_cell_type=dataset["results_df"]["first_type"],
        cell_names=dataset["cell_ids"],
        gene_names=dataset["gene_ids"],
        verbose=False,
    )
    standard = pysplit.purify_counts(dataset["counts"], **kwargs)
    aggressive = pysplit.purify_counts(
        dataset["counts"], remove_residual_contamination=True, **kwargs
    )
    assert aggressive["purified_counts"].sum() < standard["purified_counts"].sum()

    # Every gene the cleaning drops from a cell's primary profile is removed
    # entirely. (Genes that survive may *keep* more than under standard SPLIT:
    # the cleaned reference shrinks the denominator as well, so the retained
    # fraction of a specific marker goes up - that is the point of the step.)
    cleaned = aggressive["reference"]
    purified = aggressive["purified_counts"].todense()
    primary = aggressive["cell_meta"]["primary_cell_type"].to_numpy()
    for i, cell_type in enumerate(primary):
        dropped = np.asarray(cleaned.loc[cell_type].to_numpy() == 0)
        assert np.all(np.asarray(purified)[i, dropped] == 0)
