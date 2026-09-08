"""Post-processing and conversion of deconvolution results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysplit
from conftest import CELL_TYPES


def _result(results_df, weights, weights_doublet, **kwargs):
    return pysplit.RCTDResult(
        results_df=results_df,
        weights=weights,
        weights_doublet=weights_doublet,
        reference=None,
        **kwargs,
    )


def test_confident_singlets_lose_their_arbitrary_second_type():
    """One cell type above min_weight -> confident singlet, second_type dropped."""
    results_df = pd.DataFrame(
        {
            "spot_class": ["doublet_certain", "doublet_certain"],
            "first_type": ["Tumor", "Tumor"],
            "second_type": ["Tcell", "Tcell"],
        },
        index=["confident", "genuine_doublet"],
    )
    weights = pd.DataFrame(
        [[0.99, 0.01, 0.0, 0.0], [0.60, 0.40, 0.0, 0.0]],
        index=results_df.index,
        columns=CELL_TYPES,
    )
    weights_doublet = pd.DataFrame(
        [[0.99, 0.01], [0.6, 0.4]],
        index=results_df.index,
        columns=["first_type", "second_type"],
    )
    out = pysplit.run_post_process_rctd(
        _result(results_df, weights, weights_doublet), min_weight=0.05, verbose=False
    )
    df = out.results_df
    assert df.loc["confident", "spot_class"] == "singlet"
    assert pd.isna(df.loc["confident", "second_type"])
    assert df.loc["confident", "n_candidates"] == 1
    # the genuine doublet is untouched
    assert df.loc["genuine_doublet", "spot_class"] == "doublet_certain"
    assert df.loc["genuine_doublet", "second_type"] == "Tcell"


def test_cells_without_any_candidate_type_are_rejected():
    results_df = pd.DataFrame(
        {"spot_class": ["singlet"], "first_type": ["Tumor"], "second_type": ["Tcell"]},
        index=["empty"],
    )
    weights = pd.DataFrame([[0.01, 0.01, 0.01, 0.01]], index=["empty"], columns=CELL_TYPES)
    weights_doublet = pd.DataFrame(
        [[0.5, 0.5]], index=["empty"], columns=["first_type", "second_type"]
    )
    out = pysplit.run_post_process_rctd(
        _result(results_df, weights, weights_doublet), min_weight=0.05, verbose=False
    )
    assert out.results_df.loc["empty", "spot_class"] == "reject"
    assert pd.isna(out.results_df.loc["empty", "first_type"])


def test_post_processing_adds_classes_and_coordinates(rctd):
    df = rctd.results_df
    assert rctd.post_processed
    for column in (
        "weight_first_type",
        "weight_second_type",
        "first_type_class",
        "second_type_class",
        "same_class",
        "annot_max_weight",
        "annot_max_doublet_weight",
        "w1_larger_w2",
        "rctd_weights_entropy",
        "x",
        "y",
    ):
        assert column in df.columns, column
    assert rctd.results_df_raw is not None


def test_purify_input_drops_rejects_and_keeps_two_types(rctd):
    converted = pysplit.rctd_to_purify_input(rctd)
    weights = converted["weights"]
    df = rctd.results_df

    rejected = df.index[df["spot_class"].astype(object) == "reject"]
    assert not set(rejected) & set(weights.index)

    # confidently assigned cells keep at most their two assigned types
    confident = df.index[df["spot_class"].astype(object).isin(["singlet", "doublet_certain"])]
    confident = [c for c in confident if c in weights.index]
    n_nonzero = (weights.loc[confident] > 0).sum(axis=1)
    assert n_nonzero.max() <= 2

    # uncertain doublets keep their full decomposition
    uncertain = df.index[df["spot_class"].astype(object) == "doublet_uncertain"]
    uncertain = [c for c in uncertain if c in weights.index]
    if uncertain:
        assert np.allclose(
            weights.loc[uncertain].to_numpy(),
            rctd.weights.loc[uncertain, weights.columns].to_numpy(),
        )


def test_purify_input_weights_match_the_reported_doublet_weights(rctd):
    converted = pysplit.rctd_to_purify_input(rctd)
    weights = converted["weights"]
    df = rctd.results_df
    certain = [
        c
        for c in df.index[df["spot_class"].astype(object) == "doublet_certain"]
        if c in weights.index
    ]
    for cell in certain[:5]:
        row = df.loc[cell]
        assert weights.loc[cell, row["first_type"]] == pytest.approx(
            row["weight_first_type"]
        )
        assert weights.loc[cell, row["second_type"]] == pytest.approx(
            row["weight_second_type"]
        )


def test_purify_requires_post_processing(dataset):
    raw = pysplit.RCTDResult(
        results_df=dataset["results_df"],
        weights=dataset["weights"],
        weights_doublet=dataset["weights_doublet"],
        reference=dataset["reference"],
    )
    with pytest.raises(ValueError, match="run_post_process_rctd"):
        raw.check_post_processed()


def test_entropy_matches_the_analytic_value():
    assert pysplit.entropy([1, 1]) == pytest.approx(np.log(2))
    assert pysplit.entropy([1, 0, 0]) == pytest.approx(0.0)
    assert pysplit.entropy([2, 2, 2, 2]) == pytest.approx(np.log(4))


@pytest.mark.parametrize("as_type", ["dict", "series", "dataframe"])
def test_class_df_accepts_dict_series_or_dataframe(dataset, as_type):
    """rctd-py hands out a dict, the R package a data.frame - accept both."""
    mapping = dict(dataset["class_df"])
    class_df = {
        "dict": mapping,
        "series": pd.Series(mapping),
        "dataframe": pd.DataFrame({"class": pd.Series(mapping)}),
    }[as_type]
    result = pysplit.run_post_process_rctd(
        pysplit.RCTDResult(
            results_df=dataset["results_df"],
            weights=dataset["weights"],
            weights_doublet=dataset["weights_doublet"],
            reference=dataset["reference"],
            class_df=class_df,
        ),
        verbose=False,
    )
    assert isinstance(result.class_df, pd.Series)
    assert "first_type_class" in result.results_df.columns
    assert result.results_df["first_type_class"].notna().any()
