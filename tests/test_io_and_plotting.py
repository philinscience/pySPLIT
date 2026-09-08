"""rctd-py interoperability and the pie visualisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysplit
from conftest import CELL_TYPES, make_dataset

pa = pytest.importorskip("pyarrow")
h5py = pytest.importorskip("h5py")


def _write_rctdpy_dir(tmp_path, dataset):
    """Write a synthetic dataset in the layout rctd-py produces."""
    cell_ids = dataset["cell_ids"]
    df = dataset["results_df"]
    label_to_int = {"reject": 0, "singlet": 1, "doublet_certain": 2,
                    "doublet_uncertain": 3}

    pd.DataFrame({"cell_id": cell_ids}).to_parquet(tmp_path / "cell_ids.parquet")
    dataset["weights"].reset_index(drop=True).to_parquet(tmp_path / "weights.parquet")
    pd.DataFrame(
        {
            "w_1": dataset["weights_doublet"]["first_type"].to_numpy(),
            "w_2": dataset["weights_doublet"]["second_type"].to_numpy(),
        }
    ).to_parquet(tmp_path / "weights_doublet.parquet")
    pd.DataFrame(
        {
            "spot_class": [label_to_int[s] for s in df["spot_class"]],
            "first_type_name": df["first_type"].to_numpy(),
            "second_type_name": df["second_type"].to_numpy(),
            "first_class": np.zeros(len(df), dtype=bool),
            "second_class": np.zeros(len(df), dtype=bool),
            "min_score": df["min_score"].to_numpy(),
            "singlet_score": df["singlet_score"].to_numpy(),
        }
    ).to_parquet(tmp_path / "spot_results.parquet")
    pd.DataFrame({"cell_type_names": CELL_TYPES}).to_parquet(tmp_path / "metadata.parquet")
    pd.DataFrame({"cell_id": cell_ids}).to_parquet(tmp_path / "pixel_mask.parquet")

    reference = dataset["reference"]
    with h5py.File(tmp_path / "reference_profiles.h5", "w") as handle:
        handle["profiles"] = reference.to_numpy()  # cell types x genes
        handle["gene_names"] = np.array(
            [g.encode() for g in reference.columns], dtype="S32"
        )
        handle["cell_type_names"] = np.array(
            [t.encode() for t in reference.index], dtype="S32"
        )
    return tmp_path


def test_read_rctdpy_roundtrip(tmp_path, dataset):
    save_dir = _write_rctdpy_dir(tmp_path, dataset)
    rctd = pysplit.read_rctdpy(
        save_dir, class_df=dataset["class_df"], coords=dataset["coords"]
    )
    assert rctd.post_processed
    assert rctd.cell_types == CELL_TYPES
    assert list(rctd.results_df.index) == dataset["cell_ids"]
    assert np.allclose(
        rctd.reference.to_numpy(), dataset["reference"].to_numpy()
    )
    assert "first_type_class" in rctd.results_df.columns
    assert set(rctd.results_df["spot_class"].astype(object).dropna()) <= set(
        pysplit.SPOT_CLASS_LEVELS
    )
    assert {"x", "y"} <= set(rctd.results_df.columns)


def test_read_rctdpy_output_purifies(tmp_path, dataset, adata):
    rctd = pysplit.read_rctdpy(
        _write_rctdpy_dir(tmp_path, dataset), class_df=dataset["class_df"]
    )
    purified = pysplit.purify(adata, rctd=rctd, verbose=False)
    assert purified.n_obs > 0
    assert float(purified.X.sum()) < float(adata.X.sum())


def test_read_rctdpy_rejects_unknown_spot_class(tmp_path, dataset):
    save_dir = _write_rctdpy_dir(tmp_path, dataset)
    spot_results = pd.read_parquet(save_dir / "spot_results.parquet")
    spot_results.loc[0, "spot_class"] = 99
    spot_results.to_parquet(save_dir / "spot_results.parquet")
    with pytest.raises(ValueError, match="Unmapped spot_class"):
        pysplit.read_rctdpy(save_dir)


# --------------------------------------------------------------------------- #
def test_pie_dataframe_weights_sum_to_one_per_cell(rctd):
    plt = pytest.importorskip("matplotlib")
    pie_df = pysplit.pie_dataframe(rctd)
    shares = pie_df[rctd.cell_types].to_numpy()
    # a cell either has an assignment (weights sum to 1) or none at all
    totals = shares.sum(axis=1)
    assert np.all(np.isclose(totals, 1.0) | np.isclose(totals, 0.0))
    assert {"x", "y", "cell_id"} <= set(pie_df.columns)


def test_plot_pie_draws_one_patch_group_per_cell(rctd):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pie_df = pysplit.pie_dataframe(rctd)
    cropped = pysplit.crop_pie_dataframe(pie_df, cell_id=pie_df.index[5], radius=40)
    ax = pysplit.plot_pie(cropped, legend=True)
    assert len(ax.patches) >= len(cropped)
    assert ax.get_aspect() == 1.0


def test_plot_pie_around_cell_marks_the_focal_cell(rctd):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pie_df = pysplit.pie_dataframe(rctd)
    cell = pie_df.index[10]
    ax = pysplit.plot_pie_around_cell(pie_df, cell_id=cell, radius=50)
    stars = [line for line in ax.lines if line.get_marker() == "*"]
    assert len(stars) == 1
    assert stars[0].get_xdata()[0] == pytest.approx(float(pie_df.loc[cell, "x"]))


# --------------------------------------------------------------------------- #
# in-memory rctd-py bridge
# --------------------------------------------------------------------------- #
class _FakeDoubletResult:
    """Mimics rctd-py's DoubletResult: integer codes, no name arrays."""

    def __init__(self, dataset, pixel_mask=None):
        df = dataset["results_df"]
        self.cell_type_names = CELL_TYPES
        label_to_int = {"reject": 0, "singlet": 1, "doublet_certain": 2,
                        "doublet_uncertain": 3}
        type_pos = {t: i for i, t in enumerate(CELL_TYPES)}
        keep = slice(None) if pixel_mask is None else np.flatnonzero(pixel_mask)
        self.spot_class = np.array(
            [label_to_int[s] for s in df["spot_class"]]
        )[keep]
        self.first_type = np.array([type_pos[t] for t in df["first_type"]])[keep]
        self.second_type = np.array(
            [type_pos.get(t, -1) for t in df["second_type"]]
        )[keep]
        self.weights = dataset["weights"].to_numpy()[keep]
        self.weights_doublet = dataset["weights_doublet"].to_numpy()[keep]
        self.min_score = df["min_score"].to_numpy()[keep]
        self.singlet_score = df["singlet_score"].to_numpy()[keep]
        self.pixel_mask = pixel_mask


def test_from_rctd_py_maps_integer_codes_to_labels(dataset):
    result = pysplit.from_rctd_py(
        _FakeDoubletResult(dataset),
        reference=dataset["reference"],
        cell_ids=dataset["cell_ids"],
        class_df=dataset["class_df"],
        coords=dataset["coords"],
    )
    assert result.post_processed
    assert result.cell_types == CELL_TYPES
    assert list(result.results_df.index) == dataset["cell_ids"]
    assert set(result.results_df["spot_class"].astype(object).dropna()) <= set(
        pysplit.SPOT_CLASS_LEVELS
    )
    assert "first_type_class" in result.results_df.columns
    # labels round-trip for the cells post-processing did not relabel
    untouched = result.results_df["n_candidates"] > 1
    original = dataset["results_df"].loc[untouched[untouched].index, "first_type"]
    assert (result.results_df.loc[untouched, "first_type"] == original).all()


def test_from_rctd_py_applies_the_pixel_mask(dataset):
    """RCTD drops cells failing its UMI filters; the ids must follow."""
    mask = np.ones(len(dataset["cell_ids"]), dtype=bool)
    mask[:5] = False
    result = pysplit.from_rctd_py(
        _FakeDoubletResult(dataset, pixel_mask=mask),
        reference=dataset["reference"],
        cell_ids=dataset["cell_ids"],
    )
    assert list(result.results_df.index) == dataset["cell_ids"][5:]


def test_from_rctd_py_rejects_mismatched_cell_ids(dataset):
    with pytest.raises(ValueError, match="cell id"):
        pysplit.from_rctd_py(
            _FakeDoubletResult(dataset),
            reference=dataset["reference"],
            cell_ids=dataset["cell_ids"][:10],
        )


def test_from_rctd_py_output_purifies(dataset, adata):
    result = pysplit.from_rctd_py(
        _FakeDoubletResult(dataset),
        reference=dataset["reference"],
        cell_ids=dataset["cell_ids"],
        class_df=dataset["class_df"],
    )
    purified = pysplit.purify(adata, rctd=result, verbose=False)
    assert purified.n_obs > 0
    assert float(purified.X.sum()) < float(adata.X.sum())


def test_spot_class_map_matches_rctd_py():
    """Our integer->label map must stay in step with rctd-py's own ordering."""
    rctd_mod = pytest.importorskip("rctd")
    from pysplit.rctd import RCTDPY_SPOT_CLASS_MAP

    assert list(rctd_mod.SPOT_CLASS_NAMES) == [
        RCTDPY_SPOT_CLASS_MAP[i] for i in range(len(rctd_mod.SPOT_CLASS_NAMES))
    ]


def test_reference_object_axes_are_named_from_the_object():
    """An unlabelled rctd.Reference-style array must get its axes from the object."""
    from pysplit.rctd import _reference_from_rctd_py

    class _FakeReference:
        gene_names = ["g1", "g2", "g3"]
        cell_type_names = ["A", "B"]
        profiles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # genes x types

    frame = _reference_from_rctd_py(_FakeReference(), ["A", "B"])
    assert list(frame.index) == ["A", "B"]          # cell types on rows
    assert list(frame.columns) == ["g1", "g2", "g3"]
    assert frame.loc["B", "g3"] == 6.0
