"""End-to-end pySPLIT tutorial on real 10x Xenium breast cancer data.

Reproduces the workflow of the original SPLIT Xenium vignette, in Python:

    1. download the public Xenium (spatial) and Chromium (reference) data
    2. deconvolve with rctd-py in doublet mode
    3. purify with SPLIT  (+ residual contamination removal)
    4. spatially-aware SPLIT - purify only cells with local diffusion evidence
    5. SPLIT-shift - swap primary/secondary labels where the transcriptomic
       neighbourhood disagrees
    6. quantify the result with marker-gene specificity

Data
----
Janesick, A., Shelansky, R., Gottscho, A.D. et al. "High resolution mapping of
the tumor microenvironment using integrated single-cell, spatial and in situ
analysis." Nature Communications 14, 8353 (2023).

Requirements
------------
    pip install "pysplit-st[all]" rctd-py scanpy openpyxl

Roughly 200 MB of downloads plus a ~400 MB Q-matrix cache for rctd-py, and a
few GB of RAM. Every stage caches to ``--outdir``, so re-runs are cheap and an
interrupted run resumes where it stopped.

Usage
-----
    python tutorial_xenium_breast_cancer.py                 # cropped region, ~10 min
    python tutorial_xenium_breast_cancer.py --full-section  # all 167k cells, slow
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 0. configuration
# --------------------------------------------------------------------------- #
XENIUM_BASE = (
    "https://cf.10xgenomics.com/samples/xenium/1.0.1/"
    "Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1"
)
CHROMIUM_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/7.0.1/"
    "Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer/"
    "Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer"
    "_count_sample_filtered_feature_bc_matrix.h5"
)
ANNOTATION_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41467-023-43458-x/MediaObjects/"
    "41467_2023_43458_MOESM4_ESM.xlsx"
)

# Higher-level classes. Passing these to RCTD is strongly recommended: it makes
# the deconvolution more robust and sharply reduces the number of rejected
# cells, which SPLIT would otherwise have to discard.
CELL_TYPE_TO_CLASS = {
    "B_Cells": "B cell",
    "CD4+_T_Cells": "T cell",
    "CD8+_T_Cells": "T cell",
    "IRF7+_DCs": "Myeloid",
    "LAMP3+_DCs": "Myeloid",
    "Macrophages_1": "Myeloid",
    "Macrophages_2": "Myeloid",
    "Mast_Cells": "Myeloid",
    "DCIS 1": "Epithelial",
    "DCIS 2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Myoepithelial",
    "Myoepi_KRT15+": "Myoepithelial",
    "Stromal": "Stromal",
    "Perivascular-Like": "Stromal",
    "Endothelial": "Endothelial",
}

# Marker genes used only to *score* the result, never to compute it.
MARKERS = {
    "T cell": ["CD3D", "CD3E", "CD2", "IL7R", "CD247"],
    "B cell": ["CD79A", "MS4A1", "BANK1", "CD19"],
    "Myeloid": ["ITGAX", "CD14", "CD68", "LYZ", "C1QA"],
    "Epithelial": ["EPCAM", "KRT8", "ERBB2", "CEACAM6", "KRT7"],
    "Myoepithelial": ["ACTA2", "KRT15", "MYLK", "OXTR"],
    "Endothelial": ["PECAM1", "VWF", "EGFL7", "CLDN5"],
    "Stromal": ["POSTN", "LUM", "FBLN1", "PDGFRB"],
}


def log(message: str) -> None:
    print(f"\n\033[1m>>> {message}\033[0m", flush=True)


def download(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` unless it is already there.

    Uses curl rather than urllib: on managed clusters the Python SSL trust
    store is often incomplete while curl's is fine.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name}")
        return dest
    print(f"  fetching {dest.name} ...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sSfL", "-o", str(dest), url], check=True
    )
    return dest


# --------------------------------------------------------------------------- #
# 1. data
# --------------------------------------------------------------------------- #
def load_data(outdir: Path, full_section: bool):
    """Download and cache the Xenium sample and the annotated Chromium reference."""
    import scanpy as sc

    raw = outdir / "raw"
    xenium_cache = outdir / ("xenium_full.h5ad" if full_section else "xenium_crop.h5ad")
    reference_cache = outdir / "chromium_ref.h5ad"

    if xenium_cache.exists() and reference_cache.exists():
        log("Loading cached AnnData objects")
        return sc.read_h5ad(xenium_cache), sc.read_h5ad(reference_cache)

    log("Downloading public 10x data (~200 MB, cached afterwards)")
    matrix_tar = download(f"{XENIUM_BASE}_cell_feature_matrix.tar.gz",
                          raw / "xenium_cell_feature_matrix.tar")
    cells_csv = download(f"{XENIUM_BASE}_cells.csv.gz", raw / "xenium_cells.csv.gz")
    chromium_h5 = download(CHROMIUM_URL, raw / "chromium.h5")
    annotation_xlsx = download(ANNOTATION_URL, raw / "chromium_annotation.xlsx")

    matrix_dir = raw / "cell_feature_matrix"
    if not matrix_dir.exists():
        # 10x names this ".tar.gz" but ships it uncompressed
        with tarfile.open(matrix_tar) as archive:
            archive.extractall(raw)

    log("Building the Xenium AnnData")
    xenium = sc.read_10x_mtx(matrix_dir)
    xenium.obs_names = xenium.obs_names.astype(str)
    cells = pd.read_csv(cells_csv).set_index("cell_id")
    cells.index = cells.index.astype(str)
    cells = cells.loc[xenium.obs_names]
    xenium.obs["x"] = cells["x_centroid"].to_numpy()
    xenium.obs["y"] = cells["y_centroid"].to_numpy()
    xenium.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].to_numpy()
    print(f"  full section: {xenium.n_obs:,} cells x {xenium.n_vars} genes")

    if not full_section:
        # Crop rather than subsample: SPLIT's spatial mode needs intact
        # neighbourhoods, which random subsampling destroys.
        keep = (xenium.obs["x"] > 6000) & (xenium.obs["y"] > 4000)
        xenium = xenium[keep.to_numpy()].copy()
        print(f"  cropped to x>6000, y>4000: {xenium.n_obs:,} cells")

    sc.pp.filter_cells(xenium, min_counts=10)
    print(f"  after nCount >= 10 filter: {xenium.n_obs:,} cells")
    xenium.write(xenium_cache)

    log("Building the Chromium reference")
    reference = sc.read_10x_h5(chromium_h5)
    reference.var_names_make_unique()
    annotation = pd.read_excel(annotation_xlsx, sheet_name=0).set_index("Barcode")
    reference.obs["cell_type"] = (
        annotation["Annotation"].reindex(reference.obs_names).to_numpy()
    )
    # Drop unannotated cells and the study's own hybrid (doublet) calls: a
    # reference containing doublets teaches the deconvolution that mixed
    # profiles are legitimate cell types, which is exactly what SPLIT undoes.
    keep = reference.obs["cell_type"].notna() & ~reference.obs[
        "cell_type"
    ].str.contains("Hybrid", case=False, na=True)
    reference = reference[keep.to_numpy()].copy()
    print(f"  reference: {reference.n_obs:,} cells, "
          f"{reference.obs.cell_type.nunique()} cell types")
    reference.write(reference_cache)
    return xenium, reference


# --------------------------------------------------------------------------- #
# 2. deconvolution
# --------------------------------------------------------------------------- #
def deconvolve(xenium, reference, outdir: Path):
    """Run rctd-py in doublet mode and wrap the result for pySPLIT."""
    import pickle

    import pysplit

    cache = outdir / "rctd_doublet.pkl"
    # NOTE: rctd-py wants class_df as a plain {cell_type: class} dict, unlike
    # the R package which takes a data.frame. pySPLIT accepts either.
    class_df = dict(CELL_TYPE_TO_CLASS)

    if cache.exists():
        log("Loading cached RCTD result")
        with open(cache, "rb") as handle:
            payload = pickle.load(handle)
    else:
        try:
            import rctd
        except ImportError:
            sys.exit(
                "rctd-py is required for this step:  pip install rctd-py\n"
                "(Or supply your own deconvolution: SPLIT only needs a "
                "cells x cell-types weight matrix and a reference.)"
            )
        log("Running RCTD in doublet mode (this is the slow step)")
        # rctd-py auto-downloads a ~400 MB Q-matrix cache to ~/.cache/rctd on
        # first use. If its urllib call fails on SSL, fetch it by hand:
        #   curl -L -o ~/.cache/rctd/q_matrices.npz \
        #     https://github.com/p-gueguen/rctd-py/releases/download/v0.1.1/q_matrices.npz
        started = time.time()
        rctd_reference = rctd.Reference(
            reference, cell_type_col="cell_type", min_UMI=10, cell_min=25
        )
        # rctd-py sets compile=True, which hands the solver to torch.compile.
        # That pays off on a GPU but costs many minutes of one-time
        # torchinductor compilation on CPU, so switch it off when there is no
        # GPU to compile for.
        import torch

        on_gpu = torch.cuda.is_available()
        print(f"  device: {'cuda' if on_gpu else 'cpu'}, compile={on_gpu}")
        config = rctd.RCTDConfig(
            UMI_min=10,
            counts_MIN=10,
            UMI_min_sigma=100,
            class_df=class_df,
            compile=on_gpu,
        )
        result = rctd.run_rctd(xenium, rctd_reference, mode="doublet", config=config)
        print(f"  RCTD finished in {time.time() - started:.0f}s")
        # get_profiles_for_genes() silently drops genes absent from the
        # reference, so intersect first or the frame axes will not line up.
        reference_genes = {str(g) for g in rctd_reference.gene_names}
        genes = [str(g) for g in xenium.var_names if str(g) in reference_genes]
        missing = xenium.n_vars - len(genes)
        if missing:
            print(f"  {missing} panel gene(s) absent from the reference, dropped")
        payload = {
            "spot_class": np.asarray(result.spot_class),
            "first_type": np.asarray(result.first_type),
            "second_type": np.asarray(result.second_type),
            "weights": np.asarray(result.weights, dtype=float),
            "weights_doublet": np.asarray(result.weights_doublet, dtype=float),
            "cell_type_names": [str(t) for t in result.cell_type_names],
            "pixel_mask": np.asarray(result.pixel_mask),
            "min_score": np.asarray(result.min_score, dtype=float),
            "singlet_score": np.asarray(result.singlet_score, dtype=float),
            # mean profiles restricted to the panel, as cell_types x genes
            "reference": pd.DataFrame(
                np.asarray(rctd_reference.get_profiles_for_genes(genes)).T,
                index=[str(t) for t in result.cell_type_names],
                columns=genes,
            ),
            "cell_ids": list(map(str, xenium.obs_names)),
        }
        with open(cache, "wb") as handle:
            pickle.dump(payload, handle)

    log("Post-processing the deconvolution for SPLIT")
    rctd_result = pysplit.from_rctd_py(
        _SimpleResult(payload),
        reference=payload["reference"],
        cell_ids=payload["cell_ids"],
        class_df=class_df,
        coords=xenium.obs[["x", "y"]],
    )
    counts = rctd_result.results_df["spot_class"].value_counts(normalize=True) * 100
    print("  spot class distribution (%):")
    print(counts.round(1).to_string())
    return rctd_result


class _SimpleResult:
    """Adapter so a cached dict looks like an rctd-py DoubletResult."""

    def __init__(self, payload: dict) -> None:
        for key, value in payload.items():
            setattr(self, key, value)


# --------------------------------------------------------------------------- #
# 3-5. purification
# --------------------------------------------------------------------------- #
def purify_all(xenium, rctd_result):
    """Run the three SPLIT modes and return them for comparison."""
    import scanpy as sc

    import pysplit

    log("SPLIT purification")
    purified = pysplit.purify(
        xenium, rctd=rctd_result, remove_residual_contamination=True
    )
    kept = 100 * float(purified.X.sum()) / float(xenium.X.sum())
    print(f"  {purified.n_obs:,} cells retained, {kept:.1f}% of counts kept")

    # Deconvolution results and metrics must live on the raw object, since that
    # is what the balancing functions read.
    xenium = xenium[purified.obs_names].copy()
    xenium.obs = xenium.obs.join(rctd_result.results_df, rsuffix="_rctd")

    log("Spatial neighbourhood - local diffusion of the secondary cell type")
    spatial_nw = pysplit.build_spatial_network(
        xenium, basis="spatial", k_knn=20, prune=True, radius=15  # micrometres
    )
    pysplit.spatial_metrics(spatial_nw, rctd_result)
    xenium.obs = xenium.obs.join(spatial_nw.to_dataframe())
    score = xenium.obs["neighborhood_weights_second_type"]
    print(f"  diffusion score: median {score.median():.3f}, "
          f"{100 * (score > 0.05).mean():.0f}% of cells above 0.05")
    by_class = xenium.obs.groupby("spot_class", observed=True)[
        "neighborhood_weights_second_type"
    ].median()
    print("  median diffusion score by spot class:")
    print(by_class.round(3).to_string())

    log("Spatially-aware SPLIT - purify only cells with local evidence")
    spatially_aware = pysplit.balance_by_score(
        xenium, purified, threshold=0.05,
        score_name="neighborhood_weights_second_type",
    )
    n_purified = int((spatially_aware.obs["purification_status"] == "purified").sum())
    print(f"  {n_purified:,}/{spatially_aware.n_obs:,} cells purified "
          f"({100 * n_purified / spatially_aware.n_obs:.0f}%), the rest kept raw")

    log("Transcriptomic neighbourhood - for SPLIT-shift")
    normalised = xenium.copy()
    sc.pp.normalize_total(normalised)
    sc.pp.log1p(normalised)
    sc.pp.pca(normalised, n_comps=50)
    xenium.obsm["X_pca"] = normalised.obsm["X_pca"]
    del normalised
    transcriptomic_nw = pysplit.build_transcriptomics_network(
        xenium, basis="X_pca", dims=range(50), k_knn=100
    )
    pysplit.transcriptomics_metrics(transcriptomic_nw, rctd_result)
    xenium.obs = xenium.obs.join(transcriptomic_nw.to_dataframe(), rsuffix="_tr")
    agreement = xenium.obs["first_type_neighborhood_agreement"]
    print(f"  {100 * (agreement == False).mean():.0f}% of cells disagree with "
          "their transcriptomic neighbourhood")

    log("SPLIT-shift - swap primary/secondary labels where warranted")
    shifted = pysplit.balance_by_score(
        xenium, purified, threshold=0.05,
        score_name="neighborhood_weights_second_type", swap_labels=True,
    )
    print(f"  {int(shifted.obs['swap'].sum()):,} cell(s) relabelled")

    return xenium, {
        "Raw": (xenium, "first_type"),
        "SPLIT": (purified, "first_type"),
        "Spatially-aware SPLIT": (spatially_aware, "first_type"),
        "SPLIT-shift": (shifted, "first_type"),
    }


# --------------------------------------------------------------------------- #
# 6. evaluation
# --------------------------------------------------------------------------- #
def marker_specificity(adata, label_column: str) -> tuple[float, float]:
    """Mean share of a cell's counts falling on its own class's markers.

    A purely presentational metric - none of these genes are used by SPLIT.
    Higher is better: it means less of a cell's signal belongs to other
    lineages. Also returns the share landing on *foreign* lineage markers,
    which is what contamination looks like.
    """
    import scipy.sparse as sp

    classes = adata.obs[label_column].map(CELL_TYPE_TO_CLASS)
    available = {
        cls: [g for g in genes if g in adata.var_names]
        for cls, genes in MARKERS.items()
    }
    positions = {
        cls: adata.var_names.get_indexer(genes) for cls, genes in available.items()
    }
    matrix = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    totals = np.asarray(matrix.sum(axis=1)).ravel()

    own, foreign = [], []
    for cls, columns in positions.items():
        rows = np.flatnonzero((classes == cls).to_numpy() & (totals > 0))
        if rows.size == 0 or len(columns) == 0:
            continue
        block = matrix[rows]
        own_counts = np.asarray(block[:, columns].sum(axis=1)).ravel()
        other_columns = np.concatenate(
            [positions[c] for c in positions if c != cls] or [np.array([], int)]
        )
        other_counts = np.asarray(block[:, other_columns].sum(axis=1)).ravel()
        own.append(own_counts / totals[rows])
        foreign.append(other_counts / totals[rows])
    return float(np.mean(np.concatenate(own))), float(
        np.mean(np.concatenate(foreign))
    )


def evaluate(variants: dict) -> pd.DataFrame:
    log("Marker-gene specificity (higher own / lower foreign is better)")
    rows = []
    for name, (adata, column) in variants.items():
        own, foreign = marker_specificity(adata, column)
        rows.append(
            {
                "variant": name,
                "cells": adata.n_obs,
                "own-lineage markers": own,
                "foreign-lineage markers": foreign,
                "own/foreign ratio": own / foreign if foreign else np.nan,
            }
        )
    table = pd.DataFrame(rows).set_index("variant")
    print(table.round(4).to_string())
    return table


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path,
        default=Path(os.environ.get("PYSPLIT_TUTORIAL_DIR", "pysplit_tutorial")),
        help="cache directory for downloads and intermediates",
    )
    parser.add_argument(
        "--full-section", action="store_true",
        help="use all ~167k cells instead of the cropped region (much slower)",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import pysplit

    print(f"pySPLIT {pysplit.__version__}  ->  {args.outdir.resolve()}")

    xenium, reference = load_data(args.outdir, args.full_section)
    rctd_result = deconvolve(xenium, reference, args.outdir)
    xenium, variants = purify_all(xenium, rctd_result)
    table = evaluate(variants)
    table.to_csv(args.outdir / "marker_specificity.csv")

    log("Done")
    print(f"  results written to {args.outdir / 'marker_specificity.csv'}")
    print("  cite: Bilous et al., Nature Methods (2026), "
          "doi:10.1038/s41592-026-03089-8")


if __name__ == "__main__":
    main()
