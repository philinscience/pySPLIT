# pySPLIT — Spatial Purification of Layered Intracellular Transcripts

> ### ⚠️ This is an unofficial Python reimplementation, not the original package
>
> **pySPLIT is a Python port of the [SPLIT](https://github.com/bdsc-tds/SPLIT)
> R package (v0.3.0) by Mariia Bilous and colleagues at the Biomedical Data
> Science Center, CHUV / University of Lausanne.**
>
> The SPLIT method, algorithm and all of the science behind it are entirely
> theirs, published in *Nature Methods* (2026) —
> [doi:10.1038/s41592-026-03089-8](https://doi.org/10.1038/s41592-026-03089-8).
> This repository contributes only a translation of their R code into Python.
>
> * **Not affiliated with or endorsed by** the original authors.
> * **The original R package is the reference implementation.** For published
>   results, or if numbers from the two disagree, trust
>   [bdsc-tds/SPLIT](https://github.com/bdsc-tds/SPLIT).


Targets **imaging-based spatial transcriptomics (Xenium)** and the
scanpy / AnnData ecosystem.

> Visium/VisiumHD-specific material from the original package is intentionally
> **not** ported. The core purification is platform-agnostic, so it works on
> VisiumHD data too — but none of the Visium tutorials, helpers or defaults
> were carried over.

## What SPLIT does

Segmented cells in imaging-based spatial data pick up transcripts that belong to
their neighbours. Given a per-cell deconvolution (RCTD in doublet mode, or any
other tool) and a single-cell reference, SPLIT rescales every observed count by
the share of expected expression attributable to the cell's **own** primary cell
type:

```
                       w[cell, t1] · R[t1, gene]
purified[cell, gene] = ───────────────────────────── · counts[cell, gene]
                       Σ_t w[cell, t] · R[t, gene]
```

`t1` is the cell's primary cell type, `w` its deconvolution weights and `R` the
reference profiles. A gene with no reference support in `t1` gets a zero
numerator, so it is removed entirely.

On the public 10x Xenium breast cancer sample (see the
[tutorial](docs/tutorial.md)), this raises each cell's share of counts on its
own lineage's markers by 24%, cuts the share on *foreign* lineage markers by
30%, and improves the ratio between them by 76% — while keeping 68% of counts.

On simulated spillover with known ground truth (`examples/`), it lifts mean
marker-gene purity from 0.75 to 0.91 while keeping 82% of counts, and leaves
uncontaminated cells untouched (0.911 → 0.911).

## Installation

```bash
pip install -e ".[all]"          # from a clone
```

Core dependencies are `numpy`, `scipy`, `pandas`, `anndata` and `scikit-learn`.
Extras: `plot` (matplotlib), `rctdpy` (pyarrow + h5py, for reading `rctd-py`
output), `test` (pytest).

## Tutorial

**[docs/tutorial.md](docs/tutorial.md)** walks the complete workflow on real
public data — the 10x Xenium FFPE Human Breast Cancer sample (167,780 cells ×
313 genes) deconvolved against the matched Chromium reference (26,031 annotated
cells, 17 cell types), from the
[Janesick et al. (2023)](https://doi.org/10.1038/s41467-023-43458-x) study used
by the original SPLIT vignette.

It comes in two runnable forms, both downloading the data themselves and
caching each stage:

**Notebook, with figures** —
[`examples/tutorial_xenium_breast_cancer.ipynb`](examples/tutorial_xenium_breast_cancer.ipynb).
Committed with its outputs, so the plots render on GitHub without running
anything.

```bash
pip install "pysplit-st[all]" rctd-py scanpy openpyxl
jupyter lab examples/tutorial_xenium_breast_cancer.ipynb
```

**Script**, for a headless or batch run:

```bash
python examples/tutorial_xenium_breast_cancer.py                 # cropped region
python examples/tutorial_xenium_breast_cancer.py --full-section  # all 167k cells
```

Both cover RCTD deconvolution via [`rctd-py`](https://github.com/p-gueguen/rctd-py),
all three SPLIT modes, and score the outcome with marker-gene specificity on
genes SPLIT never sees. The notebook shares the script's data-loading and
deconvolution helpers, so the two cannot drift apart.

Two smaller, dependency-free examples run on simulated data with known ground
truth, and double as end-to-end tests:

```bash
python examples/simulate_and_validate.py    # purification recovers true profiles
python examples/xenium_workflow.py          # diffusion score finds contamination
```

## Quick start

### Annotation-method agnostic (recommended)

All you need is a cells × cell-types weight matrix and a cell-types × genes
reference:

```python
import pysplit

purified = pysplit.purify(
    adata,                                   # cells x genes raw counts
    weights=weights,                         # DataFrame, cells x cell types
    reference=reference,                     # DataFrame, cell types x genes
    primary_cell_type=labels,                # optional; defaults to arg-max of weights
    remove_residual_contamination=True,      # v0.3.0 feature, see below
)
```

`purified` is an `AnnData`: `X` holds the purified counts, `layers['counts_raw']`
the aligned raw counts, `obs` the per-cell metadata, `uns['pysplit']` the run
parameters. Either reference orientation is accepted — the cell-type axis is
detected from the labels.

### From an RCTD-style doublet-mode result

```python
import pysplit

# Wrap a deconvolution result...
rctd = pysplit.RCTDResult(
    results_df=results_df,          # spot_class, first_type, second_type per cell
    weights=weights,                # cells x cell types (full decomposition)
    weights_doublet=weights_doublet,# cells x 2 (primary / secondary weight)
    reference=reference,            # cell types x genes
    class_df=class_df,              # optional cell type -> class map (recommended)
    coords=coords,                  # optional x/y
)

# ...make it SPLIT-compatible (required), then purify
rctd = pysplit.run_post_process_rctd(rctd)
purified = pysplit.purify(adata, rctd=rctd, remove_residual_contamination=True)
```

Output of the Python [`rctd-py`](https://github.com/p-gueguen/rctd-py) loads
directly, already post-processed — either from the object it just returned:

```python
import rctd

reference = rctd.Reference(chromium, cell_type_col="cell_type")
result = rctd.run_rctd(xenium, reference, mode="doublet")

rctd_result = pysplit.from_rctd_py(
    result,
    reference=profiles,                  # cell types x genes
    cell_ids=list(xenium.obs_names),     # pixel_mask is applied for you
    class_df=class_df,
    coords=xenium.obs[["x", "y"]],
)
```

...or from a saved `rctd-py` output directory (the layout the R package's
`reconstruct_rctd_from_rctdpy()` expects):

```python
rctd_result = pysplit.read_rctdpy("rctd_out/", class_df=class_df, coords=coords)
```

`run_post_process_rctd` is not optional. RCTD reports a `second_type` even for
cells where a single cell type holds essentially all the weight; post-processing
relabels those as confident singlets with no secondary type, so purification
leaves them alone instead of stripping a genuine phenotype.

### Residual contamination removal (v0.3.0)

Standard SPLIT removes contamination coming from cell types *in* the
deconvolution model. `remove_residual_contamination=True` additionally zeroes
genes with no **specific** reference support in the cell's assigned type —
non-specific background that proportional redistribution cannot touch. The
specificity threshold is derived from the reference (75th percentile of the
per-gene min/max ratio), so there is nothing to tune.

Expect a further 2–5% of counts removed and measurably better marker
specificity. Note that a *surviving* gene may keep a **larger** fraction than
under standard SPLIT: cleaning shrinks the denominator too, which is the point.

### Spatially-aware SPLIT

Purifying every cell over-corrects phenotypes that are rare or absent in the
reference. Instead, purify only cells whose secondary signal is actually present
in their physical surroundings:

```python
sp_nw = pysplit.build_spatial_network(adata, k_knn=20, prune=True, radius=15)  # µm
pysplit.spatial_metrics(sp_nw, rctd)
adata.obs = adata.obs.join(sp_nw.to_dataframe())

balanced = pysplit.balance_by_score(
    adata, purified,
    score_name="neighborhood_weights_second_type",   # the local diffusion score
    threshold=0.05,                                  # lower -> purify more cells
)
```

`neighborhood_weights_second_type` is the share of a cell's spatial
neighbourhood made up of its own secondary cell type. On a simulated tumour /
stroma sample it averages 0.85 for genuinely contaminated cells and 0.00 for
clean ones.

### SPLIT-shift

When contamination is strong enough, the deconvolution assigns a cell to the
*contaminating* type. SPLIT-shift swaps the primary and secondary labels for
cells whose transcriptomic neighbourhood agrees with neither their primary type
nor its class, while matching their secondary class — and takes the **residual**
as such a cell's profile:

```python
import scanpy as sc
sc.pp.pca(adata)

tr_nw = pysplit.build_transcriptomics_network(adata, k_knn=100)
pysplit.transcriptomics_metrics(tr_nw, rctd)
adata.obs = adata.obs.join(tr_nw.to_dataframe())

shifted = pysplit.balance_by_score(adata, purified, threshold=0.05, swap_labels=True)
```

Pre-swap values are preserved in `*_before_swap` columns and `obs['swap']`
records what happened.

### Reassigning removed transcripts

Removed counts physically belong to *some* nearby cell. Rather than discarding
them, redistribute them to the spatial neighbours they most plausibly came from:

```python
corrected = pysplit.reassign_residual_counts(
    raw_counts=adata[purified.obs_names, purified.var_names].X,
    purified_counts=purified.X,
    neighborhood=sp_nw,                                # after spatial_metrics()
    purification_status=purified.obs["purification_status"],
    cell_names=list(purified.obs_names),
    mode="count_proportional",                         # or "uniform"
)
```

### Keeping both halves of a cell

```python
split = pysplit.split_cells(adata, purified)     # <cell>_1 purified, <cell>_2 residual
```

### Pie visualisation

```python
pie_df = pysplit.pie_dataframe(rctd)
pysplit.plot_pie_around_cell(pie_df, cell_id="cell_42", radius=50, palette=pal)
```

Slices are the primary/secondary weights, the pie edge marks the primary cell
type, and the centre marker encodes the spot class (coloured dot =
`doublet_certain`, black dot = `doublet_uncertain`, cross = `reject`, nothing =
confident `singlet`).

## R → Python name mapping

| R (SPLIT)                                            | Python (pySPLIT)                          |
| ---------------------------------------------------- | ----------------------------------------- |
| `purify()`                                           | `purify()`                                |
| `rctd_free_purify()`                                  | `purify_counts()`                         |
| `rctd_based_purify()`, `purify_counts_with_rctd()`   | `purify(rctd=...)`                        |
| `run_post_process_RCTD()`                            | `run_post_process_rctd()`                 |
| `convert_rctd_result_to_purify_input()`              | `rctd_to_purify_input()`                  |
| `reconstruct_rctd_from_rctdpy()`                     | `read_rctdpy()`, or `from_rctd_py()` for an in-memory result |
| `build_spatial_network()`                            | `build_spatial_network()`                 |
| `build_transcriptomics_network()`                    | `build_transcriptomics_network()`         |
| `compute_neighborhood()`                             | `compute_neighborhood()`                  |
| `add_rctd_to_neighborhood()`                         | `add_deconvolution_to_neighborhood()`     |
| `add_infiltration_metrics_to_neighborhood()`         | `add_infiltration_metrics()`              |
| `add_neighborhood_weight_composition()`              | `add_neighborhood_weight_composition()`   |
| `add_cell_types_neighborhood_weights()`              | `add_cell_type_neighborhood_weights()`    |
| `add_neigborhood_weight_composition_of_spilling_cell_type()` | `add_spilling_type_weights()`     |
| `add_individual_w1/w2_of_second_type_in_neighborhood()` | folded into `add_spilling_type_weights()` |
| `add_max_weight_of_spilling_type_in_neighborhood()`  | folded into `add_spilling_type_weights()` |
| `add_neigborhood_nCount_of_spilling_cell_type()`     | `add_spilling_type_ncount()`              |
| `add_spatial_metric()`                               | `spatial_metrics()`                       |
| `add_transcriptomics_metric()`                       | `transcriptomics_metrics()`               |
| `add_annotation_from_neighbors()`                    | `add_annotation_from_neighbors()`         |
| `add_neighborhood_annotation_certainty()`            | `add_neighborhood_annotation_certainty()` |
| `neighborhood_analysis_to_metadata()`                | `neighborhood_to_dataframe()` / `.to_dataframe()` |
| `balance_raw_and_purified_data_by_score()`           | `balance_by_score()`                      |
| `balance_raw_and_purified_data_by_spot_class()`      | `balance_by_spot_class()`                 |
| `balance_split()`                                    | `balance_split()`                         |
| `split_cells()`                                      | `split_cells()`                           |
| `compute_swapping_score()`                           | `compute_swapping_score()`                |
| `build_reassigment_operator()`                       | `build_reassignment_operator()`           |
| `reassign_residual_counts()`                         | `reassign_residual_counts()`              |
| `get_pieplot_df()`                                   | `pie_dataframe()`                         |
| `crop_pie_df_to_cell()`                              | `crop_pie_dataframe()`                    |
| `plot_pie()`, `plot_pie_around_cell()`, `plot_pie_by_coordinates()` | same names       |
| `DO_*` arguments                                     | snake_case without the prefix (`remove_residual_contamination`, `swap_labels`, …) |

Not ported: `update_score_mat_RCTD()` (deprecated upstream and unused),
`save_pieplot()` (use `matplotlib.pyplot.savefig`), and the Seurat/Visium-specific
plumbing.

## Deliberate differences from the R package

* **Orientation.** Matrices are `cells × genes` throughout, matching AnnData;
  the R package uses `genes × cells`. Reference frames accept either
  orientation.
* **Sparse-aware core.** The purification factor is evaluated only at observed
  (non-zero) counts instead of materialising a dense cells × genes product.
  Mathematically identical, but it turns the R implementation's tens of
  gigabytes of peak memory into something proportional to the number of stored
  counts. `chunk_size` therefore only bounds memory and never changes results
  (asserted by a test).
* **Bug fixes carried into the port** (each documented in the code):
  * `add_neighborhood_weight_composition` excludes the focal cell properly. The
    R version drops the first element *after* filtering to annotated cells,
    which silently drops a real neighbour when the focal cell is unannotated.
  * `compute_swapping_score`'s `min_swapping_score` returned the max in R
    (copy-paste slip); here it is the min. The function also takes a plain
    per-cell DataFrame instead of relying on a global `RCTD` object, which the R
    version does by mistake.
* **`NA` semantics.** Missing labels never match each other, matching R's
  `NA == NA -> NA` rather than Python's `None == None -> True`.
* **Deprecated arguments** (`DO_parallel`, `n_workers`, `DO_purify_singlets`,
  `gene_list`) are simply absent. `DO_purify_singlets=TRUE` is the behaviour you
  get by default: post-processing already strips the arbitrary secondary label
  from confident singlets, so they pass through unchanged.

## Notes and caveats

* Purified counts are **fractional**, not integers. Tools requiring integer
  counts need rounding first.
* `reject` cells are dropped from `purify()` output — they have no trustworthy
  annotation. Filter your raw object to match before comparing.
* Cells absent from the deconvolution are dropped, with a warning.
* Normalisation, PCA, UMAP and clustering are scanpy's job; pySPLIT only
  produces the purified counts and metadata.

## Tests

```bash
pip install -e ".[all]"
pytest            # 66 tests
```

Beyond unit tests, `examples/simulate_and_validate.py` and
`examples/xenium_workflow.py` run the full pipeline on simulated data with a
known ground truth and assert that purification raises marker specificity and
that the diffusion score separates contaminated from clean cells.
`examples/tutorial_xenium_breast_cancer.py` does the same on real data — see
[docs/tutorial.md](docs/tutorial.md).


## Acknowledgements

All credit for SPLIT goes to its authors and to the
[BDSC-TDS group](https://github.com/bdsc-tds) at CHUV / University of Lausanne.
Questions about the *method* are best directed to the
[original repository](https://github.com/bdsc-tds/SPLIT/issues) or to
Mariia Bilous (Mariia.Bilous@chuv.ch). Please raise issues here only for
problems specific to this Python translation.

## License

GPL-3.0-or-later, matching the [original package](https://github.com/bdsc-tds/SPLIT)
— a port of GPL-licensed work stays under the same terms. See `LICENSE` for the
full text and [`NOTICE`](NOTICE) for the upstream attribution.
