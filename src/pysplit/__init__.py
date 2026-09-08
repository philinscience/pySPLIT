"""pySPLIT - Spatial Purification of Layered Intracellular Transcripts.

UNOFFICIAL Python reimplementation of the `SPLIT
<https://github.com/bdsc-tds/SPLIT>`_ R package (v0.3.0) by Mariia Bilous and
colleagues (BDSC, CHUV / University of Lausanne). The method and all of the
science behind it are theirs, published in *Nature Methods* (2026),
doi:10.1038/s41592-026-03089-8; this package only translates their R code to
Python. Not affiliated with or endorsed by the original authors, and the R
package remains the reference implementation.

**Always cite the original paper for the method** (``pysplit.__citation__``).
If this implementation was useful in itself, it may be cited as software too -
see ``CITATION.cff``.

Targets imaging-based spatial transcriptomics (Xenium) and the scanpy /
AnnData ecosystem.

SPLIT removes transcript spillover from segmented cells. Given a per-cell
deconvolution (e.g. RCTD in doublet mode) and a single-cell reference, each
observed count is rescaled by the share of expected expression attributable to
the cell's own primary cell type, so contaminating signal is proportionally
removed.

Typical workflow
----------------
.. code-block:: python

    import pysplit
    import scanpy as sc

    # 1. wrap a deconvolution result and make it SPLIT-compatible
    rctd = pysplit.read_rctdpy("rctd_out/", class_df=class_df)   # already post-processed

    # 2. purify
    purified = pysplit.purify(adata, rctd=rctd, remove_residual_contamination=True)

    # 3. optional: purify only cells with local contamination evidence
    sp_nw = pysplit.build_spatial_network(adata, radius=15, k_knn=20)
    pysplit.spatial_metrics(sp_nw, rctd)
    adata.obs = adata.obs.join(sp_nw.to_dataframe())
    balanced = pysplit.balance_by_score(adata, purified, threshold=0.05)

    # 4. optional: SPLIT-shift - swap primary/secondary labels
    sc.pp.pca(adata)
    tr_nw = pysplit.build_transcriptomics_network(adata, k_knn=100)
    pysplit.transcriptomics_metrics(tr_nw, rctd)
    adata.obs = adata.obs.join(tr_nw.to_dataframe())
    shifted = pysplit.balance_by_score(adata, purified, threshold=0.05, swap_labels=True)

See the ``README.md`` for the R-to-Python name mapping.
"""

from __future__ import annotations

import logging

from ._utils import SPOT_CLASS_LEVELS, entropy, row_normalize
from .api import purify
from .balance import (
    balance_by_score,
    balance_by_spot_class,
    balance_split,
    shift_labels,
    split_cells,
)
from .neighborhood import (
    add_annotation_from_neighbors,
    add_cell_type_neighborhood_weights,
    add_deconvolution_to_neighborhood,
    add_infiltration_metrics,
    add_neighborhood_annotation_certainty,
    add_neighborhood_weight_composition,
    add_spilling_type_ncount,
    add_spilling_type_weights,
    compute_swapping_score,
    neighborhood_to_dataframe,
    spatial_metrics,
    transcriptomics_metrics,
)
from .neighbors import (
    Neighborhood,
    build_spatial_network,
    build_transcriptomics_network,
    compute_neighborhood,
)
from .purify import auto_belonging_threshold, clean_reference, purify_counts
from .reassign import build_reassignment_operator, reassign_residual_counts
from .rctd import (
    RCTDResult,
    from_rctd_py,
    read_rctdpy,
    rctd_to_purify_input,
    run_post_process_rctd,
)

__version__ = "0.3.0"

#: Version of the SPLIT R package this port tracks.
__upstream_version__ = "0.3.0"
__upstream_url__ = "https://github.com/bdsc-tds/SPLIT"
#: The method citation. Always cite this.
__citation__ = (
    "Bilous M, Buszta D, Bac J, Kang S, Dong Y, Tissot S, Andre S, "
    "Alexandre-Gaveta M, Voize C, Peters S, Homicsko K, Gottardo R. "
    "Resolving sensitivity, specificity and signal contamination in Xenium "
    "spatial transcriptomics. Nature Methods (2026). "
    "doi:10.1038/s41592-026-03089-8"
)
#: Optional software citation for this implementation.
__software_citation__ = (
    "Putze P. pySPLIT: a Python implementation of SPLIT for spatial transcript "
    "purification (version 0.3.0), 2026. https://github.com/philinscience/pySPLIT"
)

__all__ = [
    "__version__",
    "__upstream_version__",
    "__upstream_url__",
    "__citation__",
    "__software_citation__",
    # purification
    "purify",
    "purify_counts",
    "clean_reference",
    "auto_belonging_threshold",
    # deconvolution results
    "RCTDResult",
    "run_post_process_rctd",
    "from_rctd_py",
    "rctd_to_purify_input",
    "read_rctdpy",
    "SPOT_CLASS_LEVELS",
    # networks
    "Neighborhood",
    "compute_neighborhood",
    "build_spatial_network",
    "build_transcriptomics_network",
    # neighbourhood metrics
    "add_deconvolution_to_neighborhood",
    "add_infiltration_metrics",
    "add_neighborhood_weight_composition",
    "add_cell_type_neighborhood_weights",
    "add_spilling_type_weights",
    "add_spilling_type_ncount",
    "add_annotation_from_neighbors",
    "add_neighborhood_annotation_certainty",
    "spatial_metrics",
    "transcriptomics_metrics",
    "neighborhood_to_dataframe",
    "compute_swapping_score",
    # combining raw and purified data
    "balance_by_score",
    "balance_by_spot_class",
    "balance_split",
    "split_cells",
    "shift_labels",
    # residual reassignment
    "build_reassignment_operator",
    "reassign_residual_counts",
    # helpers
    "entropy",
    "row_normalize",
]


def set_verbosity(level: int | str = logging.INFO) -> None:
    """Set pySPLIT's log level; attaches a stream handler on first use."""
    logger = logging.getLogger("pysplit")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


set_verbosity(logging.INFO)


def __getattr__(name: str):
    """Lazily expose the plotting helpers so matplotlib stays optional."""
    if name in ("pie_dataframe", "plot_pie", "plot_pie_around_cell",
                "plot_pie_by_coordinates", "crop_pie_dataframe"):
        from . import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module 'pysplit' has no attribute {name!r}")
