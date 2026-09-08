"""Full Xenium workflow on a simulated tumour/stroma sample.

Run:  python examples/xenium_workflow.py

Walks the whole pipeline - deconvolution result -> purify -> spatial network ->
spatially-aware SPLIT -> SPLIT-shift - and checks that the local diffusion
score separates genuinely contaminated cells from clean ones.
"""
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import pysplit

pysplit.set_verbosity("WARNING")
rng = np.random.default_rng(3)
CT = ["Tumor", "Tcell", "Myeloid", "Stromal"]
N = 1500

# Two spatial regions: a tumour mass (left) and a stromal region (right).
xy = rng.uniform(0, 200, (N, 2))
in_tumour = xy[:, 0] < 100
true_type = np.where(in_tumour,
                     rng.choice(["Tumor", "Tcell"], N, p=[0.85, 0.15]),
                     rng.choice(["Stromal", "Myeloid"], N, p=[0.8, 0.2]))
# a cell is contaminated by whatever dominates its own region
region_dominant = np.where(in_tumour, "Tumor", "Stromal")
contaminated = (true_type != region_dominant) & (rng.random(N) < 0.7)
second = np.where(contaminated, region_dominant, None)
frac = np.where(contaminated, rng.uniform(0.25, 0.45, N), 0.0)

pos = {t: i for i, t in enumerate(CT)}
W = np.zeros((N, 4))
W[np.arange(N), [pos[t] for t in true_type]] = 1 - frac
for i in np.flatnonzero(contaminated):
    W[i, pos[second[i]]] += frac[i]

G = 120
ref = pd.DataFrame(rng.uniform(1e-4, 5e-4, (4, G)), index=CT,
                   columns=[f"g{i}" for i in range(G)])
for i, t in enumerate(CT):
    ref.loc[t, ref.columns[i*30:(i+1)*30]] = rng.uniform(0.005, 0.02, 30)
ref = ref.div(ref.sum(1), axis=0)

prof = W @ ref.to_numpy()
depth = rng.integers(150, 500, N)
counts = rng.poisson(prof / prof.sum(1, keepdims=True) * depth[:, None])
cells = [f"c{i}" for i in range(N)]

adata = ad.AnnData(sparse.csr_matrix(counts.astype(float)),
                   obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=list(ref.columns)))
adata.obsm["spatial"] = xy

spot_class = np.where(contaminated, "doublet_certain", "singlet")
results_df = pd.DataFrame(
    {"spot_class": spot_class, "first_type": true_type, "second_type": second},
    index=cells)
rctd = pysplit.RCTDResult(
    results_df=results_df,
    weights=pd.DataFrame(W, index=cells, columns=CT),
    weights_doublet=pd.DataFrame(
        np.column_stack([1 - frac, np.where(contaminated, frac, np.nan)]),
        index=cells, columns=["first_type", "second_type"]),
    reference=ref,
    class_df=pd.Series({"Tumor": "Epithelial", "Tcell": "Immune",
                        "Myeloid": "Immune", "Stromal": "Stromal"}),
    coords=pd.DataFrame(xy, index=cells, columns=["x", "y"]),
)
rctd = pysplit.run_post_process_rctd(rctd, verbose=False)

purified = pysplit.purify(adata, rctd=rctd, verbose=False)
adata.obs = adata.obs.join(rctd.results_df)

sp_nw = pysplit.build_spatial_network(adata, k_knn=20, prune=True, radius=15)
pysplit.spatial_metrics(sp_nw, rctd)
adata.obs = adata.obs.join(sp_nw.to_dataframe())

score = adata.obs["neighborhood_weights_second_type"]
print("diffusion score by ground truth:")
print(f"  genuinely contaminated : mean {score[contaminated].mean():.3f}")
print(f"  clean                  : mean {score[~contaminated].mean():.3f}")

balanced = pysplit.balance_by_score(adata, purified, threshold=0.05)
n_pur = (balanced.obs["purification_status"] == "purified").to_numpy()
truth = balanced.obs.index.map(dict(zip(cells, contaminated))).to_numpy(dtype=bool)
tp, fp = int((n_pur & truth).sum()), int((n_pur & ~truth).sum())
fn = int((~n_pur & truth).sum())
print(f"\nspatially-aware SPLIT @0.05: purified {n_pur.sum()} cells")
print(f"  recall of contaminated cells : {tp/(tp+fn):.2%}")
print(f"  precision                    : {tp/max(tp+fp,1):.2%}")

adata.obsm["X_pca"] = np.asarray(adata.X.todense())[:, :20]
tr_nw = pysplit.build_transcriptomics_network(adata, dims=range(20), k_knn=50)
pysplit.transcriptomics_metrics(tr_nw, rctd)
adata.obs = adata.obs.join(tr_nw.to_dataframe(), rsuffix="_tr")
shifted = pysplit.balance_by_score(adata, purified, threshold=0.05, swap_labels=True)
print(f"\nSPLIT-shift swapped {int(shifted.obs['swap'].sum())} label(s)")
assert score[contaminated].mean() > score[~contaminated].mean() * 2
print("OK: the diffusion score separates real contamination from clean cells.")
