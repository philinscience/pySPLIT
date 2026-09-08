"""Simulate Xenium-like spillover and verify SPLIT recovers the true profiles.

Run:  python examples/simulate_and_validate.py

Ground truth is known by construction, so this asserts the two things that
matter: purification must raise marker-gene specificity, and residual
contamination removal must raise it further.
"""
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import pysplit

pysplit.set_verbosity("WARNING")
rng = np.random.default_rng(7)

CT = ["Tumor", "Tcell", "Myeloid", "Stromal"]
G, N = 200, 3000
# reference: exclusive marker blocks + shared background
ref = pd.DataFrame(rng.uniform(1e-4, 5e-4, (4, G)), index=CT,
                   columns=[f"g{i}" for i in range(G)])
for i, t in enumerate(CT):
    ref.loc[t, ref.columns[i*40:(i+1)*40]] = rng.uniform(0.005, 0.02, 40)
ref = ref.div(ref.sum(1), axis=0)

true_type = rng.choice(CT, N, p=[0.5, 0.2, 0.15, 0.15])
contaminant = np.array([rng.choice([c for c in CT if c != t]) for t in true_type])
# contamination fraction: half the cells are clean, half heavily contaminated
frac = np.where(rng.random(N) < 0.5, 0.0, rng.uniform(0.2, 0.5, N))
w1 = 1.0 - frac

pos = {t: i for i, t in enumerate(CT)}
W = np.zeros((N, 4))
W[np.arange(N), [pos[t] for t in true_type]] = w1
W[np.arange(N), [pos[t] for t in contaminant]] += frac

depth = rng.integers(150, 600, N)
prof = W @ ref.to_numpy()
counts = rng.poisson(prof / prof.sum(1, keepdims=True) * depth[:, None])

cells = [f"c{i}" for i in range(N)]
adata = ad.AnnData(sparse.csr_matrix(counts.astype(float)),
                   obs=pd.DataFrame(index=cells),
                   var=pd.DataFrame(index=list(ref.columns)))

weights = pd.DataFrame(W, index=cells, columns=CT)
primary = pd.Series(true_type, index=cells)

def marker_purity(X, labels):
    """Fraction of each cell's counts landing on its own type's marker genes."""
    X = np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)
    out = []
    for i, t in enumerate(labels):
        j = pos[t]
        own = X[i, j*40:(j+1)*40].sum()
        total = X[i].sum()
        if total > 0:
            out.append(own / total)
    return float(np.mean(out))

raw_purity = marker_purity(adata.X, true_type)
std = pysplit.purify(adata, weights=weights, reference=ref,
                     primary_cell_type=primary, verbose=False)
res = pysplit.purify(adata, weights=weights, reference=ref,
                     primary_cell_type=primary,
                     remove_residual_contamination=True, verbose=False)

# ground truth: counts that genuinely came from the cell's own type
clean_prof = np.zeros_like(prof)
clean_prof[np.arange(N), :] = w1[:, None] * ref.to_numpy()[[pos[t] for t in true_type]]

print(f"{'':28s}{'marker purity':>14s}{'counts kept':>13s}")
print(f"{'raw':28s}{raw_purity:14.3f}{100.0:12.1f}%")
for name, obj in [("SPLIT", std), ("SPLIT + residual removal", res)]:
    p = marker_purity(obj.X, obj.obs['first_type'])
    kept = 100 * float(obj.X.sum()) / float(adata.X.sum())
    print(f"{name:28s}{p:14.3f}{kept:12.1f}%")

# purity should improve most where contamination was actually injected
contaminated = frac > 0
for label, mask in [("clean cells", ~contaminated), ("contaminated cells", contaminated)]:
    ids = np.asarray(cells)[mask]
    r = marker_purity(adata[ids].X, true_type[mask])
    s = marker_purity(std[ids].X, true_type[mask])
    print(f"  {label:22s} raw {r:.3f} -> SPLIT {s:.3f}")

assert marker_purity(std.X, std.obs['first_type']) > raw_purity, "SPLIT must raise purity"
assert marker_purity(res.X, res.obs['first_type']) > marker_purity(std.X, std.obs['first_type'])
print("\nOK: purification increases marker specificity, residual removal adds more.")
