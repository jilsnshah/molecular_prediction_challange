"""
Controlled benchmark: does molecular *representation* or *model* drive
dipole-moment prediction accuracy?

All experiments run under one protocol: 5-fold cross-validation on 20,000
labeled molecules, MAE/R2 reported as mean +/- std across folds. Every model
sees the same folds (fixed seed) so comparisons are paired.

Representations derived from the 29x29 Coulomb matrix (permutation="sorted_l2"):
  full  : flattened matrix                (841 dims)
  triu  : upper triangle incl. diagonal   (435 dims, drops redundant symmetry)
  eig   : sorted eigenvalue spectrum      ( 29 dims, PROVABLY permutation-invariant)

The eig representation is the scientific control for the permutation-invariance
hypothesis: the Coulomb-matrix eigenspectrum is invariant to atom ordering by
construction, so if invariance were the bottleneck, eig should dominate.
"""
import json, time
import numpy as np
import pandas as pd
from scipy import stats
from ase import Atoms
from dscribe.descriptors import CoulombMatrix
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

SEED = 42
NMAX = 29  # global max atom count -> fixed 29x29 padded matrices

def read_xyz(path):
    L = open(path).readlines(); n = int(L[0]); syms, pos = [], []
    for ln in L[2:2+n]:
        p = ln.split(); syms.append(p[0]); pos.append([float(x) for x in p[1:4]])
    return Atoms(symbols=syms, positions=pos)

tr = pd.read_csv("train.csv")
y = tr["dipole_moment"].values.astype(np.float32)
mols = [read_xyz(f"structures_train/{n}.xyz") for n in tr["molecule_name"]]
natoms = np.array([len(m) for m in mols])
cm = CoulombMatrix(n_atoms_max=NMAX, permutation="sorted_l2")
Xcm = cm.create(mols, n_jobs=8).reshape(len(mols), NMAX, NMAX).astype(np.float32)
N = len(y)
print(f"loaded {N} molecules, natoms {natoms.min()}-{natoms.max()}")

# ---- representations ---------------------------------------------------------
triu = np.triu_indices(29)
def rep_full(X): return X.reshape(len(X), -1)
def rep_triu(X): return X[:, triu[0], triu[1]]
def rep_eig(X):
    # symmetric -> real eigenvalues; sort descending for a canonical spectrum
    w = np.linalg.eigvalsh(X)          # ascending
    return np.sort(w, axis=1)[:, ::-1]
REPS = {"full": rep_full, "triu": rep_triu, "eig": rep_eig}

def xgb():  # fixed config across all runs
    return XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        random_state=SEED, objective="reg:absoluteerror")
def ridge(): return Ridge(alpha=10.0)
MODELS = {"xgboost": xgb, "ridge": ridge}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(kf.split(np.arange(N)))

# ---- 1. benchmark: representation x model, paired 5-fold CV ------------------
results = {}
fold_mae = {}  # (rep,model) -> [per-fold MAE] for significance tests
for rname, rfn in REPS.items():
    Xr = rfn(Xcm).astype(np.float32)
    for mname, mfn in MODELS.items():
        maes, r2s = [], []
        for tr, va in folds:
            # drop zero-variance padding cols (molecules < 29 atoms -> all-zero
            # rows/cols) BEFORE scaling, else StandardScaler divides by zero
            pre = make_pipeline(VarianceThreshold(0.0), StandardScaler()).fit(Xr[tr])
            Xt, Xv = pre.transform(Xr[tr]), pre.transform(Xr[va])
            m = mfn().fit(Xt, y[tr])
            p = m.predict(Xv)
            maes.append(mean_absolute_error(y[va], p))
            r2s.append(r2_score(y[va], p))
        key = f"{rname}/{mname}"
        fold_mae[key] = maes
        results[key] = dict(dim=int(Xr.shape[1]),
                            mae=float(np.mean(maes)), mae_std=float(np.std(maes)),
                            r2=float(np.mean(r2s)),  r2_std=float(np.std(r2s)))
        print(f"{key:20s} dim={Xr.shape[1]:4d}  MAE {np.mean(maes):.4f}+/-{np.std(maes):.4f}  R2 {np.mean(r2s):.4f}+/-{np.std(r2s):.4f}")

# ---- 2. significance: does full beat eig for XGBoost? (paired across folds) --
a, b = fold_mae["full/xgboost"], fold_mae["eig/xgboost"]
t_p = stats.ttest_rel(a, b).pvalue
w_p = stats.wilcoxon(a, b).pvalue if len(set(np.sign(np.array(a)-np.array(b)))) > 0 else 1.0
sig = dict(full_xgb_mae=float(np.mean(a)), eig_xgb_mae=float(np.mean(b)),
           delta=float(np.mean(b)-np.mean(a)), paired_t_p=float(t_p), wilcoxon_p=float(w_p))
print(f"\nfull vs eig (xgboost): delta MAE={sig['delta']:+.4f}  paired-t p={t_p:.2e}")

# ---- 3. interpretability + error analysis on best rep -----------------------
best = min((k for k in results if k.endswith("xgboost")), key=lambda k: results[k]["mae"])
brep = best.split("/")[0]
Xb = REPS[brep](Xcm).astype(np.float32)
tr, va = folds[0]
pre = make_pipeline(VarianceThreshold(0.0), StandardScaler()).fit(Xb[tr])
m = xgb().fit(pre.transform(Xb[tr]), y[tr])
pred = m.predict(pre.transform(Xb[va]))
kept = pre[0].get_support(indices=True)          # original feature indices retained
gains = m.feature_importances_
top = kept[np.argsort(gains)[::-1][:10]]         # map back to original rep indices

# error vs molecule size (heavy-atom / total-atom count)
res = np.abs(pred - y[va]); nv = natoms[va]
bins = [(3,8),(9,12),(13,16),(17,26)]
size_err = {f"{lo}-{hi}": float(res[(nv>=lo)&(nv<=hi)].mean()) for lo,hi in bins}
size_corr = float(np.corrcoef(nv, res)[0,1])
print(f"\nbest rep = {brep}; top feat idx (by gain): {top.tolist()}")
print(f"MAE by n_atoms bin: {size_err}")
print(f"corr(|residual|, n_atoms) = {size_corr:+.3f}")

out = dict(protocol="5-fold CV, seed=42, N=20000",
           benchmark=results, significance=sig,
           best_xgb_rep=brep, top_feature_idx=top.tolist(),
           mae_by_size=size_err, err_size_corr=size_corr,
           generated=time.strftime("%Y-%m-%d %H:%M"))
with open("research_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote research_results.json")
