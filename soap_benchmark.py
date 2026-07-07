"""
SOAP under the SAME 5-fold protocol as research_benchmark.py, plus an
aggregation ablation (average='inner' vs 'outer'): does *how* atom-level SOAP is
pooled to a molecule vector matter more than the model? Results merged into
research_results.json alongside the Coulomb representations.
"""
import json, time
import numpy as np
import pandas as pd
from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

SEED = 42
SCRATCH = "/private/tmp/claude-501/-Users-jils-Projects-final-team58/704829af-ad30-4b40-80ae-11a6a90c4807/scratchpad"
SPECIES = [1, 6, 7, 8, 9]

def read_xyz(path):
    L = open(path).readlines(); n = int(L[0]); syms, pos = [], []
    for ln in L[2:2+n]:
        p = ln.split(); syms.append(p[0]); pos.append([float(x) for x in p[1:4]])
    return Atoms(symbols=syms, positions=pos)

tr = pd.read_csv("train.csv")
y = tr["dipole_moment"].values.astype(np.float32)
mols = [read_xyz(f"structures_train/{n}.xyz") for n in tr["molecule_name"]]
N = len(y)
print(f"{N} molecules")

def featurize(avg):
    s = SOAP(species=SPECIES, periodic=False, r_cut=5.0, n_max=8, l_max=6,
             average=avg, sparse=False)
    return s.create(mols, n_jobs=8).astype(np.float32)

def xgb():
    return XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        random_state=SEED, objective="reg:absoluteerror")
def ridge(): return Ridge(alpha=10.0)
MODELS = {"xgboost": xgb, "ridge": ridge}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(kf.split(np.arange(N)))

new = {}
for avg in ["inner", "outer"]:
    t0 = time.time(); X = featurize(avg)
    print(f"SOAP-{avg} dim={X.shape[1]} ({time.time()-t0:.0f}s)")
    for mname, mfn in MODELS.items():
        maes, r2s = [], []
        for tr_i, va in folds:
            pre = make_pipeline(VarianceThreshold(0.0), StandardScaler()).fit(X[tr_i])
            m = mfn().fit(pre.transform(X[tr_i]), y[tr_i])
            p = m.predict(pre.transform(X[va]))
            maes.append(mean_absolute_error(y[va], p)); r2s.append(r2_score(y[va], p))
        key = f"soap-{avg}/{mname}"
        new[key] = dict(dim=int(X.shape[1]), mae=float(np.mean(maes)),
                        mae_std=float(np.std(maes)), r2=float(np.mean(r2s)),
                        r2_std=float(np.std(r2s)))
        print(f"{key:20s} MAE {np.mean(maes):.4f}+/-{np.std(maes):.4f}  R2 {np.mean(r2s):.4f}+/-{np.std(r2s):.4f}")

# merge into existing results
res = json.load(open("research_results.json"))
res["benchmark"].update(new)
best = min((k for k in res["benchmark"] if k.endswith("xgboost")),
           key=lambda k: res["benchmark"][k]["mae"])
res["best_overall_xgb"] = best
json.dump(res, open("research_results.json", "w"), indent=2)
print(f"\nbest overall (xgboost) = {best}  MAE={res['benchmark'][best]['mae']:.4f}")
