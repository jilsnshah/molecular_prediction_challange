"""
Evaluate the Coulomb-image CNN (model.py CoulombCNN) under the SAME 5-fold
protocol as research_benchmark.py, so it is directly comparable (paired folds,
seed=42) to the XGBoost/Ridge x representation arms.

Faithful to the original training recipe (train.py): raw Coulomb matrix as a
1x29x29 image (no feature normalization), L1 loss, Adam lr=1e-3, early stopping
on the validation fold. Only the split changes: single 80/20 -> 5-fold CV.
"""
import json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
sys.path.insert(0, ".")
from model import CoulombCNN

SEED = 42
SCRATCH = "/private/tmp/claude-501/-Users-jils-Projects-final-team58/704829af-ad30-4b40-80ae-11a6a90c4807/scratchpad"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device", dev)

d = np.load(f"{SCRATCH}/cm.npz", allow_pickle=True)
X = d["Xtr"].astype(np.float32)          # (20000,29,29)
y = d["y"].astype(np.float32)
N = len(y)

# reproduce sklearn KFold(n_splits=5, shuffle=True, random_state=42) EXACTLY:
# it permutes indices with the seed, then takes 5 sequential contiguous folds.
perm = np.random.RandomState(SEED).permutation(N)
folds = np.array_split(perm, 5)

def run_fold(va_idx):
    va = va_idx
    tr = np.setdiff1d(np.arange(N), va)
    Xt = torch.tensor(X[tr]).unsqueeze(1).to(dev); yt = torch.tensor(y[tr]).unsqueeze(1).to(dev)
    Xv = torch.tensor(X[va]).unsqueeze(1).to(dev); yv = torch.tensor(y[va]).unsqueeze(1).to(dev)
    model = CoulombCNN(29).to(dev)
    opt = optim.Adam(model.parameters(), lr=1e-3); crit = nn.L1Loss()
    bs, best, patience, wait = 64, float("inf"), 10, 0
    ntr = len(tr)
    for epoch in range(100):
        model.train(); order = torch.randperm(ntr)
        for i in range(0, ntr, bs):
            b = order[i:i+bs]
            opt.zero_grad(); loss = crit(model(Xt[b]), yt[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(Xv)
            vmae = torch.mean(torch.abs(p - yv)).item()
        if vmae < best - 1e-4:
            best, wait = vmae, 0
            ss_res = torch.sum((p - yv)**2).item(); ss_tot = torch.sum((yv - yv.mean())**2).item()
            best_r2 = 1 - ss_res/ss_tot
        else:
            wait += 1
            if wait >= patience: break
    return best, best_r2

maes, r2s = [], []
t0 = time.time()
for k, va in enumerate(folds):
    m, r = run_fold(va)
    maes.append(m); r2s.append(r)
    print(f"fold {k}: MAE {m:.4f}  R2 {r:.4f}  ({time.time()-t0:.0f}s)")
mae, mstd, r2, r2std = np.mean(maes), np.std(maes), np.mean(r2s), np.std(r2s)
print(f"\nCNN/coulomb-image  MAE {mae:.4f}+/-{mstd:.4f}  R2 {r2:.4f}+/-{r2std:.4f}")

res = json.load(open("research_results.json"))
res["benchmark"]["coulomb-image/cnn"] = dict(dim="29x29", mae=float(mae),
    mae_std=float(mstd), r2=float(r2), r2_std=float(r2std))
json.dump(res, open("research_results.json", "w"), indent=2)
print("updated research_results.json")
