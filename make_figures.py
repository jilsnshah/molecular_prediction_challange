"""Generate publication-style figures from research_results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open("research_results.json"))
b = res["benchmark"]
os.makedirs("figures", exist_ok=True)

# Fig 1: MAE by representation x model (grouped bars, error bars = fold std)
reps = ["soap-inner", "soap-outer", "full", "triu", "eig"]
labels = ["SOAP\ninner", "SOAP\nouter", "Coulomb\nfull", "Coulomb\ntriu", "Coulomb\neig*"]
x = np.arange(len(reps)); w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i, (m, c) in enumerate([("xgboost", "#2b6cb0"), ("ridge", "#dd6b20")]):
    mae = [b[f"{r}/{m}"]["mae"] for r in reps]
    err = [b[f"{r}/{m}"]["mae_std"] for r in reps]
    ax.bar(x + (i-0.5)*w, mae, w, yerr=err, capsize=3, label=m, color=c)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("MAE (Debye)  -- lower better")
ax.set_title("Representation drives accuracy more than model\n5-fold CV, N=20,000  (*eig = provably permutation-invariant)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("figures/benchmark.png", dpi=140)

# Fig 2: error vs molecule size
mbs = res["mae_by_size"]
fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
ax2.bar(list(mbs.keys()), list(mbs.values()), color="#2f855a")
ax2.set_xlabel("molecule size (n atoms)"); ax2.set_ylabel("MAE (Debye)")
ax2.set_title(f"Error scales inversely with size (r={res['err_size_corr']:.2f})")
ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout(); fig2.savefig("figures/error_by_size.png", dpi=140)
print("wrote figures/benchmark.png, figures/error_by_size.png")
