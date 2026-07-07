"""
Featurize all molecules with SOAP (molecule-level, average='inner'), matching
the descriptor that produced the project's best single-split result. Cached to
npz so the CV benchmark reuses it without recomputing (~expensive) SOAP.
"""
import glob, time
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from dscribe.descriptors import SOAP

SCRATCH = "/private/tmp/claude-501/-Users-jils-Projects-final-team58/704829af-ad30-4b40-80ae-11a6a90c4807/scratchpad"
SPECIES = [1, 6, 7, 8, 9]  # H C N O F

def read_xyz(path):
    L = open(path).readlines()
    n = int(L[0])
    syms, pos = [], []
    for ln in L[2:2+n]:
        p = ln.split()
        syms.append(p[0]); pos.append([float(x) for x in p[1:4]])
    return Atoms(symbols=syms, positions=pos)

def load_split(names, folder):
    return [read_xyz(f"{folder}/{n}.xyz") for n in names]

import pandas as pd
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
mols_tr = load_split(tr["molecule_name"], "structures_train")
mols_te = load_split(te["ID"], "structures_test")
print(f"read {len(mols_tr)} train + {len(mols_te)} test molecules")

soap = SOAP(species=SPECIES, periodic=False, r_cut=5.0, n_max=8, l_max=6,
            average="inner", sparse=False)
t0 = time.time()
Xtr = soap.create(mols_tr, n_jobs=8).astype(np.float32)
Xte = soap.create(mols_te, n_jobs=8).astype(np.float32)
print(f"SOAP dim={Xtr.shape[1]}  featurized in {time.time()-t0:.0f}s")
np.savez_compressed(f"{SCRATCH}/soap.npz", Xtr=Xtr, Xte=Xte)
print("wrote soap.npz")
