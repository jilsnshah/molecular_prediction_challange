import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from ase import Atoms
from dscribe.descriptors import CoulombMatrix
from tqdm import tqdm

class MoleculeImageDataset(Dataset):
    def __init__(self, csv_file, cache_dir):
        self.df = pd.read_csv(csv_file)
        self.cache_dir = cache_dir
        
        # Determine if it's train or test by checking if 'dipole_moment' exists
        self.is_train = 'dipole_moment' in self.df.columns
        
        # The identifier column is either 'molecule_name' (train) or 'ID' (test)
        self.id_col = 'molecule_name' if 'molecule_name' in self.df.columns else 'ID'
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_id = row[self.id_col]
        
        # Load pre-computed tensor
        tensor_path = os.path.join(self.cache_dir, f"{mol_id}.pt")
        if not os.path.exists(tensor_path):
            raise FileNotFoundError(f"Cached tensor not found: {tensor_path}. Run cache_data() first.")
            
        # Matrix shape is expected to be (N, N), we add a channel dimension to make it (1, N, N)
        matrix = torch.load(tensor_path, weights_only=True)
        image_tensor = matrix.unsqueeze(0).float()
        
        if self.is_train:
            target = torch.tensor([row['dipole_moment']], dtype=torch.float32)
            return image_tensor, target
        else:
            return image_tensor, mol_id

def read_xyz(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[0])
        atom_data = lines[2:2 + num_atoms]
        atoms = []
        coords = []
        for line in atom_data:
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        return atoms, coords

def cache_data(csv_file, struct_dir, cache_dir, n_atoms_max=29):
    """
    Reads xyz files, computes 2D Coulomb Matrix, and saves as .pt files
    """
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    id_col = 'molecule_name' if 'molecule_name' in df.columns else 'ID'
    
    # Create CoulombMatrix and let it flatten by default, we'll reshape it later
    cm = CoulombMatrix(n_atoms_max=n_atoms_max, permutation="sorted_l2")
    
    print(f"Caching Coulomb Matrices for {len(df)} molecules into {cache_dir}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mol_id = row[id_col]
        xyz_path = os.path.join(struct_dir, f"{mol_id}.xyz")
        tensor_path = os.path.join(cache_dir, f"{mol_id}.pt")
        
        if os.path.exists(tensor_path):
            continue
            
        if not os.path.exists(xyz_path):
            # We don't print warning to avoid spamming 19k times if files are genuinely missing
            # But the dataset assumes files are present.
            continue
            
        atoms, coords = read_xyz(xyz_path)
        molecule = Atoms(symbols=atoms, positions=coords)
        
        matrix_np = cm.create(molecule)
        
        # In dscribe v2, create() returns a flattened 1D array of shape (N*N,) for a single molecule
        # We need to reshape it back to 2D (N, N)
        matrix_np = matrix_np.reshape((n_atoms_max, n_atoms_max))
        
        tensor = torch.tensor(matrix_np, dtype=torch.float32)
        torch.save(tensor, tensor_path)
        
if __name__ == "__main__":
    print("Pre-computing training data...")
    cache_data('train.csv', 'structures_train', 'data_cache/train')
    
    print("Pre-computing testing data...")
    cache_data('test.csv', 'structures_test', 'data_cache/test')
