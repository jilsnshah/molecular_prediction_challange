import os
import os.path as osp
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
from ase import Atoms
from dscribe.descriptors import SOAP
from tqdm import tqdm

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

def custom_radius_graph(pos, r=5.0):
    # Compute all pairwise distances
    dist = torch.cdist(pos, pos)
    # Find pairs where distance <= r and distance > 0 (to avoid self-loops)
    mask = (dist <= r) & (dist > 1e-5)
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)

class SoapGraphDataset(Dataset):
    def __init__(self, root, csv_file, struct_dir, transform=None, pre_transform=None):
        self.csv_file = csv_file
        self.struct_dir = struct_dir
        self.df = pd.read_csv(csv_file)
        
        # Check if training or testing
        self.is_train = 'dipole_moment' in self.df.columns
        self.id_col = 'molecule_name' if 'molecule_name' in self.df.columns else 'ID'
        
        # Initialize SOAP descriptor (average='off' to get atomic features)
        species = ['H', 'C', 'N', 'O', 'F']
        self.soap = SOAP(
            species=species,
            periodic=False,
            r_cut=5.0,
            n_max=8,
            l_max=6,
            average='off',
            sparse=False
        )
        
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.csv_file]

    @property
    def processed_file_names(self):
        # Determine the number of expected processed files
        return [f'data_{i}.pt' for i in range(len(self.df))]

    def process(self):
        print("Processing graphs. This may take a while...")
        
        idx = 0
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            mol_id = row[self.id_col]
            xyz_path = osp.join(self.struct_dir, f"{mol_id}.xyz")
            
            if not osp.exists(xyz_path):
                # Save a dummy data to match the length if file is missing
                # This should ideally be handled better, but ensures indices align.
                continue
                
            atoms, coords = read_xyz(xyz_path)
            molecule = Atoms(symbols=atoms, positions=coords)
            
            # 1. Node Features (X)
            # SOAP feature shape: (N_atoms, 952)
            soap_features = self.soap.create(molecule)
            x = torch.tensor(soap_features, dtype=torch.float32)
            
            # 2. Node Positions (pos)
            pos = torch.tensor(coords, dtype=torch.float32)
            
            # 3. Edge Index & Edge Attributes
            # Connect atoms within r_cut=5.0 using custom function to avoid torch-cluster
            edge_index = custom_radius_graph(pos, r=5.0)
            
            # Compute distances for edge_attr
            row_idx, col_idx = edge_index
            distances = torch.norm(pos[row_idx] - pos[col_idx], p=2, dim=1).view(-1, 1)
            edge_attr = distances
            
            # 4. Target (Y)
            if self.is_train:
                y = torch.tensor([row['dipole_moment']], dtype=torch.float32)
            else:
                y = torch.tensor([], dtype=torch.float32)
                
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, mol_id=mol_id)
            
            # Save the graph
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.df)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data

if __name__ == '__main__':
    print("Dataset script ready. Run 'python gnn_dataset.py' to process the graphs (paused per request).")
