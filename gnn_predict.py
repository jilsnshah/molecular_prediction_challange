import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import os
from tqdm import tqdm

from gnn_dataset import SoapGraphDataset
from gnn_model import SoapGNN

def predict_gnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading test dataset (may trigger caching if not done)...")
    try:
        test_dataset = SoapGraphDataset(root='gnn_data/test', csv_file='test.csv', struct_dir='structures_test')
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SoapGNN(input_dim=5740, hidden_dim=128).to(device)
    
    model_path = 'best_gnn_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first using gnn_train.py.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = []

    print("Running GNN inference...")
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            outputs = model(data)
            
            preds = outputs.squeeze().cpu().numpy()
            
            # If batch_size=1, preds is a scalar, so we handle it
            if preds.ndim == 0:
                preds = [preds]
                
            for mol_id, pred in zip(data.mol_id, preds):
                results.append({'ID': mol_id, 'dipole_moment': pred})

    submission_df = pd.DataFrame(results)
    submission_file = 'gnn_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"Predictions saved to {submission_file}")

if __name__ == "__main__":
    predict_gnn()
