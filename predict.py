import torch
from torch.utils.data import DataLoader
from dataset import MoleculeImageDataset
from model import CoulombCNN
import pandas as pd
import os
from tqdm import tqdm

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading test dataset...")
    try:
        test_dataset = MoleculeImageDataset(csv_file='test.csv', cache_dir='data_cache/test')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python dataset.py' to generate the cached tensors first.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = CoulombCNN(input_size=29).to(device)
    
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = []

    print("Running inference...")
    with torch.no_grad():
        for images, mol_ids in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            # Outputs shape is (Batch, 1)
            preds = outputs.cpu().numpy().flatten()
            
            for mol_id, pred in zip(mol_ids, preds):
                results.append({'ID': mol_id, 'dipole_moment': pred})

    # Save to CSV
    submission_df = pd.DataFrame(results)
    submission_file = 'cnn_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"Predictions saved to {submission_file}")

if __name__ == "__main__":
    predict()
