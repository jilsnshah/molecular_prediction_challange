import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import os

from gnn_dataset import SoapGraphDataset
from gnn_model import SoapGNN

def train_gnn(epochs=100, batch_size=32, lr=0.001, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading PyG Dataset (may trigger caching if not done)...")
    try:
        # Load the dataset directly. If processed files exist, it loads them.
        dataset = SoapGraphDataset(root='gnn_data/train', csv_file='train.csv', struct_dir='structures_train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # PyG's DataLoader handles batching of Graph Data objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SoapGNN(input_dim=5740, hidden_dim=128).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_mae = float('inf')
    patience_counter = 0

    print("Starting GNN training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Since outputs shape is (Batch, 1) and y is (Batch,), we need to unsqueeze y or squeeze outputs
            loss = criterion(outputs.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.num_graphs
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Current Loss: {loss.item():.4f}", flush=True)
            
        train_loss /= len(train_loader.dataset)

        # Validation
        print(f"Starting Validation for Epoch {epoch+1}...", flush=True)
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs.squeeze(), data.y)
                
                val_loss += loss.item() * data.num_graphs
                all_preds.extend(outputs.squeeze().cpu().numpy().tolist())
                all_targets.extend(data.y.cpu().numpy().tolist())
                
        val_loss /= len(val_loader.dataset)
        r2 = r2_score(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}/{epochs} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | Val R2: {r2:.4f}")

        # Early stopping and model saving
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gnn_model.pth')
            print("--> Saved new best GNN model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

if __name__ == "__main__":
    train_gnn()
