import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import MoleculeImageDataset
from model import CoulombCNN
from sklearn.metrics import r2_score
import os

def train_model(epochs=100, batch_size=64, lr=0.001, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading dataset...")
    try:
        full_dataset = MoleculeImageDataset(csv_file='train.csv', cache_dir='data_cache/train')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python dataset.py' to generate the cached tensors first.")
        return
        
    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Initialize Model
    model = CoulombCNN(input_size=29).to(device)
    criterion = nn.L1Loss() # Mean Absolute Error
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_mae = float('inf')
    patience_counter = 0

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        # Calculate R2 Score
        r2 = r2_score(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}/{epochs} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | Val R2: {r2:.4f}")
        
        # Save best model and Early Stopping logic
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("--> Saved new best model!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

if __name__ == "__main__":
    train_model(epochs=100)
