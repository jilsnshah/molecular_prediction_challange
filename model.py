import torch
import torch.nn as nn
import torch.nn.functional as F

class CoulombCNN(nn.Module):
    def __init__(self, input_size=29):
        super(CoulombCNN, self).__init__()
        
        # Input shape: (Batch, 1, 29, 29)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After conv1 + pool1: (Batch, 16, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After conv2 + pool2: (Batch, 32, 7, 7)
        
        # Calculate flattened dimension.
        # 29 // 2 = 14
        # 14 // 2 = 7
        self.flattened_dim = 32 * 7 * 7
        
        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, self.flattened_dim)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test the model with dummy data
    model = CoulombCNN()
    dummy_input = torch.randn(4, 1, 29, 29)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
