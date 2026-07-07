import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SoapGNN(torch.nn.Module):
    def __init__(self, input_dim=5740, hidden_dim=128):
        super(SoapGNN, self).__init__()
        
        # 1. Node Embedding: Compress the 952-dim SOAP vector
        # This prevents over-parameterization during message passing
        self.node_embed = nn.Linear(input_dim, hidden_dim)
        
        # 2. Message Passing Layers
        # GCNConv aggregates neighborhood information
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 3. Readout / Prediction Head
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, data):
        # PyG batches graphs dynamically into a single giant graph. 
        # The 'batch' tensor keeps track of which node belongs to which molecule.
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial projection
        x = F.relu(self.node_embed(x))
        
        # Message passing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global Pooling (aggregates node embeddings into a global molecule embedding)
        x = global_mean_pool(x, batch)
        
        # MLP for final prediction
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        out = self.fc2(x)
        
        return out
