import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero, Linear
from torch_geometric.data import HeteroData

class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(128, hidden_dim)        # MAG240M features are 128-dim
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = Linear(hidden_dim, 153)     # 153 L1 classes
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.classifier(x)


def get_model(data: HeteroData, hidden_dim: int = 1024):
    """
    Wraps homogeneous GraphSAGE into heterogeneous version using to_hetero
    """
    base_model = SimpleGraphSAGE(hidden_dim=hidden_dim)
    return to_hetero(base_model, data.metadata(), aggr='sum')

