import torch
import torch_geometric.nn as nn

class model_GraphSAGE(torch.nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super().__init__()
    self.h_layer_1 = nn.SAGEConv(in_dim, hidden_dim)
    self.h_layer_2 = nn.SAGEConv(hidden_dim, out_dim)

  def forward(self, data):
      x = data["paper"].x
      x = self.h_layer_1(x, data.edge_index_dict[('paper', 'cites', 'paper')])
      x = torch.nn.functional.relu(x)

      x = self.h_layer_2(x, data.edge_index_dict[('paper', 'cites', 'paper')])
      x = torch.nn.functional.relu(x)

      return (x) # softmax is calculated in the loss function.