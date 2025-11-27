import torch
import torch.nn as nn
#!pip install torch_geometric
#!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
#!pip install pandas
import torch_geometric
from torch_geometric.nn import to_hetero
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import HGTLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import HeteroConv
from torch_geometric.transforms import ToUndirected
from torch.nn.functional import cross_entropy,mse_loss
from torch_geometric.utils import index_to_mask, mask_to_index, mask_select
from torch.nn import Linear, ReLU, Softmax
from tqdm import tqdm
from inductive import to_inductive
import torch.nn.functional as F

dataset = OGB_MAG(root='GNN-SSL-Project-for-Deep-Learning/data/',
                  transform=ToUndirected(),
                  preprocess="metapath2vec")[0]

node_type = "paper"
dataset = to_inductive(dataset.clone(), node_type)

# num_neighbors = [15, 10, 5]
num_neighbors = [8, 5]
batch_size = 256

train_batch = NeighborLoader(dataset, 
                        num_neighbors=num_neighbors, 
                        input_nodes=('paper', dataset['paper'].train_mask),
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=4)

batch = next(iter(train_batch))

print("node_types:", batch.node_types)
print("edge_types:", batch.edge_types)
print("metadata():", batch.metadata())


for ntype in batch.node_types:
    x = batch[ntype].x
    print(f"{ntype}: x is None? {x is None}", end="")
    if x is not None:
        print(", shape=", tuple(x.shape))
    else:
        print()

# test_batch = NeighborLoader(dataset, 
#                         num_neighbors=num_neighbors, 
#                         input_nodes=('paper', dataset['paper'].test_mask),
#                         batch_size=batch_size, 
#                         shuffle=True, 
#                         num_workers=0)
# val_batch = NeighborLoader(dataset, 
#                         num_neighbors=num_neighbors, 
#                         input_nodes=('paper', dataset['paper'].val_mask),
#                         batch_size=batch_size, 
#                         shuffle=True, 
#                         num_workers=0)


# class PartialLearnableEncoder(nn.Module):
#     """
#     Some nodes get learnable embeddings.
#     Others use original features + Gaussian noise.
#     Works on CUDA automatically.
#     """
#     def __init__(self, num_nodes, in_dim, out_dim, learnable_mask, noise_std=0.1, device=None):
#         super().__init__()

#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = device

#         # Store mask — move to right device
#         self.learnable_mask = learnable_mask.to(device)   # Bool [num_nodes]
#         self.noise_std = noise_std

#         # Create mapping (node -> learnable embedding index)
#         learnable_indices = torch.nonzero(self.learnable_mask).squeeze().to(device)

#         node_to_learn_index = -torch.ones(num_nodes, dtype=torch.long, device=device)
#         node_to_learn_index[learnable_indices] = torch.arange(
#             len(learnable_indices), device=device
#         )
#         self.register_buffer("node_to_learn_index", node_to_learn_index)

#         # Learnable embeddings
#         self.learn_emb = nn.Parameter(
#             torch.randn(len(learnable_indices), out_dim, device=device)
#         )

#         # Projection for non-learnable nodes
#         self.proj = nn.Linear(in_dim, out_dim).to(device)


#     def forward(self, x):
#         """
#         x: [num_nodes, in_dim]
#         returns: [num_nodes, out_dim]
#         """
#         x = x.to(self.device)

#         num_nodes = x.size(0)
#         out = torch.zeros(num_nodes, self.learn_emb.size(1), device=self.device)

#         # --- Learnable nodes ---
#         learn_mask = self.learnable_mask
#         # learn_idx = self.node_to_learn_index[learn_mask]   # indices inside learn_emb
#         # learn_idx = torch.nonzero(self.learnable_mask).squeeze().to(device)
#         out[learn_mask] = self.learn_emb

#         # --- Non-learnable nodes ---
#         not_mask = ~learn_mask
#         if not_mask.any():
#             noise = torch.randn_like(x[not_mask]) * self.noise_std
#             out[not_mask] = self.proj(x[not_mask] + noise)

#         return out
    


# num_nodes = batch["paper"].num_nodes
# learnable_mask = torch.rand(num_nodes) < 0.25

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# encoder = PartialLearnableEncoder(
#     num_nodes=num_nodes,
#     in_dim=batch["paper"].x.size(1),
#     out_dim=128,
#     learnable_mask=learnable_mask,
#     noise_std=0.1,
#     device=device
# ).to(device)


# paper_x = batch["paper"].x.to(device)
# paper_x = encoder(paper_x)

# print(paper_x)

num_nodes = batch["paper"].num_nodes

class GraphMAEEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.proj = nn.Linear(in_dim, out_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x, mask=None):
        if mask is None:
            keep_ratio = 1 - self.mask_ratio
            mask = torch.rand(x.size(0), device=x.device) < keep_ratio

        out = self.proj(x)  # project all first (more efficient)

        if (~mask).any():
            out[~mask] = self.mask_token  # override masked

        if mask.any() and keep_ratio < 1.0:
            # Optional scaling of visible nodes
            out[mask] = out[mask] / keep_ratio

        return out, mask

# 1. Fake MAG-like data (instant, no download)

num_nodes = 100_000       
feat_dim  = 768           
x = torch.randn(num_nodes, feat_dim)   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)       

print(f"x shape: {x.shape}, device: {x.device}")

encoder = GraphMAEEncoder(in_dim=feat_dim, out_dim=512).to(device)
# Forward pass
with torch.no_grad():
    h, mask = encoder(x)               

print(f"Output shape: {h.shape}")        
print(f"Kept nodes : {mask.sum().item():,} / {num_nodes:,} ({mask.float().mean():.3f})")
print(f"Mask token norm: {encoder.mask_token.norm():.4f}")

# Check that masked nodes really have the mask token
masked_h = h[~mask]
print(f"All masked nodes equal to mask_token? {torch.allclose(masked_h, encoder.mask_token)}")