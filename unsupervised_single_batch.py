import torch
import torch.nn as nn
#!pip install torch_geometric
#!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
#!pip install pandas
import torch_geometric
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import HGTLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import HeteroConv
from torch_geometric.transforms import ToUndirected
from torch.nn.functional import cross_entropy, mse_loss
from torch_geometric.utils import index_to_mask, mask_to_index, mask_select
from torch.nn import Linear, ReLU, Softmax
from tqdm import tqdm
from inductive import to_inductive

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
                             input_nodes=(
                                 'paper', dataset['paper'].train_mask),
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)
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

batch = next(iter(train_batch))


class graphSAGE_ENCODER(nn.Module):
    def __init__(self, edge_types, hidden_dim):
        super().__init__()
        self.conv1 = HeteroConv({edge_type: SAGEConv(
            (-1, -1), hidden_dim) for edge_type in edge_types}, aggr='sum')
        self.conv2 = HeteroConv({edge_type: SAGEConv(
            (-1, -1), hidden_dim) for edge_type in edge_types}, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: ReLU()(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class graphSAGE_DECODER(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_dict):
        return self.decoder(x_dict['paper'])


hidden_dim = batch['paper'].x.shape[1]
out_channels = max(batch['paper'].y).item() + 1

encoder = graphSAGE_ENCODER(batch.edge_types, hidden_dim)
decoder = graphSAGE_DECODER(hidden_dim, out_channels)


def build_x_dict(batch):
    x_dict = {}
    for node_type in batch.node_types:
        x_dict[node_type] = batch[node_type].x
    return x_dict


def mask_x_dict(x_dict, p=0.75):
    mask = {}
    x_dict_masked = {}
    for k, v in x_dict.items():
        mask[k] = torch.rand(v.shape[0]) < p
        x_dict_masked[k] = x_dict[k].clone()*(mask[k].unsqueeze(-1)) + unk_emb[k](
            torch.zeros_like(mask[k], dtype=torch.long))*(~mask[k].unsqueeze(-1))
    return x_dict_masked, mask


def formatting2loss(out, x_dict, mask):
    out = torch.cat(tuple([out[k][~mask[k]] for k in x_dict.keys()]), dim=0)
    x = torch.cat(tuple([x_dict[k][~mask[k]] for k in x_dict.keys()]), dim=0)
    return out, x


unk_emb = nn.ModuleDict({
    t: nn.Embedding(1, 128)
    for t in dataset.node_types
})

opt_encoder = torch.optim.Adam(
    list(encoder.parameters())+list(unk_emb.parameters()), lr=0.01)
opt_decoder = torch.optim.Adam(decoder.parameters(), lr=0.01)


epochs = 3
print("Start of training...")
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    i = 1
    for batch in train_batch:
        # Train the encoder
        encoder.train()
        total_loss_encoder = 0
        x_dict = build_x_dict(batch)
        edge_index_dict = {
            edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}
        x_dict_masked, mask = mask_x_dict(x_dict, p=0.9)
        out = encoder(x_dict_masked, edge_index_dict)
        out_formatted, x_formatted = formatting2loss(out, x_dict, mask)
        loss_encoder = mse_loss(out_formatted, x_formatted)
        opt_encoder.zero_grad()
        loss_encoder.backward()
        opt_encoder.step()
        opt_encoder.zero_grad()
        total_loss_encoder += loss_encoder.item()

        # Train the decoder
        decoder.train()
        out = {k: v.detach() for k, v in out.items()}
        total_loss_decoder = 0
        venue_pred = decoder(out)
        loss_decoder = cross_entropy(venue_pred, batch['paper'].y)
        opt_decoder.zero_grad()
        loss_decoder.backward()
        opt_decoder.step()
        opt_decoder.zero_grad()
        total_loss_decoder += loss_decoder.item()

        print(f"Epoch {epoch}, batch {i}")
        i += 1

    prediction = torch.argmax(Softmax(dim=1)(venue_pred), dim=-1)
    acc = (prediction == batch['paper'].y).sum()/batch['paper'].y.shape[0]
    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {total_loss_encoder:.4f}, Accuracy: {acc:.4f}")
