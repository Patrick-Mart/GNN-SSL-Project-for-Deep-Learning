print("Begin package importation...\n")
import torch
import torch.nn as nn
#!pip install torch_geometric
#!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
#!pip install pandas
import torch_geometric
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import HeteroConv
from torch_geometric.transforms import ToUndirected
from torch.nn.functional import mse_loss
from torch.nn import Linear, ReLU
from inductive import to_inductive

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

print("Load dataset...\n")
dataset = OGB_MAG(root='GNN-SSL-Project-for-Deep-Learning/data/',
                  transform=ToUndirected(),
                  preprocess="metapath2vec")[0]

node_type = "paper"
dataset_inductive = to_inductive(dataset.clone(), node_type)
dataset = dataset.to(device)
dataset_inductive = dataset_inductive.to(device)


# HYPERPARAMETERS
num_neighbors = [8, 5]
batch_size = 256
hidden_dim = 256
out_channels = dataset['paper'].x.shape[1]

print("Create batches...\n")

train_batch = NeighborLoader(dataset_inductive, 
                        num_neighbors=num_neighbors, 
                        input_nodes=('paper', dataset_inductive['paper'].train_mask),
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=0)

test_batch = NeighborLoader(dataset, 
                        num_neighbors=num_neighbors, 
                        input_nodes=('paper', dataset['paper'].test_mask),
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=0)

val_batch = NeighborLoader(dataset, 
                        num_neighbors=num_neighbors, 
                        input_nodes=('paper', dataset['paper'].val_mask),
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=0)



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
    def __init__(self, node_types, hidden_dim, output_dim):
        super().__init__()
        self.decoder = Linear(hidden_dim,output_dim)

    def forward(self, x_dict):
        x_dict = {k:self.decoder(v) for (k,v) in x_dict.items()}
        return x_dict



print("Creating models...\n")
encoder = graphSAGE_ENCODER(dataset.edge_types, hidden_dim).to(device)
decoder = graphSAGE_DECODER(dataset.node_types, hidden_dim, out_channels).to(device)


def build_x_dict(batch):
    x_dict = {}
    for node_type in batch.node_types:
        x_dict[node_type] = batch[node_type].x.to(device)
    return x_dict


def mask_x_dict(x_dict, p=0.6):
    mask = {}
    x_dict_masked = {}
    for k, v in x_dict.items():
        mask[k] = (torch.rand(v.shape[0]) < p).to(device)
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
}).to(device)

opt = torch.optim.Adam(
    list(encoder.parameters())+list(unk_emb.parameters()) + list(decoder.parameters()), lr=0.01)


epochs = 50
print("Start of training...")
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    i = 1
    for batch in train_batch:
        encoder.train()
        decoder.train()
        total_loss_train = 0
        x_dict = build_x_dict(batch)
        edge_index_dict = {
            edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}
        x_dict_masked, mask = mask_x_dict(x_dict, p=0.4)
        out = encoder(x_dict_masked, edge_index_dict)
        out = decoder(out)
        out_formatted, x_formatted = formatting2loss(out, x_dict, mask)
        loss = mse_loss(out_formatted, x_formatted)
        opt.zero_grad()
        loss.backward()
        opt.step()
        opt.zero_grad()
        total_loss_train += loss.item()
        print(f"Epoch {epoch}, batch {i}, Training loss: {loss.item():.4f}")
        i += 1
        '''
        if i==5:
            break
        '''
    
    i=1
    for batch in val_batch:
        encoder.eval()
        decoder.eval()
        total_loss_val = 0
        x_dict = build_x_dict(batch)
        edge_index_dict = {
            edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types}
        x_dict_masked, mask = mask_x_dict(x_dict, p=0.4)
        out = encoder(x_dict_masked, edge_index_dict)
        out = decoder(out)
        out_formatted, x_formatted = formatting2loss(out, x_dict, mask)
        loss = mse_loss(out_formatted, x_formatted)
        total_loss_val += loss.item()
        print(f"Epoch {epoch}, batch {i}, Validation loss: {loss.item():.4f}")
        i += 1
        '''
        if i==5:
            break
        '''
        

    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {total_loss_train:.4f}, Val Loss: {total_loss_val:.4f}")
