import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import HGTLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import HeteroConv
from torch_geometric.transforms import ToUndirected
from torch.nn.functional import cross_entropy
from torch.nn import Linear, ReLU, Softmax
from tqdm import tqdm

print(torch.__version__)
if torch.cuda.is_available():
    print("CUDA runtime version (used by PyTorch):", torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = OGB_MAG(root='GNN-SSL-Project-for-Deep-Learning/data/',
                  transform=ToUndirected())[0]
print(dataset)




num_neighbors = [15, 10, 5]
batch_size = 64

train_batch = NeighborLoader(dataset, 
                        num_neighbors=num_neighbors, 
                        input_nodes=('paper', dataset['paper'].train_mask),
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
i=0
for batch in train_batch:
    print(batch)
    if i>5:
        break
    i+=1
    


batch = next(iter(train_batch))
class graphSAGESINGLE(nn.Module):
    def __init__(self,edge_types,hidden_dim,output_dim):
        super().__init__()
        self.conv1 = HeteroConv({edge_type : SAGEConv((-1,-1),hidden_dim) for edge_type in edge_types},aggr='sum')
        self.conv2 = HeteroConv({edge_type : SAGEConv((-1,-1),output_dim) for edge_type in edge_types},aggr='sum')

    def forward(self,x_dict,edge_index_dict):
        x_dict = self.conv1(x_dict,edge_index_dict)
        x_dict = {k:ReLU()(v) for k,v in x_dict.items()}
        x_dict = self.conv2(x_dict,edge_index_dict)
        return x_dict['paper']

hidden_dim = 16
out_channels = max(batch['paper'].y).item() + 1

model = graphSAGESINGLE(batch.edge_types,hidden_dim,out_channels)

featless = [t for t in batch.node_types if 'x' not in batch[t]]
print(featless)
emb = nn.ModuleDict({
    t: nn.Embedding(dataset[t].num_nodes, 128)
    for t in featless
})

def build_x_dict(batch):
    x_dict = {}
    for node_type in batch.node_types:
        if node_type in featless:
            x_dict[node_type] = emb[node_type](batch[node_type].n_id)
        else:
            x_dict[node_type] = batch[node_type].x
    return x_dict

x_dict = build_x_dict(batch)
edge_index_dict = {edge_type : batch[edge_type].edge_index for edge_type in batch.edge_types}

opt = torch.optim.Adam(list(model.parameters())+list(emb.parameters()), lr=0.01)


epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    x_dict = build_x_dict(batch)
    edge_index_dict = {edge_type : batch[edge_type].edge_index for edge_type in batch.edge_types}
    out = model(x_dict,edge_index_dict)
    loss = cross_entropy(out, batch['paper'].y)
    prediction = torch.argmax(Softmax(dim=1)(out),dim=-1)
    acc = (prediction == batch['paper'].y).sum()/batch['paper'].y.shape[0]
    opt.zero_grad()
    loss.backward()
    opt.step()
    total_loss += loss.item() 
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, accuracy: {acc}")


x_dict = build_x_dict(batch)
edge_index_dict = {edge_type : batch[edge_type].edge_index for edge_type in batch.edge_types}
out = model(x_dict,edge_index_dict)
prediction = torch.argmax(Softmax(dim=1)(out),dim=-1)

acc = (prediction == batch['paper'].y).sum()/batch['paper'].y.shape[0]
print(acc)




hidden_channels = 64
out_channels = max(dataset['paper'].y).item() + 1


featless = [t for t in dataset.node_types if 'x' not in dataset[t]]
emb = nn.ModuleDict({
    t: nn.Embedding(dataset[t].num_nodes, 128)
    for t in featless
})
emb = emb.to(device)

class graphSAGEmodel(nn.Module):
    def __init__(self, edge_types, hidden, out_channels):
        super().__init__()
        edge_types = edge_types
        # layer 1
        self.conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden) for et in edge_types},
            aggr='sum'
        )
        # layer 2
        self.conv2 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden) for et in edge_types},
            aggr='sum'
        )
        self.head = Linear(hidden, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        #x_dict = {k: ReLU(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        # return logits for papers only
        return self.head(x_dict['paper'])
    
model = graphSAGEmodel(dataset.edge_types, hidden_channels, out_channels)
model = model.to(device)
opt = torch.optim.Adam(list(model.parameters())+list(emb.parameters()), lr=0.01)
print(model)



def build_x_dict(batch):
    x_dict = {}
    for node_type in batch.node_types:
        if node_type in featless:
            x_dict[node_type] = emb[node_type](batch[node_type].n_id)
        else:
            x_dict[node_type] = batch[node_type].x
    return x_dict


@torch.no_grad()
def infer(loader):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        bs = batch['paper'].batch_size
        x_dict = build_x_dict(batch)
        edge_index_dict = {edge_type : batch[edge_type].edge_index for edge_type in batch.edge_types}
        logits = model(x_dict, edge_index_dict)[:bs]
        preds.append(logits.argmax(-1).cpu())
        ys.append(batch['paper'].y[:bs].view(-1).cpu())
    return torch.cat(preds), torch.cat(ys)

num_epochs = 5
for epoch in range(1, num_epochs):
    model.train()
    tr_loss = 0
    for batch in tqdm(train_batch, desc=f"Epoch {epoch}/{num_epochs}"):
        batch = batch.to(device)
        x_dict = build_x_dict(batch)
        edge_index_dict = {edge_type : batch[edge_type].edge_index for edge_type in batch.edge_types}
        logits = model(x_dict, edge_index_dict)
        loss = cross_entropy(logits, batch['paper'].y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tr_loss += loss.item()
    val_pred, val_true = infer(val_batch)
    val_metric = (val_pred == val_true).sum().item() / val_true.size(0)
    print(f"Epoch {epoch:02d} | loss {tr_loss:.4f} | val {val_metric:.4f}")
