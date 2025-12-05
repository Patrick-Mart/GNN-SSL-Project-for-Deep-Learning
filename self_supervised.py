print("Importing libraries...")
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
from inductive import to_inductive
from tqdm import tqdm
import matplotlib.pyplot as plt
from unsupervised_features import graphSAGE_ENCODER

print(torch.__version__)
if torch.cuda.is_available():
    print("CUDA runtime version (used by PyTorch):", torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If your computer does not have enough memory, set temp = True to use preprocessed features
temp = False

if temp:
    dataset = OGB_MAG(root='GNN-SSL-Project-for-Deep-Learning/data/',
                  transform=ToUndirected(),
                  preprocess="metapath2vec")[0]
else:
    dataset = OGB_MAG(root='GNN-SSL-Project-for-Deep-Learning/data/',
                  transform=ToUndirected())[0]
    
node_type = "paper"
dataset_inductive = to_inductive(dataset.clone(), node_type)
dataset = dataset.to(device)
dataset_inductive = dataset_inductive.to(device)

# HYPERPARAMETERS
num_epochs = 5
num_neighbors = [5, 5, 5]
batch_size = 128
hidden_channels = 256
out_channels = max(dataset['paper'].y).item() + 1

print("Create batches...\n")

train_batch = NeighborLoader(dataset_inductive, 
                        num_neighbors=num_neighbors, 
                        input_nodes=('paper', dataset_inductive['paper'].train_mask),
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=4)

test_batch = NeighborLoader(dataset, 
                        num_neighbors=num_neighbors, 
                        input_nodes=('paper', dataset['paper'].test_mask),
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=4)

val_batch = NeighborLoader(dataset, 
                        num_neighbors=num_neighbors, 
                        input_nodes=('paper', dataset['paper'].val_mask),
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=4)




print("Create embeddings for no feature nodes...\n")
if temp:
    featless=[]
else:
    featless = [t for t in dataset.node_types if 'x' not in dataset[t]]
    emb = nn.ModuleDict({
        t: nn.Embedding(dataset[t].num_nodes, 128)
        for t in featless
    })
    # emb = emb.to(device)



class graphSAGEmodel(nn.Module):
    def __init__(self, edge_types, hidden, out_channels):
        super().__init__()
        # edge_types = edge_types
        self.encoder = graphSAGE_ENCODER(dataset.edge_types, hidden_channels).to(device)

        state = torch.load("best_encoder_b128_h256.pth", map_location=device)
        self.encoder.load_state_dict(state)

        # Freeze encoder to prevent retraining it
        for p in self.encoder.parameters():
            p.requires_grad = False

        # self.encoder = encoder.load_state_dict(torch.load("best_encoder_b128_h256.pth"))
        self.head = Linear(hidden, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # return logits for papers only
        x_dict = self.encoder((x_dict, edge_index_dict))
        return self.head(x_dict['paper'])
    
model = graphSAGEmodel(dataset.edge_types, hidden_channels, out_channels)
model = model.to(device)
if temp:
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
else:
    opt = torch.optim.Adam(list(model.parameters())+list(emb.parameters()), lr=0.01)


def build_x_dict(batch):
    x_dict = {}
    for node_type in batch.node_types:
        if node_type in featless:
            x_dict[node_type] = emb[node_type](batch[node_type].n_id)
        else:
            x_dict[node_type] = batch[node_type].x
        x_dict[node_type] = x_dict[node_type].to(device)
    return x_dict


train_accs = []
val_accs = []

@torch.no_grad()
def infer(loader):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        bs = batch['paper'].batch_size
        x_dict = build_x_dict(batch)
        edge_index_dict = {edge_type : batch[edge_type].edge_index for edge_type in batch.edge_types}
        logits = model(x_dict, edge_index_dict)[:bs]
        preds.append(logits.argmax(-1).cpu())
        ys.append(batch['paper'].y[:bs].view(-1).cpu())
    return torch.cat(preds), torch.cat(ys)

print("Start training...\n")
for epoch in range(1, num_epochs):
    model.train()
    tr_loss = 0
    train_correct = 0
    train_total = 0

    for batch in tqdm(train_batch, desc=f"Epoch {epoch}/{num_epochs}"):
        x_dict = build_x_dict(batch)
        edge_index_dict = {edge_type : batch[edge_type].edge_index for edge_type in batch.edge_types}
        logits = model(x_dict, edge_index_dict)#['paper']        # add paper
        loss = cross_entropy(logits, batch['paper'].y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tr_loss += loss.item()


        # # Training accuracy on the current batch
        # pred = logits.argmax(dim=-1)
        # train_acc = (pred == batch['paper'].y.squeeze()).float().mean().item()

    

    print("Evaluating...\n")
    val_pred, val_true = infer(val_batch)
    val_metric = (val_pred == val_true).sum().item() / val_true.size(0)
    print(f"Epoch {epoch:02d} | loss {tr_loss:.4f} | val {val_metric:.4f}\n")

    train_pred, train_true = infer(train_batch)
    train_metric = (train_pred == train_true).sum().item() / train_true.size(0)

    train_accs.append(train_metric)
    val_accs.append(val_metric)

print("train")
print(train_accs)
print("val")
print(val_accs)


plt.figure(figsize=(10, 6))


plt.plot([i for i in range(1, num_epochs)], train_accs, label="training", marker="o", linewidth=2)
plt.plot([i for i in range(1, num_epochs)], val_accs, label="validation", marker="o", linewidth=2)

plt.legend(title="Dataset", fontsize=10, title_fontsize=12, loc="best")
plt.xlabel("Index (time step, node, etc.)")
plt.ylabel("Value")
plt.title("Comparison of arrays")
plt.grid(True, alpha=0.3)
plt.tight_layout()


plt.savefig(f"GNN-SSL-Project-for-Deep-Learning/results/self_supervised_accuracies_h{hidden_channels}_b{batch_size}.png", dpi=300, bbox_inches='tight')