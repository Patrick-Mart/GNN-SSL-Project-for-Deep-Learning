import torch
import torch_sparse
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NodeLoader
from torch_geometric.loader import LinkLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import HGTLoader
import torch_geometric.transforms as T
import data as dataset_tools
from model import model_GraphSAGE 


data = OGB_MAG(root="data", transform = T.ToUndirected())[0]

node_type = "paper" # target node type
data_inductive = dataset_tools.to_inductive(data.clone(), node_type)


### alternative loader

# loader = HGTLoader(
#     data,
#     # Sample 512 nodes per type and per iteration for 4 iterations
#     num_samples={key: [512] * 4 for key in data.node_types},
#     # Use a batch size of 128 for sampling training nodes of type paper
#     batch_size=128,
#     input_nodes=('paper', data['paper'].train_mask),
# )
# batch = next(iter(loader))


neighbor_loader = NeighborLoader(data_inductive,
                                 num_neighbors=[100, 50, 30],
                                 batch_size=15,
                                 shuffle=True,
                                 input_nodes=('paper', data['paper'].train_mask))
nei_batch = next(iter(neighbor_loader))



print(nei_batch.size())
print(nei_batch.node_types)
print(nei_batch.edge_types)
for each in nei_batch.node_types:
  if each == "paper":
    print(nei_batch[each].x)
  else:
    print(each)
# print(batch.x) data have no attr x
# print(batch.adj_t) data have no attr x
# print(data["paper"].adj_t) does not work
# print(batch["paper"].adj_t)
print((nei_batch.x_dict["paper"]).size())
print(nei_batch.edge_index_dict[('paper', 'cites', 'paper')].size())


# Training loop ------------------------------------------------

model_active = model_GraphSAGE(-1, 500, 349)

optimizer = torch.optim.SGD(model_active.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

model_active.train()
for epoch in range(5):
    optimizer.zero_grad()
    out = model_active(nei_batch)
    loss = criterion(out, nei_batch["paper"].y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')