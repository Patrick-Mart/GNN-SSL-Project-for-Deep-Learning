import torch
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import DataLoader 

batch_size = 16

data = OGB_MAG('../')
loader = DataLoader(data, batch_size=batch_size, shuffle=True)

for batch in loader:
    print(batch)

