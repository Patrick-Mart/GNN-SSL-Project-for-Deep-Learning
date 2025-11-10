import torch
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader


hetero_data = OGB_MAG('../')[0]

loader = NeighborLoader(
    hetero_data,
    # Sample 30 neighbors for each node and edge type for 2 iterations
    num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
    # Use a batch size of 128 for sampling training nodes of type paper
    batch_size=128,
    input_nodes=('paper', hetero_data['paper'].train_mask),
)

sampled_hetero_data = next(iter(loader))
print(sampled_hetero_data['paper'].batch_size)

print(torch.__version__)

# print(pyg_lib.__version__)


