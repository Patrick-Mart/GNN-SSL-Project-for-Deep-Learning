import torch
from torch_geometric.data import HeteroData
import os
from typing import Tuple
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import OGB_MAG
import torch_geometric.transforms as T

def to_inductive(data: HeteroData, node_type: str) -> HeteroData:
    """
    A function that removes all val/test node features and edges between train nodes and val/test nodes.

    """
    train_mask = data[node_type].train_mask
    train_mask_idxs = torch.where(train_mask)[0]
    N_train = len(train_mask_idxs)

    # define new edge index
    new_paper_idxs = torch.full((len(train_mask),), -1, device=train_mask.device)
    new_paper_idxs[train_mask] = torch.arange(N_train, device=train_mask.device)

    # restrict node_type to only include train split
    data[node_type].x = data[node_type].x[train_mask]
    data[node_type].y = data[node_type].y[train_mask]
    data[node_type].year = data[node_type].year[train_mask]
    data[node_type].train_mask = torch.ones((N_train), dtype=torch.bool, device=train_mask.device)
    data[node_type].val_mask = torch.zeros((N_train), dtype=torch.bool, device=train_mask.device)
    data[node_type].test_mask = torch.zeros((N_train), dtype=torch.bool, device=train_mask.device)

    # find edges with node_type as either source or destination
    edge_types = list(data.edge_index_dict.keys())
    edge_type_mask = [(e[0] == node_type, e[-1] == node_type) for e in edge_types]

    edge_index_dict = data.edge_index_dict

    for i, edge_type in enumerate(edge_types):
        if not any(edge_type_mask[i]):
            continue

        edge_index = edge_index_dict[edge_type]
        src_mask = torch.ones((edge_index.size(1)), dtype=bool)
        dst_mask = torch.ones((edge_index.size(1)), dtype=bool)

        # mask paper nodes in edge index not part of train
        if edge_type[0] == node_type:
            src_mask = new_paper_idxs[edge_index[0]] != -1

        if edge_type[-1] == node_type:
            dst_mask = new_paper_idxs[edge_index[1]] != -1

        edge_mask = src_mask & dst_mask
        filtered_edge_index = edge_index[:, edge_mask]

        if edge_type[0] == node_type:
            filtered_edge_index[0] = new_paper_idxs[filtered_edge_index[0]]

        if edge_type[-1] == node_type:
            filtered_edge_index[1] = new_paper_idxs[filtered_edge_index[1]]

        data[edge_type]['edge_index'] = filtered_edge_index

    return data


class MAG240MInductiveDataset:
    def __init__(self, root: str = "../data"):
        root = os.path.expanduser(root)
        dataset = OGB_MAG(root="..data/", transform = T.ToUndirected())[0]
        self.data = dataset
        #self.split_idx = dataset.get_idx_split()

        # Create inductive version (only train papers + their subgraph)
        print("Creating inductive split (removing val/test papers and cross edges)...")
        self.inductive_data = to_inductive(self.data.clone(), node_type='paper')

    def get_data(self) -> Tuple[HeteroData, dict]:
        return self.inductive_data #, self.split_idx  # split_idx kept for reference