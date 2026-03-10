"""
Dataset loader for CommunityExpertCL.
Supports: cora, citeseer, coauthor-cs, amazon-computers (all via PyG).
All graphs are undirected.
"""

import heapq
import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected
from torch.utils.data import Dataset


SUPPORTED_DATASETS = {'cora', 'citeseer', 'coauthor-cs', 'amazon-computers'}


class GraphDataset(Dataset):
    """Undirected graph dataset for community-expert continual learning."""

    def __init__(self, dataset, data_path):
        self.dataset = dataset
        self.data_path = data_path
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Supported: {SUPPORTED_DATASETS}"
            )
        self.data, self.id_by_class = self._load_data()

    def __getitem__(self, idx):
        return {
            'node_id': idx,
            'labels': self.data.y[idx].to(torch.long),
        }

    def __len__(self):
        return self.data.x.shape[0]

    def _load_data(self):
        if self.dataset in ('cora', 'citeseer'):
            return self._load_planetoid()
        elif self.dataset == 'coauthor-cs':
            return self._load_coauthor()
        elif self.dataset == 'amazon-computers':
            return self._load_amazon()

    def _load_planetoid(self):
        from torch_geometric.datasets import Planetoid
        name_map = {'cora': 'Cora', 'citeseer': 'CiteSeer'}
        print(f"Loading {self.dataset} from PyG Planetoid...")
        pyg_dataset = Planetoid(root=self.data_path, name=name_map[self.dataset])
        return self._process(pyg_dataset[0])

    def _load_coauthor(self):
        from torch_geometric.datasets import Coauthor
        print("Loading Coauthor CS from PyG...")
        pyg_dataset = Coauthor(root=self.data_path, name='CS')
        return self._process(pyg_dataset[0])

    def _load_amazon(self):
        from torch_geometric.datasets import Amazon
        print("Loading Amazon Computers from PyG...")
        pyg_dataset = Amazon(root=self.data_path, name='Computers')
        return self._process(pyg_dataset[0])

    def _process(self, data):
        """Common processing pipeline: undirected, remove isolated, self-loops."""
        data.y = data.y.to(torch.long)
        if data.y.dim() > 1:
            data.y = data.y.squeeze(-1)

        edge_index = to_undirected(data.edge_index)
        edge_index, x, y = self._remove_isolated_nodes(edge_index, data.x, data.y)

        self.original_edge_index = edge_index.clone()

        edge_index_sl, _ = add_self_loops(edge_index)
        new_data = Data(x=x, edge_index=edge_index_sl, y=y)

        id_by_class = self._build_and_reorder_classes(new_data)
        self._print_info(new_data, id_by_class)
        return new_data, id_by_class

    @staticmethod
    def _remove_isolated_nodes(edge_index, x, y):
        num_nodes = x.size(0)
        has_edge = torch.zeros(num_nodes, dtype=torch.bool)
        has_edge[edge_index[0].unique()] = True
        has_edge[edge_index[1].unique()] = True

        num_removed = (~has_edge).sum().item()
        if num_removed > 0:
            print(f"Removing {num_removed} isolated nodes...")
            old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
            old_indices = torch.where(has_edge)[0]
            old_to_new[old_indices] = torch.arange(old_indices.size(0))

            x = x[has_edge]
            y = y[has_edge]

            new_src = old_to_new[edge_index[0]]
            new_dst = old_to_new[edge_index[1]]
            valid = (new_src >= 0) & (new_dst >= 0)
            edge_index = torch.stack([new_src[valid], new_dst[valid]])

        return edge_index, x, y

    @staticmethod
    def _build_and_reorder_classes(data):
        """Reorder labels so larger classes get smaller indices."""
        labels = data.y
        class_list = labels.unique().numpy()
        id_by_class = {int(i): [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)

        num_nodes = [len(v) for v in id_by_class.values()]
        sorted_class_idx = heapq.nlargest(
            len(class_list), enumerate(num_nodes), key=lambda x: x[1]
        )

        old_classes = list(id_by_class.keys())
        class_mapping = {}
        for new_id, (sorted_idx, _) in enumerate(sorted_class_idx):
            class_mapping[old_classes[sorted_idx]] = new_id

        for old_id, new_id in class_mapping.items():
            labels[id_by_class[old_id]] = new_id

        class_list = labels.unique().numpy()
        id_by_class = {int(i): [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)

        return id_by_class

    def _print_info(self, data, id_by_class):
        print(f"\n{'='*50}")
        print(f"Dataset: {self.dataset} (undirected)")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges (with self-loops): {data.edge_index.shape[1]}")
        print(f"  Edges (no self-loops): {self.original_edge_index.shape[1]}")
        print(f"  Classes: {data.y.max().item() + 1}")
        print(f"  Feature dim: {data.x.shape[1]}")
        print(f"  Samples per class:")
        for cls in sorted(id_by_class.keys()):
            print(f"    Class {cls}: {len(id_by_class[cls])}")
        print(f"{'='*50}\n")
