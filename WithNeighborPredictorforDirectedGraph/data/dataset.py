"""
Dataset loader for directed expert continual learning.
Supports:
  - wikics: from local wiki-cs-dataset-master/dataset/data.json (has directed edges)
  - ogbn-arxiv: from OGB (has directed edges)

All datasets produce both directed_edge_index and undirected_edge_index.
"""

import os
import json
import heapq
import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected
from torch.utils.data import Dataset

try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False


class GraphDataset(Dataset):
    """Graph dataset with directed and undirected edges."""

    def __init__(self, dataset, data_path):
        self.dataset = dataset
        self.data_path = data_path
        self.data, self.id_by_class = self._load_data()

    def __getitem__(self, idx):
        return {
            'node_id': idx,
            'labels': self.data.y[idx].to(torch.long),
        }

    def __len__(self):
        return self.data.x.shape[0]

    def _load_data(self):
        """Dispatch to appropriate loader based on dataset name."""
        if self.dataset == 'wikics':
            return self._load_wikics()
        elif self.dataset == 'ogbn-arxiv':
            return self._load_ogbn_arxiv()
        else:
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. "
                f"Supported: wikics, ogbn-arxiv"
            )

    # ------------------------------------------------------------------
    # WikiCS loader (from local JSON, has directed edges)
    # ------------------------------------------------------------------
    def _load_wikics(self):
        """Load WikiCS from data.json with directed edges."""
        # Check for processed cache
        cache_path = os.path.join(self.data_path, 'wikics_directed_processed.pt')
        if os.path.exists(cache_path):
            print("Loading processed WikiCS data from cache...")
            cached = torch.load(cache_path, weights_only=False)
            self.directed_edge_index = cached['directed_edge_index']
            self.undirected_edge_index = cached['undirected_edge_index']
            data = Data(
                x=cached['x'],
                edge_index=cached['edge_index_selfloop'],
                y=cached['y'],
            )
            id_by_class = cached['id_by_class']
            self._print_info(data, id_by_class)
            return data, id_by_class

        # Load data.json
        json_path = os.path.join(self.data_path, 'data.json')
        print(f"Loading WikiCS from {json_path} ...")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        features = torch.FloatTensor(np.array(raw['features']))
        labels = torch.LongTensor(np.array(raw['labels']))
        num_nodes = features.size(0)

        # Build directed edge_index from links
        edges = []
        for i, nbs in enumerate(raw['links']):
            for j in nbs:
                edges.append([i, j])

        if len(edges) == 0:
            raise ValueError("No edges found in data.json")

        directed_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Remove isolated nodes
        directed_edge_index, features, labels = self._remove_isolated_nodes(
            directed_edge_index, features, labels
        )

        # Store directed edge_index (no self-loops)
        self.directed_edge_index = directed_edge_index

        # Build undirected edge_index (directed + reverse, deduplicated)
        rev_edge_index = directed_edge_index.flip(0)
        undirected = torch.cat([directed_edge_index, rev_edge_index], dim=1)
        undirected = torch.unique(undirected, dim=1)
        self.undirected_edge_index = undirected

        # Add self-loops for GCN
        undirected_with_selfloop, _ = add_self_loops(undirected)

        data = Data(x=features, edge_index=undirected_with_selfloop, y=labels)

        # Reorder labels by class size
        id_by_class = self._build_and_reorder_classes(data)

        # Save cache
        print(f"Saving processed data to {cache_path} ...")
        torch.save({
            'x': data.x,
            'y': data.y,
            'edge_index_selfloop': data.edge_index,
            'directed_edge_index': self.directed_edge_index,
            'undirected_edge_index': self.undirected_edge_index,
            'id_by_class': id_by_class,
        }, cache_path)

        self._print_info(data, id_by_class)
        return data, id_by_class

    # ------------------------------------------------------------------
    # ogbn-arxiv loader (natively directed)
    # ------------------------------------------------------------------
    def _load_ogbn_arxiv(self):
        """Load ogbn-arxiv from OGB. Has directed edges natively."""
        if not OGB_AVAILABLE:
            raise ImportError(
                "ogb not installed. Install with: pip install ogb"
            )

        print(f"Loading OGB dataset: ogbn-arxiv ...")
        ogb_dataset = PygNodePropPredDataset(
            name='ogbn-arxiv', root=self.data_path
        )
        data = ogb_dataset[0]

        # Labels: squeeze to 1D, ensure long type
        if data.y.dim() > 1:
            data.y = data.y.squeeze(-1)
        data.y = data.y.to(torch.long)

        directed_raw = data.edge_index
        undirected_raw = to_undirected(directed_raw)

        # Find non-isolated nodes (from undirected view)
        num_nodes = data.x.size(0)
        has_edge = torch.zeros(num_nodes, dtype=torch.bool)
        has_edge[undirected_raw[0].unique()] = True
        has_edge[undirected_raw[1].unique()] = True
        non_isolated = has_edge

        num_removed = (~non_isolated).sum().item()
        if num_removed > 0:
            print(f"Removing {num_removed} isolated nodes...")
            old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
            old_indices = torch.where(non_isolated)[0]
            old_to_new[old_indices] = torch.arange(old_indices.size(0))

            data.x = data.x[non_isolated]
            data.y = data.y[non_isolated]

            # Remap directed edges
            d_src = old_to_new[directed_raw[0]]
            d_dst = old_to_new[directed_raw[1]]
            d_valid = (d_src >= 0) & (d_dst >= 0)
            directed_ei = torch.stack([d_src[d_valid], d_dst[d_valid]])

            # Remap undirected edges
            u_src = old_to_new[undirected_raw[0]]
            u_dst = old_to_new[undirected_raw[1]]
            u_valid = (u_src >= 0) & (u_dst >= 0)
            undirected_ei = torch.stack([u_src[u_valid], u_dst[u_valid]])
        else:
            directed_ei = directed_raw
            undirected_ei = undirected_raw

        self.directed_edge_index = directed_ei
        self.undirected_edge_index = undirected_ei

        # Add self-loops for GCN (on undirected)
        edge_index_sl, _ = add_self_loops(undirected_ei)
        new_data = Data(x=data.x, edge_index=edge_index_sl, y=data.y)

        # Build id_by_class and reorder labels
        id_by_class = self._build_and_reorder_classes(new_data)

        self._print_info(new_data, id_by_class)
        return new_data, id_by_class

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _remove_isolated_nodes(edge_index, x, y):
        """Remove nodes with no edges."""
        num_nodes = x.size(0)
        has_edge = torch.zeros(num_nodes, dtype=torch.bool)
        has_edge[edge_index[0].unique()] = True
        has_edge[edge_index[1].unique()] = True
        non_isolated = has_edge

        num_removed = (~non_isolated).sum().item()
        if num_removed > 0:
            print(f"Removing {num_removed} isolated nodes...")
            old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
            old_indices = torch.where(non_isolated)[0]
            old_to_new[old_indices] = torch.arange(old_indices.size(0))

            x = x[non_isolated]
            y = y[non_isolated]

            new_src = old_to_new[edge_index[0]]
            new_dst = old_to_new[edge_index[1]]
            valid = (new_src >= 0) & (new_dst >= 0)
            edge_index = torch.stack([new_src[valid], new_dst[valid]])

        return edge_index, x, y

    @staticmethod
    def _build_and_reorder_classes(data):
        """Build id_by_class mapping, reorder labels so larger classes get smaller indices."""
        labels = data.y
        class_list = labels.unique().numpy()
        id_by_class = {int(i): [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)

        # Sort by class size (largest first)
        class_counts = [(cls, len(nodes)) for cls, nodes in id_by_class.items()]
        sorted_class_counts = sorted(class_counts, key=lambda x: x[1], reverse=True)

        # Create mapping: old_class_id -> new_class_id
        class_mapping = {}
        for new_id, (old_id, _) in enumerate(sorted_class_counts):
            class_mapping[old_id] = new_id

        # Apply remapping
        for old_id, new_id in class_mapping.items():
            class_indices = id_by_class[old_id]
            labels[class_indices] = new_id

        # Rebuild id_by_class
        class_list = labels.unique().numpy()
        id_by_class = {int(i): [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)

        return id_by_class

    def _print_info(self, data, id_by_class):
        """Print dataset statistics."""
        print(f"\n{'='*50}")
        print(f"Dataset: {self.dataset}")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Directed edges: {self.directed_edge_index.shape[1]}")
        print(f"  Undirected edges: {self.undirected_edge_index.shape[1]}")
        print(f"  Edges (with self-loops): {data.edge_index.shape[1]}")
        print(f"  Classes: {data.y.max().item() + 1}")
        print(f"  Feature dim: {data.x.shape[1]}")
        print(f"  Samples per class:")
        for cls in sorted(id_by_class.keys()):
            print(f"    Class {cls}: {len(id_by_class[cls])}")
        print(f"{'='*50}\n")
