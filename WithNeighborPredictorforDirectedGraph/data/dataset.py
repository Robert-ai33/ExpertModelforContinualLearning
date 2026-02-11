"""
WikiCS dataset loader for directed expert continual learning.
Loads from wiki-cs-dataset-master/dataset/data.json.
Produces both directed and undirected edge indices.
Removes isolated nodes.
"""

import os
import json
import heapq
import itertools
import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.utils.data import Dataset


class WikiCSDataset(Dataset):
    """WikiCS graph dataset with directed and undirected edges."""

    def __init__(self, data_path):
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
        """Load WikiCS dataset, remove isolated nodes, build directed/undirected edges."""

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

        # Remove isolated nodes (in_degree + out_degree == 0)
        has_edge = torch.zeros(num_nodes, dtype=torch.bool)
        has_edge[directed_edge_index[0].unique()] = True
        has_edge[directed_edge_index[1].unique()] = True
        non_isolated = has_edge

        num_removed = (~non_isolated).sum().item()
        if num_removed > 0:
            print(f"Removing {num_removed} isolated nodes...")

            old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
            old_indices = torch.where(non_isolated)[0]
            old_to_new[old_indices] = torch.arange(old_indices.size(0))

            features = features[non_isolated]
            labels = labels[non_isolated]

            new_src = old_to_new[directed_edge_index[0]]
            new_dst = old_to_new[directed_edge_index[1]]
            valid = (new_src >= 0) & (new_dst >= 0)
            directed_edge_index = torch.stack([new_src[valid], new_dst[valid]])

        # Store directed edge_index (no self-loops)
        self.directed_edge_index = directed_edge_index

        # Build undirected edge_index (directed + reverse, deduplicated)
        rev_edge_index = directed_edge_index.flip(0)
        undirected = torch.cat([directed_edge_index, rev_edge_index], dim=1)
        undirected = torch.unique(undirected, dim=1)
        self.undirected_edge_index = undirected

        # Add self-loops for GCN
        undirected_with_selfloop, _ = add_self_loops(undirected)

        # Create Data object
        data = Data(
            x=features,
            edge_index=undirected_with_selfloop,
            y=labels,
        )

        # Reorder labels by class size (larger class -> smaller index)
        labels = data.y
        class_list = labels.unique().numpy()
        id_by_class = {int(i): [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)

        class_counts = [(cls, len(nodes)) for cls, nodes in id_by_class.items()]
        sorted_class_counts = sorted(class_counts, key=lambda x: x[1], reverse=True)

        for new_label, (old_label, _) in enumerate(sorted_class_counts):
            class_indices = id_by_class[old_label]
            labels[class_indices] = new_label

        # Rebuild id_by_class with new labels
        class_list = labels.unique().numpy()
        id_by_class = {int(i): [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)

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

    def _print_info(self, data, id_by_class):
        """Print dataset statistics."""
        print(f"\n{'='*50}")
        print(f"Dataset: WikiCS (Directed)")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Directed edges: {self.directed_edge_index.shape[1]}")
        print(f"  Undirected edges: {self.undirected_edge_index.shape[1]}")
        print(f"  Classes: {data.y.max().item() + 1}")
        print(f"  Feature dim: {data.x.shape[1]}")
        print(f"  Samples per class:")
        for cls in sorted(id_by_class.keys()):
            print(f"    Class {cls}: {len(id_by_class[cls])}")
        print(f"{'='*50}\n")
