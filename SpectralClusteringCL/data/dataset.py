"""
Dataset loader for graph continual learning with spectral clustering.
Supports: cora, citeseer, wikics (via HuggingFace), ogbn-arxiv, ogbn-products (via OGB),
          coauthor-cs (via PyG).
All graphs are treated as undirected.
"""

import heapq
import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected
from torch.utils.data import Dataset

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False

# PyG built-in datasets
PYG_DATASETS = {'coauthor-cs', 'amazon-computers'}


class GraphDataset(Dataset):
    """Graph dataset for continual learning with spectral clustering."""

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
        """Load dataset, dispatch to appropriate loader."""
        if self.dataset.startswith('ogbn-'):
            return self._load_ogbn()
        elif self.dataset in PYG_DATASETS:
            return self._load_pyg()
        else:
            return self._load_standard()

    def _load_ogbn(self):
        """Load OGB node property prediction dataset."""
        if not OGB_AVAILABLE:
            raise ImportError(
                "ogb not installed. Install with: pip install ogb"
            )

        print(f"Loading OGB dataset: {self.dataset} ...")
        ogb_dataset = PygNodePropPredDataset(
            name=self.dataset, root=self.data_path
        )
        data = ogb_dataset[0]

        # Labels: squeeze to 1D
        if data.y.dim() > 1:
            data.y = data.y.squeeze(-1)

        # Ensure long type
        data.y = data.y.to(torch.long)

        # Make undirected (ogbn-arxiv is directed)
        edge_index = to_undirected(data.edge_index)

        # Remove isolated nodes
        edge_index, data.x, data.y = self._remove_isolated_nodes(
            edge_index, data.x, data.y
        )

        # Store original edge_index (without self-loops) for neighbor prediction
        self.original_edge_index = edge_index.clone()

        # Add self-loops for message passing
        edge_index_sl, _ = add_self_loops(edge_index)

        new_data = Data(x=data.x, edge_index=edge_index_sl, y=data.y)

        # Build id_by_class and reorder labels (larger classes get smaller indices)
        id_by_class = self._build_and_reorder_classes(new_data)

        self._print_info(new_data, id_by_class)
        return new_data, id_by_class

    def _load_pyg(self):
        """Load PyG built-in dataset (e.g., Coauthor CS, Amazon Computers)."""
        if self.dataset == 'coauthor-cs':
            from torch_geometric.datasets import Coauthor
            print(f"Loading Coauthor CS dataset...")
            pyg_dataset = Coauthor(root=self.data_path, name='CS')
        elif self.dataset == 'amazon-computers':
            from torch_geometric.datasets import Amazon
            print(f"Loading Amazon Computers dataset...")
            pyg_dataset = Amazon(root=self.data_path, name='Computers')
        else:
            raise ValueError(f"Unknown PyG dataset: {self.dataset}")

        data = pyg_dataset[0]
        data.y = data.y.to(torch.long)

        # Already undirected, but ensure
        edge_index = to_undirected(data.edge_index)

        # Remove isolated nodes
        edge_index, data.x, data.y = self._remove_isolated_nodes(
            edge_index, data.x, data.y
        )

        self.original_edge_index = edge_index.clone()

        edge_index_sl, _ = add_self_loops(edge_index)
        new_data = Data(x=data.x, edge_index=edge_index_sl, y=data.y)

        id_by_class = self._build_and_reorder_classes(new_data)
        self._print_info(new_data, id_by_class)
        return new_data, id_by_class

    def _load_standard(self):
        """Load standard dataset (cora, citeseer, wikics) from HuggingFace."""
        path = self.data_path + self.dataset + ".pt"

        try:
            data = torch.load(path, weights_only=False)
        except FileNotFoundError:
            if HF_AVAILABLE:
                print(f"Downloading {self.dataset} from HuggingFace...")
                hf_hub_download(
                    repo_id="YYYumo/LLM4GCL",
                    filename=self.dataset + ".pt",
                    repo_type="dataset",
                    local_dir=self.data_path,
                )
                data = torch.load(path, weights_only=False)
            else:
                raise FileNotFoundError(
                    f"Dataset {self.dataset} not found at {path}. "
                    "Install huggingface_hub to download: pip install huggingface_hub"
                )

        edge_index_raw = data.edge_index

        # Make undirected
        edge_index_raw = to_undirected(edge_index_raw)

        # Remove isolated nodes
        edge_index_raw, data.x, data.y = self._remove_isolated_nodes(
            edge_index_raw, data.x, data.y
        )

        # Store original edge_index (without self-loops)
        self.original_edge_index = edge_index_raw.clone()

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index_raw)

        new_data = Data(x=data.x, edge_index=edge_index, y=data.y)

        # Build id_by_class and reorder labels
        id_by_class = self._build_and_reorder_classes(new_data)

        self._print_info(new_data, id_by_class)
        return new_data, id_by_class

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
        num_nodes = [len(v) for _, v in id_by_class.items()]
        sorted_class_idx = heapq.nlargest(
            len(class_list),
            enumerate(num_nodes),
            key=lambda x: x[1],
        )

        # Create mapping: old_class_id -> new_class_id
        old_classes = list(id_by_class.keys())
        class_mapping = {}
        for new_id, (sorted_idx, _) in enumerate(sorted_class_idx):
            old_id = old_classes[sorted_idx]
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
