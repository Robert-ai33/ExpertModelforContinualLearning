"""
Dataset loader for graph continual learning.
Supports: cora, citeseer, wikics
"""

import copy
import heapq
import torch

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.utils.data import Dataset

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TextDataset(Dataset):
    """Graph dataset with text attributes."""
    
    def __init__(self, dataset, data_path):
        self.dataset = dataset
        self.data_path = data_path
        
        self.data, self.id_by_class = self._load_data()
    
    def __getitem__(self, idx):
        item = {
            'node_id': idx,
            'labels': self.data.y[idx].to(torch.long),
        }
        return item
    
    def __len__(self):
        return self.data.x.shape[0]
    
    def _load_data(self):
        """Load dataset from file or download from HuggingFace."""
        path = self.data_path + self.dataset + ".pt"
        
        # Try to load from local file first
        try:
            data = torch.load(path)
        except FileNotFoundError:
            if HF_AVAILABLE:
                print(f"Downloading {self.dataset} from HuggingFace...")
                hf_hub_download(
                    repo_id="YYYumo/LLM4GCL",
                    filename=self.dataset + ".pt",
                    repo_type="dataset",
                    local_dir=self.data_path
                )
                data = torch.load(path)
            else:
                raise FileNotFoundError(
                    f"Dataset {self.dataset} not found at {path}. "
                    "Install huggingface_hub to download: pip install huggingface_hub"
                )
        
        # Remove isolated nodes (nodes with no edges)
        edge_index_raw = data.edge_index
        num_nodes = data.x.size(0)
        has_edge = torch.zeros(num_nodes, dtype=torch.bool)
        has_edge[edge_index_raw[0].unique()] = True
        has_edge[edge_index_raw[1].unique()] = True
        non_isolated = has_edge

        num_removed = (~non_isolated).sum().item()
        if num_removed > 0:
            print(f"Removing {num_removed} isolated nodes...")

            old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
            old_indices = torch.where(non_isolated)[0]
            old_to_new[old_indices] = torch.arange(old_indices.size(0))

            data.x = data.x[non_isolated]
            data.y = data.y[non_isolated]

            new_src = old_to_new[edge_index_raw[0]]
            new_dst = old_to_new[edge_index_raw[1]]
            valid = (new_src >= 0) & (new_dst >= 0)
            edge_index_raw = torch.stack([new_src[valid], new_dst[valid]])

        # Store original edge_index (without self-loops) for neighbor prediction
        self.original_edge_index = edge_index_raw.clone()
        
        # Add self-loops for GCN
        edge_index, _ = add_self_loops(edge_index_raw)
        
        new_data = Data(
            x=data.x,
            edge_index=edge_index,
            y=data.y,
        )
        data = new_data
        
        # Build id_by_class mapping
        labels = data.y
        class_list = labels.unique().numpy()
        id_by_class = {i: [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)
        
        # Sort classes by number of samples (largest first)
        num_nodes = [len(v) for _, v in id_by_class.items()]
        sorted_class_idx = heapq.nlargest(
            labels.max().item() + 1, 
            enumerate(num_nodes), 
            key=lambda x: x[1]
        )
        
        # Re-order labels so that larger classes have smaller indices
        for i, (class_id, _) in enumerate(sorted_class_idx):
            class_indices = id_by_class[class_id]
            labels[class_indices] = i
        
        # Rebuild id_by_class with new labels
        class_list = labels.unique().numpy()
        id_by_class = {i: [] for i in class_list}
        for idx, cla in enumerate(labels):
            id_by_class[cla.item()].append(idx)
        
        # Print dataset info
        print(f"\n{'='*50}")
        print(f"Dataset: {self.dataset}")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        print(f"  Classes: {data.y.max().item() + 1}")
        print(f"  Feature dim: {data.x.shape[1]}")
        print(f"  Samples per class:")
        for cls in sorted(id_by_class.keys()):
            print(f"    Class {cls}: {len(id_by_class[cls])}")
        print(f"{'='*50}\n")
        
        return data, id_by_class
