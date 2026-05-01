"""
TextGraphDataset: text-attributed graph loader for the SimGCL baseline.

This is a *separate* dataset class from ``GraphDataset`` -- by design, it is
only used by LLM-based methods (currently only ``simgcl``) so that adding
text plumbing does not perturb any of the feature-only baselines.

Implementation notes:
- Source data is the LLM4GCL Hugging Face release (``YYYumo/LLM4GCL``),
  which packs ``raw_texts`` directly into the ``.pt`` file along with
  ``x``, ``y``, ``edge_index``.
- The class re-ordering rule (largest class -> smallest index) is identical
  to ``GraphDataset._build_and_reorder_classes``, so downstream consumers
  (TaskLoader, EXP_SETTINGS class_splits) see the same numbering convention.
- Public surface mimics ``GraphDataset``:
    self.data            : torch_geometric.data.Data
                           (x, edge_index w/ self-loops, y,
                            raw_texts: List[str], label_texts: List[str])
    self.id_by_class     : dict[int, list[int]]
    self.original_edge_index : Tensor (no self-loops)
- ``__getitem__`` returns a dict that satisfies *both* the existing
  GraphDataset consumers (uses node_id, labels) AND the LLM4GCL prompt
  builder (uses raw_text, label_text).

Dataset-name mapping to the HF release:
    cora            -> cora
    citeseer        -> citeseer
    wikics          -> wikics
    ogbn-arxiv      -> arxiv
    ogbn-products   -> products

Notes for ``ogbn-products`` and ``ogbn-arxiv``:
LLM4GCL applies a label-filtering step that drops some classes and remaps
indices. The resulting (node count, class count, class numbering) therefore
differs from PyG's raw OGB. Class splits in ``EXP_SETTINGS`` are still
applied verbatim, but they refer to a different underlying class identity
than when running feature-only baselines on the same dataset. For
publication-quality direct comparisons, restrict to cora / citeseer /
wikics where both pipelines see the same graph.
"""

import copy
import heapq
import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


SUPPORTED_TEXT_DATASETS = {
    'cora', 'citeseer', 'wikics', 'ogbn-arxiv', 'ogbn-products',
}

_HF_NAME_MAP = {
    'cora': 'cora',
    'citeseer': 'citeseer',
    'wikics': 'wikics',
    'ogbn-arxiv': 'arxiv',
    'ogbn-products': 'products',
}

# Datasets that LLM4GCL filters down before saving (delete_label / empty_label
# in their data_loader). We must apply the same surgery so node indices, edge
# index and labels stay consistent.
_LLM4GCL_FILTER = {
    'products': {
        'empty_label': [29, 33],
        'delete_label': [22, 26, 27, 30, 34, 35, 38, 39, 40, 41, 43],
    },
}

_LABEL_TEXT_JSON = os.path.join(os.path.dirname(__file__), 'label_text.json')


class TextGraphDataset(Dataset):
    """Text-attributed graph dataset, drop-in for ``GraphDataset`` consumers.

    Args:
        dataset: dataset name, must be in ``SUPPORTED_TEXT_DATASETS``.
        data_path: directory where the HF ``.pt`` will be cached.

    Raises:
        ValueError: if ``dataset`` has no associated text release.
    """

    def __init__(self, dataset, data_path):
        if dataset not in SUPPORTED_TEXT_DATASETS:
            raise ValueError(
                f"TextGraphDataset does not support '{dataset}'. "
                f"Datasets without natural raw text (coauthor-cs, "
                f"amazon-computers, cora-full) cannot be used with "
                f"LLM-based methods. "
                f"Supported: {sorted(SUPPORTED_TEXT_DATASETS)}"
            )

        self.dataset = dataset
        self.data_path = data_path
        self.data, self.id_by_class = self._load_data()
        self.raw_texts = self.data.raw_texts
        self.label_texts = self.data.label_texts

    def __getitem__(self, idx):
        return {
            'node_id': idx,
            'labels': self.data.y[idx].to(torch.long),
            'raw_text': self.raw_texts[idx],
            'label_text': self.label_texts[int(self.data.y[idx])],
        }

    def __len__(self):
        return self.data.x.shape[0]

    def _load_data(self):
        hf_name = _HF_NAME_MAP[self.dataset]
        data, label_texts = _download_and_load_hf(
            hf_name, self.data_path, _LABEL_TEXT_JSON)
        data = _apply_llm4gcl_filter(hf_name, data)
        data, id_by_class, label_texts = _reorder_classes_by_size(
            data, label_texts)

        edge_index_no_sl = data.edge_index.clone()
        edge_index_sl, _ = add_self_loops(edge_index_no_sl)
        new_data = Data(
            x=data.x,
            edge_index=edge_index_sl,
            y=data.y,
            raw_texts=data.raw_texts,
            label_texts=label_texts,
        )

        # Mirror GraphDataset surface: ``original_edge_index`` is the
        # no-self-loop variant (used by TaskLoader._create_task_subgraph).
        self.original_edge_index = edge_index_no_sl

        _print_info(self.dataset, new_data, id_by_class)
        return new_data, id_by_class


# ----------------------------------------------------------------------
# Helpers (free functions to keep the class body short)
# ----------------------------------------------------------------------

def _download_and_load_hf(hf_name, data_path, label_text_json_path):
    """Download ``<hf_name>.pt`` from LLM4GCL HF and load the label text JSON."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "TextGraphDataset requires huggingface_hub. "
            "Install with: pip install huggingface_hub"
        ) from e

    import json

    os.makedirs(data_path, exist_ok=True)
    cached = os.path.join(data_path, f"{hf_name}.pt")
    if not os.path.exists(cached):
        print(f"Downloading {hf_name}.pt from HuggingFace 'YYYumo/LLM4GCL' ...")
        hf_hub_download(
            repo_id="YYYumo/LLM4GCL",
            filename=f"{hf_name}.pt",
            repo_type="dataset",
            local_dir=data_path,
        )
    else:
        print(f"Using cached {cached}")

    data = torch.load(cached, weights_only=False)

    with open(label_text_json_path, 'r', encoding='utf-8') as f:
        label_texts = json.load(f)[hf_name]

    return data, label_texts


def _apply_llm4gcl_filter(hf_name, data):
    """Apply LLM4GCL's label-deletion / re-mapping for products/arxiv_23.

    We currently only need ``products`` (LLM4GCL's ``arxiv_23`` is not in our
    supported set; OGB's ``arxiv`` requires no filtering).
    """
    if hf_name not in _LLM4GCL_FILTER:
        return data

    spec = _LLM4GCL_FILTER[hf_name]
    empty_label = spec['empty_label']
    delete_label = spec['delete_label']

    mask = ~torch.isin(data.y, torch.tensor(delete_label))
    to_remove = (~mask).nonzero(as_tuple=True)[0]
    remaining = torch.arange(data.x.size(0))[
        ~torch.isin(torch.arange(data.x.size(0)), to_remove)
    ]

    edge_mask = (
        ~torch.isin(data.edge_index[0], to_remove)
        & ~torch.isin(data.edge_index[1], to_remove)
    )
    data.edge_index = data.edge_index[:, edge_mask]

    node_map = {old.item(): new for new, old in enumerate(remaining)}
    src = []
    dst = []
    for i in range(data.edge_index.size(1)):
        s = data.edge_index[0, i].item()
        d = data.edge_index[1, i].item()
        if s in node_map and d in node_map:
            src.append(node_map[s])
            dst.append(node_map[d])
    data.edge_index = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0)

    data.x = data.x[mask]
    data.raw_texts = [data.raw_texts[i] for i in range(len(data.raw_texts)) if mask[i]]

    delete_extended = sorted(delete_label + empty_label)
    new_labels = []
    for label in data.y[mask].tolist():
        new_labels.append(label - sum(label > x for x in delete_extended))
    data.y = torch.tensor(new_labels, dtype=torch.long)
    data.num_nodes = data.x.size(0)
    return data


def _reorder_classes_by_size(data, label_texts):
    """Reorder classes so the largest gets index 0; matches GraphDataset."""
    labels = data.y.clone().to(torch.long)
    class_list = labels.unique().numpy()
    id_by_class = {int(i): [] for i in class_list}
    for idx, cla in enumerate(labels):
        id_by_class[int(cla)].append(idx)

    num_per_class = [len(id_by_class[c]) for c in class_list]
    sorted_class_idx = heapq.nlargest(
        len(class_list), enumerate(num_per_class), key=lambda x: x[1]
    )

    old_classes = list(id_by_class.keys())
    sorted_label_texts = copy.deepcopy(label_texts)
    new_labels = labels.clone()
    new_id_by_class = {}
    class_mapping = {}
    for new_id, (sorted_idx, _) in enumerate(sorted_class_idx):
        old_id = old_classes[sorted_idx]
        class_mapping[old_id] = new_id
        new_id_by_class[new_id] = id_by_class[old_id]
        new_labels[id_by_class[old_id]] = new_id
        if sorted_idx < len(label_texts):
            sorted_label_texts[new_id] = label_texts[sorted_idx]

    data.y = new_labels
    return data, new_id_by_class, sorted_label_texts


def _print_info(dataset, data, id_by_class):
    print(f"\n{'='*50}")
    print(f"Text Dataset: {dataset}")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Edges (with self-loops): {data.edge_index.shape[1]}")
    print(f"  Classes: {data.y.max().item() + 1}")
    print(f"  Feature dim: {data.x.shape[1]}")
    print(f"  raw_texts: {len(data.raw_texts)}")
    print(f"  label_texts: {len(data.label_texts)}")
    print(f"  Samples per class:")
    for cls in sorted(id_by_class.keys()):
        print(f"    Class {cls}: {len(id_by_class[cls])}")
    print(f"{'='*50}\n")
