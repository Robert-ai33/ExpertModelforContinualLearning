"""
TextTaskLoader: LLM-aware task loader that wraps the existing ``TaskLoader``.

Design principle: do not touch the feature-only ``TaskLoader``; instead
*compose* it and expose an 8-tuple ``get_task()`` that matches the SimGCL
(and other LLM4GCL GLM methods) calling convention. The CL scenario --
class splits, train/valid/test indices, CGLB cumulative subgraphs -- is
inherited verbatim from the user's pipeline; only ``raw_texts`` /
``label_texts`` and a few API renamings are layered on top.

Returned tuple from ``get_task(task_id, subset=-1)``::

    (class_src, class_dst,
     text_dataset_iso, text_dataset_joint,
     train_loader, valid_loader,
     test_loader_isolate, test_loader_joint)

For SimGCL's purposes (per-node prompts on a cumulative graph), both
``text_dataset_iso`` and ``text_dataset_joint`` are session-k cumulative
graph wrappers -- evaluation always operates on the same CGLB-aligned
subgraph the user's other baselines see. They are kept as separate slots
only to honour the upstream API.
"""

import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data

from .task_loader import TaskLoader


class TextTaskLoader:
    """LLM-aware adapter over ``TaskLoader``."""

    def __init__(self, batch_size, text_graph_dataset, class_splits,
                 split_S, split_t, split_v):
        self.batch_size = batch_size
        self.text_graph_dataset = text_graph_dataset
        self.data = text_graph_dataset.data

        # Reuse user's TaskLoader for CGLB index splits + subgraphs.
        self._inner = TaskLoader(
            batch_size=batch_size,
            graph_dataset=text_graph_dataset,
            class_splits=class_splits,
            split_S=split_S,
            split_t=split_t,
            split_v=split_v,
        )

        # Re-export attributes that downstream code (SimGCLBaseline) reads.
        self.train_idx_per_task = self._inner.train_idx_per_task
        self.valid_idx_per_task = self._inner.valid_idx_per_task
        self.test_idx_per_task = self._inner.test_idx_per_task
        self.test_idx_joint = self._inner.test_idx_joint
        self.subgraph_per_task = self._inner.subgraph_per_task
        self.subgraph_isolated = self._inner.subgraph_isolated
        self.subgraph_joint = self._inner.subgraph_joint
        self.sessions = self._inner.sessions
        self.class_splits = self._inner.class_splits
        self.all_classes = self._inner.all_classes
        self.id_by_class = self._inner.id_by_class

        # Per-session text dataset wrappers (cumulative subgraph topology).
        # A wrapper differs from the parent only by ``data.edge_index`` so
        # that the SimGCL prompt builder sees CGLB-cumulative neighbours.
        self._session_wrappers = self._build_session_wrappers()

    # ------------------------------------------------------------------
    # SimGCL-facing API
    # ------------------------------------------------------------------

    def get_task(self, task_id, subset=-1):
        """8-tuple matching LLM4GCL.SimGCL's call signature."""
        if task_id >= self.sessions:
            raise ValueError(
                f"task_id {task_id} >= total sessions {self.sessions}")

        train_idx = list(self.train_idx_per_task[task_id])
        valid_idx = list(self.valid_idx_per_task[task_id])
        test_idx_isolate = list(self.test_idx_per_task[task_id])
        test_idx_joint = list(self.test_idx_joint[task_id])

        if subset is not None and subset != -1:
            train_idx = self._stratified_sample(
                train_idx, self.data.y[train_idx], subset)
            valid_idx = self._stratified_sample(
                valid_idx, self.data.y[valid_idx], subset)

        text_dataset = self._session_wrappers[task_id]

        train_loader = DataLoader(
            Subset(text_dataset, train_idx),
            batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(
            Subset(text_dataset, valid_idx),
            batch_size=self.batch_size, shuffle=False)
        test_loader_isolate = DataLoader(
            Subset(text_dataset, test_idx_isolate),
            batch_size=self.batch_size, shuffle=False)
        test_loader_joint = DataLoader(
            Subset(text_dataset, test_idx_joint),
            batch_size=self.batch_size, shuffle=False)

        # class_src / class_dst follow SimGCL's NCIL convention:
        # class_src = first class added in this session (after sort)
        # class_dst = exclusive upper bound of all classes seen so far
        sorted_curr = sorted(self.class_splits[task_id])
        class_src = int(sorted_curr[0])
        cumulative = sorted(
            set(c for k in range(task_id + 1) for c in self.class_splits[k]))
        class_dst = int(cumulative[-1]) + 1

        # text_dataset_iso and text_dataset_joint are intentionally the same
        # cumulative wrapper: under the user's CGLB protocol every prompt /
        # eval operates on the cumulative subgraph for session task_id, so
        # there is no separate "isolated" topology in this pipeline.
        return (class_src, class_dst,
                text_dataset, text_dataset,
                train_loader, valid_loader,
                test_loader_isolate, test_loader_joint)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_session_wrappers(self):
        """Per-session shallow copies of ``text_graph_dataset`` whose
        ``data.edge_index`` mirrors the CGLB cumulative subgraph for that
        session. ``raw_texts`` / ``label_texts`` / ``y`` are full-graph
        tensors so that ``Subset(..., test_idx_joint)`` indexing keeps
        working with the original node IDs.
        """
        wrappers = []
        base = self.text_graph_dataset
        for k in range(self.sessions):
            sub = self._inner.subgraph_joint[k]
            new_data = Data(
                x=base.data.x,
                edge_index=sub['edge_index'],
                y=base.data.y,
                raw_texts=base.data.raw_texts,
                label_texts=base.data.label_texts,
            )
            wrapper = copy.copy(base)
            wrapper.data = new_data
            wrapper.raw_texts = base.raw_texts
            wrapper.label_texts = base.label_texts
            wrappers.append(wrapper)
        return wrappers

    @staticmethod
    def _stratified_sample(indices, labels, n_samples):
        """LLM4GCL-style stratified subsampling (pulled from data_spliter.py)."""
        label_to_indices = defaultdict(list)
        for idx, label in zip(indices, labels):
            label_to_indices[label.item()].append(idx)

        unique_labels = list(label_to_indices.keys())
        label_counts = [len(label_to_indices[l]) for l in unique_labels]
        total_label = sum(label_counts) if label_counts else 1
        proportions = np.array(label_counts) / total_label
        samples_per_label = (proportions * n_samples).astype(int)
        samples_per_label = np.maximum(samples_per_label, 1)
        total = int(samples_per_label.sum())

        while total > n_samples:
            max_idx = int(np.argmax(samples_per_label))
            if samples_per_label[max_idx] > 1:
                samples_per_label[max_idx] -= 1
                total -= 1
            else:
                break

        sampled = []
        for i, label in enumerate(unique_labels):
            sample_size = int(samples_per_label[i])
            pool = label_to_indices[label]
            sampled.extend(random.sample(pool, min(sample_size, len(pool))))
        return sampled
