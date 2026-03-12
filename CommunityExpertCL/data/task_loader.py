"""
Task loader for CommunityExpertCL.

Subgraph construction:
- Training subgraph: target nodes + external neighbors from seen classes only (current + past)
- Joint test subgraph: all seen classes, external neighbors restricted to seen classes only

Data splitting per class:
- train: ceil(N * t / S)
- valid: ceil(N * v / S), capped by remaining
- test: remaining nodes
- If only 1 node, goes to train
"""

import math
import random
import torch

from torch.utils.data import Subset, DataLoader


class TaskLoader:
    """Task loader with ratio-based train/valid splitting."""

    def __init__(self, batch_size, graph_dataset, class_splits,
                 split_S, split_t, split_v):
        self.batch_size = batch_size
        self.graph_dataset = graph_dataset
        self.data = graph_dataset.data
        self.id_by_class = graph_dataset.id_by_class
        self.original_edge_index = graph_dataset.original_edge_index

        self.class_splits = class_splits
        self.sessions = len(class_splits)
        self.split_S = split_S
        self.split_t = split_t
        self.split_v = split_v

        self.all_classes = sorted(
            set(c for split in class_splits for c in split)
        )

        self._split_data()

        print(f"TaskLoader initialized:")
        print(f"  Sessions: {self.sessions}")
        print(f"  Class splits: {self.class_splits}")
        print(f"  Split ratio: t/S={split_t}/{split_S}, v/S={split_v}/{split_S}")

    def _compute_split(self, node_num):
        """Compute train/valid/test counts from ratio-based splitting."""
        S, t, v = self.split_S, self.split_t, self.split_v

        if node_num == 0:
            return 0, 0, 0
        if node_num == 1:
            return 1, 0, 0

        train_num = math.ceil(node_num * t / S)
        remaining = node_num - train_num
        valid_num = min(math.ceil(node_num * v / S), remaining)
        test_num = remaining - valid_num

        if test_num < 0:
            test_num = 0
            valid_num = remaining

        return train_num, valid_num, test_num

    def _split_data(self):
        """Split data into train/valid/test for each session."""
        self.train_idx_per_task = []
        self.valid_idx_per_task = []
        self.test_idx_per_task = []
        self.test_idx_joint = []

        self.subgraph_per_task = []
        self.subgraph_joint = []
        self.subgraph_isolated = []

        cumulative_classes = []

        for session_id, classes in enumerate(self.class_splits):
            train_idx = []
            valid_idx = []
            test_idx = []

            for cla in classes:
                if cla not in self.id_by_class:
                    print(f"Warning: Class {cla} not found, skipping...")
                    continue

                node_idx = self.id_by_class[cla].copy()
                node_num = len(node_idx)
                train_num, valid_num, test_num = self._compute_split(node_num)

                random.shuffle(node_idx)
                train_idx.extend(node_idx[:train_num])
                valid_idx.extend(node_idx[train_num:train_num + valid_num])
                test_idx.extend(node_idx[train_num + valid_num:
                                         train_num + valid_num + test_num])

            self.train_idx_per_task.append(train_idx)
            self.valid_idx_per_task.append(valid_idx)
            self.test_idx_per_task.append(test_idx)

            if session_id == 0:
                self.test_idx_joint.append(test_idx.copy())
            else:
                self.test_idx_joint.append(self.test_idx_joint[-1] + test_idx)

            cumulative_classes.extend(classes)

            # Training subgraph: external neighbors from seen classes only
            curr_subgraph = self._create_task_subgraph(
                classes, allowed_external_classes=list(cumulative_classes)
            )
            self.subgraph_per_task.append(curr_subgraph)

            # Isolated subgraph: NO external neighbors (only this task's classes)
            isolated_subgraph = self._create_task_subgraph(
                classes, allowed_external_classes=[]
            )
            self.subgraph_isolated.append(isolated_subgraph)

            # Joint test subgraph: external neighbors restricted to seen classes
            joint_subgraph = self._create_task_subgraph(
                list(cumulative_classes),
                allowed_external_classes=list(cumulative_classes)
            )
            self.subgraph_joint.append(joint_subgraph)

            print(f"  Session {session_id}: classes={classes}, "
                  f"train={len(train_idx)}, valid={len(valid_idx)}, "
                  f"test={len(test_idx)}, "
                  f"subgraph_nodes={len(curr_subgraph['all_nodes'])}, "
                  f"isolated_nodes={len(isolated_subgraph['all_nodes'])}, "
                  f"joint_nodes={len(joint_subgraph['all_nodes'])}")

    def _create_task_subgraph(self, class_ids, allowed_external_classes=None):
        """
        Create task subgraph.

        Args:
            class_ids: target class IDs
            allowed_external_classes: if None, all external neighbors allowed;
                otherwise only nodes from these classes

        Returns dict with target_nodes, external_neighbors, all_nodes,
        edge_index (with self-loops), edge_index_no_selfloop, x, y.
        """
        target_idx = []
        for cls in class_ids:
            if cls in self.id_by_class:
                target_idx.extend(self.id_by_class[cls])
        target_idx_set = set(target_idx)

        num_nodes = len(self.data.y)
        target_mask = torch.zeros(num_nodes, dtype=torch.bool)
        target_mask[list(target_idx_set)] = True

        edge_index = self.original_edge_index
        src, dst = edge_index[0], edge_index[1]

        edges_from_target = target_mask[src]
        edges_to_target = target_mask[dst]
        connected_edges = edges_from_target | edges_to_target

        neighbor_nodes = set()
        for i in torch.where(connected_edges)[0].tolist():
            neighbor_nodes.add(src[i].item())
            neighbor_nodes.add(dst[i].item())

        external_neighbors = neighbor_nodes - target_idx_set

        if allowed_external_classes is not None:
            allowed_nodes = set()
            for cls in allowed_external_classes:
                if cls in self.id_by_class:
                    allowed_nodes.update(self.id_by_class[cls])
            external_neighbors = external_neighbors & allowed_nodes

        all_nodes = target_idx_set | external_neighbors
        all_nodes_list = sorted(all_nodes)

        all_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool)
        all_nodes_mask[all_nodes_list] = True

        edge_in_subgraph = all_nodes_mask[src] & all_nodes_mask[dst]
        edge_has_target = target_mask[src] | target_mask[dst]
        valid_edges = edge_in_subgraph & edge_has_target
        subgraph_edge_index = edge_index[:, valid_edges]

        edge_index_with_sl = self.data.edge_index
        src_sl, dst_sl = edge_index_with_sl[0], edge_index_with_sl[1]
        edge_in_sub_sl = all_nodes_mask[src_sl] & all_nodes_mask[dst_sl]
        edge_has_target_sl = target_mask[src_sl] | target_mask[dst_sl]
        is_selfloop = src_sl == dst_sl
        valid_sl = edge_in_sub_sl & (edge_has_target_sl | is_selfloop)
        subgraph_edge_index_sl = edge_index_with_sl[:, valid_sl]

        return {
            'target_nodes': sorted(target_idx_set),
            'external_neighbors': sorted(external_neighbors),
            'all_nodes': all_nodes_list,
            'edge_index': subgraph_edge_index_sl,
            'edge_index_no_selfloop': subgraph_edge_index,
            'x': self.data.x,
            'y': self.data.y,
        }

    def get_task(self, task_id):
        """
        Get data for a specific task.

        Returns:
            curr_classes, all_classes_so_far,
            subgraph, joint_subgraph,
            train_loader, valid_loader, test_loader_joint
        """
        if task_id >= self.sessions:
            raise ValueError(f"Task {task_id} >= total sessions {self.sessions}")

        curr_classes = self.class_splits[task_id]
        all_classes_so_far = sorted(
            set(c for i in range(task_id + 1) for c in self.class_splits[i])
        )

        train_idx = self.train_idx_per_task[task_id]
        valid_idx = self.valid_idx_per_task[task_id]
        test_idx_joint = self.test_idx_joint[task_id]

        subgraph = self.subgraph_per_task[task_id]
        joint_subgraph = self.subgraph_joint[task_id]

        train_loader = DataLoader(
            Subset(self.graph_dataset, train_idx),
            batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            Subset(self.graph_dataset, valid_idx),
            batch_size=self.batch_size, shuffle=False
        )
        test_loader_joint = DataLoader(
            Subset(self.graph_dataset, test_idx_joint),
            batch_size=self.batch_size, shuffle=False
        )

        return (curr_classes, all_classes_so_far,
                subgraph, joint_subgraph,
                train_loader, valid_loader,
                test_loader_joint)
