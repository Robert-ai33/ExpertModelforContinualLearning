"""
Task loader for class-incremental continual learning with spectral clustering.

Subgraph construction:
- Each session has a set of target classes
- Target nodes: all nodes of target classes
- External neighbors: nodes directly connected to target nodes AND belonging
  to previously-seen classes only (no future class nodes)
- Edges: both endpoints in subgraph, at least one is target node
- External-to-external edges are removed
- Produces edge_index (with self-loops) and edge_index_no_selfloop

Test: only joint test (accumulated test set), no isolated test.
"""

import random
import torch

from torch.utils.data import Subset, DataLoader


class TaskLoader:
    """Task loader for spectral clustering continual learning."""

    def __init__(self, batch_size, graph_dataset, class_splits,
                 train_shots, valid_shots, test_shots):
        self.batch_size = batch_size
        self.graph_dataset = graph_dataset
        self.data = graph_dataset.data
        self.id_by_class = graph_dataset.id_by_class
        self.original_edge_index = graph_dataset.original_edge_index

        self.class_splits = class_splits
        self.sessions = len(class_splits)
        self.train_shots = train_shots
        self.valid_shots = valid_shots
        self.test_shots = test_shots

        # All classes used
        self.all_classes = []
        for split in class_splits:
            self.all_classes.extend(split)
        self.all_classes = sorted(set(self.all_classes))

        # Split data
        self._split_data()

        print(f"TaskLoader initialized:")
        print(f"  Sessions: {self.sessions}")
        print(f"  Class splits: {self.class_splits}")
        print(f"  All classes: {self.all_classes}")

    def _split_data(self):
        """Split data into train/valid/test for each session."""
        self.train_idx_per_task = []
        self.valid_idx_per_task = []
        self.test_idx_per_task = []
        self.test_idx_joint = []

        self.subgraph_per_task = []
        self.subgraph_joint = []

        cumulative_classes = []

        for session_id, classes in enumerate(self.class_splits):
            train_idx = []
            valid_idx = []
            test_idx = []

            for cla in classes:
                if cla not in self.id_by_class:
                    print(f"Warning: Class {cla} not found in dataset, skipping...")
                    continue

                node_idx = self.id_by_class[cla].copy()
                node_num = len(node_idx)

                total_shots = self.train_shots + self.valid_shots + self.test_shots
                if node_num < total_shots:
                    train_num = int(node_num * self.train_shots / total_shots)
                    valid_num = int(node_num * self.valid_shots / total_shots)
                    test_num = int(node_num * self.test_shots / total_shots)
                    remainder = node_num - train_num - valid_num - test_num
                    train_num += remainder
                else:
                    train_num = self.train_shots
                    valid_num = self.valid_shots
                    test_num = self.test_shots

                random.shuffle(node_idx)

                train_idx.extend(node_idx[:train_num])
                valid_idx.extend(node_idx[train_num:train_num + valid_num])
                test_idx.extend(
                    node_idx[train_num + valid_num:train_num + valid_num + test_num]
                )

            self.train_idx_per_task.append(train_idx)
            self.valid_idx_per_task.append(valid_idx)
            self.test_idx_per_task.append(test_idx)

            # Cumulative test indices (for joint evaluation)
            if session_id == 0:
                self.test_idx_joint.append(test_idx.copy())
            else:
                self.test_idx_joint.append(self.test_idx_joint[-1] + test_idx)

            # Previously seen classes (sessions before current)
            prev_seen_classes = list(cumulative_classes)

            # Create current session subgraph
            # External neighbors restricted to previously-seen classes only
            curr_subgraph = self._create_task_subgraph(
                classes, allowed_external_classes=prev_seen_classes
            )
            self.subgraph_per_task.append(curr_subgraph)

            # Create cumulative subgraph (all seen classes including current)
            cumulative_classes.extend(classes)
            # For joint subgraph, target = all cumulative classes,
            # external neighbors restricted to cumulative (= no unseen classes)
            joint_subgraph = self._create_task_subgraph(
                cumulative_classes, allowed_external_classes=cumulative_classes
            )
            self.subgraph_joint.append(joint_subgraph)

    def _create_task_subgraph(self, class_ids, allowed_external_classes=None):
        """
        Create task subgraph.

        Rules:
        - Target nodes: all nodes of given classes
        - External neighbors: nodes directly connected to target nodes AND
          belonging to allowed_external_classes (previously-seen classes)
        - Keep edges: both endpoints in subgraph AND at least one is target
        - External-to-external edges are removed

        Args:
            class_ids: list of class IDs for target nodes
            allowed_external_classes: list of class IDs whose nodes can serve
                as external neighbors. If None, all connected nodes are allowed.

        Returns dict with:
        - target_nodes, external_neighbors, all_nodes
        - edge_index (with self-loops, for general use)
        - edge_index_no_selfloop (without self-loops, for neighbor prediction)
        - x, y (full graph features/labels for global indexing)
        """
        # Target nodes
        target_idx = []
        for cls in class_ids:
            if cls in self.id_by_class:
                target_idx.extend(self.id_by_class[cls])
        target_idx_set = set(target_idx)

        num_nodes = len(self.data.y)
        target_mask = torch.zeros(num_nodes, dtype=torch.bool)
        target_mask[list(target_idx_set)] = True

        # Find external neighbors using original edges (no self-loops)
        edge_index = self.original_edge_index
        src, dst = edge_index[0], edge_index[1]

        edges_from_target = target_mask[src]
        edges_to_target = target_mask[dst]
        connected_edges = edges_from_target | edges_to_target

        neighbor_nodes = set()
        connected_indices = torch.where(connected_edges)[0]
        for i in connected_indices.tolist():
            neighbor_nodes.add(src[i].item())
            neighbor_nodes.add(dst[i].item())

        external_neighbors = neighbor_nodes - target_idx_set

        # Filter external neighbors: only keep nodes from allowed classes
        if allowed_external_classes is not None:
            allowed_nodes = set()
            for cls in allowed_external_classes:
                if cls in self.id_by_class:
                    allowed_nodes.update(self.id_by_class[cls])
            external_neighbors = external_neighbors & allowed_nodes

        all_nodes = target_idx_set | external_neighbors
        all_nodes_list = sorted(list(all_nodes))

        all_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool)
        all_nodes_mask[all_nodes_list] = True

        # Filter edges: both in subgraph AND at least one is target
        edge_in_subgraph = all_nodes_mask[src] & all_nodes_mask[dst]
        edge_has_target = target_mask[src] | target_mask[dst]
        valid_edges = edge_in_subgraph & edge_has_target
        subgraph_edge_index = edge_index[:, valid_edges]

        # Edge index with self-loops (for general use)
        edge_index_with_sl = self.data.edge_index
        src_sl, dst_sl = edge_index_with_sl[0], edge_index_with_sl[1]
        edge_in_sub_sl = all_nodes_mask[src_sl] & all_nodes_mask[dst_sl]
        edge_has_target_sl = target_mask[src_sl] | target_mask[dst_sl]
        is_selfloop = src_sl == dst_sl
        valid_sl = edge_in_sub_sl & (edge_has_target_sl | is_selfloop)
        subgraph_edge_index_sl = edge_index_with_sl[:, valid_sl]

        return {
            'target_nodes': sorted(list(target_idx_set)),
            'external_neighbors': sorted(list(external_neighbors)),
            'all_nodes': all_nodes_list,
            'edge_index': subgraph_edge_index_sl,          # with self-loops
            'edge_index_no_selfloop': subgraph_edge_index,  # without self-loops
            'x': self.data.x,  # full graph features (global indexing)
            'y': self.data.y,  # full graph labels (global indexing)
        }

    def get_task(self, task_id):
        """
        Get data loaders for a specific task.

        Returns:
            curr_classes: classes in current task
            all_classes_so_far: all classes seen so far
            subgraph: current task subgraph
            joint_subgraph: cumulative subgraph
            train_loader: training data loader
            valid_loader: validation data loader
            test_loader_joint: joint test data loader
        """
        if task_id >= self.sessions:
            raise ValueError(f"Task id {task_id} >= total sessions {self.sessions}")

        curr_classes = self.class_splits[task_id]

        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.class_splits[i])
        all_classes_so_far = sorted(set(all_classes_so_far))

        # Indices
        train_idx = self.train_idx_per_task[task_id]
        valid_idx = self.valid_idx_per_task[task_id]
        test_idx_joint = self.test_idx_joint[task_id]

        # Subgraphs
        subgraph = self.subgraph_per_task[task_id]
        joint_subgraph = self.subgraph_joint[task_id]

        # Data loaders
        train_dataset = Subset(self.graph_dataset, train_idx)
        valid_dataset = Subset(self.graph_dataset, valid_idx)
        test_dataset_joint = Subset(self.graph_dataset, test_idx_joint)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader_joint = DataLoader(
            test_dataset_joint, batch_size=self.batch_size, shuffle=False
        )

        return (curr_classes, all_classes_so_far,
                subgraph, joint_subgraph,
                train_loader, valid_loader,
                test_loader_joint)
