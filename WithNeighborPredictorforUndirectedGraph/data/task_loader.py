"""
Task loader for class-incremental continual learning.

实现新的子图构建逻辑:
- 每个阶段有一个类别范围
- 提供的图是整体图的所有这个阶段类别范围内节点组成的子图
- 类别范围内节点之间的边全部保留
- 加上和类别内节点直接相连的节点
- 但只保留类别内节点到类别外节点的边
- 类别外节点之间的边删除
"""

import copy
import random
import torch
import numpy as np

from torch.utils.data import Subset, DataLoader
from collections import defaultdict


class TaskLoader:
    """
    Task loader that supports flexible class splits.
    
    子图构建规则:
    - 类别内节点之间的边全部保留
    - 类别外邻居节点被包含
    - 只保留类别内节点到类别外节点的边
    - 类别外节点之间的边删除
    """
    
    def __init__(self, batch_size, text_dataset, class_splits, 
                 train_shots, valid_shots, test_shots):
        self.batch_size = batch_size
        self.text_dataset = text_dataset
        self.data = text_dataset.data
        self.id_by_class = text_dataset.id_by_class
        # 使用原始edge_index（不含自环）进行邻居预测
        self.original_edge_index = text_dataset.original_edge_index
        
        self.class_splits = class_splits
        self.sessions = len(class_splits)
        self.train_shots = train_shots
        self.valid_shots = valid_shots
        self.test_shots = test_shots
        
        # Get all classes that will be used
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
        self.valid_idx_joint = []  # Cumulative valid indices
        self.test_idx_joint = []  # Cumulative test indices
        
        # 存储每个阶段的子图数据
        self.subgraph_per_task = []  # 用于训练的子图
        self.subgraph_joint = []     # 累积所有已见类别的子图
        
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
                
                # Determine split sizes
                if node_num < (self.train_shots + self.valid_shots + self.test_shots):
                    train_num = int(node_num * 0.5)
                    valid_num = int(node_num * 0.1)
                    test_num = int(node_num * 0.4)
                    if train_num + valid_num + test_num > node_num:
                        train_num = node_num - valid_num - test_num
                else:
                    train_num = self.train_shots
                    valid_num = self.valid_shots
                    test_num = self.test_shots
                
                random.shuffle(node_idx)
                
                train_idx.extend(node_idx[:train_num])
                valid_idx.extend(node_idx[train_num:train_num + valid_num])
                test_idx.extend(node_idx[train_num + valid_num:train_num + valid_num + test_num])
            
            self.train_idx_per_task.append(train_idx)
            self.valid_idx_per_task.append(valid_idx)
            self.test_idx_per_task.append(test_idx)
            
            # Cumulative valid indices (for joint validation)
            if session_id == 0:
                self.valid_idx_joint.append(valid_idx.copy())
            else:
                self.valid_idx_joint.append(self.valid_idx_joint[-1] + valid_idx)
            
            # Cumulative test indices (for joint evaluation)
            if session_id == 0:
                self.test_idx_joint.append(test_idx.copy())
            else:
                self.test_idx_joint.append(self.test_idx_joint[-1] + test_idx)
            
            # 创建当前阶段的子图
            curr_subgraph = self._create_task_subgraph(classes)
            self.subgraph_per_task.append(curr_subgraph)
            
            # 创建累积子图（包含所有已见类别）
            cumulative_classes.extend(classes)
            joint_subgraph = self._create_task_subgraph(cumulative_classes)
            self.subgraph_joint.append(joint_subgraph)
    
    def _create_task_subgraph(self, class_ids):
        """
        创建任务子图，按照用户指定的规则:
        
        1. 获取所有类别内节点（target nodes）
        2. 获取所有与类别内节点直接相连的类别外节点（neighbor nodes）
        3. 边的保留规则:
           - 类别内节点之间的边全部保留
           - 类别内节点到类别外节点的边保留
           - 类别外节点之间的边删除
        
        返回一个包含子图信息的字典:
        {
            'target_nodes': 类别内节点列表,
            'all_nodes': 所有节点列表（类别内+类别外邻居）,
            'edge_index': 子图的边索引（使用全局节点ID）,
            'edge_index_no_selfloop': 不含自环的边索引,
            'x': 所有节点的特征,
            'y': 所有节点的标签
        }
        """
        # 获取类别内节点（target nodes）
        target_idx = []
        for cls in class_ids:
            if cls in self.id_by_class:
                target_idx.extend(self.id_by_class[cls])
        target_idx_set = set(target_idx)
        
        # 创建target节点的mask
        num_nodes = len(self.data.y)
        target_mask = torch.zeros(num_nodes, dtype=torch.bool)
        target_mask[list(target_idx_set)] = True
        
        # 使用原始edge_index（不含自环）找邻居
        edge_index = self.original_edge_index
        src, dst = edge_index[0], edge_index[1]
        
        # 找到所有与target节点直接相连的节点
        # 边: target -> other 或 other -> target
        edges_from_target = target_mask[src]
        edges_to_target = target_mask[dst]
        
        # 所有与target相连的边
        connected_edges = edges_from_target | edges_to_target
        
        # 获取所有邻居节点（包括target节点和类别外邻居）
        neighbor_nodes = set()
        for i in range(edge_index.size(1)):
            if connected_edges[i]:
                neighbor_nodes.add(src[i].item())
                neighbor_nodes.add(dst[i].item())
        
        # 类别外邻居节点
        external_neighbors = neighbor_nodes - target_idx_set
        
        # 所有节点 = 类别内节点 + 类别外邻居
        all_nodes = target_idx_set | external_neighbors
        all_nodes_list = sorted(list(all_nodes))
        
        # 创建all_nodes的mask
        all_nodes_mask = torch.zeros(num_nodes, dtype=torch.bool)
        all_nodes_mask[all_nodes_list] = True
        
        # 筛选边：
        # 保留的边需要满足：
        # 1. 两端都在all_nodes中
        # 2. 至少一端是target节点（排除外部节点之间的边）
        edge_in_subgraph = all_nodes_mask[src] & all_nodes_mask[dst]  # 两端都在子图中
        edge_has_target = target_mask[src] | target_mask[dst]         # 至少一端是target
        valid_edges = edge_in_subgraph & edge_has_target
        
        # 提取有效边
        subgraph_edge_index = edge_index[:, valid_edges]
        
        # 同样处理带自环的edge_index（用于GCN）
        edge_index_with_selfloop = self.data.edge_index
        src_sl, dst_sl = edge_index_with_selfloop[0], edge_index_with_selfloop[1]
        
        edge_in_subgraph_sl = all_nodes_mask[src_sl] & all_nodes_mask[dst_sl]
        edge_has_target_sl = target_mask[src_sl] | target_mask[dst_sl]
        # 对于自环边，如果在all_nodes中就保留
        is_selfloop = src_sl == dst_sl
        valid_edges_sl = edge_in_subgraph_sl & (edge_has_target_sl | is_selfloop)
        
        subgraph_edge_index_sl = edge_index_with_selfloop[:, valid_edges_sl]
        
        return {
            'target_nodes': sorted(list(target_idx_set)),
            'external_neighbors': sorted(list(external_neighbors)),
            'all_nodes': all_nodes_list,
            'edge_index': subgraph_edge_index_sl,  # 带自环，用于GCN
            'edge_index_no_selfloop': subgraph_edge_index,  # 不带自环，用于邻居预测
            'x': self.data.x,  # 使用全图特征
            'y': self.data.y,  # 使用全图标签
        }
    
    def get_task(self, task_id):
        """
        Get data loaders for a specific task.
        
        Returns:
            curr_classes: list of classes in current task
            all_classes_so_far: list of all classes seen so far
            subgraph: current task's subgraph dict
            joint_subgraph: cumulative subgraph dict
            train_loader, valid_loader, test_loader_iso, test_loader_joint
        """
        if task_id >= self.sessions:
            raise ValueError(f"Task id {task_id} >= total sessions {self.sessions}")
        
        curr_classes = self.class_splits[task_id]
        
        # All classes seen so far
        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.class_splits[i])
        all_classes_so_far = sorted(set(all_classes_so_far))
        
        # Get indices
        train_idx = self.train_idx_per_task[task_id]
        valid_idx = self.valid_idx_per_task[task_id]
        valid_idx_joint = self.valid_idx_joint[task_id]
        test_idx_iso = self.test_idx_per_task[task_id]
        test_idx_joint = self.test_idx_joint[task_id]
        
        # Get subgraphs
        subgraph = self.subgraph_per_task[task_id]
        joint_subgraph = self.subgraph_joint[task_id]
        
        # Create data loaders
        train_dataset = Subset(self.text_dataset, train_idx)
        valid_dataset = Subset(self.text_dataset, valid_idx)
        valid_dataset_joint = Subset(self.text_dataset, valid_idx_joint)
        test_dataset_iso = Subset(self.text_dataset, test_idx_iso)
        test_dataset_joint = Subset(self.text_dataset, test_idx_joint)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        valid_loader_joint = DataLoader(valid_dataset_joint, batch_size=self.batch_size, shuffle=False)
        test_loader_iso = DataLoader(test_dataset_iso, batch_size=self.batch_size, shuffle=False)
        test_loader_joint = DataLoader(test_dataset_joint, batch_size=self.batch_size, shuffle=False)
        
        return (curr_classes, all_classes_so_far, 
                subgraph, joint_subgraph,
                train_loader, valid_loader, valid_loader_joint,
                test_loader_iso, test_loader_joint)
    
    def get_all_trained_classes(self, task_id):
        """Get list of all classes trained up to task_id."""
        all_classes = []
        for i in range(task_id + 1):
            all_classes.extend(self.class_splits[i])
        return sorted(set(all_classes))
