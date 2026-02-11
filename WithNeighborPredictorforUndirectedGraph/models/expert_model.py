"""
Expert-based Continual Learning Model

模型结构:
- 每个阶段一个专家
- 每个专家包含:
  1. 分类专家: GCN(1层传播) + 下游分类层
  2. 邻居预测专家: 直接用初始embedding，降低泛化性

训练策略:
- 前cls_epochs: 训练分类模型
- 后epochs-cls_epochs: 训练邻居预测模型
- 两个模型分开训练

测试策略:
- 用邻居预测模型的准确度选择专家
- 对于每个点，如果有n个邻居，计算top-(n+u)可能为邻居的点
- 找到正确数量最多的专家
- 如果并列，比较位置之和（越小越好）
- 如果还并列，随机选择
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from utils import save_checkpoint, load_checkpoint


class ClassificationExpert(nn.Module):
    """
    分类专家: 单层GCN + 分类头
    
    GCN传播一层，融合节点本身的embedding和其所有邻居的embedding
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        
        self.dropout = dropout
        
        # 单层GCN
        self.conv = GCNConv(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        # 分类头
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def get_embeddings(self, x, edge_index):
        """获取GCN编码后的embedding"""
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def forward(self, x, edge_index):
        """前向传播，返回logits"""
        embeddings = self.get_embeddings(x, edge_index)
        logits = self.classifier(embeddings)
        return logits, embeddings


class NeighborPredictor(nn.Module):
    """
    邻居预测专家: 直接使用初始embedding
    
    使用MLP将原始特征映射到邻居预测空间
    设计上故意降低泛化性，以便区分不同专家
    """
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        # 简单的MLP，直接处理原始特征
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        """返回用于邻居预测的embedding"""
        return self.mlp(x)


class Expert(nn.Module):
    """
    单个专家，包含分类专家和邻居预测专家
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, neighbor_hidden_dim=128, dropout=0.5):
        super().__init__()
        
        self.classifier = ClassificationExpert(input_dim, hidden_dim, num_classes, dropout)
        self.neighbor_predictor = NeighborPredictor(input_dim, neighbor_hidden_dim)
    
    def classify(self, x, edge_index):
        """分类"""
        return self.classifier(x, edge_index)
    
    def get_classification_embeddings(self, x, edge_index):
        """获取分类embedding"""
        return self.classifier.get_embeddings(x, edge_index)
    
    def get_neighbor_embeddings(self, x):
        """获取邻居预测embedding"""
        return self.neighbor_predictor(x)


class ExpertCLModel(nn.Module):
    """
    包含多个专家的持续学习模型
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_experts, 
                 neighbor_hidden_dim=128, dropout=0.5):
        super().__init__()
        
        self.num_experts = num_experts
        self.num_classes = num_classes
        
        # 创建专家列表
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, num_classes, neighbor_hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        
        self.active_expert = 0
    
    def set_active_expert(self, expert_id):
        """设置当前活跃的专家"""
        self.active_expert = expert_id
    
    def classify(self, x, edge_index, expert_id=None):
        """使用指定专家进行分类"""
        if expert_id is None:
            expert_id = self.active_expert
        return self.experts[expert_id].classify(x, edge_index)
    
    def get_neighbor_embeddings(self, x, expert_id=None):
        """获取指定专家的邻居预测embedding"""
        if expert_id is None:
            expert_id = self.active_expert
        return self.experts[expert_id].get_neighbor_embeddings(x)
    
    def freeze_expert(self, expert_id):
        """冻结指定专家的参数"""
        for param in self.experts[expert_id].parameters():
            param.requires_grad = False
    
    def unfreeze_expert(self, expert_id):
        """解冻指定专家的参数"""
        for param in self.experts[expert_id].parameters():
            param.requires_grad = True
    
    def freeze_all_except(self, expert_id):
        """冻结除指定专家外的所有专家"""
        for i in range(self.num_experts):
            if i == expert_id:
                self.unfreeze_expert(i)
            else:
                self.freeze_expert(i)
    
    def set_training_phase(self, expert_id, phase):
        """
        设置训练阶段
        
        Args:
            expert_id: 专家ID
            phase: 'classification' 或 'neighbor'
        """
        self.freeze_all_except(expert_id)
        expert = self.experts[expert_id]
        
        if phase == 'classification':
            # 训练分类器，冻结邻居预测器
            for param in expert.classifier.parameters():
                param.requires_grad = True
            for param in expert.neighbor_predictor.parameters():
                param.requires_grad = False
        
        elif phase == 'neighbor':
            # 训练邻居预测器，冻结分类器
            for param in expert.classifier.parameters():
                param.requires_grad = False
            for param in expert.neighbor_predictor.parameters():
                param.requires_grad = True


class ExpertCL:
    """
    Expert-based Continual Learning Trainer
    
    训练策略:
    - 前cls_epochs: 训练分类模型（GCN + 分类头）
    - 后epochs-cls_epochs: 训练邻居预测模型
    
    测试策略:
    - 用邻居预测准确度选择专家
    """
    
    def __init__(self, task_loader, result_logger, config, checkpoint_path,
                 dataset, seed, device):
        self.task_loader = task_loader
        self.result_logger = result_logger
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.seed = seed
        self.device = device
        
        # 模型参数
        self.input_dim = task_loader.data.x.shape[1]
        self.num_classes = task_loader.data.y.max().item() + 1
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']
        
        # 专家参数
        self.num_experts = config.get('num_experts', 5)
        
        # 邻居预测参数
        self.neighbor_hidden_dim = config.get('neighbor_hidden_dim', 128)
        self.num_neg_samples = config.get('num_neg_samples', 3)
        self.neighbor_topk_offset = config.get('neighbor_topk_offset', 3)  # 参数u
        
        # 训练参数
        self.cls_epochs = config.get('cls_epochs', 100)
        self.debug = config.get('debug', False)
        
        # 创建模型
        self.model = ExpertCLModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_experts=self.num_experts,
            neighbor_hidden_dim=self.neighbor_hidden_dim,
            dropout=self.dropout,
        ).to(device)
        
        self.current_session = 0
    
    def get_optimizer(self):
        """获取优化器"""
        return optim.Adam(
            self.model.parameters(),
            lr=float(self.config['lr']),
            weight_decay=float(self.config['weight_decay'])
        )
    
    def neighbor_prediction_loss(self, x, edge_index, expert_id):
        """
        计算邻居预测loss
        
        采样所有正边 + 随机负边（正边数量 * num_neg_samples）
        负边每次计算loss前重新随机选取
        
        Args:
            x: 节点特征（原始特征）
            edge_index: 边索引（不含自环）
            expert_id: 专家ID
        """
        h = self.model.get_neighbor_embeddings(x, expert_id)
        
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        device = x.device
        
        if num_edges == 0 or num_nodes < 2:
            return torch.tensor(0.0, device=device)
        
        # === 正样本: 所有正边 ===
        pos_src = edge_index[0]
        pos_dst = edge_index[1]
        pos_scores = (h[pos_src] * h[pos_dst]).sum(dim=1)
        
        # === 负样本: 随机采样 num_edges * num_neg_samples 条负边 ===
        num_neg_edges = num_edges * self.num_neg_samples
        
        # 随机生成负边
        neg_src = torch.randint(0, num_nodes, (num_neg_edges,), device=device)
        neg_dst = torch.randint(0, num_nodes, (num_neg_edges,), device=device)
        
        # 移除自环
        valid_mask = neg_src != neg_dst
        neg_src = neg_src[valid_mask]
        neg_dst = neg_dst[valid_mask]
        
        # 计算负样本分数
        neg_scores = (h[neg_src] * h[neg_dst]).sum(dim=1)
        
        # === 计算loss ===
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean() if neg_scores.numel() > 0 else torch.tensor(0.0, device=device)
        
        return pos_loss + neg_loss
    
    def compute_neighbor_accuracy(self, expert_id, x, edge_index, target_nodes):
        """
        计算邻居预测准确度，用于专家选择
        
        对于每个点，如果有n个邻居，计算top-(n+u)可能为邻居的点
        看看有多少真的是邻居
        
        Returns:
            correct_count: 正确预测的邻居数量
            position_sum: 真实邻居在预测中的位置之和（越小越好）
        """
        self.model.eval()
        
        with torch.no_grad():
            h = self.model.get_neighbor_embeddings(x, expert_id)
            num_nodes = h.size(0)
            device = h.device
            
            # 构建邻接矩阵
            src, dst = edge_index[0], edge_index[1]
            adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
            adj_matrix[src, dst] = True
            adj_matrix[dst, src] = True  # 对称
            
            total_correct = 0
            total_position_sum = 0
            
            for node_idx in target_nodes:
                node_idx = node_idx.item() if isinstance(node_idx, torch.Tensor) else node_idx
                
                # 获取真实邻居
                true_neighbor_mask = adj_matrix[node_idx]
                n = true_neighbor_mask.sum().item()
                
                if n == 0:
                    continue
                
                # 计算与所有节点的相似度
                scores = torch.matmul(h[node_idx], h.T)
                scores[node_idx] = float('-inf')  # 排除自己
                
                # 取top-(n+u)
                k = min(n + self.neighbor_topk_offset, num_nodes - 1)
                k = max(k, 1)
                _, top_k = torch.topk(scores, k)
                
                # 统计正确预测的邻居数量
                predicted_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                predicted_mask[top_k] = True
                
                correct = (predicted_mask & true_neighbor_mask).sum().item()
                total_correct += correct
                
                # 计算位置之和（1-indexed）
                for pos, node in enumerate(top_k.tolist()):
                    if true_neighbor_mask[node]:
                        total_position_sum += (pos + 1)
            
            return total_correct, total_position_sum
    
    def train_classification_epoch(self, session_id, subgraph, train_loader, 
                                   optimizer, curr_classes, all_classes):
        """训练一个epoch的分类模型"""
        self.model.train()
        
        expert_id = session_id % self.num_experts
        
        # 获取子图数据
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        
        total_loss = 0.0
        num_samples = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 分类
            logits, _ = self.model.classify(x, edge_index, expert_id)
            
            # 获取batch中节点的logits
            node_ids = batch['node_id']
            labels = batch['labels'].to(self.device)
            
            batch_logits = logits[node_ids]
            
            # 只使用已训练的类别
            max_class = max(all_classes) + 1
            batch_logits = batch_logits[:, :max_class]
            
            # 创建有效样本mask（只考虑当前任务的类别）
            valid_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=self.device)
            for c in curr_classes:
                valid_mask |= (labels == c)
            
            if valid_mask.sum() == 0:
                continue
            
            loss = F.cross_entropy(batch_logits[valid_mask], labels[valid_mask])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * valid_mask.sum().item()
            num_samples += valid_mask.sum().item()
        
        return total_loss / num_samples if num_samples > 0 else 0.0
    
    def train_neighbor_epoch(self, session_id, subgraph, train_loader, optimizer):
        """训练一个epoch的邻居预测模型"""
        self.model.train()
        
        expert_id = session_id % self.num_experts
        
        # 获取子图数据（使用不含自环的边）
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index_no_selfloop'].to(self.device)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 使用整个子图进行邻居预测训练
            loss = self.neighbor_prediction_loss(x, edge_index, expert_id)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate_classifier(self, session_id, subgraph, data_loader, curr_classes, all_classes):
        """
        单独验证当前专家的分类器性能（不涉及专家选择）
        
        Args:
            session_id: 当前session ID
            subgraph: 当前任务的子图
            data_loader: 验证数据loader
            curr_classes: 当前任务的类别
            all_classes: 所有已见类别
        
        Returns:
            accuracy: 分类准确率
        """
        self.model.eval()
        expert_id = session_id % self.num_experts
        
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        
        all_preds = []
        all_labels = []
        
        for batch in data_loader:
            node_ids = batch['node_id']
            labels = batch['labels'].to(self.device)
            
            # 直接使用当前专家进行分类（不经过专家选择）
            logits, _ = self.model.classify(x, edge_index, expert_id)
            batch_logits = logits[node_ids]
            
            # 只考虑已训练的类别
            max_class = max(all_classes) + 1
            batch_logits = batch_logits[:, :max_class]
            
            # 只评估当前任务类别的样本
            valid_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=self.device)
            for c in curr_classes:
                valid_mask |= (labels == c)
            
            if valid_mask.sum() == 0:
                continue
            
            preds = torch.argmax(batch_logits[valid_mask], dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels[valid_mask].cpu())
        
        if len(all_preds) == 0:
            return 0.0
        
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        
        return accuracy_score(labels, preds)
    
    @torch.no_grad()
    def evaluate_neighbor_predictor(self, session_id, subgraph, data_loader):
        """
        单独验证当前专家的邻居预测器性能（不涉及专家选择）
        
        计算验证集节点的邻居预测准确度
        
        Args:
            session_id: 当前session ID
            subgraph: 当前任务的子图
            data_loader: 验证数据loader
        
        Returns:
            accuracy: 邻居预测准确率（正确预测的邻居比例）
        """
        self.model.eval()
        expert_id = session_id % self.num_experts
        
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index_no_selfloop'].to(self.device)
        
        # 收集所有验证节点
        valid_nodes = []
        for batch in data_loader:
            valid_nodes.extend(batch['node_id'].tolist())
        
        if len(valid_nodes) == 0:
            return 0.0
        
        # 计算邻居预测准确度
        total_correct, _ = self.compute_neighbor_accuracy(
            expert_id, x, edge_index, valid_nodes
        )
        
        # 计算总的真实邻居数
        h = self.model.get_neighbor_embeddings(x, expert_id)
        num_nodes = h.size(0)
        
        src, dst = edge_index[0], edge_index[1]
        adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=self.device)
        adj_matrix[src, dst] = True
        adj_matrix[dst, src] = True
        
        total_neighbors = 0
        for node_idx in valid_nodes:
            total_neighbors += adj_matrix[node_idx].sum().item()
        
        if total_neighbors == 0:
            return 0.0
        
        return total_correct / total_neighbors
    
    def train_epoch(self, session_id, epoch, subgraph, train_loader, 
                    optimizer, curr_classes, all_classes):
        """训练一个epoch"""
        expert_id = session_id % self.num_experts
        is_cls_phase = epoch < self.cls_epochs
        
        # 设置训练阶段
        if epoch == 0:
            self.model.set_training_phase(expert_id, 'classification')
            if self.debug:
                print(f"\n[Session {session_id}] Expert {expert_id}")
                print(f"  Classes: {curr_classes}")
                print(f"  Phase 1: Classification (epochs 0-{self.cls_epochs-1})")
        elif epoch == self.cls_epochs:
            self.model.set_training_phase(expert_id, 'neighbor')
            if self.debug:
                print(f"  Phase 2: Neighbor Prediction (epochs {self.cls_epochs}+)")
        
        if is_cls_phase:
            return self.train_classification_epoch(
                session_id, subgraph, train_loader, 
                optimizer, curr_classes, all_classes
            )
        else:
            return self.train_neighbor_epoch(
                session_id, subgraph, train_loader, optimizer
            )
    
    @torch.no_grad()
    def select_expert(self, x, edge_index, node_idx):
        """
        为单个节点选择最佳专家
        
        选择策略:
        1. 计算每个专家的邻居预测准确度（正确数量）
        2. 选择正确数量最多的专家
        3. 如果并列，选择位置之和最小的
        4. 如果还并列，随机选择
        """
        num_experts = min(self.num_experts, self.current_session + 1)
        
        expert_scores = []
        for exp_id in range(num_experts):
            correct, pos_sum = self.compute_neighbor_accuracy(
                exp_id, x, edge_index, [node_idx]
            )
            expert_scores.append((exp_id, correct, pos_sum))
        
        # Step 1: 找到最大正确数量
        max_correct = max(correct for _, correct, _ in expert_scores)
        best_by_correct = [(exp_id, pos_sum) for exp_id, correct, pos_sum in expert_scores 
                          if correct == max_correct]
        
        # Step 2: 在正确数量最多的专家中，找位置之和最小的
        min_pos_sum = min(pos_sum for _, pos_sum in best_by_correct)
        best_by_pos = [exp_id for exp_id, pos_sum in best_by_correct 
                      if pos_sum == min_pos_sum]
        
        # Step 3: 随机选择
        best_expert = random.choice(best_by_pos)
        
        return best_expert, expert_scores
    
    @torch.no_grad()
    def evaluate(self, subgraph, test_loader, trained_classes):
        """
        使用专家选择进行评估
        
        对于每个测试节点:
        1. 用邻居预测选择最佳专家
        2. 用该专家的分类模型预测类别
        """
        self.model.eval()
        
        num_experts = min(self.num_experts, self.current_session + 1)
        expert_counts = {i: 0 for i in range(num_experts)}
        
        # 获取子图数据
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        edge_index_no_selfloop = subgraph['edge_index_no_selfloop'].to(self.device)
        
        all_preds = []
        all_labels = []
        
        debug_info = []
        
        for batch in test_loader:
            node_ids = batch['node_id']
            labels = batch['labels']
            
            for i, node_idx in enumerate(node_ids):
                node_idx = node_idx.item()
                
                # 选择专家（使用不含自环的边进行邻居预测）
                best_expert, expert_scores = self.select_expert(
                    x, edge_index_no_selfloop, node_idx
                )
                
                expert_counts[best_expert] += 1
                
                # 使用选中的专家进行分类
                logits, _ = self.model.classify(x, edge_index, best_expert)
                node_logits = logits[node_idx:node_idx+1]
                
                # 只考虑已训练的类别
                max_class = max(trained_classes) + 1
                node_logits = node_logits[:, :max_class]
                pred = torch.argmax(node_logits, dim=1)
                
                all_preds.append(pred.cpu())
                all_labels.append(labels[i])
                
                # Debug信息
                if self.debug and len(debug_info) < 10:
                    debug_info.append({
                        'node': node_idx,
                        'scores': [(e, c, p) for e, c, p in expert_scores],
                        'selected': best_expert,
                        'label': labels[i].item(),
                        'pred': pred.item()
                    })
        
        if len(all_preds) == 0:
            return 0.0, 0.0
        
        if self.debug:
            print(f"\n[EVAL] Trained classes: {trained_classes}")
            print(f"  Expert distribution: {expert_counts}")
            if debug_info:
                print(f"  Sample node info (first 10):")
                for info in debug_info:
                    print(f"    Node {info['node']} (label={info['label']}): "
                          f"scores={[(e, c, p) for e, c, p in info['scores']]} "
                          f"-> Expert {info['selected']}, pred={info['pred']}")
        
        preds = torch.cat(all_preds).numpy()
        labels = torch.stack(all_labels).numpy()
        
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        
        return acc, f1
    
    def fit(self, trial):
        """主训练循环"""
        # 存储每个session的joint accuracy
        joint_acc_history = []
        
        for session_id in range(self.task_loader.sessions):
            self.current_session = session_id
            
            # 每个session重置优化器
            optimizer = self.get_optimizer()
            
            # 加载之前的checkpoint
            if session_id > 0:
                load_checkpoint(self.model, self.checkpoint_path,
                               self.dataset, 'ExpertCL', self.seed)
            
            # 获取任务数据
            (curr_classes, all_classes, 
             subgraph, joint_subgraph,
             train_loader, valid_loader, valid_loader_joint,
             test_loader_iso, test_loader_joint) = self.task_loader.get_task(session_id)
            
            print(f"\n{'='*50}")
            print(f"Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"{'='*50}")
            
            # 训练循环
            best_cls_valid_acc = 0.0
            best_neighbor_valid_acc = 0.0
            cls_patience_counter = 0
            neighbor_patience_counter = 0
            
            pbar = tqdm(range(self.config['epochs']), desc=f"Session {session_id}")
            
            for epoch in pbar:
                is_cls_phase = epoch < self.cls_epochs
                
                loss = self.train_epoch(
                    session_id, epoch, subgraph, train_loader,
                    optimizer, curr_classes, all_classes
                )
                
                # 分阶段验证：只验证当前专家的对应子模型性能
                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    if is_cls_phase:
                        # 分类阶段：验证分类器性能
                        valid_acc = self.evaluate_classifier(
                            session_id, subgraph, valid_loader, curr_classes, all_classes
                        )
                        
                        if valid_acc > best_cls_valid_acc:
                            best_cls_valid_acc = valid_acc
                            cls_patience_counter = 0
                            save_checkpoint(self.model, optimizer, epoch,
                                           self.checkpoint_path, self.dataset,
                                           'ExpertCL', self.seed)
                        else:
                            cls_patience_counter += 1
                            if cls_patience_counter > self.config['patience']:
                                # 分类阶段early stop，跳到邻居预测阶段
                                pass  # 继续训练，不break
                        
                        pbar.set_postfix({'loss': f'{loss:.4f}', 'cls_val': f'{valid_acc:.4f}'})
                    else:
                        # 邻居预测阶段：验证邻居预测器性能
                        valid_acc = self.evaluate_neighbor_predictor(
                            session_id, subgraph, valid_loader
                        )
                        
                        if valid_acc > best_neighbor_valid_acc:
                            best_neighbor_valid_acc = valid_acc
                            neighbor_patience_counter = 0
                            save_checkpoint(self.model, optimizer, epoch,
                                           self.checkpoint_path, self.dataset,
                                           'ExpertCL', self.seed)
                        else:
                            neighbor_patience_counter += 1
                            if neighbor_patience_counter > self.config['patience']:
                                break
                        
                        pbar.set_postfix({'loss': f'{loss:.4f}', 'neighbor_val': f'{valid_acc:.4f}'})
                else:
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            # 加载最佳模型
            load_checkpoint(self.model, self.checkpoint_path,
                           self.dataset, 'ExpertCL', self.seed)
            
            # 测试当前任务
            test_acc_iso, _ = self.evaluate(subgraph, test_loader_iso, all_classes)
            test_acc_joint, _ = self.evaluate(joint_subgraph, test_loader_joint, all_classes)
            
            # 评估之前的任务
            acc_list = []
            for prev_session in range(session_id):
                prev_subgraph = self.task_loader.subgraph_per_task[prev_session]
                prev_test_idx = self.task_loader.test_idx_per_task[prev_session]
                prev_test_dataset = torch.utils.data.Subset(
                    self.task_loader.text_dataset, prev_test_idx
                )
                prev_test_loader = torch.utils.data.DataLoader(
                    prev_test_dataset, batch_size=self.config['batch_size'], shuffle=False
                )
                prev_acc, _ = self.evaluate(prev_subgraph, prev_test_loader, all_classes)
                acc_list.append(prev_acc)
            acc_list.append(test_acc_iso)
            
            # 记录joint accuracy
            joint_acc_history.append({
                'session': session_id,
                'classes': all_classes.copy(),
                'joint_acc': test_acc_joint
            })
            
            print(f"\nSession {session_id} Results:")
            print(f"  Isolate Acc: {test_acc_iso:.4f}")
            print(f"  Joint Acc (all {len(all_classes)} classes): {test_acc_joint:.4f}")
            
            self.result_logger.add_results(acc_list, test_acc_joint)
        
        # 打印accuracy matrix
        self.result_logger.print_matrix()
        
        # 打印joint accuracy summary
        print(f"\n{'='*60}")
        print("Joint Accuracy Summary (All Trained Classes at Each Session)")
        print(f"{'='*60}")
        for record in joint_acc_history:
            print(f"  Session {record['session']}: Classes {record['classes']} -> Acc: {record['joint_acc']:.4f}")
        print(f"{'='*60}\n")
        
        return self.result_logger
