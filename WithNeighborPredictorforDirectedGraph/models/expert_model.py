"""
Directed Expert-based Continual Learning Model

Model structure:
- Each stage has one expert
- Each expert contains:
  1. Classification Expert: GCN(1-layer) + classifier (on undirected graph)
  2. Out-Neighbor Predictor: predicts A->B (on directed graph)
  3. In-Neighbor Predictor: predicts A<-B (on directed graph)

Neighbor predictors:
- Main node A: GCN embedding -> main_mlp -> normalize
- Other node B: raw embedding -> other_mlp -> normalize
- Out-predictor trains on edges A->B where A is target class node
- In-predictor trains on edges B->A where A is target class node

Training strategy:
- Phase 1 (cls_epochs): Train classification model
- Phase 2 (out_epochs): Train out-neighbor predictor (GCN frozen)
- Phase 3 (remaining): Train in-neighbor predictor (GCN frozen)

Expert selection:
- Count n1 (out-degree), n2 (in-degree) for test node
- If n1 >= n2: use out-predictor; else: use in-predictor
- Compute top-(n+u) predicted neighbors, count correct
- Select expert with most correct predictions
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
    """Classification expert: single-layer GCN + classifier head."""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv = GCNConv(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def get_embeddings(self, x, edge_index):
        """Get GCN-encoded embeddings."""
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        """Forward pass, returns logits and embeddings."""
        embeddings = self.get_embeddings(x, edge_index)
        logits = self.classifier(embeddings)
        return logits, embeddings


class DirectedNeighborPredictor(nn.Module):
    """
    Directed neighbor predictor.

    - main_mlp: processes GCN embedding of the main node (target class node)
    - other_mlp: processes raw embedding of the other node
    - Both outputs are L2-normalized
    """

    def __init__(self, gcn_dim, raw_dim, hidden_dim=128):
        super().__init__()
        self.main_mlp = nn.Sequential(
            nn.Linear(gcn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.other_mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward_main(self, gcn_embed):
        """Process GCN embedding of main node, normalize."""
        h = self.main_mlp(gcn_embed)
        return F.normalize(h, p=2, dim=-1)

    def forward_other(self, raw_embed):
        """Process raw embedding of other node, normalize."""
        h = self.other_mlp(raw_embed)
        return F.normalize(h, p=2, dim=-1)


class Expert(nn.Module):
    """Single expert with classifier + out-predictor + in-predictor."""

    def __init__(self, input_dim, hidden_dim, num_classes,
                 neighbor_hidden_dim=128, dropout=0.5):
        super().__init__()
        self.classifier = ClassificationExpert(input_dim, hidden_dim, num_classes, dropout)
        self.out_predictor = DirectedNeighborPredictor(hidden_dim, input_dim, neighbor_hidden_dim)
        self.in_predictor = DirectedNeighborPredictor(hidden_dim, input_dim, neighbor_hidden_dim)

    def classify(self, x, edge_index):
        return self.classifier(x, edge_index)

    def get_gcn_embeddings(self, x, edge_index):
        return self.classifier.get_embeddings(x, edge_index)


class DirectedExpertCLModel(nn.Module):
    """Model containing multiple directed experts."""

    def __init__(self, input_dim, hidden_dim, num_classes, num_experts,
                 neighbor_hidden_dim=128, dropout=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes

        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, num_classes, neighbor_hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        self.active_expert = 0

    def classify(self, x, edge_index, expert_id=None):
        if expert_id is None:
            expert_id = self.active_expert
        return self.experts[expert_id].classify(x, edge_index)

    def freeze_expert(self, expert_id):
        for param in self.experts[expert_id].parameters():
            param.requires_grad = False

    def unfreeze_expert(self, expert_id):
        for param in self.experts[expert_id].parameters():
            param.requires_grad = True

    def freeze_all_except(self, expert_id):
        for i in range(self.num_experts):
            if i == expert_id:
                self.unfreeze_expert(i)
            else:
                self.freeze_expert(i)

    def set_training_phase(self, expert_id, phase):
        """
        Set training phase for a specific expert.

        Args:
            expert_id: Expert ID
            phase: 'classification', 'out_neighbor', or 'in_neighbor'
        """
        self.freeze_all_except(expert_id)
        expert = self.experts[expert_id]

        if phase == 'classification':
            for param in expert.classifier.parameters():
                param.requires_grad = True
            for param in expert.out_predictor.parameters():
                param.requires_grad = False
            for param in expert.in_predictor.parameters():
                param.requires_grad = False
        elif phase == 'out_neighbor':
            for param in expert.classifier.parameters():
                param.requires_grad = False
            for param in expert.out_predictor.parameters():
                param.requires_grad = True
            for param in expert.in_predictor.parameters():
                param.requires_grad = False
        elif phase == 'in_neighbor':
            for param in expert.classifier.parameters():
                param.requires_grad = False
            for param in expert.out_predictor.parameters():
                param.requires_grad = False
            for param in expert.in_predictor.parameters():
                param.requires_grad = True


class DirectedExpertCL:
    """
    Directed Expert-based Continual Learning Trainer.

    Three-phase training per session:
    1. Classification on undirected graph
    2. Out-neighbor prediction on directed graph
    3. In-neighbor prediction on directed graph
    """

    def __init__(self, task_loader, result_logger, config, checkpoint_path,
                 seed, device):
        self.task_loader = task_loader
        self.result_logger = result_logger
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.device = device

        # Model parameters
        self.input_dim = task_loader.data.x.shape[1]
        self.num_classes = task_loader.data.y.max().item() + 1
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']

        # Expert parameters
        self.num_experts = config.get('num_experts', 5)
        self.neighbor_hidden_dim = config.get('neighbor_hidden_dim', 128)
        self.num_neg_samples = config.get('num_neg_samples', 3)
        self.neighbor_topk_offset = config.get('neighbor_topk_offset', 3)

        # Training parameters
        self.cls_epochs = config.get('cls_epochs', 100)
        self.out_epochs = config.get('out_epochs', 100)
        self.debug = config.get('debug', False)

        # Create model
        self.model = DirectedExpertCLModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_experts=self.num_experts,
            neighbor_hidden_dim=self.neighbor_hidden_dim,
            dropout=self.dropout,
        ).to(device)

        self.current_session = 0

    # ======================== Training Functions ========================

    def train_classification_epoch(self, session_id, subgraph, train_loader,
                                   optimizer, curr_classes, all_classes):
        """Train classifier for one epoch on undirected graph."""
        self.model.train()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)

        total_loss = 0.0
        num_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()

            logits, _ = self.model.classify(x, edge_index, expert_id)

            node_ids = batch['node_id']
            labels = batch['labels'].to(self.device)
            batch_logits = logits[node_ids]

            max_class = max(all_classes) + 1
            batch_logits = batch_logits[:, :max_class]

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

    def out_neighbor_prediction_loss(self, x, gcn_embed, directed_ei,
                                     train_nodes, expert_id):
        """
        Compute out-neighbor prediction loss.

        Positive: A->B where both A and B are in train_nodes
        Negative: random (A, B') where A->B' does NOT exist, both in train_nodes
        """
        predictor = self.model.experts[expert_id].out_predictor
        device = x.device

        src, dst = directed_ei[0], directed_ei[1]

        train_tensor = torch.tensor(sorted(train_nodes), device=device)
        src_in_train = torch.isin(src, train_tensor)
        dst_in_train = torch.isin(dst, train_tensor)
        both_in_train = src_in_train & dst_in_train
        pos_src = src[both_in_train]
        pos_dst = dst[both_in_train]

        num_pos = pos_src.size(0)
        if num_pos == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Positive scores
        h_main = predictor.forward_main(gcn_embed[pos_src])
        h_other = predictor.forward_other(x[pos_dst])
        pos_scores = (h_main * h_other).sum(dim=1)

        # Negative sampling: both endpoints from train_nodes
        num_neg = num_pos * self.num_neg_samples
        max_id = x.size(0)
        pos_keys = pos_src.long() * max_id + pos_dst.long()

        oversample = int(num_neg * 1.5) + 100

        neg_A_idx = torch.randint(0, len(train_tensor), (oversample,), device=device)
        neg_B_idx = torch.randint(0, len(train_tensor), (oversample,), device=device)
        neg_A = train_tensor[neg_A_idx]
        neg_B = train_tensor[neg_B_idx]

        # Filter: no self-loops, no positive edges
        neg_keys = neg_A.long() * max_id + neg_B.long()
        not_self = neg_A != neg_B
        not_positive = ~torch.isin(neg_keys, pos_keys)
        valid = not_self & not_positive

        neg_A = neg_A[valid][:num_neg]
        neg_B = neg_B[valid][:num_neg]

        if neg_A.size(0) == 0:
            neg_loss = torch.tensor(0.0, device=device)
        else:
            h_neg_main = predictor.forward_main(gcn_embed[neg_A])
            h_neg_other = predictor.forward_other(x[neg_B])
            neg_scores = (h_neg_main * h_neg_other).sum(dim=1)
            neg_loss = -F.logsigmoid(-neg_scores).mean()

        pos_loss = -F.logsigmoid(pos_scores).mean()

        return pos_loss + neg_loss

    def in_neighbor_prediction_loss(self, x, gcn_embed, directed_ei,
                                    train_nodes, expert_id):
        """
        Compute in-neighbor prediction loss.

        Positive: B->A where both A and B are in train_nodes
        Negative: random (A, B') where B'->A does NOT exist, both in train_nodes
        """
        predictor = self.model.experts[expert_id].in_predictor
        device = x.device

        src, dst = directed_ei[0], directed_ei[1]

        train_tensor = torch.tensor(sorted(train_nodes), device=device)
        src_in_train = torch.isin(src, train_tensor)
        dst_in_train = torch.isin(dst, train_tensor)
        both_in_train = src_in_train & dst_in_train
        pos_B = src[both_in_train]  # departure point
        pos_A = dst[both_in_train]  # arrival point (main node)

        num_pos = pos_A.size(0)
        if num_pos == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Positive scores: A is main node, B is other node
        h_main = predictor.forward_main(gcn_embed[pos_A])
        h_other = predictor.forward_other(x[pos_B])
        pos_scores = (h_main * h_other).sum(dim=1)

        # Negative sampling: both endpoints from train_nodes
        num_neg = num_pos * self.num_neg_samples
        max_id = x.size(0)
        pos_keys = pos_A.long() * max_id + pos_B.long()

        oversample = int(num_neg * 1.5) + 100

        neg_A_idx = torch.randint(0, len(train_tensor), (oversample,), device=device)
        neg_B_idx = torch.randint(0, len(train_tensor), (oversample,), device=device)
        neg_A = train_tensor[neg_A_idx]
        neg_B_prime = train_tensor[neg_B_idx]

        # Filter: no self-loops, no positive edges
        neg_keys = neg_A.long() * max_id + neg_B_prime.long()
        not_self = neg_A != neg_B_prime
        not_positive = ~torch.isin(neg_keys, pos_keys)
        valid = not_self & not_positive

        neg_A = neg_A[valid][:num_neg]
        neg_B_prime = neg_B_prime[valid][:num_neg]

        if neg_A.size(0) == 0:
            neg_loss = torch.tensor(0.0, device=device)
        else:
            h_neg_main = predictor.forward_main(gcn_embed[neg_A])
            h_neg_other = predictor.forward_other(x[neg_B_prime])
            neg_scores = (h_neg_main * h_neg_other).sum(dim=1)
            neg_loss = -F.logsigmoid(-neg_scores).mean()

        pos_loss = -F.logsigmoid(pos_scores).mean()

        return pos_loss + neg_loss

    def train_out_neighbor_epoch(self, session_id, subgraph, gcn_embed,
                                 optimizer, train_nodes):
        """Train out-predictor for one epoch using only train_nodes."""
        self.model.train()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        directed_ei = subgraph['directed_edge_index'].to(self.device)

        optimizer.zero_grad()
        loss = self.out_neighbor_prediction_loss(
            x, gcn_embed, directed_ei, train_nodes, expert_id
        )
        loss.backward()
        optimizer.step()

        return loss.item()

    def train_in_neighbor_epoch(self, session_id, subgraph, gcn_embed,
                                optimizer, train_nodes):
        """Train in-predictor for one epoch using only train_nodes."""
        self.model.train()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        directed_ei = subgraph['directed_edge_index'].to(self.device)

        optimizer.zero_grad()
        loss = self.in_neighbor_prediction_loss(
            x, gcn_embed, directed_ei, train_nodes, expert_id
        )
        loss.backward()
        optimizer.step()

        return loss.item()

    # ======================== Validation Functions ========================

    @torch.no_grad()
    def evaluate_classifier(self, session_id, subgraph, data_loader,
                            curr_classes, all_classes):
        """Validate classifier independently (no expert selection)."""
        self.model.eval()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)

        all_preds = []
        all_labels = []

        for batch in data_loader:
            node_ids = batch['node_id']
            labels = batch['labels'].to(self.device)

            logits, _ = self.model.classify(x, edge_index, expert_id)
            batch_logits = logits[node_ids]

            max_class = max(all_classes) + 1
            batch_logits = batch_logits[:, :max_class]

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
    def evaluate_out_predictor(self, session_id, subgraph, data_loader):
        """Validate out-predictor accuracy on validation set."""
        self.model.eval()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        directed_ei = subgraph['directed_edge_index'].to(self.device)

        gcn_embed = self.model.experts[expert_id].get_gcn_embeddings(x, edge_index)
        predictor = self.model.experts[expert_id].out_predictor

        # Build out-adjacency
        src, dst = directed_ei[0], directed_ei[1]
        num_nodes = x.size(0)
        out_adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=self.device)
        out_adj[src, dst] = True

        valid_nodes = []
        for batch in data_loader:
            valid_nodes.extend(batch['node_id'].tolist())

        total_correct = 0
        total_neighbors = 0

        h_main_all = predictor.forward_main(gcn_embed)
        h_other_all = predictor.forward_other(x)

        for node_idx in valid_nodes:
            true_out = out_adj[node_idx]
            n = true_out.sum().item()
            if n == 0:
                continue

            scores = torch.matmul(h_main_all[node_idx], h_other_all.T)
            scores[node_idx] = float('-inf')

            k = min(n + self.neighbor_topk_offset, num_nodes - 1)
            k = max(k, 1)
            _, top_k = torch.topk(scores, k)

            predicted = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            predicted[top_k] = True
            correct = (predicted & true_out).sum().item()
            total_correct += correct
            total_neighbors += n

        return total_correct / total_neighbors if total_neighbors > 0 else 0.0

    @torch.no_grad()
    def evaluate_in_predictor(self, session_id, subgraph, data_loader):
        """Validate in-predictor accuracy on validation set."""
        self.model.eval()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        directed_ei = subgraph['directed_edge_index'].to(self.device)

        gcn_embed = self.model.experts[expert_id].get_gcn_embeddings(x, edge_index)
        predictor = self.model.experts[expert_id].in_predictor

        # Build in-adjacency: in_adj[A, B] = True means B->A exists
        src, dst = directed_ei[0], directed_ei[1]
        num_nodes = x.size(0)
        in_adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=self.device)
        in_adj[dst, src] = True  # For edge src->dst, dst has in-neighbor src

        valid_nodes = []
        for batch in data_loader:
            valid_nodes.extend(batch['node_id'].tolist())

        total_correct = 0
        total_neighbors = 0

        h_main_all = predictor.forward_main(gcn_embed)
        h_other_all = predictor.forward_other(x)

        for node_idx in valid_nodes:
            true_in = in_adj[node_idx]
            n = true_in.sum().item()
            if n == 0:
                continue

            scores = torch.matmul(h_main_all[node_idx], h_other_all.T)
            scores[node_idx] = float('-inf')

            k = min(n + self.neighbor_topk_offset, num_nodes - 1)
            k = max(k, 1)
            _, top_k = torch.topk(scores, k)

            predicted = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            predicted[top_k] = True
            correct = (predicted & true_in).sum().item()
            total_correct += correct
            total_neighbors += n

        return total_correct / total_neighbors if total_neighbors > 0 else 0.0

    # ======================== Expert Selection ========================

    @torch.no_grad()
    def compute_directed_neighbor_accuracy(self, expert_id, x, edge_index,
                                           directed_edge_index, node_idx, use_out):
        """
        Compute directed neighbor prediction accuracy for expert selection.

        Args:
            expert_id: Expert ID
            x: Node features
            edge_index: Undirected edge index with self-loops (for GCN)
            directed_edge_index: Directed edge index
            node_idx: Target node index
            use_out: True for out-predictor, False for in-predictor

        Returns:
            correct_count: Number of correctly predicted neighbors
            position_sum: Sum of positions of true neighbors in prediction
        """
        self.model.eval()
        expert = self.model.experts[expert_id]
        gcn_embed = expert.get_gcn_embeddings(x, edge_index)

        predictor = expert.out_predictor if use_out else expert.in_predictor

        src, dst = directed_edge_index[0], directed_edge_index[1]
        num_nodes = x.size(0)
        device = x.device

        if use_out:
            # True out-neighbors: B where node_idx->B
            mask = src == node_idx
            true_neighbors = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            true_neighbors[dst[mask]] = True
        else:
            # True in-neighbors: B where B->node_idx
            mask = dst == node_idx
            true_neighbors = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            true_neighbors[src[mask]] = True

        n = true_neighbors.sum().item()
        if n == 0:
            return 0, 0

        h_main = predictor.forward_main(gcn_embed[node_idx:node_idx + 1])
        h_other = predictor.forward_other(x)
        scores = torch.matmul(h_main, h_other.T).squeeze(0)
        scores[node_idx] = float('-inf')

        k = min(n + self.neighbor_topk_offset, num_nodes - 1)
        k = max(k, 1)
        _, top_k = torch.topk(scores, k)

        predicted = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        predicted[top_k] = True
        correct = (predicted & true_neighbors).sum().item()

        pos_sum = 0
        for pos, node in enumerate(top_k.tolist()):
            if true_neighbors[node]:
                pos_sum += (pos + 1)

        return correct, pos_sum

    @torch.no_grad()
    def select_expert(self, x, edge_index, directed_edge_index, node_idx):
        """
        Select best expert for a node.

        Uses out-predictor if out-degree >= in-degree, else in-predictor.
        """
        num_experts = min(self.num_experts, self.current_session + 1)

        src, dst = directed_edge_index[0], directed_edge_index[1]
        n1 = (src == node_idx).sum().item()  # out-degree
        n2 = (dst == node_idx).sum().item()  # in-degree

        use_out = (n1 >= n2)

        n = n1 if use_out else n2
        if n == 0:
            # No directed edges, random selection
            return random.randint(0, num_experts - 1), []

        expert_scores = []
        for exp_id in range(num_experts):
            correct, pos_sum = self.compute_directed_neighbor_accuracy(
                exp_id, x, edge_index, directed_edge_index, node_idx, use_out
            )
            expert_scores.append((exp_id, correct, pos_sum))

        max_correct = max(c for _, c, _ in expert_scores)
        best_by_correct = [(e, p) for e, c, p in expert_scores if c == max_correct]
        min_pos = min(p for _, p in best_by_correct)
        best_by_pos = [e for e, p in best_by_correct if p == min_pos]

        return random.choice(best_by_pos), expert_scores

    # ======================== Evaluation ========================

    @torch.no_grad()
    def evaluate(self, subgraph, test_loader, trained_classes):
        """
        Full evaluation with expert selection.

        For each test node:
        1. Select best expert via directed neighbor prediction
        2. Classify using that expert's classifier on undirected graph
        """
        self.model.eval()

        num_experts = min(self.num_experts, self.current_session + 1)
        expert_counts = {i: 0 for i in range(num_experts)}

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        directed_ei = subgraph['directed_edge_index'].to(self.device)

        all_preds = []
        all_labels = []

        for batch in test_loader:
            node_ids = batch['node_id']
            labels = batch['labels']

            for i, node_idx in enumerate(node_ids):
                node_idx = node_idx.item()

                best_expert, _ = self.select_expert(
                    x, edge_index, directed_ei, node_idx
                )
                expert_counts[best_expert] += 1

                logits, _ = self.model.classify(x, edge_index, best_expert)
                node_logits = logits[node_idx:node_idx + 1]

                max_class = max(trained_classes) + 1
                node_logits = node_logits[:, :max_class]
                pred = torch.argmax(node_logits, dim=1)

                all_preds.append(pred.cpu())
                all_labels.append(labels[i])

        if len(all_preds) == 0:
            return 0.0, 0.0

        if self.debug:
            print(f"  Expert distribution: {expert_counts}")

        preds = torch.cat(all_preds).numpy()
        labels = torch.stack(all_labels).numpy()

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return acc, f1

    # ======================== Main Training Loop ========================

    def fit(self, trial):
        """Main training loop."""
        joint_acc_history = []
        in_epochs = self.config['epochs'] - self.cls_epochs - self.out_epochs

        if in_epochs <= 0:
            raise ValueError(
                f"epochs ({self.config['epochs']}) must be > "
                f"cls_epochs ({self.cls_epochs}) + out_epochs ({self.out_epochs})"
            )

        for session_id in range(self.task_loader.sessions):
            self.current_session = session_id

            # Load checkpoint from previous session
            if session_id > 0:
                load_checkpoint(self.model, self.checkpoint_path,
                                'DirectedExpertCL', self.seed)

            # Get task data
            (curr_classes, all_classes,
             subgraph, joint_subgraph,
             train_loader, valid_loader,
             test_loader_iso, test_loader_joint) = self.task_loader.get_task(session_id)

            expert_id = session_id % self.num_experts
            train_nodes = set(self.task_loader.train_idx_per_task[session_id])

            print(f"\n{'='*60}")
            print(f"Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"Expert {expert_id}")
            print(f"{'='*60}")

            # ========== Phase 1: Classification ==========
            self.model.set_training_phase(expert_id, 'classification')
            optimizer = optim.Adam(
                self.model.experts[expert_id].classifier.parameters(),
                lr=float(self.config['lr']),
                weight_decay=float(self.config['weight_decay'])
            )
            best_cls_acc = 0.0
            cls_patience = 0

            pbar = tqdm(range(self.cls_epochs),
                        desc=f"Session {session_id} - Classification")
            for epoch in pbar:
                loss = self.train_classification_epoch(
                    session_id, subgraph, train_loader,
                    optimizer, curr_classes, all_classes
                )

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    valid_acc = self.evaluate_classifier(
                        session_id, subgraph, valid_loader,
                        curr_classes, all_classes
                    )
                    if valid_acc > best_cls_acc:
                        best_cls_acc = valid_acc
                        cls_patience = 0
                        save_checkpoint(self.model, optimizer, epoch,
                                        self.checkpoint_path,
                                        'DirectedExpertCL', self.seed)
                    else:
                        cls_patience += 1

                    pbar.set_postfix({'loss': f'{loss:.4f}',
                                      'cls_val': f'{valid_acc:.4f}'})
                else:
                    pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Load best classifier
            load_checkpoint(self.model, self.checkpoint_path,
                            'DirectedExpertCL', self.seed)

            # Pre-compute frozen GCN embeddings
            self.model.eval()
            with torch.no_grad():
                x = subgraph['x'].to(self.device)
                edge_index = subgraph['edge_index'].to(self.device)
                gcn_embed = self.model.experts[expert_id].get_gcn_embeddings(
                    x, edge_index
                ).detach()

            # ========== Phase 2: Out-Predictor ==========
            self.model.set_training_phase(expert_id, 'out_neighbor')
            optimizer = optim.Adam(
                self.model.experts[expert_id].out_predictor.parameters(),
                lr=float(self.config['lr']),
                weight_decay=float(self.config['weight_decay'])
            )
            best_out_acc = 0.0
            out_patience = 0

            pbar = tqdm(range(self.out_epochs),
                        desc=f"Session {session_id} - Out-Predictor")
            for epoch in pbar:
                loss = self.train_out_neighbor_epoch(
                    session_id, subgraph, gcn_embed, optimizer, train_nodes
                )

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    valid_acc = self.evaluate_out_predictor(
                        session_id, subgraph, valid_loader
                    )
                    if valid_acc > best_out_acc:
                        best_out_acc = valid_acc
                        out_patience = 0
                        save_checkpoint(self.model, optimizer, epoch,
                                        self.checkpoint_path,
                                        'DirectedExpertCL', self.seed)
                    else:
                        out_patience += 1
                        if out_patience > self.config['patience']:
                            break

                    pbar.set_postfix({'loss': f'{loss:.4f}',
                                      'out_val': f'{valid_acc:.4f}'})
                else:
                    pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Load best out-predictor
            load_checkpoint(self.model, self.checkpoint_path,
                            'DirectedExpertCL', self.seed)

            # ========== Phase 3: In-Predictor ==========
            self.model.set_training_phase(expert_id, 'in_neighbor')
            optimizer = optim.Adam(
                self.model.experts[expert_id].in_predictor.parameters(),
                lr=float(self.config['lr']),
                weight_decay=float(self.config['weight_decay'])
            )
            best_in_acc = 0.0
            in_patience = 0

            pbar = tqdm(range(in_epochs),
                        desc=f"Session {session_id} - In-Predictor")
            for epoch in pbar:
                loss = self.train_in_neighbor_epoch(
                    session_id, subgraph, gcn_embed, optimizer, train_nodes
                )

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    valid_acc = self.evaluate_in_predictor(
                        session_id, subgraph, valid_loader
                    )
                    if valid_acc > best_in_acc:
                        best_in_acc = valid_acc
                        in_patience = 0
                        save_checkpoint(self.model, optimizer, epoch,
                                        self.checkpoint_path,
                                        'DirectedExpertCL', self.seed)
                    else:
                        in_patience += 1
                        if in_patience > self.config['patience']:
                            break

                    pbar.set_postfix({'loss': f'{loss:.4f}',
                                      'in_val': f'{valid_acc:.4f}'})
                else:
                    pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Load best model (all three components)
            load_checkpoint(self.model, self.checkpoint_path,
                            'DirectedExpertCL', self.seed)

            print(f"\n  Phase results: cls={best_cls_acc:.4f}, "
                  f"out={best_out_acc:.4f}, in={best_in_acc:.4f}")

            # ========== Testing ==========
            test_acc_iso, _ = self.evaluate(subgraph, test_loader_iso, all_classes)
            test_acc_joint, _ = self.evaluate(
                joint_subgraph, test_loader_joint, all_classes
            )

            # Evaluate previous tasks
            acc_list = []
            for prev_session in range(session_id):
                prev_subgraph = self.task_loader.subgraph_per_task[prev_session]
                prev_test_idx = self.task_loader.test_idx_per_task[prev_session]
                prev_test_dataset = torch.utils.data.Subset(
                    self.task_loader.dataset, prev_test_idx
                )
                prev_test_loader = torch.utils.data.DataLoader(
                    prev_test_dataset,
                    batch_size=self.config['batch_size'], shuffle=False
                )
                prev_acc, _ = self.evaluate(
                    prev_subgraph, prev_test_loader, all_classes
                )
                acc_list.append(prev_acc)
            acc_list.append(test_acc_iso)

            joint_acc_history.append({
                'session': session_id,
                'classes': all_classes.copy(),
                'joint_acc': test_acc_joint
            })

            print(f"\nSession {session_id} Results:")
            print(f"  Isolate Acc: {test_acc_iso:.4f}")
            print(f"  Joint Acc (all {len(all_classes)} classes): {test_acc_joint:.4f}")

            self.result_logger.add_results(acc_list, test_acc_joint)

        # Print final results
        self.result_logger.print_matrix()

        print(f"\n{'='*60}")
        print("Joint Accuracy Summary")
        print(f"{'='*60}")
        for record in joint_acc_history:
            print(f"  Session {record['session']}: "
                  f"Classes {record['classes']} -> Acc: {record['joint_acc']:.4f}")
        print(f"{'='*60}\n")

        return self.result_logger
