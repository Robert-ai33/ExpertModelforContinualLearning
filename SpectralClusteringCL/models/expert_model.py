"""
Expert-based Continual Learning Model with Leiden Clustering

Model structure:
- Each session trains one expert (expert_id = session_id % num_experts)
- Each expert contains 2 independent components:
  1. Neighbor Predictor (PredictorMLP): cosine similarity based, for expert selection
  2. Cross-class Edge Classifier (EdgeClassifierMLP): multi-signal fusion MLP
     Input: concat(x_u * x_v, |x_u - x_v|, x_u + x_v) -> MLP -> binary logit
     Loss: BCEWithLogitsLoss

Training: 2 phases per session
  - Phase 1: Neighbor predictor (contrastive loss)
  - Phase 2: Cross-class classifier (BCE loss, balanced sampling, re-randomized each epoch)
Inference: delete cross-class edges -> Leiden clustering on modified graph
Evaluation: cluster purity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import igraph as ig
import leidenalg

from tqdm import tqdm

from utils import save_checkpoint, load_checkpoint, PurityMetric


# ==============================================================================
# Model Components
# ==============================================================================

class PredictorMLP(nn.Module):
    """
    MLP predictor: maps raw features to normalized embedding space.
    Used for neighbor prediction (cosine similarity based).
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        """Returns L2-normalized embedding."""
        h = self.mlp(x)
        return F.normalize(h, p=2, dim=-1)


class EdgeClassifierMLP(nn.Module):
    """
    Edge binary classifier using multi-signal fusion.
    Input: concat(x_u * x_v, |x_u - x_v|, x_u + x_v) -> MLP -> logit
    Fuses element-wise product, absolute difference, and sum to capture
    feature co-occurrence, divergence, and shared characteristics.
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_u, x_v):
        """
        Args:
            x_u: [N, D] raw features of source nodes
            x_v: [N, D] raw features of destination nodes

        Returns:
            logits: [N] raw scores (before sigmoid)
        """
        edge_feat = torch.cat([
            x_u * x_v,
            torch.abs(x_u - x_v),
            x_u + x_v,
        ], dim=-1)
        return self.mlp(edge_feat).squeeze(-1)


class Expert(nn.Module):
    """Single expert with neighbor predictor (cosine) + cross-class classifier (edge MLP)."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.neighbor_predictor = PredictorMLP(input_dim, hidden_dim)
        self.cross_class_predictor = EdgeClassifierMLP(input_dim, hidden_dim)


class SpectralExpertCLModel(nn.Module):
    """Multi-expert model container."""

    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim) for _ in range(num_experts)
        ])

    def freeze_all_except(self, expert_id):
        """Freeze all experts except the specified one."""
        for i, expert in enumerate(self.experts):
            requires_grad = (i == expert_id)
            for param in expert.parameters():
                param.requires_grad = requires_grad

    def set_training_phase(self, expert_id, phase):
        """
        Set training phase: only unfreeze the target MLP of the target expert.

        Args:
            expert_id: expert index
            phase: 'neighbor' or 'cross_class'
        """
        self.freeze_all_except(expert_id)
        expert = self.experts[expert_id]

        # First freeze all within the expert
        for param in expert.parameters():
            param.requires_grad = False

        # Then unfreeze only the target MLP
        if phase == 'neighbor':
            for param in expert.neighbor_predictor.parameters():
                param.requires_grad = True
        elif phase == 'cross_class':
            for param in expert.cross_class_predictor.parameters():
                param.requires_grad = True


# ==============================================================================
# Leiden Community Detection
# ==============================================================================

def leiden_clustering(edge_index, num_nodes, resolution=1.0):
    """
    Leiden community detection for undirected graphs.

    The number of communities is determined automatically by the algorithm.

    Args:
        edge_index: [2, E] tensor of edges (undirected, no self-loops)
        num_nodes: number of nodes
        resolution: resolution parameter (higher -> more communities)

    Returns:
        labels: [num_nodes] array of community assignments
    """
    if edge_index.size(1) == 0:
        # No edges: each node is its own community
        return np.arange(num_nodes)

    src = edge_index[0].cpu().numpy().tolist()
    dst = edge_index[1].cpu().numpy().tolist()

    # Build igraph
    g = ig.Graph(n=num_nodes, directed=False)
    g.add_edges(list(zip(src, dst)))
    g.simplify()  # remove multi-edges and self-loops

    # Run Leiden with RBConfigurationVertexPartition
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )

    labels = np.array(partition.membership)
    return labels


def compute_cluster_purity(cluster_labels, true_labels):
    """
    Compute per-cluster purity and overall purity.
    Works with any number of clusters (determined by cluster_labels).

    Args:
        cluster_labels: [N] cluster assignments
        true_labels: [N] ground truth class labels

    Returns:
        cluster_purities: list of (cluster_id, purity, size, dominant_class)
        overall_purity: weighted average purity
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_purities = []
    total_correct = 0
    total_nodes = len(cluster_labels)

    for c in unique_clusters:
        mask = cluster_labels == c
        size = mask.sum()

        if size == 0:
            continue

        cluster_true = true_labels[mask]
        unique, counts = np.unique(cluster_true, return_counts=True)
        max_idx = np.argmax(counts)
        dominant_class = int(unique[max_idx])
        max_count = int(counts[max_idx])

        purity = max_count / size
        total_correct += max_count

        cluster_purities.append((int(c), purity, int(size), dominant_class))

    overall_purity = total_correct / total_nodes if total_nodes > 0 else 0.0
    return cluster_purities, overall_purity


# ==============================================================================
# Main Trainer
# ==============================================================================

class SpectralExpertCL:
    """
    Expert-based Continual Learning Trainer with Leiden Clustering.

    Two-phase training per session:
    1. Neighbor predictor (for expert selection)
    2. Cross-class predictor (for edge removal; positive=cross-class edges,
       negative=same-class edges, balanced sampling)

    Inference: delete cross-class edges -> Leiden clustering -> purity evaluation
    """

    def __init__(self, task_loader, config, checkpoint_path,
                 dataset, seed, device):
        self.task_loader = task_loader
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.seed = seed
        self.device = device

        # Model parameters
        self.input_dim = task_loader.data.x.shape[1]
        self.hidden_dim = config['hidden_dim']
        self.num_experts = config.get('num_experts', 5)

        # Training parameters
        self.neighbor_epochs = config.get('neighbor_epochs', 100)
        self.cross_class_epochs = config.get('cross_class_epochs', 100)
        self.num_neg_samples = config.get('num_neg_samples', 3)

        # Inference parameters
        self.enable_delete_cross_class = config.get('enable_delete_cross_class', True)
        self.cross_class_threshold = config.get('cross_class_threshold', 0.5)

        # Leiden clustering
        self.leiden_resolution = config.get('leiden_resolution', 1.0)

        self.debug = config.get('debug', False)

        # Create model
        self.model = SpectralExpertCLModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
        ).to(device)

        self.current_session = 0

    # ==================================================================
    # Training Losses
    # ==================================================================

    def neighbor_prediction_loss(self, x, edge_index, expert_id):
        """
        Compute neighbor prediction loss.
        Positive: all edges. Negative: random non-edges.
        """
        predictor = self.model.experts[expert_id].neighbor_predictor
        h = predictor(x)

        num_nodes = x.size(0)
        device = x.device

        if edge_index.size(1) == 0 or num_nodes < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Positive
        pos_src, pos_dst = edge_index[0], edge_index[1]
        pos_scores = (h[pos_src] * h[pos_dst]).sum(dim=1)

        # Negative
        num_neg = pos_src.size(0) * self.num_neg_samples
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
        valid = neg_src != neg_dst
        neg_src, neg_dst = neg_src[valid], neg_dst[valid]

        neg_scores = (h[neg_src] * h[neg_dst]).sum(dim=1)

        # Loss
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean() if neg_scores.numel() > 0 else 0.0

        return pos_loss + neg_loss

    def cross_class_prediction_loss(self, x, edge_index, labels, expert_id):
        """
        Compute cross-class edge classification loss (BCE).

        Positive (label=1): edges where endpoints have different classes
        Negative (label=0): edges where endpoints have same class
        Sample count: min(num_cross_class, num_same_class) for both.
        No random non-edge pairs. Re-randomized each call (epoch).
        """
        predictor = self.model.experts[expert_id].cross_class_predictor

        device = x.device
        src, dst = edge_index[0], edge_index[1]

        if src.size(0) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Classify edges by endpoint labels
        src_labels = labels[src]
        dst_labels = labels[dst]
        cross_class_mask = src_labels != dst_labels
        same_class_mask = ~cross_class_mask

        # Cross-class edges (positive, label=1)
        cross_src = src[cross_class_mask]
        cross_dst = dst[cross_class_mask]
        num_cross = cross_src.size(0)

        # Same-class edges (negative, label=0)
        same_src = src[same_class_mask]
        same_dst = dst[same_class_mask]
        num_same = same_src.size(0)

        if num_cross == 0 or num_same == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Balanced sampling: n_samples = min(num_cross, num_same)
        n_samples = min(num_cross, num_same)

        # Sample from positive (cross-class edges)
        if num_cross > n_samples:
            perm = torch.randperm(num_cross, device=device)[:n_samples]
            pos_src = cross_src[perm]
            pos_dst = cross_dst[perm]
        else:
            pos_src = cross_src
            pos_dst = cross_dst

        # Sample from negative (same-class edges)
        if num_same > n_samples:
            perm = torch.randperm(num_same, device=device)[:n_samples]
            neg_src = same_src[perm]
            neg_dst = same_dst[perm]
        else:
            neg_src = same_src
            neg_dst = same_dst

        # Compute logits via edge classifier
        all_src = torch.cat([pos_src, neg_src])
        all_dst = torch.cat([pos_dst, neg_dst])
        logits = predictor(x[all_src], x[all_dst])

        # Labels: 1 for cross-class, 0 for same-class
        target = torch.cat([
            torch.ones(n_samples, device=device),
            torch.zeros(n_samples, device=device),
        ])

        return F.binary_cross_entropy_with_logits(logits, target)

    # ==================================================================
    # Training Epochs
    # ==================================================================

    def train_neighbor_epoch(self, session_id, subgraph, optimizer):
        """Train neighbor predictor for one epoch."""
        self.model.train()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index_no_selfloop'].to(self.device)

        optimizer.zero_grad()
        loss = self.neighbor_prediction_loss(x, edge_index, expert_id)
        loss.backward()
        optimizer.step()

        return loss.item()

    def train_cross_class_epoch(self, session_id, subgraph, optimizer):
        """Train cross-class neighbor predictor for one epoch."""
        self.model.train()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index_no_selfloop'].to(self.device)
        labels = subgraph['y'].to(self.device)

        optimizer.zero_grad()
        loss = self.cross_class_prediction_loss(
            x, edge_index, labels, expert_id
        )
        loss.backward()
        optimizer.step()

        return loss.item()

    # ==================================================================
    # Validation
    # ==================================================================

    @torch.no_grad()
    def evaluate_neighbor_predictor(self, session_id, subgraph, valid_loader):
        """Validate neighbor predictor: compute loss on validation node edges."""
        self.model.eval()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index_no_selfloop'].to(self.device)

        # Collect validation node set
        valid_node_set = set()
        for batch in valid_loader:
            valid_node_set.update(batch['node_id'].tolist())

        valid_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
        for n in valid_node_set:
            valid_mask[n] = True

        src, dst = edge_index[0], edge_index[1]

        # Positive: edges involving at least one validation node
        involves_valid = valid_mask[src] | valid_mask[dst]
        val_src = src[involves_valid]
        val_dst = dst[involves_valid]

        if val_src.size(0) == 0:
            return float('inf')

        predictor = self.model.experts[expert_id].neighbor_predictor
        h = predictor(x)

        pos_scores = (h[val_src] * h[val_dst]).sum(dim=1)
        pos_loss = -F.logsigmoid(pos_scores).mean()

        # Negative: random non-edges
        num_neg = val_src.size(0) * self.num_neg_samples
        num_nodes = x.size(0)
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=self.device)
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=self.device)
        valid = neg_src != neg_dst
        neg_src, neg_dst = neg_src[valid], neg_dst[valid]

        if neg_src.size(0) > 0:
            neg_scores = (h[neg_src] * h[neg_dst]).sum(dim=1)
            neg_loss = -F.logsigmoid(-neg_scores).mean()
        else:
            neg_loss = 0.0

        return (pos_loss + neg_loss).item()

    @torch.no_grad()
    def evaluate_cross_class_predictor(self, session_id, subgraph, valid_loader):
        """
        Validate cross-class predictor: compute BCE loss on validation node edges.
        Same balanced sampling logic as training.
        """
        self.model.eval()
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index_no_selfloop'].to(self.device)
        labels = subgraph['y'].to(self.device)

        # Collect validation node set
        valid_node_set = set()
        for batch in valid_loader:
            valid_node_set.update(batch['node_id'].tolist())

        valid_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
        for n in valid_node_set:
            valid_mask[n] = True

        src, dst = edge_index[0], edge_index[1]

        # Edges involving at least one validation node
        involves_valid = valid_mask[src] | valid_mask[dst]
        val_src = src[involves_valid]
        val_dst = dst[involves_valid]

        if val_src.size(0) == 0:
            return float('inf')

        predictor = self.model.experts[expert_id].cross_class_predictor

        # Classify validation edges
        cross_mask = labels[val_src] != labels[val_dst]
        same_mask = ~cross_mask

        cross_src = val_src[cross_mask]
        cross_dst = val_dst[cross_mask]
        num_cross = cross_src.size(0)

        same_src = val_src[same_mask]
        same_dst = val_dst[same_mask]
        num_same = same_src.size(0)

        if num_cross == 0 or num_same == 0:
            return float('inf')

        # Balanced sampling
        n_samples = min(num_cross, num_same)

        if num_cross > n_samples:
            perm = torch.randperm(num_cross, device=self.device)[:n_samples]
            pos_src = cross_src[perm]
            pos_dst = cross_dst[perm]
        else:
            pos_src = cross_src
            pos_dst = cross_dst

        if num_same > n_samples:
            perm = torch.randperm(num_same, device=self.device)[:n_samples]
            neg_src = same_src[perm]
            neg_dst = same_dst[perm]
        else:
            neg_src = same_src
            neg_dst = same_dst

        # Compute logits via edge classifier
        all_src = torch.cat([pos_src, neg_src])
        all_dst = torch.cat([pos_dst, neg_dst])
        logits = predictor(x[all_src], x[all_dst])

        target = torch.cat([
            torch.ones(n_samples, device=self.device),
            torch.zeros(n_samples, device=self.device),
        ])

        return F.binary_cross_entropy_with_logits(logits, target).item()

    # ==================================================================
    # Inference: Graph Modification + Spectral Clustering
    # ==================================================================

    @torch.no_grad()
    def evaluate_spectral(self, joint_subgraph, test_loader, all_classes):
        """
        Full inference pipeline:
        1. Select expert + predict cross-class edges -> delete
        2. Build modified graph (target nodes only)
        3. Leiden clustering on modified graph
        4. Compute and return cluster purity

        Args:
            joint_subgraph: cumulative subgraph dict
            test_loader: joint test data loader (for collecting test nodes)
            all_classes: list of all seen classes

        Returns:
            cluster_purities, overall_purity, edge_modify_stats
        """
        self.model.eval()

        x = joint_subgraph['x'].to(self.device)
        edge_index = joint_subgraph['edge_index_no_selfloop'].to(self.device)
        labels = joint_subgraph['y']
        all_nodes = joint_subgraph['all_nodes']
        target_nodes = joint_subgraph['target_nodes']

        num_experts = min(self.num_experts, self.current_session + 1)

        # Precompute neighbor embeddings for all experts (for expert selection)
        precomputed_h = {}
        for exp_id in range(num_experts):
            h = self.model.experts[exp_id].neighbor_predictor(x)
            precomputed_h[exp_id] = h

        # Deduplicate edges: only keep (u, v) where u < v
        src, dst = edge_index[0], edge_index[1]
        mask = src < dst
        edge_src = src[mask]  # [num_unique_edges]
        edge_dst = dst[mask]

        # ---- Step 1: Delete cross-class edges (GPU batch) ----
        num_unique = edge_src.size(0)

        # Count ground-truth cross-class edges
        gt_labels_u = labels[edge_src.cpu()]
        gt_labels_v = labels[edge_dst.cpu()]
        total_cross_class = (gt_labels_u != gt_labels_v).sum().item()
        total_same_class = num_unique - total_cross_class

        if self.debug:
            print(f"  Ground truth: {num_unique} unique edges, "
                  f"{total_cross_class} cross-class ({total_cross_class/max(num_unique,1)*100:.1f}%), "
                  f"{total_same_class} same-class ({total_same_class/max(num_unique,1)*100:.1f}%)")

        if self.enable_delete_cross_class:
            if self.debug:
                print(f"  Evaluating {num_unique} unique edges "
                      f"for cross-class removal...")

            # Expert selection for edges: has edge -> select expert with highest
            # neighbor prediction score (most familiar with this pair)
            # [num_experts, num_unique_edges]
            neighbor_scores = torch.stack([
                (precomputed_h[exp_id][edge_src] *
                 precomputed_h[exp_id][edge_dst]).sum(dim=1)
                for exp_id in range(num_experts)
            ])
            best_experts_edge = neighbor_scores.argmax(dim=0)

            # Cross-class prediction via EdgeClassifierMLP (chunked to avoid OOM)
            # Adaptive chunk size based on available GPU memory
            feat_dim = x.size(1)
            if self.device.type == 'cuda':
                free_mem = torch.cuda.mem_get_info(self.device)[0]
                per_edge_bytes = feat_dim * 4 * 8  # peak memory per edge
                chunk_size = max(1024, int(free_mem * 0.3 / max(per_edge_bytes, 1)))
            else:
                chunk_size = num_unique  # CPU: no chunking needed

            # Compute logits per expert, chunked: [num_experts, num_unique_edges]
            cc_logits_list = []
            for exp_id in range(num_experts):
                predictor = self.model.experts[exp_id].cross_class_predictor
                chunks = []
                for start in range(0, num_unique, chunk_size):
                    end = min(start + chunk_size, num_unique)
                    chunk = predictor(
                        x[edge_src[start:end]], x[edge_dst[start:end]]
                    )
                    chunks.append(chunk)
                cc_logits_list.append(torch.cat(chunks))
            cc_logits = torch.stack(cc_logits_list)

            edge_idx = torch.arange(num_unique, device=self.device)
            final_cc_scores = torch.sigmoid(
                cc_logits[best_experts_edge, edge_idx]
            )

            # Keep edges where score <= threshold (NOT cross-class)
            keep_mask = final_cc_scores <= self.cross_class_threshold
            kept_src = edge_src[keep_mask]
            kept_dst = edge_dst[keep_mask]
            delete_mask = ~keep_mask
            deleted_count = delete_mask.sum().item()

            # Accuracy: among deleted edges, how many are truly cross-class?
            if deleted_count > 0:
                del_src = edge_src[delete_mask]
                del_dst = edge_dst[delete_mask]
                del_labels_u = labels[del_src.cpu()]
                del_labels_v = labels[del_dst.cpu()]
                truly_cross = (del_labels_u != del_labels_v).sum().item()
                delete_precision = truly_cross / deleted_count
            else:
                delete_precision = 0.0

            if self.debug:
                print(f"  Deleted {deleted_count} cross-class edges, "
                      f"kept {keep_mask.sum().item()} edges")
                print(f"  Delete precision: {delete_precision:.4f} "
                      f"({truly_cross if deleted_count > 0 else 0}/"
                      f"{deleted_count} truly cross-class)")
        else:
            # Skip deletion: keep all edges
            kept_src = edge_src
            kept_dst = edge_dst
            deleted_count = 0
            delete_precision = 0.0
            if self.debug:
                print(f"  [Disabled] Cross-class edge deletion skipped, "
                      f"keeping all {num_unique} edges")

        # ---- Step 2: Build modified graph (target nodes only) ----
        # Only use target nodes (nodes of seen classes), exclude external neighbors
        target_set_tensor = torch.tensor(sorted(target_nodes),
                                         device=self.device, dtype=torch.long)
        target_set = set(target_nodes)
        target_sorted = sorted(target_nodes)
        target_to_local = {g: l for l, g in enumerate(target_sorted)}
        num_target_nodes = len(target_sorted)

        # Build global->local mapping tensor for GPU re-indexing
        # max_node_id must cover all node IDs that could appear in edges
        max_node_id = max(max(all_nodes), max(target_sorted)) + 1
        g2l_tensor = torch.full((max_node_id,), -1, dtype=torch.long,
                                device=self.device)
        g2l_tensor[target_set_tensor] = torch.arange(
            num_target_nodes, device=self.device, dtype=torch.long)

        # Kept edges: filter to target-only, expand to both directions
        # kept_src, kept_dst are already on GPU (u < v form)
        kt_mask_u = g2l_tensor[kept_src] >= 0
        kt_mask_v = g2l_tensor[kept_dst] >= 0
        kt_mask = kt_mask_u & kt_mask_v
        kt_u = g2l_tensor[kept_src[kt_mask]]
        kt_v = g2l_tensor[kept_dst[kt_mask]]
        # Both directions
        modified_src_t = torch.cat([kt_u, kt_v])
        modified_dst_t = torch.cat([kt_v, kt_u])

        if modified_src_t.numel() > 0:
            modified_edge_index = torch.stack(
                [modified_src_t, modified_dst_t], dim=0
            ).cpu()
            # Deduplicate
            modified_edge_index = torch.unique(modified_edge_index, dim=1)
        else:
            modified_edge_index = torch.zeros(2, 0, dtype=torch.long)

        if self.debug:
            print(f"  Modified graph: {num_target_nodes} target nodes, "
                  f"{modified_edge_index.size(1)} edges")

        # ---- Step 3: Leiden Clustering ----
        cluster_labels = leiden_clustering(
            modified_edge_index, num_target_nodes,
            resolution=self.leiden_resolution
        )

        n_communities = len(np.unique(cluster_labels))
        if self.debug:
            print(f"  Leiden found {n_communities} communities "
                  f"(actual classes: {len(all_classes)})")

        # ---- Step 4: Compute Purity (target nodes only) ----
        target_true_labels = labels[target_sorted].cpu().numpy()

        cluster_purities, overall_purity = compute_cluster_purity(
            cluster_labels, target_true_labels
        )

        edge_modify_stats = {
            'total_edges': num_unique,
            'total_cross_class': total_cross_class,
            'deleted_count': deleted_count,
            'delete_precision': delete_precision,
        }

        return cluster_purities, overall_purity, edge_modify_stats

    # ==================================================================
    # Main Training Loop
    # ==================================================================

    def fit(self, trial):
        """Main training loop across all sessions."""
        purity_metric = PurityMetric()

        for session_id in range(self.task_loader.sessions):
            self.current_session = session_id
            expert_id = session_id % self.num_experts

            # Load checkpoint from previous session
            if session_id > 0:
                load_checkpoint(self.model, self.checkpoint_path,
                                self.dataset, 'SpectralExpertCL', self.seed)

            # Get task data
            (curr_classes, all_classes,
             subgraph, joint_subgraph,
             train_loader, valid_loader,
             test_loader_joint) = self.task_loader.get_task(session_id)

            print(f"\n{'='*60}")
            print(f"Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"Expert {expert_id}")
            print(f"{'='*60}")

            # ========== Phase 1: Neighbor Predictor ==========
            self.model.set_training_phase(expert_id, 'neighbor')
            optimizer = optim.Adam(
                self.model.experts[expert_id].neighbor_predictor.parameters(),
                lr=float(self.config['lr']),
                weight_decay=float(self.config['weight_decay']),
            )
            best_val_loss = float('inf')
            patience_counter = 0

            pbar = tqdm(range(self.neighbor_epochs),
                        desc=f"S{session_id} Neighbor Predictor")
            for epoch in pbar:
                loss = self.train_neighbor_epoch(session_id, subgraph, optimizer)

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    val_loss = self.evaluate_neighbor_predictor(
                        session_id, subgraph, valid_loader
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_checkpoint(
                            self.model, optimizer, epoch,
                            self.checkpoint_path, self.dataset,
                            'SpectralExpertCL', self.seed,
                        )
                    else:
                        patience_counter += 1
                        if patience_counter > self.config['patience']:
                            break
                    pbar.set_postfix({'loss': f'{loss:.4f}',
                                      'val_loss': f'{val_loss:.4f}'})
                else:
                    pbar.set_postfix({'loss': f'{loss:.4f}'})

            load_checkpoint(self.model, self.checkpoint_path,
                            self.dataset, 'SpectralExpertCL', self.seed)

            # ========== Phase 2: Cross-class Predictor ==========
            self.model.set_training_phase(expert_id, 'cross_class')
            optimizer = optim.Adam(
                self.model.experts[expert_id].cross_class_predictor.parameters(),
                lr=float(self.config['lr']),
                weight_decay=float(self.config['weight_decay']),
            )
            best_val_loss = float('inf')
            patience_counter = 0

            pbar = tqdm(range(self.cross_class_epochs),
                        desc=f"S{session_id} Cross-class Predictor")
            for epoch in pbar:
                loss = self.train_cross_class_epoch(
                    session_id, subgraph, optimizer
                )

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    val_loss = self.evaluate_cross_class_predictor(
                        session_id, subgraph, valid_loader
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_checkpoint(
                            self.model, optimizer, epoch,
                            self.checkpoint_path, self.dataset,
                            'SpectralExpertCL', self.seed,
                        )
                    else:
                        patience_counter += 1
                        if patience_counter > self.config['patience']:
                            break
                    pbar.set_postfix({'loss': f'{loss:.4f}',
                                      'val_loss': f'{val_loss:.4f}'})
                else:
                    pbar.set_postfix({'loss': f'{loss:.4f}'})

            load_checkpoint(self.model, self.checkpoint_path,
                            self.dataset, 'SpectralExpertCL', self.seed)

            print(f"\n  Training complete for session {session_id}")

            # ========== Evaluation: Spectral Clustering ==========
            print(f"\n  Running Leiden clustering evaluation...")

            cluster_purities, overall_purity, edge_stats = \
                self.evaluate_spectral(
                    joint_subgraph, test_loader_joint, all_classes
                )

            # Record results
            purity_metric.add_session_result(
                session_id, all_classes, cluster_purities, overall_purity
            )

            # Print session results
            print(f"\n  Session {session_id} Purity: {overall_purity:.4f}")
            print(f"  Edge stats:")
            print(f"    Total: {edge_stats['total_edges']} edges, "
                  f"ground-truth cross-class: {edge_stats['total_cross_class']}")
            print(f"    Deleted {edge_stats['deleted_count']} edges, "
                  f"delete precision: {edge_stats['delete_precision']:.4f} "
                  f"(truly cross-class ratio)")
            print(f"  Per-cluster purity ({len(cluster_purities)} communities):")
            # Separate large clusters (size > 5) and small clusters (size <= 5)
            small_clusters = {}  # size -> count
            for cluster_id, purity, size, dom_class in cluster_purities:
                if size > 5:
                    print(f"    Cluster {cluster_id}: purity={purity:.4f}, "
                          f"size={size}, dominant_class={dom_class}")
                elif size > 0:
                    small_clusters[size] = small_clusters.get(size, 0) + 1
            if small_clusters:
                parts = [f"{count}个{sz}节点社区"
                         for sz, count in sorted(small_clusters.items())]
                print(f"    小社区汇总: {', '.join(parts)}")

        # Print final results
        purity_metric.print_results()
        return purity_metric
