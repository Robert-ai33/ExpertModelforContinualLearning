"""
LiteExpertCL: Lightweight Expert-based Continual Learning for Graph Data.

Simplified variant of CommunityExpertCL:
1. Parameter-free GCN: H = (D^{-1/2} A D^{-1/2})^k X (shared, no learnable weights)
2. Per-expert MAE: mask embedding dims with learnable mask_token → decoder (Linear→ReLU→Linear)
3. Per-expert classifier: Linear → ReLU → Linear
4. Expert selection via MAE reconstruction error (same losses as CommunityExpertCL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.utils import degree
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from utils import compute_ap_af, print_cl_matrix


# ======================================================================
# Loss Functions
# ======================================================================

def scaled_cosine_error(pred, target, gamma=2):
    """(1 - cosine_similarity)^gamma, averaged over samples."""
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    return ((1 - cos_sim) ** gamma).mean()


def scaled_cosine_error_per_node(pred, target, gamma=2):
    """Per-node (1 - cosine_similarity)^gamma. Returns (N,) loss."""
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    return (1 - cos_sim) ** gamma


def pearson_correlation_loss(x, recon, eps=1e-8):
    """1 - Pearson correlation between x and recon (flattened)."""
    x_flat = x.reshape(-1)
    recon_flat = recon.reshape(-1)
    x_c = x_flat - x_flat.mean()
    recon_c = recon_flat - recon_flat.mean()
    pearson = (x_c * recon_c).sum() / (x_c.norm() * recon_c.norm() + eps)
    return 1 - pearson


def pearson_loss_per_node(x, recon, eps=1e-8):
    """Per-node 1 - Pearson correlation. x, recon: (N, D). Returns (N,) loss."""
    x_c = x - x.mean(dim=1, keepdim=True)
    recon_c = recon - recon.mean(dim=1, keepdim=True)
    dot = (x_c * recon_c).sum(dim=1)
    norm_x = x_c.norm(dim=1) + eps
    norm_recon = recon_c.norm(dim=1) + eps
    pearson = dot / (norm_x * norm_recon)
    return 1 - pearson


def norm_ratio_loss(x, recon, eps=1e-8):
    """(log(||recon||+eps) - log(||x||+eps))^2 per sample, then mean."""
    norm_x = x.norm(dim=1) + eps
    norm_recon = recon.norm(dim=1) + eps
    log_diff = torch.log(norm_recon) - torch.log(norm_x)
    return (log_diff ** 2).mean()


def norm_ratio_loss_per_node(x, recon, eps=1e-8):
    """Per-node (log(||recon||+eps) - log(||x||+eps))^2. Returns (N,) loss."""
    norm_x = x.norm(dim=1) + eps
    norm_recon = recon.norm(dim=1) + eps
    log_diff = torch.log(norm_recon) - torch.log(norm_x)
    return log_diff ** 2


# ======================================================================
# Model Components
# ======================================================================

def paramfree_gcn(x, edge_index, num_layers=1):
    """Parameter-free GCN: H = (D^{-1/2} A D^{-1/2})^k X."""
    num_nodes = x.size(0)
    row, col = edge_index[0], edge_index[1]
    deg = degree(col, num_nodes, dtype=x.dtype).clamp(min=1)
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        torch.stack([col, row]), edge_weight, (num_nodes, num_nodes)
    ).coalesce()

    h = x
    for _ in range(num_layers):
        h = torch.sparse.mm(adj, h)
    return h


class LiteExpert(nn.Module):
    """Single expert: MAE decoder + neighbor predictor + classifier."""

    def __init__(self, embed_dim, cls_hidden_dim, num_local_classes, np_hidden_dim):
        super().__init__()
        self.mae_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.xavier_uniform_(self.mask_token.unsqueeze(0))

        self.neighbor_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, np_hidden_dim),
            nn.ReLU(),
            nn.Linear(np_hidden_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.ReLU(),
            nn.Linear(cls_hidden_dim, num_local_classes),
        )


class LiteModel(nn.Module):
    """Multi-expert container (experts added lazily per session)."""

    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList()


# ======================================================================
# LiteExpertCL - Main Model
# ======================================================================

class LiteExpertCL:
    """Lightweight expert CL with parameter-free GCN + per-expert MAE/classifier."""

    EVAL_BATCH_SIZE = 8192

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device

        self.input_dim = task_loader.data.x.shape[1]
        self.num_classes = max(task_loader.all_classes) + 1
        self.max_experts = config.get('max_experts', 7)

        self.gcn_layers = config.get('gcn_layers', 1)
        self.cls_hidden_dim = config.get('cls_hidden_dim', 256)

        self.cls_epochs = config.get('cls_epochs', 200)
        self.cls_lr = float(config.get('cls_lr', 0.01))
        self.cls_wd = float(config.get('cls_weight_decay', 5e-4))

        self.mae_epochs = config.get('mae_epochs', 200)
        self.mae_lr = float(config.get('mae_lr', 1e-3))
        self.mae_wd = float(config.get('mae_weight_decay', 1e-4))
        self.mask_ratio = config.get('mask_ratio', 0.5)
        self.mae_gamma = config.get('mae_gamma', 2)
        self.pearson_weight = config.get('pearson_weight', 0.0)
        self.norm_ratio_weight = config.get('norm_ratio_weight', 0.0)
        # Loss formulation used for (a) training the MAE, (b) validating it,
        # (c) generating pseudo data during merging, and (d) routing at
        # inference. Options:
        #   'joint' -- default, scaled-cosine + pearson_weight*Pearson
        #              + norm_ratio_weight*norm-ratio (the original three-term
        #              loss).
        #   'mse'   -- plain F.mse_loss(recon, target). Pearson/norm-ratio
        #              weights are ignored in this mode, so it's a true
        #              ablation of the joint loss.
        self.mae_loss_type = str(config.get('mae_loss', 'joint')).lower()
        if self.mae_loss_type not in ('joint', 'mse'):
            raise ValueError(
                f"Unknown mae_loss '{self.mae_loss_type}'. "
                f"Expected 'joint' or 'mse'.")

        self.np_hidden_dim = config.get('np_hidden_dim', 256)
        self.np_epochs = config.get('np_epochs', 100)
        self.np_lr = float(config.get('np_lr', 1e-3))
        self.np_wd = float(config.get('np_weight_decay', 1e-4))
        self.np_neg_ratio = config.get('np_neg_ratio', 3)
        self.np_topk_ratio = config.get('np_topk_ratio', 0.05)
        self.np_enriched_row_chunk = config.get('np_enriched_row_chunk', 512)
        self.np_enriched_col_chunk = config.get('np_enriched_col_chunk', 512)
        self.neighbor_stats_weight = config.get('neighbor_stats_weight', 1.0)

        # Expert merging hyperparameters
        self.merge_pseudo_samples = config.get('merge_pseudo_samples', 256)
        self.merge_pseudo_steps = config.get('merge_pseudo_steps', 300)
        self.merge_pseudo_lr = float(config.get('merge_pseudo_lr', 0.1))
        self.merge_entropy_weight = config.get('merge_entropy_weight', 0.1)
        self.merge_diversity_weight = config.get('merge_diversity_weight', 0.1)
        self.merge_temperature = config.get('merge_temperature', 2.0)
        self.merge_distill_mae_epochs = config.get('merge_distill_mae_epochs', 200)
        self.merge_distill_cls_epochs = config.get('merge_distill_cls_epochs', 200)
        self.merge_distill_np_epochs = config.get('merge_distill_np_epochs', 200)
        self.merge_distill_np_samples = config.get('merge_distill_np_samples', 1000)
        self.merge_stats_weight = config.get('merge_stats_weight', 1.0)

        self.model = LiteModel().to(device)

        # Per-expert class mapping (populated lazily in _train_session)
        self.expert_g2l = []   # expert_g2l[eid] = {global_class: local_index}
        self.expert_l2g = []   # expert_l2g[eid] = tensor [local_index -> global_class]
        self.expert_usage_count = []  # usage count over joint TEST nodes (drives merge choice)
        self.expert_stats = []  # {mean, var, n} of training embeddings per expert
        self.expert_neighbor_stats = []  # {mean, var, n} of neighbor-enriched embeddings

        self.current_session = 0

    # ==================== Expert Creation ====================

    def _create_expert_with_fixed_init(self, num_local_classes):
        """Create expert with fixed-seed initialization so every expert starts identically."""
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None

        torch.manual_seed(42)
        if cuda_rng is not None:
            torch.cuda.manual_seed(42)

        expert = LiteExpert(
            self.input_dim, self.cls_hidden_dim, num_local_classes,
            self.np_hidden_dim
        ).to(self.device)

        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, self.device)

        return expert

    # ==================== Embedding ====================

    def _compute_embeddings(self, subgraph):
        """Compute parameter-free GCN embeddings."""
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        with torch.no_grad():
            h = paramfree_gcn(x, edge_index, num_layers=self.gcn_layers)
        return h

    # ==================== Training ====================

    def fit(self, trial):
        """Main training + evaluation loop across all sessions."""
        num_sessions = self.task_loader.sessions

        acc_matrix = []
        joint_acc_history = []
        joint_macro_history = []

        for session_id in range(num_sessions):
            self.current_session = session_id

            (curr_classes, all_classes,
             subgraph, joint_subgraph,
             train_loader, valid_loader,
             test_loader_joint) = self.task_loader.get_task(session_id)

            train_idx = self.task_loader.train_idx_per_task[session_id]
            valid_idx = self.task_loader.valid_idx_per_task[session_id]

            print(f"\n{'='*60}")
            print(f"Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}")
            print(f"{'='*60}")

            # Merge if at capacity before adding new expert
            if len(self.model.experts) >= self.max_experts:
                print(f"\n--- Merging Experts (capacity={self.max_experts}) ---")
                self._merge_least_used_experts()

            print(f"\n--- Training (Session {session_id}) ---")
            self._train_session(
                session_id, subgraph, train_idx, valid_idx,
                curr_classes)

            # Per-Task Tests on the cumulative subgraph (CGLB)
            print(f"\n--- Per-Task Tests (Session {session_id}) ---")
            eval_subgraph = self.task_loader.subgraph_per_task[session_id]
            acc_row = []
            for tid in range(session_id + 1):
                test_idx = self.task_loader.test_idx_per_task[tid]
                task_classes = self.task_loader.class_splits[tid]

                if not test_idx:
                    acc_row.append(0.0)
                    print(f"  Task {tid} (classes {task_classes}): no test nodes")
                    continue

                res = self._evaluate_subgraph(eval_subgraph, test_idx)
                acc_row.append(res['acc'])
                print(f"  Task {tid} (classes {task_classes}): "
                      f"Acc={res['acc']:.4f} "
                      f"({res['correct']}/{res['total']})")

            acc_matrix.append(acc_row)

            # Joint Test
            print(f"\n--- Joint Test (Session {session_id}) ---")
            test_idx_joint = self.task_loader.test_idx_joint[session_id]
            joint_res = self._evaluate_subgraph(joint_subgraph, test_idx_joint)

            joint_acc_history.append(joint_res['acc'])
            joint_macro_history.append(joint_res['macro_acc'])
            print(f"  Acc={joint_res['acc']:.4f} "
                  f"Macro={joint_res['macro_acc']:.4f} "
                  f"({joint_res['correct']}/{joint_res['total']})")

            # Record usage counts for merging decisions
            self.expert_usage_count = [0] * len(self.model.experts)
            assigns = joint_res['expert_assignments']
            if assigns is not None and assigns.numel() > 0:
                unique_e, counts_e = torch.unique(assigns, return_counts=True)
                for e, c in zip(unique_e, counts_e):
                    if e.item() < len(self.expert_usage_count):
                        self.expert_usage_count[e.item()] = c.item()
                dist = ", ".join([f"E{e.item()}:{c.item()}"
                                  for e, c in zip(unique_e, counts_e)])
                print(f"  Expert distribution: {dist}")

        # Final Summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")

        print_cl_matrix("CL Accuracy Matrix", acc_matrix, num_sessions)
        ap_history, af, final_ap = compute_ap_af(acc_matrix)

        print(f"\nJoint Accuracy (micro): " + ", ".join(
            [f"S{i}={joint_acc_history[i]:.4f}" for i in range(num_sessions)]))
        print(f"Joint Accuracy (macro): " + ", ".join(
            [f"S{i}={joint_macro_history[i]:.4f}" for i in range(num_sessions)]))

        return {
            'acc_matrix': acc_matrix,
            'joint_acc': joint_acc_history,
            'joint_macro_acc': joint_macro_history,
            'ap_history': ap_history,
            'af': af,
            'final_ap': final_ap,
        }

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes):
        """Train one session's expert: Phase 1 MAE, Phase 2 NP, Phase 3 Classifier."""
        # Build local class mapping: sorted for deterministic order
        sorted_classes = sorted(curr_classes)
        g2l = {c: i for i, c in enumerate(sorted_classes)}
        l2g = torch.tensor(sorted_classes, dtype=torch.long, device=self.device)

        new_expert = self._create_expert_with_fixed_init(len(sorted_classes))

        self.model.experts.append(new_expert)
        self.expert_g2l.append(g2l)
        self.expert_l2g.append(l2g)

        h = self._compute_embeddings(subgraph)
        labels = subgraph['y'].to(self.device)

        train_mask = torch.zeros(h.size(0), dtype=torch.bool, device=self.device)
        for idx in train_idx:
            train_mask[idx] = True

        curr_class_set = set(curr_classes)
        curr_class_mask = torch.zeros(h.size(0), dtype=torch.bool, device=self.device)
        for idx in range(h.size(0)):
            if labels[idx].item() in curr_class_set:
                curr_class_mask[idx] = True

        loss_mask = train_mask & curr_class_mask

        # Pre-compute local labels for training nodes (only curr_classes nodes)
        train_indices = torch.where(loss_mask)[0]

        local_train_labels = torch.tensor(
            [g2l[labels[idx].item()] for idx in train_indices.tolist()],
            dtype=torch.long, device=self.device)

        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)

        expert = self.model.experts[-1]

        # ========== Phase 1: MAE (decoder + mask_token) ==========
        self._freeze_all()
        for param in expert.mae_decoder.parameters():
            param.requires_grad = True
        expert.mask_token.requires_grad = True

        mae_params = list(expert.mae_decoder.parameters()) + [expert.mask_token]
        optimizer = optim.Adam(mae_params, lr=self.mae_lr, weight_decay=self.mae_wd)
        best_val = float('inf')
        best_mae_state = None
        patience_cnt = 0

        curr_train_indices = torch.where(loss_mask)[0]

        pbar = tqdm(range(self.mae_epochs), desc=f"S{session_id} MAE")
        for epoch in pbar:
            self.model.train()
            expert.mae_decoder.train()

            mask_matrix = self._sample_shared_mask(curr_train_indices.size(0))
            h_target = h[curr_train_indices]
            masked_h = h_target.clone()
            mask_vals = expert.mask_token.unsqueeze(0).expand_as(masked_h)
            masked_h[mask_matrix] = mask_vals[mask_matrix]

            recon = expert.mae_decoder(masked_h)

            loss = self._mae_loss(recon, h_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_mae(
                    expert, h, labels, valid_idx, curr_classes)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_cnt = 0
                    best_mae_state = {
                        'decoder': {k: v.cpu().clone() for k, v
                                    in expert.mae_decoder.state_dict().items()},
                        'mask_token': expert.mask_token.data.cpu().clone(),
                    }
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(
                    loss=f'{loss.item():.4f}',
                    type=self.mae_loss_type,
                    val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(
                    loss=f'{loss.item():.4f}',
                    type=self.mae_loss_type)

        if best_mae_state is not None:
            expert.mae_decoder.load_state_dict(
                {k: v.to(self.device) for k, v in best_mae_state['decoder'].items()})
            expert.mask_token.data = best_mae_state['mask_token'].to(self.device)

        # ========== Phase 2: Neighbor Predictor ==========
        self._train_neighbor_predictor(
            session_id, expert, h, subgraph, train_idx, train_mask,
            valid_idx, valid_ep, patience)

        # ========== Record Statistics ==========
        with torch.no_grad():
            h_train = h[train_indices]
            self.expert_stats.append({
                'mean': h_train.mean(dim=0),
                'var': h_train.var(dim=0, correction=0),
                'n': h_train.size(0),
            })
            neighbor_stats = self._compute_enriched_stats(expert, h_train)
            self.expert_neighbor_stats.append(neighbor_stats)

        # ========== Phase 3: Classifier ==========
        self._freeze_all()
        for param in expert.classifier.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(expert.classifier.parameters(),
                               lr=self.cls_lr, weight_decay=self.cls_wd)
        best_val = float('inf')
        best_cls_state = None
        patience_cnt = 0

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} CLS")
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()
            logits = expert.classifier(h[train_indices])
            loss = F.cross_entropy(logits, local_train_labels)
            loss.backward()
            optimizer.step()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(
                    expert, h, labels, valid_idx, curr_classes, g2l)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_cnt = 0
                    best_cls_state = {k: v.cpu().clone()
                                      for k, v in expert.classifier.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(loss=f'{loss.item():.4f}', val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        if best_cls_state is not None:
            expert.classifier.load_state_dict(
                {k: v.to(self.device) for k, v in best_cls_state.items()})

    # ==================== Neighbor Predictor Helpers ====================

    def _train_neighbor_predictor(self, session_id, expert, h, subgraph,
                                  train_idx, train_mask, valid_idx,
                                  valid_ep, patience):
        """Phase 2: Train neighbor predictor on edges with BOTH endpoints in training set."""
        self._freeze_all()
        for param in expert.neighbor_predictor.parameters():
            param.requires_grad = True

        edge_index = subgraph['edge_index_no_selfloop'].to(self.device)
        src, dst = edge_index[0], edge_index[1]
        has_train = train_mask[src] & train_mask[dst]
        filt_src = src[has_train]
        filt_dst = dst[has_train]
        keep = filt_src < filt_dst
        pos_src = filt_src[keep]
        pos_dst = filt_dst[keep]
        num_pos = pos_src.size(0)

        if num_pos == 0:
            print(f"  NP: No positive edges found, skipping")
            return

        lo = torch.min(src, dst)
        hi = torch.max(src, dst)
        edge_pairs = torch.stack([lo, hi], dim=1).cpu()
        edge_pairs = torch.unique(edge_pairs, dim=0)
        all_edge_set = set(map(tuple, edge_pairs.tolist()))

        train_idx_list = list(train_idx)

        optimizer = optim.Adam(expert.neighbor_predictor.parameters(),
                               lr=self.np_lr, weight_decay=self.np_wd)
        best_val = float('inf')
        best_np_state = None
        patience_cnt = 0

        pbar = tqdm(range(self.np_epochs), desc=f"S{session_id} NP")
        for epoch in pbar:
            expert.neighbor_predictor.train()

            num_neg = self.np_neg_ratio * num_pos
            neg_src, neg_dst = self._sample_negative_edges(
                all_edge_set, train_idx_list, num_neg)

            all_src = torch.cat([pos_src, neg_src])
            all_dst = torch.cat([pos_dst, neg_dst])
            all_labels = torch.cat([
                torch.ones(num_pos, device=self.device),
                torch.zeros(neg_src.size(0), device=self.device)
            ])

            perm = torch.randperm(all_src.size(0), device=self.device)
            all_src = all_src[perm]
            all_dst = all_dst[perm]
            all_labels = all_labels[perm]

            edge_emb = self._compute_edge_embedding(h, all_src, all_dst)
            logits = expert.neighbor_predictor(edge_emb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, all_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_np(expert, h, subgraph, valid_idx)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_cnt = 0
                    best_np_state = {k: v.cpu().clone()
                                     for k, v in expert.neighbor_predictor.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(loss=f'{loss.item():.4f}', val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        if best_np_state is not None:
            expert.neighbor_predictor.load_state_dict(
                {k: v.to(self.device) for k, v in best_np_state.items()})

        print(f"  NP: Trained on {num_pos} positive edges")

    def _sample_negative_edges(self, all_edge_set, train_idx_list, num_neg):
        """Sample negative edges: BOTH endpoints in train_idx, no real edge, (a,b)=(b,a)."""
        device = self.device
        neg_set = set()
        neg_src_list = []
        neg_dst_list = []

        n_train = len(train_idx_list)

        for _ in range(50):
            if len(neg_src_list) >= num_neg:
                break
            remain = (num_neg - len(neg_src_list)) * 3
            a_idx = torch.randint(n_train, (remain,))
            b_idx = torch.randint(n_train, (remain,))

            for i in range(remain):
                ai = train_idx_list[a_idx[i].item()]
                bi = train_idx_list[b_idx[i].item()]
                if ai == bi:
                    continue
                edge = (min(ai, bi), max(ai, bi))
                if edge in all_edge_set or edge in neg_set:
                    continue
                neg_set.add(edge)
                neg_src_list.append(edge[0])
                neg_dst_list.append(edge[1])
                if len(neg_src_list) >= num_neg:
                    break

        return (torch.tensor(neg_src_list, dtype=torch.long, device=device),
                torch.tensor(neg_dst_list, dtype=torch.long, device=device))

    def _compute_edge_embedding(self, h, src, dst):
        """Edge embedding: concat(h[src]+h[dst], h[src]*h[dst])."""
        h_src = h[src]
        h_dst = h[dst]
        return torch.cat([h_src + h_dst, h_src * h_dst], dim=-1)

    def _compute_enriched_embedding(self, expert, h_batch):
        """Compute neighbor-enriched embeddings using streaming top-k (memory-friendly).

        Instead of materializing an (n, n) probability matrix, computes
        pairwise probabilities in (row_chunk, col_chunk) tiles and maintains
        a running top-k per row, keeping peak memory at O(row_chunk * col_chunk).
        """
        n = h_batch.size(0)
        k = min(max(1, int(self.np_topk_ratio * n)), n - 1)
        if k <= 0:
            return h_batch

        row_chunk = self.np_enriched_row_chunk
        col_chunk = self.np_enriched_col_chunk
        device = h_batch.device

        topk_probs = torch.full((n, k), -1.0, device=device)
        topk_indices = torch.zeros((n, k), dtype=torch.long, device=device)

        for r_start in range(0, n, row_chunk):
            r_end = min(r_start + row_chunk, n)
            h_i = h_batch[r_start:r_end]  # (rc, D)
            rc = r_end - r_start

            cur_probs = topk_probs[r_start:r_end]    # (rc, k)
            cur_idx = topk_indices[r_start:r_end]     # (rc, k)

            for c_start in range(0, n, col_chunk):
                c_end = min(c_start + col_chunk, n)
                h_j = h_batch[c_start:c_end]  # (cc, D)
                cc = c_end - c_start

                sum_emb = h_i.unsqueeze(1) + h_j.unsqueeze(0)  # (rc, cc, D)
                had_emb = h_i.unsqueeze(1) * h_j.unsqueeze(0)  # (rc, cc, D)
                edge_emb = torch.cat([sum_emb, had_emb], dim=2)

                logits = expert.neighbor_predictor(
                    edge_emb.reshape(rc * cc, -1)).squeeze(-1)
                tile_probs = torch.sigmoid(logits).reshape(rc, cc)

                # zero out diagonal entries (self-loops) — non-inplace for autograd safety
                if r_start < c_end and c_start < r_end:
                    ov_r0 = max(r_start, c_start) - r_start
                    ov_c0 = max(r_start, c_start) - c_start
                    ov_len = min(r_end, c_end) - max(r_start, c_start)
                    if ov_len > 0:
                        diag_r = torch.arange(ov_r0, ov_r0 + ov_len, device=device)
                        diag_c = torch.arange(ov_c0, ov_c0 + ov_len, device=device)
                        diag_mask = torch.zeros(rc, cc, dtype=torch.bool, device=device)
                        diag_mask[diag_r, diag_c] = True
                        tile_probs = tile_probs.masked_fill(diag_mask, 0.0)

                col_global = torch.arange(c_start, c_end, device=device).unsqueeze(0).expand(rc, -1)

                merged_probs = torch.cat([cur_probs, tile_probs], dim=1)   # (rc, k+cc)
                merged_idx = torch.cat([cur_idx, col_global], dim=1)       # (rc, k+cc)

                sel_k = min(k, merged_probs.size(1))
                new_probs, pick = merged_probs.topk(sel_k, dim=1)
                cur_probs = new_probs
                cur_idx = merged_idx.gather(1, pick)

            topk_probs[r_start:r_end] = cur_probs
            topk_indices[r_start:r_end] = cur_idx

        neighbor_agg = torch.zeros_like(h_batch)                    # (n, D)
        for r_start in range(0, n, row_chunk):
            r_end = min(r_start + row_chunk, n)
            chunk_emb = h_batch[topk_indices[r_start:r_end]]     # (rc, k, D)
            chunk_w = topk_probs[r_start:r_end].unsqueeze(-1)    # (rc, k, 1)
            neighbor_agg[r_start:r_end] = (chunk_w * chunk_emb).sum(dim=1) / k
        return neighbor_agg + h_batch

    @torch.no_grad()
    def _compute_enriched_stats(self, expert, h_train):
        """Compute mean/var of neighbor-enriched embeddings for training nodes."""
        expert.neighbor_predictor.eval()
        enriched = self._compute_enriched_embedding(expert, h_train)
        return {
            'mean': enriched.mean(dim=0),
            'var': enriched.var(dim=0, correction=0),
            'n': h_train.size(0),
        }

    @torch.no_grad()
    def _validate_np(self, expert, h, subgraph, valid_idx):
        """Validate NP on edges with BOTH endpoints in validation set."""
        expert.neighbor_predictor.eval()
        edge_index = subgraph['edge_index_no_selfloop'].to(self.device)
        src, dst = edge_index[0], edge_index[1]

        valid_mask = torch.zeros(h.size(0), dtype=torch.bool, device=self.device)
        for idx in valid_idx:
            valid_mask[idx] = True

        has_valid = valid_mask[src] & valid_mask[dst]
        filt_src = src[has_valid]
        filt_dst = dst[has_valid]
        keep = filt_src < filt_dst
        pos_src = filt_src[keep]
        pos_dst = filt_dst[keep]

        if pos_src.size(0) == 0:
            return float('inf')

        edge_emb = self._compute_edge_embedding(h, pos_src, pos_dst)
        logits = expert.neighbor_predictor(edge_emb).squeeze(-1)
        labels = torch.ones_like(logits)
        val_loss = F.binary_cross_entropy_with_logits(logits, labels)

        return val_loss.item()

    # ==================== Expert Merging ====================

    def _merge_least_used_experts(self):
        """Merge the two least-used experts (by last Joint Test usage) into one."""
        num_experts = len(self.model.experts)
        if num_experts < 2:
            return

        sorted_indices = sorted(range(num_experts),
                                key=lambda i: self.expert_usage_count[i])
        idx_a, idx_b = sorted_indices[0], sorted_indices[1]

        expert_a = self.model.experts[idx_a]
        expert_b = self.model.experts[idx_b]
        l2g_a = self.expert_l2g[idx_a]
        l2g_b = self.expert_l2g[idx_b]
        classes_a = l2g_a.tolist()
        classes_b = l2g_b.tolist()

        merged_classes = sorted(set(classes_a + classes_b))
        merged_g2l = {c: i for i, c in enumerate(merged_classes)}
        merged_l2g = torch.tensor(merged_classes, dtype=torch.long,
                                  device=self.device)
        num_merged = len(merged_classes)

        a_to_merged = [merged_g2l[c] for c in classes_a]
        b_to_merged = [merged_g2l[c] for c in classes_b]

        print(f"  Merging E{idx_a} (classes {classes_a}, "
              f"usage={self.expert_usage_count[idx_a]}) and "
              f"E{idx_b} (classes {classes_b}, "
              f"usage={self.expert_usage_count[idx_b]})")

        # Step 1: Generate pseudo data for each expert (per-class count)
        stats_a = self.expert_stats[idx_a]
        stats_b = self.expert_stats[idx_b]
        neighbor_stats_a = self.expert_neighbor_stats[idx_a]
        neighbor_stats_b = self.expert_neighbor_stats[idx_b]
        pseudo_a = self._generate_pseudo_data(
            expert_a, len(classes_a), stats_a, neighbor_stats_a, tag="A")
        pseudo_b = self._generate_pseudo_data(
            expert_b, len(classes_b), stats_b, neighbor_stats_b, tag="B")

        # Step 2: Build soft labels with temperature scaling
        T = self.merge_temperature
        with torch.no_grad():
            probs_a = F.softmax(expert_a.classifier(pseudo_a) / T, dim=1)
            probs_b = F.softmax(expert_b.classifier(pseudo_b) / T, dim=1)

        soft_labels_a = torch.zeros(pseudo_a.size(0), num_merged,
                                    device=self.device)
        for li, mi in enumerate(a_to_merged):
            soft_labels_a[:, mi] = probs_a[:, li]

        soft_labels_b = torch.zeros(pseudo_b.size(0), num_merged,
                                    device=self.device)
        for li, mi in enumerate(b_to_merged):
            soft_labels_b[:, mi] = probs_b[:, li]

        all_pseudo = torch.cat([pseudo_a, pseudo_b], dim=0)
        all_soft_labels = torch.cat([soft_labels_a, soft_labels_b], dim=0)

        perm = torch.randperm(all_pseudo.size(0), device=self.device)
        all_pseudo = all_pseudo[perm]
        all_soft_labels = all_soft_labels[perm]

        # Step 3: Create merged expert with averaged initial weights
        merged_expert = LiteExpert(
            self.input_dim, self.cls_hidden_dim, num_merged,
            self.np_hidden_dim
        ).to(self.device)

        self._init_merged_weights(merged_expert, expert_a, expert_b,
                                  a_to_merged, b_to_merged, num_merged,
                                  stats_a, stats_b)

        self._distill_mae(merged_expert, all_pseudo)
        self._distill_classifier(merged_expert, all_pseudo, all_soft_labels)
        self._distill_neighbor_predictor(
            merged_expert, expert_a, expert_b,
            pseudo_a, pseudo_b, len(classes_a), len(classes_b))

        # Step 4: Compute merged stats for the new expert
        n_a, n_b = stats_a['n'], stats_b['n']
        n_ab = n_a + n_b
        mean_ab = (n_a * stats_a['mean'] + n_b * stats_b['mean']) / n_ab
        var_ab = ((n_a * (stats_a['var'] + stats_a['mean'] ** 2)
                   + n_b * (stats_b['var'] + stats_b['mean'] ** 2)) / n_ab
                  - mean_ab ** 2)

        neighbor_mean_ab = (n_a * neighbor_stats_a['mean']
                            + n_b * neighbor_stats_b['mean']) / n_ab
        neighbor_var_ab = (
            (n_a * (neighbor_stats_a['var'] + neighbor_stats_a['mean'] ** 2)
             + n_b * (neighbor_stats_b['var'] + neighbor_stats_b['mean'] ** 2))
            / n_ab - neighbor_mean_ab ** 2)

        # Step 5: Remove old experts (reverse order to preserve indices)
        for idx in sorted([idx_a, idx_b], reverse=True):
            del self.model.experts[idx]
            del self.expert_g2l[idx]
            del self.expert_l2g[idx]
            del self.expert_usage_count[idx]
            del self.expert_stats[idx]
            del self.expert_neighbor_stats[idx]

        # Step 6: Add merged expert
        self.model.experts.append(merged_expert)
        self.expert_g2l.append(merged_g2l)
        self.expert_l2g.append(merged_l2g)
        self.expert_usage_count.append(0)
        self.expert_stats.append({'mean': mean_ab, 'var': var_ab, 'n': n_ab})
        self.expert_neighbor_stats.append({
            'mean': neighbor_mean_ab, 'var': neighbor_var_ab, 'n': n_ab})

        print(f"  -> Merged expert manages classes {merged_classes} "
              f"(total experts: {len(self.model.experts)})")

    @torch.no_grad()
    def _init_merged_weights(self, merged, expert_a, expert_b,
                             a_to_merged, b_to_merged, num_merged,
                             stats_a, stats_b):
        """Initialize merged expert via weight-space permutation alignment + weighted averaging.

        Uses the Hungarian algorithm (LAP) to find the optimal neuron permutation
        that minimizes ||W_A - W_B P||_F before averaging weights. Parameters are
        then merged via weighted average proportional to each expert's training
        data count (stored in stats['n']).
        """
        n_a = stats_a['n']
        n_b = stats_b['n']
        alpha = n_a / (n_a + n_b)  # weight for expert_a

        # ── MAE Decoder alignment ──
        wa1 = expert_a.mae_decoder[0].weight.data   # [hidden, input]
        wb1 = expert_b.mae_decoder[0].weight.data
        ba1 = expert_a.mae_decoder[0].bias.data      # [hidden]
        bb1 = expert_b.mae_decoder[0].bias.data
        wa2 = expert_a.mae_decoder[2].weight.data    # [output, hidden]
        wb2 = expert_b.mae_decoder[2].weight.data

        # Composite cost = Σ ||Δparam_i||_2^2 over all parameters tied to each hidden neuron
        cost_mae = torch.cdist(wa1, wb1, p=2) ** 2
        cost_mae += torch.cdist(ba1.unsqueeze(1), bb1.unsqueeze(1), p=2) ** 2
        cost_mae += torch.cdist(wa2.t(), wb2.t(), p=2) ** 2

        _, col_mae = linear_sum_assignment(cost_mae.cpu().numpy())
        pm = torch.tensor(col_mae, device=self.device, dtype=torch.long)

        merged.mae_decoder[0].weight.data.copy_(alpha * wa1 + (1 - alpha) * wb1[pm])
        merged.mae_decoder[0].bias.data.copy_(alpha * ba1 + (1 - alpha) * bb1[pm])

        # Layer 2: [output, hidden] — columns correspond to hidden neurons
        merged.mae_decoder[2].weight.data.copy_(
            alpha * wa2 + (1 - alpha) * wb2[:, pm])
        merged.mae_decoder[2].bias.data.copy_(
            alpha * expert_a.mae_decoder[2].bias.data
            + (1 - alpha) * expert_b.mae_decoder[2].bias.data)

        # mask_token: weighted average proportional to training data count
        merged.mask_token.data.copy_(
            alpha * expert_a.mask_token.data
            + (1 - alpha) * expert_b.mask_token.data)

        # ── Neighbor Predictor alignment ──
        np_wa1 = expert_a.neighbor_predictor[0].weight.data  # [np_hidden, 2D]
        np_wb1 = expert_b.neighbor_predictor[0].weight.data
        np_ba1 = expert_a.neighbor_predictor[0].bias.data    # [np_hidden]
        np_bb1 = expert_b.neighbor_predictor[0].bias.data
        np_wa2 = expert_a.neighbor_predictor[2].weight.data  # [1, np_hidden]
        np_wb2 = expert_b.neighbor_predictor[2].weight.data

        cost_np = torch.cdist(np_wa1, np_wb1, p=2) ** 2
        cost_np += torch.cdist(np_ba1.unsqueeze(1), np_bb1.unsqueeze(1), p=2) ** 2
        cost_np += torch.cdist(np_wa2.t(), np_wb2.t(), p=2) ** 2

        _, col_np = linear_sum_assignment(cost_np.cpu().numpy())
        pm_np = torch.tensor(col_np, device=self.device, dtype=torch.long)

        merged.neighbor_predictor[0].weight.data.copy_(
            alpha * np_wa1 + (1 - alpha) * np_wb1[pm_np])
        merged.neighbor_predictor[0].bias.data.copy_(
            alpha * np_ba1 + (1 - alpha) * np_bb1[pm_np])
        merged.neighbor_predictor[2].weight.data.copy_(
            alpha * np_wa2 + (1 - alpha) * np_wb2[:, pm_np])
        merged.neighbor_predictor[2].bias.data.copy_(
            alpha * expert_a.neighbor_predictor[2].bias.data
            + (1 - alpha) * expert_b.neighbor_predictor[2].bias.data)

    def _generate_pseudo_data(self, expert, num_classes, stats,
                              neighbor_stats, tag=""):
        """Optimize random data to minimize MAE recon + entropy + balance + stats match."""
        expert.eval()
        for p in expert.parameters():
            p.requires_grad = False

        target_mean = stats['mean'].detach()
        target_var = stats['var'].detach()
        target_neighbor_mean = (neighbor_stats['mean'].detach()
                                if neighbor_stats is not None else None)
        target_neighbor_var = (neighbor_stats['var'].detach()
                               if neighbor_stats is not None else None)

        n = num_classes * self.merge_pseudo_samples
        h_fake = torch.randn(n, self.input_dim, device=self.device,
                             requires_grad=True)
        optimizer = optim.Adam([h_fake], lr=self.merge_pseudo_lr)

        pbar = tqdm(range(self.merge_pseudo_steps), desc=f"  GenData-{tag}")
        for step in pbar:
            optimizer.zero_grad()

            # Random per-node mask each step
            mask_matrix = self._sample_pernode_mask(n)
            mask_vals = expert.mask_token.detach().unsqueeze(0).expand(n, -1)
            masked_h = torch.where(mask_matrix, mask_vals, h_fake)

            recon = expert.mae_decoder(masked_h)
            loss_mae = self._mae_loss(recon, h_fake)

            # Per-sample confidence: minimize entropy so each sample has a clear class
            logits = expert.classifier(h_fake)
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            loss_entropy = -(probs * log_probs).sum(dim=1).mean()

            # Class balance: maximize entropy of avg prediction → uniform class coverage
            avg_probs = probs.mean(dim=0)
            loss_diversity = (avg_probs * torch.log(avg_probs + 1e-8)).sum()

            # Stats match: align fake data distribution with real training data
            fake_mean = h_fake.mean(dim=0)
            fake_var = h_fake.var(dim=0, correction=0)
            loss_stats = (F.mse_loss(fake_mean, target_mean)
                          + F.mse_loss(fake_var, target_var))

            loss_neighbor = torch.tensor(0.0, device=self.device)
            if target_neighbor_mean is not None:
                enriched_fake = self._compute_enriched_embedding(
                    expert, h_fake)
                fake_neighbor_mean = enriched_fake.mean(dim=0)
                fake_neighbor_var = enriched_fake.var(dim=0, correction=0)
                loss_neighbor = (
                    F.mse_loss(fake_neighbor_mean, target_neighbor_mean)
                    + F.mse_loss(fake_neighbor_var, target_neighbor_var))

            loss = (loss_mae
                    + self.merge_entropy_weight * loss_entropy
                    + self.merge_diversity_weight * loss_diversity
                    + self.merge_stats_weight * loss_stats
                    + self.neighbor_stats_weight * loss_neighbor)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                pbar.set_postfix(mae=f'{loss_mae.item():.4f}',
                                 ent=f'{loss_entropy.item():.4f}',
                                 bal=f'{loss_diversity.item():.4f}',
                                 stat=f'{loss_stats.item():.4f}',
                                 nei=f'{loss_neighbor.item():.4f}')

        return h_fake.detach()

    def _distill_mae(self, merged_expert, pseudo_data):
        """Train merged expert's MAE on combined pseudo data."""
        for p in merged_expert.mae_decoder.parameters():
            p.requires_grad = True
        merged_expert.mask_token.requires_grad = True

        mae_params = (list(merged_expert.mae_decoder.parameters())
                      + [merged_expert.mask_token])
        optimizer = optim.Adam(mae_params, lr=self.mae_lr,
                               weight_decay=self.mae_wd)

        n = pseudo_data.size(0)
        pbar = tqdm(range(self.merge_distill_mae_epochs), desc="  DistillMAE")
        for epoch in pbar:
            merged_expert.mae_decoder.train()

            # Random shared mask each step
            mask_matrix = self._sample_shared_mask(n)
            masked_h = pseudo_data.clone()
            mask_vals = merged_expert.mask_token.unsqueeze(0).expand_as(masked_h)
            masked_h[mask_matrix] = mask_vals[mask_matrix]

            recon = merged_expert.mae_decoder(masked_h)

            loss = self._mae_loss(recon, pseudo_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

    def _distill_classifier(self, merged_expert, pseudo_data, soft_labels):
        """Train merged expert's classifier via knowledge distillation with soft labels."""
        for p in merged_expert.classifier.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(merged_expert.classifier.parameters(),
                               lr=self.cls_lr, weight_decay=self.cls_wd)
        T = self.merge_temperature

        pbar = tqdm(range(self.merge_distill_cls_epochs), desc="  DistillCLS")
        for epoch in pbar:
            merged_expert.classifier.train()

            logits = merged_expert.classifier(pseudo_data) / T
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(soft_labels * log_probs).sum(dim=1).mean() * (T ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

    def _distill_neighbor_predictor(self, merged_expert, expert_a, expert_b,
                                    pseudo_a, pseudo_b,
                                    num_classes_a, num_classes_b):
        """Distill merged NP using soft targets from original experts' NPs.

        Per epoch, sample n_a pairs from pseudo_a and n_b pairs from pseudo_b,
        where n_a/n_b ∝ x²/y² (x, y = number of classes each expert manages).
        """
        for p in merged_expert.neighbor_predictor.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(merged_expert.neighbor_predictor.parameters(),
                               lr=self.np_lr, weight_decay=self.np_wd)

        x2 = num_classes_a ** 2
        y2 = num_classes_b ** 2
        total_sq = x2 + y2
        n = self.merge_distill_np_samples
        n_a = max(1, round(n * x2 / total_sq))
        n_b = max(1, round(n * y2 / total_sq))

        m_a = pseudo_a.size(0)
        m_b = pseudo_b.size(0)

        pbar = tqdm(range(self.merge_distill_np_epochs), desc="  DistillNP")
        for epoch in pbar:
            merged_expert.neighbor_predictor.train()

            # Sample distinct pairs from A
            idx_a1 = torch.randint(m_a, (n_a,), device=self.device)
            idx_a2 = torch.randint(m_a - 1, (n_a,), device=self.device)
            idx_a2 = idx_a2 + (idx_a2 >= idx_a1).long()

            # Sample distinct pairs from B
            idx_b1 = torch.randint(m_b, (n_b,), device=self.device)
            idx_b2 = torch.randint(m_b - 1, (n_b,), device=self.device)
            idx_b2 = idx_b2 + (idx_b2 >= idx_b1).long()

            # Edge embeddings from fixed pseudo data
            edge_emb_a = self._compute_edge_embedding(pseudo_a, idx_a1, idx_a2)
            edge_emb_b = self._compute_edge_embedding(pseudo_b, idx_b1, idx_b2)

            # Soft targets from original experts (no grad)
            with torch.no_grad():
                target_probs_a = torch.sigmoid(
                    expert_a.neighbor_predictor(edge_emb_a).squeeze(-1))
                target_probs_b = torch.sigmoid(
                    expert_b.neighbor_predictor(edge_emb_b).squeeze(-1))

            # Merged NP predictions
            merged_probs_a = torch.sigmoid(
                merged_expert.neighbor_predictor(edge_emb_a).squeeze(-1))
            merged_probs_b = torch.sigmoid(
                merged_expert.neighbor_predictor(edge_emb_b).squeeze(-1))

            all_merged = torch.cat([merged_probs_a, merged_probs_b])
            all_target = torch.cat([target_probs_a, target_probs_b])
            loss = F.binary_cross_entropy(all_merged, all_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

    # ==================== Validation ====================

    # ==================== MAE loss (joint / mse switch) ====================

    def _mae_loss(self, recon, target):
        """Scalar MAE loss used by training / validation / pseudo-data gen.

        - ``mae_loss='joint'`` (default) uses the original three-term
          combination: scaled-cosine-error + pearson_weight * Pearson
          correlation loss + norm_ratio_weight * norm-ratio loss.
        - ``mae_loss='mse'`` uses ``F.mse_loss(recon, target)``; pearson and
          norm-ratio weights are ignored, so it is a clean ablation.
        """
        if self.mae_loss_type == 'mse':
            return F.mse_loss(recon, target)

        loss = scaled_cosine_error(recon, target, gamma=self.mae_gamma)
        if self.pearson_weight > 0:
            loss = loss + self.pearson_weight * pearson_correlation_loss(
                target, recon)
        if self.norm_ratio_weight > 0:
            loss = loss + self.norm_ratio_weight * norm_ratio_loss(
                target, recon)
        return loss

    def _mae_loss_per_node(self, recon, target):
        """Per-node MAE loss used by the routing selector.

        Returns a ``(N,)`` tensor matching the batch size so that callers can
        still ``argmin`` across experts. ``'mse'`` mode uses the row-wise mean
        squared error, ``'joint'`` mode uses the per-node equivalent of the
        three-term combination.
        """
        if self.mae_loss_type == 'mse':
            return ((recon - target) ** 2).mean(dim=1)

        loss = scaled_cosine_error_per_node(recon, target, gamma=self.mae_gamma)
        if self.pearson_weight > 0:
            loss = loss + self.pearson_weight * pearson_loss_per_node(
                target, recon)
        if self.norm_ratio_weight > 0:
            loss = loss + self.norm_ratio_weight * norm_ratio_loss_per_node(
                target, recon)
        return loss

    def _sample_shared_mask(self, num_target):
        """Sample one shared mask for all nodes: (num_target, input_dim) bool."""
        num_mask = int(self.mask_ratio * self.input_dim)
        indices = torch.randperm(self.input_dim, device=self.device)[:num_mask]
        row = torch.zeros(self.input_dim, dtype=torch.bool, device=self.device)
        row[indices] = True
        return row.unsqueeze(0).expand(num_target, -1)

    def _sample_pernode_mask(self, num_target):
        """Per-node independent mask: (num_target, input_dim) bool."""
        num_mask = int(self.mask_ratio * self.input_dim)
        rand = torch.rand(num_target, self.input_dim, device=self.device)
        _, topk_indices = rand.topk(num_mask, dim=1)
        mask_matrix = torch.zeros(num_target, self.input_dim,
                                  dtype=torch.bool, device=self.device)
        mask_matrix.scatter_(1, topk_indices, True)
        return mask_matrix

    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _validate_cls(self, expert, h, labels, valid_idx, curr_classes, g2l):
        expert.classifier.eval()
        curr_set = set(curr_classes)
        valid_indices = [idx for idx in valid_idx if labels[idx].item() in curr_set]
        if not valid_indices:
            return float('inf')

        valid_t = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
        local_labels = torch.tensor(
            [g2l[labels[idx].item()] for idx in valid_indices],
            dtype=torch.long, device=self.device)
        logits = expert.classifier(h[valid_t])
        val_loss = F.cross_entropy(logits, local_labels)
        return val_loss.item()

    @torch.no_grad()
    def _validate_mae(self, expert, h, labels, valid_idx, curr_classes):
        expert.mae_decoder.eval()
        curr_set = set(curr_classes)
        valid_indices = [idx for idx in valid_idx if labels[idx].item() in curr_set]
        if not valid_indices:
            return float('inf')

        valid_t = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
        mask_matrix = self._sample_shared_mask(valid_t.size(0))

        h_val = h[valid_t]
        masked_h = h_val.clone()
        mask_vals = expert.mask_token.unsqueeze(0).expand_as(masked_h)
        masked_h[mask_matrix] = mask_vals[mask_matrix]

        recon = expert.mae_decoder(masked_h)

        val_loss = self._mae_loss(recon, h_val)

        return val_loss.item()

    # ==================== Inference ====================

    @torch.no_grad()
    def _predict_nodes(self, subgraph, target_nodes):
        """Expert selection via MAE, then classify with selected expert's classifier."""
        self.model.eval()
        h = self._compute_embeddings(subgraph)

        num_experts = len(self.model.experts)
        target_t = torch.tensor(target_nodes, device=self.device, dtype=torch.long)
        num_target = target_t.size(0)
        infer_batch = self.config.get('infer_batch_size', 0)

        expert_assignments = torch.zeros(num_target, dtype=torch.long,
                                         device=self.device)

        if infer_batch <= 0 or infer_batch >= num_target:
            expert_assignments = self._select_experts_batch(
                h, target_t, num_experts)
        else:
            for start in range(0, num_target, infer_batch):
                end = min(start + infer_batch, num_target)
                batch_targets = target_t[start:end]
                expert_assignments[start:end] = self._select_experts_batch(
                    h, batch_targets, num_experts)

        predictions = torch.zeros(num_target, dtype=torch.long, device=self.device)
        active_experts = torch.unique(expert_assignments)

        for eid in active_experts:
            mask = (expert_assignments == eid)
            eid_int = eid.item()
            expert = self.model.experts[eid_int]
            l2g = self.expert_l2g[eid_int]
            logits = expert.classifier(h[target_t[mask]])
            local_preds = logits.argmax(dim=1)
            predictions[mask] = l2g[local_preds]

        return predictions.cpu(), expert_assignments.cpu()

    @torch.no_grad()
    def _select_experts_batch(self, h, batch_targets, num_experts):
        """Select best expert per node via MAE reconstruction error."""
        num_batch = batch_targets.size(0)
        mask_matrix = self._sample_pernode_mask(num_batch)

        h_batch = h[batch_targets]
        recon_errors = torch.zeros(num_experts, num_batch, device=self.device)

        for eid in range(num_experts):
            expert = self.model.experts[eid]
            masked_h = h_batch.clone()
            mask_vals = expert.mask_token.unsqueeze(0).expand_as(masked_h)
            masked_h[mask_matrix] = mask_vals[mask_matrix]
            recon = expert.mae_decoder(masked_h)

            recon_errors[eid] = self._mae_loss_per_node(recon, h_batch)

        return recon_errors.argmin(dim=0)

    # ==================== Evaluation ====================

    @torch.no_grad()
    def _evaluate_subgraph(self, subgraph, test_idx):
        target_nodes = subgraph['target_nodes']
        labels = subgraph['y']
        target_sorted = sorted(target_nodes)
        g2l = {g: l for l, g in enumerate(target_sorted)}

        node_preds, expert_assigns = self._predict_nodes(subgraph, target_sorted)
        true_labels = labels[target_sorted].cpu()

        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}
        test_positions = []
        for gid in test_idx:
            if gid in g2l:
                lid = g2l[gid]
                test_positions.append(lid)
                pred = node_preds[lid].item()
                true = true_labels[lid].item()
                per_class_total[true] = per_class_total.get(true, 0) + 1
                if pred == true:
                    correct += 1
                    per_class_correct[true] = per_class_correct.get(true, 0) + 1
                total += 1

        acc = correct / total if total > 0 else 0.0
        per_class_acc = []
        for c in sorted(per_class_total.keys()):
            c_correct = per_class_correct.get(c, 0)
            c_total = per_class_total[c]
            per_class_acc.append(c_correct / c_total if c_total > 0 else 0.0)
        macro_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0

        # Expert assignments restricted to test nodes -- this is what drives
        # both the printed expert distribution and the merging usage count.
        if test_positions:
            test_expert_assigns = expert_assigns[
                torch.tensor(test_positions, dtype=torch.long)]
        else:
            test_expert_assigns = torch.zeros(0, dtype=torch.long)

        return {
            'acc': acc, 'macro_acc': macro_acc,
            'correct': correct, 'total': total,
            'expert_assignments': test_expert_assigns,
        }

