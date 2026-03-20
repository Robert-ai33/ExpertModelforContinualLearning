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
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.utils import degree
from tqdm import tqdm

from .prediction_model import (
    scaled_cosine_error, scaled_cosine_error_per_node,
    pearson_correlation_loss, pearson_loss_per_node,
    norm_ratio_loss, norm_ratio_loss_per_node,
)


# ======================================================================
# Model Components
# ======================================================================

def paramfree_gcn(x, edge_index, num_layers=1):
    """Parameter-free GCN: H = (D^{-1/2} A D^{-1/2})^k X."""
    num_nodes = x.size(0)
    row, col = edge_index[0], edge_index[1]
    deg = degree(col, num_nodes, dtype=x.dtype).clamp(min=1)
    deg_inv_sqrt = deg.pow(-0.5)

    h = x
    for _ in range(num_layers):
        src_norm = deg_inv_sqrt[row]
        dst_norm = deg_inv_sqrt[col]
        edge_weight = src_norm * dst_norm
        out = torch.zeros_like(h)
        out.scatter_add_(0, col.unsqueeze(1).expand_as(h[row]),
                         h[row] * edge_weight.unsqueeze(1))
        h = out
    return h


class LiteExpert(nn.Module):
    """Single expert: MAE decoder + learnable mask_token + classifier (local classes only)."""

    def __init__(self, embed_dim, cls_hidden_dim, num_local_classes):
        super().__init__()
        self.mae_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.xavier_uniform_(self.mask_token.unsqueeze(0))

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

        self.use_amp = config.get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Expert merging hyperparameters
        self.merge_pseudo_samples = config.get('merge_pseudo_samples', 256)
        self.merge_pseudo_steps = config.get('merge_pseudo_steps', 300)
        self.merge_pseudo_lr = float(config.get('merge_pseudo_lr', 0.1))
        self.merge_entropy_weight = config.get('merge_entropy_weight', 0.1)
        self.merge_diversity_weight = config.get('merge_diversity_weight', 0.1)
        self.merge_temperature = config.get('merge_temperature', 2.0)
        self.merge_distill_mae_epochs = config.get('merge_distill_mae_epochs', 200)
        self.merge_distill_cls_epochs = config.get('merge_distill_cls_epochs', 200)
        self.merge_stats_weight = config.get('merge_stats_weight', 1.0)

        self.model = LiteModel().to(device)

        # Per-expert class mapping (populated lazily in _train_session)
        self.expert_g2l = []   # expert_g2l[eid] = {global_class: local_index}
        self.expert_l2g = []   # expert_l2g[eid] = tensor [local_index -> global_class]
        self.expert_usage_count = []  # usage count from last Joint Test
        self.expert_stats = []  # {mean, var, n} of training embeddings per expert

        self.current_session = 0

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

            # Isolated Tests
            print(f"\n--- Isolated Tests (Session {session_id}) ---")
            acc_row = []
            for tid in range(session_id + 1):
                iso_subgraph = self.task_loader.subgraph_isolated[tid]
                test_idx = self.task_loader.test_idx_per_task[tid]
                task_classes = self.task_loader.class_splits[tid]

                if not test_idx:
                    acc_row.append(0.0)
                    print(f"  Task {tid} (classes {task_classes}): no test nodes")
                    continue

                res = self._evaluate_subgraph(iso_subgraph, test_idx)
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
            print(f"  Acc={joint_res['acc']:.4f} "
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

        self._print_cl_matrix("CL Accuracy Matrix", acc_matrix, num_sessions)

        print(f"\nJoint Accuracy: " + ", ".join(
            [f"S{i}={joint_acc_history[i]:.4f}" for i in range(num_sessions)]))

        return {
            'acc_matrix': acc_matrix,
            'joint_acc': joint_acc_history,
        }

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes):
        """Train one session's expert: Phase 1 classifier, Phase 2 MAE."""
        # Build local class mapping: sorted for deterministic order
        sorted_classes = sorted(curr_classes)
        g2l = {c: i for i, c in enumerate(sorted_classes)}
        l2g = torch.tensor(sorted_classes, dtype=torch.long, device=self.device)

        # Always create and append new expert
        new_expert = LiteExpert(
            self.input_dim, self.cls_hidden_dim, len(sorted_classes)
        ).to(self.device)

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

        # Record embedding statistics for pseudo data generation
        with torch.no_grad():
            h_train = h[train_indices]
            self.expert_stats.append({
                'mean': h_train.mean(dim=0),
                'var': h_train.var(dim=0, correction=0),
                'n': h_train.size(0),
            })
        local_train_labels = torch.tensor(
            [g2l[labels[idx].item()] for idx in train_indices.tolist()],
            dtype=torch.long, device=self.device)

        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)

        expert = self.model.experts[-1]

        # ========== Phase 1: Classifier ==========
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
            with autocast(enabled=self.use_amp):
                logits = expert.classifier(h[train_indices])
                loss = F.cross_entropy(logits, local_train_labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

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

        # ========== Phase 2: MAE (decoder + mask_token) ==========
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
            with autocast(enabled=self.use_amp):
                h_target = h[curr_train_indices]
                masked_h = h_target.clone()
                mask_vals = expert.mask_token.unsqueeze(0).expand_as(masked_h)
                masked_h[mask_matrix] = mask_vals[mask_matrix]

                recon = expert.mae_decoder(masked_h)

                recon_loss = scaled_cosine_error(
                    recon, h_target, gamma=self.mae_gamma)

                pearson_loss_val = torch.tensor(0.0, device=self.device)
                if self.pearson_weight > 0:
                    pearson_loss_val = pearson_correlation_loss(h_target, recon)

                norm_ratio_loss_val = torch.tensor(0.0, device=self.device)
                if self.norm_ratio_weight > 0:
                    norm_ratio_loss_val = norm_ratio_loss(h_target, recon)

                loss = (recon_loss + self.pearson_weight * pearson_loss_val
                       + self.norm_ratio_weight * norm_ratio_loss_val)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

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
                    recon=f'{recon_loss.item():.4f}',
                    pearson=f'{pearson_loss_val.item():.4f}',
                    norm=f'{norm_ratio_loss_val.item():.4f}',
                    val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(
                    recon=f'{recon_loss.item():.4f}',
                    pearson=f'{pearson_loss_val.item():.4f}',
                    norm=f'{norm_ratio_loss_val.item():.4f}')

        if best_mae_state is not None:
            expert.mae_decoder.load_state_dict(
                {k: v.to(self.device) for k, v in best_mae_state['decoder'].items()})
            expert.mask_token.data = best_mae_state['mask_token'].to(self.device)

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
        pseudo_a = self._generate_pseudo_data(expert_a, len(classes_a), stats_a, tag="A")
        pseudo_b = self._generate_pseudo_data(expert_b, len(classes_b), stats_b, tag="B")

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

        # Step 3: Create merged expert and distill
        merged_expert = LiteExpert(
            self.input_dim, self.cls_hidden_dim, num_merged
        ).to(self.device)

        self._distill_mae(merged_expert, all_pseudo)
        self._distill_classifier(merged_expert, all_pseudo, all_soft_labels)

        # Step 4: Compute merged stats for the new expert
        n_a, n_b = stats_a['n'], stats_b['n']
        n_ab = n_a + n_b
        mean_ab = (n_a * stats_a['mean'] + n_b * stats_b['mean']) / n_ab
        var_ab = ((n_a * (stats_a['var'] + stats_a['mean'] ** 2)
                   + n_b * (stats_b['var'] + stats_b['mean'] ** 2)) / n_ab
                  - mean_ab ** 2)

        # Step 5: Remove old experts (reverse order to preserve indices)
        for idx in sorted([idx_a, idx_b], reverse=True):
            del self.model.experts[idx]
            del self.expert_g2l[idx]
            del self.expert_l2g[idx]
            del self.expert_usage_count[idx]
            del self.expert_stats[idx]

        # Step 6: Add merged expert
        self.model.experts.append(merged_expert)
        self.expert_g2l.append(merged_g2l)
        self.expert_l2g.append(merged_l2g)
        self.expert_usage_count.append(0)
        self.expert_stats.append({'mean': mean_ab, 'var': var_ab, 'n': n_ab})

        print(f"  -> Merged expert manages classes {merged_classes} "
              f"(total experts: {len(self.model.experts)})")

    def _generate_pseudo_data(self, expert, num_classes, stats, tag=""):
        """Optimize random data to minimize MAE recon + entropy + balance + stats match."""
        expert.eval()
        for p in expert.parameters():
            p.requires_grad = False

        target_mean = stats['mean'].detach()
        target_var = stats['var'].detach()

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
            loss_mae = scaled_cosine_error(recon, h_fake, gamma=self.mae_gamma)
            if self.pearson_weight > 0:
                loss_mae = loss_mae + self.pearson_weight * pearson_correlation_loss(h_fake, recon)
            if self.norm_ratio_weight > 0:
                loss_mae = loss_mae + self.norm_ratio_weight * norm_ratio_loss(h_fake, recon)

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

            loss = (loss_mae
                    + self.merge_entropy_weight * loss_entropy
                    + self.merge_diversity_weight * loss_diversity
                    + self.merge_stats_weight * loss_stats)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                pbar.set_postfix(mae=f'{loss_mae.item():.4f}',
                                 ent=f'{loss_entropy.item():.4f}',
                                 bal=f'{loss_diversity.item():.4f}',
                                 stat=f'{loss_stats.item():.4f}')

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

            loss = scaled_cosine_error(recon, pseudo_data, gamma=self.mae_gamma)
            if self.pearson_weight > 0:
                loss = loss + self.pearson_weight * pearson_correlation_loss(
                    pseudo_data, recon)
            if self.norm_ratio_weight > 0:
                loss = loss + self.norm_ratio_weight * norm_ratio_loss(
                    pseudo_data, recon)

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

    # ==================== Validation ====================

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
        with autocast(enabled=self.use_amp):
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

        with autocast(enabled=self.use_amp):
            h_val = h[valid_t]
            masked_h = h_val.clone()
            mask_vals = expert.mask_token.unsqueeze(0).expand_as(masked_h)
            masked_h[mask_matrix] = mask_vals[mask_matrix]

            recon = expert.mae_decoder(masked_h)

            val_loss = scaled_cosine_error(recon, h_val, gamma=self.mae_gamma)

            if self.pearson_weight > 0:
                val_loss = val_loss + self.pearson_weight * pearson_correlation_loss(h_val, recon)

            if self.norm_ratio_weight > 0:
                val_loss = val_loss + self.norm_ratio_weight * norm_ratio_loss(h_val, recon)

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
            with autocast(enabled=self.use_amp):
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
            with autocast(enabled=self.use_amp):
                masked_h = h_batch.clone()
                mask_vals = expert.mask_token.unsqueeze(0).expand_as(masked_h)
                masked_h[mask_matrix] = mask_vals[mask_matrix]
                recon = expert.mae_decoder(masked_h)

                scaled_cos = scaled_cosine_error_per_node(
                    recon, h_batch, gamma=self.mae_gamma)
                pearson = pearson_loss_per_node(h_batch, recon)
                nr = norm_ratio_loss_per_node(h_batch, recon)

            recon_errors[eid] = (scaled_cos + self.pearson_weight * pearson
                                + self.norm_ratio_weight * nr)

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
        for gid in test_idx:
            if gid in g2l:
                lid = g2l[gid]
                if node_preds[lid].item() == true_labels[lid].item():
                    correct += 1
                total += 1

        acc = correct / total if total > 0 else 0.0
        return {
            'acc': acc, 'correct': correct, 'total': total,
            'expert_assignments': expert_assigns,
        }

    # ==================== Printing ====================

    @staticmethod
    def _print_cl_matrix(title, matrix, num_sessions):
        print(f"\n{title}:")
        header = "Session | " + " | ".join(
            [f"Task {i:5d}" for i in range(num_sessions)])
        print(header)
        print("-" * len(header))
        for sid, row in enumerate(matrix):
            parts = []
            for tid in range(num_sessions):
                if tid < len(row):
                    parts.append(f"{row[tid]:.4f} ")
                else:
                    parts.append("       ")
            print(f"   {sid}    | " + " | ".join(parts))
