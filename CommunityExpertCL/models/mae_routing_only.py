"""
MAERoutingOnlyCL: MAE-as-routing theoretical upper-bound model (ablation).

Mirrors the structure of LiteExpertCL but replaces the MAE with:
  y     = warp_mlp(h)                 (non-linear warp, D -> D)
  alpha = alpha_mlp(h)                (mask scores, D -> D)
  mask  = hard-top-K(alpha) with STE  (differentiable hard selection)
  h_hat = decoder( mask * y + (1 - mask) * mask_token )

Loss targets the original paramfree-GCN embedding h (same joint / mse loss
options as Lite). The design lets the MLP choose which *nonlinear* submanifold
of the data to fix (via warp_mlp) and which K coordinates in that submanifold
to keep (via alpha_mlp + top-K); under MSE the Bayes-optimal reconstruction is
then the conditional expectation on that non-flat hypersurface.

No expert merging, no neighbor predictor, no pseudo-data, no distillation:
each session adds one expert, experts accumulate indefinitely. The purpose is
to probe the theoretical upper bound of MAE-based routing without any
merging-induced degradation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from .lite_expert_model import (
    scaled_cosine_error,
    scaled_cosine_error_per_node,
    pearson_correlation_loss,
    pearson_loss_per_node,
    norm_ratio_loss,
    norm_ratio_loss_per_node,
    paramfree_gcn,
)
from utils import compute_ap_af, print_cl_matrix


# ======================================================================
# Expert Module
# ======================================================================

class MAERoutingExpert(nn.Module):
    """Single expert: warp_mlp + alpha_mlp + decoder + mask_token + classifier.

    No neighbor predictor -- there is no expert merging in this model.
    """

    def __init__(self, embed_dim, mae_hidden_dim, cls_hidden_dim,
                 num_local_classes):
        super().__init__()
        self.embed_dim = embed_dim

        self.warp_mlp = nn.Sequential(
            nn.Linear(embed_dim, mae_hidden_dim),
            nn.ReLU(),
            nn.Linear(mae_hidden_dim, embed_dim),
        )

        self.alpha_mlp = nn.Sequential(
            nn.Linear(embed_dim, mae_hidden_dim),
            nn.ReLU(),
            nn.Linear(mae_hidden_dim, embed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, mae_hidden_dim),
            nn.ReLU(),
            nn.Linear(mae_hidden_dim, embed_dim),
        )

        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.xavier_uniform_(self.mask_token.unsqueeze(0))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.ReLU(),
            nn.Linear(cls_hidden_dim, num_local_classes),
        )

    def forward_mae(self, h, k_keep):
        """Forward MAE: h -> h_hat.

        Args:
            h:       (N, D) GCN embedding (the reconstruction target).
            k_keep:  number of dimensions to KEEP unmasked (top-K of alpha).

        Returns:
            h_hat:   (N, D) reconstructed embedding.
        """
        y = self.warp_mlp(h)                   # (N, D)
        alpha = self.alpha_mlp(h)              # (N, D)

        # Hard top-K selection: keep the k_keep dims with largest alpha.
        _, topk_idx = alpha.topk(k_keep, dim=-1)
        mask_hard = torch.zeros_like(alpha)
        mask_hard.scatter_(-1, topk_idx, 1.0)  # 1 = keep, 0 = mask

        # Straight-Through Estimator:
        #   forward value  = mask_hard  (since soft - soft.detach() == 0)
        #   backward grad  = d sigmoid(alpha) / d alpha  flows into alpha_mlp
        mask_soft = torch.sigmoid(alpha)
        mask = mask_hard + mask_soft - mask_soft.detach()

        # Masked positions get mask_token; kept positions get y.
        tilde_y = mask * y + (1.0 - mask) * self.mask_token.unsqueeze(0)
        h_hat = self.decoder(tilde_y)
        return h_hat


class MAERoutingModel(nn.Module):
    """Multi-expert container (experts added lazily per session)."""

    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList()


# ======================================================================
# MAERoutingOnlyCL - Main Model
# ======================================================================

class MAERoutingOnlyCL:
    """MAE-as-routing upper-bound CL: one expert per session, no merging."""

    EVAL_BATCH_SIZE = 8192

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device

        self.input_dim = task_loader.data.x.shape[1]
        self.num_classes = max(task_loader.all_classes) + 1

        # --- Paramfree GCN ---
        self.gcn_layers = config.get('gcn_layers', 1)

        # --- MAE ---
        self.mae_hidden_dim = config.get('mae_hidden_dim', 256)
        self.mae_epochs = config.get('mae_epochs', 200)
        self.mae_lr = float(config.get('mae_lr', 1e-3))
        self.mae_wd = float(config.get('mae_weight_decay', 1e-4))
        self.mask_ratio = config.get('mask_ratio', 0.5)
        self.mae_gamma = config.get('mae_gamma', 2)
        self.pearson_weight = config.get('pearson_weight', 0.0)
        self.norm_ratio_weight = config.get('norm_ratio_weight', 0.0)
        self.mae_loss_type = str(config.get('mae_loss', 'joint')).lower()
        if self.mae_loss_type not in ('joint', 'mse'):
            raise ValueError(
                f"Unknown mae_loss '{self.mae_loss_type}'. "
                f"Expected 'joint' or 'mse'.")

        # --- Classifier ---
        self.cls_hidden_dim = config.get('cls_hidden_dim', 256)
        self.cls_epochs = config.get('cls_epochs', 200)
        self.cls_lr = float(config.get('cls_lr', 0.01))
        self.cls_wd = float(config.get('cls_weight_decay', 5e-4))

        # --- Number of dims kept per MAE forward ---
        # mask_ratio is the fraction MASKED; k_keep is the count retained.
        num_mask = int(self.mask_ratio * self.input_dim)
        self.k_keep = max(1, self.input_dim - num_mask)

        self.model = MAERoutingModel().to(device)

        # Per-expert class mapping (populated lazily in _train_session)
        self.expert_g2l = []
        self.expert_l2g = []

        self.current_session = 0

    # ==================== Expert Creation ====================

    def _create_expert_with_fixed_init(self, num_local_classes):
        """Create expert with fixed-seed init so every expert starts identically."""
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state(self.device)
                    if self.device.type == 'cuda' else None)

        torch.manual_seed(42)
        if cuda_rng is not None:
            torch.cuda.manual_seed(42)

        expert = MAERoutingExpert(
            self.input_dim, self.mae_hidden_dim,
            self.cls_hidden_dim, num_local_classes
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
        routing_acc_history = []

        for session_id in range(num_sessions):
            self.current_session = session_id

            (curr_classes, _all_classes,
             subgraph, joint_subgraph,
             _train_loader, _valid_loader,
             _test_loader_joint) = self.task_loader.get_task(session_id)

            train_idx = self.task_loader.train_idx_per_task[session_id]
            valid_idx = self.task_loader.valid_idx_per_task[session_id]

            print(f"\n{'='*60}")
            print(f"Session {session_id}: Classes {curr_classes}")
            print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}")
            print(f"{'='*60}")

            print(f"\n--- Training (Session {session_id}) ---")
            self._train_session(
                session_id, subgraph, train_idx, valid_idx, curr_classes)

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
            routing_acc_history.append(joint_res['routing_acc'])
            print(f"  Acc={joint_res['acc']:.4f} "
                  f"Macro={joint_res['macro_acc']:.4f} "
                  f"Routing={joint_res['routing_acc']:.4f} "
                  f"({joint_res['correct']}/{joint_res['total']})")

            assigns = joint_res['expert_assignments']
            if assigns is not None and assigns.numel() > 0:
                unique_e, counts_e = torch.unique(assigns, return_counts=True)
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
        print(f"Routing Accuracy:       " + ", ".join(
            [f"S{i}={routing_acc_history[i]:.4f}" for i in range(num_sessions)]))

        return {
            'acc_matrix': acc_matrix,
            'joint_acc': joint_acc_history,
            'joint_macro_acc': joint_macro_history,
            'routing_acc': routing_acc_history,
            'ap_history': ap_history,
            'af': af,
            'final_ap': final_ap,
        }

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes):
        """Train one session's expert: Phase 1 MAE, Phase 2 Classifier."""
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
        train_indices = torch.where(loss_mask)[0]

        local_train_labels = torch.tensor(
            [g2l[labels[idx].item()] for idx in train_indices.tolist()],
            dtype=torch.long, device=self.device)

        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)

        expert = self.model.experts[-1]

        # ========== Phase 1: MAE (warp + alpha + decoder + mask_token) ==========
        self._freeze_all()
        for module in (expert.warp_mlp, expert.alpha_mlp, expert.decoder):
            for param in module.parameters():
                param.requires_grad = True
        expert.mask_token.requires_grad = True

        mae_params = (
            list(expert.warp_mlp.parameters())
            + list(expert.alpha_mlp.parameters())
            + list(expert.decoder.parameters())
            + [expert.mask_token]
        )
        optimizer = optim.Adam(mae_params, lr=self.mae_lr,
                               weight_decay=self.mae_wd)

        best_val = float('inf')
        best_mae_state = None
        patience_cnt = 0

        pbar = tqdm(range(self.mae_epochs), desc=f"S{session_id} MAE")
        for epoch in pbar:
            self.model.train()

            h_target = h[train_indices]
            recon = expert.forward_mae(h_target, self.k_keep)
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
                        'warp_mlp': {k: v.cpu().clone() for k, v
                                     in expert.warp_mlp.state_dict().items()},
                        'alpha_mlp': {k: v.cpu().clone() for k, v
                                      in expert.alpha_mlp.state_dict().items()},
                        'decoder': {k: v.cpu().clone() for k, v
                                    in expert.decoder.state_dict().items()},
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
            expert.warp_mlp.load_state_dict(
                {k: v.to(self.device) for k, v in best_mae_state['warp_mlp'].items()})
            expert.alpha_mlp.load_state_dict(
                {k: v.to(self.device) for k, v in best_mae_state['alpha_mlp'].items()})
            expert.decoder.load_state_dict(
                {k: v.to(self.device) for k, v in best_mae_state['decoder'].items()})
            expert.mask_token.data = best_mae_state['mask_token'].to(self.device)

        # ========== Phase 2: Classifier ==========
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

    # ==================== MAE loss ====================

    def _mae_loss(self, recon, target):
        """Scalar MAE loss used by training and validation.

        joint: scaled-cosine + pearson_weight*Pearson + norm_ratio_weight*norm-ratio.
        mse:   F.mse_loss(recon, target); the two weights above are ignored.
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
        """Per-node MAE loss used by the routing selector (returns shape (N,))."""
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

    # ==================== Freeze / Validation ====================

    def _freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _validate_mae(self, expert, h, labels, valid_idx, curr_classes):
        expert.eval()
        curr_set = set(curr_classes)
        valid_indices = [idx for idx in valid_idx if labels[idx].item() in curr_set]
        if not valid_indices:
            return float('inf')

        valid_t = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
        h_val = h[valid_t]
        recon = expert.forward_mae(h_val, self.k_keep)
        val_loss = self._mae_loss(recon, h_val)
        return val_loss.item()

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

    # ==================== Inference ====================

    @torch.no_grad()
    def _predict_nodes(self, subgraph, target_nodes):
        """Route via MAE recon error, then classify with routed expert."""
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
        """Select best expert per node via MAE reconstruction error.

        No random mask sampling here -- mask is fully determined by each
        expert's alpha_mlp applied to the query h, so routing is stable
        across calls.
        """
        h_batch = h[batch_targets]
        num_batch = h_batch.size(0)
        recon_errors = torch.zeros(num_experts, num_batch, device=self.device)

        for eid in range(num_experts):
            expert = self.model.experts[eid]
            recon = expert.forward_mae(h_batch, self.k_keep)
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

        # Build global-class -> owning-expert map (for routing accuracy).
        class_to_expert = {}
        for eid, l2g in enumerate(self.expert_l2g):
            for gc in l2g.tolist():
                class_to_expert[gc] = eid

        correct = 0
        total = 0
        route_correct = 0
        route_total = 0
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

                if true in class_to_expert:
                    route_total += 1
                    if expert_assigns[lid].item() == class_to_expert[true]:
                        route_correct += 1

        acc = correct / total if total > 0 else 0.0
        per_class_acc = []
        for c in sorted(per_class_total.keys()):
            c_correct = per_class_correct.get(c, 0)
            c_total = per_class_total[c]
            per_class_acc.append(c_correct / c_total if c_total > 0 else 0.0)
        macro_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0
        routing_acc = route_correct / route_total if route_total > 0 else 0.0

        # Expert assignments restricted to test nodes -- this is the real-scenario
        # routing distribution the user cares about.
        if test_positions:
            test_expert_assigns = expert_assigns[
                torch.tensor(test_positions, dtype=torch.long)]
        else:
            test_expert_assigns = torch.zeros(0, dtype=torch.long)

        return {
            'acc': acc, 'macro_acc': macro_acc,
            'correct': correct, 'total': total,
            'routing_acc': routing_acc,
            'expert_assignments': test_expert_assigns,
        }

