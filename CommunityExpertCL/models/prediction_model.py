"""
CommunityExpertCL: Expert-based Continual Learning for Graph Data.

Multi-expert model where each expert contains:
1. Classification GCN: node classification (frozen after training)
2. Graph MAE Decoder + learnable mask_token: reconstructs node features

MAE Training (per expert, after freezing GCN):
- For ALL current-class training nodes:
  - Partial mask: randomly select λ*dim dimensions (one shared mask per epoch),
    replace those dimensions of x_v with mask_token values, keep the rest
  - Compute mean of neighbor features (excluding v)
  - Concatenate partially masked x_v with neighbor mean
  - Learnable linear W(2d -> d) + ReLU applied to concatenation
  - Decoder: Linear -> ReLU -> Linear to reconstruct x_v
- Loss: scaled cosine error
- Trainable: mae_decoder + mae_agg_linear + mask_token

Expert Selection (inference):
- Each node independently samples its own mask (λ*dim dimensions)
- Same node uses same mask across all experts (fair comparison)
- Different nodes use different masks
- Select expert with minimum reconstruction error

Node Classification:
- Use selected expert's GCN (unmasked features) to predict class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from tqdm import tqdm



# ======================================================================
# Model Components
# ======================================================================

class ClassificationGCN(nn.Module):
    """Single-layer GCN + linear classifier."""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv = GCNConv(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """Returns (logits, embeddings)."""
        h = self.conv(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        return logits, h

    def get_embeddings(self, x, edge_index):
        """Get GCN embeddings (no dropout, for MAE)."""
        h = self.conv(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MAEDecoder(nn.Module):
    """
    Graph Masked Autoencoder decoder.

    Maps hidden representation back to original feature space
    for node feature reconstruction via Linear -> ReLU -> Linear.
    """

    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PredictionExpert(nn.Module):
    """Single expert: classification GCN + MAE decoder + learnable mask token + agg layer."""

    def __init__(self, input_dim, gcn_hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.gcn = ClassificationGCN(input_dim, gcn_hidden_dim, num_classes, dropout)
        self.mae_decoder = MAEDecoder(input_dim, input_dim)
        self.mask_token = nn.Parameter(torch.zeros(input_dim))
        nn.init.xavier_uniform_(self.mask_token.unsqueeze(0))
        self.mae_agg_linear = nn.Linear(2 * input_dim, input_dim)


class PredictionModel(nn.Module):
    """Multi-expert prediction model container."""

    def __init__(self, input_dim, gcn_hidden_dim, num_classes,
                 num_experts, dropout=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.experts = nn.ModuleList([
            PredictionExpert(input_dim, gcn_hidden_dim, num_classes, dropout)
            for _ in range(num_experts)
        ])


# ======================================================================
# Helpers
# ======================================================================

def compute_masked_agg(x, edge_index, mask_token, agg_linear,
                       target_indices, num_nodes, mask_matrix):
    """
    Per-node partial-masked aggregation: concat(partially_masked_x_v, neighbor_mean) -> Linear -> ReLU.

    Args:
        mask_matrix: (num_target, feat_dim) bool tensor, per-node mask.

    For each target node v:
    1. Partial mask: copy x_v, replace dimensions where mask_matrix[i]=True with mask_token values
    2. Compute mean of neighbor features (excluding v itself)
    3. Concatenate partially masked x_v with neighbor_mean
    4. Apply learnable linear W(2d -> d) + ReLU
    """
    src, dst = edge_index[0], edge_index[1]
    feat_dim = x.size(1)

    non_self = src != dst
    src_ns, dst_ns = src[non_self], dst[non_self]

    deg = degree(dst_ns, num_nodes).clamp(min=1)
    agg = torch.zeros(num_nodes, feat_dim, device=x.device)
    agg.index_add_(0, dst_ns, x[src_ns])
    neighbor_mean = agg[target_indices] / deg[target_indices].unsqueeze(1)

    masked_x = x[target_indices].clone()
    mask_vals = mask_token.unsqueeze(0).expand_as(masked_x)
    masked_x[mask_matrix] = mask_vals[mask_matrix]
    concat = torch.cat([masked_x, neighbor_mean], dim=1)

    return F.relu(agg_linear(concat))


def scaled_cosine_error(pred, target, gamma=2):
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    return ((1 - cos_sim) ** gamma).mean()


# ======================================================================
# CommunityExpertCL - Main Model
# ======================================================================

class CommunityExpertCL:
    """Expert-based continual learning model with MAE expert selection."""

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device
        self.debug = config.get('debug', False)

        self.input_dim = task_loader.data.x.shape[1]
        self.gcn_hidden_dim = config.get('gcn_hidden_dim', 256)
        self.num_classes = max(task_loader.all_classes) + 1
        self.num_experts = config.get('num_experts', 7)
        self.dropout = config.get('gcn_dropout', 0.5)

        self.cls_epochs = config.get('cls_epochs', 200)
        self.cls_lr = float(config.get('cls_lr', 0.01))
        self.cls_wd = float(config.get('cls_weight_decay', 5e-4))

        self.mae_epochs = config.get('mae_epochs', 200)
        self.mae_lr = float(config.get('mae_lr', 1e-3))
        self.mae_wd = float(config.get('mae_weight_decay', 1e-4))
        self.mae_gamma = config.get('mae_gamma', 2)
        self.mask_ratio = config.get('mask_ratio', 0.5)
        self.use_dispersion_loss = config.get('use_dispersion_loss', True)
        self.dispersion_weight = config.get('dispersion_weight', 0.5)
        self.dispersion_margin = config.get('dispersion_margin', 0.1)

        self.model = PredictionModel(
            self.input_dim, self.gcn_hidden_dim, self.num_classes,
            self.num_experts, self.dropout
        ).to(device)

        self.current_session = 0

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

            # ========== Train ==========
            print(f"\n--- Training (Session {session_id}) ---")
            self._train_session(
                session_id, subgraph, train_idx, valid_idx,
                curr_classes, valid_loader
            )

            # ========== Isolated Tests (fill CL matrix row) ==========
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

            # ========== Joint Test ==========
            print(f"\n--- Joint Test (Session {session_id}) ---")
            test_idx_joint = self.task_loader.test_idx_joint[session_id]
            joint_res = self._evaluate_subgraph(joint_subgraph, test_idx_joint)

            joint_acc_history.append(joint_res['acc'])

            print(f"  Acc={joint_res['acc']:.4f} "
                  f"({joint_res['correct']}/{joint_res['total']})")

            assigns = joint_res['expert_assignments']
            if assigns is not None and assigns.numel() > 0:
                unique_e, counts_e = torch.unique(assigns, return_counts=True)
                dist = ", ".join([f"E{e.item()}:{c.item()}"
                                  for e, c in zip(unique_e, counts_e)])
                print(f"  Expert distribution: {dist}")

        # ========== Final Summary ==========
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
                       curr_classes, valid_loader):
        """Train one session's expert: Phase 1 GCN, Phase 2 MAE."""
        expert_id = session_id % self.num_experts

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
        for idx in train_idx:
            train_mask[idx] = True

        curr_class_set = set(curr_classes)
        curr_class_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
        for idx in range(x.size(0)):
            if labels[idx].item() in curr_class_set:
                curr_class_mask[idx] = True

        loss_mask = train_mask & curr_class_mask

        # ========== Phase 1: Classification GCN ==========
        self._freeze_all()
        gcn = self.model.experts[expert_id].gcn
        for param in gcn.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(gcn.parameters(), lr=self.cls_lr,
                               weight_decay=self.cls_wd)
        best_val = float('inf')
        best_gcn_state = None
        patience_cnt = 0
        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} GCN")
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()
            logits, _ = gcn(x, edge_index)
            loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
            loss.backward()
            optimizer.step()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(gcn, x, edge_index, labels,
                                              valid_idx, curr_classes)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_cnt = 0
                    best_gcn_state = {k: v.cpu().clone()
                                      for k, v in gcn.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(loss=f'{loss.item():.4f}', val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        if best_gcn_state is not None:
            gcn.load_state_dict({k: v.to(self.device)
                                 for k, v in best_gcn_state.items()})

        # ========== Phase 2: MAE Decoder + mask_token (frozen GCN) ==========
        self._freeze_all()
        expert = self.model.experts[expert_id]
        for param in expert.mae_decoder.parameters():
            param.requires_grad = True
        for param in expert.mae_agg_linear.parameters():
            param.requires_grad = True
        expert.mask_token.requires_grad = True

        mae_params = (list(expert.mae_decoder.parameters())
                      + list(expert.mae_agg_linear.parameters())
                      + [expert.mask_token])
        optimizer = optim.Adam(mae_params, lr=self.mae_lr,
                               weight_decay=self.mae_wd)
        best_val = float('inf')
        best_mae_state = None
        patience_cnt = 0

        num_nodes = x.size(0)
        curr_train_indices = torch.where(loss_mask)[0]

        old_tokens = None
        if session_id > 0 and self.use_dispersion_loss and self.dispersion_weight > 0:
            old_expert_ids = sorted(
                {s % self.num_experts for s in range(session_id)} - {expert_id}
            )
            if old_expert_ids:
                old_tokens_list = [
                    self.model.experts[eid].mask_token.detach()
                    for eid in old_expert_ids
                ]
                old_tokens = torch.stack(old_tokens_list).to(self.device)

        pbar = tqdm(range(self.mae_epochs), desc=f"S{session_id} MAE")
        for epoch in pbar:
            self.model.train()
            expert.mae_decoder.train()

            mask_matrix = self._sample_shared_mask(curr_train_indices.size(0))
            masked_agg = compute_masked_agg(
                x, edge_index, expert.mask_token, expert.mae_agg_linear,
                curr_train_indices, num_nodes, mask_matrix)
            recon = expert.mae_decoder(masked_agg)

            recon_loss = scaled_cosine_error(
                recon, x[curr_train_indices], gamma=self.mae_gamma)

            dispersion_loss = torch.tensor(0.0, device=self.device)
            if old_tokens is not None:
                curr_token = expert.mask_token.unsqueeze(0)
                cos_sims = F.cosine_similarity(curr_token, old_tokens, dim=1)
                dispersion_loss = F.relu(cos_sims - self.dispersion_margin).mean()

            loss = recon_loss + self.dispersion_weight * dispersion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_mae(
                    expert_id, x, edge_index, valid_idx,
                    curr_classes, labels, num_nodes)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_cnt = 0
                    best_mae_state = {
                        'decoder': {k: v.cpu().clone() for k, v
                                    in expert.mae_decoder.state_dict().items()},
                        'agg_linear': {k: v.cpu().clone() for k, v
                                       in expert.mae_agg_linear.state_dict().items()},
                        'mask_token': expert.mask_token.data.cpu().clone(),
                    }
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(
                    recon=f'{recon_loss.item():.4f}',
                    disp=f'{dispersion_loss.item():.4f}',
                    val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(
                    recon=f'{recon_loss.item():.4f}',
                    disp=f'{dispersion_loss.item():.4f}')

        if best_mae_state is not None:
            expert.mae_decoder.load_state_dict(
                {k: v.to(self.device) for k, v in best_mae_state['decoder'].items()})
            expert.mae_agg_linear.load_state_dict(
                {k: v.to(self.device) for k, v in best_mae_state['agg_linear'].items()})
            expert.mask_token.data = best_mae_state['mask_token'].to(self.device)

    # ==================== Validation ====================

    def _sample_shared_mask(self, num_target):
        """Sample one mask and broadcast to all nodes: (num_target, input_dim) bool tensor.
        All nodes share the same masked dimensions. Used in training and validation."""
        num_mask = int(self.mask_ratio * self.input_dim)
        indices = torch.randperm(self.input_dim, device=self.device)[:num_mask]
        row = torch.zeros(self.input_dim, dtype=torch.bool, device=self.device)
        row[indices] = True
        return row.unsqueeze(0).expand(num_target, -1)

    def _sample_pernode_mask(self, num_target):
        """Sample per-node independent mask: (num_target, input_dim) bool tensor.
        Each node gets its own random masked dimensions. Used in expert selection."""
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
    def _validate_cls(self, gcn, x, edge_index, labels, valid_idx, curr_classes):
        gcn.eval()
        valid_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
        for idx in valid_idx:
            valid_mask[idx] = True

        curr_set = set(curr_classes)
        curr_mask = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)
        for idx in range(x.size(0)):
            if labels[idx].item() in curr_set:
                curr_mask[idx] = True

        val_loss_mask = valid_mask & curr_mask
        if val_loss_mask.sum() == 0:
            return float('inf')

        logits, _ = gcn(x, edge_index)
        return F.cross_entropy(logits[val_loss_mask], labels[val_loss_mask]).item()

    @torch.no_grad()
    def _validate_mae(self, expert_id, x, edge_index, valid_idx,
                      curr_classes, labels, num_nodes):
        self.model.eval()
        valid_indices = []
        curr_set = set(curr_classes)
        for idx in valid_idx:
            if labels[idx].item() in curr_set:
                valid_indices.append(idx)

        if not valid_indices:
            return float('inf')

        valid_t = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
        expert = self.model.experts[expert_id]

        mask_matrix = self._sample_shared_mask(valid_t.size(0))

        masked_agg = compute_masked_agg(
            x, edge_index, expert.mask_token, expert.mae_agg_linear,
            valid_t, num_nodes, mask_matrix)
        recon = expert.mae_decoder(masked_agg)
        return scaled_cosine_error(recon, x[valid_t],
                                   gamma=self.mae_gamma).item()

    # ==================== Inference ====================

    @torch.no_grad()
    def _predict_nodes(self, subgraph, target_nodes):
        """Expert selection via MAE, then run only selected experts' GCN."""
        self.model.eval()
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)

        num_nodes = x.size(0)
        num_experts = min(self.num_experts, self.current_session + 1)
        target_t = torch.tensor(target_nodes, device=self.device, dtype=torch.long)
        num_target = target_t.size(0)

        # Phase 1: MAE expert selection (lightweight, no GCN forward)
        mask_matrix = self._sample_pernode_mask(num_target)

        recon_errors = torch.zeros(num_experts, num_target, device=self.device)
        for eid in range(num_experts):
            expert = self.model.experts[eid]
            masked_agg = compute_masked_agg(
                x, edge_index, expert.mask_token, expert.mae_agg_linear,
                target_t, num_nodes, mask_matrix)
            recon = expert.mae_decoder(masked_agg)
            cos_sim = F.cosine_similarity(recon, x[target_t], dim=1)
            recon_errors[eid] = (1 - cos_sim) ** self.mae_gamma

        expert_assignments = recon_errors.argmin(dim=0)
        del recon_errors

        # Phase 2: only run GCN for experts that were actually selected
        predictions = torch.zeros(num_target, dtype=torch.long, device=self.device)
        active_experts = torch.unique(expert_assignments)

        for eid in active_experts:
            mask = (expert_assignments == eid)
            logits, _ = self.model.experts[eid.item()].gcn(x, edge_index)
            predictions[mask] = logits[target_t[mask]].argmax(dim=1)
            del logits

        return predictions.cpu(), expert_assignments.cpu()

    # ==================== Evaluation ====================

    @torch.no_grad()
    def _evaluate_subgraph(self, subgraph, test_idx):
        target_nodes = subgraph['target_nodes']
        labels = subgraph['y']
        target_sorted = sorted(target_nodes)
        g2l = {g: l for l, g in enumerate(target_sorted)}

        node_preds, expert_assigns = self._predict_nodes(subgraph, target_sorted)
        true_labels_target = labels[target_sorted].cpu()

        correct = 0
        total = 0
        for gid in test_idx:
            if gid in g2l:
                lid = g2l[gid]
                if node_preds[lid].item() == true_labels_target[lid].item():
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
        header = "Session | " + " | ".join([f"Task {i:5d}" for i in range(num_sessions)])
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
