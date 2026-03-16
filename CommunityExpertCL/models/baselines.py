"""
Baseline continual learning models for graph node classification.

All models share the same interface:
    __init__(task_loader, config, device)
    fit(trial) -> {'acc_matrix': List[List[float]], 'joint_acc': List[float]}

Models:
    BareModel   - Fine-tune only, no CL protection
    EWCModel    - Elastic Weight Consolidation (regularization)
    MASModel    - Memory Aware Synapses (regularization)
    GEMModel    - Gradient Episodic Memory (gradient projection)
    TWPModel    - Topology-aware Weight Preserving (graph-specific regularization)
    LwFModel    - Learning without Forgetting (knowledge distillation)
    JointModel  - Joint training on all seen data (upper bound)
"""

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .prediction_model import ClassificationGCN


# ======================================================================
# Base Class
# ======================================================================

class BaseCLModel:
    """Base class for single-GCN continual learning baselines."""

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device

        self.input_dim = task_loader.data.x.shape[1]
        self.gcn_hidden_dim = config.get('gcn_hidden_dim', 256)
        self.num_classes = max(task_loader.all_classes) + 1
        self.dropout = config.get('gcn_dropout', 0.5)

        self.cls_epochs = config.get('cls_epochs', 200)
        self.cls_lr = float(config.get('cls_lr', 0.01))
        self.cls_wd = float(config.get('cls_weight_decay', 5e-4))

        self.use_amp = config.get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.gcn = ClassificationGCN(
            self.input_dim, self.gcn_hidden_dim,
            self.num_classes, self.dropout
        ).to(device)

        self.current_session = 0

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

            print(f"\n--- Training (Session {session_id}) ---")
            self._train_session(
                session_id, subgraph, train_idx, valid_idx,
                curr_classes, all_classes, joint_subgraph
            )

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
                       curr_classes, all_classes, joint_subgraph):
        raise NotImplementedError

    # ==================== Inference & Evaluation ====================

    @torch.no_grad()
    def _predict_nodes(self, subgraph, target_nodes):
        """Run GCN on subgraph and predict target nodes."""
        self.gcn.eval()
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        target_t = torch.tensor(target_nodes, device=self.device, dtype=torch.long)
        with autocast(enabled=self.use_amp):
            logits, _ = self.gcn(x, edge_index)
        return logits[target_t].argmax(dim=1).cpu()

    @torch.no_grad()
    def _evaluate_subgraph(self, subgraph, test_idx):
        """Evaluate accuracy on test_idx within the subgraph."""
        target_nodes = subgraph['target_nodes']
        labels = subgraph['y']
        target_sorted = sorted(target_nodes)
        g2l = {g: l for l, g in enumerate(target_sorted)}

        node_preds = self._predict_nodes(subgraph, target_sorted)
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
            'expert_assignments': None,
        }

    # ==================== Helpers ====================

    def _get_loss_mask(self, labels, train_idx, target_classes):
        """Build boolean mask: nodes that are in train_idx AND belong to target_classes."""
        n = labels.size(0)
        train_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        for idx in train_idx:
            train_mask[idx] = True
        cls_set = set(target_classes)
        cls_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        for idx in range(n):
            if labels[idx].item() in cls_set:
                cls_mask[idx] = True
        return train_mask & cls_mask

    @torch.no_grad()
    def _validate_cls(self, x, edge_index, labels, valid_idx, curr_classes):
        """Compute cross-entropy on validation nodes of current classes."""
        self.gcn.eval()
        mask = self._get_loss_mask(labels, valid_idx, curr_classes)
        if mask.sum() == 0:
            return float('inf')
        with autocast(enabled=self.use_amp):
            logits, _ = self.gcn(x, edge_index)
            return F.cross_entropy(logits[mask], labels[mask]).item()

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


# ======================================================================
# Bare Model (no CL strategy)
# ======================================================================

class BareModel(BaseCLModel):
    """Fine-tune on each new task without any forgetting protection."""

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes, all_classes, joint_subgraph):
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        loss_mask = self._get_loss_mask(labels, train_idx, curr_classes)

        optimizer = optim.Adam(self.gcn.parameters(), lr=self.cls_lr,
                               weight_decay=self.cls_wd)
        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)
        best_val, best_state, patience_cnt = float('inf'), None, 0

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} Bare")
        for epoch in pbar:
            self.gcn.train()
            optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits, _ = self.gcn(x, edge_index)
                loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(
                    x, edge_index, labels, valid_idx, curr_classes)
                if val_loss < best_val:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.gcn.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        if best_state:
            self.gcn.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})


# ======================================================================
# EWC - Elastic Weight Consolidation
# ======================================================================

class EWCModel(BaseCLModel):
    """
    EWC: penalizes changes to parameters important for previous tasks.
    Loss = CE + lambda * sum_i F_i * (theta_i - theta*_i)^2
    """

    def __init__(self, task_loader, config, device):
        super().__init__(task_loader, config, device)
        self.ewc_lambda = float(config.get('ewc_lambda', 1000.0))
        self.fisher = {}
        self.old_params = {}

    def _compute_fisher(self, x, edge_index, labels, loss_mask):
        """Diagonal Fisher information from classification loss."""
        self.gcn.train()
        self.gcn.zero_grad()
        logits, _ = self.gcn(x, edge_index)
        loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
        loss.backward()

        fisher = {}
        for n, p in self.gcn.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] = p.grad.data.clone() ** 2
        return fisher

    def _ewc_penalty(self):
        """Compute EWC regularization penalty."""
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.gcn.named_parameters():
            if n in self.fisher and n in self.old_params:
                penalty = penalty + (self.fisher[n] * (p - self.old_params[n]) ** 2).sum()
        return penalty

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes, all_classes, joint_subgraph):
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        loss_mask = self._get_loss_mask(labels, train_idx, curr_classes)

        optimizer = optim.Adam(self.gcn.parameters(), lr=self.cls_lr,
                               weight_decay=self.cls_wd)
        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)
        best_val, best_state, patience_cnt = float('inf'), None, 0

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} EWC")
        for epoch in pbar:
            self.gcn.train()
            optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits, _ = self.gcn(x, edge_index)
                ce_loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
            ewc_loss = self._ewc_penalty()
            loss = ce_loss + self.ewc_lambda * ewc_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(
                    x, edge_index, labels, valid_idx, curr_classes)
                if val_loss < best_val:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.gcn.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                                 ewc=f'{ewc_loss.item():.4f}',
                                 val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                                 ewc=f'{ewc_loss.item():.4f}')

        if best_state:
            self.gcn.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        # Update Fisher and reference parameters (online EWC)
        new_fisher = self._compute_fisher(x, edge_index, labels, loss_mask)
        for n in new_fisher:
            if n in self.fisher:
                self.fisher[n] = self.fisher[n] + new_fisher[n]
            else:
                self.fisher[n] = new_fisher[n]
        self.old_params = {n: p.data.clone()
                           for n, p in self.gcn.named_parameters()
                           if p.requires_grad}


# ======================================================================
# MAS - Memory Aware Synapses
# ======================================================================

class MASModel(BaseCLModel):
    """
    MAS: importance is the sensitivity of the learned function output to parameter changes.
    Omega_i = E[|grad_theta_i ||f(x)||^2|]
    Loss = CE + lambda * sum_i Omega_i * (theta_i - theta*_i)^2
    """

    def __init__(self, task_loader, config, device):
        super().__init__(task_loader, config, device)
        self.mas_lambda = float(config.get('mas_lambda', 1.0))
        self.omega = {}
        self.old_params = {}

    def _compute_importance(self, x, edge_index, target_idx):
        """Compute parameter importance based on output sensitivity."""
        self.gcn.eval()
        self.gcn.zero_grad()
        logits, _ = self.gcn(x, edge_index)
        target_t = torch.tensor(target_idx, device=self.device, dtype=torch.long)
        output_norm = logits[target_t].norm(dim=1).mean()
        output_norm.backward()

        importance = {}
        for n, p in self.gcn.named_parameters():
            if p.requires_grad and p.grad is not None:
                importance[n] = p.grad.data.abs().clone()
        return importance

    def _mas_penalty(self):
        """Compute MAS regularization penalty."""
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.gcn.named_parameters():
            if n in self.omega and n in self.old_params:
                penalty = penalty + (self.omega[n] * (p - self.old_params[n]) ** 2).sum()
        return penalty

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes, all_classes, joint_subgraph):
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        loss_mask = self._get_loss_mask(labels, train_idx, curr_classes)

        optimizer = optim.Adam(self.gcn.parameters(), lr=self.cls_lr,
                               weight_decay=self.cls_wd)
        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)
        best_val, best_state, patience_cnt = float('inf'), None, 0

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} MAS")
        for epoch in pbar:
            self.gcn.train()
            optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits, _ = self.gcn(x, edge_index)
                ce_loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
            mas_loss = self._mas_penalty()
            loss = ce_loss + self.mas_lambda * mas_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(
                    x, edge_index, labels, valid_idx, curr_classes)
                if val_loss < best_val:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.gcn.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                                 mas=f'{mas_loss.item():.4f}',
                                 val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                                 mas=f'{mas_loss.item():.4f}')

        if best_state:
            self.gcn.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        # Update importance (accumulate across tasks)
        new_omega = self._compute_importance(x, edge_index, train_idx)
        for n in new_omega:
            if n in self.omega:
                self.omega[n] = self.omega[n] + new_omega[n]
            else:
                self.omega[n] = new_omega[n]
        self.old_params = {n: p.data.clone()
                           for n, p in self.gcn.named_parameters()
                           if p.requires_grad}


# ======================================================================
# GEM - Gradient Episodic Memory
# ======================================================================

class GEMModel(BaseCLModel):
    """
    GEM: stores episodic memory per task and projects gradients to prevent
    increasing loss on previous tasks.
    """

    def __init__(self, task_loader, config, device):
        super().__init__(task_loader, config, device)
        self.gem_memory_size = config.get('gem_memory_size', 100)
        self.memories = []

    def _flatten_grad(self):
        """Flatten all parameter gradients into a single vector."""
        grads = []
        for p in self.gcn.parameters():
            if p.requires_grad:
                grads.append(p.grad.data.view(-1) if p.grad is not None
                             else torch.zeros(p.numel(), device=self.device))
        return torch.cat(grads)

    def _set_grad_from_flat(self, flat_grad):
        """Write flat gradient vector back into parameter .grad fields."""
        offset = 0
        for p in self.gcn.parameters():
            if p.requires_grad:
                numel = p.numel()
                if p.grad is not None:
                    p.grad.data.copy_(flat_grad[offset:offset + numel].view_as(p))
                offset += numel

    def _project_gradient(self, grad, ref_grads):
        """
        Project gradient to satisfy all memory constraints:
        <projected, ref_i> >= 0 for all i.
        Uses iterative projection (Syed et al. approximation).
        """
        dotg = ref_grads @ grad
        if (dotg >= 0).all():
            return grad
        projected = grad.clone()
        for i in range(ref_grads.size(0)):
            dot = torch.dot(projected, ref_grads[i])
            if dot < 0:
                projected -= (dot / (torch.dot(ref_grads[i], ref_grads[i]) + 1e-12)
                              ) * ref_grads[i]
        return projected

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes, all_classes, joint_subgraph):
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        loss_mask = self._get_loss_mask(labels, train_idx, curr_classes)

        optimizer = optim.Adam(self.gcn.parameters(), lr=self.cls_lr,
                               weight_decay=self.cls_wd)
        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)
        best_val, best_state, patience_cnt = float('inf'), None, 0

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} GEM")
        for epoch in pbar:
            self.gcn.train()
            optimizer.zero_grad()
            logits, _ = self.gcn(x, edge_index)
            loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
            loss.backward()

            if self.memories:
                new_grad = self._flatten_grad()
                ref_grads = []
                for mem in self.memories:
                    optimizer.zero_grad()
                    mx = mem['subgraph']['x'].to(self.device)
                    mei = mem['subgraph']['edge_index'].to(self.device)
                    ml = mem['subgraph']['y'].to(self.device)
                    mm = self._get_loss_mask(ml, mem['mem_idx'], mem['classes'])
                    mlogits, _ = self.gcn(mx, mei)
                    mloss = F.cross_entropy(mlogits[mm], ml[mm])
                    mloss.backward()
                    ref_grads.append(self._flatten_grad())
                ref_grads = torch.stack(ref_grads)
                projected = self._project_gradient(new_grad, ref_grads)
                self._set_grad_from_flat(projected)

            optimizer.step()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(
                    x, edge_index, labels, valid_idx, curr_classes)
                if val_loss < best_val:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.gcn.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        if best_state:
            self.gcn.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        # Store memory for current task
        mem_size = min(self.gem_memory_size, len(train_idx))
        mem_idx = random.sample(train_idx, mem_size)
        self.memories.append({
            'subgraph': subgraph,
            'mem_idx': mem_idx,
            'classes': list(curr_classes),
        })


# ======================================================================
# TWP - Topology-aware Weight Preserving
# ======================================================================

class TWPModel(BaseCLModel):
    """
    TWP: combines task-performance importance (Fisher) with
    topology-preserving importance (how params affect neighbor aggregation).
    Loss = CE + lambda * sum_i (I_task + beta * I_topo)_i * (theta_i - theta*_i)^2
    """

    def __init__(self, task_loader, config, device):
        super().__init__(task_loader, config, device)
        self.twp_lambda = float(config.get('twp_lambda', 1000.0))
        self.twp_beta = float(config.get('twp_beta', 0.01))
        self.importance = {}
        self.old_params = {}

    def _compute_task_importance(self, x, edge_index, labels, loss_mask):
        """Fisher-based importance from classification loss."""
        self.gcn.train()
        self.gcn.zero_grad()
        logits, _ = self.gcn(x, edge_index)
        loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
        loss.backward()

        imp = {}
        for n, p in self.gcn.named_parameters():
            if p.requires_grad and p.grad is not None:
                imp[n] = p.grad.data.clone() ** 2
        return imp

    def _compute_topo_importance(self, x, edge_index_no_sl):
        """Topology importance: how params affect neighborhood similarity."""
        self.gcn.train()
        self.gcn.zero_grad()
        _, h = self.gcn(x, edge_index_no_sl)
        src, dst = edge_index_no_sl[0], edge_index_no_sl[1]
        sim = F.cosine_similarity(h[src], h[dst], dim=1)
        topo_loss = -sim.mean()
        topo_loss.backward()

        imp = {}
        for n, p in self.gcn.named_parameters():
            if p.requires_grad and p.grad is not None:
                imp[n] = p.grad.data.clone() ** 2
        return imp

    def _twp_penalty(self):
        """Compute TWP regularization penalty."""
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.gcn.named_parameters():
            if n in self.importance and n in self.old_params:
                penalty = penalty + (self.importance[n] * (p - self.old_params[n]) ** 2).sum()
        return penalty

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes, all_classes, joint_subgraph):
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        loss_mask = self._get_loss_mask(labels, train_idx, curr_classes)

        optimizer = optim.Adam(self.gcn.parameters(), lr=self.cls_lr,
                               weight_decay=self.cls_wd)
        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)
        best_val, best_state, patience_cnt = float('inf'), None, 0

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} TWP")
        for epoch in pbar:
            self.gcn.train()
            optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits, _ = self.gcn(x, edge_index)
                ce_loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
            twp_loss = self._twp_penalty()
            loss = ce_loss + self.twp_lambda * twp_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(
                    x, edge_index, labels, valid_idx, curr_classes)
                if val_loss < best_val:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.gcn.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                                 twp=f'{twp_loss.item():.4f}',
                                 val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                                 twp=f'{twp_loss.item():.4f}')

        if best_state:
            self.gcn.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        # Compute combined importance
        task_imp = self._compute_task_importance(x, edge_index, labels, loss_mask)
        edge_index_no_sl = subgraph['edge_index_no_selfloop'].to(self.device)
        topo_imp = self._compute_topo_importance(x, edge_index_no_sl)

        for n in task_imp:
            combined = task_imp[n] + self.twp_beta * topo_imp.get(n, 0.0)
            if n in self.importance:
                self.importance[n] = self.importance[n] + combined
            else:
                self.importance[n] = combined

        self.old_params = {n: p.data.clone()
                           for n, p in self.gcn.named_parameters()
                           if p.requires_grad}


# ======================================================================
# LwF - Learning without Forgetting
# ======================================================================

class LwFModel(BaseCLModel):
    """
    LwF: knowledge distillation from old model on new task data.
    Loss = CE(new classes) + alpha * KL(old_softmax || new_softmax) * T^2
    """

    def __init__(self, task_loader, config, device):
        super().__init__(task_loader, config, device)
        self.lwf_alpha = float(config.get('lwf_alpha', 1.0))
        self.lwf_temperature = float(config.get('lwf_temperature', 2.0))
        self.old_model = None
        self.old_classes = []

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes, all_classes, joint_subgraph):
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        loss_mask = self._get_loss_mask(labels, train_idx, curr_classes)

        # Precompute old model soft targets (frozen)
        old_soft = None
        if self.old_model is not None and self.old_classes:
            self.old_model.eval()
            with torch.no_grad():
                old_logits, _ = self.old_model(x, edge_index)
                old_soft = F.softmax(
                    old_logits[:, self.old_classes] / self.lwf_temperature, dim=1)

        optimizer = optim.Adam(self.gcn.parameters(), lr=self.cls_lr,
                               weight_decay=self.cls_wd)
        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)
        best_val, best_state, patience_cnt = float('inf'), None, 0

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} LwF")
        for epoch in pbar:
            self.gcn.train()
            optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits, _ = self.gcn(x, edge_index)
                ce_loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])

                kd_loss = torch.tensor(0.0, device=self.device)
                if old_soft is not None:
                    new_log_soft = F.log_softmax(
                        logits[:, self.old_classes] / self.lwf_temperature, dim=1)
                    kd_loss = F.kl_div(
                        new_log_soft[loss_mask],
                        old_soft[loss_mask].detach(),
                        reduction='batchmean'
                    ) * (self.lwf_temperature ** 2)

                loss = ce_loss + self.lwf_alpha * kd_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(
                    x, edge_index, labels, valid_idx, curr_classes)
                if val_loss < best_val:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.gcn.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                                 kd=f'{kd_loss.item():.4f}',
                                 val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                                 kd=f'{kd_loss.item():.4f}')

        if best_state:
            self.gcn.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        # Save current model for next session's distillation
        self.old_model = copy.deepcopy(self.gcn).to(self.device)
        self.old_model.eval()
        self.old_classes = list(all_classes)


# ======================================================================
# Joint Training (upper bound)
# ======================================================================

class JointModel(BaseCLModel):
    """
    Joint: re-initialize and train on ALL accumulated data each session.
    This is the upper-bound reference for CL methods.
    """

    def __init__(self, task_loader, config, device):
        super().__init__(task_loader, config, device)
        self.all_train_idx = []
        self.all_valid_idx = []
        self.all_seen_classes = []

    def _train_session(self, session_id, subgraph, train_idx, valid_idx,
                       curr_classes, all_classes, joint_subgraph):
        self.all_train_idx.extend(train_idx)
        self.all_valid_idx.extend(valid_idx)
        self.all_seen_classes = list(all_classes)

        # Re-initialize model each session for true joint upper bound
        self.gcn = ClassificationGCN(
            self.input_dim, self.gcn_hidden_dim,
            self.num_classes, self.dropout
        ).to(self.device)

        x = joint_subgraph['x'].to(self.device)
        edge_index = joint_subgraph['edge_index'].to(self.device)
        labels = joint_subgraph['y'].to(self.device)
        loss_mask = self._get_loss_mask(
            labels, self.all_train_idx, self.all_seen_classes)

        optimizer = optim.Adam(self.gcn.parameters(), lr=self.cls_lr,
                               weight_decay=self.cls_wd)
        valid_ep = self.config.get('valid_epoch', 10)
        patience = self.config.get('patience', 9999)
        best_val, best_state, patience_cnt = float('inf'), None, 0

        pbar = tqdm(range(self.cls_epochs), desc=f"S{session_id} Joint")
        for epoch in pbar:
            self.gcn.train()
            optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits, _ = self.gcn(x, edge_index)
                loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            if epoch > 0 and epoch % valid_ep == 0:
                val_loss = self._validate_cls(
                    x, edge_index, labels,
                    self.all_valid_idx, self.all_seen_classes)
                if val_loss < best_val:
                    best_val, patience_cnt = val_loss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.gcn.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt > patience:
                        break
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 val=f'{val_loss:.4f}')
            else:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        if best_state:
            self.gcn.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})


# ======================================================================
# Model Registry
# ======================================================================

BASELINE_MODELS = {
    'bare': BareModel,
    'ewc': EWCModel,
    'mas': MASModel,
    'gem': GEMModel,
    'twp': TWPModel,
    'lwf': LwFModel,
    'joint': JointModel,
}
