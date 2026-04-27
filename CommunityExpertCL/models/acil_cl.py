"""
ACILCL: Analytic Class-Incremental Learning for graph continual learning.

Port of:
  Zhuang et al. "ACIL: Analytic class-incremental learning with absolute
  memorization and privacy protection." NeurIPS 2022.
  (Original impl: analytic/ACIL.py + analytic/AnalyticLinear.py + analytic/Buffer.py)

Graph adaptation:
  - Backbone      : parameter-free GCN  H = (D^{-1/2} A D^{-1/2})^K X
                    (no trainable params, no base-training phase needed).
  - RandomBuffer  : fixed random Linear(D -> P) + ReLU; P = `buffer_size`.
                    Weights are drawn ONCE (seed-controlled) and never updated.
  - Classifier    : RecursiveLinear (analytic linear). Weights are updated by
                    recursive least squares (Eqs. (9)(10) of ACIL paper) via
                    Sherman--Morrison--Woodbury, in float64.

Per session k:
    h      = paramfree_gcn(subgraph_per_task[k])    # (N, D)
    feat   = ReLU(h @ W_buf + b_buf)                # (N, P)
    feat_1 = [feat, 1]                              # bias augmentation
    Y      = one_hot(labels, num_classes)           # (N, C)
    recursive_linear.fit(feat_1[train_ids], Y[train_ids])

Absolute Memorization (Thm. 1 of ACIL): the incremental updates of (R, W)
are exactly equal to the joint ridge solution on ALL seen data, regardless
of task order. No replay buffer, no gradient descent, no forgetting.

Inference restricts argmax to classes seen so far to avoid ties with the
all-zero columns for unseen classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lite_expert_model import paramfree_gcn
from utils import compute_ap_af, print_cl_matrix


# ======================================================================
# RandomBuffer: fixed random feature expansion
# ======================================================================

class RandomBuffer(nn.Module):
    """Fixed random Linear(D -> P) + ReLU.

    Mirrors ``analytic/Buffer.py::RandomBuffer`` in the original ACIL code.
    Weights are registered as ``buffer`` (not ``Parameter``) so they never
    receive gradients and are excluded from any optimiser. They are drawn
    once under a fixed seed so every trial gets the same expansion.
    """

    def __init__(self, in_features, out_features, activation='relu',
                 seed=1998):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        gen = torch.Generator(device='cpu').manual_seed(seed)
        # Default PyTorch Linear init: kaiming_uniform_(a=sqrt(5)).
        # Equivalent bound for the uniform buffer:
        #   k = 1 / sqrt(in_features)
        k = 1.0 / (in_features ** 0.5)
        W = (torch.rand(out_features, in_features, generator=gen) * 2 - 1) * k
        b = (torch.rand(out_features, generator=gen) * 2 - 1) * k
        self.register_buffer('weight', W)
        self.register_buffer('bias', b)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation in (None, 'none', 'identity'):
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported RandomBuffer activation: {activation}")

    def forward(self, x):
        return self.act(F.linear(x, self.weight, self.bias))


# ======================================================================
# RecursiveLinear: analytic linear with recursive least-squares updates
# ======================================================================

class RecursiveLinear(nn.Module):
    """Analytic linear classifier with recursive least-squares (RLS) updates.

    Faithful to ``analytic/AnalyticLinear.py::RecursiveLinear`` in the original
    code (Eqs. (9) and (10) of the ACIL paper):

        K    = (I_b + X R X^T)^{-1}                    shape (b, b)
        R   <- R - R X^T K X R                         shape (P, P)
        W   <- W + R X^T (Y - X W)                     shape (P, C)

    where X is a minibatch of expanded features (optionally bias-augmented),
    Y is the corresponding one-hot label matrix, P is the feature size and
    C is the total number of classes.

    Internal state (``R``, ``weight``) is stored in float64 for numerical
    stability of the matrix inverse (same as the original).
    """

    def __init__(self, in_features, num_classes, gamma=1e-3, bias=True,
                 dtype=torch.float64):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.gamma = gamma
        self.use_bias = bias
        self.dtype = dtype

        P = in_features + (1 if bias else 0)
        # R = (gamma * I)^{-1} = I / gamma  ->  R0 = I / gamma.
        # Equivalently we set R directly as eye/gamma (matches original).
        self.register_buffer(
            'R', torch.eye(P, dtype=dtype) / gamma)
        self.register_buffer(
            'weight', torch.zeros(P, num_classes, dtype=dtype))

    def _augment(self, x):
        """Append a column of 1s for the bias term."""
        if not self.use_bias:
            return x
        return torch.cat([x, torch.ones(x.size(0), 1,
                                        dtype=x.dtype, device=x.device)], dim=1)

    @torch.no_grad()
    def fit(self, x, y_onehot):
        """One RLS update on a minibatch.

        Args:
            x        : (b, in_features) expanded features (RandomBuffer output).
            y_onehot : (b, num_classes) one-hot labels.
        """
        X = self._augment(x).to(self.R.dtype)
        Y = y_onehot.to(self.R.dtype)

        # K^{-1} = I_b + X R X^T     (b, b)
        XR = X @ self.R                           # (b, P)
        K_inv = torch.eye(X.size(0), device=X.device, dtype=X.dtype) \
            + XR @ X.T
        K = torch.linalg.inv(K_inv)

        # Eq. (10): R <- R - R X^T K X R = R - XR^T K XR
        self.R = self.R - XR.T @ K @ XR
        # Symmetrise to suppress numerical drift.
        self.R = 0.5 * (self.R + self.R.T)

        # Eq. (9): W <- W + R X^T (Y - X W)
        residual = Y - X @ self.weight            # (b, C)
        self.weight = self.weight + self.R @ X.T @ residual

    @torch.no_grad()
    def forward(self, x):
        """Return logits (N, num_classes) in the input's original dtype."""
        X = self._augment(x).to(self.R.dtype)
        logits = X @ self.weight
        return logits.to(x.dtype)


# ======================================================================
# ACILCL - Main Model
# ======================================================================

class ACILCL:
    """Analytic continual learning over graphs with parameter-free GCN."""

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device

        self.input_dim = task_loader.data.x.shape[1]
        self.num_classes = max(task_loader.all_classes) + 1

        # --- Paramfree GCN ---
        self.gcn_layers = config.get('gcn_layers', 2)

        # --- RandomBuffer ---
        self.buffer_size = int(config.get('acil_buffer_size', 2048))
        self.buffer_activation = config.get('acil_buffer_activation', 'relu')
        # Seed for the ONE-TIME random buffer draw. Fixed across trials so
        # the analytic solution is comparable; change only for buffer ablation.
        self.buffer_seed = int(config.get('acil_buffer_seed', 1998))

        # --- RecursiveLinear ---
        self.gamma = float(config.get('acil_gamma', 1e-3))
        self.use_bias = bool(config.get('acil_use_bias', True))

        # --- RLS batch size (within a session). The inverse cost is O(b^3),
        #     so we cap this to keep per-update memory/time reasonable on
        #     large graphs (e.g. ogbn-arxiv has thousands of train nodes per
        #     session; a single 20k x 20k inverse would be prohibitive).
        self.rls_batch_size = int(config.get('acil_rls_batch_size', 1024))

        # Build buffer + analytic head lazily (buffer_size could depend on dim).
        self.random_buffer = RandomBuffer(
            self.input_dim, self.buffer_size,
            activation=self.buffer_activation, seed=self.buffer_seed,
        ).to(device)

        self.head = RecursiveLinear(
            self.buffer_size, self.num_classes,
            gamma=self.gamma, bias=self.use_bias,
        ).to(device)

        self.current_session = 0

    # ==================== Embedding ====================

    @torch.no_grad()
    def _compute_features(self, subgraph):
        """paramfree GCN -> RandomBuffer -> expanded features."""
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        h = paramfree_gcn(x, edge_index, num_layers=self.gcn_layers)
        feat = self.random_buffer(h)
        return feat

    # ==================== Training ====================

    def fit(self, trial):
        num_sessions = self.task_loader.sessions

        acc_matrix = []
        joint_acc_history = []
        joint_macro_history = []

        for session_id in range(num_sessions):
            self.current_session = session_id

            (curr_classes, all_classes,
             subgraph, joint_subgraph,
             _train_loader, _valid_loader,
             _test_loader_joint) = self.task_loader.get_task(session_id)

            train_idx = self.task_loader.train_idx_per_task[session_id]

            print(f"\n{'=' * 60}")
            print(f"[ACIL] Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"Train: {len(train_idx)}")
            print(f"{'=' * 60}")

            # -------- Analytic fit on current session's train nodes --------
            self._fit_session(session_id, subgraph, train_idx, curr_classes)

            # -------- Per-task tests on the cumulative subgraph (CGLB) --------
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

                res = self._evaluate(eval_subgraph, test_idx, all_classes)
                acc_row.append(res['acc'])
                print(f"  Task {tid} (classes {task_classes}): "
                      f"Acc={res['acc']:.4f} ({res['correct']}/{res['total']})")
            acc_matrix.append(acc_row)

            # -------- Joint test over union of test sets so far --------
            print(f"\n--- Joint Test (Session {session_id}) ---")
            test_idx_joint = self.task_loader.test_idx_joint[session_id]
            joint_res = self._evaluate(joint_subgraph, test_idx_joint,
                                       all_classes)
            joint_acc_history.append(joint_res['acc'])
            joint_macro_history.append(joint_res['macro_acc'])
            print(f"  Acc={joint_res['acc']:.4f} "
                  f"Macro={joint_res['macro_acc']:.4f} "
                  f"({joint_res['correct']}/{joint_res['total']})")

        # -------- Final summary --------
        print(f"\n{'=' * 60}")
        print(f"[ACIL] FINAL RESULTS")
        print(f"{'=' * 60}")
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

    # ==================== Analytic Fit ====================

    @torch.no_grad()
    def _fit_session(self, session_id, subgraph, train_idx, curr_classes):
        """Fit the RecursiveLinear head on current-session training nodes.

        Matches the original ACIL flow (``ACIL.py::ACILLearner.learn``):
        propagate features on the current subgraph, expand via RandomBuffer,
        one-hot the labels, then call ``RecursiveLinear.fit`` in minibatches.
        """
        feat_full = self._compute_features(subgraph)
        labels = subgraph['y'].to(self.device)

        all_nodes_set = set(subgraph['all_nodes'])
        curr_set = set(curr_classes)

        # Keep only train nodes (a) in this subgraph and (b) with a current-
        # session class label. Same filter as lite / mae_routing to exclude
        # stray external neighbours of other tasks.
        train_filtered = [n for n in train_idx
                          if n in all_nodes_set
                          and labels[n].item() in curr_set]
        if not train_filtered:
            print(f"  [ACIL] Session {session_id}: no valid train nodes; skip.")
            return

        train_ids = torch.tensor(train_filtered, dtype=torch.long,
                                 device=self.device)
        x_train = feat_full[train_ids]                         # (N, P)
        y_train = labels[train_ids]                            # (N,)
        y_onehot = F.one_hot(y_train, self.num_classes).float()

        # Shuffle so minibatches see a class-balanced view.
        perm = torch.randperm(x_train.size(0), device=self.device)
        x_train = x_train[perm]
        y_onehot = y_onehot[perm]

        b = self.rls_batch_size
        n_batches = (x_train.size(0) + b - 1) // b
        for i in range(n_batches):
            x_b = x_train[i * b: (i + 1) * b]
            y_b = y_onehot[i * b: (i + 1) * b]
            self.head.fit(x_b, y_b)

        print(f"  [ACIL] Session {session_id}: fitted "
              f"{x_train.size(0)} nodes in {n_batches} RLS batches "
              f"(batch_size={b}).")

    # ==================== Evaluation ====================

    @torch.no_grad()
    def _evaluate(self, subgraph, test_idx, seen_classes):
        """Predict with analytic head; restrict argmax to classes seen so far."""
        feat_full = self._compute_features(subgraph)
        labels = subgraph['y']
        all_nodes_set = set(subgraph['all_nodes'])

        logits = self.head(feat_full)                          # (N_total, C)

        # Zero-out unseen classes so they never win the argmax (their columns
        # in ``self.head.weight`` are already zero, but we mask defensively
        # in case of tiny numerical noise from the RLS updates).
        unseen = [c for c in self.task_loader.all_classes
                  if c not in set(seen_classes)]
        if unseen:
            logits[:, unseen] = float('-inf')

        preds = logits.argmax(dim=1).cpu()

        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}
        for gid in test_idx:
            if gid in all_nodes_set:
                pred = preds[gid].item()
                true = labels[gid].item()
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
        macro_acc = sum(per_class_acc) / len(per_class_acc) \
            if per_class_acc else 0.0

        return {
            'acc': acc, 'macro_acc': macro_acc,
            'correct': correct, 'total': total,
        }

