"""
TEMCL: Topology-aware Embedding Memory for Continual Graph Learning.

Port of:
  Zhang et al. "Topology-aware Embedding Memory for Continual Learning
  on Expanding Networks." KDD 2024.
  Original impl: PDGNNs-main/source/Baselines/TEM_model.py (DGL-based).

PyG adaptation (this file) keeps the algorithm faithful but switches to
edge_index-based ops so it fits CommunityExpertCL's TaskLoader.

Backbone (matches PDGNNs default):
    CustomDecoupledAPPNP = APPNPConv(k, alpha) + MLP([d_in, *h_dims, n_cls])
    - neighbor_agg : parameter-free APPNP propagation
                     H_{l+1} = (1 - alpha) * D^{-1/2} A D^{-1/2} H_l + alpha * X
    - feat_trans   : MLP with dropout-linear-ReLU, no bias by default

Algorithm (class-IL, faithful to TEM_model.NET.observe):
    At each epoch of session k:
        topo_vecs = neighbor_agg(x, edge_index)             # no grad, no params
        input    = cat(topo_vecs[train_ids], TEM_vecs)      # replay concat
        labels   = cat(curr_labels, TEM_labels)
        IF first epoch of session k:
            memory sampling + append to TEM_vecs / TEM_labels
        logits     = feat_trans(input)
        logits_seen = logits[:, seen_classes]               # classifier_increase
        loss       = CE_balanced(logits_seen, remapped_labels)

Samplers (identical to TEM_utils.py):
    - random_select          : uniform per class
    - cover_max_select_01    : K-hop coverage count (O(N^2) memory)
    - cover_max_select_02    : in-degree surrogate (default in paper)

No expert-ensemble, no replay graph rebuild: each session updates the
single MLP in place using the concatenated (current, memory) embeddings.
Unseen classes are masked at eval (same convention as ACILCL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.utils import degree
from tqdm import tqdm

from utils import compute_ap_af, print_cl_matrix


# ======================================================================
# APPNP propagation (PyG port of dgl.nn.pytorch.conv.APPNPConv)
# ======================================================================

def _symmetric_norm_adj(edge_index, num_nodes, dtype):
    """Build D^{-1/2} A D^{-1/2} as a sparse COO tensor.

    Matches DGL's APPNPConv normalization on undirected graphs with
    self-loops already present in ``edge_index``:

        src_norm = out_deg^{-0.5},  dst_norm = in_deg^{-0.5}
        (A_norm)_{ij} = dst_norm[i] * A[i,j] * src_norm[j]

    For undirected graphs src_norm == dst_norm == deg^{-0.5}.
    Convention: ``adj @ h`` puts the aggregated result at destination rows.
    """
    row, col = edge_index[0], edge_index[1]
    deg = degree(col, num_nodes, dtype=dtype).clamp(min=1)
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(
        torch.stack([col, row]), edge_weight, (num_nodes, num_nodes)
    ).coalesce()
    return adj


class APPNPPropagate(nn.Module):
    """Parameter-free APPNP neighbour aggregation (no trainable params).

    H_0 = X
    for _ in range(k):
        H_{l+1} = (1 - alpha) * A_norm @ H_l + alpha * H_0

    Mirrors ``dgl.nn.pytorch.conv.APPNPConv`` (edge_drop=0.0), which is
    what ``PDGNNs-main/source/Backbones/gnns.py::CustomDecoupledAPPNP``
    instantiates as its ``neighbor_agg``.
    """

    def __init__(self, k: int = 2, alpha: float = 0.05):
        super().__init__()
        self.k = int(k)
        self.alpha = float(alpha)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adj = _symmetric_norm_adj(edge_index, num_nodes, x.dtype)
        h_0 = x
        h = x
        for _ in range(self.k):
            h = torch.sparse.mm(adj, h)
            h = (1.0 - self.alpha) * h + self.alpha * h_0
        return h


# ======================================================================
# CustomDecoupledAPPNP backbone (MLP feat_trans, matches original paper)
# ======================================================================

class CustomDecoupledAPPNP(nn.Module):
    """Decoupled APPNP backbone: parameter-free propagation + MLP.

    Replicates ``Backbones/gnns.py::CustomDecoupledAPPNP`` in PDGNNs-main.
    Default hyperparameters (paper):

        h_dims      = [256]
        dropout     = 0.0
        k           = 2       # APPNP propagation steps
        alpha       = 0.05    # teleport probability
        linear_bias = False   # applied to all MLP linears
        batch_norm  = False   # not used in the official feat_trans

    feat_trans ordering (preserved from original):

        for fc in fcs[:-1]:   x = ReLU(fc(dropout(x)))
        x = fcs[-1](dropout(x))
    """

    def __init__(self, in_dim, num_classes,
                 h_dims=(256,), k=2, alpha=0.05,
                 dropout=0.0, linear_bias=False):
        super().__init__()

        self.neighbor_agg = APPNPPropagate(k=k, alpha=alpha)

        mlp_dims = [in_dim] + list(h_dims) + [num_classes]
        fcs = []
        for i in range(1, len(mlp_dims)):
            fcs.append(nn.Linear(mlp_dims[i - 1], mlp_dims[i],
                                 bias=linear_bias))
        self.fcs = nn.ModuleList(fcs)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def feat_trans(self, x):
        for fc in self.fcs[:-1]:
            x = self.act(fc(self.dropout(x)))
        return self.fcs[-1](self.dropout(x))

    def forward(self, x, edge_index):
        h = self.neighbor_agg(x, edge_index)
        return self.feat_trans(h)


# ======================================================================
# Samplers (three strategies from PDGNNs-main/source/Baselines/TEM_utils.py)
# ======================================================================

class _SamplerBase(nn.Module):
    """Shared helpers for the three TEM samplers."""

    @staticmethod
    def _budget_size(budget, n_ids):
        if isinstance(budget, int):
            return max(1, min(budget, n_ids))
        return max(1, int(float(budget) * n_ids))

    @staticmethod
    def _fallback_random(ids, take):
        perm = torch.randperm(len(ids))[:take]
        return [ids[i.item()] for i in perm]


class RandomSelect(_SamplerBase):
    """Uniform random per-class sampling. Equivalent to ``random_select``."""

    def forward(self, ids_per_cls, budget, **kwargs):
        selected = []
        for ids in ids_per_cls:
            if not ids:
                continue
            take = self._budget_size(budget, len(ids))
            selected.extend(self._fallback_random(ids, take))
        return selected


class CoverMaxSelect01(_SamplerBase):
    """Coverage-based sampling: count how many nodes each class-node
    reaches after K-hop APPNP, then sample weighted by coverage.

    Reference: TEM_utils.py::cover_max_select_01.

    Builds an N x N float matrix via ``neighbor_agg(eye(N))``, so only
    suitable for small graphs (Cora / Citeseer / CoauthorCS scale). For
    large graphs use ``cover_max_select_02``.
    """

    def forward(self, ids_per_cls, budget, neighbor_agg_model=None,
                edge_index=None, num_nodes=None, device=None, **kwargs):
        assert neighbor_agg_model is not None
        assert edge_index is not None and num_nodes is not None
        indicators = torch.eye(num_nodes, device=device)
        covered = neighbor_agg_model(indicators, edge_index)
        cover_count = covered.bool().sum(dim=1).float()
        selected = []
        for ids in ids_per_cls:
            if not ids:
                continue
            take = self._budget_size(budget, len(ids))
            ids_t = torch.tensor(ids, device=device)
            judge = cover_count[ids_t]
            if judge.sum() <= 0:
                selected.extend(self._fallback_random(ids, take))
                continue
            prob = judge / judge.sum()
            idx = torch.multinomial(prob, take, replacement=False)
            selected.extend([ids[i.item()] for i in idx])
        return selected


class CoverMaxSelect02(_SamplerBase):
    """Degree-based proxy for coverage. The paper's default sampler.

    Reference: TEM_utils.py::cover_max_select_02.
    """

    def forward(self, ids_per_cls, budget, edge_index=None,
                num_nodes=None, device=None, **kwargs):
        assert edge_index is not None and num_nodes is not None
        # In-degree on the current subgraph's edge_index (matches DGL's
        # graph.in_degrees() used in the original sampler).
        in_deg = degree(edge_index[1], num_nodes,
                        dtype=torch.float32).to(device)
        selected = []
        for ids in ids_per_cls:
            if not ids:
                continue
            take = self._budget_size(budget, len(ids))
            ids_t = torch.tensor(ids, device=device)
            judge = in_deg[ids_t]
            if judge.sum() <= 0:
                selected.extend(self._fallback_random(ids, take))
                continue
            idx = torch.multinomial(judge, take, replacement=False)
            selected.extend([ids[i.item()] for i in idx])
        return selected


SAMPLER_REGISTRY = {
    'random_select': RandomSelect,
    'cover_max_select_01': CoverMaxSelect01,
    'cover_max_select_02': CoverMaxSelect02,
}


# ======================================================================
# TEMCL - Main Model
# ======================================================================

class TEMCL:
    """Topology-aware Embedding Memory (TEM) continual graph learning.

    Class-IL port of ``Baselines/TEM_model.py::NET.observe`` from
    PDGNNs-main. Backbone is ``CustomDecoupledAPPNP`` (paper default);
    the head is a two-layer MLP (``feat_trans``) trained with Adam on
    concatenated (current_task_features, memory_features).

    Faithful ordering preserved from the original (observe):
        1. Read OLD TEM_vecs, build input_concat.
        2. IF first epoch of this session: sample + append to TEM_vecs.
        3. Forward MLP + CE loss, backprop.
    """

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device

        self.input_dim = task_loader.data.x.shape[1]
        self.num_classes = max(task_loader.all_classes) + 1

        # --- Backbone (CustomDecoupledAPPNP defaults) ---
        self.h_dims = list(config.get('tem_hidden_dims', [256]))
        self.dropout = float(config.get('tem_dropout', 0.0))
        self.linear_bias = bool(config.get('tem_linear_bias', False))
        self.k = int(config.get('tem_k', 2))
        self.alpha = float(config.get('tem_alpha', 0.05))

        # --- Training ---
        self.epochs = int(config.get('tem_epochs', 200))
        self.lr = float(config.get('tem_lr', 0.005))
        self.weight_decay = float(config.get('tem_weight_decay', 5e-4))
        self.cls_balance = bool(config.get('tem_cls_balance', True))

        # --- Memory / sampler ---
        self.budget = int(config.get('tem_budget', 400))
        sampler_name = str(config.get('tem_sampler', 'cover_max_select_02'))
        if sampler_name not in SAMPLER_REGISTRY:
            raise ValueError(
                f"Unknown tem_sampler '{sampler_name}'. Options: "
                f"{list(SAMPLER_REGISTRY.keys())}")
        self.sampler_name = sampler_name
        self.sampler = SAMPLER_REGISTRY[sampler_name]()

        # --- Backbone (single instance, reused across sessions) ---
        self.net = CustomDecoupledAPPNP(
            in_dim=self.input_dim,
            num_classes=self.num_classes,
            h_dims=self.h_dims,
            k=self.k,
            alpha=self.alpha,
            dropout=self.dropout,
            linear_bias=self.linear_bias,
        ).to(device)

        self.opt = optim.Adam(self.net.parameters(), lr=self.lr,
                              weight_decay=self.weight_decay)

        # --- Memory: concat of topo_vecs and their global labels ---
        self.TEM_vecs = torch.empty(0, self.input_dim, device=device)
        self.TEM_labels = torch.empty(0, dtype=torch.long, device=device)
        self.current_task = -1

    # ==================== Helpers ====================

    def _filter_train_nodes(self, train_idx, subgraph, curr_classes):
        """Keep (a) nodes in this subgraph and (b) with current-session label.

        Same filter as lite / mae_routing / acil to exclude stray external
        neighbours that belong to other tasks' classes.
        """
        labels = subgraph['y']
        all_nodes_set = set(subgraph['all_nodes'])
        curr_set = set(curr_classes)
        return [n for n in train_idx
                if n in all_nodes_set and labels[n].item() in curr_set]

    def _ids_per_cls(self, train_ids_filtered, subgraph, curr_classes):
        """Partition filtered train ids by class (sorted by class)."""
        labels = subgraph['y']
        buckets = {c: [] for c in sorted(curr_classes)}
        for gid in train_ids_filtered:
            c = labels[gid].item()
            if c in buckets:
                buckets[c].append(gid)
        return [buckets[c] for c in sorted(curr_classes)]

    def _build_remap(self, seen_classes):
        """Global-class-id -> local index tensor for seen-class CE."""
        remap = torch.full((self.num_classes,), -1,
                           dtype=torch.long, device=self.device)
        for i, c in enumerate(seen_classes):
            remap[c] = i
        return remap

    def _compute_cls_weights(self, labels_local, num_seen):
        """1 / max(count_c, 1) per seen class (mirrors args.cls_balance)."""
        if not self.cls_balance:
            return None
        counts = torch.bincount(labels_local, minlength=num_seen).float()
        return 1.0 / counts.clamp(min=1.0)

    # ==================== Main Loop ====================

    def fit(self, trial):
        num_sessions = self.task_loader.sessions

        acc_matrix = []
        joint_acc_history = []
        joint_macro_history = []

        for session_id in range(num_sessions):
            (curr_classes, all_classes,
             subgraph, joint_subgraph,
             _train_loader, _valid_loader,
             _test_loader_joint) = self.task_loader.get_task(session_id)

            train_idx = self.task_loader.train_idx_per_task[session_id]

            print(f"\n{'=' * 60}")
            print(f"[TEM] Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"Train: {len(train_idx)}, "
                  f"sampler={self.sampler_name}, budget={self.budget}")
            print(f"{'=' * 60}")

            self._train_session(session_id, subgraph, train_idx,
                                curr_classes, all_classes)

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

            # -------- Joint test --------
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
        print(f"[TEM] FINAL RESULTS")
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

    # ==================== Training ====================

    def _train_session(self, session_id, subgraph, train_idx, curr_classes,
                       all_classes_so_far):
        """E epochs on this session; memory updated once at epoch 0."""
        x_full = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels_full = subgraph['y'].to(self.device)
        num_nodes = x_full.size(0)

        train_filtered = self._filter_train_nodes(train_idx, subgraph,
                                                  curr_classes)
        if not train_filtered:
            print(f"  [TEM] Session {session_id}: no valid train nodes; skip.")
            return

        train_ids_t = torch.tensor(train_filtered, dtype=torch.long,
                                   device=self.device)
        labels_train = labels_full[train_ids_t]
        ids_per_cls_train = self._ids_per_cls(train_filtered, subgraph,
                                              curr_classes)

        seen_classes = sorted(set(all_classes_so_far))
        num_seen = len(seen_classes)
        seen_idx_t = torch.tensor(seen_classes, dtype=torch.long,
                                  device=self.device)
        remap = self._build_remap(seen_classes)

        pbar = tqdm(range(self.epochs), desc=f"S{session_id} TEM")
        for epoch in pbar:
            self.net.train()

            # APPNP is parameter-free -> safe to redo each epoch with no_grad.
            with torch.no_grad():
                topo_vecs = self.net.neighbor_agg(x_full, edge_index)

            # Original observe order:
            #   (1) build input_concat from OLD TEM_vecs
            #   (2) if first epoch of this session, grow TEM_vecs in-place
            #   (3) compute loss on input_concat
            input_concat = torch.cat(
                [topo_vecs[train_ids_t], self.TEM_vecs], dim=0)
            labels_concat = torch.cat(
                [labels_train, self.TEM_labels], dim=0)

            if session_id != self.current_task:
                self.current_task = session_id
                selected_ids = self.sampler(
                    ids_per_cls_train, self.budget,
                    neighbor_agg_model=self.net.neighbor_agg,
                    edge_index=edge_index, num_nodes=num_nodes,
                    device=self.device,
                )
                if selected_ids:
                    sel_t = torch.tensor(selected_ids, dtype=torch.long,
                                         device=self.device)
                    new_vecs = topo_vecs[sel_t].detach()
                    new_labels = labels_full[sel_t].detach()
                    self.TEM_vecs = torch.cat(
                        [self.TEM_vecs, new_vecs], dim=0)
                    self.TEM_labels = torch.cat(
                        [self.TEM_labels, new_labels], dim=0)

            logits = self.net.feat_trans(input_concat)
            logits_seen = logits.index_select(1, seen_idx_t)
            local_labels = remap[labels_concat]

            loss_w = self._compute_cls_weights(local_labels, num_seen)
            loss = F.cross_entropy(logits_seen, local_labels, weight=loss_w)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if epoch == 0 or (epoch + 1) % 20 == 0:
                pbar.set_postfix(
                    loss=f'{loss.item():.4f}',
                    mem=f'{self.TEM_vecs.size(0)}',
                )

        print(f"  [TEM] Session {session_id}: trained "
              f"{len(train_filtered)} nodes, "
              f"memory size = {self.TEM_vecs.size(0)}.")

    # ==================== Evaluation ====================

    @torch.no_grad()
    def _evaluate(self, subgraph, test_idx, seen_classes_list):
        self.net.eval()
        x_full = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels_full = subgraph['y']
        all_nodes_set = set(subgraph['all_nodes'])

        topo_vecs = self.net.neighbor_agg(x_full, edge_index)
        logits = self.net.feat_trans(topo_vecs)  # (N_total, num_classes)

        # Mask unseen classes (same convention as ACIL / mae_routing).
        unseen = [c for c in self.task_loader.all_classes
                  if c not in set(seen_classes_list)]
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
                true = labels_full[gid].item()
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

