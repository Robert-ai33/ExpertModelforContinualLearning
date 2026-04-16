"""
Baseline continual learning methods for node classification.

Supported methods:
  bare     - Naive fine-tuning (no forgetting mitigation)
  joint    - Joint training on all seen data (upper bound)
  ewc      - Elastic Weight Consolidation
  mas      - Memory Aware Synapses
  lwf      - Learning without Forgetting
  gem      - Gradient Episodic Memory
  twp      - Topology-aware Weight Preserving
  ergnn    - Experience Replay GNN (CM sampler)
  cat      - Condensed Graph Memory (CaT) replay
  cosine   - COSINE (frozen backbone + prototype classifier)
  teen     - TEEN (COSINE + soft calibration of new-class prototypes)
  delome   - DeLoMe (Debiased Lossless Memory replay via gradient-matching condensation)

All methods share the same GCN backbone and evaluation pipeline.
"""

import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .gcn_backbone import GCNBackbone


# ======================================================================
# GEM utilities (adapted from CGLB, replacing quadprog with scipy)
# ======================================================================

def _store_grad(parameters, grads, grad_dims, tid):
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in parameters():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def _overwrite_grad(parameters, newgrad, grad_dims):
    cnt = 0
    for param in parameters():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            param.grad.data.copy_(newgrad[beg:en].contiguous().view(param.grad.data.size()))
        cnt += 1


def _project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """GEM gradient projection via QP solved with scipy."""
    from scipy.optimize import minimize
    memories_np = memories.detach().cpu().t().double().numpy()
    gradient_np = gradient.detach().cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]

    def objective(v):
        x = np.dot(v, memories_np) + gradient_np
        return 0.5 * np.dot(x, x)

    def jacobian(v):
        x = np.dot(v, memories_np) + gradient_np
        return np.dot(memories_np, x)

    bounds = [(0, None) for _ in range(t)]
    v0 = np.zeros(t)
    result = minimize(objective, v0, jac=jacobian, bounds=bounds, method='L-BFGS-B')
    x = np.dot(result.x, memories_np) + gradient_np
    gradient.copy_(torch.tensor(x, dtype=gradient.dtype).view(-1, 1))


# ======================================================================
# ER-GNN utilities (simplified CM sampler)
# ======================================================================

def _cm_sample(ids_per_cls_train, budget, feats, d=0.5):
    """Coverage Maximization sampler for ER-GNN."""
    budget_dist_compute = 1000
    vecs = feats.half()
    ids_selected = []
    for i, ids in enumerate(ids_per_cls_train):
        if len(ids) == 0:
            continue
        other_cls_ids = [j for j in range(len(ids_per_cls_train)) if j != i]
        sample_ids = ids if len(ids) < budget_dist_compute else random.sample(ids, budget_dist_compute)
        vecs_0 = vecs[sample_ids]

        dist_list = []
        for j in other_cls_ids:
            if len(ids_per_cls_train[j]) == 0:
                continue
            chosen = random.sample(ids_per_cls_train[j], min(budget_dist_compute, len(ids_per_cls_train[j])))
            vecs_1 = vecs[chosen]
            if len(chosen) < 26 or len(sample_ids) < 26:
                dist_list.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
            else:
                dist_list.append(torch.cdist(vecs_0, vecs_1))

        if dist_list:
            dist_ = torch.cat(dist_list, dim=-1)
            n_selected = (dist_ < d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
        else:
            rank = list(range(len(sample_ids)))

        current = rank[:min(budget, len(ids))]
        ids_selected.extend([sample_ids[j] for j in current])
    return ids_selected


# ======================================================================
# LwF distillation loss
# ======================================================================

def _multi_class_cross_entropy(logits, targets, T=2.0):
    outputs = torch.log_softmax(logits / T, dim=1)
    labels = torch.softmax(targets / T, dim=1)
    return -torch.mean(torch.sum(outputs * labels, dim=1))


# ======================================================================
# CaT (CGM) utilities - graph condensation via distribution matching
# ======================================================================

class _CaTEncoder(nn.Module):
    """Random encoder for CaT distribution matching (Linear layers wrapping GCNConv)."""

    def __init__(self, nin, nhid, nout, nlayers):
        super().__init__()
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(nin, nout))
        else:
            self.layers.append(nn.Linear(nin, nhid))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(nhid, nhid))
            self.layers.append(nn.Linear(nhid, nout))

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def encode_with_graph(self, x, edge_index, hop=1):
        """Encode with k-hop neighbor aggregation then MLP layers."""
        from torch_geometric.utils import add_self_loops, degree
        num_nodes = x.size(0)
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_sl
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        adj = torch.sparse_coo_tensor(
            torch.stack([col, row]), norm, (num_nodes, num_nodes)
        ).coalesce()

        h = x
        for _ in range(hop):
            h = torch.sparse.mm(adj, h)

        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        return h

    def encode_without_graph(self, x):
        """Encode without graph structure (pure MLP)."""
        h = x
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        return h


def _cat_condense_task(x, edge_index, train_ids, labels, classes, budget,
                       n_encoders=10, feat_lr=0.01, hid_dim=256, emb_dim=128,
                       n_layers=2, hop=1, device='cpu'):
    """
    CaT graph condensation: generate synthetic nodes for one task.

    Uses distribution matching with random encoders.
    Real nodes are encoded with graph structure; synthetic nodes without.
    """
    num_features = x.size(1)

    cls_counts = {}
    for nid in train_ids:
        c = labels[nid].item()
        cls_counts[c] = cls_counts.get(c, 0) + 1
    total_train = sum(cls_counts.values())

    budgets_per_cls = {}
    allocated = 0
    cls_list = sorted(classes)
    for i, c in enumerate(cls_list):
        ratio = cls_counts.get(c, 0) / total_train if total_train > 0 else 1.0 / len(cls_list)
        b = max(1, int(budget * ratio))
        budgets_per_cls[c] = b
        allocated += b
    gap = budget - allocated
    for i in range(abs(gap)):
        c = cls_list[i % len(cls_list)]
        budgets_per_cls[c] += 1 if gap > 0 else -1
        budgets_per_cls[c] = max(1, budgets_per_cls[c])

    labels_cond = []
    for c in cls_list:
        labels_cond.extend([c] * budgets_per_cls[c])
    labels_cond = torch.tensor(labels_cond, dtype=torch.long, device=device)
    n_cond = labels_cond.size(0)

    feat_cond = torch.nn.Parameter(torch.FloatTensor(n_cond, num_features).to(device))
    idx = 0
    for c in cls_list:
        cls_train = [nid for nid in train_ids if labels[nid].item() == c]
        bc = budgets_per_cls[c]
        if len(cls_train) > 0:
            sampled = [cls_train[random.randint(0, len(cls_train) - 1)] for _ in range(bc)]
            feat_cond.data[idx:idx + bc] = x[sampled].to(device)
        else:
            torch.nn.init.xavier_uniform_(feat_cond.data[idx:idx + bc])
        idx += bc

    opt_feat = torch.optim.Adam([feat_cond], lr=feat_lr)
    x_dev = x.to(device)
    edge_dev = edge_index.to(device)
    labels_dev = labels.to(device)

    cls_train_masks = {}
    for c in cls_list:
        mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
        for nid in train_ids:
            if labels_dev[nid].item() == c:
                mask[nid] = True
        cls_train_masks[c] = mask

    encoder = _CaTEncoder(num_features, hid_dim, emb_dim, n_layers).to(device)

    for _ in range(n_encoders):
        encoder.initialize()
        encoder.eval()

        with torch.no_grad():
            emb_real = encoder.encode_with_graph(x_dev, edge_dev, hop=hop)
            emb_real = F.normalize(emb_real, dim=1)

        emb_cond = encoder.encode_without_graph(feat_cond)
        emb_cond = F.normalize(emb_cond, dim=1)

        loss = torch.tensor(0., device=device)
        for c in cls_list:
            real_emb_c = emb_real[cls_train_masks[c]]
            cond_emb_c = emb_cond[labels_cond == c]
            if real_emb_c.size(0) == 0 or cond_emb_c.size(0) == 0:
                continue
            dist = torch.mean(real_emb_c, dim=0) - torch.mean(cond_emb_c, dim=0)
            loss = loss + torch.sum(dist ** 2)

        opt_feat.zero_grad()
        loss.backward()
        opt_feat.step()

    return feat_cond.detach(), labels_cond


# ======================================================================
# DeLoMe utilities - gradient-matching graph condensation
# ======================================================================

class _DeLoMeSGC(nn.Module):
    """Lightweight SGC used inside DeLoMe condensation (k-hop aggregation + linear)."""

    def __init__(self, input_dim, hidden_dim, output_dim, k=2):
        super().__init__()
        self.k = k
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def _sgc_agg(self, x, edge_index, num_nodes):
        from torch_geometric.utils import add_self_loops, degree
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_sl
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        adj = torch.sparse_coo_tensor(
            torch.stack([col, row]), norm, (num_nodes, num_nodes)
        ).coalesce()

        for _ in range(self.k):
            x = torch.sparse.mm(adj, x)
        return x

    def forward(self, x, edge_index, num_nodes=None):
        if num_nodes is None:
            num_nodes = x.size(0)
        h = self._sgc_agg(x, edge_index, num_nodes)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

    def reset_params(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


def _delome_condense_task(x, edge_index, train_ids, labels, classes, budget,
                          num_all_classes, hidden_dim=256, sgc_k=2,
                          condense_epochs=600, feat_lr=1e-4, device='cpu'):
    """
    DeLoMe graph condensation via gradient matching.

    For each class in *classes*, learn up to *budget* synthetic node features
    so that the gradient on synthetic data matches the gradient on real data.
    """
    num_features = x.size(1)

    cls_train = {}
    for c in classes:
        cls_train[c] = [nid for nid in train_ids if labels[nid].item() == c]

    n_syn_per_cls = {}
    labels_syn_list = []
    syn_class_ranges = {}
    for c in sorted(classes):
        n = min(budget, len(cls_train.get(c, [])))
        n = max(n, 1)
        n_syn_per_cls[c] = n
        syn_class_ranges[c] = (len(labels_syn_list), len(labels_syn_list) + n)
        labels_syn_list.extend([c] * n)

    labels_syn = torch.tensor(labels_syn_list, dtype=torch.long, device=device)
    n_syn = labels_syn.size(0)

    feat_syn = nn.Parameter(torch.FloatTensor(n_syn, num_features).to(device))
    idx_init = []
    for c in sorted(classes):
        ids = cls_train.get(c, [])
        n = n_syn_per_cls[c]
        if ids:
            sampled = [ids[random.randint(0, len(ids) - 1)] for _ in range(n)]
            idx_init.extend(sampled)
        else:
            idx_init.extend([0] * n)
    feat_syn.data.copy_(x[idx_init].to(device))

    opt_feat = torch.optim.Adam([feat_syn], lr=feat_lr)
    x_dev = x.to(device)
    edge_dev = edge_index.to(device)
    labels_dev = labels.to(device)

    self_loop_edge = torch.arange(n_syn, device=device).unsqueeze(0).repeat(2, 1)

    for epoch in range(condense_epochs):
        sgc = _DeLoMeSGC(num_features, hidden_dim, num_all_classes, k=sgc_k).to(device)
        sgc.reset_params()
        sgc_params = list(sgc.parameters())
        sgc.train()

        for c in sorted(classes):
            ids_c = cls_train.get(c, [])
            if not ids_c:
                continue
            ids_t = torch.tensor(ids_c, dtype=torch.long, device=device)

            sgc.zero_grad()
            logits_real = sgc(x_dev, edge_dev)
            loss_real = F.cross_entropy(logits_real[ids_t], labels_dev[ids_t])
            gw_real = torch.autograd.grad(loss_real, sgc_params)
            gw_real = [g.detach().clone() for g in gw_real]

            s, e = syn_class_ranges[c]
            logits_syn = sgc(feat_syn, self_loop_edge, num_nodes=n_syn)
            loss_syn = F.cross_entropy(logits_syn[s:e], labels_syn[s:e])
            gw_syn = torch.autograd.grad(loss_syn, sgc_params, create_graph=True)

            coeff = n_syn_per_cls[c] / max(n_syn_per_cls.values())
            loss_match = coeff * _grad_match_loss(gw_syn, gw_real)

            opt_feat.zero_grad()
            loss_match.backward()
            opt_feat.step()

    return feat_syn.detach(), labels_syn


def _grad_match_loss(gw_syn, gw_real):
    """MSE distance between synthetic and real gradients."""
    vec_real = torch.cat([g.reshape(-1) for g in gw_real])
    vec_syn = torch.cat([g.reshape(-1) for g in gw_syn])
    return torch.sum((vec_syn - vec_real) ** 2)


# ======================================================================
# BaselineCL
# ======================================================================

class BaselineCL:
    """Unified baseline continual learning framework."""

    METHODS = ['bare', 'joint', 'ewc', 'mas', 'lwf', 'gem', 'twp', 'ergnn', 'cat',
               'cosine', 'teen', 'delome']

    def __init__(self, task_loader, config, device, method='bare'):
        assert method in self.METHODS, f"Unknown method: {method}"
        self.task_loader = task_loader
        self.config = config
        self.device = device
        self.method = method

        self.input_dim = task_loader.data.x.size(1)
        self.num_classes = len(task_loader.all_classes)
        self.hidden_dim = config.get('gcn_hidden_dim', 256)
        self.num_layers = config.get('gcn_layers', 2)
        self.gcn_dropout = config.get('gcn_dropout', 0.0)
        self.epochs = config.get('baseline_epochs', 200)
        self.lr = config.get('baseline_lr', 0.005)
        self.weight_decay = config.get('baseline_weight_decay', 5e-4)

        self._init_method_params()

    def _init_method_params(self):
        cfg = self.config
        self.ewc_lambda = cfg.get('ewc_lambda', 10000.0)
        self.mas_lambda = cfg.get('mas_lambda', 1.0)
        self.lwf_lambda = cfg.get('lwf_lambda', 1.0)
        self.lwf_T = cfg.get('lwf_temperature', 2.0)
        self.gem_margin = cfg.get('gem_margin', 0.5)
        self.gem_n_memories = cfg.get('gem_n_memories', 100)
        self.twp_lambda_l = cfg.get('twp_lambda_l', 10000.0)
        self.twp_beta = cfg.get('twp_beta', 0.01)
        self.ergnn_budget = cfg.get('ergnn_budget', 100)
        self.ergnn_d = cfg.get('ergnn_d', 0.5)
        self.ergnn_replay_weight = cfg.get('ergnn_replay_weight', 1.0)
        self.cat_budget = cfg.get('cat_budget', 20)
        self.cat_n_encoders = cfg.get('cat_n_encoders', 10)
        self.cat_feat_lr = cfg.get('cat_feat_lr', 0.01)
        self.cat_hid_dim = cfg.get('cat_hid_dim', 256)
        self.cat_emb_dim = cfg.get('cat_emb_dim', 128)
        self.cat_n_layers = cfg.get('cat_n_layers', 2)
        self.cat_hop = cfg.get('cat_hop', 1)
        self.cosine_T = cfg.get('cosine_T', 16.0)
        self.teen_T = cfg.get('teen_T', 16.0)
        self.teen_softmax_t = cfg.get('teen_softmax_t', 20.0)
        self.teen_shift_weight = cfg.get('teen_shift_weight', 0.1)
        self.delome_budget = cfg.get('delome_budget', 60)
        self.delome_tro = cfg.get('delome_tro', 1.0)
        self.delome_condense_epochs = cfg.get('delome_condense_epochs', 600)
        self.delome_feat_lr = cfg.get('delome_feat_lr', 1e-4)
        self.delome_sgc_k = cfg.get('delome_sgc_k', 2)
        self.delome_hidden_dim = cfg.get('delome_hidden_dim', 256)
        self.delome_cls_balance = cfg.get('delome_cls_balance', 'logita')

    def _create_model(self):
        return GCNBackbone(
            self.input_dim, self.hidden_dim, self.num_classes,
            num_layers=self.num_layers, dropout=self.gcn_dropout
        ).to(self.device)

    # ==================== Main Loop ====================

    def fit(self, trial):
        num_sessions = self.task_loader.sessions

        if self.method in ('cosine', 'teen'):
            net = GCNBackbone(
                self.input_dim, self.hidden_dim, self.hidden_dim,
                num_layers=self.num_layers, dropout=self.gcn_dropout
            ).to(self.device)
        else:
            net = self._create_model()

        state = self._init_state(net, num_sessions)

        if self.method in ('cosine', 'teen'):
            opt = torch.optim.Adam(
                list(net.parameters()) + list(state['fc'].parameters()),
                lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        acc_matrix = []
        joint_acc_history = []
        joint_macro_history = []

        for session_id in range(num_sessions):
            (curr_classes, all_classes,
             subgraph, joint_subgraph,
             train_loader, valid_loader,
             test_loader_joint) = self.task_loader.get_task(session_id)

            train_idx = self.task_loader.train_idx_per_task[session_id]
            valid_idx = self.task_loader.valid_idx_per_task[session_id]

            print(f"\n{'='*60}")
            print(f"[{self.method.upper()}] Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}")
            print(f"{'='*60}")

            if self.method in ('cosine', 'teen'):
                state['seen_classes'] = list(all_classes)

            if self.method == 'joint':
                net = self._create_model()
                opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            if self.method in ('cosine', 'teen') and session_id > 0:
                self._cosine_proto_update(net, state, session_id, subgraph, train_idx)
            else:
                self._train_session(net, opt, session_id, subgraph, train_idx,
                                    all_classes, state)

            # Isolated Tests
            print(f"\n--- Isolated Tests (Session {session_id}) ---")
            acc_row = []
            for tid in range(session_id + 1):
                iso_subgraph = self.task_loader.subgraph_isolated[tid]
                test_idx = self.task_loader.test_idx_per_task[tid]
                task_classes = self.task_loader.class_splits[tid]

                if not test_idx:
                    acc_row.append(0.0)
                    continue

                res = self._evaluate(net, iso_subgraph, test_idx, state=state)
                acc_row.append(res['acc'])
                print(f"  Task {tid} (classes {task_classes}): "
                      f"Acc={res['acc']:.4f} ({res['correct']}/{res['total']})")
            acc_matrix.append(acc_row)

            # Joint Test
            print(f"\n--- Joint Test (Session {session_id}) ---")
            test_idx_joint = self.task_loader.test_idx_joint[session_id]
            joint_res = self._evaluate(net, joint_subgraph, test_idx_joint, state=state)
            joint_acc_history.append(joint_res['acc'])
            joint_macro_history.append(joint_res['macro_acc'])
            print(f"  Acc={joint_res['acc']:.4f} Macro={joint_res['macro_acc']:.4f} "
                  f"({joint_res['correct']}/{joint_res['total']})")

            self._post_session(net, opt, session_id, subgraph, train_idx,
                               all_classes, state)

        # Final Summary
        print(f"\n{'='*60}")
        print(f"[{self.method.upper()}] FINAL RESULTS")
        print(f"{'='*60}")
        self._print_cl_matrix("CL Accuracy Matrix", acc_matrix, num_sessions)
        print(f"\nJoint Accuracy (micro): " + ", ".join(
            [f"S{i}={joint_acc_history[i]:.4f}" for i in range(num_sessions)]))
        print(f"Joint Accuracy (macro): " + ", ".join(
            [f"S{i}={joint_macro_history[i]:.4f}" for i in range(num_sessions)]))

        return {
            'acc_matrix': acc_matrix,
            'joint_acc': joint_acc_history,
            'joint_macro_acc': joint_macro_history,
        }

    # ==================== State Init ====================

    def _init_state(self, net, num_sessions):
        state = {}
        if self.method == 'ewc':
            state['fisher'] = {}
            state['optpar'] = {}
        elif self.method == 'mas':
            state['fisher'] = []
            state['optpar'] = []
        elif self.method == 'lwf':
            state['prev_model'] = None
        elif self.method == 'gem':
            state['memory_train_idx'] = {}
            state['memory_subgraphs'] = {}
            grad_dims = [p.data.numel() for p in net.parameters()]
            state['grad_dims'] = grad_dims
            state['grads'] = torch.zeros(sum(grad_dims), num_sessions).to(self.device)
        elif self.method == 'twp':
            state['fisher_loss'] = {}
            state['fisher_att'] = {}
            state['optpar'] = {}
        elif self.method == 'ergnn':
            state['buffer_node_ids'] = []
            state['aux_data'] = None
        elif self.method == 'cat':
            state['memory_bank'] = []
        elif self.method in ('cosine', 'teen'):
            state['fc'] = nn.Linear(self.hidden_dim, self.num_classes).to(self.device)
        elif self.method == 'delome':
            state['memory_bank'] = []
            state['cond_num'] = {}
            state['adjustments'] = None
        return state

    # ==================== COSINE/TEEN prototype update ====================

    @torch.no_grad()
    def _cosine_proto_update(self, net, state, session_id, subgraph, train_idx):
        """Update fc prototypes for new-session classes (COSINE/TEEN, session > 0)."""
        net.eval()
        fc = state['fc']
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        all_nodes_set = set(subgraph['all_nodes'])

        h = net(x, edge_index)

        task_classes = self.task_loader.class_splits[session_id]
        for c in task_classes:
            cls_train = [n for n in train_idx if n in all_nodes_set
                         and labels[n].item() == c]
            if cls_train:
                cls_ids = torch.tensor(cls_train, dtype=torch.long, device=self.device)
                fc.weight.data[c] = h[cls_ids].mean(dim=0)

        if self.method == 'teen':
            base_class = len(self.task_loader.class_splits[0])
            class_src = task_classes[0]
            class_dst = task_classes[-1] + 1
            base_protos = F.normalize(fc.weight.data[:base_class].clone(), p=2, dim=-1)
            cur_protos = F.normalize(fc.weight.data[class_src:class_dst].clone(), p=2, dim=-1)
            weights = torch.mm(cur_protos, base_protos.T) * self.teen_softmax_t
            norm_weights = torch.softmax(weights, dim=1)
            delta = torch.matmul(norm_weights, base_protos)
            delta = F.normalize(delta, p=2, dim=-1)
            updated = (1 - self.teen_shift_weight) * cur_protos + self.teen_shift_weight * delta
            fc.weight.data[class_src:class_dst] = updated

    # ==================== Training ====================

    def _train_session(self, net, opt, session_id, subgraph, train_idx,
                       all_classes, state):
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        all_nodes_set = set(subgraph['all_nodes'])

        # Use global node IDs directly (x/labels are full-graph, logits indexed by global ID)
        train_ids = torch.tensor([n for n in train_idx if n in all_nodes_set],
                                 dtype=torch.long, device=self.device)

        if self.method in ('cosine', 'teen'):
            fc = state['fc']
            net.train()
            fc.train()
            for epoch in range(self.epochs):
                opt.zero_grad()
                h = net(x, edge_index)
                logits = fc(h)
                loss = F.cross_entropy(logits[train_ids], labels[train_ids])
                loss.backward()
                opt.step()
            return
            
        if self.method == 'joint':
            all_train_idx = []
            for sid in range(session_id + 1):
                all_train_idx.extend(self.task_loader.train_idx_per_task[sid])
            train_ids = torch.tensor([n for n in all_train_idx if n in all_nodes_set],
                                     dtype=torch.long, device=self.device)
            train_labels = labels[train_ids]
            class_counts = torch.bincount(train_labels, minlength=self.num_classes).float().clamp(min=1)
            class_weight = (1.0 / class_counts)
            class_weight = class_weight * (self.num_classes / class_weight.sum())

        if self.method == 'lwf' and state['prev_model'] is not None:
            prev_model = state['prev_model']
            prev_model.eval()
            with torch.no_grad():
                prev_logits = prev_model(x, edge_index)
        else:
            prev_logits = None

        net.train()
        for epoch in range(self.epochs):
            opt.zero_grad()
            logits = net(x, edge_index)
            if self.method == 'joint':
                loss = F.cross_entropy(logits[train_ids], labels[train_ids],
                                       weight=class_weight)
            else:
                loss = F.cross_entropy(logits[train_ids], labels[train_ids])

            # Method-specific regularization
            if self.method == 'ewc':
                for tt in range(session_id):
                    for i, p in enumerate(net.parameters()):
                        fisher_val = state['fisher'][tt][i]
                        optpar_val = state['optpar'][tt][i]
                        loss += (self.ewc_lambda / 2.0) * (fisher_val * (p - optpar_val).pow(2)).sum()

            elif self.method == 'mas':
                if session_id > 0 and len(state['fisher']) > 0:
                    for i, p in enumerate(net.parameters()):
                        loss += (self.mas_lambda / 2.0) * (state['fisher'][i] * (p - state['optpar'][i]).pow(2)).sum()

            elif self.method == 'lwf' and prev_logits is not None:
                dist_loss = _multi_class_cross_entropy(
                    logits[train_ids], prev_logits[train_ids], self.lwf_T)
                loss += self.lwf_lambda * dist_loss

            elif self.method == 'twp':
                loss.backward(retain_graph=True)
                grad_norm = sum(p.grad.data.clone().norm(p=1) for p in net.parameters() if p.grad is not None)

                for tt in range(session_id):
                    for i, p in enumerate(net.parameters()):
                        fl = state['fisher_loss'][tt][i]
                        fa = state['fisher_att'][tt][i]
                        reg = (self.twp_lambda_l * fl + self.twp_lambda_l * fa)
                        reg = reg * (p - state['optpar'][tt][i]).pow(2)
                        loss += reg.sum()

                loss = loss + self.twp_beta * grad_norm
                opt.zero_grad()

            elif self.method == 'gem' and session_id > 0:
                loss.backward()
                _store_grad(net.parameters, state['grads'], state['grad_dims'], session_id)

                for old_t in range(session_id):
                    old_sub = state['memory_subgraphs'][old_t]
                    old_train_ids = state['memory_train_idx'][old_t]
                    ox = old_sub['x'].to(self.device)
                    oe = old_sub['edge_index'].to(self.device)
                    oy = old_sub['y'].to(self.device)
                    old_ids = torch.tensor(old_train_ids, dtype=torch.long, device=self.device)
                    net.zero_grad()
                    old_logits = net(ox, oe)
                    old_loss = F.cross_entropy(old_logits[old_ids], oy[old_ids])
                    old_loss.backward()
                    _store_grad(net.parameters, state['grads'], state['grad_dims'], old_t)

                indx = torch.arange(session_id, device=self.device)
                dotp = torch.mm(state['grads'][:, session_id].unsqueeze(0),
                                state['grads'][:, indx])
                if (dotp < 0).any():
                    _project2cone2(state['grads'][:, session_id].unsqueeze(1),
                                   state['grads'][:, indx], self.gem_margin)
                    _overwrite_grad(net.parameters, state['grads'][:, session_id], state['grad_dims'])
                opt.step()
                continue

            elif self.method == 'ergnn' and session_id > 0 and state['aux_data'] is not None:
                aux_x, aux_edge, aux_y, aux_ids = state['aux_data']
                aux_logits = net(aux_x, aux_edge)
                loss_aux = F.cross_entropy(aux_logits[aux_ids], aux_y[aux_ids])
                loss = loss + self.ergnn_replay_weight * loss_aux

            elif self.method == 'cat' and session_id > 0 and len(state['memory_bank']) > 0:
                all_feat = torch.cat([m['feat'] for m in state['memory_bank']], dim=0)
                all_lab = torch.cat([m['labels'] for m in state['memory_bank']], dim=0)
                n_cond = all_feat.size(0)
                self_loops = torch.arange(n_cond, device=self.device).unsqueeze(0).repeat(2, 1)
                cond_logits = net(all_feat, self_loops)
                loss_replay = F.cross_entropy(cond_logits, all_lab)
                loss = loss + loss_replay

            elif self.method == 'delome':
                if state['adjustments'] is not None:
                    loss = F.cross_entropy(
                        logits[train_ids] + state['adjustments'], labels[train_ids])
                if session_id > 0 and len(state['memory_bank']) > 0:
                    for mem in state['memory_bank']:
                        mem_feat = mem['feat']
                        mem_lab = mem['labels']
                        n_mem = mem_feat.size(0)
                        sl = torch.arange(n_mem, device=self.device).unsqueeze(0).repeat(2, 1)
                        mem_logits = net(mem_feat, sl)
                        if state['adjustments'] is not None:
                            loss_aux = F.cross_entropy(
                                mem_logits + state['adjustments'], mem_lab)
                        else:
                            loss_aux = F.cross_entropy(mem_logits, mem_lab)
                        loss = loss + loss_aux

            loss.backward()
            opt.step()

    # ==================== Post-Session Hooks ====================

    def _post_session(self, net, opt, session_id, subgraph, train_idx,
                      all_classes, state):
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        all_nodes_set = set(subgraph['all_nodes'])

        # Global node IDs for indexing into full-graph logits/labels
        train_ids = torch.tensor([n for n in train_idx if n in all_nodes_set],
                                 dtype=torch.long, device=self.device)

        if self.method == 'ewc':
            net.zero_grad()
            logits = net(x, edge_index)
            loss = F.cross_entropy(logits[train_ids], labels[train_ids])
            loss.backward()
            state['fisher'][session_id] = []
            state['optpar'][session_id] = []
            for p in net.parameters():
                state['optpar'][session_id].append(p.data.clone())
                state['fisher'][session_id].append(p.grad.data.clone().pow(2))

        elif self.method == 'mas':
            net.zero_grad()
            logits = net(x, edge_index)
            logits_sq = logits[train_ids].pow(2)
            loss = logits_sq.mean()
            loss.backward()
            new_fisher = []
            new_optpar = []
            for p in net.parameters():
                new_optpar.append(p.data.clone())
                new_fisher.append(p.grad.data.clone().pow(2))
            if len(state['fisher']) > 0:
                for i in range(len(new_fisher)):
                    state['fisher'][i] = (state['fisher'][i] * session_id + new_fisher[i]) / (session_id + 1)
                state['optpar'] = new_optpar
            else:
                state['fisher'] = new_fisher
                state['optpar'] = new_optpar

        elif self.method == 'lwf':
            state['prev_model'] = copy.deepcopy(net)

        elif self.method == 'gem':
            mem_ids = train_idx[:self.gem_n_memories]
            state['memory_train_idx'][session_id] = mem_ids
            state['memory_subgraphs'][session_id] = subgraph

        elif self.method == 'twp':
            net.zero_grad()
            logits = net(x, edge_index)
            loss = F.cross_entropy(logits[train_ids], labels[train_ids])
            loss.backward(retain_graph=True)
            state['fisher_loss'][session_id] = []
            state['fisher_att'][session_id] = []
            state['optpar'][session_id] = []
            for p in net.parameters():
                state['optpar'][session_id].append(p.data.clone())
                state['fisher_loss'][session_id].append(p.grad.data.clone().pow(2))
            net.zero_grad()
            logits2 = net(x, edge_index)
            l2 = logits2[train_ids].norm()
            l2.backward()
            for p in net.parameters():
                state['fisher_att'][session_id].append(p.grad.data.clone().pow(2))

        elif self.method == 'ergnn':
            task_classes = self.task_loader.class_splits[session_id]
            ids_per_cls_train = []
            for c in task_classes:
                cls_ids = [n for n in train_idx if n in all_nodes_set
                           and labels[n].item() == c]
                ids_per_cls_train.append(cls_ids)

            feats = x
            sampled = _cm_sample(ids_per_cls_train, self.ergnn_budget, feats, self.ergnn_d)
            state['buffer_node_ids'].extend(sampled)

            if len(state['buffer_node_ids']) > 0:
                all_buf_train = list(state['buffer_node_ids'])
                cumul_classes = []
                for sid in range(session_id + 1):
                    cumul_classes.extend(self.task_loader.class_splits[sid])
                buf_subgraph = self.task_loader._create_task_subgraph(
                    list(set(cumul_classes)),
                    allowed_external_classes=list(set(cumul_classes))
                )
                buf_nodes_set = set(buf_subgraph['all_nodes'])
                buf_ids = [n for n in all_buf_train if n in buf_nodes_set]
                if buf_ids:
                    buf_ids_t = torch.tensor(buf_ids, dtype=torch.long, device=self.device)
                    state['aux_data'] = (
                        buf_subgraph['x'].to(self.device),
                        buf_subgraph['edge_index'].to(self.device),
                        buf_subgraph['y'].to(self.device),
                        buf_ids_t,
                    )

        elif self.method in ('cosine', 'teen'):
            if session_id == 0:
                for p in net.parameters():
                    p.requires_grad = False
                    
        elif self.method == 'cat':
            task_classes = self.task_loader.class_splits[session_id]
            sub_all_nodes = subgraph['all_nodes']
            global_to_local = {g: l for l, g in enumerate(sub_all_nodes)}

            local_x = x[sub_all_nodes]
            local_labels = labels[sub_all_nodes]

            src, dst = subgraph['edge_index']
            local_src = torch.tensor([global_to_local[s.item()] for s in src],
                                     dtype=torch.long)
            local_dst = torch.tensor([global_to_local[d.item()] for d in dst],
                                     dtype=torch.long)
            local_edge_index = torch.stack([local_src, local_dst], dim=0)

            local_train_ids = [global_to_local[n] for n in train_idx
                               if n in global_to_local]

            feat_cond, labels_cond = _cat_condense_task(
                local_x, local_edge_index, local_train_ids, local_labels,
                task_classes, self.cat_budget,
                n_encoders=self.cat_n_encoders,
                feat_lr=self.cat_feat_lr,
                hid_dim=self.cat_hid_dim,
                emb_dim=self.cat_emb_dim,
                n_layers=self.cat_n_layers,
                hop=self.cat_hop,
                device=self.device,
            )
            state['memory_bank'].append({
                'feat': feat_cond,
                'labels': labels_cond,
            })
            print(f"  [CaT] Condensed task {session_id}: "
                  f"{feat_cond.size(0)} synthetic nodes for classes {task_classes}")

        elif self.method == 'delome':
            task_classes = self.task_loader.class_splits[session_id]
            sub_all_nodes = subgraph['all_nodes']
            global_to_local = {g: l for l, g in enumerate(sub_all_nodes)}

            local_x = x[sub_all_nodes]
            local_labels = labels[sub_all_nodes]

            src, dst = subgraph['edge_index']
            local_src = torch.tensor([global_to_local[s.item()] for s in src],
                                     dtype=torch.long)
            local_dst = torch.tensor([global_to_local[d.item()] for d in dst],
                                     dtype=torch.long)
            local_edge_index = torch.stack([local_src, local_dst], dim=0)

            local_train_ids = [global_to_local[n] for n in train_idx
                               if n in global_to_local]

            feat_cond, labels_cond = _delome_condense_task(
                local_x, local_edge_index, local_train_ids, local_labels,
                task_classes, self.delome_budget,
                num_all_classes=self.num_classes,
                hidden_dim=self.delome_hidden_dim,
                sgc_k=self.delome_sgc_k,
                condense_epochs=self.delome_condense_epochs,
                feat_lr=self.delome_feat_lr,
                device=self.device,
            )
            state['memory_bank'].append({
                'feat': feat_cond,
                'labels': labels_cond,
            })
            print(f"  [DeLoMe] Condensed task {session_id}: "
                  f"{feat_cond.size(0)} synthetic nodes for classes {task_classes}")

            for c in task_classes:
                c_count = (labels_cond == c).sum().item()
                if c_count > 0:
                    state['cond_num'][c] = c_count

            if self.delome_cls_balance == 'logita':
                all_nodes_set = set(sub_all_nodes)
                current_counts = []
                for c in task_classes:
                    c_ids = [n for n in train_idx if n in all_nodes_set
                             and labels[n].item() == c]
                    current_counts.append(len(c_ids))

                freq_list = []
                for sid in range(session_id + 1):
                    for c in self.task_loader.class_splits[sid]:
                        if sid < session_id:
                            freq_list.append(state['cond_num'].get(c, 1))
                        else:
                            idx_in_task = task_classes.index(c)
                            freq_list.append(current_counts[idx_in_task])
                freq_arr = np.array(freq_list, dtype=np.float64)
                freq_arr = freq_arr / freq_arr.sum()
                adj_vals = np.log(freq_arr ** self.delome_tro + 1e-12)
                full_adj = torch.zeros(self.num_classes, device=self.device)
                all_seen = []
                for sid in range(session_id + 1):
                    all_seen.extend(self.task_loader.class_splits[sid])
                for i, c in enumerate(all_seen):
                    if c < self.num_classes:
                        full_adj[c] = adj_vals[i]
                state['adjustments'] = full_adj

    # ==================== Evaluation ====================

    @torch.no_grad()
    def _evaluate(self, net, subgraph, test_idx, state=None):
        net.eval()
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y']
        all_nodes_set = set(subgraph['all_nodes'])

        if self.method in ('cosine', 'teen') and state is not None:
            fc = state['fc']
            h = net(x, edge_index)
            T = self.cosine_T if self.method == 'cosine' else self.teen_T
            logits = F.linear(
                F.normalize(h, p=2, dim=-1),
                F.normalize(fc.weight, p=2, dim=-1)
            ) * T
            seen_classes = state.get('seen_classes')
            if seen_classes is not None:
                unseen_classes = [c for c in self.task_loader.all_classes if c not in seen_classes]
                if unseen_classes:
                    logits[:, unseen_classes] = -1e9
        else:
            logits = net(x, edge_index)
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
        macro_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0

        net.train()
        return {
            'acc': acc, 'macro_acc': macro_acc,
            'correct': correct, 'total': total,
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
