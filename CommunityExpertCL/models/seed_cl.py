"""
SEED (Selection of Experts with Ensemble of Distributions) adapted for graph
node classification.

Paper: "Divide and not forget: Ensemble of selectively trained experts in
continual learning", ICLR 2024.  Original code works on image streams
(ResNet + per-class GMM).  This port keeps SEED's algorithmic core
(multi-expert backbone pool + per-class GMM + KL-based expert selection
+ Bayes aggregation) and replaces the image-specific pieces
(ResNet, DataLoader, horizontal flip, BatchNorm freezing) with
graph-native equivalents (GCN backbone, subgraph forward pass).

Algorithmic mapping from original SEED to this graph port:

  SEED (image)                         | Graph port
  ------------------------------------ | ------------------------------------
  ResNet backbone per expert           | GCN backbone per expert
  images DataLoader                    | full-subgraph forward on task_loader
  target -= model.task_offset          | per-expert global->local class map
  features[:, bb_num] (stacked)        | per-expert features (computed lazy)
  torch.flip(images) augmentation      | removed (no direct graph analog)
  freeze BatchNorm2d in finetune       | removed (GCN has no BN2d)
  GMM on 64-d ResNet features          | GMM on feat_dim GCN embeddings
  MSE feature-KD during finetune       | same (MSE on GCN embeddings)
  KL divergence over new-class MVNs    | same (via GMM -> MultivariateNormal)

Expected training / inference flow (per session t):

  1.  If len(experts) < max_experts:
        spawn a new expert, train its GCN backbone + local classifier with CE
        on the task-t training nodes.
  2.  Else:
        pick the expert with highest mean KL over new-class distributions
        (most "over-confident separation" on the new task) and finetune it
        with CE + MSE feature distillation against its own frozen copy.
  3.  For every existing expert, fit one GMM per new class on its feature
        embeddings (drifted old-class GMMs stay untouched, matching SEED).
  4.  Evaluation: run every expert on the eval subgraph, Bayes-aggregate
        softmax(log_prob / tau) across experts.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from .gcn_backbone import GCNBackbone
from .seed_gmm import GaussianMixture


# ----------------------------------------------------------------------
# Expert module
# ----------------------------------------------------------------------


class SeedExpert(nn.Module):
    """Single SEED expert: GCN feature extractor + temporary linear head.

    The fc head is only used during CE training / finetuning.  For GMM
    fitting and inference we call :meth:`features` and discard the head.
    """

    def __init__(self, input_dim, hidden_dim, feat_dim, num_local_classes,
                 num_layers=2, dropout=0.0):
        super().__init__()
        self.backbone = GCNBackbone(
            input_dim, hidden_dim, feat_dim,
            num_layers=num_layers, dropout=dropout,
        )
        self.fc = nn.Linear(feat_dim, num_local_classes)

    def forward(self, x, edge_index, return_features=False):
        feat = self.backbone(x, edge_index)
        logits = self.fc(feat)
        if return_features:
            return logits, feat
        return logits

    def features(self, x, edge_index):
        return self.backbone(x, edge_index)

    def reset_fc(self, num_local_classes, device):
        self.fc = nn.Linear(self.fc.in_features, num_local_classes).to(device)


# ----------------------------------------------------------------------
# Main SEED-graph class
# ----------------------------------------------------------------------


def softmax_temperature(x, dim, tau=1.0):
    return torch.softmax(x / tau, dim=dim)


class SEEDCL:
    """SEED adapted for graph continual learning (node classification)."""

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device

        self.input_dim = task_loader.data.x.size(1)
        self.num_classes = max(task_loader.all_classes) + 1

        # Backbone geometry.  Default to 64-d features -- same as SEED's
        # ResNet-32 output.  Graph datasets often have small per-class
        # sample sizes, so a compact embedding makes full-covariance GMMs
        # tractable when the user enables them.
        self.hidden_dim = int(config.get('seed_hidden_dim', 64))
        self.feat_dim = int(config.get('seed_feat_dim', self.hidden_dim))
        self.num_layers = int(config.get('gcn_layers', 2))
        self.dropout = float(config.get('gcn_dropout', 0.0))

        # Training hyperparameters
        self.nepochs = int(config.get('seed_nepochs',
                                      config.get('baseline_epochs', 200)))
        self.ftepochs = int(config.get('seed_ftepochs', 100))
        self.lr = float(config.get('seed_lr',
                                   config.get('baseline_lr', 0.005)))
        self.wd = float(config.get('seed_weight_decay',
                                   config.get('baseline_weight_decay', 5e-4)))
        self.ftwd = float(config.get('seed_ftwd', 0.0))
        self.clipgrad = float(config.get('seed_clipgrad', 10000.0))

        # SEED specifics
        self.max_experts = int(config.get('seed_max_experts', 5))
        self.gmms = int(config.get('seed_gmms', 1))
        self.alpha = float(config.get('seed_alpha', 0.99))
        self.tau = float(config.get('seed_tau', 3.0))
        self.use_multivariate = bool(config.get('seed_use_multivariate', False))
        self.use_nmc = bool(config.get('seed_use_nmc', False))
        self.initialization_strategy = str(
            config.get('seed_init_strategy', 'first'))
        assert self.initialization_strategy in ('first', 'random')

        # State: ModuleList so buffers move with .to(), plus per-expert
        # class -> GMM dict.
        self.experts = nn.ModuleList().to(device)
        self.experts_distributions: list[dict[int, GaussianMixture]] = []

    # ------------------------------------------------------------------
    # Expert factory
    # ------------------------------------------------------------------

    def _create_expert(self, num_local_classes):
        return SeedExpert(
            self.input_dim, self.hidden_dim, self.feat_dim,
            num_local_classes, self.num_layers, self.dropout,
        ).to(self.device)

    def _spawn_expert(self, num_local_classes, t):
        """Spawn a new expert.  If init_strategy='first' and t>0, copy
        expert 0's backbone weights (SEED's default, speeds up warm-up).
        """
        expert = self._create_expert(num_local_classes)
        if self.initialization_strategy == 'first' and t > 0 and len(self.experts) > 0:
            expert.backbone.load_state_dict(
                copy.deepcopy(self.experts[0].backbone.state_dict()))
        return expert

    # ------------------------------------------------------------------
    # Subgraph unpacking helper
    # ------------------------------------------------------------------

    def _unpack(self, subgraph, train_idx, curr_class_set):
        """Move tensors to device and compute filtered train-id tensor."""
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        all_nodes_set = set(subgraph['all_nodes'])

        train_ids_list = [
            n for n in train_idx
            if n in all_nodes_set and labels[n].item() in curr_class_set
        ]
        train_ids = torch.tensor(train_ids_list, dtype=torch.long,
                                 device=self.device)
        return x, edge_index, labels, all_nodes_set, train_ids, train_ids_list

    # ------------------------------------------------------------------
    # Training: new expert (session_id < max_experts)
    # ------------------------------------------------------------------

    def _train_expert(self, session_id, curr_classes, subgraph, train_idx):
        sorted_classes = sorted(curr_classes)
        g2l = {c: i for i, c in enumerate(sorted_classes)}

        expert = self._spawn_expert(len(sorted_classes), session_id)
        self.experts.append(expert)

        curr_class_set = set(sorted_classes)
        x, edge_index, labels, _, train_ids, train_ids_list = self._unpack(
            subgraph, train_idx, curr_class_set)

        if train_ids.numel() == 0:
            print(f"  [SEED] session {session_id}: no training nodes, skip")
            return

        local_labels = torch.tensor(
            [g2l[labels[n].item()] for n in train_ids_list],
            dtype=torch.long, device=self.device)

        optimizer = torch.optim.SGD(
            expert.parameters(), lr=self.lr,
            weight_decay=self.wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[
                int(self.nepochs * 0.3),
                int(self.nepochs * 0.6),
                int(self.nepochs * 0.8),
            ], gamma=0.1)

        pbar = tqdm(range(self.nepochs), desc=f"S{session_id} SEED-train")
        for _epoch in pbar:
            expert.train()
            optimizer.zero_grad()
            logits = expert(x, edge_index)
            loss = F.cross_entropy(logits[train_ids], local_labels)
            loss.backward()
            if self.clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(
                    expert.parameters(), self.clipgrad)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

    # ------------------------------------------------------------------
    # Finetuning: existing expert (session_id >= max_experts)
    # ------------------------------------------------------------------

    def _finetune_expert(self, bb_to_ft, session_id, curr_classes,
                         subgraph, train_idx):
        sorted_classes = sorted(curr_classes)
        g2l = {c: i for i, c in enumerate(sorted_classes)}

        expert = self.experts[bb_to_ft]

        # Snapshot the pre-finetune backbone for feature distillation
        old_backbone = copy.deepcopy(expert.backbone)
        for p in old_backbone.parameters():
            p.requires_grad = False
        old_backbone.eval()

        # Swap the classifier head to match current task's cardinality
        expert.reset_fc(len(sorted_classes), self.device)

        curr_class_set = set(sorted_classes)
        x, edge_index, labels, _, train_ids, train_ids_list = self._unpack(
            subgraph, train_idx, curr_class_set)

        if train_ids.numel() == 0:
            print(f"  [SEED] session {session_id}: no training nodes, skip")
            return

        local_labels = torch.tensor(
            [g2l[labels[n].item()] for n in train_ids_list],
            dtype=torch.long, device=self.device)

        optimizer = torch.optim.SGD(
            expert.parameters(), lr=self.lr,
            weight_decay=self.ftwd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[
                int(self.ftepochs * 0.3),
                int(self.ftepochs * 0.6),
                int(self.ftepochs * 0.8),
            ], gamma=0.1)

        pbar = tqdm(range(self.ftepochs),
                    desc=f"S{session_id} SEED-ft(E{bb_to_ft})")
        for _epoch in pbar:
            expert.train()
            optimizer.zero_grad()

            with torch.no_grad():
                old_feat = old_backbone(x, edge_index)
            logits, feat = expert(x, edge_index, return_features=True)

            ce_loss = F.cross_entropy(logits[train_ids], local_labels)
            kd_loss = F.mse_loss(feat, old_feat)
            loss = (1.0 - self.alpha) * ce_loss + self.alpha * kd_loss

            loss.backward()
            if self.clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(
                    expert.parameters(), self.clipgrad)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(ce=f'{ce_loss.item():.4f}',
                             kd=f'{kd_loss.item():.4f}')

    # ------------------------------------------------------------------
    # Choose expert to finetune (max-KL overlap heuristic from SEED)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _choose_backbone_to_finetune(self, curr_classes, subgraph, train_idx):
        """Fit temporary new-class GMMs for every expert, pick the one whose
        new-class distributions have the highest mean pairwise KL.

        In SEED this is argmax over ``expert_overlap`` (mean of the KL
        matrix over new classes): intuitively the expert whose feature
        space spreads the new classes the furthest is the safest to
        update, since its old-class representation still differs the
        most from the new task's optimum.
        """
        overlap = torch.zeros(len(self.experts))
        for bb_num in range(len(self.experts)):
            tmp = self._fit_gmms_for_classes(
                bb_num, curr_classes, subgraph, train_idx)
            class_list = sorted(tmp.keys())
            if len(class_list) < 2:
                overlap[bb_num] = 0.0
                continue

            mvns = []
            for c in class_list:
                try:
                    mvns.append(self._gmm_to_mvn(tmp[c]))
                except Exception:
                    mvns.append(None)

            kls = []
            for i, m_i in enumerate(mvns):
                if m_i is None:
                    continue
                for j, m_j in enumerate(mvns):
                    if i == j or m_j is None:
                        continue
                    try:
                        kls.append(float(
                            torch.distributions.kl_divergence(m_i, m_j)))
                    except Exception:
                        pass
            overlap[bb_num] = sum(kls) / max(len(kls), 1)

        bb_to_ft = int(overlap.argmax().item())
        print(f"  [SEED] expert overlap (higher=more spread): "
              f"{[round(v, 3) for v in overlap.tolist()]} "
              f"-> finetune expert {bb_to_ft}")
        return bb_to_ft

    @staticmethod
    def _gmm_to_mvn(gmm):
        """Convert a 1-component GMM into a torch MultivariateNormal.

        Works for both ``diag`` and ``full`` covariance variants.  Raises
        if covariance is singular (the caller should catch and skip).
        """
        mu = gmm.mu.data
        if mu.dim() == 3:
            mu = mu[0, 0]
        elif mu.dim() == 2:
            mu = mu[0]
        var = gmm.var.data
        if gmm.covariance_type == 'full':
            cov = var[0, 0]
        else:
            cov = torch.diag(var[0, 0])
        # Small ridge for numerical stability
        cov = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device)
        return MultivariateNormal(mu, covariance_matrix=cov)

    # ------------------------------------------------------------------
    # GMM fitting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _fit_gmms_for_classes(self, bb_num, classes, subgraph, train_idx):
        """Fit one GMM per class in ``classes`` using expert ``bb_num``.

        Returns a dict ``{global_class_id: GaussianMixture}``.
        """
        expert = self.experts[bb_num]
        expert.eval()

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        all_nodes_set = set(subgraph['all_nodes'])

        feats = expert.features(x, edge_index).detach()

        cov_type = 'full' if self.use_multivariate else 'diag'
        out: dict[int, GaussianMixture] = {}

        for c in classes:
            cls_ids = [n for n in train_idx
                       if n in all_nodes_set and labels[n].item() == c]
            if len(cls_ids) < 2:
                continue

            cls_t = torch.tensor(cls_ids, dtype=torch.long, device=self.device)
            cls_feat = feats[cls_t]

            gmm = self._try_fit_gmm(cls_feat, cov_type)
            if gmm is None and cov_type == 'full':
                # Full cov singular: degrade to diag
                gmm = self._try_fit_gmm(cls_feat, 'diag')
            if gmm is None:
                continue

            if gmm.mu.data.dim() == 2:
                gmm.mu.data = gmm.mu.data.unsqueeze(1)
            out[c] = gmm

        return out

    def _try_fit_gmm(self, cls_feat, cov_type):
        """Fit a GMM with progressive eps back-off on singular covariance.

        ``torch.linalg.cholesky`` raises ``RuntimeError`` (or a subclass
        ``torch._C._LinAlgError`` on newer torch versions) when the
        covariance matrix is singular -- we widen eps and retry.
        """
        eps = 1e-6
        for _ in range(5):
            try:
                gmm = GaussianMixture(
                    self.gmms, cls_feat.shape[1],
                    covariance_type=cov_type, eps=eps,
                ).to(self.device)
                gmm.fit(cls_feat, delta=1e-3, n_iter=100)
                return gmm
            except RuntimeError:
                eps *= 10.0
            except Exception:
                return None
        return None

    def _create_distributions(self, curr_classes, subgraph, train_idx):
        """Fit GMMs for ``curr_classes`` using every existing expert.

        Old-class GMMs are intentionally left untouched -- this mirrors
        SEED's expansion-only design.  Drift after finetuning is handled
        implicitly by the MSE feature-KD during ``_finetune_expert``.
        """
        for bb_num in range(len(self.experts)):
            new = self._fit_gmms_for_classes(
                bb_num, curr_classes, subgraph, train_idx)
            self.experts_distributions[bb_num].update(new)

    # ------------------------------------------------------------------
    # Inference (Bayes aggregation over experts)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, subgraph, test_idx):
        all_nodes_set = set(subgraph['all_nodes'])
        labels = subgraph['y']
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)

        num_experts = len(self.experts)
        if num_experts == 0:
            return {'acc': 0.0, 'macro_acc': 0.0, 'correct': 0, 'total': 0}

        N = x.size(0)

        # Gather features per expert (single forward per expert).
        feats_list = []
        for expert in self.experts:
            expert.eval()
            feats_list.append(expert.features(x, edge_index))

        log_probs = torch.full(
            (N, num_experts, self.num_classes), -1e8, device=self.device)
        # valid[e, c] = True if expert e has a GMM for class c
        valid = torch.zeros(
            num_experts, self.num_classes, dtype=torch.bool,
            device=self.device)

        for e, dists in enumerate(self.experts_distributions):
            feat_e = feats_list[e]
            for c, gmm in dists.items():
                if c < 0 or c >= self.num_classes:
                    continue
                try:
                    if self.use_nmc:
                        mu = gmm.mu.data
                        mu = mu[0, 0] if mu.dim() == 3 else mu[0]
                        lp = -torch.cdist(
                            feat_e, mu.unsqueeze(0)).squeeze(-1)
                    else:
                        lp = gmm.score_samples(feat_e)
                    log_probs[:, e, c] = lp
                    valid[e, c] = True
                except Exception:
                    continue

        # Softmax over classes per (node, expert), then sum over experts
        # weighted by number of experts that actually have that class.
        lp_soft = softmax_temperature(log_probs, dim=2, tau=self.tau)
        denom = valid.sum(dim=0).clamp(min=1).float().unsqueeze(0)  # (1, C)
        conf = lp_soft.sum(dim=1) / denom
        preds = conf.argmax(dim=1).cpu()

        correct = 0
        total = 0
        per_class_correct: dict[int, int] = {}
        per_class_total: dict[int, int] = {}
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
            c_tot = per_class_total[c]
            per_class_acc.append(per_class_correct.get(c, 0) / c_tot
                                 if c_tot > 0 else 0.0)
        macro_acc = (sum(per_class_acc) / len(per_class_acc)
                     if per_class_acc else 0.0)

        return {'acc': acc, 'macro_acc': macro_acc,
                'correct': correct, 'total': total}

    # ------------------------------------------------------------------
    # Outer loop
    # ------------------------------------------------------------------

    def fit(self, trial):
        num_sessions = self.task_loader.sessions
        acc_matrix = []
        joint_acc_history = []
        joint_macro_history = []

        for session_id in range(num_sessions):
            (curr_classes, all_classes, subgraph, joint_subgraph,
             _, _, _) = self.task_loader.get_task(session_id)
            train_idx = self.task_loader.train_idx_per_task[session_id]
            valid_idx = self.task_loader.valid_idx_per_task[session_id]

            print(f"\n{'=' * 60}")
            print(f"[SEED] Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}")
            print(f"{'=' * 60}")

            if len(self.experts) < self.max_experts:
                print(f"  [SEED] training new expert "
                      f"(id={len(self.experts)})")
                self._train_expert(
                    session_id, curr_classes, subgraph, train_idx)
                self.experts_distributions.append({})
            else:
                bb_to_ft = self._choose_backbone_to_finetune(
                    curr_classes, subgraph, train_idx)
                self._finetune_expert(
                    bb_to_ft, session_id, curr_classes, subgraph, train_idx)

            print(f"  [SEED] fitting GMMs for "
                  f"{len(curr_classes)} new classes "
                  f"x {len(self.experts)} experts")
            self._create_distributions(curr_classes, subgraph, train_idx)

            print(f"\n--- Isolated Tests (Session {session_id}) ---")
            acc_row = []
            for tid in range(session_id + 1):
                iso_subgraph = self.task_loader.subgraph_isolated[tid]
                test_idx = self.task_loader.test_idx_per_task[tid]
                task_classes = self.task_loader.class_splits[tid]
                if not test_idx:
                    acc_row.append(0.0)
                    continue
                res = self._evaluate(iso_subgraph, test_idx)
                acc_row.append(res['acc'])
                print(f"  Task {tid} (classes {task_classes}): "
                      f"Acc={res['acc']:.4f} "
                      f"({res['correct']}/{res['total']})")
            acc_matrix.append(acc_row)

            print(f"\n--- Joint Test (Session {session_id}) ---")
            test_idx_joint = self.task_loader.test_idx_joint[session_id]
            joint_res = self._evaluate(joint_subgraph, test_idx_joint)
            joint_acc_history.append(joint_res['acc'])
            joint_macro_history.append(joint_res['macro_acc'])
            print(f"  Acc={joint_res['acc']:.4f} "
                  f"Macro={joint_res['macro_acc']:.4f} "
                  f"({joint_res['correct']}/{joint_res['total']})")

        print(f"\n{'=' * 60}")
        print("[SEED] FINAL RESULTS")
        print(f"{'=' * 60}")
        self._print_cl_matrix("CL Accuracy Matrix", acc_matrix, num_sessions)
        print(f"\nJoint Accuracy (micro): " + ", ".join(
            [f"S{i}={joint_acc_history[i]:.4f}"
             for i in range(num_sessions)]))
        print(f"Joint Accuracy (macro): " + ", ".join(
            [f"S{i}={joint_macro_history[i]:.4f}"
             for i in range(num_sessions)]))

        return {
            'acc_matrix': acc_matrix,
            'joint_acc': joint_acc_history,
            'joint_macro_acc': joint_macro_history,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
