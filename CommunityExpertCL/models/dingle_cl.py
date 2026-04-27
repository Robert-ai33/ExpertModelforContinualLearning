"""
DINGLE adapted to the CommunityExpertCL graph class-IL pipeline.

Original paper: "DINGLE: Disentangled Knowledge Distillation for Graph
Class-Incremental Learning" (the public release lives at
``DINGLE-main/{main.py, dis_model.py, utils_metric.py}``).

Algorithmic mapping from the original DINGLE script to this port:

  DINGLE (FSCIL on graphs)               | CommunityExpertCL (class-IL)
  -------------------------------------- | ------------------------------------
  pre-saved ``dataset/{name}_stream/``   | TaskLoader subgraphs (cumulative
  base / streaming npz files               + isolated, per-session)
  argparse + per-config yaml             | configs/config_dingle.yaml
  GCN encoder + DisModel + Classifier    | GCNBackbone + DisModel + Classifier
  classifier rebuilt every iteration     | single global Linear(num_classes),
                                          unseen logits masked at eval
  fresh GCN per streaming session        | continuous GCN across sessions
                                          (toggleable via config)
  per-session optim_generator with        | three optimizer groups, the
  duplicate-key dict bug (only            | three DisModel sub-modules each
  decoder updated)                        | get their own param group
  ``combine_all_params`` reads .pkl      | ``combine_dis_state_dicts`` reads
  files from disk                         | an in-memory list of state-dicts
  per-class top-30 closest-to-mean,      | same selection rule, capped by
  hard-fail on small classes              | ``len(node_ids)``
  evaluation = per-task acc + AP/AF      | per-task isolated tests (acc_matrix)
                                          + joint test on cumulative test set

Design choices that depart from the original (each one is a *necessary*
class-IL adaptation, never a re-derivation of the method):

1. Single global classifier ``Linear(z_c_dim, num_classes)``. The original
   recreates ``Classifier(z_c.shape[1], class_num)`` inside every training
   iteration with a brand-new ``nn.Linear``, so its CE gradient never
   updates classifier weights at all. We fix this latent bug by training
   one classifier; at eval we mask logits for not-yet-seen classes to
   ``-1e9`` (same convention as ``cosine`` / ``teen`` baselines).
2. ``DisModel`` parameter groups in ``optim_generator``: original uses a
   *single* dict literal with three ``'params'`` keys, which under Python
   semantics keeps only the last entry (``decoder``). We use a list of
   three dicts so ``encoder_c`` / ``encoder_s`` / ``decoder`` all train --
   without this fix the disentanglement encoders never get optimized.
3. GCN backbone persists across sessions by default. Original re-inits a
   fresh ``get_model(...)`` at every streaming session, which is sensible
   only in FSCIL (where each session has a few-shot novel class-set and
   the buffer carries old-class signal). Under our class-IL splits each
   session has plenty of training nodes, so a fresh GCN would catastrophic
   ally forget all previously consolidated representations. Toggleable via
   ``dingle_fresh_gcn_per_session`` (default ``False``).
4. ``ft_lr`` is honoured: original defines it but never references it in
   ``main.py`` (both base and streaming use ``args.lr``). We expose
   ``dingle_lr`` (base) and ``dingle_ft_lr`` (streaming) separately so
   users can tune the FT phase if needed.
5. CE is restricted to the *current session's class subset*. Because we
   keep one global classifier across sessions (item 1), a softmax CE over
   all ``num_classes`` would push past-class rows down whenever the
   current session's labels never include them, collapsing past-class
   accuracy to 0. The original DINGLE never sees this collapse because it
   rebuilds ``Classifier(.,. class_num)`` per iteration -- so past rows
   stay at their (random) re-init values rather than being suppressed.
   Restricting CE to ``curr_classes`` is the literal class-IL analogue:
   past-class rows are simply frozen at their last-trained values, the
   exact behaviour the disentangled-feature KD was designed to protect.

Everything else (DisModel architecture, the four reconstruction-style
losses + GAN, the buffer-of-closest-nodes mechanism, the linear-decay
weighted teacher fusion of ``encoder_c`` with simple-mean of the other
sub-modules, and the reverse-sign style-KD) stays byte-faithful to the
original release.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .gcn_backbone import GCNBackbone
from .dingle_modules import (
    DisModel,
    combine_dis_state_dicts,
    get_closest_nodes,
)
from utils import compute_ap_af, print_cl_matrix


class DINGLECL:
    """DINGLE adapted for graph class-incremental node classification."""

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device

        self.input_dim = task_loader.data.x.size(1)
        self.num_classes = max(task_loader.all_classes) + 1

        # ---- Backbone geometry (matches DINGLE config_*_stream.yaml fields) ----
        # ``hidden_feature``  in DINGLE = GCN output dim = DisModel input dim
        # ``semantic_feature``           = MLP intermediate dim inside DisModel
        # ``output_dim``                 = z_c / z_s dim
        self.hidden_feature = int(
            config.get('dingle_hidden_feature',
                       config.get('gcn_hidden_dim', 256))
        )
        self.semantic_feature = int(config.get('dingle_semantic_feature', 64))
        self.output_dim = int(config.get('dingle_output_dim', 64))
        self.enc_layer = int(config.get('dingle_enc_layer', 2))
        self.dec_layer = int(config.get('dingle_dec_layer', 2))
        self.dis_layer = int(config.get('dingle_dis_layer', 2))
        self.batch_norm = bool(config.get('dingle_batch_norm', False))

        # ---- GCN backbone (consistent with the rest of the project) ----
        self.gcn_layers = int(config.get('gcn_layers', 2))
        self.gcn_dropout = float(config.get('gcn_dropout', 0.0))
        self.gcn_hidden_dim = int(config.get('gcn_hidden_dim',
                                             self.hidden_feature))
        # When dingle_hidden_feature differs from gcn_hidden_dim, the GCN
        # output is set to dingle_hidden_feature so DisModel input matches.

        # ---- Training schedule ----
        self.epochs = int(config.get('dingle_epochs',
                                     config.get('baseline_epochs', 200)))
        self.ft_epochs = int(config.get('dingle_ft_epochs', self.epochs))
        self.lr = float(config.get('dingle_lr',
                                   config.get('baseline_lr', 1e-3)))
        self.ft_lr = float(config.get('dingle_ft_lr', self.lr))
        self.weight_decay = float(
            config.get('dingle_weight_decay',
                       config.get('baseline_weight_decay', 5e-4))
        )

        # ---- Loss weights (original defines loss_ratio but never uses it) ----
        self.dis_loss_weight = float(config.get('dingle_dis_loss_weight', 1.0))
        self.kd_loss_weight = float(config.get('dingle_kd_loss_weight', 1.0))

        # ---- Buffer ----
        self.buffer_per_class = int(config.get('dingle_buffer_per_class', 30))

        # ---- Adaptation flags ----
        self.fresh_gcn_per_session = bool(
            config.get('dingle_fresh_gcn_per_session', False)
        )
        self.fresh_classifier_per_session = bool(
            config.get('dingle_fresh_classifier_per_session', False)
        )

        # ---- Persistent state across sessions ----
        self.gcn = None  # type: ignore[assignment]
        self.classifier = None  # type: ignore[assignment]
        self.history_dis_states: List[dict] = []  # CPU state-dicts, oldest first
        # Buffer is a (M, hidden_feature) tensor of FROZEN GCN embeddings
        # captured at the time each session ended (matches the original's
        # heterogeneous buffer behaviour where session-k entries reflect
        # the GCN weights at end-of-session-k).
        self.buffer = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_gcn(self):
        return GCNBackbone(
            self.input_dim,
            self.gcn_hidden_dim,
            self.hidden_feature,
            num_layers=self.gcn_layers,
            dropout=self.gcn_dropout,
        ).to(self.device)

    def _build_dis(self):
        return DisModel(
            self.hidden_feature,
            self.semantic_feature,
            self.output_dim,
            self.enc_layer,
            self.dec_layer,
            self.dis_layer,
            batch_norm=self.batch_norm,
        ).to(self.device)

    def _build_classifier(self):
        return nn.Linear(self.output_dim, self.num_classes).to(self.device)

    # ------------------------------------------------------------------
    # Subgraph unpacking
    # ------------------------------------------------------------------

    def _unpack(self, subgraph, train_idx, label_filter=None):
        """Move tensors to device and produce a filtered train-id tensor.

        ``label_filter`` (optional set of class ids) restricts the train
        nodes to a specific class subset (e.g. the current task's classes).
        """
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)
        labels = subgraph['y'].to(self.device)
        all_nodes_set = set(subgraph['all_nodes'])

        if label_filter is None:
            keep = [n for n in train_idx if n in all_nodes_set]
        else:
            keep = [
                n for n in train_idx
                if n in all_nodes_set and labels[n].item() in label_filter
            ]
        train_ids = torch.tensor(keep, dtype=torch.long, device=self.device)
        return x, edge_index, labels, all_nodes_set, train_ids, keep

    # ------------------------------------------------------------------
    # Per-session training
    # ------------------------------------------------------------------

    def _train_session(self, session_id, curr_classes, subgraph, train_idx):
        """One DINGLE training session.

        Replicates the original DINGLE's ``base session`` (session 0) and
        ``streaming session`` (session j>0) loops in a single function,
        gated by the presence of historical DisModel snapshots.
        """
        is_streaming = len(self.history_dis_states) > 0

        # ---- Backbone / classifier instantiation ----
        if self.gcn is None or (is_streaming and self.fresh_gcn_per_session):
            self.gcn = self._build_gcn()
        if self.classifier is None or (
            is_streaming and self.fresh_classifier_per_session
        ):
            self.classifier = self._build_classifier()

        # ---- DisModel student / teacher ----
        student_dis = self._build_dis()
        teacher_dis = None
        if is_streaming:
            teacher_state = combine_dis_state_dicts(self.history_dis_states)
            # Move CPU state-dict tensors back to device before loading.
            teacher_state_dev = {
                k: v.to(self.device) for k, v in teacher_state.items()
            }
            teacher_dis = self._build_dis()
            teacher_dis.load_state_dict(teacher_state_dev)
            student_dis.load_state_dict(teacher_state_dev)
            teacher_dis.eval()
            for p in teacher_dis.parameters():
                p.requires_grad = False

        # ---- Optimizers (three groups, original-faithful) ----
        # Original DINGLE bug: optim_generator was a single dict with three
        # 'params' keys (so only ``decoder`` got stepped). Here we use a
        # list of three dicts so all three sub-modules update.
        # We also include the classifier in optimizer_encoder so CE actually
        # trains it (original creates a fresh classifier every iteration).
        n_epochs = self.ft_epochs if is_streaming else self.epochs
        cur_lr = self.ft_lr if is_streaming else self.lr

        optimizer_encoder = torch.optim.Adam(
            list(self.gcn.parameters()) + list(self.classifier.parameters()),
            lr=cur_lr, weight_decay=self.weight_decay,
        )
        optim_generator = torch.optim.Adam([
            {'params': student_dis.encoder_c.parameters()},
            {'params': student_dis.encoder_s.parameters()},
            {'params': student_dis.decoder.parameters()},
        ], lr=cur_lr)
        optim_discriminator = torch.optim.Adam(
            student_dis.discriminator.parameters(), lr=cur_lr,
        )

        # ---- Data tensors ----
        x, edge_index, labels, _, train_ids, train_ids_list = self._unpack(
            subgraph, train_idx, label_filter=set(curr_classes),
        )
        if train_ids.numel() == 0:
            print(f"  [DINGLE] session {session_id}: no training nodes, skip")
            return student_dis

        # ---- CE class scoping (current-session classes only) ----
        # Original DINGLE rebuilds a classifier of size ``class_num`` every
        # session, but never optimizes it -- so past-class rows in the
        # original are simply random.  In our class-IL port we keep a single
        # global classifier, which means a naive softmax CE over all
        # ``num_classes`` would *systematically suppress* past-class rows
        # whenever the current session's labels never include them (the
        # CE gradient on non-target rows is "push the inner product down").
        # That collapse is what was causing past-task accuracy to fall to
        # exactly 0.0.  We therefore restrict CE to the current session's
        # class subset, which is the literal class-IL analogue of the
        # original "per-session classifier" idea: past-class rows stay
        # frozen at their last-trained values.
        cur_cls_sorted = sorted(curr_classes)
        cur_cls_t = torch.tensor(cur_cls_sorted, dtype=torch.long,
                                 device=self.device)
        g2l_local = {c: i for i, c in enumerate(cur_cls_sorted)}
        local_labels = torch.tensor(
            [g2l_local[labels[n].item()] for n in train_ids_list],
            dtype=torch.long, device=self.device,
        )

        # ---- KD inputs (fixed buffer for the entire session) ----
        kd_active = is_streaming and self.buffer is not None and self.buffer.numel() > 0
        if kd_active:
            buffer_emb = self.buffer.to(self.device)
        else:
            buffer_emb = None

        # ---- Training loop ----
        desc = f"S{session_id} DINGLE-{'ft' if is_streaming else 'base'}"
        pbar = tqdm(range(n_epochs), desc=desc)
        for _epoch in pbar:
            self.gcn.train()
            student_dis.train()
            self.classifier.train()

            optimizer_encoder.zero_grad()
            optim_generator.zero_grad()
            optim_discriminator.zero_grad()

            embeddings = self.gcn(x, edge_index)
            (z_c, _z_s,
             loss_gen, loss_rec_x, loss_rec_c, loss_rec_s,
             loss_disc) = student_dis(embeddings)

            logits = self.classifier(z_c)
            # CE only over current-session class columns; ``local_labels``
            # are the train nodes' labels remapped to local [0, |cur|).
            ce_loss = F.cross_entropy(
                logits[train_ids][:, cur_cls_t], local_labels,
            )

            dis_loss = (loss_gen + loss_rec_x + loss_rec_c + loss_rec_s
                        + loss_disc)

            kd_loss = z_c.new_zeros(())
            if kd_active:
                z_c_st = student_dis.encoder_c(buffer_emb)
                z_s_st = student_dis.encoder_s(buffer_emb)
                with torch.no_grad():
                    z_c_te = teacher_dis.encoder_c(buffer_emb)
                    z_s_te = teacher_dis.encoder_s(buffer_emb)
                # Reverse-sign KD: pull content closer to teacher,
                # push style farther from teacher.
                kd_loss = (F.mse_loss(z_c_st, z_c_te)
                           - F.mse_loss(z_s_st, z_s_te))

            total_loss = (ce_loss
                          + self.dis_loss_weight * dis_loss
                          + self.kd_loss_weight * kd_loss)

            total_loss.backward()
            optimizer_encoder.step()
            optim_generator.step()
            optim_discriminator.step()

            pbar.set_postfix(
                ce=f'{ce_loss.item():.3f}',
                dis=f'{dis_loss.item():.3f}',
                kd=(f'{kd_loss.item():.3f}' if kd_active else 'n/a'),
            )

        # ---- End-of-session: extend buffer with closest-to-mean nodes ----
        with torch.no_grad():
            self.gcn.eval()
            embeddings_eval = self.gcn(x, edge_index)
            cls_to_train_ids = {}
            for n in train_ids.tolist():
                c = labels[n].item()
                cls_to_train_ids.setdefault(c, []).append(n)
            _, new_embs = get_closest_nodes(
                embeddings_eval, cls_to_train_ids,
                per_class_budget=self.buffer_per_class,
            )

        if new_embs.numel() > 0:
            new_embs_cpu = new_embs.detach().cpu()
            if self.buffer is None:
                self.buffer = new_embs_cpu
            else:
                self.buffer = torch.cat([self.buffer, new_embs_cpu], dim=0)

        return student_dis

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, dis_model, subgraph, test_idx, seen_classes):
        """Evaluate on a (sub)graph using the current (gcn, dis_model, fc).

        ``seen_classes`` is the set of class ids visible up to the current
        session; logits for any other class are masked to -1e9 to keep
        argmax inside the seen-class space (matches the convention used by
        ``cosine`` / ``teen``).
        """
        all_nodes_set = set(subgraph['all_nodes'])
        labels = subgraph['y']
        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)

        self.gcn.eval()
        dis_model.eval()
        self.classifier.eval()

        embeddings = self.gcn(x, edge_index)
        z_c = dis_model.encoder_c(embeddings)
        logits = self.classifier(z_c)

        if seen_classes is not None:
            unseen = [c for c in range(self.num_classes) if c not in seen_classes]
            if unseen:
                logits[:, unseen] = -1e9

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
            c_tot = per_class_total[c]
            per_class_acc.append(
                per_class_correct.get(c, 0) / c_tot if c_tot > 0 else 0.0
            )
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

        # The "current" DisModel that drives evaluation after each session.
        current_dis = None

        for session_id in range(num_sessions):
            (curr_classes, all_classes, subgraph, joint_subgraph,
             _, _, _) = self.task_loader.get_task(session_id)
            train_idx = self.task_loader.train_idx_per_task[session_id]
            valid_idx = self.task_loader.valid_idx_per_task[session_id]

            print(f"\n{'=' * 60}")
            print(f"[DINGLE] Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"Train: {len(train_idx)}, Valid: {len(valid_idx)}")
            print(f"{'=' * 60}")

            current_dis = self._train_session(
                session_id, curr_classes, subgraph, train_idx,
            )

            # Snapshot the just-trained student DisModel for future fusion.
            cpu_state = {
                k: v.detach().cpu().clone()
                for k, v in current_dis.state_dict().items()
            }
            self.history_dis_states.append(cpu_state)

            seen_classes_set = set(all_classes)

            # ---- Per-task tests on the cumulative subgraph (CGLB) ----
            # Each cell (k, t) of the CL accuracy matrix evaluates the model
            # on subgraph_per_task[k] (the cumulative subgraph used during
            # session-k training) restricted to test_idx_per_task[t].
            print(f"\n--- Per-Task Tests (Session {session_id}) ---")
            eval_subgraph = self.task_loader.subgraph_per_task[session_id]
            acc_row = []
            for tid in range(session_id + 1):
                test_idx = self.task_loader.test_idx_per_task[tid]
                task_classes = self.task_loader.class_splits[tid]
                if not test_idx:
                    acc_row.append(0.0)
                    continue
                res = self._evaluate(current_dis, eval_subgraph,
                                     test_idx, seen_classes_set)
                acc_row.append(res['acc'])
                print(f"  Task {tid} (classes {task_classes}): "
                      f"Acc={res['acc']:.4f} "
                      f"({res['correct']}/{res['total']})")
            acc_matrix.append(acc_row)

            # ---- Joint test on cumulative test set ----
            print(f"\n--- Joint Test (Session {session_id}) ---")
            test_idx_joint = self.task_loader.test_idx_joint[session_id]
            joint_res = self._evaluate(
                current_dis, joint_subgraph,
                test_idx_joint, seen_classes_set,
            )
            joint_acc_history.append(joint_res['acc'])
            joint_macro_history.append(joint_res['macro_acc'])
            print(f"  Acc={joint_res['acc']:.4f} "
                  f"Macro={joint_res['macro_acc']:.4f} "
                  f"({joint_res['correct']}/{joint_res['total']})")

        print(f"\n{'=' * 60}")
        print("[DINGLE] FINAL RESULTS")
        print(f"{'=' * 60}")
        print_cl_matrix("CL Accuracy Matrix", acc_matrix, num_sessions)
        ap_history, af, final_ap = compute_ap_af(acc_matrix)
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
            'ap_history': ap_history,
            'af': af,
            'final_ap': final_ap,
        }
