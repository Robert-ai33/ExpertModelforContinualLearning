"""
DINGLE building blocks for the CommunityExpertCL graph class-IL pipeline.

Faithfully ported from the original DINGLE-main release:

- ``MLP``                : same as ``dis_model.MLP``
- ``DisModel``           : same as ``dis_model.DisModel`` (E_c / E_s / D / Disc
                            with the four reconstruction-style losses + GAN)
- ``get_closest_nodes``  : same as ``utils_metric.get_closest_nodes`` but the
                            per-class budget is configurable and capped by the
                            actual sample count (the original hard-codes 30,
                            which crashes on classes with <30 train nodes -- a
                            real concern under our class-IL split ratios)
- ``combine_dis_state_dicts``
                          : same teacher-fusion math as
                            ``utils_metric.combine_all_params`` but operates
                            on an in-memory list of state-dicts so we don't
                            need the disk-backed ``model_pkl/*.pkl`` cache.

All other deviations from the original (single global classifier, continuous
GCN backbone across sessions, fixed optimizer parameter groups) live in
``dingle_cl.DINGLECL``; this file stays intentionally close to the original
DisModel implementation.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# MLP -- byte-identical to DINGLE-main/dis_model.py:MLP
# ----------------------------------------------------------------------


class MLP(nn.Module):
    """Multi-layer perceptron used by every component of ``DisModel``.

    Faithful to the original: ``num_layers`` counts layers excluding the
    input. ``num_layers == 1`` collapses to a single ``nn.Linear``. With
    ``batch_norm=True`` a ``BatchNorm1d`` is inserted between every two
    linear layers (after the linear, before ReLU). The original DINGLE
    main script always passes ``batch_norm=False`` for graph experiments.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 batch_norm=True):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.norm_layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            for _ in range(num_layers - 1):
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        for layer in range(self.num_layers - 1):
            if self.batch_norm:
                x = F.relu(self.norm_layers[layer](self.layers[layer](x)))
            else:
                x = F.relu(self.layers[layer](x))
        return self.layers[self.num_layers - 1](x)


# ----------------------------------------------------------------------
# DisModel -- faithful to DINGLE-main/dis_model.py:DisModel
# ----------------------------------------------------------------------


class DisModel(nn.Module):
    """Disentangled encoder/decoder + GAN block from DINGLE.

    The forward pass returns the four reconstruction-style losses and the
    discriminator loss exactly as in the original implementation:

    - ``L_rec_x``      : MSE(D(z_c, z_s), h)         -- reconstruction
    - ``L_rec_c``      : MSE(E_c(D(z_c, s_noise)), z_c)
                          -- content invariant to a random style
    - ``L_rec_s``      : MSE(E_s(D(z_c, s_noise)), s_noise)
                          -- style channel actually carries the random style
    - ``L_disc``       : BCE for the GAN discriminator on (h, D(z_c, s))
    - ``L_gen``        : BCE for the generator (decoder) trying to fool D

    The four shapes (``hidden_dim``, ``output_dim`` etc.) follow the original
    DINGLE config naming where ``input_dim`` is the GNN embedding dim
    (``hidden_feature``), ``hidden_dim`` the MLP intermediate dim
    (``semantic_feature``) and ``output_dim`` the ``z_c`` / ``z_s`` dim
    (``output_dim`` in DINGLE configs).
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 enc_layer, dec_layer, dis_layer, batch_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder_c = MLP(input_dim, hidden_dim, output_dim,
                             enc_layer, batch_norm)
        self.encoder_s = MLP(input_dim, hidden_dim, output_dim,
                             enc_layer, batch_norm)
        self.decoder = MLP(2 * output_dim, hidden_dim, input_dim,
                           dec_layer, batch_norm)
        self.discriminator = MLP(input_dim, hidden_dim, 1,
                                 dis_layer, batch_norm)

    def forward(self, x):
        X_c = self.encoder_c(x)
        X_s = self.encoder_s(x)
        X_rec = self.decoder(torch.cat([X_c, X_s], dim=1))

        s = torch.randn_like(X_s).to(X_s.device)
        dec_noise = self.decoder(torch.cat([X_c, s], dim=1))
        X_c_rec = self.encoder_c(dec_noise)
        X_s_rec = self.encoder_s(dec_noise)

        loss_rec_x = F.mse_loss(X_rec, x)
        loss_rec_c = F.mse_loss(X_c_rec, X_c)
        loss_rec_s = F.mse_loss(X_s_rec, s)

        out_pos = self.discriminator(x)
        out_neg = self.discriminator(dec_noise.detach())
        ones = torch.ones_like(out_pos).to(out_pos.device)
        zeros = torch.zeros_like(out_pos).to(out_neg.device)

        out_disc = torch.cat([out_pos, out_neg])
        labels = torch.cat([ones, zeros])
        loss_discriminator = F.binary_cross_entropy_with_logits(out_disc, labels)

        out = self.discriminator(dec_noise)
        loss_generator = F.binary_cross_entropy_with_logits(out, ones)

        return (X_c, X_s,
                loss_generator, loss_rec_x, loss_rec_c, loss_rec_s,
                loss_discriminator)


# ----------------------------------------------------------------------
# Buffer sampling -- adapted from utils_metric.get_closest_nodes
# ----------------------------------------------------------------------


def get_closest_nodes(node_embeddings, label_dict, per_class_budget=30):
    """Per-class top-k nodes closest to the class-mean embedding.

    Args:
        node_embeddings: ``(N, D)`` tensor (typically GCN output).
        label_dict: mapping ``{global_class_id: [global_node_ids...]}``.
        per_class_budget: max nodes kept per class (original DINGLE = 30).
            We additionally cap by ``len(node_ids)`` so classes with fewer
            training samples (common under our class-IL split) don't crash.

    Returns:
        sample_ids:   sorted list of selected global node ids.
        sample_embs:  ``(sum_k, D)`` tensor of the corresponding embeddings.
    """
    device = node_embeddings.device
    sample_ids = []
    for label, node_ids in label_dict.items():
        if len(node_ids) == 0:
            continue
        ids_t = torch.as_tensor(node_ids, dtype=torch.long, device=device)
        cls_emb = node_embeddings[ids_t]
        mean_emb = cls_emb.mean(dim=0, keepdim=True)
        dists = torch.norm(cls_emb - mean_emb, dim=1)

        k = min(per_class_budget, ids_t.numel())
        top_local = torch.topk(dists, k, largest=False).indices
        sample_ids.extend(ids_t[top_local].tolist())

    sample_ids.sort()
    if not sample_ids:
        return [], torch.empty(0, node_embeddings.size(1), device=device)
    sample_idx_t = torch.as_tensor(sample_ids, dtype=torch.long, device=device)
    sample_embs = node_embeddings[sample_idx_t]
    return sample_ids, sample_embs


# ----------------------------------------------------------------------
# Teacher fusion -- port of utils_metric.combine_all_params
# ----------------------------------------------------------------------


def combine_dis_state_dicts(history_state_dicts):
    """Fuse historical DisModel state-dicts into a single teacher state.

    Mirrors the original ``combine_all_params``:

    - ``encoder_c``: linear-decay weighted sum, ``(n - idx) / S`` with
      ``S = 1 + 2 + ... + n``. Most recent session gets the largest weight.
    - ``encoder_s`` / ``decoder`` / ``discriminator``: simple arithmetic
      mean over all sessions (same as the original code's
      ``+= current; / file_count`` pattern).

    Edge case (n == 1) matches the original: return the lone state-dict
    untouched, which is the natural identity element of the fusion.

    Args:
        history_state_dicts: list of ``DisModel.state_dict()`` snapshots,
            ordered oldest-first. ``len(...) >= 1`` is required by the
            caller (we never fuse for the very first streaming session
            either, since session 0 always saves before session 1 reads).

    Returns:
        ``OrderedDict`` ready for ``DisModel.load_state_dict``.
    """
    n = len(history_state_dicts)
    if n == 0:
        raise ValueError("combine_dis_state_dicts requires >= 1 snapshot")
    if n == 1:
        # Original returns current_params directly when n==1
        return OrderedDict(history_state_dicts[0])

    S = sum(range(1, n + 1))  # 1 + 2 + ... + n
    encoder_c_weighted = OrderedDict()
    fused = OrderedDict()

    for idx, snap in enumerate(history_state_dicts):
        for key, value in snap.items():
            if "encoder_c" in key:
                weight = (n - idx) / S
                if key not in encoder_c_weighted:
                    encoder_c_weighted[key] = weight * value.clone()
                else:
                    encoder_c_weighted[key] = (
                        encoder_c_weighted[key] + weight * value
                    )
            else:
                if key not in fused:
                    fused[key] = value.clone()
                else:
                    fused[key] = fused[key] + value

    out = OrderedDict()
    for snap in history_state_dicts:
        ref_keys = list(snap.keys())
        break

    for key in ref_keys:
        if "encoder_c" in key:
            out[key] = encoder_c_weighted[key]
        else:
            out[key] = (fused[key] / float(n)).type(fused[key].dtype)
    return out
