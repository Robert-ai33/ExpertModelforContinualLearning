"""
SimGCL baseline (LLM4GCL port).

Algorithm (kept faithful to ``LLM4GCL/models/GLM/SimGCL.py``):
- Session 0: fine-tune the LM backbone (LLaMA causal-LM loss with LoRA
  4-bit quantisation) on the *current* text-attributed graph. The
  cosine-classifier head ``fc`` is not trained; its weights are
  populated via prototype averaging.
- Subsequent sessions: no LM training. Only ``update_proto`` is called
  to copy class-mean embeddings into ``fc.weight`` for the new classes
  introduced this session.
- Inference: temperature-scaled cosine similarity between mean-pooled
  hidden states and ``fc.weight`` (= class prototypes).

The outer ``fit`` loop is rewritten to match the user's CGLB protocol:
row k of ``acc_matrix`` is computed by evaluating the cumulative
session-k subgraph on each ``test_idx_per_task[t]`` for ``t <= k``.
The returned dictionary mirrors ``LiteExpertCL.fit``'s shape so the
result plays nicely with ``run_all.py``-style aggregation.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils import compute_ap_af, print_cl_matrix

from .lm_backbone import build_backbone
from .simgcl_prompts import get_instruction_prompts


IGNORE_INDEX = -100


def _mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


# ======================================================================
# LM wrapper with cosine classifier head
# ======================================================================

class LMModel(nn.Module):
    """LLaMA backbone + cosine-classifier head."""

    def __init__(self, lm_type, max_length, model_path, output_dim,
                 lora_config, dropout, att_dropout, T, device):
        super().__init__()
        self.lm_type = lm_type
        self.device = device
        self.max_length = max_length
        self.T = T

        self.lm = build_backbone(
            lm_type, model_path, lora_config, dropout, att_dropout, device)
        self.hidden_dim = self.lm.hidden_dim
        self.fc = nn.Linear(self.hidden_dim, output_dim).to(device)

    # ------------------------------------------------------------------
    # Training forward (returns object with ``.loss`` and ``.hidden_states``)
    # ------------------------------------------------------------------

    def forward(self, instructions, class_labels=None):
        return self._llama_forward(instructions)

    def _llama_forward(self, instructions):
        tok = self.lm.tokenizer
        emb = self.lm.embeddings
        pad_emb = emb(torch.tensor(tok.pad_token_id, device=self.device)).unsqueeze(0)
        bos_id = tok.bos_token_id
        bos_list = [bos_id] if bos_id is not None else []
        eos_id = tok.eos_token_id

        batch_embeds, batch_attn, batch_labels = [], [], []
        for item in instructions:
            ctx = '[INST]' + item["Context"]
            q = item["Question"] + '[/INST]'
            ans = item["Answer"]
            ctx_tok = tok(ctx, add_special_tokens=False)
            q_tok = tok(q, add_special_tokens=False)
            ans_tok = tok(ans, add_special_tokens=False)

            max_text = (self.max_length
                        - len(q_tok.input_ids + ans_tok.input_ids + [eos_id]) - 1)
            input_ids = (
                bos_list
                + ctx_tok.input_ids[:max_text]
                + q_tok.input_ids
                + ans_tok.input_ids
                + [eos_id]
            )
            inputs_embeds = emb(torch.tensor(input_ids, device=self.device))
            label_ids = (
                [IGNORE_INDEX] * (
                    len(bos_list)
                    + len(ctx_tok.input_ids[:max_text])
                    + len(q_tok.input_ids))
                + ans_tok.input_ids
                + [eos_id]
            )
            batch_embeds.append(inputs_embeds)
            batch_attn.append([1] * len(input_ids))
            batch_labels.append(label_ids)

        max_len = max(x.size(0) for x in batch_embeds)
        for i in range(len(batch_embeds)):
            pad_len = max_len - batch_embeds[i].size(0)
            batch_embeds[i] = torch.cat([
                pad_emb.repeat(pad_len, 1), batch_embeds[i]])
            batch_attn[i] = [0] * pad_len + batch_attn[i]
            batch_labels[i] = [IGNORE_INDEX] * pad_len + batch_labels[i]

        inputs_embeds = torch.stack(batch_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attn, device=self.device)
        labels = torch.tensor(batch_labels, device=self.device)

        return self.lm(inputs_embeds, attention_mask, labels=labels)

    # ------------------------------------------------------------------
    # Eval-time forwards (no LM loss, just hidden states / cosine logits)
    # ------------------------------------------------------------------

    def embedding_forward(self, instructions):
        return self._llama_embedding_forward(instructions)

    def _llama_embedding_forward(self, instructions):
        tok = self.lm.tokenizer
        emb = self.lm.embeddings
        pad_emb = emb(torch.tensor(tok.pad_token_id, device=self.device)).unsqueeze(0)
        bos_id = tok.bos_token_id
        bos_list = [bos_id] if bos_id is not None else []

        batch_embeds, batch_attn = [], []
        for item in instructions:
            ctx = '[INST]' + item["Context"]
            q = item["Question"] + '[/INST]'
            ctx_tok = tok(ctx, add_special_tokens=False)
            q_tok = tok(q, add_special_tokens=False)
            max_text = self.max_length - len(q_tok.input_ids) - 1
            input_ids = (
                bos_list
                + ctx_tok.input_ids[:max_text]
                + q_tok.input_ids
            )
            inputs_embeds = emb(torch.tensor(input_ids, device=self.device))
            batch_embeds.append(inputs_embeds)
            batch_attn.append([1] * len(input_ids))

        max_len = max(x.size(0) for x in batch_embeds)
        for i in range(len(batch_embeds)):
            pad_len = max_len - batch_embeds[i].size(0)
            batch_embeds[i] = torch.cat([
                pad_emb.repeat(pad_len, 1), batch_embeds[i]])
            batch_attn[i] = [0] * pad_len + batch_attn[i]

        inputs_embeds = torch.stack(batch_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attn, device=self.device)
        outputs = self.lm(inputs_embeds, attention_mask)
        return outputs, attention_mask

    def cosine_forward(self, instructions):
        outputs, attn = self.embedding_forward(instructions)
        last = outputs.hidden_states[-1]
        x = _mean_pooling(last, attn)
        x = F.linear(F.normalize(x, p=2, dim=-1),
                     F.normalize(self.fc.weight, p=2, dim=-1))
        return self.T * x

    def hidden_pooling(self, instructions):
        outputs, attn = self.embedding_forward(instructions)
        last = outputs.hidden_states[-1]
        return _mean_pooling(last, attn), attn

    def generate_text(self, instructions):
        """Optional: causal-LM text generation (kept for parity with the
        upstream evaluate path; never invoked under the cosine pipeline).
        """
        # For brevity we omit the upstream generate plumbing; the cosine
        # pipeline does not call this method. Implement on demand.
        raise NotImplementedError(
            "generate_text is intentionally inert in this CL pipeline; "
            "use cosine_forward instead.")


# ======================================================================
# SimGCL CL baseline -- public API matches LiteExpertCL.fit
# ======================================================================

class SimGCLBaseline:
    """LLM-based CL baseline: fine-tune at session 0, then prototype updates.

    Performance considerations and tunables live in
    ``configs/config_simgcl.yaml``. Backbone is LLaMA-3.2-{1B,3B} with
    4-bit NF4 quantisation + LoRA, faithful to LLM4GCL's SimGCL.
    """

    def __init__(self, task_loader, config, device):
        self.task_loader = task_loader
        self.config = config
        self.device = device

        self.dataset = config['dataset']
        self.session_num = task_loader.sessions
        self.num_class = int(task_loader.data.y.max().item()) + 1

        self.lm_type = config['lm']
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.lora_config = config.get('LoRA', {'use_lora': False})
        self.dropout = config.get('dropout', 0.1)
        self.att_dropout = config.get('att_dropout', 0.1)
        self.max_length = config.get('max_length', 256)
        self.model_path = config.get('model_path', './model_cache/')

        self.T = config.get('T', 1.0)
        self.sample_num = config.get('sample_num', 50)
        self.hop = tuple(config.get('hop', [10, 10]))
        self.mode = config.get('mode', 'neighbors')
        self.include_label = config.get('include_label', False)
        self.max_node_text_len = config.get('max_node_text_len', 128)

        self.epochs = int(config.get('epochs', 10))
        self.valid_epoch = int(config.get('valid_epoch', 1))
        self.warmup_epochs = int(config.get('warmup_epochs', 1))
        self.min_lr = float(config.get('min_lr', 5e-6))
        self.grad_steps = int(config.get('grad_steps', 2))
        self.patience = int(config.get('patience', 5))

        self.checkpoint_path = config.get('checkpoint_path', './checkpoints/simgcl')
        self.seed = int(config.get('_seed', 0))

        self.model = LMModel(
            self.lm_type, self.max_length, self.model_path,
            self.num_class, self.lora_config, self.dropout, self.att_dropout,
            self.T, device,
        )

    # ------------------------------------------------------------------
    # Optimizer / scheduler (matches LLM4GCL's adjust_learning_rate)
    # ------------------------------------------------------------------

    def get_optimizer(self):
        params = [p for _, p in self.model.named_parameters() if p.requires_grad]
        return optim.AdamW(
            [{'params': params, 'lr': self.lr,
              'weight_decay': self.weight_decay}],
            betas=(0.9, 0.95),
        )

    def _adjust_lr(self, param_group, epoch_frac):
        import math
        if epoch_frac < self.warmup_epochs:
            lr = self.lr * epoch_frac / max(self.warmup_epochs, 1)
        else:
            denom = max(self.epochs - self.warmup_epochs, 1)
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch_frac - self.warmup_epochs) / denom))
        param_group['lr'] = lr

    # ------------------------------------------------------------------
    # Per-class prototype update (== ``update_proto`` upstream)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_proto(self, text_dataset, train_loader, class_src, class_dst,
                      label_index):
        self.model.eval()
        embeds, labels = [], []
        for batch in train_loader:
            if batch['node_id'].size(0) < 2:
                continue
            instructions = get_instruction_prompts(
                batch['node_id'], text_dataset.data,
                text_dataset.data.raw_texts, label_index, class_dst,
                self.dataset, self.hop, self.mode,
                self.include_label, self.max_node_text_len)
            pooled, _ = self.model.hidden_pooling(instructions)
            embeds.extend(pooled)
            labels.extend(batch['labels'].to(self.device))

        if not embeds:
            return

        embeds = torch.stack(embeds, dim=0)
        labels = torch.stack(labels, dim=0)

        for cls in range(class_src, class_dst):
            sel = (labels == cls).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            proto = embeds[sel].mean(dim=0)
            self.model.fc.weight.data[cls] = proto

    # ------------------------------------------------------------------
    # Train / eval primitives
    # ------------------------------------------------------------------

    def _train_epoch(self, curr_session, curr_epoch, text_dataset,
                     train_loader, optimizer, class_dst, label_index):
        self.model.train()
        all_loss, n_samples = 0.0, 0
        for step, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                continue
            optimizer.zero_grad()
            instructions = get_instruction_prompts(
                batch['node_id'], text_dataset.data,
                text_dataset.data.raw_texts, label_index, class_dst,
                self.dataset, self.hop, self.mode,
                self.include_label, self.max_node_text_len)
            class_labels = batch['labels'].to(self.device).long()
            outputs = self.model(instructions, class_labels=class_labels)
            loss = outputs.loss
            loss.backward()
            all_loss += loss.item() * batch['node_id'].size(0)
            n_samples += batch['node_id'].size(0)
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if (step + 1) % self.grad_steps == 0:
                self._adjust_lr(
                    optimizer.param_groups[0],
                    step / max(len(train_loader), 1) + curr_epoch)
            optimizer.step()
        return all_loss / max(n_samples, 1)

    @torch.no_grad()
    def _cosine_eval(self, text_dataset, loader, class_num, label_index,
                     compute_macro=False):
        self.model.eval()
        preds_all, labels_all = [], []
        for batch in loader:
            if batch['node_id'].size(0) < 2:
                continue
            instructions = get_instruction_prompts(
                batch['node_id'], text_dataset.data,
                text_dataset.data.raw_texts, label_index, class_num,
                self.dataset, self.hop, self.mode,
                self.include_label, self.max_node_text_len)
            logits = self.model.cosine_forward(instructions)
            logits = logits[:, : class_num]
            preds = logits.argmax(dim=1).cpu()
            labels = batch['labels'].cpu()
            preds_all.append(preds)
            labels_all.append(labels)
        if not preds_all:
            return 0.0, 0.0
        preds = torch.cat(preds_all).numpy()
        labels = torch.cat(labels_all).numpy()
        acc = float(accuracy_score(labels, preds))
        if compute_macro:
            macro = float(f1_score(labels, preds, average='macro', zero_division=0))
            # Per-class accuracy macro (matches LiteExpertCL._evaluate_subgraph)
            per_class = []
            unique = sorted(set(int(x) for x in labels.tolist()))
            for c in unique:
                mask = labels == c
                if mask.sum() > 0:
                    per_class.append(float((preds[mask] == c).mean()))
            macro_acc = float(sum(per_class) / max(len(per_class), 1))
            return acc, macro_acc
        return acc, 0.0

    # ------------------------------------------------------------------
    # Checkpoint helpers (per-trial save/restore, mirrors LLM4GCL)
    # ------------------------------------------------------------------

    def _ckpt_path(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        return os.path.join(
            self.checkpoint_path,
            f"{self.dataset}_{self.lm_type}_seed{self.seed}.pt")

    def _save_best(self):
        path = self._ckpt_path()
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()
                 if v.requires_grad}
        try:
            torch.save({'model': state}, path)
        except Exception as e:
            print(f"  [warn] failed to save checkpoint: {e}")

    def _reload_best(self):
        path = self._ckpt_path()
        if not os.path.exists(path):
            return
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)
        except Exception as e:
            print(f"  [warn] failed to reload checkpoint: {e}")

    # ------------------------------------------------------------------
    # Main fit (CGLB protocol, return dict matching LiteExpertCL)
    # ------------------------------------------------------------------

    def fit(self, trial):
        self.seed = int(self.config.get('_seed', trial))
        optimizer = self.get_optimizer()

        # Build cumulative label_index per session for prompt construction.
        label_index_isolate, label_index_joint = [], []
        for k in range(self.session_num):
            tr = list(self.task_loader.train_idx_per_task[k])
            va = list(self.task_loader.valid_idx_per_task[k])
            label_index_isolate.append(tr + va)
            if k == 0:
                label_index_joint.append(label_index_isolate[k])
            else:
                label_index_joint.append(
                    label_index_isolate[k] + label_index_joint[-1])

        # ============== Phase 0: session-0 fine-tuning ==============
        (class_src0, class_dst0,
         text_iso, text_jot,
         train_loader, valid_loader,
         _, _) = self.task_loader.get_task(0)

        progress_bar = tqdm(range(self.epochs))
        progress_bar.set_description(f'SimGCL Train | Iter {trial}')
        tolerate, best_acc_valid, ever_saved = 0, 0.0, False
        for epoch in range(self.epochs):
            loss = self._train_epoch(
                0, epoch, text_iso, train_loader, optimizer,
                class_dst0, label_index_isolate[0])
            progress_bar.write(
                f"Session 0 | Epoch {epoch} | Loss {loss:.4f}")
            if epoch > 0 and epoch % self.valid_epoch == 0:
                acc_valid, _ = self._cosine_eval(
                    text_iso, valid_loader, class_dst0,
                    label_index_isolate[0])
                progress_bar.write(
                    f"Session 0 | Epoch {epoch} | Valid Acc {acc_valid:.4f} "
                    f"| Tolerate {tolerate}")
                if acc_valid > best_acc_valid:
                    best_acc_valid = acc_valid
                    tolerate = 0
                    self._save_best()
                    ever_saved = True
                else:
                    tolerate += 1
                    if tolerate > self.patience:
                        break
            progress_bar.set_postfix(loss=f"{loss:.4f}",
                                     best=f"{best_acc_valid:.4f}")
            progress_bar.update(1)
        progress_bar.close()

        if ever_saved:
            self._reload_best()

        # ============== Phase 1: per-session prototypes + CGLB eval ==============
        acc_matrix = []
        joint_acc_history = []
        joint_macro_history = []

        bs = getattr(self.task_loader, 'batch_size', 8)

        for k in range(self.session_num):
            print(f"\n{'='*60}")
            print(f"SimGCL Session {k}")
            print(f"{'='*60}")

            (class_src, class_dst,
             text_iso, text_jot,
             train_loader_k, _,
             _, _) = self.task_loader.get_task(k, subset=self.sample_num)

            self._update_proto(
                text_iso, train_loader_k, class_src, class_dst,
                label_index_isolate[k])

            # Per-task accuracy on cumulative subgraph (CGLB protocol):
            # row k uses session-k cumulative wrapper, evaluated on
            # ``test_idx_per_task[t]`` for each prior task t <= k.
            print(f"--- Per-Task Tests (Session {k}) ---")
            acc_row = []
            for t in range(k + 1):
                test_idx_t = self.task_loader.test_idx_per_task[t]
                if not test_idx_t:
                    acc_row.append(0.0)
                    continue
                loader_t = DataLoader(
                    Subset(text_iso, test_idx_t),
                    batch_size=bs, shuffle=False)
                acc_t, _ = self._cosine_eval(
                    text_iso, loader_t, class_dst,
                    label_index_isolate[k])
                acc_row.append(acc_t)
                print(f"  Task {t}: Acc={acc_t:.4f} ({len(test_idx_t)} nodes)")
            acc_matrix.append(acc_row)

            # Joint (cumulative) test
            test_idx_jot = self.task_loader.test_idx_joint[k]
            loader_jot = DataLoader(
                Subset(text_jot, test_idx_jot),
                batch_size=bs, shuffle=False)
            joint_micro, joint_macro = self._cosine_eval(
                text_jot, loader_jot, class_dst,
                label_index_joint[k], compute_macro=True)
            joint_acc_history.append(joint_micro)
            joint_macro_history.append(joint_macro)
            print(f"--- Joint Test (Session {k}) ---")
            print(f"  Micro={joint_micro:.4f} Macro={joint_macro:.4f} "
                  f"({len(test_idx_jot)} nodes)")

        print(f"\n{'='*60}")
        print("SimGCL FINAL RESULTS")
        print(f"{'='*60}")
        print_cl_matrix("CL Accuracy Matrix", acc_matrix, self.session_num)
        ap_history, af, final_ap = compute_ap_af(acc_matrix)
        print(f"Final AP: {final_ap:.4f}    AF: {af:+.4f}")

        return {
            'acc_matrix': acc_matrix,
            'joint_acc': joint_acc_history,
            'joint_macro_acc': joint_macro_history,
            'ap_history': ap_history,
            'af': af,
            'final_ap': final_ap,
        }
