"""
LM backbone for the SimGCL baseline (LLM4GCL-faithful).

Supported choices:
    'llama-1b'   -> meta-llama/Llama-3.2-1B  (4-bit + LoRA)
    'llama-3b'   -> meta-llama/Llama-3.2-3B  (4-bit + LoRA)

Weights are resolved in this order:
  1. Local ModelScope snapshot under ``model_path`` (default
     ``./model_cache/LLM-Research/Llama-3___2-{1B,3B}``). Pre-download via:
         from modelscope import snapshot_download
         snapshot_download('LLM-Research/Llama-3.2-1B', cache_dir='./model_cache')
         snapshot_download('LLM-Research/Llama-3.2-3B', cache_dir='./model_cache')
     This path is offline, no Hugging Face account / token required.
  2. Hugging Face hub fallback (``meta-llama/...``). Gated, requires
     ``HF_TOKEN`` and an approved access request.
  3. Explicit override via env vars ``LLAMA_1B_PATH`` / ``LLAMA_3B_PATH``.

The LLaMA path follows ``LLM4GCL/backbones/LM/LLaMA.py`` 1:1 (4-bit nf4
quantisation via bitsandbytes, LoRA on q_proj/v_proj, ``[INST]...[/INST]``
prompt, causal-LM forward returning ``loss/logits/hidden_states``).
"""

import os

import torch
import torch.nn as nn


SUPPORTED_LM_TYPES = ('llama-1b', 'llama-3b')


def build_backbone(lm_type, model_path, lora_config, dropout, att_dropout, device):
    """Factory: returns a ``LLaMANet`` instance."""
    lm_type = lm_type.lower()
    if lm_type not in SUPPORTED_LM_TYPES:
        raise ValueError(
            f"Unsupported lm '{lm_type}'. "
            f"Choose from {SUPPORTED_LM_TYPES}.")
    return LLaMANet(lm_type, model_path, lora_config, dropout, att_dropout, device)


# ======================================================================
# LLaMA backbone (4-bit + LoRA), aligned with LLM4GCL
# ======================================================================
#
# Resolution order for the actual weights:
#   1. Local ModelScope snapshot under ``model_path`` (preferred; works
#      offline, no HF token needed). ModelScope replaces dots with
#      ``___`` on Windows, so ``Llama-3.2-1B`` becomes ``Llama-3___2-1B``.
#   2. Hugging Face hub fallback (gated, requires ``HF_TOKEN``).
# A user can also force a path via env var ``LLAMA_1B_PATH`` /
# ``LLAMA_3B_PATH`` for full control.

_LLAMA_HF_MAP = {
    'llama-1b': ('meta-llama/Llama-3.2-1B', 2048),
    'llama-3b': ('meta-llama/Llama-3.2-3B', 3072),
}

_LLAMA_MODELSCOPE_DIRS = {
    'llama-1b': [
        os.path.join('LLM-Research', 'Llama-3___2-1B'),
        os.path.join('LLM-Research', 'Llama-3.2-1B'),
    ],
    'llama-3b': [
        os.path.join('LLM-Research', 'Llama-3___2-3B'),
        os.path.join('LLM-Research', 'Llama-3.2-3B'),
    ],
}

_LLAMA_ENV_OVERRIDE = {
    'llama-1b': 'LLAMA_1B_PATH',
    'llama-3b': 'LLAMA_3B_PATH',
}


def _resolve_llama_source(lm_type, model_path):
    """Pick a local ModelScope snapshot if present, else the HF id.

    Returns ``(name_or_path, is_local)``.
    """
    env_key = _LLAMA_ENV_OVERRIDE.get(lm_type)
    env_path = os.environ.get(env_key) if env_key else None
    if env_path and os.path.isdir(env_path):
        return env_path, True

    for rel in _LLAMA_MODELSCOPE_DIRS.get(lm_type, ()):
        candidate = os.path.join(model_path, rel)
        if os.path.isdir(candidate) and os.path.isfile(
            os.path.join(candidate, 'config.json')
        ):
            return candidate, True

    return _LLAMA_HF_MAP[lm_type][0], False


class LLaMANet(nn.Module):
    """4-bit quantised LLaMA-3.2-{1B,3B} with LoRA adapters."""

    def __init__(self, lm_type, model_path, lora_config, dropout, att_dropout, device):
        super().__init__()
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                       BitsAndBytesConfig)
        except ImportError as e:
            raise ImportError(
                "LLaMA backbone needs `transformers`, `peft`, `bitsandbytes`. "
                "Install with: pip install transformers peft bitsandbytes"
            ) from e

        self.lm_type = lm_type
        _, self.hidden_dim = _LLAMA_HF_MAP[lm_type]
        self.lora_config = lora_config
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.max_ans_length = 32

        self.model_name, is_local = _resolve_llama_source(lm_type, model_path)
        if is_local:
            print(f"[LLaMANet] Loading {lm_type} from local snapshot: "
                  f"{self.model_name}")
            access_token = None
            from_pretrained_kwargs = {'local_files_only': True}
        else:
            print(f"[LLaMANet] Local snapshot not found for {lm_type}; "
                  f"falling back to Hugging Face hub: {self.model_name}")
            access_token = os.environ.get('HF_TOKEN', None)
            from_pretrained_kwargs = {'cache_dir': model_path}

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=access_token, **from_pretrained_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map={"": device.index if device.type == 'cuda' else 'cpu'},
            quantization_config=quant_config,
            token=access_token,
            **from_pretrained_kwargs,
        )

        model.config.dropout = self.dropout
        model.config.attention_dropout = self.att_dropout
        model.config.output_hidden_states = True

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        )
        self.tokenizer.padding_side = 'left'
        self.embeddings = model.get_input_embeddings()

        if lora_config.get('use_lora', True):
            cfg = LoraConfig(
                r=lora_config['lora_r'],
                lora_alpha=lora_config['lora_alpha'],
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_config['lora_dropout'],
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(model, cfg)
        else:
            self.model = model

    def forward(self, input, attention_mask, labels=None):
        kwargs = (
            {"input_ids": input}
            if input.dtype in (torch.int32, torch.int64) and input.dim() == 2
            else {"inputs_embeds": input}
        )
        return self.model(
            **kwargs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def generate(self, input, attention_mask):
        kwargs = (
            {"input_ids": input}
            if input.dtype in (torch.int32, torch.int64) and input.dim() == 2
            else {"inputs_embeds": input}
        )
        outputs = self.model.generate(
            **kwargs,
            attention_mask=attention_mask,
            max_new_tokens=self.max_ans_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
