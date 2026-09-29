"""Vendored LoRA merge, DoRA and prefix-tuning oracles from HuggingFace PEFT.

Upstream:   https://github.com/huggingface/peft
Commit:     b8674c86183a5dee38d0c3ede392e189593025e5
Files:      src/peft/tuners/lora/layer.py, src/peft/tuners/lora/dora.py,
            src/peft/peft_model.py
Symbols:    Linear.get_delta_weight, Linear.merge, Linear.unmerge,
            DoraLinearLayer.get_weight_norm, DoraLinearLayer.update_layer,
            DoraLinearLayer.forward, PeftModel.get_prompt,
            PeftModelForCausalLM.forward (prefix attention mask)

Upstream:   https://github.com/huggingface/transformers
Commit:     ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae
File:       src/transformers/modeling_attn_mask_utils.py
Symbols:    AttentionMaskConverter.to_4d, _make_causal_mask, _expand_mask

Retrieved:  2026-09-29
License:    Apache License 2.0
            Copyright the HuggingFace Inc. team. Licensed under the Apache
            License, Version 2.0; you may obtain a copy at
            http://www.apache.org/licenses/LICENSE-2.0

WHY THIS FILE EXISTS
--------------------
Ground truth for lora_merge, dora and prefix_tuning. Each exercise states its
contract in its own words; these functions follow PEFT's code path instead, so a
shared misunderstanding between an exercise's evaluator and its reference
solution shows up as a disagreement here.

WHAT WAS CHANGED, AND WHY
-------------------------
The arithmetic is transcribed line for line. Removed infrastructure:

  * Named adapters, adapter dicts and module state become plain tensors.
  * ``fan_in_fan_out`` is fixed to False, so ``transpose`` is the identity.
  * ``safe_merge``, ``lora_bias``, fp16-on-CPU casts, caching, FSDP gathering,
    quantized-weight dequantization and deprecation warnings are dropped.
  * DoRA's ``lora_B(lora_A(x))`` module calls become ``F.linear`` with the same
    weights; ``get_lora_weight`` computes ``lora_B @ lora_A`` through the same
    identity-matrix trick.
  * ``get_prompt`` receives the flattened prefix tensor directly instead of
    running a ``PrefixEncoder``; ``num_transformer_submodules`` is fixed to 1.
  * The transformers mask helpers return boolean "may attend" masks instead of
    additive ``finfo.min`` masks; the rows and columns they select are unchanged.

NOTHING HERE IMPORTS A TASK SOLUTION. This module must stay independent of the
answers it is used to check.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "peft_causal_lm_attention_mask",
    "peft_dora_forward",
    "peft_dora_init_magnitude",
    "peft_get_delta_weight",
    "peft_merge",
    "peft_prefix_past_key_values",
    "peft_unmerge",
]


# --------------------------------------------------------------------------
# LoRA merge (peft/tuners/lora/layer.py, class Linear)
# --------------------------------------------------------------------------

def peft_get_delta_weight(lora_A_weight: torch.Tensor, lora_B_weight: torch.Tensor, scaling: float) -> torch.Tensor:
    # output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
    return (lora_B_weight @ lora_A_weight) * scaling


def peft_merge(base_weight: torch.Tensor, lora_A_weight, lora_B_weight, scaling) -> None:
    """Non-safe merge of a vanilla LoRA adapter, in place on ``base_weight``."""
    delta_weight = peft_get_delta_weight(lora_A_weight, lora_B_weight, scaling)
    base_weight.data += delta_weight


def peft_unmerge(base_weight: torch.Tensor, lora_A_weight, lora_B_weight, scaling) -> None:
    """Unmerge of a vanilla LoRA adapter, in place on ``base_weight``."""
    orig_dtype = base_weight.dtype
    delta_weight = peft_get_delta_weight(lora_A_weight, lora_B_weight, scaling)
    base_weight.data -= delta_weight.to(orig_dtype)


# --------------------------------------------------------------------------
# DoRA (peft/tuners/lora/dora.py, class DoraLinearLayer)
# --------------------------------------------------------------------------

def _get_weight_norm(weight: torch.Tensor, lora_weight: torch.Tensor, scaling: float) -> torch.Tensor:
    # calculate L2 norm of weight matrix, column-wise
    weight = weight + scaling * lora_weight
    weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
    return weight_norm


def _get_lora_weight(lora_A_weight: torch.Tensor, lora_B_weight: torch.Tensor) -> torch.Tensor:
    x_eye = torch.eye(lora_A_weight.shape[1], device=lora_A_weight.device, dtype=lora_A_weight.dtype)
    lora_weight = F.linear(F.linear(x_eye, lora_A_weight), lora_B_weight).T
    return lora_weight


def peft_dora_init_magnitude(base_weight: torch.Tensor, lora_A_weight, lora_B_weight, scaling) -> torch.Tensor:
    """``update_layer``: the initial magnitude vector."""
    lora_weight = lora_B_weight @ lora_A_weight
    return _get_weight_norm(base_weight, lora_weight, scaling)


def peft_dora_forward(x, *, base_weight, base_bias, lora_A_weight, lora_B_weight, magnitude, scaling):
    """Full DoRA layer output: base layer result plus ``DoraLinearLayer.forward``'s correction."""
    base_result = F.linear(x, base_weight, base_bias)

    lora_weight = _get_lora_weight(lora_A_weight, lora_B_weight)
    lora_weight = lora_weight.to(x.dtype)

    weight = base_weight.to(x.dtype)
    weight_norm = _get_weight_norm(weight, lora_weight.detach(), scaling)
    weight_norm = weight_norm.detach()
    mag_norm_scale = (magnitude / weight_norm).view(1, -1)

    lora_result = F.linear(F.linear(x, lora_A_weight), lora_B_weight)

    bias = base_bias
    result = base_result
    if bias is not None:
        base_result = base_result - bias

    result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling
    return result + result_dora


# --------------------------------------------------------------------------
# Prefix tuning (peft/peft_model.py) and the causal mask it relies on
# (transformers/modeling_attn_mask_utils.py)
# --------------------------------------------------------------------------

def peft_prefix_past_key_values(prefix: torch.Tensor, *, batch_size, num_layers, num_attention_heads, token_dim):
    """``get_prompt`` for PREFIX_TUNING: flattened prefix -> per-layer (key, value).

    ``prefix`` is the PrefixEncoder output for one sequence, shaped
    ``(num_virtual_tokens, num_layers * 2 * token_dim)``. Returns a list with one
    ``(key, value)`` pair per layer, each ``(batch, heads, num_virtual_tokens, head_dim)``.
    """
    num_virtual_tokens = prefix.shape[0]
    past_key_values = prefix.unsqueeze(0).expand(batch_size, -1, -1)
    past_key_values = past_key_values.reshape(
        batch_size,
        num_virtual_tokens,
        num_layers * 2,
        num_attention_heads,
        token_dim // num_attention_heads,
    )
    # Transpose: 2 x [num_layers, batch_size, num_heads, num_virtual_tokens, head_dim]
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    return [(layer[0], layer[1]) for layer in past_key_values]


def _make_causal_mask(tgt_len: int, past_key_values_length: int) -> torch.Tensor:
    mask_cond = torch.arange(tgt_len)
    # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0) marks the attended slots
    allowed = mask_cond < (mask_cond + 1).view(tgt_len, 1)
    if past_key_values_length > 0:
        allowed = torch.cat([torch.ones(tgt_len, past_key_values_length, dtype=torch.bool), allowed], dim=-1)
    return allowed


def peft_causal_lm_attention_mask(attention_mask: torch.Tensor, num_virtual_tokens: int) -> torch.Tensor:
    """Boolean ``(batch, 1, T, num_virtual_tokens + T)`` mask a causal LM sees under prefix tuning.

    PEFT prepends ``num_virtual_tokens`` ones to the 2D mask; transformers then builds
    a causal mask whose past length is the prefix length and combines the two.
    """
    batch_size, query_length = attention_mask.shape
    prefix_attention_mask = torch.ones(batch_size, num_virtual_tokens).to(attention_mask.device)
    attention_mask = torch.cat((prefix_attention_mask, attention_mask.to(prefix_attention_mask.dtype)), dim=1)

    key_value_length = attention_mask.shape[1]
    past_key_values_length = key_value_length - query_length
    causal = _make_causal_mask(query_length, past_key_values_length)
    expanded = attention_mask[:, None, None, :].expand(batch_size, 1, query_length, key_value_length).bool()
    return causal[None, None] & expanded
