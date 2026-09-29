"""Ground-truth oracles transcribed from HuggingFace PEFT.

Same rules as `torch_judge.harness.rl.vendored`: each module carries its upstream
URL, pinned commit, symbols and license, the arithmetic is transcribed line for
line, and only infrastructure that cannot run offline is removed.
`tests/quality/test_peft_vendored_oracles.py` compares every PEFT reference
solution against these functions.

No module in this package may import `torch_judge.tasks`.

Licenses of vendored material:
  * hf_peft.py — Apache License 2.0, the HuggingFace Inc. team (PEFT and transformers)
"""

from torch_judge.harness.peft.vendored.hf_peft import (
    peft_causal_lm_attention_mask,
    peft_dora_forward,
    peft_dora_init_magnitude,
    peft_get_delta_weight,
    peft_merge,
    peft_prefix_past_key_values,
    peft_unmerge,
)

__all__ = [
    "peft_causal_lm_attention_mask",
    "peft_dora_forward",
    "peft_dora_init_magnitude",
    "peft_get_delta_weight",
    "peft_merge",
    "peft_prefix_past_key_values",
    "peft_unmerge",
]
