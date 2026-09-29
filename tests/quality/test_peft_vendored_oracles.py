"""Every PEFT-path reference solution, checked against HuggingFace PEFT's own arithmetic.

The mutation gate in `test_peft_path_mutations.py` proves each evaluator rejects wrong
implementations. These tests prove the contract itself matches upstream, by comparing
each reference solution with code transcribed into `torch_judge/harness/peft/vendored/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torch_judge.harness.peft.vendored import (
    peft_causal_lm_attention_mask,
    peft_dora_forward,
    peft_dora_init_magnitude,
    peft_get_delta_weight,
    peft_merge,
    peft_prefix_past_key_values,
    peft_unmerge,
)
from torch_judge.tasks import get_task

VENDORED = Path(__file__).resolve().parents[2] / "torch_judge/harness/peft/vendored"


def reference(task_id: str):
    task = get_task(task_id)
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def test_vendored_oracles_do_not_import_tasks():
    for path in VENDORED.glob("*.py"):
        imports = [line for line in path.read_text().splitlines() if line.startswith(("import ", "from "))]
        assert not any("torch_judge.tasks" in line for line in imports), f"{path.name} imports task solutions"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lora_merge_matches_peft(seed):
    torch.manual_seed(seed)
    layer = reference("lora_merge")(9, 5, rank=3, alpha=7.0)
    with torch.no_grad():
        layer.lora_A.normal_()
        layer.lora_B.normal_()
    base = layer.linear.weight.detach().clone()
    A, B = layer.lora_A.detach(), layer.lora_B.detach()

    expected = base.clone()
    peft_merge(expected, A, B, 7.0 / 3)
    layer.merge()
    torch.testing.assert_close(layer.linear.weight.detach(), expected)
    torch.testing.assert_close(expected - base, peft_get_delta_weight(A, B, 7.0 / 3))

    peft_unmerge(expected, A, B, 7.0 / 3)
    layer.unmerge()
    torch.testing.assert_close(layer.linear.weight.detach(), expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dora_matches_peft(seed):
    torch.manual_seed(seed)
    layer = reference("dora")(11, 6, rank=4, alpha=12.0).double()
    scaling = 12.0 / 4
    W = layer.linear.weight.detach()
    torch.testing.assert_close(
        layer.magnitude.detach(),
        peft_dora_init_magnitude(W, layer.lora_A.detach(), layer.lora_B.detach(), scaling),
    )
    with torch.no_grad():
        layer.lora_A.normal_()
        layer.lora_B.normal_()
        layer.magnitude.mul_(torch.rand(6, dtype=torch.float64) + 0.5)
        layer.linear.bias.normal_()

    x = torch.randn(3, 4, 11, dtype=torch.float64)
    out = layer(x)
    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()

    A = layer.lora_A.detach().clone().requires_grad_()
    B = layer.lora_B.detach().clone().requires_grad_()
    m = layer.magnitude.detach().clone().requires_grad_()
    want = peft_dora_forward(
        x, base_weight=W, base_bias=layer.linear.bias.detach(),
        lora_A_weight=A, lora_B_weight=B, magnitude=m, scaling=scaling,
    )
    (want * upstream).sum().backward()

    torch.testing.assert_close(out.detach(), want.detach())
    torch.testing.assert_close(layer.lora_A.grad, A.grad)
    torch.testing.assert_close(layer.lora_B.grad, B.grad)
    torch.testing.assert_close(layer.magnitude.grad, m.grad)


def _hf_prefix_attention(layer, x, mask, prefix_k, prefix_v):
    """Causal self-attention the way a transformers model runs it with PEFT's past_key_values."""
    B, T, D = x.shape
    H, P, hd = prefix_k.shape[1], prefix_k.shape[2], prefix_k.shape[3]
    q = layer.q_proj(x).view(B, T, H, hd).transpose(1, 2)
    k = torch.cat([prefix_k, layer.k_proj(x).view(B, T, H, hd).transpose(1, 2)], dim=2)
    v = torch.cat([prefix_v, layer.v_proj(x).view(B, T, H, hd).transpose(1, 2)], dim=2)
    allowed = peft_causal_lm_attention_mask(mask, P)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    return layer.o_proj(out.transpose(1, 2).reshape(B, T, D))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prefix_tuning_matches_peft_past_key_values(seed):
    torch.manual_seed(seed)
    num_layers, heads, d_model, num_virtual = 3, 2, 8, 4
    layer = reference("prefix_tuning")(d_model, heads, num_virtual).double()
    flat_prefix = torch.randn(num_virtual, num_layers * 2 * d_model, dtype=torch.float64)
    x = torch.randn(3, 6, d_model, dtype=torch.float64)
    mask = torch.ones(3, 6, dtype=torch.long)
    mask[1, :2] = 0  # left padding
    mask[2, 4:] = 0  # right padding

    past = peft_prefix_past_key_values(
        flat_prefix, batch_size=3, num_layers=num_layers, num_attention_heads=heads, token_dim=d_model,
    )
    for key, value in past:
        with torch.no_grad():
            layer.prefix_k.copy_(key[0])
            layer.prefix_v.copy_(value[0])
        torch.testing.assert_close(
            layer(x, attention_mask=mask),
            _hf_prefix_attention(layer, x, mask, key, value),
        )
