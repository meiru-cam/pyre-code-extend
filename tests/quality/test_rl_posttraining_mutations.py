"""Mutation gates and metadata contracts for the post-training RL path."""

from __future__ import annotations

from pathlib import Path

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


PER_TOKEN_LOGPROBS_MUTATIONS = [
    Mutation("log_after_softmax", """def per_token_logprobs(logits, input_ids):
    log_probs = torch.softmax(logits, dim=-1).log()
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
"""),
    Mutation("normalize_over_positions", """def per_token_logprobs(logits, input_ids):
    log_probs = torch.log_softmax(logits, dim=1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
"""),
    Mutation("returns_probs_not_logprobs", """def per_token_logprobs(logits, input_ids):
    probs = torch.softmax(logits, dim=-1)
    return torch.gather(probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
"""),
    Mutation("applies_causal_shift", """def per_token_logprobs(logits, input_ids):
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    shifted = input_ids[:, 1:]
    return torch.gather(log_probs, dim=-1, index=shifted.unsqueeze(-1)).squeeze(-1)
"""),
    Mutation("detached", """def per_token_logprobs(logits, input_ids):
    log_probs = torch.log_softmax(logits, dim=-1).detach()
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
"""),
    Mutation("reduced_to_scalar", """def per_token_logprobs(logits, input_ids):
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1).mean()
"""),
    Mutation("argmax_instead_of_realized", """def per_token_logprobs(logits, input_ids):
    return torch.log_softmax(logits, dim=-1).max(dim=-1).values
"""),
    Mutation("casts_to_float32", """def per_token_logprobs(logits, input_ids):
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
"""),
]


GROUP_ADVANTAGE_MUTATIONS = [
    Mutation("global_normalization", """def group_relative_advantage(rewards, group_size, eps=1e-4):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("bad group_size")
    if rewards.numel() % group_size != 0:
        raise ValueError("bad group_size")
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)
"""),
    Mutation("mean_only_no_scale", """def group_relative_advantage(rewards, group_size, eps=1e-4):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("bad group_size")
    if rewards.numel() % group_size != 0:
        raise ValueError("bad group_size")
    grouped = rewards.reshape(-1, group_size)
    return (grouped - grouped.mean(dim=-1, keepdim=True)).reshape(-1)
"""),
    Mutation("scale_only_no_centering", """def group_relative_advantage(rewards, group_size, eps=1e-4):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("bad group_size")
    if rewards.numel() % group_size != 0:
        raise ValueError("bad group_size")
    grouped = rewards.reshape(-1, group_size)
    return (grouped / (grouped.std(dim=-1, unbiased=False, keepdim=True) + eps)).reshape(-1)
"""),
    Mutation("eps_added_after_division", """def group_relative_advantage(rewards, group_size, eps=1e-4):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("bad group_size")
    if rewards.numel() % group_size != 0:
        raise ValueError("bad group_size")
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=False, keepdim=True)
    return ((grouped - mean) / std + eps).reshape(-1)
"""),
    Mutation("sample_std_bessel", """def group_relative_advantage(rewards, group_size, eps=1e-4):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("bad group_size")
    if rewards.numel() % group_size != 0:
        raise ValueError("bad group_size")
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=True, keepdim=True)
    return ((grouped - mean) / (std + eps)).reshape(-1)
"""),
    Mutation("reduces_across_groups", """def group_relative_advantage(rewards, group_size, eps=1e-4):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("bad group_size")
    if rewards.numel() % group_size != 0:
        raise ValueError("bad group_size")
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=0, keepdim=True)
    std = grouped.std(dim=0, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + eps)).reshape(-1)
"""),
    Mutation("accepts_invalid_group_size", """def group_relative_advantage(rewards, group_size, eps=1e-4):
    grouped = rewards.reshape(-1, max(int(group_size), 1))
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + eps)).reshape(-1)
"""),
]


K3_KL_MUTATIONS = [
    Mutation("k1_plain_difference", """def k3_kl_penalty(logprobs, ref_logprobs):
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError("shape mismatch")
    return logprobs - ref_logprobs
"""),
    Mutation("k2_squared", """def k3_kl_penalty(logprobs, ref_logprobs):
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError("shape mismatch")
    log_ratio = ref_logprobs - logprobs
    return log_ratio * log_ratio / 2.0
"""),
    Mutation("reversed_log_ratio", """def k3_kl_penalty(logprobs, ref_logprobs):
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError("shape mismatch")
    log_ratio = logprobs - ref_logprobs
    return torch.exp(log_ratio) - log_ratio - 1.0
"""),
    Mutation("wrong_constant_sign", """def k3_kl_penalty(logprobs, ref_logprobs):
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError("shape mismatch")
    log_ratio = ref_logprobs - logprobs
    return torch.exp(log_ratio) - log_ratio + 1.0
"""),
    Mutation("missing_linear_term", """def k3_kl_penalty(logprobs, ref_logprobs):
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError("shape mismatch")
    log_ratio = ref_logprobs - logprobs
    return torch.exp(log_ratio) - 1.0
"""),
    Mutation("detached", """def k3_kl_penalty(logprobs, ref_logprobs):
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError("shape mismatch")
    log_ratio = (ref_logprobs - logprobs).detach()
    return torch.exp(log_ratio) - log_ratio - 1.0
"""),
    Mutation("reduced_to_scalar", """def k3_kl_penalty(logprobs, ref_logprobs):
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError("shape mismatch")
    log_ratio = ref_logprobs - logprobs
    return (torch.exp(log_ratio) - log_ratio - 1.0).mean()
"""),
    Mutation("broadcasts_mismatched_shapes", """def k3_kl_penalty(logprobs, ref_logprobs):
    log_ratio = ref_logprobs - logprobs
    return torch.exp(log_ratio) - log_ratio - 1.0
"""),
]


RESPONSE_MASK_MUTATIONS = [
    Mutation("keeps_prompt_positions", """def response_token_mask(prompt_lens, total_lens, max_len):
    if prompt_lens.shape != total_lens.shape:
        raise ValueError("shape mismatch")
    if (prompt_lens < 0).any() or (total_lens < prompt_lens).any() or (total_lens > max_len).any():
        raise ValueError("bad lengths")
    positions = torch.arange(max_len, device=prompt_lens.device).unsqueeze(0)
    return positions < total_lens.unsqueeze(-1)
"""),
    Mutation("keeps_padding_positions", """def response_token_mask(prompt_lens, total_lens, max_len):
    if prompt_lens.shape != total_lens.shape:
        raise ValueError("shape mismatch")
    if (prompt_lens < 0).any() or (total_lens < prompt_lens).any() or (total_lens > max_len).any():
        raise ValueError("bad lengths")
    positions = torch.arange(max_len, device=prompt_lens.device).unsqueeze(0)
    return positions >= prompt_lens.unsqueeze(-1)
"""),
    Mutation("drops_final_token", """def response_token_mask(prompt_lens, total_lens, max_len):
    if prompt_lens.shape != total_lens.shape:
        raise ValueError("shape mismatch")
    if (prompt_lens < 0).any() or (total_lens < prompt_lens).any() or (total_lens > max_len).any():
        raise ValueError("bad lengths")
    positions = torch.arange(max_len, device=prompt_lens.device).unsqueeze(0)
    return (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1) - 1)
"""),
    Mutation("off_by_one_start", """def response_token_mask(prompt_lens, total_lens, max_len):
    if prompt_lens.shape != total_lens.shape:
        raise ValueError("shape mismatch")
    if (prompt_lens < 0).any() or (total_lens < prompt_lens).any() or (total_lens > max_len).any():
        raise ValueError("bad lengths")
    positions = torch.arange(max_len, device=prompt_lens.device).unsqueeze(0)
    return (positions > prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
"""),
    Mutation("returns_float_mask", """def response_token_mask(prompt_lens, total_lens, max_len):
    if prompt_lens.shape != total_lens.shape:
        raise ValueError("shape mismatch")
    if (prompt_lens < 0).any() or (total_lens < prompt_lens).any() or (total_lens > max_len).any():
        raise ValueError("bad lengths")
    positions = torch.arange(max_len, device=prompt_lens.device).unsqueeze(0)
    keep = (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
    return keep.float()
"""),
    Mutation("accepts_inconsistent_lengths", """def response_token_mask(prompt_lens, total_lens, max_len):
    positions = torch.arange(max_len, device=prompt_lens.device).unsqueeze(0)
    return (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
"""),
]


FORMAT_REWARD_MUTATIONS = [
    Mutation("substring_presence_only", '''def rlvr_format_reward(completion, eos_token):
    if not eos_token or not completion.endswith(eos_token):
        return 0.0
    body = completion[: -len(eos_token)]
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    return 1.0 if all(tag in body for tag in tags) else 0.0
'''),
    Mutation("ignores_eos", '''def rlvr_format_reward(completion, eos_token):
    body = completion
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(body.count(tag) != 1 for tag in tags):
        return 0.0
    positions = [body.index(tag) for tag in tags]
    if positions != sorted(positions):
        return 0.0
    think = body[positions[0] + 7 : positions[1]]
    answer = body[positions[2] + 8 : positions[3]]
    ideal = "<think>" + think + "</think>" + "<answer>" + answer + "</answer>"
    if body.strip() != ideal.strip() or not think.strip() or not answer.strip():
        return 0.5
    return 1.0
'''),
    Mutation("ignores_ordering", '''def rlvr_format_reward(completion, eos_token):
    if not eos_token or not completion.endswith(eos_token):
        return 0.0
    body = completion[: -len(eos_token)]
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(body.count(tag) != 1 for tag in tags):
        return 0.0
    return 1.0
'''),
    Mutation("binary_no_middle_tier", '''def rlvr_format_reward(completion, eos_token):
    if not eos_token or not completion.endswith(eos_token):
        return 0.0
    body = completion[: -len(eos_token)]
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(body.count(tag) != 1 for tag in tags):
        return 0.0
    positions = [body.index(tag) for tag in tags]
    if positions != sorted(positions):
        return 0.0
    think = body[positions[0] + 7 : positions[1]]
    answer = body[positions[2] + 8 : positions[3]]
    ideal = "<think>" + think + "</think>" + "<answer>" + answer + "</answer>"
    return 1.0 if body.strip() == ideal.strip() else 0.0
'''),
    Mutation("accepts_empty_blocks", '''def rlvr_format_reward(completion, eos_token):
    if not eos_token or not completion.endswith(eos_token):
        return 0.0
    body = completion[: -len(eos_token)]
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(body.count(tag) != 1 for tag in tags):
        return 0.0
    positions = [body.index(tag) for tag in tags]
    if positions != sorted(positions):
        return 0.0
    think = body[positions[0] + 7 : positions[1]]
    answer = body[positions[2] + 8 : positions[3]]
    ideal = "<think>" + think + "</think>" + "<answer>" + answer + "</answer>"
    return 1.0 if body.strip() == ideal.strip() else 0.5
'''),
    Mutation("returns_bool", '''def rlvr_format_reward(completion, eos_token):
    if not eos_token or not completion.endswith(eos_token):
        return False
    body = completion[: -len(eos_token)]
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(body.count(tag) != 1 for tag in tags):
        return False
    positions = [body.index(tag) for tag in tags]
    return positions == sorted(positions)
'''),
]


_GRPO_GUARDS = """    if logprobs.shape != ref_logprobs.shape or logprobs.shape != mask.shape:
        raise ValueError("shape mismatch")
    if advantages.shape != logprobs.shape[:1]:
        raise ValueError("advantages shape mismatch")
    if beta < 0:
        raise ValueError("negative beta")
"""

GRPO_TOKEN_LOSS_MUTATIONS = [
    Mutation("divides_by_padded_size", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    keep = mask.to(logprobs.dtype)
    if keep.sum() == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.unsqueeze(-1) * logprobs - beta * kl
    return -(objective * keep).mean()
"""),
    Mutation("per_response_normalization", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    keep = mask.to(logprobs.dtype)
    if keep.sum() == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.unsqueeze(-1) * logprobs - beta * kl
    per_row = (objective * keep).sum(dim=-1) / keep.sum(dim=-1).clamp(min=1.0)
    return -per_row.mean()
"""),
    Mutation("kl_sign_flipped", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.unsqueeze(-1) * logprobs + beta * kl
    return -(objective * keep).sum() / denominator
"""),
    Mutation("ignores_mask", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.unsqueeze(-1) * logprobs - beta * kl
    return -objective.mean()
"""),
    Mutation("drops_kl_term", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    objective = advantages.unsqueeze(-1) * logprobs
    return -(objective * keep).sum() / denominator
"""),
    Mutation("advantage_not_broadcast", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.mean() * logprobs - beta * kl
    return -(objective * keep).sum() / denominator
"""),
    Mutation("sign_not_negated", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.unsqueeze(-1) * logprobs - beta * kl
    return (objective * keep).sum() / denominator
"""),
    Mutation("nan_on_empty_mask", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    keep = mask.to(logprobs.dtype)
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.unsqueeze(-1) * logprobs - beta * kl
    return -(objective * keep).sum() / keep.sum()
"""),
    Mutation("detaches_logprobs", """def grpo_token_loss(logprobs, ref_logprobs, advantages, mask, beta=0.04):
""" + _GRPO_GUARDS + """    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    frozen = logprobs.detach()
    log_ratio = ref_logprobs - frozen
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    objective = advantages.unsqueeze(-1) * frozen - beta * kl
    return -(objective * keep).sum() / denominator
"""),
]


_PPO_SIGNATURE = (
    "def ppo_clipped_policy_loss(logprobs, old_logprobs, advantages, mask, "
    "clip_eps=0.2, dual_clip=None):\n"
)

_PPO_GUARDS = """    shapes = (logprobs.shape, old_logprobs.shape, advantages.shape, mask.shape)
    if len(set(shapes)) != 1:
        raise ValueError("shape mismatch")
    if clip_eps <= 0:
        raise ValueError("bad clip_eps")
    if dual_clip is not None and dual_clip <= 1.0:
        raise ValueError("bad dual_clip")
    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=logprobs.dtype, device=logprobs.device)
    ratio = torch.exp(logprobs - old_logprobs)
"""

PPO_CLIPPED_MUTATIONS = [
    Mutation("max_instead_of_min", _PPO_SIGNATURE + _PPO_GUARDS + """    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token = -torch.max(unclipped, clipped)
    if dual_clip is not None:
        per_token = torch.where(advantages < 0, torch.min(per_token, -dual_clip * advantages), per_token)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("clipped_branch_only", _PPO_SIGNATURE + _PPO_GUARDS + """    per_token = -(ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages)
    if dual_clip is not None:
        per_token = torch.where(advantages < 0, torch.min(per_token, -dual_clip * advantages), per_token)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("no_clipping_at_all", _PPO_SIGNATURE + _PPO_GUARDS + """    per_token = -(ratio * advantages)
    if dual_clip is not None:
        per_token = torch.where(advantages < 0, torch.min(per_token, -dual_clip * advantages), per_token)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("clamps_the_product", _PPO_SIGNATURE + _PPO_GUARDS + """    unclipped = ratio * advantages
    clipped = (ratio * advantages).clamp(1.0 - clip_eps, 1.0 + clip_eps)
    per_token = -torch.min(unclipped, clipped)
    if dual_clip is not None:
        per_token = torch.where(advantages < 0, torch.min(per_token, -dual_clip * advantages), per_token)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("dual_clip_applied_everywhere", _PPO_SIGNATURE + _PPO_GUARDS + """    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token = -torch.min(unclipped, clipped)
    if dual_clip is not None:
        per_token = torch.min(per_token, -dual_clip * advantages)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("dual_clip_ignored", _PPO_SIGNATURE + _PPO_GUARDS + """    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token = -torch.min(unclipped, clipped)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("negate_before_min", _PPO_SIGNATURE + _PPO_GUARDS + """    unclipped = -(ratio * advantages)
    clipped = -(ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages)
    per_token = torch.min(unclipped, clipped)
    if dual_clip is not None:
        per_token = torch.where(advantages < 0, torch.min(per_token, -dual_clip * advantages), per_token)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("ignores_mask", _PPO_SIGNATURE + _PPO_GUARDS + """    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token = -torch.min(unclipped, clipped)
    if dual_clip is not None:
        per_token = torch.where(advantages < 0, torch.min(per_token, -dual_clip * advantages), per_token)
    return per_token.mean()
"""),
    Mutation("detaches_logprobs", _PPO_SIGNATURE + _PPO_GUARDS + """    ratio = torch.exp(logprobs.detach() - old_logprobs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token = -torch.min(unclipped, clipped)
    if dual_clip is not None:
        per_token = torch.where(advantages < 0, torch.min(per_token, -dual_clip * advantages), per_token)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("reversed_ratio", _PPO_SIGNATURE + _PPO_GUARDS + """    ratio = torch.exp(old_logprobs - logprobs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token = -torch.min(unclipped, clipped)
    if dual_clip is not None:
        per_token = torch.where(advantages < 0, torch.min(per_token, -dual_clip * advantages), per_token)
    return (per_token * keep).sum() / denominator
"""),
]


_VALUE_SIGNATURE = "def ppo_value_loss(values, old_values, returns, mask, clip_eps=0.2):\n"

_VALUE_GUARDS = """    shapes = (values.shape, old_values.shape, returns.shape, mask.shape)
    if len(set(shapes)) != 1:
        raise ValueError("shape mismatch")
    if clip_eps <= 0:
        raise ValueError("bad clip_eps")
    keep = mask.to(values.dtype)
    denominator = keep.sum()
    if denominator == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
"""

PPO_VALUE_MUTATIONS = [
    Mutation("min_instead_of_max", _VALUE_SIGNATURE + _VALUE_GUARDS + """    clipped_value = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    per_token = 0.5 * torch.min((values - returns) ** 2, (clipped_value - returns) ** 2)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("clamps_the_prediction", _VALUE_SIGNATURE + _VALUE_GUARDS + """    clipped_value = values.clamp(-clip_eps, clip_eps)
    per_token = 0.5 * torch.max((values - returns) ** 2, (clipped_value - returns) ** 2)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("clamps_the_error", _VALUE_SIGNATURE + _VALUE_GUARDS + """    unclipped = (values - returns) ** 2
    per_token = 0.5 * unclipped.clamp(-clip_eps, clip_eps)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("missing_half_factor", _VALUE_SIGNATURE + _VALUE_GUARDS + """    clipped_value = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    per_token = torch.max((values - returns) ** 2, (clipped_value - returns) ** 2)
    return (per_token * keep).sum() / denominator
"""),
    Mutation("no_clipping_at_all", _VALUE_SIGNATURE + _VALUE_GUARDS + """    per_token = 0.5 * (values - returns) ** 2
    return (per_token * keep).sum() / denominator
"""),
    Mutation("ignores_mask", _VALUE_SIGNATURE + _VALUE_GUARDS + """    clipped_value = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    per_token = 0.5 * torch.max((values - returns) ** 2, (clipped_value - returns) ** 2)
    return per_token.mean()
"""),
    Mutation("detaches_values", _VALUE_SIGNATURE + _VALUE_GUARDS + """    frozen = values.detach()
    clipped_value = old_values + (frozen - old_values).clamp(-clip_eps, clip_eps)
    per_token = 0.5 * torch.max((frozen - returns) ** 2, (clipped_value - returns) ** 2)
    return (per_token * keep).sum() / denominator
"""),
]


_GAE_SIGNATURE = "def gae_advantage(rewards, values, dones, gamma=0.99, lam=0.95):\n"

_GAE_GUARDS = """    if rewards.shape != dones.shape:
        raise ValueError("shape mismatch")
    horizon = rewards.shape[-1]
    if values.shape[:-1] != rewards.shape[:-1] or values.shape[-1] != horizon + 1:
        raise ValueError("values horizon mismatch")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("bad gamma")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("bad lam")
    not_done = 1.0 - dones.to(rewards.dtype)
"""

GAE_MUTATIONS = [
    Mutation("forward_recursion", _GAE_SIGNATURE + _GAE_GUARDS + """    deltas = rewards + gamma * values[..., 1:] * not_done - values[..., :-1]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in range(horizon):
        running = deltas[..., t] + gamma * lam * not_done[..., t] * running
        advantages[..., t] = running
    return advantages
"""),
    Mutation("bootstraps_across_done", _GAE_SIGNATURE + _GAE_GUARDS + """    deltas = rewards + gamma * values[..., 1:] - values[..., :-1]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(horizon)):
        running = deltas[..., t] + gamma * lam * not_done[..., t] * running
        advantages[..., t] = running
    return advantages
"""),
    Mutation("accumulates_across_done", _GAE_SIGNATURE + _GAE_GUARDS + """    deltas = rewards + gamma * values[..., 1:] * not_done - values[..., :-1]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(horizon)):
        running = deltas[..., t] + gamma * lam * running
        advantages[..., t] = running
    return advantages
"""),
    Mutation("missing_gamma_in_recursion", _GAE_SIGNATURE + _GAE_GUARDS + """    deltas = rewards + gamma * values[..., 1:] * not_done - values[..., :-1]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(horizon)):
        running = deltas[..., t] + lam * not_done[..., t] * running
        advantages[..., t] = running
    return advantages
"""),
    Mutation("td_residual_only", _GAE_SIGNATURE + _GAE_GUARDS + """    return rewards + gamma * values[..., 1:] * not_done - values[..., :-1]
"""),
    Mutation("off_by_one_value_slice", _GAE_SIGNATURE + _GAE_GUARDS + """    deltas = rewards + gamma * values[..., :-1] * not_done - values[..., 1:]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(horizon)):
        running = deltas[..., t] + gamma * lam * not_done[..., t] * running
        advantages[..., t] = running
    return advantages
"""),
    Mutation("accepts_short_values", """def gae_advantage(rewards, values, dones, gamma=0.99, lam=0.95):
    horizon = rewards.shape[-1]
    not_done = 1.0 - dones.to(rewards.dtype)
    padded = values if values.shape[-1] == horizon + 1 else torch.cat(
        [values, torch.zeros_like(values[..., :1])], dim=-1
    )
    deltas = rewards + gamma * padded[..., 1:] * not_done - padded[..., :-1]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(horizon)):
        running = deltas[..., t] + gamma * lam * not_done[..., t] * running
        advantages[..., t] = running
    return advantages
"""),
]


_GSPO_SIGNATURE = "def gspo_sequence_ratio(logprobs, old_logprobs, mask):\n"

_GSPO_GUARDS = """    shapes = (logprobs.shape, old_logprobs.shape, mask.shape)
    if len(set(shapes)) != 1:
        raise ValueError("shape mismatch")
    keep = mask.to(logprobs.dtype)
"""

GSPO_MUTATIONS = [
    Mutation("exponentiates_before_aggregating", _GSPO_SIGNATURE + _GSPO_GUARDS + """    ratio = torch.exp(logprobs - old_logprobs) * keep
    count = keep.sum(dim=-1)
    return ratio.sum(dim=-1) / count.clamp(min=1.0)
"""),
    Mutation("sums_without_normalizing", _GSPO_SIGNATURE + _GSPO_GUARDS + """    log_ratio = (logprobs - old_logprobs) * keep
    return torch.exp(log_ratio.sum(dim=-1))
"""),
    Mutation("divides_by_padded_length", _GSPO_SIGNATURE + _GSPO_GUARDS + """    log_ratio = (logprobs - old_logprobs) * keep
    return torch.exp(log_ratio.sum(dim=-1) / logprobs.shape[-1])
"""),
    Mutation("ignores_mask", _GSPO_SIGNATURE + _GSPO_GUARDS + """    log_ratio = logprobs - old_logprobs
    return torch.exp(log_ratio.mean(dim=-1))
"""),
    Mutation("reversed_log_ratio", _GSPO_SIGNATURE + _GSPO_GUARDS + """    log_ratio = (old_logprobs - logprobs) * keep
    count = keep.sum(dim=-1)
    return torch.exp(log_ratio.sum(dim=-1) / count.clamp(min=1.0))
"""),
    Mutation("token_level_not_sequence_level", _GSPO_SIGNATURE + _GSPO_GUARDS + """    return torch.exp(logprobs - old_logprobs)
"""),
    Mutation("nan_on_empty_row", _GSPO_SIGNATURE + _GSPO_GUARDS + """    log_ratio = (logprobs - old_logprobs) * keep
    return torch.exp(log_ratio.sum(dim=-1) / keep.sum(dim=-1))
"""),
    Mutation("detaches_logprobs", _GSPO_SIGNATURE + _GSPO_GUARDS + """    log_ratio = (logprobs.detach() - old_logprobs) * keep
    count = keep.sum(dim=-1)
    return torch.exp(log_ratio.sum(dim=-1) / count.clamp(min=1.0))
"""),
]


_DAPO_SIGNATURE = "def dapo_dynamic_sampling(rewards, group_size, tol=0.0):\n"

_DAPO_GUARDS = """    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("bad group_size")
    if rewards.numel() % group_size != 0:
        raise ValueError("bad group_size")
    if tol < 0:
        raise ValueError("bad tol")
    grouped = rewards.reshape(-1, group_size)
"""

DAPO_MUTATIONS = [
    Mutation("drops_only_all_correct", _DAPO_SIGNATURE + _DAPO_GUARDS + """    keep = grouped.min(dim=-1).values < grouped.max(dim=-1).values
    keep = keep | (grouped.max(dim=-1).values < 1.0)
    return keep.repeat_interleave(group_size)
"""),
    Mutation("filters_individual_responses", _DAPO_SIGNATURE + _DAPO_GUARDS + """    mean = grouped.mean(dim=-1, keepdim=True)
    return ((grouped - mean).abs() > tol).reshape(-1)
"""),
    Mutation("tiles_instead_of_repeating", _DAPO_SIGNATURE + _DAPO_GUARDS + """    spread = grouped.max(dim=-1).values - grouped.min(dim=-1).values
    keep = spread > tol
    return keep.repeat(group_size)
"""),
    Mutation("keeps_constant_groups", _DAPO_SIGNATURE + _DAPO_GUARDS + """    return torch.ones(rewards.numel(), dtype=torch.bool, device=rewards.device)
"""),
    Mutation("inverted_decision", _DAPO_SIGNATURE + _DAPO_GUARDS + """    spread = grouped.max(dim=-1).values - grouped.min(dim=-1).values
    keep = spread <= tol
    return keep.repeat_interleave(group_size)
"""),
    Mutation("ignores_tolerance", _DAPO_SIGNATURE + _DAPO_GUARDS + """    spread = grouped.max(dim=-1).values - grouped.min(dim=-1).values
    keep = spread > 0.0
    return keep.repeat_interleave(group_size)
"""),
    Mutation("accepts_invalid_group_size", """def dapo_dynamic_sampling(rewards, group_size, tol=0.0):
    grouped = rewards.reshape(-1, max(int(group_size), 1))
    spread = grouped.max(dim=-1).values - grouped.min(dim=-1).values
    return (spread > tol).repeat_interleave(max(int(group_size), 1))
"""),
]


_VINE_SIGNATURE = "def vineppo_mc_value(rollout_rewards, final_reward):\n"

_VINE_GUARDS = """    if rollout_rewards.ndim != 2:
        raise ValueError("not two-dimensional")
    states, continuations = rollout_rewards.shape
    if states == 0 or continuations == 0:
        raise ValueError("empty rollout matrix")
"""

VINEPPO_MUTATIONS = [
    Mutation("averages_across_states", _VINE_SIGNATURE + _VINE_GUARDS + """    values = rollout_rewards.mean(dim=0)
    bootstrap = final_reward.reshape(1).to(values.dtype)
    next_value = torch.cat([values[1:], bootstrap])
    return values, next_value - values
"""),
    Mutation("sign_flipped_advantage", _VINE_SIGNATURE + _VINE_GUARDS + """    values = rollout_rewards.mean(dim=-1)
    bootstrap = final_reward.reshape(1).to(values.dtype)
    next_value = torch.cat([values[1:], bootstrap])
    return values, values - next_value
"""),
    Mutation("drops_the_bootstrap", _VINE_SIGNATURE + _VINE_GUARDS + """    values = rollout_rewards.mean(dim=-1)
    next_value = torch.cat([values[1:], torch.zeros(1, dtype=values.dtype, device=values.device)])
    return values, next_value - values
"""),
    Mutation("one_advantage_broadcast", _VINE_SIGNATURE + _VINE_GUARDS + """    values = rollout_rewards.mean(dim=-1)
    overall = final_reward.to(values.dtype) - values[0]
    return values, overall.expand_as(values).clone()
"""),
    Mutation("returns_raw_rewards_as_values", _VINE_SIGNATURE + _VINE_GUARDS + """    values = rollout_rewards[:, 0]
    bootstrap = final_reward.reshape(1).to(values.dtype)
    next_value = torch.cat([values[1:], bootstrap])
    return values, next_value - values
"""),
    Mutation("sums_instead_of_averaging", _VINE_SIGNATURE + _VINE_GUARDS + """    values = rollout_rewards.sum(dim=-1)
    bootstrap = final_reward.reshape(1).to(values.dtype)
    next_value = torch.cat([values[1:], bootstrap])
    return values, next_value - values
"""),
    Mutation("accepts_malformed_input", """def vineppo_mc_value(rollout_rewards, final_reward):
    flat = rollout_rewards.reshape(-1, rollout_rewards.shape[-1]) if rollout_rewards.ndim > 1 else rollout_rewards.reshape(1, -1)
    values = flat.mean(dim=-1)
    bootstrap = final_reward.reshape(1).to(values.dtype)
    next_value = torch.cat([values[1:], bootstrap])
    return values, next_value - values
"""),
]


_ASSEMBLY_SIGNATURE = (
    "def rollout_batch_assembly(prompt_ids, generation_ids, advantages, pad_id=0):\n"
)

_ASSEMBLY_GUARDS = """    batch_size = len(prompt_ids)
    if batch_size == 0 or len(generation_ids) != batch_size:
        raise ValueError("bad batch")
    if advantages.ndim != 1 or advantages.shape[0] != batch_size:
        raise ValueError("bad advantages")
    device = advantages.device
    prompt_lens = torch.tensor([len(p) for p in prompt_ids], device=device)
    gen_lens = torch.tensor([len(g) for g in generation_ids], device=device)
    total_lens = prompt_lens + gen_lens
"""

_ASSEMBLY_FILL = """    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long, device=device)
    for row, (prompt, generation) in enumerate(zip(prompt_ids, generation_ids)):
        tokens = list(prompt) + list(generation)
        if tokens:
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    positions = torch.arange(width, device=device).unsqueeze(0)
"""

ROLLOUT_ASSEMBLY_MUTATIONS = [
    Mutation("mask_covers_the_prompt", _ASSEMBLY_SIGNATURE + _ASSEMBLY_GUARDS + """    width = int(total_lens.max())
""" + _ASSEMBLY_FILL + """    mask = positions < total_lens.unsqueeze(-1)
    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": advantages.unsqueeze(-1) * mask.to(advantages.dtype),
        "lengths": gen_lens,
    }
"""),
    Mutation("mask_covers_the_padding", _ASSEMBLY_SIGNATURE + _ASSEMBLY_GUARDS + """    width = int(total_lens.max())
""" + _ASSEMBLY_FILL + """    mask = positions >= prompt_lens.unsqueeze(-1)
    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": advantages.unsqueeze(-1) * mask.to(advantages.dtype),
        "lengths": gen_lens,
    }
"""),
    Mutation("advantage_not_zeroed_outside_mask", _ASSEMBLY_SIGNATURE + _ASSEMBLY_GUARDS + """    width = int(total_lens.max())
""" + _ASSEMBLY_FILL + """    mask = (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
    token_advantages = advantages.unsqueeze(-1).expand(-1, width).clone()
    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": token_advantages,
        "lengths": gen_lens,
    }
"""),
    Mutation("width_from_generations_only", _ASSEMBLY_SIGNATURE + _ASSEMBLY_GUARDS + """    width = int(gen_lens.max())
""" + _ASSEMBLY_FILL + """    mask = (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": advantages.unsqueeze(-1) * mask.to(advantages.dtype),
        "lengths": gen_lens,
    }
"""),
    Mutation("drops_the_prompt_tokens", _ASSEMBLY_SIGNATURE + _ASSEMBLY_GUARDS + """    width = int(total_lens.max())
    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long, device=device)
    for row, generation in enumerate(generation_ids):
        tokens = list(generation)
        if tokens:
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    positions = torch.arange(width, device=device).unsqueeze(0)
    mask = (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": advantages.unsqueeze(-1) * mask.to(advantages.dtype),
        "lengths": gen_lens,
    }
"""),
    Mutation("lengths_include_the_prompt", _ASSEMBLY_SIGNATURE + _ASSEMBLY_GUARDS + """    width = int(total_lens.max())
""" + _ASSEMBLY_FILL + """    mask = (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": advantages.unsqueeze(-1) * mask.to(advantages.dtype),
        "lengths": total_lens,
    }
"""),
    Mutation("left_padded_layout", _ASSEMBLY_SIGNATURE + _ASSEMBLY_GUARDS + """    width = int(total_lens.max())
    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long, device=device)
    for row, (prompt, generation) in enumerate(zip(prompt_ids, generation_ids)):
        tokens = list(prompt) + list(generation)
        if tokens:
            input_ids[row, width - len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
    positions = torch.arange(width, device=device).unsqueeze(0)
    mask = (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": advantages.unsqueeze(-1) * mask.to(advantages.dtype),
        "lengths": gen_lens,
    }
"""),
    Mutation("accepts_mismatched_batch", """def rollout_batch_assembly(prompt_ids, generation_ids, advantages, pad_id=0):
    batch_size = min(len(prompt_ids), len(generation_ids), advantages.reshape(-1).shape[0])
    device = advantages.device
    prompt_lens = torch.tensor([len(p) for p in prompt_ids[:batch_size]], device=device)
    gen_lens = torch.tensor([len(g) for g in generation_ids[:batch_size]], device=device)
    total_lens = prompt_lens + gen_lens
    width = int(total_lens.max()) if batch_size else 0
    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long, device=device)
    for row in range(batch_size):
        tokens = list(prompt_ids[row]) + list(generation_ids[row])
        if tokens:
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    positions = torch.arange(width, device=device).unsqueeze(0)
    mask = (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
    flat = advantages.reshape(-1)[:batch_size]
    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": flat.unsqueeze(-1) * mask.to(flat.dtype),
        "lengths": gen_lens,
    }
"""),
]


_STEP_SIGNATURE = "def grpo_train_step(model, ref_model, optimizer, batch, beta=0.04):\n"

_STEP_GUARDS = """    required = ("input_ids", "mask", "advantages")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError("missing keys")
    if beta < 0:
        raise ValueError("negative beta")
    input_ids = batch["input_ids"]
    mask = batch["mask"]
    advantages = batch["advantages"]
"""

_STEP_FORWARD = """    logits = model(input_ids)
    logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        ref_logits = ref_model(input_ids)
        ref_logprobs = torch.gather(
            torch.log_softmax(ref_logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    weighted_logprobs = advantages.unsqueeze(-1) * logprobs
    objective = weighted_logprobs - beta * kl
    loss = -(objective * keep).sum() / denominator
"""

_STEP_RETURN = """    return {
        "loss": loss.detach(),
        "pg_loss": (-(weighted_logprobs * keep).sum() / denominator).detach(),
        "kl": ((kl * keep).sum() / denominator).detach(),
        "num_tokens": denominator.detach(),
    }
"""

GRPO_TRAIN_STEP_MUTATIONS = [
    Mutation("missing_zero_grad", _STEP_SIGNATURE + _STEP_GUARDS + _STEP_FORWARD + """    loss.backward()
    optimizer.step()
""" + _STEP_RETURN),
    Mutation("zero_grad_after_backward", _STEP_SIGNATURE + _STEP_GUARDS + _STEP_FORWARD + """    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
""" + _STEP_RETURN),
    Mutation("steps_twice", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
""" + _STEP_FORWARD + """    loss.backward()
    optimizer.step()
    optimizer.step()
""" + _STEP_RETURN),
    Mutation("never_steps", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
""" + _STEP_FORWARD + """    loss.backward()
""" + _STEP_RETURN),
    Mutation("detaches_logprobs_before_loss", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
    logits = model(input_ids)
    logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1).detach()
    with torch.no_grad():
        ref_logits = ref_model(input_ids)
        ref_logprobs = torch.gather(
            torch.log_softmax(ref_logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    weighted_logprobs = advantages.unsqueeze(-1) * logprobs
    objective = weighted_logprobs - beta * kl
    loss = -(objective * keep).sum() / denominator
    if loss.requires_grad:
        loss.backward()
    optimizer.step()
""" + _STEP_RETURN),
    Mutation("returns_attached_metrics", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
""" + _STEP_FORWARD + """    loss.backward()
    optimizer.step()
    return {
        "loss": loss,
        "pg_loss": -(weighted_logprobs * keep).sum() / denominator,
        "kl": (kl * keep).sum() / denominator,
        "num_tokens": denominator,
    }
"""),
    Mutation("kl_sign_flipped", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
    logits = model(input_ids)
    logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        ref_logits = ref_model(input_ids)
        ref_logprobs = torch.gather(
            torch.log_softmax(ref_logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    weighted_logprobs = advantages.unsqueeze(-1) * logprobs
    objective = weighted_logprobs + beta * kl
    loss = -(objective * keep).sum() / denominator
    loss.backward()
    optimizer.step()
""" + _STEP_RETURN),
    Mutation("divides_by_padded_size", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
    logits = model(input_ids)
    logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        ref_logits = ref_model(input_ids)
        ref_logprobs = torch.gather(
            torch.log_softmax(ref_logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
    keep = mask.to(logprobs.dtype)
    denominator = torch.tensor(float(logprobs.numel()))
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    weighted_logprobs = advantages.unsqueeze(-1) * logprobs
    objective = weighted_logprobs - beta * kl
    loss = -(objective * keep).sum() / denominator
    loss.backward()
    optimizer.step()
""" + _STEP_RETURN),
    Mutation("reported_kl_scaled_by_beta", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
""" + _STEP_FORWARD + """    loss.backward()
    optimizer.step()
    return {
        "loss": loss.detach(),
        "pg_loss": (-(weighted_logprobs * keep).sum() / denominator).detach(),
        "kl": (beta * (kl * keep).sum() / denominator).detach(),
        "num_tokens": denominator.detach(),
    }
"""),
    Mutation("ignores_advantages", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
    logits = model(input_ids)
    logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        ref_logits = ref_model(input_ids)
        ref_logprobs = torch.gather(
            torch.log_softmax(ref_logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    weighted_logprobs = logprobs
    objective = weighted_logprobs - beta * kl
    loss = -(objective * keep).sum() / denominator
    loss.backward()
    optimizer.step()
""" + _STEP_RETURN),
    Mutation("reference_forward_with_grad", _STEP_SIGNATURE + _STEP_GUARDS + """    optimizer.zero_grad()
    logits = model(input_ids)
    logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    for parameter in ref_model.parameters():
        parameter.requires_grad_(True)
    ref_logits = ref_model(input_ids)
    ref_logprobs = torch.gather(
        torch.log_softmax(ref_logits, dim=-1), dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    keep = mask.to(logprobs.dtype)
    denominator = keep.sum()
    log_ratio = ref_logprobs - logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    weighted_logprobs = advantages.unsqueeze(-1) * logprobs
    objective = weighted_logprobs - beta * kl
    loss = -(objective * keep).sum() / denominator
    loss.backward()
    optimizer.step()
""" + _STEP_RETURN),
]


RL_TASK_IDS = [
    "per_token_logprobs",
    "group_relative_advantage",
    "k3_kl_penalty",
    "response_token_mask",
    "rlvr_format_reward",
    "grpo_token_loss",
    "ppo_clipped_policy_loss",
    "ppo_value_loss",
    "gae_advantage",
    "gspo_sequence_ratio",
    "dapo_dynamic_sampling",
    "vineppo_mc_value",
    "rollout_batch_assembly",
    "grpo_train_step",
    "agentic_rollout_loop",
]


_ROLLOUT_SIGNATURE = (
    "def agentic_rollout_loop(model, tools, initial_messages, max_turns, reward_fn):\n"
)

_ROLLOUT_GUARD = """    if max_turns < 1:
        raise ValueError("bad max_turns")
"""

AGENTIC_ROLLOUT_MUTATIONS = [
    Mutation("tool_output_marked_trainable", _ROLLOUT_SIGNATURE + _ROLLOUT_GUARD + """    messages = [dict(m) for m in initial_messages]
    trainable = [m.get("role") == "assistant" for m in messages]
    turns = tool_calls = 0
    stop_reason = "max_turns"
    while turns < max_turns:
        response = model.call(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response.get("content", "")})
        trainable.append(True)
        requested = response.get("tool_calls") or []
        if not requested:
            stop_reason = "finished"
            break
        for call in requested:
            name = call["name"]
            if name not in tools:
                raise ValueError("unknown tool")
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
            trainable.append(True)
    return {"messages": messages, "trainable": trainable, "turns": turns,
            "tool_calls": tool_calls, "stop_reason": stop_reason,
            "reward": float(reward_fn(messages))}
"""),
    Mutation("trainable_falls_out_of_step", _ROLLOUT_SIGNATURE + _ROLLOUT_GUARD + """    messages = [dict(m) for m in initial_messages]
    trainable = [m.get("role") == "assistant" for m in messages]
    turns = tool_calls = 0
    stop_reason = "max_turns"
    while turns < max_turns:
        response = model.call(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response.get("content", "")})
        trainable.append(True)
        requested = response.get("tool_calls") or []
        if not requested:
            stop_reason = "finished"
            break
        for call in requested:
            name = call["name"]
            if name not in tools:
                raise ValueError("unknown tool")
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
    return {"messages": messages, "trainable": trainable, "turns": turns,
            "tool_calls": tool_calls, "stop_reason": stop_reason,
            "reward": float(reward_fn(messages))}
"""),
    Mutation("truncation_reported_as_finished", _ROLLOUT_SIGNATURE + _ROLLOUT_GUARD + """    messages = [dict(m) for m in initial_messages]
    trainable = [m.get("role") == "assistant" for m in messages]
    turns = tool_calls = 0
    while turns < max_turns:
        response = model.call(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response.get("content", "")})
        trainable.append(True)
        requested = response.get("tool_calls") or []
        if not requested:
            break
        for call in requested:
            name = call["name"]
            if name not in tools:
                raise ValueError("unknown tool")
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
            trainable.append(False)
    return {"messages": messages, "trainable": trainable, "turns": turns,
            "tool_calls": tool_calls, "stop_reason": "finished",
            "reward": float(reward_fn(messages))}
"""),
    Mutation("budget_off_by_one", _ROLLOUT_SIGNATURE + _ROLLOUT_GUARD + """    messages = [dict(m) for m in initial_messages]
    trainable = [m.get("role") == "assistant" for m in messages]
    turns = tool_calls = 0
    stop_reason = "max_turns"
    while turns <= max_turns:
        response = model.call(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response.get("content", "")})
        trainable.append(True)
        requested = response.get("tool_calls") or []
        if not requested:
            stop_reason = "finished"
            break
        for call in requested:
            name = call["name"]
            if name not in tools:
                raise ValueError("unknown tool")
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
            trainable.append(False)
    return {"messages": messages, "trainable": trainable, "turns": turns,
            "tool_calls": tool_calls, "stop_reason": stop_reason,
            "reward": float(reward_fn(messages))}
"""),
    Mutation("only_first_tool_call_runs", _ROLLOUT_SIGNATURE + _ROLLOUT_GUARD + """    messages = [dict(m) for m in initial_messages]
    trainable = [m.get("role") == "assistant" for m in messages]
    turns = tool_calls = 0
    stop_reason = "max_turns"
    while turns < max_turns:
        response = model.call(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response.get("content", "")})
        trainable.append(True)
        requested = response.get("tool_calls") or []
        if not requested:
            stop_reason = "finished"
            break
        call = requested[0]
        name = call["name"]
        if name not in tools:
            raise ValueError("unknown tool")
        result = tools[name].invoke(call.get("arguments", {}))
        tool_calls += 1
        messages.append({"role": "tool", "name": name, "content": result})
        trainable.append(False)
    return {"messages": messages, "trainable": trainable, "turns": turns,
            "tool_calls": tool_calls, "stop_reason": stop_reason,
            "reward": float(reward_fn(messages))}
"""),
    Mutation("mutates_initial_messages", _ROLLOUT_SIGNATURE + _ROLLOUT_GUARD + """    messages = initial_messages
    trainable = [m.get("role") == "assistant" for m in messages]
    turns = tool_calls = 0
    stop_reason = "max_turns"
    while turns < max_turns:
        response = model.call(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response.get("content", "")})
        trainable.append(True)
        requested = response.get("tool_calls") or []
        if not requested:
            stop_reason = "finished"
            break
        for call in requested:
            name = call["name"]
            if name not in tools:
                raise ValueError("unknown tool")
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
            trainable.append(False)
    return {"messages": messages, "trainable": trainable, "turns": turns,
            "tool_calls": tool_calls, "stop_reason": stop_reason,
            "reward": float(reward_fn(messages))}
"""),
    Mutation("continues_after_a_final_turn", _ROLLOUT_SIGNATURE + _ROLLOUT_GUARD + """    messages = [dict(m) for m in initial_messages]
    trainable = [m.get("role") == "assistant" for m in messages]
    turns = tool_calls = 0
    stop_reason = "max_turns"
    while turns < max_turns:
        response = model.call(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response.get("content", "")})
        trainable.append(True)
        requested = response.get("tool_calls") or []
        if not requested:
            stop_reason = "finished"
            continue
        for call in requested:
            name = call["name"]
            if name not in tools:
                raise ValueError("unknown tool")
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
            trainable.append(False)
    return {"messages": messages, "trainable": trainable, "turns": turns,
            "tool_calls": tool_calls, "stop_reason": stop_reason,
            "reward": float(reward_fn(messages))}
"""),
    Mutation("accepts_unknown_tool", _ROLLOUT_SIGNATURE + _ROLLOUT_GUARD + """    messages = [dict(m) for m in initial_messages]
    trainable = [m.get("role") == "assistant" for m in messages]
    turns = tool_calls = 0
    stop_reason = "max_turns"
    while turns < max_turns:
        response = model.call(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response.get("content", "")})
        trainable.append(True)
        requested = response.get("tool_calls") or []
        if not requested:
            stop_reason = "finished"
            break
        for call in requested:
            name = call["name"]
            if name not in tools:
                continue
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
            trainable.append(False)
    return {"messages": messages, "trainable": trainable, "turns": turns,
            "tool_calls": tool_calls, "stop_reason": stop_reason,
            "reward": float(reward_fn(messages))}
"""),
]


@pytest.mark.parametrize("task_id", RL_TASK_IDS)
def test_rl_task_metadata_is_valid(task_id):
    task = get_task(task_id)
    assert task is not None, f"unknown task {task_id!r}"
    validate_task(task_id, task, known_ids=None)


@pytest.mark.parametrize("task_id", RL_TASK_IDS)
def test_rl_task_has_pinned_code_provenance(task_id):
    sources = get_task(task_id)["sources"]
    code_sources = [source for source in sources if source["kind"] == "code"]
    assert code_sources, f"{task_id} needs at least one pinned code source"
    assert any(source["kind"] == "paper" for source in sources), f"{task_id} needs a paper source"


@pytest.mark.parametrize("task_id", RL_TASK_IDS)
def test_rl_task_is_english_only(task_id):
    """Scope decision 2026-09-12: this path ships English only."""
    task = get_task(task_id)
    for key in ("title_zh", "description_zh", "hint_zh"):
        assert key not in task, f"{task_id} must not carry {key}; the RL path is English only"


def test_per_token_logprobs_provenance():
    sources = {
        source["symbol"]: source
        for source in get_task("per_token_logprobs")["sources"]
        if source["kind"] == "code"
    }
    assert "compute_pg_loss" in sources
    assert sources["compute_pg_loss"]["path"] == "nano_r1_script.py"


def test_group_relative_advantage_provenance():
    sources = {
        source["symbol"]: source
        for source in get_task("group_relative_advantage")["sources"]
        if source["kind"] == "code"
    }
    assert "create_training_episodes" in sources
    assert sources["create_training_episodes"]["path"] == "nano_r1_script.py"


@pytest.mark.parametrize("_repeat", range(3))
def test_per_token_logprobs_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("per_token_logprobs", PER_TOKEN_LOGPROBS_MUTATIONS)
    ) == {m.name for m in PER_TOKEN_LOGPROBS_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_group_relative_advantage_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("group_relative_advantage", GROUP_ADVANTAGE_MUTATIONS)
    ) == {m.name for m in GROUP_ADVANTAGE_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_k3_kl_penalty_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("k3_kl_penalty", K3_KL_MUTATIONS)
    ) == {m.name for m in K3_KL_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_response_token_mask_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("response_token_mask", RESPONSE_MASK_MUTATIONS)
    ) == {m.name for m in RESPONSE_MASK_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_rlvr_format_reward_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("rlvr_format_reward", FORMAT_REWARD_MUTATIONS)
    ) == {m.name for m in FORMAT_REWARD_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_grpo_token_loss_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("grpo_token_loss", GRPO_TOKEN_LOSS_MUTATIONS)
    ) == {m.name for m in GRPO_TOKEN_LOSS_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_ppo_clipped_policy_loss_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("ppo_clipped_policy_loss", PPO_CLIPPED_MUTATIONS)
    ) == {m.name for m in PPO_CLIPPED_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_ppo_value_loss_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("ppo_value_loss", PPO_VALUE_MUTATIONS)
    ) == {m.name for m in PPO_VALUE_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_gae_advantage_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("gae_advantage", GAE_MUTATIONS)
    ) == {m.name for m in GAE_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_gspo_sequence_ratio_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("gspo_sequence_ratio", GSPO_MUTATIONS)
    ) == {m.name for m in GSPO_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_dapo_dynamic_sampling_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("dapo_dynamic_sampling", DAPO_MUTATIONS)
    ) == {m.name for m in DAPO_MUTATIONS}


def test_gae_advantage_does_not_collide_with_the_graph_autoencoder():
    """Naming constraint 2026-09-12: the id `gae` belongs to the Graph Autoencoder."""
    graph_autoencoder = get_task("gae")
    assert graph_autoencoder is not None
    assert graph_autoencoder["function_name"] == "gae"
    rl_estimator = get_task("gae_advantage")
    assert rl_estimator["function_name"] == "gae_advantage"
    assert "Graph Autoencoder" in rl_estimator["description_en"]


@pytest.mark.parametrize("_repeat", range(3))
def test_vineppo_mc_value_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("vineppo_mc_value", VINEPPO_MUTATIONS)
    ) == {m.name for m in VINEPPO_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_rollout_batch_assembly_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("rollout_batch_assembly", ROLLOUT_ASSEMBLY_MUTATIONS)
    ) == {m.name for m in ROLLOUT_ASSEMBLY_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_grpo_train_step_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("grpo_train_step", GRPO_TRAIN_STEP_MUTATIONS)
    ) == {m.name for m in GRPO_TRAIN_STEP_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_agentic_rollout_loop_reference_and_mutations(_repeat):
    assert set(
        assert_mutations_rejected("agentic_rollout_loop", AGENTIC_ROLLOUT_MUTATIONS)
    ) == {m.name for m in AGENTIC_ROLLOUT_MUTATIONS}


def test_the_rl_path_is_complete():
    """All 15 tickets of the 2026-09-12 backlog: P1-P5, S1-S8 (S9 cut), I1-I2."""
    import json

    paths = json.loads(
        (Path(__file__).resolve().parents[2] / "web/src/lib/paths.json").read_text()
    )["paths"]
    listed = [p for p in paths if p["id"] == "rl-posttraining"]
    assert listed, "the rl-posttraining path is not registered"
    assert listed[0]["problems"] == RL_TASK_IDS, (
        "the path order must match the backlog's primitive -> subsystem -> integrative "
        "progression"
    )
    assert len(RL_TASK_IDS) == 15


def test_rl_harness_never_imports_a_task_solution():
    """The harness must be independent of the answers it grades."""
    import torch_judge.harness.rl.tiny_policy as tiny_policy

    source = Path(tiny_policy.__file__).read_text()
    assert "torch_judge.tasks" not in source, "the RL harness must not import task solutions"


def test_rl_harness_fixtures_are_deterministic():
    import torch

    from torch_judge.harness.rl import TinyPolicy, seeded_rollout_batch

    first, second = seeded_rollout_batch(seed=11), seeded_rollout_batch(seed=11)
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["mask"], second["mask"])
    assert torch.equal(first["advantages"], second["advantages"])
    # The batch must be genuinely ragged, or the masking mutations cannot be caught.
    assert len(set(first["mask"].sum(dim=-1).tolist())) > 1

    a, b = TinyPolicy(seed=3), TinyPolicy(seed=3)
    for (_, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p.detach(), q.detach())


def test_layer_two_is_complete():
    """S1 to S8 of the 2026-09-12 backlog, minus the cut S9."""
    subsystem_ids = {
        "grpo_token_loss",
        "ppo_clipped_policy_loss",
        "ppo_value_loss",
        "gae_advantage",
        "gspo_sequence_ratio",
        "dapo_dynamic_sampling",
        "vineppo_mc_value",
        "rollout_batch_assembly",
    }
    assert subsystem_ids <= set(RL_TASK_IDS)
    for task_id in subsystem_ids:
        assert get_task(task_id) is not None, task_id
    # S9 pairwise_reward_loss was cut as a duplicate of the existing reward_model.
    assert get_task("pairwise_reward_loss") is None
    assert get_task("reward_model") is not None


def test_ppo_family_shares_one_reduction_convention():
    """The three PPO-branch losses must all reduce over the same masked token count."""
    for task_id in ("grpo_token_loss", "ppo_clipped_policy_loss", "ppo_value_loss"):
        description = get_task(task_id)["description_en"]
        assert "mask" in description, task_id
        assert "zero scalar" in description, f"{task_id} must define the empty-mask case"


def test_ppo_clipped_is_distinct_from_the_legacy_sequence_level_exercise():
    """Overlap audit 2026-09-12: this must not restate the existing ppo_loss."""
    new = get_task("ppo_clipped_policy_loss")
    legacy = get_task("ppo_loss")
    assert new["function_name"] != legacy["function_name"]
    # The token-level contract is what separates them: a mask and a dual-clip bound.
    description = new["description_en"]
    assert "mask" in description and "dual_clip" in description
    assert "mask" not in legacy["description_en"]
    assert "ppo_loss" in new["advisory_prerequisites"]


def test_grpo_token_loss_builds_on_the_primitives():
    """S1 is the convergence point of P1 to P4; its prerequisites must say so."""
    prerequisites = set(get_task("grpo_token_loss")["advisory_prerequisites"])
    assert prerequisites == {
        "per_token_logprobs",
        "group_relative_advantage",
        "k3_kl_penalty",
        "response_token_mask",
    }
