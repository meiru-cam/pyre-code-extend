"""Deterministic tiny policy and rollout fixtures for RL post-training exercises.

Every object here is seeded and CPU-only so an evaluator can assert an exact
parameter delta rather than a plausible-looking metric. Nothing in this module
imports a task solution.
"""

from __future__ import annotations

import torch
from torch import nn

from torch_judge.harness import HarnessFailure


class TinyPolicy(nn.Module):
    """A few-thousand-parameter causal language model with deterministic weights.

    The forward pass maps token ids to per-position vocabulary logits, which is
    the only interface the RL exercises need. There is no attention and no
    causal structure: positions are independent, so a learner cannot accidentally
    pass an exercise by exploiting sequence structure the contract never mentions.
    """

    def __init__(self, vocab_size: int = 8, hidden: int = 4, seed: int = 0) -> None:
        super().__init__()
        if vocab_size < 1 or hidden < 1:
            raise HarnessFailure(
                f"vocab_size and hidden must be positive, got {vocab_size} and {hidden}"
            )
        generator = torch.Generator().manual_seed(seed)
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(vocab_size, hidden, generator=generator) * 0.5
            )
            self.head.weight.copy_(
                torch.randn(vocab_size, hidden, generator=generator) * 0.5
            )
            self.head.bias.copy_(torch.randn(vocab_size, generator=generator) * 0.1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(input_ids))


def frozen_copy(model: TinyPolicy) -> TinyPolicy:
    """Return a detached, eval-mode clone usable as a reference model."""
    vocab_size, hidden = model.embedding.weight.shape
    clone = TinyPolicy(vocab_size=vocab_size, hidden=hidden)
    clone.load_state_dict(model.state_dict())
    for parameter in clone.parameters():
        parameter.requires_grad_(False)
    clone.eval()
    return clone


def drifted_reference(model: TinyPolicy, seed: int = 1, scale: float = 0.3) -> TinyPolicy:
    """A frozen reference whose weights differ from ``model``.

    ``frozen_copy`` produces a reference that agrees with the policy everywhere, so
    the KL term is identically zero. That is the correct state at the start of
    training and the wrong fixture for testing that the KL term is wired up at all.
    """
    clone = frozen_copy(model)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in clone.parameters():
            parameter.add_(
                torch.randn(parameter.shape, generator=generator) * scale
            )
    return clone


def seeded_rollout_batch(
    batch_size: int = 4,
    seq_len: int = 5,
    vocab_size: int = 8,
    prompt_len: int = 2,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a deterministic padded rollout batch.

    Returns ``input_ids`` ``(B, T)``, a boolean ``mask`` ``(B, T)`` that is True
    only on generated positions, and per-response ``advantages`` ``(B,)`` that are
    already standardized to sum to approximately zero.
    """
    if prompt_len >= seq_len:
        raise HarnessFailure(
            f"prompt_len {prompt_len} must be smaller than seq_len {seq_len}"
        )
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)

    # Ragged generation lengths, but always at least one generated token. Cycling
    # the available lengths guarantees the batch is genuinely ragged whenever it is
    # wide enough, rather than leaving it to a lucky seed.
    span = seq_len - prompt_len
    gen_lens = torch.tensor([(i % span) + 1 for i in range(batch_size)])
    total_lens = gen_lens + prompt_len
    positions = torch.arange(seq_len).unsqueeze(0)
    mask = (positions >= prompt_len) & (positions < total_lens.unsqueeze(-1))

    raw = torch.randn(batch_size, generator=generator)
    advantages = raw - raw.mean()
    return {"input_ids": input_ids, "mask": mask, "advantages": advantages}


def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached clones of every parameter, for an exact before-and-after delta."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def assert_parameters_changed(
    before: dict[str, torch.Tensor], model: nn.Module, message: str
) -> None:
    """Require at least one parameter to have moved."""
    moved = any(
        not torch.equal(before[name], p.detach())
        for name, p in model.named_parameters()
        if name in before
    )
    if not moved:
        raise AssertionError(message)


def assert_parameters_unchanged(
    before: dict[str, torch.Tensor], model: nn.Module, message: str
) -> None:
    """Require every parameter to be exactly where it was."""
    for name, parameter in model.named_parameters():
        if name in before and not torch.equal(before[name], parameter.detach()):
            raise AssertionError(f"{message} (parameter {name!r} moved)")


def reference_grpo_step(
    model: TinyPolicy,
    ref_model: TinyPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    beta: float = 0.04,
) -> dict[str, torch.Tensor]:
    """An independent implementation of one GRPO step, used only as an oracle.

    This mirrors the exercise contract but is written from the formula rather than
    from the reference solution, so a learner's step can be compared against an
    exact parameter delta produced outside their code.
    """
    input_ids = batch["input_ids"]
    mask = batch["mask"]
    advantages = batch["advantages"]

    optimizer.zero_grad()
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
    objective = advantages.unsqueeze(-1) * logprobs - beta * kl
    loss = -(objective * keep).sum() / denominator
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.detach(),
        "pg_loss": (-(advantages.unsqueeze(-1) * logprobs) * keep).sum().detach()
        / denominator,
        "kl": (kl * keep).sum().detach() / denominator,
        "num_tokens": denominator.detach(),
    }
