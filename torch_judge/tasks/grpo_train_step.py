"""One complete GRPO optimization step — the capstone of the RL post-training path."""

TASK = {
    "title": "One Complete GRPO Training Step",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "grpo_train_step",
    "description_en": r"""Run one full GRPO optimization step: forward both models, build the loss, and update the policy exactly once.

**Signature:** `grpo_train_step(model, ref_model, optimizer, batch, beta=0.04) -> dict`

**Parameters:**
- `model` — the policy being trained. `model(input_ids)` returns logits of shape `(B, T, V)`.
- `ref_model` — a frozen reference model with the same interface.
- `optimizer` — a PyTorch optimizer already bound to the policy's parameters.
- `batch` — a dictionary with `input_ids` `(B, T)` integer, `mask` `(B, T)` boolean, and `advantages` `(B,)` float.
- `beta` — non-negative KL coefficient.

**Returns:** a dictionary with exactly the keys `loss`, `pg_loss`, `kl` and `num_tokens`, each a **detached** scalar tensor.

**The step, in order:**

1. `optimizer.zero_grad()`.
2. Forward the policy once: `logits = model(input_ids)`.
3. Gather per-token log-probabilities of the realized tokens: log-softmax over the vocabulary, then select `input_ids`. The tensors are already aligned; do not shift.
4. Forward the reference model once under `torch.no_grad()` and gather its log-probabilities the same way.
5. Build the loss, with `keep` the mask cast to the log-probability dtype and `denominator = keep.sum()`:

       kl        = exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
       objective = advantages.unsqueeze(-1) * logprobs - beta * kl
       loss      = -(objective * keep).sum() / denominator

6. `loss.backward()` once.
7. `optimizer.step()` once.
8. Return the metrics, each detached:

       pg_loss    = (-(advantages.unsqueeze(-1) * logprobs) * keep).sum() / denominator
       kl         = (kl * keep).sum() / denominator
       num_tokens = denominator

**Constraints:**
- Call `zero_grad` before the forward pass, exactly once.
- Exactly one forward pass of each model, one `backward`, one `step`.
- Wrap only the reference forward in `torch.no_grad()`, covering both the forward call and the gather.
- Detach each metric when building the return dictionary — not earlier, while building the loss.
- The reported `kl` is the unweighted penalty. Do not scale it by `beta`.
- `ref_model` parameters are unchanged and carry no gradient afterwards.
- Raise a `ValueError` when `batch` is missing any of the three keys, or when `beta` is negative.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Five ways this goes wrong, none of which changes a shape.**

**Forgetting `zero_grad`.** The first step looks perfect. The second accumulates the first step's gradients on top of its own, so the update is wrong from step two onward and the loss curve still descends.

**Stepping twice, or backward twice.** One batch is one update. A duplicated step doubles the effective learning rate silently.

**Running the reference model with gradients.** Its parameters do not move if the optimizer is not bound to them, but the graph still grows and the KL term stops being a constant anchor.

**Returning attached tensors.** Holding a metric that still carries a graph keeps the whole step's activations alive, and memory grows every iteration until the process dies.

**Detaching too early.** Detaching the log-probabilities before building the loss produces a scalar that looks right and a gradient that is identically zero, so the model never moves at all.

The last two are opposite mistakes and both are silent, which is why the evaluator checks an exact parameter delta against an independently written step rather than accepting a plausible-looking loss value.""",
    "advisory_prerequisites": [
        "grpo_token_loss",
        "per_token_logprobs",
        "k3_kl_penalty",
        "rollout_batch_assembly",
        "adam",
    ],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Write the seven steps down in order before writing any code, then ask of each one: how many times does it run? Which of the two models needs a gradient, and what context manager expresses that for the other one? When you build the metrics dictionary, ask what each tensor is still holding onto — and separately, ask what would break if you detached those same tensors one step earlier, while building the loss. Finally: if you ran your step twice on the same batch, what would have to be true for the second update to be identical to the first?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "The body is short; the discipline is in the ordering. Start with `optimizer.zero_grad()` — before the forward pass, not after the backward — so a caller that loops sees a clean slate each iteration. Forward the policy normally, and wrap only the reference forward in `with torch.no_grad():`; that block must cover both the forward call and the gather, or the gather re-attaches the result. Build the loss exactly as `grpo_token_loss` does, reusing `keep` and `denominator` so the metrics and the loss share one denominator. Call `backward` once on the total loss and `step` once. Only when constructing the return dictionary, call `.detach()` on each scalar — the two failure modes here are opposite and both silent, so detach late and detach everything. Note that `pg_loss` and `kl` are diagnostics recomputed from tensors you already hold; they are not summed into the loss a second time.",
        },
    ],
    "model_connections": [
        "simple_GRPO's GRPO_step performs this exact sequence with a reference model served from a separate process, which is why its reference log-probabilities arrive already detached.",
        "nano-aha-moment's training loop calls compute_pg_loss inside a gradient-accumulation loop and steps once per accumulation window, which is the same discipline spread across micro-batches.",
        "OpenRLHF wraps the step in a Ray actor with the reference model on its own device; the no-grad boundary around the reference forward is what makes that split possible.",
        "verl and slime both separate the rollout worker from the training worker, so this function is the whole of what the training worker does per batch.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One function holds the entire update, so the ordering discipline is visible in one place rather than spread across a training loop.",
            "Returning detached metrics makes the step safe to call in a loop without leaking activations.",
            "Sharing one denominator between the loss and the diagnostics means a reported KL is the same number the loss used.",
        ],
        "cons": [
            "A single step hides gradient accumulation, clipping, learning-rate scheduling and mixed precision, all of which a real loop needs.",
            "Keeping a full reference model in memory doubles the parameter footprint, which is why production stacks put it on another device or drop it entirely.",
            "The metrics are diagnostics only; a descending loss with healthy KL still says nothing about whether the reward is going up.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "compute_pg_loss",
            "license": "MIT",
            "adapted": "The policy and reference forward passes, the masked per-token objective with a k3 KL penalty, and the detached metric dictionary returned to the training loop.",
            "simplifications": "One step over one batch with a plain optimizer: no gradient accumulation, no distributed reduction, no mixed precision, no temperature and no inference engine.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.03300",
            "section": "4.1 From PPO to GRPO, iterative training procedure",
        },
    ],
    "tests": [
        {
            "name": "Metrics contract and a single update",
            "behavior": "gradient.flow",
            "code": r"""
import torch
from torch_judge.harness.rl import TinyPolicy, drifted_reference, seeded_rollout_batch, parameter_snapshot

model = TinyPolicy()
ref = drifted_reference(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
batch = seeded_rollout_batch()
before = parameter_snapshot(model)

metrics = {fn}(model, ref, optimizer, batch, beta=0.04)
assert set(metrics) == {'loss', 'pg_loss', 'kl', 'num_tokens'}, sorted(metrics)
for key, value in metrics.items():
    assert isinstance(value, torch.Tensor), f'{key} is {type(value).__name__}'
    assert value.ndim == 0, f'{key} has shape {tuple(value.shape)}'
    assert not value.requires_grad, f'{key} is still attached to the graph'
assert int(metrics['num_tokens']) == int(batch['mask'].sum()), metrics['num_tokens']
assert any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()), 'the policy did not move'
""",
        },
        {
            "name": "The reference model is frozen",
            "behavior": "state.invariant",
            "code": r"""
import torch
from torch_judge.harness.rl import TinyPolicy, drifted_reference, seeded_rollout_batch, parameter_snapshot, assert_parameters_unchanged

model = TinyPolicy()
ref = drifted_reference(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
before_ref = parameter_snapshot(ref)
{fn}(model, ref, optimizer, seeded_rollout_batch(), beta=0.04)
assert_parameters_unchanged(before_ref, ref, 'the reference model must not be updated')
for name, parameter in ref.named_parameters():
    assert parameter.grad is None, f'reference parameter {name} received a gradient'
""",
        },
        {
            "name": "Matches an independent step exactly",
            "visibility": "unshown",
            "behavior": "gradient.flow",
            "failure_message": "The parameter delta disagrees with an independently written GRPO step. Check the loss formula, the denominator, and that exactly one optimizer step runs.",
            "code": r"""
import torch
from torch_judge.harness.rl import (
    TinyPolicy, drifted_reference, seeded_rollout_batch, reference_grpo_step
)

for seed, beta in ((0, 0.04), (3, 0.0), (7, 0.5)):
    batch = seeded_rollout_batch(seed=seed)
    mine = TinyPolicy(seed=seed)
    theirs = TinyPolicy(seed=seed)
    ref_a = drifted_reference(mine)
    ref_b = drifted_reference(theirs)
    opt_a = torch.optim.SGD(mine.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(theirs.parameters(), lr=0.1)

    got = {fn}(mine, ref_a, opt_a, batch, beta=beta)
    want = reference_grpo_step(theirs, ref_b, opt_b, batch, beta=beta)

    assert torch.allclose(got['loss'], want['loss'], atol=1e-6), f"loss {got['loss'].item()} vs {want['loss'].item()}"
    assert torch.allclose(got['kl'], want['kl'], atol=1e-6), f"kl {got['kl'].item()} vs {want['kl'].item()}"
    assert torch.allclose(got['pg_loss'], want['pg_loss'], atol=1e-6), f"pg_loss {got['pg_loss'].item()} vs {want['pg_loss'].item()}"
    for (name, a), (_, b) in zip(mine.named_parameters(), theirs.named_parameters()):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-6), f'parameter {name} differs after the step'
""",
        },
        {
            "name": "Gradients are cleared between steps",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "A second step on the same batch did not reproduce an independent second step. The most likely cause is a missing optimizer.zero_grad() before the forward pass.",
            "code": r"""
import torch
from torch_judge.harness.rl import (
    TinyPolicy, drifted_reference, seeded_rollout_batch, reference_grpo_step
)

batch = seeded_rollout_batch(seed=2)
mine = TinyPolicy(seed=2)
theirs = TinyPolicy(seed=2)
ref_a = drifted_reference(mine)
ref_b = drifted_reference(theirs)
opt_a = torch.optim.SGD(mine.parameters(), lr=0.1)
opt_b = torch.optim.SGD(theirs.parameters(), lr=0.1)

for step in range(3):
    {fn}(mine, ref_a, opt_a, batch, beta=0.04)
    reference_grpo_step(theirs, ref_b, opt_b, batch, beta=0.04)
    for (name, a), (_, b) in zip(mine.named_parameters(), theirs.named_parameters()):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-5), f'step {step}: parameter {name} diverged'
""",
        },
        {
            "name": "Exactly one optimizer step per call",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The optimizer was stepped more or fewer than once. One batch is one update.",
            "code": r"""
import torch
from torch_judge.harness.rl import TinyPolicy, drifted_reference, seeded_rollout_batch

class CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.zeroed = 0
    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)
    def zero_grad(self, *args, **kwargs):
        self.zeroed += 1
        return super().zero_grad(*args, **kwargs)

model = TinyPolicy()
ref = drifted_reference(model)
optimizer = CountingSGD(model.parameters(), lr=0.1)
{fn}(model, ref, optimizer, seeded_rollout_batch(), beta=0.04)
assert optimizer.steps == 1, f'expected 1 optimizer step, got {optimizer.steps}'
assert optimizer.zeroed == 1, f'expected 1 zero_grad call, got {optimizer.zeroed}'
""",
        },
        {
            "name": "The loss actually drives the parameters",
            "visibility": "unshown",
            "behavior": "gradient.flow",
            "failure_message": "The policy did not move, or moved without regard to the advantages. Detaching the log-probabilities before the loss gives a plausible scalar and a zero gradient.",
            "code": r"""
import torch
from torch_judge.harness.rl import TinyPolicy, drifted_reference, seeded_rollout_batch, parameter_snapshot

batch = seeded_rollout_batch(seed=5)
model = TinyPolicy(seed=5)
ref = drifted_reference(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
before = parameter_snapshot(model)
{fn}(model, ref, optimizer, batch, beta=0.04)
moved = {n: (before[n] - p.detach()).abs().max() for n, p in model.named_parameters()}
assert max(moved.values()) > 1e-6, f'no parameter moved: {moved}'

# Flipping the sign of every advantage must move the policy the other way.
flipped = dict(batch)
flipped['advantages'] = -batch['advantages']
model_b = TinyPolicy(seed=5)
ref_b = drifted_reference(model_b)
opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
{fn}(model_b, ref_b, opt_b, flipped, beta=0.04)
same = all(torch.allclose(a.detach(), b.detach(), atol=1e-7)
           for (_, a), (_, b) in zip(model.named_parameters(), model_b.named_parameters()))
assert not same, 'flipping the advantages produced an identical update; they are not reaching the loss'
""",
        },
        {
            "name": "beta controls the KL contribution",
            "visibility": "unshown",
            "behavior": "rl.kl_estimator",
            "failure_message": "The beta coefficient did not change the loss, or the reported kl depends on beta. kl is the unweighted penalty; beta weights it inside the loss only.",
            "code": r"""
import torch
from torch_judge.harness.rl import TinyPolicy, drifted_reference, seeded_rollout_batch

batch = seeded_rollout_batch(seed=4)
results = {}
for beta in (0.0, 0.5):
    model = TinyPolicy(seed=4)
    ref = drifted_reference(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    results[beta] = {fn}(model, ref, optimizer, batch, beta=beta)

assert not torch.allclose(results[0.0]['loss'], results[0.5]['loss'], atol=1e-6), 'beta did not affect the loss'
# The reported KL is the raw penalty, so it does not depend on beta.
assert torch.allclose(results[0.0]['kl'], results[0.5]['kl'], atol=1e-6), 'the reported kl should not be scaled by beta'
assert float(results[0.0]['kl']) > 0, 'the drifted reference should produce a positive KL'
# With beta = 0 the loss is exactly the policy-gradient term.
assert torch.allclose(results[0.0]['loss'], results[0.0]['pg_loss'], atol=1e-6), 'at beta=0 loss must equal pg_loss'
""",
        },
        {
            "name": "Metrics do not retain the graph",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "A returned metric still carries grad_fn. Detach each scalar when building the dictionary, not before building the loss.",
            "code": r"""
import torch
from torch_judge.harness.rl import TinyPolicy, drifted_reference, seeded_rollout_batch

model = TinyPolicy()
ref = drifted_reference(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
metrics = {fn}(model, ref, optimizer, seeded_rollout_batch(), beta=0.04)
for key, value in metrics.items():
    assert value.grad_fn is None, f'{key} still carries grad_fn {value.grad_fn}'
    assert not value.requires_grad, f'{key} still requires grad'
""",
        },
        {
            "name": "Rejects a malformed batch or a negative beta",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "A malformed batch or a negative beta was accepted. Validate the three required keys and the sign of beta.",
            "code": r"""
import torch
from torch_judge.harness.rl import TinyPolicy, drifted_reference, seeded_rollout_batch

model = TinyPolicy()
ref = drifted_reference(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
full = seeded_rollout_batch()

for missing in ('input_ids', 'mask', 'advantages'):
    partial = {k: v for k, v in full.items() if k != missing}
    try:
        {fn}(model, ref, optimizer, partial, beta=0.04)
    except ValueError:
        continue
    raise AssertionError(f'a batch missing {missing!r} should raise ValueError')

try:
    {fn}(model, ref, optimizer, full, beta=-0.1)
except ValueError:
    pass
else:
    raise AssertionError('a negative beta should raise ValueError')
""",
        },
    ],
    "solution": '''import torch


def grpo_train_step(model, ref_model, optimizer, batch, beta=0.04):
    required = ("input_ids", "mask", "advantages")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"batch is missing {missing}")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")

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
    weighted_logprobs = advantages.unsqueeze(-1) * logprobs
    objective = weighted_logprobs - beta * kl
    loss = -(objective * keep).sum() / denominator

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.detach(),
        "pg_loss": (-(weighted_logprobs * keep).sum() / denominator).detach(),
        "kl": ((kl * keep).sum() / denominator).detach(),
        "num_tokens": denominator.detach(),
    }
''',
}
