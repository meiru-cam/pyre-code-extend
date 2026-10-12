"""VLA action adapter: turn a padded, normalized, delta action chunk into the command a robot env accepts."""

from ._interview import interview

# Plain-loop model of the contract, written independently of the reference.
_MODEL = r"""
import math
import torch

def m_adapt(actions, q01, q99, state, mask, binarize=True, invert=True):
    D = len(q01)
    H, P = actions.shape[-2], actions.shape[-1]
    lead = tuple(actions.shape[:-2])
    flat_a = actions.reshape(-1, H, P).tolist()
    flat_s = state.reshape(-1, D).tolist()
    lo, hi, m = q01.tolist(), q99.tolist(), mask.tolist()
    out = []
    for chunk, s in zip(flat_a, flat_s):
        rows = []
        for step in chunk:
            row = []
            for d in range(D):
                v = (step[d] + 1.0) / 2.0 * (hi[d] - lo[d] + 1e-6) + lo[d]
                if m[d]:
                    v = v + s[d]
                row.append(v)
            g = 2.0 * row[-1] - 1.0
            if binarize:
                g = 0.0 if g == 0 else math.copysign(1.0, g)
            if invert:
                g = -g
            row[-1] = g
            rows.append(row)
        out.append(rows)
    return torch.tensor(out, dtype=actions.dtype).reshape(lead + (H, D))

def m_case(g, lead, H, D, P, dtype=torch.float64):
    actions = torch.rand(lead + (H, P), generator=g, dtype=dtype) * 2.4 - 1.2
    q01 = torch.rand(D, generator=g, dtype=dtype) * 2 - 1.5
    q99 = q01 + torch.rand(D, generator=g, dtype=dtype) * 2 + 0.1
    q01[-1], q99[-1] = 0.0, 1.0
    state = torch.randn(lead + (D,), generator=g, dtype=dtype) * 3
    mask = torch.rand(D, generator=g) < 0.6
    mask[-1] = False
    return actions, q01, q99, state, mask
"""

TASK = {
    "title": "VLA Action Adapter: Model Output to Env Command",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "adapt_actions",
    "description_en": r"""Convert an action chunk predicted by a VLA policy into the commands a LIBERO-style robot environment accepts: unnormalize, drop padding, turn deltas into absolute targets, and convert the gripper.

**Signature:** `adapt_actions(actions, q01, q99, state, delta_mask, binarize_gripper=True, invert_gripper=True) -> Tensor`

**Parameters:**
- `actions` — float tensor `(..., H, P)`: one or more chunks of `H` normalized actions from the model, padded to the model width `P`.
- `q01`, `q99` — float tensors `(D,)`: per-dimension 1st and 99th percentiles of the robot's real actions, with `1 <= D <= P`.
- `state` — float tensor `(..., D)` with the same leading dimensions as `actions`: the robot state when each chunk was predicted, in env units.
- `delta_mask` — bool tensor `(D,)`: `True` where the model predicts a change relative to `state`.
- `binarize_gripper`, `invert_gripper` — bools.

**Returns:** a new float tensor `(..., H, D)` with the dtype of `actions`. Apply these steps in order:
- Keep the first `D` of the `P` model dims and drop the rest (they are padding). Unnormalize each kept dim:

        x = (a + 1) / 2 * (q99 - q01 + 1e-6) + q01

- Delta to absolute: on every dim where `delta_mask` is `True`, add `state[d]`. Each chunk has one state, added to all `H` steps of that chunk.
- Gripper: the last of the `D` dims, index `D - 1`, is the gripper, in `[0, 1]` with 0 = close and 1 = open. Map it to `g = 2 * x - 1`. If `binarize_gripper`, replace `g` with `sign(g)`, where `sign(0) = 0`. If `invert_gripper`, replace `g` with `-g`.

**Constraints:**
- A gripper value of exactly 0.5 gives `g = 0`, and binarizing keeps it at 0. It never becomes +1 or -1.
- Take the sign after mapping to `[-1, 1]`, never of the raw `[0, 1]` value.
- Do not clip anything. Model outputs outside `[-1, 1]` unnormalize to values outside `[q01, q99]`.
- Never compute statistics from `actions`; use the given `q01` and `q99`.
- Do not modify any input tensor.
- All float inputs share the dtype of `actions`.
- Raise `ValueError` if `actions` has fewer than 2 dims, `q01`, `q99` or `delta_mask` is not `(D,)`, `P < D`, or `state.shape != actions.shape[:-2] + (D,)`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why an adapter.** A VLA is trained on many robots at once, so its raw output is in a shared, normalized and padded space. openpi's pi0 model config defaults to `action_dim = 32` numbers per step, while a LIBERO action has 7: six end-effector deltas and a gripper. The env cannot read the raw model output directly.

**Unnormalize.** openpi uses quantile normalization for every model type except pi0, which uses z-scores. Its quantile inverse is the formula above. When the stats are narrower than the model output, the extra padded dims pass through unchanged and are sliced off later by the robot's output transform (`LiberoOutputs` keeps `[..., :7]`). The `1e-6` matches the one in the forward normalization, `(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`, so a round trip returns the original action up to float rounding.

**Delta to absolute.** Some openpi configs train on joint deltas relative to the current state and convert back with `AbsoluteActions`. That transform adds the state to the masked dims of every step in the chunk. A mask such as `make_bool_mask(6, -1)` covers six joints and leaves the gripper absolute. The state is the one at prediction time, so later steps in the chunk are offsets from that same state, not from each other.

**The gripper.** SimpleVLA-RL's docstring explains that the RLDS data loader aligns the gripper as 0 = close, 1 = open, while the environments it targets expect -1 = open, +1 = close. Its LIBERO rollout converts every action with `normalize_gripper_action(a, binarize=True)` and then `invert_gripper_action`. Binarizing uses `np.sign`, so an output of exactly 0.5 becomes 0, a command that is neither open nor close. Flipping commutes with sign, `sign(-g) = -sign(g)`, so it does not matter whether you flip before or after binarizing.

**Where this chain comes from.** The steps chain two codebases. openpi's output transforms do the first three, and SimpleVLA-RL's LIBERO rollout does the gripper step. openpi's own LIBERO example sends the continuous gripper value unchanged. Neither project clips the action in these steps.""",
    "advisory_prerequisites": ["vla_action_bins", "pi0_flow_matching_chunk"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which of the P model dims carry real actions, and when can you drop the others? What shape must the state take so that one state is added to all H steps of its chunk? What is sign(0), and what gripper value produces it?"},
        {"level": 2, "kind": "analysis", "content": "Validate the shapes first. Slice actions to the first D dims and apply the unnormalize formula with broadcasting. Build the delta offset as state where delta_mask is True and 0 elsewhere, unsqueeze it at dim -2, and add it. Then rebuild the last dim as 2x - 1, apply torch.sign if binarizing and negate if inverting. Return a new tensor."},
    ],
    "model_connections": [
        "openpi's inference output pipeline runs Unnormalize, then AbsoluteActions (when the data config uses delta actions), then a robot-specific output transform such as LiberoOutputs that keeps the first 7 dims.",
        "SimpleVLA-RL's LIBERO rollout maps the gripper from [0, 1] to [-1, 1], binarizes it with np.sign and flips its sign before every env.step.",
        "OpenVLA's predict_action also unnormalizes with q01/q99, but uses 0.5 * (n + 1) * (q99 - q01) + q01 with no epsilon and a per-dimension mask that leaves unmasked dims normalized.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One shared padded, normalized action space lets a single model train on robots with different action dims and units.",
            "Delta actions relative to the current state are often easier to learn than absolute joint targets, and the conversion back is exact.",
        ],
        "cons": [
            "Every robot needs correct statistics, a correct delta mask and the right gripper convention. A wrong sign or an off-by-one slice does not raise; the robot just moves wrong.",
            "Binarizing the gripper throws away continuous grasp control, and sign(0) = 0 sends a neutral command for an output of exactly 0.5.",
        ],
    },
    "sources": [
        {'kind': 'code',
         'url': 'https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py',
         'commit': '215abfb217dbac7d5f1273282331b9b1866c0479',
         'path': 'src/openpi/transforms.py',
         'symbol': 'Unnormalize._unnormalize_quantile, AbsoluteActions, make_bool_mask, pad_to_dim',
         'license': 'Apache-2.0',
         'adapted': 'Quantile unnormalize (x + 1) / 2 * (q99 - q01 + 1e-6) + q01 on the first D dims, then the state added '
                    'on masked dims, broadcast over the chunk axis, in torch.',
         'simplifications': 'Plain function on tensors instead of dataclass transforms over a data dict; padded dims are '
                            'dropped at once instead of passed through Unnormalize and sliced by the robot output transform; '
                            'the state arrives raw instead of round-tripped through Normalize/Unnormalize; the mask must '
                            'have length D (upstream allows a shorter mask); the result is a new tensor (upstream '
                            'AbsoluteActions adds in place).'},
        {'kind': 'code',
         'url': 'https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py',
         'commit': '215abfb217dbac7d5f1273282331b9b1866c0479',
         'path': 'src/openpi/policies/libero_policy.py',
         'symbol': 'LiberoOutputs',
         'license': 'Apache-2.0',
         'adapted': 'Keep the first D dims of the padded model output.',
         'simplifications': 'D is the length of q01 instead of the hard-coded 7.'},
        {'kind': 'code',
         'url': 'https://github.com/PRIME-RL/SimpleVLA-RL/blob/7c51662df27b586f9e8a1ab35fcf849f2b8852f9/verl/utils/libero_utils.py',
         'commit': '7c51662df27b586f9e8a1ab35fcf849f2b8852f9',
         'path': 'verl/utils/libero_utils.py',
         'symbol': 'normalize_gripper_action, invert_gripper_action',
         'license': 'MIT',
         'adapted': 'Last dim mapped 2 * x - 1, binarized with sign (sign(0) = 0) and negated, in torch, in the order '
                    'used by verl/workers/rollout/rob_rollout.py.',
         'simplifications': 'Binarize and invert are flags of one function. SimpleVLA-RL applies these steps to OpenVLA '
                            'actions unnormalized with its own masked formula; the exercise applies them after openpi\'s '
                            'quantile unnormalize, a combination neither project runs as one pipeline.'},
        {'kind': 'paper',
         'url': 'https://arxiv.org/abs/2509.09674',
         'section': 'Li et al. 2025 (SimpleVLA-RL), Section 3.1 Interactive VLA Rollout',
         'note': 'The RL framework whose LIBERO rollout applies the gripper conversion.'},
    ],
    "tests": [
        {"name": "Unnormalize, drop padding, add the delta and convert the gripper", "behavior": "tensor.shape", "code": r"""
import torch
actions = torch.tensor([[0.0, 0.0, 1.0, 99.0],
                        [1.0, -1.0, -1.0, -99.0]], dtype=torch.float64)   # H=2, P=4
q01 = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
q99 = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
state = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
delta_mask = torch.tensor([True, False, False])
out = {fn}(actions, q01, q99, state, delta_mask)
# dim 0: unnormalized 0 and 1, plus state 10; dim 1: absolute 1 and 0
# gripper: x = 1 -> g = 1 -> sign 1 -> flipped -1;  x = 0 -> g = -1 -> flipped +1
want = torch.tensor([[10.0, 1.0, -1.0], [11.0, 0.0, 1.0]], dtype=torch.float64)
assert out.shape == (2, 3) and out.dtype == torch.float64, (out.shape, out.dtype)
assert torch.allclose(out, want, atol=1e-5), out
"""},
        {"name": "A gripper of exactly 0.5 binarizes to 0", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
q01 = torch.tensor([0.0, 0.5], dtype=torch.float64)
q99 = torch.tensor([1.0, 1.0], dtype=torch.float64)
actions = torch.tensor([[-1.0, -1.0], [-1.0, 0.0]], dtype=torch.float64)
state = torch.zeros(2, dtype=torch.float64)
mask = torch.tensor([False, False])
# gripper: a = -1 unnormalizes to q01 = 0.5 exactly, so g = 0; a = 0 gives 0.75 + 5e-7, so g > 0
out = {fn}(actions, q01, q99, state, mask)
assert out[:, 1].tolist() == [0.0, -1.0], out
cont = {fn}(actions, q01, q99, state, mask, binarize_gripper=False)
assert torch.allclose(cont[:, 1], torch.tensor([0.0, -0.5], dtype=torch.float64), atol=1e-5), cont
"""},
        {"name": "Matches a seeded loop model", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Unnormalize with (a + 1) / 2 * (q99 - q01 + 1e-6) + q01, add the state on masked dims, then map the last dim to 2x - 1, sign and negate.", "code": _MODEL + r"""
for seed, lead, H, D, P in ((0, (), 5, 7, 32), (1, (3,), 4, 7, 8), (2, (2, 2), 3, 4, 4), (3, (1,), 6, 2, 5), (4, (4,), 2, 14, 32), (5, (), 1, 1, 3)):
    g = torch.Generator().manual_seed(seed)
    actions, q01, q99, state, mask = m_case(g, lead, H, D, P)
    for binarize in (True, False):
        for invert in (True, False):
            out = {fn}(actions, q01, q99, state, mask, binarize_gripper=binarize, invert_gripper=invert)
            want = m_adapt(actions, q01, q99, state, mask, binarize, invert)
            assert out.shape == lead + (H, D), (seed, out.shape)
            assert torch.allclose(out, want, atol=1e-12, rtol=0), (seed, binarize, invert, (out - want).abs().max())
"""},
        {"name": "Uses the first D model dims and drops the padding", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "The real action dims are the first D of the P model outputs; the trailing P - D dims are padding to drop.", "code": _MODEL + r"""
g = torch.Generator().manual_seed(11)
actions, q01, q99, state, mask = m_case(g, (2,), 3, 3, 6)
actions[..., 3:] = float("nan")
out = {fn}(actions, q01, q99, state, mask)
assert out.shape == (2, 3, 3), out.shape
assert torch.isfinite(out).all(), out
assert torch.allclose(out, m_adapt(actions, q01, q99, state, mask), atol=1e-12, rtol=0), out
"""},
        {"name": "The state is added only on masked dims", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Add state[d] only where delta_mask[d] is True; the other dims, including the gripper, stay absolute.", "code": r"""
import torch
g = torch.Generator().manual_seed(5)
actions = torch.rand(4, 6, generator=g, dtype=torch.float64) * 2 - 1
q01 = torch.tensor([-1.0, -2.0, 0.0, 1.0, 0.0], dtype=torch.float64)
q99 = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0], dtype=torch.float64)
mask = torch.tensor([True, False, True, False, False])
state = torch.tensor([5.0, -7.0, 0.25, 100.0, 0.75], dtype=torch.float64)
out = {fn}(actions, q01, q99, state, mask, binarize_gripper=False, invert_gripper=False)
base = {fn}(actions, q01, q99, torch.zeros(5, dtype=torch.float64), mask, binarize_gripper=False, invert_gripper=False)
diff = out - base
want = torch.tensor([5.0, 0.0, 0.25, 0.0, 0.0], dtype=torch.float64).expand(4, 5)
assert torch.allclose(diff, want, atol=1e-12), diff
"""},
        {"name": "Each chunk's one state is added to all its steps", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "state has shape (..., D), one per chunk; broadcast it over the H steps (dim -2), not over the batch.", "code": _MODEL + r"""
g = torch.Generator().manual_seed(21)
actions, q01, q99, state, mask = m_case(g, (3,), 3, 4, 6)
mask = torch.tensor([True, True, True, False])
state = torch.tensor([[1.0, 2.0, 3.0, 0.0], [10.0, 20.0, 30.0, 0.0], [100.0, 200.0, 300.0, 0.0]], dtype=torch.float64)
out = {fn}(actions, q01, q99, state, mask)
base = {fn}(actions, q01, q99, torch.zeros_like(state), mask)
for b in range(3):
    for h in range(3):
        assert torch.allclose(out[b, h, :3] - base[b, h, :3], state[b, :3], atol=1e-9), (b, h, out[b, h], base[b, h])
assert torch.allclose(out, m_adapt(actions, q01, q99, state, mask), atol=1e-12, rtol=0)
"""},
        {"name": "Gripper: sign after mapping to [-1, 1], with sign(0) = 0", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Map the gripper to 2x - 1 first, then take sign; an exact 0.5 must give 0, not +1 or -1.", "code": r"""
import torch
q01 = torch.tensor([0.0, 0.5], dtype=torch.float64)
q99 = torch.tensor([1.0, 1.5], dtype=torch.float64)
# gripper a = -1 -> x = 0.5 exactly; a = -0.6 -> x ~ 0.7; a = 1 -> x ~ 1.5 (beyond q99, not clipped)
actions = torch.tensor([[0.0, -1.0], [0.0, -0.6], [0.0, 1.0]], dtype=torch.float64)
state = torch.zeros(2, dtype=torch.float64)
mask = torch.tensor([False, False])
flip = {fn}(actions, q01, q99, state, mask)
assert flip[:, 1].tolist() == [0.0, -1.0, -1.0], flip
keep = {fn}(actions, q01, q99, state, mask, invert_gripper=False)
assert keep[:, 1].tolist() == [0.0, 1.0, 1.0], keep
q01b = torch.tensor([0.0, 0.0], dtype=torch.float64)
q99b = torch.tensor([1.0, 1.0], dtype=torch.float64)
# x = (a + 1) / 2, so a = -0.4 gives x = 0.3 and g = -0.4: sign -1, flipped to +1
low = {fn}(torch.tensor([[0.0, -0.4]], dtype=torch.float64), q01b, q99b, state, mask)
assert low[0, 1].item() == 1.0, low
"""},
        {"name": "Continuous gripper keeps 2x - 1 and the flags act independently", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "binarize_gripper only controls the sign step and invert_gripper only the negation; with both off the gripper is 2x - 1.", "code": r"""
import torch
q01 = torch.tensor([-1.0, 0.0], dtype=torch.float64)
q99 = torch.tensor([1.0, 1.0], dtype=torch.float64)
actions = torch.tensor([[0.5, -0.5], [0.5, 0.8]], dtype=torch.float64)
state = torch.zeros(2, dtype=torch.float64)
mask = torch.tensor([False, False])
x = (actions[:, 1] + 1) / 2 * (1 + 1e-6)
g = 2 * x - 1
cases = {(False, False): g, (False, True): -g, (True, False): torch.sign(g), (True, True): -torch.sign(g)}
for (binarize, invert), want in cases.items():
    out = {fn}(actions, q01, q99, state, mask, binarize_gripper=binarize, invert_gripper=invert)
    assert torch.allclose(out[:, 1], want, atol=1e-12), (binarize, invert, out)
    assert torch.allclose(out[:, 0], torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-5), out
"""},
        {"name": "No clipping: values beyond [-1, 1] extrapolate linearly", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Do not clip the model output or the unnormalized action; apply the linear formula to every value.", "code": r"""
import torch
q01 = torch.tensor([0.0, 0.0], dtype=torch.float64)
q99 = torch.tensor([2.0, 1.0], dtype=torch.float64)
actions = torch.tensor([[3.0, 0.0], [-2.0, 0.0]], dtype=torch.float64)
out = {fn}(actions, q01, q99, torch.zeros(2, dtype=torch.float64), torch.tensor([False, False]))
assert torch.allclose(out[:, 0], torch.tensor([4.0, -1.0], dtype=torch.float64), atol=1e-5), out
"""},
        {"name": "Keeps dtype and leaves the inputs untouched", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Return a new tensor in the dtype of actions and never write into actions, state or the statistics.", "code": _MODEL + r"""
g = torch.Generator().manual_seed(3)
actions, q01, q99, state, mask = m_case(g, (2,), 4, 7, 7, dtype=torch.float32)
mask[0] = True
copies = [t.clone() for t in (actions, q01, q99, state, mask)]
out = {fn}(actions, q01, q99, state, mask)
assert out.dtype == torch.float32, out.dtype
for before, after in zip(copies, (actions, q01, q99, state, mask)):
    assert torch.equal(before, after), "an input was modified"
want = m_adapt(actions.double(), q01.double(), q99.double(), state.double(), mask).float()
assert torch.allclose(out, want, atol=1e-4), (out - want).abs().max()
"""},
        {"name": "Rejects mismatched shapes", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError when P < D, a statistic or the mask is not (D,), state is not actions.shape[:-2] + (D,), or actions has fewer than 2 dims.", "code": r"""
import torch
f = torch.float64
ok = dict(actions=torch.zeros(2, 3, 5, dtype=f), q01=torch.zeros(4, dtype=f), q99=torch.ones(4, dtype=f),
          state=torch.zeros(2, 4, dtype=f), delta_mask=torch.zeros(4, dtype=torch.bool))
{fn}(**ok)
bad = [
    dict(ok, actions=torch.zeros(2, 3, 3, dtype=f)),
    dict(ok, q99=torch.ones(5, dtype=f)),
    dict(ok, q01=torch.zeros(1, 4, dtype=f), q99=torch.ones(1, 4, dtype=f)),
    dict(ok, delta_mask=torch.zeros(3, dtype=torch.bool)),
    dict(ok, state=torch.zeros(4, dtype=f)),
    dict(ok, state=torch.zeros(3, 4, dtype=f)),
    dict(ok, actions=torch.zeros(5, dtype=f), state=torch.zeros(4, dtype=f)),
]
for i, kwargs in enumerate(bad):
    try:
        {fn}(**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    ],
    "solution": '''import torch

def adapt_actions(actions, q01, q99, state, delta_mask, binarize_gripper=True, invert_gripper=True):
    if actions.ndim < 2:
        raise ValueError("actions must have shape (..., H, P)")
    if q01.ndim != 1 or q01.shape[0] < 1:
        raise ValueError("q01 must have shape (D,)")
    D = q01.shape[0]
    if q99.shape != q01.shape or delta_mask.shape != q01.shape:
        raise ValueError("q99 and delta_mask must have shape (D,)")
    if actions.shape[-1] < D:
        raise ValueError("model width P is smaller than the action dim D")
    if tuple(state.shape) != tuple(actions.shape[:-2]) + (D,):
        raise ValueError("state must have shape actions.shape[:-2] + (D,)")

    # Unnormalize the real dims; the padded tail is dropped.
    x = (actions[..., :D] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    # Delta -> absolute: one state per chunk, broadcast over the H steps.
    offset = torch.where(delta_mask, state, torch.zeros_like(state))
    x = x + offset.unsqueeze(-2)

    # Gripper: [0, 1] -> [-1, 1], optional sign (sign(0) = 0), optional flip.
    g = 2.0 * x[..., -1:] - 1.0
    if binarize_gripper:
        g = torch.sign(g)
    if invert_gripper:
        g = -g
    return torch.cat([x[..., :-1], g], dim=-1).to(actions.dtype)
''',
    "interview_questions": interview(
        concept=[
            "Why can a VLA's raw output not be sent straight to the robot, and what does each step of the adapter undo?",
            "What does a delta mask such as make_bool_mask(6, -1) mean, and why is the gripper usually left absolute?",
        ],
        deep_dive=[
            "In a chunk of H delta actions, is each step relative to the previous step or to the state at prediction time? How does that show up in the broadcast?",
            "What does np.sign return for a gripper value of exactly 0.5, and why does it matter whether you take the sign before or after mapping to [-1, 1]?",
            "Does it matter whether the gripper is flipped before or after binarizing? Why or why not?",
        ],
        tradeoffs=[
            "Binarized versus continuous gripper commands: what does each buy for RL rollouts and for fine grasping?",
            "Quantile (q01/q99) versus z-score normalization for actions: how does each handle outliers and the padded dims?",
        ],
    ),
}
