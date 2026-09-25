"""Convert completed rollouts into a token-masked RL training batch."""

TASK = {
    "title": "Rollout to Train Boundary",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "rollout_train_boundary",
    "description_en": r"""Prepare completed trajectories for a policy-gradient trainer.

**Signature:** `rollout_train_boundary(trajectories) -> dict`

Each trajectory is a dictionary with aligned lists `response_ids` and
`response_mask`, plus a non-empty per-step `rewards` list. The mask is true
only for policy-generated
tokens; false positions are environment/tool tokens. Return exactly:
`input_ids`, `loss_mask`, `advantages`, and `episode_ids`. Concatenate
trajectories in input order. For each trajectory, compute its scalar return as
the sum of `rewards`, broadcast that value to every token, then set
advantages and loss_mask to zero on non-policy positions. Use episode ids
0, 1, ... so the trainer can reconstruct boundaries. Do not normalize across
episodes and do not perform an optimizer step.

Raise `ValueError` for a missing field, mismatched id/mask lengths, or an empty
trajectory.

────────────────

**Background — context only.** This is the seam between rollout and training.
VERL's trajectory/DataProto and Slime's Data Buffer carry generated tokens plus
masks and rewards; the trainer computes the actual loss later. A tool response
is real context but not a policy action, so it stays in `input_ids` while
being excluded from `loss_mask`. Broadcasting one episode return is a
deliberately simple credit-assignment rule, not a claim that every turn was
equally useful.""",
    "advisory_prerequisites": ["agent_env_adapter", "grpo_token_loss", "rollout_batch_assembly"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which fields are concatenated, and which are derived? Does a tool token disappear from input_ids or only from the loss mask? Should rewards from one episode normalize together with another episode?"},
        {"level": 2, "kind": "analysis", "content": "Validate that ids and response_mask have the same non-zero length, and that rewards is non-empty (rewards are per environment step, not per token). Compute return = sum(rewards) once per trajectory. For each token append the id, episode id, and either return/true or zero/false according to response_mask. Keep episode order stable."},
    ],
    "model_connections": [
        "VERL's DataProto is the batch boundary between agent rollout workers and PPO/GRPO training workers.",
        "Slime's Data Buffer performs the same handoff while allowing rollout and training to run asynchronously.",
    ],
    "sources": [{
        "kind": "code",
        "url": "https://github.com/THUDM/slime",
        "commit": "5bae5bb7928d65906e24e99f5b5d3f99a4daaf93",
        "path": "slime/rollout/sglang_rollout.py",
        "symbol": "generate_rollout",
        "license": "Apache-2.0",
        "adapted": "The batch boundary keeps generated tokens, masks, rewards, and sample identity together for training.",
        "simplifications": "Uses Python lists and scalar returns instead of tensors, Data Buffer metadata, log probabilities, and Megatron batches."
    }],
    "tests": [
        {"name": "Masks environment tokens and preserves episode boundaries", "behavior": "rl.masking", "code": r"""
out = {fn}([
    {"response_ids": [1, 2, 3], "response_mask": [True, False, True], "rewards": [1.0, 2.0]},
    {"response_ids": [4, 5], "response_mask": [False, True], "rewards": [-1.0]},
])
assert out == {
    "input_ids": [1, 2, 3, 4, 5],
    "loss_mask": [True, False, True, False, True],
    "advantages": [3.0, 0.0, 3.0, 0.0, -1.0],
    "episode_ids": [0, 0, 0, 1, 1],
}
"""},
        {"name": "Does not normalize returns across episodes", "visibility": "unshown", "behavior": "rl.advantage", "failure_message": "Advantages must be per-trajectory returns broadcast over policy tokens, not a batch-wide normalization.", "code": r"""
out = {fn}([
    {"response_ids": [1], "response_mask": [True], "rewards": [10.0]},
    {"response_ids": [2], "response_mask": [True], "rewards": [20.0]},
])
assert out["advantages"] == [10.0, 20.0]
"""},
        {"name": "Seeded episodes preserve ids and independently derived advantages", "visibility": "unshown", "behavior": "rl.rollout_assembly", "failure_message": "Episode ids, token masks, and rewards must remain aligned for variable-length trajectories.", "code": r"""
import random
for seed in (11, 23, 37):
    rng = random.Random(seed)
    trajectories=[]
    for episode in range(4):
        length=rng.randrange(1, 5)
        trajectories.append({
            "response_ids": [episode * 10 + i for i in range(length)],
            "response_mask": [bool(rng.randrange(2)) for _ in range(length)],
            "rewards": [rng.randint(-3, 5) for _ in range(rng.randrange(1, 4))],
        })
    got={fn}(trajectories)
    expected={"input_ids":[], "loss_mask":[], "advantages":[], "episode_ids":[]}
    for episode, trajectory in enumerate(trajectories):
        for token, trainable in zip(trajectory["response_ids"], trajectory["response_mask"]):
            expected["input_ids"].append(token)
            expected["loss_mask"].append(trainable)
            expected["advantages"].append(sum(trajectory["rewards"]) if trainable else 0.0)
            expected["episode_ids"].append(episode)
    assert got == expected, (seed, got, expected)
"""},
        {"name": "Rejects malformed or empty trajectories", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Every trajectory must have non-empty ids and rewards, and response_ids must align with response_mask.", "code": r"""
bad = [
    {},
    {"response_ids": [1], "response_mask": [True, False], "rewards": [1]},
    {"response_ids": [], "response_mask": [], "rewards": []},
]
for item in bad:
    try: {fn}([item])
    except ValueError: pass
    else: raise AssertionError("malformed trajectory accepted")
"""},
        {"name": "Returns fresh output lists", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "The batch leaked caller-owned lists or mutated a trajectory in place.", "code": r"""
trajectory = {"response_ids": [7], "response_mask": [True], "rewards": [2.0]}
out = {fn}([trajectory])
out["input_ids"].append(99)
assert trajectory == {"response_ids": [7], "response_mask": [True], "rewards": [2.0]}
assert out["input_ids"] == [7, 99]
"""},
    ],
    "solution": '''def rollout_train_boundary(trajectories):
    output = {"input_ids": [], "loss_mask": [], "advantages": [], "episode_ids": []}
    for episode_id, trajectory in enumerate(trajectories):
        try:
            ids = trajectory["response_ids"]
            mask = trajectory["response_mask"]
            rewards = trajectory["rewards"]
        except (KeyError, TypeError):
            raise ValueError("trajectory is missing a required field")
        if not ids or not rewards or len(ids) != len(mask):
            raise ValueError("trajectory fields must be non-empty and aligned")
        total_reward = sum(float(reward) for reward in rewards)
        for token_id, trainable in zip(ids, mask):
            output["input_ids"].append(token_id)
            output["episode_ids"].append(episode_id)
            output["loss_mask"].append(bool(trainable))
            output["advantages"].append(total_reward if trainable else 0.0)
    return output
''',
}
