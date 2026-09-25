"""Filter and shard a verl-style DataProto while preserving aligned metadata."""

TASK = {
    "title": "VERL DataProto Filter and Chunk",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "verl_dataproto_filter_chunk",
    "description_en": r"""Select trainable trajectories from a VERL-style DataProto and split them for workers.

**Signature:** `verl_dataproto_filter_chunk(data, chunks) -> list[DataProto]`

`data.batch["response_mask"]` is a boolean tensor of shape (N, T).
A row is trainable if any token is True. Call `data.select_idxs(indices)`
to retain exactly those rows, then call `.chunk(chunks)` to form equal
worker shards in original order. The DataProto API carries tensor fields,
non-tensor fields such as sample ids, and meta info together. Do not rebuild
a tensor-only batch. Raise ValueError if `chunks` is not a positive integer,
if no trainable rows remain, or if the selected row count is not divisible
by `chunks`.

The grader provides a CPU `MiniDataProto` with the relevant VERL methods;
you do not need to install VERL.

────────────────────────────────

**Background — context only.** VERL's DataProto is a transport object across
rollout and training workers. Filtering only `response_mask` or only
`input_ids` silently misaligns samples, rewards and non-tensor ids. Using
`select_idxs` before `chunk` keeps every field aligned. VERL's real
`chunk` requires equal-size shards when padding is disabled.""",
    "advisory_prerequisites": ["response_token_mask", "rollout_batch_assembly"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which axis describes samples? What DataProto method selects the same rows in every field? When is an equal-size chunk valid?"},
        {"level": 2, "kind": "analysis", "content": "Reduce response_mask.any(dim=1), turn True positions into an index list, validate its size, then return data.select_idxs(indices).chunk(chunks). Do not select each tensor separately."},
    ],
    "model_connections": [
        "VERL's DataProto.select_idxs filters tensor and non-tensor data by the same sample indices.",
        "VERL's DataProto.chunk splits the selected batch into equal worker shards while preserving meta_info.",
    ],
    "sources": [
        {"kind": "code", "url": "https://github.com/verl-project/verl", "commit": "12ebe0cb4d300c58449fb6c675379e8700015c51", "path": "verl/protocol.py", "symbol": "DataProto.select_idxs; DataProto.chunk", "license": "Apache-2.0", "adapted": "Composes VERL's row-selection and equal-chunk operations on a rollout batch.", "simplifications": "A CPU MiniDataProto mimics the two methods so the exercise grades without Ray or tensordict."},
    ],
    "tests": [
        {"name": "Preserves tensor rows and sample ids in worker shards", "behavior": "rl.rollout_assembly", "code": r"""
import torch
from torch_judge.harness.rl import MiniDataProto
data = MiniDataProto(
    {"response_mask": torch.tensor([[True,False],[False,False],[True,True],[True,False],[False,False]]),
     "input_ids": torch.tensor([[10,11],[20,21],[30,31],[40,41],[50,51]])},
    {"sample_id": ["a","b","c","d","e"]}, {"rollout_version": 7})
out = {fn}(data, 3)
assert len(out) == 3
assert [part.non_tensor_batch["sample_id"] for part in out] == [["a"],["c"],["d"]]
assert [part.batch["input_ids"][0,0].item() for part in out] == [10,30,40]
assert all(part.meta_info == {"rollout_version": 7} for part in out)
""" },
        {"name": "Keeps all fields aligned after filtering", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Filtering must use DataProto.select_idxs so non-tensor ids remain aligned with tensor rows.", "code": r"""
import torch
from torch_judge.harness.rl import MiniDataProto
data = MiniDataProto(
    {"response_mask": torch.tensor([[False],[True],[True],[False],[True]]),
     "reward": torch.tensor([0., 11., 22., 0., 44.])},
    {"sample_id": ["z","a","b","y","c"]})
out = {fn}(data, 1)[0]
assert out.non_tensor_batch["sample_id"] == ["a","b","c"]
assert torch.equal(out.batch["reward"], torch.tensor([11.,22.,44.]))
assert len(data) == 5
""" },
        {"name": "Seeded worker shards match independently selected row ids", "visibility": "unshown", "behavior": "rl.rollout_assembly", "failure_message": "Every shard must preserve selected tensor and non-tensor rows in their original order.", "code": r"""
import random, torch
from torch_judge.harness.rl import MiniDataProto
for seed in (11, 23, 37):
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(8), 4))
    mask = torch.zeros((8, 3), dtype=torch.bool)
    for row in chosen: mask[row, rng.randrange(3)] = True
    data = MiniDataProto(
        {"response_mask": mask, "reward": torch.arange(8, dtype=torch.float32)},
        {"sample_id": [f"id-{i}" for i in range(8)]}, {"seed": seed})
    shards = {fn}(data, 2)
    assert len(shards) == 2 and all(len(shard) == 2 for shard in shards)
    for worker, shard in enumerate(shards):
        rows = chosen[worker * 2:(worker + 1) * 2]
        assert shard.non_tensor_batch["sample_id"] == [f"id-{i}" for i in rows]
        assert torch.equal(shard.batch["reward"], torch.tensor(rows, dtype=torch.float32))
        assert shard.meta_info == {"seed": seed}
    assert len(data) == 8
""" },
        {"name": "Rejects an empty or uneven worker split", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "No trainable rows or a non-divisible row count cannot form equal worker chunks.", "code": r"""
import torch
from torch_judge.harness.rl import MiniDataProto
for mask,chunks in [([[False],[False]],1), ([[True],[False],[True]],3), ([[True]],0)]:
    data=MiniDataProto({"response_mask":torch.tensor(mask)})
    try: {fn}(data,chunks)
    except ValueError: pass
    else: raise AssertionError("invalid chunk request accepted")
""" },
    ],
    "solution": '''def verl_dataproto_filter_chunk(data, chunks):
    import torch
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks <= 0:
        raise ValueError("chunks must be positive")
    mask = data.batch["response_mask"]
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("response_mask must be a boolean matrix")
    indices = torch.nonzero(mask.any(dim=1), as_tuple=False).flatten().tolist()
    if not indices or len(indices) % chunks:
        raise ValueError("selected rows cannot form equal chunks")
    return data.select_idxs(indices).chunk(chunks)
''',
}
