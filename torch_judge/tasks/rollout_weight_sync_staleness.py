"""Route Slime-style rollout samples across a weight-sync boundary."""

TASK = {
    "title": "Rollout Weight Sync and Staleness",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "rollout_weight_sync_staleness",
    "description_en": r"""Decide which rollout samples train and when an inference engine can receive new weights.

**Signature:** `rollout_weight_sync_staleness(samples, trainer_version, serving_version, max_staleness, in_flight, sync_weights) -> dict`

Each sample is a dictionary with `sample_id`, non-negative integer
`weight_version`, and `status`: `"complete"`, `"aborted"`, or
`"failed"`. For complete samples, compute
`lag = trainer_version - weight_version`. Send a complete sample to
`train` only when `0 <= lag <= max_staleness`. Put stale or aborted
samples in `retry`, and failed samples in `dropped`. Each output list
contains the original sample dictionaries in input order.

The trainer may synchronize the serving engine to `trainer_version` only
when `in_flight == 0`. If the versions differ and no trajectory is active,
call `sync_weights(trainer_version)` exactly once, after routing the samples.
Otherwise leave serving_version unchanged and report `sync_deferred=True`
when a sync is waiting. Return exactly `train`, `retry`, `dropped`,
`serving_version`, and `sync_deferred`.

Raise ValueError for negative or boolean version/count values, a serving
version ahead of the trainer, duplicate sample ids, unknown statuses, or a
sample weight version ahead of the trainer.

────────────────────────────────

**Background — context only.** Slime's training loop updates rollout weights
at a training boundary, and its distributed weight updater pauses generation
around synchronization. Its fully async worker requeues aborted groups. This
exercise uses a stricter drain-before-sync policy and makes sample staleness
explicit. The allowed lag is a teaching policy, not a built-in Slime default.""",
    "advisory_prerequisites": ["fully_async_rollout_buffer", "rollout_train_boundary"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which samples are safe to train under the version limit? When may the serving engine be updated without changing weights mid-trajectory?"},
        {"level": 2, "kind": "analysis", "content": "Validate versions and unique ids, then route each sample by status and complete-sample lag. After routing, if serving_version differs from trainer_version, call sync_weights only when in_flight is zero; otherwise report the deferred sync."},
    ],
    "model_connections": [
        "Slime train.py updates rollout weights after a training round; its distributed updater pauses and resumes generation during synchronization.",
        "Slime fully_async_rollout.py requeues ABORTED groups through the Data Buffer.",
        "VERL also separates rollout weights from trainer weights, so a trajectory's generating version matters when training is asynchronous.",
    ],
    "sources": [
        {"kind": "code", "url": "https://github.com/THUDM/slime", "commit": "8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b", "path": "train.py", "symbol": "train", "license": "Apache-2.0", "adapted": "Synchronize rollout weights at a training-loop boundary.", "simplifications": "Uses explicit integer versions and a drain-before-sync rule in place of Ray actors and distributed weight broadcast."},
        {"kind": "code", "url": "https://github.com/THUDM/slime", "commit": "8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b", "path": "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py", "symbol": "update_weights", "license": "Apache-2.0", "adapted": "Prevent weight updates from changing rollout weights mid-generation.", "simplifications": "Defers until no in-flight work rather than calling Slime's pause/continue generation methods."},
        {"kind": "code", "url": "https://github.com/THUDM/slime", "commit": "5bae5bb7928d65906e24e99f5b5d3f99a4daaf93", "path": "slime/rollout/fully_async_rollout.py", "symbol": "AsyncRolloutWorker._make_done_cb", "license": "Apache-2.0", "adapted": "Requeue aborted rollout groups before training.", "simplifications": "Classifies individual sample dictionaries rather than Sample groups in a background worker."},
        {"kind": "paper", "url": "https://arxiv.org/abs/1802.01561", "section": "2. IMPALA architecture and policy lag"}
    ],
    "tests": [
        {"name": "Routes complete, stale, aborted and failed samples", "behavior": "rl.rollout_assembly", "code": r"""
samples = [
    {"sample_id":"a","weight_version":8,"status":"complete"},
    {"sample_id":"b","weight_version":7,"status":"complete"},
    {"sample_id":"c","weight_version":9,"status":"aborted"},
    {"sample_id":"d","weight_version":9,"status":"failed"},
]
calls=[]
out={fn}(samples, trainer_version=10, serving_version=9, max_staleness=2,
         in_flight=0, sync_weights=calls.append)
assert [x["sample_id"] for x in out["train"]] == ["a"]
assert [x["sample_id"] for x in out["retry"]] == ["b","c"]
assert [x["sample_id"] for x in out["dropped"]] == ["d"]
assert out["serving_version"] == 10 and out["sync_deferred"] is False
assert calls == [10]
""" },
        {"name": "Defers sync while a trajectory is active", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Weights must not change while rollout generation is in flight.", "code": r"""
calls=[]
sample={"sample_id":"x","weight_version":3,"status":"complete"}
out={fn}([sample],5,3,2,1,calls.append)
assert out["train"] == [sample]
assert out["serving_version"] == 3 and out["sync_deferred"] is True
assert calls == []
""" },
        {"name": "Exact staleness boundary is trainable", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "A sample with lag exactly max_staleness should remain trainable.", "code": r"""
samples=[{"sample_id":"x","weight_version":2,"status":"complete"},
         {"sample_id":"y","weight_version":1,"status":"complete"}]
out={fn}(samples,5,5,3,0,lambda version:None)
assert [x["sample_id"] for x in out["train"]] == ["x"]
assert [x["sample_id"] for x in out["retry"]] == ["y"]
assert out["sync_deferred"] is False
""" },
        {"name": "Seeded status and lag routing conserves every sample", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Routing must classify each input exactly once and synchronize only when idle.", "code": r"""
import random
for seed in (11, 23, 37):
    rng = random.Random(seed)
    trainer = 8
    samples = [
        {"sample_id": str(i), "weight_version": rng.randrange(0, trainer + 1),
         "status": rng.choice(("complete", "aborted", "failed"))}
        for i in range(12)
    ]
    serving = rng.randrange(0, trainer + 1)
    in_flight = seed % 2
    calls = []
    out = {fn}(samples, trainer, serving, 3, in_flight, calls.append)
    want = {"train": [], "retry": [], "dropped": []}
    for sample in samples:
        if sample["status"] == "failed": bucket = "dropped"
        elif sample["status"] == "aborted" or sample["weight_version"] < trainer - 3: bucket = "retry"
        else: bucket = "train"
        want[bucket].append(sample)
    for bucket in want: assert out[bucket] == want[bucket], (seed, bucket, out[bucket])
    must_sync = serving != trainer and in_flight == 0
    assert calls == ([trainer] if must_sync else [])
    assert out["serving_version"] == (trainer if must_sync else serving)
    assert out["sync_deferred"] == (serving != trainer and in_flight > 0)
""" },
        {"name": "Rejects duplicate ids and future weights", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Samples need unique ids and cannot be generated by a future trainer version.", "code": r"""
sample={"sample_id":"a","weight_version":2,"status":"complete"}
for samples in ([sample,sample],[{"sample_id":"b","weight_version":9,"status":"complete"}]):
    try: {fn}(samples,5,5,1,0,lambda version:None)
    except ValueError: pass
    else: raise AssertionError("invalid samples accepted")
""" },
    ],
    "solution": '''def rollout_weight_sync_staleness(samples, trainer_version, serving_version, max_staleness, in_flight, sync_weights):
    for value in (trainer_version, serving_version, max_staleness, in_flight):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("versions and counts must be non-negative integers")
    if serving_version > trainer_version:
        raise ValueError("serving version is ahead of trainer")
    train, retry, dropped, seen = [], [], [], set()
    for sample in samples:
        try:
            sample_id = sample["sample_id"]
            version = sample["weight_version"]
            status = sample["status"]
        except (KeyError, TypeError):
            raise ValueError("malformed sample")
        if sample_id in seen or status not in ("complete", "aborted", "failed"):
            raise ValueError("duplicate id or unknown status")
        seen.add(sample_id)
        if not isinstance(version, int) or isinstance(version, bool) or version < 0 or version > trainer_version:
            raise ValueError("invalid sample version")
        if status == "failed":
            dropped.append(sample)
        elif status == "aborted" or trainer_version - version > max_staleness:
            retry.append(sample)
        else:
            train.append(sample)
    sync_deferred = serving_version != trainer_version and in_flight > 0
    if serving_version != trainer_version and not sync_deferred:
        sync_weights(trainer_version)
        serving_version = trainer_version
    return {
        "train": train, "retry": retry, "dropped": dropped,
        "serving_version": serving_version, "sync_deferred": sync_deferred,
    }
''',
}
