"""Mutation gates and metadata contracts for the streaming LLM path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from tests.quality.test_harness_paths_mutations import _seeded_task_variant

ROOT = Path(__file__).resolve().parents[2]
PATH_ID = "streaming-llm"

NEW_TASKS = [
    "sse_decoder",
    "chat_stream_accumulator",
    "incremental_detokenize",
    "stop_string_holdback",
    "request_output_collector",
    "attention_sink_cache",
    "sink_cache_rerotation",
    "h2o_kv_eviction",
    "duplex_streaming_session",
    "embodied_action_stream",
    "posed_frame_context",
]

# (name, text in the reference solution, replacement). Each replacement is one realistic bug.
MUTATIONS = {
    "sse_decoder": [
        ("CRLF split counts twice", "            if self._skip_lf:\n                self._skip_lf = False\n                if byte == 10:\n                    continue\n", ""),
        ("decodes each chunk", "        events = []\n        for byte in chunk:", "        events = []\n        chunk = chunk.decode(\"utf-8\", errors=\"replace\").encode(\"utf-8\")\n        for byte in chunk:"),
        ("strips all leading spaces", "        if value.startswith(\" \"):\n            value = value[1:]", "        value = value.lstrip(\" \")"),
        ("id resets on dispatch", "self._event, self._data, self._retry, self._pending = None, [], None, False",
         "self._event, self._data, self._retry, self._pending, self._last_id = None, [], None, False, None"),
        ("blank line always dispatches", "            if not self._pending:\n                return None\n", ""),
        ("data joined without newline", "\"data\": \"\\n\".join(self._data)", "\"data\": \"\".join(self._data)"),
        ("accepts NUL ids", "            if \"\\0\" in value:\n                return None\n", ""),
        ("lenient retry", "if not value or any(c not in \"0123456789\" for c in value):", "if not value.strip().lstrip(\"-\").isdigit():"),
        ("keeps going after DONE", "            if self.done:\n                break\n", ""),
        ("unknown field marks pending", "        else:\n            return None\n        self._pending = True", "        self._pending = True"),
    ],
    "chat_stream_accumulator": [
        ("copies first indexed list", "            if not _is_indexed(value):", "            if True:"),
        ("concatenates index and type", "if key == \"index\" or key == \"type\":", "if False:"),
        ("None overwrites", "                    current.insert(index, entry)\n    return acc",
         "                    current.insert(index, entry)\n        else:\n            acc[key] = value\n    return acc"),
        ("numbers not added", "elif _is_number(current) and _is_number(value):", "elif False:"),
        ("always extends lists", "if primitive and (current or not _is_indexed(value)):", "if primitive:"),
        ("merges into the last call", "accumulate_delta(current[index], entry)", "accumulate_delta(current[-1], entry)"),
        ("dict delta replaces", "            accumulate_delta(current, value)", "            acc[key] = value"),
        ("None accumulator kept", "if key not in acc or acc[key] is None:", "if key not in acc:"),
        ("no entry validation", "if not isinstance(entry, dict) or type(entry.get(\"index\")) is not int:", "if not isinstance(entry, dict):"),
    ],
    "incremental_detokenize": [
        ("decodes the token alone", "        new_text = self.convert(self.tokens[self.prefix_offset:])", "        new_text = prefix_text + self.convert([token])"),
        ("emits replacement characters", " or new_text.endswith(\"\\ufffd\")", ""),
        ("advances while holding", "            return \"\"\n        self.prefix_offset", "            self.prefix_offset, self.read_offset = self.read_offset, len(self.tokens)\n            return \"\"\n        self.prefix_offset"),
        ("window collapses", "        self.prefix_offset = self.read_offset\n", "        self.prefix_offset = len(self.tokens)\n"),
        ("no prompt context", "        self.tokens = list(prompt_tokens[-(initial_offset + 2):]) if prompt_tokens else []", "        self.tokens = []"),
        ("whole prompt window", "self.prefix_offset = max(self.read_offset - initial_offset, 0)", "self.prefix_offset = 0"),
        ("slices by token count", "        return new_text[len(prefix_text):]", "        return new_text[self.read_offset - self.prefix_offset:]" ),
    ],
    "stop_string_holdback": [
        ("holds back the full length", "max(len(s) for s in self.stop) - 1 if", "max(len(s) for s in self.stop) if"),
        ("holds back one too few", "max(len(s) for s in self.stop) - 1 if", "max(len(s) for s in self.stop) - 2 if"),
        ("holds back when included", "if self.stop and not self.include else 0", "if self.stop else 0"),
        ("no overlap", "len(self.output_text) - new_chars - len(s) + 1", "len(self.output_text) - new_chars"),
        ("rescans everything", "self.output_text.find(s, max(0, len(self.output_text) - new_chars - len(s) + 1))", "self.output_text.find(s)"),
        ("earliest start wins", "            end = index + len(s)\n            if best_end is None or end < best_end:", "            end = index + len(s)\n            if best_end is None or index < best_index:"),
        ("last string wins ties", "if best_end is None or end < best_end:", "if best_end is None or end <= best_end:"),
        ("first listed wins", "if best_end is None or end < best_end:", "if best_end is None:"),
        ("min_tokens ignored", "            if self.min_tokens and self.num_tokens <= self.min_tokens:\n                check_from = len(self.output_text)\n", ""),
        ("min_tokens blocks straddling stops", "                check_from = len(self.output_text)\n        if not self.stop", "                check_from = len(self.output_text) + max((len(s) for s in self.stop), default=0)\n        if not self.stop"),
        ("include truncates at start", "self.output_text[:best_end] if self.include else", "self.output_text[:best_index] if self.include else"),
        ("finished still holds", "(0 if finished else self.hold)", "self.hold"),
    ],
    "request_output_collector": [
        ("latest output wins", "            self._merge(self.output, item)", "            self.output = copy.deepcopy(item)"),
        ("output replaces a pending exception", "        elif not isinstance(self.output, Exception):\n            self._merge(self.output, item)",
         "        elif isinstance(self.output, Exception):\n            self.output = copy.deepcopy(item)\n        else:\n            self._merge(self.output, item)"),
        ("finished overwritten", "pending[\"finished\"] or new[\"finished\"]", "new[\"finished\"]"),
        ("merges by position", "if current[\"index\"] == completion[\"index\"]:", "if True:"),
        ("always aggregates", "                    if self.aggregate:", "                    if True:"),
        ("never aggregates", "                    if self.aggregate:", "                    if False:"),
        ("token ids dropped", "                        current[\"token_ids\"].extend(completion[\"token_ids\"])\n", ""),
        ("stores the caller's dict", "self.output = item if isinstance(item, Exception) else copy.deepcopy(item)", "self.output = item"),
        ("slot not cleared", "            self.output = None\n            self.ready.clear()", "            self.ready.clear()"),
        ("get does not wait", "        while self.output is None:\n            await self.ready.wait()\n", ""),
    ],
    "attention_sink_cache": [
        ("no sinks", "        self.start_size = start_size\n", "        self.start_size = 0\n"),
        ("evicts without room for the new tokens", "cut = length - self.recent_size + n", "cut = length - self.recent_size"),
        ("queries one position late", "query_pos = torch.arange(total - n, total, device=q.device)", "query_pos = torch.arange(total - n + 1, total + 1, device=q.device)"),
        ("caches rotated keys", "        self.keys = torch.cat([self.keys, k], dim=1)",
         "        self.keys = torch.cat([self.keys, _rope(k, torch.arange(self.keys.shape[1], self.keys.shape[1] + n), self.base)], dim=1)"),
        ("no causal mask", "        scores = scores.masked_fill(key_pos[None, :] > query_pos[:, None], float(\"-inf\"))\n", ""),
        ("mask hides the diagonal", "key_pos[None, :] > query_pos[:, None]", "key_pos[None, :] >= query_pos[:, None]"),
        ("interleaved rope pairs", "torch.cat([inv_freq, inv_freq])", "inv_freq.repeat_interleave(2)"),
        ("no scaling", " / math.sqrt(q.shape[-1])", ""),
        ("accepts oversized chunks", "        if not 1 <= n <= self.recent_size:\n            raise ValueError(\"need 1 <= n <= recent_size\")\n", ""),
    ],
    "sink_cache_rerotation": [
        ("no re-rotation", "        kept = _rotate(key_cache[:, num_sink_tokens + shift:], -shift * freqs)", "        kept = key_cache[:, num_sink_tokens + shift:]"),
        ("rotates forward", "-shift * freqs", "shift * freqs"),
        ("rotates the sinks too", "        sinks = key_cache[:, :num_sink_tokens]", "        sinks = _rotate(key_cache[:, :num_sink_tokens], -shift * freqs)"),
        ("evicts the oldest tokens", "kept = _rotate(key_cache[:, num_sink_tokens + shift:]", "kept = _rotate(key_cache[:, num_sink_tokens:key_cache.shape[1] - shift]"),
        ("new keys at text positions", "positions = torch.arange(start, start + n,", "positions = torch.arange(start + shift, start + shift + n,"),
        ("values not evicted", "        value_cache = torch.cat([value_cache[:, :num_sink_tokens], value_cache[:, num_sink_tokens + shift:]], dim=1)\n", "        value_cache = value_cache[:, shift:]\n"),
        ("interleaved rope pairs", "freqs = torch.cat([inv_freq, inv_freq])", "freqs = inv_freq.repeat_interleave(2)"),
        ("evicts only when over by one", "shift = max(0, key_cache.shape[1] + n - window_length)", "shift = max(0, key_cache.shape[1] + n - window_length - 1)"),
        ("accepts oversized chunks", "    if not 1 <= n <= window_length - num_sink_tokens:\n        raise ValueError(\"need 1 <= n <= window_length - num_sink_tokens\")\n", ""),
    ],
    "h2o_kv_eviction": [
        ("no accumulation", "            scores[:, :length - n] += self.scores\n", "            pass\n"),
        ("misaligned accumulation", "scores[:, :length - n] += self.scores", "scores[:, n:] += self.scores"),
        ("recent tokens compete", "candidates = scores[:, :length - self.recent_size]", "candidates = scores[:, :length - self.recent_size + 1]"),
        ("keeps the lightest", "torch.argsort(-candidates, dim=-1, stable=True)", "torch.argsort(candidates, dim=-1, stable=True)"),
        ("heavy hitters out of order", "order[:, :self.heavy_size].sort(dim=-1).values", "order[:, :self.heavy_size]"),
        ("one selection for all heads", "candidates = scores[:, :length - self.recent_size]", "candidates = scores.sum(0, keepdim=True).expand_as(scores)[:, :length - self.recent_size]"),
        ("scores not evicted", "        self.scores = torch.gather(scores, 1, keep)\n", ""),
        ("uses the last query only", "scores = attn.sum(dim=1)", "scores = attn[:, -1]"),
        ("no shape check", "        if attn.shape[2] != length:\n            raise ValueError(\"attention must cover every cached key\")\n", ""),
    ],
    "duplex_streaming_session": [
        ("never strips BOS", "if self.history and self.bos_id is not None and new_ids[:1] == [self.bos_id]:", "if False:"),
        ("strips BOS on the first chunk", "if self.history and self.bos_id is not None and", "if self.bos_id is not None and"),
        ("BOS id 0 ignored", "self.history and self.bos_id is not None and", "self.history and self.bos_id and"),
        ("KV counts the last sampled token", "self.kv_len = len(sequence) + max(len(out) - 1, 0)", "self.kv_len = len(sequence) + len(out)"),
        ("output not truncated", "out = list(output_ids[:max_new_tokens])", "out = list(output_ids)"),
        ("commits at begin", "        self._inflight = (sequence, max_new_tokens)\n", "        self._inflight = (sequence, max_new_tokens)\n        self.history = sequence\n"),
        ("allows concurrent requests", "        if self._inflight is not None:\n            raise RuntimeError(\"streaming session already has an active request\")\n", ""),
        ("returns internal list", "\"input_ids\": list(sequence)", "\"input_ids\": sequence"),
        ("cached_len is the history length", "\"cached_len\": self.kv_len", "\"cached_len\": len(self.history)"),
    ],
    "embodied_action_stream": [
        ("no preemption", "        events = [e for e in events if e[0] < tick]\n", ""),
        ("held keys not released", "        for key in sorted(_held(events)):", "        for key in []:"),
        ("keeps events at the arrival tick", "events = [e for e in events if e[0] < tick]", "events = [e for e in events if e[0] <= tick]"),
        ("skips bad actions", "                if action is None:\n                    return None", "                if action is None:\n                    continue"),
        ("rejected chunk preempts", "            rejected.append(index)\n            continue", "            rejected.append(index)\n            events = [e for e in events if e[0] < tick]\n            continue"),
        ("press releases in the same tick", "events.append((cursor + 1, seq + 1, \"up\", arg))", "events.append((cursor, seq + 1, \"up\", arg))"),
        ("no clamp", "max(-100, min(100, int(p)))", "int(p)"),
        ("text before a segment", "        if match.start() != pos:\n            return None\n", ""),
        ("text after the segments", "    if pos != len(text):\n        return None\n", ""),
        ("wait 0 accepted", "int(parts[1]) >= 1", "int(parts[1]) >= 0"),
        ("thinking spoken aloud", "        if kind == \"say\":", "        if kind in (\"say\", \"think\"):"),
        ("held keys ignore tick order", "sorted(events, key=lambda e: (e[0], e[1]))", "events"),
    ],
    "posed_frame_context": [
        ("yaw does not wrap", "    return abs((a - b + 180) % 360 - 180)", "    return abs(a - b)"),
        ("no recent frames", "chosen = set(range(max(0, n - recent), n))", "chosen = set()"),
        ("ties prefer older frames", "others.sort(key=lambda i: (score(i), -i))", "others.sort(key=lambda i: (score(i), i))"),
        ("no direction filter", " and _angle(self.frames[i][2], yaw) <= max_angle]", "]"),
        ("strict angle limit", "_angle(self.frames[i][2], yaw) <= max_angle", "_angle(self.frames[i][2], yaw) < max_angle"),
        ("angle ignored in score", " + angle_weight * math.radians(_angle(frame_yaw, yaw))", ""),
        ("angle in degrees", "math.radians(_angle(frame_yaw, yaw))", "_angle(frame_yaw, yaw)"),
        ("ranked order returned", "return [self.frames[i][0] for i in sorted(chosen)]", "return [self.frames[i][0] for i in list(range(max(0, n - recent), n)) + others[:k - len(set(range(max(0, n - recent), n)))]]"),
        ("k extra frames", "chosen.update(others[:k - len(chosen)])", "chosen.update(others[:k])"),
        ("duplicate ids", "        if frame_id in self.ids:\n            raise ValueError(f\"frame {frame_id!r} already added\")\n", ""),
        ("manhattan distance", "math.dist(pos, position)", "sum(abs(a - b) for a, b in zip(pos, position))"),
    ],
}


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("task_id,targets", MUTATIONS.items())
def test_mutations_rejected_across_distinct_seeds(task_id, targets, repeat):
    original = get_task(task_id)["solution"]
    mutations = []
    for name, old, new in targets:
        assert old in original, f"mutation target drifted: {task_id}/{name}"
        mutations.append(Mutation(name, original.replace(old, new, 1)))
    rejected = assert_mutations_rejected(
        task_id, mutations, require_unshown=True,
        task_override=_seeded_task_variant(task_id, repeat),
    )
    assert set(rejected) == {name for name, _, _ in targets}


@pytest.mark.parametrize("task_id", NEW_TASKS)
def test_task_metadata_is_valid_and_cited(task_id):
    task = get_task(task_id)
    validate_task(task_id, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
    assert task["interview_questions"]
    assert task["sources"]


def test_every_task_has_mutations():
    assert set(MUTATIONS) == set(NEW_TASKS)


def test_streaming_path_lists_every_new_task():
    paths = {p["id"]: p for p in json.loads((ROOT / "web/src/lib/paths.json").read_text())["paths"]}
    assert set(NEW_TASKS) <= set(paths[PATH_ID]["problems"])
    starters = json.loads((ROOT / "web/src/lib/starters.json").read_text())
    assert all(task_id in starters for task_id in NEW_TASKS)
