"""Mutation gates and metadata contracts for the DeepSeek Harness and Pi agent paths."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

ROOT = Path(__file__).resolve().parents[2]

NEW_TASKS = {
    "dsh_event_bus": "deepseek-harness",
    "dsh_repeat_tool_reminder": "deepseek-harness",
    "dsh_tool_result_pruner": "deepseek-harness",
    "dsh_open_turn_closers": "deepseek-harness",
    "dsh_tool_pipeline": "deepseek-harness",
    "dsh_tool_timeout": "deepseek-harness",
    "dsh_fold_surface": "deepseek-harness",
    "dsh_system_prompt_reconcile": "deepseek-harness",
    "pi_agent_loop": "pi-agent",
    "pi_transform_messages": "pi-agent",
    "pi_multi_edit": "pi-agent",
    "pi_file_mutation_queue": "pi-agent",
    "pi_compaction_cut_point": "pi-agent",
    "pi_truncate_output": "pi-agent",
    "pi_streaming_json": "pi-agent",
    "pi_parallel_tool_batch": "pi-agent",
}

# (name, text in the reference solution, replacement). Each replacement is one realistic bug.
MUTATIONS = {
    "dsh_event_bus": [
        ("no snapshot", "return [record[0] for record in self._hooks.get(name, [])]",
         "return _LiveView(self._hooks.get(name, []))"),
        ("prepend appends", "hooks.insert(0, record)", "hooks.append(record)"),
        ("dispose always true", "            return False\n\n        return dispose", "            return True\n\n        return dispose"),
        ("once disposes after running", "            dispose()\n            return listener(*args)",
         "            result = listener(*args)\n            dispose()\n            return result"),
        ("bail treats falsy as bail", "return value is not None and value is not False", "return bool(value)"),
        ("serial does not await", "            if inspect.isawaitable(result):\n                result = await result\n            if self._bailed(result):",
         "            if self._bailed(result):"),
        ("parallel runs one by one", "        results = await asyncio.gather(*(run(l) for l in self._snapshot(name)), return_exceptions=True)",
         "        results = []\n        for l in self._snapshot(name):\n            try:\n                await run(l)\n            except Exception as e:\n                results.append(e)"),
        ("parallel raises only the first error", "raise ExceptionGroup(f\"{len(errors)} listener(s) of {name!r} failed\", errors)", "raise errors[0]"),
        ("waterfall ignores veto", "            return inner(*args)\n\n        return next()",
         "            return inner(*args)\n\n        result = next()\n        while queue:\n            next()\n        return result"),
    ],
    "dsh_repeat_tool_reminder": [
        ("key ignores arguments", "key = (name, canonical)", "key = (name,)"),
        ("unsorted keys", "json.dumps(arguments, sort_keys=True,", "json.dumps(arguments, sort_keys=False,"),
        ("excluded tools reset the chain", "        if agent is None or not self._tracked(name):\n            return None",
         "        if agent is None:\n            return None\n        if not self._tracked(name):\n            self.chains.pop(agent, None)\n            return None"),
        ("one chain for all agents", "chain = self.chains.get(agent)", "chain = self.chains.get('shared')"),
        ("gentle on every threshold", "if count == self.thresholds[0]:", "if True:"),
        ("unanchored patterns", "p.fullmatch(name) for p in self.include", "p.search(name) for p in self.include"),
        ("key uses the preview", "        key = (name, canonical)", "        key = (name, canonical[:self.preview_chars])"),
        ("reminder appended last", "[reminder, *downstream.get(\"additional_contexts\", [])]", "[*downstream.get(\"additional_contexts\", []), reminder]"),
        ("mutates downstream", "        return {**downstream, \"additional_contexts\": [reminder, *downstream.get(\"additional_contexts\", [])]}",
         "        downstream[\"additional_contexts\"] = [reminder, *downstream.get(\"additional_contexts\", [])]\n        return downstream"),
        ("counts after next", "        reminder = self._observe(agent, tool_name, arguments)\n        downstream = next()",
         "        downstream = next()\n        downstream = next()\n        reminder = self._observe(agent, tool_name, arguments)"),
        ("user message resets everyone", "self.chains.pop(agent, None)\n", "self.chains.clear()\n"),
    ],
    "dsh_tool_result_pruner": [
        ("marker in every intersecting block", "if start < removed_end and end > removed_start and not marker_placed:",
         "if start < removed_end and end > removed_start:"),
        ("keeps empty text blocks", "        if new_text:\n            pruned.append", "        if True:\n            pruned.append"),
        ("drops non-text blocks", "        if block[\"type\"] != \"text\":\n            pruned.append(block)\n            continue",
         "        if block[\"type\"] != \"text\":\n            continue"),
        ("per-block budgets", "removed_end = total - tail_chars", "removed_end = removed_start + (total - head_chars - tail_chars) // max(1, len(blocks))"),
        ("prunes at the threshold", "if total <= threshold_chars:", "if total < threshold_chars:"),
        ("no convergence check", "if head_chars + len(MARKER) + tail_chars > threshold_chars:", "if head_chars + tail_chars > threshold_chars:"),
        ("accepts float budgets", "if type(threshold_chars) is not int or threshold_chars <= 0:", "if threshold_chars <= 0:"),
    ],
    "dsh_open_turn_closers": [
        ("step end keeps pending", "if kind in (\"turn/start\", \"turn/end\", \"step/end\"):", "if kind in (\"turn/start\", \"turn/end\"):"),
        ("replace answers a call", "if (entry is not None and data[\"surface_op\"] == \"append\"", "if (entry is not None"),
        ("ignores the step of the answer", "and entry[\"turn\"] == data[\"turn\"] and entry[\"step\"] == data[\"step\"]):", "):"),
        ("re-request keeps started", "pending[call_id] = {\"turn\": data[\"turn\"], \"step\": data[\"step\"], \"call_seq\": None}",
         "pending[call_id] = {\"turn\": data[\"turn\"], \"step\": data[\"step\"], \"call_seq\": pending.get(call_id, {}).get(\"call_seq\")}"),
        ("no step closer", "    if open_step is not None:\n        seq += 1\n        closers.append",
         "    if False:\n        seq += 1\n        closers.append"),
        ("closes balanced logs", "if not events or open_turn is None:", "if not events:"),
        ("swapped error codes", "\"TOOL_OUTCOME_UNKNOWN\" if started else \"TOOL_NOT_STARTED\"", "\"TOOL_NOT_STARTED\" if started else \"TOOL_OUTCOME_UNKNOWN\""),
        ("seq restarts at last", "    seq, time = last[\"seq\"], last[\"time\"]", "    seq, time = last[\"seq\"] - 1, last[\"time\"]"),
    ],
    "pi_transform_messages": [
        ("same model ignores api", "same = (msg[\"provider\"] == model[\"provider\"] and msg[\"api\"] == model[\"api\"]",
         "same = (msg[\"provider\"] == model[\"provider\"]"),
        ("keeps redacted thinking cross-model", "                        if same:\n                            content.append(block)\n                        continue",
         "                        content.append(block)\n                        continue"),
        ("keeps thought signature", "if not same and \"thought_signature\" in call:", "if False:"),
        ("results keep old ids", "            if new_id is not None and new_id != msg[\"tool_call_id\"]:", "            if False:"),
        ("replays errored turns", "            if msg.get(\"stop_reason\") in (\"error\", \"aborted\"):\n                continue", "            pass"),
        ("system messages not held", "            (held if pending else result).append(msg)", "            result.append(msg)"),
        ("user does not close calls", "        elif role == \"user\":\n            close()\n", "        elif role == \"user\":\n"),
        ("duplicate placeholders", "            if not previous_was_placeholder:\n                result.append", "            if True:\n                result.append"),
        ("mutates input", "            msg = {**msg, \"content\": content}", "            msg[\"content\"] = content"),
        ("no trailing close", "            result.append(msg)\n    close()\n    return result", "            result.append(msg)\n    return result"),
    ],
    "pi_multi_edit": [
        ("fuzzy writes the normalized file", "    if not fuzzy:\n        result = _apply(base, matches)", "    if True:\n        result = _apply(base, matches)"),
        ("no uniqueness check", "        if _norm(base).count(_norm(old)) > 1:", "        if base.count(old) > 1:"),
        ("sequential matching", "    for old, new in edits:\n        index, length = base.find(old), len(old)",
         "    for old, new in edits:\n        base = _apply(base, matches)\n        matches = []\n        index, length = base.find(old), len(old)"),
        ("no overlap check", "        if s1 + l1 > s2:", "        if False:"),
        ("keeps CR endings", "    return result.replace(\"\\n\", \"\\r\\n\") if crlf else result", "    return result"),
        ("no quote folding", "_TABLE = {**_SINGLE, **_DOUBLE, **_DASH, **_SPACE}", "_TABLE = {**_DASH, **_SPACE}"),
        ("keeps trailing spaces", "    text = \"\\n\".join(line.rstrip() for line in text.split(\"\\n\"))\n", ""),
        ("no-op allowed", "    if result == original:", "    if False:"),
        ("forward application", "for start, length, new in sorted(replacements, reverse=True):", "for start, length, new in sorted(replacements):"),
    ],
    "pi_file_mutation_queue": [
        ("key without realpath", "key = os.path.realpath(os.path.abspath(path))", "key = os.path.abspath(path)"),
        ("no waiting", "                await prev.wait()\n            except", "                pass\n            except"),
        ("cancelled caller releases early", "                async def hand_over():\n                    await prev.wait()\n                    self._release(key, done)",
         "                async def hand_over():\n                    self._release(key, done)"),
        ("cancelled caller never releases", "                task = asyncio.ensure_future(hand_over())\n                self._background.add(task)\n                task.add_done_callback(self._background.discard)\n", ""),
        ("failure keeps the key", "        try:\n            return await fn()\n        finally:\n            self._release(key, done)",
         "        result = await fn()\n        self._release(key, done)\n        return result"),
        ("deletes the key unconditionally", "        if self._tails.get(key) is done:\n            del self._tails[key]", "        self._tails.pop(key, None)"),
    ],
    "pi_compaction_cut_point": [
        ("cuts at tool results", "cut_points = [i for i, e in enumerate(entries) if e[\"role\"] not in (\"toolResult\", \"meta\")]",
         "cut_points = [i for i, e in enumerate(entries) if e[\"role\"] != \"meta\"]"),
        ("picks the cut before the crossing", "cut = next((c for c in cut_points if c >= i), cut_points[-1])",
         "cut = next((c for c in reversed(cut_points) if c <= i), cut_points[0])"),
        ("strict budget comparison", "if accumulated >= keep_recent_tokens:", "if accumulated > keep_recent_tokens:"),
        ("no meta pull-back", "    while cut > 0 and entries[cut - 1][\"role\"] == \"meta\":\n        cut -= 1\n", ""),
        ("assistant starts turns", "TURN_STARTS = {\"user\", \"bashExecution\",", "TURN_STARTS = {\"user\", \"assistant\", \"bashExecution\","),
        ("split turn ignores missing start", "\"is_split_turn\": turn_start != -1}", "\"is_split_turn\": True}"),
        ("accepts unknown roles", "        if entry[\"role\"] not in ROLES:", "        if False:"),
    ],
    "pi_agent_loop": [
        ("executes truncated calls", "        if message[\"stop_reason\"] == \"length\":", "        if False:"),
        ("ignores follow-ups", "        follow = get_follow_up()\n", "        follow = []\n"),
        ("no steering between steps", "            pending = get_steering()\n        follow", "            pending = []\n        follow"),
        ("terminate on any result", "return results, bool(results) and not all(terminates)", "return results, bool(results) and not any(terminates)"),
        ("exceptions propagate", "        try:\n            outcome = execute_tool(call)\n        except Exception as error:",
         "        try:\n            outcome = execute_tool(call)\n        except ZeroDivisionError as error:"),
        ("continues after an error", "            if reply[\"stop_reason\"] in (\"error\", \"aborted\"):", "            if reply[\"stop_reason\"] == \"aborted\":"),
        ("model sees a live list", "reply = model(list(context))", "reply = model(context)"),
        ("ignores initial steering", "    pending = get_steering()\n    while True:", "    pending = []\n    while True:"),
    ],
}

MUTATIONS.update({
    "dsh_tool_pipeline": [
        ("guards run on deny", "            if decision[\"kind\"] == \"allow\":\n                for guard", "            if True:\n                for guard"),
        ("no guards", "                    reason = guard(call)\n", "                    reason = None\n"),
        ("denials skip post-execute", "                result = _error(reason)\n", "                return _error(reason)\n"),
        ("pipeline failures reach post-execute", "        except Exception as error:\n            result = _error(str(error))\n        for observer",
         "        except Exception as error:\n            result = self._post(call, _error(str(error)))\n        for observer"),
        ("body errors escape", "        try:\n            return {\"is_error\": False, \"content\": self.tools[name](call[\"arguments\"])}\n        except Exception as error:\n            return _error(str(error))",
         "        return {\"is_error\": False, \"content\": self.tools[name](call[\"arguments\"])}"),
        ("unknown tool without code", "_error(f'unknown tool \"{name}\"', \"UNKNOWN_TOOL\")", "_error(f'unknown tool \"{name}\"')"),
        ("ask allowed without channel", "        if self.approve is None:\n            return {\"kind\": \"deny\"", "        if self.approve is None:\n            return {\"kind\": \"allow\"}\n            return {\"kind\": \"deny\""),
        ("block keeps tool contexts", "            if extra:\n                blocked[\"additional_contexts\"] = extra",
         "            contexts = list(result.get(\"additional_contexts\") or []) + extra\n            if contexts:\n                blocked[\"additional_contexts\"] = contexts"),
        ("observer failure stops others", "            try:\n                observer(call, result)\n            except Exception:\n                pass", "            observer(call, result)"),
        ("reversed middleware", "    queue = list(middleware)", "    queue = list(reversed(middleware))"),
        ("cancel treated as allow", "        if decision[\"kind\"] == \"cancel\":", "        if False:"),
    ],
    "dsh_tool_timeout": [
        ("abort not idempotent", "        if signal.aborted:\n            return\n        signal.aborted = True", "        signal.aborted = True"),
        ("late listener ignored", "        if self.aborted:\n            fn(self.reason)\n        else:\n            self._listeners.append(fn)", "        self._listeners.append(fn)"),
        ("unscoped timeout", "        if timeout_of(d.signal, \"TOOL_TIMEOUT\") is not None:", "        if timeout_of(d.signal) is not None:"),
        ("no restore on error", "    finally:\n        execution[\"signal\"] = upstream\n        d.dispose()", "    finally:\n        d.dispose()\n    execution[\"signal\"] = upstream"),
        ("no dispose", "        execution[\"signal\"] = upstream\n        d.dispose()", "        execution[\"signal\"] = upstream"),
        ("returns without waiting", "        result = await next()\n",
         "        task = asyncio.ensure_future(next())\n        done = asyncio.Event()\n        d.signal.add_listener(lambda r: done.set())\n        task.add_done_callback(lambda t: done.set())\n        await done.wait()\n        result = task.result() if task.done() else {}\n"),
        ("timer ignores upstream", "    signal = timer.signal if upstream is None else any_signal([upstream, timer.signal])", "    signal = timer.signal"),
        ("zero timeout arms a timer", "    if timeout <= 0:", "    if timeout < 0:"),
        ("wrapper passes no budget through", "    if timeout is None:\n        return await next()", "    if timeout is None:\n        timeout = 0.001"),
    ],
    "dsh_fold_surface": [
        ("replace appends", "        nodes[start_idx:end_idx + 1] = [seq]", "        del nodes[start_idx:end_idx + 1]\n        nodes.append(seq)"),
        ("exclusive end", "        shadowed = nodes[start_idx:end_idx + 1]", "        shadowed = nodes[start_idx:end_idx]"),
        ("no content-only check", "            if _without_content(events[shadowed[0]][\"data\"]) != _without_content(event[\"data\"]):", "            if False:"),
        ("no head protection", "        if start_idx == 0 and events[nodes[0]][\"type\"] == \"system/message\":", "        if False:"),
        ("non-message ops ignored", "            if op is not None:\n                raise ValueError(f\"{event['type']} cannot carry a surface_op\")", "            pass"),
        ("no seq check", "        if seq != index:", "        if False:"),
        ("messages from the log", "        \"messages\": [events[s][\"data\"][\"message\"] for s in nodes],",
         "        \"messages\": [e[\"data\"][\"message\"] for e in events if e[\"type\"] in MESSAGE_TYPES],"),
    ],
    "dsh_system_prompt_reconcile": [
        ("capable route rewrites head", "    if in_history_capable and not new_series:", "    if in_history_capable and False:"),
        ("new series still appends", "    if in_history_capable and not new_series:", "    if in_history_capable:"),
        ("compares with head", "        effective = next((text for _, text in reversed(system_nodes) if text), None)", "        effective = head_text"),
        ("blanks empty nodes too", "    later = [(seq, text) for seq, text in system_nodes[1:] if text]", "    later = list(system_nodes[1:])"),
        ("head before later nodes", "        ops = [(\"replace\", seq, \"\") for seq, _ in later]\n        if head_text != target:\n            ops.append((\"replace\", head_seq, target))",
         "        ops = [(\"replace\", head_seq, target)] if head_text != target else []\n        ops += [(\"replace\", seq, \"\") for seq, _ in later]"),
        ("empty rendering appends", "    if rendered == \"\":\n        return consolidate(\"\")", "    if rendered == \"\" and not in_history_capable:\n        return consolidate(\"\")"),
        ("no node appends only non-empty", "    if not system_nodes:\n        return [(\"append\", rendered)]", "    if not system_nodes:\n        return [(\"append\", rendered)] if rendered else []"),
        ("always rewrites head", "        if head_text != target:\n            ops.append", "        if True:\n            ops.append"),
    ],
    "pi_truncate_output": [
        ("trailing newline adds a line", "    if content.endswith(\"\\n\"):\n        parts.pop()\n", ""),
        ("newline always counted", "            cost = _size(line) + (1 if kept else 0)\n            if used + cost > max_bytes:\n                truncated_by = \"bytes\"\n                break",
         "            cost = _size(line) + 1\n            if used + cost > max_bytes:\n                truncated_by = \"bytes\"\n                break"),
        ("chars instead of bytes", "    return len(text.encode(\"utf-8\"))", "    return len(text)"),
        ("partial backs up past the limit", "        start += 1\n", "        start -= 1\n"),
        ("no partial tail line", "                if not kept:\n                    kept.append(_suffix_within(line, max_bytes))\n                    result[\"last_line_partial\"] = True\n", ""),
        ("head keeps partial first line", "        if _size(lines[0]) > max_bytes:", "        if False:"),
        ("either limit enough", "    if len(lines) <= max_lines and total_bytes <= max_bytes:", "    if len(lines) <= max_lines or total_bytes <= max_bytes:"),
        ("line limit off by one", "        for line in lines[:max_lines]:", "        for line in lines[:max_lines + 1]:"),
    ],
    "pi_streaming_json": [
        ("drops partial strings", "        text = json.loads(s[i:j] + '\"')\n", "        text = _MISSING\n        return text, n, False\n"),
        ("keeps partial literals", "            if literal.startswith(s[i:]):\n                return _MISSING, n, False", "            if literal.startswith(s[i:]):\n                return parsed, n, False"),
        ("keeps dangling keys", "                if s[j] != \":\":\n                    raise _Invalid()\n                j += 1",
         "                if s[j] != \":\":\n                    raise _Invalid()\n                j += 1\n                if ws(j) == n:\n                    result[key] = None\n                    return result, n, False"),
        ("no repair", "    fixed = repair(text)\n", "    fixed = text\n"),
        ("no control escaping", "        elif ord(ch) < 0x20:\n            out.append(_CONTROL.get(ch, f\"\\\\u{ord(ch):04x}\"))", "        elif False:\n            pass"),
        ("repaired text first", "    for candidate in (text, fixed):", "    for candidate in (fixed, text):"),
        ("trailing garbage accepted", "    if done and ws(end) != n:\n        raise _Invalid()\n", ""),
        ("numbers kept raw", "            token = token.rstrip(\"+-.eE\")\n", ""),
        ("keeps dangling surrogate", "        if text and 0xD800 <= ord(text[-1]) <= 0xDBFF:\n            text = text[:-1]\n", ""),
    ],
    "pi_parallel_tool_batch": [
        ("runs during preparation", "        async def settle(call, outcome):\n            return outcome if outcome is not None else await run(call, True)\n\n        outcomes = await asyncio.gather(*(settle(c, o) for c, o in pending))\n        recorded = list(zip([c for c, _ in pending], outcomes))",
         "        outcomes = []\n        for c, o in pending:\n            outcomes.append(o if o is not None else await run(c, True))\n        recorded = list(zip([c for c, _ in pending], outcomes))"),
        ("ignores sequential tools", "    sequential = mode == \"sequential\" or any(\n        tools.get(c[\"name\"], {}).get(\"sequential\") for c in calls)", "    sequential = mode == \"sequential\""),
        ("completion order", "        outcomes = await asyncio.gather(*(settle(c, o) for c, o in pending))\n        recorded = list(zip([c for c, _ in pending], outcomes))",
         "        done_order = []\n        async def track(c, o):\n            r = await settle(c, o)\n            done_order.append((c, r))\n        await asyncio.gather(*(track(c, o) for c, o in pending))\n        recorded = done_order"),
        ("exceptions escape", "            except Exception as error:\n                outcome = _error(str(error))", "            except ZeroDivisionError as error:\n                outcome = _error(str(error))"),
        ("terminate on any", "all(outcome[\"terminate\"] for _, outcome in recorded)", "any(outcome[\"terminate\"] for _, outcome in recorded)"),
        ("block ignores terminate", "                    outcome = _error(decision.get(\"reason\") or \"Tool execution was blocked\",\n                                     decision.get(\"terminate\") is True)",
         "                    outcome = _error(decision.get(\"reason\") or \"Tool execution was blocked\")"),
        ("no abort before run", "        if check_abort and is_aborted():", "        if False:"),
        ("late end for early outcomes", "        if outcome is not None:\n            emit({\"type\": \"end\", \"id\": call[\"id\"]})\n        return outcome", "        return outcome"),
    ],
})

# Helper appended to the event-bus solution so the "no snapshot" mutation can iterate the live list.
_EVENT_BUS_HELPER = '''

class _LiveView:
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        i = 0
        while i < len(self.records):
            yield self.records[i][0]
            i += 1

    def __len__(self):
        return len(self.records)

    def pop(self, index):
        return self.records.pop(index)[0]
'''

_SEED_TUPLE = re.compile(r"for seed in \(([\d, ]+)\)")
_MANUAL_SEED = re.compile(r"manual_seed\((\d+)\)")


def _seeded_task_variant(task_id: str, repeat: int) -> dict:
    """Shift every seed in the test cases so a mutation cannot pass by luck."""
    task = copy.deepcopy(get_task(task_id))
    for case in task["tests"]:
        case["code"] = _SEED_TUPLE.sub(
            lambda m: "for seed in (" + ", ".join(str(int(s) + 100 * repeat) for s in m.group(1).split(",")) + ",)",
            case["code"],
        )
        case["code"] = _MANUAL_SEED.sub(lambda m: f"manual_seed({int(m.group(1)) + 100 * repeat})", case["code"])
    return task


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("task_id,targets", MUTATIONS.items())
def test_mutations_rejected_across_distinct_seeds(task_id, targets, repeat):
    original = get_task(task_id)["solution"]
    mutations = []
    for name, old, new in targets:
        assert old in original, f"mutation target drifted: {task_id}/{name}"
        code = original.replace(old, new, 1)
        if task_id == "dsh_event_bus":
            code += _EVENT_BUS_HELPER
        mutations.append(Mutation(name, code))
    rejected = assert_mutations_rejected(
        task_id, mutations, require_unshown=True,
        task_override=_seeded_task_variant(task_id, repeat),
    )
    assert set(rejected) == {name for name, _, _ in targets}


@pytest.mark.parametrize("task_id", NEW_TASKS)
def test_task_metadata_is_valid_and_cites_a_pinned_source(task_id):
    task = get_task(task_id)
    validate_task(task_id, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
    assert task["interview_questions"]
    assert any(source["kind"] == "code" for source in task["sources"])


def test_harness_paths_list_every_new_task():
    paths = {p["id"]: p for p in json.loads((ROOT / "web/src/lib/paths.json").read_text())["paths"]}
    for task_id, path_id in NEW_TASKS.items():
        assert task_id in paths[path_id]["problems"], (task_id, path_id)
    starters = json.loads((ROOT / "web/src/lib/starters.json").read_text())
    assert all(task_id in starters for task_id in NEW_TASKS)
