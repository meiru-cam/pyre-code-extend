"""Multi-turn agentic rollout — assembling a tool-using episode into an RL trajectory."""

TASK = {
    "title": "Agentic Rollout Loop",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "agentic_rollout_loop",
    "description_en": r"""Run a multi-turn tool-calling episode and assemble it into a trajectory an RL loss can consume.

**Signature:** `agentic_rollout_loop(model, tools, initial_messages, max_turns, reward_fn) -> dict`

**Parameters:**
- `model` — has `model.call(messages) -> response`. The response is a dictionary with a `content` string and, optionally, a `tool_calls` list whose entries each have a `name` and an `arguments` dictionary.
- `tools` — a dictionary mapping a tool name to an object with `invoke(arguments) -> result`.
- `initial_messages` — a list of message dictionaries, each with at least a `role` and a `content`.
- `max_turns` — the maximum number of model calls. At least 1.
- `reward_fn` — `reward_fn(messages) -> float`. Scores the finished conversation.

**Returns:** a dictionary with exactly six keys:

- `messages` — the full conversation, in order.
- `trainable` — a list of booleans, the same length as `messages`. True exactly on messages whose role is `assistant`.
- `turns` — the number of model calls made.
- `tool_calls` — the number of tool invocations made.
- `stop_reason` — `finished` when the model returned a turn with no tool calls, or `max_turns` when the budget ran out first.
- `reward` — the float returned by `reward_fn` on the final message list.

**The loop, per turn:**

1. Call the model with the conversation so far.
2. Append an assistant message holding the response's `content`. Mark it trainable.
3. If the response requested no tools, set `stop_reason` to `finished` and stop.
4. Otherwise invoke each requested tool in order, appending one message per result with role `tool`, a `name`, and the result as `content`. Mark each **not** trainable.
5. If the turn budget is exhausted, stop with `stop_reason` of `max_turns`.

**Constraints:**
- `trainable` is True exactly on assistant messages, and always the same length as `messages`.
- Never call the model after a turn that requested no tools.
- Never exceed `max_turns` model calls. A truncated episode reports `max_turns`, not `finished`.
- Invoke tools in the order the response lists them, one message per result.
- Do not mutate `initial_messages`. Copy before appending.
- Raise a `ValueError` when `max_turns` is less than 1, or when a response names a tool that is not in `tools`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**The trainable flags are the whole point.** A rollout mixes two kinds of text that look identical once tokenized: what the policy produced, and what the environment wrote back. Only the first carries gradient. Marking tool output trainable teaches the model to predict text it never chose and will not choose at inference — the multi-turn version of training on the prompt, and invisible in the loss value. Downstream, `trainable` becomes the per-token mask that `grpo_token_loss` divides by.

**One reward for the whole trajectory.** The reward is computed once, at the end, and belongs to every trainable token equally. This is the same credit-assignment limitation `group_relative_advantage` has within a single response, now stretched across turns: a trajectory that succeeded after a useless tool call rewards that useless call exactly as much as the useful one. `vineppo_mc_value` is the estimator that addresses this, at a cost.

**The budget is a hard stop.** A model that never stops calling tools must be cut off, and the trajectory it produced up to that point is still a valid training example. Reporting `max_turns` rather than `finished` is what lets a training loop tell a truncated episode from a completed one; conflating them makes a runaway agent look like a successful one in every metric.""",
    "advisory_prerequisites": [
        "grpo_train_step",
        "rollout_batch_assembly",
        "tool_registry",
        "budgeted_agent_loop",
    ],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Two lists grow together and must never fall out of step — what invariant relates their lengths, and where in the loop is it easiest to break? Walk through a model that asks for two tools in one turn: how many messages get appended, how many are trainable, and what does `turns` become? Now the exit conditions: there are two, and they produce different stop reasons — which one is checked inside the loop body and which one is the loop condition itself? Finally, if the caller passes the same `initial_messages` list to two rollouts in a row, what would go wrong without a copy?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Copy first: `messages = [dict(m) for m in initial_messages]`, and build `trainable` alongside it from each message's role, so the two lists are the same length from the very first line. Structure the loop as `while turns < max_turns`, which makes `max_turns` the fall-through stop reason and `finished` the one you set explicitly with a `break`. Inside: call the model, increment `turns`, append the assistant message with `True`, then read `response.get(\"tool_calls\") or []` — the `or` matters because a missing key and an empty list must behave the same. If that list is empty, set `finished` and break. Otherwise loop over it, validate each name against `tools` before invoking, and append one message with `False` per result. The bug that survives every count assertion is appending to `messages` in one branch and forgetting `trainable` in another; keeping the two appends adjacent on every path is what prevents it.",
        },
    ],
    "model_connections": [
        "nano-aha-moment's create_training_episodes assembles single-turn generations into episodes; a tool-using agent needs this multi-turn version, where the assembly must additionally distinguish policy text from environment text.",
        "OpenRLHF's agent rollout keeps an action mask per turn for the same reason: only assistant tokens enter the loss, and tool results are context.",
        "verl and slime both run rollout in a separate worker from training, so a trajectory like this one is the unit that crosses between them.",
        "The exercise reuses the deterministic ScriptedModel and FakeTool fixtures from the agent-runtime path, so no live model or network is involved in grading.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One structure carries everything the loss needs: the text, the gradient mask, the budget outcome and the reward.",
            "Separating trainable from non-trainable at assembly time means no downstream loss has to know what a tool is.",
            "A hard turn budget makes a runaway agent produce a usable, labelled training example instead of hanging.",
        ],
        "cons": [
            "One terminal reward for a whole trajectory gives every turn identical credit, including the useless ones.",
            "Tool results sit in context and consume the budget without ever being trained on, so long tool outputs are pure overhead.",
            "Turn-level granularity cannot express a partially correct turn, which is what per-step estimators exist to fix.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "create_training_episodes",
            "license": "MIT",
            "adapted": "The assembly of a generation into a training episode carrying its own mask and one reward per trajectory.",
            "simplifications": "Extended to multiple turns with tool results interleaved, which upstream does not do; upstream generates one response per prompt with no environment interaction. Operates on message dictionaries rather than token ids, so there is no tokenizer and no padding.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2501.12948",
            "section": "2.3 Reinforcement learning with multi-stage training",
        },
    ],
    "tests": [
        {
            "name": "A two-turn episode with one tool call",
            "behavior": "rl.trajectory",
            "code": r"""
from torch_judge.harness.rl import (
    scripted_agent, scripted_tool, assistant_turn, final_turn, trajectory_reward_from_marker
)

model = scripted_agent([
    assistant_turn('let me look it up', 'search', {'query': 'x'}),
    final_turn('the answer is 42'),
] + [final_turn('extra')] * 5)
tools = {'search': scripted_tool('search', ['result: 42'] + ['spare'] * 5)}
out = {fn}(model, tools, [{'role': 'user', 'content': 'what is x?'}], 5,
           trajectory_reward_from_marker('42'))

assert set(out) == {'messages', 'trainable', 'turns', 'tool_calls', 'stop_reason', 'reward'}, sorted(out)
assert [m['role'] for m in out['messages']] == ['user', 'assistant', 'tool', 'assistant'], out['messages']
assert out['trainable'] == [False, True, False, True], out['trainable']
assert out['turns'] == 2 and out['tool_calls'] == 1, (out['turns'], out['tool_calls'])
assert out['stop_reason'] == 'finished', out['stop_reason']
assert out['reward'] == 1.0, out['reward']
""",
        },
        {
            "name": "Only assistant messages are trainable",
            "behavior": "rl.trajectory",
            "code": r"""
from torch_judge.harness.rl import (
    scripted_agent, scripted_tool, assistant_turn, final_turn, trajectory_reward_from_marker
)

model = scripted_agent([
    assistant_turn('step one', 'search', {'query': 'a'}),
    assistant_turn('step two', 'search', {'query': 'b'}),
    final_turn('done'),
] + [final_turn('extra')] * 9)
tools = {'search': scripted_tool('search', ['first', 'second'] + ['spare'] * 9)}
out = {fn}(model, tools, [{'role': 'system', 'content': 'be brief'}], 9,
           trajectory_reward_from_marker('done'))
assert out['turns'] == 3, f"the third turn ends the episode, got {out['turns']}"

assert len(out['trainable']) == len(out['messages']), (len(out['trainable']), len(out['messages']))
for message, flag in zip(out['messages'], out['trainable']):
    assert flag == (message['role'] == 'assistant'), f"{message['role']} marked {flag}"
""",
        },
        {
            "name": "Matches an independent rollout oracle",
            "visibility": "unshown",
            "behavior": "rl.trajectory",
            "failure_message": "The trajectory disagrees with an independently written rollout loop. Check the message order, the trainable flags, and the turn and tool counts.",
            "code": r"""
from torch_judge.harness.rl import (
    scripted_agent, scripted_tool, assistant_turn, final_turn,
    reference_agentic_rollout, trajectory_reward_from_marker,
)

scripts = [
    ([assistant_turn('a', 'search', {'query': '1'}), final_turn('done ok')], ['r1'], 5),
    ([assistant_turn('a', 'search', {'query': '1'}),
      assistant_turn('b', 'search', {'query': '2'}),
      final_turn('done ok')], ['r1', 'r2'], 5),
    ([final_turn('done ok')], [], 3),
    ([assistant_turn('a', 'search', {'query': '1'}),
      assistant_turn('b', 'search', {'query': '2'})], ['r1', 'r2'], 2),
]
# Spare turns and results pad every script past its budget, so a loop that runs
# too long is reported as a disagreement with the oracle rather than as an
# exhausted fixture.
def padded(turns, results, budget):
    slack = budget + 4
    return (scripted_agent(list(turns) + [final_turn('extra')] * slack),
            {'search': scripted_tool('search', list(results) + ['spare'] * slack)})

for turns, results, budget in scripts:
    start = [{'role': 'user', 'content': 'go'}]
    model, tools = padded(turns, results, budget)
    mine = {fn}(model, tools, start, budget, trajectory_reward_from_marker('ok'))
    model, tools = padded(turns, results, budget)
    theirs = reference_agentic_rollout(model, tools, start, budget,
                                       trajectory_reward_from_marker('ok'))
    for key in ('trainable', 'turns', 'tool_calls', 'stop_reason', 'reward'):
        assert mine[key] == theirs[key], f'{key}: {mine[key]} vs {theirs[key]}'
    assert [m['role'] for m in mine['messages']] == [m['role'] for m in theirs['messages']]
""",
        },
        {
            "name": "The turn budget is a hard stop",
            "visibility": "unshown",
            "behavior": "budget.enforcement",
            "failure_message": "The loop ran past max_turns, or reported the wrong stop reason. A truncated episode must be labelled max_turns, not finished.",
            "code": r"""
from torch_judge.harness.rl import (
    scripted_agent, scripted_tool, assistant_turn, trajectory_reward_from_marker
)

# A model that never stops asking for tools.
turns = [assistant_turn(f'step {i}', 'search', {'query': str(i)}) for i in range(10)]
tools = {'search': scripted_tool('search', [f'r{i}' for i in range(10)])}
out = {fn}(scripted_agent(turns), tools, [{'role': 'user', 'content': 'go'}], 3,
           trajectory_reward_from_marker('never'))

assert out['turns'] == 3, f"expected 3 turns, got {out['turns']}"
assert out['stop_reason'] == 'max_turns', out['stop_reason']
assert out['tool_calls'] == 3, out['tool_calls']
assert out['reward'] == 0.0, out['reward']
# A truncated trajectory is still a usable training example.
assert sum(out['trainable']) == 3, out['trainable']
""",
        },
        {
            "name": "Stops immediately when no tool is requested",
            "visibility": "unshown",
            "behavior": "events.ordering",
            "failure_message": "The model was called again after a turn with no tool calls. That turn ends the rollout.",
            "code": r"""
from torch_judge.harness.rl import scripted_agent, final_turn, trajectory_reward_from_marker

# The script holds spare turns on purpose: a loop that keeps going after a
# tool-free turn must be caught by the call count, not by the script running out.
model = scripted_agent([final_turn('done ok')] + [final_turn('extra')] * 6)
out = {fn}(model, {}, [{'role': 'user', 'content': 'go'}], 7,
           trajectory_reward_from_marker('ok'))
assert len(model.calls) == 1, f'the model was called {len(model.calls)} times, expected 1'
assert out['turns'] == 1, f"expected 1 turn, got {out['turns']}"
assert out['stop_reason'] == 'finished', out['stop_reason']
assert out['tool_calls'] == 0, out['tool_calls']
assert len(out['messages']) == 2, out['messages']
""",
        },
        {
            "name": "Several tools in one turn are invoked in order",
            "visibility": "unshown",
            "behavior": "events.ordering",
            "failure_message": "Multiple tool calls in one turn were dropped or reordered. One turn may request several tools; each gets its own message.",
            "code": r"""
from torch_judge.harness.rl import scripted_agent, scripted_tool, final_turn, trajectory_reward_from_marker

multi = {'role': 'assistant', 'content': 'both',
         'tool_calls': [{'name': 'alpha', 'arguments': {'query': 'a'}},
                        {'name': 'beta', 'arguments': {'query': 'b'}}]}
model = scripted_agent([multi, final_turn('done ok')] + [final_turn('extra')] * 6)
tools = {'alpha': scripted_tool('alpha', ['A'] + ['spare'] * 6),
         'beta': scripted_tool('beta', ['B'] + ['spare'] * 6)}
out = {fn}(model, tools, [{'role': 'user', 'content': 'go'}], 5,
           trajectory_reward_from_marker('ok'))

assert out['turns'] == 2, out['turns']
assert out['tool_calls'] == 2, out['tool_calls']
roles = [m['role'] for m in out['messages']]
assert roles == ['user', 'assistant', 'tool', 'tool', 'assistant'], roles
tool_contents = [m['content'] for m in out['messages'] if m['role'] == 'tool']
assert tool_contents == ['A', 'B'], f'tools ran out of order: {tool_contents}'
assert out['trainable'] == [False, True, False, False, True], out['trainable']
""",
        },
        {
            "name": "The caller's message list is not mutated",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "initial_messages was modified in place. Copy it, or a second rollout inherits the first one's conversation.",
            "code": r"""
from torch_judge.harness.rl import (
    scripted_agent, scripted_tool, assistant_turn, final_turn, trajectory_reward_from_marker
)

start = [{'role': 'user', 'content': 'go'}]
snapshot = [dict(m) for m in start]

def build():
    return (scripted_agent([assistant_turn('a', 'search', {'query': '1'}),
                            final_turn('done ok')] + [final_turn('extra')] * 6),
            {'search': scripted_tool('search', ['r1'] + ['spare'] * 6)})

model, tools = build()
first = {fn}(model, tools, start, 5, trajectory_reward_from_marker('ok'))
assert start == snapshot, f'initial_messages was mutated: {start}'

model, tools = build()
second = {fn}(model, tools, start, 5, trajectory_reward_from_marker('ok'))
assert len(first['messages']) == len(second['messages']), 'the second rollout inherited state'
""",
        },
        {
            "name": "Trainable flags stay aligned with messages",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The two lists fell out of step. Append to messages and trainable together on every path.",
            "code": r"""
from torch_judge.harness.rl import (
    scripted_agent, scripted_tool, assistant_turn, final_turn, trajectory_reward_from_marker
)

for budget in (1, 2, 3, 4, 8):
    # The third scripted turn ends the episode. Spare turns follow it so that a
    # loop which keeps going is caught by the turn count, not by an exhausted script.
    turns = [assistant_turn('a', 'search', {'query': '1'}),
             assistant_turn('b', 'search', {'query': '2'}),
             final_turn('done ok')] + [final_turn('extra')] * 8
    tools = {'search': scripted_tool('search', ['r1', 'r2'] + ['spare'] * 8)}
    out = {fn}(scripted_agent(turns), tools,
               [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'go'}],
               budget, trajectory_reward_from_marker('ok'))
    assert len(out['trainable']) == len(out['messages']), (budget, len(out['trainable']), len(out['messages']))
    assert sum(out['trainable']) == out['turns'], (budget, sum(out['trainable']), out['turns'])
    assert out['turns'] == min(budget, 3), (budget, out['turns'])
    for message, flag in zip(out['messages'], out['trainable']):
        assert flag == (message['role'] == 'assistant'), (budget, message['role'], flag)
""",
        },
        {
            "name": "Rejects a bad budget or an unknown tool",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An invalid max_turns or a response naming a missing tool was accepted. Both must raise ValueError.",
            "code": r"""
from torch_judge.harness.rl import (
    scripted_agent, scripted_tool, assistant_turn, final_turn, trajectory_reward_from_marker
)

reward = trajectory_reward_from_marker('ok')
start = [{'role': 'user', 'content': 'go'}]

for bad in (0, -1):
    try:
        {fn}(scripted_agent([final_turn('done ok')]), {}, start, bad, reward)
    except ValueError:
        continue
    raise AssertionError(f'max_turns={bad} should raise ValueError')

model = scripted_agent([assistant_turn('a', 'missing', {'query': '1'}), final_turn('done ok')])
try:
    {fn}(model, {'search': scripted_tool('search', ['r'])}, start, 5, reward)
except ValueError:
    pass
else:
    raise AssertionError('an unknown tool name should raise ValueError')
""",
        },
    ],
    "solution": '''def agentic_rollout_loop(model, tools, initial_messages, max_turns, reward_fn):
    if max_turns < 1:
        raise ValueError(f"max_turns must be at least 1, got {max_turns}")

    messages = [dict(message) for message in initial_messages]
    trainable = [message.get("role") == "assistant" for message in messages]
    turns = 0
    tool_calls = 0
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
                raise ValueError(f"response named unknown tool {name!r}")
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
            trainable.append(False)

    return {
        "messages": messages,
        "trainable": trainable,
        "turns": turns,
        "tool_calls": tool_calls,
        "stop_reason": stop_reason,
        "reward": float(reward_fn(messages)),
    }
''',
}
