"""RLVR format reward — a deterministic, verifiable reward over response structure."""

TASK = {
    "title": "RLVR Format Reward",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "rlvr_format_reward",
    "description_en": r"""Implement the graded structural reward that teaches a model to separate its reasoning from its final answer.

**Signature:** `rlvr_format_reward(completion, eos_token) -> float`

**Parameters:**
- `completion` — the decoded string the policy produced.
- `eos_token` — the end-of-sequence string the completion is required to end with.

**Returns:** a Python float, one of 0.0, 0.5 or 1.0.

The target structure is a reasoning block followed by an answer block, using the literal tags `think` and `answer` written as XML-style open and close tags.

**Grading rules,** applied in order:

1. If `completion` does not end with `eos_token`, return 0.0. Otherwise remove that suffix and call the remainder the body.
2. If the body does not contain each of the four tags exactly once, return 0.0.
3. If the four tags do not appear in the order open-think, close-think, open-answer, close-answer, return 0.0.
4. If the body is exactly the think block followed by the answer block — apart from whitespace between and around them — and neither block's content is empty or whitespace-only, return 1.0.
5. Otherwise return 0.5.

**Constraints:**
- Return a float, not a bool and not an int.
- Count each tag. A repeated tag scores 0.0 even though every tag appears.
- Compare the four tag positions. Answer-before-reasoning scores 0.0.
- Grade structure only. Do not parse or judge the answer's content.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the 0.5 tier exists.** It covers a response whose tags are all present, unique and correctly ordered, but which also carries text outside the blocks or leaves a block empty. A binary reward gives a model no gradient to climb from nearly formatted to formatted, and early in training almost every sample is nearly formatted.

**Why verifiable rewards.** This function is a complete reward model: deterministic, free to evaluate, and impossible to exploit by producing text that merely looks confident. That is the whole premise of RLVR. The tradeoff is that it only rewards what can be checked mechanically, so it is always paired with a task-correctness term — a policy left alone with a format reward will learn to emit perfectly formatted nonsense.""",
    "advisory_prerequisites": [],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Start at the end: what string method tells you whether the completion terminates correctly, and how do you remove that suffix without accidentally stripping a longer or shorter piece? For the tags, what method counts occurrences rather than merely detecting one, and why does the difference matter for a model that has learned to repeat itself? Once you know each tag appears once, what does comparing their positions tell you about ordering? Finally, for the top tier you must distinguish 'exactly these two blocks' from 'these two blocks plus extra text' — if you rebuild the ideal string from the two block contents, what comparison decides the tier?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Work in four stages so each rule maps to a few lines. First `completion.endswith(eos_token)`; if true, slice the body with a negative index of `len(eos_token)`, guarding against an empty eos_token. Second, call `body.count(tag)` for each of the four tags and require every count to be exactly one — this is what rejects a repeated block, which `in` would accept. Third, take `body.index(tag)` for each and require the four indices to be strictly increasing. Fourth, slice the two contents from between their tags and compare a reconstructed ideal against the stripped body: if the body with whitespace removed at the edges equals the think block followed directly by the answer block, also joined without extra text, and both contents are non-empty after stripping, the response earns 1.0. Everything that reaches this point and fails that last comparison earns 0.5. Avoid a single regular expression that tries to do all four stages at once — it collapses the tiers into pass or fail, which is exactly the gradient this reward exists to provide.",
        },
    ],
    "model_connections": [
        "nano-aha-moment's format_reward_func grades the same structure into three tiers of 0, 0.5 and 1.0, and its compute_reward adds that to a separate correctness term.",
        "simple_GRPO's reward_format checks the identical think and answer tag structure and returns a positive or negative constant, which its gen_samples sums with reward_correct.",
        "DeepSeek-R1-Zero's training used a rule-based reward with exactly this split between a format reward and an accuracy reward, and reported that a purely learned reward model invited reward hacking at scale.",
        "Production RLVR pipelines keep every reward term deterministic and offline for the same reason this exercise is gradeable at all: the reward is a function, not a model.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Deterministic and free, so it can be evaluated for every sample of every rollout without a reward-model forward pass.",
            "Cannot be gamed by persuasive text, unlike a learned preference model.",
            "The graded 0.5 tier gives a usable gradient early in training when almost no sample is fully formatted.",
        ],
        "cons": [
            "It rewards structure only, so a policy optimizing it alone converges to well-formatted nonsense.",
            "The tag vocabulary is hard-coded, so changing the response schema silently invalidates every past reward.",
            "Discrete tiers mean a large population of samples share an identical reward, which shrinks the effective group variance that GRPO's advantage depends on.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "format_reward_func",
            "license": "MIT",
            "adapted": "The idea of a graded rather than binary structural reward over a think-then-answer completion, and the end-of-sequence requirement.",
            "simplifications": "The tier rules differ from upstream and are defined by this contract. Upstream is specific to the Countdown task: its 0.5 tier means the response is well formed but the answer body is not a pure arithmetic expression, whereas this exercise's 0.5 means the tags are present, unique and ordered but the response carries extra text or an empty block. Upstream also synthetically prepends an opening think tag, because its prompt template ends with one, and requires a literal newline between the two blocks; neither applies here. This is a divergent contract, not a transcription, so it has no vendored oracle.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2501.12948",
            "section": "2.2.2 Reward Modeling",
        },
    ],
    "tests": [
        {
            "name": "Well-formed completion scores the top tier",
            "behavior": "rl.reward_verifiable",
            "code": r"""
text = '<think>two plus two</think><answer>4</answer><eos>'
out = {fn}(text, '<eos>')
assert isinstance(out, float), f'expected a float, got {type(out).__name__}'
assert out == 1.0, out
# Whitespace between and around the blocks is allowed.
assert {fn}('  <think>a</think>\n<answer>b</answer>  <eos>', '<eos>') == 1.0
""",
        },
        {
            "name": "Missing end-of-sequence scores zero",
            "behavior": "rl.reward_verifiable",
            "code": r"""
assert {fn}('<think>a</think><answer>b</answer>', '<eos>') == 0.0
# Correct structure but a different terminator is still a truncated response.
assert {fn}('<think>a</think><answer>b</answer><pad>', '<eos>') == 0.0
""",
        },
        {
            "name": "Extra text outside the blocks scores the middle tier",
            "visibility": "unshown",
            "behavior": "rl.reward_verifiable",
            "failure_message": "A completion with all four tags present, unique and ordered but carrying text outside the blocks must score 0.5, not 0.0 and not 1.0.",
            "code": r"""
cases = ('Sure! <think>a</think><answer>b</answer><eos>',
         '<think>a</think> and so <answer>b</answer><eos>',
         '<think>a</think><answer>b</answer> hope that helps<eos>')
for text in cases:
    score = {fn}(text, '<eos>')
    assert score == 0.5, f'{text!r} scored {score}, expected 0.5'
""",
        },
        {
            "name": "Empty blocks score the middle tier",
            "visibility": "unshown",
            "behavior": "rl.reward_verifiable",
            "failure_message": "An empty or whitespace-only think or answer block must score 0.5, not 1.0. The tags are correct but the response carries no content.",
            "code": r"""
assert {fn}('<think></think><answer>b</answer><eos>', '<eos>') == 0.5
assert {fn}('<think>a</think><answer>   </answer><eos>', '<eos>') == 0.5
assert {fn}('<think>  </think><answer>\n</answer><eos>', '<eos>') == 0.5
""",
        },
        {
            "name": "Repeated tags score zero",
            "visibility": "unshown",
            "behavior": "rl.reward_verifiable",
            "failure_message": "A repeated tag was accepted. Count each tag and require exactly one occurrence; a substring test accepts a model that loops.",
            "code": r"""
cases = ('<think>a</think><think>b</think><answer>c</answer><eos>',
         '<think>a</think><answer>b</answer><answer>c</answer><eos>',
         '<think>a</think></think><answer>b</answer><eos>')
for text in cases:
    assert {fn}(text, '<eos>') == 0.0, f'{text!r} should score 0.0'
""",
        },
        {
            "name": "Wrong tag order scores zero",
            "visibility": "unshown",
            "behavior": "rl.reward_verifiable",
            "failure_message": "Tags out of order were accepted. Compare the four tag positions and require them to increase.",
            "code": r"""
cases = ('<answer>b</answer><think>a</think><eos>',
         '</think><think>a<answer>b</answer><eos>',
         '<think>a</think></answer>b<answer><eos>')
for text in cases:
    score = {fn}(text, '<eos>')
    assert score == 0.0, f'{text!r} scored {score}, expected 0.0'
""",
        },
        {
            "name": "Missing tags score zero",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "A completion missing one of the four tags was not rejected. All four must be present exactly once.",
            "code": r"""
cases = ('<think>a</think><eos>',
         '<answer>b</answer><eos>',
         '<think>a<answer>b</answer><eos>',
         '<eos>',
         'no tags at all<eos>')
for text in cases:
    assert {fn}(text, '<eos>') == 0.0, f'{text!r} should score 0.0'
""",
        },
        {
            "name": "Returns exactly one of the three tiers",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "A value outside {0.0, 0.5, 1.0} or a non-float was returned. The reward is a graded float with three levels.",
            "code": r"""
samples = ('<think>a</think><answer>b</answer><eos>',
           'x<think>a</think><answer>b</answer><eos>',
           '<think>a</think><eos>',
           '',
           '<eos>',
           '<think></think><answer></answer><eos>')
for text in samples:
    out = {fn}(text, '<eos>')
    assert isinstance(out, float) and not isinstance(out, bool), f'{text!r} returned {type(out).__name__}'
    assert out in (0.0, 0.5, 1.0), f'{text!r} returned {out}'
""",
        },
        {
            "name": "Multi-line reasoning is accepted",
            "visibility": "unshown",
            "behavior": "rl.reward_verifiable",
            "failure_message": "A newline inside the think block broke the match. Reasoning spans many lines, so the block content must not be restricted to a single line.",
            "code": r"""
text = '<think>step one\nstep two\nstep three</think><answer>42</answer><eos>'
assert {fn}(text, '<eos>') == 1.0, {fn}(text, '<eos>')
""",
        },
    ],
    "solution": '''THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
TAGS = (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)


def rlvr_format_reward(completion, eos_token):
    if not eos_token or not completion.endswith(eos_token):
        return 0.0
    body = completion[: -len(eos_token)]

    if any(body.count(tag) != 1 for tag in TAGS):
        return 0.0

    positions = [body.index(tag) for tag in TAGS]
    if positions != sorted(positions):
        return 0.0

    think = body[positions[0] + len(THINK_OPEN) : positions[1]]
    answer = body[positions[2] + len(ANSWER_OPEN) : positions[3]]

    ideal = THINK_OPEN + think + THINK_CLOSE + ANSWER_OPEN + answer + ANSWER_CLOSE
    if body.strip() != ideal.strip():
        gap = body[positions[1] + len(THINK_CLOSE) : positions[2]]
        prefix = body[: positions[0]]
        suffix = body[positions[3] + len(ANSWER_CLOSE) :]
        if prefix.strip() or suffix.strip() or gap.strip():
            return 0.5
    if not think.strip() or not answer.strip():
        return 0.5
    return 1.0
''',
}
