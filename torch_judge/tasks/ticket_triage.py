"""Triage support tickets with a fake model: a tagged reply format, a forgiving parser with one retry, a scoring harness that never leaks examples, and a sign test for prompt changes."""

from ._interview import interview

# A fake model that replays fixed replies per ticket and records every call.
_MODEL = r"""
import itertools, math, random
from fractions import Fraction

LABELS = ["refund", "outage", "question", "login"]

EXAMPLES = [
    {"text": "The app logs me out whenever I switch networks.", "label": "login", "reason": "Cannot stay signed in."},
    {"text": "Your status page says green but uploads fail for everyone here.", "label": "outage", "reason": "A service is down."},
]

class FakeModel:
    # replies[ticket text] = (first reply, reply to the retry); a third call for one ticket fails the test
    def __init__(self, replies):
        self.replies, self.calls, self.seen = replies, [], {}
    def __call__(self, *, system, prompt, temperature):
        self.calls.append({"system": system, "prompt": prompt, "temperature": temperature})
        found = [t for t in self.replies if t in prompt]
        assert len(found) == 1, ("the prompt should hold exactly one scored ticket's text", prompt[:200])
        text = found[0]
        k = self.seen[text] = self.seen.get(text, 0) + 1
        assert k <= 2, ("complete was called a third time for", text)
        first, second = self.replies[text]
        if k == 2:
            assert first in prompt, ("the retry prompt must include the failed reply", first, prompt[:300])
        return first if k == 1 else second

def tagged(label, reason="because"):
    return f"<label>{label}</label>\n<reason>{reason}</reason>"

def m_score(tickets, predictions):
    n = len(tickets)
    pred = [predictions.get(t["id"]) for t in tickets]
    out = {"accuracy": Fraction(sum(p == t["label"] for p, t in zip(pred, tickets)), n),
           "parse_failure_rate": Fraction(sum(p is None for p in pred), n), "per_label": {}}
    for lab in LABELS:
        tp = sum(p == lab and t["label"] == lab for p, t in zip(pred, tickets))
        fp = sum(p == lab and t["label"] != lab for p, t in zip(pred, tickets))
        fn = sum(p != lab and t["label"] == lab for p, t in zip(pred, tickets))
        out["per_label"][lab] = {"precision": Fraction(tp, tp + fp) if tp + fp else 0,
                                 "recall": Fraction(tp, tp + fn) if tp + fn else 0}
    return out

def close(a, b):
    return abs(float(a) - float(b)) < 1e-9

def check_score(got, want, label):
    assert close(got["accuracy"], want["accuracy"]), (label, "accuracy", got["accuracy"], float(want["accuracy"]))
    assert close(got["parse_failure_rate"], want["parse_failure_rate"]), (label, "parse_failure_rate", got["parse_failure_rate"])
    assert set(got["per_label"]) == set(LABELS), (label, sorted(got["per_label"]))
    for lab in LABELS:
        for k in ("precision", "recall"):
            assert close(got["per_label"][lab][k], want["per_label"][lab][k]), (label, lab, k, got["per_label"][lab][k], float(want["per_label"][lab][k]))

def m_sign_p(a_only, b_only):
    n = a_only + b_only
    probs = [Fraction(math.comb(n, k), 2 ** n) for k in range(n + 1)]
    return float(sum(p for p in probs if p <= probs[a_only]))

def rand_tickets(rng, n, prefix="t"):
    return [{"id": f"{prefix}{i:02d}", "text": f"ticket {prefix}{i:02d}: case number {rng.randrange(10**6)}.", "label": rng.choice(LABELS)}
            for i in range(n)]
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "protocol.validation", "code": _MODEL + r"""
ticket = "I was billed after cancelling; please return the charge."
model = FakeModel({ticket: ("Looks like a refund request after a cancellation.", "Sure.\n<label> Refund </label>\n<reason>Wants a charge returned.</reason>")})
got = {fn}(model, LABELS).classify(ticket, EXAMPLES)
assert got == {"label": "refund", "reason": "Wants a charge returned.", "retried": True}, got
assert len(model.calls) == 2, len(model.calls)
"""},
    {"name": "Part 1: the parser", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
     "failure_message": "parse_reply got a reply wrong: tags surrounded by text, padded or upper-case labels, tags spanning lines, a reason before the label, a missing reason, a label not in labels, or a second label tag after an invalid first one.",
     "code": _MODEL + r"""
p = {fn}(None, LABELS)
cases = [
    ("<label>outage</label><reason>down</reason>", {"label": "outage", "reason": "down"}),
    ("Sure! Here you go:\n<label>\n  LOGIN \n</label>\n<reason>\nlocked out\n</reason>\nThanks.", {"label": "login", "reason": "locked out"}),
    ("<reason>asks how</reason> then <label>Question</label>", {"label": "question", "reason": "asks how"}),
    ("<label>refund</label>", {"label": "refund", "reason": ""}),
    ("<label>refund</label><reason>a</reason><label>login</label><reason>b</reason>", {"label": "refund", "reason": "a"}),
    ("<label>billing</label><reason>x</reason>", None),
    ("<label>billing</label><label>refund</label>", None),
    ("refund", None),
    ("<label></label><reason>x</reason>", None),
    ("<LABEL>refund</LABEL>", None),
]
for reply, want in cases:
    got = p.parse_reply(reply)
    assert got == want, (reply, got, want)
"""},
    {"name": "Part 1: prompts, retries and temperature", "part": 1, "visibility": "unshown", "behavior": "retry.classification",
     "failure_message": "classify left a label or a worked example (text, label tag or reason tag) out of the system prompt, left the ticket out of the prompt, retried a reply that parsed, did not retry exactly once, dropped the temperature, or returned the wrong dict.",
     "code": _MODEL + r"""
ticket = "I was charged for a seat nobody uses."
model = FakeModel({ticket: (tagged("refund", "unused seat"), "unused")})
got = {fn}(model, LABELS).classify(ticket, EXAMPLES, temperature=0.7)
assert got == {"label": "refund", "reason": "unused seat", "retried": False}, got
assert len(model.calls) == 1, len(model.calls)
call = model.calls[0]
assert call["temperature"] == 0.7, call["temperature"]
assert ticket in call["prompt"], "the prompt must hold the ticket text"
for lab in LABELS:
    assert lab in call["system"], ("label missing from the system prompt", lab)
for ex in EXAMPLES:
    for piece in (ex["text"], f"<label>{ex['label']}</label>", f"<reason>{ex['reason']}</reason>"):
        assert piece in call["system"], ("worked example missing from the system prompt", piece)
ticket = "Where do I rename a project?"
model = FakeModel({ticket: ("question, I think", "<label>maybe</label>")})
got = {fn}(model, LABELS).classify(ticket, EXAMPLES, temperature=0.3)
assert got == {"label": None, "reason": "", "retried": True}, got
assert [c["temperature"] for c in model.calls] == [0.3, 0.3], [c["temperature"] for c in model.calls]
assert ticket in model.calls[1]["prompt"], "the retry prompt must hold the ticket text"
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "metrics.averaging", "code": _MODEL + r"""
tickets = [{"id": "k1", "label": "refund"}, {"id": "k2", "label": "refund"}, {"id": "k3", "label": "outage"},
           {"id": "k4", "label": "question"}, {"id": "k5", "label": "login"}]
predictions = {"k1": "refund", "k2": None, "k3": "question", "k4": "question", "k5": "login"}
got = {fn}(None, LABELS).score(tickets, predictions)
assert close(got["accuracy"], 0.6) and close(got["parse_failure_rate"], 0.2), got
want = {"refund": (1.0, 0.5), "outage": (0.0, 0.0), "question": (0.5, 1.0), "login": (1.0, 1.0)}
for lab, (pr, rc) in want.items():
    assert close(got["per_label"][lab]["precision"], pr) and close(got["per_label"][lab]["recall"], rc), (lab, got["per_label"][lab])
"""},
    {"name": "Part 2: scoring, evaluation and leaks", "part": 2, "visibility": "unshown", "behavior": "metrics.averaging",
     "failure_message": "score disagreed with exact precision, recall, accuracy and parse-failure counts (missing predictions count as None), evaluate did not classify every ticket once or miscounted retries, compare listed the wrong changed ids, or a worked example that is also a scored ticket did not raise ValueError before any call.",
     "code": _MODEL + r"""
p = {fn}(None, LABELS)
for seed in range(150):
    rng = random.Random(seed)
    tickets = rand_tickets(rng, rng.randint(1, 12))
    predictions = {t["id"]: rng.choice(LABELS + [None]) for t in tickets if rng.random() < 0.9}
    check_score(p.score(tickets, predictions), m_score(tickets, predictions), seed)
rng = random.Random(7)
tickets = rand_tickets(rng, 9)
replies, labels_a = {}, {}
for i, t in enumerate(tickets):
    kind = i % 3
    lab = rng.choice(LABELS)
    replies[t["text"]] = [(tagged(lab), "x"), ("no tags", tagged(lab)), ("no tags", "still none")][kind]
    labels_a[t["id"]] = None if kind == 2 else lab
model = FakeModel(replies)
res = {fn}(model, LABELS).evaluate(tickets, EXAMPLES, temperature=0.0)
assert res["predictions"] == labels_a, res["predictions"]
assert res["retry_count"] == 6, res["retry_count"]
assert len(model.calls) == 15, len(model.calls)
check_score(res, m_score(tickets, labels_a), "evaluate")
model = FakeModel({t["text"]: (tagged("refund"), "x") for t in tickets})
leak = EXAMPLES + [{"text": tickets[4]["text"], "label": tickets[4]["label"], "reason": "r"}]
try:
    {fn}(model, LABELS).evaluate(tickets, leak)
except Exception as e:
    assert type(e).__name__ == "ValueError", type(e).__name__
else:
    raise AssertionError("a worked example that is a scored ticket must raise ValueError")
assert model.calls == [], "evaluate called the model before rejecting a leaked example"
class Switch:
    # answers from one table for prompts built with EXAMPLES[:1], from another otherwise
    def __init__(self, a, b):
        self.a, self.b = a, b
    def __call__(self, *, system, prompt, temperature):
        table = self.a if EXAMPLES[1]["text"] not in system + prompt else self.b
        return next(tagged(v) for k, v in table.items() if k in prompt)
a = {t["text"]: rng.choice(LABELS) for t in tickets}
b = {k: (v if i % 2 else rng.choice(LABELS)) for i, (k, v) in enumerate(a.items())}
cmp = {fn}(Switch(a, b), LABELS).compare(tickets, EXAMPLES[:1], EXAMPLES)
want = sorted(t["id"] for t in tickets if a[t["text"]] != b[t["text"]])
assert cmp["changed"] == want, (cmp["changed"], want)
assert cmp["a"]["predictions"] == {t["id"]: a[t["text"]] for t in tickets}, cmp["a"]["predictions"]
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "metrics.ties", "code": _MODEL + r"""
tickets = [{"id": f"s{i}", "label": "refund"} for i in range(10)]
a = {"predictions": {f"s{i}": ("refund" if i >= 6 else "login") for i in range(10)}}
b = {"predictions": {f"s{i}": "refund" for i in range(10)}}
got = {fn}(None, LABELS).paired_improvement(tickets, a, b)
assert got["a_only"] == 0 and got["b_only"] == 6, got
assert close(got["p_value"], 2 / 64) and got["winner"] == "b", got
a["predictions"]["s6"] = "login"
b["predictions"]["s6"] = None
a["predictions"]["s7"], b["predictions"]["s7"] = "refund", "outage"
got = {fn}(None, LABELS).paired_improvement(tickets, a, b)
assert (got["a_only"], got["b_only"]) == (1, 6), got
assert close(got["p_value"], 16 / 128) and got["winner"] is None, got
"""},
    {"name": "Part 3: sign test on random splits", "part": 3, "visibility": "unshown", "behavior": "metrics.ties",
     "failure_message": "paired_improvement miscounted the discordant tickets, gave a p-value other than the exact two-sided sign test (1.0 with no discordant tickets), or named a winner when p_value was not strictly below alpha.",
     "code": _MODEL + r"""
p = {fn}(None, LABELS)
for seed in range(200):
    rng = random.Random(500 + seed)
    tickets = rand_tickets(rng, rng.randint(0, 30))
    ra = {"predictions": {t["id"]: (t["label"] if rng.random() < 0.6 else rng.choice(LABELS + [None])) for t in tickets}}
    rb = {"predictions": {t["id"]: (t["label"] if rng.random() < 0.7 else rng.choice(LABELS + [None])) for t in tickets}}
    alpha = rng.choice([0.01, 0.05, 0.2])
    got = p.paired_improvement(tickets, ra, rb, alpha=alpha)
    ok = lambda r, t: r["predictions"].get(t["id"]) == t["label"]
    a_only = sum(ok(ra, t) and not ok(rb, t) for t in tickets)
    b_only = sum(ok(rb, t) and not ok(ra, t) for t in tickets)
    assert (got["a_only"], got["b_only"]) == (a_only, b_only), (seed, got, a_only, b_only)
    want_p = m_sign_p(a_only, b_only)
    assert close(got["p_value"], want_p), (seed, got["p_value"], want_p)
    want_w = None if want_p >= alpha else ("b" if b_only > a_only else "a")
    assert got["winner"] == want_w, (seed, got["winner"], want_w, want_p, alpha)
tickets = [{"id": f"e{i}", "label": "login"} for i in range(5)]
got = p.paired_improvement(tickets, {"predictions": {}}, {"predictions": {t["id"]: "login" for t in tickets}}, alpha=0.0625)
assert close(got["p_value"], 0.0625) and got["winner"] is None, ("p_value equal to alpha is not significant", got)
"""},
]

TASK = {
    "title": "Ticket Triage with an LLM",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "TicketTriage",
    "description_en": r"""Write `TicketTriage`, which labels support tickets by prompting a language model, parses its reply, and measures how well a prompt does. The tests replace the model with a fake `complete` that replays fixed replies, so only your prompt structure, parsing and arithmetic are graded.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `TicketTriage` passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `TicketTriage(complete, labels)`: `complete(system=..., prompt=..., temperature=...)` returns the model's reply as a string, and `labels` is the list of allowed label names, in lower case.
- A ticket is a dict with `"id"`, `"text"` and `"label"`. A worked example is a dict with `"text"`, `"label"` and `"reason"`.
- The reply format asks for one `<label>...</label>` tag holding a label, then one `<reason>...</reason>` tag. Tag names are lower case exactly as written.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** applied roles test whether you treat a prompt like code: a written output contract, a parser that tolerates harmless noise, an evaluation that does not cheat, and a statistical answer to "is the new prompt better?". Each later part adds one requirement: a scoring harness, then a significance test.

**Where it is used:** classification with a model, such as routing tickets, tagging content or grading answers, runs this loop. A tagged output format, one retry, held-out examples and paired comparisons keep the numbers honest.

Adapted from the prompting and evaluation question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded. The notebook functions become a `TicketTriage` class with the labels passed in (the tests use refund, outage, question and login), the live model becomes a fake `complete`, and the tickets are new. The system prompt is checked only for labels and worked examples, not for label definitions or the tie-break rule, and the retry prompt need not say why the reply failed. `alpha` and `winner` are new. The colleague's buggy evaluation cell is left out; its bugs become rules of `score` and `evaluate`. The error-analysis write-up is left out, and the decision rule becomes the graded `winner` field.""",
    "parts": [
        {
            "title": "Prompt, parser and one retry",
            "description_en": r"""**Signature:** `TicketTriage(complete, labels)` with `parse_reply(reply) -> dict | None` and `classify(ticket_text, examples, temperature=0.0) -> dict`

- `parse_reply` finds the first `<label>` tag anywhere in the reply; text around and between tags is ignored, and tags may span lines. If there is no such tag, or its content, stripped and lower-cased, is not in `labels`, it returns `None`.
- Otherwise it returns `{"label": <that label>, "reason": <the stripped content of the first <reason> tag, or "">}`.
- `classify` calls `complete` with a system prompt and a prompt, passing `temperature` through. The system prompt names every label and shows each worked example with its text, `<label>{label}</label>` and `<reason>{reason}</reason>`. The prompt holds the ticket text.
- If the reply does not parse, `classify` calls `complete` exactly once more, with the same system prompt and temperature and a prompt holding the ticket text and the failed reply, and parses that reply instead.
- It returns `{"label", "reason", "retried"}`. `retried` is `True` exactly when a second call was made; when neither reply parses, `label` is `None` and `reason` is `""`.

**Example:** with `labels = ["refund", "outage", "question", "login"]`, the ticket `"I was billed after cancelling; please return the charge."`:
- the first reply is `"Looks like a refund request after a cancellation."`, which has no tag
- the retry's reply is `"Sure.\n<label> Refund </label>\n<reason>Wants a charge returned.</reason>"`
- `classify` returns `{"label": "refund", "reason": "Wants a charge returned.", "retried": True}` after two calls""",
        },
        {
            "title": "An evaluation harness",
            "description_en": r"""Keep Part 1. Add `score(tickets, predictions)`, `evaluate(tickets, examples, temperature=0.0)` and `compare(tickets, examples_a, examples_b, temperature=0.0)`.

- `score` makes no model calls. `predictions` maps ticket id to a label or `None`, and a missing id counts as `None`. It returns `accuracy` (correct over all tickets), `parse_failure_rate` (`None` over all tickets), and `per_label[label] = {"precision", "recall"}` for every label.
- A `None` prediction is never correct and never a false positive. Precision is `0.0` for a label never predicted, and recall is `0.0` for a label no ticket has.
- `evaluate` first raises `ValueError` if any worked example's text equals a scored ticket's text, before any call. It then classifies every ticket once and returns `score`'s dict plus `"predictions"` and `"retry_count"`, the number of tickets that were retried.
- `compare` evaluates both example sets on the same tickets and returns `{"a", "b", "changed"}`, where `changed` is the sorted list of ids whose predictions differ.

**Example:** five tickets `k1` to `k5` with true labels refund, refund, outage, question, login and predictions refund, `None`, question, question, login:
- `accuracy` is `0.6` and `parse_failure_rate` is `0.2`
- refund has precision `1.0` and recall `0.5`; outage, never predicted, has `0.0` and `0.0`; question has `0.5` and `1.0`; login has `1.0` and `1.0`""",
        },
        {
            "title": "Is the new prompt better?",
            "description_en": r"""Keep Parts 1–2. Add `paired_improvement(tickets, result_a, result_b, alpha=0.05) -> dict`. The two results come from `evaluate` on the same tickets.

- `a_only` counts tickets that `result_a` predicts correctly and `result_b` does not; `b_only` is the reverse.
- `p_value` is the exact two-sided sign test: with `n = a_only + b_only`, take `2 * P(X <= min(a_only, b_only))` for `X ~ Binomial(n, 0.5)`, capped at `1.0`. It is `1.0` when `n == 0`.
- `winner` is `"b"` or `"a"`, whichever has the larger count, when `p_value < alpha`; otherwise it is `None`.

**Example:** on ten tickets, `b` alone is right on six and `a` alone on none:
- `p_value` is `2/64`, below `0.05`, so `winner` is `"b"`
- if `a` alone is also right on one more ticket, `p_value` is `16/128 = 0.125` and `winner` is `None`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which harmless differences should the parser accept: text around the tags, spaces or a newline inside them, a capital letter? Which should it reject? When the first reply fails, what does the model need to see to answer correctly the second time?"},
        {"level": 2, "kind": "analysis", "content": "Search, don't match: a non-greedy pattern like <label>(.*?)</label> with DOTALL finds the first tag anywhere. Strip and lower-case its content, then check it against labels. Build the system prompt from the labels and the examples in the exact tag format. On a failed parse, send one more prompt holding the ticket and the bad reply, and record retried."},
    ],
    "model_connections": [
        "Model-based classifiers and graders use a strict tagged output format, parse it leniently, and retry once with the parse error fed back.",
        "Prompt and model changes are judged on held-out sets with paired tests, because two prompts scored on the same items disagree on only a few of them.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Tags give the parser one clear target, while text around them is tolerated.",
            "Keeping scoring a pure function of predictions lets you rescore without calling the model again.",
            "A paired sign test uses only the tickets where the prompts disagree, which is the evidence that matters.",
        ],
        "cons": [
            "A retry doubles the cost for every reply that fails to parse.",
            "Exact-text leak checks miss examples that are paraphrases of scored tickets.",
            "With few disagreeing tickets the sign test cannot reach significance, whatever the split.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import math
import re

LABEL_TAG = re.compile(r"<label>(.*?)</label>", re.DOTALL)
REASON_TAG = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)


class TicketTriage:
    def __init__(self, complete, labels):
        self.complete = complete  # complete(system=..., prompt=..., temperature=...) -> reply text
        self.labels = list(labels)

    # -- part 1: prompt, parser, one retry
    def system_prompt(self, examples):
        shots = "\n\n".join(
            f"Ticket: {ex['text']}\n<label>{ex['label']}</label>\n<reason>{ex['reason']}</reason>" for ex in examples)
        return (
            "You triage support tickets. Choose exactly one label from: " + ", ".join(self.labels) + ".\n"
            "If a ticket fits several labels, pick the one for what must be fixed first.\n"
            "Reply with exactly one <label>...</label> tag holding the label, then one <reason>...</reason> "
            "tag with a short justification, and nothing else.\n\nWorked examples:\n\n" + shots)

    def parse_reply(self, reply):
        match = LABEL_TAG.search(reply)  # the first tag anywhere; text around the tags is ignored
        if not match or match.group(1).strip().lower() not in self.labels:
            return None
        reason = REASON_TAG.search(reply)
        return {"label": match.group(1).strip().lower(), "reason": reason.group(1).strip() if reason else ""}

    def classify(self, ticket_text, examples, temperature=0.0):
        system = self.system_prompt(examples)
        prompt = f"Ticket: {ticket_text}"
        reply = self.complete(system=system, prompt=prompt, temperature=temperature)
        parsed = self.parse_reply(reply)
        retried = parsed is None
        if retried:
            retry = (f"{prompt}\n\nYour previous reply was:\n{reply}\n\nIt did not parse: it needs one "
                     f"<label> tag holding one of {', '.join(self.labels)}. Reply again in the required format.")
            parsed = self.parse_reply(self.complete(system=system, prompt=retry, temperature=temperature))
        if parsed is None:
            return {"label": None, "reason": "", "retried": True}
        return {**parsed, "retried": retried}

    # -- part 2: scoring apart from calling the model
    def score(self, tickets, predictions):
        n = len(tickets)
        pred = {t["id"]: predictions.get(t["id"]) for t in tickets}  # a missing prediction counts as None
        per_label = {}
        for label in self.labels:
            predicted = [t for t in tickets if pred[t["id"]] == label]
            actual = [t for t in tickets if t["label"] == label]
            hits = sum(1 for t in predicted if t["label"] == label)
            per_label[label] = {"precision": hits / len(predicted) if predicted else 0.0,
                                "recall": hits / len(actual) if actual else 0.0}
        return {"accuracy": sum(1 for t in tickets if pred[t["id"]] == t["label"]) / n,
                "parse_failure_rate": sum(1 for t in tickets if pred[t["id"]] is None) / n,
                "per_label": per_label}

    def evaluate(self, tickets, examples, temperature=0.0):
        scored = {t["text"] for t in tickets}
        leaked = [ex["text"] for ex in examples if ex["text"] in scored]
        if leaked:  # a worked example would show the model the answer it is graded on
            raise ValueError(f"worked example is a scored ticket: {leaked[0]!r}")
        results = {t["id"]: self.classify(t["text"], examples, temperature) for t in tickets}
        predictions = {i: r["label"] for i, r in results.items()}
        return {**self.score(tickets, predictions), "predictions": predictions,
                "retry_count": sum(1 for r in results.values() if r["retried"])}

    def compare(self, tickets, examples_a, examples_b, temperature=0.0):
        a = self.evaluate(tickets, examples_a, temperature)
        b = self.evaluate(tickets, examples_b, temperature)
        changed = sorted(t["id"] for t in tickets if a["predictions"][t["id"]] != b["predictions"][t["id"]])
        return {"a": a, "b": b, "changed": changed}

    # -- part 3: is the difference real?
    def paired_improvement(self, tickets, result_a, result_b, alpha=0.05):
        a_only = b_only = 0
        for t in tickets:
            a_ok = result_a["predictions"].get(t["id"]) == t["label"]
            b_ok = result_b["predictions"].get(t["id"]) == t["label"]
            a_only += a_ok and not b_ok
            b_only += b_ok and not a_ok
        n = a_only + b_only
        tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n
        p_value = min(1.0, 2 * tail)  # two-sided exact sign test; n == 0 gives 1.0
        winner = None
        if p_value < alpha:
            winner = "b" if b_only > a_only else "a"
        return {"a_only": a_only, "b_only": b_only, "p_value": p_value, "winner": winner}
''',
    "interview_questions": interview(
        concept=[
            "Why search for the first label tag instead of matching the whole reply, and which replies does each approach reject?",
            "What should a retry prompt contain so the model can fix its format the second time?",
        ],
        deep_dive=[
            "How does the parser handle a label tag whose content is not an allowed label, and why return None rather than guess?",
        ],
        tradeoffs=[
            "Why must a worked example never be one of the scored tickets, and how would you catch near-duplicates?",
            "Why count a parse failure as wrong rather than leaving it out of accuracy?",
            "Why compare two prompts with a paired sign test on the tickets where they disagree instead of comparing their accuracies?",
            "With temperature above zero, how would you change the evaluation so one lucky run does not decide?",
        ],
    ),
}
