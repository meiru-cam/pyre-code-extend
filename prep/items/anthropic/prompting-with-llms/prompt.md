A notebook defines `complete`, a function that sends one turn to a language model and returns its reply as plain text.

```py
def complete(system: str, prompt: str, temperature: float = 0.0) -> str:
    """Sends `prompt` to a language model under the given system instructions and returns its text
    reply. Calling complete with the same system and prompt at temperature 0.0 returns the same
    string every time it is called; above 0.0 it may return a different string on each call."""
```

The task is to triage incoming support tickets. A *ticket* is a dict with `id`, `text` and `label`; `TICKETS` is a list of 28 such tickets, and `label` is one of the four strings in `LABELS`, defined as follows.

- `billing`: about a charge, invoice, payment method, refund, or subscription price.
- `bug`: the product is not behaving as documented — an error, a crash, or a control that stops working.
- `how_to`: the sender wants to know how to do something the product already supports.
- `access`: the sender cannot get into their account — login, password, a verification code, or account ownership.

The *tie-break rule*: when a ticket's wording touches more than one category, its label is the action support must take right now, not any secondary cause mentioned in passing. A ticket whose immediate problem is that the sender cannot sign in is labelled `access` even when a billing issue is the stated cause, and a ticket phrased as a question about a control that does not work is labelled `bug`, not `how_to`, whenever that control is supposed to work already.

```python
LABELS = ["billing", "bug", "how_to", "access"]

TICKETS = [
    {"id": "t01", "text": "You charged me twice for my March subscription, can you refund the extra payment?", "label": "billing"},
    {"id": "t02", "text": "My invoice shows a $49 charge but my plan should only cost $29 a month.", "label": "billing"},
    {"id": "t03", "text": "I cancelled my subscription last week but was still billed today.", "label": "billing"},
    {"id": "t04", "text": "Can I get a receipt for tax purposes for my last three payments?", "label": "billing"},
    {"id": "t05", "text": "The annual plan discount wasn't applied at checkout, I paid full price.", "label": "billing"},
    {"id": "t06", "text": "I want to downgrade my plan and get a prorated refund for the unused days.", "label": "billing"},
    {"id": "t07", "text": "My credit card was charged in a currency I did not select at signup.", "label": "billing"},
    {"id": "t08", "text": "The export as CSV option is grayed out and I can't click it, how do I enable it?", "label": "bug"},
    {"id": "t09", "text": "Every time I upload a PNG larger than 2MB the page just goes blank.", "label": "bug"},
    {"id": "t10", "text": "Search results are missing items that were there yesterday, nothing was deleted.", "label": "bug"},
    {"id": "t11", "text": "The dashboard chart shows negative numbers for a metric that can never be negative.", "label": "bug"},
    {"id": "t12", "text": "Dark mode resets to light mode every time I refresh the page.", "label": "bug"},
    {"id": "t13", "text": "Saving a draft sometimes duplicates it into two identical drafts.", "label": "bug"},
    {"id": "t14", "text": "The mobile app crashes immediately after I tap the notifications bell.", "label": "bug"},
    {"id": "t15", "text": "How do I invite a teammate to my workspace?", "label": "how_to"},
    {"id": "t16", "text": "Is there a way to schedule a report to send every Monday morning?", "label": "how_to"},
    {"id": "t17", "text": "Where do I change the default currency shown on invoices?", "label": "how_to"},
    {"id": "t18", "text": "How can I export my project data as a CSV file?", "label": "how_to"},
    {"id": "t19", "text": "What's the difference between the viewer and editor roles?", "label": "how_to"},
    {"id": "t20", "text": "How do I set up a keyboard shortcut for creating a new task?", "label": "how_to"},
    {"id": "t21", "text": "Can you point me to where I turn off email notifications for comments?", "label": "how_to"},
    {"id": "t22", "text": "I can't log in because it says my payment failed and my account is suspended until I update my card.", "label": "access"},
    {"id": "t23", "text": "My account got locked after too many failed login attempts, please unlock it.", "label": "access"},
    {"id": "t24", "text": "Two-factor authentication won't accept the code from my authenticator app.", "label": "access"},
    {"id": "t25", "text": "I'm logged out every few minutes even though I checked 'stay signed in'.", "label": "access"},
    {"id": "t26", "text": "I forgot my password and the reset email never arrives.", "label": "access"},
    {"id": "t27", "text": "Single sign-on redirects me back to the login page without any error.", "label": "access"},
    {"id": "t28", "text": "My teammate left the company and I need to be added as the new account owner.", "label": "access"},
]
```

A colleague's first attempt at evaluating a triage prompt is below. It runs without raising an exception, and it prints a number, but that number is wrong: the cell has exactly four bugs.

```py
LABELS = ["billing", "bug", "how_to", "access"]
SYSTEM = "You are a support triage assistant. Read the ticket and decide which category it belongs to."


def build_prompt(ticket_text, examples):
    shots = "\n\n".join(f"Ticket: {ex['text']}\nLabel: {ex['label']}" for ex in examples)
    return f"{shots}\n\nTicket: {ticket_text}\nLabel:"


LABEL_RE = re.compile(r"^([a-z]+)$")


def parse_label(reply):
    match = LABEL_RE.match(reply.strip())
    return match.group(1) if match else None


def run_eval(tickets):
    examples = tickets[:2]
    correct = 0
    for t in tickets[:20]:
        reply = complete(system=SYSTEM, prompt=build_prompt(t["text"], examples))
        pred = parse_label(reply)
        if pred == t["label"]:
            correct += 1
    return correct / len(tickets)


print(f"accuracy: {run_eval(TICKETS):.2%}")
```

### Part 1 — Prompt, output contract and a robust parser

Design a system prompt for this task and a parser for the model's reply, then combine them into `classify`.

State in the system prompt: the four label definitions and the tie-break rule above; that the reply must contain exactly one `<label>...</label>` tag whose content is one of the four label names, followed by one `<reason>...</reason>` tag with a short justification, and nothing else; and one or two *worked examples* in that same tag format — each a dict with `text`, `label` and `reason`, shown to the model as a demonstration of a ticket already correctly classified. Every worked example must be a ticket that is never one of the 28 in `TICKETS` — an example drawn from the tickets being scored would show the model the answer to a question it is about to be asked again.

```py
def parse_reply(reply: str) -> dict | None:
    """Extracts a prediction from one raw reply. Finds the first `<label>...</label>` tag anywhere
    in `reply` (any text before, between or after the tags is ignored); if none is present, or its
    content -- stripped of surrounding whitespace and compared case-insensitively -- is not one of
    LABELS, returns None. Otherwise returns {"label": <the matched label, lower-cased>, "reason":
    the content of the first `<reason>...</reason>` tag if present, else ""}."""


def classify(ticket_text: str, examples: list[dict], temperature: float = 0.0) -> dict:
    """Builds a prompt from `ticket_text` and the worked `examples` (each a dict with "text",
    "label" and "reason"), calls complete(), and parses the reply with parse_reply. If parsing
    fails, calls complete() exactly once more, with a prompt that includes the failed reply and
    states why it did not parse, and parses that second reply instead. Returns {"label": ... (None
    if both attempts fail to parse), "reason": ... ("" if label is None), "retried": True iff a
    second call was made}."""
```

```text
ticket_text = "Can I get a receipt for tax purposes for my last three payments?"

# first call: the model answers in prose instead of the required tags
complete(system, prompt) -> "This is a billing question -- they want a receipt for past payments."
parse_reply(...) -> None                                    # no <label> tag anywhere

# classify() retries once, with the parse failure fed back into a new prompt
complete(system, retry_prompt) -> ('<label>billing</label>\n'
                                    '<reason>Requests a receipt for past payments.</reason>')
parse_reply(...) -> {"label": "billing", "reason": "Requests a receipt for past payments."}

classify(ticket_text, examples) -> {"label": "billing",
                                     "reason": "Requests a receipt for past payments.",
                                     "retried": True}
```

### Part 2 — An evaluation harness

Separate scoring from calling the model: `score` computes metrics from a predictions dict with no calls to `complete`, and `evaluate` combines it with `classify`.

```py
def score(tickets: list[dict], predictions: dict[str, str | None]) -> dict:
    """Pure function of `tickets` (using each entry's "id" and "label") and `predictions` (ticket id
    -> predicted label, or None for a ticket classify() could not parse). Returns:
    "accuracy": (# tickets whose prediction equals their true label) / len(tickets) -- a missing or
      None prediction is never equal to any label, so it counts as incorrect, not excluded;
    "parse_failure_rate": (# tickets with a None prediction) / len(tickets);
    "per_label": {label: {"precision": ..., "recall": ...} for label in LABELS}, using the standard
      multi-class definitions (precision is 0.0 for a label that was never predicted; a None
      prediction can lower a label's recall but is never a false positive for any label, since it
      was not predicted as anything)."""


def evaluate(tickets: list[dict], examples: list[dict], temperature: float = 0.0) -> dict:
    """Classifies every ticket in `tickets` with classify(ticket["text"], examples, temperature),
    then returns score(tickets, predictions) with two keys added: "predictions" (the dict used) and
    "retry_count" (the number of tickets whose classify() call had "retried": True)."""


def compare(tickets: list[dict], examples_a: list[dict], examples_b: list[dict],
            temperature: float = 0.0) -> dict:
    """Runs evaluate(tickets, examples_a, temperature) and evaluate(tickets, examples_b,
    temperature) on the same tickets. Returns {"a": <first result>, "b": <second result>,
    "changed": sorted list of ticket ids where the two predicted labels differ}."""
```

Then find the bugs. The pipeline shown above has exactly four of them: state each one (which line causes it and why it is wrong) and give a corrected pipeline built from `classify` and `score` above. Fixed, it must score every one of the 28 tickets exactly once, must never show the model a worked example that is also one of the tickets being scored, and must accept a correctly tagged reply regardless of what text surrounds the tags.

```text
tickets = [{"id": "x1", "label": "billing"}, {"id": "x2", "label": "billing"},
           {"id": "x3", "label": "bug"},      {"id": "x4", "label": "bug"},
           {"id": "x5", "label": "how_to"},   {"id": "x6", "label": "access"}]
predictions = {"x1": "billing", "x2": "access", "x3": "bug", "x4": None,
               "x5": "how_to", "x6": "access"}
# x2 is wrong (predicted access, true billing); x4 failed to parse (true bug); the other four are correct

score(tickets, predictions) -> {
    "accuracy": 4 / 6,                                    # x1, x3, x5, x6 correct
    "parse_failure_rate": 1 / 6,                          # x4
    "per_label": {
        "billing": {"precision": 1.0, "recall": 0.5},     # the 1 ticket predicted "billing" (x1) is right,
                                                            # but only 1 of the 2 true billing tickets was recalled
        "bug":     {"precision": 1.0, "recall": 0.5},      # x4's parse failure costs bug a recall point
        "how_to":  {"precision": 1.0, "recall": 1.0},
        "access":  {"precision": 0.5, "recall": 1.0},      # x2's wrong guess costs access its precision
    },
}
```

### Part 3 — Error analysis and deciding what is a real improvement

Read the tickets `evaluate` gets wrong with your Part 1 prompt, and for each one, decide from the label definitions and the tie-break rule which category it should have been, and why the wording could mislead a first pass. Add one or two further worked examples to the prompt that address the pattern you find — never an example that is one of the 28 tickets — and re-run `evaluate` with the enlarged example set.

```py
def paired_improvement(tickets: list[dict], result_a: dict, result_b: dict) -> dict:
    """Compares two evaluate() results computed on the same tickets. Among tickets where exactly one
    side predicted the true label, counts "a_only" (a correct, b incorrect) and "b_only" (b correct,
    a incorrect). Returns {"a_only": ..., "b_only": ..., "p_value": the two-sided exact binomial
    (sign) test p-value for a_only against n = a_only + b_only trials at p = 0.5 -- the probability,
    if the two prompts were equally good and every discordant ticket were a coin flip, of a split at
    least this lopsided; 1.0 when a_only + b_only == 0}."""
```

```text
tickets = [{"id": "y1", "label": "bug"}, {"id": "y2", "label": "access"}, {"id": "y3", "label": "billing"}]
result_a = {"predictions": {"y1": "how_to", "y2": "billing", "y3": "billing"}}   # wrong on y1, y2
result_b = {"predictions": {"y1": "bug",    "y2": "access",  "y3": "billing"}}   # right on all three

paired_improvement(tickets, result_a, result_b)
-> {"a_only": 0, "b_only": 2, "p_value": 0.5}
# both disagreements favor b and none favor a -- but with only two disagreements, the exact
# two-sided sign test can never report p below 0.5, whichever way a two-item split goes
```

State precisely how you would use `paired_improvement`'s output here to decide whether `EXEMPLARS_V2` is a genuine improvement over `EXEMPLARS_V1`, and not a difference a coin flip could as easily have produced.
