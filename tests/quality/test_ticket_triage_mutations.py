"""Mutation gate for the multi-part ticket triage exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "ticket_triage"

MUTATIONS = [
    ("whole reply must match", 1, [("match = LABEL_TAG.search(reply)", "match = LABEL_TAG.fullmatch(reply.strip())")]),
    ("tags cannot span lines", 1, [("LABEL_TAG = re.compile(r\"<label>(.*?)</label>\", re.DOTALL)", "LABEL_TAG = re.compile(r\"<label>(.*?)</label>\")")]),
    ("label not stripped", 1, [("if not match or match.group(1).strip().lower() not in self.labels:", "if not match or match.group(1).lower() not in self.labels:")]),
    ("greedy tag", 1, [("r\"<label>(.*?)</label>\"", "r\"<label>(.*)</label>\"")]),
    ("first valid tag wins", 1, [("        match = LABEL_TAG.search(reply)  # the first tag anywhere; text around the tags is ignored\n",
                                  "        match = next((m for m in LABEL_TAG.finditer(reply) if m.group(1).strip().lower() in self.labels), None)\n")]),
    ("reason not stripped", 1, [("reason.group(1).strip() if reason", "reason.group(1) if reason")]),
    ("no retry", 1, [("        retried = parsed is None\n        if retried:", "        retried = parsed is None\n        if False:")]),
    ("retry omits the failed reply", 1, [("Your previous reply was:\\n{reply}\\n\\n", "Your previous reply did not parse.\\n\\n")]),
    ("temperature dropped on retry", 1, [("parsed = self.parse_reply(self.complete(system=system, prompt=retry, temperature=temperature))", "parsed = self.parse_reply(self.complete(system=system, prompt=retry, temperature=0.0))")]),
    ("examples left out", 1, [("\"tag with a short justification, and nothing else.\\n\\nWorked examples:\\n\\n\" + shots)", "\"tag with a short justification, and nothing else.\")")]),
    ("retried flag lost", 1, [("        return {**parsed, \"retried\": retried}", "        return {**parsed, \"retried\": False}")]),
    ("failures excluded from accuracy", 2, [("return {\"accuracy\": sum(1 for t in tickets if pred[t[\"id\"]] == t[\"label\"]) / n,", "return {\"accuracy\": sum(1 for t in tickets if pred[t[\"id\"]] == t[\"label\"]) / max(1, sum(1 for t in tickets if pred[t[\"id\"]] is not None)),")]),
    ("missing id ignored", 2, [("pred = {t[\"id\"]: predictions.get(t[\"id\"]) for t in tickets}  # a missing prediction counts as None", "pred = {t[\"id\"]: predictions.get(t[\"id\"], t[\"label\"]) for t in tickets}")]),
    ("precision over all tickets", 2, [("\"precision\": hits / len(predicted) if predicted else 0.0", "\"precision\": hits / len(tickets)")]),
    ("recall counts parse failures out", 2, [("actual = [t for t in tickets if t[\"label\"] == label]", "actual = [t for t in tickets if t[\"label\"] == label and pred[t[\"id\"]] is not None]")]),
    ("no leak check", 2, [("        if leaked:  # a worked example would show the model the answer it is graded on", "        if False:")]),
    ("retry count wrong", 2, [("\"retry_count\": sum(1 for r in results.values() if r[\"retried\"])", "\"retry_count\": sum(1 for r in results.values() if r[\"label\"] is None)")]),
    ("changed unsorted", 2, [("changed = sorted(t[\"id\"] for t in tickets", "changed = list(t[\"id\"] for t in tickets[::-1]")]),
    ("one-sided p", 3, [("p_value = min(1.0, 2 * tail)", "p_value = min(1.0, tail)")]),
    ("tail excludes the observed split", 3, [("for i in range(min(a_only, b_only) + 1)", "for i in range(min(a_only, b_only))")]),
    ("winner without significance", 3, [("        if p_value < alpha:", "        if a_only != b_only:")]),
    ("winner at alpha", 3, [("        if p_value < alpha:", "        if p_value <= alpha:")]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
