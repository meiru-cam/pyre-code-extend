"""Mutation gate for the multi-part rate limiter bug hunt exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "rate_limiter"

MUTATIONS = [
    ("stops counting at the edge", 1, [("while log and now - log[0] > period:", "while log and now - log[0] >= period:")]),
    ("one request too many", 1, [("fits = fits and len(log) < limit", "fits = fits and len(log) <= limit")]),
    ("users share a log", 1, [("logs = self.logs[user_id]", "logs = self.logs[None]")]),
    ("rejected requests recorded", 1, [("            if fits:\n                for log in logs:", "            if True:\n                for log in logs:")]),
    ("each rule records as it passes", 2, [("                fits = fits and len(log) < limit\n", "                fits = fits and len(log) < limit\n                if fits:\n                    log.append(now)\n"),
                                           ("            if fits:\n                for log in logs:  # recorded under every rule, or under none\n                    log.append(now)\n", "")]),
    ("only the first rule records", 2, [("                for log in logs:  # recorded under every rule, or under none\n                    log.append(now)", "                logs[0].append(now)")]),
    ("last rule ignored", 3, [("for (limit, period), log in zip(self.rules, logs):", "for (limit, period), log in zip(self.rules[:-1], logs):")]),
    ("no lock", 4, [("        with self.lock:  # deciding", "        if True:  # deciding")]),
    ("lock released before recording", 4, [("            self.checkpoint()\n            if fits:\n                for log in logs:  # recorded under every rule, or under none\n                    log.append(now)\n            return fits\n",
                                              "            pass\n        self.checkpoint()\n        if fits:\n            for log in logs:\n                log.append(now)\n        return fits\n")]),
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
