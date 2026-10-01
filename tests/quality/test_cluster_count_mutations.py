"""Mutation gate for the multi-part cluster count exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import (
    assert_first_fails_in_part,
    assert_part_mutations_rejected,
    first_failing_part,
)

TASK_ID = "cluster_count"

MUTATIONS = [
    ("leaves count 0", 1, [('"count": (lambda node: 1,', '"count": (lambda node: 0,')]),
    ("machines forget themselves", 1, [("lambda node, answers: 1 + sum(answers)", "lambda node, answers: sum(answers)")]),
    ("finishes on the first answer", 1, [('            if not state["pending"]:\n', "            if True:\n")]),
    ("one query at a time", 1, [('            _, query_type, request_id = message\n            state = self._state.get(request_id)',
                                 '            _, query_type, request_id = message\n            request_id = "only"\n            state = self._state.get(request_id)')]),
    ("subtrees in arrival order", 2, [('combine(self, [state["answers"][c] for c in self.children_ids])',
                                       'combine(self, list(state["answers"].values()))')]),
    ("leaf topology is a bare id", 2, [('"topology": (lambda node: (node.node_id, []),', '"topology": (lambda node: node.node_id,')]),
    ("never resends", 3, [('if child in state["pending"] and self._tick - state["asked_at"][child] >= self.TIMEOUT:',
                           'if False:')]),
    ("resends to every child", 3, [('if child in state["pending"] and self._tick - state["asked_at"][child] >= self.TIMEOUT:',
                                    'if not state["done"] and self._tick - state["asked_at"][child] >= self.TIMEOUT:')]),
    ("repeated request restarts the work", 3, [('            if state is not None:\n                if state["done"]:\n                    self._reply(request_id)\n                return\n',
                                                '            if state is not None and state["done"]:\n                self._reply(request_id)\n                return\n')]),
    ("finished machine asks its children again", 3, [('            if state is not None:\n                if state["done"]:\n                    self._reply(request_id)\n                return\n',
                                                      '            if state is not None and not state["done"]:\n                return\n')]),
    ("timeout too short", 3, [("TIMEOUT = 3", "TIMEOUT = 1")]),
    ("timeout too long", 3, [("TIMEOUT = 3", "TIMEOUT = 12")]),
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


def test_a_solution_without_retries_passes_parts_1_and_2():
    task = get_task(TASK_ID)
    no_ticks = task["solution"].replace("    def on_tick(self):", "    def _unused(self):")
    assert no_ticks != task["solution"]
    assert first_failing_part(task, no_ticks) == 3
