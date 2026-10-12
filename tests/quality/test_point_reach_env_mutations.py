"""Mutation gate for the point_reach_env exercise (Gymnasium reset/step contract)."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from tests.quality.parts_gate import mutant

TASK_ID = "point_reach_env"

_SEEDING = """        if seed is not None:
            self._generator = torch.Generator().manual_seed(seed)
        elif self._generator is None:
            self._generator = torch.Generator()
            self._generator.seed()
"""

# (name, [(text in the reference, replacement), ...]).
EDITS = [
    ("success_reported_as_truncation", [
        ("terminated = success", "terminated = False"),
        ("truncated = self._elapsed >= self.max_steps", "truncated = self._elapsed >= self.max_steps or success"),
    ]),
    ("reseeds_every_unseeded_reset", [
        (_SEEDING, """        if seed is not None:
            self._last_seed = seed
        self._generator = torch.Generator().manual_seed(getattr(self, "_last_seed", 0))
"""),
    ]),
    ("truncates_one_step_late", [
        ("truncated = self._elapsed >= self.max_steps", "truncated = self._elapsed > self.max_steps"),
    ]),
    ("success_suppresses_truncation", [
        ("truncated = self._elapsed >= self.max_steps", "truncated = self._elapsed >= self.max_steps and not success"),
    ]),
    ("step_after_end_allowed", [("self._needs_reset = terminated or truncated", "self._needs_reset = False")]),
    ("state_aliases_position", [('"state": self._pos.clone()', '"state": self._pos')]),
    ("init_state_aliased", [
        ("torch.as_tensor(init_state, dtype=torch.float32).clone()", "torch.as_tensor(init_state, dtype=torch.float32)"),
        ("init_state[:2].clone(), init_state[2:].clone()", "init_state[:2], init_state[2:]"),
    ]),
    ("init_state_consumes_generator", [
        ("            self._pos, self._goal = init_state[:2].clone(), init_state[2:].clone()",
         "            torch.rand(4, generator=self._generator)\n"
         "            self._pos, self._goal = init_state[:2].clone(), init_state[2:].clone()"),
    ]),
    ("goal_drawn_before_start", [
        ("            self._pos = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1\n"
         "            self._goal = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1",
         "            self._goal = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1\n"
         "            self._pos = torch.rand(2, generator=g, dtype=torch.float32) * 2 - 1"),
    ]),
    ("strict_success_threshold", [("<= self.goal_radius", "< self.goal_radius")]),
    ("action_not_clipped", [(".clamp(-self.max_action, self.max_action)", "")]),
    ("position_not_clipped", [("self._pos = (self._pos + delta).clamp(-1.0, 1.0)", "self._pos = self._pos + delta")]),
    ("step_counter_not_reset", [
        ("        self._elapsed = 0\n        self._needs_reset = False", "        self._needs_reset = False"),
    ]),
    ("rows_and_columns_swapped", [
        ("image[0, self._pixel(self._goal[1]), self._pixel(self._goal[0])] = 0.5",
         "image[0, self._pixel(self._goal[0]), self._pixel(self._goal[1])] = 0.5"),
        ("image[0, self._pixel(self._pos[1]), self._pixel(self._pos[0])] = 1.0",
         "image[0, self._pixel(self._pos[0]), self._pixel(self._pos[1])] = 1.0"),
    ]),
    ("goal_drawn_over_agent", [
        ("        image[0, self._pixel(self._goal[1]), self._pixel(self._goal[0])] = 0.5\n"
         "        image[0, self._pixel(self._pos[1]), self._pixel(self._pos[0])] = 1.0",
         "        image[0, self._pixel(self._pos[1]), self._pixel(self._pos[0])] = 1.0\n"
         "        image[0, self._pixel(self._goal[1]), self._pixel(self._goal[0])] = 0.5"),
    ]),
    ("pixel_index_not_clamped", [
        ("min(int(math.floor((float(c) + 1.0) / 2.0 * self.image_size)), self.image_size - 1)",
         "int(math.floor((float(c) + 1.0) / 2.0 * (self.image_size - 1)))"),
    ]),
    ("flags_as_tensors", [
        ("success = self._success()\n", "success = torch.tensor(self._success())\n"),
    ]),
]


def mutations() -> list[Mutation]:
    original = get_task(TASK_ID)["solution"]
    return [Mutation(name, mutant(original, edits)) for name, edits in EDITS]


@pytest.mark.parametrize("_repeat", range(3))
def test_point_reach_env_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, mutations(), require_unshown=True)
    assert set(rejected) == {name for name, _ in EDITS}


def test_point_reach_env_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task)
    assert all(len(source["commit"]) == 40 for source in task["sources"] if source["kind"] == "code")
