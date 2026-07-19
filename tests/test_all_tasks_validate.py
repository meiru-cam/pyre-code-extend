"""Every registered task must satisfy the schema."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge.tasks import TASKS
from torch_judge.tasks._schema import validate_task


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_task_validates(task_id):
    validate_task(task_id, TASKS[task_id], known_ids=set(TASKS))
