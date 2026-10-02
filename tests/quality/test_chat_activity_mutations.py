"""Mutation gate for the multi-part chat activity tracker exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "chat_activity"

_FORGET = '''            if current[2]:
                self._bump(chat[0], -1)  # its latest ping just left the window
            del self._chats[chat]
'''

MUTATIONS = [
    ("window edge excluded", 1, [("while self._events and self._events[0][0] < edge:", "while self._events and self._events[0][0] <= edge:")]),
    ("counts never deleted", 1, [("            if not self._counts[chat]:\n                del self._counts[chat]\n", "")]),
    ("count walks the events", 1, [("        return self._counts.get((user_id, chat_id), 0)",
                                    "        return sum(1 for _, chat in self._events if chat == (user_id, chat_id))")]),
    ("chat_id alone", 1, [("        chat = (user_id, chat_id)\n", "        chat = chat_id\n"),
                          ("self._counts.get((user_id, chat_id), 0)", "self._counts.get(chat_id, 0)")]),
    ("users kept at zero", 1, [("        if count:\n            self._active[user_id] = count", "        if True:\n            self._active[user_id] = count")]),
    ("close ignored", 2, [("active = ping is not None and (close is None or ping > close)", "active = ping is not None")]),
    ("ties go to ping", 2, [("active = ping is not None and (close is None or ping > close)",
                             "active = ping is not None and (close is None or ping[0] >= close[0])")]),
    ("closed chats never forgotten", 2, [(_FORGET, '''            if current[2]:
                self._bump(chat[0], -1)  # its latest ping just left the window
                del self._chats[chat]
''')]),
    ("unknown type accepted", 2, [('        if event_type not in ("ping", "close"):\n            raise ValueError', '        if False:\n            raise ValueError')]),
    ("clock follows the latest arrival", 3, [("self._clock = t if self._clock is None else max(self._clock, t)", "self._clock = t")]),
    ("arrival order decides", 3, [("ping = stamp if ping is None else max(ping, stamp)", "ping = stamp"),
                                  ("close = stamp if close is None else max(close, stamp)", "close = stamp")]),
    ("now ignored", 3, [("        if now is not None:\n            self._advance(now)\n", "")]),
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
