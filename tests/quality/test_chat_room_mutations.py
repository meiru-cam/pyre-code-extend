"""Mutation gate for the multi-part chat bot refactoring exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "chat_room"

MUTATIONS = [
    ("reminder at the end minute", 1, [("if user in text and now < end]", "if user in text and now <= end]")]),
    ("reminders sorted by name", 1, [("for user, end in self._until.items() if", "for user, end in sorted(self._until.items()) if")]),
    ("focus applied before reminders", 1, [("        lines = self._reminders(text)  # decided before this message's own /focus takes effect\n        minutes = self._minutes(text)\n        if minutes is not None:\n            end = self._clock.now + minutes\n            self._until[sender] = end  # reassigning a key keeps its place in the dict\n",
                                            "        minutes = self._minutes(text)\n        if minutes is not None:\n            self._until[sender] = self._clock.now + minutes\n        lines = self._reminders(text)\n        if minutes is not None:\n            end = self._clock.now + minutes\n")]),
    ("replaced session moves last", 1, [("            self._until[sender] = end  # reassigning a key keeps its place in the dict\n",
                                         "            self._until.pop(sender, None)\n            self._until[sender] = end\n")]),
    ("votes case-sensitive", 1, [("        letter = arg.upper()\n", "        letter = arg\n")]),
    ("one-option poll accepted", 1, [("return pieces if len(pieces) >= 3 else None", "return pieces if len(pieces) >= 2 else None")]),
    ("state shared at class level", 1, [("    def __init__(self, cheer_counts, bus=None):\n        self._counts = cheer_counts\n",
                                         "    _shared = {}\n\n    def __init__(self, cheer_counts, bus=None):\n        self._counts = CheerBot._shared\n")]),
    ("unicode digits accepted", 1, [("not (arg.isascii() and arg.isdigit())", "not arg.isdigit()")]),
    ("tab after the command word", 1, [("return text[len(prefix):].strip() if text.startswith(prefix) else None",
                                        "return text[len(prefix):].strip() if text.startswith(prefix) or text.startswith(command + \"\\t\") else None")]),
    ("log returned live", 1, [("        return list(self._log)\n", "        return self._log\n")]),
    ("bots in reverse order", 1, [("        for bot in self._bots:\n", "        for bot in reversed(self._bots):\n")]),
    ("no failure isolation", 2, [("            except Exception:\n", "            except ZeroDivisionError:\n")]),
    ("clashes allowed", 2, [("        if taken:\n            raise ValueError", "        if False:\n            raise ValueError")]),
    ("claim kept on rejection", 2, [("        taken = [command for command in commands if command in self._claimed]\n        if taken:\n",
                                     "        taken = [command for command in commands if command in self._claimed]\n        self._claimed.update(commands)\n        if taken:\n")]),
    ("handle without can_handle", 1, [("                if bot.can_handle(sender, text):\n                    self._log.extend(bot.handle(sender, text))\n",
                                       "                bot.can_handle(sender, text)\n                self._log.extend(bot.handle(sender, text))\n")]),
    ("poll stays open", 3, [("            self._poll.clear()  # later votes see no poll in progress\n", "")]),
    ("closes one vote late", 3, [("votes[letter] >= self._close_at:", "votes[letter] > self._close_at:")]),
    ("handlers in reverse", 3, [('        for handler in self._handlers.get(event.type, []):', '        for handler in reversed(self._handlers.get(event.type, [])):')]),
    ("cheer line before the vote", 3, [("                lines += self._bus.publish(", "                lines[:0] = self._bus.publish(")]),
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
