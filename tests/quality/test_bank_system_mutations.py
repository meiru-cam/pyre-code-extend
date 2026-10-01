"""Mutation gate for the multi-part bank ledger exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "bank_system"

MUTATIONS = [
    ("duplicate account allowed", 1, [("        if account_id in self._live:\n            return False\n", "")]),
    ("cannot withdraw the whole balance", 1, [(
        "if account is None or amount > self._available(account_id, timestamp):",
        "if account is None or amount >= self._available(account_id, timestamp):",
    )]),
    ("deposit to a missing account", 1, [(
        "        account = self._live.get(account_id)\n        if account is None:\n            return None\n        account.change(timestamp, amount)",
        "        account = self._live.setdefault(account_id, _Account(timestamp))\n        account.change(timestamp, amount)",
    )]),
    ("ties sorted descending", 2, [("key=lambda item: (-item[1].outgoing, item[0])", "key=lambda item: (-item[1].outgoing, [-ord(c) for c in item[0]])")]),
    ("outgoing never counted", 2, [("        account.outgoing += amount\n", "")]),
    ("top n off by one", 2, [("for account_id, account in ranked[:n]]", "for account_id, account in ranked[:max(n - 1, 1)]]")]),
    ("holds ignored", 3, [(
        "        return self._live[account_id].balance - held\n",
        "        return self._live[account_id].balance\n",
    )]),
    ("window excludes its last tick", 3, [("return not self.accepted and timestamp <= self.created + WINDOW", "return not self.accepted and timestamp < self.created + WINDOW")]),
    ("counter advances on failure", 3, [
        ("        if amount > self._available(source_id, timestamp):\n            return None\n        transfer_id",
         "        if amount > self._available(source_id, timestamp):\n            self._transfers[f\"failed{len(self._transfers)}\"] = _Transfer(None, None, 0, -10 ** 9)\n            return None\n        transfer_id"),
    ]),
    ("anyone may accept", 3, [("if t is None or t.target != account_id or not t.pending(timestamp):", "if t is None or not t.pending(timestamp):")]),
    ("accept twice", 3, [("return not self.accepted and timestamp <= self.created + WINDOW", "return timestamp <= self.created + WINDOW")]),
    ("merge drops outgoing", 4, [("        keep.outgoing += gone.outgoing\n", "")]),
    ("merge leaves incoming transfers on b", 4, [("                if t.target == b:\n                    t.target = a\n", "")]),
    ("merge leaves outgoing transfers on b", 4, [("                if t.source == b:\n                    t.source = a\n", "")]),
    ("merge credit missing from history", 4, [(
        "        keep.change(timestamp, gone.balance)\n",
        "        keep.balance += gone.balance\n",
    )]),
    ("history ignores the creation time", 4, [("        return account.balances[index - 1] if index else None", "        return account.balances[max(index - 1, 0)]")]),
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
