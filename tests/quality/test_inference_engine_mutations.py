"""Mutation gate for the multi-part inference engine exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "inference_engine"

MUTATIONS = [
    ("no room for the new token", 1, [("need = -(-(len(seq.token_ids) + 1) // BLOCK_SIZE)", "need = -(-len(seq.token_ids) // BLOCK_SIZE)")]),
    ("newest request first", 1, [("            self.waiting.popleft()\n            used +=", "            seq = self.sequences[self.waiting.pop()]\n            used +=")]),
    ("ids start at one", 1, [("seq = Sequence(len(self.sequences), prompt, max_tokens)", "seq = Sequence(len(self.sequences) + 1, prompt, max_tokens)")]),
    ("decode newest first", 2, [("for seq in list(self.running):", "for seq in reversed(self.running):")]),
    ("decode tokens not budgeted", 2, [("        used = decoded\n", "        used = 0\n")]),
    ("one sequence too many", 2, [("len(self.running) >= self.max_seqs:", "len(self.running) > self.max_seqs:")]),
    ("short requests jump the queue", 2, [("            if self.token_budget is not None and used + len(seq.token_ids) > self.token_budget:\n                break  # nothing is admitted out of order\n            self.waiting.popleft()\n",
                                          "            if self.token_budget is not None and used + len(seq.token_ids) > self.token_budget:\n                fits = [r for r in self.waiting if used + len(self.sequences[r].token_ids) <= self.token_budget]\n                if not fits:\n                    break\n                seq = self.sequences[fits[0]]\n            self.waiting.remove(seq.request_id)\n")]),
    ("decode always allocates", 2, [("        if len(seq.token_ids) + 1 > len(seq.block_ids) * BLOCK_SIZE:\n            fresh", "        if True:\n            fresh")]),
    ("finished keep their blocks", 3, [("        if seq.is_stopped():\n            self._release(seq, Status.FINISHED)", "        if seq.is_stopped():\n            seq.status = Status.FINISHED\n            self.running.remove(seq)")]),
    ("finished keep running", 3, [("        if seq.is_stopped():\n            self._release(seq, Status.FINISHED)", "        if seq.is_stopped() and len(seq.generated) > seq.max_tokens:\n            self._release(seq, Status.FINISHED)")]),
    ("no reuse", 4, [("            block_id = self.allocator.lookup_cached(content_hash)\n", "            block_id = None\n")]),
    ("blocks filled by decode never registered", 4, [("        if len(seq.token_ids) % BLOCK_SIZE == 0:  # the last block just became full", "        if False:")]),
    ("prompt blocks never registered", 4, [("        for i in range(len(hits), full):\n            self.allocator.register", "        for i in range(0):\n            self.allocator.register")]),
    ("reuse past a miss", 4, [("                break  # reuse only an unbroken run of leading blocks", "                continue")]),
    ("blocks freed in reverse order", 4, [("        self.allocator.free(seq.block_ids)", "        self.allocator.free(seq.block_ids[::-1])")]),
    ("own victim still generates", 5, [("            if fresh is None:\n                return False", "            if fresh is None:\n                seq.token_ids.append(next_token(seq.token_ids))\n                return False")]),
    ("re-admitted keeps its old slot", 5, [("        self.running.append(seq)\n", "        self.running.insert(sum(s.request_id < seq.request_id for s in self.running), seq)\n")]),
    ("preempted sequences still decode", 5, [("            if seq.status is Status.RUNNING and self._decode(seq):", "            if self._decode(seq):")]),
    ("oldest preempted", 5, [("            victim = self.running[-1]", "            victim = self.running[0]")]),
    ("preempted to the back", 5, [("        self.waiting.appendleft(seq.request_id)", "        self.waiting.append(seq.request_id)")]),
    ("admission goes on after a preemption", 5, [("        while self.waiting and not self.preempted:", "        while self.waiting:")]),
    ("readmission billed for the prompt only", 5, [("            if self.token_budget is not None and used + len(seq.token_ids) > self.token_budget:", "            if self.token_budget is not None and used + len(seq.prompt) > self.token_budget:"),
                                                     ("            used += len(seq.token_ids)", "            used += len(seq.prompt)")]),
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
