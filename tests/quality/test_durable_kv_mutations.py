"""Mutation gate for the multi-part durable key-value store exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import (
    assert_first_fails_in_part,
    assert_part_mutations_rejected,
    first_failing_part,
)

TASK_ID = "durable_kv"

# The natural part 1 answer: one file, rewritten in place by every save.
_ONE_FILE = '''class KVStore:
    def __init__(self, fs):
        self.fs = fs
        self.data = {}

    def put(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def save(self):
        out = bytearray()
        for text in (s for pair in self.data.items() for s in pair):
            raw = text.encode("utf-8", "surrogatepass")
            out += len(raw).to_bytes(8, "big") + raw
        self.fs.write("store", bytes(out))

    def load(self):
        blob, self.data, pos, items = self.fs.read("store") or b"", {}, 0, []
        while pos < len(blob):
            end = pos + 8 + int.from_bytes(blob[pos:pos + 8], "big")
            items.append(blob[pos + 8:end].decode("utf-8", "surrogatepass"))
            pos = end
        self.data = dict(zip(items[::2], items[1::2]))
'''

_DELIMITED = '''def _serialize(data):
    return "".join(f"{k}:{v}\\n" for k, v in data.items()).encode("utf-8", "surrogatepass")


def _deserialize(blob):
    lines = blob.decode("utf-8", "surrogatepass").split("\\n")[:-1]
    return dict(line.split(":", 1) for line in lines)


def _unused_serialize(data):
'''

_SAVE_HEAD = '''        names = set()
        for i in range(0, len(blob), cap):  # a cut may fall inside a length or a character
'''
_COMMIT = '''        # The commit point: one atomic write switches load() to the new generation.
        self.fs.write(self.MANIFEST, generation.to_bytes(8, "big") + len(names).to_bytes(8, "big"))
'''
_CLEANUP = '''        for name in self.fs.list():  # also chunks of interrupted saves, not only the previous one
            if name.startswith("chunk_") and name not in names:
                self.fs.delete(name)
'''
_REPLAY_LOOP = '''        while (record := _read_record(data, pos)) is not None:  # stop at the first bad record
            op, key, value, pos = record
'''
_TRUNCATE = '''        if pos < len(data):  # cut the bad tail off, or later appends land behind it
            self.fs.write(self.LOG, data[:pos])
'''
_COMPACT = '''        self.fs.write(self.LOG, b"".join(_encode_record(PUT, k, v) for k, v in self._data.items()))
'''

MUTATIONS = [
    ("strict UTF-8", 1, [('raw = s.encode("utf-8", "surrogatepass")', 'raw = s.encode("utf-8")')]),
    ("length in characters", 1, [('return len(raw).to_bytes(8, "big") + raw', 'return len(s).to_bytes(8, "big") + raw')]),
    ("delimiter format", 1, [("def _serialize(data):\n", _DELIMITED)]),
    ("json format", 1, [("def _serialize(data):\n    return b", "def _serialize(data):\n    import json\n    json.dumps(data)\n    return b")]),
    ("nothing saved raises", 1, [("        if manifest is None:\n            self._data = {}\n            return\n", "")]),
    ("load keeps unsaved puts", 1, [("        if manifest is None:\n            self._data = {}\n            return\n",
                                     "        if manifest is None:\n            return\n")]),
    ("one file ignores the cap", 2, [("cap = self.fs.max_file_size or max(len(blob), 1)", "cap = max(len(blob), 1)")]),
    ("chunks overwritten in place", 2, [('generation = int.from_bytes(old[:8], "big") + 1 if old is not None else 0', "generation = 0")]),
    ("commit before the chunks", 2, [(_COMMIT, ""), (_SAVE_HEAD, '        self.fs.write(self.MANIFEST, generation.to_bytes(8, "big") + (-(-len(blob) // cap)).to_bytes(8, "big"))\n' + _SAVE_HEAD)]),
    ("old chunks deleted before the commit", 2, [(_COMMIT + _CLEANUP, _CLEANUP.replace("name not in names", "not name.startswith(f\"chunk_{generation}_\")") + _COMMIT)]),
    ("only the previous generation cleaned", 2, [('if name.startswith("chunk_") and name not in names:', 'if name.startswith(f"chunk_{generation - 1}_"):')]),
    ("no cleanup", 2, [(_CLEANUP, "")]),
    ("delete not logged", 3, [("            self.fs.append(self.LOG, _encode_record(DELETE, key))\n", "            pass\n")]),
    ("tombstone is an empty put", 3, [("_encode_record(DELETE, key))", "_encode_record(PUT, key, \"\"))")]),
    ("put rewrites the log", 3, [("            self.fs.append(self.LOG, _encode_record(PUT, key, value))",
                                  "            self.fs.write(self.LOG, (self.fs.read(self.LOG) or b\"\") + _encode_record(PUT, key, value))")]),
    ("bad tail kept", 3, [(_TRUNCATE, "")]),
    ("checksum skips the length", 3, [("zlib.crc32(header + payload)", "zlib.crc32(payload)"),
                                      ("zlib.crc32(data[pos:pos + 8] + payload)", "zlib.crc32(payload)")]),
    ("no checksum check", 3, [("    if zlib.crc32(data[pos:pos + 8] + payload) != int.from_bytes(data[pos + 8:pos + 12], \"big\"):\n        return None\n", "")]),
    ("load re-slices per record", 3, [(_REPLAY_LOOP, "        while (record := _read_record(data[pos:], 0)) is not None:\n            op, key, value, used = record\n            pos += used\n")]),
    ("compact appends", 3, [(_COMPACT, _COMPACT.replace("self.fs.write(", "self.fs.append("))]),
    ("compact deletes first", 3, [(_COMPACT, "        self.fs.delete(self.LOG)\n" + _COMPACT)]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_one_rewritten_file_passes_part_1_only():
    assert first_failing_part(get_task(TASK_ID), _ONE_FILE) == 2


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
