"""Mutation gate for the multi-part file deduplication exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import (
    assert_first_fails_in_part,
    assert_part_mutations_rejected,
    first_failing_part,
)

TASK_ID = "file_dedup"

# The usual first answer: hash every regular file in full, one at a time.
_HASH_EVERYTHING = '''import hashlib
import os
from collections import defaultdict


def find_duplicate_files(root_path):
    by_digest = defaultdict(list)
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as f:
                    by_digest[hashlib.sha256(f.read()).digest()].append(path)
            except OSError:
                continue
    return sorted((sorted(g) for g in by_digest.values() if len(g) >= 2), key=lambda g: g[0])
'''

MUTATIONS = [
    ("walks linked directories", 1, [("os.walk(root_path):", "os.walk(root_path, followlinks=True):")]),
    ("reports file links", 1, [("if not os.path.islink(path) and os.path.isfile(path):", "if os.path.isfile(path):")]),
    ("singletons reported", 1, [("out.extend(g for g in by_key.values() if len(g) >= 2)", "out.extend(by_key.values())")]),
    ("groups in reverse order", 1, [("key=lambda group: group[0])", "key=lambda group: group[0], reverse=True)")]),
    ("empty files dropped", 1, [("if keys[path] is not None:", "if keys[path]:")]),
    ("unreadable file raises", 1, [("            return f.read(SAMPLE_SIZE)\n    except OSError:\n        return None",
                                     "            return f.read(SAMPLE_SIZE)\n    except FileNotFoundError:\n        return None")]),
    ("sample trusted as proof", 1, [("        groups = _regroup(groups, _digest, pool)\n", "")]),
    ("no size filter", 2, [("groups = [paths for paths in by_size.values() if len(paths) >= 2]",
                            "groups = [[path for paths in by_size.values() for path in paths]]")]),
    ("no sample stage", 2, [("        groups = _regroup(groups, _sample, pool)\n", "")]),
    ("sample too large", 2, [("SAMPLE_SIZE = 4096", "SAMPLE_SIZE = 65536")]),
    ("reads whole files at once", 2, [("            while chunk := f.read(CHUNK_SIZE):\n                hasher.update(chunk)",
                                       "            hasher.update(f.read())")]),
    ("reads one file at a time", 3, [("keys = dict(zip(paths, pool.map(key_of, paths)))", "keys = dict(zip(paths, map(key_of, paths)))")]),
    ("max_workers ignored", 3, [("ThreadPoolExecutor(max_workers=max_workers)", "ThreadPoolExecutor(max_workers=8)")]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_hashing_everything_passes_part_1_only():
    assert first_failing_part(get_task(TASK_ID), _HASH_EVERYTHING) == 2


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
