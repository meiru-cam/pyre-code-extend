"""Mutation gate for the multi-part config store exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "config_store"

MUTATIONS = [
    ("DELETE kept", 1, [("            if value is self.DELETE:\n                continue\n", "            if False:\n                continue\n")]),
    ("dict overlay replaces", 1, [("                result[key] = self.deep_merge(inner if isinstance(inner, dict) else {}, value)\n",
                                   "                result[key] = copy.deepcopy(value)\n")]),
    ("base values shared", 1, [("result = {key: copy.deepcopy(value) for key, value in base.items() if key not in overlay}",
                                "result = {key: value for key, value in base.items() if key not in overlay}")]),
    ("overlay values shared", 1, [("            else:\n                result[key] = copy.deepcopy(value)\n", "            else:\n                result[key] = value\n")]),
    ("bases right to left", 1, [('for base in own.get("_base_", []):', 'for base in reversed(own.get("_base_", [])):')]),
    ("own keys merged first", 1, [('return self.deep_merge(result, {k: v for k, v in own.items() if k != "_base_"})',
                                   'return self.deep_merge({k: v for k, v in own.items() if k != "_base_"}, result)')]),
    ("cycle message from the root", 1, [("loop = list(_path[_path.index(name):]) + [name]", "loop = list(_path) + [name]")]),
    ("missing last key created", 2, [("            if keys[-1] not in node and not plus:\n", "            if False:\n")]),
    ("scalar on the path replaced", 2, [("                if not isinstance(node[key], dict):\n", "                if not isinstance(node[key], dict) and not plus:\n")]),
    ("bare words accepted", 2, [('    raise OverrideSyntaxError(f"bad value {text!r}")\n', "    return text\n")]),
    ("exponent not a float", 2, [('return float(text) if any(c in text for c in ".eE") else int(text)', 'return float(text) if "." in text else int(text)')]),
    ("None embedded as text", 2, [('            raise InterpolationError(f"{path} is {_type_name(value)}, which cannot sit inside a string")\n', "            return str(value)\n")]),
    ("bools embedded as Python", 2, [('                return "true" if value else "false"\n', "                return str(value)\n")]),
    ("whole reference stringified", 2, [("                return copy.deepcopy(value_at(whole.group(1), stack))\n",
                                         "                return text(value_at(whole.group(1), stack), whole.group(1))\n")]),
    ("interpolation loop from the root", 2, [("loop = list(stack[stack.index(path):]) + [path]", "loop = list(stack) + [path]")]),
    ("lists not interpolated", 2, [("                return [walk(v, here, stack) for v in node]\n", "                return list(node)\n")]),
    ("bool accepted as int", 3, [("ok = (isinstance(value, kind) and not (isinstance(value, bool) and kind is not bool)) or (",
                                  "ok = isinstance(value, kind) or (")]),
    ("int rejected for float", 3, [(") or (\n                    kind is float and isinstance(value, int) and not isinstance(value, bool))", ")")]),
    ("defaults required", 3, [("                elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:\n", "                else:\n")]),
    ("extra keys ignored", 3, [('                    errors.append((prefix + name, "not in schema"))\n', "                    pass\n")]),
    ("list items unchecked", 3, [('                        check(item, typing.get_args(kind)[0], f"{path}.{i}")\n', "                        pass\n")]),
    ("first error only", 3, [("        record(config, schema, \"\")\n        return errors\n", "        record(config, schema, \"\")\n        return errors[:1]\n")]),
    ("non-dict config accepted", 3, [('            raise TypeError("config must be a dict")\n', "            return []\n")]),
    ("lists left mutable", 4, [("            return tuple(self.freeze(v) for v in config)\n", "            return [self.freeze(v) for v in config]\n")]),
    ("shallow freeze", 4, [("return types.MappingProxyType({k: self.freeze(v) for k, v in config.items()})", "return types.MappingProxyType(dict(config))")]),
    ("2.0 hashes like 2", 4, [('        return f"f{value.hex()}"\n', '        return f"i{int(value)}" if value == int(value) else f"f{value.hex()}"\n')]),
    ("-0.0 hashes like 0.0", 4, [('        return f"f{value.hex()}"\n', '        return f"f{(value + 0.0).hex()}"\n')]),
    ("True hashes like 1", 4, [('        return "b1" if value else "b0"\n', '        return "i1" if value else "i0"\n')]),
    ("key order matters", 4, [("for k in sorted(value)", "for k in value")]),
    ("NaN hashed", 4, [('            raise ValueError("NaN and infinity cannot be hashed")\n', "            pass\n")]),
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
