"""Mutation gate for the multi-part image pipeline exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "image_pipeline"

MUTATIONS = [
    ("grayscale truncates", 1, [("114 * rgb[..., 2] + 500) // 1000", "114 * rgb[..., 2]) // 1000")]),
    ("grayscale overflows uint8", 1, [("rgb = image.astype(np.int64)", "rgb = image")]),
    ("flips swapped", 1, [("return image[:, ::-1].copy()", "return image[::-1].copy()")]),
    ("flip returns a view", 1, [("return image[:, ::-1].copy()", "return image[:, ::-1]")]),
    ("scale picks the top-left pixel", 1, [("rows = (2 * np.arange(nh) + 1) * h // (2 * nh)", "rows = np.arange(nh) * h // nh")]),
    ("scale can reach zero", 1, [("nh, nw = max(1, round(h * step[\"factor\"])), max(1, round(w * step[\"factor\"]))", "nh, nw = round(h * step[\"factor\"]), round(w * step[\"factor\"])")]),
    ("blur floors the mean", 1, [("return ((total + count // 2) // count)", "return ((total) // count)")]),
    ("blur divides by the full window", 1, [("count = np.outer(bottom - top, right - left)", "count = np.full((h, w), (2 * r + 1) ** 2)")]),
    ("rotate turns clockwise", 1, [("np.rot90(image, (step[\"angle\"] // 90) % 4)", "np.rot90(image, -(step[\"angle\"] // 90) % 4)")]),
    ("unknown type ignored", 1, [("raise ValueError(f\"unknown step type {kind!r}\")", "return image.copy()")]),
    ("run loads per pipeline", 1, [("            source = self.load(name)  # once per image, shared by every pipeline\n            for pipeline_name, steps in pipelines.items():\n                self.save(name, pipeline_name, self.run_pipeline(source, steps))",
                                   "            for pipeline_name, steps in pipelines.items():\n                self.save(name, pipeline_name, self.run_pipeline(self.load(name), steps))")]),
    ("parallel runs one at a time", 2, [("with ThreadPoolExecutor(max_workers=max_workers) as pool:", "with ThreadPoolExecutor(max_workers=1) as pool:")]),
    ("parallel ignores the bound", 2, [("with ThreadPoolExecutor(max_workers=max_workers) as pool:", "with ThreadPoolExecutor(max_workers=8) as pool:")]),
    ("parallel swallows errors", 2, [("                future.result()  # re-raise a worker's error here", "                future.exception()")]),
    ("parallel loads per pipeline", 2, [("        source = self.load(name)\n        for pipeline_name, steps in pipelines.items():\n            self.save(name, pipeline_name, self.run_pipeline(source, steps))",
                                         "        for pipeline_name, steps in pipelines.items():\n            self.save(name, pipeline_name, self.run_pipeline(self.load(name), steps))")]),
    ("shared runs every pipeline alone", 3, [("        self._fan_out(image_names, pipelines, max_workers, self._one_image_shared)", "        self._fan_out(image_names, pipelines, max_workers, self._one_image)")]),
    ("step keys depend on order", 3, [("key = tuple(sorted(step.items()))", "key = tuple(step.items())")]),
    ("empty pipelines not saved", 3, [("            self.save(name, pipeline_name, source)\n", "            pass\n")]),
    ("every intermediate kept", 3, [("            result = self.apply_step(image, step)\n", "            result = self.apply_step(image, step)\n            self.__dict__.setdefault('kept', []).append(result)\n")]),
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
