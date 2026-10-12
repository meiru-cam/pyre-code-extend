"""Provenance hygiene for the oracles in `torch_judge/harness/embodied_rl/vendored/`."""

from __future__ import annotations

from pathlib import Path

import pytest

VENDORED_DIR = Path(__file__).resolve().parents[2] / "torch_judge" / "harness" / "embodied_rl" / "vendored"
MODULES = sorted(path.name for path in VENDORED_DIR.glob("*.py") if path.name != "__init__.py")


def test_every_module_is_listed_with_its_license():
    package_doc = (VENDORED_DIR / "__init__.py").read_text()
    for module in MODULES:
        assert module in package_doc, f"{module} is missing from the package license list"


def test_vendored_modules_never_import_a_task_solution():
    """Match import statements, not prose — the package docstring names the rule."""
    for module in MODULES:
        for line in (VENDORED_DIR / module).read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "torch_judge.tasks" not in stripped, f"{module} imports task solutions: {stripped}"


@pytest.mark.parametrize("module", MODULES)
def test_vendored_modules_carry_provenance_and_license(module):
    source = (VENDORED_DIR / module).read_text()
    for marker in ("Upstream:", "Commit:", "License:", "WHAT WAS CHANGED"):
        assert marker in source, f"{module} is missing {marker!r}"
    assert "https://" in source or "arXiv" in source, f"{module} has no upstream URL"
