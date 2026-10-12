"""Local CI that runs the slow mutation tests only for the exercises a change touches.

Always runs: tsc + vitest, the core pytest suite and the script tests (about 30 seconds).
Adds the tests/quality files that a change can affect:
- a changed tests/quality/test_*.py file runs itself;
- a changed torch_judge/tasks/<id>.py runs every quality file that names "<id>";
- a change to shared grading code runs all of tests/quality.

`--full` runs everything. The changed files are the working tree against the merge base
with origin/main, plus untracked files.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUALITY = ROOT / "tests" / "quality"

# Changes here can alter how every exercise is graded or mutated.
SHARED = re.compile(
    r"^(torch_judge/(engine\.py|harness/|tasks/_[^/]*\.py|tasks/__init__\.py)"
    r"|tests/quality/(mutation_runner|parts_gate|__init__)\.py"
    r"|tests/conftest\.py|pyproject\.toml)"
)
TASK = re.compile(r"^torch_judge/tasks/([a-z0-9_]+)\.py$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def changed_files() -> list[str]:
    base = git("merge-base", "HEAD", "origin/main").strip()
    tracked = git("diff", "--name-only", base).split()
    untracked = git("ls-files", "--others", "--exclude-standard").split()
    return sorted(set(tracked + untracked))


def quality_targets(changed: list[str]) -> list[str] | None:
    """Quality test files to run; None means all of them."""
    if any(SHARED.match(path) for path in changed):
        return None
    quality_files = sorted(QUALITY.glob("test_*.py"))
    targets: set[Path] = set()
    for path in changed:
        if path.startswith("tests/quality/test_") and (ROOT / path).exists():
            targets.add(ROOT / path)
        match = TASK.match(path)
        if match:
            name = re.compile(rf"[\"']{match.group(1)}[\"']")
            targets.update(f for f in quality_files if name.search(f.read_text()))
    return sorted(str(t.relative_to(ROOT)) for t in targets)


def run(label: str, cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n== {label}", flush=True)
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"ci failed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="run every test, in parallel")
    args = parser.parse_args()

    pytest = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
    run("web: tsc", ["npx", "tsc", "--noEmit", "-p", "."], cwd=ROOT / "web")
    run("web: vitest", ["npx", "vitest", "run"], cwd=ROOT / "web")

    if args.full:
        run("pytest: everything", [*pytest, "-n", "auto"])
        return

    run("pytest: core and scripts", [*pytest, "tests", "scripts", "--ignore=tests/quality"])
    targets = quality_targets(changed_files())
    if targets is None:
        run("pytest: all quality (shared code changed)", [*pytest, "-n", "auto", "tests/quality"])
    elif targets:
        run(f"pytest: {len(targets)} quality file(s)", [*pytest, "-n", "auto", *targets])
    else:
        print("\n== pytest: no quality tests affected")


if __name__ == "__main__":
    main()
