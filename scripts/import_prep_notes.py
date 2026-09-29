#!/usr/bin/env python3
"""Adapt the questions listed in prep/sources.json from Schuture interview-notes checkouts.

Usage:
    python scripts/import_prep_notes.py --openai <checkout> --anthropic <checkout>

Writes prep/items/<company>/<slug>/{meta.json,prompt.md,reference.md} for every source. The notes' prose is
CC BY-NC 4.0, so every item keeps its source URL; see prep/NOTICE.md.

The site's markdown renderer has no tables, links or HTML, so tables become dash lists,
links become "text (url)" and <details> wrappers are dropped. It does render italics and
TeX in its extended mode, which the prep pages use, so those pass through.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from export_prep import ROUNDS, RUBRIC_ROUNDS

ROOT = Path(__file__).parent.parent
ITEMS_DIR = ROOT / "prep" / "items"
SOURCES = ROOT / "prep" / "sources.json"

SOURCE_REPOS = {
    "openai": "https://github.com/Schuture/OpenAI-Interview-Notes",
    "anthropic": "https://github.com/Schuture/Anthropic-Interview-Notes",
}

# Notes categories are prep rounds as-is, except that "coding" becomes "ml-coding" when the
# item links to exercises.
CATEGORIES = frozenset(ROUNDS) - {"ml-coding"}

# Design answers are judged on these even where the reference has no section for them.
_DESIGN_RUBRIC_EXTRAS = ("Failure modes and recovery", "Trade-offs and alternatives rejected")

# Reference-solution sections that coach delivery rather than name a thing to cover.
_NOT_RUBRIC = {"follow-ups", "prep outline", "slide outline", "choosing the project"}


def _read_meta(path: Path) -> dict[str, object]:
    """Read the flat subset of YAML these meta files use, without a YAML dependency."""
    meta: dict[str, object] = {}
    key = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if match:
            key, value = match.group(1), match.group(2).strip()
            if value in (">-", ">", "|"):
                meta[key] = ""
                continue
            if value.startswith("[") and value.endswith("]"):
                meta[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1].replace("''", "'")
            meta[key] = value
        elif key and line.startswith("  "):
            meta[key] = f"{meta[key]} {line.strip()}".strip()
    return meta


def _with_fences(lines: list[str]):
    """Yield (line, in_code); fence lines themselves count as code."""
    in_code = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
            yield line, True
        else:
            yield line, in_code


def _sections(text: str) -> dict[str, str]:
    """Split a README on its level-2 headings."""
    parts: dict[str, str] = {}
    current = None
    buffer: list[str] = []
    for line, in_code in _with_fences(text.splitlines()):
        if not in_code and line.startswith("## "):
            if current:
                parts[current] = "\n".join(buffer).strip()
            current, buffer = line[3:].strip().lower(), []
            continue
        buffer.append(line)
    if current:
        parts[current] = "\n".join(buffer).strip()
    return parts


def _table_to_list(rows: list[str]) -> list[str]:
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], [r for r in cells[1:] if not all(set(c) <= set("-: ") for c in r)]
    out = []
    for row in body:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
        if len(header) == 2 and len(row) >= 2:
            out.append(f"- **{row[0]}**: {row[1]}")
        else:
            out.append("- " + " · ".join(pairs))
    return out


def to_site_markdown(text: str) -> str:
    """Rewrite GitHub markdown into what MarkdownContent.tsx renders."""
    out: list[str] = []
    table: list[str] = []
    for line, in_code in _with_fences(text.splitlines()):
        if not in_code and line.lstrip().startswith("|"):
            table.append(line)
            continue
        if table:
            out.extend(_table_to_list(table))
            table = []
        if not in_code:
            # The site shows the reference behind its own reveal, so drop the notes' <details>.
            line = re.sub(r"</?details>|<summary>.*?</summary>", "", line)
            if not line.strip() and out and not out[-1].strip():
                continue
            line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
            line = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", line)
            line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
            line = re.sub(r"^(\s*)\d+\.\s+", r"\1- ", line)
            line = re.sub(r"^#### ", "### ", line)
        out.append(line)
    if table:
        out.extend(_table_to_list(table))
    return "\n".join(_unwrap(out)).strip() + "\n"


def _is_block_start(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith(("#", "- ", "* ", "```", "$$"))
    )


def _unwrap(lines: list[str]) -> list[str]:
    """Join hard-wrapped lines: the renderer starts a new list item or paragraph per line."""
    out: list[str] = []
    in_code = in_math = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if not in_code and stripped == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_code or in_math:
            out.append(line)
            continue
        prev = out[-1].strip() if out else ""
        joinable = prev and not prev.startswith(("#", "```", "$$")) and prev != "$$"
        if joinable and not _is_block_start(line):
            out[-1] = f"{out[-1].rstrip()} {stripped}"
        else:
            out.append(line)
    return out


def _headings(section: str) -> list[str]:
    heads = []
    for line, in_code in _with_fences(section.splitlines()):
        if not in_code and line.startswith("### "):
            heads.append(line[4:].strip())
    return heads


def rubric_for(round_: str, problem: str, reference: str) -> list[str]:
    if round_ not in RUBRIC_ROUNDS:
        return []
    # Behavioral prompts list their themes under the problem; design prompts do not.
    heads = _headings(problem) if round_ == "behavioral" else []
    heads = [h for h in heads or _headings(reference) if h.lower() not in _NOT_RUBRIC]
    if round_ == "system-design":
        heads += [extra for extra in _DESIGN_RUBRIC_EXTRAS if extra not in heads]
    return heads


def import_item(checkout: Path, company: str, rel: str, exercises: list[str]) -> Path:
    source_dir = checkout / rel
    category, slug = rel.split("/")
    meta = _read_meta(source_dir / "meta.yaml")
    parts = _sections((source_dir / "README.md").read_text(encoding="utf-8"))
    problem = parts.get("problem", "")
    reference = parts.get("reference solution", "")
    if category not in CATEGORIES:
        raise ValueError(f"{rel}: unknown notes category {category!r}")
    round_ = category
    if round_ == "coding" and exercises:
        round_ = "ml-coding"

    item = {
        "title": meta["title"],
        "summary": meta.get("summary", ""),
        "round": round_,
        "kind": meta.get("kind", ""),
        "difficulty": str(meta.get("difficulty", "")).capitalize() or None,
        "frequency": meta.get("frequency") or None,
        "format": meta.get("format") or None,
        "topics": meta.get("topics", []),
        "source": f"{SOURCE_REPOS[company]}/tree/main/{rel}",
        "rubric": rubric_for(round_, problem, reference),
        "exercises": exercises,
    }
    target = ITEMS_DIR / company / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / "meta.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "prompt.md").write_text(to_site_markdown(problem), encoding="utf-8")
    (target / "reference.md").write_text(to_site_markdown(reference), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for company in sorted(SOURCE_REPOS):
        parser.add_argument(f"--{company}", type=Path, required=True, help=f"{company} notes checkout")
    args = parser.parse_args(argv)
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    for source in sources:
        checkout = getattr(args, source["company"])
        target = import_item(checkout, source["company"], source["item"], source.get("exercises", []))
        print(f"Wrote {target.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
