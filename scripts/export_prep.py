#!/usr/bin/env python3
"""Export company interview prep items from prep/ for the frontend."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
PREP_DIR = ROOT / "prep"
OUTPUT = ROOT / "web" / "src" / "lib" / "prep.json"

sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS

# Companies with their own tab; every other company is listed under "other".
FEATURED_COMPANIES = ("openai", "anthropic")
ROUNDS = ("ml-coding", "coding", "system-design", "behavioral", "take-home")
RUBRIC_ROUNDS = frozenset({"system-design", "behavioral"})


class PrepError(ValueError):
    pass


def _check(condition: bool, item_id: str, message: str) -> None:
    if not condition:
        raise PrepError(f"{item_id}: {message}")


def _item_entry(company: str, item_dir: Path, known_exercises: set[str]) -> dict[str, Any]:
    item_id = f"{company}-{item_dir.name}"
    meta = json.loads((item_dir / "meta.json").read_text(encoding="utf-8"))
    prompt = (item_dir / "prompt.md").read_text(encoding="utf-8").strip()
    reference_path = item_dir / "reference.md"
    reference = reference_path.read_text(encoding="utf-8").strip() if reference_path.exists() else ""

    round_ = meta["round"]
    rubric = meta.get("rubric", [])
    exercises = meta.get("exercises", [])
    _check(round_ in ROUNDS, item_id, f"unknown round {round_!r}")
    _check(bool(prompt), item_id, "prompt.md is empty")
    _check(meta.get("source", "").startswith("https://"), item_id, "source must be an https URL")
    _check(round_ not in RUBRIC_ROUNDS or bool(rubric), item_id, f"{round_} items need a rubric")
    _check(round_ != "ml-coding" or bool(exercises), item_id, "ml-coding items need exercises")
    unknown = [e for e in exercises if e not in known_exercises]
    _check(not unknown, item_id, f"unknown exercises {unknown}")

    return {
        "id": item_id,
        "company": company,
        "title": meta["title"],
        "summary": meta.get("summary", ""),
        "round": round_,
        "kind": meta.get("kind", ""),
        "difficulty": meta.get("difficulty"),
        "frequency": meta.get("frequency"),
        "format": meta.get("format"),
        "topics": meta.get("topics", []),
        "source": meta["source"],
        "rubric": rubric,
        "exercises": exercises,
        "prompt": prompt,
        "reference": reference,
    }


def _external_entry(link: dict[str, Any]) -> dict[str, Any]:
    item_id = f"{link['company']}-{link['slug']}"
    _check(link["round"] in ROUNDS, item_id, f"unknown round {link['round']!r}")
    _check(link["url"].startswith("https://"), item_id, "url must be an https URL")
    return {
        "id": item_id,
        "company": link["company"],
        "title": link["title"],
        "round": link["round"],
        "url": link["url"],
    }


def build_prep_catalog(
    prep_dir: Path = PREP_DIR, known_exercises: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    known = set(TASKS) if known_exercises is None else known_exercises
    items = [
        _item_entry(company_dir.name, item_dir, known)
        for company_dir in sorted((prep_dir / "items").iterdir()) if company_dir.is_dir()
        for item_dir in sorted(company_dir.iterdir()) if (item_dir / "meta.json").exists()
    ]
    external_path = prep_dir / "external.json"
    external = json.loads(external_path.read_text(encoding="utf-8")) if external_path.exists() else []
    links = [_external_entry(link) for link in external]

    ids = [entry["id"] for entry in items + links]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise PrepError(f"duplicate prep ids {duplicates}")
    return {"items": items, "links": links}


def main() -> None:
    data = build_prep_catalog()
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Written {len(data['items'])} prep items and {len(data['links'])} links "
        f"to {OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
