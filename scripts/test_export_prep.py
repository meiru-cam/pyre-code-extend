"""Tests for export_prep.py and the prep importer's markdown rewriting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from export_prep import OUTPUT, PrepError, build_prep_catalog
from import_prep_notes import rubric_for, to_site_markdown


def _write_item(root: Path, company: str, slug: str, meta: dict, prompt: str = "Design it.") -> None:
    item = root / "items" / company / slug
    item.mkdir(parents=True)
    (item / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (item / "prompt.md").write_text(prompt, encoding="utf-8")


def _design_meta(**overrides) -> dict:
    meta = {
        "title": "URL shortener",
        "round": "system-design",
        "source": "https://example.com/url-shortener",
        "rubric": ["Requirements and scale"],
    }
    meta.update(overrides)
    return meta


def test_committed_prep_json_matches_sources():
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == build_prep_catalog()


def test_every_committed_item_keeps_its_attribution_link():
    for item in build_prep_catalog()["items"]:
        assert item["source"].startswith("https://github.com/Schuture/"), item["id"]


def test_item_id_joins_company_and_slug(tmp_path):
    _write_item(tmp_path, "openai", "url-shortener", _design_meta())
    [item] = build_prep_catalog(tmp_path, known_exercises=set())["items"]
    assert item["id"] == "openai-url-shortener"
    assert item["reference"] == ""


def test_design_round_without_rubric_is_rejected(tmp_path):
    _write_item(tmp_path, "openai", "url-shortener", _design_meta(rubric=[]))
    with pytest.raises(PrepError, match="need a rubric"):
        build_prep_catalog(tmp_path, known_exercises=set())


def test_unknown_exercise_is_rejected(tmp_path):
    meta = _design_meta(round="ml-coding", rubric=[], exercises=["no_such_task"])
    _write_item(tmp_path, "openai", "loss", meta)
    with pytest.raises(PrepError, match="unknown exercises"):
        build_prep_catalog(tmp_path, known_exercises={"cross_entropy"})


def test_duplicate_ids_across_items_and_links_are_rejected(tmp_path):
    _write_item(tmp_path, "openai", "url-shortener", _design_meta())
    link = {"company": "openai", "slug": "url-shortener", "title": "x", "round": "coding",
            "url": "https://example.com"}
    (tmp_path / "external.json").write_text(json.dumps([link]), encoding="utf-8")
    with pytest.raises(PrepError, match="duplicate"):
        build_prep_catalog(tmp_path, known_exercises=set())


def test_site_markdown_joins_wrapped_list_items_but_not_code():
    source = "- first line\n  continues here\n- second\n\n```py\na = 1\nb = 2\n```\n"
    assert to_site_markdown(source) == "- first line continues here\n- second\n\n```py\na = 1\nb = 2\n```\n"


def test_site_markdown_turns_tables_into_lists_and_drops_details():
    source = "<details>\n<summary>Show</summary>\n\n| Story | Failure |\n| --- | --- |\n| Mistake | Blames others |\n</details>\n"
    assert to_site_markdown(source) == "- **Mistake**: Blames others\n"


def test_behavioral_rubric_prefers_prompt_themes():
    problem = "Intro\n\n### Motivation\n\n### Values stories\n"
    reference = "### Motivation\n\n### Prep outline\n"
    assert rubric_for("behavioral", problem, reference) == ["Motivation", "Values stories"]
    assert rubric_for("system-design", "", "### Architecture\n### Follow-ups\n") == ["Architecture"]
