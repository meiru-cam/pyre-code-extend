"""Tests for web_json.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from web_json import WebJsonError, dump_web_json


def test_dump_matches_plain_json_dumps():
    data = {"a": ["x", "中文", "\U0001F600", chr(0x41)], "b": {"c": "safe \\n text"}}
    assert dump_web_json(data) == json.dumps(data, ensure_ascii=False, indent=2)


@pytest.mark.parametrize("text", ['such as "\\ud800"', "\\uDFFF", "\\uDbFf", "\\u{d800}", "\\u{0dc00}"])
def test_rejects_surrogate_escape_text(text):
    with pytest.raises(WebJsonError, match=r"\$\.problems\[0\]\.code: .*chr\(0xD"):
        dump_web_json({"problems": [{"code": text}]})


def test_rejects_surrogate_escape_in_a_key():
    with pytest.raises(WebJsonError, match=r"\$\.a key '.*ud800'"):
        dump_web_json({"a": {"\\ud800": 1}})


def test_rejects_lone_surrogate_character():
    with pytest.raises(WebJsonError, match="U\\+D800"):
        dump_web_json(["x", chr(0xD800)])


@pytest.mark.parametrize("text", ["chr(0xD800)", "\\u00e9", "\\ud7ff", "\\ue000", "\\U0001F600", "\\x00", "\\u{dc000}", "\\u{d800"])
def test_accepts_other_escapes(text):
    assert json.loads(dump_web_json({"s": text})) == {"s": text}
