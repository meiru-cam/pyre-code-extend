"""Serialize data for web/src/lib/*.json, refusing text that breaks the browser bundle."""

from __future__ import annotations

import json
import re
from typing import Any

# The client build inlines these JSON files as JSON.parse('...') strings, and its minifier turns
# the escape text \ud800 (any surrogate, \uD800-\uDFFF, or \u{d800}) into a lone surrogate there,
# so every problem page fails to parse. Write such a string in Python as chr(0xD800) instead.
_SURROGATE_HEX = r"([dD][89a-fA-F][0-9a-fA-F]{2})"
SURROGATE_ESCAPE = re.compile(rf"\\u(?:{_SURROGATE_HEX}|\{{0*{_SURROGATE_HEX}\}})")
LONE_SURROGATE = re.compile("[\ud800-\udfff]")


class WebJsonError(ValueError):
    pass


def _check_strings(value: Any, where: str) -> None:
    if isinstance(value, str):
        if match := SURROGATE_ESCAPE.search(value):
            raise WebJsonError(
                f"{where}: the text {match.group()} breaks the client bundle; write it as chr(0x{(match[1] or match[2]).upper()})"
            )
        if match := LONE_SURROGATE.search(value):
            raise WebJsonError(f"{where}: lone surrogate U+{ord(match.group()):04X} cannot be written as UTF-8")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_strings(key, f"{where} key {key!r}")
            _check_strings(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_strings(item, f"{where}[{index}]")


def dump_web_json(data: Any) -> str:
    _check_strings(data, "$")
    return json.dumps(data, ensure_ascii=False, indent=2)
