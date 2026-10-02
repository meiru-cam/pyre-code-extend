"""Mutation gate for the multi-part concurrent web crawler exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "web_crawler"

MUTATIONS = [
    ("fragment kept", 1, [('    return url.split("#", 1)[0]  #', '    return url  #')]),
    ("port compared", 1, [("if urlparse(url).hostname == host and url not in seen:", "if urlparse(url).netloc == host and url not in seen:")]),
    ("subdomains in scope", 1, [("if urlparse(url).hostname == host and url not in seen:", "if (urlparse(url).hostname or '').endswith(host) and url not in seen:")]),
    ("failures propagate", 1, [("        try:\n            return parser.get_urls(url)\n        except Exception:  # a failed page still counts as crawled; it just has no links\n            return []\n",
                                "        return parser.get_urls(url)\n")]),
    ("start not normalised", 1, [("        start = _normalise(start_url)\n        host, seen, queue", "        start = start_url\n        host, seen, queue")]),
    ("pool crawl is serial", 2, [("        start = _normalise(start_url)\n        host, seen = urlparse(start).hostname, {start}\n        # only this thread",
                                  "        return self.crawl(start_url, html_parser)\n        start = _normalise(start_url)\n        host, seen = urlparse(start).hostname, {start}\n        # only this thread")]),
    ("pool not bounded", 2, [("ThreadPoolExecutor(max_workers=max_workers)", "ThreadPoolExecutor(max_workers=64)")]),
    ("no semaphore", 3, [("            async with gate:  # at most `concurrency` blocking calls in flight\n                links = await asyncio.to_thread(self._fetch, html_parser, url)\n",
                          "            links = await asyncio.to_thread(self._fetch, html_parser, url)\n")]),
    ("blocking call on the loop", 3, [("links = await asyncio.to_thread(self._fetch, html_parser, url)", "links = self._fetch(html_parser, url)")]),
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
