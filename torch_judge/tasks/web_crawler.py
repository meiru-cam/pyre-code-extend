"""A same-host web crawl: breadth-first, then with a thread pool, then with bounded asyncio."""

from ._interview import interview

# A fake parser that counts calls and overlap, random link graphs, and a slow model of the rules.
_HELPERS = r"""
import asyncio, random, sys, threading, time
from collections import Counter
from urllib.parse import urlparse

class Parser:
    def __init__(self, graph, delay=0.0, fail=()):
        self.graph, self.delay, self.fail = graph, delay, set(fail)
        self.calls, self.lock, self.inflight, self.peak = Counter(), threading.Lock(), 0, 0
    def get_urls(self, url):
        with self.lock:
            self.calls[url] += 1
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        try:
            if self.delay:
                time.sleep(self.delay)
            if url in self.fail:
                raise ConnectionError(f"cannot load {url}")
            return list(self.graph.get(url, []))
        finally:
            with self.lock:
                self.inflight -= 1

def model_crawl(start, graph, fail=()):
    norm = lambda u: u.split("#", 1)[0]
    start = norm(start)
    host, seen, queue = urlparse(start).hostname, {start}, [start]
    while queue:
        url = queue.pop(0)
        links = [] if url in fail else graph.get(url, [])
        for link in links:
            link = norm(link)
            if urlparse(link).hostname == host and link not in seen:
                seen.add(link)
                queue.append(link)
    return sorted(seen)

def random_graph(rng, pages=25):
    bases = ["http://a.io", "https://a.io", "http://a.io:8080", "http://b.io", "http://sub.a.io"]
    urls = [f"{rng.choice(bases[:3])}/p{i}" for i in range(pages)]
    def link():
        if rng.random() < 0.15:
            return f"{rng.choice(bases[3:])}/p{rng.randrange(pages)}"
        u = rng.choice(urls)
        return u + rng.choice(["", "", "#top", "#s2"])
    graph = {u: [link() for _ in range(rng.randint(0, 5))] for u in urls}
    fail = {u for u in urls if rng.random() < 0.1}
    return urls[0], graph, fail

def star(n, host="http://wide.io"):
    return f"{host}/", {f"{host}/": [f"{host}/p{i}" for i in range(n)]}

EXAMPLE = {
    "http://wiki.local/start": ["http://wiki.local/a", "http://wiki.local:8080/b"],
    "http://wiki.local/a": ["http://media.wiki.local/x.png", "http://wiki.local/c#refs", "https://wiki.local/a"],
    "http://wiki.local:8080/b": ["http://wiki.local/a"],
    "https://wiki.local/a": [],
}
EXAMPLE_FAIL = {"http://wiki.local/c"}
EXAMPLE_RESULT = ["http://wiki.local/a", "http://wiki.local/c", "http://wiki.local/start", "http://wiki.local:8080/b", "https://wiki.local/a"]
"""

TASK = {
    "title": "Concurrent Web Crawler",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "WebCrawler",
    "description_en": r"""Build `WebCrawler`, which finds every page on one host reachable from a start URL, first one page at a time and then concurrently.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `WebCrawler` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `html_parser.get_urls(url)` returns the absolute URLs linked from the page at `url`. It is slow, safe to call from several threads at once, and may raise for a page that fails to load. A page that raises still counts as crawled; it just has no links.
- Normalise a URL by removing its fragment: `#` and everything after it. Nothing else changes, so `http://x.io/a` and `https://x.io/a` are different pages.
- A link is in scope when `urlparse(link).hostname` equals the start URL's hostname. A subdomain is a different host; scheme and port do not matter for scope.
- Call `get_urls` exactly once for each normalised in-scope URL reached, the start URL included, and never for anything else.
- Return the normalised URLs crawled, as a `list` sorted in ascending string order.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the single-threaded crawl is a breadth-first search; the work is keeping exactly-once fetches and a concurrency bound when many fetches overlap, and each later part adds one requirement.

**Where it is used:** web-scale crawlers gather pretraining text, and documentation crawlers feed retrieval systems; both overlap slow network calls without fetching a page twice.

Adapted from the concurrent web crawler question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "One page at a time",
            "description_en": r"""**Signature:** `WebCrawler()`, `crawl(start_url, html_parser) -> list[str]`

- Follow the rules above. Any visiting order works, since the result is sorted.

**Example**, start URL `http://wiki.local/start`:
- `start` links to `http://wiki.local/a` and `http://wiki.local:8080/b`; `:8080/b` links back to `http://wiki.local/a`
- `http://wiki.local/a` links to `http://media.wiki.local/x.png`, `http://wiki.local/c#refs` and `https://wiki.local/a`
- `https://wiki.local/a` has no links, and `get_urls("http://wiki.local/c")` raises
- the result is `["http://wiki.local/a", "http://wiki.local/c", "http://wiki.local/start", "http://wiki.local:8080/b", "https://wiki.local/a"]`: `/a` is fetched once though two pages link to it, `media.wiki.local` is another host, and the failed page is still listed""",
        },
        {
            "title": "A thread pool",
            "description_en": r"""Keep Part 1 and add a crawl that overlaps calls to `get_urls`.

**Signature:** `crawl_concurrent(start_url, html_parser, max_workers=8) -> list[str]`

- Same result and the same exactly-once rule as `crawl`, for any `max_workers` of `1` or more.
- At no moment are more than `max_workers` calls to `get_urls` running.
- Calls must actually overlap: with many slow pages, the crawl takes about `1 / max_workers` of the one-at-a-time time, not all of it.

**Example**, the same graph with `max_workers=3`: the same five URLs, with exactly five calls to `get_urls`, and never more than three at once.""",
        },
        {
            "title": "Bounded asyncio",
            "description_en": r"""Keep Parts 1–2 and add a coroutine version.

**Signature:** `async crawl_async(start_url, html_parser, concurrency=8) -> list[str]`

- Same result and the same exactly-once rule, for any `concurrency` of `1` or more. It is called as `asyncio.run(crawler.crawl_async(...))`.
- At no moment are more than `concurrency` calls to `get_urls` running, however many pages are waiting.
- `get_urls` is an ordinary blocking function. While it runs, the event loop must stay free to run other tasks.

**Example**, a start page linking to nine pages with `concurrency=3`: all ten URLs come back, and at most three calls are ever running together.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "When should a URL be marked as seen: when it is fetched, or when it is first discovered? What happens to a page linked from two pages in the queue if you wait until it is fetched? Which form of a URL goes into the seen set?"},
        {"level": 2, "kind": "analysis", "content": "Normalise with url.split('#', 1)[0]. Keep host = urlparse(start).hostname, seen = {start} and a deque [start]. Pop a URL, call get_urls inside try/except (an exception means no links), and for each link, normalise it; if its hostname is host and it is not in seen, add it to seen and to the deque. Return sorted(seen)."},
    ],
    "model_connections": [
        "Pretraining corpora such as Common Crawl come from crawlers that dedupe URLs and overlap thousands of slow fetches.",
        "Retrieval and agent tools crawl documentation sites to build an index, with a concurrency limit to stay polite to the host.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Marking a URL seen when it is discovered, not when it is fetched, makes each page fetched exactly once.",
            "Letting only the coordinating thread touch the seen set avoids locks entirely.",
            "A semaphore bounds asyncio concurrency without capping how many tasks exist.",
        ],
        "cons": [
            "Fragment stripping is the only normalisation; trailing slashes and query order still create duplicates.",
            "Threads that only sleep on I/O are cheap, but each still costs memory; asyncio scales further.",
            "Treating a failed page as crawled hides transient errors that a retry would fix.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
p = Parser(EXAMPLE, fail=EXAMPLE_FAIL)
assert {fn}().crawl("http://wiki.local/start", p) == EXAMPLE_RESULT
assert all(n == 1 for n in p.calls.values()) and sorted(p.calls) == EXAMPLE_RESULT, p.calls
"""},
        {"name": "Part 1: random link graphs", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random link graph, the crawl result or the set of fetched URLs differed from the rules: strip only the fragment, compare hostnames, fetch each in-scope URL once, and keep failed pages.",
         "code": _HELPERS + r"""
c = {fn}()
rng = random.Random(1)
for trial in range(200):
    start, graph, fail = random_graph(rng)
    if trial % 5 == 0:
        start += "#frag"
    p = Parser(graph, fail=fail)
    want = model_crawl(start, graph, fail)
    assert c.crawl(start, p) == want, trial
    assert sorted(p.calls) == want and all(n == 1 for n in p.calls.values()), trial
p = Parser({}, fail={"http://solo.io/x"})
assert c.crawl("http://solo.io/x", p) == ["http://solo.io/x"], "a failing start page is still crawled"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "scheduler.concurrency", "code": _HELPERS + r"""
p = Parser(EXAMPLE, fail=EXAMPLE_FAIL)
assert {fn}().crawl_concurrent("http://wiki.local/start", p, max_workers=3) == EXAMPLE_RESULT
assert sum(p.calls.values()) == 5 and p.peak <= 3
"""},
        {"name": "Part 2: exactly once under contention", "part": 2, "visibility": "unshown", "behavior": "scheduler.concurrency",
         "failure_message": "With many threads finding the same links at once, a URL was fetched twice, missed, or more than max_workers calls ran together; decide what is new in one place or under a lock.",
         "code": _HELPERS + r"""
c = {fn}()
old = sys.getswitchinterval()
sys.setswitchinterval(1e-6)
try:
    rng = random.Random(2)
    for trial in range(30):
        start, graph, fail = random_graph(rng, pages=30)
        hub = list(graph)
        for u in hub:
            graph[u] = graph[u] + rng.sample(hub, 6)  # many pages share many links
        workers = rng.choice([1, 2, 3, 8])
        p = Parser(graph, delay=0.001, fail=fail)
        want = model_crawl(start, graph, fail)
        assert c.crawl_concurrent(start, p, max_workers=workers) == want, trial
        assert sorted(p.calls) == want and all(n == 1 for n in p.calls.values()), (trial, p.calls.most_common(2))
        assert p.peak <= workers, f"{p.peak} calls ran together with max_workers={workers}"
finally:
    sys.setswitchinterval(old)
"""},
        {"name": "Part 2: calls overlap", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Forty pages that each take 100 ms were not fetched in parallel; submit every new page to the pool instead of waiting for each one.",
         "code": _HELPERS + r"""
start, graph = star(40)
p = Parser(graph, delay=0.1)
t0 = time.perf_counter()
got = {fn}().crawl_concurrent(start, p, max_workers=8)
elapsed = time.perf_counter() - t0
assert len(got) == 41 and p.peak <= 8
assert elapsed < 2.0, f"took {elapsed:.2f}s; one page at a time takes about 4s"
assert p.peak >= 2, "no two calls ever ran together"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "scheduler.concurrency", "code": _HELPERS + r"""
p = Parser(EXAMPLE, fail=EXAMPLE_FAIL)
assert asyncio.run({fn}().crawl_async("http://wiki.local/start", p, concurrency=2)) == EXAMPLE_RESULT
assert sum(p.calls.values()) == 5 and p.peak <= 2
start, graph = star(9)
p = Parser(graph, delay=0.02)
assert len(asyncio.run({fn}().crawl_async(start, p, concurrency=3))) == 10 and p.peak <= 3
"""},
        {"name": "Part 3: bounded, exact and non-blocking", "part": 3, "visibility": "unshown", "behavior": "scheduler.concurrency",
         "failure_message": "crawl_async fetched a URL twice, ran more than `concurrency` calls together, did not overlap calls, or blocked the event loop while get_urls ran; run get_urls in a thread and bound it with a semaphore.",
         "code": _HELPERS + r"""
c = {fn}()
rng = random.Random(3)
for trial in range(40):
    start, graph, fail = random_graph(rng)
    limit = rng.choice([1, 2, 5])
    p = Parser(graph, delay=0.001, fail=fail)
    want = model_crawl(start, graph, fail)
    assert asyncio.run(c.crawl_async(start, p, concurrency=limit)) == want, trial
    assert sorted(p.calls) == want and all(n == 1 for n in p.calls.values()), trial
    assert p.peak <= limit, f"{p.peak} calls ran together with concurrency={limit}"

async def with_heartbeat(parser, start, limit):
    ticks, done = [], False
    async def beat():
        while not done:
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)
    task = asyncio.create_task(beat())
    t0 = time.perf_counter()
    result = await c.crawl_async(start, parser, concurrency=limit)
    elapsed = time.perf_counter() - t0
    done = True
    await task
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    return result, elapsed, max(gaps) if gaps else elapsed

start, graph = star(12)
p = Parser(graph, delay=0.5)
result, elapsed, worst_gap = asyncio.run(with_heartbeat(p, start, 6))
assert len(result) == 13 and p.peak <= 6
assert elapsed < 3.5, f"took {elapsed:.2f}s; one page at a time takes about 6.5s"
assert worst_gap < 0.3, f"the event loop was blocked for {worst_gap * 1000:.0f} ms while get_urls ran"
"""},
    ],
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import asyncio
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from urllib.parse import urlparse


def _normalise(url):
    return url.split("#", 1)[0]  # only the fragment goes; scheme, host, port and path stay


class WebCrawler:
    def _new_links(self, links, host, seen):
        """The normalised in-scope links not seen before; marks them seen."""
        fresh = []
        for link in links:
            url = _normalise(link)
            if urlparse(url).hostname == host and url not in seen:
                seen.add(url)
                fresh.append(url)
        return fresh

    @staticmethod
    def _fetch(parser, url):
        try:
            return parser.get_urls(url)
        except Exception:  # a failed page still counts as crawled; it just has no links
            return []

    def crawl(self, start_url, html_parser):
        start = _normalise(start_url)
        host, seen, queue = urlparse(start).hostname, {start}, deque([start])
        while queue:
            url = queue.popleft()
            queue.extend(self._new_links(self._fetch(html_parser, url), host, seen))
        return sorted(seen)

    def crawl_concurrent(self, start_url, html_parser, max_workers=8):
        start = _normalise(start_url)
        host, seen = urlparse(start).hostname, {start}
        # only this thread touches `seen`; workers just fetch, so no lock is needed
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pending = {pool.submit(self._fetch, html_parser, start)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    for url in self._new_links(future.result(), host, seen):
                        pending.add(pool.submit(self._fetch, html_parser, url))
        return sorted(seen)

    async def crawl_async(self, start_url, html_parser, concurrency=8):
        start = _normalise(start_url)
        host, seen = urlparse(start).hostname, {start}
        gate = asyncio.Semaphore(concurrency)
        tasks = set()

        async def visit(url):
            async with gate:  # at most `concurrency` blocking calls in flight
                links = await asyncio.to_thread(self._fetch, html_parser, url)
            for new in self._new_links(links, host, seen):
                tasks.add(asyncio.create_task(visit(new)))

        tasks.add(asyncio.create_task(visit(start)))
        while tasks:
            done, _ = await asyncio.wait(tasks)
            tasks.difference_update(done)
        return sorted(seen)
''',
    "interview_questions": interview(
        concept=[
            "Why mark a URL as seen when it is discovered rather than when it is fetched?",
            "Why must the seen set hold normalised URLs, and what does normalisation change here?",
        ],
        deep_dive=[
            "What does a breadth-first crawl cost in calls and memory for P pages and L links?",
        ],
        tradeoffs=[
            "In a thread-pool crawl, which thread should update the seen set, and what race appears if workers do it without a lock?",
            "How do you bound in-flight calls in asyncio without limiting how many tasks you create?",
            "Why must a blocking call run in a thread when you use asyncio, and what breaks if it does not?",
            "What else would you normalise or respect in a production crawler, such as robots.txt or per-host rate limits?",
        ],
    ),
}
