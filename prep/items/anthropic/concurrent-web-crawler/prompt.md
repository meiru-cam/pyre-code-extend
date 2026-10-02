A *crawl* starting at `start_url` visits every page reachable from it whose hostname matches `start_url`'s, using a supplied helper:

```py
class HtmlParser:
    def get_urls(self, url: str) -> list[str]:
        """Returns the absolute URLs of every link found on the page at `url`."""
        ...
```

`get_urls` is I/O-bound and may take a long time to return; it is safe to call from several threads at once. It can also raise an exception, for a page that fails to load: a page whose call raises is still counted as crawled (something else linked to it, and the crawler did attempt to fetch it, so it belongs in the result), but it contributes no further links, and the crawl continues with whatever else has already been discovered. `start_url`, and every URL `get_urls` returns, is absolute, with an explicit scheme and hostname; none of them is a relative path, so no relative-URL resolution is ever needed.

Two URLs normalise to the same string, and are therefore the same page, once their *fragment* — the `#` and everything after it — is removed; normalisation changes nothing else about a URL. `http://a.com/x#top` and `http://a.com/x#bottom` both normalise to `http://a.com/x` and count as one page, fetched once. A link is *in scope* for the crawl exactly when `urlparse(link).hostname` equals `urlparse(start_url).hostname`, compared as plain strings (`hostname` is already lower-cased by `urlparse`): a subdomain is a different host (`shop.a.com` is not `a.com`), and so is an unrelated domain, but a link's scheme and port play no part in this comparison — an `https` link, or one naming an explicit port, is still in scope as long as its hostname matches. Scheme and port are not stripped by normalisation, though, so they still affect a URL's identity: `http://a.com/x` and `https://a.com/x` are two different normalised URLs, both in scope, fetched independently. The link graph may contain cycles, the same link repeated on one page or across several, and self-links; each normalised, in-scope URL is fetched — `get_urls` called on it — at most once, however many pages link to it or however many times.

A crawl function returns every normalised URL it fetched, as a list sorted lexicographically (ascending, ordinary string order), including `start_url` itself.

### Part 1 — Single-threaded crawl

```py
def crawl(start_url: str, html_parser: HtmlParser) -> list[str]: ...
```

`crawl` may visit pages in any order that only ever fetches a page once a link to it has been discovered (breadth-first is the natural choice, and the one traced below); since the result is a sorted list, the order pages are visited in never shows up in the output.

```text
start_url = "http://a.com/index"

http://a.com/index    -> http://a.com/about#team, http://b.com/x,
                          http://a.com/index, https://a.com/index
http://a.com/about    -> http://a.com/index#top, http://a.com/products
https://a.com/index   -> http://a.com/products
http://a.com/products -> http://a.com/products

crawl(start_url, parser):
  visit http://a.com/index                # the start page
    -> http://a.com/about#team   normalises to .../about,   new,               queued
    -> http://b.com/x            hostname "b.com" != "a.com",                  skipped
    -> http://a.com/index        already visited (a self-link),                skipped
    -> https://a.com/index       hostname still "a.com" (scheme is not
                                  compared); a NEW normalised URL, since
                                  normalisation does not touch the scheme,      queued
  visit http://a.com/about
    -> http://a.com/index#top    normalises to .../index, already visited,     skipped
    -> http://a.com/products     new,                                          queued
  visit https://a.com/index
    -> http://a.com/products     already visited (queued just above),         skipped
  visit http://a.com/products
    -> http://a.com/products     itself, already visited (a 1-page cycle),    skipped

crawl(start_url, parser) == [
    "http://a.com/about", "http://a.com/index",
    "http://a.com/products", "https://a.com/index",
]
```

### Part 2 — Concurrent crawl with threads

Implement `crawl_concurrent`, with the same contract as `crawl`, that overlaps the latency of `get_urls` across up to `max_workers` threads running at once instead of fetching one page at a time.

```py
def crawl_concurrent(start_url: str, html_parser: HtmlParser, max_workers: int = 8) -> list[str]: ...
```

For any `max_workers`, the returned list must be exactly what `crawl` returns for the same graph, and `get_urls` must still be called exactly once per crawled URL: concurrency is only allowed to change how fast the answer arrives, never which pages are fetched or how many times each one is. On the graph above, `crawl_concurrent(start_url, parser, max_workers=2)` still returns the same four URLs and calls `get_urls` exactly four times in total, however those four calls happen to interleave across threads. At no point may more than `max_workers` calls to `get_urls` be running at the same time.

### Part 3 — Bounded asyncio crawl

Implement `crawl_async`, an `async def` coroutine function with the same contract as `crawl`, that uses `asyncio` so that at most `concurrency` calls to `get_urls` are ever in flight together. `get_urls` itself is an ordinary, blocking function, not a coroutine, so it cannot simply be awaited; it has to be driven in a way that does not block the event loop for the duration of a call.

```py
async def crawl_async(start_url: str, html_parser: HtmlParser, concurrency: int = 8) -> list[str]: ...
```

For any `concurrency`, `crawl_async` must return exactly what `crawl` returns for the same graph, again calling `get_urls` exactly once per crawled URL and never running more than `concurrency` calls to it at once — with `concurrency = 2` and ten pages linked directly from `start_url`, for instance, at most two of those ten calls to `get_urls` are ever running together, no matter how many of the ten already have a task created for them.
