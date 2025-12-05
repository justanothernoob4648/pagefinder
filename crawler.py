from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup

USER_AGENT = "PageRankPlus/1.0"
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)


# ----------------------- URL helpers ----------------------- #
def clean_url(url: str) -> Optional[str]:
    """Normalize a URL by removing fragments/query, normalizing scheme/host, and trimming slashes."""
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
        return None

    netloc = parsed.netloc.lower()
    if not netloc:
        return None

    path = parsed.path or "/"
    path = path.rstrip("/") or "/"

    cleaned = parsed._replace(
        scheme=(parsed.scheme or "http").lower(),
        netloc=netloc,
        path=path,
        params="",
        query="",
        fragment="",
    )
    return urlunparse(cleaned)


def is_internal_link(url: str, root_netloc: str) -> bool:
    """Check if a link belongs to the same domain (ignoring leading www)."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    base = root_netloc.lower()
    if host == base:
        return True
    if host.startswith("www."):
        host = host[4:]
    if base.startswith("www."):
        base = base[4:]
    return host == base


# ----------------------- Extraction helpers ----------------------- #
def extract_links(soup: BeautifulSoup, base_url: str, root_netloc: str) -> Set[str]:
    """Extract and normalize same-domain anchor links from a page."""
    links: Set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        resolved = urljoin(base_url, href)
        cleaned = clean_url(resolved)
        if not cleaned:
            continue
        if not is_internal_link(cleaned, root_netloc):  # check that the link is within the same domain
            continue
        links.add(cleaned)
    return links


def extract_navigation_links(
    soup: BeautifulSoup, base_url: str, root_netloc: str
) -> Set[str]:
    """Collect links inside nav/header/menu/ul elements as crawl seeds."""
    nav_links: Set[str] = set()
    for tag_name in ("nav", "header", "menu", "ul"):
        for tag in soup.find_all(tag_name):
            nav_links.update(extract_links(tag, base_url, root_netloc))
    return nav_links




# ----------------------- Crawling ----------------------- #
async def fetch_html(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch a page and return text content (no content-type filtering)."""
    try:
        async with session.get(url) as resp:
            if resp.status >= 400:
                return None
            try:
                return await resp.text()
            except UnicodeDecodeError:
                raw = await resp.read()
                return raw.decode(errors="ignore")
    except (aiohttp.ClientError, asyncio.TimeoutError, asyncio.CancelledError):
        return None


def _extract_title(soup: BeautifulSoup) -> str:
    return soup.title.string.strip() if soup.title and soup.title.string else ""


async def _crawl_async(
    entry_url: str, max_depth: int
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    graph: Dict[str, List[str]] = defaultdict(list)
    titles: Dict[str, str] = {}
    snippets: Dict[str, str] = {}
    seen: Set[str] = set()

    entry_clean = clean_url(entry_url)
    if not entry_clean:
        raise ValueError(f"Invalid entry URL: {entry_url}")
    root_netloc = urlparse(entry_clean).netloc

    async with aiohttp.ClientSession(
        timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    ) as session:
        home_html = await fetch_html(session, entry_clean)
        if not home_html:
            return graph, titles, snippets

        home_soup = BeautifulSoup(home_html, "html.parser")
        titles[entry_clean] = _extract_title(home_soup)
        home_text = home_soup.get_text(" ", strip=True)
        snippets[entry_clean] = home_text[:300]

        nav_links = extract_navigation_links(home_soup, entry_clean, root_netloc)
        graph[entry_clean] = sorted(nav_links)
        seen.add(entry_clean)
        print(f"[crawl] seed={entry_clean} nav_links={len(nav_links)}", flush=True) #for tracing purposes

        queue = deque((link, 1) for link in nav_links)
        for link in nav_links:
            print(f"[crawl] nav-link -> {link}", flush=True)

        while queue:
            link, depth = queue.popleft()
            if depth > max_depth or link in seen:
                continue

            print(f"[crawl] depth={depth} url={link}", flush=True)
            seen.add(link)

            html = await fetch_html(session, link)
            if not html:
                graph.setdefault(link, [])
                continue

            soup = BeautifulSoup(html, "html.parser")
            titles[link] = _extract_title(soup)
            text = soup.get_text(" ", strip=True)
            snippets[link] = text[:300]

            links = extract_links(soup, link, root_netloc)
            graph[link] = sorted(links)

            if depth < max_depth:
                for child in list(links)[:15]:  # limit breadth to 15 children
                    if child not in seen:
                        queue.append((child, depth + 1))

    return graph, titles, snippets, seen, root_netloc


def crawl_website(entry_url: str, max_depth: int = 2):
    """
    Crawl from a site's navigation links and build a link graph.

    Returns (graph, url_to_index, index_to_url).
    Also populates crawl_website.titles, crawl_website.snippets, crawl_website.seen,
    and crawl_website.root_netloc for later incremental expansion.
    """
    graph, titles, snippets, seen, root_netloc = asyncio.run(
        _crawl_async(entry_url, max_depth=max_depth)
    )

    all_urls: Set[str] = set(graph.keys())
    for children in graph.values():
        all_urls.update(children)

    url_to_index = {url: idx for idx, url in enumerate(sorted(all_urls))}
    index_to_url = {idx: url for url, idx in url_to_index.items()}

    crawl_website.titles = titles
    crawl_website.snippets = snippets
    crawl_website.seen = seen
    crawl_website.root_netloc = root_netloc
    return graph, url_to_index, index_to_url


# ----------------------- Incremental crawl ----------------------- #
async def _crawl_subset(
    seeds: Iterable[str],
    seen: Set[str],
    graph: Dict[str, List[str]],
    titles: Dict[str, str],
    snippets: Dict[str, str],
    root_netloc: str,
    max_pages: int,
) -> int:
    added = 0
    queue: deque[Tuple[str, int]] = deque((s, 0) for s in seeds if s not in seen)
    if not queue:
        return 0

    async with aiohttp.ClientSession(
        timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    ) as session:
        while queue and added < max_pages:
            link, depth = queue.popleft()
            if link in seen:
                continue

            seen.add(link)
            html = await fetch_html(session, link)
            if not html:
                graph.setdefault(link, [])
                continue

            soup = BeautifulSoup(html, "html.parser")
            titles[link] = _extract_title(soup)
            text = soup.get_text(" ", strip=True)
            snippets[link] = text[:300]

            links = extract_links(soup, link, root_netloc)
            graph[link] = sorted(links)
            added += 1

            if depth < 1:  # one hop deeper
                for child in list(links)[:10]:
                    if child not in seen:
                        queue.append((child, depth + 1))
    return added


def expand_frontier(
    entry_url: str,
    seeds: Iterable[str],
    graph: Dict[str, List[str]],
    titles: Dict[str, str],
    snippets: Dict[str, str],
    seen: Set[str],
    root_netloc: Optional[str] = None,
    max_pages: int = 10,
) -> int:
    """
    Incrementally crawl a small set of seeds (and one hop of their children).
    Updates graph/titles/snippets/seen in place and returns number of pages added.
    """
    root = root_netloc or urlparse(entry_url).netloc
    return asyncio.run(
        _crawl_subset(seeds, seen, graph, titles, snippets, root, max_pages=max_pages)
    )
