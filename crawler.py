from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup

USER_AGENT = "PageRankPlus/1.0"
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)


# URL helpers
def _normalize_netloc(netloc: str) -> str:
    netloc = netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def clean_url(url: str) -> Optional[str]:

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

    parsed = urlparse(url)
    host = _normalize_netloc(parsed.netloc)
    base = _normalize_netloc(root_netloc)
    return host == base


def is_allowed_link(url: str, allowed_netlocs: Set[str]) -> bool:

    parsed = urlparse(url)
    host = _normalize_netloc(parsed.netloc)
    return host in allowed_netlocs


# Extraction helpers
def extract_links(soup: BeautifulSoup, base_url: str, root_netloc: str) -> Set[str]:

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


def extract_links_allowed(
    soup: BeautifulSoup, base_url: str, allowed_netlocs: Set[str]
) -> Set[str]:

    links: Set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        resolved = urljoin(base_url, href)
        cleaned = clean_url(resolved)
        if not cleaned:
            continue
        if not is_allowed_link(cleaned, allowed_netlocs):
            continue
        links.add(cleaned)
    return links


def extract_navigation_links(
    soup: BeautifulSoup, base_url: str, root_netloc: str
) -> Set[str]:

    nav_links: Set[str] = set()
    for tag_name in ("nav", "header", "menu", "ul"):
        for tag in soup.find_all(tag_name):
            nav_links.update(extract_links(tag, base_url, root_netloc))
    return nav_links




# Crawling
async def fetch_html(session: aiohttp.ClientSession, url: str) -> Optional[str]:

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
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], Set[str], str]:
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
            return graph, titles, snippets, seen, root_netloc

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


# PASS 1: discover external domains --------------
async def _discover_domains_async(
    entry_url: str, max_depth: int
) -> Tuple[Dict[str, List[str]], Set[str], str]:
    
    graph: Dict[str, List[str]] = defaultdict(list)
    seen: Set[str] = set()

    entry_clean = clean_url(entry_url)
    if not entry_clean:
        raise ValueError(f"Invalid entry URL: {entry_url}")
    root_netloc = urlparse(entry_clean).netloc
    root_base = _normalize_netloc(root_netloc)

    async with aiohttp.ClientSession(
        timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    ) as session:
        home_html = await fetch_html(session, entry_clean)
        if not home_html:
            return graph, seen, root_netloc

        home_soup = BeautifulSoup(home_html, "html.parser")
        seen.add(entry_clean)

        nav_links = extract_navigation_links(home_soup, entry_clean, root_netloc)
        outbound_root: Set[str] = set(nav_links)

        # Also capture all anchors on the home page (external domains become domain:// nodes).
        for anchor in home_soup.find_all("a", href=True):
            href = anchor.get("href")
            resolved = urljoin(entry_clean, href)
            cleaned = clean_url(resolved)
            if not cleaned:
                continue
            netloc = urlparse(cleaned).netloc
            base = _normalize_netloc(netloc)
            if base == root_base:
                outbound_root.add(cleaned)
            else:
                domain_node = f"domain://{base}"
                if domain_node not in graph:
                    #print(f"[home - discover] new external domain = {base}", flush=True) #tracing
                    graph[domain_node] = [domain_node]  # self-loop to avoid sink spreading
                #print(f"[home - edge to ext. domain] {entry_clean} -> {domain_node}", flush=True) #tracing
                outbound_root.add(domain_node)

        graph[entry_clean] = sorted(outbound_root)

        queue = deque((link, 1) for link in nav_links)

        while queue:
            link, depth = queue.popleft()
            if depth > max_depth or link in seen:
                continue

            print(f"[crawl] visiting depth={depth} url={link}", flush=True)
            seen.add(link)
            html = await fetch_html(session, link)
            if not html:
                graph.setdefault(link, [])
                continue

            soup = BeautifulSoup(html, "html.parser")
            internal_links: Set[str] = set()
            outbound: Set[str] = set()

            for anchor in soup.find_all("a", href=True):
                href = anchor.get("href")
                resolved = urljoin(link, href)
                cleaned = clean_url(resolved)
                if not cleaned:
                    continue
                netloc = urlparse(cleaned).netloc
                base = _normalize_netloc(netloc)
                if base == root_base:
                    internal_links.add(cleaned)
                else:
                    domain_node = f"domain://{base}"
                    if domain_node not in graph:
                        #print(f"[discover] new external domain = {base}", flush=True) #tracing
                        graph[domain_node] = [domain_node]  # self-loop to avoid sink spreading
                    #print(f"[edge to ext. domain] {link} -> {domain_node}", flush=True) #tracing
                    outbound.add(domain_node)

            outbound.update(internal_links)
            graph[link] = sorted(outbound)

            if depth < max_depth:
                for child in sorted(internal_links)[:15]:
                    if child not in seen:
                        queue.append((child, depth + 1))

    return graph, seen, root_netloc


def discover_external_domains(entry_url: str, max_depth: int, top_k: int) -> Set[str]:

    from markov import build_markov_matrix, compute_pagerank

    graph, seen, root_netloc = asyncio.run(
        _discover_domains_async(entry_url, max_depth=max_depth)
    )

    all_nodes: Set[str] = set(graph.keys())
    for children in graph.values():
        all_nodes.update(children)

    url_to_index = {url: idx for idx, url in enumerate(sorted(all_nodes))}
    M = build_markov_matrix(graph, url_to_index)
    pagerank = compute_pagerank(M)

    external_scores: List[Tuple[str, float]] = []
    for url, idx in url_to_index.items():
        if url.startswith("domain://"):
            external_scores.append((url, float(pagerank[idx]) if idx < len(pagerank) else 0.0))

    external_scores.sort(key=lambda x: x[1], reverse=True)

    #tracing: print top 10 external domains and their scores
    top_preview = external_scores[:10]
    if top_preview:
        print("[whitelist] top external domains:")
        for url, score in top_preview:
            print(f"  {url} -> {score:.6f}")
    #end trace

    # Interactive whitelist
    added = 0
    ind = 0
    whitelist: Set[str] = set()
    while added<top_k and ind<len(external_scores):
        url, _ = external_scores[ind]
        normUrl = url[len("domain://") :]
        resp = input(f"\nWould you like to include domain - {normUrl}? [Y/N]: ").strip().lower()
        if resp in {"y", "yes"}:
            added+=1
            whitelist.add(_normalize_netloc(normUrl))
        ind+=1

    # # Automated whitelist
    # whitelist = {_normalize_netloc(url[len("domain://") :]) for url, _ in external_scores[:top_k]}

    # Always include root domain
    whitelist.add(_normalize_netloc(root_netloc))
    return whitelist


# Pass 1: crawl with whitelist --------------
async def _crawl_allowed_async(
    entry_url: str, allowed_netlocs: Set[str], max_depth: int
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], Set[str]]:
    allowed_netlocs = {_normalize_netloc(n) for n in allowed_netlocs}
    graph: Dict[str, List[str]] = defaultdict(list)
    titles: Dict[str, str] = {}
    snippets: Dict[str, str] = {}
    seen: Set[str] = set()

    entry_clean = clean_url(entry_url)
    if not entry_clean:
        raise ValueError(f"Invalid entry URL: {entry_url}")

    async with aiohttp.ClientSession(
        timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    ) as session:
        home_html = await fetch_html(session, entry_clean)
        if not home_html:
            return graph, titles, snippets, seen

        home_soup = BeautifulSoup(home_html, "html.parser")
        titles[entry_clean] = _extract_title(home_soup)
        home_text = home_soup.get_text(" ", strip=True)
        snippets[entry_clean] = home_text[:300]

        nav_links = extract_links_allowed(home_soup, entry_clean, allowed_netlocs)
        graph[entry_clean] = sorted(nav_links)
        seen.add(entry_clean)

        queue = deque((link, 1) for link in nav_links)

        while queue:
            link, depth = queue.popleft()
            if depth > max_depth or link in seen:
                continue

            print(f"[crawl] visiting depth={depth} url={link}", flush=True)
            seen.add(link)

            html = await fetch_html(session, link)
            if not html:
                graph.setdefault(link, [])
                continue

            soup = BeautifulSoup(html, "html.parser")
            titles[link] = _extract_title(soup)
            text = soup.get_text(" ", strip=True)
            snippets[link] = text[:300]

            links = extract_links_allowed(soup, link, allowed_netlocs)
            graph[link] = sorted(links)

            if depth < max_depth:
                for child in sorted(links)[:15]:
                    if child not in seen:
                        queue.append((child, depth + 1))

    return graph, titles, snippets, seen


def crawl_with_whitelist(
    entry_url: str, allowed_netlocs: Set[str], max_depth: int = 2
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], Set[str]]:

    return asyncio.run(_crawl_allowed_async(entry_url, allowed_netlocs, max_depth=max_depth))


# Incremental crawl
async def _crawl_subset(
    seeds: Iterable[str],
    seen: Set[str],
    graph: Dict[str, List[str]],
    titles: Dict[str, str],
    snippets: Dict[str, str],
    allowed_netlocs: Set[str],
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

            print(f"[crawl] visiting depth={depth} url={link}", flush=True)
            seen.add(link)
            html = await fetch_html(session, link)
            if not html:
                graph.setdefault(link, [])
                continue

            soup = BeautifulSoup(html, "html.parser")
            titles[link] = _extract_title(soup)
            text = soup.get_text(" ", strip=True)
            snippets[link] = text[:300]

            links = extract_links_allowed(soup, link, allowed_netlocs)
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
    allowed_netlocs: Optional[Set[str]] = None,
    max_pages: int = 10,
) -> int:

    if not allowed_netlocs:
        root = urlparse(entry_url).netloc
        allowed_netlocs = {_normalize_netloc(root)}
    else:
        allowed_netlocs = {_normalize_netloc(d) for d in allowed_netlocs}
    return asyncio.run(
        _crawl_subset(seeds, seen, graph, titles, snippets, allowed_netlocs, max_pages=max_pages)
    )
