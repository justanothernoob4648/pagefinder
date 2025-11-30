from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "PagerankCrawler/1.0"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}


def canonicalize_url(url: str) -> Optional[str]:
    """
    Normalize a URL for consistent graph nodes.

    - Resolves scheme/netloc casing.
    - Removes fragments and query strings.
    - Strips trailing slashes except for the root path.
    Returns None for non-http(s) URLs or missing hosts.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None

    netloc = parsed.netloc.lower()
    if not netloc:
        return None

    path = parsed.path or "/"
    path = path.rstrip("/")
    if path == "":
        path = "/"

    cleaned = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=netloc,
        path=path,
        params="",
        query="",  # drop queries to reduce duplicate variants
        fragment="",
    )
    return urlunparse(cleaned)


def crawl_website(root_url: str, max_pages: int = 200) -> Dict[str, Set[str]]:
    """
    Crawl a single domain starting at root_url and collect link graph edges.

    Returns a mapping of canonical URL -> set of canonical URLs it links to.
    Pages outside the starting domain are ignored.
    """
    canonical_root = canonicalize_url(root_url)
    if not canonical_root:
        raise ValueError(f"Invalid root URL: {root_url}")

    allowed_domain = urlparse(canonical_root).netloc

    def strip_www(host: str) -> str:
        return host[4:] if host.startswith("www.") else host

    base_domain = strip_www(allowed_domain)

    def in_domain(netloc: str) -> bool:
        host = netloc.lower()
        return host == allowed_domain or strip_www(host) == base_domain

    to_visit = deque([canonical_root])
    visited: Set[str] = set()
    links: Dict[str, Set[str]] = {}

    while to_visit and len(visited) < max_pages:
        current = to_visit.popleft()
        if current in visited:
            continue

        try:
            response = requests.get(
                current, headers=DEFAULT_HEADERS, timeout=10, allow_redirects=True
            )
        except requests.RequestException:
            continue

        canonical_current = canonicalize_url(response.url)
        if not canonical_current or not in_domain(urlparse(canonical_current).netloc):
            continue
        if canonical_current in visited:
            continue

        visited.add(canonical_current)

        if response.status_code >= 400:
            continue

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            continue

        links.setdefault(canonical_current, set())

        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            raw_href = anchor.get("href")
            if not raw_href:
                continue

            resolved = urljoin(canonical_current, raw_href)
            normalized = canonicalize_url(resolved)
            if not normalized:
                continue
            if not in_domain(urlparse(normalized).netloc):
                continue

            links[canonical_current].add(normalized)

            if (
                normalized not in visited
                and normalized not in to_visit
                and len(visited) + len(to_visit) < max_pages
            ):
                to_visit.append(normalized)

    return links
