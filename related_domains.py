#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
import numpy as np
from bs4 import BeautifulSoup, Tag

from ppr import PPRConfig, ConvergenceStats, PersonalizedPageRank



# Config ------------------------------------------

@dataclass
class CrawlConfig:
    """Crawler config"""
    seed_domain: str
    max_pages_total: int = 2000
    max_pages_per_host: int = 200
    max_depth: int = 3
    politeness_delay: float = 0.01  # seconds between requests
    timeout: float = 15.0  # request timeout in seconds
    max_retries: int = 2
    user_agent: str = "RelatedDomainsAgent/1.0 (+https://github.com/related-domains)"
    allowed_schemes: Tuple[str, ...] = ("http", "https")
    # Additional domains to treat as first-party (e.g., ["troy.k12.mi.us"] for troycolts.org)
    alias_domains: List[str] = field(default_factory=list)


@dataclass
class TemplateFilterConfig:
    """Settings for filtering out nav/footer/header (to avoid repeats)"""
    remove_nav: bool = True
    remove_footer: bool = True
    remove_header: bool = True
    remove_aside: bool = True
    filter_tokens: Tuple[str, ...] = (
        "nav", "navbar", "menu", "footer", "header", "sidebar", "breadcrumb",
    )


# URL and Host Util ------------------------------------------

def strip_www_for_seed(host: str, seed_apex: str) -> str:
    """Remove www. prefix to avoid double counting"""
    if not host.startswith("www."):
        return host
    without_www = host[4:]  # Remove "www."
    # Only strip if it's first-party (seed or subdomain)
    if without_www == seed_apex or without_www.endswith("." + seed_apex):
        return without_www
    return host  # Keep www for external domains


def normalize_host(netloc: str, seed_apex: str) -> str:
    """Lowercase, strip port, strip www."""
    # Strip port
    host = netloc.lower().split(":")[0]
    # Strip www.
    host = strip_www_for_seed(host, seed_apex)
    return host


def extract_host(url: str, seed_apex: str) -> Optional[str]:
    """Pull the normalized hostname out of a URL."""
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return normalize_host(parsed.netloc, seed_apex)
    except Exception:
        return None


def is_first_party(host: str, seed_apex: str, alias_domains: List[str] = None) -> bool:
    """True if host belongs to seed domain or its aliases."""
    if host == seed_apex or host.endswith("." + seed_apex):
        return True
    if alias_domains:
        for alias in alias_domains:
            if host == alias or host.endswith("." + alias):
                return True
    return False


def normalize_url(url: str) -> Optional[str]:
    """Clean up URL: lowercase, remove fragments, ensure path exists."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None
        # Remove fragment, ensure path is at least "/"
        path = parsed.path if parsed.path else "/"
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            parsed.query,
            ""  # Remove fragment
        ))
        return normalized
    except Exception:
        return None


def resolve_url(base_url: str, href: str, allowed_schemes: Tuple[str, ...] = ("http", "https")) -> Optional[str]:
    """Turn a relative href into an absolute URL. Filters out mailto/javascript/etc."""
    if not href:
        return None

    href = href.strip()

    # Filter out non-navigational schemes
    href_lower = href.lower()
    if any(href_lower.startswith(s) for s in ("mailto:", "tel:", "javascript:", "data:", "#")):
        return None

    try:
        # Handle protocol-relative URLs
        if href.startswith("//"):
            base_parsed = urlparse(base_url)
            href = base_parsed.scheme + ":" + href

        # Resolve relative URLs
        resolved = urljoin(base_url, href)
        parsed = urlparse(resolved)

        # Filter by scheme
        if parsed.scheme.lower() not in allowed_schemes:
            return None

        return normalize_url(resolved)
    except Exception:
        return None


def get_homepage_url(host: str, scheme: str = "https") -> str:
    """Build homepage URL like https://cmu.edu/"""
    return f"{scheme}://{host}/"


def is_homepage(url: str) -> bool:
    """True if URL is a root page (path is / or empty)."""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        return path == "" and not parsed.query
    except Exception:
        return False



# Template Filtering ------------------------------------------

class TemplateFilter:
    """Strips nav/footer/sidebar from HTML before extracting links."""

    def __init__(self, config: TemplateFilterConfig):
        self.config = config
        # Pre-compile regex for token matching
        self._token_pattern = re.compile(
            "|".join(re.escape(t) for t in config.filter_tokens),
            re.IGNORECASE
        )

    def _has_template_class_or_id(self, element: Tag) -> bool:
        """True if element's class/id looks like boilerplate."""
        # Check classes
        classes = element.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        for cls in classes:
            if self._token_pattern.search(cls):
                return True

        # Check id
        element_id = element.get("id", "")
        if element_id and self._token_pattern.search(element_id):
            return True

        return False

    def filter_template_elements(self, soup: BeautifulSoup) -> None:
        """Remove boilerplate elements from soup in place."""
        # Remove semantic template elements
        tags_to_remove = []
        if self.config.remove_nav:
            tags_to_remove.append("nav")
        if self.config.remove_footer:
            tags_to_remove.append("footer")
        if self.config.remove_header:
            tags_to_remove.append("header")
        if self.config.remove_aside:
            tags_to_remove.append("aside")

        for tag_name in tags_to_remove:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Remove elements with template class/id tokens
        # We iterate over a copy since we're modifying during iteration
        for element in soup.find_all(True):  # True matches all tags
            if self._has_template_class_or_id(element):
                element.decompose()

    def extract_links_with_counts(
        self,
        soup: BeautifulSoup,
        base_url: str,
        page_is_homepage: bool,
        seed_apex: str,
        allowed_schemes: Tuple[str, ...] = ("http", "https")
    ) -> Dict[str, int]:
        """Get links from page, counting how many point to each host."""
        # Apply template filtering only if NOT homepage
        if not page_is_homepage:
            self.filter_template_elements(soup)

        # Extract and count links
        host_counts: Dict[str, int] = defaultdict(int)

        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            resolved = resolve_url(base_url, href, allowed_schemes)
            if not resolved:
                continue

            target_host = extract_host(resolved, seed_apex)
            if target_host:
                host_counts[target_host] += 1  # Count every occurrence

        return dict(host_counts)



# Domain Graph ------------------------------------------

class DomainGraph:
    """Graph of domains connected by links. Edge weight = link count."""

    def __init__(self, seed_host: str):
        self.seed_host = seed_host
        self.nodes: Set[str] = set()
        self.first_party_hosts: Set[str] = set()
        self.external_hosts: Set[str] = set()
        # edges[source][target] = weight
        self.edges: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_node(self, host: str, is_first_party: bool) -> None:
        self.nodes.add(host)
        if is_first_party:
            self.first_party_hosts.add(host)
        else:
            self.external_hosts.add(host)

    def add_edge(self, source_host: str, target_host: str, weight: int = 1) -> None:
        """Add or increment link weight between hosts."""
        self.edges[source_host][target_host] += weight

    def get_subdomains(self) -> Set[str]:
        """First-party hosts minus the seed itself."""
        return self.first_party_hosts - {self.seed_host}

    def get_out_degree(self, host: str) -> int:
        """Total outgoing link weight."""
        if host not in self.edges:
            return 0
        return sum(self.edges[host].values())

    def is_dangling(self, host: str) -> bool:
        """True if node has no outgoing links."""
        return self.get_out_degree(host) == 0

    def to_column_stochastic_matrix(
        self,
        host_to_index: Dict[str, int],
        restart_vector: np.ndarray
    ) -> np.ndarray:
        """Build transition matrix for PageRank. Dangling nodes use restart vector."""
        n = len(host_to_index)
        M = np.zeros((n, n))

        for source, targets in self.edges.items():
            if source not in host_to_index:
                continue
            src_idx = host_to_index[source]
            total_weight = sum(targets.values())

            if total_weight == 0:
                # Dangling node: column is restart vector
                M[:, src_idx] = restart_vector
            else:
                for target, weight in targets.items():
                    if target not in host_to_index:
                        continue
                    tgt_idx = host_to_index[target]
                    M[tgt_idx, src_idx] = weight / total_weight

        # Handle nodes not in edges (implicit dangling)
        for host, idx in host_to_index.items():
            if host not in self.edges or self.get_out_degree(host) == 0:
                M[:, idx] = restart_vector

        return M

    def get_stats(self) -> Dict[str, int]:
        total_edges = sum(sum(targets.values()) for targets in self.edges.values())
        return {
            "total_nodes": len(self.nodes),
            "first_party_hosts": len(self.first_party_hosts),
            "external_hosts": len(self.external_hosts),
            "total_edge_weight": total_edges,
            "subdomains": len(self.get_subdomains()),
        }



# Async Crawler ------------------------------------------

class RelatedDomainsCrawler:
    """Crawls seed domain via BFS, building a graph of linked domains."""

    def __init__(
        self,
        crawl_config: CrawlConfig,
        template_config: TemplateFilterConfig,
        verbose: bool = False
    ):
        self.crawl_config = crawl_config
        self.template_filter = TemplateFilter(template_config)
        self.verbose = verbose

        # Normalize seed domain
        self.seed_apex = crawl_config.seed_domain.lower()
        if self.seed_apex.startswith("www."):
            self.seed_apex = self.seed_apex[4:]

        # Initialize graph
        self.graph = DomainGraph(self.seed_apex)
        self.graph.add_node(self.seed_apex, is_first_party=True)

        # Crawl state
        self.visited_urls: Set[str] = set()
        self.pages_per_host: Dict[str, int] = defaultdict(int)
        self.total_pages_crawled: int = 0
        self.host_homepages_queued: Set[str] = set()

    def _is_first_party(self, host: str) -> bool:
        return is_first_party(host, self.seed_apex, self.crawl_config.alias_domains)

    def _should_crawl(self, url: str, depth: int) -> bool:
        """Check limits and first-party status."""
        if url in self.visited_urls:
            return False
        if depth > self.crawl_config.max_depth:
            return False
        if self.total_pages_crawled >= self.crawl_config.max_pages_total:
            return False

        host = extract_host(url, self.seed_apex)
        if not host or not self._is_first_party(host):
            return False
        if self.pages_per_host[host] >= self.crawl_config.max_pages_per_host:
            return False

        return True

    async def _fetch_with_retry(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Tuple[Optional[str], int]:
        """Fetch URL, retrying on failure. Returns (html, status)."""
        for attempt in range(self.crawl_config.max_retries + 1):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.crawl_config.timeout),
                    allow_redirects=True,
                    headers={"User-Agent": self.crawl_config.user_agent}
                ) as response:
                    # Only process HTML content
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" not in content_type.lower():
                        return None, response.status

                    html = await response.text(errors="ignore")
                    return html, response.status

            except asyncio.TimeoutError:
                if self.verbose:
                    print(f"  Timeout (attempt {attempt + 1}): {url}")
            except aiohttp.ClientError as e:
                if self.verbose:
                    print(f"  Error (attempt {attempt + 1}): {url} - {e}")
            except Exception as e:
                if self.verbose:
                    print(f"  Unexpected error: {url} - {e}")
                break

        return None, 0

    async def _process_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
        depth: int
    ) -> List[Tuple[str, int]]:
        """Fetch page, extract links, update graph. Returns new URLs to crawl."""
        # Get source host (attribute to ORIGINAL URL, not redirect)
        source_host = extract_host(url, self.seed_apex)
        if not source_host:
            return []

        # Mark as visited and update counts
        self.visited_urls.add(url)
        self.pages_per_host[source_host] += 1
        self.total_pages_crawled += 1

        if self.verbose:
            print(f"[{self.total_pages_crawled}] Crawling: {url} (depth={depth})")

        # Fetch page
        html, status = await self._fetch_with_retry(session, url)
        if not html:
            return []

        # Parse HTML
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            if self.verbose:
                print(f"  Parse error: {e}")
            return []

        # Check if homepage
        page_is_homepage = is_homepage(url)

        # Extract links with counts (applies template filtering if not homepage)
        host_counts = self.template_filter.extract_links_with_counts(
            soup,
            url,
            page_is_homepage,
            self.seed_apex,
            self.crawl_config.allowed_schemes
        )

        # Update graph and collect URLs to crawl
        next_urls: List[Tuple[str, int]] = []

        for target_host, count in host_counts.items():
            target_is_first_party = self._is_first_party(target_host)

            # Add node and edge to graph
            self.graph.add_node(target_host, target_is_first_party)
            self.graph.add_edge(source_host, target_host, count)

            # If first-party, we may want to crawl it
            if target_is_first_party:
                # Ensure homepage is queued for new hosts
                if target_host not in self.host_homepages_queued:
                    homepage = get_homepage_url(target_host)
                    if homepage not in self.visited_urls:
                        # Add homepage with priority (lower depth)
                        next_urls.insert(0, (homepage, depth + 1))
                    self.host_homepages_queued.add(target_host)

        # Also extract individual URLs for BFS (not just hosts)
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            resolved = resolve_url(href, url, self.crawl_config.allowed_schemes)
            if not resolved:
                continue

            target_host = extract_host(resolved, self.seed_apex)
            if target_host and self._is_first_party(target_host):
                if resolved not in self.visited_urls:
                    next_urls.append((resolved, depth + 1))

        return next_urls

    async def crawl(self) -> DomainGraph:
        """Run BFS crawl, return the domain graph."""
        # Initialize queue with seed homepage
        queue: deque[Tuple[str, int]] = deque()
        seed_homepage = get_homepage_url(self.seed_apex)
        queue.append((seed_homepage, 0))
        self.host_homepages_queued.add(self.seed_apex)

        # Also try www version in case it redirects
        www_homepage = get_homepage_url("www." + self.seed_apex)
        queue.append((www_homepage, 0))

        print(f"Starting crawl of {self.seed_apex}...")
        print(f"  Max pages total: {self.crawl_config.max_pages_total}")
        print(f"  Max pages per host: {self.crawl_config.max_pages_per_host}")
        print(f"  Max depth: {self.crawl_config.max_depth}")
        print()

        connector = aiohttp.TCPConnector(limit=10, limit_per_host=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            while queue and self.total_pages_crawled < self.crawl_config.max_pages_total:
                url, depth = queue.popleft()

                if not self._should_crawl(url, depth):
                    continue

                # Process page and get new URLs
                new_urls = await self._process_page(session, url, depth)

                # Add new URLs to queue
                for new_url, new_depth in new_urls:
                    if new_url not in self.visited_urls:
                        queue.append((new_url, new_depth))

                # Politeness delay
                await asyncio.sleep(self.crawl_config.politeness_delay)

        print(f"\nCrawl complete. Total pages: {self.total_pages_crawled}")
        return self.graph



# Results Reporter ------------------------------------------

class ResultsReporter:
    """Print results to console."""

    @staticmethod
    def print_crawl_stats(
        graph: DomainGraph,
        pages_per_host: Dict[str, int],
        convergence: ConvergenceStats
    ) -> None:
        stats = graph.get_stats()

        print("\n" + "=" * 60)
        print("CRAWL STATISTICS")
        print("=" * 60)
        print(f"  Total pages crawled: {sum(pages_per_host.values())}")
        print(f"  First-party hosts: {stats['first_party_hosts']}")
        print(f"  External hosts: {stats['external_hosts']}")
        print(f"  Subdomains discovered: {stats['subdomains']}")
        print(f"  Total edge weight: {stats['total_edge_weight']}")

        print("\n  Pages per host (top 20):")
        sorted_hosts = sorted(pages_per_host.items(), key=lambda x: -x[1])[:20]
        for host, count in sorted_hosts:
            print(f"    {host}: {count}")

        print("\n  PPR Convergence:")
        print(f"    Iterations: {convergence.iterations}")
        print(f"    Final L1 diff: {convergence.final_l1_diff:.2e}")
        print(f"    Converged: {'Yes' if convergence.converged else 'No'}")

    @staticmethod
    def print_ranked_hosts(
        scores: np.ndarray,
        host_to_index: Dict[str, int],
        graph: DomainGraph,
        top_k: int = 50
    ) -> None:
        if len(scores) == 0:
            print("\nNo hosts to rank.")
            return

        # Create index to host mapping
        index_to_host = {i: h for h, i in host_to_index.items()}

        # Sort by score
        ranked_indices = np.argsort(-scores)

        print("\n" + "=" * 60)
        print(f"TOP {min(top_k, len(scores))} RELATED DOMAINS (PPR Score)")
        print("=" * 60)

        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            host = index_to_host[idx]
            score = scores[idx]
            host_type = "1P" if host in graph.first_party_hosts else "EXT"
            print(f"  {rank:3d}. [{host_type}] {host:40s} {score:.6f}")

    @staticmethod
    def print_raw_edge_weights(
        graph: DomainGraph,
        top_k: int = 20
    ) -> None:
        print("\n" + "=" * 60)
        print(f"RAW EDGE WEIGHTS (from {graph.seed_host})")
        print("=" * 60)

        if graph.seed_host not in graph.edges:
            print("  No outgoing edges from seed.")
            return

        edges = graph.edges[graph.seed_host]
        sorted_edges = sorted(edges.items(), key=lambda x: -x[1])[:top_k]

        for target, weight in sorted_edges:
            host_type = "1P" if target in graph.first_party_hosts else "EXT"
            print(f"  → [{host_type}] {target:40s}: {weight}")



# CLI and Main ------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Related Domains Agent - Rank domains using Personalized PageRank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python related_domains.py cmu.edu
    python related_domains.py cmu.edu --max-pages-total 500 --max-depth 2
    python related_domains.py stanford.edu --top-k 100 --verbose
        """
    )

    parser.add_argument(
        "seed_domain",
        help="Seed domain to crawl (e.g., 'cmu.edu')"
    )

    # Crawling options
    crawl_group = parser.add_argument_group("Crawling Options")
    crawl_group.add_argument(
        "--max-pages-total",
        type=int,
        default=2000,
        help="Maximum pages to crawl total (default: 2000)"
    )
    crawl_group.add_argument(
        "--max-pages-per-host",
        type=int,
        default=200,
        help="Maximum pages per host (default: 200)"
    )
    crawl_group.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum BFS depth (default: 3)"
    )
    crawl_group.add_argument(
        "--politeness-delay",
        type=float,
        default=0.1,
        help="Seconds between requests (default: 0.1)"
    )
    crawl_group.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds (default: 15)"
    )
    crawl_group.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per request (default: 2)"
    )
    crawl_group.add_argument(
        "--alias",
        type=str,
        action="append",
        default=[],
        dest="alias_domains",
        metavar="DOMAIN",
        help="Additional domain to treat as first-party (can be repeated, e.g., --alias troy.k12.mi.us)"
    )

    # PageRank options
    ppr_group = parser.add_argument_group("PageRank Options")
    ppr_group.add_argument(
        "--damping",
        type=float,
        default=0.85,
        help="Damping factor alpha (default: 0.85)"
    )
    ppr_group.add_argument(
        "--seed-weight",
        type=float,
        default=0.9,
        help="Weight for seed in restart vector (default: 0.9)"
    )
    ppr_group.add_argument(
        "--convergence",
        type=float,
        default=1e-10,
        help="L1 convergence threshold (default: 1e-10)"
    )
    ppr_group.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Maximum PPR iterations (default: 200)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top domains to show (default: 50)"
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    # Build configurations
    crawl_config = CrawlConfig(
        seed_domain=args.seed_domain,
        max_pages_total=args.max_pages_total,
        max_pages_per_host=args.max_pages_per_host,
        max_depth=args.max_depth,
        politeness_delay=args.politeness_delay,
        timeout=args.timeout,
        max_retries=args.retries,
        alias_domains=args.alias_domains,
    )

    ppr_config = PPRConfig(
        damping=args.damping,
        seed_weight=args.seed_weight,
        subdomain_weight=1.0 - args.seed_weight,  # Complement to 1
        convergence_threshold=args.convergence,
        max_iterations=args.max_iterations,
    )

    template_config = TemplateFilterConfig()

    # Print header
    print("=" * 60)
    print(f"RELATED DOMAINS AGENT - {args.seed_domain}")
    print("=" * 60)

    # Run crawler
    crawler = RelatedDomainsCrawler(crawl_config, template_config, args.verbose)
    graph = await crawler.crawl()

    # Handle edge case: tiny or empty graph
    if len(graph.nodes) < 2:
        print("\nWarning: Graph has fewer than 2 nodes. Results may be trivial.")

    # Run PPR
    ppr = PersonalizedPageRank(ppr_config)
    scores, host_to_index, convergence = ppr.compute(graph)

    # Print results
    ResultsReporter.print_crawl_stats(graph, crawler.pages_per_host, convergence)
    ResultsReporter.print_ranked_hosts(scores, host_to_index, graph, args.top_k)
    ResultsReporter.print_raw_edge_weights(graph, top_k=20)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nCrawl interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
