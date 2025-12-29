#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import heapq
import os

# Load .env file if avail
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # no env, jus use env vars directly
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
import numpy as np
from bs4 import BeautifulSoup

# Import from related_domains.py for whitelist
from related_domains import (
    CrawlConfig as RDCrawlConfig,
    TemplateFilterConfig,
    RelatedDomainsCrawler,
)
from ppr import PPRConfig, PersonalizedPageRank

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None



# Configuration Dataclasses ------------------------------------------

@dataclass
class WhitelistConfig:
    """Stage 1 settings."""
    seed_domain: str
    top_n: int = 10
    max_pages: int = 500
    max_depth: int = 2


@dataclass
class APIConfig:
    """Embedding API settings. Only OpenAI works (OpenRouter has no embeddings)."""
    provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    def get_client(self) -> OpenAI:
        if OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")

        if self.provider == "openrouter":
            raise ValueError(
                "OpenRouter does not support embedding models. "
                "Please use --api-provider openai instead. "
                "Make sure OPENAI_API_KEY is set in your .env file."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        return OpenAI(api_key=api_key, timeout=60)


@dataclass
class ScoringConfig:
    """Scoring thresholds and chunking settings."""
    chunk_size: int = 5000
    chunk_overlap: int = 0
    max_chunks: int = 2
    keep_threshold: float = 0.40
    expand_threshold: float = 0.35


@dataclass
class CrawlConfig:
    """Stage 2 crawler settings."""
    whitelisted_domains: List[str] = field(default_factory=list)
    max_pages_total: int = 1000
    max_pages_per_domain: int = 200
    max_depth: int = 4
    politeness_delay: float = 0.1
    timeout: float = 20.0
    max_retries: int = 1
    user_agent: str = "TaskPagesAgent/1.0 (+https://github.com/task-pages)"


@dataclass
class HybridConfig:
    """Settings for expanding crawl when initial results are weak."""
    similarity_threshold: float = 0.6
    expansion_depth: int = 2
    expansion_max_pages_per_seed: int = 75
    top_seeds: int = 3


@dataclass
class PageResult:
    """A crawled page with its relevance score."""
    url: str
    title: str
    score: float
    best_chunk: str
    depth: int = 0


@dataclass
class CrawlStats:
    """Summary of what happened during the crawl."""
    total_pages_crawled: int = 0
    pages_per_domain: Dict[str, int] = field(default_factory=dict)
    max_depth_reached: int = 0
    hybrid_triggered: bool = False
    hybrid_pages_added: int = 0
    embedding_model: str = ""
    keep_threshold: float = 0.0
    expand_threshold: float = 0.0



# Stopwords for keyword extraction ------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "how", "when",
    "where", "why", "find", "get", "show", "tell", "give", "list",
    "about", "all", "also", "any", "each", "every", "if", "into",
    "just", "more", "most", "no", "not", "only", "other", "out",
    "over", "same", "some", "such", "than", "then", "there", "through",
    "too", "under", "up", "very", "want", "me", "my", "your", "his", "her"
}



# URL and Host Utilities ------------------------------------------

def normalize_host(netloc: str) -> str:
    """Lowercase, strip port, strip www."""
    host = netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def extract_host(url: str) -> Optional[str]:
    """Pull hostname from URL."""
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return normalize_host(parsed.netloc)
    except Exception:
        return None


def is_whitelisted(host: str, whitelist: List[str]) -> bool:
    """True if host or its parent domain is in the whitelist."""
    for allowed in whitelist:
        if host == allowed or host.endswith("." + allowed):
            return True
    return False


def normalize_url(url: str) -> Optional[str]:
    """Clean up URL: lowercase, remove fragments."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None
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


def resolve_url(base_url: str, href: str) -> Optional[str]:
    """Turn relative href into absolute URL."""
    if not href:
        return None

    href = href.strip()
    href_lower = href.lower()

    # Filter non-navigational schemes
    if any(href_lower.startswith(s) for s in ("mailto:", "tel:", "javascript:", "data:", "#")):
        return None

    try:
        if href.startswith("//"):
            base_parsed = urlparse(base_url)
            href = base_parsed.scheme + ":" + href

        resolved = urljoin(base_url, href)
        parsed = urlparse(resolved)

        if parsed.scheme.lower() not in ("http", "https"):
            return None

        return normalize_url(resolved)
    except Exception:
        return None


def get_homepage_url(host: str, scheme: str = "https") -> str:
    return f"{scheme}://{host}/"


def extract_task_keywords(task_query: str) -> List[str]:
    """Pull meaningful words from task query for matching URLs/anchors."""
    tokens = re.findall(r"[a-z]+", task_query.lower())
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return keywords


def url_contains_keywords(url: str, keywords: List[str]) -> bool:
    """True if URL contains any of the keywords."""
    url_lower = url.lower()
    return any(kw in url_lower for kw in keywords)


# Content Extractor ------------------------------------------

class ContentExtractor:
    """Strips boilerplate and extracts main text from HTML."""

    # Tags to remove entirely
    REMOVE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript"]

    # Class/ID tokens indicating boilerplate
    BOILERPLATE_TOKENS = [
        "nav", "navbar", "menu", "footer", "header", "sidebar",
        "breadcrumb", "advertisement", "ad-", "social", "share"
    ]

    def __init__(self):
        self._token_pattern = re.compile(
            "|".join(re.escape(t) for t in self.BOILERPLATE_TOKENS),
            re.IGNORECASE
        )

    def _is_boilerplate(self, element) -> bool:
        """True if element's class/id looks like nav/footer/etc."""
        # Guard against non-Tag elements (NavigableString, etc.)
        if element is None or not hasattr(element, 'get'):
            return False
        classes = element.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        for cls in classes:
            if cls and self._token_pattern.search(cls):
                return True
        element_id = element.get("id", "")
        if element_id and self._token_pattern.search(element_id):
            return True
        return False

    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Get clean text after removing nav/footer/sidebar junk."""
        # Work on a copy to avoid modifying original
        soup_copy = BeautifulSoup(str(soup), "html.parser")

        # Remove script, style, and semantic boilerplate tags
        for tag_name in self.REMOVE_TAGS:
            for element in soup_copy.find_all(tag_name):
                element.decompose()

        # Remove elements with boilerplate class/id
        # Collect elements first, then decompose (avoid modifying while iterating)
        to_remove = []
        for element in soup_copy.find_all(True):
            if element.attrs is not None and self._is_boilerplate(element):
                to_remove.append(element)
        for element in to_remove:
            try:
                element.decompose()
            except Exception:
                pass

        # Get clean text
        text = soup_copy.get_text(" ", strip=True)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_title(self, soup: BeautifulSoup) -> str:
        """Get title tag or first h1."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        # Fallback to first h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""

    def is_valid_page(self, text: str, min_length: int = 100) -> bool:
        """True if page has enough content to score."""
        return len(text) >= min_length



# Semantic Scorer ------------------------------------------

class SemanticScorer:
    """Scores pages against task using OpenAI embeddings. Chunks long pages."""

    def __init__(self, api_config: APIConfig, scoring_config: ScoringConfig):
        self.api_config = api_config
        self.scoring_config = scoring_config
        self.client = api_config.get_client()
        self.task_embedding: Optional[np.ndarray] = None
        self.task_keywords: List[str] = []
        self._cache: Dict[str, np.ndarray] = {}

    def set_task(self, task_query: str) -> None:
        """Cache embedding and keywords for the task."""
        self.task_embedding = self._embed(task_query)
        self.task_keywords = extract_task_keywords(task_query)

    def _embed(self, text: str) -> np.ndarray:
        """Get embedding vector, cached."""
        if not text:
            return np.zeros(1)

        # Check cache
        cache_key = text[:500]  # Use first 500 chars as key
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            resp = self.client.embeddings.create(
                model=self.api_config.embedding_model,
                input=text[:8000]  # Limit input length
            )
            vec = np.array(resp.data[0].embedding, dtype=float)
            self._cache[cache_key] = vec
            return vec
        except Exception as e:
            print(f"  Embedding error: {e}")
            return np.zeros(1)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        if a.size <= 1 or b.size <= 1:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _chunk_text(self, text: str) -> List[str]:
        """Break text into chunks for embedding."""
        chunk_size = self.scoring_config.chunk_size
        overlap = self.scoring_config.chunk_overlap

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
            if start >= len(text) - overlap:
                break

        return chunks if chunks else [text]

    def score_page(self, text: str) -> Tuple[float, str]:
        """Score page against task. Returns (best_score, best_chunk)."""
        if not text or self.task_embedding is None:
            return 0.0, ""

        chunks = self._chunk_text(text)
        # Limit chunks to reduce API calls (most relevant content is at top)
        chunks = chunks[:self.scoring_config.max_chunks]
        best_score = 0.0
        best_chunk = ""

        for chunk in chunks:
            chunk_emb = self._embed(chunk)
            sim = self._cosine(self.task_embedding, chunk_emb)
            if sim > best_score:
                best_score = sim
                best_chunk = chunk

        return best_score, best_chunk[:500]  # Truncate for display

    def score_anchor_text(self, anchor_text: str) -> float:
        """How well anchor text matches task keywords."""
        if not anchor_text or not self.task_keywords:
            return 0.0

        anchor_lower = anchor_text.lower()
        matches = sum(1 for kw in self.task_keywords if kw in anchor_lower)
        return min(1.0, matches / max(1, len(self.task_keywords)))

    def score_url_tokens(self, url: str) -> float:
        """How well URL path matches task keywords."""
        if not url or not self.task_keywords:
            return 0.0

        parsed = urlparse(url)
        path_tokens = re.findall(r"[a-z]+", parsed.path.lower())
        matches = sum(1 for kw in self.task_keywords if kw in path_tokens)
        return min(1.0, matches / max(1, len(self.task_keywords)))



# Priority Frontier (Best-First Queue) ------------------------------------------

@dataclass(order=True)
class FrontierItem:
    """URL waiting to be crawled, with priority."""
    priority: float  # Negative for max-heap
    url: str = field(compare=False)
    depth: int = field(compare=False)
    anchor_text: str = field(compare=False)


class PriorityFrontier:
    """Priority queue of URLs. Higher priority = crawled first."""

    def __init__(self):
        self._heap: List[FrontierItem] = []
        self._seen: Set[str] = set()

    def __len__(self) -> int:
        return len(self._heap)

    def push(
        self,
        url: str,
        depth: int,
        priority: float,
        anchor_text: str = ""
    ) -> bool:
        """Add URL to queue. Returns False if already seen."""
        if url in self._seen:
            return False
        self._seen.add(url)
        # Use negative priority for max-heap behavior
        heapq.heappush(self._heap, FrontierItem(-priority, url, depth, anchor_text))
        return True

    def pop(self) -> Optional[FrontierItem]:
        """Pop the highest priority URL."""
        if not self._heap:
            return None
        item = heapq.heappop(self._heap)
        # Restore positive priority for external use
        return FrontierItem(-item.priority, item.url, item.depth, item.anchor_text)

    def has_seen(self, url: str) -> bool:
        return url in self._seen

    def mark_seen(self, url: str) -> None:
        self._seen.add(url)

    @staticmethod
    def compute_priority(
        parent_similarity: float,
        anchor_relevance: float,
        url_relevance: float
    ) -> float:
        """Weighted combo of parent score, anchor text, and URL tokens."""
        return (
            0.3 * parent_similarity +
            0.4 * anchor_relevance +
            0.3 * url_relevance
        )



# Task-Gated Crawler  ------------------------------------------

class TaskGatedCrawler:
    """Best-first crawler that only expands links from relevant pages."""

    # Hub page tokens - pages that should always expand even with low scores
    HUB_TOKENS = [
        "admission", "admissions", "tuition", "fees", "cost", "financial",
        "student", "accounts", "directory", "index", "sitemap", "about",
        "programs", "academics", "faculty", "departments", "schools"
    ]

    def __init__(
        self,
        task_query: str,
        crawl_config: CrawlConfig,
        scoring_config: ScoringConfig,
        api_config: APIConfig,
        verbose: bool = False
    ):
        self.task_query = task_query
        self.crawl_config = crawl_config
        self.scoring_config = scoring_config
        self.verbose = verbose

        # Initialize scorer
        self.scorer = SemanticScorer(api_config, scoring_config)
        self.scorer.set_task(task_query)

        # Initialize extractor
        self.extractor = ContentExtractor()

        # Initialize frontier
        self.frontier = PriorityFrontier()

        # Results storage
        self.pages: Dict[str, PageResult] = {}
        self.pages_per_domain: Dict[str, int] = defaultdict(int)
        self.total_crawled: int = 0
        self.max_depth_reached: int = 0

    def _is_hub_page(self, url: str, title: str) -> bool:
        """True if page looks like a directory/index that should always expand."""
        url_lower = url.lower()
        title_lower = title.lower()
        return any(
            token in url_lower or token in title_lower
            for token in self.HUB_TOKENS
        )

    def _should_expand(self, page_score: float, url: str, title: str) -> bool:
        """True if we should follow this page's links."""
        if page_score >= self.scoring_config.expand_threshold:
            return True
        if self._is_hub_page(url, title):
            return True
        return False

    def _can_crawl_domain(self, host: str) -> bool:
        """True if we haven't hit the per-domain limit."""
        return self.pages_per_domain[host] < self.crawl_config.max_pages_per_domain

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[str]:
        """Fetch HTML, respecting whitelist on redirects."""
        for attempt in range(self.crawl_config.max_retries + 1):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.crawl_config.timeout),
                    allow_redirects=True,
                    headers={"User-Agent": self.crawl_config.user_agent}
                ) as response:
                    # Check if redirected outside whitelist
                    final_host = extract_host(str(response.url))
                    if final_host and not is_whitelisted(final_host, self.crawl_config.whitelisted_domains):
                        if self.verbose:
                            print(f"  Redirect outside whitelist: {url} -> {response.url}")
                        return None

                    # Only process HTML
                    headers = response.headers
                    content_type = headers.get("Content-Type", "") if headers else ""
                    if "text/html" not in content_type.lower():
                        return None

                    return await response.text(errors="ignore")

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

        return None

    def _extract_links(
        self,
        soup: BeautifulSoup,
        base_url: str
    ) -> List[Tuple[str, str]]:
        """Get (url, anchor_text) pairs, filtered to whitelist."""
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            resolved = resolve_url(base_url, href)
            if not resolved:
                continue

            target_host = extract_host(resolved)
            if not target_host:
                continue

            # WHITELIST ENFORCEMENT (hard constraint)
            if not is_whitelisted(target_host, self.crawl_config.whitelisted_domains):
                continue

            anchor_text = anchor.get_text(strip=True)
            links.append((resolved, anchor_text))

        return links

    async def _process_page(
        self,
        session: aiohttp.ClientSession,
        item: FrontierItem
    ) -> None:
        """Fetch, score, store result, and maybe expand links."""
        url = item.url
        depth = item.depth

        # Get host
        host = extract_host(url)
        if not host:
            return

        # Check domain budget
        if not self._can_crawl_domain(host):
            return

        # Fetch page
        html = await self._fetch_page(session, url)
        if not html:
            return

        # Parse
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return

        # Extract content
        title = self.extractor.extract_title(soup)
        text = self.extractor.extract_main_content(soup)

        # Validate
        if not self.extractor.is_valid_page(text):
            return

        # Update counts
        self.total_crawled += 1
        self.pages_per_domain[host] += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)

        # Compute semantic score
        score, best_chunk = self.scorer.score_page(text)

        # Store result
        self.pages[url] = PageResult(
            url=url,
            title=title,
            score=score,
            best_chunk=best_chunk,
            depth=depth
        )

        if self.verbose:
            print(f"[{self.total_crawled:4d}] Score: {score:.3f} | {url[:70]}")

        # Decide whether to expand
        if not self._should_expand(score, url, title):
            return

        # Don't expand beyond max depth
        if depth >= self.crawl_config.max_depth:
            return

        # Extract and enqueue links
        links = self._extract_links(soup, url)
        for link_url, anchor_text in links:
            if self.frontier.has_seen(link_url):
                continue

            # Compute priority
            anchor_rel = self.scorer.score_anchor_text(anchor_text)
            url_rel = self.scorer.score_url_tokens(link_url)
            priority = PriorityFrontier.compute_priority(score, anchor_rel, url_rel)

            self.frontier.push(link_url, depth + 1, priority, anchor_text)

    async def crawl(self) -> List[PageResult]:
        """Run best-first crawl. Returns pages sorted by score."""
        # Seed with homepages (priority based on PPR rank)
        for i, domain in enumerate(self.crawl_config.whitelisted_domains):
            homepage = get_homepage_url(domain)
            # Top 1 = 1.0, Top 2-3 = 0.75, Top 4+ = 0.5
            if i == 0:
                priority = 0.5
            elif i <= 2:
                priority = 0.2
            else:
                priority = 0.1
            self.frontier.push(homepage, 0, priority)

        print(f"Crawling with best-first strategy...")
        print(f"  Whitelisted domains: {len(self.crawl_config.whitelisted_domains)}")
        print(f"  Max pages: {self.crawl_config.max_pages_total}")
        print(f"  Max depth: {self.crawl_config.max_depth}")
        print()

        connector = aiohttp.TCPConnector(limit=10, limit_per_host=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            while self.frontier and self.total_crawled < self.crawl_config.max_pages_total:
                item = self.frontier.pop()
                if item is None:
                    break

                await self._process_page(session, item)
                await asyncio.sleep(self.crawl_config.politeness_delay)

        print(f"\nCrawl complete. Total pages: {self.total_crawled}")

        # Return sorted by score
        return sorted(self.pages.values(), key=lambda p: -p.score)



# Hybrid Expander ------------------------------------------

class HybridExpander:
    """Re-crawls from top pages when initial results are weak."""

    def __init__(
        self,
        config: HybridConfig,
        crawl_config: CrawlConfig,
        scoring_config: ScoringConfig,
        api_config: APIConfig,
        verbose: bool = False
    ):
        self.config = config
        self.crawl_config = crawl_config
        self.scoring_config = scoring_config
        self.api_config = api_config
        self.verbose = verbose

    def should_trigger(self, top_score: float) -> bool:
        """True if top result is below threshold."""
        return top_score < self.config.similarity_threshold

    async def expand(
        self,
        seed_pages: List[PageResult],
        existing_pages: Dict[str, PageResult],
        task_query: str
    ) -> Tuple[List[PageResult], int]:
        """Crawl deeper from top pages. Returns (new_pages, count)."""
        # Select top seeds
        seeds = seed_pages[:self.config.top_seeds]
        if not seeds:
            return [], 0

        print(f"\n=== Hybrid Expansion ===")
        print(f"Expanding from {len(seeds)} seed pages...")

        new_pages: Dict[str, PageResult] = {}

        for seed in seeds:
            if self.verbose:
                print(f"  Seed: {seed.url}")

            # Create focused crawler
            focused_config = CrawlConfig(
                whitelisted_domains=self.crawl_config.whitelisted_domains,
                max_pages_total=self.config.expansion_max_pages_per_seed,
                max_pages_per_domain=self.config.expansion_max_pages_per_seed,
                max_depth=self.config.expansion_depth,
                politeness_delay=self.crawl_config.politeness_delay,
                timeout=self.crawl_config.timeout,
                max_retries=self.crawl_config.max_retries,
            )

            crawler = TaskGatedCrawler(
                task_query,
                focused_config,
                self.scoring_config,
                self.api_config,
                verbose=self.verbose
            )

            # Seed from this page
            crawler.frontier.push(seed.url, 0, seed.score)
            # Mark existing pages as seen
            for url in existing_pages:
                crawler.frontier.mark_seen(url)

            # Crawl
            results = await crawler.crawl()

            # Collect new pages
            for page in results:
                if page.url not in existing_pages and page.url not in new_pages:
                    new_pages[page.url] = page

        count = len(new_pages)
        print(f"Expansion added {count} new pages")

        return list(new_pages.values()), count



# Results Reporter ------------------------------------------

class ResultsReporter:
    """Print results to console."""

    @staticmethod
    def print_results(pages: List[PageResult], top_k: int = 10) -> None:
        print("\n" + "=" * 70)
        print(f"TOP {min(top_k, len(pages))} RESULTS")
        print("=" * 70)

        for i, page in enumerate(pages[:top_k], 1):
            print(f"\n{i}. [{page.score:.3f}] {page.url}")
            print(f"   Title: {page.title[:70]}...")
            excerpt = page.best_chunk[:200].replace("\n", " ")
            print(f"   Excerpt: \"{excerpt}...\"")

    @staticmethod
    def print_diagnostics(stats: CrawlStats, whitelist: List[str]) -> None:
        print("\n" + "=" * 70)
        print("DIAGNOSTICS")
        print("=" * 70)

        print(f"Total pages crawled: {stats.total_pages_crawled}")
        print(f"Max depth reached: {stats.max_depth_reached}")

        print("\nPages per domain:")
        sorted_domains = sorted(stats.pages_per_domain.items(), key=lambda x: -x[1])
        for domain, count in sorted_domains[:10]:
            print(f"  - {domain}: {count}")

        print(f"\nHybrid expansion: {'Triggered' if stats.hybrid_triggered else 'Not triggered'}")
        if stats.hybrid_triggered:
            print(f"  Pages added: {stats.hybrid_pages_added}")

        print(f"\nEmbedding model: {stats.embedding_model}")
        print(f"Thresholds: keep={stats.keep_threshold}, expand={stats.expand_threshold}")
        print(f"Whitelisted domains: {len(whitelist)}")




# Whitelist Builder ------------------------------------------

async def build_whitelist(
    seed_domain: str,
    top_n: int = 10,
    max_pages: int = 500,
    max_depth: int = 2,
    verbose: bool = False
) -> List[str]:
    """Run Stage 1 to get top N related domains."""
    print("=" * 70)
    print("Stage 1: Building Whitelist")
    print("=" * 70)
    print(f"Running related_domains on {seed_domain}...")

    # Configure crawl
    crawl_config = RDCrawlConfig(
        seed_domain=seed_domain,
        max_pages_total=max_pages,
        max_depth=max_depth,
    )

    # Run crawler
    crawler = RelatedDomainsCrawler(crawl_config, TemplateFilterConfig(), verbose=verbose)
    graph = await crawler.crawl()

    # Run PPR
    ppr = PersonalizedPageRank(PPRConfig())
    scores, host_to_index, convergence = ppr.compute(graph)

    # Get top N domains
    index_to_host = {i: h for h, i in host_to_index.items()}
    ranked_indices = np.argsort(-scores)

    whitelist = []
    for idx in ranked_indices[:top_n]:
        host = index_to_host[idx]
        whitelist.append(host)

    print(f"\nTop {top_n} domains by PPR:")
    for i, host in enumerate(whitelist, 1):
        score = scores[host_to_index[host]]
        print(f"  {i}. {host} ({score:.4f})")

    return whitelist



# CLI and Main ------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task-Relevant Pages Agent - Find pages matching a task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python task_pages.py --seed cmu.edu --task "Find tuition fees"
    python task_pages.py --seed stanford.edu --task "PhD requirements" --top-k 20
    python task_pages.py --seed mit.edu --task "Faculty directory" --verbose
        """
    )

    # Required
    parser.add_argument(
        "--seed",
        required=True,
        help="Seed domain for whitelist building"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Natural language task/query"
    )

    # Whitelist building
    wl_group = parser.add_argument_group("Whitelist Building")
    wl_group.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Use top N domains from PPR (default: 10)"
    )
    wl_group.add_argument(
        "--whitelist-pages",
        type=int,
        default=100,
        help="Max pages for whitelist crawl (default: 500)"
    )
    wl_group.add_argument(
        "--whitelist-depth",
        type=int,
        default=2,
        help="Max depth for whitelist crawl (default: 2)"
    )

    # API
    api_group = parser.add_argument_group("API")
    api_group.add_argument(
        "--api-provider",
        choices=["openai"],
        default="openai",
        help="API provider for embeddings (only openai supported)"
    )
    api_group.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model (default: text-embedding-3-small)"
    )

    # Crawling
    crawl_group = parser.add_argument_group("Crawling")
    crawl_group.add_argument(
        "--max-pages-total",
        type=int,
        default=200,
        help="Maximum pages to crawl (default: 1000)"
    )
    crawl_group.add_argument(
        "--max-pages-per-domain",
        type=int,
        default=20,
        help="Max pages per domain (default: 200)"
    )
    crawl_group.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum crawl depth (default: 4)"
    )

    # Scoring
    score_group = parser.add_argument_group("Scoring")
    score_group.add_argument(
        "--keep-threshold",
        type=float,
        default=0.3,
        help="Score threshold for candidate pages (default: 0.4)"
    )
    score_group.add_argument(
        "--expand-threshold",
        type=float,
        default=0.15,
        help="Score threshold for link expansion (default: 0.2, higher = more selective)"
    )

    # Hybrid
    hybrid_group = parser.add_argument_group("Hybrid Expansion")
    hybrid_group.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Trigger expansion if top-1 < this (default: 0.6)"
    )
    hybrid_group.add_argument(
        "--expansion-depth",
        type=int,
        default=3,
        help="Depth for hybrid expansion (default: 3)"
    )
    hybrid_group.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to show (default: 10)"
    )

    # Output
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    # Build whitelist (Stage 1)
    whitelist = await build_whitelist(
        seed_domain=args.seed,
        top_n=args.top_n,
        max_pages=args.whitelist_pages,
        max_depth=args.whitelist_depth,
        verbose=args.verbose
    )

    if not whitelist:
        print("Error: No domains in whitelist. Exiting.")
        return

    # Configure Stage 2
    api_config = APIConfig(
        provider=args.api_provider,
        embedding_model=args.embedding_model
    )

    scoring_config = ScoringConfig(
        keep_threshold=args.keep_threshold,
        expand_threshold=args.expand_threshold
    )

    crawl_config = CrawlConfig(
        whitelisted_domains=whitelist,
        max_pages_total=args.max_pages_total,
        max_pages_per_domain=args.max_pages_per_domain,
        max_depth=args.max_depth
    )

    hybrid_config = HybridConfig(
        similarity_threshold=args.similarity_threshold,
        expansion_depth=args.expansion_depth
    )

    # Run Stage 2
    print("\n" + "=" * 70)
    print("Stage 2: Task-Relevant Pages")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Keywords: {extract_task_keywords(args.task)}")
    print()

    crawler = TaskGatedCrawler(
        args.task,
        crawl_config,
        scoring_config,
        api_config,
        verbose=args.verbose
    )

    results = await crawler.crawl()

    # Check for hybrid expansion
    stats = CrawlStats(
        total_pages_crawled=crawler.total_crawled,
        pages_per_domain=dict(crawler.pages_per_domain),
        max_depth_reached=crawler.max_depth_reached,
        embedding_model=args.embedding_model,
        keep_threshold=args.keep_threshold,
        expand_threshold=args.expand_threshold
    )

    if results:
        top_score = results[0].score
        expander = HybridExpander(
            hybrid_config, crawl_config, scoring_config, api_config, args.verbose
        )

        if expander.should_trigger(top_score):
            stats.hybrid_triggered = True
            new_pages, count = await expander.expand(
                results,
                crawler.pages,
                args.task
            )
            stats.hybrid_pages_added = count

            # Merge and re-rank
            all_pages = list(crawler.pages.values()) + new_pages
            results = sorted(all_pages, key=lambda p: -p.score)
            stats.total_pages_crawled += count

    # Print results
    ResultsReporter.print_results(results, args.top_k)
    ResultsReporter.print_diagnostics(stats, whitelist)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
