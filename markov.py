from __future__ import annotations

from collections import defaultdict
import os
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
try:
    from openai import OpenAI
except ImportError:  # Allow code to run without OpenAI installed
    OpenAI = None


# ----------------------- Matrix + PageRank ----------------------- #
def build_markov_matrix(graph: Dict[str, List[str]], url_to_index: Dict[str, int]) -> np.ndarray:
    """Construct a column-stochastic Markov matrix from the crawl graph."""
    n = len(url_to_index)
    if n == 0:
        return np.zeros((0, 0))

    M = np.zeros((n, n), dtype=float)

    def _normalize_column(col_idx: int, outbound: Iterable[str]) -> None:
        targets = [url_to_index[t] for t in outbound if t in url_to_index]
        if not targets:     #if the current node is not outbound to anything
            M[:, col_idx] = 1.0 / n
            return
        weight = 1.0 / len(targets)
        for tgt in targets:
            M[tgt, col_idx] = weight

    for url, idx in url_to_index.items():
        outbound = graph.get(url, [])
        _normalize_column(idx, outbound)

    for col in range(n):    #normalize columns
        col_sum = M[:, col].sum()
        if col_sum == 0:
            M[:, col] = 1.0 / n
        else:
            M[:, col] /= col_sum

    return M


def compute_pagerank(M: np.ndarray, damping: float = 0.85, eps: float = 1e-6) -> np.ndarray:
    """Iteratively compute PageRank until convergence."""
    n = M.shape[0]
    if n == 0:
        return np.array([])

    rank = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - damping) / n)

    while True:
        prev = rank
        rank = teleport + damping * (M @ prev)
        if np.linalg.norm(rank - prev, ord=1) < eps or np.linalg.norm(rank - prev) < eps:
            break

    return rank


# ----------------------- Semantic ranking ----------------------- #

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "in",
    "of",
    "to",
    "at",
    "by",
    "on",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "this",
    "that",
    "as",
    "from",
    "about",
    "how",
    "what",
    "where",
    "when",
    "why",
    "who",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, keep alphanumerics, drop stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t and t not in STOPWORDS]


def _text_to_embedding(text: str) -> Dict[str, float]:
    """
    Bag-of-words with unigram and bigram frequencies.
    Bigrams help catch phrases like "tuition fee" or "financial aid".
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}

    freq: Dict[str, float] = defaultdict(float)
    for token in tokens:
        freq[token] += 1.0

    # Add bigrams with a slight boost
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i+1]}"
        freq[bigram] += 2.0

    total = float(sum(freq.values()))
    for token in list(freq.keys()):
        freq[token] /= total
    return freq


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = sum(v * v for v in vec_a.values()) ** 0.5
    norm_b = sum(v * v for v in vec_b.values()) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_pages(
    goal: str,
    titles: Dict[str, str],
    snippets: Dict[str, str],
    pagerank: np.ndarray,
    url_to_index: Dict[str, int],
) -> List[Tuple[str, float]]:
    """Combine PageRank and chunk-level semantic similarity to produce final scores."""
    goal_embed = _text_to_embedding(goal)
    scores: List[Tuple[str, float]] = []
    for url, idx in url_to_index.items():
        pr_score = float(pagerank[idx]) if idx < len(pagerank) else 0.0
        snippet = snippets.get(url) or titles.get(url, "")
        semantic_score = _cosine_similarity(goal_embed, _text_to_embedding(snippet))
        final_score = 0 * pr_score + 1 * semantic_score
        scores.append((url, final_score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


class LLMSemanticRanker:
    """
    Uses OpenAI embeddings to score how well each page matches a user goal.
    Provide OPENAI_API_KEY in the environment before enabling this.
    """

    def __init__(self, model: str | None = None) -> None:
        if OpenAI is None:
            raise ImportError("openai package not installed. Add it to requirements and reinstall.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment.")

        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        # Set a short timeout so we fail fast if networking is blocked.
        self.client = OpenAI(api_key=api_key, timeout=60)
        self._cache: Dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(1)
        if text in self._cache:
            return self._cache[text]
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=float)
        self._cache[text] = vec
        return vec

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def rank_pages(
        self,
        goal: str,
        titles: Dict[str, str],
        snippets: Dict[str, str],
        pagerank: np.ndarray,
        url_to_index: Dict[str, int],
        pagerank_weight: float = 0.01,
    ) -> List[Tuple[str, float]]:
        pagerank_weight = max(0.0, min(1.0, pagerank_weight))
        semantic_weight = 1.0 - pagerank_weight

        goal_embed = self._embed(goal)
        scores: List[Tuple[str, float]] = []
        for url, idx in url_to_index.items():
            pr_score = float(pagerank[idx]) if idx < len(pagerank) else 0.0
            text = snippets.get(url) or titles.get(url, "")
            page_embed = self._embed(text)
            semantic_score = self._cosine(goal_embed, page_embed)
            final_score = pagerank_weight * pr_score + semantic_weight * semantic_score
            scores.append((url, final_score))
        return sorted(scores, key=lambda item: item[1], reverse=True)
