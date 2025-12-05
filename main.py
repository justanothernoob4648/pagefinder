from __future__ import annotations

import argparse
import os
from typing import List, Optional

from crawler import crawl_website, expand_frontier
from markov import (
    LLMSemanticRanker,
    build_markov_matrix,
    compute_pagerank,
    rank_pages,
)


def load_env_from_file(path: str = ".env") -> None:
    """Lightweight .env loader to set env vars if they aren't already set."""
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, val = stripped.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        return

#input: python main.py <url> --use-llm --hybrid-expand "help me log into my account"
# or without the goal in order to enter multiple tasks in the same run
# ----------------------- Main driver ----------------------- #
def main(
    entry_url: str,
    goals: Optional[List[str]] = None,
    use_llm: bool = False,
    llm_model: Optional[str] = None,
    pagerank_weight: float = 0.2,
    hybrid_expand: bool = False,
    hybrid_threshold: float = 0.3,  #when to trigger hybrid expansion
    hybrid_max_new: int = 5,        #how many new pages to fetch
    hybrid_frontier_k: int = 3,     #how many (top k promising) pages to use as the frontier for expansion
) -> None:
    # Load env vars (e.g., OPENAI_API_KEY) from .env if present.
    load_env_from_file()

    graph, url_to_index, index_to_url = crawl_website(entry_url)
    titles = getattr(crawl_website, "titles", {})
    snippets = getattr(crawl_website, "snippets", {})
    seen = getattr(crawl_website, "seen", set())
    root_netloc = getattr(crawl_website, "root_netloc", None)

    def _recompute_rankings() -> tuple:
        all_urls = set(graph.keys())
        for children in graph.values():
            all_urls.update(children)
        new_url_to_index = {url: idx for idx, url in enumerate(sorted(all_urls))}
        markov = build_markov_matrix(graph, new_url_to_index)
        new_pagerank = compute_pagerank(markov)
        return new_url_to_index, new_pagerank

    url_to_index, pagerank = _recompute_rankings()
    llm_ranker: Optional[LLMSemanticRanker] = None
    if use_llm:
        try:
            llm_ranker = LLMSemanticRanker(model=llm_model)
            print(
                f"LLM semantic ranking enabled (model={llm_ranker.model}, "
                f"pagerank_weight={pagerank_weight})."
            )
        except Exception as exc:
            print(f"LLM ranking unavailable: {exc}. Falling back to bag-of-words.")

    print(f"Crawled {len(url_to_index)} pages. Ready for ranking queries.")

    def _maybe_expand(goal: str, ranked: list[tuple[str, float]]) -> bool:
        nonlocal url_to_index, pagerank
        if not (hybrid_expand and llm_ranker):
            return False
        if not ranked:
            return False
        best_score = ranked[0][1]
        if best_score >= hybrid_threshold:
            return False

        frontier: List[str] = []
        for url, _ in ranked[:hybrid_frontier_k]:
            frontier.extend(graph.get(url, []))
        frontier = [u for u in frontier if u not in seen]
        if not frontier:
            return False

        print(f"[hybrid] Triggered (best_score={best_score:.4f} < {hybrid_threshold}); exploring frontier.")
        new_pages = expand_frontier(
            entry_url,
            frontier[:hybrid_max_new],
            graph,
            titles,
            snippets,
            seen,
            root_netloc=root_netloc,
            max_pages=hybrid_max_new,
        )
        if new_pages:
            url_to_index, pagerank = _recompute_rankings()
            print(
                f"[hybrid] Expanded frontier with {new_pages} pages (threshold={hybrid_threshold})."
            )
            return True
        return False

    def _rank_and_print(goal: str) -> None:
        if llm_ranker:
            try:
                ranked = llm_ranker.rank_pages(
                    goal, titles, snippets, pagerank, url_to_index, pagerank_weight
                )
            except Exception as exc:
                print(f"[llm] Ranking failed, falling back to bag-of-words: {exc}")
                ranked = rank_pages(goal, titles, snippets, pagerank, url_to_index)
        else:
            ranked = rank_pages(goal, titles, snippets, pagerank, url_to_index)

        if _maybe_expand(goal, ranked):
            if llm_ranker:
                try:
                    ranked = llm_ranker.rank_pages(
                        goal, titles, snippets, pagerank, url_to_index, pagerank_weight
                    )
                except Exception as exc:
                    print(f"[llm] Ranking failed after expand; fallback: {exc}")
                    ranked = rank_pages(goal, titles, snippets, pagerank, url_to_index)

        print(f"\nGoal: {goal}")
        print("Top 10 pages:")
        for url, score in ranked[:10]:
            title = titles.get(url, "")
            snippet = snippets.get(url, "")
            print(f"- {url} | score={score:.4f} | title={title} | snippet='{snippet[:100]}'")

    if goals:
        for goal in goals:
            _rank_and_print(goal)
    else:
        try:
            while True:
                goal = input("\nEnter goal (or blank to quit): ").strip()
                if not goal:
                    break
                _rank_and_print(goal)
        except (EOFError, KeyboardInterrupt):
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl a site and rank pages for a task/goal."
    )
    parser.add_argument("entry_url", help="Root URL to crawl.")
    parser.add_argument(
        "goals",
        nargs="*",
        help="Optional goals/tasks to rank immediately; otherwise prompts interactively.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use OpenAI embeddings to score semantic similarity.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override embedding model name (default: text-embedding-3-small).",
    )
    parser.add_argument(
        "--pagerank-weight",
        type=float,
        default=0.2,
        help="Blend weight for PageRank when using LLM similarity (0-1).",
    )
    parser.add_argument(
        "--hybrid-expand",
        action="store_true",
        help="If similarity is low, expand crawl from top candidates and retry ranking.",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=0.3,
        help="Trigger expansion if best score is below this value.",
    )
    parser.add_argument(
        "--hybrid-max-new",
        type=int,
        default=5,
        help="Maximum new pages to crawl during a hybrid expansion.",
    )
    parser.add_argument(
        "--hybrid-frontier-k",
        type=int,
        default=3,
        help="Use outbound links from top-k pages as the expansion frontier.",
    )
    args = parser.parse_args()

    main(
        args.entry_url,
        args.goals if args.goals else None,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        pagerank_weight=args.pagerank_weight,
        hybrid_expand=args.hybrid_expand,
        hybrid_threshold=args.hybrid_threshold,
        hybrid_max_new=args.hybrid_max_new,
        hybrid_frontier_k=args.hybrid_frontier_k,
    )
