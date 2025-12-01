from __future__ import annotations

from crawler import crawl_website
from markov import build_markov_matrix, compute_pagerank, rank_pages


# ----------------------- Main driver ----------------------- #
def main(entry_url: str, goals: Optional[List[str]] = None) -> None:
    graph, url_to_index, index_to_url = crawl_website(entry_url)
    titles = getattr(crawl_website, "titles", {})
    snippets = getattr(crawl_website, "snippets", {})

    markov = build_markov_matrix(graph, url_to_index)
    pagerank = compute_pagerank(markov)

    print(f"Crawled {len(url_to_index)} pages. Ready for ranking queries.")

    def _rank_and_print(goal: str) -> None:
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
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <entry_url> [goal1 goal2 ...]")
        sys.exit(1)

    entry = sys.argv[1]
    goals = sys.argv[2:]
    main(entry, goals if goals else None)
