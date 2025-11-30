from __future__ import annotations

import argparse
from typing import List

import numpy as np

from crawler import crawl_website
from graph_builder import (
    build_adjacency_matrix,
    build_node_index,
    normalize_adjacency_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl a site and build PageRank inputs.")
    parser.add_argument("--url", required=True, help="Root URL to crawl (e.g., https://example.com)")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Maximum number of pages to visit (default: 200)",
    )
    return parser.parse_args()


def save_nodes(node_order: List[str]) -> None:
    with open("nodes.csv", "w", encoding="utf-8") as f:
        f.write("index,url\n")
        for idx, url in enumerate(node_order):
            f.write(f"{idx},{url}\n")


def main() -> None:
    args = parse_args()

    links = crawl_website(args.url, max_pages=args.max_pages)
    node_index = build_node_index(links)
    adjacency_matrix = build_adjacency_matrix(links, node_index)
    normalized_matrix = normalize_adjacency_matrix(adjacency_matrix)

    ordered_nodes = [url for url, _ in sorted(node_index.items(), key=lambda kv: kv[1])]
    np.savetxt("adjacency_matrix.csv", adjacency_matrix, fmt="%d", delimiter=",")
    np.savetxt(
        "adjacency_matrix_normalized.csv",
        normalized_matrix,
        fmt="%.6f",
        delimiter=",",
    )
    save_nodes(ordered_nodes)

    edge_count = int(adjacency_matrix.sum())
    sample_nodes = ordered_nodes[:5]

    print(f"Crawled {len(ordered_nodes)} pages (limit {args.max_pages}).")
    print(f"Edges: {edge_count}")
    if sample_nodes:
        print("Sample nodes:")
        for node in sample_nodes:
            print(f" - {node}")


if __name__ == "__main__":
    main()
