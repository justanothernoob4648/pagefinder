from __future__ import annotations

from typing import Dict, Set

import numpy as np


def build_node_index(links: Dict[str, Set[str]]) -> Dict[str, int]:
    """
    Build a deterministic ordering of nodes.

    Returns a mapping from URL to its index in the sorted node list.
    """
    nodes = set(links.keys())
    for targets in links.values():
        nodes.update(targets)

    ordered_nodes = sorted(nodes)
    return {url: idx for idx, url in enumerate(ordered_nodes)}


def build_adjacency_matrix(
    links: Dict[str, Set[str]], node_index: Dict[str, int]
) -> np.ndarray:
    """
    Convert link mapping to an adjacency matrix.

    A[i][j] = 1 if node i links to node j, else 0.
    """
    size = len(node_index)
    adjacency = np.zeros((size, size), dtype=int)

    for source, targets in links.items():
        src_idx = node_index.get(source)
        if src_idx is None:
            continue
        for target in targets:
            tgt_idx = node_index.get(target)
            if tgt_idx is None:
                continue
            adjacency[src_idx, tgt_idx] = 1

    return adjacency


def normalize_adjacency_matrix(adjacency: np.ndarray) -> np.ndarray:
    """
    Normalize rows to create a row-stochastic matrix.

    Rows without outgoing links remain all zeros (dangling nodes).
    """
    row_sums = adjacency.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(adjacency, dtype=float)

    non_zero_rows = row_sums[:, 0] > 0
    normalized[non_zero_rows] = adjacency[non_zero_rows] / row_sums[non_zero_rows]

    return normalized
