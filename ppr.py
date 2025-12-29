#!/usr/bin/env python3
"""Personalized PageRank algorithm implementation on domain graph."""

from dataclasses import dataclass
from typing import Dict, Set, Tuple

import numpy as np


@dataclass
class PPRConfig:
    """PageRank config"""
    damping: float = 0.85  # damping factor, or Pr(no restart)
    seed_weight: float = 0.75  # weight for e_seed in restart vector
    subdomain_weight: float = 0.25  # weight for u_subdomains in restart vector
    convergence_threshold: float = 1e-10  # L1 norm threshold for detecting convg.
    max_iterations: int = 200


@dataclass
class ConvergenceStats:
    """Data Class to store how well PPR converged"""
    iterations: int
    final_l1_diff: float
    converged: bool


class PersonalizedPageRank:
    """PageRank biased toward seed domain. Higher score = more related"""

    def __init__(self, config: PPRConfig):
        self.config = config

    def build_restart_vector(
        self,
        seed_host: str,
        subdomains: Set[str],
        host_to_index: Dict[str, int]
    ) -> np.ndarray:
        """Build restart vector v using weights in config."""
        n = len(host_to_index)
        v = np.zeros(n)

        seed_idx = host_to_index.get(seed_host)
        if seed_idx is None:
            # Safety: REQUIRES(seed in graph) - shld b true
            return np.ones(n) / n

        if not subdomains:
            # No subdomains: v = e_seed (all restart vec mass on seed)
            v[seed_idx] = 1.0
        else:
            # v = seed_weight * e_seed + subdomain_weight * u_subdomains
            v[seed_idx] = self.config.seed_weight

            # Filter to subdomains that exist in the graph
            valid_subdomains = [s for s in subdomains if s in host_to_index]
            if valid_subdomains:
                subdomain_mass = self.config.subdomain_weight / len(valid_subdomains)
                for subdomain in valid_subdomains:
                    v[host_to_index[subdomain]] += subdomain_mass
            else:
                # No valid subdomains: give all mass to seed
                v[seed_idx] += self.config.subdomain_weight

        # Ensure it sums to 1 (handle float)
        v = v / v.sum()
        return v

    def compute(
        self,
        graph,  # DomainGraph
    ) -> Tuple[np.ndarray, Dict[str, int], ConvergenceStats]:
        """Run PageRank via power iteration. Returns (scores, host_to_index, stats)."""
        # Build host to index map
        hosts = sorted(graph.nodes)
        host_to_index = {host: i for i, host in enumerate(hosts)}
        n = len(hosts)

        if n == 0:
            return np.array([]), {}, ConvergenceStats(0, 0.0, True)

        print(f"\nRunning Personalized PageRank on {n} nodes...")

        # Build restart vector using above func
        subdomains = graph.get_subdomains()
        v = self.build_restart_vector(graph.seed_host, subdomains, host_to_index)

        print(f"  Seed: {graph.seed_host}")
        print(f"  Subdomains in restart vector: {len(subdomains)}")
        print(f"  Damping factor: {self.config.damping}")

        # Build transition matrix (replace dangling nodes w/ restart vector v)
        M = graph.to_column_stochastic_matrix(host_to_index, v)

        # Power iteration
        x = v.copy()  # Start from restart distribution
        alpha = self.config.damping

        for iteration in range(self.config.max_iterations):
            x_new = alpha * (M @ x) + (1 - alpha) * v
            diff = np.abs(x_new - x).sum()  # L1 norm (sum of abs val differences of each element)
            x = x_new

            if diff < self.config.convergence_threshold:    # less than defined threshold
                stats = ConvergenceStats(iteration + 1, diff, True)
                print(f"  Converged after {iteration + 1} iterations (L1 diff: {diff:.2e})")
                return x, host_to_index, stats

        stats = ConvergenceStats(self.config.max_iterations, diff, False)
        print(f"  Did not converge after {self.config.max_iterations} iterations (L1 diff: {diff:.2e})")
        return x, host_to_index, stats
