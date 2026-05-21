from __future__ import annotations

import itertools
import math
import random
from collections import deque
from math import comb
from typing import Any, Callable, Hashable, Sequence

import networkx as nx
import numpy as np
from scipy.special import expit

from experiments.llm_based_sorting.plackett_luce.bayesian_inference import gibbs_step


ItemId = Hashable
Observation = dict[str, Any]
PosteriorSampler = Callable[[Sequence[Observation]], np.ndarray]
ExperimentRunner = Callable[[list[ItemId]], Sequence[ItemId]]


def _binary_entropy(probabilities: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def observations_to_index_rankings(
    observations: Sequence[Observation],
    item_to_index: dict[ItemId, int],
) -> list[tuple[int, ...]]:
    rankings: list[tuple[int, ...]] = []
    for observation in observations:
        ranking = tuple(item_to_index[item] for item in observation["ranking"])
        rankings.append(ranking)
    return rankings


def sample_plackett_luce_score_posterior(
    observations: Sequence[Observation],
    item_to_index: dict[ItemId, int],
    *,
    num_samples: int = 200,
    burn_in: int = 100,
    thinning: int = 2,
    prior_shape: float = 1.5,
    prior_rate: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample posterior score draws for a Plackett-Luce model.

    The Gibbs sampler operates on positive skill parameters `lambda`. This
    adapter returns `log(lambda)` so downstream code can use logistic score
    differences directly.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thinning <= 0:
        raise ValueError("thinning must be positive.")

    num_items = len(item_to_index)
    rng = random.Random(seed)
    rankings = observations_to_index_rankings(observations, item_to_index)
    lambdas = [1.0] * num_items

    if not rankings:
        prior_samples = np.empty((num_samples, num_items), dtype=float)
        for sample_index in range(num_samples):
            lambda_sample = [
                rng.gammavariate(prior_shape, 1.0 / prior_rate)
                for _ in range(num_items)
            ]
            prior_samples[sample_index, :] = np.log(lambda_sample)
        return prior_samples

    total_steps = burn_in + num_samples * thinning
    collected = np.empty((num_samples, num_items), dtype=float)
    write_index = 0
    for step in range(total_steps):
        lambdas, _ = gibbs_step(
            rankings,
            lambdas,
            prior_shape=prior_shape,
            prior_rate=prior_rate,
            rng=rng,
        )
        if step < burn_in:
            continue
        if (step - burn_in) % thinning != 0:
            continue
        collected[write_index, :] = np.log(np.asarray(lambdas, dtype=float))
        write_index += 1

    return collected


class RankingActiveLearner:
    def __init__(
        self,
        items: Sequence[ItemId],
        K: int,
        posterior_sampler: PosteriorSampler,
        experiment_runner: ExperimentRunner,
        warm_start_repeats: int = 3,
        batch_size: int = 10,
        max_overlap: int | None = None,
        direct_active_iterations: int = 1,
        max_direct_candidates: int = 1000,
        random_seed: int = 0,
    ):
        if K < 2:
            raise ValueError("K must be at least 2.")
        if len(items) < K:
            raise ValueError("Need at least K items.")
        if warm_start_repeats < 0:
            raise ValueError("warm_start_repeats must be non-negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if direct_active_iterations < 0:
            raise ValueError("direct_active_iterations must be non-negative.")

        self.items = list(items)
        self.K = K
        self.posterior_sampler = posterior_sampler
        self.experiment_runner = experiment_runner
        self.warm_start_repeats = warm_start_repeats
        self.batch_size = batch_size
        self.max_overlap = K - 1 if max_overlap is None else max_overlap
        self.direct_active_iterations = direct_active_iterations
        self.max_direct_candidates = max_direct_candidates
        self.random = random.Random(random_seed)

        self.item_to_index = {item: index for index, item in enumerate(self.items)}
        if len(self.item_to_index) != len(self.items):
            raise ValueError("Items must be unique.")

        self.observations: list[Observation] = []
        self.exposure_counts = {item: 0 for item in self.items}
        self.posterior_samples: np.ndarray | None = None
        self.iteration = 0
        self._direct_active_completed = 0

    def run_iteration(self) -> list[Observation]:
        """
        Runs one active-learning iteration.
        """
        if self.needs_warm_start():
            phase = "warm_start"
            batch = [self.run_warm_start_iteration()]
        else:
            self.update_posterior()
            if self._direct_active_completed < self.direct_active_iterations:
                phase = "direct_active"
                batch = [self.select_direct_active_set(self.posterior_samples)]
                self._direct_active_completed += 1
            else:
                phase = "mst_batch"
                utility_matrix = self.compute_pairwise_utilities(self.posterior_samples)
                batch = self.select_batch_from_mst(utility_matrix)

        observations = self.run_experiments(batch, phase=phase)
        self.update_posterior()
        self.iteration += 1
        return observations

    def needs_warm_start(self) -> bool:
        return min(self.exposure_counts.values()) < self.warm_start_repeats

    def run_warm_start_iteration(self) -> tuple[int, ...]:
        """
        Runs one random/broad K-way experiment.
        """
        ranked_items = sorted(
            self.items,
            key=lambda item: (self.exposure_counts[item], self.random.random()),
        )
        selected = ranked_items[: self.K]
        return tuple(self.item_to_index[item] for item in selected)

    def update_posterior(self) -> np.ndarray:
        """
        Calls existing Plackett-Luce posterior sampler.
        """
        self.posterior_samples = self.posterior_sampler(self.observations)
        return self.posterior_samples

    def compute_pairwise_utilities(self, posterior_samples: np.ndarray) -> np.ndarray:
        """
        Returns U[i, j] matrix of pairwise mutual information scores.
        """
        if posterior_samples.ndim != 2:
            raise ValueError("posterior_samples must have shape (num_samples, num_items).")
        num_items = posterior_samples.shape[1]
        utilities = np.zeros((num_items, num_items), dtype=float)

        for i in range(num_items):
            for j in range(i + 1, num_items):
                probabilities = expit(posterior_samples[:, i] - posterior_samples[:, j])
                p_bar = probabilities.mean()
                utility = float(_binary_entropy(np.array([p_bar]))[0] - _binary_entropy(probabilities).mean())
                utilities[i, j] = utility
                utilities[j, i] = utility

        return utilities

    def build_mst(self, U: np.ndarray) -> nx.Graph:
        """
        Uses networkx.maximum_spanning_tree.
        """
        graph = nx.Graph()
        num_items = U.shape[0]
        graph.add_nodes_from(range(num_items))
        for i in range(num_items):
            for j in range(i + 1, num_items):
                graph.add_edge(i, j, weight=float(U[i, j]))
        return nx.maximum_spanning_tree(graph, weight="weight")

    def extract_k_neighborhoods(self, T: nx.Graph) -> list[tuple[int, ...]]:
        """
        Returns candidate K-item sets from MST neighborhoods.
        """
        candidates: set[tuple[int, ...]] = set()
        for start in T.nodes:
            queue: deque[int] = deque([start])
            seen: set[int] = set()
            ordered: list[int] = []

            while queue and len(ordered) < self.K:
                node = queue.popleft()
                if node in seen:
                    continue
                seen.add(node)
                ordered.append(node)
                for neighbor in sorted(T.neighbors(node)):
                    if neighbor not in seen:
                        queue.append(neighbor)

            if len(ordered) == self.K:
                candidates.add(tuple(sorted(ordered)))

        return sorted(candidates)

    def score_set_by_internal_mst(self, S: Sequence[int], U: np.ndarray) -> float:
        """
        Scores a K-set by max spanning tree over pairwise utilities inside S.
        """
        graph = nx.Graph()
        graph.add_nodes_from(S)
        for i, j in itertools.combinations(S, 2):
            graph.add_edge(i, j, weight=float(U[i, j]))
        tree = nx.maximum_spanning_tree(graph, weight="weight")
        return float(sum(data["weight"] for _, _, data in tree.edges(data=True)))

    def select_batch_from_mst(self, U: np.ndarray) -> list[tuple[int, ...]]:
        """
        Returns list of K-item sets for the next batch.
        """
        tree = self.build_mst(U)
        candidates = self.extract_k_neighborhoods(tree)
        scored = sorted(
            ((candidate, self.score_set_by_internal_mst(candidate, U)) for candidate in candidates),
            key=lambda pair: pair[1],
            reverse=True,
        )

        batch: list[tuple[int, ...]] = []
        for candidate, _ in scored:
            if self._is_acceptable_overlap(candidate, batch):
                batch.append(candidate)
            if len(batch) >= self.batch_size:
                break

        return batch

    def run_experiments(self, batch: Sequence[Sequence[int]], *, phase: str) -> list[Observation]:
        """
        Calls experiment_runner on each K-set and stores rankings.
        """
        new_observations: list[Observation] = []
        for candidate in batch:
            batch_items = [self.items[index] for index in candidate]
            ranking = list(self.experiment_runner(batch_items))
            if set(ranking) != set(batch_items) or len(ranking) != len(batch_items):
                raise ValueError("experiment_runner must return a full ranking of the provided items.")

            observation = {
                "items": batch_items,
                "ranking": ranking,
                "metadata": {
                    "phase": phase,
                    "iteration": self.iteration,
                },
            }
            self.observations.append(observation)
            new_observations.append(observation)
            for item in batch_items:
                self.exposure_counts[item] += 1

        return new_observations

    def select_direct_active_set(self, posterior_samples: np.ndarray | None) -> tuple[int, ...]:
        if posterior_samples is None:
            raise ValueError("Posterior samples must be available for direct active selection.")
        utility_matrix = self.compute_pairwise_utilities(posterior_samples)
        candidates = self._enumerate_or_sample_candidate_sets()
        scored = [
            (candidate, self.score_set_by_internal_mst(candidate, utility_matrix))
            for candidate in candidates
        ]
        best_candidate, _ = max(scored, key=lambda pair: pair[1])
        return best_candidate

    def _enumerate_or_sample_candidate_sets(self) -> list[tuple[int, ...]]:
        num_items = len(self.items)
        if comb(num_items, self.K) <= self.max_direct_candidates:
            return [tuple(candidate) for candidate in itertools.combinations(range(num_items), self.K)]

        candidates: set[tuple[int, ...]] = set()
        while len(candidates) < self.max_direct_candidates:
            draw = tuple(sorted(self.random.sample(range(num_items), self.K)))
            candidates.add(draw)
        return sorted(candidates)

    def _is_acceptable_overlap(
        self,
        candidate: Sequence[int],
        selected: Sequence[Sequence[int]],
    ) -> bool:
        candidate_set = set(candidate)
        for other in selected:
            if len(candidate_set.intersection(other)) > self.max_overlap:
                return False
        return True
