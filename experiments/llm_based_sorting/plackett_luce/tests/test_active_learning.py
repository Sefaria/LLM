from __future__ import annotations

import numpy as np

from experiments.llm_based_sorting.plackett_luce.active_learning import (
    RankingActiveLearner,
    sample_plackett_luce_score_posterior,
)


def _mock_experiment_runner(items: list[int]) -> list[int]:
    hidden_scores = {
        0: 5.0,
        1: 4.0,
        2: 3.0,
        3: 2.0,
        4: 1.0,
    }
    return sorted(items, key=lambda item: hidden_scores[item], reverse=True)


def test_pairwise_utilities_are_symmetric_with_zero_diagonal() -> None:
    learner = RankingActiveLearner(
        items=[0, 1, 2],
        K=2,
        posterior_sampler=lambda _: np.zeros((3, 3)),
        experiment_runner=_mock_experiment_runner,
    )
    posterior_samples = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.8, 0.1, -0.9],
            [1.1, -0.2, -0.7],
        ]
    )

    utilities = learner.compute_pairwise_utilities(posterior_samples)

    assert utilities.shape == (3, 3)
    assert np.allclose(utilities, utilities.T)
    assert np.allclose(np.diag(utilities), 0.0)


def test_extract_k_neighborhoods_returns_unique_connected_candidates() -> None:
    learner = RankingActiveLearner(
        items=[0, 1, 2, 3],
        K=3,
        posterior_sampler=lambda _: np.zeros((3, 4)),
        experiment_runner=_mock_experiment_runner,
    )
    utility_matrix = np.array(
        [
            [0.0, 4.0, 0.5, 0.1],
            [4.0, 0.0, 3.0, 0.2],
            [0.5, 3.0, 0.0, 2.0],
            [0.1, 0.2, 2.0, 0.0],
        ]
    )

    tree = learner.build_mst(utility_matrix)
    neighborhoods = learner.extract_k_neighborhoods(tree)

    assert neighborhoods == [(0, 1, 2), (1, 2, 3)]


def test_run_iteration_warm_start_then_active_updates_observations_and_posterior() -> None:
    items = [0, 1, 2, 3, 4]

    def posterior_sampler(observations):
        return sample_plackett_luce_score_posterior(
            observations,
            {item: index for index, item in enumerate(items)},
            num_samples=20,
            burn_in=10,
            thinning=1,
            seed=0,
        )

    learner = RankingActiveLearner(
        items=items,
        K=3,
        posterior_sampler=posterior_sampler,
        experiment_runner=_mock_experiment_runner,
        warm_start_repeats=1,
        batch_size=2,
        direct_active_iterations=1,
        random_seed=0,
    )

    warm_start_rounds = 0
    while learner.needs_warm_start():
        observations = learner.run_iteration()
        assert observations[0]["metadata"]["phase"] == "warm_start"
        warm_start_rounds += 1

    assert warm_start_rounds >= 2
    assert min(learner.exposure_counts.values()) >= 1
    assert learner.posterior_samples is not None
    assert learner.posterior_samples.shape == (20, len(items))

    direct_active_observations = learner.run_iteration()
    assert direct_active_observations[0]["metadata"]["phase"] == "direct_active"

    mst_batch_observations = learner.run_iteration()
    assert all(observation["metadata"]["phase"] == "mst_batch" for observation in mst_batch_observations)
    assert 1 <= len(mst_batch_observations) <= learner.batch_size


def test_select_batch_from_mst_respects_overlap_constraint() -> None:
    learner = RankingActiveLearner(
        items=[0, 1, 2, 3, 4],
        K=3,
        posterior_sampler=lambda _: np.zeros((3, 5)),
        experiment_runner=_mock_experiment_runner,
        batch_size=3,
        max_overlap=1,
    )
    utility_matrix = np.array(
        [
            [0.0, 5.0, 4.0, 0.1, 0.1],
            [5.0, 0.0, 4.5, 0.2, 0.2],
            [4.0, 4.5, 0.0, 3.5, 3.0],
            [0.1, 0.2, 3.5, 0.0, 4.8],
            [0.1, 0.2, 3.0, 4.8, 0.0],
        ]
    )

    batch = learner.select_batch_from_mst(utility_matrix)

    for index, candidate in enumerate(batch):
        for other in batch[index + 1 :]:
            assert len(set(candidate).intersection(other)) <= 1
