from __future__ import annotations

import random

from experiments.llm_based_sorting.plackett_luce.bayesian_inference import (
    gibbs_step,
    posterior_skill_shape_rates,
    sample_latent_variables,
)


def test_sample_latent_variables_returns_one_positive_value_per_nonterminal_rank() -> None:
    rankings = [
        (0, 1, 2),
        (2, 0, 1),
        (1, 0),
    ]
    lambdas = [1.2, 0.8, 2.4]

    latent = sample_latent_variables(rankings, lambdas, rng=random.Random(0))

    assert len(latent) == len(rankings)
    assert [len(row) for row in latent] == [2, 2, 1]
    assert all(value > 0 for row in latent for value in row)


def test_posterior_skill_shape_rates_matches_closed_form_example() -> None:
    rankings = [
        (0, 1, 2),
        (2, 0, 1),
    ]
    latent = [
        [0.5, 1.0],
        [0.25, 0.75],
    ]

    shapes, rates = posterior_skill_shape_rates(
        rankings,
        latent,
        num_items=3,
        prior_shape=2.0,
        prior_rate=3.0,
    )

    assert shapes == [4.0, 3.0, 3.0]
    assert rates == [4.5, 5.5, 4.75]


def test_gibbs_step_returns_positive_skills_and_matching_latent_layout() -> None:
    rankings = [
        (0, 1, 2),
        (2, 0, 1),
    ]
    initial_lambdas = [1.0, 1.5, 2.0]

    updated_lambdas, latent = gibbs_step(
        rankings,
        initial_lambdas,
        prior_shape=1.5,
        prior_rate=0.5,
        rng=random.Random(0),
    )

    assert len(updated_lambdas) == 3
    assert all(value > 0 for value in updated_lambdas)
    assert [len(row) for row in latent] == [2, 2]
