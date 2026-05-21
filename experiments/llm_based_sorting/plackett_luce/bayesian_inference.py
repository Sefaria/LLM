"""
Scaffolding for Bayesian inference routines for the Plackett-Luce model.

Pseudocode summary from Caron and Doucet (2010), Section 4:

Given:
    rankings: n rankings, where ranking i is rho_i = (rho_i1, ..., rho_i p_i)
    lambdas: positive skill parameters lambda_1, ..., lambda_K
    prior: lambda_k ~ Gamma(shape=prior_shape, rate=prior_rate)

Latent-variable augmentation:
    For each ranking i and each rank position j in {1, ..., p_i - 1},
    introduce Z_ij ~ Exponential(rate = sum_{m=j}^{p_i} lambda_{rho_im})

Posterior conditional for lambda_k:
    shape_k = prior_shape + w_k
    rate_k = prior_rate + sum_{i=1}^n sum_{j=1}^{p_i - 1} delta_ijk * Z_ij

    where:
        w_k = number of rankings where item k is not placed last
        delta_ijk = 1 if item k is still available at stage j of ranking i,
                    else 0

One Gibbs step:
    1. Sample all latent Z_ij given current lambdas
    2. Compute posterior Gamma parameters for each lambda_k
    3. Sample each lambda_k independently from its Gamma conditional
"""

from __future__ import annotations

import random
from typing import Sequence


Ranking = Sequence[int]
LatentVariables = Sequence[Sequence[float]]


def _coerce_rng(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random.Random()


def _validate_rankings(rankings: Sequence[Ranking], num_items: int) -> None:
    for ranking in rankings:
        if len(ranking) < 2:
            raise ValueError("Each ranking must contain at least two items.")
        if len(set(ranking)) != len(ranking):
            raise ValueError("Each ranking must contain unique item ids.")
        for item in ranking:
            if item < 0 or item >= num_items:
                raise ValueError(f"Ranking item {item} is out of bounds for {num_items} items.")


def _validate_latent_layout(rankings: Sequence[Ranking], latent_variables: LatentVariables) -> None:
    if len(rankings) != len(latent_variables):
        raise ValueError("Each ranking must have a matching latent-variable row.")
    for ranking, row in zip(rankings, latent_variables):
        expected = len(ranking) - 1
        if len(row) != expected:
            raise ValueError(
                f"Ranking of length {len(ranking)} requires {expected} latent values; got {len(row)}."
            )


def sample_latent_variables(
    rankings: Sequence[Ranking],
    lambdas: Sequence[float],
    *,
    rng=None,
) -> list[list[float]]:
    """
    Sample one latent variable for each non-terminal rank position.

    Pseudocode:

    1. Initialize `latent = []`.
    2. For each ranking `rho_i` in `rankings`:
       a. Initialize `row = []`.
       b. For each position `j` from 0 to `len(rho_i) - 2`:
          i.   Compute the remaining-choice set
               `remaining = rho_i[j:]`.
          ii.  Compute the exponential rate
               `rate = sum(lambdas[item] for item in remaining)`.
          iii. Draw `z_ij ~ Exponential(rate=rate)` using `rng`.
          iv.  Append `z_ij` to `row`.
       c. Append `row` to `latent`.
    3. Return `latent`.

    Notes:
    - Each ranking of length `p_i` contributes exactly `p_i - 1` latent values.
    - Every sampled latent value must be strictly positive.
    """
    rng = _coerce_rng(rng)
    _validate_rankings(rankings, len(lambdas))

    if any(value <= 0 for value in lambdas):
        raise ValueError("All skill parameters must be strictly positive.")

    latent: list[list[float]] = []
    for ranking in rankings:
        row: list[float] = []
        for stage in range(len(ranking) - 1):
            rate = sum(lambdas[item] for item in ranking[stage:])
            row.append(rng.expovariate(rate))
        latent.append(row)
    return latent


def posterior_skill_shape_rates(
    rankings: Sequence[Ranking],
    latent_variables: LatentVariables,
    num_items: int,
    *,
    prior_shape: float = 1.0,
    prior_rate: float = 0.0,
) -> tuple[list[float], list[float]]:
    """
    Return Gamma posterior shape and rate parameters for each skill.

    Pseudocode:

    1. Initialize:
       `shapes = [prior_shape] * num_items`
       `rates = [prior_rate] * num_items`
    2. For each ranking `rho_i` and matching latent row `z_i`:
       a. Let `p_i = len(rho_i)`.
       b. For each stage `j` from 0 to `p_i - 2`:
          i.   Let `winner = rho_i[j]`.
          ii.  Increment the winner's shape count:
               `shapes[winner] += 1`
               This is the `w_k` contribution from the paper.
          iii. For every item in `rho_i[j:]`:
               add `z_i[j]` to that item's rate:
               `rates[item] += z_i[j]`
               This is the `delta_ijk * Z_ij` contribution.
    3. Return `(shapes, rates)`.

    Notes:
    - The last-ranked item in a ranking does not receive a shape increment from
      that ranking.
    - Items absent from a ranking receive no contribution from that ranking.
    """
    _validate_rankings(rankings, num_items)
    _validate_latent_layout(rankings, latent_variables)

    if prior_shape <= 0:
        raise ValueError("prior_shape must be strictly positive.")
    if prior_rate < 0:
        raise ValueError("prior_rate must be non-negative.")

    shapes = [float(prior_shape)] * num_items
    rates = [float(prior_rate)] * num_items

    for ranking, row in zip(rankings, latent_variables):
        for stage, z_value in enumerate(row):
            if z_value <= 0:
                raise ValueError("Latent variables must be strictly positive.")
            winner = ranking[stage]
            shapes[winner] += 1.0
            for item in ranking[stage:]:
                rates[item] += z_value

    return shapes, rates


def gibbs_step(
    rankings: Sequence[Ranking],
    lambdas: Sequence[float],
    *,
    prior_shape: float = 1.0,
    prior_rate: float = 0.0,
    rng=None,
) -> tuple[list[float], list[list[float]]]:
    """
    Perform one Gibbs update over latent variables and skill parameters.

    Pseudocode:

    1. Sample latent variables:
       `latent = sample_latent_variables(rankings, lambdas, rng=rng)`
    2. Compute posterior Gamma parameters:
       `shapes, rates = posterior_skill_shape_rates(
            rankings,
            latent,
            num_items=len(lambdas),
            prior_shape=prior_shape,
            prior_rate=prior_rate,
        )`
    3. For each item `k`:
       a. Sample
          `lambda_k_new ~ Gamma(shape=shapes[k], rate=rates[k])`
       b. Store the sampled value in `updated_lambdas[k]`.
    4. Return `(updated_lambdas, latent)`.

    Practical implementation note:
    - Python RNG APIs typically parameterize Gamma by shape and scale, not rate.
      If using `random.Random.gammavariate(alpha, beta)`, pass
      `alpha = shape` and `beta = 1.0 / rate`.
    """
    rng = _coerce_rng(rng)
    latent = sample_latent_variables(rankings, lambdas, rng=rng)
    shapes, rates = posterior_skill_shape_rates(
        rankings,
        latent,
        num_items=len(lambdas),
        prior_shape=prior_shape,
        prior_rate=prior_rate,
    )

    updated_lambdas: list[float] = []
    for shape, rate in zip(shapes, rates):
        if rate <= 0:
            raise ValueError(
                "Posterior Gamma rate must be strictly positive. "
                "Use a positive prior_rate or ensure each item receives data support."
            )
        updated_lambdas.append(rng.gammavariate(shape, 1.0 / rate))

    return updated_lambdas, latent
