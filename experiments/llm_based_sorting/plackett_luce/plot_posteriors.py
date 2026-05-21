from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.llm_based_sorting.plackett_luce.bayesian_inference import gibbs_step


def build_demo_rankings() -> tuple[list[tuple[int, ...]], list[float]]:
    """
    Reuse the same small three-item structure as the tests, but pair it with a
    simple set of ground-truth skill values and enough repeated observations to
    yield visible posterior separation.
    """
    true_lambdas = [3.0, 1.5, 2.25]
    rankings = (
        [(0, 1, 2)] * 10
        + [(0, 2, 1)] * 8
        + [(2, 0, 1)] * 6
        + [(1, 0, 2)] * 3
    )
    return rankings, true_lambdas


def collect_posterior_samples(
    rankings: list[tuple[int, ...]],
    *,
    num_items: int,
    prior_shape: float,
    prior_rate: float,
    burn_in: int,
    num_samples: int,
    thinning: int,
    seed: int,
) -> list[list[float]]:
    rng = random.Random(seed)
    lambdas = [1.0] * num_items
    traces = [[] for _ in range(num_items)]

    total_steps = burn_in + num_samples * thinning
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
        for item_index, value in enumerate(lambdas):
            traces[item_index].append(value)

    return traces


def plot_posteriors(samples: list[list[float]], true_lambdas: list[float], output_path: Path) -> None:
    fig, axes = plt.subplots(len(samples), 1, figsize=(9, 8), sharex=True)
    if len(samples) == 1:
        axes = [axes]

    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    for item_index, (ax, item_samples) in enumerate(zip(axes, samples)):
        ax.hist(item_samples, bins=30, color=colors[item_index % len(colors)], alpha=0.85)
        mean_value = sum(item_samples) / len(item_samples)
        ax.axvline(mean_value, color="black", linestyle="--", linewidth=1)
        ax.axvline(true_lambdas[item_index], color="#c62828", linestyle="-", linewidth=1.5)
        ax.set_ylabel(f"Item {item_index}")
        ax.set_title(f"Posterior of lambda_{item_index}")
        ax.text(
            0.98,
            0.92,
            f"true={true_lambdas[item_index]:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    axes[-1].set_xlabel("Skill value")
    fig.suptitle("Plackett-Luce Posterior Samples from Gibbs Updates", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rankings, true_lambdas = build_demo_rankings()
    samples = collect_posterior_samples(
        rankings,
        num_items=3,
        prior_shape=1.5,
        prior_rate=0.5,
        burn_in=200,
        num_samples=1000,
        thinning=3,
        seed=0,
    )

    output_path = Path(__file__).resolve().parent / "plackett_luce_posterior_demo.png"
    plot_posteriors(samples, true_lambdas, output_path)
    print(f"Saved posterior plot to {output_path}")


if __name__ == "__main__":
    main()
