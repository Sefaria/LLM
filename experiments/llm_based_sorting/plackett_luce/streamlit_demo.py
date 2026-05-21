from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from experiments.llm_based_sorting.plackett_luce.active_learning import (
    RankingActiveLearner,
    sample_plackett_luce_score_posterior,
)


@dataclass
class DemoConfig:
    num_items: int
    k: int
    warm_start_repeats: int
    batch_size: int
    direct_active_iterations: int
    posterior_samples: int
    posterior_burn_in: int
    posterior_thinning: int
    posterior_seed: int
    oracle_type: str
    oracle_seed: int


def config_signature(config: DemoConfig) -> tuple[Any, ...]:
    return (
        config.num_items,
        config.k,
        config.warm_start_repeats,
        config.batch_size,
        config.direct_active_iterations,
        config.posterior_samples,
        config.posterior_burn_in,
        config.posterior_thinning,
        config.posterior_seed,
        config.oracle_type,
        config.oracle_seed,
    )


def build_true_scores(num_items: int) -> np.ndarray:
    """
    Build a score landscape with broad ordering plus local clusters/near-ties.
    """
    base = np.linspace(1.8, -1.8, num_items)
    wave = 0.35 * np.sin(np.linspace(0.0, 3.5 * np.pi, num_items))
    cluster_offsets = np.array([0.18 * ((index % 4) - 1.5) for index in range(num_items)], dtype=float)
    scores = base + wave + cluster_offsets
    return scores


def sample_pl_ranking(items: list[int], score_lookup: dict[int, float], rng: random.Random) -> list[int]:
    remaining = list(items)
    ranking: list[int] = []
    while remaining:
        weights = [math.exp(score_lookup[item]) for item in remaining]
        chosen = rng.choices(remaining, weights=weights, k=1)[0]
        ranking.append(chosen)
        remaining.remove(chosen)
    return ranking


def build_experiment_runner(
    score_lookup: dict[int, float],
    oracle_type: str,
    oracle_seed: int,
):
    rng = random.Random(oracle_seed)

    def run_experiment(items: list[int]) -> list[int]:
        if oracle_type == "Deterministic":
            return sorted(items, key=lambda item: score_lookup[item], reverse=True)
        return sample_pl_ranking(items, score_lookup, rng)

    return run_experiment


def build_posterior_sampler(config: DemoConfig, items: list[int]):
    item_to_index = {item: index for index, item in enumerate(items)}

    def posterior_sampler(observations):
        return sample_plackett_luce_score_posterior(
            observations,
            item_to_index,
            num_samples=config.posterior_samples,
            burn_in=config.posterior_burn_in,
            thinning=config.posterior_thinning,
            seed=config.posterior_seed,
        )

    return posterior_sampler


def create_learner(config: DemoConfig) -> RankingActiveLearner:
    items = list(range(config.num_items))
    true_scores = build_true_scores(config.num_items)
    score_lookup = {item: float(true_scores[item]) for item in items}
    posterior_sampler = build_posterior_sampler(config, items)
    experiment_runner = build_experiment_runner(score_lookup, config.oracle_type, config.oracle_seed)

    learner = RankingActiveLearner(
        items=items,
        K=config.k,
        posterior_sampler=posterior_sampler,
        experiment_runner=experiment_runner,
        warm_start_repeats=config.warm_start_repeats,
        batch_size=config.batch_size,
        direct_active_iterations=config.direct_active_iterations,
        random_seed=config.posterior_seed,
    )
    return learner


def initialize_session(config: DemoConfig) -> None:
    learner = create_learner(config)
    st.session_state.demo_config_signature = config_signature(config)
    st.session_state.learner = learner
    st.session_state.true_scores = build_true_scores(config.num_items)
    st.session_state.last_observations = []
    st.session_state.last_utility_matrix = np.zeros((config.num_items, config.num_items), dtype=float)
    st.session_state.last_tree = nx.Graph()
    st.session_state.last_batch_candidates = []


def ensure_session(config: DemoConfig) -> RankingActiveLearner:
    current_signature = st.session_state.get("demo_config_signature")
    if current_signature != config_signature(config) or "learner" not in st.session_state:
        initialize_session(config)
    return st.session_state.learner


def update_visual_state(learner: RankingActiveLearner) -> None:
    if learner.posterior_samples is None:
        learner.update_posterior()
    utility_matrix = learner.compute_pairwise_utilities(learner.posterior_samples)
    tree = learner.build_mst(utility_matrix)
    batch_candidates = learner.select_batch_from_mst(utility_matrix)

    st.session_state.last_utility_matrix = utility_matrix
    st.session_state.last_tree = tree
    st.session_state.last_batch_candidates = batch_candidates


def run_steps(step_count: int) -> None:
    learner: RankingActiveLearner = st.session_state.learner
    for _ in range(step_count):
        st.session_state.last_observations = learner.run_iteration()
    update_visual_state(learner)


def posterior_summary_frame(learner: RankingActiveLearner, true_scores: np.ndarray) -> pd.DataFrame:
    if learner.posterior_samples is None:
        learner.update_posterior()
    posterior_means = learner.posterior_samples.mean(axis=0)
    posterior_stds = learner.posterior_samples.std(axis=0)

    frame = pd.DataFrame(
        {
            "item": learner.items,
            "true_score": true_scores,
            "posterior_mean": posterior_means,
            "posterior_std": posterior_stds,
            "exposures": [learner.exposure_counts[item] for item in learner.items],
        }
    )
    frame["true_rank"] = frame["true_score"].rank(ascending=False, method="dense").astype(int)
    frame["posterior_rank"] = frame["posterior_mean"].rank(ascending=False, method="dense").astype(int)
    return frame.sort_values("item", ascending=True).reset_index(drop=True)


def plot_posterior_summary(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x_positions = np.arange(len(frame))
    ax.errorbar(
        x_positions,
        frame["posterior_mean"],
        yerr=2.0 * frame["posterior_std"],
        fmt="o",
        color="#1b9e77",
        ecolor="#1b9e77",
        capsize=4,
        label="Posterior mean +/- 2 std",
    )
    ax.scatter(
        x_positions,
        frame["true_score"],
        color="#d95f02",
        marker="x",
        s=70,
        label="True score",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(item) for item in frame["item"]])
    ax.set_xlabel("Item")
    ax.set_ylabel("Score")
    ax.set_title("Posterior Summary vs True Scores")
    ax.legend()
    fig.tight_layout()
    return fig


def posterior_ranked_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("posterior_mean", ascending=False).reset_index(drop=True)


def plot_heatmap(matrix: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Item")
    ax.set_ylabel("Item")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_tree(tree: nx.Graph, frame: pd.DataFrame, highlighted_sets: list[tuple[int, ...]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if tree.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No MST available yet", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    positions = nx.spring_layout(tree, seed=0, weight="weight")
    highlighted_nodes = set()
    if highlighted_sets:
        highlighted_nodes.update(highlighted_sets[0])

    node_colors = []
    for node in tree.nodes:
        if node in highlighted_nodes:
            node_colors.append("#d95f02")
        else:
            node_colors.append("#1b9e77")

    labels = {
        row["item"]: f'{int(row["item"])}\nmu={row["posterior_mean"]:.2f}'
        for _, row in frame.iterrows()
    }
    edge_labels = {(u, v): f'{data["weight"]:.2f}' for u, v, data in tree.edges(data=True)}

    nx.draw_networkx(tree, pos=positions, ax=ax, node_color=node_colors, with_labels=False)
    nx.draw_networkx_labels(tree, pos=positions, labels=labels, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(tree, pos=positions, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_title("Maximum Spanning Tree of Pairwise Utilities")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def observations_frame(observations: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for observation in observations:
        rows.append(
            {
                "phase": observation["metadata"]["phase"],
                "iteration": observation["metadata"]["iteration"],
                "items": observation["items"],
                "ranking": observation["ranking"],
            }
        )
    return pd.DataFrame(rows)


def render_sidebar() -> DemoConfig:
    st.sidebar.header("Controls")
    num_items = st.sidebar.slider("Number of items", min_value=5, max_value=40, value=16, step=1)
    k = st.sidebar.slider("Items per experiment (K)", min_value=2, max_value=min(6, num_items), value=3, step=1)
    warm_start_repeats = st.sidebar.slider("Warm-start repeats", min_value=1, max_value=5, value=2, step=1)
    batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=8, value=3, step=1)
    direct_active_iterations = st.sidebar.slider("Direct-active rounds", min_value=0, max_value=5, value=1, step=1)
    posterior_samples = st.sidebar.slider("Posterior samples", min_value=20, max_value=400, value=120, step=20)
    posterior_burn_in = st.sidebar.slider("Burn-in", min_value=0, max_value=300, value=60, step=10)
    posterior_thinning = st.sidebar.slider("Thinning", min_value=1, max_value=10, value=2, step=1)
    posterior_seed = st.sidebar.number_input("Posterior seed", min_value=0, value=0, step=1)
    oracle_type = st.sidebar.selectbox("Oracle", options=["Deterministic", "Noisy Plackett-Luce"])
    oracle_seed = st.sidebar.number_input("Oracle seed", min_value=0, value=0, step=1)

    return DemoConfig(
        num_items=num_items,
        k=k,
        warm_start_repeats=warm_start_repeats,
        batch_size=batch_size,
        direct_active_iterations=direct_active_iterations,
        posterior_samples=posterior_samples,
        posterior_burn_in=posterior_burn_in,
        posterior_thinning=posterior_thinning,
        posterior_seed=posterior_seed,
        oracle_type=oracle_type,
        oracle_seed=oracle_seed,
    )


def main() -> None:
    st.set_page_config(page_title="Plackett-Luce Active Learning Demo", layout="wide")
    st.title("Plackett-Luce Active Learning Demo")
    st.write(
        "Interactive synthetic demo for warm start, direct active selection, "
        "and MST-based batching under a Plackett-Luce posterior sampler."
    )

    config = render_sidebar()
    learner = ensure_session(config)
    if learner.posterior_samples is None:
        update_visual_state(learner)

    controls_col1, controls_col2, controls_col3, controls_col4 = st.columns([1, 1, 1, 2])
    with controls_col1:
        if st.button("Reset demo", use_container_width=True):
            initialize_session(config)
            learner = st.session_state.learner
            update_visual_state(learner)
    with controls_col2:
        if st.button("Run next iteration", use_container_width=True):
            run_steps(1)
            learner = st.session_state.learner
    with controls_col3:
        if st.button("Run 5 iterations", use_container_width=True):
            run_steps(5)
            learner = st.session_state.learner
    with controls_col4:
        st.caption(
            "The oracle is synthetic. Deterministic sorts by hidden truth; "
            "Noisy Plackett-Luce samples rankings from hidden scores."
        )

    true_scores = st.session_state.true_scores
    frame = posterior_summary_frame(learner, true_scores)
    ranked_frame = posterior_ranked_frame(frame)
    utility_matrix = st.session_state.last_utility_matrix
    tree = st.session_state.last_tree
    candidate_sets = st.session_state.last_batch_candidates

    current_phase = "warm_start" if learner.needs_warm_start() else (
        "direct_active" if learner._direct_active_completed < learner.direct_active_iterations else "mst_batch"
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Iteration", learner.iteration)
    metric_col2.metric("Observations", len(learner.observations))
    metric_col3.metric("Current phase", current_phase)
    metric_col4.metric("Min exposure", min(learner.exposure_counts.values()))

    chart_col1, chart_col2 = st.columns([3, 2])
    with chart_col1:
        st.pyplot(plot_posterior_summary(frame), use_container_width=True)
    with chart_col2:
        st.dataframe(ranked_frame, use_container_width=True, hide_index=True)

    lower_col1, lower_col2 = st.columns(2)
    with lower_col1:
        st.pyplot(plot_heatmap(utility_matrix, "Pairwise Mutual Information"), use_container_width=True)
    with lower_col2:
        st.pyplot(plot_tree(tree, frame, candidate_sets), use_container_width=True)

    st.subheader("Selected candidate K-sets from MST")
    if candidate_sets:
        candidate_table = pd.DataFrame(
            {
                "rank": list(range(1, len(candidate_sets) + 1)),
                "candidate_set": [list(candidate) for candidate in candidate_sets],
            }
        )
        st.dataframe(candidate_table, use_container_width=True, hide_index=True)
    else:
        st.write("No candidate sets available yet.")

    st.subheader("Latest observations")
    latest_observations = st.session_state.last_observations
    if latest_observations:
        st.dataframe(observations_frame(latest_observations), use_container_width=True, hide_index=True)
    else:
        st.write("No iterations have been run yet.")

    st.subheader("Full observation history")
    if learner.observations:
        st.dataframe(observations_frame(learner.observations), use_container_width=True, hide_index=True)
    else:
        st.write("Observation history is empty.")


if __name__ == "__main__":
    main()
