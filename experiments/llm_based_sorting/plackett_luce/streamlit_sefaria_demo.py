from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from experiments.llm_based_sorting.plackett_luce.active_learning import RankingActiveLearner
from experiments.llm_based_sorting.plackett_luce.sefaria_ranking_pipeline import (
    SefariaQuotingCommentaryRankingConfig,
    build_claude_relevance_runner,
    build_posterior_sampler,
)
from experiments.llm_based_sorting.plackett_luce.sefaria_retrieval import (
    LocalSefariaProjectQuotingCommentaryRetriever,
    QuotingCommentaryItem,
    SefariaApiQuotingCommentaryRetriever,
)


DEFAULT_RELEVANCE_PROMPT = (
    "Rank these quoting-commentary passages by how interesting and engaging they are for a reader. "
    "Prefer passages that contain something surprising, vivid, narratively rich, emotionally striking, "
    "or especially memorable. Favor commentary that tells a story, introduces a dramatic image, sharpens "
    "a conflict, reveals an unexpected interpretation, or gives a striking insight that would make someone "
    "want to keep reading. Rank lower passages that are mainly technical, repetitive, dry, purely linguistic, "
    "or only weakly engaging even if they are informative."
)


@dataclass
class DemoConfig:
    ref: str
    relevance_prompt: str
    language: str
    retrieval_mode: str
    base_url: str
    sefaria_project_path: str
    k: int
    warm_start_repeats: int
    batch_size: int
    direct_active_iterations: int
    posterior_samples: int
    posterior_burn_in: int
    posterior_thinning: int
    posterior_seed: int
    anthropic_model: str
    anthropic_temperature: float
    anthropic_max_tokens: int
    run_iterations: int
    max_loaded_items: int
    item_sample_seed: int


def config_signature(config: DemoConfig) -> tuple[Any, ...]:
    return (
        config.ref,
        config.relevance_prompt,
        config.language,
        config.retrieval_mode,
        config.base_url,
        config.sefaria_project_path,
        config.k,
        config.warm_start_repeats,
        config.batch_size,
        config.direct_active_iterations,
        config.posterior_samples,
        config.posterior_burn_in,
        config.posterior_thinning,
        config.posterior_seed,
        config.anthropic_model,
        config.anthropic_temperature,
        config.anthropic_max_tokens,
        config.run_iterations,
        config.max_loaded_items,
        config.item_sample_seed,
    )


def render_sidebar() -> DemoConfig:
    st.sidebar.header("Controls")
    ref = st.sidebar.text_input("Ref", value="Genesis 1:1")
    relevance_prompt = st.sidebar.text_area("Relevance prompt", value=DEFAULT_RELEVANCE_PROMPT, height=180)
    language = st.sidebar.selectbox("Language", options=["en", "he"], index=0)
    retrieval_mode = st.sidebar.selectbox("Retrieval mode", options=["local_project", "api"], index=0)
    base_url = st.sidebar.text_input("Sefaria base URL", value="http://localhost:8000")
    sefaria_project_path = st.sidebar.text_input(
        "Sefaria project path",
        value="/Users/yon/projects/Sefaria-Project",
    )
    k = st.sidebar.slider("Items per experiment (K)", min_value=2, max_value=6, value=4, step=1)
    warm_start_repeats = st.sidebar.slider("Warm-start repeats", min_value=1, max_value=5, value=2, step=1)
    batch_size = st.sidebar.slider("MST batch size", min_value=1, max_value=8, value=3, step=1)
    direct_active_iterations = st.sidebar.slider("Direct-active rounds", min_value=0, max_value=5, value=1, step=1)
    posterior_samples = st.sidebar.slider("Posterior samples", min_value=20, max_value=400, value=120, step=20)
    posterior_burn_in = st.sidebar.slider("Burn-in", min_value=0, max_value=300, value=60, step=10)
    posterior_thinning = st.sidebar.slider("Thinning", min_value=1, max_value=10, value=2, step=1)
    posterior_seed = st.sidebar.number_input("Posterior seed", min_value=0, value=0, step=1)
    anthropic_model = st.sidebar.text_input("Anthropic model", value="claude-sonnet-4-6")
    anthropic_temperature = st.sidebar.number_input("Anthropic temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    anthropic_max_tokens = st.sidebar.number_input("Anthropic max tokens", min_value=128, max_value=4096, value=1024, step=128)
    run_iterations = st.sidebar.number_input("Run iterations per click", min_value=1, max_value=100, value=5, step=1)
    max_loaded_items = st.sidebar.number_input("Max loaded items (0 = all)", min_value=0, max_value=500, value=0, step=1)
    item_sample_seed = st.sidebar.number_input("Item sampling seed", min_value=0, value=0, step=1)

    return DemoConfig(
        ref=ref.strip(),
        relevance_prompt=relevance_prompt.strip(),
        language=language,
        retrieval_mode=retrieval_mode,
        base_url=base_url.strip(),
        sefaria_project_path=sefaria_project_path.strip(),
        k=k,
        warm_start_repeats=warm_start_repeats,
        batch_size=batch_size,
        direct_active_iterations=direct_active_iterations,
        posterior_samples=posterior_samples,
        posterior_burn_in=posterior_burn_in,
        posterior_thinning=posterior_thinning,
        posterior_seed=posterior_seed,
        anthropic_model=anthropic_model.strip(),
        anthropic_temperature=float(anthropic_temperature),
        anthropic_max_tokens=int(anthropic_max_tokens),
        run_iterations=int(run_iterations),
        max_loaded_items=int(max_loaded_items),
        item_sample_seed=int(item_sample_seed),
    )


def to_pipeline_config(config: DemoConfig) -> SefariaQuotingCommentaryRankingConfig:
    return SefariaQuotingCommentaryRankingConfig(
        ref=config.ref,
        relevance_prompt=config.relevance_prompt,
        language=config.language,
        retrieval_mode=config.retrieval_mode,
        base_url=config.base_url,
        sefaria_project_path=config.sefaria_project_path,
        k=config.k,
        warm_start_repeats=config.warm_start_repeats,
        batch_size=config.batch_size,
        direct_active_iterations=config.direct_active_iterations,
        posterior_samples=config.posterior_samples,
        posterior_burn_in=config.posterior_burn_in,
        posterior_thinning=config.posterior_thinning,
        posterior_seed=config.posterior_seed,
        anthropic_model=config.anthropic_model,
        anthropic_temperature=config.anthropic_temperature,
        anthropic_max_tokens=config.anthropic_max_tokens,
    )


def build_retriever(config: DemoConfig):
    if config.retrieval_mode == "api":
        return SefariaApiQuotingCommentaryRetriever(base_url=config.base_url)
    return LocalSefariaProjectQuotingCommentaryRetriever(
        sefaria_project_path=config.sefaria_project_path
    )


def build_item_table(items: list[QuotingCommentaryItem], *, language: str) -> pd.DataFrame:
    rows = []
    for item_id, item in enumerate(items):
        body, excerpt_language = item.render_with_fallback(language=language)
        excerpt = body.replace("\n", " ")
        if len(excerpt) > 180:
            excerpt = excerpt[:177] + "..."
        rows.append(
            {
                "item_id": item_id,
                "source_ref": item.source_ref,
                "anchor_ref": item.anchor_ref,
                "title": item.collective_title_en or item.index_title,
                "category": item.category,
                "excerpt_language": excerpt_language,
                "excerpt": excerpt,
            }
        )
    return pd.DataFrame(rows)


def create_learner(
    config: DemoConfig,
    items: list[QuotingCommentaryItem],
) -> RankingActiveLearner:
    item_ids = list(range(len(items)))
    posterior_sampler = build_posterior_sampler(
        item_ids,
        num_samples=config.posterior_samples,
        burn_in=config.posterior_burn_in,
        thinning=config.posterior_thinning,
        seed=config.posterior_seed,
    )
    experiment_runner = build_claude_relevance_runner(
        items,
        relevance_prompt=config.relevance_prompt,
        model=config.anthropic_model,
        temperature=config.anthropic_temperature,
        max_tokens=config.anthropic_max_tokens,
    )
    return RankingActiveLearner(
        items=item_ids,
        K=config.k,
        posterior_sampler=posterior_sampler,
        experiment_runner=experiment_runner,
        warm_start_repeats=config.warm_start_repeats,
        batch_size=config.batch_size,
        direct_active_iterations=config.direct_active_iterations,
        random_seed=config.posterior_seed,
    )


def maybe_sample_items(
    items: list[QuotingCommentaryItem],
    *,
    max_loaded_items: int,
    item_sample_seed: int,
) -> list[QuotingCommentaryItem]:
    if max_loaded_items <= 0 or len(items) <= max_loaded_items:
        return items

    rng = random.Random(item_sample_seed)
    sampled_indices = sorted(rng.sample(range(len(items)), max_loaded_items))
    return [items[index] for index in sampled_indices]


def initialize_session(config: DemoConfig, items: list[QuotingCommentaryItem]) -> RankingActiveLearner:
    learner = create_learner(config, items)
    st.session_state.sefaria_demo_config_signature = config_signature(config)
    st.session_state.sefaria_items = items
    st.session_state.sefaria_item_table = build_item_table(items, language=config.language)
    st.session_state.sefaria_total_retrieved_items = st.session_state.get("sefaria_total_retrieved_items", len(items))
    st.session_state.sefaria_learner = learner
    st.session_state.sefaria_last_observations = []
    st.session_state.sefaria_last_utility_matrix = np.zeros((len(items), len(items)), dtype=float)
    st.session_state.sefaria_last_tree = nx.Graph()
    st.session_state.sefaria_last_batch_candidates = []
    st.session_state.sefaria_last_error = None
    update_visual_state(learner)
    return learner


def load_ref(config: DemoConfig) -> RankingActiveLearner:
    if not config.ref:
        raise ValueError("Ref is required.")

    retriever = build_retriever(config)
    retrieved_items = retriever.fetch(config.ref)
    items = maybe_sample_items(
        retrieved_items,
        max_loaded_items=config.max_loaded_items,
        item_sample_seed=config.item_sample_seed,
    )
    if len(items) < config.k:
        raise ValueError(
            f"Retrieved only {len(items)} quoting-commentary items for {config.ref}. Need at least K={config.k}."
        )
    st.session_state.sefaria_total_retrieved_items = len(retrieved_items)
    return initialize_session(config, items)


def update_visual_state(learner: RankingActiveLearner) -> None:
    if learner.posterior_samples is None:
        learner.update_posterior()
    utility_matrix = learner.compute_pairwise_utilities(learner.posterior_samples)
    tree = learner.build_mst(utility_matrix)
    batch_candidates = learner.select_batch_from_mst(utility_matrix)

    st.session_state.sefaria_last_utility_matrix = utility_matrix
    st.session_state.sefaria_last_tree = tree
    st.session_state.sefaria_last_batch_candidates = batch_candidates


def run_steps(step_count: int) -> None:
    learner: RankingActiveLearner = st.session_state.sefaria_learner
    for _ in range(step_count):
        st.session_state.sefaria_last_observations = learner.run_iteration()
    update_visual_state(learner)


def current_phase(learner: RankingActiveLearner) -> str:
    if learner.needs_warm_start():
        return "warm_start"
    if learner._direct_active_completed < learner.direct_active_iterations:
        return "direct_active"
    return "mst_batch"


def posterior_summary_frame(
    learner: RankingActiveLearner,
    items: list[QuotingCommentaryItem],
    *,
    language: str,
) -> pd.DataFrame:
    if learner.posterior_samples is None:
        learner.update_posterior()
    posterior_means = learner.posterior_samples.mean(axis=0)
    posterior_stds = learner.posterior_samples.std(axis=0)

    rows = []
    for item_id, item in enumerate(items):
        body, excerpt_language = item.render_with_fallback(language=language)
        excerpt = body.replace("\n", " ")
        if len(excerpt) > 140:
            excerpt = excerpt[:137] + "..."
        rows.append(
            {
                "item_id": item_id,
                "title": item.collective_title_en or item.index_title,
                "source_ref": item.source_ref,
                "posterior_mean": float(posterior_means[item_id]),
                "posterior_std": float(posterior_stds[item_id]),
                "exposures": learner.exposure_counts[item_id],
                "excerpt_language": excerpt_language,
                "excerpt": excerpt,
            }
        )

    frame = pd.DataFrame(rows)
    frame["posterior_rank"] = frame["posterior_mean"].rank(ascending=False, method="dense").astype(int)
    return frame.sort_values("item_id", ascending=True).reset_index(drop=True)


def plot_posterior_summary(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.8))
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
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(item_id) for item_id in frame["item_id"]], rotation=0)
    ax.set_xlabel("Item")
    ax.set_ylabel("Score")
    ax.set_title("Posterior Summary")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_heatmap(matrix: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Item")
    ax.set_ylabel("Item")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_tree(
    tree: nx.Graph,
    frame: pd.DataFrame,
    highlighted_sets: list[tuple[int, ...]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if tree.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No MST available yet", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    positions = nx.spring_layout(tree, seed=0, weight="weight")
    highlighted_nodes = set(highlighted_sets[0]) if highlighted_sets else set()
    node_colors = ["#d95f02" if node in highlighted_nodes else "#1b9e77" for node in tree.nodes]
    labels = {
        int(row["item_id"]): f'{int(row["item_id"])}\nmu={row["posterior_mean"]:.2f}'
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


def observations_frame(observations: list[dict[str, Any]], items: list[QuotingCommentaryItem]) -> pd.DataFrame:
    rows = []
    for observation in observations:
        rows.append(
            {
                "phase": observation["metadata"]["phase"],
                "iteration": observation["metadata"]["iteration"],
                "items": observation["items"],
                "item_refs": [items[item_id].source_ref for item_id in observation["items"]],
                "ranking": observation["ranking"],
                "ranking_refs": [items[item_id].source_ref for item_id in observation["ranking"]],
            }
        )
    return pd.DataFrame(rows)


def candidate_sets_frame(
    candidate_sets: list[tuple[int, ...]],
    learner: RankingActiveLearner,
    items: list[QuotingCommentaryItem],
) -> pd.DataFrame:
    if not candidate_sets:
        return pd.DataFrame()

    utility_matrix = st.session_state.sefaria_last_utility_matrix
    rows = []
    for rank, candidate in enumerate(candidate_sets, start=1):
        rows.append(
            {
                "rank": rank,
                "candidate_set": list(candidate),
                "refs": [items[item_id].source_ref for item_id in candidate],
                "score": learner.score_set_by_internal_mst(candidate, utility_matrix),
            }
        )
    return pd.DataFrame(rows)


def render_item_texts(items: list[QuotingCommentaryItem], *, language: str) -> None:
    st.subheader("Item Texts")
    for item_id, item in enumerate(items):
        title = item.collective_title_en or item.index_title
        with st.expander(f"Item {item_id}: {title} | {item.source_ref}", expanded=False):
            body, displayed_language = item.render_with_fallback(language=language)
            st.caption(
                f"Anchor ref: {item.anchor_ref} | Category: {item.category} | Display language: {displayed_language}"
            )
            st.write(body or "[Empty text]")


def render_loaded_view(config: DemoConfig) -> None:
    learner: RankingActiveLearner = st.session_state.sefaria_learner
    items: list[QuotingCommentaryItem] = st.session_state.sefaria_items
    item_table: pd.DataFrame = st.session_state.sefaria_item_table
    frame = posterior_summary_frame(learner, items, language=config.language)
    ranked_frame = frame.sort_values("posterior_mean", ascending=False).reset_index(drop=True)
    utility_matrix = st.session_state.sefaria_last_utility_matrix
    tree = st.session_state.sefaria_last_tree
    candidate_sets = st.session_state.sefaria_last_batch_candidates

    total_llm_calls = len(learner.observations)
    total_retrieved_items = st.session_state.get("sefaria_total_retrieved_items", len(items))

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
    metric_col1.metric("Loaded items", len(items))
    metric_col2.metric("Retrieved items", total_retrieved_items)
    metric_col3.metric("Iteration", learner.iteration)
    metric_col4.metric("Observations", len(learner.observations))
    metric_col5.metric("Current phase", current_phase(learner))
    metric_col6.metric("LLM calls", total_llm_calls)

    metric_col7, metric_col8 = st.columns(2)
    metric_col7.metric("Min exposure", min(learner.exposure_counts.values()))
    metric_col8.metric("Anthropic key", "Present" if os.getenv("ANTHROPIC_API_KEY") else "Missing")

    st.subheader("Item Index Map")
    st.dataframe(item_table, use_container_width=True, hide_index=True)

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

    st.subheader("Selected Candidate K-Sets")
    candidate_table = candidate_sets_frame(candidate_sets, learner, items)
    if not candidate_table.empty:
        st.dataframe(candidate_table, use_container_width=True, hide_index=True)
    else:
        st.write("No candidate sets available yet.")

    st.subheader("Latest Observations")
    latest_observations = st.session_state.sefaria_last_observations
    if latest_observations:
        st.dataframe(observations_frame(latest_observations, items), use_container_width=True, hide_index=True)
    else:
        st.write("No iterations have been run yet.")

    st.subheader("Full Observation History")
    if learner.observations:
        st.dataframe(observations_frame(learner.observations, items), use_container_width=True, hide_index=True)
    else:
        st.write("Observation history is empty.")

    render_item_texts(items, language=config.language)


def main() -> None:
    st.set_page_config(page_title="Sefaria Quoting Commentary Active Learning", layout="wide")
    st.title("Sefaria Quoting Commentary Active Learning")
    st.write(
        "Load quoting-commentary passages for a Sefaria ref, rank them with the active-learning loop, "
        "and inspect the posterior, pairwise utility heatmap, MST, selected sets, and raw texts."
    )

    config = render_sidebar()
    config_key = config_signature(config)
    session_key = st.session_state.get("sefaria_demo_config_signature")
    config_changed = session_key is not None and session_key != config_key

    controls_col1, controls_col2, controls_col3, controls_col4 = st.columns([1, 1, 1, 2])
    with controls_col1:
        if st.button("Load ref", use_container_width=True):
            try:
                with st.spinner(f"Loading quoting commentary for {config.ref}..."):
                    load_ref(config)
            except Exception as exc:
                st.session_state.sefaria_last_error = str(exc)
    with controls_col2:
        if st.button("Reset learner", use_container_width=True):
            try:
                items = st.session_state.get("sefaria_items")
                if items is None or config_changed:
                    with st.spinner(f"Reloading quoting commentary for {config.ref}..."):
                        load_ref(config)
                else:
                    initialize_session(config, items)
            except Exception as exc:
                st.session_state.sefaria_last_error = str(exc)
    with controls_col3:
        run_disabled = "sefaria_learner" not in st.session_state
        if st.button("Run next iteration", use_container_width=True, disabled=run_disabled):
            try:
                with st.spinner("Running one active-learning iteration..."):
                    run_steps(1)
                    st.session_state.sefaria_last_error = None
            except Exception as exc:
                st.session_state.sefaria_last_error = str(exc)
    with controls_col4:
        run_disabled = "sefaria_learner" not in st.session_state
        if st.button(f"Run {config.run_iterations} iterations", use_container_width=True, disabled=run_disabled):
            try:
                with st.spinner(f"Running {config.run_iterations} active-learning iterations..."):
                    run_steps(config.run_iterations)
                    st.session_state.sefaria_last_error = None
            except Exception as exc:
                st.session_state.sefaria_last_error = str(exc)

    if config_changed:
        st.info("Sidebar settings changed. Click `Load ref` or `Reset learner` to apply them.")

    if st.session_state.get("sefaria_last_error"):
        st.error(st.session_state.sefaria_last_error)

    if "sefaria_learner" not in st.session_state:
        st.info(
            "Choose a ref and relevance prompt, then click `Load ref`. "
            "After loading, you can run the algorithm iteration by iteration."
        )
        return

    render_loaded_view(config)


if __name__ == "__main__":
    main()
