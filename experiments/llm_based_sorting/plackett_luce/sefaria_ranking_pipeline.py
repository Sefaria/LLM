from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from experiments.llm_based_sorting.plackett_luce.active_learning import (
    RankingActiveLearner,
    sample_plackett_luce_score_posterior,
)
from experiments.llm_based_sorting.plackett_luce.experiment_runners import (
    ClaudeListwiseRankingExperimentRunner,
)
from experiments.llm_based_sorting.plackett_luce.sefaria_retrieval import (
    LocalSefariaProjectQuotingCommentaryRetriever,
    QuotingCommentaryItem,
    SefariaApiQuotingCommentaryRetriever,
)


@dataclass
class SefariaQuotingCommentaryRankingConfig:
    ref: str
    relevance_prompt: str
    language: str = "en"
    retrieval_mode: str = "local_project"
    base_url: str = "http://localhost:8000"
    sefaria_project_path: str = "/Users/yon/projects/Sefaria-Project"
    k: int = 4
    warm_start_repeats: int = 2
    batch_size: int = 3
    direct_active_iterations: int = 1
    total_iterations: int = 12
    posterior_samples: int = 160
    posterior_burn_in: int = 80
    posterior_thinning: int = 2
    posterior_seed: int = 0
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_temperature: float = 0.0
    anthropic_max_tokens: int = 1024
    output_path: str | None = None


def build_default_item_text(item: QuotingCommentaryItem, *, language: str) -> str:
    text, text_language = item.render_with_fallback(language=language)
    return "\n".join(
        [
            f"Source ref: {item.source_ref}",
            f"Anchor ref: {item.anchor_ref}",
            f"Title: {item.collective_title_en or item.index_title}",
            f"Text language: {text_language}",
            "Text:",
            text,
        ]
    )


def build_claude_relevance_runner(
    items: Sequence[QuotingCommentaryItem],
    *,
    relevance_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> ClaudeListwiseRankingExperimentRunner:
    item_lookup = {index: item for index, item in enumerate(items)}

    def render(item_id: int) -> str:
        return build_default_item_text(item_lookup[item_id], language="en")

    return ClaudeListwiseRankingExperimentRunner(
        item_renderer=render,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=(
            "You are a careful Jewish-text ranking judge. "
            "You will receive several quoting-commentary passages anchored to the same base ref. "
            "Rank them from most relevant to least relevant according to the user-provided criterion. "
            "Return only valid JSON."
        ),
        ranking_instruction=(
            "Relevance criterion:\n"
            f"{relevance_prompt.strip()}\n\n"
            "Return JSON of the form {\"ranking\": [0, 2, 1]} where each number is a Local index. "
            "Rank the items from best to worst. Use each Local index exactly once."
        ),
    )


def build_posterior_sampler(
    item_ids: Sequence[int],
    *,
    num_samples: int,
    burn_in: int,
    thinning: int,
    seed: int,
) -> Callable[[Sequence[dict[str, Any]]], np.ndarray]:
    item_to_index = {item_id: item_id for item_id in item_ids}

    def posterior_sampler(observations: Sequence[dict[str, Any]]) -> np.ndarray:
        return sample_plackett_luce_score_posterior(
            observations,
            item_to_index,
            num_samples=num_samples,
            burn_in=burn_in,
            thinning=thinning,
            seed=seed,
        )

    return posterior_sampler


def build_results(
    learner: RankingActiveLearner,
    items: Sequence[QuotingCommentaryItem],
    config: SefariaQuotingCommentaryRankingConfig,
) -> list[dict[str, Any]]:
    if learner.posterior_samples is None:
        learner.update_posterior()

    posterior_means = learner.posterior_samples.mean(axis=0)
    posterior_stds = learner.posterior_samples.std(axis=0)

    results: list[dict[str, Any]] = []
    for item_id, item in enumerate(items):
        result = {
            "item_id": item_id,
            "source_ref": item.source_ref,
            "anchor_ref": item.anchor_ref,
            "index_title": item.index_title,
            "collective_title_en": item.collective_title_en,
            "collective_title_he": item.collective_title_he,
            "category": item.category,
            "text": item.text,
            "he": item.he,
            "posterior_mean": float(posterior_means[item_id]),
            "posterior_std": float(posterior_stds[item_id]),
            "exposure_count": int(learner.exposure_counts[item_id]),
            "raw_link": item.raw_link,
        }
        results.append(result)

    results.sort(key=lambda row: row["posterior_mean"], reverse=True)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
        row["relevance_prompt"] = config.relevance_prompt
        row["ref"] = config.ref

    return results


def run_sefaria_quoting_commentary_ranking(
    config: SefariaQuotingCommentaryRankingConfig,
    *,
    retriever: Any | None = None,
) -> dict[str, Any]:
    if retriever is None:
        if config.retrieval_mode == "api":
            retriever = SefariaApiQuotingCommentaryRetriever(base_url=config.base_url)
        elif config.retrieval_mode == "local_project":
            retriever = LocalSefariaProjectQuotingCommentaryRetriever(
                sefaria_project_path=config.sefaria_project_path
            )
        else:
            raise ValueError("retrieval_mode must be one of: 'api', 'local_project'.")
    retrieved_items = retriever.fetch(config.ref)
    if len(retrieved_items) < config.k:
        raise ValueError(
            f"Retrieved only {len(retrieved_items)} quoting-commentary items for {config.ref}. "
            f"Need at least K={config.k}."
        )

    item_ids = list(range(len(retrieved_items)))
    posterior_sampler = build_posterior_sampler(
        item_ids,
        num_samples=config.posterior_samples,
        burn_in=config.posterior_burn_in,
        thinning=config.posterior_thinning,
        seed=config.posterior_seed,
    )
    experiment_runner = build_claude_relevance_runner(
        retrieved_items,
        relevance_prompt=config.relevance_prompt,
        model=config.anthropic_model,
        temperature=config.anthropic_temperature,
        max_tokens=config.anthropic_max_tokens,
    )

    learner = RankingActiveLearner(
        items=item_ids,
        K=config.k,
        posterior_sampler=posterior_sampler,
        experiment_runner=experiment_runner,
        warm_start_repeats=config.warm_start_repeats,
        batch_size=config.batch_size,
        direct_active_iterations=config.direct_active_iterations,
        random_seed=config.posterior_seed,
    )

    for _ in range(config.total_iterations):
        learner.run_iteration()

    results = build_results(learner, retrieved_items, config)
    payload = {
        "config": {
            "ref": config.ref,
            "relevance_prompt": config.relevance_prompt,
            "language": config.language,
            "retrieval_mode": config.retrieval_mode,
            "base_url": config.base_url,
            "sefaria_project_path": config.sefaria_project_path,
            "k": config.k,
            "warm_start_repeats": config.warm_start_repeats,
            "batch_size": config.batch_size,
            "direct_active_iterations": config.direct_active_iterations,
            "total_iterations": config.total_iterations,
            "posterior_samples": config.posterior_samples,
            "posterior_burn_in": config.posterior_burn_in,
            "posterior_thinning": config.posterior_thinning,
            "posterior_seed": config.posterior_seed,
            "anthropic_model": config.anthropic_model,
            "anthropic_temperature": config.anthropic_temperature,
            "anthropic_max_tokens": config.anthropic_max_tokens,
        },
        "num_retrieved_items": len(retrieved_items),
        "num_observations": len(learner.observations),
        "observations": learner.observations,
        "results": results,
    }
    return payload


def write_results(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path
