from __future__ import annotations

from typing import Mapping

from experiments.llm_based_sorting.plackett_luce.experiment_runners import (
    ClaudeListwiseRankingExperimentRunner,
    ItemId,
    make_item_renderer_from_lookup,
)


def build_claude_ranking_runner(
    item_text_lookup: Mapping[ItemId, str],
) -> ClaudeListwiseRankingExperimentRunner:
    """
    Edit this function when you want to plug in your real LLM experiment.

    Expected usage:

        runner = build_claude_ranking_runner(item_text_lookup)
        learner = RankingActiveLearner(
            items=list(item_text_lookup),
            K=4,
            posterior_sampler=posterior_sampler,
            experiment_runner=runner,
            ...
        )

    The active learner itself stays unchanged. Only this outer injected
    component depends on Claude.
    """
    return ClaudeListwiseRankingExperimentRunner(
        item_renderer=make_item_renderer_from_lookup(item_text_lookup),
        ranking_instruction=(
            "Rank these items from best to worst for the retrieval/ranking task at hand. "
            "Return only valid JSON like {\"ranking\": [2, 0, 1, 3]} using the Local index values."
        ),
    )
