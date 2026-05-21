from __future__ import annotations

from experiments.llm_based_sorting.plackett_luce.sefaria_ranking_pipeline import (
    SefariaQuotingCommentaryRankingConfig,
    run_sefaria_quoting_commentary_ranking,
)
from experiments.llm_based_sorting.plackett_luce.sefaria_retrieval import (
    QuotingCommentaryItem,
)


class _FakeRetriever:
    def fetch(self, tref: str):
        return [
            QuotingCommentaryItem(
                anchor_ref=tref,
                source_ref=f"Commentary {i}",
                category="Quoting Commentary",
                index_title="Commentary",
                collective_title_en=f"Commentator {i}",
                collective_title_he="",
                text=[f"text {i}"],
                he=[f"he {i}"],
                raw_link={"sourceRef": f"Commentary {i}"},
            )
            for i in range(6)
        ]


def test_pipeline_returns_sorted_results_with_metadata(monkeypatch) -> None:
    from experiments.llm_based_sorting.plackett_luce import sefaria_ranking_pipeline as pipeline

    class _FakeRunner:
        def __call__(self, items: list[int]) -> list[int]:
            return sorted(items)

    monkeypatch.setattr(
        pipeline,
        "build_claude_relevance_runner",
        lambda *args, **kwargs: _FakeRunner(),
    )

    config = SefariaQuotingCommentaryRankingConfig(
        ref="Genesis 1:1",
        relevance_prompt="Prefer more informative passages.",
        k=3,
        total_iterations=4,
        posterior_samples=20,
        posterior_burn_in=10,
        posterior_thinning=1,
    )

    payload = run_sefaria_quoting_commentary_ranking(config, retriever=_FakeRetriever())

    assert payload["num_retrieved_items"] == 6
    assert payload["num_observations"] > 0
    assert len(payload["results"]) == 6
    assert payload["results"][0]["posterior_mean"] >= payload["results"][-1]["posterior_mean"]
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["ref"] == "Genesis 1:1"
