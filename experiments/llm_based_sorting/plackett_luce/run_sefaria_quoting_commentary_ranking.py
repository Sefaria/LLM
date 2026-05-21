from __future__ import annotations

from pprint import pprint

from experiments.llm_based_sorting.plackett_luce.sefaria_ranking_pipeline import (
    SefariaQuotingCommentaryRankingConfig,
    run_sefaria_quoting_commentary_ranking,
    write_results,
)


# Edit parameters here. No CLI arguments are required.
CONFIG = SefariaQuotingCommentaryRankingConfig(
    ref="Genesis 1:1",
    relevance_prompt=(
        "Rank these quoting-commentary passages by how interesting and engaging they are for a reader. "
        "Prefer passages that contain something surprising, vivid, narratively rich, emotionally striking, "
        "or especially memorable. Favor commentary that tells a story, introduces a dramatic image, sharpens "
        "a conflict, reveals an unexpected interpretation, or gives a striking insight that would make someone "
        "want to keep reading. Rank lower passages that are mainly technical, repetitive, dry, purely linguistic, "
        "or only weakly engaging even if they are informative."
    ),
    language="en",
    retrieval_mode="local_project",
    base_url="http://localhost:8000",
    sefaria_project_path="/Users/yon/projects/Sefaria-Project",
    k=4,
    warm_start_repeats=2,
    batch_size=3,
    direct_active_iterations=1,
    total_iterations=12,
    posterior_samples=160,
    posterior_burn_in=80,
    posterior_thinning=2,
    posterior_seed=0,
    anthropic_model="claude-sonnet-4-6",
    anthropic_temperature=0.0,
    anthropic_max_tokens=1024,
    output_path="experiments/llm_based_sorting/plackett_luce/output/quoting_commentary_ranking.json",
)


def main() -> None:
    payload = run_sefaria_quoting_commentary_ranking(CONFIG)
    output_path = write_results(payload, CONFIG.output_path)

    print(f"Wrote results to {output_path}")
    print("Top ranked items:")
    for row in payload["results"][:10]:
        pprint(
            {
                "rank": row["rank"],
                "posterior_mean": row["posterior_mean"],
                "posterior_std": row["posterior_std"],
                "source_ref": row["source_ref"],
                "collective_title_en": row["collective_title_en"],
                "exposure_count": row["exposure_count"],
            }
        )


if __name__ == "__main__":
    main()
