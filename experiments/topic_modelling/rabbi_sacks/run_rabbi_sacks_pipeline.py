#!/usr/bin/env python3
"""
Pipeline for generating topic predictions and LLM-filtered tags
for Rabbi Jonathan Sacks references.

Stage 1  → collect segment refs, predict candidate slugs, write JSONL.
Stage 2  → load that JSONL, filter slugs with an LLM, write JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
SEFARIA_ROOT = REPO_ROOT.parent / "Sefaria-Project"
TOPIC_MODELLING_ROOT = Path(__file__).resolve().parents[1]

if str(SEFARIA_ROOT) not in sys.path:
    sys.path.insert(0, str(SEFARIA_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sefaria.settings")

import django

django.setup()

from sefaria.model import Ref, Topic, IndexSet, Version  # noqa: E402
from sefaria.model.topic import AuthorTopic, RefTopicLinkSet  # noqa: E402

import experiments.topic_modelling.make_predictions as mp  # noqa: E402
from experiments.topic_modelling.llm_filtering import SequentialRefTopicFilter  # noqa: E402
from experiments.topic_modelling.utils import LabelledRef  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_community.vectorstores import Chroma  # noqa: E402

CHROMA_PATH = TOPIC_MODELLING_ROOT / ".chromadb_openai"
if CHROMA_PATH.exists():
    mp.db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=mp.embeddings,
    )

DEFAULT_AUTHOR_SLUG = "jonathan-sacks"


@dataclass
class PipelineConfig:
    output_dir: Path
    author_slug: str = DEFAULT_AUTHOR_SLUG
    limit: Optional[int] = None
    llm_model: str = "gpt-4o-mini"
    max_topics: int = 15
    candidate_k: int = 500
    threshold_factor: float = 2.5
    debug: bool = False


@dataclass
class PipelineResult:
    stage1_path: Path
    stage2_path: Path
    refs_processed: int


def _score_to_weight(score: float) -> float:
    l2 = mp.euclidean_relevance_to_l2(score)
    cosine = mp.l2_to_cosine_similarity(l2)
    return mp.cosine_to_one_minus_sine(cosine) ** 5


def _segment_refs_for_index(index) -> List[tuple[int, str]]:
    """Collect (order_id, tref) pairs for every segment in an index."""
    rows: List[tuple[int, str]] = []

    def action(_text, en_tref, he_tref, _version_obj):
        tref = en_tref or he_tref
        if not tref:
            return
        try:
            oref = Ref(tref)
        except Exception:
            return
        rows.append((oref.order_id(), oref.normal()))

    # Mirror the Sefaria-Data pattern: iterate the author’s versions and walk through
    # every segment to ensure we only collect refs that actually exist in the text.
    for language in ("en", "he"):
        version = Version().load({"title": index.title, "language": language})
        if not version:
            continue
        version.walk_thru_contents(action)
        if rows:
            break

    if not rows:
        for ref in index.all_segment_refs():
            try:
                rows.append((ref.order_id(), ref.normal()))
            except Exception:
                continue

    return rows


def fetch_author_segments(author_slug: str) -> List[str]:
    """
    Collect every library segment attributed to the author by scanning the indexes
    that list this author in their metadata (pattern used across Sefaria-Data scripts).
    Falls back to topic-link based sources if no authored indexes are found.
    """
    topic = Topic.init(author_slug)
    if topic is None or not isinstance(topic, AuthorTopic):
        raise ValueError(f"Unknown author slug or not an author: {author_slug}")

    indexes = IndexSet({"authors": author_slug})
    if indexes.count() == 0:
        return []

    ordered_refs: List[tuple[int, str]] = []
    for index in indexes:
        try:
            ordered_refs.extend(_segment_refs_for_index(index))
        except Exception:
            continue

    # Fallback to linked refs (legacy data) if we did not yield anything
    if not ordered_refs:
        links = RefTopicLinkSet({"toTopic": author_slug, "is_sheet": False})
        for link in links:
            for expanded in getattr(link, "expandedRefs", []):
                try:
                    ordered_refs.append((Ref(expanded).order_id(), expanded))
                except Exception:
                    continue

    ordered_refs.sort(key=lambda pair: pair[0])

    seen: set[str] = set()
    results: List[str] = []
    for _, tref in ordered_refs:
        if tref in seen:
            continue
        seen.add(tref)
        results.append(tref)

    return results


def stage_one_predict(refs: List[str], out_path: Path, cfg: PipelineConfig) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        for ref in tqdm(refs, desc="Predicting topic slugs"):
            text = mp.get_ref_text_with_fallback(Ref(ref), "en", auto_translate=True)
            if not text.strip():
                fh.write(json.dumps({"ref": ref, "slugs": []}, ensure_ascii=False) + "\n")
                continue

            docs = mp.get_closest_docs_by_text_similarity(text, k=cfg.candidate_k)
            freq_map = mp.get_recommended_slugs_weighted_frequency_map(
                docs, _score_to_weight, ref_to_ignore=ref
            )
            best_slugs = mp.get_keys_above_mean(freq_map, cfg.threshold_factor)
            fh.write(json.dumps({"ref": ref, "slugs": best_slugs}, ensure_ascii=False) + "\n")


def stage_two_filter(predictions_path: Path, out_path: Path, cfg: PipelineConfig) -> None:
    llm = ChatOpenAI(model=cfg.llm_model, temperature=0)
    filterer = SequentialRefTopicFilter(llm=llm, max_topics=cfg.max_topics, debug=cfg.debug)

    with predictions_path.open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in tqdm(src, desc="Filtering slugs with LLM"):
            row = json.loads(line)
            lr = LabelledRef(row["ref"], row.get("slugs", []))
            if not lr.slugs:
                dst.write(json.dumps({"ref": lr.ref, "slugs": []}, ensure_ascii=False) + "\n")
                continue
            kept = filterer.filter_ref(lr)
            dst.write(json.dumps({"ref": lr.ref, "slugs": kept}, ensure_ascii=False) + "\n")


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    stage1_path = cfg.output_dir / f"stage1_predictions_{run_id}.jsonl"
    stage2_path = cfg.output_dir / f"stage2_filtered_{run_id}.jsonl"

    refs = fetch_author_segments(cfg.author_slug)
    if cfg.limit:
        refs = refs[: cfg.limit]

    stage_one_predict(refs, stage1_path, cfg)
    stage_two_filter(stage1_path, stage2_path, cfg)

    return PipelineResult(stage1_path=stage1_path, stage2_path=stage2_path, refs_processed=len(refs))


def parse_args(argv: Optional[List[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run Rabbi Sacks topic-tagging pipeline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Directory for JSONL outputs.",
    )
    parser.add_argument(
        "--author-slug",
        default=DEFAULT_AUTHOR_SLUG,
        help="Topic slug for the author (default: jonathan-sacks).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of segment refs to process.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="Chat model to use for filtering.",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=15,
        help="Maximum slugs to keep after filtering.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Number of nearest neighbours to consider in Stage 1.",
    )
    parser.add_argument(
        "--threshold-factor",
        type=float,
        default=2.5,
        help="Std-dev multiplier for slug selection threshold.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging for LLM prompts.",
    )

    args = parser.parse_args(argv)
    return PipelineConfig(
        output_dir=args.output_dir,
        author_slug=args.author_slug,
        limit=args.limit,
        llm_model=args.llm_model,
        max_topics=args.max_topics,
        candidate_k=args.candidate_k,
        threshold_factor=args.threshold_factor,
        debug=args.debug,
    )


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)
    result = run_pipeline(cfg)
    print(f"Stage 1 → {result.stage1_path}")
    print(f"Stage 2 → {result.stage2_path}")
    print(f"Processed {result.refs_processed} refs.")


if __name__ == "__main__":
    main()
