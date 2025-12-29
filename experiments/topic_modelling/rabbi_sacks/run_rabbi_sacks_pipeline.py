#!/usr/bin/env python3
"""
Pipeline for generating topic predictions and LLM-filtered tags
for Rabbi Jonathan Sacks references.

Stage 1  → collect segment refs, predict candidate slugs, write JSONL.
Stage 2  → load that JSONL, filter slugs with an LLM, write reviewable CSV.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import csv

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

try:
    from langchain_anthropic import ChatAnthropic  # noqa: E402
except ImportError:  # pragma: no cover - optional dependency
    ChatAnthropic = None

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
    llm_provider: str = "gpt"
    llm_workers: int = 1
    anthropic_max_output_tokens: int = 512
    max_topics: int = 15
    candidate_k: int = 500
    threshold_factor: float = 2.5
    debug: bool = False


@dataclass
class PipelineResult:
    stage1_path: Path
    stage2_path: Path
    refs_processed: int


# Edit this object to configure the pipeline without CLI arguments.
PIPELINE_CONFIG = PipelineConfig(
    output_dir=Path(__file__).resolve().parent / "runs",
    author_slug=DEFAULT_AUTHOR_SLUG,
    limit=None,
    llm_model="gpt-4o-mini",
    llm_provider="gpt",
    llm_workers=4,
    anthropic_max_output_tokens=512,
    max_topics=15,
    candidate_k=500,
    threshold_factor=2.5,
    debug=False,
)


def _score_to_weight(score: float) -> float:
    l2 = mp.euclidean_relevance_to_l2(score)
    cosine = mp.l2_to_cosine_similarity(l2)
    return mp.cosine_to_one_minus_sine(cosine) ** 5


def _create_chat_model(cfg: PipelineConfig):
    provider = cfg.llm_provider.lower()
    if provider in {"gpt", "openai"}:
        return ChatOpenAI(model=cfg.llm_model, temperature=0)
    if provider in {"claude", "anthropic"}:
        if ChatAnthropic is None:
            raise RuntimeError(
                "langchain_anthropic is not installed. Install it or choose --llm-provider gpt."
            )
        return ChatAnthropic(
            model=cfg.llm_model,
            temperature=0,
            max_output_tokens=cfg.anthropic_max_output_tokens,
        )
    raise ValueError(f"Unsupported llm provider: {cfg.llm_provider}")


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
            try:
                text = mp.get_ref_text_with_fallback(Ref(ref), "en", auto_translate=True)
            except Exception as exc:
                print(f"⚠️  failed to fetch text for {ref}: {exc}")
                text = ""

            if text.strip():
                try:
                    docs = mp.get_closest_docs_by_text_similarity(text, k=cfg.candidate_k)
                    freq_map = mp.get_recommended_slugs_weighted_frequency_map(
                        docs, _score_to_weight, ref_to_ignore=ref
                    )
                    slugs = mp.get_keys_above_mean(freq_map, cfg.threshold_factor)
                except Exception as exc:
                    print(f"⚠️  embedding lookup failed for {ref}: {exc}")
                    slugs = []
            else:
                slugs = []

            fh.write(
                json.dumps({"ref": ref, "slugs": slugs, "text": text}, ensure_ascii=False) + "\n"
            )


def stage_two_filter(predictions_path: Path, out_path: Path, cfg: PipelineConfig) -> None:
    labelled: List[LabelledRef] = []
    texts: dict[str, str] = {}
    with predictions_path.open("r", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            labelled.append(LabelledRef(row["ref"], row.get("slugs", [])))
            texts[row["ref"]] = row.get("text", "")

    filterer = SequentialRefTopicFilter(
        llm=_create_chat_model(cfg),
        max_topics=cfg.max_topics,
        debug=cfg.debug,
    )
    filtered_map = filterer.filter_refs(labelled, max_workers=cfg.llm_workers)

    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["Ref", "Slugs", "Text"])
        writer.writeheader()
        for lr in labelled:
            kept = filtered_map.get(lr.ref, [])
            text = texts.get(lr.ref, "")
            if not text:
                try:
                    text = mp.get_ref_text_with_fallback(Ref(lr.ref), "en", auto_translate=True)
                except Exception as exc:
                    print(f"⚠️  failed to fetch text for {lr.ref} during CSV write: {exc}")
                    text = ""
            writer.writerow(
                {
                    "Ref": lr.ref,
                    "Slugs": "; ".join(kept),
                    "Text": text,
                }
            )


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    stage1_path = cfg.output_dir / f"stage1_predictions_{run_id}.jsonl"
    stage2_path = cfg.output_dir / f"stage2_filtered_{run_id}.csv"

    refs = fetch_author_segments(cfg.author_slug)
    if cfg.limit:
        refs = refs[: cfg.limit]

    stage_one_predict(refs, stage1_path, cfg)
    stage_two_filter(stage1_path, stage2_path, cfg)

    return PipelineResult(stage1_path=stage1_path, stage2_path=stage2_path, refs_processed=len(refs))


def main(argv: Optional[List[str]] = None) -> None:
    result = run_pipeline(PIPELINE_CONFIG)
    print(f"Stage 1 → {result.stage1_path}")
    print(f"Stage 2 → {result.stage2_path}")
    print(f"Processed {result.refs_processed} refs.")


if __name__ == "__main__":
    PIPELINE_CONFIG = PipelineConfig(
        output_dir=Path(__file__).resolve().parent / "runs",
        author_slug=DEFAULT_AUTHOR_SLUG,
        limit=None,
        llm_model="gpt-4o-mini",
        llm_provider="gpt",
        llm_workers=4,
        anthropic_max_output_tokens=512,
        max_topics=15,
        candidate_k=500,
        threshold_factor=2.5,
        debug=False,
    )
    main()
