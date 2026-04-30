import json
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import django
from tqdm import tqdm

import generate_eval_queries_and_qrels as eval_gen
import sample_refdata_segment_refs as sampler
from connection_settings import CONNECTION
from sefaria.model.text import Ref
from sefaria.system.exceptions import InputError

django.setup()


BASE_DIR = Path(__file__).resolve().parent

INPUT_DOCUMENTS_PATH = BASE_DIR / "output" / "eval_dataset_sanity_check" / "documents.jsonl"
OUTPUT_CONFOUNDING_PATH = BASE_DIR / "output" / "eval_dataset_sanity_check" / "confounding_documents.jsonl"
OUTPUT_METADATA_PATH = BASE_DIR / "output" / "eval_dataset_sanity_check" / "confounding_documents_metadata.json"
WRITE_MERGED_DOCUMENTS = True
OUTPUT_MERGED_DOCUMENTS_PATH = BASE_DIR / "output" / "eval_dataset_sanity_check" / "documents_with_confounders.jsonl"

USE_REMOTE = True
REMOTE_DB = CONNECTION["mongo_db"]
MIN_PAGESHEETRANK = 2.0
REMOTE_READ_CACHE_ENABLED = True
REMOTE_READ_CACHE_PATH = BASE_DIR / "output" / "cache" / "sampled_refs_sanity_check_cache.jsonl"
REMOTE_READ_CACHE_MODE = "candidate_pool"

SEED = 613
CONFOUNDERS_PER_DOCUMENT = 10
REQUIRE_SAME_CATEGORY = True
REQUIRE_DIFFERENT_SOURCE = False
REQUIRE_SAME_SOURCE = False
TOKEN_TOLERANCE_RATIO = 0.15
MIN_TOKEN_TOLERANCE = 8
MAX_ACCEPTABLE_RELATIVE_ERROR = 0.25
MAX_EXPANSION_STEPS = 12
MAX_SEGMENTS_PER_DOCUMENT = 8
MAX_SEED_ATTEMPTS_PER_DOCUMENT = 600
VERBOSE = True


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as fin:
        row_iter = tqdm(fin, desc=f"Reading {path.name}") if VERBOSE else fin
        for line in row_iter:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def token_count(text: str) -> int:
    return len(text.split()) if text else 0


def target_token_band(target_tokens: int) -> tuple[int, int]:
    tolerance = max(MIN_TOKEN_TOLERANCE, int(round(target_tokens * TOKEN_TOLERANCE_RATIO)))
    return max(1, target_tokens - tolerance), target_tokens + tolerance


@lru_cache(maxsize=None)
def get_ref_descriptor(tref: str) -> tuple[Optional[str], Optional[str]]:
    try:
        oref = Ref(tref)
    except Exception:
        return None, None
    try:
        return oref.index.get_primary_category(), oref.index.title
    except Exception:
        return None, None


@lru_cache(maxsize=None)
def get_ref_obj(tref: str) -> Optional[Ref]:
    try:
        return Ref(tref)
    except Exception:
        return None


@lru_cache(maxsize=None)
def get_clean_text(tref: str, lang: str) -> Optional[str]:
    return eval_gen.get_ref_text(tref, lang)


def refs_overlap(left_tref: str, right_tref: str) -> bool:
    left = get_ref_obj(left_tref)
    right = get_ref_obj(right_tref)
    if left is None or right is None:
        return False
    try:
        return bool(left.overlaps(right))
    except Exception:
        return False


def overlaps_any(tref: str, other_refs: list[str]) -> bool:
    return any(refs_overlap(tref, other_tref) for other_tref in other_refs)


def segment_count_for_ref(tref: str) -> int:
    oref = get_ref_obj(tref)
    if oref is None:
        return 1
    try:
        return len(oref.all_segment_refs())
    except Exception:
        return 1


def build_candidate_option(
    start_tref: str,
    end_tref: str,
    side: str,
    lang: str,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> Optional[dict]:
    if side == "left":
        next_start = eval_gen.get_prev_segment_tref(start_tref)
        next_end = end_tref
        if not next_start:
            return None
    else:
        next_start = start_tref
        next_end = eval_gen.get_next_segment_tref(end_tref)
        if not next_end:
            return None
    candidate_ref = eval_gen.build_range_ref(next_start, next_end)
    text = get_clean_text(candidate_ref, lang)
    if not text:
        return None
    tokens = token_count(text)
    if segment_count_for_ref(candidate_ref) > MAX_SEGMENTS_PER_DOCUMENT:
        return None
    if min_tokens <= tokens <= max_tokens:
        score = (0, abs(tokens - target_tokens))
    elif tokens < min_tokens:
        score = (1, min_tokens - tokens)
    else:
        score = (2, tokens - max_tokens)
    return {
        "ref": candidate_ref,
        "start_ref": next_start,
        "end_ref": next_end,
        "text": text,
        "tokens": tokens,
        "score": score,
    }


def expand_seed_to_target_length(seed_ref: str, lang: str, target_tokens: int) -> Optional[dict]:
    text = get_clean_text(seed_ref, lang)
    if not text:
        return None

    min_tokens, max_tokens = target_token_band(target_tokens)
    current_start = seed_ref
    current_end = seed_ref
    current_ref = seed_ref
    current_text = text
    current_tokens = token_count(text)

    if min_tokens <= current_tokens <= max_tokens:
        return {
            "ref": current_ref,
            "text": current_text,
            "token_count": current_tokens,
            "steps": 0,
            "segment_count": segment_count_for_ref(current_ref),
        }
    if current_tokens > max_tokens:
        relative_error = abs(current_tokens - target_tokens) / max(1, target_tokens)
        if relative_error <= MAX_ACCEPTABLE_RELATIVE_ERROR:
            return {
                "ref": current_ref,
                "text": current_text,
                "token_count": current_tokens,
                "steps": 0,
                "segment_count": segment_count_for_ref(current_ref),
            }
        return None

    for step in range(1, MAX_EXPANSION_STEPS + 1):
        options = []
        for side in ("left", "right"):
            option = build_candidate_option(
                current_start,
                current_end,
                side,
                lang,
                target_tokens,
                min_tokens,
                max_tokens,
            )
            if option is not None:
                options.append(option)
        if not options:
            break

        best = min(options, key=lambda item: item["score"])
        current_ref = best["ref"]
        current_start = best["start_ref"]
        current_end = best["end_ref"]
        current_text = best["text"]
        current_tokens = best["tokens"]

        if min_tokens <= current_tokens <= max_tokens:
            return {
                "ref": current_ref,
                "text": current_text,
                "token_count": current_tokens,
                "steps": step,
                "segment_count": segment_count_for_ref(current_ref),
            }

    relative_error = abs(current_tokens - target_tokens) / max(1, target_tokens)
    if relative_error <= MAX_ACCEPTABLE_RELATIVE_ERROR:
        return {
            "ref": current_ref,
            "text": current_text,
            "token_count": current_tokens,
            "steps": MAX_EXPANSION_STEPS,
            "segment_count": segment_count_for_ref(current_ref),
        }
    return None


def candidate_matches_metadata(seed_ref: str, source_doc: dict) -> bool:
    source_metadata = source_doc.get("metadata", {})
    target_category = source_metadata.get("category")
    target_source = source_metadata.get("source")
    seed_category, seed_source = get_ref_descriptor(seed_ref)

    if REQUIRE_SAME_CATEGORY and target_category and seed_category != target_category:
        return False
    if REQUIRE_DIFFERENT_SOURCE and target_source and seed_source == target_source:
        return False
    if REQUIRE_SAME_SOURCE and target_source and seed_source != target_source:
        return False
    return True


def make_confounder_doc_id(tref: str, lang: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", tref).strip("_").lower()
    return f"conf_{lang}_{slug}"


def sample_confounders_for_document(
    source_doc: dict,
    candidate_pool: list[dict],
    occupied_refs: list[str],
    used_final_refs: set[str],
    rng: random.Random,
) -> list[dict]:
    source_metadata = source_doc["metadata"]
    target_lang = source_metadata["lang"]
    target_ref = source_metadata["ref"]
    target_source = source_metadata.get("source")
    target_category = source_metadata.get("category")
    target_tokens = token_count(source_doc["text"])

    chosen = []
    tried_seed_refs = set()
    attempts = 0

    while len(chosen) < CONFOUNDERS_PER_DOCUMENT and attempts < MAX_SEED_ATTEMPTS_PER_DOCUMENT:
        row = candidate_pool[rng.randrange(len(candidate_pool))]
        seed_ref = row["ref"]
        attempts += 1

        if seed_ref in tried_seed_refs:
            continue
        tried_seed_refs.add(seed_ref)

        if not candidate_matches_metadata(seed_ref, source_doc):
            continue
        if overlaps_any(seed_ref, occupied_refs):
            continue

        seed_text = get_clean_text(seed_ref, target_lang)
        if not seed_text:
            continue

        expanded = expand_seed_to_target_length(seed_ref, target_lang, target_tokens)
        if expanded is None:
            continue

        final_ref = expanded["ref"]
        if final_ref in used_final_refs:
            continue
        if overlaps_any(final_ref, occupied_refs):
            continue

        final_category, final_source = get_ref_descriptor(final_ref)
        if REQUIRE_SAME_CATEGORY and target_category and final_category != target_category:
            continue
        if REQUIRE_DIFFERENT_SOURCE and target_source and final_source == target_source:
            continue
        if REQUIRE_SAME_SOURCE and target_source and final_source != target_source:
            continue

        used_final_refs.add(final_ref)
        occupied_refs.append(final_ref)
        chosen.append(
            {
                "doc_id": make_confounder_doc_id(final_ref, target_lang),
                "text": expanded["text"],
                "metadata": {
                    "ref": final_ref,
                    "seed_ref": seed_ref,
                    "matched_doc_id": source_doc["doc_id"],
                    "matched_ref": target_ref,
                    "url": f"https://www.sefaria.org/{Ref(final_ref).url()}",
                    "source": final_source,
                    "category": final_category,
                    "lang": target_lang,
                    "pagesheetrank": row.get("pagesheetrank"),
                    "is_confounder": True,
                    "target_token_count": target_tokens,
                    "confounder_token_count": expanded["token_count"],
                    "length_ratio": expanded["token_count"] / max(1, target_tokens),
                    "expansion_steps": expanded["steps"],
                    "segment_count": expanded["segment_count"],
                    "sampling_attempts_used": attempts,
                },
            }
        )

    if len(chosen) < CONFOUNDERS_PER_DOCUMENT:
        log(
            f"{source_doc['doc_id']}: found {len(chosen)}/{CONFOUNDERS_PER_DOCUMENT} "
            f"confounders after {attempts} seed attempts"
        )
    return chosen


def load_candidate_pool() -> list[dict]:
    sampler.VERBOSE = VERBOSE
    sampler.REMOTE_DB = REMOTE_DB
    sampler.REMOTE_READ_CACHE_ENABLED = REMOTE_READ_CACHE_ENABLED
    sampler.REMOTE_READ_CACHE_PATH = REMOTE_READ_CACHE_PATH
    sampler.REMOTE_READ_CACHE_MODE = REMOTE_READ_CACHE_MODE
    with sampler.ref_data_collection(USE_REMOTE, REMOTE_DB) as collection:
        if USE_REMOTE:
            return sampler.get_remote_candidate_pool(collection, MIN_PAGESHEETRANK)
        return sampler.get_candidate_refs(collection, MIN_PAGESHEETRANK)


def main() -> None:
    start = time.perf_counter()
    rng = random.Random(SEED)

    log(f"Reading source documents from {INPUT_DOCUMENTS_PATH}")
    source_documents = read_jsonl(INPUT_DOCUMENTS_PATH)
    candidate_pool = load_candidate_pool()
    log(f"Loaded {len(candidate_pool)} seed candidates from RefData")

    occupied_refs = [doc["metadata"]["ref"] for doc in source_documents]
    used_final_refs = set(occupied_refs)
    confounders = []

    doc_iter = tqdm(source_documents, desc="Building confounders") if VERBOSE else source_documents
    for source_doc in doc_iter:
        confounders.extend(
            sample_confounders_for_document(
                source_doc,
                candidate_pool,
                occupied_refs,
                used_final_refs,
                rng,
            )
        )

    eval_gen.write_jsonl(OUTPUT_CONFOUNDING_PATH, confounders)
    if WRITE_MERGED_DOCUMENTS:
        eval_gen.write_jsonl(OUTPUT_MERGED_DOCUMENTS_PATH, source_documents + confounders)

    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_documents_path": str(INPUT_DOCUMENTS_PATH),
        "output_confounding_path": str(OUTPUT_CONFOUNDING_PATH),
        "output_merged_documents_path": str(OUTPUT_MERGED_DOCUMENTS_PATH) if WRITE_MERGED_DOCUMENTS else None,
        "use_remote": USE_REMOTE,
        "remote_db": REMOTE_DB,
        "min_pagesheetrank": MIN_PAGESHEETRANK,
        "seed": SEED,
        "confounders_per_document": CONFOUNDERS_PER_DOCUMENT,
        "require_same_category": REQUIRE_SAME_CATEGORY,
        "require_different_source": REQUIRE_DIFFERENT_SOURCE,
        "require_same_source": REQUIRE_SAME_SOURCE,
        "token_tolerance_ratio": TOKEN_TOLERANCE_RATIO,
        "min_token_tolerance": MIN_TOKEN_TOLERANCE,
        "max_acceptable_relative_error": MAX_ACCEPTABLE_RELATIVE_ERROR,
        "max_expansion_steps": MAX_EXPANSION_STEPS,
        "max_segments_per_document": MAX_SEGMENTS_PER_DOCUMENT,
        "max_seed_attempts_per_document": MAX_SEED_ATTEMPTS_PER_DOCUMENT,
        "source_documents_count": len(source_documents),
        "confounding_documents_count": len(confounders),
        "merged_documents_count": len(source_documents) + len(confounders) if WRITE_MERGED_DOCUMENTS else None,
    }
    OUTPUT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_METADATA_PATH.open("w") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)
    log(f"Wrote confounder metadata to {OUTPUT_METADATA_PATH}")
    log(f"Completed confounder build in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
