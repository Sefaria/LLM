"""
Profiling script for LLMParallelAmbiguousResolver.

This script profiles the performance of the LLMParallelAmbiguousResolver by:
1. Fetching random linker output documents that contain ambiguous citations
2. Extracting individual ambiguity groups (by charRange) from each document
3. Processing each group independently to measure resolution performance

Each "sample" in the output CSV represents one ambiguity group - a single citation
location that has multiple ambiguous alternative resolutions. This allows for
fine-grained performance analysis at the individual citation level.

Output CSV columns:
- sample_index: Sequential index of the ambiguity group
- doc_index: Index of the source linker output document
- group_index: Index of the group within its document
- citing_ref: Reference of the document containing the citation
- citing_language: Language of the citing text (typically "he" for Hebrew)
- citing_version: Version title of the citing text
- citation_text: First 100 characters of the citation text
- char_range: Character range of the citation in the source document
- num_alternatives: Number of ambiguous alternative refs for this citation
- alternative_refs: Pipe-separated list of all alternative refs
- resolved: Whether the ambiguity was successfully resolved (True/False)
- resolved_ref: The resolved reference (if resolved)
- match_source: Source of the match (dicta/search)
- dicta_seconds: Time spent in Dicta API calls
- elasticsearch_seconds: Time spent in Elasticsearch queries
- total_seconds: Total resolution time for this group
- llm_tokens: Total tokens used by LLM calls
- llm_tokens_by_model: JSON breakdown of tokens by model
- error: Error message if any exception occurred
"""

import csv
import json
import os
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List

from llm_parallel_resolver import LLMParallelAmbiguousResolver
from utils import get_random_ambiguous_linker_outputs
from util.general import run_parallel


_thread_local = threading.local()


def _get_resolver() -> LLMParallelAmbiguousResolver:
    resolver = getattr(_thread_local, "resolver", None)
    if resolver is None:
        resolver = LLMParallelAmbiguousResolver(debug=False)
        _thread_local.resolver = resolver
    return resolver


def _extract_ambiguity_groups(linker_output: dict) -> List[Dict[str, Any]]:
    """
    Extract individual ambiguity groups (by charRange) from a linker output document.
    Each group represents one citation location with multiple ambiguous alternatives.
    """
    spans = linker_output.get("spans", [])
    ambiguous_citations = [
        span for span in spans
        if span.get("ambiguous") is True and span.get("type") == "citation"
    ]

    # Group by charRange
    char_range_groups = defaultdict(list)
    for span in ambiguous_citations:
        char_range = tuple(span.get('charRange', []))
        if char_range:
            char_range_groups[char_range].append(span)

    # Convert to list of groups with metadata
    groups = []
    for char_range, span_group in char_range_groups.items():
        # Get the refs for this group (all alternatives)
        refs = [span.get("ref") for span in span_group if span.get("ref")]
        groups.append({
            "char_range": list(char_range),
            "text": span_group[0].get("text", "") if span_group else "",
            "num_alternatives": len(span_group),
            "refs": refs,
            "spans": span_group,
        })

    return groups


def _process_group(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Process a single ambiguity group."""
    resolver = _get_resolver()
    linker_output = item["linker_output"]
    group = item["group"]
    doc_index = item["doc_index"]
    group_index = item["group_index"]

    # Create a synthetic linker_output with only this group's spans
    synthetic_output = {
        "ref": linker_output.get("ref"),
        "language": linker_output.get("language"),
        "versionTitle": linker_output.get("versionTitle"),
        "spans": group["spans"],
    }

    profile: Dict[str, Any] = {}
    start = time.perf_counter()
    resolved = False
    resolved_ref = ""
    match_source = ""
    error = ""

    try:
        result = resolver.resolve(synthetic_output, profile=profile)
        if result and result.get("resolved_groups", 0) > 0:
            resolved = True
            # Get the first resolution
            resolutions = result.get("resolutions", [])
            if resolutions:
                resolution = resolutions[0]
                resolved_ref = resolution.get("resolved_ref", "")
                match_source = resolution.get("match_source", "")
    except Exception as exc:
        error = str(exc)

    total_seconds = time.perf_counter() - start

    citing_ref = linker_output.get("ref", "")
    citing_language = linker_output.get("language", "")
    citing_version = linker_output.get("versionTitle", "")

    return {
        "sample_index": index,
        "doc_index": doc_index,
        "group_index": group_index,
        "citing_ref": citing_ref,
        "citing_language": citing_language,
        "citing_version": citing_version,
        "citation_text": group["text"][:100],  # First 100 chars
        "char_range": json.dumps(group["char_range"]),
        "num_alternatives": group["num_alternatives"],
        "alternative_refs": " | ".join(group["refs"]),
        "resolved": str(resolved),
        "resolved_ref": resolved_ref,
        "match_source": match_source,
        "dicta_seconds": f"{profile.get('dicta_seconds', 0.0):.4f}",
        "elasticsearch_seconds": f"{profile.get('es_seconds', 0.0):.4f}",
        "total_seconds": f"{total_seconds:.4f}",
        "llm_tokens": str(profile.get("llm_tokens", 0)),
        "llm_tokens_by_model": json.dumps(profile.get("llm_tokens_by_model", {}), ensure_ascii=True),
        "error": error,
    }


def main() -> None:
    # Configuration
    num_documents = 200  # Number of linker output documents to fetch
    workers = 10
    seed = 616
    use_remote = True
    use_cache = True
    span_type = "citation"
    output_path = "experiments/linker/resolve_ambiguous/llm_parallel_ambiguous_resolver_profile.csv"

    print(f"Fetching {num_documents} random ambiguous linker outputs...")
    linker_outputs = get_random_ambiguous_linker_outputs(
        n=num_documents,
        use_remote=use_remote,
        seed=seed,
        use_cache=use_cache,
        progress=True,
        span_type=span_type,
    )

    if not linker_outputs:
        raise RuntimeError("No ambiguous linker outputs found. Check database connectivity or filters.")

    print(f"Extracting ambiguity groups from {len(linker_outputs)} documents...")

    # Extract all ambiguity groups from all documents
    all_groups = []
    sample_index = 0
    for doc_index, linker_output in enumerate(linker_outputs):
        groups = _extract_ambiguity_groups(linker_output)
        for group_index, group in enumerate(groups):
            all_groups.append({
                "item": {
                    "linker_output": linker_output,
                    "group": group,
                    "doc_index": doc_index,
                    "group_index": group_index,
                },
                "index": sample_index
            })
            sample_index += 1

    print(f"Found {len(all_groups)} total ambiguity groups to process")

    if not all_groups:
        raise RuntimeError("No ambiguity groups found in the fetched linker outputs.")

    def unit_func(payload: Dict[str, Any]) -> Dict[str, Any]:
        return _process_group(payload["item"], payload["index"])

    rows = run_parallel(all_groups, unit_func, max_workers=workers, desc="Resolving groups")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "sample_index",
        "doc_index",
        "group_index",
        "citing_ref",
        "citing_language",
        "citing_version",
        "citation_text",
        "char_range",
        "num_alternatives",
        "alternative_refs",
        "resolved",
        "resolved_ref",
        "match_source",
        "dicta_seconds",
        "elasticsearch_seconds",
        "total_seconds",
        "llm_tokens",
        "llm_tokens_by_model",
        "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary statistics
    total = len(rows)
    resolved_count = sum(1 for r in rows if r["resolved"] == "True")
    error_count = sum(1 for r in rows if r["error"])

    print(f"\n{'='*80}")
    print(f"Results written to: {output_path}")
    print(f"{'='*80}")
    print(f"Total ambiguity groups processed: {total}")
    print(f"Successfully resolved: {resolved_count} ({resolved_count/total*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/total*100:.1f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

