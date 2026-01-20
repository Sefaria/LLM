import csv
import json
import os
import threading
import time
from typing import Any, Dict

from llm_parallel_resolver import LLMParallelResolver
from utils import get_random_non_segment_links_with_chunks
from util.general import run_parallel


_thread_local = threading.local()


def _get_resolver() -> LLMParallelResolver:
    resolver = getattr(_thread_local, "resolver", None)
    if resolver is None:
        resolver = LLMParallelResolver(debug=False)
        _thread_local.resolver = resolver
    return resolver


def _process_item(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    resolver = _get_resolver()
    profile: Dict[str, Any] = {}
    start = time.perf_counter()
    resolved = False
    resolved_ref = ""
    error = ""
    try:
        result = resolver.resolve(item["link"], item["chunk"], profile=profile)
        if result is not None:
            resolved = True
            resolved_ref = result.get("resolved_ref") or ""
    except Exception as exc:
        error = str(exc)
    total_seconds = time.perf_counter() - start

    chunk = item.get("chunk") or {}
    link = item.get("link") or {}
    citing_ref = chunk.get("ref", "")
    citing_language = chunk.get("language", "")
    non_segment_ref = ""
    for tref in link.get("refs", []) or []:
        try:
            from sefaria.model.text import Ref
            oref = Ref(tref)
            if not oref.is_segment_level():
                non_segment_ref = oref.normal()
                break
        except Exception:
            continue

    def detect_language_for_ref(tref: str) -> str:
        if not tref:
            return ""
        try:
            from sefaria.model.text import Ref
            oref = Ref(tref)
            has_he = bool(oref.text("he").as_string())
            has_en = bool(oref.text("en").as_string())
        except Exception:
            return ""
        if has_he and has_en:
            return "he/en"
        if has_he:
            return "he"
        if has_en:
            return "en"
        return ""

    target_text_language = detect_language_for_ref(non_segment_ref)
    cited_text_language = detect_language_for_ref(resolved_ref)

    return {
        "sample_index": index,
        "citing_ref": citing_ref,
        "citing_language": citing_language,
        "non_segment_ref": non_segment_ref,
        "target_text_language": target_text_language,
        "resolved_ref": resolved_ref,
        "cited_text_language": cited_text_language,
        "dicta_seconds": f"{profile.get('dicta_seconds', 0.0):.4f}",
        "elasticsearch_seconds": f"{profile.get('es_seconds', 0.0):.4f}",
        "total_seconds": f"{total_seconds:.4f}",
        "llm_tokens": str(profile.get("llm_tokens", 0)),
        "llm_tokens_by_model": json.dumps(profile.get("llm_tokens_by_model", {}), ensure_ascii=True),
        "resolved": str(resolved),
        "error": error,
    }


def main() -> None:
    samples_count = 1000
    workers = 10
    seed = 616
    use_remote = True
    use_cache = True
    output_path = "experiments/linker/resolve_ambiguous/llm_parallel_resolver_profile.csv"

    samples = get_random_non_segment_links_with_chunks(
        n=samples_count,
        use_remote=use_remote,
        seed=seed,
        use_cache=use_cache,
        progress=True,
    )

    if not samples:
        raise RuntimeError("No samples found. Check database connectivity or filters.")

    def unit_func(payload: Dict[str, Any]) -> Dict[str, Any]:
        return _process_item(payload["item"], payload["index"])

    indexed_items = [{"item": item, "index": i} for i, item in enumerate(samples)]
    rows = run_parallel(indexed_items, unit_func, max_workers=workers, desc="Resolving")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "sample_index",
        "citing_ref",
        "citing_language",
        "non_segment_ref",
        "target_text_language",
        "resolved_ref",
        "cited_text_language",
        "dicta_seconds",
        "elasticsearch_seconds",
        "total_seconds",
        "llm_tokens",
        "llm_tokens_by_model",
        "resolved",
        "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
