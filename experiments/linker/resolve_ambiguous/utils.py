import os
import json
import random
from contextlib import contextmanager
from typing import List

import django
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder
import paramiko

django.setup()

from sefaria.model.text import Ref
from sefaria.system.database import db

from connection_settings import CONNECTION


@contextmanager
def links_collection(use_remote: bool):
    """
    Yield the links collection, optionally via the remote SSH-tunneled Mongo used in analysis scripts.
    """
    if not use_remote:
        yield db.links
        return

    # sshtunnel relies on paramiko.DSSKey which is removed in newer paramiko versions.
    if not hasattr(paramiko, "DSSKey"):
        paramiko.DSSKey = paramiko.RSAKey

    server = SSHTunnelForwarder(
        (CONNECTION["ssh_host"], CONNECTION["ssh_port"]),
        ssh_username=CONNECTION["ssh_user"],
        ssh_pkey=os.path.expanduser(CONNECTION["ssh_pkey"]),
        remote_bind_address=(CONNECTION["remote_bind_host"], CONNECTION["remote_bind_port"]),
    )
    server.start()
    client = MongoClient(f"mongodb://127.0.0.1:{server.local_bind_port}")
    try:
        yield client[CONNECTION["mongo_db"]][CONNECTION["mongo_collection"]]
    finally:
        client.close()
        server.stop()


@contextmanager
def linker_output_collection(use_remote: bool):
    """
    Yield the linker_output collection, optionally via the remote SSH-tunneled Mongo used in analysis scripts.
    """
    if not use_remote:
        # For local, assume the collection exists in the same database
        yield db["linker_output"]
        return

    # sshtunnel relies on paramiko.DSSKey which is removed in newer paramiko versions.
    if not hasattr(paramiko, "DSSKey"):
        paramiko.DSSKey = paramiko.RSAKey

    server = SSHTunnelForwarder(
        (CONNECTION["ssh_host"], CONNECTION["ssh_port"]),
        ssh_username=CONNECTION["ssh_user"],
        ssh_pkey=os.path.expanduser(CONNECTION["ssh_pkey"]),
        remote_bind_address=(CONNECTION["remote_bind_host"], CONNECTION["remote_bind_port"]),
    )
    server.start()
    client = MongoClient(f"mongodb://127.0.0.1:{server.local_bind_port}")
    try:
        # Use the same database as links collection (sefaria-library-linker2)
        yield client[CONNECTION["mongo_db"]]["linker_output"]
    finally:
        client.close()
        server.stop()


def get_ambiguous_linker_outputs(
    use_remote: bool = False,
    limit: int = None,
    progress: bool = False,
    span_type: str = "citation",
) -> list:
    """
    Return all linker output objects from the linker_output collection that have ambiguous resolution.
    Uses the query: { spans : { $elemMatch : { ambiguous : true, type : <span_type> } } }

    Args:
        use_remote: If True, connect to remote MongoDB via SSH tunnel
        limit: Optional limit on the number of results to return
        progress: If True, show progress bar (requires tqdm)
        span_type: Filter for span type. Defaults to "citation". Set to None for all types.

    Returns:
        List of linker output documents with ambiguous spans of the specified type
    """
    results: list = []

    with linker_output_collection(use_remote) as collection:
        # Build query based on whether span_type is specified
        if span_type:
            query = {"spans": {"$elemMatch": {"ambiguous": True, "type": span_type}}}
        else:
            query = {"spans": {"$elemMatch": {"ambiguous": True}}}

        cursor = collection.find(query)
        if limit:
            cursor = cursor.limit(limit)

        total = collection.count_documents(query) if progress else None

        cursor_iter = cursor
        if progress and total:
            try:
                from tqdm import tqdm
                cursor_iter = tqdm(cursor, desc="Fetching ambiguous linker outputs", total=min(total, limit) if limit else total)
            except Exception:
                pass

        for doc in cursor_iter:
            results.append(doc)

    return results


def get_random_ambiguous_linker_outputs(
    n: int,
    use_remote: bool = False,
    seed: int = None,
    use_cache: bool = False,
    progress: bool = False,
    span_type: str = "citation",
) -> list:
    """
    Return up to n random linker output documents that have ambiguous spans of the specified type.

    Performance improvements:
    - Avoids per-iteration `skip(idx)` calls (which are O(idx) in MongoDB).
    - Fetches a batch of random candidates with a single aggregate query (or a single scan fallback).
    - Supports caching for deterministic results with the same seed.

    Args:
        n: Number of random ambiguous linker outputs to return
        use_remote: If True, connect to remote MongoDB via SSH tunnel
        seed: Random seed for reproducibility. If None, results are non-deterministic.
        use_cache: If True, cache results to disk and reuse on subsequent calls with same parameters
        progress: If True, show progress bar (requires tqdm)
        span_type: Filter for span type. Defaults to "citation". Set to None for all types.

    Returns:
        List of up to n random linker output documents with ambiguous spans of the specified type
    """
    cache_key = f"ambiguous-{'remote' if use_remote else 'local'}-n{n}-seed{seed}-type{span_type}"
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    if use_cache and os.path.isfile(cache_file):
        try:
            with open(cache_file, "r") as fin:
                return json.load(fin)
        except Exception:
            pass

    rng = random.Random(seed)
    results: list = []
    seen_ids: set[str] = set()

    with linker_output_collection(use_remote) as collection:
        # Build query based on whether span_type is specified
        if span_type:
            query = {"spans": {"$elemMatch": {"ambiguous": True, "type": span_type}}}
        else:
            query = {"spans": {"$elemMatch": {"ambiguous": True}}}

        total = collection.count_documents(query)
        if total == 0:
            return results

        # Determine sample size - over-sample to ensure we get enough after filtering
        sample_size = min(total, n * 3)

        # First try to use Mongo's $sample for fast random selection
        try:
            pipeline = [
                {"$match": query},
                {"$sample": {"size": sample_size}},
            ]
            candidates: List[dict] = list(collection.aggregate(pipeline))
        except Exception:
            # Fallback: scan IDs once, then sample in memory deterministically via `seed`
            id_cursor = collection.find(query, {"_id": 1})
            ids = [doc["_id"] for doc in id_cursor]
            if not ids:
                return results
            rng.shuffle(ids)
            chosen_ids = ids[:sample_size]
            candidates = list(
                collection.find(
                    {"_id": {"$in": chosen_ids}},
                )
            )

        candidate_iter = candidates
        if progress:
            try:
                from tqdm import tqdm
                candidate_iter = tqdm(
                    candidate_iter,
                    desc="Scanning ambiguous linker outputs",
                    total=len(candidates),
                )
            except Exception:
                pass

        for doc in candidate_iter:
            if len(results) >= n:
                break

            doc_id = str(doc.get("_id"))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            results.append(doc)

    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w") as fout:
                json.dump(results, fout, default=str)
        except Exception:
            pass

    return results


def get_random_non_segment_links_with_chunks(
    n: int,
    use_remote: bool = False,
    seed: int = None,
    use_cache: bool = False,
    progress: bool = False,
) -> list:
    """
    Return up to n random links whose refs are not segment-level (e.g., book or chapter level),
    along with their corresponding marked_up_text_chunks entry. Only links with a chunk match are returned.

    Performance improvements vs. previous version:
    - Avoids per-iteration `skip(idx)` calls (which are O(idx) in MongoDB).
    - Fetches a batch of random candidates with a single aggregate query (or a single scan fallback).
    """
    cache_key = f"{'remote' if use_remote else 'local'}-n{n}-seed{seed}"
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    if use_cache and os.path.isfile(cache_file):
        try:
            with open(cache_file, "r") as fin:
                return json.load(fin)
        except Exception:
            pass

    rng = random.Random(seed)
    results: list = []
    seen_ids: set[str] = set()

    with links_collection(use_remote) as links:
        query = {"generated_by": "add_links_from_text"}
        total = links.count_documents(query)
        if total == 0:
            return results

        chunks_col = links.database["marked_up_text_chunks"]

        # Over-sample to compensate for filtering (book-level / segment-level / no-chunk).
        max_attempts = max(n * 20, n + 10)
        sample_size = min(total, max_attempts * 3)

        # First try to use Mongo's $sample for fast random selection.
        try:
            pipeline = [
                {"$match": query},
                {"$sample": {"size": sample_size}},
                {"$project": {"refs": 1, "generated_by": 1}},
            ]
            candidates: List[dict] = list(links.aggregate(pipeline))
        except Exception:
            # Fallback: scan IDs once, then sample in memory deterministically via `seed`.
            id_cursor = links.find(query, {"_id": 1})
            ids = [doc["_id"] for doc in id_cursor]
            if not ids:
                return results
            rng.shuffle(ids)
            chosen_ids = ids[:sample_size]
            candidates = list(
                links.find(
                    {"_id": {"$in": chosen_ids}},
                    {"refs": 1, "generated_by": 1},
                )
            )

        candidate_iter = candidates
        if progress:
            try:
                from tqdm import tqdm

                candidate_iter = tqdm(
                    candidate_iter,
                    desc="Scanning sampled links",
                    total=len(candidates),
                )
            except Exception:
                pass

        for link in candidate_iter:
            if len(results) >= n:
                break

            link_id = str(link.get("_id"))
            if link_id in seen_ids:
                continue
            seen_ids.add(link_id)

            if link.get("generated_by") != "add_links_from_text":
                continue

            refs = link.get("refs") or []
            for tref in refs:
                try:
                    oref = Ref(tref)
                except Exception:
                    continue

                if oref.is_book_level() or oref.is_segment_level():
                    # Skip book-level and segment-level refs; we want higher-level only.
                    continue

                chunk = _find_chunk_for_link(link, chunks_col)
                if chunk:
                    results.append({"link": link, "chunk": chunk})
                    break  # move on to next link once we got a valid chunk

    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w") as fout:
                json.dump(results, fout, default=str)
        except Exception:
            pass

    return results


def _find_chunk_for_link(link: dict, chunks_col):
    """
    Attempt to locate the marked_up_text_chunks document for this link by:
    - For each ref in the link, find a chunk with that ref
    - And ensure the other ref from the link appears as a citation in spans.ref
    """
    refs = link.get("refs") or []
    for i, source_ref in enumerate(refs):
        for j, cited_ref in enumerate(refs):
            if i == j:
                continue
            chunk = chunks_col.find_one({"ref": source_ref, "spans.ref": cited_ref})
            if chunk:
                return chunk
    return None

if __name__ == '__main__':
    # Test get_random_ambiguous_linker_outputs with caching and seed
    print("Testing get_random_ambiguous_linker_outputs (random with seed)...")
    random_ambiguous = get_random_ambiguous_linker_outputs(
        n=5,
        use_remote=True,
        seed=42,
        use_cache=False,
        progress=True
    )
    print(f"Found {len(random_ambiguous)} random ambiguous citation outputs\n")

    for i, output in enumerate(random_ambiguous, 1):
        print(f"{i}. Ref: {output.get('ref')}")
        ambiguous_spans = [span for span in output.get('spans', []) if span.get('ambiguous') and span.get('type') == 'citation']
        print(f"   Ambiguous citation spans: {len(ambiguous_spans)}")

        # Group by charRange to show unique citation locations
        from collections import defaultdict
        char_range_groups = defaultdict(list)
        for span in ambiguous_spans:
            char_range = tuple(span.get('charRange', []))
            if char_range:
                char_range_groups[char_range].append(span)

        print(f"   Unique citation locations: {len(char_range_groups)}")
        for j, (char_range, spans) in enumerate(list(char_range_groups.items())[:2], 1):
            print(f"   {j}. charRange={list(char_range)}: {len(spans)} alternative(s)")
            print(f"      Text: '{spans[0].get('text', 'N/A')[:40]}'")
            print(f"      Refs: {[s.get('ref', 'N/A') for s in spans[:3]]}")
        print()

    print("="*80 + "\n")

    # Test get_ambiguous_linker_outputs - default is citations only (non-random)
    print("Testing get_ambiguous_linker_outputs (non-random, limit 3)...")
    ambiguous_citations = get_ambiguous_linker_outputs(use_remote=True, limit=3, progress=True)
    print(f"Found {len(ambiguous_citations)} ambiguous citation outputs\n")

    for i, output in enumerate(ambiguous_citations, 1):
        print(f"{i}. Ref: {output.get('ref')}")
        ambiguous_spans = [span for span in output.get('spans', []) if span.get('ambiguous') and span.get('type') == 'citation']
        print(f"   Ambiguous citation spans: {len(ambiguous_spans)}")
        for span in ambiguous_spans[:2]:
            print(f"   - Text: '{span.get('text', 'N/A')[:50]}' → Ref: {span.get('ref', 'N/A')}")
        print()

    print("\n" + "="*80 + "\n")

    # Original test
    # random_links = get_random_non_segment_links_with_chunks(5, use_remote=True, seed=613, use_cache=False)
    # for item in random_links:
    #     print(item)
