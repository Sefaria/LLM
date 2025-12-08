import os
import json
import random
from contextlib import contextmanager

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
    random_links = get_random_non_segment_links_with_chunks(5, use_remote=True, seed=613, use_cache=False)
    for item in random_links:
        print(item)
