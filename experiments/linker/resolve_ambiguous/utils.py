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
    n: int, use_remote: bool = False, seed: int = None, use_cache: bool = False
) -> list:
    """
    Return up to n random links whose refs are not segment-level (e.g., book or chapter level),
    along with their corresponding marked_up_text_chunks entry. Only links with a chunk match are returned.
    Randomly samples offsets instead of scanning the whole collection.
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
    query = {"generated_by": "add_links_from_text"}
    results = []
    seen_ids = set()

    with links_collection(use_remote) as links:
        total = links.count_documents(query)
        if total == 0:
            return results

        chunks_col = links.database["marked_up_text_chunks"]

        # Over-sample attempts to account for links that won't yield chunks.
        max_attempts = max(n * 20, n + 10)
        attempts = 0
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            idx = rng.randrange(total)
            doc_cursor = links.find(query, {"refs": 1, "generated_by": 1}).skip(idx).limit(1)
            link = next(doc_cursor, None)
            if not link:
                continue
            if link.get("generated_by") != "add_links_from_text":
                continue
            link_id = str(link.get("_id"))
            if link_id in seen_ids:
                continue
            seen_ids.add(link_id)

            refs = link.get("refs") or []
            for tref in refs:
                try:
                    oref = Ref(tref)
                except Exception:
                    continue
                if not oref.is_segment_level():
                    chunk = _find_chunk_for_link(link, chunks_col)
                    if chunk:
                        results.append({"link": link, "chunk": chunk})
                    break

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
