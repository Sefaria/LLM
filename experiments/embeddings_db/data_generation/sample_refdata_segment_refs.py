import json
import os
import random
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import django
import paramiko
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder
from tqdm import tqdm

django.setup()

from sefaria.model.text import Ref
from sefaria.system.exceptions import InputError

from connection_settings import CONNECTION

N = 100
MIN_PAGESHEETRANK = 1.0
SEED = None
USE_REMOTE = True
REMOTE_DB = CONNECTION["mongo_db"]
OUTPUT_PATH = None
OUTPUT_DESTINATION = "stdout"  # stdout, jsonl, or mongo
OUTPUT_MONGO_USE_REMOTE = False
OUTPUT_MONGO_DB = CONNECTION["mongo_db"]
OUTPUT_MONGO_COLLECTION = "sampled_segment_refs_to_embed"
VERBOSE = True
REMOTE_READ_CACHE_ENABLED = False
REMOTE_READ_CACHE_PATH = None
REMOTE_READ_CACHE_MODE = "sampled_rows"  # sampled_rows or candidate_pool


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def is_segment_ref(tref: str) -> bool:
    try:
        return Ref(tref).is_segment_level()
    except InputError:
        return False


@contextmanager
def ref_data_collection(use_remote: bool, remote_db: str):
    if not use_remote:
        from sefaria.system.database import db

        yield db.ref_data
        return

    # sshtunnel relies on paramiko.DSSKey, which is removed in newer paramiko versions.
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
        yield client[remote_db][CONNECTION["mongo_collection"]]
    finally:
        client.close()
        server.stop()


@contextmanager
def output_mongo_collection(use_remote: bool, mongo_db: str, collection_name: str):
    if not use_remote:
        from sefaria.system.database import db

        yield db.client[mongo_db][collection_name]
        return

    # sshtunnel relies on paramiko.DSSKey, which is removed in newer paramiko versions.
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
        yield client[mongo_db][collection_name]
    finally:
        client.close()
        server.stop()


def normalize_ref_data_row(row: dict) -> dict:
    return {
        "ref": row["ref"],
        "pagesheetrank": row["pagesheetrank"],
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(f"{cache_path.suffix}.meta.json")


def read_cache_metadata(cache_path: Path) -> Optional[dict]:
    metadata_path = cache_metadata_path(cache_path)
    if not metadata_path.exists():
        return None
    with metadata_path.open("r") as fin:
        return json.load(fin)


def write_cache_metadata(cache_path: Path, metadata: dict) -> None:
    metadata_path = cache_metadata_path(cache_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)


def expected_cache_metadata(min_pagesheetrank: float) -> dict:
    return {
        "min_pagesheetrank": min_pagesheetrank,
        "remote_db": REMOTE_DB,
        "cache_mode": REMOTE_READ_CACHE_MODE,
    }


def cache_matches_settings(cache_path: Path, min_pagesheetrank: float) -> bool:
    metadata = read_cache_metadata(cache_path)
    if metadata is None:
        return False
    return metadata == expected_cache_metadata(min_pagesheetrank)


def get_candidate_refs(collection, min_pagesheetrank: float) -> list[dict]:
    query = {"pagesheetrank": {"$gt": min_pagesheetrank}}
    total = collection.count_documents(query) if VERBOSE else None
    cursor = collection.find(
        query,
        {"_id": 0, "ref": 1, "pagesheetrank": 1},
    )
    cursor_iter = tqdm(cursor, desc="Scanning RefData rows", total=total) if VERBOSE else cursor
    candidates = []
    for row in cursor_iter:
        if is_segment_ref(row["ref"]):
            candidates.append(normalize_ref_data_row(row))
    log(f"Found {len(candidates)} segment refs with pagesheetrank > {min_pagesheetrank}")
    return candidates


def load_remote_cache_rows(cache_path: Path) -> list[dict]:
    start = time.perf_counter()
    rows = read_jsonl(cache_path)
    log(f"Loaded {len(rows)} rows from remote-read cache {cache_path} in {time.perf_counter() - start:.2f}s")
    return rows


def get_remote_candidate_pool(collection, min_pagesheetrank: float) -> list[dict]:
    cache_path = Path(REMOTE_READ_CACHE_PATH) if REMOTE_READ_CACHE_PATH is not None else None
    if REMOTE_READ_CACHE_ENABLED and REMOTE_READ_CACHE_MODE == "candidate_pool" and cache_path is not None and cache_path.exists():
        if cache_matches_settings(cache_path, min_pagesheetrank):
            return load_remote_cache_rows(cache_path)
        log(f"Ignoring stale candidate-pool cache at {cache_path}; settings changed")

    candidates = get_candidate_refs(collection, min_pagesheetrank)
    if REMOTE_READ_CACHE_ENABLED and REMOTE_READ_CACHE_MODE == "candidate_pool" and cache_path is not None:
        write_jsonl(candidates, cache_path)
        write_cache_metadata(cache_path, expected_cache_metadata(min_pagesheetrank))
    return candidates


def sample_refs_with_seed(collection, n: int, min_pagesheetrank: float, seed: int) -> list[dict]:
    start = time.perf_counter()
    log(f"Loading deterministic candidate pool with seed={seed}")
    candidates = get_remote_candidate_pool(collection, min_pagesheetrank)
    if n > len(candidates):
        raise ValueError(
            f"Requested {n} refs, but only found {len(candidates)} segment refs "
            f"with pagesheetrank > {min_pagesheetrank}."
        )

    rng = random.Random(seed)
    rows = rng.sample(candidates, n)
    log(f"Sampled {len(rows)} refs deterministically in {time.perf_counter() - start:.2f}s")
    return rows


def sample_refs_from_mongo(collection, n: int, min_pagesheetrank: float) -> list[dict]:
    start = time.perf_counter()
    if REMOTE_READ_CACHE_ENABLED and REMOTE_READ_CACHE_MODE == "candidate_pool" and REMOTE_READ_CACHE_PATH is not None:
        candidates = get_remote_candidate_pool(collection, min_pagesheetrank)
        if n > len(candidates):
            raise ValueError(
                f"Requested {n} refs, but only found {len(candidates)} segment refs "
                f"with pagesheetrank > {min_pagesheetrank}."
            )
        sampled = random.sample(candidates, n)
        log(f"Sampled {len(sampled)} refs from cached candidate pool in {time.perf_counter() - start:.2f}s")
        return sampled

    total = collection.count_documents({"pagesheetrank": {"$gt": min_pagesheetrank}})
    log(f"Found {total} RefData rows with pagesheetrank > {min_pagesheetrank}")
    if n > total:
        raise ValueError(f"Requested {n} refs, but fewer RefData rows match pagesheetrank > {min_pagesheetrank}.")

    sampled_by_ref = {}
    batch_size = max(n * 3, 100)
    max_attempts = 20

    attempts = range(max_attempts)
    if VERBOSE:
        attempts = tqdm(attempts, desc="Sampling segment refs", total=max_attempts)
    for attempt in attempts:
        log(f"Sampling attempt {attempt + 1}/{max_attempts}; collected {len(sampled_by_ref)}/{n}")
        rows = collection.aggregate([
            {"$match": {
                "pagesheetrank": {"$gt": min_pagesheetrank},
                "ref": {"$nin": list(sampled_by_ref)},
            }},
            {"$sample": {"size": batch_size}},
            {"$project": {"_id": 0, "ref": 1, "pagesheetrank": 1}},
        ])
        for row in rows:
            if is_segment_ref(row["ref"]):
                sampled_by_ref[row["ref"]] = normalize_ref_data_row(row)
                if len(sampled_by_ref) == n:
                    sampled = list(sampled_by_ref.values())
                    log(f"Sampled {len(sampled)} refs in {time.perf_counter() - start:.2f}s")
                    return sampled

    raise ValueError(
        f"Only found {len(sampled_by_ref)} segment refs with pagesheetrank > {min_pagesheetrank} "
        f"after {max_attempts} sampling attempts."
    )


def sample_refs(
    n: int,
    min_pagesheetrank: float = 1.0,
    seed: Optional[int] = None,
    use_remote: bool = False,
    remote_db: str = CONNECTION["mongo_db"],
) -> list[dict]:
    if (
        use_remote
        and REMOTE_READ_CACHE_ENABLED
        and REMOTE_READ_CACHE_MODE == "sampled_rows"
        and REMOTE_READ_CACHE_PATH is not None
    ):
        cache_path = Path(REMOTE_READ_CACHE_PATH)
        if cache_path.exists():
            if cache_matches_settings(cache_path, min_pagesheetrank):
                return load_remote_cache_rows(cache_path)
            log(f"Ignoring stale sampled-rows cache at {cache_path}; settings changed")

    log(f"Opening {'remote' if use_remote else 'local'} RefData collection")
    with ref_data_collection(use_remote, remote_db) as collection:
        if seed is not None:
            rows = sample_refs_with_seed(collection, n, min_pagesheetrank, seed)
        else:
            rows = sample_refs_from_mongo(collection, n, min_pagesheetrank)

    if (
        use_remote
        and REMOTE_READ_CACHE_ENABLED
        and REMOTE_READ_CACHE_MODE == "sampled_rows"
        and REMOTE_READ_CACHE_PATH is not None
    ):
        cache_path = Path(REMOTE_READ_CACHE_PATH)
        write_jsonl(rows, cache_path)
        write_cache_metadata(cache_path, expected_cache_metadata(min_pagesheetrank))

    return rows


def write_jsonl(rows: list[dict], output_path: Path) -> None:
    start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        row_iter = tqdm(rows, desc="Writing sampled refs JSONL") if VERBOSE else rows
        for row in row_iter:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"Wrote {len(rows)} sampled refs to {output_path} in {time.perf_counter() - start:.2f}s")


def write_mongo(rows: list[dict], use_remote: bool, mongo_db: str, collection_name: str) -> str:
    start = time.perf_counter()
    sample_run_id = str(uuid.uuid4())
    sampled_at = datetime.now(timezone.utc)
    docs = [
        {
            **row,
            "sample_run_id": sample_run_id,
            "sampled_at": sampled_at,
            "min_pagesheetrank": MIN_PAGESHEETRANK,
            "seed": SEED,
            "sample_source": "remote" if USE_REMOTE else "local",
        }
        for row in rows
    ]
    with output_mongo_collection(use_remote, mongo_db, collection_name) as collection:
        if docs:
            collection.insert_many(docs)
    log(
        f"Wrote {len(rows)} sampled refs to {'remote' if use_remote else 'local'} Mongo "
        f"{mongo_db}.{collection_name} in {time.perf_counter() - start:.2f}s"
    )
    return sample_run_id


def main() -> None:
    start = time.perf_counter()
    log("Starting RefData segment-ref sampling")
    rows = sample_refs(N, MIN_PAGESHEETRANK, SEED, USE_REMOTE, REMOTE_DB)
    if OUTPUT_DESTINATION == "jsonl":
        if OUTPUT_PATH is None:
            raise ValueError("OUTPUT_PATH must be set when OUTPUT_DESTINATION is 'jsonl'.")
        output_path = Path(OUTPUT_PATH)
        write_jsonl(rows, output_path)
        log(f"Wrote {len(rows)} sampled refs to {output_path}")
    elif OUTPUT_DESTINATION == "mongo":
        sample_run_id = write_mongo(rows, OUTPUT_MONGO_USE_REMOTE, OUTPUT_MONGO_DB, OUTPUT_MONGO_COLLECTION)
        target = f"{OUTPUT_MONGO_DB}.{OUTPUT_MONGO_COLLECTION}"
        location = "remote" if OUTPUT_MONGO_USE_REMOTE else "local"
        log(f"Wrote {len(rows)} sampled refs to {location} Mongo collection {target} with sample_run_id={sample_run_id}")
    elif OUTPUT_DESTINATION == "stdout":
        for row in rows:
            print(row["ref"])
    else:
        raise ValueError("OUTPUT_DESTINATION must be one of: stdout, jsonl, mongo.")
    log(f"Sampling script completed in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
