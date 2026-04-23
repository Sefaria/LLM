import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import django
import paramiko
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder

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


def get_candidate_refs(collection, min_pagesheetrank: float) -> list[dict]:
    cursor = collection.find(
        {"pagesheetrank": {"$gt": min_pagesheetrank}},
        {"_id": 0, "ref": 1, "pagesheetrank": 1},
    )
    candidates = []
    for row in cursor:
        if is_segment_ref(row["ref"]):
            candidates.append(normalize_ref_data_row(row))
    return candidates


def sample_refs_with_seed(collection, n: int, min_pagesheetrank: float, seed: int) -> list[dict]:
    candidates = get_candidate_refs(collection, min_pagesheetrank)
    if n > len(candidates):
        raise ValueError(
            f"Requested {n} refs, but only found {len(candidates)} segment refs "
            f"with pagesheetrank > {min_pagesheetrank}."
        )

    rng = random.Random(seed)
    return rng.sample(candidates, n)


def sample_refs_from_mongo(collection, n: int, min_pagesheetrank: float) -> list[dict]:
    if n > collection.count_documents({"pagesheetrank": {"$gt": min_pagesheetrank}}):
        raise ValueError(f"Requested {n} refs, but fewer RefData rows match pagesheetrank > {min_pagesheetrank}.")

    sampled_by_ref = {}
    batch_size = max(n * 3, 100)
    max_attempts = 20

    for _ in range(max_attempts):
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
                    return list(sampled_by_ref.values())

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
    with ref_data_collection(use_remote, remote_db) as collection:
        if seed is not None:
            return sample_refs_with_seed(collection, n, min_pagesheetrank, seed)
        return sample_refs_from_mongo(collection, n, min_pagesheetrank)


def write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_mongo(rows: list[dict], use_remote: bool, mongo_db: str, collection_name: str) -> str:
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
    return sample_run_id


def main() -> None:
    rows = sample_refs(N, MIN_PAGESHEETRANK, SEED, USE_REMOTE, REMOTE_DB)
    if OUTPUT_DESTINATION == "jsonl":
        if OUTPUT_PATH is None:
            raise ValueError("OUTPUT_PATH must be set when OUTPUT_DESTINATION is 'jsonl'.")
        output_path = Path(OUTPUT_PATH)
        write_jsonl(rows, output_path)
        print(f"Wrote {len(rows)} sampled refs to {output_path}")
    elif OUTPUT_DESTINATION == "mongo":
        sample_run_id = write_mongo(rows, OUTPUT_MONGO_USE_REMOTE, OUTPUT_MONGO_DB, OUTPUT_MONGO_COLLECTION)
        target = f"{OUTPUT_MONGO_DB}.{OUTPUT_MONGO_COLLECTION}"
        location = "remote" if OUTPUT_MONGO_USE_REMOTE else "local"
        print(f"Wrote {len(rows)} sampled refs to {location} Mongo collection {target} with sample_run_id={sample_run_id}")
    elif OUTPUT_DESTINATION == "stdout":
        for row in rows:
            print(row["ref"])
    else:
        raise ValueError("OUTPUT_DESTINATION must be one of: stdout, jsonl, mongo.")


if __name__ == "__main__":
    main()
