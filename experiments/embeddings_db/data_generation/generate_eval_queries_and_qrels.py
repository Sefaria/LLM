import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import django
import paramiko
from anthropic import Anthropic
from openai import OpenAI
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder
from tqdm import tqdm

django.setup()

from sefaria.model.text import Ref
from sefaria.system.exceptions import InputError

from connection_settings import CONNECTION

INPUT_SOURCE = "mongo"  # jsonl or mongo
INPUT_JSONL_PATH = None
INPUT_MONGO_USE_REMOTE = False
INPUT_MONGO_DB = CONNECTION["mongo_db"]
INPUT_MONGO_COLLECTION = "sampled_segment_refs_to_embed"
INPUT_SAMPLE_RUN_ID = None
LIMIT = None
INPUT_REMOTE_CACHE_ENABLED = False
INPUT_REMOTE_CACHE_PATH = None

TEXT_LANGS = ["en", "he"]
TEXT_LANG_SELECTION_MODE = "prefer_he_fallback_en"  # prefer_he_fallback_en or all_available
QUERY_LANGUAGE_NAMES = {
    "en": "English",
    "he": "Hebrew",
}
QUERIES_PER_TYPE_PER_DOC = 3
QUERY_TYPES = ["keyword", "question", "sentence"]
QUERY_TYPES_PER_DOC = 2
QUERY_TYPE_SAMPLE_SEED = 613
LLM_MAX_WORKERS = 4
LLM_PROVIDER = "anthropic"  # anthropic or openai
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_MAX_TOKENS = 4096
OPENAI_MODEL = "gpt-4o-mini"
LAST_LLM_MODEL_USED = None
_thread_local = threading.local()

OUTPUT_DESTINATION = "jsonl"  # jsonl or mongo
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval_dataset"
OUTPUT_MONGO_USE_REMOTE = False
OUTPUT_MONGO_DB = CONNECTION["mongo_db"]
OUTPUT_MONGO_PREFIX = "embedding_eval"
VERBOSE = True


def log(message: str) -> None:
    if VERBOSE:
        print(message)


@contextmanager
def mongo_collection(use_remote: bool, mongo_db: str, collection_name: str):
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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    log(f"Reading sampled refs from {path}")
    with path.open("r") as fin:
        line_iter = tqdm(fin, desc="Reading sampled refs JSONL") if VERBOSE else fin
        for line in line_iter:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log(f"Loaded {len(rows)} sampled refs from JSONL")
    return rows


def load_sampled_refs() -> list[dict]:
    start = time.perf_counter()
    if INPUT_SOURCE == "jsonl":
        if INPUT_JSONL_PATH is None:
            raise ValueError("INPUT_JSONL_PATH must be set when INPUT_SOURCE is 'jsonl'.")
        rows = read_jsonl(Path(INPUT_JSONL_PATH))
    elif INPUT_SOURCE == "mongo":
        if INPUT_MONGO_USE_REMOTE and INPUT_REMOTE_CACHE_ENABLED and INPUT_REMOTE_CACHE_PATH is not None:
            cache_path = Path(INPUT_REMOTE_CACHE_PATH)
            if cache_path.exists():
                rows = read_jsonl(cache_path)
                if LIMIT is not None:
                    rows = rows[:LIMIT]
                    log(f"Applied LIMIT={LIMIT}; using {len(rows)} sampled refs")
                log(f"Loaded {len(rows)} sampled refs from remote-read cache {cache_path} in {time.perf_counter() - start:.2f}s")
                return rows
        query = {}
        if INPUT_SAMPLE_RUN_ID is not None:
            query["sample_run_id"] = INPUT_SAMPLE_RUN_ID
        with mongo_collection(INPUT_MONGO_USE_REMOTE, INPUT_MONGO_DB, INPUT_MONGO_COLLECTION) as collection:
            total = collection.count_documents(query) if VERBOSE else None
            cursor = collection.find(query, {"_id": 0})
            cursor_iter = tqdm(cursor, desc="Loading sampled refs from Mongo", total=total) if VERBOSE else cursor
            rows = list(cursor_iter)
        if INPUT_MONGO_USE_REMOTE and INPUT_REMOTE_CACHE_ENABLED and INPUT_REMOTE_CACHE_PATH is not None:
            write_jsonl(Path(INPUT_REMOTE_CACHE_PATH), rows)
    else:
        raise ValueError("INPUT_SOURCE must be one of: jsonl, mongo.")

    if LIMIT is not None:
        rows = rows[:LIMIT]
        log(f"Applied LIMIT={LIMIT}; using {len(rows)} sampled refs")
    log(f"Loaded sampled refs in {time.perf_counter() - start:.2f}s")
    return rows


def make_doc_id(tref: str, lang: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", tref).strip("_").lower()
    return f"doc_{lang}_{slug}"


def get_ref_text(tref: str, lang: str) -> Optional[str]:
    try:
        text = Ref(tref).text(lang).as_string()
    except (InputError, ValueError):
        return None
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def choose_langs_for_ref(available_by_lang: dict[str, Optional[str]]) -> list[str]:
    available_langs = [lang for lang in TEXT_LANGS if available_by_lang.get(lang)]
    if TEXT_LANG_SELECTION_MODE == "all_available":
        return available_langs
    if TEXT_LANG_SELECTION_MODE == "prefer_he_fallback_en":
        if available_by_lang.get("he"):
            return ["he"]
        if available_by_lang.get("en"):
            return ["en"]
        return []
    raise ValueError("TEXT_LANG_SELECTION_MODE must be one of: prefer_he_fallback_en, all_available.")


def build_documents(sampled_refs: list[dict]) -> list[dict]:
    start = time.perf_counter()
    documents = []
    seen_doc_ids = set()
    ref_iter = tqdm(sampled_refs, desc="Fetching Sefaria texts") if VERBOSE else sampled_refs
    missing_by_lang = {lang: 0 for lang in TEXT_LANGS}
    for row in ref_iter:
        tref = row["ref"]
        texts_by_lang = {}
        for lang in TEXT_LANGS:
            text = get_ref_text(tref, lang)
            texts_by_lang[lang] = text
            if text is None:
                missing_by_lang[lang] = missing_by_lang.get(lang, 0) + 1
        for lang in choose_langs_for_ref(texts_by_lang):
            doc_id = make_doc_id(tref, lang)
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            oref = Ref(tref)
            documents.append({
                "doc_id": doc_id,
                "text": texts_by_lang[lang],
                "metadata": {
                    "ref": tref,
                    "url": f"https://www.sefaria.org/{oref.url()}",
                    "source": oref.index.title,
                    "category": oref.index.get_primary_category(),
                    "lang": lang,
                    "pagesheetrank": row.get("pagesheetrank"),
                    "sample_run_id": row.get("sample_run_id"),
                },
            })
    log(f"Built {len(documents)} documents in {time.perf_counter() - start:.2f}s")
    if VERBOSE:
        for lang, count in missing_by_lang.items():
            if count:
                print(f"Missing {lang} text for {count} refs")
    return documents


def choose_query_types_for_doc(doc: dict) -> list[str]:
    if QUERY_TYPES_PER_DOC >= len(QUERY_TYPES):
        return list(QUERY_TYPES)
    seed = f"{QUERY_TYPE_SAMPLE_SEED}:{doc['doc_id']}"
    rng = random.Random(seed)
    chosen = rng.sample(QUERY_TYPES, QUERY_TYPES_PER_DOC)
    return [query_type for query_type in QUERY_TYPES if query_type in chosen]


def make_query_type_prompt(doc: dict, query_type: str) -> str:
    query_lang = doc["metadata"]["lang"]
    query_language_name = QUERY_LANGUAGE_NAMES.get(query_lang, query_lang)
    compact_doc = {
        "doc_id": doc["doc_id"],
        "ref": doc["metadata"]["ref"],
        "category": doc["metadata"]["category"],
        "lang": query_lang,
        "text": doc["text"][:2400],
    }
    return f"""
You are creating an information-retrieval evaluation dataset for Jewish text search.

Create exactly {QUERIES_PER_TYPE_PER_DOC} {query_type} retrieval queries for the supplied document.
This is NOT question answering. A query should map to useful documents.

Rules:
- Write every query in {query_language_name}.
- Query type must be: {query_type}.
- The supplied document is highly relevant to every query you create.
- Prefer realistic user search language over citation wording.
- Focus on the content of the text itself, not on metadata about the ref.
- Avoid queries that are mainly based on the title, category, book name, author name, or other reference metadata unless that information is explicitly central to the passage content.
- Keep keyword queries short.
- Question queries should be natural user questions.
- Sentence queries should be natural sentence-length search inputs, not questions.

Return only valid JSON. Do not use markdown fences.
Shape:
{{
  "queries": [
    {{"text": "...", "reason": "why this document is highly relevant"}}
  ]
}}

Document:
{json.dumps(compact_doc, ensure_ascii=False)}
""".strip()


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def call_openai_for_query_type(client: OpenAI, doc: dict, query_type: str) -> dict:
    log(f"Calling OpenAI {OPENAI_MODEL} for {doc['doc_id']} {query_type}")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You create careful retrieval evaluation datasets."},
            {"role": "user", "content": make_query_type_prompt(doc, query_type)},
        ],
    )
    return json.loads(response.choices[0].message.content)


def call_anthropic_for_query_type(client: Anthropic, doc: dict, query_type: str) -> dict:
    global LAST_LLM_MODEL_USED
    log(f"Calling Anthropic {ANTHROPIC_MODEL} for {doc['doc_id']} {query_type}")
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        temperature=0,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system="You create careful retrieval evaluation datasets. Return only valid JSON.",
        messages=[
            {"role": "user", "content": make_query_type_prompt(doc, query_type)},
        ],
    )
    LAST_LLM_MODEL_USED = ANTHROPIC_MODEL
    content = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return parse_json_response(content)


def create_llm_client():
    if LLM_PROVIDER == "anthropic":
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")), call_anthropic_for_query_type
    if LLM_PROVIDER == "openai":
        return OpenAI(), call_openai_for_query_type
    raise ValueError("LLM_PROVIDER must be one of: anthropic, openai.")


def get_thread_llm_client():
    client = getattr(_thread_local, "llm_client", None)
    if client is None:
        client, _ = create_llm_client()
        _thread_local.llm_client = client
    return client


def call_llm_for_query_type(doc: dict, query_type: str) -> dict:
    client = get_thread_llm_client()
    if LLM_PROVIDER == "anthropic":
        return call_anthropic_for_query_type(client, doc, query_type)
    if LLM_PROVIDER == "openai":
        return call_openai_for_query_type(client, doc, query_type)
    raise ValueError("LLM_PROVIDER must be one of: anthropic, openai.")


def get_llm_model_name() -> str:
    if LLM_PROVIDER == "anthropic":
        return LAST_LLM_MODEL_USED or ANTHROPIC_MODEL
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    return LLM_PROVIDER


def normalize_query_type_output(output: dict, doc: dict, query_type: str, job_index: int) -> tuple[list[dict], list[dict]]:
    queries = []
    qrels = []
    query_lang = doc["metadata"]["lang"]
    for query_index, raw_query in enumerate(output.get("queries", [])):
        if query_index >= QUERIES_PER_TYPE_PER_DOC:
            break
        query_text = (raw_query.get("text") or "").strip()
        if not query_text:
            continue
        query_id = f"q_{query_lang}_{query_type}_{job_index}_{query_index}"
        queries.append({
            "query_id": query_id,
            "text": query_text,
            "type": query_type,
            "lang": query_lang,
        })
        qrels.append({
            "query_id": query_id,
            "doc_id": doc["doc_id"],
            "relevance": 2,
            "reason": (raw_query.get("reason") or "").strip(),
        })
    return queries, qrels


def generate_queries_and_qrels(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    jobs = [
        {"doc": doc, "query_type": query_type, "index": index}
        for index, (doc, query_type) in enumerate(
            (doc, query_type)
            for doc in documents
            for query_type in choose_query_types_for_doc(doc)
        )
    ]
    all_queries = []
    all_qrels = []
    start = time.perf_counter()

    def run_job(job: dict) -> tuple[list[dict], list[dict]]:
        output = call_llm_for_query_type(job["doc"], job["query_type"])
        return normalize_query_type_output(output, job["doc"], job["query_type"], job["index"])

    progress = tqdm(total=len(jobs), desc="Generating query jobs") if VERBOSE else None
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            queries, qrels = future.result()
            all_queries.extend(queries)
            all_qrels.extend(qrels)
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()

    log(f"Generated {len(all_queries)} queries and {len(all_qrels)} qrels in {time.perf_counter() - start:.2f}s")
    return all_queries, all_qrels


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        row_iter = tqdm(rows, desc=f"Writing {path.name}") if VERBOSE else rows
        for row in row_iter:
            fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    log(f"Wrote {len(rows)} rows to {path}")


def write_dataset_jsonl(documents: list[dict], queries: list[dict], qrels: list[dict]) -> None:
    start = time.perf_counter()
    write_jsonl(OUTPUT_DIR / "documents.jsonl", documents)
    write_jsonl(OUTPUT_DIR / "queries.jsonl", queries)
    write_jsonl(OUTPUT_DIR / "qrels.jsonl", qrels)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_provider": LLM_PROVIDER,
        "llm_model": get_llm_model_name(),
        "text_langs": TEXT_LANGS,
        "text_lang_selection_mode": TEXT_LANG_SELECTION_MODE,
        "queries_per_type_per_doc": QUERIES_PER_TYPE_PER_DOC,
        "query_types": QUERY_TYPES,
        "query_types_per_doc": QUERY_TYPES_PER_DOC,
        "query_type_sample_seed": QUERY_TYPE_SAMPLE_SEED,
        "input_source": INPUT_SOURCE,
        "input_mongo_collection": INPUT_MONGO_COLLECTION,
        "input_sample_run_id": INPUT_SAMPLE_RUN_ID,
        "documents_count": len(documents),
        "queries_count": len(queries),
        "qrels_count": len(qrels),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "metadata.json").open("w") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)
    log(f"Wrote JSONL dataset to {OUTPUT_DIR} in {time.perf_counter() - start:.2f}s")


def write_dataset_mongo(documents: list[dict], queries: list[dict], qrels: list[dict]) -> str:
    start = time.perf_counter()
    dataset_run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc)

    def with_run(row: dict) -> dict:
        return {
            **row,
            "dataset_run_id": dataset_run_id,
            "generated_at": generated_at,
            "llm_provider": LLM_PROVIDER,
            "llm_model": get_llm_model_name(),
        }

    collections = {
        f"{OUTPUT_MONGO_PREFIX}_documents": [with_run(row) for row in documents],
        f"{OUTPUT_MONGO_PREFIX}_queries": [with_run(row) for row in queries],
        f"{OUTPUT_MONGO_PREFIX}_qrels": [with_run(row) for row in qrels],
    }
    for collection_name, rows in collections.items():
        log(f"Writing {len(rows)} rows to Mongo collection {OUTPUT_MONGO_DB}.{collection_name}")
        with mongo_collection(OUTPUT_MONGO_USE_REMOTE, OUTPUT_MONGO_DB, collection_name) as collection:
            if rows:
                collection.insert_many(rows)
    log(f"Wrote Mongo dataset in {time.perf_counter() - start:.2f}s")
    return dataset_run_id


def main() -> None:
    start = time.perf_counter()
    log("Starting eval query/qrel generation")
    sampled_refs = load_sampled_refs()
    documents = build_documents(sampled_refs)

    log(f"Using {LLM_PROVIDER} model {get_llm_model_name()}")
    all_queries, all_qrels = generate_queries_and_qrels(documents)

    if OUTPUT_DESTINATION == "jsonl":
        write_dataset_jsonl(documents, all_queries, all_qrels)
        log(f"Wrote dataset to {OUTPUT_DIR}")
    elif OUTPUT_DESTINATION == "mongo":
        dataset_run_id = write_dataset_mongo(documents, all_queries, all_qrels)
        log(f"Wrote dataset to Mongo with dataset_run_id={dataset_run_id}")
    else:
        raise ValueError("OUTPUT_DESTINATION must be one of: jsonl, mongo.")
    log(f"Eval generation completed in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
