import json
import os
import re
import uuid
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

TEXT_LANGS = ["en", "he"]
QUERY_LANGUAGE_NAMES = {
    "en": "English",
    "he": "Hebrew",
}
BATCH_SIZE = 8
QUERIES_PER_BATCH = 12
LLM_PROVIDER = "anthropic"  # anthropic or openai
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
ANTHROPIC_MAX_TOKENS = 4096
OPENAI_MODEL = "gpt-4o-mini"

OUTPUT_DESTINATION = "jsonl"  # jsonl or mongo
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval_dataset"
OUTPUT_MONGO_USE_REMOTE = False
OUTPUT_MONGO_DB = CONNECTION["mongo_db"]
OUTPUT_MONGO_PREFIX = "embedding_eval"


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
    with path.open("r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_sampled_refs() -> list[dict]:
    if INPUT_SOURCE == "jsonl":
        if INPUT_JSONL_PATH is None:
            raise ValueError("INPUT_JSONL_PATH must be set when INPUT_SOURCE is 'jsonl'.")
        rows = read_jsonl(Path(INPUT_JSONL_PATH))
    elif INPUT_SOURCE == "mongo":
        query = {}
        if INPUT_SAMPLE_RUN_ID is not None:
            query["sample_run_id"] = INPUT_SAMPLE_RUN_ID
        with mongo_collection(INPUT_MONGO_USE_REMOTE, INPUT_MONGO_DB, INPUT_MONGO_COLLECTION) as collection:
            rows = list(collection.find(query, {"_id": 0}))
    else:
        raise ValueError("INPUT_SOURCE must be one of: jsonl, mongo.")

    if LIMIT is not None:
        rows = rows[:LIMIT]
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


def build_documents(sampled_refs: list[dict]) -> list[dict]:
    documents = []
    seen_doc_ids = set()
    for row in sampled_refs:
        tref = row["ref"]
        for lang in TEXT_LANGS:
            doc_id = make_doc_id(tref, lang)
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            text = get_ref_text(tref, lang)
            if text is None:
                continue

            oref = Ref(tref)
            documents.append({
                "doc_id": doc_id,
                "text": text,
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
    return documents


def chunk_list(items: list[dict], batch_size: int) -> list[list[dict]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def group_documents_by_lang(documents: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for doc in documents:
        lang = doc["metadata"]["lang"]
        grouped.setdefault(lang, []).append(doc)
    return grouped


def make_prompt(documents: list[dict], query_lang: str) -> str:
    query_language_name = QUERY_LANGUAGE_NAMES.get(query_lang, query_lang)
    compact_docs = [
        {
            "doc_id": doc["doc_id"],
            "ref": doc["metadata"]["ref"],
            "category": doc["metadata"]["category"],
            "lang": doc["metadata"]["lang"],
            "text": doc["text"][:1800],
        }
        for doc in documents
    ]
    return f"""
You are creating an information-retrieval evaluation dataset for Jewish text search.

Given a batch of documents, create search queries and relevance judgments.
This is NOT question answering. A query should map to useful documents.

Relevance scale:
2 = highly relevant: directly satisfies the query intent.
1 = relevant: useful but partial, indirect, too broad, too narrow, or missing a key aspect.
0 = irrelevant: not useful. Do not output relevance 0 rows.

Rules:
- Create exactly {QUERIES_PER_BATCH} queries.
- Write every query in {query_language_name}.
- Query types must be one of: keyword, question, sentence.
- Each query should have at least one qrel with relevance 2.
- Only use doc_ids from the supplied documents.
- Store only relevance 1 or 2.
- Prefer realistic user search language over overly specific citation wording.
- Include a short reason for each qrel.

Return only valid JSON with this shape. Do not wrap it in markdown fences:
{{
  "queries": [
    {{
      "text": "...",
      "type": "keyword | question | sentence",
      "qrels": [
        {{"doc_id": "...", "relevance": 2, "reason": "..."}}
      ]
    }}
  ]
}}

Documents:
{json.dumps(compact_docs, ensure_ascii=False)}
""".strip()


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def call_openai_for_batch(client: OpenAI, documents: list[dict], query_lang: str) -> dict:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You create careful retrieval evaluation datasets."},
            {"role": "user", "content": make_prompt(documents, query_lang)},
        ],
    )
    return json.loads(response.choices[0].message.content)


def call_anthropic_for_batch(client: Anthropic, documents: list[dict], query_lang: str) -> dict:
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        temperature=0,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system="You create careful retrieval evaluation datasets. Return only valid JSON.",
        messages=[
            {"role": "user", "content": make_prompt(documents, query_lang)},
        ],
    )
    content = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return parse_json_response(content)


def create_llm_client():
    if LLM_PROVIDER == "anthropic":
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")), call_anthropic_for_batch
    if LLM_PROVIDER == "openai":
        return OpenAI(), call_openai_for_batch
    raise ValueError("LLM_PROVIDER must be one of: anthropic, openai.")


def get_llm_model_name() -> str:
    if LLM_PROVIDER == "anthropic":
        return ANTHROPIC_MODEL
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    return LLM_PROVIDER


def normalize_llm_output(batch_output: dict, documents: list[dict], batch_index: int, query_lang: str) -> tuple[list[dict], list[dict]]:
    valid_doc_ids = {doc["doc_id"] for doc in documents}
    queries = []
    qrels = []
    for query_index, raw_query in enumerate(batch_output.get("queries", [])):
        query_text = (raw_query.get("text") or "").strip()
        query_type = raw_query.get("type")
        if not query_text or query_type not in {"keyword", "question", "sentence"}:
            continue

        query_id = f"q_{query_lang}_{batch_index}_{query_index}"
        query = {
            "query_id": query_id,
            "text": query_text,
            "type": query_type,
            "lang": query_lang,
        }
        query_qrels = []
        for raw_qrel in raw_query.get("qrels", []):
            doc_id = raw_qrel.get("doc_id")
            relevance = raw_qrel.get("relevance")
            if doc_id not in valid_doc_ids or relevance not in {1, 2}:
                continue
            query_qrels.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": relevance,
                "reason": (raw_qrel.get("reason") or "").strip(),
            })

        if query_qrels:
            queries.append(query)
            qrels.extend(query_qrels)
    return queries, qrels


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_dataset_jsonl(documents: list[dict], queries: list[dict], qrels: list[dict]) -> None:
    write_jsonl(OUTPUT_DIR / "documents.jsonl", documents)
    write_jsonl(OUTPUT_DIR / "queries.jsonl", queries)
    write_jsonl(OUTPUT_DIR / "qrels.jsonl", qrels)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_provider": LLM_PROVIDER,
        "llm_model": get_llm_model_name(),
        "text_langs": TEXT_LANGS,
        "batch_size": BATCH_SIZE,
        "queries_per_batch": QUERIES_PER_BATCH,
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


def write_dataset_mongo(documents: list[dict], queries: list[dict], qrels: list[dict]) -> str:
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
        with mongo_collection(OUTPUT_MONGO_USE_REMOTE, OUTPUT_MONGO_DB, collection_name) as collection:
            if rows:
                collection.insert_many(rows)
    return dataset_run_id


def main() -> None:
    sampled_refs = load_sampled_refs()
    documents = build_documents(sampled_refs)

    client, call_llm_for_batch = create_llm_client()
    all_queries = []
    all_qrels = []
    batch_index = 0
    for lang, lang_documents in group_documents_by_lang(documents).items():
        for document_batch in chunk_list(lang_documents, BATCH_SIZE):
            batch_output = call_llm_for_batch(client, document_batch, lang)
            queries, qrels = normalize_llm_output(batch_output, document_batch, batch_index, lang)
            all_queries.extend(queries)
            all_qrels.extend(qrels)
            batch_index += 1
            print(f"Processed {lang} batch {batch_index}: {len(queries)} queries, {len(qrels)} qrels")

    if OUTPUT_DESTINATION == "jsonl":
        write_dataset_jsonl(documents, all_queries, all_qrels)
        print(f"Wrote dataset to {OUTPUT_DIR}")
    elif OUTPUT_DESTINATION == "mongo":
        dataset_run_id = write_dataset_mongo(documents, all_queries, all_qrels)
        print(f"Wrote dataset to Mongo with dataset_run_id={dataset_run_id}")
    else:
        raise ValueError("OUTPUT_DESTINATION must be one of: jsonl, mongo.")


if __name__ == "__main__":
    main()
