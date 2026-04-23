import os
from pathlib import Path
from typing import Optional

import generate_eval_queries_and_qrels as eval_gen
import sample_refdata_segment_refs as sampler
from connection_settings import CONNECTION


# ---------------------------------------------------------------------------
# Edit all pipeline parameters here.
# ---------------------------------------------------------------------------

RUN_SAMPLING = True

# Sampling input.
SAMPLE_N = 100
SAMPLE_MIN_PAGESHEETRANK = 1.0
SAMPLE_SEED = None
SAMPLE_USE_REMOTE = True
SAMPLE_REMOTE_DB = CONNECTION["mongo_db"]

# Optional sampled-ref intermediate output.
# Use "none", "jsonl", or "mongo".
SAMPLE_OUTPUT_DESTINATION = "mongo"
SAMPLE_OUTPUT_JSONL_PATH = Path(__file__).resolve().parent / "output" / "sampled_refs.jsonl"
SAMPLE_OUTPUT_MONGO_USE_REMOTE = False
SAMPLE_OUTPUT_MONGO_DB = CONNECTION["mongo_db"]
SAMPLE_OUTPUT_MONGO_COLLECTION = "sampled_segment_refs_to_embed"

# If RUN_SAMPLING is False, load sampled refs from here.
# Use "jsonl" or "mongo".
EXISTING_SAMPLE_INPUT_SOURCE = "mongo"
EXISTING_SAMPLE_JSONL_PATH = SAMPLE_OUTPUT_JSONL_PATH
EXISTING_SAMPLE_MONGO_USE_REMOTE = False
EXISTING_SAMPLE_MONGO_DB = CONNECTION["mongo_db"]
EXISTING_SAMPLE_MONGO_COLLECTION = "sampled_segment_refs_to_embed"
EXISTING_SAMPLE_RUN_ID = None
EXISTING_SAMPLE_LIMIT = None

# Document text and LLM generation.
TEXT_LANGS = ["en", "he"]
BATCH_SIZE = 8
QUERIES_PER_BATCH = 12
LLM_PROVIDER = "anthropic"  # anthropic or openai
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
ANTHROPIC_MAX_TOKENS = 4096
OPENAI_MODEL = "gpt-4o-mini"

# Final dataset output.
# Use "jsonl" or "mongo".
DATASET_OUTPUT_DESTINATION = "jsonl"
DATASET_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval_dataset"
DATASET_OUTPUT_MONGO_USE_REMOTE = False
DATASET_OUTPUT_MONGO_DB = CONNECTION["mongo_db"]
DATASET_OUTPUT_MONGO_PREFIX = "embedding_eval"


def configure_sampler() -> None:
    sampler.N = SAMPLE_N
    sampler.MIN_PAGESHEETRANK = SAMPLE_MIN_PAGESHEETRANK
    sampler.SEED = SAMPLE_SEED
    sampler.USE_REMOTE = SAMPLE_USE_REMOTE
    sampler.REMOTE_DB = SAMPLE_REMOTE_DB
    sampler.OUTPUT_DESTINATION = SAMPLE_OUTPUT_DESTINATION
    sampler.OUTPUT_PATH = SAMPLE_OUTPUT_JSONL_PATH
    sampler.OUTPUT_MONGO_USE_REMOTE = SAMPLE_OUTPUT_MONGO_USE_REMOTE
    sampler.OUTPUT_MONGO_DB = SAMPLE_OUTPUT_MONGO_DB
    sampler.OUTPUT_MONGO_COLLECTION = SAMPLE_OUTPUT_MONGO_COLLECTION


def configure_eval_generator() -> None:
    eval_gen.TEXT_LANGS = TEXT_LANGS
    eval_gen.BATCH_SIZE = BATCH_SIZE
    eval_gen.QUERIES_PER_BATCH = QUERIES_PER_BATCH
    eval_gen.LLM_PROVIDER = LLM_PROVIDER
    eval_gen.ANTHROPIC_MODEL = ANTHROPIC_MODEL
    eval_gen.ANTHROPIC_MAX_TOKENS = ANTHROPIC_MAX_TOKENS
    eval_gen.OPENAI_MODEL = OPENAI_MODEL
    eval_gen.OUTPUT_DESTINATION = DATASET_OUTPUT_DESTINATION
    eval_gen.OUTPUT_DIR = DATASET_OUTPUT_DIR
    eval_gen.OUTPUT_MONGO_USE_REMOTE = DATASET_OUTPUT_MONGO_USE_REMOTE
    eval_gen.OUTPUT_MONGO_DB = DATASET_OUTPUT_MONGO_DB
    eval_gen.OUTPUT_MONGO_PREFIX = DATASET_OUTPUT_MONGO_PREFIX


def sample_refs() -> tuple[list[dict], Optional[str]]:
    configure_sampler()
    rows = sampler.sample_refs(SAMPLE_N, SAMPLE_MIN_PAGESHEETRANK, SAMPLE_SEED, SAMPLE_USE_REMOTE, SAMPLE_REMOTE_DB)

    sample_run_id = None
    if SAMPLE_OUTPUT_DESTINATION == "jsonl":
        sampler.write_jsonl(rows, Path(SAMPLE_OUTPUT_JSONL_PATH))
        print(f"Wrote {len(rows)} sampled refs to {SAMPLE_OUTPUT_JSONL_PATH}")
    elif SAMPLE_OUTPUT_DESTINATION == "mongo":
        sample_run_id = sampler.write_mongo(
            rows,
            SAMPLE_OUTPUT_MONGO_USE_REMOTE,
            SAMPLE_OUTPUT_MONGO_DB,
            SAMPLE_OUTPUT_MONGO_COLLECTION,
        )
        location = "remote" if SAMPLE_OUTPUT_MONGO_USE_REMOTE else "local"
        target = f"{SAMPLE_OUTPUT_MONGO_DB}.{SAMPLE_OUTPUT_MONGO_COLLECTION}"
        print(f"Wrote {len(rows)} sampled refs to {location} Mongo collection {target} with sample_run_id={sample_run_id}")
    elif SAMPLE_OUTPUT_DESTINATION == "none":
        print(f"Sampled {len(rows)} refs without writing an intermediate output")
    else:
        raise ValueError("SAMPLE_OUTPUT_DESTINATION must be one of: none, jsonl, mongo.")

    return rows, sample_run_id


def load_existing_sampled_refs() -> list[dict]:
    eval_gen.INPUT_SOURCE = EXISTING_SAMPLE_INPUT_SOURCE
    eval_gen.INPUT_JSONL_PATH = EXISTING_SAMPLE_JSONL_PATH
    eval_gen.INPUT_MONGO_USE_REMOTE = EXISTING_SAMPLE_MONGO_USE_REMOTE
    eval_gen.INPUT_MONGO_DB = EXISTING_SAMPLE_MONGO_DB
    eval_gen.INPUT_MONGO_COLLECTION = EXISTING_SAMPLE_MONGO_COLLECTION
    eval_gen.INPUT_SAMPLE_RUN_ID = EXISTING_SAMPLE_RUN_ID
    eval_gen.LIMIT = EXISTING_SAMPLE_LIMIT
    return eval_gen.load_sampled_refs()


def generate_dataset(sampled_refs: list[dict]) -> None:
    configure_eval_generator()

    documents = eval_gen.build_documents(sampled_refs)
    client, call_llm_for_batch = eval_gen.create_llm_client()

    all_queries = []
    all_qrels = []
    batch_index = 0
    for lang, lang_documents in eval_gen.group_documents_by_lang(documents).items():
        for document_batch in eval_gen.chunk_list(lang_documents, BATCH_SIZE):
            batch_output = call_llm_for_batch(client, document_batch, lang)
            queries, qrels = eval_gen.normalize_llm_output(batch_output, document_batch, batch_index, lang)
            all_queries.extend(queries)
            all_qrels.extend(qrels)
            batch_index += 1
            print(f"Processed {lang} batch {batch_index}: {len(queries)} queries, {len(qrels)} qrels")

    if DATASET_OUTPUT_DESTINATION == "jsonl":
        eval_gen.write_dataset_jsonl(documents, all_queries, all_qrels)
        print(f"Wrote dataset to {DATASET_OUTPUT_DIR}")
    elif DATASET_OUTPUT_DESTINATION == "mongo":
        dataset_run_id = eval_gen.write_dataset_mongo(documents, all_queries, all_qrels)
        print(f"Wrote dataset to Mongo with dataset_run_id={dataset_run_id}")
    else:
        raise ValueError("DATASET_OUTPUT_DESTINATION must be one of: jsonl, mongo.")


def main() -> None:
    if RUN_SAMPLING:
        sampled_refs, _sample_run_id = sample_refs()
    else:
        sampled_refs = load_existing_sampled_refs()
        print(f"Loaded {len(sampled_refs)} existing sampled refs")

    generate_dataset(sampled_refs)


if __name__ == "__main__":
    main()
