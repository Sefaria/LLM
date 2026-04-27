import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import generate_eval_queries_and_qrels as eval_gen
import sample_refdata_segment_refs as sampler
from connection_settings import CONNECTION


# ---------------------------------------------------------------------------
# Edit `ACTIVE_PRESET` to switch between saved pipeline configurations.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ACTIVE_PRESET = "sanity_check"  # sanity_check or default

PIPELINE_PRESETS = {
    "sanity_check": {
        "VERBOSE": True,
        "RUN_SAMPLING": True,
        "SAMPLE_N": 100,
        "SAMPLE_MIN_PAGESHEETRANK": 2.0,
        "SAMPLE_SEED": 613,
        "SAMPLE_USE_REMOTE": True,
        "SAMPLE_REMOTE_DB": CONNECTION["mongo_db"],
        "SAMPLE_OUTPUT_DESTINATION": "jsonl",
        "SAMPLE_OUTPUT_JSONL_PATH": BASE_DIR / "output" / "sampled_refs_sanity_check.jsonl",
        "SAMPLE_OUTPUT_MONGO_USE_REMOTE": False,
        "SAMPLE_OUTPUT_MONGO_DB": CONNECTION["mongo_db"],
        "SAMPLE_OUTPUT_MONGO_COLLECTION": "sampled_segment_refs_to_embed",
        "SAMPLE_REMOTE_READ_CACHE_ENABLED": True,
        "SAMPLE_REMOTE_READ_CACHE_PATH": BASE_DIR / "output" / "cache" / "sampled_refs_sanity_check_cache.jsonl",
        "SAMPLE_REMOTE_READ_CACHE_MODE": "candidate_pool",
        "EXISTING_SAMPLE_INPUT_SOURCE": "jsonl",
        "EXISTING_SAMPLE_JSONL_PATH": BASE_DIR / "output" / "sampled_refs_sanity_check.jsonl",
        "EXISTING_SAMPLE_MONGO_USE_REMOTE": False,
        "EXISTING_SAMPLE_MONGO_DB": CONNECTION["mongo_db"],
        "EXISTING_SAMPLE_MONGO_COLLECTION": "sampled_segment_refs_to_embed",
        "EXISTING_SAMPLE_RUN_ID": None,
        "EXISTING_SAMPLE_LIMIT": None,
        "EXISTING_SAMPLE_REMOTE_CACHE_ENABLED": False,
        "EXISTING_SAMPLE_REMOTE_CACHE_PATH": BASE_DIR / "output" / "cache" / "existing_sampled_refs_sanity_check.jsonl",
        "TEXT_LANGS": ["en", "he"],
        "TEXT_LANG_SELECTION_MODE": "prefer_he_fallback_en",
        "CONTEXT_EXPANSION_ENABLED": True,
        "CONTEXT_EXPANSION_MAX_STEPS": 3,
        "CONTEXT_EXPANSION_MAX_WORKERS": 4,
        "CONTEXT_EXPANSION_PROVIDER": "openai",
        "CONTEXT_EXPANSION_OPENAI_MODEL": "gpt-4o-mini",
        "CONTEXT_EXPANSION_ANTHROPIC_MODEL": os.getenv("ANTHROPIC_CONTEXT_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")),
        "QUERIES_PER_TYPE_PER_DOC": 3,
        "QUERY_TYPES_PER_DOC": 2,
        "QUERY_TYPE_SAMPLE_SEED": 613,
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "ANTHROPIC_MAX_TOKENS": 4096,
        "OPENAI_MODEL": "gpt-4o-mini",
        "LLM_MAX_WORKERS": 4,
        "DATASET_OUTPUT_DESTINATION": "jsonl",
        "DATASET_OUTPUT_DIR": BASE_DIR / "output" / "eval_dataset_sanity_check",
        "DATASET_OUTPUT_MONGO_USE_REMOTE": False,
        "DATASET_OUTPUT_MONGO_DB": CONNECTION["mongo_db"],
        "DATASET_OUTPUT_MONGO_PREFIX": "embedding_eval",
    },
    "default": {
        "VERBOSE": True,
        "RUN_SAMPLING": True,
        "SAMPLE_N": 100,
        "SAMPLE_MIN_PAGESHEETRANK": 1.0,
        "SAMPLE_SEED": None,
        "SAMPLE_USE_REMOTE": True,
        "SAMPLE_REMOTE_DB": CONNECTION["mongo_db"],
        "SAMPLE_OUTPUT_DESTINATION": "mongo",
        "SAMPLE_OUTPUT_JSONL_PATH": BASE_DIR / "output" / "sampled_refs.jsonl",
        "SAMPLE_OUTPUT_MONGO_USE_REMOTE": False,
        "SAMPLE_OUTPUT_MONGO_DB": CONNECTION["mongo_db"],
        "SAMPLE_OUTPUT_MONGO_COLLECTION": "sampled_segment_refs_to_embed",
        "SAMPLE_REMOTE_READ_CACHE_ENABLED": True,
        "SAMPLE_REMOTE_READ_CACHE_PATH": BASE_DIR / "output" / "cache" / "sampled_refs_cache.jsonl",
        "SAMPLE_REMOTE_READ_CACHE_MODE": "candidate_pool",
        "EXISTING_SAMPLE_INPUT_SOURCE": "mongo",
        "EXISTING_SAMPLE_JSONL_PATH": BASE_DIR / "output" / "sampled_refs.jsonl",
        "EXISTING_SAMPLE_MONGO_USE_REMOTE": False,
        "EXISTING_SAMPLE_MONGO_DB": CONNECTION["mongo_db"],
        "EXISTING_SAMPLE_MONGO_COLLECTION": "sampled_segment_refs_to_embed",
        "EXISTING_SAMPLE_RUN_ID": None,
        "EXISTING_SAMPLE_LIMIT": None,
        "EXISTING_SAMPLE_REMOTE_CACHE_ENABLED": False,
        "EXISTING_SAMPLE_REMOTE_CACHE_PATH": BASE_DIR / "output" / "cache" / "existing_sampled_refs.jsonl",
        "TEXT_LANGS": ["en", "he"],
        "TEXT_LANG_SELECTION_MODE": "prefer_he_fallback_en",
        "CONTEXT_EXPANSION_ENABLED": True,
        "CONTEXT_EXPANSION_MAX_STEPS": 3,
        "CONTEXT_EXPANSION_MAX_WORKERS": 4,
        "CONTEXT_EXPANSION_PROVIDER": "openai",
        "CONTEXT_EXPANSION_OPENAI_MODEL": "gpt-4o-mini",
        "CONTEXT_EXPANSION_ANTHROPIC_MODEL": os.getenv("ANTHROPIC_CONTEXT_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")),
        "QUERIES_PER_TYPE_PER_DOC": 3,
        "QUERY_TYPES_PER_DOC": 2,
        "QUERY_TYPE_SAMPLE_SEED": 613,
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "ANTHROPIC_MAX_TOKENS": 4096,
        "OPENAI_MODEL": "gpt-4o-mini",
        "LLM_MAX_WORKERS": 4,
        "DATASET_OUTPUT_DESTINATION": "jsonl",
        "DATASET_OUTPUT_DIR": BASE_DIR / "output" / "eval_dataset",
        "DATASET_OUTPUT_MONGO_USE_REMOTE": False,
        "DATASET_OUTPUT_MONGO_DB": CONNECTION["mongo_db"],
        "DATASET_OUTPUT_MONGO_PREFIX": "embedding_eval",
    },
}

CONFIG = deepcopy(PIPELINE_PRESETS[ACTIVE_PRESET])

VERBOSE = CONFIG["VERBOSE"]
RUN_SAMPLING = CONFIG["RUN_SAMPLING"]
SAMPLE_N = CONFIG["SAMPLE_N"]
SAMPLE_MIN_PAGESHEETRANK = CONFIG["SAMPLE_MIN_PAGESHEETRANK"]
SAMPLE_SEED = CONFIG["SAMPLE_SEED"]
SAMPLE_USE_REMOTE = CONFIG["SAMPLE_USE_REMOTE"]
SAMPLE_REMOTE_DB = CONFIG["SAMPLE_REMOTE_DB"]
SAMPLE_OUTPUT_DESTINATION = CONFIG["SAMPLE_OUTPUT_DESTINATION"]
SAMPLE_OUTPUT_JSONL_PATH = CONFIG["SAMPLE_OUTPUT_JSONL_PATH"]
SAMPLE_OUTPUT_MONGO_USE_REMOTE = CONFIG["SAMPLE_OUTPUT_MONGO_USE_REMOTE"]
SAMPLE_OUTPUT_MONGO_DB = CONFIG["SAMPLE_OUTPUT_MONGO_DB"]
SAMPLE_OUTPUT_MONGO_COLLECTION = CONFIG["SAMPLE_OUTPUT_MONGO_COLLECTION"]
SAMPLE_REMOTE_READ_CACHE_ENABLED = CONFIG["SAMPLE_REMOTE_READ_CACHE_ENABLED"]
SAMPLE_REMOTE_READ_CACHE_PATH = CONFIG["SAMPLE_REMOTE_READ_CACHE_PATH"]
SAMPLE_REMOTE_READ_CACHE_MODE = CONFIG["SAMPLE_REMOTE_READ_CACHE_MODE"]
EXISTING_SAMPLE_INPUT_SOURCE = CONFIG["EXISTING_SAMPLE_INPUT_SOURCE"]
EXISTING_SAMPLE_JSONL_PATH = CONFIG["EXISTING_SAMPLE_JSONL_PATH"]
EXISTING_SAMPLE_MONGO_USE_REMOTE = CONFIG["EXISTING_SAMPLE_MONGO_USE_REMOTE"]
EXISTING_SAMPLE_MONGO_DB = CONFIG["EXISTING_SAMPLE_MONGO_DB"]
EXISTING_SAMPLE_MONGO_COLLECTION = CONFIG["EXISTING_SAMPLE_MONGO_COLLECTION"]
EXISTING_SAMPLE_RUN_ID = CONFIG["EXISTING_SAMPLE_RUN_ID"]
EXISTING_SAMPLE_LIMIT = CONFIG["EXISTING_SAMPLE_LIMIT"]
EXISTING_SAMPLE_REMOTE_CACHE_ENABLED = CONFIG["EXISTING_SAMPLE_REMOTE_CACHE_ENABLED"]
EXISTING_SAMPLE_REMOTE_CACHE_PATH = CONFIG["EXISTING_SAMPLE_REMOTE_CACHE_PATH"]
TEXT_LANGS = CONFIG["TEXT_LANGS"]
TEXT_LANG_SELECTION_MODE = CONFIG["TEXT_LANG_SELECTION_MODE"]
CONTEXT_EXPANSION_ENABLED = CONFIG["CONTEXT_EXPANSION_ENABLED"]
CONTEXT_EXPANSION_MAX_STEPS = CONFIG["CONTEXT_EXPANSION_MAX_STEPS"]
CONTEXT_EXPANSION_MAX_WORKERS = CONFIG["CONTEXT_EXPANSION_MAX_WORKERS"]
CONTEXT_EXPANSION_PROVIDER = CONFIG["CONTEXT_EXPANSION_PROVIDER"]
CONTEXT_EXPANSION_OPENAI_MODEL = CONFIG["CONTEXT_EXPANSION_OPENAI_MODEL"]
CONTEXT_EXPANSION_ANTHROPIC_MODEL = CONFIG["CONTEXT_EXPANSION_ANTHROPIC_MODEL"]
QUERIES_PER_TYPE_PER_DOC = CONFIG["QUERIES_PER_TYPE_PER_DOC"]
QUERY_TYPES_PER_DOC = CONFIG["QUERY_TYPES_PER_DOC"]
QUERY_TYPE_SAMPLE_SEED = CONFIG["QUERY_TYPE_SAMPLE_SEED"]
LLM_PROVIDER = CONFIG["LLM_PROVIDER"]
ANTHROPIC_MODEL = CONFIG["ANTHROPIC_MODEL"]
ANTHROPIC_MAX_TOKENS = CONFIG["ANTHROPIC_MAX_TOKENS"]
OPENAI_MODEL = CONFIG["OPENAI_MODEL"]
LLM_MAX_WORKERS = CONFIG["LLM_MAX_WORKERS"]
DATASET_OUTPUT_DESTINATION = CONFIG["DATASET_OUTPUT_DESTINATION"]
DATASET_OUTPUT_DIR = CONFIG["DATASET_OUTPUT_DIR"]
DATASET_OUTPUT_MONGO_USE_REMOTE = CONFIG["DATASET_OUTPUT_MONGO_USE_REMOTE"]
DATASET_OUTPUT_MONGO_DB = CONFIG["DATASET_OUTPUT_MONGO_DB"]
DATASET_OUTPUT_MONGO_PREFIX = CONFIG["DATASET_OUTPUT_MONGO_PREFIX"]


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def configure_sampler() -> None:
    sampler.VERBOSE = VERBOSE
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
    sampler.REMOTE_READ_CACHE_ENABLED = SAMPLE_REMOTE_READ_CACHE_ENABLED
    sampler.REMOTE_READ_CACHE_PATH = SAMPLE_REMOTE_READ_CACHE_PATH
    sampler.REMOTE_READ_CACHE_MODE = SAMPLE_REMOTE_READ_CACHE_MODE


def configure_eval_generator() -> None:
    eval_gen.VERBOSE = VERBOSE
    eval_gen.TEXT_LANGS = TEXT_LANGS
    eval_gen.TEXT_LANG_SELECTION_MODE = TEXT_LANG_SELECTION_MODE
    eval_gen.CONTEXT_EXPANSION_ENABLED = CONTEXT_EXPANSION_ENABLED
    eval_gen.CONTEXT_EXPANSION_MAX_STEPS = CONTEXT_EXPANSION_MAX_STEPS
    eval_gen.CONTEXT_EXPANSION_MAX_WORKERS = CONTEXT_EXPANSION_MAX_WORKERS
    eval_gen.CONTEXT_EXPANSION_PROVIDER = CONTEXT_EXPANSION_PROVIDER
    eval_gen.CONTEXT_EXPANSION_OPENAI_MODEL = CONTEXT_EXPANSION_OPENAI_MODEL
    eval_gen.CONTEXT_EXPANSION_ANTHROPIC_MODEL = CONTEXT_EXPANSION_ANTHROPIC_MODEL
    eval_gen.QUERIES_PER_TYPE_PER_DOC = QUERIES_PER_TYPE_PER_DOC
    eval_gen.QUERY_TYPES_PER_DOC = QUERY_TYPES_PER_DOC
    eval_gen.QUERY_TYPE_SAMPLE_SEED = QUERY_TYPE_SAMPLE_SEED
    eval_gen.LLM_PROVIDER = LLM_PROVIDER
    eval_gen.ANTHROPIC_MODEL = ANTHROPIC_MODEL
    eval_gen.ANTHROPIC_MAX_TOKENS = ANTHROPIC_MAX_TOKENS
    eval_gen.OPENAI_MODEL = OPENAI_MODEL
    eval_gen.LLM_MAX_WORKERS = LLM_MAX_WORKERS
    eval_gen.OUTPUT_DESTINATION = DATASET_OUTPUT_DESTINATION
    eval_gen.OUTPUT_DIR = DATASET_OUTPUT_DIR
    eval_gen.OUTPUT_MONGO_USE_REMOTE = DATASET_OUTPUT_MONGO_USE_REMOTE
    eval_gen.OUTPUT_MONGO_DB = DATASET_OUTPUT_MONGO_DB
    eval_gen.OUTPUT_MONGO_PREFIX = DATASET_OUTPUT_MONGO_PREFIX


def sample_refs() -> tuple[list[dict], Optional[str]]:
    configure_sampler()
    start = time.perf_counter()
    log(
        f"Sampling {SAMPLE_N} refs from {'remote' if SAMPLE_USE_REMOTE else 'local'} "
        f"Mongo with pagesheetrank > {SAMPLE_MIN_PAGESHEETRANK}"
    )
    rows = sampler.sample_refs(SAMPLE_N, SAMPLE_MIN_PAGESHEETRANK, SAMPLE_SEED, SAMPLE_USE_REMOTE, SAMPLE_REMOTE_DB)
    log(f"Sampled {len(rows)} refs in {time.perf_counter() - start:.2f}s")

    sample_run_id = None
    if SAMPLE_OUTPUT_DESTINATION == "jsonl":
        sampler.write_jsonl(rows, Path(SAMPLE_OUTPUT_JSONL_PATH))
        log(f"Wrote {len(rows)} sampled refs to {SAMPLE_OUTPUT_JSONL_PATH}")
    elif SAMPLE_OUTPUT_DESTINATION == "mongo":
        sample_run_id = sampler.write_mongo(
            rows,
            SAMPLE_OUTPUT_MONGO_USE_REMOTE,
            SAMPLE_OUTPUT_MONGO_DB,
            SAMPLE_OUTPUT_MONGO_COLLECTION,
        )
        location = "remote" if SAMPLE_OUTPUT_MONGO_USE_REMOTE else "local"
        target = f"{SAMPLE_OUTPUT_MONGO_DB}.{SAMPLE_OUTPUT_MONGO_COLLECTION}"
        log(f"Wrote {len(rows)} sampled refs to {location} Mongo collection {target} with sample_run_id={sample_run_id}")
    elif SAMPLE_OUTPUT_DESTINATION == "none":
        log(f"Sampled {len(rows)} refs without writing an intermediate output")
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
    eval_gen.INPUT_REMOTE_CACHE_ENABLED = EXISTING_SAMPLE_REMOTE_CACHE_ENABLED
    eval_gen.INPUT_REMOTE_CACHE_PATH = EXISTING_SAMPLE_REMOTE_CACHE_PATH
    start = time.perf_counter()
    rows = eval_gen.load_sampled_refs()
    log(f"Loaded {len(rows)} existing sampled refs in {time.perf_counter() - start:.2f}s")
    return rows


def generate_dataset(sampled_refs: list[dict]) -> None:
    configure_eval_generator()

    start = time.perf_counter()
    log(
        f"Building documents for languages: {', '.join(TEXT_LANGS)} "
        f"with selection mode {TEXT_LANG_SELECTION_MODE}"
    )
    documents = eval_gen.build_documents(sampled_refs)
    log(f"Built {len(documents)} documents in {time.perf_counter() - start:.2f}s")

    log(f"Using {LLM_PROVIDER} model {eval_gen.get_llm_model_name()}")
    all_queries, all_qrels = eval_gen.generate_queries_and_qrels(documents)

    write_start = time.perf_counter()
    if DATASET_OUTPUT_DESTINATION == "jsonl":
        eval_gen.write_dataset_jsonl(documents, all_queries, all_qrels)
        log(f"Wrote dataset to {DATASET_OUTPUT_DIR} in {time.perf_counter() - write_start:.2f}s")
    elif DATASET_OUTPUT_DESTINATION == "mongo":
        dataset_run_id = eval_gen.write_dataset_mongo(documents, all_queries, all_qrels)
        log(f"Wrote dataset to Mongo with dataset_run_id={dataset_run_id} in {time.perf_counter() - write_start:.2f}s")
    else:
        raise ValueError("DATASET_OUTPUT_DESTINATION must be one of: jsonl, mongo.")


def main() -> None:
    pipeline_start = time.perf_counter()
    log(f"Starting pipeline preset '{ACTIVE_PRESET}'")
    if RUN_SAMPLING:
        sampled_refs, _sample_run_id = sample_refs()
    else:
        sampled_refs = load_existing_sampled_refs()

    generate_dataset(sampled_refs)
    log(f"Pipeline completed in {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
