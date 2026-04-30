import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from embedding_eval.cache import EmbeddingCache
    from embedding_eval.data_loader import load_dataset
    from embedding_eval.experiment_grid import build_experiment_grid, config_run_id, config_to_dict
    from embedding_eval.gemini_embedder import EmbeddingError, GeminiEmbedder, l2_normalize_vector
    from embedding_eval.metrics import compute_first_relevant_rank, compute_mrr, compute_ndcg, compute_recall
    from embedding_eval.ranking import l2_normalize, rank_documents
    from embedding_eval.reports import write_breakdowns_csv, write_jsonl, write_run_config, write_summary_csv
    from embedding_eval.text_variants import render_document_text, render_query_text, strip_hebrew_diacritics
else:
    from .cache import EmbeddingCache
    from .data_loader import load_dataset
    from .experiment_grid import build_experiment_grid, config_run_id, config_to_dict
    from .gemini_embedder import EmbeddingError, GeminiEmbedder, l2_normalize_vector
    from .metrics import compute_first_relevant_rank, compute_mrr, compute_ndcg, compute_recall
    from .ranking import l2_normalize, rank_documents
    from .reports import write_breakdowns_csv, write_jsonl, write_run_config, write_summary_csv
    from .text_variants import render_document_text, render_query_text, strip_hebrew_diacritics


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "data_generation" / "output" / "eval_dataset_sanity_check"
OUTPUT_DIR = BASE_DIR / "output" / "gemini_eval"
CACHE_PATH = OUTPUT_DIR / "embedding_cache.sqlite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
INCLUDE_GEMINI_EMBEDDING_2_IF_AVAILABLE = True
INCLUDE_CONFOUNDERS = True
FLUSH_EMBEDDING_CACHE = False
STRIP_NIQQUD = True
VERBOSE = True
EMBEDDING_MAX_WORKERS = 4
ADAPTIVE_WORKERS_ENABLED = True
ADAPTIVE_WORKERS_MIN = 1
ADAPTIVE_WORKERS_MAX = 6
ADAPTIVE_WORKERS_SUCCESS_STREAK_FOR_INCREASE = 3
K_VALUES = [1, 3, 5, 10, 20, 50]


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def progress(iterable, **kwargs):
    if VERBOSE:
        return tqdm(iterable, **kwargs)
    return iterable


def make_mode_cache_string(config, is_query: bool) -> str:
    side = "query" if is_query else "doc"
    task_type = config.query_task_type if is_query else config.doc_task_type
    return f"{side}|setup={config.embedding_task_setup}|task_type={task_type}"


def prepare_text(text: str) -> str:
    if STRIP_NIQQUD:
        text = strip_hebrew_diacritics(text)
    return text


def embed_text_with_cache(embedder: GeminiEmbedder, cache: EmbeddingCache, model: str, text: str, output_dimensionality: int, mode_string: str, task_type: object = None) -> list[float]:
    cached = cache.get(model, output_dimensionality, mode_string, text)
    if cached is not None:
        return cached
    vector = embedder.embed_text(
        model=model,
        text=text,
        output_dimensionality=output_dimensionality,
        task_type=task_type,
    )
    cache.put(model, output_dimensionality, mode_string, text, vector)
    return vector


def embed_items_parallel(items: list[dict], worker_fn, desc: str, max_workers: int) -> tuple[list[dict], list[dict]]:
    successes = []
    failures = []
    progress_bar = tqdm(total=len(items), desc=desc, leave=False) if VERBOSE else None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                successes.append(worker_fn_result(future.result(), item))
            except EmbeddingError as exc:
                failures.append({"item": item, "error": str(exc)})
            finally:
                if progress_bar is not None:
                    progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    return successes, failures


def worker_fn_result(result: dict, item: dict) -> dict:
    return {**item, **result}


def is_rate_limit_failure(error_text: str) -> bool:
    lowered = error_text.lower()
    return "429" in lowered or "rate limit" in lowered or "resource_exhausted" in lowered


def aggregate_metric(values: list[float]) -> float:
    if not values:
        return 0.0
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("inf")
    return sum(finite_values) / len(finite_values)


def compute_query_metrics(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> dict:
    metrics = {}
    for k in K_VALUES:
        metrics[f"ndcg@{k}"] = compute_ndcg(ranked_doc_ids, relevant_doc_ids, k)
    for k in K_VALUES:
        metrics[f"recall@{k}"] = compute_recall(ranked_doc_ids, relevant_doc_ids, k)
    for k in K_VALUES:
        metrics[f"mrr@{k}"] = compute_mrr(ranked_doc_ids, relevant_doc_ids, k)
    metrics["mean_first_relevant_rank"] = compute_first_relevant_rank(ranked_doc_ids, relevant_doc_ids)
    return metrics


def summarize_group(rows: list[dict]) -> dict:
    summary = {}
    metric_names = [
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "ndcg@50",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "mrr@1",
        "mrr@3",
        "mrr@5",
        "mrr@10",
        "mrr@20",
        "mrr@50",
        "mean_first_relevant_rank",
    ]
    for metric_name in metric_names:
        summary[metric_name] = aggregate_metric([row["metrics"][metric_name] for row in rows])
    summary["query_count"] = len(rows)
    return summary


def maybe_normalize_embeddings(config, matrix: np.ndarray) -> np.ndarray:
    if config.normalize_embeddings:
        return l2_normalize(matrix)
    return matrix


def build_report_row(config, run_id: str, summary: dict) -> dict:
    return {
        "run_id": run_id,
        "model": config.model,
        "embedding_task_setup": config.embedding_task_setup,
        "output_dimensionality": config.output_dimensionality,
        "doc_text_variant": config.doc_text_variant,
        "query_text_variant": config.query_text_variant,
        "similarity_metric": config.similarity_metric,
        "normalize_embeddings": config.normalize_embeddings,
        "ndcg@1": summary["ndcg@1"],
        "ndcg@3": summary["ndcg@3"],
        "ndcg@5": summary["ndcg@5"],
        "ndcg@10": summary["ndcg@10"],
        "ndcg@20": summary["ndcg@20"],
        "recall@10": summary["recall@10"],
        "recall@50": summary["recall@50"],
        "mrr@10": summary["mrr@10"],
        "mean_first_relevant_rank": summary["mean_first_relevant_rank"],
    }


def pareto_front(rows: list[dict], metric_names: list[str]) -> list[dict]:
    front = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            greater_or_equal = all(other[name] >= row[name] for name in metric_names)
            strictly_greater = any(other[name] > row[name] for name in metric_names)
            if greater_or_equal and strictly_greater:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def run() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running embedding evaluation.")

    log(f"Loading dataset from {DATASET_DIR}")
    dataset = load_dataset(DATASET_DIR, include_confounders=INCLUDE_CONFOUNDERS)
    log(
        f"Loaded {len(dataset['documents'])} documents "
        f"({len(dataset['judged_documents'])} judged + {len(dataset['confounding_documents'])} confounders), "
        f"{len(dataset['queries'])} queries, "
        f"{len(dataset['relevant_doc_ids_by_query'])} judged queries"
    )
    embedder = GeminiEmbedder(api_key=GEMINI_API_KEY)
    cache = EmbeddingCache(CACHE_PATH)
    if FLUSH_EMBEDDING_CACHE:
        cache.clear()
        log(f"Flushed embedding cache at {CACHE_PATH}")

    include_embedding_2 = False
    available_models = []
    try:
        available_models = embedder.list_models()
        include_embedding_2 = INCLUDE_GEMINI_EMBEDDING_2_IF_AVAILABLE and "gemini-embedding-2" in available_models
    except Exception as exc:
        log(f"Model discovery failed: {exc}. Continuing with gemini-embedding-001 only.")

    configs = build_experiment_grid(include_gemini_embedding_2=include_embedding_2)
    log(f"Running {len(configs)} hyperparameter configurations")
    summary_rows = []
    per_query_rows = []
    breakdown_rows = []
    failures = []
    current_workers = EMBEDDING_MAX_WORKERS
    success_streak = 0

    config_iter = progress(
        enumerate(configs, start=1),
        total=len(configs),
        desc="Configs",
    )
    for config_index, config in config_iter:
        run_id = config_run_id(config)
        log(
            f"[{config_index}/{len(configs)}] Running {run_id} "
            f"(workers={current_workers}) {json.dumps(config_to_dict(config), sort_keys=True)}"
        )

        def embed_document_worker(document: dict) -> dict:
            final_text = prepare_text(
                render_document_text(document, config.doc_text_variant),
            )
            vector = embed_text_with_cache(
                embedder=embedder,
                cache=cache,
                model=config.model,
                text=final_text,
                output_dimensionality=config.output_dimensionality,
                mode_string=make_mode_cache_string(config, is_query=False),
                task_type=config.doc_task_type,
            )
            if config.model == "gemini-embedding-001" and config.output_dimensionality != 3072:
                vector = l2_normalize_vector(vector)
            return {"vector": vector}

        doc_successes, doc_failures = embed_items_parallel(
            dataset["documents"],
            embed_document_worker,
            desc=f"Docs {run_id}",
            max_workers=current_workers,
        )
        for failure in doc_failures:
            failures.append(
                {
                    "run_id": run_id,
                    "kind": "document",
                    "doc_id": failure["item"]["doc_id"],
                    "error": failure["error"],
                }
            )

        doc_successes_by_id = {row["doc_id"]: row["vector"] for row in doc_successes}
        successful_doc_ids = [
            document["doc_id"]
            for document in dataset["documents"]
            if document["doc_id"] in doc_successes_by_id
        ]
        doc_vectors = [doc_successes_by_id[doc_id] for doc_id in successful_doc_ids]

        if not doc_vectors:
            continue

        doc_matrix = np.array(doc_vectors, dtype=float)
        doc_matrix = maybe_normalize_embeddings(config, doc_matrix)

        if doc_matrix.shape[1] != config.output_dimensionality:
            raise ValueError(f"Document embedding dimensions mismatch for {run_id}: expected {config.output_dimensionality}, got {doc_matrix.shape[1]}")

        judged_queries = [query for query in dataset["queries"] if dataset["relevant_doc_ids_by_query"].get(query["query_id"])]
        def embed_query_worker(query: dict) -> dict:
            relevant_doc_ids = dataset["relevant_doc_ids_by_query"].get(query["query_id"])
            final_query_text = prepare_text(
                render_query_text(query, config.query_text_variant),
            )
            query_vector = embed_text_with_cache(
                embedder=embedder,
                cache=cache,
                model=config.model,
                text=final_query_text,
                output_dimensionality=config.output_dimensionality,
                mode_string=make_mode_cache_string(config, is_query=True),
                task_type=config.query_task_type,
            )
            if config.model == "gemini-embedding-001" and config.output_dimensionality != 3072:
                query_vector = l2_normalize_vector(query_vector)
            return {
                "vector": query_vector,
                "relevant_doc_ids": relevant_doc_ids,
            }

        query_successes, query_failures = embed_items_parallel(
            judged_queries,
            embed_query_worker,
            desc=f"Queries {run_id}",
            max_workers=current_workers,
        )
        for failure in query_failures:
            failures.append(
                {
                    "run_id": run_id,
                    "kind": "query",
                    "query_id": failure["item"]["query_id"],
                    "error": failure["error"],
                }
            )

        rate_limit_failures = [
            failure for failure in (doc_failures + query_failures)
            if is_rate_limit_failure(failure["error"])
        ]
        if ADAPTIVE_WORKERS_ENABLED:
            if rate_limit_failures:
                new_workers = max(ADAPTIVE_WORKERS_MIN, current_workers - 1)
                if new_workers != current_workers:
                    log(
                        f"{run_id} saw {len(rate_limit_failures)} rate-limit failures; "
                        f"reducing workers {current_workers} -> {new_workers}"
                    )
                current_workers = new_workers
                success_streak = 0
            else:
                success_streak += 1
                if (
                    success_streak >= ADAPTIVE_WORKERS_SUCCESS_STREAK_FOR_INCREASE
                    and current_workers < ADAPTIVE_WORKERS_MAX
                ):
                    new_workers = current_workers + 1
                    log(
                        f"{run_id} had {success_streak} clean configs; "
                        f"increasing workers {current_workers} -> {new_workers}"
                    )
                    current_workers = new_workers
                    success_streak = 0

        query_successes_by_id = {row["query_id"]: row for row in query_successes}
        run_query_rows = []
        for query in judged_queries:
            row = query_successes_by_id.get(query["query_id"])
            if row is None:
                continue
            relevant_doc_ids = row["relevant_doc_ids"]
            query_vector = row["vector"]

            query_array = np.array(query_vector, dtype=float)
            if config.normalize_embeddings:
                norm = np.linalg.norm(query_array)
                if norm != 0:
                    query_array = query_array / norm

            if query_array.shape[0] != doc_matrix.shape[1]:
                raise ValueError(
                    f"Query/document dimension mismatch for {run_id}: "
                    f"query={query_array.shape[0]} docs={doc_matrix.shape[1]}"
                )

            ranked = rank_documents(query_array, successful_doc_ids, doc_matrix, config.similarity_metric)
            ranked_doc_ids = [doc_id for doc_id, _score in ranked]
            metrics = compute_query_metrics(ranked_doc_ids, relevant_doc_ids)
            top_10 = [
                {
                    "doc_id": doc_id,
                    "score": score,
                    "rank": rank,
                    "is_relevant": doc_id in relevant_doc_ids,
                }
                for rank, (doc_id, score) in enumerate(ranked[:10], start=1)
            ]
            query_row = {
                "run_id": run_id,
                "query_id": query["query_id"],
                "query_text": query.get("text", ""),
                "query_type": query.get("type"),
                "query_lang": query.get("lang"),
                "query_category": dataset["query_category_by_id"].get(query["query_id"], "unknown"),
                "positive_doc_ids": sorted(relevant_doc_ids),
                "top_10": top_10,
                "metrics": metrics,
            }
            run_query_rows.append(query_row)

        if not run_query_rows:
            log(f"{run_id} produced no query results")
            continue

        summary = summarize_group(run_query_rows)
        log(
            f"{run_id} summary: ndcg@10={summary['ndcg@10']:.4f} "
            f"recall@10={summary['recall@10']:.4f} mrr@10={summary['mrr@10']:.4f} "
            f"queries={summary['query_count']}"
        )
        summary_rows.append(build_report_row(config, run_id, summary))
        per_query_rows.extend(run_query_rows)

        grouping_specs = [
            ("overall", lambda row: "overall"),
            ("query_type", lambda row: row["query_type"] or "unknown"),
            ("query_lang", lambda row: row["query_lang"] or "unknown"),
            ("document_category", lambda row: row["query_category"] or "unknown"),
        ]
        for grouping_name, key_fn in grouping_specs:
            grouped = defaultdict(list)
            for row in run_query_rows:
                grouped[key_fn(row)].append(row)
            for group_value, group_rows in grouped.items():
                group_summary = summarize_group(group_rows)
                breakdown_rows.append({
                    "run_id": run_id,
                    "grouping": grouping_name,
                    "group_value": group_value,
                    **build_report_row(config, run_id, group_summary),
                    "query_count": len(group_rows),
                })

    summary_rows.sort(key=lambda row: row["ndcg@10"], reverse=True)

    write_summary_csv(OUTPUT_DIR / "results_summary.csv", summary_rows)
    write_jsonl(OUTPUT_DIR / "per_query_results.jsonl", per_query_rows)
    write_breakdowns_csv(OUTPUT_DIR / "results_breakdowns.csv", breakdown_rows)
    write_run_config(
        OUTPUT_DIR / "run_config.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": str(DATASET_DIR),
            "output_dir": str(OUTPUT_DIR),
            "cache_path": str(CACHE_PATH),
            "strip_niqqud": STRIP_NIQQUD,
            "include_confounders": INCLUDE_CONFOUNDERS,
            "judged_documents_count": len(dataset["judged_documents"]),
            "confounding_documents_count": len(dataset["confounding_documents"]),
            "documents_count": len(dataset["documents"]),
            "available_models": available_models,
            "include_gemini_embedding_2": include_embedding_2,
            "config_count": len(configs),
            "configs": [config_to_dict(config) | {"run_id": config_run_id(config)} for config in configs],
            "k_values": K_VALUES,
            "primary_metric": "ndcg@10",
            "secondary_metrics": ["recall@10", "mrr@10", "ndcg@20", "recall@50"],
            "positive_only_qrels_caveat": "Unjudged pairs are treated as implicit 0.",
            "failures": failures,
        },
    )
    log(f"Wrote reports to {OUTPUT_DIR}")
    if failures:
        log(f"Recorded {len(failures)} embedding failures in run_config.json")

    log("Top 10 configurations by nDCG@10:")
    for row in summary_rows[:10]:
        log(
            f"{row['run_id']} ndcg@10={row['ndcg@10']:.4f} recall@50={row['recall@50']:.4f} "
            f"mrr@10={row['mrr@10']:.4f} model={row['model']} setup={row['embedding_task_setup']} "
            f"dim={row['output_dimensionality']} doc_variant={row['doc_text_variant']} "
            f"query_variant={row['query_text_variant']} sim={row['similarity_metric']} norm={row['normalize_embeddings']}"
        )

    pareto_rows = pareto_front(summary_rows, ["ndcg@10", "recall@50", "mrr@10"])
    log("Pareto-good configurations across nDCG@10, Recall@50, and MRR@10:")
    for row in pareto_rows:
        log(
            f"{row['run_id']} ndcg@10={row['ndcg@10']:.4f} recall@50={row['recall@50']:.4f} mrr@10={row['mrr@10']:.4f}"
        )

    cache.close()


if __name__ == "__main__":
    run()
