import csv
import json
from pathlib import Path


SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "embedding_task_setup",
    "output_dimensionality",
    "doc_text_variant",
    "query_text_variant",
    "similarity_metric",
    "normalize_embeddings",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "recall@10",
    "recall@50",
    "mrr@10",
    "mean_first_relevant_rank",
]


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})


def write_breakdowns_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_run_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
