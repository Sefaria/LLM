#!/usr/bin/env python
"""
Evaluate topic‑tagging predictions against a LangSmith dataset,
track per‑slug false positives, and persist the histogram as
both JSON and CSV when the run completes.
"""

from __future__ import annotations

from collections import Counter
import json
import csv
from datetime import datetime
from pathlib import Path

from langsmith import evaluate, Client
from langsmith.schemas import Example, Run
from langchain_openai import ChatOpenAI

# Uncomment if you actually use the cache in this project.
# from langchain_community.cache import SQLiteCache

from experiments.topic_modelling.llm_filtering import SequentialRefTopicFilter
from experiments.topic_modelling.utils import DataHandler

# ──────────────────────────────────────────────────────────
# Global aggregation container
# ──────────────────────────────────────────────────────────
false_positive_counter: Counter[str] = Counter()

# ──────────────────────────────────────────────────────────
# Helpers and configuration
# ──────────────────────────────────────────────────────────
llm_filter = SequentialRefTopicFilter(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    max_topics=15,
)

dh = DataHandler(predicted_filename="../../evaluation_data/predictions.jsonl")
predicted = dh.get_predicted()


def get_example(x: dict) -> dict | None:
    """Convert raw LangSmith row → model input & gold-standard slugs."""
    ref = x.get("ref")
    if ref is None:
        return None

    predicted_lr = next(lr for lr in predicted if lr.ref == ref)
    filtered_slugs = llm_filter.filter_ref(predicted_lr)
    return {"ref": ref, "slugs": filtered_slugs}


def check_answer(root_run: Run, example: Example) -> dict:
    """Compute precision/recall/F1 and record false‑positive slugs."""
    try:
        ref = root_run.outputs.get("ref")
        predicted_lr = next(lr for lr in dh.get_predicted() if lr.ref == ref)

        gold_set = set(example.outputs.get("slugs")) & set(predicted_lr.slugs)
        pred_set = set(root_run.outputs.get("slugs"))

        overlap = gold_set & pred_set
        false_pos = sorted(pred_set - gold_set)
        false_neg = sorted(gold_set - pred_set)

        # Aggregate false positives across the whole experiment
        for slug in false_pos:
            false_positive_counter[slug] += 1

        tp = len(overlap)
        fp = len(false_pos)
        fn = len(false_neg)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "score": f1,                    # LangSmith expects 'score' (0‑1)
            "precision": precision,
            "recall": recall,
            "n_gold": len(gold_set),
            "n_pred": len(pred_set),
            "true_positives": sorted(overlap),
            "false_positives": false_pos,
            "false_negatives": false_neg,
        }

    except Exception as exc:  # pragma: no cover
        return {"score": 0.0, "error": str(exc)}


# ──────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = Client()  # noqa: F841  # keep for future run‑metadata logging

    evaluate(
        get_example,
        data="topic-tagging-gold",
        evaluators=[check_answer],
        experiment_prefix="topic_tagging_experiment",
    )

    # Persist the false‑positive histogram
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("experiment_artifacts")
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / f"false_positive_counts_{timestamp}.json"
    csv_path = out_dir / f"false_positive_counts_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(dict(false_positive_counter), fp, indent=2, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["slug", "count"])
        for slug, count in false_positive_counter.most_common():
            writer.writerow([slug, count])

    print("✅  False‑positive histogram saved to:")
    print(f"   • {json_path}")
    print(f"   • {csv_path}")
