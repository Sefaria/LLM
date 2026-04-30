import math


def compute_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for index, relevance in enumerate(relevances[:k], start=1):
        if relevance <= 0:
            continue
        dcg += (2 ** relevance - 1) / math.log2(index + 1)
    return dcg


def compute_idcg(num_relevant: int, k: int) -> float:
    ideal_relevances = [1] * min(num_relevant, k)
    return compute_dcg(ideal_relevances, k)


def compute_ndcg(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    relevances = [1 if doc_id in relevant_doc_ids else 0 for doc_id in ranked_doc_ids[:k]]
    dcg = compute_dcg(relevances, k)
    idcg = compute_idcg(len(relevant_doc_ids), k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def compute_mrr(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    for index, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / index
    return 0.0


def compute_first_relevant_rank(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return float(index)
    return float("inf")

