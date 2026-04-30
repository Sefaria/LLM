import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset(dataset_dir: Path, include_confounders: bool = False) -> dict:
    documents = read_jsonl(dataset_dir / "documents.jsonl")
    confounding_documents = read_jsonl(dataset_dir / "confounding_documents.jsonl") if include_confounders else []
    queries = read_jsonl(dataset_dir / "queries.jsonl")
    qrels = read_jsonl(dataset_dir / "qrels.jsonl")
    all_documents = documents + confounding_documents

    documents_by_id = {doc["doc_id"]: doc for doc in all_documents}
    queries_by_id = {query["query_id"]: query for query in queries}
    relevant_doc_ids_by_query = {}
    for qrel in qrels:
        if qrel.get("relevance", 0) <= 0:
            continue
        relevant_doc_ids_by_query.setdefault(qrel["query_id"], set()).add(qrel["doc_id"])

    query_category_by_id = {}
    for query_id, doc_ids in relevant_doc_ids_by_query.items():
        categories = sorted({
            documents_by_id[doc_id].get("metadata", {}).get("category", "unknown")
            for doc_id in doc_ids
            if doc_id in documents_by_id
        })
        if not categories:
            query_category_by_id[query_id] = "unknown"
        elif len(categories) == 1:
            query_category_by_id[query_id] = categories[0]
        else:
            query_category_by_id[query_id] = "mixed"

    return {
        "documents": all_documents,
        "judged_documents": documents,
        "confounding_documents": confounding_documents,
        "queries": queries,
        "qrels": qrels,
        "documents_by_id": documents_by_id,
        "queries_by_id": queries_by_id,
        "relevant_doc_ids_by_query": relevant_doc_ids_by_query,
        "query_category_by_id": query_category_by_id,
    }
