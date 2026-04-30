import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def score_documents(
    query_vector: np.ndarray,
    document_matrix: np.ndarray,
    similarity_metric: str,
) -> np.ndarray:
    if similarity_metric == "dot":
        return document_matrix @ query_vector
    if similarity_metric == "cosine":
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(document_matrix, axis=1)
        denom = doc_norms * query_norm
        denom[denom == 0] = 1.0
        return (document_matrix @ query_vector) / denom
    if similarity_metric == "euclidean":
        return -np.linalg.norm(document_matrix - query_vector, axis=1)
    raise ValueError(f"Unknown similarity metric: {similarity_metric}")


def rank_documents(
    query_vector: np.ndarray,
    document_ids: list[str],
    document_matrix: np.ndarray,
    similarity_metric: str,
) -> list[tuple[str, float]]:
    scores = score_documents(query_vector, document_matrix, similarity_metric)
    order = np.argsort(-scores)
    return [(document_ids[index], float(scores[index])) for index in order]

