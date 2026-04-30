import hashlib
import itertools
import json

from .gemini_embedder import GeminiEmbeddingConfig


OUTPUT_DIMS = [256, 512, 768, 1536, 3072]
DOC_TEXT_VARIANTS = ["raw_text"]
QUERY_TEXT_VARIANTS = ["raw_query"]
SIMILARITY_METRICS = ["cosine", "dot", "euclidean"]

EMBEDDING_001_TASK_SETUPS = {
    "retrieval": ("RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"),
    "semantic_similarity": ("SEMANTIC_SIMILARITY", "SEMANTIC_SIMILARITY"),
    "classification": ("CLASSIFICATION", "CLASSIFICATION"),
    "clustering": ("CLUSTERING", "CLUSTERING"),
    "question_answering": ("QUESTION_ANSWERING", "RETRIEVAL_DOCUMENT"),
    "fact_verification": ("FACT_VERIFICATION", "RETRIEVAL_DOCUMENT"),
    "code_retrieval": ("CODE_RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"),
}


def config_to_dict(config: GeminiEmbeddingConfig) -> dict:
    return {
        "model": config.model,
        "embedding_task_setup": config.embedding_task_setup,
        "output_dimensionality": config.output_dimensionality,
        "doc_text_variant": config.doc_text_variant,
        "query_text_variant": config.query_text_variant,
        "similarity_metric": config.similarity_metric,
        "normalize_embeddings": config.normalize_embeddings,
        "query_task_type": config.query_task_type,
        "doc_task_type": config.doc_task_type,
    }


def config_run_id(config: GeminiEmbeddingConfig) -> str:
    payload = json.dumps(config_to_dict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_experiment_grid(include_gemini_embedding_2: bool) -> list[GeminiEmbeddingConfig]:
    configs = []

    for output_dimensionality, doc_text_variant, query_text_variant, similarity_metric in itertools.product(
        OUTPUT_DIMS,
        DOC_TEXT_VARIANTS,
        QUERY_TEXT_VARIANTS,
        SIMILARITY_METRICS,
    ):
        normalize_options = [True] if output_dimensionality != 3072 else [False, True]
        for normalize_embeddings in normalize_options:
            for embedding_task_setup, (query_task_type, doc_task_type) in EMBEDDING_001_TASK_SETUPS.items():
                configs.append(
                    GeminiEmbeddingConfig(
                        model="gemini-embedding-001",
                        embedding_task_setup=embedding_task_setup,
                        output_dimensionality=output_dimensionality,
                        doc_text_variant=doc_text_variant,
                        query_text_variant=query_text_variant,
                        similarity_metric=similarity_metric,
                        normalize_embeddings=normalize_embeddings,
                        query_task_type=query_task_type,
                        doc_task_type=doc_task_type,
                    )
                )

    if include_gemini_embedding_2:
        for output_dimensionality, doc_text_variant, query_text_variant, similarity_metric in itertools.product(
            OUTPUT_DIMS,
            DOC_TEXT_VARIANTS,
            QUERY_TEXT_VARIANTS,
            SIMILARITY_METRICS,
        ):
            for normalize_embeddings in [False, True]:
                configs.append(
                    GeminiEmbeddingConfig(
                        model="gemini-embedding-2",
                        embedding_task_setup="default",
                        output_dimensionality=output_dimensionality,
                        doc_text_variant=doc_text_variant,
                        query_text_variant=query_text_variant,
                        similarity_metric=similarity_metric,
                        normalize_embeddings=normalize_embeddings,
                    )
                )

    return configs
