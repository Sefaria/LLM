import re


HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")


def strip_hebrew_diacritics(text: str) -> str:
    return HEBREW_DIACRITICS_RE.sub("", text or "")


def render_document_text(document: dict, variant: str) -> str:
    metadata = document.get("metadata", {})
    text = document.get("text", "")
    if variant == "raw_text":
        return text
    if variant == "metadata_plus_text":
        return (
            f"ref: {metadata.get('ref', '')}\n"
            f"source: {metadata.get('source', '')}\n"
            f"category: {metadata.get('category', '')}\n"
            f"lang: {metadata.get('lang', '')}\n"
            f"text: {text}"
        )
    if variant == "minimal_metadata_plus_text":
        return (
            f"source: {metadata.get('source', '')}\n"
            f"ref: {metadata.get('ref', '')}\n"
            f"text: {text}"
        )
    raise ValueError(f"Unknown document text variant: {variant}")


def render_query_text(query: dict, variant: str) -> str:
    if variant == "raw_query":
        return query.get("text", "")
    raise ValueError(f"Unknown query text variant: {variant}")


def apply_mode_prefix(text: str, mode: str, is_query: bool) -> str:
    if mode in {"raw", "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"}:
        return text
    raise ValueError(f"Unknown text prefix mode: {mode}")
