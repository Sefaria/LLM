import json
from collections import defaultdict
from pathlib import Path

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Streamlit is required. Install it with: pip install streamlit") from exc


DATASET_DIR = Path(__file__).resolve().parent / "output" / "eval_dataset_sanity_check"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data
def load_dataset(dataset_dir_str: str):
    d = Path(dataset_dir_str)
    documents = read_jsonl(d / "documents.jsonl")
    queries = read_jsonl(d / "queries.jsonl")
    qrels = read_jsonl(d / "qrels.jsonl")
    return documents, queries, qrels


def main():
    st.set_page_config(page_title="Eval Dataset Inspector", layout="wide")

    dataset_dir_str = st.sidebar.text_input("Dataset directory", str(DATASET_DIR))
    documents, queries, qrels = load_dataset(dataset_dir_str)

    # Build lookups
    queries_by_id = {q["query_id"]: q for q in queries}
    qrels_by_doc = defaultdict(list)
    for qrel in qrels:
        qrels_by_doc[qrel["doc_id"]].append(qrel)

    # Group docs by ref
    docs_by_ref: dict[str, dict[str, dict]] = defaultdict(dict)  # ref -> lang -> doc
    for doc in documents:
        meta = doc.get("metadata", {})
        ref = meta.get("ref")
        lang = meta.get("lang")
        if ref and lang:
            docs_by_ref[ref][lang] = doc

    all_refs = sorted(docs_by_ref)
    if not all_refs:
        st.warning("No data loaded.")
        return

    # Sidebar: pick a ref
    st.sidebar.markdown(f"**{len(all_refs)} refs**")
    ref_filter = st.sidebar.text_input("Filter refs", "").strip().lower()
    filtered_refs = [r for r in all_refs if ref_filter in r.lower()] if ref_filter else all_refs
    selected_ref = st.sidebar.selectbox("Ref", filtered_refs)

    if not selected_ref:
        return

    st.header(selected_ref)

    # Show text for each language
    lang_docs = docs_by_ref[selected_ref]
    cols = st.columns(len(lang_docs)) if lang_docs else []
    for col, (lang, doc) in zip(cols, sorted(lang_docs.items())):
        col.subheader(lang.upper())
        col.text_area(lang.upper(), doc.get("text", ""), height=200, key=f"text-{doc['doc_id']}", label_visibility="collapsed")

    # Show all queries linked to any doc of this ref
    st.divider()
    st.subheader("Queries")

    all_qrels = []
    for lang, doc in lang_docs.items():
        for qrel in qrels_by_doc.get(doc["doc_id"], []):
            all_qrels.append((lang, qrel))

    if not all_qrels:
        st.info("No queries linked to this ref.")
        return

    all_qrels.sort(key=lambda x: (-x[1].get("relevance", 0), x[0]))

    rows = []
    for doc_lang, qrel in all_qrels:
        q = queries_by_id.get(qrel.get("query_id"), {})
        rows.append({
            "relevance": qrel.get("relevance"),
            "doc_lang": doc_lang,
            "query_lang": q.get("lang"),
            "type": q.get("type"),
            "query": q.get("text"),
            "reason": qrel.get("reason"),
        })

    st.dataframe(rows, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
