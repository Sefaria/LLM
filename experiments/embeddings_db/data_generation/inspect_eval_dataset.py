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


def build_ref_summaries(documents: list[dict], qrels_by_doc: dict) -> list[dict]:
    docs_by_ref = defaultdict(dict)
    for doc in documents:
        meta = doc.get("metadata", {})
        ref = meta.get("ref")
        lang = meta.get("lang")
        if ref and lang:
            docs_by_ref[ref][lang] = doc

    summaries = []
    for ref, lang_docs in docs_by_ref.items():
        query_count = sum(len(qrels_by_doc.get(doc["doc_id"], [])) for doc in lang_docs.values())
        summaries.append({
            "ref": ref,
            "langs": sorted(lang_docs),
            "query_count": query_count,
            "docs": lang_docs,
        })
    return sorted(summaries, key=lambda row: row["ref"])


def main():
    st.set_page_config(page_title="Eval Dataset Inspector", layout="wide")

    dataset_dir_str = st.sidebar.text_input("Dataset directory", str(DATASET_DIR))
    documents, queries, qrels = load_dataset(dataset_dir_str)

    # Build lookups
    queries_by_id = {q["query_id"]: q for q in queries}
    qrels_by_doc = defaultdict(list)
    for qrel in qrels:
        qrels_by_doc[qrel["doc_id"]].append(qrel)

    ref_summaries = build_ref_summaries(documents, qrels_by_doc)
    if not ref_summaries:
        st.warning("No data loaded.")
        return

    # Sidebar: pick a ref
    st.sidebar.markdown(f"**{len(ref_summaries)} refs**")
    ref_filter = st.sidebar.text_input("Filter refs", "").strip().lower()
    filtered_summaries = [row for row in ref_summaries if ref_filter in row["ref"].lower()] if ref_filter else ref_summaries
    if not filtered_summaries:
        st.warning("No refs match the current filter.")
        return

    st.sidebar.caption(f"{len(filtered_summaries)} matching refs")

    filtered_refs = [row["ref"] for row in filtered_summaries]

    # Use the selectbox key as the single source of truth so clicks are never overridden
    if "_ref_selectbox" not in st.session_state or st.session_state._ref_selectbox not in filtered_refs:
        st.session_state._ref_selectbox = filtered_refs[0]

    selected_index = filtered_refs.index(st.session_state._ref_selectbox)

    nav_cols = st.sidebar.columns(2)
    if nav_cols[0].button("◀ Prev") and selected_index > 0:
        st.session_state._ref_selectbox = filtered_refs[selected_index - 1]
        st.rerun()
    if nav_cols[1].button("Next ▶") and selected_index < len(filtered_refs) - 1:
        st.session_state._ref_selectbox = filtered_refs[selected_index + 1]
        st.rerun()

    selected_ref = st.sidebar.selectbox(
        "Ref",
        filtered_refs,
        key="_ref_selectbox",
        format_func=lambda ref: next(
            f"{filtered_refs.index(ref) + 1}. {ref} [{' / '.join(row['langs'])}] ({row['query_count']} queries)"
            for row in filtered_summaries
            if row["ref"] == ref
        ),
    )

    if not selected_ref:
        return

    st.header(selected_ref)

    # Show text for each language
    lang_docs = next(row["docs"] for row in ref_summaries if row["ref"] == selected_ref)
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
