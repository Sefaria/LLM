import sys
import os
import csv
import logging
import random
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sefaria.settings")
import django
django.setup()

from sefaria.model.link import LinkSet
from sefaria.model.text import Ref
from sefaria.helper.llm.topic_prompt import make_topic_prompt_source
from sklearn.cluster import AffinityPropagation, HDBSCAN
from functools import partial
import numpy as np
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from llm_cluster_optimizer.llm_clusterer import LLMClusterOptimizer, SummarizedCluster
from llm_cluster_optimizer.util import run_parallel

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


_embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    task_type="CLUSTERING",
)


def embed_text(text: str) -> np.ndarray:
    return np.array(_embedder.embed_query(text))


def get_text_from_ref(oref: Ref) -> tuple[str, str | None]:
    """Returns (text, ignore_reason). ignore_reason is None if text was retrieved."""
    try:
        source = make_topic_prompt_source(oref, '', with_commentary=False)
    except AttributeError as e:
        if 'is_virtual' in str(e):
            return '', 'text not in local DB (index_node=None)'
        return '', f'AttributeError: {e}'
    except Exception as e:
        return '', str(e)
    en = source.text.get('en', '')
    he = source.text.get('he', '')
    if en.strip():
        return en, None
    if he.strip():
        return he, None
    return '', 'empty text (both en and he blank)'


def summarize_relation_to_target(target_text: str, linked_text: str) -> str | None:
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    system = SystemMessage(content=(
        "You are a Jewish scholar. Given a target passage and a linked source, "
        "write a one-sentence summary (≤20 words) of what the linked source says "
        "that is relevant to the target passage. "
        "Be very specific about the unique angle the linked source has on understanding the target passage. "
        "Wrap the summary in <summary> tags."
    ))
    human = HumanMessage(content=(
        f"<target>{target_text}</target>\n"
        f"<linked>{linked_text}</linked>"
    ))
    response = llm.invoke([system, human])
    content = response.content
    if "<summary>" in content and "</summary>" in content:
        return content.split("<summary>")[1].split("</summary>")[0].strip()
    return content.strip() or None


class SidebarClusterer(LLMClusterOptimizer):
    def cluster(self, texts: list[str]) -> list[SummarizedCluster]:
        embeddings = self._embed_parallel(texts, desc="Embedding summaries")
        best_clusters = None
        highest_score = 0
        for cluster_model in self._cluster_models:
            curr_labels = self._get_labels(embeddings, cluster_model)
            curr_clusters = self._build_clusters(curr_labels, embeddings, texts)
            summarized = self._summarize_clusters(curr_clusters)
            summarized = self._optimize_collapse_similar_clusters(summarized)
            score = self._calculate_clustering_score(summarized)
            if score > highest_score:
                highest_score = score
                best_clusters = summarized
        return best_clusters or []


def cluster_sidebar(tref: str) -> None:
    oref = Ref(tref)
    target_text = get_text_from_ref(oref)
    if not target_text:
        print(f"Could not retrieve text for target ref: {tref}")
        return

    linked_refs = []  # list of (raw_tref_str, Ref)
    ignored: list[dict] = []
    oref_normal = oref.normal()

    def find_section_parent(raw_refs):
        for i, s in enumerate(raw_refs):
            if s == oref_normal:
                continue
            try:
                r = Ref(s)
                if r.index_node is not None and r.contains(oref):
                    return s, raw_refs[1 - i]
            except Exception:
                pass
        return None

    for link in LinkSet(oref):
        raw_refs = getattr(link, 'refs', ['?', '?'])
        try:
            opposite = link.ref_opposite(oref)
            if opposite is None:
                result = find_section_parent(raw_refs)
                if result is None:
                    ignored.append({"ref": " | ".join(raw_refs), "reason": "ref_opposite=None: no section-level parent found"})
                else:
                    parent_str, opp_raw = result
                    ignored.append({"ref": opp_raw, "reason": f"linked via section ref {parent_str}, not segment-level"})
                continue
            ref_obj = opposite if isinstance(opposite, Ref) else Ref(opposite)
            opp_raw = raw_refs[0] if raw_refs[1] == oref_normal else raw_refs[1]
            linked_refs.append((opp_raw, ref_obj))
        except Exception as e:
            ignored.append({"ref": " | ".join(raw_refs), "reason": f"exception: {e}"})
            continue

    print(f"Found {len(linked_refs)} linked refs for {tref}")

    texts_and_refs = []
    for raw, ref in linked_refs:
        text, reason = get_text_from_ref(ref)
        if text.strip():
            texts_and_refs.append((ref, text))
        else:
            ignored.append({"ref": raw, "reason": reason or "empty text"})

    print(f"{len(texts_and_refs)} refs with text, {len(ignored)} ignored")

    csv_path = f"ignored_refs_{tref.replace(' ', '_').replace(':', '_')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ref", "reason"])
        writer.writeheader()
        writer.writerows(ignored)
    print(f"Ignored refs written to {csv_path}")

    if not texts_and_refs:
        print("No texts found")
        return

    refs, linked_texts = zip(*texts_and_refs)

    print("Summarizing each linked ref's relation to the target passage...")
    summarize = partial(summarize_relation_to_target, target_text)
    summaries = run_parallel(list(linked_texts), summarize, max_workers=25,
                             desc="summarizing relations")
    summaries_and_refs = [(ref, s) for ref, s in zip(refs, summaries) if s]
    print(f"{len(summaries_and_refs)} summaries generated")

    if not summaries_and_refs:
        print("No summaries generated")
        return

    refs_by_summary, summaries = {}, []
    for ref, s in summaries_and_refs:
        refs_by_summary[s] = ref
        summaries.append(s)

    optimizer = SidebarClusterer(
        cluster_models=[
            HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method="eom", cluster_selection_epsilon=0.65),
            HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method="leaf", cluster_selection_epsilon=0.5),
            AffinityPropagation(damping=0.7, max_iter=1000, convergence_iter=100),
        ],
        embedding_fn=embed_text,
        verbose=True,
        num_embed_workers=25
    )

    clusters = optimizer.cluster(list(summaries))
    clusters.sort(key=len, reverse=True)

    print(f"\n--- Clusters for {tref} ---")
    for cluster in clusters:
        print(f"({len(cluster)}) {cluster.summary}")
        sample = random.sample(cluster.items, min(len(cluster), 10))
        for summary in sample:
            ref = refs_by_summary.get(summary, "?")
            print(f"  - {ref}: {summary}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cluster_sidebar.py <tref>")
        sys.exit(1)
    cluster_sidebar(sys.argv[1])
