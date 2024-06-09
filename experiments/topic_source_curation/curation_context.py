"""
Add helpful context to the curated sources that will be used for prompt generation
"""
import random
from util.cluster import Cluster
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from experiments.topic_source_curation.cluster import SummarizedSource, embed_text_openai
from topic_prompt.uniqueness_of_source import get_context_of_source
from sklearn.metrics.pairwise import cosine_similarity
from util.general import run_parallel
import numpy as np


def _find_closest_clusters(clusters: list[Cluster], target_cluster: Cluster, n: int):
    cluster_embeddings = run_parallel(
        [c.summary for c in clusters] + [target_cluster.summary],
        embed_text_openai,
        max_workers=40, desc="embedding clusters", disable=True
    )
    similarities = cosine_similarity([cluster_embeddings[-1]], cluster_embeddings[:-1])[0]
    closest_indices = np.argsort(similarities)[-n:][::-1]
    clusters = np.array(clusters)
    return clusters[closest_indices]


def get_context_for_source(source: SummarizedSource, clusters: list[Cluster], topic: Topic) -> str:
    tref = source.source.ref
    source_cluster = next((c for c in clusters if tref in {item.source.ref for item in c.items}), None)
    other_sources: list[TopicPromptSource] = [s.source for s in source_cluster.items if s.source.ref != tref]
    closest_clusters = _find_closest_clusters([c for c in clusters if c.summary != source_cluster.summary], source_cluster, 10)
    icluster = 0
    while len(other_sources) < 10 and icluster < len(closest_clusters):
        other_sources += [s.source for s in closest_clusters[icluster].items]
        icluster += 1
    random.shuffle(other_sources)
    return get_context_of_source(source.source, topic, other_sources)
