"""
Given clusters tries to
1) Choose the "best" clusters such that each cluster represents an idea that should be part of the final topic page curation
2) For each chosen cluster, choose the "best" source to be curated. Best here needs to fulfill a few criteria including
    - Category quota: sources should be from diverse categories in Sefaria
    - Fundamental sources: funadmental sources from Tanakh and Talmud etc. should be chosen. These should be few, made 2-3
    - Interesting sources: the rest of the sources should represent interesting ideas for a newcomer to Sefaria
"""
import django
django.setup()
from sefaria.pagesheetrank import pagerank_rank_ref_list
from sefaria.model.text import Ref
from sefaria.recommendation_engine import RecommendationEngine
import voyageai
from tqdm import tqdm
from experiments.topic_source_curation_v2.cluster import Cluster, embed_text_openai, get_text_from_source
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sklearn.metrics import pairwise_distances
from util.pipeline import Artifact
from functools import reduce
from statistics import mean, stdev
import numpy as np


def choose_ideal_sources_for_clusters(clusters: list[Cluster]) -> list[TopicPromptSource]:
    return Artifact(clusters).pipe(choose_ideal_clusters, 20).pipe(choose_ideal_sources).data


def choose_primary_sources(clusters: list[Cluster]) -> tuple[list[TopicPromptSource], list[Cluster]]:
    orefs, cluster_labels = zip(*reduce(lambda x, y: x + [(Ref(item.source.ref), y.label) for item in y.items], clusters, []))
    ref_clusters = RecommendationEngine.cluster_close_refs(orefs, cluster_labels, 2)
    orefs, cluster_labels = [], []
    for ref_cluster in ref_clusters:
        curr_refs = [data['ref'] for data in ref_cluster]
        curr_labels = [data['data'] for data in ref_cluster]
        if curr_refs[0].primary_category == "Commentary":
            # don't combine commentary refs
            orefs += curr_refs
            cluster_labels += curr_labels
        else:
            orefs.append(curr_refs[0].to(curr_refs[-1]))
            cluster_labels.append(curr_labels[0])  # assume they're all in the same cluster
    trefs, pageranks = zip(*pagerank_rank_ref_list(orefs))
    max_ref = trefs[0]
    thresh = mean(pageranks) + 2 * stdev(pageranks)
    is_big = pageranks[0] > thresh
    print(max_ref, "IS BIG:", is_big, pageranks[0], thresh)
    # TODO need to combine sources before PR
    return [clusters[0].items[0]], [clusters[0]]


def choose_ideal_clusters(clusters: list[Cluster], max_clusters: int) -> list[Cluster]:
    """
    Sorted in descending order from outliers to central
    Choose a few central clusters and a few outliers
    Also might want to use custom GPT sort to find "best" clusters based on various criteria
    """
    # sorted_clusters = _sort_by_highest_avg_pairwise_distance(clusters)
    return [c for c in clusters if len(clusters) > 1]

def choose_ideal_sources(source_clusters: list[Cluster], verbose=True) -> list[TopicPromptSource]:
    """
    Criteria could be:
        Pagerank based on link graph for topic page. Higher means more relevant
        Pagerank delta == Global PR - Local PR. Need to decide what is good
        Highest average pairwise cosine distance. Higher means more unique
        Fulfills category quota. Want to choose sources from different categories
    """
    ideal_sources = []
    choose_primary_sources(source_clusters)
    for cluster in tqdm(source_clusters, desc='choose ideal sources', disable=not verbose):
        pass
       # ideal_sources += [choose_ideal_source_from_cluster(cluster)]
    return ideal_sources


def choose_ideal_source_from_cluster(cluster: Cluster) -> TopicPromptSource:
    if len(cluster) == 1:
        return cluster.items[0]
    vo = voyageai.Client()
    output = vo.rerank(cluster.summary, [get_text_from_source(item) for item in cluster.items], "rerank-lite-1")
    best_idx = output.results[0].index
    return cluster.items[best_idx]




def _get_highest_avg_pairwise_distance_indices(embeddings: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(embeddings, metric='cosine')
    sum_distances = np.sum(distances, axis=1)
    avg_distances = sum_distances / (len(embeddings) - 1)
    sorted_indices = np.argsort(avg_distances)[::-1]  # Sort in descending order
    return sorted_indices

def _sort_by_highest_avg_pairwise_distance(clusters: list[Cluster]) -> list[Cluster]:
    embeddings = np.array([embed_text_openai(c.summary) for c in clusters])
    sorted_indices = _get_highest_avg_pairwise_distance_indices(embeddings)
    return [clusters[i] for i in sorted_indices]
