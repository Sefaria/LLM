"""
Given clusters tries to
1) Choose the "best" clusters such that each cluster represents an idea that should be part of the final topic page curation
2) For each chosen cluster, choose the "best" source to be curated. Best here needs to fulfill a few criteria including
    - Category quota: sources should be from diverse categories in Sefaria
    - Fundamental sources: funadmental sources from Tanakh and Talmud etc. should be chosen. These should be few, made 2-3
    - Interesting sources: the rest of the sources should represent interesting ideas for a newcomer to Sefaria
"""
from experiments.topic_source_curation_v2.cluster import Cluster
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sklearn.metrics import pairwise_distances
from util.pipeline import Artifact
import numpy as np


def choose_ideal_sources_for_clusters(clusters: list[Cluster]) -> list[TopicPromptSource]:
    return Artifact(clusters).pipe(choose_ideal_clusters, 20).pipe(choose_ideal_sources_for_clusters).data


def choose_ideal_clusters(clusters: list[Cluster], max_clusters: int) -> list[Cluster]:
    sorted_clusters = _sort_by_highest_avg_pairwise_distance(clusters)
    """
    Sorted in descending order from outliers to central
    Choose a few central clusters and a few outliers
    Also might want to use custom GPT sort to find "best" clusters based on various criteria
    """

def choose_ideal_sources(source_clusters: list[Cluster]) -> list[TopicPromptSource]:
    """
    Criteria could be:
        Pagerank based on link graph for topic page. Higher means more relevant
        Pagerank delta == Global PR - Local PR. Need to decide what is good
        Highest average pairwise cosine distance. Higher means more unique
        Fulfills category quota. Want to choose sources from different categories
    """
    pass


def _get_highest_avg_pairwise_distance_indices(embeddings: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(embeddings, metric='cosine')
    sum_distances = np.sum(distances, axis=1)
    avg_distances = sum_distances / (len(embeddings) - 1)
    sorted_indices = np.argsort(avg_distances)[::-1]  # Sort in descending order
    return sorted_indices

def _sort_by_highest_avg_pairwise_distance(clusters: list[Cluster]) -> list[Cluster]:
    embeddings = np.array([c.embedding for c in clusters])
    sorted_indices = _get_highest_avg_pairwise_distance_indices(embeddings)
    return [clusters[i] for i in sorted_indices]
