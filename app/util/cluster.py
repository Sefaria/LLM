from abc import ABC, abstractmethod
from typing import Any, Callable, Union, TypeVar
from functools import reduce, partial
from tqdm import tqdm
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation
import random
from dataclasses import dataclass
from numpy import ndarray
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import HumanMessage, SystemMessage, AbstractMessage
from util.general import get_by_xml_tag, run_parallel
import numpy as np

T = TypeVar('T')
RANDOM_SEED = 567454
random.seed(RANDOM_SEED)


class AbstractClusterItem(ABC):
    @abstractmethod
    def get_str_to_embed(self) -> str:
        pass

    @abstractmethod
    def get_str_to_summarize(self) -> str:
        pass


@dataclass
class Cluster:
    label: int
    embeddings: list[ndarray]
    items: list[AbstractClusterItem]
    summary: str = None

    def merge(self, other: 'Cluster') -> 'Cluster':
        return Cluster(self.label, self.embeddings + other.embeddings, self.items + other.items, self.summary)

    def __len__(self):
        return len(self.items)

    def __eq__(self, other):
        if not isinstance(other, Cluster):
            return False
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.label, len(self), self.summary))


class AbstractClusterer(ABC):

    def __init__(self, embedding_fn: Callable[[str], ndarray], get_cluster_summary: Callable[[list[str]], str] = None, verbose=True):
        """
        :param embedding_fn:
        :param get_cluster_summary: function that takes a list of strings to summarize (sampled from cluster items) and returns a summary of the strings.
        """
        self._embedding_fn = embedding_fn
        self._get_cluster_summary = get_cluster_summary
        self._verbose = verbose

    def embed_parallel(self, items: list[T], key: Callable[[T], ndarray], **kwargs):
        return run_parallel([key(item) for item in items], self._embedding_fn, disable=not self._verbose, **kwargs)

    @abstractmethod
    def cluster_items(self, items: list[AbstractClusterItem]) -> list[Cluster]:
        pass

    def cluster_and_summarize(self, items: list[AbstractClusterItem]) -> list[Cluster]:
        clusters = self.cluster_items(items)
        return self.summarize_clusters(clusters)

    @staticmethod
    def _default_get_cluster_summary(strs_to_summarize: list[str]) -> str:
        llm = ChatOpenAI("gpt-4o", 0)
        system = SystemMessage(content="Given a few ideas (wrapped in <idea> "
                                       "XML tags) output a summary of the"
                                       "ideas. Wrap the output in <summary> tags. Summary"
                                       "should be no more than 10 words.")
        human = HumanMessage(content=f"<idea>{'</idea><idea>'.join(strs_to_summarize)}</idea>")
        response = llm([system, human])
        return get_by_xml_tag(response.content, "summary")

    def summarize_cluster(self, cluster: Cluster, sample_size=5) -> Cluster:
        """
        :param cluster: Cluster to summarize
        :param sample_size: Maximum number of items to sample from a cluster. If len(cluster) < sample_size, then all items in the cluster will be chosen.
        :return: the same cluster object with the `summary` attribute set.
        """
        get_cluster_summary = self._get_cluster_summary or self._default_get_cluster_summary
        sample = random.sample(cluster.items, min(len(cluster), sample_size))
        strs_to_summarize = [item.get_str_to_summarize() for item in sample]
        if len(cluster) == 1:
            cluster.summary = cluster.items[0].get_str_to_summarize()
            return cluster
        cluster.summary = get_cluster_summary(strs_to_summarize)
        return cluster

    def summarize_clusters(self, clusters: list[Cluster], **kwargs) -> list[Cluster]:
        return run_parallel(clusters, partial(self.summarize_cluster, **kwargs),
                            max_workers=25, desc='summarize source clusters', disable=not self._verbose)

    @staticmethod
    def _build_clusters_from_cluster_results(labels, embeddings, items):
        clusters = []
        noise_items = []
        noise_embeddings = []
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            curr_embeddings = [embeddings[j] for j in indices]
            curr_items = [items[j] for j in indices]
            if label == -1:
                noise_items += curr_items
                noise_embeddings += curr_embeddings
                continue
            clusters += [Cluster(label, curr_embeddings, curr_items)]
        return clusters, noise_items, noise_embeddings


def _guess_optimal_n_clusters(embeddings, get_model, verbose=True):
    if len(embeddings) <= 1:
        return len(embeddings)

    best_sil_coeff = -1
    best_num_clusters = 0
    MAX_MIN_CLUSTERS = 3  # the max start of the search for optimal cluster number.
    n_cluster_start = min(len(embeddings), MAX_MIN_CLUSTERS)
    n_cluster_end = len(embeddings)//2
    if n_cluster_end < (n_cluster_start + 1):
        n_cluster_start = 2
        n_cluster_end = n_cluster_start + 1
    n_clusters = range(n_cluster_start, n_cluster_end)
    for n_cluster in tqdm(n_clusters, total=len(n_clusters), desc='guess optimal clustering', disable=not verbose):
        model = get_model(n_cluster).fit(embeddings)
        sil_coeff = silhouette_score(embeddings, model.labels_, metric='cosine', random_state=RANDOM_SEED)
        if sil_coeff > best_sil_coeff:
            best_sil_coeff = sil_coeff
            best_num_clusters = n_cluster
    if verbose:
        print("Best N", best_num_clusters, "Best silhouette score", round(best_sil_coeff, 4))
    return best_num_clusters


def get_agglomerative_clustering_labels_with_optimal_silhouette_score(embeddings: list[np.ndarray]):
    n_clusters = _guess_optimal_n_clusters(embeddings, lambda n: AgglomerativeClustering(n_clusters=n, linkage='average', metric='cosine'))
    return AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='cosine').fit(embeddings).labels_


class SklearnClusterer(AbstractClusterer):

    def __init__(self, embedding_fn: Callable[[str], ndarray],
                 get_cluster_labels: Callable[[list[ndarray]], ndarray],
                 get_cluster_summary: Callable[[list[str]], str] = None,
                 verbose: bool = True, breakup_large_clusters: bool = True):
        """
        :param embedding_fn:
        :param get_cluster_model:
        :param verbose:
        """
        super().__init__(embedding_fn, get_cluster_summary, verbose)
        self._get_cluster_labels = get_cluster_labels
        self._breakup_large_clusters = breakup_large_clusters

    def clone(self, **kwargs) -> 'SklearnClusterer':
        """
        Return new object with all the same data except modifications specified in kwargs
        """
        init_params = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                key = key[1:]
            init_params[key] = value
        return self.__class__(**{**init_params, **kwargs})

    def _get_large_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        from statistics import mean, stdev
        large_clusters = []
        for cluster in clusters:
            other_cluster_lens = [len(c) for c in clusters if c != cluster]
            if len(other_cluster_lens) <= 1:
                continue
            if len(cluster) > (mean(other_cluster_lens) + 3*stdev(other_cluster_lens)):
                large_clusters.append(cluster)
        return large_clusters

    def _recluster_large_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        large_clusters = set(self._get_large_clusters(clusters))
        other_clusters = [c for c in clusters if c not in large_clusters]
        if len(large_clusters) == 0:
            return clusters
        affinity_clusterer = self.clone(
            get_cluster_labels=lambda x: AffinityPropagation(damping=0.7, max_iter=1000, convergence_iter=100).fit(x).predict(x),
            breakup_large_clusters=False
        )
        items_to_recluster = reduce(lambda x, y: x + y.items, large_clusters, [])
        reclustered_clusters = affinity_clusterer.cluster_items(items_to_recluster)
        print("RECLUSTERED LEN", len(reclustered_clusters))
        return other_clusters + reclustered_clusters

    def cluster_items(self, items: list[AbstractClusterItem]) -> list[Cluster]:
        """
        :param items: Generic list of items to cluster
        :return: list of Cluster objects
        """
        embeddings = self.embed_parallel(items, lambda x: x.get_str_to_embed(), max_workers=40, desc="embedding items for clustering")
        labels = self._get_cluster_labels(embeddings)
        clusters, noise_items, noise_embeddings = self._build_clusters_from_cluster_results(labels, embeddings, items)
        if self._breakup_large_clusters:
            clusters = self._recluster_large_clusters(clusters)
        if len(noise_items) > 0:
            noise_labels = AffinityPropagation(damping=0.7, max_iter=1000, convergence_iter=100).fit(noise_embeddings).predict(noise_embeddings)
            noise_clusters, _, _ = self._build_clusters_from_cluster_results(noise_labels, noise_embeddings, noise_items)
            if self._verbose:
                print("LEN NOISE_CLUSTERS", len(noise_clusters))
        else:
            noise_clusters = []
        return clusters + noise_clusters


class OptimizingClusterer(AbstractClusterer):

    IDEAL_NUM_CLUSTERS = 20

    def __init__(self, embedding_fn: Callable[[str], ndarray], clusterers: list[AbstractClusterer], verbose=True):
        super().__init__(embedding_fn, verbose=verbose)
        self._clusterers = clusterers

    def _embed_cluster_summaries(self, summarized_clusters: list[Cluster]):
        return self.embed_parallel(
            summarized_clusters, lambda x: x.summary, max_workers=40, desc="embedding cluster summaries to score"
        )

    def _optimize_collapse_similar_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        embeddings = self._embed_cluster_summaries(clusters)
        distances = pairwise_distances(embeddings, metric='cosine')
        highest_clustering_score = 0
        best_clusters = None
        best_thresh = None
        for threshold in [0.2, 0.25, 0.3]:
            temp_clusters = self._collapse_similar_clusters(clusters, distances, threshold)
            temp_score = self._calculate_clustering_score(temp_clusters)
            if temp_score > highest_clustering_score:
                highest_clustering_score = temp_score
                best_clusters = temp_clusters
                best_thresh = threshold
        print("Best threshold", best_thresh)
        return best_clusters

    @staticmethod
    def _collapse_similar_clusters(clusters: list[Cluster], distances: ndarray, threshold: float) -> list[Cluster]:
        num_clusters = len(clusters)

        # Keep track of which clusters have been merged
        merged = [False] * num_clusters
        new_clusters = []

        for i in range(num_clusters):
            if not merged[i]:
                merged[i] = True
                merged_cluster = clusters[i]
                for j in range(i + 1, num_clusters):
                    if not merged[j] and distances[i, j] < threshold:
                        merged[j] = True
                        merged_cluster = merged_cluster.merge(clusters[j])
                new_clusters.append(merged_cluster)
        return new_clusters

    def _calculate_clustering_score(self, summarized_clusters: list[Cluster]) -> float:
        embeddings = self._embed_cluster_summaries(summarized_clusters)
        distances = pairwise_distances(embeddings, metric='cosine')
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        num_clusters = len(summarized_clusters)
        avg_min_distance = sum(min_distances)/len(min_distances)
        max_clusters = 2*self.IDEAL_NUM_CLUSTERS  # used to ensure closeness_to_ideal_score is bounded
        closeness_to_ideal_score = (1 - (min(abs(num_clusters - self.IDEAL_NUM_CLUSTERS), max_clusters)/max_clusters)) * 0.7
        clustering_score = closeness_to_ideal_score + avg_min_distance
        return clustering_score

    def cluster_items(self, items: list[AbstractClusterItem]) -> list[Cluster]:
        best_clusters = None
        highest_clustering_score = 0
        chosen_i = None
        for i, clusterer in enumerate(self._clusterers):
            curr_clusters = clusterer.cluster_items(items)
            summarized_clusters = clusterer.summarize_clusters(curr_clusters)
            summarized_clusters = self._optimize_collapse_similar_clusters(summarized_clusters)
            clustering_score = self._calculate_clustering_score(summarized_clusters)
            # print("CLUSTER SCORE: ", clustering_score)
            if clustering_score > highest_clustering_score:
                highest_clustering_score = clustering_score
                best_clusters = summarized_clusters
                chosen_i = i
        print("CLUSTERER CHOSEN", chosen_i)
        return best_clusters

    def cluster_and_summarize(self, items: list[AbstractClusterItem]) -> list[Cluster]:
        clusters = self.cluster_items(items)
        return self._clusterers[0].summarize_clusters(clusters)
