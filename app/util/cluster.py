from abc import ABC, abstractmethod
from typing import Any, Callable, Union
from functools import reduce, partial
from tqdm import tqdm
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans
import random
from dataclasses import dataclass
from numpy import ndarray
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import HumanMessage, SystemMessage, AbstractMessage
from util.general import get_by_xml_tag, run_parallel
import numpy as np

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

    def __len__(self):
        return len(self.items)


class AbstractClusterer(ABC):

    @abstractmethod
    def cluster_items(self, items: list[AbstractClusterItem], cluster_noise: bool = False) -> list[Cluster]:
        pass

    @abstractmethod
    def cluster_and_summarize(self, items: list[AbstractClusterItem]) -> list[Cluster]:
        pass


def _guess_optimal_kmeans_clustering(embeddings, verbose=True):
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
        kmeans = KMeans(n_clusters=n_cluster, n_init='auto', random_state=RANDOM_SEED).fit(embeddings)
        labels = kmeans.labels_
        sil_coeff = silhouette_score(embeddings, labels, metric='cosine', random_state=RANDOM_SEED)
        if sil_coeff > best_sil_coeff:
            best_sil_coeff = sil_coeff
            best_num_clusters = n_cluster
    if verbose:
        print("Best silhouette score", round(best_sil_coeff, 4))
    return best_num_clusters


def make_kmeans_algo_with_optimal_silhouette_score(embeddings: list[np.ndarray]):
    n_clusters = _guess_optimal_kmeans_clustering(embeddings)
    return KMeans(n_clusters=n_clusters, n_init='auto', random_state=RANDOM_SEED)


class StandardClusterer(AbstractClusterer):

    def __init__(self, embedding_fn: Callable[[str], ndarray], 
                 get_cluster_algo: Callable[[list[ndarray]], Union[KMeans, HDBSCAN]],
                 get_cluster_summary: Callable[[list[str]], str] = None,
                 verbose: bool = True):
        """
        :param embedding_fn:
        :param get_cluster_algo:
        :param get_cluster_summary: function that takes a list of strings to summarize (sampled from cluster items) and returns a summary of the strings.
        :param verbose:
        """
        self.embedding_fn = embedding_fn
        self.get_cluster_algo = get_cluster_algo
        self.verbose = verbose
        self.get_cluster_summary = get_cluster_summary or self._default_get_cluster_summary


    def clone(self, **kwargs) -> 'StandardClusterer':
        """
        Return new object with all the same data except modifications specified in kwargs
        """
        return self.__class__(**{**self.__dict__, **kwargs})

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
        sample = random.sample(cluster.items, min(len(cluster), sample_size))
        strs_to_summarize = [item.get_str_to_summarize() for item in sample]
        if len(cluster) == 1:
            cluster.summary = cluster.items[0].get_str_to_summarize()
            return cluster
        cluster.summary = self.get_cluster_summary(strs_to_summarize)
        return cluster

    def summarize_clusters(self, clusters: list[Cluster], **kwargs) -> list[Cluster]:
        return run_parallel(clusters, partial(self.summarize_cluster, **kwargs),
                            max_workers=25, desc='summarize source clusters', disable=not self.verbose)

    def cluster_items(self, items: list[AbstractClusterItem], cluster_noise: bool = False) -> list[Cluster]:
        """
        :param items: Generic list of items to cluster
        :param cluster_noise:
        :return: list of Cluster objects
        """
        embeddings = run_parallel([item.get_str_to_embed() for item in items], self.embedding_fn, max_workers=40, desc="embedding items for clustering", disable=not self.verbose)
        cluster_results = self.get_cluster_algo(embeddings).fit(embeddings)
        clusters, noise_items, noise_embeddings = self._build_clusters_from_cluster_results(cluster_results, embeddings, items)
        if cluster_noise:
            noise_results = make_kmeans_algo_with_optimal_silhouette_score(noise_embeddings).fit(noise_embeddings)
            noise_clusters, _, _ = self._build_clusters_from_cluster_results(noise_results, noise_embeddings, noise_items)
            if self.verbose:
                print("LEN NOISE_CLUSTERS", len(noise_clusters))
        else:
            noise_clusters = []
        return clusters + noise_clusters

    def cluster_and_summarize(self, items: list[AbstractClusterItem], **kwargs) -> list[Cluster]:
        clusters = self.cluster_items(items)
        return self.summarize_clusters(clusters, **kwargs)


    def _build_clusters_from_cluster_results(self, cluster_results: Union[KMeans, HDBSCAN], embeddings, items):
        clusters = []
        noise_items = []
        noise_embeddings = []
        for label in set(cluster_results.labels_):
            indices = np.where(cluster_results.labels_ == label)[0]
            curr_embeddings = [embeddings[j] for j in indices]
            curr_items = [items[j] for j in indices]
            if label == -1:
                noise_items += curr_items
                noise_embeddings += curr_embeddings
                if self.verbose:
                    print('noise cluster', len(curr_items))
                continue
            clusters += [Cluster(label, curr_embeddings, curr_items)]
        return clusters, noise_items, noise_embeddings



class HDBSCANOptimizerClusterer(AbstractClusterer):
    IDEAL_NUM_CLUSTERS = 20
    HDBSCAN_PARAM_OPTS = {
        "min_samples": [1, 1],
        "min_cluster_size": [2, 2],
        "cluster_selection_method": ["eom", "leaf"],
        "cluster_selection_epsilon": [0.65, 0.5],
    }

    def __init__(self, clusterer: StandardClusterer, verbose=True):
        # TODO param to avoid clustering noise cluster which may mess up optimizer
        self.clusterer = clusterer
        self.param_search_len = len(self.HDBSCAN_PARAM_OPTS["min_samples"])
        self.verbose = verbose

    def _embed_cluster_summaries(self, summarized_clusters: list[Cluster]):
        return run_parallel(
            [c.summary for c in summarized_clusters],
            self.clusterer.embedding_fn,
            max_workers=40, desc="embedding items for clustering", disable=not self.verbose
        )

    def _calculate_clustering_score(self, summarized_clusters: list[Cluster], verbose=True) -> float:
        embeddings = self._embed_cluster_summaries(summarized_clusters)
        distances = pairwise_distances(embeddings, metric='cosine')
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        num_clusters = len(summarized_clusters)
        avg_min_distance = sum(min_distances)/len(min_distances)
        max_clusters = 5*self.IDEAL_NUM_CLUSTERS  # used to ensure closeness_to_ideal_score is bounded
        closeness_to_ideal_score = (1 - (min(abs(num_clusters - self.IDEAL_NUM_CLUSTERS), max_clusters)/max_clusters)) * 0.7
        clustering_score = closeness_to_ideal_score + avg_min_distance
        return clustering_score

    def _get_ith_hdbscan_params(self, i):
        return reduce(lambda x, y: {**x, y[0]: y[1][i]}, self.HDBSCAN_PARAM_OPTS.items(), {})

    def cluster_items(self, items: list[AbstractClusterItem], cluster_noise: bool = False) -> list[Cluster]:
        best_clusterer = None
        highest_clustering_score = 0
        for i in range(self.param_search_len):
            curr_hdbscan_obj = HDBSCAN(**self._get_ith_hdbscan_params(i))
            curr_clusterer = self.clusterer.clone(get_cluster_algo=lambda x: curr_hdbscan_obj, verbose=False)
            curr_clusters = self.clusterer.cluster_items(items, cluster_noise=True)
            summarized_clusters = self.clusterer.summarize_clusters(curr_clusters)
            clustering_score = self._calculate_clustering_score(summarized_clusters)
            if clustering_score > highest_clustering_score:
                highest_clustering_score = clustering_score
                best_clusterer = curr_clusterer
                print("best hdbscan params", self._get_ith_hdbscan_params(i))
        return best_clusterer.cluster_items(items, cluster_noise=cluster_noise)

    def cluster_and_summarize(self, items: list[AbstractClusterItem]) -> list[Cluster]:
        clusters = self.cluster_items(items, cluster_noise=True)
        summarized_clusters = self.clusterer.summarize_clusters(clusters)
        if self.verbose:
            print(f'---SUMMARIES--- ({len(summarized_clusters)})')
            for cluster in summarized_clusters:
                print('\t-', len(cluster.items), cluster.summary.strip())
        return summarized_clusters


