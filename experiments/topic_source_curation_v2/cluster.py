from typing import Any, Callable, Union
from functools import partial
from tqdm import tqdm
from basic_langchain.embeddings import VoyageAIEmbeddings
from basic_langchain.embeddings import OpenAIEmbeddings
import hdbscan
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from numpy import ndarray
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from basic_langchain.chat_models import ChatOpenAI, ChatAnthropic
from basic_langchain.schema import HumanMessage, SystemMessage
from experiments.topic_source_curation_v2.common import run_parallel
from util.general import get_by_xml_tag
from util.pipeline import Artifact
import numpy as np

RANDOM_SEED = 567454
random.seed(RANDOM_SEED)


@dataclass
class Cluster:
    label: int
    embeddings: list[ndarray]
    items: list[Any]
    summary: str = None

    def __len__(self):
        return len(self.items)

@dataclass
class SummarizedSource:
    source: TopicPromptSource
    summary: str

    def __init__(self, source: Union[TopicPromptSource, dict], summary: str, embedding: np.ndarray = None):
        self.source = source if isinstance(source, TopicPromptSource) else TopicPromptSource(**source)
        self.summary = summary
        self.embedding = np.array(embedding) if embedding is not None else None

    def serialize(self) -> dict:
        serial = asdict(self)
        serial['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        return serial


def get_clustered_sources_based_on_summaries(sources: list[TopicPromptSource], topic: Topic) -> list[Cluster]:
    """
    Clusters sources based on LLM summaries of the source text
    :param sources:
    :param topic:
    :return:
    """
    key = lambda source: source.summary
    return Artifact(sources).pipe(_summarize_sources_parallel, topic).pipe(_cluster_sources, key, topic).pipe(summarize_source_clusters, topic).data


def get_text_from_source(source: Union[TopicPromptSource, SummarizedSource]) -> str:
    if isinstance(source, SummarizedSource):
        source = source.source
    text = source.text
    return text.get('en', text.get('he', 'N/A'))


def get_clustered_sources(sources: list[TopicPromptSource]) -> list[Cluster]:
    """
    Clusters sources based on the source text. Faster than `get_clustered_sources_based_on_summaries` since it doesn't require summarization
    :param sources:
    :return:
    """
    return Artifact(sources).pipe(_cluster_sources, get_text_from_source).data


def _summarize_source(llm: object, topic_str: str, source: TopicPromptSource):
    source_text = source.text['en'] if len(source.text['en']) > 0 else source.text['he']
    if len(source_text) == 0:
        return None
    summary = summarize_based_on_uniqueness(source_text, topic_str, llm, "English")
    if summary is None:
        return None
    return SummarizedSource(source, summary)


def _summarize_sources_parallel(sources: list[TopicPromptSource], topic: Topic, verbose=True) -> list[SummarizedSource]:
    llm = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
    topic_str = f"Title: '{topic.title}'. Description: '{_get_topic_desc_str(topic)}'."
    return run_parallel(sources, partial(_summarize_source, llm, topic_str), 2,
                        desc="summarize sources", disable=not verbose)

def embed_text_openai(text):
    return np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(text))

def embed_text_voyageai(text):
    return np.array(VoyageAIEmbeddings(model="voyage-large-2-instruct").embed_query(text))

def _make_kmeans_algo(embeddings: np.ndarray, items, topic):
    n_clusters = _guess_optimal_clustering(embeddings)
    return KMeans(n_clusters=n_clusters, n_init='auto', random_state=RANDOM_SEED)

def _make_hdbscan_algo(embeddings: np.ndarray, items, topic, verbose=True):
    IDEAL_NUM_CLUSTERS = 20
    hdbscan_params = {
        "min_samples": [1, 1],
        "min_cluster_size": [2, 2],
        "cluster_selection_method": ["eom", "leaf"],
        "cluster_selection_epsilon": [0.65, 0.5],
    }
    best_hdbscan_obj = None
    highest_clustering_score = 0
    for i in range(len(hdbscan_params["min_samples"])):
        curr_params = {}
        for key, val in hdbscan_params.items():
            curr_params[key] = val[i]
        curr_hdbscan_obj = hdbscan.HDBSCAN(**curr_params)
        results = curr_hdbscan_obj.fit(embeddings)
        curr_clusters, _, _ = _build_clusters_from_cluster_results(results, embeddings, items)
        summarized_clusters = summarize_source_clusters(curr_clusters, topic, verbose=False)
        cluster_summary_embeddings = run_parallel([c.summary for c in summarized_clusters], embed_text_openai, max_workers=40, desc="embedding items for clustering", disable=not verbose)
        distances = pairwise_distances(cluster_summary_embeddings, metric='cosine')
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        num_clusters = len(set(results.labels_))
        avg_min_distance = sum(min_distances)/len(min_distances)
        closeness_to_ideal_score = (1 - (min(abs(num_clusters - IDEAL_NUM_CLUSTERS), 5*IDEAL_NUM_CLUSTERS)/(5*IDEAL_NUM_CLUSTERS))) * 0.7
        clustering_score = closeness_to_ideal_score + avg_min_distance
        print("CURR PARAMS", curr_params, avg_min_distance, num_clusters, closeness_to_ideal_score, clustering_score)
        if clustering_score > highest_clustering_score:
            highest_clustering_score = clustering_score
            print("IDEAL PARAMS", curr_params)
            best_hdbscan_obj = curr_hdbscan_obj

    return best_hdbscan_obj

def _cluster_sources(sources: list[SummarizedSource], key: Callable[[SummarizedSource], str], topic) -> list[Cluster]:

    openai_clusters = cluster_items(sources, key, embed_text_openai, _make_hdbscan_algo, topic)
    # openai_clusters = cluster_items(sources, key, embed_text_openai, _make_kmeans_algo)
    # voyageai_clusters = cluster_items(sources, key, embed_text_voyageai)
    return openai_clusters

def _get_cluster_size_by_source(clusters: list[Cluster]) -> dict:
    pass

def _get_topic_desc_str(topic: Topic) -> str:
    topic_desc = f'{topic.title["en"]}'
    if topic.description.get('en', False) and False:
        topic_desc += f': {topic.description["en"]}'
    return topic_desc

def summarize_source_clusters(clusters: list[Cluster], topic, verbose=True) -> list[Cluster]:
    topic_desc = _get_topic_desc_str(topic)
    summarized_clusters = run_parallel(clusters, partial(summarize_cluster, context=topic_desc, key=get_text_from_source), max_workers=20, desc='summarize source clusters', disable=not verbose)
    if verbose:
        print('---SUMMARIES---')
        for cluster in summarized_clusters:
            print('\t-', len(cluster.items), cluster.summary.strip())
    return summarized_clusters

def summarize_cluster(cluster: Cluster, context: str, key: Callable[[Any], str], sample_size=5) -> Cluster:
    sample = random.sample(cluster.items, min(len(cluster), sample_size))
    strs_to_summarize = [key(item) for item in sample]
    if len(cluster) == 1:
        # assumes items have a summary field
        cluster.summary = cluster.items[0].summary
        return cluster
    llm = ChatOpenAI("gpt-4o", 0)
    system = SystemMessage(content="You are a Jewish scholar familiar with Torah. Given a few ideas (wrapped in <idea> "
                                   "XML tags) about a given topic (wrapped in <topic> XML tags) output a summary of the"
                                   "ideas as they related to the topic. Wrap the output in <summary> tags. Summary"
                                   "should be no more than 10 words.")
    human = HumanMessage(content=f"<topic>{context}</topic><idea>{'</idea><idea>'.join(strs_to_summarize)}</idea>")
    response = llm([system, human])
    summary = get_by_xml_tag(response.content, "summary")
    cluster.summary = summary
    return cluster


def cluster_items(items: list[Any], key: Callable[[Any], str], embedding_fn: Callable[[str], ndarray],
                  get_cluster_algo: Callable, topic, verbose=True) -> list[Cluster]:
    """
    :param items: Generic list of items to cluster
    :param key: function that takes an item from `items` and returns a string to pass to `embedding_fn`
    :param embedding_fn: Given a str (from `key` function) return its embedding
    :param get_cluster_algo:
    :param verbose:
    :return: list of Cluster objects
    """
    embeddings = run_parallel([key(item) for item in items], embedding_fn, max_workers=40, desc="embedding items for clustering", disable=not verbose)
    cluster_results = get_cluster_algo(embeddings, items, topic).fit(embeddings)
    clusters, noise_items, noise_embeddings = _build_clusters_from_cluster_results(cluster_results, embeddings, items, verbose)
    noise_results = _make_kmeans_algo(noise_embeddings, items, topic).fit(noise_embeddings)
    noise_clusters, _, _ = _build_clusters_from_cluster_results(noise_results, noise_embeddings, noise_items, verbose)
    if verbose:
        print("LEN NOISE_CLUSTERS", len(noise_clusters))
    return clusters + noise_clusters


def _build_clusters_from_cluster_results(cluster_results, embeddings, items, verbose=True):
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
            if verbose:
                print('noise cluster', len(curr_items))
            continue
        clusters += [Cluster(label, curr_embeddings, curr_items)]
    return clusters, noise_items, noise_embeddings

def _guess_optimal_clustering(embeddings, verbose=True):
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
    print("Best silhouette score", round(best_sil_coeff, 4))
    return best_num_clusters
