from typing import Any, Callable, Union
from tqdm import tqdm
from langchain_voyageai import VoyageAIEmbeddings
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from numpy import ndarray
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from basic_langchain.chat_models import ChatAnthropic
from basic_langchain.schema import HumanMessage, SystemMessage
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
        serial['embedding'] = self.embedding.tolist()
        return serial


def get_clustered_sources_based_on_summaries(sources: list[TopicPromptSource], topic: Topic) -> list[Cluster]:
    """
    Clusters sources based on LLM summaries of the source text
    :param sources:
    :param topic:
    :return:
    """
    key = lambda source: source.summary
    return Artifact(sources).pipe(_summarize_sources_parallel, topic).pipe(_cluster_sources, key).pipe(summarize_source_clusters, topic).data


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


def _summarize_source(source: TopicPromptSource, llm: object, topic_str: str):
    source_text = source.text['en'] if len(source.text['en']) > 0 else source.text['he']
    if len(source_text) == 0:
        return None
    summary = summarize_based_on_uniqueness(source_text, topic_str, llm, "English")
    if summary is None:
        return None
    return SummarizedSource(source, summary)


def _summarize_sources_parallel(sources: list[TopicPromptSource], topic: Topic, verbose=True) -> list[SummarizedSource]:
    llm = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
    topic_str = f"Title: '{topic.title}'. Description: '{topic.description.get('en', 'N/A')}'."

    def _summarize_source_pbar(pbar, source, llm, topic_str):
        summarized_source = _summarize_source(source, llm, topic_str)
        with pbar.get_lock():
            pbar.update(1)
        return summarized_source


    with tqdm(total=len(sources), desc=f'summarize_topic_page: {topic.title["en"]}', disable=not verbose) as pbar:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for source in sources:
                futures.append(executor.submit(_summarize_source_pbar, pbar, source, llm, topic_str))

    summaries = [future.result() for future in futures if future.result() is not None]
    return summaries

def embed_text(text):
    return np.array(VoyageAIEmbeddings(model="voyage-large-2-instruct", batch_size=1).embed_query(text))

def _cluster_sources(sources: list[SummarizedSource], key: Callable[[SummarizedSource], str]) -> list[Cluster]:
    return cluster_items(sources, key, embed_text)

def _get_topic_desc_str(topic: Topic) -> str:
    topic_desc = f'{topic.title["en"]}'
    if topic.description.get('en', False) and False:
        topic_desc += f': {topic.description["en"]}'
    return topic_desc

def summarize_source_clusters(clusters: list[Cluster], topic, verbose=True) -> list[Cluster]:
    summarized_clusters = [summarize_cluster(cluster, _get_topic_desc_str(topic), get_text_from_source) for cluster in tqdm(clusters, desc='summarize source clusters', disable=not verbose)]
    if verbose:
        print('---SUMMARIES---')
        for cluster in summarized_clusters:
            print('\t-', len(cluster.items), cluster.summary.strip())
    return summarized_clusters

def summarize_cluster(cluster: Cluster, context: str, key: Callable[[Any], str], sample_size=5) -> Cluster:
    sample = random.sample(cluster.items, min(len(cluster), sample_size))
    strs_to_summarize = [key(item) for item in sample]
    llm = ChatAnthropic("claude-3-opus-20240229", 0)
    system = SystemMessage(content="You are a Jewish scholar familiar with Torah. Given a few ideas (wrapped in <idea> "
                                   "XML tags) about a given topic (wrapped in <topic> XML tags) output a summary of the"
                                   "ideas as they related to the topic. Wrap the output in <summary> tags. Summary"
                                   "should be no more than 10 words.")
    human = HumanMessage(content=f"<topic>{context}</topic><idea>{'</idea><idea>'.join(strs_to_summarize)}</idea>")
    response = llm([system, human])
    summary = get_by_xml_tag(response.content, "summary")
    cluster_dict = asdict(cluster)
    cluster_dict.pop('summary', None)
    return Cluster(summary=summary, **cluster_dict)


def cluster_items(items: list[Any], key: Callable[[Any], str], embedding_fn: Callable[[str], ndarray], verbose=True) -> list[Cluster]:
    """
    :param items: Generic list of items to cluster
    :param key: function that takes an item from `items` and returns a string to pass to `embedding_fn`
    :param embedding_fn: Given a str (from `key` function) return its embedding
    :return: list of Cluster objects
    """
    embeddings = [embedding_fn(key(item)) for item in tqdm(items, desc="embedding items for clustering", disable=not verbose)]
    n_clusters = _guess_optimal_clustering(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=RANDOM_SEED).fit(embeddings)
    clusters = []
    for label in set(kmeans.labels_):
        indices = np.where(kmeans.labels_ == label)[0]
        curr_embeddings = [embeddings[j] for j in indices]
        curr_items = [items[j] for j in indices]
        clusters += [Cluster(label, curr_embeddings, curr_items)]
    return clusters


def _guess_optimal_clustering(embeddings, verbose=True):
    best_sil_coeff = -1
    best_num_clusters = 0
    n_clusters = range(2, len(embeddings)//2)
    for n_cluster in tqdm(n_clusters, total=len(n_clusters), desc='guess optimal clustering', disable=not verbose):
        kmeans = KMeans(n_clusters=n_cluster, n_init='auto', random_state=RANDOM_SEED).fit(embeddings)
        labels = kmeans.labels_
        sil_coeff = silhouette_score(embeddings, labels, metric='cosine', random_state=RANDOM_SEED)
        if sil_coeff > best_sil_coeff:
            best_sil_coeff = sil_coeff
            best_num_clusters = n_cluster
    return best_num_clusters
