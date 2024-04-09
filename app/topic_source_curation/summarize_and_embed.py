"""
This is an experiment in what happens when we summarize and embed a toipc page
will clear clusters emerge that make it easy to curate?
This would be magical and amazing. I assume there will be no issues whatsoever

Current approach:
- Summarize top sources on a topic page as they relate to the topic
    - need to modify how "top" is defined here
    - need a method of getting a large pool of sources
- Embed each summary
- Cluster embeddings using k means and silhouette score to determine ideal clustering
    - Number of clusters can be very large, they will be trimmed in the next step
- Remove clusters with few elements
    - anything with fewer than 4 elements
    - CONSIDER REMOVING THIS STEP
- Calculate centroids for remaining clusters?
    - Use KNN to segment all embeddings into one of these centroids?
- Sample 5 summaries from each cluster, ask GPT to summarize the summaries
    - This provides a human-readable version of the cluster
- Cluster the clusters?
- Sort the cluster summaries by how "interesting" they are
    - Use cmp_to_key where the comparison function asks GPT to decide which of two summaries is more interesting
- Choose top 15-20 cluster summaries
- For each cluster chosen, choose one source that is the most interesting / fulfills category quota
    - Needs thought
"""
import random
from typing import Any
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
from topic_source_curation.common import get_exported_topic_pages
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness
from util.general import get_by_xml_tag
from sefaria_llm_interface.topic_source_curation import CuratedTopic
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from basic_langchain.embeddings import OpenAIEmbeddings
from basic_langchain.chat_models import ChatOpenAI, ChatAnthropic
from basic_langchain.schema import SystemMessage, HumanMessage
from dataclasses import dataclass

random.seed = 567454

@dataclass
class SummarizedSource:
    source: TopicPromptSource
    summary: str
    embedding: np.ndarray = None


@dataclass
class SourceCluster:
    label: int
    summarized_sources: list[SummarizedSource]
    cluster_summary: str
    embedding: np.ndarray = None

    def __len__(self):
        return len(self.summarized_sources)

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return self.label == other.label


def summarize_topic_page(curated_topic: CuratedTopic) -> list[SummarizedSource]:
    # llm = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    topic = curated_topic.topic
    topic_str = f"Title: '{topic.title}'. Description: '{topic.description.get('en', 'N/A')}'."
    summaries: list[SummarizedSource] = []
    for source in tqdm(curated_topic.sources, desc=f'summarize_topic_page: {topic.title["en"]}', disable=not verbose):
        summary = summarize_based_on_uniqueness(source.text.get('en', source.text['he']), topic_str, llm)
        summaries.append(SummarizedSource(source, summary))
    return summaries


def embed_stuff(documents):
    llm = OpenAIEmbeddings(model='text-embedding-3-large')
    embeddings = llm.embed_documents(documents)
    return embeddings


def guess_optimal_clustering(embeddings):
    best_sil_coeff = -1
    best_num_clusters = 0
    n_clusters = range(2, len(embeddings)//2)
    for n_cluster in tqdm(n_clusters, total=len(n_clusters), desc='guess optimal clustering', disable=not verbose):
        kmeans = KMeans(n_clusters=n_cluster, n_init='auto', random_state=random.seed).fit(embeddings)
        labels = kmeans.labels_
        sil_coeff = silhouette_score(embeddings, labels, metric='cosine', random_state=random.seed)
        if sil_coeff > best_sil_coeff:
            best_sil_coeff = sil_coeff
            best_num_clusters = n_cluster
    return best_num_clusters

def filter_small_clusters(items, kmeans, threshold) -> list[list[Any]]:
    """

    :param items: list of items that were clustered
    :param kmeans: kmeans object from sklearn
    :param threshold: minimum size of cluster that will be kept. anything with fewer elements will be discarded
    :return: clusters where each cluster has at least `threshold` elements
    """
    large_cluster_labels = []
    large_clusters = []
    for label in set(kmeans.labels_):
        indices = np.where(kmeans.labels_ == label)[0]
        if len(indices) >= threshold:
            large_cluster_labels.append(indices)
    for i, indices in enumerate(large_cluster_labels):
        large_clusters += [[items[j] for j in indices]]
    print('Num large clusters:', len(large_cluster_labels))
    return large_clusters

def get_highest_avg_pairwise_distance_indices(embeddings: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(embeddings, metric='cosine')
    sum_distances = np.sum(distances, axis=1)
    avg_distances = sum_distances / (len(embeddings) - 1)
    sorted_indices = np.argsort(avg_distances)[::-1]  # Sort in descending order
    return sorted_indices

def sort_by_highest_avg_pairwise_distance(clusters: list[SourceCluster]):
    embeddings = np.array([c.embedding for c in clusters])
    sorted_indices = get_highest_avg_pairwise_distance_indices(embeddings)
    return [clusters[i] for i in sorted_indices]


def summarize_cluster(topic, summarized_sources: list[SummarizedSource]):
    summaries = [s.summary for s in summarized_sources]
    sample = random.sample(summaries, min(len(summaries), 5))
    # for x in sample:
    #     print(x['source'].ref)
    llm = ChatOpenAI("gpt-3.5-turbo", 0)
    system = SystemMessage(content="You are a Jewish scholar familiar with Torah. Given a few ideas (wrapped in <idea> "
                                   "XML tags) about a given topic (wrapped in <topic> XML tags) output a summary of the"
                                   "ideas as they related to the topic. Wrap the output in <summary> tags. Summary"
                                   "should be no more than 10 words.")
    topic_desc = ''
    if topic.description.get('en', False) and False:
        topic_desc = f': {topic.description["en"]}'
    human = HumanMessage(content=f"<topic>{topic.title['en']}{topic_desc}</topic><idea>{'</idea><idea>'.join(sample)}</idea>")
    response = llm([system, human])
    return get_by_xml_tag(response.content, "summary")


def summarize_all_clusters(topic, clusters):
    cluster_summaries = []
    for cluster in tqdm(clusters, desc="summarize clusters", disable=not verbose):
        cluster_summaries += [summarize_cluster(topic, cluster)]
    return cluster_summaries


def get_gpt_compare(system_prompt, human_prompt_generator, llm):
    def gpt_compare(a, b) -> int:
        response = llm([system_prompt, human_prompt_generator(a, b)])
        print(a)
        print(b)
        print(response.content, int(response.content)*2 - 3)
        return int(response.content)*2 - 3
    return gpt_compare


def sort_by_interest_to_newcomer(documents: list[str]):
    from functools import cmp_to_key
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system = SystemMessage(content="You are a Jewish teacher tasked with finding Torah sources relevant and interesting "
                                   "to a newcomer to Judaism who is curious to explore. Given two ideas (listed in a "
                                   "numbered list with each list item on its own line) output the list item of the"
                                   " idea which would be most interesting to this student. The only output should be "
                                   "either '1' or '2'")
    human_generator = lambda a, b: HumanMessage(content=f"1) {a}\n2) {b}")
    documents.sort(key=cmp_to_key(get_gpt_compare(system, human_generator, llm)))
    for document in documents:
        print(document)


def get_source_clusters(n_clusters: int, topic: Topic, summarized_sources: list[SummarizedSource], min_size: int) -> list[SourceCluster]:
    embeddings = [s.embedding for s in summarized_sources]
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random.seed).fit(embeddings)
    clusters = []
    for label in tqdm(set(kmeans.labels_), desc="summarizing clusters", disable=not verbose):
        indices = np.where(kmeans.labels_ == label)[0]
        curr_sources = [summarized_sources[j] for j in indices]
        if len(curr_sources) < min_size:
            continue
        cluster_summary = summarize_cluster(topic, curr_sources)
        clusters += [SourceCluster(label, curr_sources, cluster_summary)]
    cluster_embeddings = embed_stuff([c.cluster_summary for c in clusters])
    for cluster, embedding in zip(clusters, cluster_embeddings):
        cluster.embedding = embedding
    return clusters

def add_embeddings_to_sources(summarized_sources: list[SummarizedSource]) -> list[SummarizedSource]:
    embeddings = np.array(embed_stuff([s.summary for s in summarized_sources]))
    for embedding, source in zip(embeddings, summarized_sources):
        source.embedding = embedding
    return summarized_sources

def add_embeddings_to_clusters(clusters: list[SourceCluster]) -> list[SourceCluster]:
    embeddings = np.array(embed_stuff([s.cluster_summary for s in clusters]))
    for embedding, cluster in zip(embeddings, clusters):
        cluster.embedding = embedding
    return cluster

def summarize_and_embed(curated_topic: CuratedTopic):
    # make unique
    source_by_ref = {s.ref: s for s in curated_topic.sources}
    curated_topic.sources = list(source_by_ref.values())
    summarized_sources = summarize_topic_page(curated_topic)
    summarized_sources = add_embeddings_to_sources(summarized_sources)
    n_clusters = guess_optimal_clustering([s.embedding for s in summarized_sources])
    print(f"Optimal Clustering: {n_clusters}")
    source_clusters = get_source_clusters(n_clusters, curated_topic.topic, summarized_sources, 3)
    print(f"Num source clusters: {len(source_clusters)}")
    # sort_by_interest_to_newcomer(cluster_summaries)
    source_clusters = sort_by_highest_avg_pairwise_distance(source_clusters)
    for cluster in source_clusters:
        print(cluster.cluster_summary)


if __name__ == '__main__':
    verbose = True
    topic_pages = get_exported_topic_pages()
    summarize_and_embed(topic_pages[1])


