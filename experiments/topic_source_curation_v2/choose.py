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
from experiments.topic_source_curation_v2.cluster import Cluster, SummarizedSource, embed_text_openai, get_text_from_source
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sklearn.metrics import pairwise_distances
from util.pipeline import Artifact
from functools import reduce
from statistics import mean, stdev
import numpy as np
from basic_langchain.schema import HumanMessage, SystemMessage
from basic_langchain.chat_models import ChatOpenAI
from sefaria_llm_interface.common.topic import Topic
from solver import solve_clusters

def choose_ideal_sources_for_clusters(clusters: list[Cluster], topic: Topic) -> list[TopicPromptSource]:
    # return Artifact(clusters).pipe(sort_clusters, 20, topic).pipe(choose_ideal_sources).data
    return Artifact(clusters).pipe(sort_clusters, 20, topic).pipe(solve_clusters).data


def choose_primary_sources(clusters: list[Cluster]) -> list[str]:
    """
    Returns list of primary sources as trefs
    :param clusters:
    :return:
    """
    orefs = reduce(lambda x, y: x + [Ref(item.source.ref) for item in y.items], clusters, [])
    trefs, pageranks = zip(*pagerank_rank_ref_list(orefs))
    max_ref = trefs[0]
    thresh = mean(pageranks) + 2 * stdev(pageranks)
    is_big = pageranks[0] > thresh
    print(max_ref, "IS BIG:", is_big, pageranks[0], thresh)
    return [max_ref]


def choose_ideal_clusters(clusters: list[Cluster], max_clusters: int) -> list[Cluster]:
    """
    Sorted in descending order from outliers to central
    Choose a few central clusters and a few outliers
    Also might want to use custom GPT sort to find "best" clusters based on various criteria
    """
    # sorted_clusters = _sort_by_highest_avg_pairwise_distance(clusters)
    # sorted_clusters = _sort_clusters_by_instruction(clusters)
    return [c for c in clusters if len(clusters) > 1]

def sort_clusters(clusters: list[Cluster], max_clusters: int, topic:Topic) -> list[Cluster]:
    # sorted_clusters = _sort_by_highest_avg_pairwise_distance(clusters)
    sorted_clusters = _sort_clusters_by_instruction(clusters, topic)
    for cluster in sorted_clusters:
        cluster.items = _sort_within_cluster(cluster, topic)
    return sorted_clusters

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

def _sort_by_query_voayageai(documents: list[str], query:str):
    vo = voyageai.Client()
    output = vo.rerank(query, documents, "rerank-lite-1")
    sorted_by_relevance = [result.document for result in output.results]
    # best_idx = output.results[0].index
    return sorted_by_relevance




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
def get_gpt_compare(system_prompt, human_prompt_generator, llm):
    content_to_val = {"1":-1, "2":1, "0":0}
    def gpt_compare(a, b) -> int:
        response = llm([system_prompt, human_prompt_generator(a, b)])
        # print(a)
        # print(b)
        # print(response.content, content_to_val.get(response.content, 0))
        return content_to_val.get(response.content, 0)

    return gpt_compare


def sort_by_instruction(documents,  comparison_instruction, key_extraction_func=lambda x:x):
    from functools import cmp_to_key
    message_suffix = " The only output should be either '1' or '2' or '0'"
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system = SystemMessage(
        content=
        comparison_instruction
        +message_suffix)
    human_generator = lambda a, b: HumanMessage(content=f"1) {key_extraction_func(a)}\n2) {key_extraction_func(b)}")
    documents.sort(key=cmp_to_key(get_gpt_compare(system, human_generator, llm)))
    # for document in documents:
    #     print(document)
    return documents
def _apply_sorting_to_clusters(clusters: list[Cluster], summaries:[str]) -> list[Cluster]:
    sorted = []
    for summary in summaries:
        cluster = [cluster for cluster in clusters if cluster.summary==summary][0]
        sorted.append(cluster)
    return sorted
# def _apply_sorting_to_sources_within_cluster(cluster: Cluster, summaries:[str]) -> list[Cluster]:
#     sorted = []
#     summarized_sources = cluster.items
#     for summary in summaries:
#         summarized_source = [summarized_source for summarized_source in summarized_sources if summarized_source.summary==summary][0]
#         sorted.append(summarized_source)
#     cluster.items = sorted
#     return sorted
def _apply_sorting_to_sources_within_cluster(cluster: Cluster, texts:[str]) -> list[Cluster]:
    sorted = []
    summarized_sources = cluster.items
    for text in texts:
        # summarized_source = [summarized_source for summarized_source in summarized_sources if summarized_source.source.text["en"] == text][0]
        summarized_source = [summarized_source for summarized_source in summarized_sources if summarized_source.summary == text][0]
        sorted.append(summarized_source)
    cluster.items = sorted
    return sorted
def _sort_clusters_by_instruction(clusters: list[Cluster], topic: Topic) -> list[Cluster]:
    summaries = [c.summary for c in clusters]

    interestingness_instruction =f"""You are an expert in Jewish laws, history and traditions, wishing to teach your student about {topic.title} in light of Jewish tradition.
    Given 2 topics related to {topic.title}, output the index of the one which presents a more interesting, surprising, wild and/or non-trivial information regarding {topic.title}, which might be captivating and intriguing to your students. If both are equally non-trivial and interesting with regards to {topic.title}, output 0.  
    """
    fundamentalness_instruction ="""You are an expert in Jewish laws, history and traditions, wishing to teach your student about the historical person of Cyrus in light of Jewish tradition.
    Given 2 topics related to king Cyrus, output the index of the one which presents a more fundamental and basic fact about Cyrus, one that your students should learn first before learning the other. If both are equally fundamental and no one is more important than the other, output 0.  
    """
    # interesting = sort_by_instruction(summaries[:], interestingness_instruction)
    interesting = sort_by_instruction(clusters, interestingness_instruction, lambda cluster: cluster.summary)
    # fundamental = sort_by_instruction(summaries[:], fundamentalness_instruction)


    # interesting_voyageai = _sort_by_query_voayageai(summaries, "I am looking for the most surpsring, interesting, wild and non-trivial sources about king Cyrus as is reflected in Jewish sources and tradition")
    # from pprint import pprint
    # pprint("GPT INTERESTINGNESS SORT:")
    # pprint(interesting)
    #
    # pprint("VOYAGEAI INTERESTINGESS SORT:")
    # pprint(interesting_voyageai)
    a = "halt"

    return interesting

def _sort_within_cluster(cluster: Cluster, topic: Topic):
    if len(cluster.items) <= 1:
        return cluster
    interestingness_instruction = f"""You are an expert in Jewish laws, history and traditions, wishing to teach your student about {topic.title} in light of Jewish tradition.
        Given 2 texts related to {topic.title}, output the index of the one which presents a more interesting, surprising, wild and/or non-trivial information regarding {topic.title}, which might be captivating and intriguing to your students. If both are equally non-trivial and interesting, output 0.  
        """
    interesting = sort_by_instruction(cluster.items, interestingness_instruction, lambda item: item.summary)
    return interesting
