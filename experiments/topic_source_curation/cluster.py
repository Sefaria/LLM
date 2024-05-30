from typing import Union
from functools import partial, reduce
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import HumanMessage, SystemMessage
import random
from hdbscan import HDBSCAN
from sklearn.cluster import AffinityPropagation
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from basic_langchain.embeddings import VoyageAIEmbeddings, OpenAIEmbeddings
from util.pipeline import Artifact
from util.general import get_by_xml_tag
from util.cluster import Cluster, OptimizingClusterer, SklearnClusterer, AbstractClusterItem
from experiments.topic_source_curation.common import get_topic_str_for_prompts
from experiments.topic_source_curation.summarized_source import SummarizedSource
import numpy as np

RANDOM_SEED = 567454
random.seed(RANDOM_SEED)


def get_clustered_sources_based_on_summaries(sources: list[TopicPromptSource], topic: Topic) -> list[Cluster]:
    """
    Clusters sources based on LLM summaries of the source text
    :param sources:
    :param topic:
    :return:
    """
    return Artifact(sources).pipe(_cluster_sources, topic).data


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


def embed_text_openai(text):
    return np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(text))


def embed_text_voyageai(text):
    return np.array(VoyageAIEmbeddings(model="voyage-large-2-instruct").embed_query(text))


def _get_cluster_summary_based_on_topic(topic_desc, strs_to_summarize):
    llm = ChatOpenAI("gpt-4o", 0)
    system = SystemMessage(content="You are a Jewish scholar familiar with Torah. Given a few ideas (wrapped in <idea> "
                                   "XML tags) about a given topic (wrapped in <topic> XML tags) output a summary of the "
                                   "ideas as they related to the topic. Wrap the output in <summary> tags. Summary "
                                   "should be no more than 10 words.")
    human = HumanMessage(content=f"<topic>{topic_desc}</topic><idea>{'</idea><idea>'.join(strs_to_summarize)}</idea>")
    response = llm([system, human])
    summary = get_by_xml_tag(response.content, "summary")
    if not summary and '<summary>' in response.content:
        summary = response.content.replace('<summary>', '').replace('</summary>', '')
    return summary or 'N/A'


HDBSCAN_PARAM_OPTS = {
    "min_samples": [1, 1],
    "min_cluster_size": [2, 2],
    "cluster_selection_method": ["eom", "leaf"],
    "cluster_selection_epsilon": [0.65, 0.5],
}


def _get_ith_hdbscan_params(i):
    return reduce(lambda x, y: {**x, y[0]: y[1][i]}, HDBSCAN_PARAM_OPTS.items(), {})


def _cluster_sources(sources: list[SummarizedSource], topic) -> list[Cluster]:
    topic_desc = get_topic_str_for_prompts(topic, verbose=False)
    clusterers = []
    for i in range(len(HDBSCAN_PARAM_OPTS['min_samples'])):
        hdbscan_params = _get_ith_hdbscan_params(i)
        temp_clusterer = SklearnClusterer(embed_text_openai,
                                           lambda x: HDBSCAN(**hdbscan_params).fit(x).labels_,
                                           partial(_get_cluster_summary_based_on_topic, topic_desc), verbose=False)
        clusterers.append(temp_clusterer)

    # temp_clusterer = SklearnClusterer(embed_text_openai,
    #                                    lambda x: AffinityPropagation(damping=0.7).fit(x).predict(x),
    #                                    partial(_get_cluster_summary_based_on_topic, topic_desc))
    # clusterers.append(temp_clusterer)
    clusterer_optimizer = OptimizingClusterer(embed_text_openai, clusterers)
    return clusterer_optimizer.cluster_and_summarize(sources)
