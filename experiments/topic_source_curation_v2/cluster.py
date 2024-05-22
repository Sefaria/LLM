from typing import Callable, Union
from functools import partial
from tqdm import tqdm
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import HumanMessage, SystemMessage
import random
from hdbscan import HDBSCAN
from dataclasses import dataclass, asdict
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from basic_langchain.chat_models import ChatAnthropic
from basic_langchain.embeddings import VoyageAIEmbeddings, OpenAIEmbeddings
from util.pipeline import Artifact
from util.general import get_by_xml_tag, run_parallel
from util.cluster import Cluster, HDBSCANOptimizerClusterer, StandardClusterer, AbstractClusterItem
import numpy as np

RANDOM_SEED = 567454
random.seed(RANDOM_SEED)


@dataclass
class SummarizedSource(AbstractClusterItem):
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

    def get_str_to_summarize(self) -> str:
        return self.summary

    def get_str_to_embed(self) -> str:
        return self.summary



def get_clustered_sources_based_on_summaries(sources: list[TopicPromptSource], topic: Topic) -> list[Cluster]:
    """
    Clusters sources based on LLM summaries of the source text
    :param sources:
    :param topic:
    :return:
    """
    return Artifact(sources).pipe(_summarize_sources_parallel, topic).pipe(_cluster_sources, topic).data


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

def _get_cluster_summary_based_on_topic(topic_desc, strs_to_summarize):
    llm = ChatOpenAI("gpt-4o", 0)
    system = SystemMessage(content="You are a Jewish scholar familiar with Torah. Given a few ideas (wrapped in <idea> "
                                   "XML tags) about a given topic (wrapped in <topic> XML tags) output a summary of the"
                                   "ideas as they related to the topic. Wrap the output in <summary> tags. Summary"
                                   "should be no more than 10 words.")
    human = HumanMessage(content=f"<topic>{topic_desc}</topic><idea>{'</idea><idea>'.join(strs_to_summarize)}</idea>")
    response = llm([system, human])
    return get_by_xml_tag(response.content, "summary")

def _cluster_sources(sources: list[SummarizedSource], topic) -> list[Cluster]:
    topic_desc = _get_topic_desc_str(topic)
    # get_cluster_algo will be optimized by HDBSCANOptimizerClusterer
    clusterer = StandardClusterer(embed_text_openai, lambda x: HDBSCAN(),
                                  partial(_get_cluster_summary_based_on_topic, topic_desc))
    clusterer_optimizer = HDBSCANOptimizerClusterer(clusterer)
    return clusterer_optimizer.cluster_and_summarize(sources)

def _get_topic_desc_str(topic: Topic) -> str:
    topic_desc = f'{topic.title["en"]}'
    if topic.description.get('en', False) and False:
        topic_desc += f': {topic.description["en"]}'
    return topic_desc
