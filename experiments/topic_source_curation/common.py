import json

import numpy as np

from sefaria_llm_interface.common.topic import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.topic_source_curation import CuratedTopic
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import SystemMessage, HumanMessage
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness
from basic_langchain.embeddings import OpenAIEmbeddings
from basic_langchain.chat_models import ChatOpenAI
from util.general import embedding_distance
from tqdm import tqdm

BAD_DATASET = "input/bad_curation.json"
GOOD_DATASET = "input/good_curation.json"
EXPORTED_TOPIC_PAGES = "input/exported_topic_pages.json"


def _get_dataset(filename):
    with open(filename, 'r') as fin:
        examples = json.load(fin)
        return [CuratedTopic(**example) for example in examples]


def get_datasets():
    return _get_dataset(BAD_DATASET), _get_dataset(GOOD_DATASET)


def get_exported_topic_pages():
    return _get_dataset(EXPORTED_TOPIC_PAGES)
