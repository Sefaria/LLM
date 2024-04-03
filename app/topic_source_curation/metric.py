"""
Ideas for how to measure metrics:

- Diverse: Sources are "different" from one another
    - Vectors are different
        - Calculate average distance between vectors (higher is better)
    - Answers different questions about topic
        - For each source, get question it answers about topic
        - Calculate average distance between vectors (higher is better)
- Related: Sources are closely tied to the topic
    - Basic: Source is about topic (binary decision)
    - Pagerank: what is relative pagerank for this source given the graph of other source related to topic?
    - Average distance between vectors (lower is better)
"""
import json

import numpy as np

from sefaria_llm_interface.common.topic import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.topic_source_curation import CuratedTopic
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import SystemMessage, HumanMessage
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness
from basic_langchain.embeddings import OpenAIEmbeddings
from util.general import embedding_distance
from tqdm import tqdm


BAD_DATASET = "input/bad_curation.json"
GOOD_DATASET = "input/good_curation.json"


def get_dataset(filename):
    with open(filename, 'r') as fin:
        examples = json.load(fin)
        return [CuratedTopic(**example) for example in examples]


def get_question_answered(source: TopicPromptSource, topic: Topic):
    print("Get question for", source.ref, topic.title['en'])
    topic_str = f"Title: '{topic.title}'. Description: '{topic.description.get('en', 'N/A')}'."
    return summarize_based_on_uniqueness(source.text['en'], topic_str)


def get_questions_for_topic(curate_topic: CuratedTopic) -> list[str]:
    questions = []
    for source in tqdm(curate_topic.sources, desc=f'get questions for topic "{curate_topic.topic.title["en"]}"'):
        questions.append(get_question_answered(source, curate_topic.topic))
        print(questions[-1])
        # print(source.text['en'])
        # print('----')
    return questions


def mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    total = 0.0
    count = 0
    for i, embedding in enumerate(embeddings):
        if i > embeddings.shape[0]/2:
            break
        for j, other_embedding in enumerate(embeddings):
            if i == j: continue
            total += embedding_distance(embedding, other_embedding)
            count += 1
    return total/count


def mean_embedding_distance(embedding, other_embeddings) -> float:
    total = 0.0
    for other_embedding in other_embeddings:
        total += embedding_distance(embedding, other_embedding)
    return total / len(other_embeddings)


if __name__ == '__main__':
    llm = OpenAIEmbeddings()
    # good = get_dataset(BAD_DATASET)
    # for example in good[:5]:
    #     questions = get_questions_for_topic(example)
    #     embeddings = llm.embed_documents(questions)
    #     print(mean_pairwise_cosine_distance(embeddings))
    embeddings = llm.embed_documents(["God's decision to end all life on earth due to rampant lawlessness, leading to the Great Flood, as communicated to Noah.", "the divine regret over human wickedness, leading to the decision to wipe out all life on earth, an event later known as the Flood."])

    print(mean_pairwise_cosine_distance(embeddings))
