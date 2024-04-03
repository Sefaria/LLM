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
from sefaria_llm_interface.common.topic import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.topic_source_curation import CuratedTopic
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import SystemMessage, HumanMessage
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness


BAD_DATASET = "input/bad_curation.json"
GOOD_DATASET = "input/good_curation.json"


def get_dataset(filename):
    with open(filename, 'r') as fin:
        examples = json.load(fin)
        return [CuratedTopic(**example) for example in examples]


def get_question_answered(source: TopicPromptSource, topic: Topic):
    print("Get question for", source.ref, topic.title['en'])
    topic_str = f"Title: '{topic.title}'. Description: '{topic.description['en']}'."
    return summarize_based_on_uniqueness(source.text['en'], topic_str)


def get_questions_for_topic(curate_topic: CuratedTopic) -> list[str]:
    questions = []
    for source in curate_topic.sources:
        questions.append(get_question_answered(source, curate_topic.topic))
        print(questions[-1])
        print(source.text['en'])
        print('----')
    return questions



if __name__ == '__main__':
    good = get_dataset(GOOD_DATASET)
    example = good[0]
    get_questions_for_topic(example)
