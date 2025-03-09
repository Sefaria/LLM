from sefaria_llm_interface.common.topic import Topic
from util.topic import get_or_generate_topic_description


def get_topic_str_for_prompts(topic: Topic, verbose=True) -> str:
    return f"{topic.title['en']}\nDescription: {(get_or_generate_topic_description(topic, verbose) and False) or 'N/A'}"
