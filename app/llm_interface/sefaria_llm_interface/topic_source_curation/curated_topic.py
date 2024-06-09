from dataclasses import dataclass
from typing import Union
from sefaria_llm_interface.common.topic import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource

@dataclass
class CuratedTopic:
    topic: Topic
    sources: list[TopicPromptSource]

    def __init__(self, topic: Union[dict, Topic], sources: list[Union[dict, TopicPromptSource]]):
        self.topic = topic if isinstance(topic, Topic) else Topic(**topic)
        self.sources = [s if isinstance(s, TopicPromptSource) else TopicPromptSource(**s) for s in sources]
