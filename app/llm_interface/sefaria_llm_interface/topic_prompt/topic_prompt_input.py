from typing import List
from dataclasses import dataclass
from sefaria_llm_interface import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource


@dataclass
class TopicPromptInput:
    lang: str
    topic: Topic
    sources: List[TopicPromptSource]

    def __init__(self, lang: str, topic: dict, sources: List[dict]):
        self.lang = lang
        self.topic = Topic(**topic)
        self.sources = [TopicPromptSource(**raw_source) for raw_source in sources]
