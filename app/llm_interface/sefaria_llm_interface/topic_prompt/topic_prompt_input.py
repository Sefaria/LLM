from typing import List
from dataclasses import dataclass
from sefaria_llm_interface import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource


@dataclass
class TopicPromptInput:
    lang: str
    topic: Topic
    sources: List[TopicPromptSource]

    @staticmethod
    def create(serial) -> 'TopicPromptInput':
        return TopicPromptInput(**{
            **serial,
            "topic": Topic(**serial['topic']),
            "sources": [TopicPromptSource.deserialize(raw_source) for raw_source in serial['sources']]
        })
