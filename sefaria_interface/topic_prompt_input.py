from typing import List
from dataclasses import dataclass
from .topic import Topic
from .topic_prompt_source import TopicPromptSource


@dataclass
class TopicPromptInput:
    lang: str
    topic: Topic
    sources: List[TopicPromptSource]

    @staticmethod
    def deserialize(serial) -> 'TopicPromptInput':
        return TopicPromptInput(**{
            **serial,
            "topic": Topic(**serial['topic']),
            "sources": [TopicPromptSource.deserialize(raw_source) for raw_source in serial['sources']]
        })


