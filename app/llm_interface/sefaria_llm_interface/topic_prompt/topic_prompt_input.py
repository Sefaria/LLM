from typing import List, Union
from dataclasses import dataclass
from sefaria_llm_interface import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource


@dataclass
class TopicPromptInput:
    lang: str
    topic: Topic
    sources: List[TopicPromptSource]

    def __init__(self, lang: str, topic: Union[dict, Topic], sources: List[Union[dict, TopicPromptSource]]):
        self.lang = lang
        self.topic = Topic(**topic)
        self.sources = [s if isinstance(s, TopicPromptSource) else TopicPromptSource(**s) for s in sources]
