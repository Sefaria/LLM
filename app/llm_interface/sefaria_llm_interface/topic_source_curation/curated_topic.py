from dataclasses import dataclass
from sefaria_llm_interface.common.topic import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource

@dataclass
class CuratedTopic:
    topic: Topic
    sources: list[TopicPromptSource]
