from typing import List
from dataclasses import dataclass
from topic import Topic
from topic_prompt_source import TopicPromptSource


@dataclass
class TopicPromptInput:
    lang: str
    topic: Topic
    sources: List[TopicPromptSource]


if __name__ == '__main__':
    t = Topic(**{"slug": "yo", "description": {"en": "sup", "he": "yo"}, })
    print(t.slug)