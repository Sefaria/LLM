"""
Classes for instantiating objects received from the topic prompt generator
"""
from typing import List, Union
from dataclasses import dataclass


@dataclass
class TopicPrompt:
    title: str
    prompt: str
    ref: str
    slug: str


@dataclass
class TopicPromptGenerationOutput:
    lang: str
    prompts: List[TopicPrompt]

    def __init__(self, lang: str, prompts: List[Union[dict, TopicPrompt]]):
        self.lang = lang
        self.prompts = [p if isinstance(p, TopicPrompt) else TopicPrompt(**p) for p in prompts]
