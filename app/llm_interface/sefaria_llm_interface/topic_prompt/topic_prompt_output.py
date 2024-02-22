"""
Classes for instantiating objects received from the topic prompt generator
"""
from typing import List
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

    def __init__(self, lang: str, prompts: List[dict]):
        self.lang = lang
        self.prompts = [TopicPrompt(**raw_prompt) for raw_prompt in prompts]
