import dataclasses
from typing import List
from sefaria_llm_interface.topic_prompt.topic_prompt_source import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic


@dataclasses.dataclass
class Toprompt:
    topic: Topic
    source: TopicPromptSource
    why: str
    what: str
    title: str

    @property
    def prompt_string(self):
        return f"{self.title}\n\n{self.prompt}"

    @property
    def prompt(self):
        return f"{self.why} {self.what}"

    def serialize(self):
        return {
            "title": self.title,
            "prompt": self.prompt,
            "slug": self.topic.slug,
            "ref": self.source.ref,
        }


class TopromptOptions:

    def __init__(self, toprompts: List[Toprompt]):
        self.toprompts = toprompts
        self.source = toprompts[0].source
        self.topic = toprompts[0].topic

    def get_titles(self):
        return [toprompt.title for toprompt in self.toprompts]

    def get_prompts(self):
        return [toprompt.prompt for toprompt in self.toprompts]
