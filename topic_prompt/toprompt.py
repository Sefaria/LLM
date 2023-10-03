import dataclasses
from typing import List
from sefaria.model.topic import Topic
from sefaria.model.text import Ref


@dataclasses.dataclass
class Toprompt:
    topic: Topic
    oref: Ref
    prompt: str
    title: str

    @property
    def prompt_string(self):
        return f"{self.title}\n\n{self.prompt}"


class TopromptOptions:

    def __init__(self, toprompts: List[Toprompt]):
        self.toprompts = toprompts
        self.oref = toprompts[0].oref
        self.topic = toprompts[0].topic

    def get_titles(self):
        return [toprompt.title for toprompt in self.toprompts]

    def get_prompts(self):
        return [toprompt.prompt for toprompt in self.toprompts]
