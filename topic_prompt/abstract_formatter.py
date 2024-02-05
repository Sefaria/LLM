from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Tuple, Union
from toprompt import TopromptOptions, Toprompt


class AbstractFormatter(ABC):

    def __init__(self, toprompt_options_list: List[TopromptOptions], gold_standard_prompts: List[Toprompt] = None):
        self.toprompt_options_list = toprompt_options_list
        self.gold_standard_prompts = gold_standard_prompts
        self._by_topic: Dict[str, Dict[str, Union[List[TopromptOptions], List[Toprompt]]]] = self._organize_by_topic()

    def _organize_by_topic(self) -> Dict[str, Dict[str, Union[List[TopromptOptions], List[Toprompt]]]]:
        by_topic = defaultdict(lambda: {"toprompt_options": [], "gold_standard_prompts": []})
        gold_standard_prompts = self.gold_standard_prompts or [None] * len(self.toprompt_options_list)
        for toprompt_options, gold_standard_prompt in zip(self.toprompt_options_list, gold_standard_prompts):
            curr_dict = by_topic[toprompt_options.topic.slug]
            curr_dict["toprompt_options"] += [toprompt_options]
            curr_dict["gold_standard_prompts"] += [gold_standard_prompt]
        return by_topic

    @abstractmethod
    def save(self, filename: str) -> None:
        pass

