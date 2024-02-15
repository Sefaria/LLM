from typing import Optional, Any, List, Tuple, Dict
from abstract_formatter import AbstractFormatter
from toprompt import TopromptOptions, Toprompt
import csv
from app.sefaria_interface.topic import Topic


class CSVFormatter(AbstractFormatter):

    def _get_csv_rows(self):
        rows = []
        for slug, toprompt_data in self._by_topic.items():
            topic = self._slug_topic_map[slug]
            rows += self._get_csv_rows_for_topic(topic, toprompt_data['toprompt_options'], toprompt_data['gold_standard_prompts'])
        return rows

    def _get_csv_rows_for_topic(self, topic: Topic, toprompt_options_list: List[TopromptOptions], gold_standard_prompts: List[Toprompt]) -> List[dict]:
        rows = []
        for toprompt_options, gold_standard_prompt in zip(toprompt_options_list, gold_standard_prompts):
            rows += [self._get_csv_row(topic, toprompt_options, gold_standard_prompt)]

        return rows

    @staticmethod
    def _get_csv_row(topic, toprompt_options: TopromptOptions, gold_standard_prompt: Toprompt) -> Dict:
        row = {
            "Slug": topic.slug,
            "Topic": topic.title['en'],
            "Ref": toprompt_options.source.ref,
        }
        if gold_standard_prompt:
            row["Gold Standard"] = gold_standard_prompt.prompt_string,
        for i, toprompt in enumerate(toprompt_options.toprompts):
            row[f"Option {i+1}"] = toprompt.prompt_string

        return row

    def save(self, filename: str) -> None:
        rows = self._get_csv_rows()
        with open(filename, 'w') as fout:
            cout = csv.DictWriter(fout, ['Slug', 'Topic', 'Ref', 'Option 1', 'Option 2', 'Option 3', 'Gold Standard'])
            cout.writeheader()
            cout.writerows(rows)
