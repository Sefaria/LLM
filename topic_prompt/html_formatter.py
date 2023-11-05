from typing import List, Dict, Tuple, Union
from collections import defaultdict
from toprompt import TopromptOptions, Toprompt
from util.general import get_raw_ref_text
from sefaria.model.topic import Topic
from abstract_formatter import AbstractFormatter
import math


class HTMLFormatter(AbstractFormatter):

    @staticmethod
    def _get_css_rules():
        return """
        body {
            width: 800px;
            display: flex;
            align-items: center;
            flex-direction: column;
            margin-right: auto;
            margin-left: auto;
        }
        .topic-prompt {
            display: flex;
            flex-direction: column;
            align-items: center;
        } 
        .he {
            direction: rtl;
        }
        td p, td h3 {
            margin-left: 15px;
            margin-right: 15px;
        }
        """

    def _get_full_html(self) -> str:
        html = f"<html><style>{self._get_css_rules()}</style><body>"
        for slug, toprompt_data in self._by_topic.items():
            topic = Topic.init(slug)
            html += self._get_html_for_topic(topic, toprompt_data['toprompt_options'], toprompt_data['gold_standard_prompts'])
        html += "</body></html>"
        return html

    def _get_html_for_topic(self, topic: Topic, toprompt_options_list: List[TopromptOptions], gold_standard_prompts: List[Toprompt]) -> str:
        return f"""
        <h1>{topic.get_primary_title("en")}</h1>
        <div>
            {''.join(
            self._get_html_for_toprompt_options(toprompt_options, gold_standard_prompt) for toprompt_options, gold_standard_prompt in zip(toprompt_options_list, gold_standard_prompts)
        )}
        </div>
        """

    def _get_html_for_toprompt_options(self, toprompt_options: TopromptOptions, gold_standard_prompt: Toprompt) -> str:
        oref = toprompt_options.oref
        all_toprompts = toprompt_options.toprompts
        has_gold_standard = gold_standard_prompt is not None
        if has_gold_standard:
            all_toprompts += [gold_standard_prompt]
        gold_style = 'border: 2px solid gold; background-color: #fffaf0;'
        all_toprompts_html = [f"""
        <div style="{gold_style if i == len(all_toprompts) - 1 and has_gold_standard else ''}">
        <h3>{toprompt.title}</h3>
        <p>{toprompt.prompt}</p> 
        </div>
        """ for i, toprompt in enumerate(all_toprompts)]
        return f"""
        <div class="topic-prompt">
            <h2>{oref.normal()}</h2>
            {HTMLFormatter._get_n_column_table(2, all_toprompts_html)}
            <h3>Text</h3>
            <p class="he">{get_raw_ref_text(oref, "he")}</p>
            <p>{get_raw_ref_text(oref, "en")}</p>
        </div>
        """

    @staticmethod
    def _get_n_column_table(n, items):
        table_width = 800
        td_style = f'"width: {table_width/n}px;"'
        return f"""
        <table style="table-layout: fixed;">
        {"".join(f"<tr>{''.join(f'<td style={td_style}>{item}</td>' for item in items[i*n:(i*n)+n])}</tr>" for i in range(math.ceil(len(items) / n)))}
        </table>
        """

    def save(self, filename):
        html = self._get_full_html()
        with open(filename, "w") as fout:
            fout.write(html)
