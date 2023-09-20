from typing import List, Dict, Tuple
from collections import defaultdict
from toprompt import TopromptOptions, Toprompt
from util.general import get_raw_ref_text
from sefaria.model.topic import Topic


class HTMLFormatter:

    def __init__(self, toprompt_options_list: List[TopromptOptions], gold_standard_prompts: List[Tuple[str, str]]):
        self.toprompt_options_list = toprompt_options_list
        self.gold_standard_prompts = gold_standard_prompts

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

    def _get_full_html(self, by_topic: Dict[str, List[Toprompt]]) -> str:
        html = f"<html><style>{self._get_css_rules()}</style><body>"
        for slug, toprompt_data in by_topic.items():
            topic = Topic.init(slug)
            html += self._get_html_for_topic(topic, toprompt_data['toprompt_options'], toprompt_data['gold_standard_prompts'])
        html += "</body></html>"
        return html

    def _organize_by_topic(self):
        by_topic = defaultdict(lambda: {"toprompt_options": [], "gold_standard_prompts": []})
        for toprompt_options, gold_standard_prompt in zip(self.toprompt_options_list, self.gold_standard_prompts):
            curr_dict = by_topic[toprompt_options.topic.slug]
            curr_dict["toprompt_options"] += [toprompt_options]
            curr_dict["gold_standard_prompts"] += [gold_standard_prompt]
        return by_topic

    def _get_html_for_topic(self, topic: Topic, toprompt_options_list: List[TopromptOptions], gold_standard_prompts: List[Tuple[str, str]]) -> str:
        return f"""
        <h1>{topic.get_primary_title("en")}</h1>
        <div>
            {''.join(
            self._get_html_for_toprompt_options(toprompt_options, gold_standard_prompt) for toprompt_options, gold_standard_prompt in zip(toprompt_options_list, gold_standard_prompts)
        )}
        </div>
        """

    def _get_html_for_toprompt_options(self, toprompt_options: TopromptOptions, gold_standard_prompt: Tuple[str, str]) -> str:
        oref = toprompt_options.oref
        gold_title, gold_prompt = gold_standard_prompt
        return f"""
        <div class="topic-prompt">
            <h2>{oref.normal()}</h2>
            <table>
            <tr><td><h3>{"</h3></td><td><h3>".join(toprompt_options.get_titles())}</h3></td></tr>
            <tr><td><p>{"</p></td><td><p>".join(toprompt_options.get_prompts())}</p></td></tr>
            </table>
            <h3>Gold Standard</h3>
            <h4>{gold_title}</h4>
            <p>{gold_prompt}</p>
            <h3>Text</h3>
            <p class="he">{get_raw_ref_text(oref, "he")}</p>
            <p>{get_raw_ref_text(oref, "en")}</p>
        </div>
        """

    def save(self, filename):
        html = self._get_full_html(self._organize_by_topic())
        with open(filename, "w") as fout:
            fout.write(html)
