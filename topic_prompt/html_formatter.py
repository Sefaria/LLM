from typing import List, Dict
from collections import defaultdict
from toprompt import TopromptOptions, Toprompt
from util.general import get_raw_ref_text
from sefaria.model.topic import Topic


class HTMLFormatter:

    def __init__(self, toprompt_options_list: List[TopromptOptions]):
        self.toprompt_options_list = toprompt_options_list

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
        for slug, toprompt_options_list in by_topic.items():
            topic = Topic.init(slug)
            html += self._get_html_for_topic(topic, toprompt_options_list)
        html += "</body></html>"
        return html

    def _organize_by_topic(self):
        by_topic = defaultdict(list)
        for toprompt_options in self.toprompt_options_list:
            by_topic[toprompt_options.topic.slug] += [toprompt_options]
        return by_topic

    def _get_html_for_topic(self, topic: Topic, toprompt_options_list: List[TopromptOptions]) -> str:
        return f"""
        <h1>{topic.get_primary_title("en")}</h1>
        <div>
            {''.join(
            self._get_html_for_toprompt_options(toprompt_options) for toprompt_options in toprompt_options_list
        )}
        </div>
        """

    def _get_html_for_toprompt_options(self, toprompt_options: TopromptOptions) -> str:
        oref = toprompt_options.oref
        return f"""
        <div class="topic-prompt">
            <h2>{oref.normal()}</h2>
            <table>
            <tr><td><h3>{"</h3></td><td><h3>".join(toprompt_options.get_titles())}</h3></td></tr>
            <tr><td><p>{"</p></td><td><p>".join(toprompt_options.get_prompts())}</p></td></tr>
            </table>
            <h3>Text</h3>
            <p class="he">{get_raw_ref_text(oref, "he")}</p>
            <p>{get_raw_ref_text(oref, "en")}</p>
        </div>
        """

    def save(self, filename):
        html = self._get_full_html(self._organize_by_topic())
        with open(filename, "w") as fout:
            fout.write(html)
