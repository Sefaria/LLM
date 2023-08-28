import dataclasses

import django
django.setup()
import re
from util.sentencizer import sentencize
from tqdm import tqdm
from util.openai import count_tokens_openai
from collections import defaultdict
from typing import List, Any, Tuple, Dict
from sefaria.model import *
from sheet_interface import get_topic_and_orefs
from get_normalizer import get_normalizer
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.cache import SQLiteCache

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, BasePromptTemplate
from langchain.schema import HumanMessage
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
import langchain
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


normalizer = get_normalizer()


class TopromptLLMOutput(BaseModel):
    why: str = Field(description="Why should I care about this source? Limit to one sentence.")
    what: str = Field(description="What do I need to know in order to be able to understand this source? Limit to one sentence.")
    title: str = Field(description="contextualizes the source within the topic. DO NOT mention the source book in the title.")


@dataclasses.dataclass
class Toprompt:
    topic: Topic
    oref: Ref
    prompt: str
    title: str


class TopromptOptions:

    def __init__(self, toprompts: List[Toprompt]):
        self.toprompts = toprompts
        self.oref = toprompts[0].oref
        self.topic = toprompts[0].topic

    def get_titles(self):
        return [toprompt.title for toprompt in self.toprompts]

    def get_prompts(self):
        return [toprompt.prompt for toprompt in self.toprompts]


class TopromptLLMPrompt:

    def __init__(self, lang: str, topic: Topic, oref: Ref):
        self.lang: str = lang
        self.topic: Topic = topic
        self.oref: Ref = oref

    def get(self) -> BasePromptTemplate:
        example_generator = TopromptExampleGenerator(self.lang)
        examples = example_generator.get()
        example_prompt = PromptTemplate.from_template('Source Text: {source_text}\nTopic: {topic}\nOutput: {{{{"why": "{why}", '
                                                      '"what": "{what}", "title": "{title}"}}}}')
        intro_prompt = TopromptLLMPrompt._get_introduction_prompt() + self._get_formatting_prompt()
        input_prompt = self._get_input_prompt()
        format_instructions = _get_output_parser().get_format_instructions()

        example_selector = LengthBasedExampleSelector(
            examples=examples,
            example_prompt=example_prompt,
            max_length=7500-count_tokens_openai(intro_prompt+" "+input_prompt+" "+format_instructions),
            get_text_length=count_tokens_openai
        )
        prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=intro_prompt,
            suffix=input_prompt,
            partial_variables={"format_instructions": format_instructions},
            input_variables=[],
        )
        return prompt

    @staticmethod
    def _get_introduction_prompt() -> str:
        return (
            "# Identity\n"
            "You are a Jewish scholar knowledgeable in all texts relating to Torah, Talmud, Midrash etc. You are writing "
            "for people curious in learning more about Judaism."
            "# Task\n"
            "Write description of a Jewish text such that it persuades the reader to read the full source. The description "
            "should orient them with the essential information they need in order to learn the text. "
            "The title should contextualize the source within the topic; it should be inviting and specific to the source."
            "\n"
        )

    @staticmethod
    def _get_formatting_prompt() -> str:
        return (
            "# Input Format: Input has the following format:\n"
            "Topic: <topic>\n"
            "Source Text: <source text>\n"
            "Source Author: <author>\n"
            "Source Publication Year: <year>\n"
            "Source Book Description (optional): <book description>"
            "Commentary (optional): when it exists, use commentary to inform understanding of `Source Text`. DO NOT refer to the commentary in the final output. Only use the commentary to help understand the source.\n"
        )

    def _get_input_prompt(self) -> str:
        return (
                "{format_instructions} " +
                "# Input:\n" + self._get_input_prompt_details()
        )

    def _get_input_prompt_details(self) -> str:
        index = self.oref.index
        desc_attr = f"{self.lang}Desc"
        book_desc = getattr(index, desc_attr, "N/A")  # getattr(index, "enShortDesc", getattr(index, "enDesc", "N/A"))
        composition_time_period = index.composition_time_period()
        pub_year = composition_time_period.period_string(self.lang) if composition_time_period else "N/A"
        try:
            author_name = Topic.init(index.authors[0]).get_primary_title(self.lang) if len(index.authors) > 0 else "N/A"
        except AttributeError:
            author_name = "N/A"
        source_text = get_ref_text_with_fallback(self.oref, self.lang)
        category = index.get_primary_category()
        prompt = f"Topic: {self.topic.get_primary_title('en')}\n" \
                 f"Source Text: {source_text}\n" \
                 f"Source Author: {author_name}\n" \
                 f"Source Publication Year: {pub_year}"
        if True:  # category not in {"Talmud", "Midrash", "Tanakh"}:
            prompt += f"\nSource Book Description: {book_desc}"
        if category in {"Tanakh"}:
            from summarize_commentary.summarize_commentary import summarize_commentary
            commentary_summary = summarize_commentary(self.oref.normal(), self.topic.slug, company='anthropic')
            prompt += f"\nCommentary: {commentary_summary}"
        return prompt


class ToppromptExample:

    def __init__(self, lang, ref_topic_link: RefTopicLink):
        self.lang = lang
        self.topic = Topic.init(ref_topic_link.toTopic)
        self.source_text = normalizer.normalize(Ref(ref_topic_link.ref).text(lang).ja().flatten_to_string())
        prompt_dict = ref_topic_link.descriptions[lang]
        self.title = prompt_dict['title']
        prompt = prompt_dict['prompt']
        prompt_sents = sentencize(prompt)
        assert len(prompt_sents) == 2
        self.why = prompt_sents[0]
        self.what = prompt_sents[1]

    def serialize(self):
        return {
            "topic": self.topic.get_primary_title(self.lang),
            "source_text": self.source_text,
            "title": self.title,
            "why": self.why,
            "what": self.what,
        }


class TopromptExampleGenerator:

    def __init__(self, lang: str):
        self.lang: str = lang

    def get(self) -> List[dict]:
        toprompts = self._get_existing_toprompts()
        examples = []
        for itopic, ref_topic_link in enumerate(toprompts):
            examples += [ToppromptExample(lang, ref_topic_link)]
        return [example.serialize() for example in examples]

    def _get_existing_toprompts(self):
        link_set = RefTopicLinkSet(self._get_query_for_ref_topic_link_with_prompt())
        # make unique by toTopic
        slug_link_map = {}
        for link in link_set:
            slug_link_map[link.toTopic] = link
        return list(slug_link_map.values())

    def _get_query_for_ref_topic_link_with_prompt(self, slug=None):
        query = {f"descriptions.{self.lang}": {"$exists": True}}
        if slug is not None:
            query['toTopic'] = slug
        return query


def _get_output_parser():
    return PydanticOutputParser(pydantic_object=TopromptLLMOutput)


def get_raw_ref_text(oref: Ref, lang: str) -> str:
    return oref.text(lang).ja().flatten_to_string()


def get_ref_text_with_fallback(oref: Ref, lang: str) -> str:
    raw_text = get_raw_ref_text(oref, lang)
    if len(raw_text) == 0:
        other_lang = "en" if lang == "he" else "he"
        raw_text = get_raw_ref_text(oref, other_lang)

    return normalizer.normalize(raw_text)


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


def _get_toprompt_options(lang: str, topic: Topic, oref: Ref) -> TopromptOptions:
    # TODO pull out formatting from _get_input_prompt_details
    full_language = "English" if lang == "en" else "Hebrew"
    llm_prompt = TopromptLLMPrompt(lang, topic, oref).get()
    llm = ChatOpenAI(model="gpt-4")
    human_message = HumanMessage(content=llm_prompt.format())
    responses = []
    topic_prompts = []
    sescondary_prompt = PromptTemplate.from_template(f"Generate another set of description and title. Refer back to the "
                                                     f"examples provided to stick to the same writing style.\n"
                                                     "{format_instructions}",
                                                     partial_variables={"format_instructions": _get_output_parser().get_format_instructions()})
    for i in range(3):
        response = llm([human_message] + responses)
        responses += [response, HumanMessage(content=sescondary_prompt.format())]

        output_parser = _get_output_parser()
        parsed_output = output_parser.parse(response.content)
        toprompt_text = parsed_output.why + "  ||| " + parsed_output.what
        topic_prompts += [Toprompt(topic, oref, toprompt_text, parsed_output.title)]
    return TopromptOptions(topic_prompts)


def _get_topprompts_for_sheet_id(lang, sheet_id: int) -> List[TopromptOptions]:
    topic, orefs = get_topic_and_orefs(sheet_id)
    toprompt_options = []
    for oref in tqdm(orefs, desc="get toprompts for sheet"):
        toprompt_options += [_get_toprompt_options(lang, topic, oref)]
    return toprompt_options


def output_toprompts_for_sheet_id_list(lang: str, sheet_ids: List[int]) -> None:
    toprompt_options = []
    for sheet_id in sheet_ids:
        toprompt_options += _get_topprompts_for_sheet_id(lang, sheet_id)
    formatter = HTMLFormatter(toprompt_options)
    formatter.save("output/topic_prompts.html")


if __name__ == '__main__':
    # sheet_ids = [502699]  # [502699, 502661, 499080, 498250, 500844]
    sheet_ids = [504798]
    llm_company = "claude"
    lang = "en"
    output_toprompts_for_sheet_id_list(lang, sheet_ids)
