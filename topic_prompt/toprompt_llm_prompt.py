import csv
import random

from util.openai import count_tokens_openai
from util.sentencizer import sentencize
from util.general import get_raw_ref_text, get_ref_text_with_fallback
from typing import List
from sefaria.model import *
from pydantic import BaseModel, Field

from langchain.output_parsers import PydanticOutputParser
from langchain import PromptTemplate, BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
random.seed(23223)


class TopromptLLMOutput(BaseModel):
    why: str = Field(description="Why should I care about this source? Limit to one sentence. Do NOT summarize the source. The goal is to engage the reader which summarizing.")
    what: str = Field(description="What do I need to know in order to be able to understand this source? Limit to one sentence. Do NOT summarize the source. The goal is to engage the reader which summarizing.")
    title: str = Field(description="contextualizes the source within the topic. DO NOT mention the source book in the title.")


class TopromptLLMPrompt:

    def __init__(self, lang: str, topic: Topic, oref: Ref):
        self.lang: str = lang
        self.topic: Topic = topic
        self.oref: Ref = oref

    def get(self) -> BasePromptTemplate:
        example_generator = TopromptExampleGenerator(self.lang)
        examples = example_generator.get()
        example_prompt = PromptTemplate.from_template('<source_text>{source_text}</source_text>\n'
                                                      '<topic>{topic}</topic>\n'
                                                      '<output>{{{{'
                                                      '"why": "{why}", "what": "{what}", "title": "{title}"'
                                                      '}}}}</output>')
        intro_prompt = TopromptLLMPrompt._get_introduction_prompt() + self._get_formatting_prompt()
        input_prompt = self._get_input_prompt()
        format_instructions = get_output_parser().get_format_instructions()

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
            "<identity>\n"
            "You are a Jewish scholar knowledgeable in all texts relating to Torah, Talmud, Midrash etc. You are writing "
            "for people curious in learning more about Judaism."
            "</identity>"
            "<task>\n"
            "Write description of a Jewish text such that it persuades the reader to read the full source. The description "
            "should orient them with the essential information they need in order to learn the text. "
            "The title should contextualize the source within the topic; it should be inviting and specific to the source."
            "</task>"
            "\n"
        )

    @staticmethod
    def _get_formatting_prompt() -> str:
        return (
            "<input_format>Input has the following format:\n"
            "<topic>Name of the topic</topic>\n"
            "<source_text>Text of the source</source_text>\n"
            "<author>Author of the source</author>\n"
            "<publication_year>Year the source was published</publication_year>\n"
            "<book_description>Optional. Description of the source book</book_description>"
            "<commentary> (optional): when it exists, use commentary to inform understanding of `<source_text>`. DO NOT"
            "refer to the commentary in the final output. Only use the commentary to help understand the source."
            "</commentary>\n"
            "</input_format>"
        )

    def _get_input_prompt(self) -> str:
        return (
                "{format_instructions} " +
                "<input>\n" + self._get_input_prompt_details() + "</input>"
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
        prompt = f"<topic>{self.topic.get_primary_title('en')}</topic>\n" \
                 f"<source_text>{source_text}</source_text>\n" \
                 f"<author>{author_name}</author>\n" \
                 f"<publication_year>{pub_year}</publication_year>"
        if True:  # category not in {"Talmud", "Midrash", "Tanakh"}:
            prompt += f"\n<book_description>{book_desc}</book_description>"
        if category in {"Tanakh"}:
            from summarize_commentary.summarize_commentary import summarize_commentary
            commentary_summary = summarize_commentary(self.oref.normal(), self.topic.slug, company='anthropic')
            prompt += f"\n<commentary>{commentary_summary}</commentary>"
        return prompt


class ToppromptExample:

    _hard_coded_sents = {
        'In biblical sources, the Temple is presented as God\'s home. This work of rabbinic interpretations on the Book of Exodus asks the question‚ "Where is God?" in light of the destruction of the Temple.': [
            "In biblical sources, the Temple is presented as God's home.",
            'This work of rabbinic interpretations on the Book of Exodus asks the question‚ "Where is God?" in light of the destruction of the Temple.',
        ],
        'Why is the shofar called a shofar? What does it mean? This ancient midrash from the land of Israel points out that the word “shofar” is spelled in the same order and the same letters as the Hebrew verb that means “to improve” and thereby suggests its meaning.': [
           'Why is the shofar called a shofar? What does it mean?',
           'This ancient midrash from the land of Israel points out that the word “shofar” is spelled in the same order and the same letters as the Hebrew verb that means “to improve” and thereby suggests its meaning.',
        ],
    }

    def __init__(self, lang, ref_topic_link: RefTopicLink):
        self.lang = lang
        self.topic = Topic.init(ref_topic_link.toTopic)
        self.source_text = get_raw_ref_text(Ref(ref_topic_link.ref), lang)
        prompt_dict = ref_topic_link.descriptions[lang]
        self.title = prompt_dict['title']
        prompt = prompt_dict['prompt']
        prompt_sents = self._hard_coded_sents.get(prompt, sentencize(prompt))
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
        # toprompts = self._get_existing_toprompts()
        toprompts = self._get_training_set()
        examples = []
        for itopic, ref_topic_link in enumerate(toprompts):
            examples += [ToppromptExample(self.lang, ref_topic_link)]
        return [example.serialize() for example in examples]

    def _get_training_set(self) -> List[RefTopicLink]:
        ref_topic_links = []
        with open("input/topic_prompt_training_set.csv", "r") as fin:
            cin = csv.DictReader(fin)
            for row in cin:
                if len(RefTopicLinkSet(self._get_query_for_ref_topic_link_with_prompt(slug=row["Slug"]))) == 0:
                    print(row["Slug"])
                    continue

                ref_topic_links += [RefTopicLink(
                    attrs={
                        "ref": row["Reference"],
                        "toTopic": row["Slug"],
                        "descriptions": {
                            self.lang: {
                                "prompt": row["Prompt"],
                                "title": row["Title"],
                            }
                        }
                    }
                )]
        random.shuffle(ref_topic_links)
        return ref_topic_links

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


def get_output_parser():
    return PydanticOutputParser(pydantic_object=TopromptLLMOutput)
