import json
from dataclasses import dataclass
import random

from util.openai import count_tokens_openai
from topic_prompt.uniqueness_of_source import get_uniqueness_of_source
from topic_prompt.contextualize import get_context
from typing import List
from sefaria_interface.topic import Topic
from sefaria_interface.topic_prompt_source import TopicPromptSource
from pydantic import BaseModel, Field

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate, BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
random.seed(23223)


class TopromptLLMOutput(BaseModel):
    why: str = Field(description="Why should I care about this source? Focus on <unique_aspect> to explain why the user "
                                 "should care about this source. Limit to ONE sentence.")
    what: str = Field(description="What do I need to know in order to be able to understand this source? Limit to one "
                                  "sentence. Do NOT summarize the source. The goal is to engage the reader without "
                                  "summarizing.")
    title: str = Field(description="Contextualizes the source within the topic. DO NOT mention the source book in the "
                                   "title.")


class TopromptLLMPrompt:

    def __init__(self, lang: str, topic: Topic, source: TopicPromptSource, other_sources: List[TopicPromptSource]):
        self.lang: str = lang
        self.topic: Topic = topic
        self.source: TopicPromptSource = source
        self.other_sources: List[TopicPromptSource] = other_sources

    def get(self) -> BasePromptTemplate:
        example_generator = TopromptExampleGenerator(self.lang)
        examples = example_generator.get()
        example_prompt = PromptTemplate.from_template('<topic>{topic}</topic>\n'
                                                      '<unique_aspect>{unique_aspect}</unique_aspect>'
                                                      '<context>{context}</context>'
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
            "<author>Author of the source</author>\n"
            "<publication_year>Year the source was published</publication_year>\n"
            "<book_description>Optional. Description of the source book</book_description>"
            "<commentary> (optional): when it exists, use commentary to inform understanding of `<unique_aspect>`. DO NOT"
            " refer to the commentary in the final output. Only use the commentary to help understand the source."
            "</commentary>\n"
            "<unique_aspect> Unique perspective this source has on the topic. Use this to understand why a user would "
            "want to learn this source for this topic.</unique_aspect>\n"
            "<context> (optional): when it exists this provides further context about the source. Use this to provide"
            " more context to the reader."
            "</input_format>"
        )

    def _get_input_prompt(self) -> str:
        return (
                "{format_instructions} " +
                "<input>\n" + self._get_input_prompt_details() + "</input>"
        )

    def _get_book_description(self, source: TopicPromptSource):
        """
        Modify book description so its more specific for Yerushalmi and Mishnah. This helps the LLM better describe the source
        :param source:
        :return:
        """
        title = source.book_title[self.lang]
        book_description = source.book_description.get(self.lang, "N/A")
        if "Yerushalmi" in source.categories:
            book_description = book_description.replace(title.replace("Jerusalem Talmud ", ""), title)
        if source.categories[0] == "Mishnah":
            book_description = book_description.replace(title.replace("Mishnah ", ""), title)
        return book_description

    def _get_input_prompt_details(self) -> str:
        book_desc = self._get_book_description(self.source)
        context = get_context(self.source)
        unique_aspect = get_uniqueness_of_source(self.source, self.topic, self.other_sources)
        prompt = f"<topic>{self.topic.title['en']}</topic>\n" \
                 f"<author>{self.source.author_name}</author>\n" \
                 f"<publication_year>{self.source.comp_date}</publication_year>\n" \
                 f"<unique_aspect>{unique_aspect}</unique_aspect>\n" \
                 f"<context>{context}</context>"

        if True:  # category not in {"Talmud", "Midrash", "Tanakh"}:
            prompt += f"\n<book_description>{book_desc}</book_description>"
        if self.source.commentary:
            from summarize_commentary.summarize_commentary import summarize_commentary
            commentary_summary = summarize_commentary(self.source, self.topic, company='anthropic')
            prompt += f"\n<commentary>{commentary_summary}</commentary>"
        return prompt


@dataclass
class ToppromptExample:
    lang: str
    topic: str
    title: str
    why: str
    what: str
    unique_aspect: str
    context: str

    def serialize(self):
        return {
            "topic": self.topic,
            "title": self.title,
            "why": self.why,
            "what": self.what,
            "unique_aspect": self.unique_aspect,
            "context": self.context,
        }


class TopromptExampleGenerator:

    def __init__(self, lang: str):
        self.lang: str = lang

    def get(self) -> List[dict]:
        examples = self._get_training_set()
        return [example.serialize() for example in examples]

    def _get_training_set(self) -> List[ToppromptExample]:
        with open("input/topic_prompt_training_set.json", "r") as fin:
            raw_examples = json.load(fin)
            return [ToppromptExample(lang=self.lang, **raw_example) for raw_example in raw_examples]


def get_output_parser():
    return PydanticOutputParser(pydantic_object=TopromptLLMOutput)
