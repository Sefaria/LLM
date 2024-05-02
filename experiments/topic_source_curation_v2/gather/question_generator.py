"""
Defines classes for generating questions about a topic
questions are meant to be used as queries to a vector store
which will be used to gather many sources about a topic to curate
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from basic_langchain.schema import SystemMessage, HumanMessage
from basic_langchain.chat_models import ChatAnthropic
import requests
from readability import Document
import re
import csv
from sefaria_llm_interface.common.topic import Topic


def create_multi_source_question_generator() -> 'AbstractQuestionGenerator':
    return MultiSourceQuestionGenerator([
        TemplatedQuestionGenerator("input/templated_questions_by_type.csv"),
        WebPageQuestionGenerator(_TopicURLMapping())
    ])


class AbstractQuestionGenerator(ABC):
    @abstractmethod
    def generate(self, topic: Topic) -> list[str]:
        pass



class MultiSourceQuestionGenerator(AbstractQuestionGenerator):

    def __init__(self, question_generators: list[AbstractQuestionGenerator]):
        self._question_generators = question_generators

    def generate(self, topic: Topic) -> list[str]:
        questions = []
        for generator in self._question_generators:
            questions.extend(generator.generate(topic))
        return questions


class TemplatedQuestionGenerator(AbstractQuestionGenerator):

    def __init__(self, templated_questions_by_type_filename: str):
        self.templated_questions_by_type = self.__get_templated_questions_by_type(templated_questions_by_type_filename)

    @staticmethod
    def __get_templated_questions_by_type(templated_questions_by_type_filename: str) -> dict[str, list[str]]:
        questions_by_type = defaultdict(list)
        with open(templated_questions_by_type_filename, "r") as fin:
            cin = csv.DictReader(templated_questions_by_type_filename)
            for row in cin:
                questions_by_type[row['Type']] += [row['Question']]
        return questions_by_type

    @staticmethod
    def __get_type_for_topic(topic: Topic) -> str:
        pass

    def generate(self, topic: Topic) -> list[str]:
        return self.templated_questions_by_type[self.__get_type_for_topic(topic)]


class _TopicURLMapping:
    slug_url_mapping = "input/Topic Webpage mapping for question generation - Sheet1.csv"

    def __init__(self):
        self._raw_mapping = self._get_raw_mapping()

    def __getitem__(self, item) -> list[str]:
        return self._raw_mapping[item]

    def _get_raw_mapping(self):
        mapping = defaultdict(list)
        with open(self.slug_url_mapping, "r") as fin:
            cin = csv.DictReader(fin)
            for row in cin:
                mapping[row['slug']] += [row['url']]
        return mapping


class WebPageQuestionGenerator(AbstractQuestionGenerator):

    def __init__(self, topic_url_mapping: _TopicURLMapping):
        self._topic_url_mapping = topic_url_mapping

    def _get_urls_for_topic(self, topic: Topic) -> list[str]:
        return self._topic_url_mapping[topic.slug]

    def generate(self, topic: Topic) -> list[str]:
        urls = self._get_urls_for_topic(topic)
        questions = []
        for url in urls:
            questions += self._generate_for_url(url)
        return questions

    def _generate_for_url(self, url: str) -> list[str]:
        webpage_text = self._get_webpage_text(url)
        llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
        system = SystemMessage(content="You are a Jewish teacher looking to stimulate students to pose questions about Jewish topics. Your students don't have a strong background in Judaism but are curious to learn more. Given text about a Jewish topic, wrapped in <text>, output a list of questions that this student would ask in order to learn more about this topic. Wrap each question in a <question> tag.")
        human = HumanMessage(content=f"<text>{webpage_text}</text>")
        response = llm([system, human])
        questions = []
        for match in re.finditer(r"<question>(.*?)</question>", response.content):
            questions += [match.group(1)]
        return questions

    @staticmethod
    def _get_webpage_text(url: str) -> str:
        response = requests.get(url)
        doc = Document(response.content)
        return f"{doc.title()}\n{doc.summary()}"

