"""
Defines classes for generating questions about a topic
questions are meant to be used as queries to a vector store
which will be used to gather many sources about a topic to curate
"""
from tqdm import tqdm
import django
django.setup()
from sefaria.model.topic import Topic as SefariaTopic
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
    # return MultiSourceQuestionGenerator([
    #     TemplatedQuestionGenerator("gather/input/templated_questions_by_type.csv"),
    #     WebPageQuestionGenerator(_TopicURLMapping())
    # ])
    # return MultiSourceQuestionGenerator([
    #     TemplatedQuestionCategoryAwareGenerator("gather/input/templated_questions_and_categories_by_type.csv"),
    #     WebPageQuestionGenerator(_TopicURLMapping())
    # ])
    return MultiSourceQuestionGenerator([
        LlmExpandedTemplatedQuestionGenerator("gather/input/templated_questions_to_expand_by_type.csv"),
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
    """
    learning team types:
    holiday
    figure
    story
    liturgy
    mitzvah
    ritual object
    customs
    rabbinic principles
    idea/belief
    """

    ontology_type_to_learning_team_type_map = {

    }

    def __init__(self, templated_questions_by_type_filename: str):
        self.templated_questions_by_type = self.__get_templated_questions_by_type(templated_questions_by_type_filename)

    @staticmethod
    def __get_templated_questions_by_type(templated_questions_by_type_filename: str) -> dict[str, list[str]]:
        questions_by_type = defaultdict(list)
        with open(templated_questions_by_type_filename, "r") as fin:
            cin = csv.DictReader(fin)
            for row in cin:
                questions_by_type[row['Type']] += [row['Question']]
        return questions_by_type

    @staticmethod
    def __get_type_for_topic(topic: Topic) -> str:
        naive_map = {"stars": "neutral-object"}
        return(naive_map[topic.slug])


    def generate(self, topic: Topic) -> list[str]:
        return [q.format(topic.title['en']) for q in self.templated_questions_by_type[self.__get_type_for_topic(topic)]]
class TemplatedQuestionCategoryAwareGenerator(AbstractQuestionGenerator):
    """
    learning team types:
    holiday
    figure
    story
    liturgy
    mitzvah
    ritual object
    customs
    rabbinic principles
    idea/belief
    """

    ontology_type_to_learning_team_type_map = {

    }

    def __init__(self, templated_questions_by_type_filename: str):
        self.templated_questions_and_categories_by_type = self.__get_templated_questions_by_type(templated_questions_by_type_filename)

    @staticmethod
    def __get_templated_questions_by_type(templated_questions_by_type_filename: str) -> dict[str, list[(str, list[str])]]:
        questions_and_categories_by_type = defaultdict(list)
        with open(templated_questions_by_type_filename, "r") as fin:
            cin = csv.DictReader(fin)
            for row in cin:
                questions_and_categories_by_type[row['Type']].append((row['Question'],
                                                                      None if not row["Categories"] else
                                                                      row["Categories"].split(",") if "," in row["Categories"]
                                                                      else [row["Categories"]]))
        return questions_and_categories_by_type

    @staticmethod
    def __get_type_for_topic(topic: Topic) -> str:
        naive_map = {"stars": "neutral-object"}
        return(naive_map[topic.slug])


    def generate(self, topic: Topic) -> list[str]:
        return [(q[0].format(topic.title['en']), q[1]) for q in self.templated_questions_and_categories_by_type[self.__get_type_for_topic(topic)]]

class LlmExpandedTemplatedQuestionGenerator(AbstractQuestionGenerator):
    """
    learning team types:
    holiday
    figure
    story
    liturgy
    mitzvah
    ritual object
    customs
    rabbinic principles
    idea/belief
    """

    ontology_type_to_learning_team_type_map = {

    }

    def __init__(self, templated_questions_by_type_filename: str):
        self.templated_questions_by_type = self.__get_templated_questions_by_type(templated_questions_by_type_filename)

    @staticmethod
    def __get_templated_questions_by_type(templated_questions_by_type_filename: str) -> dict[str, list[str]]:
        questions_by_type = defaultdict(list)
        with open(templated_questions_by_type_filename, "r") as fin:
            cin = csv.DictReader(fin)
            for row in cin:
                questions_by_type[row['Type']] += [row['Question']]
        return questions_by_type

    def _expand_question(self, seed_question: str):
        llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
        system = SystemMessage(content="You are a Jewish teacher well versed in all Jewish texts and customs. Given a general question about a topic in Judasim wrapped in <text> tag, produce a multiple more specific questions exploring specific aspects of the general topic. wrap each question in <question> tag")
        human = HumanMessage(content=f"<text>{seed_question}</text>")
        response = llm([system, human])
        questions = []
        for match in re.finditer(r"<question>(.*?)</question>", response.content):
            questions += [match.group(1)]
        return questions

    @staticmethod
    def __get_type_for_topic(topic: Topic) -> str:
        naive_map = {"stars": "neutral-object",
                     "jesse": "biblical-figure",
                     "friendship": "idea/belief",
                     "bread": "idea/belief",
                     "ulla": "idea/belief"}
        return(naive_map[topic.slug])


    def generate(self, topic: Topic, verbose=True) -> list[str]:
        if verbose:
            print('---LLM QUESTION EXPANDER---')
        questions = []
        for q in self.templated_questions_by_type[self.__get_type_for_topic(topic)]:
            q = q.replace("{}", topic.title['en'])
            expanded = self._expand_question(q)
            if verbose:
                print('----')
                print('\toriginal question:', q)
                for expanded_q in expanded:
                    print('\t-', expanded_q)
            questions.extend(expanded)
        return questions


class _TopicURLMapping:
    slug_url_mapping = "gather/input/Topic Webpage mapping for question generation - Sheet1.csv"

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

    def _get_urls_for_topic_from_mapping(self, topic: Topic) -> list[str]:
        return self._topic_url_mapping[topic.slug]

    def _get_urls_for_topic_from_topic_object(self, topic: Topic) -> list[str]:
        sefaria_topic = SefariaTopic.init(topic.slug)
        assert isinstance(sefaria_topic, SefariaTopic)
        url_fields = [["enWikiLink", "heWikiLink"], ["enNliLink", "heNliLink"], ["jeLink"]]
        urls = []
        for fields_by_priority in url_fields:
            for field in fields_by_priority:
                value = sefaria_topic.get_property(field)
                if value is not None:
                    urls.append(value)
                    break
        return urls

    def _get_urls_for_topic(self, topic: Topic) -> list[str]:
        return self._get_urls_for_topic_from_mapping(topic) + self._get_urls_for_topic_from_topic_object(topic)


    def generate(self, topic: Topic, verbose=True) -> list[str]:
        urls = self._get_urls_for_topic(topic)
        questions = []
        for url in tqdm(urls, desc="Generating questions from urls", disable=not verbose):
            questions += self._generate_for_url(url)
        if verbose:
            print('---WEB-DERIVED QUESTIONS---')
            for question in questions:
                print('\t-', question)

        return questions

    def _generate_for_url(self, url: str) -> list[str]:
        webpage_text = self._get_webpage_text(url)
        llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
        system = SystemMessage(content="You are a Jewish teacher well versed in all Jewish texts and customs. Given text about a Jewish topic, wrapped in <text>, summary the text and output the most important bullet points the text discusses. Wrap each bullet point in a <bullet_point> tag.")
        human = HumanMessage(content=f"<text>{webpage_text}</text>")
        response = llm([system, human])
        questions = []
        for match in re.finditer(r"<bullet_point>(.*?)</bullet_point>", response.content):
            questions += [match.group(1)]
        return questions

    @staticmethod
    def _get_webpage_text(url: str) -> str:
        response = requests.get(url)
        doc = Document(response.content)
        return f"{doc.title()}\n{doc.summary()}"

