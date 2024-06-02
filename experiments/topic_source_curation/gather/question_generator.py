"""
Defines classes for generating questions about a topic
questions are meant to be used as queries to a vector store
which will be used to gather many sources about a topic to curate
"""
from tqdm import tqdm
import django
django.setup()
from functools import reduce, partial
from abc import ABC, abstractmethod
from collections import defaultdict
from basic_langchain.schema import SystemMessage, HumanMessage
from basic_langchain.chat_models import ChatOpenAI, ChatAnthropic
from util.topic import get_urls_for_topic_from_topic_object
from util.webpage import get_webpage_text
from util.general import run_parallel, get_by_xml_tag
import re
import csv
from sefaria_llm_interface.common.topic import Topic
from experiments.topic_source_curation.common import get_topic_str_for_prompts


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
    def generate(self, topic: Topic, verbose=True) -> list[str]:
        pass



class MultiSourceQuestionGenerator(AbstractQuestionGenerator):

    def __init__(self, question_generators: list[AbstractQuestionGenerator]):
        self._question_generators = question_generators

    def generate(self, topic: Topic, verbose=True) -> list[str]:
        questions = []
        for generator in self._question_generators:
            questions.extend(generator.generate(topic, verbose=verbose))
        return questions

    @staticmethod
    def _translate_questions_parallel(questions: list[str], lang: str, verbose=True) -> list[str]:
        return run_parallel(questions, partial(MultiSourceQuestionGenerator._translate_question, lang=lang), max_workers=2, desc="translation questions", disable=not verbose)


    @staticmethod
    def _translate_question(question: str, lang: str) -> str:
        llm = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
        system = SystemMessage(content=f"Given text about a Jewish topic output the translation of the text into {lang}. Question is wrapped in <text> tags. Output translation in <translation> tags.")
        human = HumanMessage(content=f"<text>{question}</text>")
        response = llm([system, human])
        return get_by_xml_tag(response.content, "translation")


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


    def generate(self, topic: Topic, verbose=True) -> list[str]:
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


    def generate(self, topic: Topic, verbose=True) -> list[str]:
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

    def _expand_question(self, topic_str: str, seed_question: str):
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        system = SystemMessage(content="You are a Jewish teacher well versed in all Jewish texts and customs. Given a general question about a topic in Judaism,  produce a multiple answers to the question that are specific. Topic is wrapped in <topic> tags and text is wrapped in <text> tags. Wrap each answer in an <answer> tag")
        human = HumanMessage(content=f"<topic>{topic_str}</topic>\n<text>{seed_question}</text>")
        response = llm([system, human])
        questions = []
        for match in re.finditer(r"<answer>(.*?)</answer>", response.content.replace('\n', ' ')):
            questions += [match.group(1)]
        return questions

    @staticmethod
    def __get_type_for_topic(topic: Topic) -> str:
        return "generic"

    def generate(self, topic: Topic, verbose=True) -> list[str]:
        if verbose:
            print('---LLM QUESTION EXPANDER---')
        topic_str = get_topic_str_for_prompts(topic)
        original_questions = [q.format(topic.title['en']) for q in self.templated_questions_by_type[self.__get_type_for_topic(topic)]]
        expanded_questions = run_parallel(original_questions, partial(self._expand_question, topic_str), max_workers=20, desc="llm question expander", disable=not verbose)
        if verbose:
            for original_q, expanded_qs in zip(original_questions, expanded_questions):
                print('----')
                print('\toriginal question:', original_q)
                for expanded_q in expanded_qs:
                    print('\t-', expanded_q)
        return reduce(lambda a, b: a + b, expanded_questions)


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

    def _get_urls_for_topic(self, topic: Topic) -> list[str]:
        return self._get_urls_for_topic_from_mapping(topic) + get_urls_for_topic_from_topic_object(topic)

    def generate(self, topic: Topic, verbose=True) -> list[str]:
        urls = self._get_urls_for_topic(topic)
        questions = []
        for url in tqdm(urls, desc="Generating questions from urls", disable=not verbose):
            questions += self._generate_for_url(topic, url)
        if verbose:
            print('---WEB-DERIVED QUESTIONS---')
            for question in questions:
                print('\t-', question)

        return questions

    @staticmethod
    def _generate_for_url(topic: Topic, url: str) -> list[str]:
        webpage_text = get_webpage_text(url)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        system = SystemMessage(content="You are a Jewish teacher well versed in all Jewish texts and customs. Given text about a Jewish topic, summary the text and output the most important bullet points the text discusses. Topic is wrapped in <topic> tags and text is wrapped in <text> tags. Wrap each bullet point in a <bullet_point> tag.")
        human = HumanMessage(content=f"<topic>{get_topic_str_for_prompts(topic, verbose=False)}</topic>\n<text>{webpage_text}</text>")
        response = llm([system, human])
        questions = []
        for match in re.finditer(r"<bullet_point>(.*?)</bullet_point>", response.content):
            questions += [match.group(1)]
        return questions



