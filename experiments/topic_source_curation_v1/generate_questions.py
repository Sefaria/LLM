import csv
from collections import defaultdict
from basic_langchain.schema import SystemMessage, HumanMessage
from basic_langchain.chat_models import ChatAnthropic
import requests
from readability import Document
import re


def get_urls_for_slug(topic_slug: str) -> list[str]:
    topic_url_mapping = _TopicURLMapping()
    return topic_url_mapping[topic_slug]


def generate_questions_from_url_list(urls: list[str]) -> list[str]:
    questions = []
    for url in urls:
        text = _get_webpage_text(url)
        temp_questions = _generate_questions(text)
        questions += temp_questions
    return questions


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


def _generate_questions(text: str) -> list[str]:
    llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
    system = SystemMessage(content="You are a Jewish teacher looking to stimulate students to pose questions about Jewish topics. Your students don't have a strong background in Judaism but are curious to learn more. Given text about a Jewish topic, wrapped in <text>, output a list of questions that this student would ask in order to learn more about this topic. Wrap each question in a <question> tag.")
    human = HumanMessage(content=f"<text>{text}</text>")
    response = llm([system, human])
    questions = []
    for match in re.finditer(r"<question>(.*?)</question>", response.content):
        questions += [match.group(1)]
    return questions


def _get_webpage_text(url: str) -> str:
    response = requests.get(url)
    doc = Document(response.content)
    return f"{doc.title()}\n{doc.summary()}"
