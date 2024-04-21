import csv
from tqdm import tqdm
from basic_langchain.schema import SystemMessage, HumanMessage
from basic_langchain.chat_models import ChatAnthropic
import requests
from readability import Document
import re


INPUT = "input/Topic Webpage mapping for question generation - Sheet1.csv"

def get_mapping():
    with open(INPUT, "r") as fin:
        cin = csv.DictReader(fin)
        return {row['slug']: row['url'] for row in cin}


def get_webpage_text(url: str) -> str:
    response = requests.get(url)
    doc = Document(response.content)
    return f"{doc.title()}\n{doc.summary()}"


def generate_questions(text: str) -> list[str]:
    llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
    system = SystemMessage(content="You are a Jewish teacher looking to stimulate students to pose questions about Jewish topics. Your students don't have a strong background in Judaism but are curious to learn more. Given text about a Jewish topic, wrapped in <text>, output a list of questions that this student would ask in order to learn more about this topic. Wrap each question in a <question> tag.")
    human = HumanMessage(content=f"<text>{text}</text>")
    response = llm([system, human])
    questions = []
    for match in re.finditer(r"<question>(.*?)</question>", response.content):
        questions += [match.group(1)]
    return questions



if __name__ == '__main__':
    mapping = get_mapping()
    webpage_text = get_webpage_text(mapping['alexandria'])
    print(generate_questions(webpage_text))

