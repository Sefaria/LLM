import django
django.setup()
from sefaria.model.text import Ref, library

import langchain
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

from util.general import get_raw_ref_text
import typer
from tqdm import tqdm


def get_topics_for_title(title: str, lang: str):
    index = library.get_index(title)
    for segment_oref in tqdm(index.all_segment_refs()[1:]):
        print('-----')
        print(segment_oref.normal())
        print("AFTER", get_topics_for_tref(segment_oref, lang))


def get_topics_for_tref(oref: Ref, lang: str):
    text = get_raw_ref_text(oref, lang)
    return get_raw_topics(text)


def get_raw_topics(text):
    system_message = SystemMessage(content=
                                   "You are an intelligent Jewish scholar who is knowledgeable in all aspects of the Torah and Jewish texts.\n"
                                    "# Task\n"
                                    "Output list of high-level topics discussed by the input\n"
                                   "Topics should be important enough that they would warrant an entry in the index in the back of a book\n"
                                   "Each topic should be on its own line with no extra text.\n"
                                   "Topics should be short. They should be written as if they are titles of encyclopedia entries. Therefore, they should be understandable when read independent of the source text.\n"
                                   "Citations are not topics. E.g. Genesis 1:4 is not a topic\n"
                                   "Topics should be written assuming a Torah context. Phrases like \"Torah perspective\", \"in Judaism\", \"in the Torah\" and \"Biblical Narrative\" should not appear in a topic.\n"
                                   "ONLY output topics with no explanatory text."
                                   "# Example topics\n"
                                   "Teruma\n"
                                   "Parashat Noach\n"
                                   "Abraham\n"
                                   "Shabbat\n"
                                   "# Examples that are NOT topics\n"
                                   "Dispute between Rabbi Akiva and Rabbi Yehoshua\n"
                                   "Opinions on how to shake lulav\n"
                                   )
    user_prompt = PromptTemplate.from_template("# Input\n{text}")
    human_message = HumanMessage(content=user_prompt.format(text=text))

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    # llm = ChatAnthropic(model="claude-2", temperature=0)

    response = llm([system_message, human_message])
    print("BEFORE\n", response.content)
    print('---')
    human_refine = HumanMessage(content="Of the topics above, list the most fundamental topics for understanding the source text. Exclude topics that are very specific.")
    response2 = llm([system_message, human_message, response, human_refine])
    return response2.content


if __name__ == '__main__':
    typer.run(get_topics_for_title)


