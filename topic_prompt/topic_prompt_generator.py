import csv
import re

from tqdm import tqdm
from typing import List
from sheet_interface import get_topic_and_orefs
from html_formatter import HTMLFormatter
from sefaria.model.topic import Topic
from sefaria.model.text import Ref
from toprompt_llm_prompt import TopromptLLMPrompt, get_output_parser
from toprompt import Toprompt, TopromptOptions

import langchain
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain import PromptTemplate
from langchain.schema import HumanMessage
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def _get_toprompt_options(lang: str, topic: Topic, oref: Ref, num_tries=1) -> TopromptOptions:
    # TODO pull out formatting from _get_input_prompt_details
    full_language = "English" if lang == "en" else "Hebrew"
    llm_prompt = TopromptLLMPrompt(lang, topic, oref).get()
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    human_message = HumanMessage(content=llm_prompt.format())
    responses = []
    topic_prompts = []
    secondary_prompt = PromptTemplate.from_template(f"Generate another set of description and title. Refer back to the "
                                                     f"examples provided to stick to the same writing style.\n"
                                                     "{format_instructions}",
                                                     partial_variables={"format_instructions": get_output_parser().get_format_instructions()})
    for i in range(num_tries):
        curr_response = llm([human_message] + responses)
        responses += [curr_response]
        if i < num_tries-1:
            responses += [HumanMessage(content=secondary_prompt.format())]

        output_parser = get_output_parser()
        parsed_output = output_parser.parse(curr_response.content)
        toprompt_text = parsed_output.why + " " + parsed_output.what

        # improve title
        if ":" in parsed_output.title:
            print("OLD")
            print(parsed_output.title)
            parsed_output.title = _improve_title(responses)
            print("NEW")
            print(parsed_output.title)
            print("---")

        topic_prompts += [Toprompt(topic, oref, toprompt_text, parsed_output.title)]

    return TopromptOptions(topic_prompts)


def _improve_title(curr_responses):
    better_title_prompt = PromptTemplate.from_template(f"Rewrite the title, rephrasing to avoid using a colon."
                                                       f"Wrap the title in <title> tags. It should at most"
                                                       f"five words and grab the reader's attention.")
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    title_response = llm(curr_responses + [HumanMessage(content=better_title_prompt.format())])
    return re.search(r'<title>(.+?)</title>', title_response.content).group(1)


def _get_topprompts_for_sheet_id(lang, sheet_id: int) -> List[TopromptOptions]:
    topic, orefs = get_topic_and_orefs(sheet_id)
    toprompt_options = []
    for oref in tqdm(orefs, desc="get toprompts for sheet"):
        toprompt_options += [_get_toprompt_options(lang, topic, oref)]
        break
    return toprompt_options


def output_toprompts_for_sheet_id_list(lang: str, sheet_ids: List[int]) -> None:
    toprompt_options = []
    for sheet_id in sheet_ids:
        toprompt_options += _get_topprompts_for_sheet_id(lang, sheet_id)
    formatter = HTMLFormatter(toprompt_options)
    formatter.save("output/topic_prompts.html")


def _get_validation_set():
    validation_set = []
    with open("input/topic_prompt_validation_set.csv", "r") as fin:
        cin = csv.DictReader(fin)
        for row in cin:
            validation_set += [(Topic.init(row['Slug']), Ref(row['Reference']), row['Title'], row['Prompt '])]
    return validation_set


def output_toprompts_for_validation_set(lang):
    validation_set = _get_validation_set()
    toprompt_options = []
    gold_standard_prompts = []
    for topic, oref, title, prompt in tqdm(validation_set):
        toprompt_options += [_get_toprompt_options(lang, topic, oref)]
        gold_standard_prompts += [Toprompt(topic, oref, prompt, title)]
    formatter = HTMLFormatter(toprompt_options, gold_standard_prompts)
    formatter.save("output/topic_prompts.html")


if __name__ == '__main__':
    # sheet_ids = [502699]  # [502699, 502661, 499080, 498250, 500844]
    # sheet_ids = [498250]
    lang = "en"
    # output_toprompts_for_sheet_id_list(lang, sheet_ids)
    output_toprompts_for_validation_set(lang)
