import csv
import re

from tqdm import tqdm
from typing import List
from sheet_interface import get_topic_and_orefs
from html_formatter import HTMLFormatter
from csv_formatter import CSVFormatter
from sefaria.model.topic import Topic
from sefaria.model.text import Ref
from toprompt_llm_prompt import TopromptLLMPrompt, get_output_parser
from toprompt import Toprompt, TopromptOptions
from differentiate_writing import repeated_phrase, differentiate_writing

import langchain
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def _get_toprompt_options(lang: str, topic: Topic, oref: Ref, other_orefs: List[Ref], num_tries=1, phrase_to_avoid=None) -> TopromptOptions:
    # TODO pull out formatting from _get_input_prompt_details
    full_language = "English" if lang == "en" else "Hebrew"
    llm_prompt = TopromptLLMPrompt(lang, topic, oref, other_orefs).get()
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
        parsed_output.title = _improve_title_and_validate_improvement(responses, parsed_output.title)

        topic_prompts += [Toprompt(topic, oref, parsed_output.why, parsed_output.what, parsed_output.title)]

    # phrase to avoid
    if phrase_to_avoid:
        avoid_prompt = PromptTemplate.from_template("Rewrite the description and title but avoid using the phrase "
                                                    "\"{phrase}\". Refer back to the "
                                                    f"examples provided to stick to the same writing style.\n"
                                                    "{format_instructions}",
                                                    partial_variables={"phrase": phrase_to_avoid, "format_instructions": get_output_parser().get_format_instructions()})
        curr_response = llm([human_message] + responses + [HumanMessage(content=avoid_prompt.format())])
        output_parser = get_output_parser()
        parsed_output = output_parser.parse(curr_response.content)
        parsed_output.title = _improve_title_and_validate_improvement(responses + [curr_response], parsed_output.title)
        topic_prompts[-1] = Toprompt(topic, oref, parsed_output.why, parsed_output.what, parsed_output.title)

    return TopromptOptions(topic_prompts)


def _improve_title_and_validate_improvement(gpt_responses, title):
    counter = 0
    while ":" in title and counter < 5:
        title = _improve_title(gpt_responses, title)
        if not title: return
        counter += 1
    return title


def _improve_title(curr_responses, curr_title):
    better_title_prompt = PromptTemplate.from_template(f"Current title is: {curr_title}. "
                                                       f"Rewrite the title, rephrasing to avoid using a colon."
                                                       f" Wrap the title in <title> tags. It should at most"
                                                       f" five words and grab the reader's attention.")
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    title_response = llm(curr_responses + [HumanMessage(content=better_title_prompt.format())])
    title_match = re.search(r'<title>(.+?)</title>', title_response.content)
    if title_match is None:
        return
    new_title = title_match.group(1)
    new_title = re.sub(r'^"', '', new_title)
    new_title = re.sub(r'"$', '', new_title)
    return new_title


def _get_topprompts_for_sheet_id(lang, sheet_id: int) -> List[TopromptOptions]:
    topic, orefs = get_topic_and_orefs(sheet_id)
    toprompt_options = []
    for oref in tqdm(orefs, desc="get toprompts for sheet"):
        other_orefs = [r for r in orefs if r.normal() != oref.normal()]
        toprompt_options += [_get_toprompt_options(lang, topic, oref, other_orefs, num_tries=1)]
    toprompt_options = differentiate_prompts(toprompt_options, orefs, lang, topic)
    return toprompt_options


def differentiate_prompts(toprompt_options: List[TopromptOptions], orefs, lang, topic):
    """
    Going to assume we're just focusing on the first option of each toprompt_option
    :param toprompt_options:
    :return:
    """
    differentiated = []
    seen_phrases = set()
    for i, (oref, toprompt_option) in tqdm(enumerate(zip(orefs, toprompt_options)), total=len(orefs)):
        why = toprompt_option.toprompts[0].why
        other_whys = [option.toprompts[0].why for j, option in enumerate(toprompt_options) if j != i]
        phrase = repeated_phrase(why, other_whys).strip()
        if phrase not in seen_phrases:
            seen_phrases.add(phrase)
        elif phrase is None:
            pass
        else:
            new_why = differentiate_writing(why, phrase)
            toprompt_option.toprompts[0].why = new_why
        differentiated += [toprompt_option]
    return differentiated


def output_toprompts_for_sheet_id_list(lang: str, sheet_ids: List[int]) -> None:
    toprompt_options = []
    for sheet_id in sheet_ids:
        toprompt_options += _get_topprompts_for_sheet_id(lang, sheet_id)
    formatter = HTMLFormatter(toprompt_options)
    formatter.save("output/sheet_topic_prompts.html")
    csv_formatter = CSVFormatter(toprompt_options)
    csv_formatter.save("output/sheet_topic_prompts.csv")


def _get_validation_set():
    validation_set = []
    with open("input/topic_prompt_validation_set.csv", "r") as fin:
        cin = csv.DictReader(fin)
        for row in cin:
            validation_set += [(Topic.init(row['Slug']), Ref(row['Reference']), row['Title'], row['Prompt '])]
    return validation_set


# TODO not maintaining for now since some assumptions have been made that break validation set
def output_toprompts_for_validation_set(lang):
    validation_set = _get_validation_set()
    toprompt_options = []
    gold_standard_prompts = []
    for topic, oref, title, prompt in tqdm(validation_set):
        toprompt_options += [_get_toprompt_options(lang, topic, oref)]
        gold_standard_prompts += [Toprompt(topic, oref, prompt, title)]
    html_formatter = HTMLFormatter(toprompt_options, gold_standard_prompts)
    html_formatter.save("output/validation_topic_prompts.html")
    csv_formatter = CSVFormatter(toprompt_options, gold_standard_prompts)
    csv_formatter.save("output/validation_topic_prompts.csv")


def _get_top_n_orefs_for_topic(slug, top_n=10) -> List[Ref]:
    from sefaria.helper.topic import get_topic

    out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
    return [Ref(d['ref']) for d in out['refs']['about']['refs'][:top_n]]


def output_toprompts_for_topic_page(lang, slug, top_n=10):
    topic = Topic.init(slug)
    orefs = _get_top_n_orefs_for_topic(slug, top_n)
    toprompt_options = []
    for oref in tqdm(orefs, desc="get toprompts for topic page"):
        other_oref = [r for r in orefs if r.normal() != oref.normal()]
        toprompt_options += [_get_toprompt_options(lang, topic, oref, other_oref, num_tries=3)]
    formatter = HTMLFormatter(toprompt_options)
    formatter.save("output/topic_page_topic_prompts.html")
    csv_formatter = CSVFormatter(toprompt_options)
    csv_formatter.save("output/topic_page_topic_prompts.csv")


if __name__ == '__main__':
    sheet_ids = [515293]
    lang = "en"
    output_toprompts_for_sheet_id_list(lang, sheet_ids)
    # output_toprompts_for_validation_set(lang)
    # output_toprompts_for_topic_page(lang, 'peace')
