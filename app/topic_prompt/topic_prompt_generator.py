import re
from loguru import logger

from tqdm import tqdm
from typing import List
from sefaria_llm_interface.topic_prompt import TopicPromptInput, TopicPromptSource
from sefaria_llm_interface import Topic
from topic_prompt.toprompt_llm_prompt import TopromptLLMPrompt, get_output_parser
from topic_prompt.toprompt import Toprompt, TopromptOptions
from topic_prompt.differentiate_writing import repeated_phrase
from util.general import escape_json_inner_quotes

from langchain.prompts import PromptTemplate
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import HumanMessage, AbstractMessage

def _get_toprompt_options(lang: str, topic: Topic, source: TopicPromptSource, other_sources: List[TopicPromptSource],
                          num_tries=1, phrase_to_avoid=None) -> (TopromptOptions, list[AbstractMessage]):
    # TODO pull out formatting from _get_input_prompt_details
    llm_prompt = TopromptLLMPrompt(lang, topic, source, other_sources).get()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
        parsed_output = output_parser.parse(escape_json_inner_quotes(curr_response.content))
        parsed_output.title = _remove_colon_from_title_with_validation(responses, parsed_output.title)

        topic_prompts += [Toprompt(topic, source, parsed_output.why, parsed_output.what, parsed_output.title)]

    # phrase to avoid
    if phrase_to_avoid:
        avoid_prompt = PromptTemplate.from_template("Rewrite the description and title but avoid using the phrase "
                                                    "\"{phrase}\". Refer back to the "
                                                    f"examples provided to stick to the same writing style.\n"
                                                    "{format_instructions}",
                                                    partial_variables={"phrase": phrase_to_avoid, "format_instructions": get_output_parser().get_format_instructions()})
        curr_response = llm([human_message] + responses + [HumanMessage(content=avoid_prompt.format())])
        output_parser = get_output_parser()
        parsed_output = output_parser.parse(escape_json_inner_quotes(curr_response.content))
        parsed_output.title = _remove_colon_from_title_with_validation(responses + [curr_response], parsed_output.title)
        topic_prompts[-1] = Toprompt(topic, source, parsed_output.why, parsed_output.what, parsed_output.title)

    return TopromptOptions(topic_prompts), responses


def _remove_colon_from_title_with_validation(gpt_responses, title):
    counter = 0
    remove_colon_prompt = f"Rewrite the title, rephrasing to avoid using a colon."
    while ":" in title and counter < 5:
        title = _improve_title(gpt_responses, title, remove_colon_prompt)
        if not title: return
        counter += 1
    return title


def _improve_title(curr_responses, curr_title, rewrite_prompt):
    better_title_prompt = PromptTemplate.from_template(f"Current title is: {curr_title}. "
                                                       f"{rewrite_prompt}"
                                                       f" Wrap the title in <title> tags. It should at most"
                                                       f" five words and grab the reader's attention.")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    title_response = llm(curr_responses + [HumanMessage(content=better_title_prompt.format())])
    title_match = re.search(r'<title>(.+?)</title>', title_response.content)
    if title_match is None:
        return
    new_title = title_match.group(1)
    new_title = re.sub(r'^"', '', new_title)
    new_title = re.sub(r'"$', '', new_title)
    return new_title


def _differentiate_prompts(toprompt_options: List[TopromptOptions], tp_input: TopicPromptInput):
    """
    Going to assume we're just focusing on the first option of each toprompt_option
    :param toprompt_options:
    :param tp_input:
    :return:
    """
    differentiated = []
    seen_phrases = set()
    for i, (source, toprompt_option) in tqdm(enumerate(zip(tp_input.sources, toprompt_options)), total=len(tp_input.sources), desc='differentiate'):
        other_toprompt_options = [option for j, option in enumerate(toprompt_options) if j != i]
        phrase = _get_repeated_phrase_from_key(toprompt_option, other_toprompt_options, 'why')
        title_phrase = _get_repeated_phrase_from_key(toprompt_option, other_toprompt_options, 'title')
        if title_phrase and not phrase:
            phrase = title_phrase
        if phrase is None:
            pass
        elif phrase not in seen_phrases:
            seen_phrases.add(phrase)
        else:
            other_sources = [other_source for other_source in tp_input.sources if other_source.ref != source.ref]
            diff_prompt_option, _ = _get_toprompt_options(tp_input.lang, tp_input.topic, source, other_sources, num_tries=1, phrase_to_avoid=phrase)
            toprompt_option = diff_prompt_option
        differentiated += [toprompt_option]
    return differentiated


def _differentiate_titles(toprompt_options: List[TopromptOptions], llm_responses: list[list[AbstractMessage]], topic: Topic):
    """
    Unlike prompts, when titles repeat we can usually rewrite the title without rewriting the whole prompt
    :param toprompt_options:
    :param llm_responses:
    :return:
    """
    differentiated = []
    seen_phrases = set()
    for i, (toprompt_option, temp_llm_responses) in tqdm(enumerate(zip(toprompt_options, llm_responses)), total=len(llm_responses), desc='differentiate titles'):
        other_toprompt_options = [option for j, option in enumerate(toprompt_options) if j != i]
        phrase = _get_repeated_phrase_from_key(toprompt_option, other_toprompt_options, 'title')
        if phrase is None:
            pass
        elif phrase not in seen_phrases:
            seen_phrases.add(phrase)
        else:
            diff_title_prompt = f"This title will appear on a page all about {topic.title['en']}. Rewrite the title given users already know the context and don't want to see '{topic.title['en']}' all over the page repeated. DO NOT use any punctuation marks in the new title."
            new_title = _improve_title(temp_llm_responses, toprompt_option.toprompts[0].title, diff_title_prompt)
            print("---REWROTE TITLE---")
            print("\tREPEATED:", phrase)
            print('\tOLD:', toprompt_option.toprompts[0].title)
            print('\tNEW:', new_title)
            toprompt_option.toprompts[0].title = new_title
        differentiated += [toprompt_option]
    return differentiated


def _get_repeated_phrase_from_key(toprompt_option: TopromptOptions, other_toprompt_options: list[TopromptOptions], key: str):
    value = getattr(toprompt_option.toprompts[0], key)
    other_values = [getattr(to.toprompts[0], key) for to in other_toprompt_options]
    phrase = repeated_phrase(value, other_values)
    return phrase


def get_toprompts(tp_input: TopicPromptInput):
    toprompt_options = []
    llm_responses = []
    for source in tqdm(tp_input.sources, desc=f"get toprompts for serial input"):
        other_sources = [other_source for other_source in tp_input.sources if other_source.ref != source.ref]
        toprompt_option, temp_llm_responses = _get_toprompt_options(tp_input.lang, tp_input.topic, source, other_sources, num_tries=1)
        toprompt_options += [toprompt_option]
        llm_responses += [temp_llm_responses]
    toprompt_options = _differentiate_prompts(toprompt_options, tp_input)
    toprompt_options = _differentiate_titles(toprompt_options, llm_responses, tp_input.topic)
    return toprompt_options


def init_logger():
    with open("out.log", "w") as fin:
        pass
    logger.remove(0)
    logger.add("out.log", level="TRACE", format="{message}")
