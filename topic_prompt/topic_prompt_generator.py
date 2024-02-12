import re
from loguru import logger

from tqdm import tqdm
from typing import List
from sefaria_interface.topic_prompt_input import TopicPromptInput
from sefaria_interface.topic_prompt_source import TopicPromptSource
from sefaria_interface.topic import Topic
from topic_prompt.toprompt_llm_prompt import TopromptLLMPrompt, get_output_parser
from topic_prompt.toprompt import Toprompt, TopromptOptions
from topic_prompt.differentiate_writing import repeated_phrase

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage


def _get_toprompt_options(lang: str, topic: Topic, source: TopicPromptSource, other_sources: List[TopicPromptSource],
                          num_tries=1, phrase_to_avoid=None) -> TopromptOptions:
    # TODO pull out formatting from _get_input_prompt_details
    llm_prompt = TopromptLLMPrompt(lang, topic, source, other_sources).get()
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
        parsed_output = output_parser.parse(curr_response.content)
        parsed_output.title = _improve_title_and_validate_improvement(responses + [curr_response], parsed_output.title)
        topic_prompts[-1] = Toprompt(topic, source, parsed_output.why, parsed_output.what, parsed_output.title)

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


def differentiate_prompts(toprompt_options: List[TopromptOptions], tp_input: TopicPromptInput):
    """
    Going to assume we're just focusing on the first option of each toprompt_option
    :param toprompt_options:
    :param tp_input:
    :return:
    """
    differentiated = []
    seen_phrases = set()
    for i, (source, toprompt_option) in tqdm(enumerate(zip(tp_input.sources, toprompt_options)), total=len(tp_input.sources), desc='differentiate'):
        why = toprompt_option.toprompts[0].why
        other_sources = [other_source for other_source in tp_input.sources if other_source.ref != source.ref]
        other_whys = [option.toprompts[0].why for j, option in enumerate(toprompt_options) if j != i]
        phrase = repeated_phrase(why, other_whys)
        if phrase is None:
            pass
        elif phrase not in seen_phrases:
            seen_phrases.add(phrase)
        else:
            diff_prompt_option = _get_toprompt_options(tp_input.lang, tp_input.topic, source, other_sources, num_tries=1, phrase_to_avoid=phrase)
            toprompt_option = diff_prompt_option
        differentiated += [toprompt_option]
    return differentiated


def get_toprompts(tp_input: TopicPromptInput):
    toprompt_options = []
    for source in tqdm(tp_input.sources, desc=f"get toprompts for serial input"):
        other_sources = [other_source for other_source in tp_input.sources if other_source.ref != source.ref]
        toprompt_options += [_get_toprompt_options(tp_input.lang, tp_input.topic, source, other_sources, num_tries=1)]
    toprompt_options = differentiate_prompts(toprompt_options, tp_input)
    return toprompt_options


def init_logger():
    with open("out.log", "w") as fin:
        pass
    logger.remove(0)
    logger.add("out.log", level="TRACE", format="{message}")
