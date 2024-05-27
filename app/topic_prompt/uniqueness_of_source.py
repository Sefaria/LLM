"""
Given all the sources curated for a topic, determine what is unique about this source
"""
import json
import re
from functools import reduce
from typing import List
from util.general import get_source_text_with_fallback, get_by_xml_tag
from util.topic import get_or_generate_topic_description
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface import Topic

from langchain.prompts import PromptTemplate
from basic_langchain.schema import HumanMessage, SystemMessage
from basic_langchain.chat_models import ChatOpenAI


def _get_prompt_inputs(source, other_sources: List[TopicPromptSource], topic: Topic):
    topic_title = topic.title['en']
    topic_description = get_or_generate_topic_description(topic, verbose=False)
    comparison_sources_list = []
    max_len = 7000
    for other_source in other_sources:
        other_text = get_source_text_with_fallback(other_source, "en", auto_translate=True)
        curr_len = reduce(lambda a, b: a + len(b), comparison_sources_list, 0)
        if curr_len + len(other_text) < max_len:
            comparison_sources_list += [other_text]
    return {
        "topic_title": topic_title,
        "topic_description": topic_description,
        "input_source": get_source_text_with_fallback(source, "en", auto_translate=True),
        "comparison_sources": json.dumps(comparison_sources_list),
        "hint": source.context_hint,
    }


def get_uniqueness_of_source(source: TopicPromptSource, topic: Topic, other_sources: List[TopicPromptSource]) -> str:
    return _get_uniqueness_of_source_as_compared_to_other_sources(source, other_sources, topic)


def get_context_of_source(source: TopicPromptSource, topic: Topic, other_sources: List[TopicPromptSource]) -> str:
    prompt_inputs = _get_prompt_inputs(source, other_sources, topic)
    del prompt_inputs['hint']
    system_message = SystemMessage(content=
                                   f"You are an expert in Jewish laws, history and traditions, wishing to teach your students a text related to {topic.title['en']}.\n"
                                   "# Task\n"
                                   f"Given a list of Jewish texts about '{topic.title['en']}', output helpful context for your students that doesn't exist in inputSource. This context should be gleaned from comparisonSources.\n"
                                   "# Input format\n"
                                   "Input will be in JSON format with the following structure\n"
                                   '{'
                                   '"topicTitle": "Title of the topic the sources are focusing on",'
                                   '"topicDescription": "Description of the topic",'
                                   '"inputSource": "Text of the source we want context for",'
                                   '"comparisonSources": "List of text of sources to compare `inputSource` to"'
                                   '}\n'
                                   "# Output format\n"
                                   "Output context for `inputSource`, extracted from `comparisonSources`, that will "
                                   "help your students understand the source. Output should be wrapped in <context> tags. Limit context to 50 words or less."
                                   )
    prompt = PromptTemplate.from_template('{{{{'
                                          '"topicTitle": "{topic_title}", "topicDescription": "{topic_description}",'
                                          '"inputSource": "{input_source}", "comparisonSources": {comparison_sources}'
                                          '}}}}')
    human_message = HumanMessage(content=prompt.format(**prompt_inputs))
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    response = llm([system_message, human_message])
    return get_by_xml_tag(response.content, 'context')


def summarize_based_on_uniqueness(text: str, uniqueness: str, llm, lang) -> str:
    system_message = SystemMessage(content=
                                   "You are an intelligent Jewish scholar who is knowledgeable in all aspects of the Torah and Jewish texts.\n"
                                   "# Task\n"
                                   "Given a Jewish text and an idea mentioned in this text, write a summary of the text"
                                   " that focuses on this idea.\n" 
                                   "# Input format\n"
                                   "Input will be in XML format with the following structure:\n"
                                   "<text> text to be summarized according to idea </text>\n"
                                   "<idea> idea mentioned in the text </idea>\n"
                                   "# Output format\n"
                                   f"A summary of the text that focuses on the idea, in 50 words or less, written in {lang}.\n"
                                   "Wrap the summary in <summary> tags."
                                   "Summary should start with the words \"The text discusses...\""
                                   )
    prompt = PromptTemplate.from_template("<text>{text}</text>\n<idea>{idea}</idea>")
    human_message = HumanMessage(content=prompt.format(text=text, idea=uniqueness))
    response = llm([system_message, human_message])
    match = re.search(r"<summary>\s*The text discusses (.+?)\s*</summary>", response.content)
    if not match:
        return None
    return match.group(1)


def _get_uniqueness_of_source_as_compared_to_other_sources(source: TopicPromptSource, other_sources: List[TopicPromptSource], topic: Topic) -> str:
    uniqueness_preamble = "The input source emphasizes"
    prompt_inputs = _get_prompt_inputs(source, other_sources, topic)
    system_message = SystemMessage(content=
                                   "You are an intelligent Jewish scholar who is knowledgeable in all aspects of the Torah and Jewish texts.\n"
                                   "# Task\n"
                                   "Given a list of Jewish texts about a certain topic, output the aspect that differentiates the input source from the other sources.\n"
                                   "# Input format\n"
                                   "Input will be in JSON format with the following structure\n"
                                   '{'
                                   '"topicTitle": "Title of the topic the sources are focusing on",'
                                   '"topicDescription": "Description of the topic",'
                                   '"inputSource": "Text of the source we want to differentiate from `comparisonSources`",'
                                   '"comparisonSources": "List of text of sources to compare `inputSource` to",'
                                   '"hint": "A hint as to what makes inputSource unique. This hint gives a very good indication as to what makes input source unique. Use it!'
                                   '}\n'
                                   "# Output format\n"
                                   "Output a summary that explains the aspect of `inputSource` that differentiates it "
                                   "from `comparisonSources`.\n"
                                   # "Summary should be no more than 20 words.\n"
                                   "Only mention the `inputSource`. Don't mention the `comparisonSources`.\n"
                                   f'Summary should complete the following sentence: "{uniqueness_preamble}...".'
                                   )
    prompt = PromptTemplate.from_template('{{{{'
                                          '"topicTitle": "{topic_title}", "topicDescription": "{topic_description}",'
                                          '"inputSource": "{input_source}", "comparisonSources": {comparison_sources},'
                                          '"hint": "{hint}"'
                                          '}}}}')
    human_message = HumanMessage(content=prompt.format(**prompt_inputs))
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    response = llm([system_message, human_message])
    uniqueness = re.sub(fr'^"?{uniqueness_preamble}\s*', '', response.content)
    uniqueness = re.sub(r'"$', '', uniqueness)
    return uniqueness
