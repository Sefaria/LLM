"""
Provide context for a source
"""
import django
django.setup()
from sefaria.model import *
from util.general import get_ref_text_with_fallback, get_by_xml_tag
import re

import langchain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def _get_context_ref(segment_oref: Ref):
    if segment_oref.primary_category == "Tanakh":
        return segment_oref.section_ref()
    elif segment_oref.index.get_primary_corpus() == "Bavli":
        passage = Passage.containing_segment(segment_oref)
        return passage.ref()
    return None


def context_from_section(segment_oref: Ref, context_oref: Ref, context_hint: str) -> str:
    """
    Given a segment ref, provide context from its section ref
    :param segment_oref:
    :param context_oref:
    :param context_hint:
    :return: context from section ref
    """
    segment_text = get_ref_text_with_fallback(segment_oref, "en", auto_translate=True)
    section_text = get_ref_text_with_fallback(context_oref, "en", auto_translate=True)
    system_message = SystemMessage(content="# Identity\nYou are a Jewish scholar familiar with all Torah texts.\n"
                                           "# Task\nGiven a segment of Torah text and surrounding context text, output"
                                           "a summary of the relevant parts of the context that will help users"
                                           " understand the segment of Torah text."
                                           "# Input format\n"
                                           "- Segment of Torah text surrounded by <segment> XML tags.\n"
                                           "- Context text surrounded by <context> XML tags.\n"
                                           "- Hint as to what the relevant context is surrounded by <hint> XML tags. This hint gives a very good indication as to the relevant context. Use it!\n"
                                           "# Output format\n"
                                           "Summary of the relevant context text to help users understand <segment>"
                                           " text. Output should be surrounded in <relevant_context> XML"
                                           " tags. No more than 50 words. Summary should start with the word 'The"
                                           " context describes'.")
    human_message = HumanMessage(content=f"<segment>{segment_text}</segment>\n"
                                         f"<context>{section_text}</context>\n"
                                         f"<hint>{context_hint}</hint>")
    llm = ChatAnthropic(model="claude-2", temperature=0, max_tokens_to_sample=100000)
    response = llm([system_message, human_message])
    context = get_by_xml_tag(response.content, "relevant_context").strip()
    if context is None:
        return response.content
    context = re.sub(r"^The context describes ", "", context)
    return context


def general_context(oref, context_hint):
    text = get_ref_text_with_fallback(oref, "en", auto_translate=True)
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_message = SystemMessage(content=f"""
    Given a text from the Jewish cannon, add any relevant context that would help a user understand this text from a
    Jewish perspective. Relevant context may be:
    If this text is a prayer, when was it recited and why?
    Historical significance to Jewish history
    How this text is viewed nowadays by Jewish people
    
    DO NOT offer an interpretation or explanation of the text. Only offer helpful context.
    
    Hint as to what the relevant context is surrounded by <hint> XML tags. 
    This hint gives a very good indication as to the relevant context. Use it!
    
    Limit to 50 words or less.
    """)

    prompt = PromptTemplate.from_template("Citation: {citation}\nText: {text}\nHint: {hint}")
    human_message = HumanMessage(content=prompt.format(text=text, citation=oref.normal(), hint=context_hint))
    response = llm([system_message, human_message])
    return response.content


def get_context(oref: Ref, context_hint=None):
    context_oref = _get_context_ref(oref)
    if context_oref:
        return context_from_section(oref, context_oref, context_hint)
    return general_context(oref, context_hint)


if __name__ == '__main__':
    # print(get_context(Ref("Nehemiah 8:14-16")))
    print(get_context(Ref("Berakhot 61b:9-10")))

