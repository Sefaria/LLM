"""
Provide context for a source
"""
from sefaria_interface.topic_prompt_source import TopicPromptSource
from util.general import get_source_text_with_fallback, get_by_xml_tag
import re

from langchain.prompts import PromptTemplate
from basic_langchain.schema import HumanMessage, SystemMessage
from basic_langchain.chat_models import ChatAnthropic, ChatOpenAI


def context_from_surrounding_text(source: TopicPromptSource) -> str:
    """
    Given a source, provide context from its surrounding_text
    :return: context from surrounding_text
    """
    segment_text = get_source_text_with_fallback(source, "en", auto_translate=True)
    section_text = get_source_text_with_fallback(source, "en", auto_translate=True)
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
                                         f"<hint>{source.context_hint}</hint>")
    llm = ChatAnthropic(model="claude-2.1", temperature=0)
    response = llm([system_message, human_message])
    context = get_by_xml_tag(response.content, "relevant_context").strip()
    if context is None:
        return response.content
    context = re.sub(r"^The context describes ", "", context)
    return context


def general_context(source: TopicPromptSource):
    text = get_source_text_with_fallback(source, "en", auto_translate=True)
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
    human_message = HumanMessage(content=prompt.format(text=text, citation=source.ref, hint=source.context_hint))
    response = llm([system_message, human_message])
    return response.content


def get_context(source: TopicPromptSource):
    if source.surrounding_text:
        return context_from_surrounding_text(source)
    return general_context(source)


if __name__ == '__main__':
    # print(get_context(Ref("Nehemiah 8:14-16")))
    pass

