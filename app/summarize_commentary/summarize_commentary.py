from typing import List
from sefaria_llm_interface import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource, TopicPromptCommentary
from util.openai_utils import get_completion_openai, count_tokens_openai
from basic_langchain.chat_models import ChatAnthropic
from basic_langchain.schema import HumanMessage


def get_prompt(source: TopicPromptSource, topic: Topic, commentary: str):
    topic_name, topic_description = get_topic_prompt(topic)
    prompt = (
        f"# Input:\n"
        f"1) Commentary: commentary on the verse {source.ref}.\n"
        f"2) Topic: topic which relates to this verse\n"
        f"3) Topic Description: description of topic\n"
        f"# Task: Summarize the main points discussed by the commentators. Only include points that relate to the"
        f" topic \"{topic_name}\".\n"
        f"# Output: Numbered list of main points, only when relating to the topic \"{topic_name}\".\n"
        f"-----\n"
        f"# Input:\n1) Topic: {topic_name}\n2) Topic Description: {topic_description}\n3) Commentary: {commentary}"
    )
    return prompt


def get_topic_prompt(topic: Topic):
    return topic.title['en'], topic.description.get('en', '')


def truncate_commentary(commentary: List[TopicPromptCommentary], max_tokens=7000):
    commentary_text = ""
    for comment in commentary:
        commentary_text += f"Source: {comment.ref}\n{comment.text.get('en', 'N/A')}\n"
        if count_tokens_openai(commentary_text) > max_tokens:
            break
    return commentary_text


def summarize_commentary(source: TopicPromptSource, topic: Topic, company='openai'):
    commentary_text = truncate_commentary(source.commentary)
    prompt = get_prompt(source, topic, commentary_text)

    if company == 'openai':
        num_tokens = count_tokens_openai(prompt)
        print(f"Number of commentary tokens: {num_tokens}")
        completion = get_completion_openai(prompt)
    elif company == 'anthropic':
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
        completion = llm([HumanMessage(content=prompt)]).content
    else:
        raise Exception("No valid company passed. Options are 'openai' or 'anthropic'.")
    return completion


def print_summarized_commentary(tref, topic_slug):
    completion = summarize_commentary(tref, topic_slug)
    print(completion)


if __name__ == '__main__':
    print_summarized_commentary('Exodus 10:1-2', 'haggadah')
