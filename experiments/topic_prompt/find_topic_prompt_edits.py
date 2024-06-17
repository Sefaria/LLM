"""
Goal is to find topic prompts that were edited by the learning team
"""
import django
django.setup()
from sefaria.model import *
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import SystemMessage, HumanMessage
from util.general import get_by_xml_list, get_by_xml_tag, run_parallel
import random

random.seed(34243)


def find_substantive_difference(texts):
    a, b = texts
    system = SystemMessage(content="Given two paragraphs, output the essential difference between them as a paraphrase."
                                   " The paraphrase should include the whole phrase with the difference."
                                   " If there are multiple differences, output each phrase with a difference."
                                   " Input text A wrapped in <text_a> tags and text B wrapped in <text_b> tags."
                                   " Output each difference in a <difference> tag."
                                   " Inside each <difference> tag include the original paraphrase from text A in"
                                   " <text_a> tags and the original paraphrase from text B in <text_b> tags.")
    human = HumanMessage(content=f"<text>{a}</text>\n<text>{b}</text>")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm([system, human])
    raw_differences = get_by_xml_list(response.content, "difference")
    differences = []
    for diff in raw_differences:
        a = get_by_xml_tag(diff, 'text_a')
        b = get_by_xml_tag(diff, 'text_b')
        differences.append((a, b))
    return differences


def find_substantive_difference_testing():
    a = """Rabbi Meir's ordination was a daring act of defiance against Roman persecution, ensuring the survival of Jewish legal traditions. The Epistle of Rav Sherira Gaon recounts the bravery of Rabbi Yehudah ben Bava, who risked his life to ordain rabbis during a time of severe Roman oppression."""
    b = """Rabbi Meir's ordination was a daring act of defiance against Roman persecution, ensuring the survival of Jewish legal traditions. The 10th-century Epistle of Rav Sherira Gaon recounts the bravery of Rabbi Yehudah ben Bava, who risked his life to ordain several rabbis, among them Rabbi Meir, during a time of severe Roman oppression."""
    diffs = find_substantive_difference(a, b)
    for d in diffs:
        print(d)


def find_differences_in_all_links():
    count = 0
    prompts_with_diffs = []
    for link in RefTopicLinkSet({"descriptions.en.ai_prompt": {"$exists": True}}):
        description = link.descriptions['en']
        if description['ai_prompt'] != description['prompt']:
            count += 1
            prompts_with_diffs += [(description['ai_prompt'], description['prompt'])]
    diffs_list = run_parallel(prompts_with_diffs, find_substantive_difference, max_workers=30, desc='find diffs')
    for diffs in diffs_list:
        for a_diff, b_diff in diffs:
            print('----')
            print(a_diff)
            print(b_diff)

    print(count)


if __name__ == '__main__':
    # find_substantive_difference_testing()
    find_differences_in_all_links()
