import django
django.setup()
import csv
from tqdm import tqdm
from sefaria.model import *
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import SystemMessage, HumanMessage
from openai import BadRequestError
from util.general import get_by_xml_tag, run_parallel
from util.sefaria_specific import get_ref_text_with_fallback
from experiments.topic_source_curation.common import get_topic_str_for_prompts
from sefaria.helper.llm.topic_prompt import make_llm_topic
from functools import partial
from collections import defaultdict


def _is_text_about_topic(topic_description, general_topic_example, text):
    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    system = SystemMessage(content="You are a Jewish scholar. Given a topic description wrapped in <topic> and a text, "
                                   "wrapped in <text>, output 'Yes' if <text> is about <topic> and 'No' if <text> is "
                                   f"not about <topic>. Wrap output in <answer> tags. If topic is general (e.g. {general_topic_example}), only output 'Yes' if <text> is highly relevant to the topic, not just tangentially related.")
    human = HumanMessage(content=f"<topic>{topic_description}</topic>\n<text>{text}</text>")
    try:
        response = llm([system, human])
    except BadRequestError:
        return False
    answer = get_by_xml_tag(response.content, 'answer').strip()
    if answer.strip().lower() not in {'yes', 'no'}:
        print(f"Answer not in Yes or No: {answer}")
        return False
    return answer == 'Yes'


def _get_refs_relevant_to_topic(topic, refs: list[tuple[Ref, float]]):
    topic_description = topic.title['en']  # get_topic_str_for_prompts(topic, verbose=False)
    is_general = Topic.init(topic.slug).numSources > 100
    unit_func = partial(_is_text_about_topic, topic_description, topic.title['en'] if is_general else "Torah")
    texts = []
    for ref, score in refs:
        try:
            texts.append(get_ref_text_with_fallback(ref, 'en'))
        except:
            continue
    is_about_topic_list = run_parallel(texts, unit_func, 100, desc="filter irrelevant sources", disable=True)
    is_about_refs = []
    for ref, is_about in zip(refs, is_about_topic_list):
        if is_about:
            is_about_refs.append(ref)
    return is_about_refs


def _organize_links_by_slug(slug__ref_list: list[tuple[str, str, float]]):
    links_by_slug = defaultdict(list)
    for slug, ref, score in slug__ref_list:
        links_by_slug[slug].append((ref, score))
    return links_by_slug


def get_all_relevant_links():
    with open("/Users/nss/Downloads/topic links - reflinks by score.csv", "r") as fin:
        cin = csv.DictReader(fin)
        slug__ref_list = [(row['toTopic'], row['ref'], float(row['score'])) for row in cin]
    good_rows = []
    bad_rows = []
    links_by_slug = _organize_links_by_slug(slug__ref_list)
    for slug, trefs in tqdm(links_by_slug.items(), total=len(links_by_slug)):
        relevant_refs = _get_refs_relevant_to_topic(make_llm_topic(Topic.init(slug)), [(Ref(tref), score) for tref, score in trefs])
        for ref, score in relevant_refs:
            good_rows.append({"slug": slug, "ref": ref.normal(), "score": score})
        for tref in (set(trefs) - {r.normal() for r, s in relevant_refs}):
            bad_rows.append({"slug": slug, "ref": tref})
    with open("/Users/nss/Downloads/topic links final.csv", "w") as fout:
        cout = csv.DictWriter(fout, ['slug', 'ref', 'score'])
        cout.writeheader()
        cout.writerows(good_rows)
    with open("/Users/nss/Downloads/topic links final bad.csv", "w") as fout:
        cout = csv.DictWriter(fout, ['slug', 'ref'])
        cout.writeheader()
        cout.writerows(bad_rows)


if __name__ == '__main__':
    get_all_relevant_links()

"""
tikkun, kehillah, torah, religion, derekh eretz, psychology, political thought
"""