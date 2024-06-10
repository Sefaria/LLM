"""
A way to generate and save the output of topic prompt generation
without using celery
"""
from collections import defaultdict

import django
django.setup()
from sefaria.helper.llm.topic_prompt import make_topic_prompt_input, get_ref_context_hints_by_lang
from sefaria_llm_interface.topic_prompt import TopicPrompt
from sefaria.model import Topic as SefariaTopic
from topic_prompt.topic_prompt_generator import get_toprompts
from topic_prompt.html_formatter import HTMLFormatter
import json


def get_topic_prompts_from_file(filepath):
    with open(filepath, 'r') as fin:
        ref_topic_links = json.load(fin)
    by_slug = defaultdict(list)
    for ref_topic_link in ref_topic_links:
        by_slug[ref_topic_link['toTopic']] += [ref_topic_link]
    for slug, temp_ref_topic_links in by_slug.items():
        topic = SefariaTopic.init(slug)
        for lang, ref__context_hints in get_ref_context_hints_by_lang(temp_ref_topic_links).items():
            orefs, context_hints = zip(*ref__context_hints)
            tp_input = make_topic_prompt_input('en', topic, orefs, context_hints)
            toprompt_options_list = get_toprompts(tp_input)
            formatter = HTMLFormatter(toprompt_options_list)
            formatter.save('../experiments/topic_prompt/output/{}.html'.format(slug))


if __name__ == '__main__':
    get_topic_prompts_from_file('../experiments/topic_source_curation/output/ref_topic_links.json')