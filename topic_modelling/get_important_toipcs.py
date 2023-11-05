"""
A heuristic for a getting a list of important topics that we care about for topic tagging
"""

from typing import List
from tqdm import tqdm
import django
django.setup()
from sefaria.model import *


def get_important_topics() -> List[Topic]:
    topic_set = TopicSet({"numSources": 10})


def topics_by_good_data_sources(min_links=5) -> List[Topic]:
    good_data_sources = ['aspaklaria', 'aspaklaria-edited-by-sefaria', 'sefaria', 'sefer-haagada']
    topic_set = TopicSet({"numSources": {"$gte": min_links}})
    important_topics = []
    for topic in topic_set:
        ref_links = topic.link_set(_class="refTopic", query_kwargs={"dataSource": {"$in": good_data_sources}})
        if ref_links.count() >= min_links:
            important_topics += [topic]
    return important_topics


def topics_only_sheets(all_topics):
    out = []
    for topic in tqdm(all_topics, desc="only sheets", total=all_topics.count()):
        ref_links = topic.link_set(_class="refTopic")
        if not all(l.is_sheet for l in ref_links): continue
        out += [topic]
    return out


def get_inverse_topic_list(topic_list, all_topics):
    only_sheets_slugs = {t.slug for t in topics_only_sheets(all_topics)}
    good_slugs = {t.slug for t in topic_list}
    inverse_topic_list = []
    for topic in all_topics:
        if topic.slug in good_slugs or topic.slug in only_sheets_slugs: continue
        inverse_topic_list += [topic]
    return inverse_topic_list


if __name__ == '__main__':
    all_topics = TopicSet()
    good = topics_by_good_data_sources()
    print(len(good))
    bad = get_inverse_topic_list(good, all_topics)
    for b in bad:
        print(b.slug, b.get_primary_title('en'))
    print(len(bad))

"""
sure signals:
- over 10 sources
- over 5 aspaklaria sources
- subclass person

everything:
- unless 
"""