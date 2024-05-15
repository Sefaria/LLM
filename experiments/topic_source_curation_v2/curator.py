"""
Main
"""
from experiments.topic_source_curation_v2.gather.source_gatherer import gather_sources_about_topic
from experiments.topic_source_curation_v2.cluster import get_clustered_sources_based_on_summaries
from experiments.topic_source_curation_v2.choose import choose_ideal_sources_for_clusters
from sefaria.helper.llm.topic_prompt import _make_llm_topic
from sefaria_llm_interface.common.topic import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from util.pipeline import Artifact
from dataclasses import asdict
import csv
import random
import json
import django
django.setup()
from sefaria.model.topic import Topic as SefariaTopic

TOPICS_TO_CURATE_CSV_PATH = 'input/Topic project plan - 1000 topics pages product - list of all topic slugs.csv'

def get_topics_to_curate():
    topics = []
    with open(TOPICS_TO_CURATE_CSV_PATH, "r") as fin:
        cin = csv.DictReader(fin)
        for row in cin:
            slug = row['slug'].strip()
            try:
                topics += [_make_llm_topic(SefariaTopic.init(slug))]
            except:
                print("Slug doesn't exist", row['slug'])
                continue
    return topics

def save_gathered_sources(sources: list[TopicPromptSource], topic: Topic) -> None:
    with open(f"_cache/gathered_sources_{topic.slug}.json", "w") as fout:
        json.dump([asdict(s) for s in sources], fout, indent=2, ensure_ascii=False)

def load_gathered_sources(topic: Topic) -> list[TopicPromptSource]:
    with open(f"_cache/gathered_sources_{topic.slug}.json", "r") as fin:
        raw_sources = json.load(fin)
        return [TopicPromptSource(**s) for s in raw_sources]

def curate_topic(topic: Topic) -> list[TopicPromptSource]:
    return (Artifact(topic)
            .pipe(gather_sources_about_topic)
            .pipe(save_gathered_sources, topic))
            # .pipe(load_gathered_sources)
            # .pipe(get_clustered_sources_based_on_summaries, topic)
            # .pipe(choose_ideal_sources_for_clusters).data)

if __name__ == '__main__':
    topics = get_topics_to_curate()
    for topic in random.sample(topics, 10):
        print("CURATING", topic.slug)
        sources = curate_topic(topic)
    # print('---CURATION---')
    # print('num sources', len(sources))
    # for source in sources:
    #     s = source.source
    #     print('---')
    #     print('\t-', s.ref)
    #     print('\t-', s.text['en'])


