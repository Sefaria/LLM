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
import csv
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

def curate_topic(topic: Topic) -> list[TopicPromptSource]:
    return (Artifact(topic)
            .pipe(gather_sources_about_topic)
            .pipe(get_clustered_sources_based_on_summaries, topic)
            .pipe(choose_ideal_sources_for_clusters).data)

if __name__ == '__main__':
    slug = "friendship"
    topic = _make_llm_topic(SefariaTopic.init(slug))
    curate_topic(topic)
