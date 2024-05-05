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
import django
django.setup()
from sefaria.model.topic import Topic as SefariaTopic

def curate_topic(topic: Topic) -> list[TopicPromptSource]:
    return (Artifact(topic)
            .pipe(gather_sources_about_topic)
            .pipe(get_clustered_sources_based_on_summaries, topic)
            .pipe(choose_ideal_sources_for_clusters).data)

if __name__ == '__main__':
    slug = "stars"
    topic = _make_llm_topic(SefariaTopic.init(slug))
    curate_topic(topic)

