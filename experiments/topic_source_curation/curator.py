"""
Main
"""
from tqdm import tqdm
from functools import partial
from experiments.topic_source_curation.cluster import SummarizedSource
from experiments.topic_source_curation.choose import choose
from experiments.topic_source_curation.cache import load_sources, save_sources, load_clusters, save_clusters
from experiments.topic_source_curation.curation_context import get_context_for_source
from experiments.topic_source_curation.gather.source_gatherer import gather_sources_about_topic
from experiments.topic_source_curation.cluster import get_clustered_sources_based_on_summaries
from sefaria.helper.llm.topic_prompt import make_llm_topic
from sefaria_llm_interface.common.topic import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from util.pipeline import Artifact
from util.general import run_parallel
import csv
import random
import json
import django
django.setup()
from sefaria.model.text import library
from sefaria.model.topic import Topic as SefariaTopic

random.seed(45612)
TOPICS_TO_CURATE_CSV_PATH = 'input/Topic project plan - 1000 topics pages product - list of all topic slugs.csv'


def get_topics_to_curate():
    topics = []
    with open(TOPICS_TO_CURATE_CSV_PATH, "r") as fin:
        cin = csv.DictReader(fin)
        for row in cin:
            if len(row['curated']) > 0:
                continue
            slug = row['slug'].strip()
            try:
                topics += [make_llm_topic(SefariaTopic.init(slug))]
            except:
                print("Slug doesn't exist", row['slug'])
                continue
    return topics


def save_curation(data, topic: Topic) -> list[SummarizedSource]:
    sources, clusters = data
    contexts = run_parallel(sources, partial(get_context_for_source, topic=topic, clusters=clusters), max_workers=20, desc="Get source context")
    out = [{
        "ref": source.source.ref,
        "context": contexts[isource]
    } for (isource, source) in enumerate(sources)]
    with open(f"output/curation_{topic.slug}.json", "w") as fout:
        json.dump(out, fout, ensure_ascii=False, indent=2)
    return sources


def curate_topic(topic: Topic) -> list[TopicPromptSource]:
    return (Artifact(topic)
            .pipe(gather_sources_about_topic)
            .pipe(save_sources, topic)
            # .pipe(load_sources)
            .pipe(get_clustered_sources_based_on_summaries, topic)
            .pipe(save_clusters, topic)
            # .pipe(load_clusters)
            .pipe(choose, topic)
            .pipe(save_curation, topic).data
            )


if __name__ == '__main__':
    library.rebuild_toc()
    topics = get_topics_to_curate()
    print(len(topics))
    for t in topics[356:]:
        print("CURATING", t.slug)
        try:
            curated_sources = curate_topic(t)
        except:
            print(f"FAILED", t.slug)
