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
from util.topic import get_or_generate_topic_description
import csv
import random
import json
import django
django.setup()
from sefaria.model.text import library
from sefaria.model.topic import Topic as SefariaTopic

random.seed(45612)
TOPICS_TO_CURATE_CSV_PATH = 'input/Topic project plan - 1000 topics pages product - list of all topic slugs.csv'


def get_topics_to_curate(return_all=False) -> list[Topic]:
    topics = []
    with open(TOPICS_TO_CURATE_CSV_PATH, "r") as fin:
        cin = csv.DictReader(fin)
        for row in cin:
            if len(row['curated'].strip()) > 0 and not return_all:
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
    topic.description['en'] = get_or_generate_topic_description(topic, verbose=False)
    # contexts = run_parallel(sources, partial(get_context_for_source, topic=topic, clusters=clusters), max_workers=20, desc="Get source context")
    # not finding context helpful
    contexts = ['']*len(sources)
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


def get_topics_that_havent_been_curated_yet(root=None) -> list[Topic]:
    """
    Get all filenames from output folder
    :return:
    """
    from os import listdir
    from os.path import isfile, join
    import re
    slugs_curated = {re.sub(r"curation_(.*)\.json", r"\1", f) for f in listdir("output") if isfile(join("output", f)) and re.match("curation_(.*)\.json", f)}
    if root is None:
        topics_to_curate = get_topics_to_curate(return_all=True)
    else:
        topics_to_curate = [make_llm_topic(t) for t in root.topics_by_link_type_recursively(only_leaves=True)]

    topics_not_yet_curated = []
    for topic in topics_to_curate:
        if topic.slug in slugs_curated:
            continue
        topics_not_yet_curated.append(topic)
    for topic in topics_not_yet_curated:
        print(topic.slug)
    return topics_not_yet_curated


if __name__ == '__main__':
    library.rebuild_toc()
    topics = get_topics_that_havent_been_curated_yet()
    print(len(topics))
    for t in topics:
        print("CURATING", t.slug)
        try:
            curated_sources = curate_topic(t)
        except Exception as e:
            print(f"FAILED", t.slug)
            print(e)
