"""
Creates dataset for measuring success of topic source curation
"""
import csv
import json
import django
import random
from tqdm import tqdm
from collections import defaultdict
from dataclasses import asdict
django.setup()
from sefaria.model import *
from sefaria.helper.topic import get_topic
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source, _make_llm_topic
from sefaria_llm_interface.topic_source_curation.curated_topic import CuratedTopic


random.seed(613)


def get_top_sources_from_slug(slug, top_n=10) -> list[str]:
    out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
    try:
        trefs = [d['ref'] for d in out['refs']['about']['refs'][:top_n]]
        return [tref for tref in trefs if Ref.is_ref(tref)]
    except KeyError:
        print('No refs found for {}'.format(slug))
        return []


def get_bad_curation_dataset() -> dict[str, list[str]]:
    with open("../input/bad topic pages (for source curation data set) - Sheet1.csv", "r") as fin:
        cin = csv.DictReader(fin)
        slugs = [row['slug'] for row in cin]
    return {slug: get_top_sources_from_slug(slug) for slug in slugs}


def get_good_curation_dataset(n=50) -> dict[str, list[str]]:
    links = RefTopicLinkSet({"descriptions.en.title": {"$exists": True}})
    dataset = defaultdict(list)
    for link in links:
        dataset[link.toTopic] += [link.ref]
    dataset = {k: v for k, v in random.sample(list(dataset.items()), k=n)}
    return dataset


def annotate_dataset(dataset: dict[str, list[str]]) -> list[dict]:
    annotated_dataset = []
    for slug, trefs in tqdm(dataset.items(), total=len(dataset), desc='annotate_dataset'):
        annotated_dataset.append(CuratedTopic(
            _make_llm_topic(Topic.init(slug)),
            [_make_topic_prompt_source(Ref(tref), '', with_commentary=False) for tref in trefs]
        ))
    return [asdict(curated_topic) for curated_topic in annotated_dataset]




def output_dataset(name, dataset):
    with open(f'../input/{name}.json', 'w') as fout:
        json.dump(dataset, fout, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    library.rebuild_toc()
    output_dataset('bad_curation', annotate_dataset(get_bad_curation_dataset()))
    output_dataset('good_curation', annotate_dataset(get_good_curation_dataset()))
