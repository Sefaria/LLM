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
from topic_source_curation.sefaria_project_specific.common import get_top_sources_from_slug, convert_dataset_to_curated_topics


random.seed(613)


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


def output_dataset(name, dataset):
    with open(f'../input/{name}.json', 'w') as fout:
        json.dump(dataset, fout, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    library.rebuild_toc()
    output_dataset('bad_curation', convert_dataset_to_curated_topics(get_bad_curation_dataset()))
    output_dataset('good_curation', convert_dataset_to_curated_topics(get_good_curation_dataset()))
