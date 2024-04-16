"""
Export topic pages data
"""
import json
import django
django.setup()
from topic_source_curation.sefaria_project_specific.common import get_top_sources_from_slug, convert_dataset_to_curated_topics

def export_topic_page(slugs: list[str]):
    dataset = {slug: get_top_sources_from_slug(slug, top_n=500) for slug in slugs}
    curated_topics = convert_dataset_to_curated_topics(dataset)
    with open('../input/exported_topic_pages.json', 'w') as fout:
        json.dump(curated_topics, fout, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    export_topic_page(['abraham', 'shabbat', 'dogs'])