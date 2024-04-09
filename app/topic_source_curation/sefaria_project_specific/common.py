from tqdm import tqdm
import django
django.setup()
from sefaria.model.text import Ref
from sefaria.model.topic import Topic
from dataclasses import asdict
from sefaria.helper.topic import get_topic
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source, _make_llm_topic
from sefaria_llm_interface.topic_source_curation import CuratedTopic


def convert_dataset_to_curated_topics(dataset: dict[str, list[str]]) -> list[dict]:
    annotated_dataset = []
    for slug, trefs in tqdm(dataset.items(), total=len(dataset), desc='annotate_dataset'):
        annotated_dataset.append(CuratedTopic(
            _make_llm_topic(Topic.init(slug)),
            [_make_topic_prompt_source(Ref(tref), '', with_commentary=False) for tref in trefs]
        ))
    return [asdict(curated_topic) for curated_topic in annotated_dataset]


def get_top_sources_from_slug(slug, top_n=10) -> list[str]:
    out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
    try:
        trefs = [d['ref'] for d in out['refs']['about']['refs'][:top_n]]
        return [tref for tref in trefs if Ref.is_ref(tref)]
    except KeyError:
        print('No refs found for {}'.format(slug))
        return []

