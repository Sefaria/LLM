from tqdm import tqdm
import django
django.setup()
from sefaria.model.text import Ref
from sefaria.model.topic import Topic
from dataclasses import asdict
from sefaria.helper.topic import get_topic
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source, _make_llm_topic
from sefaria_llm_interface.topic_source_curation import CuratedTopic

def filter_subset_refs_old(orefs: list[Ref]) -> list[Ref]:
    orefs.sort(key=lambda x: x.order_id())
    deduped_orefs = []
    skip_next = False
    for ioref, oref in enumerate(orefs[:-1]):
        if skip_next:
            skip_next = False
            continue
        next_oref = orefs[ioref+1]
        if oref.index.title != next_oref.index.title:
            # optimization, the easiest cases to check for
            deduped_orefs.append(oref)
        elif oref.contains(next_oref):
            # will be dealt with in next iteration
            continue
        elif next_oref.contains(oref):
            # unfortunately Ref.order_id() doesn't consistently put larger refs before smaller ones
            # e.g. Tosafot on Berakhot 2 precedes Tosafot on Berakhot Chapter 1...
            # check if next match actually contains this match
            deduped_orefs += [oref]
            skip_next = True
        else:
            deduped_orefs += [oref]
    if len(orefs) > 0:
        # never dealt with last oref
        deduped_orefs += [orefs[-1]]
    return deduped_orefs

def filter_subset_refs(orefs: list[Ref]) -> list[Ref]:
    orefs.sort(key=lambda x: x.order_id())
    deduped_orefs = []
    for ioref, oref in enumerate(orefs[:-1]):
        if len(deduped_orefs) == 0:
            deduped_orefs += [oref]
            continue
        prev_oref = deduped_orefs[-1]
        if oref.index.title != prev_oref.index.title:
            # optimization, the easiest cases to check for
            deduped_orefs += [oref]
        elif oref.contains(prev_oref):
            # will be dealt with in next iteration
            continue
        elif prev_oref.contains(oref):
            # unfortunately Ref.order_id() doesn't consistently put larger refs before smaller ones
            # e.g. Tosafot on Berakhot 2 precedes Tosafot on Berakhot Chapter 1...
            # check if next match actually contains this match
            deduped_orefs.pop()
            deduped_orefs += [oref]
        else:
            deduped_orefs += [oref]
    if len(orefs) > 0:
        # never dealt with last oref
        deduped_orefs += [orefs[-1]]
    return deduped_orefs


def filter_subset_refs_from_dataset(dataset: dict[str, list[str]]) -> dict[str, list[str]]:
    for slug, trefs in dataset.items():
        orefs = [Ref(tref) for tref in trefs]
        dataset[slug] = [oref.normal() for oref in filter_subset_refs(orefs)]
    return dataset


def convert_dataset_to_curated_topics(dataset: dict[str, list[str]]) -> list[dict]:
    dataset = filter_subset_refs_from_dataset(dataset)
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

        trefs = [d['ref'] for d in out['refs']['about']['refs'] if not d['is_sheet']]
        return [tref for tref in trefs[:top_n] if Ref.is_ref(tref)]
    except KeyError:
        print('No refs found for {}'.format(slug))
        return []

