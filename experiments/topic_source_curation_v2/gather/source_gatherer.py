"""
- TopicPageSourceGetter
    - depends on export of topic ref links
    - get(slug: str) -> List[TopicPromptSource]

- SourceGatherer
    def init(topic_page_source_getter, source_querier, combo_question_generator)
    def gather(slug) -> List[TopicPromptSource]
        make sources unique before returning
        and filter_subset_refs

- filter_sources_about_topic(topic, sources: List[TopicPromptSource) -> List[TopicPromptSource]
    Artifact(topic, sources).pipe(cluster_sources).pipe(label_clusters).pipe(filter_clusters_not_about_topic).pipe(convert_clusters_to_list)
"""
import django
django.setup()
from sefaria.model.text import Ref
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from sefaria.helper.topic import get_topic
from experiments.topic_source_curation_v2.gather.source_querier import SourceQuerierFactory, AbstractSourceQuerier
from experiments.topic_source_curation_v2.gather.question_generator import create_multi_source_question_generator, AbstractQuestionGenerator
from util.pipeline import Artifact


def gather_sources_about_topic(topic: Topic) -> list[TopicPromptSource]:
    source_gatherer = _create_source_gatherer()
    return (Artifact(topic)
            .pipe(source_gatherer.gather)
            .pipe(_make_sources_unique)
            .pipe(_filter_sources_about_topic, topic).data
           )

def _make_sources_unique(sources: list[TopicPromptSource]) -> list[TopicPromptSource]:
    orefs = [Ref(source.ref) for source in sources]
    sources_by_tref = {source.ref: source for source in sources}
    unique_trefs = [oref.normal() for oref in filter_subset_refs(orefs)]
    return [sources_by_tref[tref] for tref in unique_trefs]

def _filter_sources_about_topic(sources: list[TopicPromptSource], topic: Topic) -> list[TopicPromptSource]:
    pass
    # Artifact(topic, sources).pipe(cluster_sources).pipe(label_clusters).pipe(filter_clusters_not_about_topic).pipe(convert_clusters_to_list)

def _create_source_gatherer() -> 'SourceGatherer':
    return SourceGatherer(
        TopicPageSourceGetter(),
        SourceQuerierFactory.create('chroma'),
        create_multi_source_question_generator()
    )


class SourceGatherer:

    def __init__(self,
                 topic_page_source_getter: 'TopicPageSourceGetter',
                 source_querier: AbstractSourceQuerier,
                 question_generator: AbstractQuestionGenerator
                ):
        self.topic_page_source_getter = topic_page_source_getter
        self.source_querier = source_querier
        self.question_generator = question_generator

    def gather(self, topic: Topic) -> list[TopicPromptSource]:
        questions = self.question_generator.generate(topic)
        sources: list[TopicPromptSource] = self.topic_page_source_getter.get(topic)
        for question in questions:
            temp_sources, _ = self.source_querier.query(question, 1000, 0.9)
            sources.extend(temp_sources)
        return sources


class TopicPageSourceGetter:

    @staticmethod
    def get(topic: Topic) -> list[TopicPromptSource]:
        return [_make_topic_prompt_source(Ref(tref), '', with_commentary=False) for tref in TopicPageSourceGetter._get_top_trefs_from_slug(topic.slug, None)]

    @staticmethod
    def _get_top_trefs_from_slug(slug, top_n=10) -> list[str]:
        out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
        try:
            trefs = [d['ref'] for d in out['refs']['about']['refs'] if not d['is_sheet']]
            return [tref for tref in trefs[:top_n] if Ref.is_ref(tref)]
        except KeyError:
            print('No refs found for {}'.format(slug))
            return []


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
