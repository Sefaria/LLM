from typing import Any, Callable
import django
django.setup()
from sefaria.model.text import Ref
from tqdm import tqdm
from functools import reduce, partial
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source, _make_llm_topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from basic_langchain.chat_models import ChatAnthropic
from basic_langchain.schema import HumanMessage, SystemMessage
from util.general import get_by_xml_tag
from sefaria.helper.topic import get_topic
from experiments.topic_source_curation_v2.gather.source_querier import SourceQuerierFactory, AbstractSourceQuerier
from experiments.topic_source_curation_v2.gather.question_generator import create_multi_source_question_generator, AbstractQuestionGenerator
from experiments.topic_source_curation_v2.cluster import get_text_from_source, Cluster
from experiments.topic_source_curation_v2.common import filter_invalid_refs, run_parallel
from util.pipeline import Artifact
from sefaria.model.topic import Topic as SefariaTopic


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union
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
    return _get_items_relevant_to_topic(sources, get_text_from_source, topic)

def _create_source_gatherer() -> 'SourceGatherer':
    return (
        SourceGatherer(
        # CategoryAwareSourceGatherer(
        TopicPageSourceGetter(),
        SourceQuerierFactory.create('chroma'),
        create_multi_source_question_generator()
    ))


class SourceGatherer:

    def __init__(self,
                 topic_page_source_getter: 'TopicPageSourceGetter',
                 source_querier: AbstractSourceQuerier,
                 question_generator: AbstractQuestionGenerator
                ):
        self.topic_page_source_getter = topic_page_source_getter
        self.source_querier = source_querier
        self.question_generator = question_generator

    def gather(self, topic: Topic, verbose=True) -> list[TopicPromptSource]:
        questions = self.question_generator.generate(topic)
        sources: list[TopicPromptSource] = self.topic_page_source_getter.get(topic)
        sources = []
        for question in tqdm(questions, desc='gather sources', disable=not verbose):
            temp_sources, _ = self.source_querier.query(question, 10, 0.2)
            sources.extend(temp_sources)
        print(f'total sources: {len(sources)}')
        return sources

class CategoryAwareSourceGatherer:

    def __init__(self,
                 topic_page_source_getter: 'TopicPageSourceGetter',
                 source_querier: AbstractSourceQuerier,
                 question_generator: AbstractQuestionGenerator
                ):
        self.topic_page_source_getter = topic_page_source_getter
        self.source_querier = source_querier
        self.question_generator = question_generator

    def gather(self, topic: Topic) -> list[TopicPromptSource]:
        questions_and_categories = self.question_generator.generate(topic)
        sources: list[TopicPromptSource] = self.topic_page_source_getter.get(topic)
        for question_and_cats in questions_and_categories:
            question, cats = question_and_cats[0], question_and_cats[1]
            temp_sources, _ = self.source_querier.query(question, 500, 0.2)
            temp_sources = self._filter_by_category(temp_sources, cats)
            temp_sources = self._add_base_text_sources(temp_sources)
            sources.extend(temp_sources)
        return sources
    def _filter_by_category(self, sources: list[TopicPromptSource], categories):
        if not categories:
            return sources
        filtered = [source for source in sources if any(cat in source.categories for cat in categories)]
        return filtered

    def _extract_base_text_source_from_commentary(self, source: TopicPromptSource):
        sefaria_commentary_ref = Ref(source.ref)
        if sefaria_commentary_ref.is_commentary():
            commentary_links = [link for link in sefaria_commentary_ref.linkset().array() if link.type == "commentary"]
            if not commentary_links:
                return None
            linked_refs = commentary_links[0].refs
            base_text_ref = linked_refs[0] if Ref(linked_refs[0]).book != sefaria_commentary_ref.book else linked_refs[1]
            return _make_topic_prompt_source(Ref(base_text_ref), '', with_commentary=False)
        else:
            return None
    def _add_base_text_sources(self, sources: list[TopicPromptSource]):
        extracted_sources = []
        for source in sources:
            base_text_source = self._extract_base_text_source_from_commentary(source)
            if base_text_source:
                extracted_sources.append(base_text_source)
        return sources + extracted_sources



class TopicPageSourceGetter:

    @staticmethod
    def get(topic: Topic) -> list[TopicPromptSource]:
        return [_make_topic_prompt_source(Ref(tref), '', with_commentary=False) for tref in TopicPageSourceGetter._get_top_trefs_from_slug(topic.slug, None)]

    @staticmethod
    def _get_top_trefs_from_slug(slug, top_n=10) -> list[str]:
        out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
        try:
            trefs = [d['ref'] for d in out['refs']['about']['refs'] if not d['is_sheet']]
            trefs = filter_invalid_refs(trefs[:top_n])
            return trefs
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

def _get_topic_description(topic: Topic):
    return f"{topic.title['en']}\nDescription: {topic.description.get('en', 'N/A')}"

def _get_items_relevant_to_topic(items: list[Any], key: Callable[[Any], str], topic: Topic, verbose=True):
    topic_description = _get_topic_description(topic)
    unit_func = partial(_is_text_about_topic, topic_description)
    is_about_topic_list = run_parallel([key(item) for item in items], unit_func, 2,
                                       desc="filter irrelevant sources", disable=not verbose)
    filtered_items = []
    if verbose:
        print("---FILTERING---")
    for is_about_topic, item in zip(is_about_topic_list, items):
        if is_about_topic:
            filtered_items += [item]
        else:
            if verbose:
                print(item.ref)
                print(key(item))
    if verbose:
        print('after filtering: ', len(filtered_items))
    return filtered_items

def _is_text_about_topic(topic_description, text):
    llm = ChatAnthropic(model='claude-3-opus-20240229', temperature=0)
    system = SystemMessage(content="You are a Jewish scholar. Given a topic description wrapped in <topic> and a text, "
                                   "wrapped in <text>, output 'Yes' if <text> is about <topic> and 'No' if <text> is "
                                   "not about <topic>. Wrap output in <answer> tags.")
    human = HumanMessage(content=f"<topic>{topic_description}</topic>\n<text>{text}</text>")
    response = llm([system, human])
    answer = get_by_xml_tag(response.content, 'answer').strip()
    if answer not in {'Yes', 'No'}:
        print(f"Answer not in Yes or No: {answer}")
        return False
    return answer == 'Yes'

if __name__ == "__main__":
    topic = _make_llm_topic(SefariaTopic.init('jesse'))
    gather_sources_about_topic(topic)