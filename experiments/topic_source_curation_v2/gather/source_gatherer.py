from typing import Any, Callable
import django
django.setup()
from sefaria.model.text import Ref
from openai import BadRequestError
from tqdm import tqdm
from functools import reduce, partial
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source, _make_llm_topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
from sefaria.recommendation_engine import RecommendationEngine
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import HumanMessage, SystemMessage
from util.general import get_by_xml_tag, run_parallel
from util.topic import get_top_trefs_from_slug
from util.pipeline import Artifact
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness
from experiments.topic_source_curation_v2.common import get_topic_str_for_prompts
from experiments.topic_source_curation_v2.gather.source_querier import SourceQuerierFactory, AbstractSourceQuerier
from experiments.topic_source_curation_v2.gather.question_generator import create_multi_source_question_generator, AbstractQuestionGenerator, WebPageQuestionGenerator
from experiments.topic_source_curation_v2.cluster import get_text_from_source
from experiments.topic_source_curation_v2.summarized_source import SummarizedSource
from sefaria.model.topic import Topic as SefariaTopic


def gather_sources_about_topic(topic: Topic) -> list[TopicPromptSource]:
    source_gatherer = _create_source_gatherer()
    return (Artifact(topic)
            .pipe(source_gatherer.gather)
            .pipe(_make_sources_unique)
            .pipe(_summarize_sources_parallel, topic)
            .pipe(_filter_sources_about_topic, topic, _get_summary_from_source)
            .pipe(_filter_sources_about_topic, topic, get_text_from_source)
            .pipe(_filter_targum_thats_redundant)
            .pipe(_combine_close_sources).data
           )

def _does_text_add_information(texts):
    text1, text2 = texts
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system = SystemMessage(content="Given two texts from the Torah cannon, text1 and text2, output 'Yes' if text2 adds significant information not present in text1. Output 'No' if text2 has basically the same information as text. text1 is wrapped in <text1> tags. text2 is wrapped in <text2> tags. Output 'Yes' or 'No' wrapped in <answer> tags.")
    human = HumanMessage(content=f"<text1>{text1}</text1>\n<text2>{text2}</text2>")
    response = llm([system, human])
    adds_info = get_by_xml_tag(response.content, 'answer').lower().strip() == 'yes'
    return adds_info

def _filter_targum_thats_redundant(sources: list[SummarizedSource], verbose=True) -> list[SummarizedSource]:
    """
    Many targum sources don't add any more information than the pasuk. Filter these out.
    :param sources:
    :return:
    """
    from sefaria.model.link import LinkSet
    out_sources = []
    texts_to_check = []
    sources_to_check = []
    for source in sources:
        s = source.source
        if "|".join(s.categories).startswith("Tanakh|Targum"):
            links = LinkSet({"refs": s.ref, "type": "targum"})
            tanakh_ref = None
            for link in links:
                tanakh_ref = link.ref_opposite(Ref(s.ref))
                if tanakh_ref.primary_category == "Tanakh":
                    break
            tanakh_source = _make_topic_prompt_source(tanakh_ref, '', with_commentary=False)
            texts_to_check.append((tanakh_source.text['en'], s.text['en']))
            sources_to_check.append(source)
        else:
            out_sources += [source]
    do_sources_add_info = run_parallel(texts_to_check, _does_text_add_information, max_workers=100, desc="checking if sources add info", disable=not verbose)
    for does_add_info, source in zip(do_sources_add_info, sources_to_check):
        if does_add_info:
            out_sources += [source]
            print(f"Targum adds info! {source.source.ref}\n{source.source.text['en']}")
        # else:
        #     print(f"Targum adds NO info! {source.source.ref}\n{source.source.text['en']}")
    return out_sources

def _combine_close_sources(sources: list[SummarizedSource]) -> list[SummarizedSource]:
    """
    Currently only combines sources in Tanakh because segments tend to be small and highly related
    :param sources:
    :return:
    """
    orefs = [Ref(source.source.ref) for source in sources]
    ref_clusters = RecommendationEngine.cluster_close_refs(orefs, sources, 2)
    clustered_sources = []
    for ref_cluster in ref_clusters:
        curr_refs = [data['ref'] for data in ref_cluster]
        curr_sources = [data['data'] for data in ref_cluster]
        if curr_refs[0].primary_category == "Tanakh":
            ranged_oref = curr_refs[0].to(curr_refs[-1])
            if len(curr_refs) > 1:
                print("COMBINED", ranged_oref, " | ".join(r.normal() for r in curr_refs))
            new_source = _make_topic_prompt_source(ranged_oref, '', with_commentary=False)
            clustered_sources.append(SummarizedSource(new_source, ". ".join([s.summary for s in curr_sources])))
        else:
            # don't combine commentary refs
            clustered_sources += curr_sources
    return clustered_sources

def _make_sources_unique(sources: list[TopicPromptSource]) -> list[TopicPromptSource]:
    orefs = [Ref(source.ref) for source in sources]
    sources_by_tref = {source.ref: source for source in sources}
    unique_trefs = [oref.normal() for oref in filter_subset_refs(orefs)]
    return [sources_by_tref[tref] for tref in unique_trefs]


def _get_summary_from_source(source: SummarizedSource) -> str:
    return source.summary


def _filter_sources_about_topic(sources: list[SummarizedSource], topic: Topic, key) -> list[SummarizedSource]:
    return _get_items_relevant_to_topic(sources, key, topic)

def _create_source_gatherer() -> 'SourceGatherer':
    return (
        SourceGatherer(
        # CategoryAwareSourceGatherer(
        TopicPageSourceGetter(),
        SourceQuerierFactory.create('chroma_all'),
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
        questions = self.question_generator.generate(topic, verbose=False)
        topic_page_sources: list[TopicPromptSource] = self.topic_page_source_getter.get(topic)
        retrieved_sources = []
        for question in tqdm(questions, desc='gather sources', disable=not verbose):
            temp_sources, _ = self.source_querier.query(question, 100, 0.5)
            retrieved_sources.extend(temp_sources)
        if verbose:
            print(f'RECALL', self._calculate_recall_of_sources(topic_page_sources, retrieved_sources))
        return topic_page_sources + retrieved_sources

    @staticmethod
    def _calculate_recall_of_sources(topic_page_sources: list[TopicPromptSource], retrieved_sources: list[TopicPromptSource]) -> float:
        """
        A bit pedantic is how specific the metric is, but I wanted to make sure it's 100% accurate
        :param topic_page_sources:
        :param retrieved_sources:
        :return:
        """
        topic_page_sources = _make_sources_unique(topic_page_sources)
        retrieved_sources = _make_sources_unique(retrieved_sources)
        topic_page_trefs = set()
        for source in topic_page_sources:
            topic_page_trefs |= {subref.normal() for subref in Ref(source.ref).range_list()}
        num_recalled = 0
        recalled_trefs = set()
        for source in retrieved_sources:
            for subref in Ref(source.ref).range_list():
                if subref.normal() in topic_page_trefs:
                    num_recalled += 1
                    recalled_trefs.add(subref.normal())
                    break
        not_recalled_trefs = set()
        for source in topic_page_sources:
            recalled = False
            for subref in Ref(source.ref).range_list():
                if subref.normal() in recalled_trefs:
                    recalled = True
                    break
            if not recalled:
                not_recalled_trefs.add(source.ref)

        return num_recalled / len(topic_page_sources)


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
        return [_make_topic_prompt_source(Ref(tref), '', with_commentary=False) for tref in get_top_trefs_from_slug(topic.slug, None)]


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

def _summarize_source(llm: object, topic_str: str, source: TopicPromptSource):
    source_text = source.text['en'] if len(source.text['en']) > 0 else source.text['he']
    if len(source_text) == 0:
        return None
    summary = summarize_based_on_uniqueness(source_text, topic_str, llm, "English")
    if summary is None:
        return None
    return SummarizedSource(source, summary)


def _summarize_sources_parallel(sources: list[TopicPromptSource], topic: Topic, verbose=True) -> list[SummarizedSource]:
    llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)
    topic_str = get_topic_str_for_prompts(topic)
    return run_parallel(sources, partial(_summarize_source, llm, topic_str), 100,
                        desc="summarize sources", disable=not verbose)


def _get_items_relevant_to_topic(items: list[Any], key: Callable[[Any], str], topic: Topic, verbose=True):
    topic_description = get_topic_str_for_prompts(topic)
    unit_func = partial(_is_text_about_topic, topic_description)
    is_about_topic_list = run_parallel([key(item) for item in items], unit_func, 100,
                                       desc="filter irrelevant sources", disable=not verbose)
    filtered_items = []
    if verbose:
        print("---FILTERING---")
    for is_about_topic, item in zip(is_about_topic_list, items):
        if is_about_topic:
            filtered_items += [item]
        else:
            pass
            # if verbose:
            #     print(item.ref)
            #     print(key(item))
    if verbose:
        print('after filtering: ', len(filtered_items))
    return filtered_items

def _is_text_about_topic(topic_description, text):
    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    system = SystemMessage(content="You are a Jewish scholar. Given a topic description wrapped in <topic> and a text, "
                                   "wrapped in <text>, output 'Yes' if <text> is about <topic> and 'No' if <text> is "
                                   "not about <topic>. Wrap output in <answer> tags.")
    human = HumanMessage(content=f"<topic>{topic_description}</topic>\n<text>{text}</text>")
    try:
        response = llm([system, human])
    except BadRequestError:
        return False
    answer = get_by_xml_tag(response.content, 'answer').strip()
    if answer.strip().lower() not in {'yes', 'no'}:
        print(f"Answer not in Yes or No: {answer}")
        return False
    return answer == 'Yes'

if __name__ == "__main__":
    topic = _make_llm_topic(SefariaTopic.init('jesse'))
    gather_sources_about_topic(topic)