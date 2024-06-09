"""
Given clusters tries to
1) Choose the "best" clusters such that each cluster represents an idea that should be part of the final topic page curation
2) For each chosen cluster, choose the "best" source to be curated. Best here needs to fulfill a few criteria including
    - Category quota: sources should be from diverse categories in Sefaria
    - Fundamental sources: funadmental sources from Tanakh and Talmud etc. should be chosen. These should be few, made 2-3
    - Interesting sources: the rest of the sources should represent interesting ideas for a newcomer to Sefaria
"""
from typing import TypeVar, Callable
import django
django.setup()
from sefaria.pagesheetrank import pagerank_rank_ref_list
from sefaria.model.text import Ref
from sefaria.client.wrapper import get_links
from sefaria.recommendation_engine import RecommendationEngine
import voyageai
from tqdm import tqdm
from experiments.topic_source_curation.cluster import Cluster, SummarizedSource, embed_text_openai, get_text_from_source
from experiments.topic_source_curation.common import get_topic_str_for_prompts
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from util.general import run_parallel, get_by_xml_tag
from sklearn.metrics import pairwise_distances
from util.pipeline import Artifact
from functools import reduce, partial
from statistics import mean, stdev
import numpy as np
from basic_langchain.schema import HumanMessage, SystemMessage
from basic_langchain.chat_models import ChatOpenAI
from sefaria_llm_interface.common.topic import Topic
from experiments.topic_source_curation.solver import solve_clusters
from experiments.topic_source_curation.scripts.analyze_gathered_sources import save_clusters_and_chosen_sources_to_html

T = TypeVar('T')


class CurationOption:
    """
    Represents a set of sources that is an option for curating a certain topic
    """
    def __init__(self, interesting_sources: list[SummarizedSource], not_interesting_sources: list[SummarizedSource], penalties: list[str]):
        self._interesting_sources = interesting_sources
        self._not_interesting_sources = not_interesting_sources
        self._penalties = penalties

    @property
    def sources(self) -> list[SummarizedSource]:
        return self._interesting_sources + self._not_interesting_sources

    @property
    def not_interesting_sources(self) -> list[SummarizedSource]:
        return self._not_interesting_sources

    @property
    def interesting_sources(self) -> list[SummarizedSource]:
        return self._interesting_sources

    @property
    def penalties(self) -> list[str]:
        return self._penalties

    def score(self, ranked_trefs: list[str]) -> float:
        """
        Score between 0 and 1 for how good this curation is
        Interesting sources are all considered to have equal weight
        Not interesting sources are ranked by `ranked_trefs` with a max weight 0.5 * interesting source weight
        :param ranked_trefs:
        :return:
        """
        not_interesting_ranks = [(len(ranked_trefs) - ranked_trefs.index(s.source.ref)) for s in self._not_interesting_sources]
        not_interesting_scores = [0.5 * (rank/len(ranked_trefs)) for rank in not_interesting_ranks]
        return (len(self._interesting_sources) + sum(not_interesting_scores))/len(self.sources)

# def choose_ideal_sources_for_clusters(clusters: list[Cluster], topic: Topic) -> list[TopicPromptSource]:
#     # return Artifact(clusters).pipe(sort_clusters, 20, topic).pipe(choose_ideal_sources).data
#     return Artifact(clusters).pipe(sort_clusters, 20, topic).pipe(solve_clusters).data


def choose(clusters: list[Cluster], topic: Topic) -> (list[SummarizedSource], list[Cluster]):
    primary_sources_trefs = choose_primary_sources(clusters)
    clusters = _put_primary_source_in_own_cluster(clusters, primary_sources_trefs)
    sorted_clusters = sort_clusters(clusters)
    sorted_items = _sort_by_highest_avg_pairwise_distance(reduce(lambda x, y: x + y.items, clusters, []), lambda x: x.summary)
    link_pairs = _get_link_pairs_to_avoid(sorted_items)
    chosen_sources, chosen_penalties, not_interesting_trefs = solve_clusters_iteratively(clusters, topic, sorted_items, primary_sources_trefs, link_pairs)
    chosen_sources = (Artifact(chosen_sources)
                      .pipe(_sort_sources_by_gpt_instruction, topic)
                      .pipe(_put_primary_sources_first, primary_sources_trefs)
                      .pipe(_remove_not_interesting_sources, not_interesting_trefs)
                      .pipe(_remove_duplicate_books).data
                      )
    save_clusters_and_chosen_sources_to_html(topic, sorted_clusters, chosen_sources, chosen_penalties, primary_sources_trefs, not_interesting_trefs)
    return chosen_sources, clusters


def _remove_not_interesting_sources(chosen_sources: list[SummarizedSource], not_interesting_trefs: list[str]) -> list[SummarizedSource]:
    return filter(lambda source: source.source.ref not in not_interesting_trefs, chosen_sources)


def _remove_duplicate_books(chosen_sources: list[SummarizedSource]) -> list[SummarizedSource]:
    """
    Prioritize sources higher in the list which have been sorted to be more interesting
    :param chosen_sources:
    :return:
    """
    seen_books = set()
    out_sources = []
    for source in chosen_sources:
        if source.source.book_title['en'] in seen_books:
            continue
        seen_books.add(source.source.book_title['en'])
        out_sources.append(source)
    return out_sources


def _get_link_pairs_to_avoid(sources: list[SummarizedSource]) -> set[tuple[str, str]]:
    link_pairs = []
    seg_to_orig_ref_map = {}
    seg_tref_set = set()
    for source in sources:
        seg_trefs = [r.normal() for r in Ref(source.source.ref).all_segment_refs()]
        for seg in seg_trefs:
            seg_to_orig_ref_map[seg] = source.source.ref
        seg_tref_set |= set(seg_trefs)
    for source in sources:
        links = get_links(source.source.ref, with_text=False)
        for link in links:
            if link['category'] not in {'Targum', 'Commentary'}:
                continue
            matching_trefs = set(link['anchorRefExpanded']) & seg_tref_set
            matching_other_side_trefs = set(r.normal() for r in Ref(link['ref']).all_segment_refs()) & seg_tref_set
            if len(matching_trefs) > 0 and len(matching_other_side_trefs) > 0:
                # both sides exist in sources
                for a_tref in matching_trefs:
                    for b_tref in matching_other_side_trefs:
                        link_pair = [seg_to_orig_ref_map[a_tref], seg_to_orig_ref_map[b_tref]]
                        link_pair.sort()
                        link_pairs.append(tuple(link_pair))
    return set(link_pairs)


def _put_primary_source_in_own_cluster(clusters: list[Cluster], primary_sources_trefs):
    primary_source_clusters = []
    for cluster in clusters:
        items_to_remove = []
        for i, (embedding, item) in enumerate(zip(cluster.embeddings, cluster.items)):
            if item.source.ref in primary_sources_trefs:
                new_label = len(clusters)+1+len(primary_source_clusters)
                primary_source_clusters.append(Cluster(new_label, [embedding], [item], item.summary))
                items_to_remove.append(i)
        for i in reversed(items_to_remove):
            cluster.embeddings.pop(i)
            cluster.items.pop(i)
    return primary_source_clusters + clusters


def _put_primary_sources_first(sources: list[SummarizedSource], primary_sources_trefs: list[str]) -> list[SummarizedSource]:
    primary_sources = [s for s in sources if s.source.ref in primary_sources_trefs]
    other_sources = [s for s in sources if s.source.ref not in primary_sources_trefs]
    return primary_sources + other_sources


def solve_clusters_iteratively(clusters: list[Cluster], topic: Topic, sorted_sources: list[SummarizedSource], primary_sources_trefs: list[str], link_pairs, verbose=True) -> (list[SummarizedSource], list[str], list[str]):
    """
    Run solve_clusters in a loop, trying to remove sources that aren't interesting
    :param clusters:
    :param topic:
    :param sorted_sources:
    :param primary_sources_trefs:
    :return:
    """
    max_iter = 10
    curation_options: list[CurationOption] = []
    not_interesting_trefs: set[str] = set()
    for _ in tqdm(range(max_iter), desc="Recursively solving clusters", disable=not verbose):
        curr_chosen_sources, curr_chosen_penalties = solve_clusters(clusters, sorted_sources, primary_sources_trefs, not_interesting_trefs, link_pairs)
        interesting, not_interesting = _bisect_sources_by_if_interesting(curr_chosen_sources, topic, verbose=False)
        curation_options.append(CurationOption(interesting, not_interesting, curr_chosen_penalties))
        if len(not_interesting) == 0:
            if verbose:
                print("--------------------")
                print("Found solution with all interesting sources!")
                print("--------------------")
            break
        chosen_not_interesting_trefs = {source.source.ref for source in not_interesting} & not_interesting_trefs
        if len(chosen_not_interesting_trefs) > 0:
            # LP was forced to choose a known not interesting source
            # there is no reason to keep on trying
            if verbose:
                print("--------------------")
                print("Was forced into solution with non-interesting sources")
                print("--------------------")
            not_interesting_trefs |= {source.source.ref for source in not_interesting}
            break
        not_interesting_trefs |= {source.source.ref for source in not_interesting}

    print("Choosing best curation...")
    best_curation = _choose_best_curation(curation_options, not_interesting_trefs, topic)

    print("All not interesting trefs")
    for tref in not_interesting_trefs:
        print('-', tref)
    print("Chosen not interesting trefs")
    for source in best_curation.not_interesting_sources:
        print('-', source.source.ref)
    return best_curation.sources, best_curation.penalties, [s.source.ref for s in best_curation.not_interesting_sources]


def _choose_best_curation(curation_options: list[CurationOption], not_interesting_trefs: set[str], topic: Topic) -> CurationOption:
    highest_num_interesting_sources = max([len(o.interesting_sources) for o in curation_options])
    tied_for_most_interesting = []
    for option in curation_options:
        if len(option.interesting_sources) == highest_num_interesting_sources:
            tied_for_most_interesting.append(option)
    if len(tied_for_most_interesting) == 1:
        best_curation = tied_for_most_interesting[0]
    else:
        sorted_not_interesting_trefs = _get_sorted_not_interesting_trefs(curation_options, not_interesting_trefs, topic)
        best_curation = _choose_best_curation_scoring_not_interesting(curation_options, sorted_not_interesting_trefs)
    return best_curation


def _choose_best_curation_scoring_not_interesting(curation_options: list[CurationOption], sorted_not_interesting_trefs: list[str]) -> CurationOption:
    return max(curation_options, key=lambda c: c.score(sorted_not_interesting_trefs))


def _get_sorted_not_interesting_trefs(curation_options: list[CurationOption], not_interesting_trefs: set[str], topic: Topic) -> list[str]:
    all_not_interesting_sources = []
    found_not_interesting_trefs = set()
    for option in curation_options:
        for source in option.sources:
            tref = source.source.ref
            if tref in not_interesting_trefs and tref not in found_not_interesting_trefs:
                all_not_interesting_sources.append(source)
                found_not_interesting_trefs.add(tref)
    return [s.source.ref for s in _sort_sources_by_gpt_instruction(all_not_interesting_sources, topic)]



def choose_primary_sources(clusters: list[Cluster]) -> list[str]:
    """
    Returns list of primary sources as trefs
    :param clusters:
    :return:
    """
    orefs = reduce(lambda x, y: x + [Ref(item.source.ref) for item in y.items], clusters, [])
    refs, pageranks = zip(*pagerank_rank_ref_list(orefs))
    max_ref = refs[0].normal()
    thresh = mean(pageranks) + 2 * stdev(pageranks)
    is_primary = pageranks[0] > thresh
    print(max_ref, "IS PRIMARY:", is_primary, round(pageranks[0], 3), round(thresh, 3))
    if is_primary:
        return [max_ref]
    return []


def choose_ideal_clusters(clusters: list[Cluster], max_clusters: int) -> list[Cluster]:
    """
    Sorted in descending order from outliers to central
    Choose a few central clusters and a few outliers
    Also might want to use custom GPT sort to find "best" clusters based on various criteria
    """
    # sorted_clusters = _sort_by_highest_avg_pairwise_distance(clusters)
    # sorted_clusters = _sort_clusters_by_instruction(clusters)
    return [c for c in clusters if len(clusters) > 1]


def sort_clusters(clusters: list[Cluster]) -> list[Cluster]:
    return _sort_by_highest_avg_pairwise_distance(clusters, lambda x: x.summary)

def choose_ideal_sources(source_clusters: list[Cluster], verbose=True) -> list[TopicPromptSource]:
    """
    Criteria could be:
        Pagerank based on link graph for topic page. Higher means more relevant
        Pagerank delta == Global PR - Local PR. Need to decide what is good
        Highest average pairwise cosine distance. Higher means more unique
        Fulfills category quota. Want to choose sources from different categories
    """
    ideal_sources = []
    choose_primary_sources(source_clusters)
    for cluster in tqdm(source_clusters, desc='choose ideal sources', disable=not verbose):
        pass
       # ideal_sources += [choose_ideal_source_from_cluster(cluster)]
    return ideal_sources


def choose_ideal_source_from_cluster(cluster: Cluster) -> TopicPromptSource:
    if len(cluster) == 1:
        return cluster.items[0]
    vo = voyageai.Client()
    output = vo.rerank(cluster.summary, [get_text_from_source(item) for item in cluster.items], "rerank-lite-1")
    best_idx = output.results[0].index
    return cluster.items[best_idx]

def _sort_by_query_voayageai(documents: list[str], query:str):
    vo = voyageai.Client()
    output = vo.rerank(query, documents, "rerank-lite-1")
    sorted_by_relevance = [result.document for result in output.results]
    # best_idx = output.results[0].index
    return sorted_by_relevance




def _get_highest_avg_pairwise_distance_indices(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 1:
        return np.array([0])
    distances = pairwise_distances(embeddings, metric='cosine')
    sum_distances = np.sum(distances, axis=1)
    avg_distances = sum_distances / (len(embeddings) - 1)
    sorted_indices = np.argsort(avg_distances)[::-1]  # Sort in descending order
    return sorted_indices


def _sort_by_highest_avg_pairwise_distance(items: list[T], key: Callable[[T], str], verbose=True) -> list[T]:
    embeddings = np.array(run_parallel([key(x) for x in items], embed_text_openai, max_workers=100, desc="Embedding summaries for interestingness sort", disable=not verbose))
    sorted_indices = _get_highest_avg_pairwise_distance_indices(embeddings)
    return [items[i] for i in sorted_indices]


def get_gpt_compare(system_prompt, human_prompt_generator, llm):
    content_to_val = {"1":-1, "2":1, "0":0}
    def gpt_compare(a, b) -> int:
        response = llm([system_prompt, human_prompt_generator(a, b)])
        # print(a)
        # print(b)
        # print(response.content, content_to_val.get(response.content, 0))
        return content_to_val.get(response.content, 0)

    return gpt_compare


def sort_by_instruction(documents,  comparison_instruction, key_extraction_func=lambda x:x):
    from functools import cmp_to_key
    message_suffix = " The only output should be either '1' or '2' or '0'"
    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    system = SystemMessage(
        content=
        comparison_instruction
        +message_suffix)
    human_generator = lambda a, b: HumanMessage(content=f"1) {key_extraction_func(a)}\n2) {key_extraction_func(b)}")
    documents.sort(key=cmp_to_key(get_gpt_compare(system, human_generator, llm)))
    # for document in documents:
    #     print(document)
    return documents
def _apply_sorting_to_clusters(clusters: list[Cluster], summaries:[str]) -> list[Cluster]:
    sorted = []
    for summary in summaries:
        cluster = [cluster for cluster in clusters if cluster.summary==summary][0]
        sorted.append(cluster)
    return sorted
# def _apply_sorting_to_sources_within_cluster(cluster: Cluster, summaries:[str]) -> list[Cluster]:
#     sorted = []
#     summarized_sources = cluster.items
#     for summary in summaries:
#         summarized_source = [summarized_source for summarized_source in summarized_sources if summarized_source.summary==summary][0]
#         sorted.append(summarized_source)
#     cluster.items = sorted
#     return sorted
def _apply_sorting_to_sources_within_cluster(cluster: Cluster, texts:[str]) -> list[Cluster]:
    sorted = []
    summarized_sources = cluster.items
    for text in texts:
        # summarized_source = [summarized_source for summarized_source in summarized_sources if summarized_source.source.text["en"] == text][0]
        summarized_source = [summarized_source for summarized_source in summarized_sources if summarized_source.summary == text][0]
        sorted.append(summarized_source)
    cluster.items = sorted
    return sorted
def _sort_clusters_by_instruction(clusters: list[Cluster], topic: Topic) -> list[Cluster]:
    summaries = [c.summary for c in clusters]

    interestingness_instruction =f"""You are an expert in Jewish laws, history and traditions, wishing to teach your student about {topic.title} in light of Jewish tradition.
    Given 2 topics related to {topic.title}, output the index of the one which presents a more interesting, surprising, wild and/or non-trivial information regarding {topic.title}, which might be captivating and intriguing to your students. If both are equally non-trivial and interesting with regards to {topic.title}, output 0.  
    """
    fundamentalness_instruction ="""You are an expert in Jewish laws, history and traditions, wishing to teach your student about the historical person of Cyrus in light of Jewish tradition.
    Given 2 topics related to king Cyrus, output the index of the one which presents a more fundamental and basic fact about Cyrus, one that your students should learn first before learning the other. If both are equally fundamental and no one is more important than the other, output 0.  
    """
    # interesting = sort_by_instruction(summaries[:], interestingness_instruction)
    interesting = sort_by_instruction(clusters, interestingness_instruction, lambda cluster: cluster.summary)
    # fundamental = sort_by_instruction(summaries[:], fundamentalness_instruction)


    # interesting_voyageai = _sort_by_query_voayageai(summaries, "I am looking for the most surpsring, interesting, wild and non-trivial sources about king Cyrus as is reflected in Jewish sources and tradition")
    # from pprint import pprint
    # pprint("GPT INTERESTINGNESS SORT:")
    # pprint(interesting)
    #
    # pprint("VOYAGEAI INTERESTINGESS SORT:")
    # pprint(interesting_voyageai)
    a = "halt"

    return interesting

def _sort_sources_by_gpt_instruction(sources: list[SummarizedSource], topic: Topic):
    if len(sources) <= 1:
        return sources
    interestingness_instruction = f"""You are an expert in Jewish laws, history and traditions, wishing to teach your student about {topic.title} in light of Jewish tradition.
        Given 2 texts related to {topic.title}, output the index of the one which presents a more interesting, surprising, wild and/or non-trivial information regarding {topic.title}, which might be captivating and intriguing to your students. If both are equally non-trivial and interesting, output 0.  
        """
    interesting = sort_by_instruction(sources, interestingness_instruction, lambda item: item.summary)
    return interesting


def _determine_if_source_is_interesting(source: SummarizedSource, topic: Topic):
    topic_str = get_topic_str_for_prompts(topic, verbose=False)
    interestingness_instruction = f"""You are an expert in Jewish laws, history and traditions, wishing to teach your 
     students a text related to {topic_str}. Output if you think the input text will be captivating and intriguing to your 
     students. The source should be highly relevant to {topic.title}. Text will be wrapped in <text> tags. Output should 
     be either 'Yes' or 'No' wrapped in <is_interesting> tags
     """
    system = SystemMessage(content=interestingness_instruction)
    text = source.source.text
    human = HumanMessage(content=f"<text>{text.get('en', text['he'])}</text>")
    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    response = llm([system, human])
    answer = get_by_xml_tag(response.content, "is_interesting")
    return answer.lower().strip() == "yes"


def _bisect_sources_by_if_interesting(sources: list[SummarizedSource], topic: Topic, verbose=True) -> (list[SummarizedSource], list[SummarizedSource]):
    is_interesting_list = run_parallel(sources, partial(_determine_if_source_is_interesting, topic=topic),
                                       max_workers=100, desc="Bisect sources by if interesting", disable=not verbose)
    interesting, boring = [], []
    for is_interesting, source in zip(is_interesting_list, sources):
        temp_list = interesting if is_interesting else boring
        temp_list.append(source)
    return interesting, boring
