"""
Goal is to get many potentially relevant sources for a topic
To be filtered and sorted at a later stage
"""
import re
import django
django.setup()
from sefaria.model.topic import Topic
from sefaria.helper.llm.topic_prompt import _make_llm_topic
from experiments.topic_source_curation.generate_questions import get_urls_for_slug, generate_questions_from_url_list
from experiments.topic_source_curation.query_sources import SourceQuerier
from experiments.topic_source_curation.common import is_text_about_topic
from app.util.pipeline import Artifact
from functools import partial
from tqdm import tqdm

def bisect_right_with_key(a, x, lo=0, hi=None, key=None):
    if key is None:
        key = lambda y: y
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if key(x) < key(a[mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo

def get_topic_description(topic_slug):
    topic = Topic.init(topic_slug)
    llm_topic = _make_llm_topic(topic)
    return f"{llm_topic.title['en']}\nDescription: {llm_topic.description.get('en', 'N/A')}"

def get_sources_relevant_to_topic_biscect(topic_slug, docs):
    topic_description = get_topic_description(topic_slug)
    def bisector(doc):
        if isinstance(doc, int):
            return doc
        # print('------\n\nrunning bisector')
        # print(doc[0].page_content)
        is_relevant = is_text_about_topic(topic_description, doc[0].page_content)
        # print(is_relevant)
        return 0 if is_relevant else 1
    idx_first_irrelevant_source = bisect_right_with_key(docs, 0, key=bisector)
    print("\n\nFIRST IRRELEVANT", idx_first_irrelevant_source)
    return docs[:idx_first_irrelevant_source]

def get_sources_relevant_to_topic_exact(topic_slug, docs):
    topic_description = get_topic_description(topic_slug)
    return list(filter(lambda doc: is_text_about_topic(topic_description, doc[0].page_content), docs))

def get_ref_from_doc(doc):
    ref_reg = re.compile(r'Reference: (.*?)\. Version')
    return ref_reg.search(doc[0].metadata['source']).group(1)

def get_all_sources(topic_slug, top_k, score_threshold, questions) -> list:
    querier = SourceQuerier()
    docs_by_ref = {}
    query_sources = partial(querier.query_sources, top_k=top_k, score_threshold=score_threshold)
    for question in tqdm(questions, desc='Querying sources'):
        print('question', question)
        temp_docs = (Artifact(question) >> query_sources >>
                     partial(get_sources_relevant_to_topic_biscect, topic_slug) >>
                     partial(get_sources_relevant_to_topic_exact, topic_slug))
        print('FINAL RELEVANT', len(temp_docs.data))
        for doc in temp_docs.data:
            docs_by_ref[get_ref_from_doc(doc)] = doc
    return list(docs_by_ref.values())

def gather_sources(topic_slug):
    sources = (Artifact(topic_slug) >> get_urls_for_slug >> generate_questions_from_url_list >>
          partial(get_all_sources, topic_slug, 1000, 0.9))
    print([get_ref_from_doc(doc) for doc in sources.data])
    print(len(sources.data))



if __name__ == '__main__':
    gather_sources('dogs')