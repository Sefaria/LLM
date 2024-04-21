"""
Goal is to get many potentially relevant sources for a topic
To be filtered and sorted at a later stage
"""
from experiments.topic_source_curation.generate_questions import get_urls_for_slug, generate_questions_from_url_list
from app.util.pipeline import Artifact

def gather_sources(topic_slug):
    qs = Artifact(topic_slug) >> get_urls_for_slug >> generate_questions_from_url_list
    print(qs._data)


if __name__ == '__main__':
    gather_sources('dogs')