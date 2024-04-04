from topic_source_curation.common import get_datasets
from sefaria_llm_interface.topic_source_curation import CuratedTopic
from collections import defaultdict


def metadata_metric(curated_topic: CuratedTopic) -> float:
    cat_counts = defaultdict(int)
    for source in curated_topic.sources:
        if source.author_name == "N/A":
            cat_counts["/".join(source.categories)] += 1
        else:
            cat_counts[source.author_name] += 1
    return len(cat_counts)/len(curated_topic.sources)


def _test_decision_boundar(good, bad, boundary):
    num_wrong = 0
    for x in good:
        if x < boundary:
            num_wrong += 1
    for x in bad:
        if x > boundary:
            num_wrong += 1
    return num_wrong

def _find_decision_boundary(good, bad):
    min_num_wrong = None
    best_boundary = None
    for x in good + bad:
        num_wrong = _test_decision_boundar(good, bad, x)
        if min_num_wrong is None or num_wrong < min_num_wrong:
            min_num_wrong = num_wrong
            best_boundary = x
    return best_boundary




if __name__ == '__main__':
    bad, good = get_datasets()
    print("GOOD")
    good_metrics = [metadata_metric(example) for example in good]
    bad_metrics = [metadata_metric(example) for example in bad]
    boundary = _find_decision_boundary(good_metrics, bad_metrics)
    for example, metric in zip(good, good_metrics):
        if metric < boundary:
            print("GOOD", example.topic.title['en'], metric)
    for example, metric in zip(bad, bad_metrics):
        if metric > boundary:
            print("BAD", example.topic.title['en'], metric)
