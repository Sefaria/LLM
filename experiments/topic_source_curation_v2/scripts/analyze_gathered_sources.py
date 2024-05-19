from experiments.topic_source_curation_v2.cache import load_sources, load_clusters
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
import django
django.setup()
from sefaria.model.topic import Topic as SefariaTopic
from sefaria.helper.llm.topic_prompt import _make_llm_topic
from collections import defaultdict
import csv


def count_cats(slug):
    sources = load_sources(_make_llm_topic(SefariaTopic.init(slug)))
    cat_counts = defaultdict(list)
    for source in sources:
        assert isinstance(source, TopicPromptSource)
        cat_counts[source.categories[0]] += [source.ref]
    for cat, trefs in cat_counts.items():
        print(f'{cat}: {len(trefs)}')
        for ref in trefs:
            print('\t', ref)


def print_clusters(slug):
    clusters = load_clusters(_make_llm_topic(SefariaTopic.init(slug)))
    for cluster in clusters:
        print(f'{cluster.summary}: {len(cluster)}')
        for item in cluster.items:
            print('\t', item.source.ref)
            print('\t\t', item.summary)


def save_clusters_to_csv(slug):
    clusters = load_clusters(_make_llm_topic(SefariaTopic.init(slug)))
    rows = []
    for cluster in clusters:
        for item in cluster.items:
            rows += [{
                "Cluster Label": cluster.label,
                "Cluster Summary": cluster.summary,
                "Source Summary": item.summary,
                "Ref": item.source.ref,
                "Text": item.source.text['en'],
            }]
    with open("output/clusters_{}.csv".format(slug), 'w') as fout:
        cout = csv.DictWriter(fout, ['Cluster Label', 'Cluster Summary', 'Source Summary', 'Ref', 'Text'])
        cout.writeheader()
        cout.writerows(rows)



if __name__ == '__main__':
    count_cats('rabbinic-authority')
    print_clusters('rabbinic-authority')
    # slugs = ['ants', 'ulla', 'achitofel', 'friendship', 'david-and-the-temple', 'cains-sacrifice', 'abraham-in-egypt']
    # slugs = ['war-with-midian', 'medicine']
    # for slug in slugs:
    #     print(slug)
    #     save_clusters_to_csv(slug)
