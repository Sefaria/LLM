from experiments.topic_source_curation.cache import load_sources, load_clusters
from experiments.topic_source_curation.cluster import Cluster, SummarizedSource
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from sefaria_llm_interface.common.topic import Topic
import django
django.setup()
from sefaria.model.topic import Topic as SefariaTopic
from sefaria.model.text import Ref
from sefaria.helper.llm.topic_prompt import make_llm_topic
from collections import defaultdict
import csv, os, glob, re, json


def count_cats(slug):
    sources = load_sources(make_llm_topic(SefariaTopic.init(slug)))
    cat_counts = defaultdict(list)
    for source in sources:
        assert isinstance(source, TopicPromptSource)
        cat_counts[source.categories[0]] += [source.ref]
    for cat, trefs in cat_counts.items():
        print(f'{cat}: {len(trefs)}')
        for ref in trefs:
            print('\t', ref)


def print_clusters(slug):
    clusters = load_clusters(make_llm_topic(SefariaTopic.init(slug)))
    for cluster in clusters:
        print(f'{cluster.summary}: {len(cluster)}')
        for item in cluster.items:
            print('\t', item.source.ref)
            print('\t\t', item.summary)


def save_clusters_to_csv(slug):
    clusters = load_clusters(make_llm_topic(SefariaTopic.init(slug)))
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


def save_clusters_to_html(slug):
    topic = make_llm_topic(SefariaTopic.init(slug))
    clusters = load_clusters(topic)
    html = _make_cluster_html_wrapper(topic, ''.join(_make_cluster_html(cluster) for cluster in clusters))
    with open("output/clusters_{}.html".format(slug), 'w') as fout:
        fout.write(html)

def save_custom_clusters_to_html(topic, clusters):
    html = _make_cluster_html_wrapper(topic, ''.join(_make_cluster_html(cluster) for cluster in clusters))
    with open("scripts/output/clusters_{}.html".format(topic.slug), 'w') as fout:
        fout.write(html)
def save_clusters_and_chosen_sources_to_html(topic, clusters, chosen_sources, chosen_penalties, primary_sources_trefs, not_interesting_trefs):
    html = _make_cluster_and_penalties_html_wrapper(topic, chosen_sources, chosen_penalties, primary_sources_trefs, not_interesting_trefs, ''.join(_make_cluster_with_chosen_sources_html(cluster, chosen_sources) for cluster in clusters))
    with open("output/clusters_and_chosen_sources_{}.html".format(topic.slug), 'w') as fout:
        fout.write(html)


def _make_cluster_html(cluster: Cluster):
    return f"""
    <details class="cluster">
        <summary>
            <h2 class="clusterSummary">
                {cluster.summary} ({len(cluster)})
            </h2>
        </summary>
        <div class="clusterSources">
        {''.join([_make_cluster_source_html(source, '') for source in cluster.items])}
        </div>
    </details>
    """
def _make_source_for_chosen_sources_html(source: SummarizedSource, chosen_sources: list[SummarizedSource]):
    if source in chosen_sources:
        return _make_cluster_chosen_source_html(source)
    else:
        return _make_cluster_source_html(source, "")

def _make_cluster_with_chosen_sources_html(cluster: Cluster, chosen_sources: list[SummarizedSource]):
    return f"""
    <details class="cluster">
        <summary>
            <h2 class="clusterSummary">
                {cluster.summary} ({len(cluster)})
            </h2>
        </summary>
        <div class="clusterSources">
        {''.join([_make_source_for_chosen_sources_html(source, chosen_sources) for source in cluster.items])}
        </div>
    </details>
    """


def _make_cluster_source_html(source: SummarizedSource, source_type: str):
    source_type2class = {
        "primary": "primary",
        "not_interesting": "not-interesting",
        "N/A": "",
        "": "",
    }
    return f"""
    <details class="clusterSource {source_type2class[source_type]}">
        <summary>
        <h3><a target="_blank" href="https://www.sefaria.org/{Ref(source.source.ref).url()}">{source.source.ref}</a> ({Ref(source.source.ref).primary_category})</h3>
        {source.summary}
        </summary>
        <blockquote class="he">
            {source.source.text['he']}
        </blockquote>
        <blockquote>
            {source.source.text['en']}
        </blockquote>
    </details>
    """

def _make_cluster_chosen_source_html(source: SummarizedSource):
    return f"""
    <div style="background-color: yellow;">
    <details class="clusterSource">
        <summary>
        <h3><a target="_blank" href="https://www.sefaria.org/{Ref(source.source.ref).url()}">{source.source.ref}</a> ({Ref(source.source.ref).primary_category})</h3>
        {source.summary}
        </summary>
        <blockquote class="he">
            {source.source.text['he']}
        </blockquote>
        <blockquote>
            {source.source.text['en']}
        </blockquote>
    </details>
    </div>
    """

def convert_list_to_html(strings):
    html_list = "<ul>\n"
    if strings:
        for string in strings:
            html_list += f"  <li>{string}</li>\n"
    else:
        html_list += f"None\n"
    html_list += "</ul>"
    return html_list
def _make_cluster_html_wrapper(topic, content):
    return f"""
    <html>
        <style>
            body {{
                max-width: 750px;
                margin-left: auto;
                margin-right: auto;
            }}
            .he {{
                direction: rtl;
                font-size: 120%;
            }}
            .cluster {{
                margin-bottom: 30px;
            }}
            .clusterSummary {{
                display: inline;
            }}
            .clusterSource {{
                margin-left: 15px;
                margin-bottom: 30px;
            }}
            .clusterSource h3 {{
                display: inline;
            }}
        </style>
        <body>
            <h1>{topic.title['en']} Clusters</h1>
            {content}
        </body>
    </html>
    """


def _make_cluster_and_penalties_html_wrapper(topic, chosen_sources, penalties, primary_sources_trefs, not_interesting_trefs, content):
    source_types = []
    for source in chosen_sources:
        if source.source.ref in primary_sources_trefs:
            source_types.append("primary")
        elif source.source.ref in not_interesting_trefs:
            source_types.append("not_interesting")
        else:
            source_types.append("N/A")
    return f"""
    <html>
        <style>
            body {{
                max-width: 750px;
                margin-left: auto;
                margin-right: auto;
            }}
            .he {{
                direction: rtl;
                font-size: 120%;
            }}
            .primary {{
                border: 4px solid #1F4E79;
            }}
            .not-interesting {{
                border: 4px solid #B22222;
            }}
            .cluster {{
                margin-bottom: 30px;
            }}
            .clusterSummary {{
                display: inline;
            }}
            .clusterSource {{
                margin-left: 15px;
                margin-bottom: 30px;
            }}
            .clusterSource h3 {{
                display: inline;
            }}
        </style>
        <body>
            <h1>{topic.title['en']} Clusters and Chosen Sources (chose {len(chosen_sources)})</h1>
            <h2>Chosen Sources (including primary source)</h2>
            {''.join(_make_cluster_source_html(source, source_type) for (source, source_type) in zip(chosen_sources, source_types))}
            <h2>Penalties</h2>
            {convert_list_to_html(penalties)}
            {content}
        </body>
    </html>
    """


def load_all_gathered_sources():
    file_pattern = os.path.join("_cache", 'gathered_sources_*')
    slugs = [re.search(r"gathered_sources_(.*)\.json", filename).group(1) for filename in glob.glob(file_pattern)]
    for slug in slugs:
        topic = SefariaTopic.init(slug)
        if topic is None:
            continue
        yield load_sources(make_llm_topic(topic)), slug


def export_gathered_sources():
    rows_by_ref = {}
    slugs_by_ref = defaultdict(list)
    count = 0
    for sources, slug in load_all_gathered_sources():
        count += 1
        for source in sources:
            assert isinstance(source, SummarizedSource)
            s = source.source
            assert isinstance(s, TopicPromptSource)
            slugs_by_ref[s.ref].append(slug)
            rows_by_ref[s.ref] = {
                "Ref": s.ref,
                "Slugs": [],
                "English": s.text['en']
            }
    rows = []
    for ref, row in rows_by_ref.items():
        row['Slugs'] = slugs_by_ref[ref]
        rows.append(row)
    print("len", len(rows))
    print('num slugs', count)
    with open('output/topic_modelling_training_set.json', 'w') as fout:
        json.dump(rows, fout, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    slugs = ['abraham-in-egypt']
    # count_cats(slug)
    # print_clusters(slug)
    # slugs = ['ants', 'ulla', 'achitofel', 'friendship', 'david-and-the-temple', 'cains-sacrifice', 'abraham-in-egypt']
    # slugs = ['war-with-midian', 'medicine']
    # for slug in slugs:
    #     print(slug)
    #     save_clusters_to_csv(slug)
    #     save_clusters_to_html(slug)
    export_gathered_sources()
