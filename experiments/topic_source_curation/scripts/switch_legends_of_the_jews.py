import django
django.setup()
from sefaria.model.topic import Topic
from sefaria.helper.llm.topic_prompt import make_llm_topic
from sefaria.model.text import Ref
from experiments.topic_source_curation.common import get_topic_str_for_prompts
from experiments.topic_source_curation.cache import load_clusters
from topic_prompt.uniqueness_of_source import summarize_based_on_uniqueness
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import os, re, json, glob
from util.general import run_parallel
import numpy as np
from basic_langchain.embeddings import OpenAIEmbeddings
from basic_langchain.chat_models import ChatOpenAI
from basic_langchain.schema import SystemMessage, HumanMessage
import csv


def embed_text_openai(text):
    return np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(text))


def embed_parallel(items, key):
    return run_parallel([key(item) for item in items], embed_text_openai, max_workers=50, disable=True)


def load_chosen_sources(slug):
    with open(f"output/curation_{slug}.json") as fin:
        return json.load(fin)


def load_all_chosen_sources():
    file_pattern = os.path.join("output", 'curation_*')
    slugs = [re.search(r"curation_(.*)\.json", filename).group(1) for filename in glob.glob(file_pattern)]
    for slug in slugs:
        yield load_chosen_sources(slug), slug



def switch_all_legends_of_the_jews():
    total_legends = 0
    rows = []
    llm = ChatOpenAI("gpt-4o-mini", temperature=0)
    for sources, slug in tqdm(list(load_all_chosen_sources())):
        sefaria_topic = Topic.init(slug)
        if sefaria_topic is None:
            topic_str = "N/A"
        else:
            topic = make_llm_topic(sefaria_topic)
            topic_str = get_topic_str_for_prompts(topic, verbose=False)
        for source in sources:
            if source['ref'].startswith("Legends of the Jews"):
                total_legends += 1
                other_refs = [s['ref'] for s in sources if not s['ref'].startswith("Legends of the Jews")]
                new_ref, used_one_away = switch_legends(source['ref'], slug, other_refs)
                old_text = Ref(source['ref']).text('en').text
                old_summary = summarize_based_on_uniqueness(old_text, topic_str, llm, "English")
                if new_ref:
                    new_text = Ref(new_ref).text('en').text
                    new_summary = summarize_based_on_uniqueness(new_text, topic_str, llm, "English")
                else:
                    new_summary = ""
                rows.append({
                    "Old Ref": source['ref'],
                    "New Ref": new_ref,
                    "In same cluster?": not used_one_away,
                    "Old Summary": old_summary,
                    "New Summary":  new_summary,
                    "Slug": slug,
                })
                if new_ref:
                    print("FOUND", source['ref'], new_ref)
    with open('output/legends_of_the_jews.csv', 'w') as fout:
        cout = csv.DictWriter(fout, ['Slug', 'Old Ref', 'New Ref', 'In same cluster?', 'Old Summary', 'New Summary'])
        cout.writeheader()
        cout.writerows(rows)
    print("TOTAL LEGENDS", total_legends)


def get_cluster_with_ref(ref, clusters):
    for cluster in clusters:
        for source in cluster.items:
            if source.source.ref == ref:
                return cluster, source


def get_closest_thing(summarized_thing, other_summarized_things):
    thing_embed = embed_text_openai(summarized_thing.summary)
    other_embeds = embed_parallel(other_summarized_things, lambda x: x.summary)
    distances = cosine_distances(np.array([thing_embed]), np.array(other_embeds))[0]
    closest_thing, closest_distance = None, 1
    for other_thing, dist in zip(other_summarized_things, distances):
        if dist < closest_distance:
            closest_thing = other_thing
            closest_distance = dist
    return closest_thing

def get_midrash_from_cluster(cluster):
    options = []
    for source in cluster.items:
        if source.source.categories[0] == 'Midrash' and source.source.book_title['en'] != 'Legends of the Jews':
            options.append(source)
    return options

def switch_legends(ref, slug, other_chosen_trefs):
    try:
        clusters = load_clusters(make_llm_topic(Topic.init(slug)))
    except:
        print("NO SLUG", slug)
        return None, None
    cluster, legends_source = get_cluster_with_ref(ref, clusters)
    if cluster is None:
        print('oh no')
        return None, None
    options = get_midrash_from_cluster(cluster)
    used_one_away = False
    if len(options) == 0:
        closest_cluster = get_closest_thing(cluster, [c for c in clusters if c != cluster])
        options = get_midrash_from_cluster(closest_cluster)
        # remove chosen refs
        options = [o for o in options if o.source.ref not in other_chosen_trefs]
        used_one_away = True
        if len(options) == 0:
            print("OH NO!!!", slug, ref)
            return None, None
    closest_source = get_closest_thing(legends_source, options)
    return closest_source.source.ref, used_one_away


if __name__ == '__main__':
    switch_all_legends_of_the_jews()

