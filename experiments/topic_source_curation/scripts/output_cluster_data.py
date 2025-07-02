import django
django.setup()
from sefaria.model import *
import os
import json
import re


def get_directory_content(directory, glob):
    return [f"{directory}/{filename}" for filename in os.listdir(directory) if re.match(glob, filename)]


def run():
    # loop through cluster json files and strip out irrelevant data
    all_clusters = {}
    cluster_glob = r"clusters_(.*)\.json"
    for cluster_file in get_directory_content("_cache", cluster_glob):
        with open(cluster_file, "r") as fin:
            data = json.load(fin)
        for cluster in data:
            cluster.pop('embeddings', None)
            for source in cluster['items']:
                source['ref'] = source['source']['ref']
                source.pop('source', None)
                source.pop('embedding', None)
        all_clusters[re.sub(cluster_glob, r'\1', cluster_file).replace('_cache/', '')] = data
    with open("output/all_clusters.json", "w") as fout:
        json.dump(all_clusters, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()
