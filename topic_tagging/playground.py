import csv
import django
django.setup()
# import typer
import json
from sefaria.model import *
# from sefaria.utils.hebrew import strip_cantillation
# import random
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
# from langchain.chat_models import ChatOpenAI
# import openai
# import re
# from sefaria.helper.normalization import NormalizerComposer, RegexNormalizer, AbstractNormalizer
# from util.general import get_removal_list

api_key = os.getenv("OPENAI_API_KEY")

def get_top_topics_slugs():
    # Read topics to promote from file
    topics_slugs = []
    with open('good_to_promote_topics.txt', 'r') as file:
        lines = file.readlines()
        topics_slugs = [line.replace("',\n", '')[2:].lower().replace(" ", '-') for line in lines]

    # Retrieve main topics list
    main_topics_list = [TopicSet({"slug": slug}).array()[0] for slug in topics_slugs]

    # Sort topics based on numSources
    key_function = lambda topic: getattr(topic, 'numSources', 0)
    top_topics = sorted(main_topics_list, key=key_function, reverse=True)[:100]

    # Write top topic slugs to CSV
    csv_file = "n_topic_slugs.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[topic.slug] for topic in top_topics])


def embed_text(query):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    text = query
    query_result = embeddings.embed_query(text)

    return query_result

def query_llm_model(template, text):
    llm = OpenAI(temperature=.7)

    prompt_template = PromptTemplate(input_variables=["text"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = answer_chain.run(text)
    return answer

def topic_slugs_list_from_csv(slugs_path_csv):
    return [row[0] for row in csv.reader(open(slugs_path_csv))]
def topic_list_from_slugs_csv(slugs_path_csv):
    slugs = topic_slugs_list_from_csv(slugs_path_csv)
    topics = []
    for slug in slugs:
        topics += [TopicSet({"slug":slug})[0]]
    return topics

def list_of_tuples_to_csv(data, file_path):
    """
    Parameters:
    - data (list of tuples): The data to be written to the CSV file.
    - file_path (str): The path to the CSV file.
    Returns:
    - None
    """
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)
def infer_topic_descriptions_to_csv(slugs_csv_path, query_template, output_csv_path):

    topic_list = topic_list_from_slugs_csv(slugs_csv_path)
    slugXdescription_list = []
    for topic in topic_list:
        print(topic.get_primary_title())
        des = query_llm_model(query_template, topic.get_primary_title())
        slug = topic.slug
        slugXdescription_list += [(slug, des)]

    list_of_tuples_to_csv(slugXdescription_list, output_csv_path)

def embed_topic_descriptions_to_jsonl(slugs_and_descriptions_csv, output_jsonl_path):
    list_of_embeddings = []
    slugXdes_list = [(row[0], row[1]) for row in csv.reader(open(slugs_and_descriptions_csv))]
    for slugXdes in slugXdes_list:
        slug = slugXdes[0]
        des = slugXdes[1]
        list_of_embeddings += [{"slug": slug, "embedding":embed_text(des)}]
    with open(output_jsonl_path, 'w') as jsonl_file:
        for item in list_of_embeddings:
            jsonl_file.write(json.dumps(item) + '\n')

def cluster_slugs():
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    import matplotlib.pyplot as plt

    # Assuming you have a list of dicts with slugs and embeddings
    data = [
        {"slug": "slug1", "embedding": np.array([0.1, 0.2, 0.3])},
        {"slug": "slug2", "embedding": np.array([0.4, 0.5, 0.6])},
        # ... more data ...
    ]

    data = [json.loads(line) for line in open('description_embeddings.jsonl', 'r')]


    # Extract embeddings into a NumPy array
    embeddings = np.array([item["embedding"] for item in data])

    # Choose the number of clusters (you might need to tune this)
    num_clusters = 5

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Assign each data point to its nearest cluster
    closest_points, _ = pairwise_distances_argmin_min(embeddings, kmeans.cluster_centers_)

    # Create a dictionary to store the clusters
    clusters = {i: [] for i in range(num_clusters)}

    # Assign each data point to its cluster in the dictionary
    for i, point_index in enumerate(closest_points):
        clusters[cluster_labels[i]].append(data[i]["slug"])

    # Print the clusters
    for cluster_label, cluster_slugs in clusters.items():
        print(f"Cluster {cluster_label + 1}: {cluster_slugs}")

    # Plot the clusters
    plt.figure(figsize=(24, 16))

    for i in range(num_clusters):
        cluster_points = embeddings[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

    # Plot the cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red',
                label='Centroids')

    # Annotate points with slugs
    for i, txt in enumerate([item["slug"] for item in data]):
        plt.annotate(txt, (embeddings[i, 0], embeddings[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    print("Hi")
    # get_embeddings()
    # template = """You are a humanities scholar specializing in Judaism. Given a topic or a term, write a description for that topic from a Jewish perspective.
    # Topic: {text}
    # Description:
    # """
    # infer_topic_descriptions_to_csv("n_topic_slugs.csv", template, "slugs_and_inferred_descriptions.csv")
    # embed_topic_descriptions_to_jsonl("slugs_and_inferred_descriptions.csv", "description_embeddings.jsonl")
    cluster_slugs()



    print("bye")