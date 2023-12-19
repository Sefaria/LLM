import csv
import django
django.setup()
# We can do the same thing with a SQLite cache
# from langchain.cache import SQLiteCache
# set_llm_cache(SQLiteCache(database_path=".langchain.db"))
# import typer
import re
import json
from sefaria.model import *
# from sefaria.utils.hebrew import strip_cantillation
# import random
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
import  numpy as np

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
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, request_timeout=120)
    text = query
    query_result = embeddings.embed_query(text)

    return query_result
def format_string(fstring, arguments):
    try:
        formatted_string = fstring.format(*arguments)
        return formatted_string
    except IndexError:
        print("Error: Number of arguments does not match placeholders in the string.")
    except KeyError:
        print("Error: Invalid placeholder in the string.")
    except Exception as e:
        print(f"Error: {e}")

def query_llm_model(template_fstirng, arguments_list):
    formatted_string = template_fstirng.format(*arguments_list)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=.5, request_timeout=120)
    user_prompt = PromptTemplate.from_template("# Input\n{text}")
    human_message = HumanMessage(content=user_prompt.format(text=formatted_string))
    answer = llm([human_message])

    return answer.content

def topic_slugs_list_from_csv(slugs_path_csv):
    return [row[0] for row in csv.reader(open(slugs_path_csv))]
def topic_list_from_slugs_csv(slugs_path_csv):
    slugs = topic_slugs_list_from_csv(slugs_path_csv)
    topics = []
    for slug in slugs:
        topics += [TopicSet({"slug":slug})[0]]
    return topics

def slugs_and_description_dict_from_csv():
    return dict((row[0], row[1]) for row in csv.reader(open('slugs_and_inferred_descriptions.csv')))

def list_of_tuples_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)
def _get_top_n_orefs_for_topic(slug, top_n=10):
    from sefaria.helper.topic import get_topic

    out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
    result = []
    for d in out['refs']['about']['refs'][:top_n]:
        try:
            result.append(Ref(d['ref']))
        except Exception as e:
            print (e)
    return result
def _get_first_k_categorically_distinct_refs(refs, k):
    distinct_categories = set()
    result = []

    for ref in refs:
        category = ref.primary_category
        if category is not None and category not in distinct_categories:
            distinct_categories.add(category)
            result.append(ref)

            if len(result) == k:
                break

    return result
def _concatenate_passages(passages, separetaor_token):
    result = ""
    for i, passage in enumerate(passages, 1):
        result += f"{separetaor_token} {i}:\n{passage}\n"

    return result
def _get_relevant_example_passages_from_slug(slug, n=3, discard_longest = 0):
    refs = _get_top_n_orefs_for_topic(slug, 200)
    refs = _get_first_k_categorically_distinct_refs(refs, n)
    def _get_length_of_ref(ref):
        return len(ref.text().text)
    for i in range(0, discard_longest):
        longest_element = max(refs, key=_get_length_of_ref)
        refs.remove(longest_element)
    text = _concatenate_passages([ref.tref + ': ' + str(ref.text().text) for ref in refs], "Passage")

    return text

def infer_topic_descriptions_to_csv(slugs_csv_path, output_csv_path):
    template = """You are a humanities scholar specializing in Judaism. Given a Topic or a term and a list of passages that relate to that topic, write a description for that topic from a Jewish perspective.
    Don't quote the passages in your answer, don't summarize the passages, but rather explain the general concept to the Topic.
    Topic: {0}
    Related Passages: {1}
    Description:
    """


    topic_list = topic_list_from_slugs_csv(slugs_csv_path)
    slugXdescription_list = []
    # for topic in topic_list:
    #     print(topic.get_primary_title())
    #     des = query_llm_model(template, topic.get_primary_title())
    #     slug = topic.slug
    #     slugXdescription_list += [(slug, des)]
    for topic in topic_list:
        print(topic.get_primary_title())
        slug = topic.slug
        # if slug != "peace":
        #     continue
        passages = _get_relevant_example_passages_from_slug(slug)
        try:
            des = query_llm_model(template, [topic.get_primary_title(), passages])
        except:
            try:
                passages = _get_relevant_example_passages_from_slug(slug, discard_longest=1)
                des = query_llm_model(template, [topic.get_primary_title(), passages])
            except:
                try:
                    passages = _get_relevant_example_passages_from_slug(slug, discard_longest=2)
                    des = query_llm_model(template, [topic.get_primary_title(), passages])
                except:
                    des = query_llm_model(template, [topic.get_primary_title(), ''])



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

def cluster_slugs(slugs_and_embeddings_jsonl):
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

    data = [json.loads(line) for line in open(slugs_and_embeddings_jsonl, 'r')]


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

def ask_llm_for_topics_from_segment(segment_text):
    infer_topics_template = ("You are a humanities scholar specializing in Judaism. "
                             + "Given the following text segment, generate an unumbered list of relevant short topics or tags separated by commas."
                             + "Topics can also be general conepts and creative, and don't have to appear in the text itself. If texts talks about a general concept that does not apear explicitly in the text, I still consider that as a valid topic for the text.\n"
                            #  +"Output must be a simple string of topics separated by commas\n"
                            #  +"Example Output:\n"
                            # +"First Topic, Second Topic, Third Topic\n\n"

                             +"The Text: {0}")
    embed_topic_template = """You are a humanities scholar specializing in Judaism. Given a topic or a term, write a description for that topic from a Jewish perspective.
    Topic: {0}
    Description:
    """
    answer = query_llm_model(infer_topics_template, [segment_text])
    # model_topics = [topic.strip() for topic in answer.split(",")]
    model_topics = [topic.strip() for topic in answer.split("\n-")]
    def embedding_distance(embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)
    existing_topics_dicts = [json.loads(line) for line in open("description_with_sources_prompts_embeddings.jsonl", 'r')]
    for topic_dict in existing_topics_dicts:
        topic_dict["embedding"] = np.array(topic_dict["embedding"])

    for topic in model_topics:
        inferred_topic_embedding = np.array(embed_text(query_llm_model(embed_topic_template, [topic])))
        sorted_data = sorted(existing_topics_dicts, key=lambda x: embedding_distance(x["embedding"], inferred_topic_embedding))
        print(f"Gpt tagged passage with the topic: {topic}, which is similar to Sefaria's topic: {sorted_data[0]['slug']}")

        slug_desc_dict = slugs_and_description_dict_from_csv()
        verifier_template = """
            "You are a humanities scholar specializing in Judaism. Given a passage, a topic and a description of that topic, return YES if the passage can be tagged with this topic. note: even if the topic is not mentioned explicitly in the passage, but the passage refers to ghe general concept of the topic, the passage can be tagged with that topic.
             if it's not a good topic tagging for the passage, return NO . if you are unsure, return NO .
             Passage: {0}
             Possible Topic: {1}
             Topic Description: {2}
        """
        nearest_slug = sorted_data[0]['slug']
        description = slug_desc_dict[nearest_slug]
        ver = query_llm_model(verifier_template, [segment_text, nearest_slug, description]).replace('# Output', '').strip()
        print(f"Verification: {ver}")
    print(model_topics)

class TopicVerifier:
    verifier_template = (
        "You are a humanities scholar specializing in Judaism. Given a passage, a topic, and a description of that topic, return YES if the passage can be tagged with this topic. "
        "Note: Even if the topic is not mentioned explicitly in the passage, but the passage refers to the general concept of the topic, or the general concept coud be found within the passage, return YES."
        "If it's not a good topic tagging for the passage, return NO. If you are unsure, return NO.\n"
        "Passage: {0}\nPossible Topic: {1}\nTopic Description: {2}"
    )

    def __init__(self, slugs_descriptions_csv):
        self.slug_descriptions_dict = {row[0]: row[1] for row in csv.reader(open(slugs_descriptions_csv, 'r'))}

    def verify_topic(self, sefaria_slug, segment_text):
        description = self.slug_descriptions_dict[sefaria_slug]
        ver_response = query_llm_model(self.verifier_template, [segment_text, sefaria_slug, description]).replace('# Output',
                                                                                                    '').strip()
        ver_approved = False
        if "YES" in ver_response:
            ver_approved = True
        if "NO" in ver_response:
            ver_approved = False
        return ver_approved

class TopicsVectorSpace:

    embed_topic_template = (
        "You are a humanities scholar specializing in Judaism. Given a topic or a term, write a description for that topic from a Jewish perspective.\n"
        "Topic: {0}\nDescription:"
    )

    def __init__(self, slugs_embeddings_jsonl):
        slug_embeddings_list_of_dicts = [json.loads(line) for line in
                                         open(slugs_embeddings_jsonl, 'r')]
        slug_embeddings_dict = {}
        for topic_dict in slug_embeddings_list_of_dicts:
            slug_embeddings_dict[topic_dict["slug"]] = np.array(topic_dict["embedding"])
        self.slug_embeddings_dict = slug_embeddings_dict

    def _embed_and_get_embedding(self, template, topic):
        return np.array(embed_text(query_llm_model(template, [topic])))
    def _embedding_distance(self, embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)
    def get_nearest_topic(self, slug):
        inferred_topic_embedding = self._embed_and_get_embedding(self.embed_topic_template, slug)
        sefaria_topic_embeddings_list = [(key, value) for key, value in self.slug_embeddings_dict.items()]
        sorted_data = sorted(sefaria_topic_embeddings_list,
                             key=lambda x: self._embedding_distance(x[1], inferred_topic_embedding))
        sefaria_slug = sorted_data[0][0]
        # print(f"GPT tagged passage with the topic: {inferred_topic}, which is similar to Sefaria's topic: {sefaria_slug}")
        return sefaria_slug

class TopicTagger:
    # Class attribute
    species = "Canis familiaris"
    # Template for inferring topics from the given text segment
    infer_topics_template = (
        "You are a humanities scholar specializing in Judaism. "
        "Given the following text segment, generate an unnumbered list of relevant short topics or tags. "
        "Topics can also be general concepts and creative, and don't have to appear in the text itself. "
        "Try to infer the 'theme' of the segment, what it tries to teach us, and generate topics accordingly"
        "If the text talks about a general concept that does not appear explicitly in the text, I still consider that as a valid topic for the text.\n"
        "The Text: {0}"
    )


    def __init__(self, topics_space: TopicsVectorSpace, verifier: TopicVerifier):
        self.verifier = verifier
        self.topics_space = topics_space


    def _get_inferred_topics(self, template, segment_text):
        answer = query_llm_model(template, [segment_text])
        return [topic.strip() for topic in answer.split("\n-")]

    def _load_existing_topics(self, jsonl_path):
        existing_topics_dicts = [json.loads(line) for line in
                                 open(jsonl_path, 'r')]
        for topic_dict in existing_topics_dicts:
            topic_dict["embedding"] = np.array(topic_dict["embedding"])
        return existing_topics_dicts

    def _embed_and_get_embedding(self, template, topic):
        return np.array(embed_text(query_llm_model(template, [topic])))

    def _get_Sefaria_nearest_topic(self, inferred_topic):
        inferred_topic_embedding = self._embed_and_get_embedding(self.embed_topic_template, inferred_topic)
        sefaria_topic_embeddings_list = [(key, value) for key, value in self.slug_embeddings_dict.items()]
        sorted_data = sorted(sefaria_topic_embeddings_list,
                             key=lambda x: self._embedding_distance(x[1], inferred_topic_embedding))
        sefaria_slug = sorted_data[0][0]
        # print(f"GPT tagged passage with the topic: {inferred_topic}, which is similar to Sefaria's topic: {sefaria_slug}")
        return sefaria_slug


    def tag_segment(self, segment_text):
        model_topics = self._get_inferred_topics(self.infer_topics_template, segment_text)
        verified_slugs = set()

        for inferred_topic in model_topics:
            sefaria_slug = self.topics_space.get_nearest_topic(inferred_topic)
            verified = self.verifier.verify_topic(sefaria_slug, segment_text)
            print(f"LLM Tag: {inferred_topic}")
            print(f"Sefaria Nearest Slug: {sefaria_slug}")
            if verified:
                # print("https://www.sefaria.org/topics/" + sefaria_slug)
                print("LLM Verification: Accept")
                verified_slugs.add(sefaria_slug)
            else:
                print("LLM Verification: Reject")

        return model_topics, set(verified_slugs)



if __name__ == '__main__':
    print("Hi")
    # get_embeddings()
    # infer_topic_descriptions_to_csv("n_topic_slugs.csv",  "slugs_and_inferred_descriptions_prompt_with_sources.csv")
    # embed_topic_descriptions_to_jsonl("slugs_and_inferred_descriptions_prompt_with_sources.csv", "description_with_sources_prompts_embeddings.jsonl")
    # cluster_slugs("description_embeddings.jsonl")
    # ask_llm_for_topics_from_segment(Ref("Sanhedrin.99a.2").text().text)
    verifier = TopicVerifier(slugs_descriptions_csv="slugs_and_inferred_descriptions_prompt_with_sources.csv")
    topics_space = TopicsVectorSpace(slugs_embeddings_jsonl="description_embeddings.jsonl")
    tagger = TopicTagger(topics_space=topics_space, verifier=verifier)
    # tagger.tag_segment(Ref("Bereshit_Rabbah.62.2").text().text)
    refs_for_presentation =[
        # "Shabbat.53b.17-18",
        # "Pirkei_Avot.3.17",
        # "Megillah.29a.4",
        # "Chagigah.5b.15",
        # "Berakhot.61a.27-61b.3",
        # "Sanhedrin.59b.13",
        # "Midrash Tanchuma, Beshalach 10:4-6",
        # "Sifra, Kedoshim, Chapter 4.12",
        # "Sefer_HaChinukh.132.2",
        # "Rabbeinu_Bahya, Shemot 21 19.1",
        # "Sanhedrin.38b.11-12",
        # "Berakhot.42a.9",
        # "Kohelet_Rabbah.1.7.1"

        # "Kohelet_Rabbah.3.11.1",
        # "Mishneh Torah, Human_Dispositions.6.6-7",
        # "Berakhot.34b.22",
        # "Sifra, Emor, Chapter 1.6",
        # "Yevamot 62b:13",
        # "Bava Metzia 85b:4-7",
        # "Yoma 82a:1-7",
        # "Sanhedrin.21b.20"

        # "Vayikra Rabbah 34:4",
        # "Bava Metzia 31b:7",
        # "Shabbat 28a:4",
        # "Berakhot 55a:13",
        # "Ramban on Genesis 5:1:1",
        # "Radak on Genesis 6:6:3",
        # "Sanhedrin 38a:12",
        # "Mesilat Yesharim 23:18-22"
        "Yevamot 49b:10"
    ]
    result_tuples = []
    for ref in refs_for_presentation:
        print(f"Ref: {Ref(ref).normal()}")
        model_topics, verified_slugs = tagger.tag_segment(Ref(ref).text().text)
        print(verified_slugs)
    #     result_tuples.append((Ref(ref).normal(), Ref(ref).text().text, str(model_topics), str(verified_slugs)))
    # with open("results.csv", 'w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerows(result_tuples)









    print("bye")