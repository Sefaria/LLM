import math

from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from collections import defaultdict
import json, csv
import random
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache


set_llm_cache(SQLiteCache(database_path=".langchain.db"))

random.seed(615)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(persist_directory=".chromadb_openai", embedding_function=embeddings)

def get_top_k_keys(d, k):
    return sorted(d, key=d.get, reverse=True)[:k]


def get_keys_above_mean(d, threshold_factor=1.0):
    if not d:
        return []
    values = list(d.values())
    mean_value = sum(values) / len(values)

    # Define the threshold as mean + threshold_factor * standard deviation
    threshold = mean_value + threshold_factor * (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5

    return [key for key, value in d.items() if value > threshold]

# def query(q):
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#     db = Chroma(persist_directory=".chromadb_openai", embedding_function=embeddings)
#     ret = db.similarity_search_with_relevance_scores(q, k=100, score_threshold=0.4)
#     for yo, score in ret:
#         print("-----")
#         print(score)
#         print(yo.page_content)
#         print(yo.metadata['Ref'])
#     print("DB size", len(db.get()["ids"]))

def convert_to_csv(sheet_rows, file_name='output.csv'):
    if not sheet_rows:
        return
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=sheet_rows[0].keys())
        writer.writeheader()
        writer.writerows(sheet_rows)
def slugs_string_to_list(slugs_string):
    return [s for s in slugs_string.split('$') if s]

def get_closest_docs_by_text_similarity(query, k=100, score_threshold=0.0):
    docs = db.similarity_search_with_relevance_scores(
        query.lower(), k=k, score_threshold=score_threshold
    )
    return docs

def get_closest_docs_by_embedding(embedding, k=100, score_threshold=0.3):
    docs = db.similarity_search_by_vector(embedding, k=k, score_threshold=score_threshold)
    return docs

def get_recommended_slugs_frequency_map(docs, ref_to_ignore="$$$"):
    recommended_slugs = defaultdict(lambda: 0)
    for doc, score in docs:
        if doc.metadata['Ref'] == ref_to_ignore:
            continue
        slugs = slugs_string_to_list(doc.metadata['Slugs'])
        for slug in slugs:
            recommended_slugs[slug] += 1
    return recommended_slugs

def get_recommended_slugs_weighted_frequency_map(docs, compute_weight_fn, ref_to_ignore="$$$"):
    recommended_slugs = defaultdict(lambda: 0)
    for doc, score in docs:
        if doc.metadata['Ref'] == ref_to_ignore:
            continue
        slugs = slugs_string_to_list(doc.metadata['Slugs'])
        for slug in slugs:
            recommended_slugs[slug] += compute_weight_fn(score)*1
    return recommended_slugs

def get_recall_of_pair(data_slugs, inferred_slugs):
    return len(set(data_slugs).intersection(inferred_slugs))

def euclidean_relevance_to_l2(euclidean_relevance_score):
    return math.sqrt(2)*(1-euclidean_relevance_score)

def l2_to_cosine_similarity(l2_distance):
    if 1 - (l2_distance ** 2) / 2 <0:
        print("Cosine Similarity:",  1 - (l2_distance ** 2) / 2)
    return 1 - (l2_distance ** 2) / 2

def cosine_to_one_minus_sine(cos_theta):
    sin_theta = math.sqrt(1 - cos_theta**2)
    # print("One Minus Sine:", 1 - sin_theta)
    return 1 - sin_theta

def cosine_to_linear(cos_theta):
    x = math.acos(cos_theta)
    y = -(math.pi / 4) * x + 1
    return y
def eval():
    with open("topic_modelling_training_set.json", 'r') as file:
        items = json.load(file)
    items_sample = random.sample(items, 100)

    num_of_slug_hits = 0
    num_of_all_data_slugs = 0

    sheet_rows = []


    for item in items_sample:
        # if "Kol Bo 46:1" not in item["Ref"]:
        #     continue
        docs = get_closest_docs_by_text_similarity(item["English"], 500)
        recommended_slugs_cosine = get_recommended_slugs_weighted_frequency_map(docs, lambda score: l2_to_cosine_similarity(euclidean_relevance_to_l2(score))
                                                                         ,item["Ref"])
        recommended_slugs_sine = get_recommended_slugs_weighted_frequency_map(docs, lambda score: (cosine_to_one_minus_sine(l2_to_cosine_similarity(euclidean_relevance_to_l2(score))))**10
                                                                         ,item["Ref"])
        recommended_slugs_linear = get_recommended_slugs_weighted_frequency_map(docs, lambda score: (cosine_to_linear(l2_to_cosine_similarity(euclidean_relevance_to_l2(score))))
                                                                                ,item["Ref"])

        best_slugs_cosine = get_keys_above_mean(recommended_slugs_cosine, 2.5)
        best_slugs_sine = get_keys_above_mean(recommended_slugs_sine, 2.5)

        best_slugs_linear = get_keys_above_mean(recommended_slugs_linear, 2.5)

        best_slugs_naive = get_keys_above_mean(get_recommended_slugs_frequency_map(docs, item["Ref"]),2.5)

        num_of_all_data_slugs += len(item["Slugs"])
        # num_of_slug_hits += get_recall_of_pair(item["Slugs"], best_slugs)
        # print("Ref:", item["Ref"])
        # print("Text:", item["English"])
        # print("Slugs from Data:", item["Slugs"])
        # print("Recommended Slugs:", best_slugs)

        sheet_rows.append(
            {
                "Ref": item["Ref"],
                "Text": item["English"],
                "Slugs from Data": item["Slugs"],
                "Recommended Slugs Cosine": best_slugs_cosine,
                "Recommended Slugs Sine": best_slugs_sine,
                "In Sine not in Cosine": set(best_slugs_sine).difference(set(best_slugs_cosine)),
                "In Cosine not in Sine": set(best_slugs_cosine).difference(set(best_slugs_sine))
            }
        )
    convert_to_csv(sheet_rows)
    # print("Recall: ", num_of_slug_hits/num_of_all_data_slugs)






if __name__ == '__main__':
    # query('כורש מכונה "משיח" (המשוח) בספר ישעיהו (ישעיהו 45:1), מונח שבדרך כלל שמור למלכים ולכוהנים יהודים, מדגיש את תפקידו הייחודי בהיסטוריה היהודית.')
    query = """
Rebbe Yitzchak said, \"Come and see when the holy one blessed be he bought down water he first brought it down in mercy to show the world that if they repent he will accept them. This is implied when it first says 'There was rain' and later it says 'there was the flood' that if they repent it will be rain of blessing and if they do not repent it will be a flood.\
"""

    # ref_to_ignore = "Zohar Chadash, Noach 81"
    # suggested_slugs = defaultdict(lambda: 0)
    # docs = get_closest_docs(query)
    # for doc, score in docs:
    #     print("-----")
    #     if doc.metadata['Ref'] == ref_to_ignore:
    #         continue
    #     print(score)
    #     print(doc.metadata['Ref'])
    #     slugs = slugs_string_to_list(doc.metadata['Slugs'])
    #     for slug in slugs:
    #         suggested_slugs[slug] += 1
    # print(suggested_slugs)
    eval()