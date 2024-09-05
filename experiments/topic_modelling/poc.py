from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from collections import defaultdict
import json, csv
import random
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

random.seed(614)
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

def eval():
    with open("topic_modelling_training_set.json", 'r') as file:
        items = json.load(file)
    items_sample = random.sample(items, 20)

    sheet_rows = []


    for item in items_sample:
        if "Haamek Davar on Exodus 13:4:1" not in item["Ref"]:
            continue
        docs = get_closest_docs_by_text_similarity(item["English"], 1000)
        recommended_slugs = get_recommended_slugs_frequency_map(docs, item["Ref"])
        best_slugs = get_keys_above_mean(recommended_slugs, 2.0)
        print("Ref:", item["Ref"])
        print("Text:", item["English"])
        print("Slugs from Data:", item["Slugs"])
        print("Recommended Slugs:", best_slugs)
        sheet_rows.append(
            {
                "Ref": item["Ref"],
                "Text": item["English"],
                "Slugs from Data": item["Slugs"],
                "Recommended Slugs:": best_slugs
            }
        )
    convert_to_csv(sheet_rows)






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