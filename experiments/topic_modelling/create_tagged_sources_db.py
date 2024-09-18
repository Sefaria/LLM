import json
# from util.general import run_parallel
from experiments.embedding.create_embedding_db import ingest_docs
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
import csv


def get_all_slugs_in_data(filename="topic_modelling_training_set.json"):
    with open(filename, 'r') as file:
        items = json.load(file)
    all_slugs_set = set()
    for item in items:
        slugs = set(item["Slugs"])
        all_slugs_set = all_slugs_set.union(slugs)
    with open("all_slugs_in_training_set.csv", mode="w", newline="") as file: csv.writer(file).writerows([[item] for item in all_slugs_set])


if __name__ == '__main__':
    get_all_slugs_in_data()

    # with open("topic_modelling_training_set.json", 'r') as file:
    #     items = json.load(file)
    # formatted_items = []
    #
    # ##turn list of slugs into one string for chromadb
    # for item in items:
    #     slugs = item["Slugs"]
    #     slugs_with_token = [slug + "$" for slug in slugs]
    #     new_slugs = "".join(slugs_with_token)
    #     item["Slugs"] = new_slugs
    #
    # for item in items:
    #     formatted = {
    #         'text': item['English'],
    #         'metadata': {key: value for key, value in item.items() if key!='English'}
    #                  }
    #     formatted_items.append(formatted)
    #
    # docs = []
    # for item in formatted_items:
    #     docs.append(Document(page_content=item['text'], metadata=item['metadata']))
    # # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # # chroma_db = Chroma.from_documents(
    # #     documents=[docs[0]], embedding=embeddings, persist_directory=".chromadb_openai"
    # # )
    # # for doc in docs:
    # #     chroma_db.add_documents([doc])
    # ingest_docs(docs)