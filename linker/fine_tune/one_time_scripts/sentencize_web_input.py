from util.general import load_mongo_docs
from util.sentencizer import sentencize
collection = "ner_en_web_input"
from db_manager import MongoProdigyDBManager
from pymongo import InsertOne
from tqdm import tqdm


def sentencize_web_input():
    my_db = MongoProdigyDBManager("ner_en_sent_web_input")
    my_db.output_collection.delete_many({})
    docs = list(load_mongo_docs(collection))
    sent_docs = []
    for doc in tqdm(docs):
        sents = sentencize(doc['text'])
        del doc['_id']
        for sent in sents:
            doc_copy = doc.copy()
            doc_copy['text'] = sent
            sent_docs += [doc_copy]
    my_db.output_collection.bulk_write([InsertOne(d) for d in sent_docs])

def output_sample():
    docs = load_mongo_docs("ner_en_sent_web_input")
    for doc in docs:
        print(doc['text'])
        print()


if __name__ == '__main__':
    # sentencize_web_input()
    output_sample()

