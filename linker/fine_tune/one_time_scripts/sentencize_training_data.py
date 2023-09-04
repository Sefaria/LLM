from db_manager import MongoProdigyDBManager
from util.general import load_mongo_docs
from util.sentencizer import sentencize
from tqdm import tqdm


def segmentize_and_save(my_db, text, meta):
    sents = sentencize(text)
    for sent in sents:
        my_db.output_collection.insert_one({"text": sent, "spans": [], "meta": meta})


if __name__ == '__main__':
    my_db = MongoProdigyDBManager('ner_en_sent_input2')
    my_db.output_collection.delete_many({})
    docs = load_mongo_docs('ner_en_web_input')
    for doc in tqdm(list(docs)):
        segmentize_and_save(my_db, doc['text'], doc['meta'])
