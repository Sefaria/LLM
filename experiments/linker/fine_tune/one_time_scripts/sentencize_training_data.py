from db_manager import MongoProdigyDBManager
from util.sefaria_specific import load_mongo_docs
from util.sentencizer import sentencize, claude_sentencizer
from util.general import run_parallel
from functools import partial
from tqdm import tqdm

import logging

# Set the logging level for 'httpx' to WARNING
httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)


def sentencize_by_type(text, sentencizer_type='claude'):
    if sentencizer_type == 'claude':
        sents = claude_sentencizer(text)
    else:
        sents = sentencize(text)
    return sents


def sentencize_training_data(input_collection, output_collection, skip, total, sentencizer_type):
    my_db = MongoProdigyDBManager(output_collection)
    my_db.output_collection.delete_many({})
    docs = load_mongo_docs(input_collection)
    docs= list(docs)[skip:skip+total]
    unit_func = partial(sentencize_by_type, sentencizer_type=sentencizer_type)
    all_sents = run_parallel([d['text'] for d in docs], unit_func, 25, desc='Sentencizing')
    for doc, sents in zip(docs, all_sents):
        for sent in sents:
            if len(sent) < 20: continue
            my_db.output_collection.insert_one({"text": sent, "spans": [], "meta": doc['meta']})


if __name__ == '__main__':
    sentencize_training_data('ner_he_input', 'ner_he_input_broken', 21500, 500, 'claude')
