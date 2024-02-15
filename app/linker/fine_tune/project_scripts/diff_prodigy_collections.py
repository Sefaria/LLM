"""
Takes two prodigy collections as input and outputs only the documents that are different.
"""
import argparse
from functools import reduce
from app.util.sefaria_specific import load_mongo_docs
from db_manager import MongoProdigyDBManager
from pymongo import InsertOne


def hash_doc(mongo_doc, include_spans=True):
    return hash((
        mongo_doc['text'],
        mongo_doc['meta']['Ref'],
        tuple(hash_span(span) for span in mongo_doc['spans']) if include_spans else None
    ))


def hash_span(span):
    return hash((span['start'], span['end'], span['label']))


def get_num_diff_spans(orig_doc, mod_doc):
    hashed_orig = {hash_span(span) for span in orig_doc['spans']}
    hashed_mod = {hash_span(span) for span in mod_doc['spans']}
    return len(hashed_mod - hashed_orig)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('original_collection')
    parser.add_argument('modified_collection')
    parser.add_argument('output_collection')
    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    original_docs = list(load_mongo_docs(args.original_collection))
    modified_docs = list(load_mongo_docs(args.modified_collection))

    hashed_original = {hash_doc(d): d for d in original_docs}
    hashed_original_wo_spans = {hash_doc(d, include_spans=False): d for d in original_docs}
    diff_docs = []
    num_diff_spans = 0
    total_spans = reduce(lambda a, b: a + len(b['spans']), original_docs, 0)
    for modified_doc in modified_docs:
        modified_hash = hash_doc(modified_doc)
        if modified_hash not in hashed_original:
            diff_docs += [modified_doc]
            modified_hash_wo_spans = hash_doc(modified_doc, include_spans=False)
            num_diff_spans += get_num_diff_spans(hashed_original_wo_spans[modified_hash_wo_spans], modified_doc)

    my_db = MongoProdigyDBManager(args.output_collection)
    my_db.output_collection.bulk_write([InsertOne(d) for d in diff_docs])
    print("Accuracy:", (total_spans-num_diff_spans)/total_spans)

