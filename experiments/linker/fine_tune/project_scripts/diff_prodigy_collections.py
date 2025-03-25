"""
Takes two prodigy collections as input and outputs only the documents that are different.
"""
import argparse
from functools import reduce
from util.sefaria_specific import load_mongo_docs
from db_manager import MongoProdigyDBManager
from pymongo import InsertOne


def hash_doc(mongo_doc, include_spans=True):
    return hash((
        mongo_doc['text'],
        mongo_doc['meta']['Ref'],
        tuple(hash_span(span) for span in sorted(mongo_doc['spans'], key=lambda span: span['start'])) if include_spans else None
    ))


def hash_span(span):
    return hash((span['start'], span['end'], span['label']))


def get_num_spans_in_common(orig_doc, mod_doc):
    hashed_orig = {hash_span(span) for span in orig_doc['spans']}
    hashed_mod = {hash_span(span) for span in mod_doc['spans']}
    return len(hashed_mod & hashed_orig)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('original_collection')
    parser.add_argument('modified_collection')
    parser.add_argument('output_collection')
    return parser


def convert_en_labels_to_he(doc):
    for span in doc['spans']:
        if span['label'] == 'Citation':
            span['label'] = 'מקור'
        elif span['label'] == 'Person':
            span['label'] = 'בן-אדם'
        else:
            raise Exception("Unknown label: {}".format(span['label']))
    return doc


def diff_prodigy_collections(original_collection, modified_collection, output_collection):
    original_docs = list(load_mongo_docs(original_collection))
    modified_docs = list(load_mongo_docs(modified_collection))

    hashed_original = {hash_doc(d): d for d in original_docs}
    hashed_original_wo_spans = {hash_doc(d, include_spans=False): d for d in original_docs}
    diff_docs = []
    num_common_spans = 0
    total_spans = reduce(lambda a, b: a + len(b['spans']), original_docs, 0)
    for modified_doc in modified_docs:
        modified_hash = hash_doc(modified_doc)
        if modified_hash not in hashed_original:
            diff_docs += [modified_doc]
            modified_hash_wo_spans = hash_doc(modified_doc, include_spans=False)
            try:
                num_common_spans += get_num_spans_in_common(hashed_original_wo_spans[modified_hash_wo_spans], modified_doc)
            except KeyError:
                print('key error')
        else:
            # docs are the same
            num_common_spans += get_num_spans_in_common(modified_doc, modified_doc)

    my_db = MongoProdigyDBManager(output_collection)
    my_db.output_collection.delete_many({})
    my_db.output_collection.bulk_write([InsertOne(convert_en_labels_to_he(d)) for d in diff_docs])
    print("Accuracy:", num_common_spans/total_spans)


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    diff_prodigy_collections(args.original_collection, args.modified_collection, args.output_collection)
