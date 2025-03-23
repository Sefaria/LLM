"""
functions for running CNN linker
"""
import django
django.setup()
from sefaria.model import *
from sefaria.model.linker.linker import LinkedDoc
from sefaria.model.linker.ref_part import span_inds, span_char_inds
from db_manager import MongoProdigyDBManager
from util.sefaria_specific import load_mongo_docs
from spacy.tokens import Doc


def linker_doc_to_mongo_doc(linker_doc: LinkedDoc, orig_mongo_doc: dict) -> dict:
    spans = []
    for resolved in linker_doc.all_resolved:
        entity = resolved.raw_entity
        start_char, end_char = entity.char_indices
        token_start, token_end = span_inds(entity.span)
        spans.append({
            'start': start_char,
            'end': end_char,
            'token_start': token_start,
            'token_end': token_end,
            'label': entity.type.value.capitalize()
        })
    return {
        **orig_mongo_doc,
        'spans': spans,
    }


def spacy_doc_to_mongo_doc(spacy_doc: Doc, orig_mongo_doc: dict) -> dict:
    spans = []
    for span in spacy_doc.ents:
        start_char, end_char = span_char_inds(span)
        token_start, token_end = span_inds(span)
        spans.append({
            'start': start_char,
            'end': end_char,
            'token_start': token_start,
            'token_end': token_end,
            'label': span.label_,
        })
    return {
        **orig_mongo_doc,
        'spans': spans,
    }


def run_linker_on_collection(input_collection, output_collection):
    my_db = MongoProdigyDBManager(output_collection)
    my_db.output_collection.delete_many({})
    mongo_docs = list(load_mongo_docs(input_collection))
    linker: Linker = library.get_linker('en')
    linker_docs = linker.bulk_link([d['text'] for d in mongo_docs], verbose=True)
    new_mongo_docs = [linker_doc_to_mongo_doc(linker_doc, mongo_doc) for linker_doc, mongo_doc in zip(linker_docs, mongo_docs)]
    my_db.output_collection.insert_many(new_mongo_docs)


def run_ref_part_model_on_collection(input_collection, output_collection):
    my_db = MongoProdigyDBManager(output_collection)
    my_db.output_collection.delete_many({})
    mongo_docs = list(load_mongo_docs(input_collection))
    linker: Linker = library.get_linker('en')
    ner = linker.get_ner().raw_ref_part_model
    spacy_docs = []
    for doc in ner.pipe([d['text'] for d in mongo_docs], batch_size=100):
        spacy_docs.append(doc)
    new_mongo_docs = [spacy_doc_to_mongo_doc(spacy_doc, mongo_doc) for spacy_doc, mongo_doc in zip(spacy_docs, mongo_docs)]
    my_db.output_collection.insert_many(new_mongo_docs)


if __name__ == '__main__':
    # run_linker_on_collection('ner_en_gpt_copper_web', 'ner_en_gpt_copper_web_cnn')
    run_ref_part_model_on_collection('ner_en_gpt_copper_combo_sub_citation', 'ner_en_gpt_copper_combo_cnn_sub_citation')
