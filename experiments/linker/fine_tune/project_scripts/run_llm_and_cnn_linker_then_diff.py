"""
A script to run the LLM and CNN linker and then compare the results.
"""
from experiments.linker.fine_tune.project_scripts.run_llm_linker import run_llm_linker_on_mongo
from experiments.linker.fine_tune.project_scripts.run_cnn_linker import run_linker_on_collection
from experiments.linker.fine_tune.project_scripts.diff_prodigy_collections import diff_prodigy_collections
from pymongo import MongoClient


def convert_group_to_person(collection_name):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["prodigy"]
    collection = db[collection_name]

    # run the update
    result = collection.update_many(
        { "spans.label": "Group" },
        { "$set": { "spans.$[elem].label": "Person" } },
        array_filters=[ { "elem.label": "Group" } ]
    )
    print("Matched {} documents and modified {} documents.".format(result.matched_count, result.modified_count))


if __name__ == '__main__':
    lang = 'en'
    input_collection = f'ner_{lang}_input_broken'
    output_llm_collection = f'ner_{lang}_gpt_copper'
    output_cnn_collection = f'ner_{lang}_cnn_copper'
    output_diff_collection = f'ner_{lang}_diff'
    run_llm_linker_on_mongo(input_collection, output_llm_collection)
    run_linker_on_collection(input_collection, output_cnn_collection, lang)
    convert_group_to_person(output_cnn_collection)
    diff_prodigy_collections(output_cnn_collection, output_llm_collection, output_diff_collection, lang)
