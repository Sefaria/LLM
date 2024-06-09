from collections import defaultdict
import json

from srsly import read_jsonl, write_jsonl
en_export_filename = "/Users/nss/Downloads/en_library.jsonl"
he_export_filename = "/Users/nss/Downloads/he_library.jsonl"
he_only_export_filename = "/Users/nss/Downloads/he_only_library.jsonl"
import django
django.setup()
from functools import reduce
from sefaria.model import *
from langchain.docstore.document import Document
from tqdm import tqdm

def make_doc_from_export_item(export_item):
    return Document(page_content=export_item['text'], metadata=export_item['metadata'])

def get_sefat_emet_to_ingest():
    export = read_jsonl(en_export_filename)
    docs = []
    for item in export:
        if item['metadata']['ref'].startswith('Sefat Emet'):
            new_meta = {}
            for k, v in item['metadata'].items():
                if isinstance(v, list):
                    v = v[0] if len(v) > 0 else ""
                if v is None:
                    v = ""
                new_meta[k] = v
            item['metadata'] = new_meta
            docs += [item]
    write_jsonl("/Users/nss/Downloads/sefat_emet_to_ingest.jsonl", docs)

def get_passages_to_ingest():
    export = read_jsonl(en_export_filename)
    tref_to_export_item = {}
    for item in export:
        meta = item["metadata"]
        tref_to_export_item[(meta['ref'], meta['versionTitle'], meta['lang'])] = item
    docs = []
    passages = PassageSet({})
    for passage in tqdm(passages, total=len(passages)):
        oref = Ref(passage.full_ref)
        for version in oref.versionset():
            if version.actualLanguage != "en": continue
            try:
                passage_export_items = [tref_to_export_item[(seg_oref.normal(), version.versionTitle, version.language)] for seg_oref in oref.range_list()]
            except KeyError:
                print("Skipping", oref.normal(), version.versionTitle, version.language)
                continue
            passage_text = " ".join(item['text'] for item in passage_export_items).strip()
            docs += [{
                "text": passage_text,
                "metadata": {
                    **passage_export_items[0]["metadata"],
                    "ref": oref.normal(),
                    "url":  f"https://www.sefaria.org/{oref.url()}",
                    "pagerank": max(item['metadata']['pagerank'] for item in passage_export_items),
                    "docType": "section",
                    "associatedTopicIDs": list(set(reduce(lambda x, y: x + y, [item['metadata']['associatedTopicIDs'] for item in passage_export_items]))),
                    "associatedTopicNames": list(set(reduce(lambda x, y: x + y, [item['metadata']['associatedTopicNames'] for item in passage_export_items])))
                }
            }]
    return docs


def get_hebrew_that_has_no_english_to_ingest():
    en_export = read_jsonl(en_export_filename)
    he_export = read_jsonl(he_export_filename)
    he_only_export = []
    en_refs = set()
    for item in en_export:
        en_refs.add(item['metadata']['ref'])
    for item in he_export:
        if item['metadata']['ref'] in en_refs:
            continue
        he_only_export += [item]
    write_jsonl(he_only_export_filename, he_only_export)

def count_tokens_in_export(export):
    from util.openai_utils import count_tokens_openai
    count = 0
    for item in tqdm(list(export)):
        count += count_tokens_openai(item['text'])
    print('Num Tokens', count)


def print_random_segs():
    import random
    export = list(read_jsonl(he_only_export_filename))
    export = export[:len(export)//4]
    sample = random.sample(export, 1000)
    for item in sample:
        print(item['metadata']['ref'])


def get_metadata_value_ranges():
    export = read_jsonl(en_export_filename)
    fields = ['docType', 'primaryDocCategory', 'authorIDs', 'authorNames', 'dataOrigin', 'compositionPlace', 'associatedTopicIDs', 'associatedTopicNames', 'isTranslation']
    unique_values = defaultdict(set)
    for item in export:
        for field in fields:
            value = item['metadata'][field]
            if isinstance(value, list):
                unique_values[field] |= set(value)
            else:
                unique_values[field].add(value)
    out = {field: list(values) for field, values in unique_values.items()}
    with open('output/metadata_ranges.json', 'w') as fout:
        json.dump(out, fout, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # docs = get_passages_to_ingest()
    # write_jsonl("/Users/nss/Downloads/passages_to_ingest.jsonl", docs)
    # print(docs[0])
    # print(len(docs))
    # get_hebrew_that_has_no_english_to_ingest()
    # count_tokens_in_export(read_jsonl(he_only_export_filename))
    # print_random_segs()
    # get_sefat_emet_to_ingest()
    get_metadata_value_ranges()
