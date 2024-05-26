from srsly import read_jsonl, write_jsonl
en_export_filename = "/Users/nss/Downloads/en_library.jsonl"
he_export_filename = "/Users/nss/Downloads/he_library.jsonl"
he_only_export_filename = "/Users/nss/Downloads/he_only_library.jsonl"
import django
django.setup()
from sefaria.model import *
from langchain.docstore.document import Document
from tqdm import tqdm

def make_doc_from_export_item(export_item):
    return Document(page_content=export_item['text'], metadata=export_item['metadata'])

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
            docs += [make_doc_from_export_item({
                "text": passage_text,
                "metadata": {
                    **passage_export_items[0]["metadata"],
                    "ref": oref.normal(),
                    "url":  f"https://www.sefaria.org/{oref.url()}",
                    "pagerank": max(item['metadata']['pagerank'] for item in passage_export_items)
                }
            })]
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


if __name__ == '__main__':
    # docs = get_passages_to_ingest()
    # print(docs[0])
    # print(len(docs))
    # get_hebrew_that_has_no_english_to_ingest()
    # count_tokens_in_export(read_jsonl(he_only_export_filename))
    print_random_segs()
