from srsly import read_jsonl
export_filename = "/Users/nss/Downloads/en_library.jsonl"
import django
django.setup()
from sefaria.model import *
from langchain.docstore.document import Document
from tqdm import tqdm

def make_doc_from_export_item(export_item):
    return Document(page_content=export_item['text'], metadata=export_item['metadata'])

def get_passages_to_ingest():
    export = read_jsonl(export_filename)
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

if __name__ == '__main__':
    docs = get_passages_to_ingest()
    print(docs[0])
    print(len(docs))
