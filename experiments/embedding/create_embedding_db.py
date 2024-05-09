from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from time import sleep
from langchain_text_splitters import CharacterTextSplitter
from srsly import read_jsonl
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
from voyageai.error import RateLimitError
import os
import json
import django
django.setup()
from sefaria.model import *
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

export_filename = "/Users/nss/Downloads/en_library.jsonl"

def add_herzog_tanakh_passages():
    PassageSet({"type": {"$in": ["Herzog Level 2", "Herzog Level 3"]}}).delete()
    herzog_filenames = ["embedding/output/herzog_level_2_wo_overlaps.json", "embedding/output/herzog_level_3_wo_overlaps.json"]
    level_nums = [2, 3]
    for herzog_filename, level_num in zip(herzog_filenames, level_nums):
        with open(herzog_filename, "r") as fin:
            passages = json.load(fin)
            for passage in passages:
                oref = Ref(passage["b_ref"]).to(Ref(passage["e_ref"]))
                Passage({
                    "full_ref": oref.normal(),
                    "ref_list": oref.range_list(),
                    "type": f"Herzog Level {level_num}"
                }).save()




def ingest_document(doc, db, pbar):
    try:
        db.add_documents([doc])
    except RateLimitError:
        sleep(60)
        ingest_document(doc, db, pbar)
        return

    with pbar.get_lock():
        pbar.update(1)

def make_doc_from_export_item(export_item):
    return Document(page_content=export_item['text'], metadata=export_item['metadata'])

def make_metadata_from_ref(oref: Ref, version: Version, pr: float):
    return {
        "url": f"https://www.sefaria.org/{oref.url()}",
        "ref": oref.normal(), "versionTitle": version.versionTitle, "lang": version.actualLanguage,
        "docCategory": version.get_index().get_primary_category(),
        'dataQuality': 'user' if version.versionTitle == "Sefaria Community Translation" else 'professional',
        'pagerank': pr,
    }

def ingest_passages():
    ref_pr_map = {ref_data.ref: ref_data.pagesheetrank for ref_data in RefDataSet()}
    docs = []
    passage_set = PassageSet()
    for passage in tqdm(passage_set, desc='Creating passages', total=len(passage_set)):
        oref = Ref(passage.full_ref)
        chunk = oref.text('en')
        passage_text = re.sub('<[^>]+>', '', TextChunk.strip_itags(chunk.as_string()))
        try:
            version = chunk.version()
        except Exception as e:
            for sub_ref in oref.range_list():
                try:
                    version = sub_ref.text('en').version()
                except:
                    continue
        try:
            pr = sum(ref_pr_map[sub_ref.normal()] for sub_ref in oref.range_list())/len(oref.range_list())
        except Exception as e:
            print(e)
            continue
        metadata = make_metadata_from_ref(oref, version, pr)
        doc = Document(page_content=passage_text, metadata=metadata)
        docs.append(doc)
    ingest_docs(docs)

def ingest_export(start):
    export = read_jsonl(export_filename)
    docs = [make_doc_from_export_item(item) for item in export]
    ingest_docs(docs, start)


def ingest_docs(docs, start=0):
    embeddings = VoyageAIEmbeddings(model="voyage-large-2-instruct", batch_size=32)
    chroma_db = Chroma(persist_directory="embedding/.chromadb", embedding_function=embeddings)
    # chroma_db = Chroma.from_documents(
    #     documents=[docs[0]], embedding=embeddings, persist_directory="embedding/.chromadb"
    # )

    with tqdm(total=len(docs)-1, desc="Ingesting documents") as pbar:
        pbar.update(start)
        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
            futures = []
            for d in docs[start:]:
                future = executor.submit(ingest_document, d, chroma_db, pbar)
                futures.append(future)
            for future in futures:
                future.result()

def query():
    query = "Abraham our forefather"
    embeddings = VoyageAIEmbeddings(model="voyage-large-2-instruct", batch_size=32)
    db = Chroma(persist_directory="embedding/.chromadb", embedding_function=embeddings)
    # ret = db.similarity_search_with_relevance_scores(query)
    # for yo in ret:
    #     print(yo)
    # print("DB size", len(db.get()["ids"]))

if __name__ == '__main__':
    # ingest_export(855836)
    ingest_passages()
    # query()
    # add_herzog_tanakh_passages()
