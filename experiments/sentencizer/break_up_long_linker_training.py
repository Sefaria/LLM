from util.sentencizer import claude_sentencizer
import django
django.setup()
from sefaria.helper.linker import load_spacy_model
from spacy import Language
from thefuzz import fuzz
from functools import reduce
from db_manager import MongoProdigyDBManager
from tqdm import tqdm
from sefaria.model.linker.ref_part import span_inds, span_char_inds
from util.general import run_parallel


def load_mongo_docs(min_training_text_len, unique_by_metadata=True, *db_manager_args) -> list[dict]:
    print(db_manager_args)
    my_db = MongoProdigyDBManager(*db_manager_args)
    data = [d for d in my_db.output_collection.find({}) if len(d['text']) > min_training_text_len]
    # make data unique
    if unique_by_metadata:
        data = list({(tuple(sorted(d['meta'].items(), key=lambda x: x[0])), d['text']): d for d in data}.values())
    return data


def break_up_docs(docs: list[dict], nlp: Language) -> list[dict]:
    broken_up_docs: list[list[dict]] = run_parallel(docs, lambda doc: break_up_doc(doc, nlp), max_workers=50, desc="Breaking up")
    return reduce(lambda x, y: x + y, broken_up_docs)


def break_up_doc(doc: dict, nlp: Language) -> list[dict]:
    text = doc['text']
    thresh = 800
    if len(text) <= thresh: return [doc]
    sentences = claude_sentencizer(text)
    new_docs = []
    curr_start = 0
    spacy_doc = nlp(text)
    curr_end_index = 0
    for sentence in sentences:
        sent_search_start = max(curr_end_index-10, 0)
        sent_index, sent_score = best_substring_match_index(text[sent_search_start:], sentence)
        curr_end_index = sent_index + len(sentence) + sent_search_start
        # guarantee span ends on a token
        try:
            potential_span = spacy_doc.char_span(curr_start, curr_end_index, alignment_mode='expand')
        except:
            continue
        _, curr_end_index = span_char_inds(potential_span)
        potential_sub_text = text[curr_start:curr_end_index]
        if len(potential_sub_text) > thresh/2:
            # create sub doc
            new_docs.append(create_sub_doc(potential_sub_text, doc['meta']['Ref']))
            curr_start = curr_end_index
    final_text = text[curr_start:]
    if len(final_text) >= 3:
        new_docs.append(create_sub_doc(final_text, doc['meta']['Ref']))
    # add spans
    curr_token = 0
    spans_taken = 0
    for sub_doc in new_docs:
        end_token = curr_token + len(nlp(sub_doc['text']))
        spacy_sub_doc = nlp(sub_doc['text'])
        prev_span_end = 0
        for span in sorted(doc['spans'], key=lambda span: span['start']):
            if span['token_start'] >= curr_token and span['token_end'] < end_token:
                span_text = text[span['start']: span['end']]
                span_start = sub_doc['text'][prev_span_end:].index(span_text) + prev_span_end
                span_end = span_start + len(span_text)
                spacy_span = spacy_sub_doc.char_span(span_start, span_end)
                span_token_start, span_token_end = span_inds(spacy_span)
                sub_doc['spans'].append({
                    "start": span_start,
                    "end": span_end,
                    "token_start": span_token_start,
                    "token_end": span_token_end,
                    "label": span['label'],
                })
                spans_taken += 1
                prev_span_end = span_end
        curr_token = end_token
    if spans_taken < len(doc['spans']):
        print("FEWER SPANS WERE TAKEN!", doc['meta']['Ref'])

    return new_docs


def create_sub_doc(text, ref):
    return {
        "meta": {"Ref": ref},
        "text": text.strip(),
        "spans": [],
    }


def best_substring_match_index(long_string, short_string):
    best_score = -1
    best_index = -1

    for i in range(len(long_string) - len(short_string) + 1):
        substring = long_string[i:i+len(short_string)]
        score = fuzz.ratio(short_string, substring)

        if score > best_score:
            best_score = score
            best_index = i

    return best_index, best_score


def run():
    nlp = load_spacy_model("/Users/nss/sefaria/models/ref_he")
    docs = load_mongo_docs(0, False, 'ner_he_input')
    docs = docs[2000:3000]
    docs = break_up_docs(docs, nlp)
    my_db = MongoProdigyDBManager('ner_he_input_broken')
    my_db.output_collection.delete_many({})
    my_db.output_collection.insert_many(docs)


if __name__ == '__main__':
    run()

