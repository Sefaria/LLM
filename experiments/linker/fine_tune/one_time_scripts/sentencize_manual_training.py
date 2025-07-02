import csv
import re
from db_manager import MongoProdigyDBManager
from util.sentencizer import sentencize


def get_texts():
    texts = []
    with open("/Users/nss/Downloads/English Linker Refs to Train on - To train text.csv", "r") as fin:
        cin = csv.reader(fin)
        for row in cin:
            text = row[0].strip()
            for sentence in sentencize(text):
                texts.append(sentence)
    final_texts = []
    roman_reg = "[xivlc]+"
    prev_text = None
    for text in texts:
        if prev_text is not None and re.search(fr"\s{roman_reg}\.\s*$", prev_text):
            print(prev_text)
            text = prev_text + " " + text
        if re.search(fr"\s{roman_reg}\.\s*$", text):
            prev_text = text
            continue
        if re.search(fr"^{roman_reg}", text):
            print(text)
            final_texts[-1] += " " + text
            prev_text = text
            continue
        final_texts.append(text)
        prev_text = text
    return final_texts


if __name__ == '__main__':
    texts = get_texts()
    db = MongoProdigyDBManager('manual_en_web_input', 'localhost', 27017)
    for i, t in enumerate(texts):
        doc = {"text": t, "spans": [], "meta": {"id": i}}
        db.output_collection.insert_one(doc)