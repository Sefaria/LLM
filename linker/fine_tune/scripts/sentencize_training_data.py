from segmentizer.segmentizer import segmentize
from util.general import load_mongo_docs


def run(text):
    sents = segmentize(text)
    for sent in sents:
        print(sent)
        print("")
        print("%%#$#$")
        print("")


if __name__ == '__main__':
    docs = load_mongo_docs('ner_en_input2')
    for doc in docs:
        print(doc['text'])
        run(doc['text'])
        break