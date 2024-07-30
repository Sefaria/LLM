import django
django.setup()
from sefaria.model import *
import csv
from collections import defaultdict


def run():
    links = RefTopicLinkSet({"descriptions.en.ai_title": {"$exists": True}, "descriptions.en.prompt": {"$exists": True}})
    book_counts = defaultdict(int)
    author_counts = defaultdict(int)
    ngrams = defaultdict(lambda: defaultdict(int))
    print(len(links))
    for link in links:
        link.ref = link.ref.replace("&amp;", "&").replace("&quot;", '"')
        oref = Ref(link.ref)
        book = oref.index.title
        authors = oref.index.author_objects()
        if len(authors) > 0:
            author_counts[authors[0].slug] += 1
        book_counts[book] += 1
        for ngram_size in range(1, 5):
            for field in ["title", "prompt"]:
                text = link.descriptions['en'][field]
                words = text.split()
                for jword in range(len(words)-ngram_size+1):
                    ngram = words[jword:jword+ngram_size]
                    ngrams[ngram_size][" ".join(ngram)] += 1
    for ngram_size in range(1, 5):
        with open("../output/ab_counts_ngram_{}.csv".format(ngram_size), "w") as fin:
            cin = csv.DictWriter(fin, fieldnames=["ngram", "count"])
            cin.writeheader()
            rows = [{"ngram": ngram, "count": count} for ngram, count in sorted(ngrams[ngram_size].items(), key=lambda item: item[1], reverse=True)]
            cin.writerows(rows[:2000])

    with open("../output/book_counts.csv", "w") as fin:
        cin = csv.DictWriter(fin, ['book', 'count'])
        cin.writeheader()
        cin.writerows({"book": book, "count": count} for book, count in sorted(book_counts.items(), key=lambda item: item[1], reverse=True))

    with open("../output/authors_counts.csv", "w") as fin:
        cin = csv.DictWriter(fin, ['author', 'count'])
        cin.writeheader()
        cin.writerows({"author": author, "count": count} for author, count in sorted(author_counts.items(), key=lambda item: item[1], reverse=True))



if __name__ == '__main__':
    run()

