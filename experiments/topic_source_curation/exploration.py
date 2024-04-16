"""
Functions to explore existing topic curations
"""
from topic_source_curation.common import get_datasets
from collections import defaultdict
import django
django.setup()
from sefaria.model import *


if __name__ == '__main__':
    bad, good = get_datasets()
    counts = defaultdict(int)
    for example in good:
        for source in example.sources:
            oref = Ref(source.ref)
            index = oref.index
            if index.get_primary_corpus():
                counts[index.get_primary_corpus()] += 1
            elif len(getattr(index, 'authors', [])) == 1:
                counts[index.authors[0]] += 1
            else:
                counts[index.title] += 1
    for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(k, v)

"""
quick takeaways:
- sources
    - Tanakh, Mishnah, Midrash Rabbah, Bavli, Yerushalmi, MT, SA, Siddurim, Rashi, Zohar
    - Pirkei DeRabbi Eliezer, Tanchuma, Ibn Ezra, Ramban, Mekhilta
- Acharonim that give major insights
    - maharl
    - ramchal
    - Shnei Luchot Habrit
    - Nachman of Breslov
    - levi-yitzchak-of-berditchev
    - tzadok-hakohen-of-lublin
- Modern english
    - erica brown
    - Peninei Halacha
    - Eliezer Berkovitz

This is a fairly comprehensive list of everything that comes up 3 times or more
"""


