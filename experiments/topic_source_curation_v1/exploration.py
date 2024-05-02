"""
Functions to explore existing topic curations
"""
from experiments.topic_source_curation_v1.common import get_datasets
from collections import defaultdict
import django
django.setup()
from sefaria.model import *
from tqdm import tqdm


if __name__ == '__main__':
    bad, good = get_datasets()
    counts = defaultdict(list)
    for example in tqdm(good):
        for source in example.sources:
            oref = Ref(source.ref)
            index = oref.index
            he_versions = VersionSet({"title": index.title, "actualLanguage": "he"}).array()
            if len(he_versions) == 0:
                continue
            default_he = he_versions[0]
            en_versions = VersionSet({"title": index.title, "actualLanguage": "en"}).array()
            if len(en_versions) == 0:
                continue
            default_en = en_versions[0]
            he_wc = default_he.word_count()
            en_wc = default_en.word_count()
            if en_wc/he_wc > 0.5:
                # print(f"Skipping {index.title}. {default_en.versionTitle}")
                continue
            print(index.title)
            if index.get_primary_corpus():
                counts[index.get_primary_corpus()] += [index]
            elif len(getattr(index, 'authors', [])) == 1:
                counts[index.authors[0]] += [index]
            else:
                counts[index.title] += [index]
    for k, v in sorted(counts.items(), key=lambda x: len(x[1]), reverse=True):
        print("-----")
        print(k)
        for vv in v:
            print('\t', vv)


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


