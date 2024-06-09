from bs4 import BeautifulSoup


def get_trefs_from_page(dir, slug):
    with open(f"{dir}/clusters_and_chosen_sources_{slug}.html", "r") as fin:
        soup = BeautifulSoup(fin.read(), "lxml")
        for link in soup.select('body > details.clusterSource a'):
            yield link.get_text()


def compare_clusterings(dir1, dir2, slug):
    trefs1 = set(get_trefs_from_page(dir1, slug))
    trefs2 = set(get_trefs_from_page(dir2, slug))
    print("in 1 but not 2")
    for tref in (trefs1 - trefs2):
        print('\t', tref)
    print("in 2 but not 1")
    for tref in (trefs2 - trefs1):
        print('\t', tref)


def compare_all_clusterings(dir1, dir2, slug_list):
    for slug in slug_list:
        compare_clusterings(dir1, dir2, slug)


if __name__ == '__main__':
    compare_all_clusterings(
        "/Users/nss/Downloads/old clustering",
        "/Users/nss/sefaria/llm/experiments/topic_source_curation/output",
        ["maaser-sheni"]
    )


"""

Moriah
    Old: Good: 2 Bad: 1
    New: Good: 6 Bad: 1
beyond-the-letter-of-the-law
    Old: Good: 2 Bad: Neutral: 4
    New: Good: 4 Bad: 1 Neurtral: 1
maaser-sheni
    Old: Good: 3 Bad: 1 Neutral: 1
    New: Good: 3 Bad:  Neurtral: 2 
"""
