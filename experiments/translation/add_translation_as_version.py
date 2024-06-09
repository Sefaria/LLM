import django
django.setup()
from sefaria.model import *
from sefaria.tracker import modify_bulk_text
from sefaria.utils.util import get_directory_content
import csv
import re
from tqdm import tqdm

TRANSLATION_DIR = "one_time_scripts/output"

def add_translation_as_version(title, text_map):
    index = Index().load({"title": title})
    vtitle = "Claude v3 Opus Translation"
    if Version().load({"title": title, "versionTitle": vtitle}) is not None:
        print("Skipping version", title)
        return
    version = Version(
        {
            "chapter": index.nodes.create_skeleton(),
            "versionTitle": vtitle,
            "versionSource": "https://www.anthropic.com/",
            "language": "en",
            "actualLanguage": "en",
            "title": index.title
        }
    )
    modify_bulk_text(5842, version, text_map, skip_links=True)


def get_title_from_filename(filename):
    return re.search(r"/([^/]+)_translation\.csv$", filename).group(1)


def add_all_translation_files_as_versions():
    for filename in tqdm(get_directory_content(TRANSLATION_DIR)):
        with open(filename, "r") as fin:
            title = get_title_from_filename(filename)
            cin = csv.DictReader(fin)
            text_map = {row['ref']: row['english'] for row in cin}
            print(title)
            try:
                add_translation_as_version(title, text_map)
            except AttributeError:
                print("Attribute Error", title)
                continue


def count_auto_translated_words():
    vs = VersionSet({"versionTitle": "Claude v3 Opus Translation"})
    for version in vs:
        if version.title.startswith("Rashi"):
            continue
        print(version.title)
    print(vs.word_count())


def count_voyage_tokens():
    from transformers import AutoTokenizer
    from srsly import read_jsonl
    tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    export = read_jsonl('/Users/nss/Downloads/en_library.jsonl')
    count = 0
    for entry in tqdm(export):
        count += len(tokenizer.tokenize(entry['text']))
    print(count)




if __name__ == '__main__':
    add_all_translation_files_as_versions()
    count_auto_translated_words()
    # count_voyage_tokens()
