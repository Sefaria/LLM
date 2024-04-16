import django
django.setup()
from sefaria.model import *
from translation.translation import translate_text
from util.sefaria_specific import get_normalized_ref_text
from tqdm import tqdm
import csv

def translate_book(title):
    fout = open(f"output/{title}_translation.csv", "w")
    cout = csv.DictWriter(fout, ['ref', 'hebrew', 'english'])
    cout.writeheader()
    index = library.get_index(title)
    assert isinstance(index, Index)
    for segment_oref in tqdm(index.all_segment_refs(), desc=f"Translate {title}"):
        segment_text = get_normalized_ref_text(segment_oref, "he")
        translation = translate_text(segment_text)
        cout.writerow({"ref": segment_oref.normal(), "hebrew": segment_text, "english": translation})
    fout.close()

def translate_books(titles):
    for title in titles:
        translate_book(title)


if __name__ == '__main__':
    translate_books(["Noam Elimelech", "Tanna Debei Eliyahu Rabbah", "Tanna debei Eliyahu Zuta", "Ba'al Shem Tov"])