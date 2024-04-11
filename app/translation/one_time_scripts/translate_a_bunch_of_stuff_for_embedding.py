import django
django.setup()
from sefaria.model import *
from translation.poc import translate_text
from util.sefaria_specific import get_normalized_ref_text
from tqdm import tqdm

def translate_book(title):
    index = library.get_index(title)
    assert isinstance(index, Index)
    for segment_oref in tqdm(index.all_segment_refs(), desc=f"Translate {title}"):
        segment_text = get_normalized_ref_text(segment_oref, "he")
        translation = translate_text(segment_text)
        print(segment_oref.normal())
        print(translation)


if __name__ == '__main__':
    translate_book("Netivot Olam")