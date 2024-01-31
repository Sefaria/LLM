"""
Utilities that are dependent on Sefaria Project code
"""

import django
django.setup()
from db_manager import MongoProdigyDBManager
from sefaria.model.text import Ref
from sefaria.helper.normalization import NormalizerComposer


def get_normalizer():
    return NormalizerComposer(['unidecode', 'br-tag', 'itag', 'html', 'maqaf', 'cantillation', 'double-space'])


normalizer = get_normalizer()


def load_mongo_docs(*db_manager_args):
    my_db = MongoProdigyDBManager(*db_manager_args)
    return my_db.output_collection.find({})


def get_raw_ref_text(oref: Ref, lang: str) -> str:
    return oref.text(lang).ja().flatten_to_string()


def get_normalized_ref_text(oref: Ref, lang: str) -> str:
    return normalizer.normalize(get_raw_ref_text(oref, lang))


def get_ref_text_with_fallback(oref: Ref, lang: str, auto_translate=False) -> str:
    raw_text = get_raw_ref_text(oref, lang)
    if len(raw_text) == 0:
        if auto_translate and lang == "en":
            from translation.poc import translate_segment
            raw_text = translate_segment(oref.normal())
        else:
            other_lang = "en" if lang == "he" else "he"
            raw_text = get_raw_ref_text(oref, other_lang)

    return normalizer.normalize(raw_text)


