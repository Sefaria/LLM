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


def get_ref_text_with_fallback(oref: Ref, lang: str) -> str:
    raw_text = get_raw_ref_text(oref, lang)
    if len(raw_text) == 0:
        other_lang = "en" if lang == "he" else "he"
        raw_text = get_raw_ref_text(oref, other_lang)

    return normalizer.normalize(raw_text)
