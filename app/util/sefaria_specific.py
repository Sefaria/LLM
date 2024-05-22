"""
Utilities that are dependent on Sefaria Project code
"""

import django
django.setup()
from db_manager import MongoProdigyDBManager
from sefaria.model.text import Ref
from sefaria.helper.normalization import NormalizerComposer
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source
from sefaria_llm_interface.topic_prompt import TopicPromptSource


def filter_invalid_refs(trefs, key=None):
    key = key or (lambda x: x)
    out = []
    for tref in trefs:
        try:
            Ref(key(tref))
        except:
            continue
        out += [tref]
    return out


def get_normalizer():
    return NormalizerComposer(['unidecode', 'br-tag', 'itag', 'html', 'maqaf', 'cantillation', 'double-space'])


normalizer = get_normalizer()


def load_mongo_docs(*db_manager_args):
    my_db = MongoProdigyDBManager(*db_manager_args)
    return my_db.output_collection.find({})


def get_raw_ref_text(oref: Ref, lang: str, vtitle=None) -> str:
    return oref.text(lang, vtitle=vtitle).ja().flatten_to_string()


def get_normalized_ref_text(oref: Ref, lang: str, vtitle=None) -> str:
    return normalizer.normalize(get_raw_ref_text(oref, lang, vtitle))


def get_ref_text_with_fallback(oref: Ref, lang: str, auto_translate=False) -> str:
    raw_text = get_raw_ref_text(oref, lang)
    if len(raw_text) == 0:
        if auto_translate and lang == "en":
            from translation.translation import translate_segment
            raw_text = translate_segment(oref.normal())
        else:
            other_lang = "en" if lang == "he" else "he"
            raw_text = get_raw_ref_text(oref, other_lang)

    return normalizer.normalize(raw_text)


def convert_trefs_to_sources(trefs) -> list[TopicPromptSource]:
    return [_make_topic_prompt_source(Ref(tref), '', with_commentary=False) for tref in trefs]
