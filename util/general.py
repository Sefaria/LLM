import django
django.setup()
from db_manager import MongoProdigyDBManager
from sefaria.model.text import Ref
from sefaria.helper.normalization import NormalizerComposer
import diff_match_patch


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


def get_removal_list(orig, new):
    """
    Calculate list of text removed (or added) in order to convert `orig` into `new`
    :param orig: original string
    :param new: new string. assumption is it is quite similar to `orig`
    :return: list where each element is ((start_char, end_char), text_added).
    If text was deleted, `text_added` will be '' and the range will cover the range of text deleted.
    If text was added, the range will be zero-length
    """
    checker = diff_match_patch.diff_match_patch()
    diff = checker.diff_main(orig, new)
    removal_list = []
    curr_start_char = 0
    for diff_type, diff_text in diff:
        if diff_type == 0:
            curr_start_char += len(diff_text)
        elif diff_type == 1:
            removal_list += [((curr_start_char, curr_start_char), diff_text)]
            curr_start_char += len(diff_text)
        elif diff_type == -1:
            removal_list += [((curr_start_char, curr_start_char + len(diff_text)), '')]
    removal_list.sort(key=lambda x: (x[0][0], (x[0][1]-x[0][0])))
    return removal_list
