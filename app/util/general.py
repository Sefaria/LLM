from sefaria_llm_interface.topic_prompt import TopicPromptSource
import diff_match_patch
import re


def get_source_text_with_fallback(source: TopicPromptSource, lang: str, auto_translate=False) -> str:
    text = source.text.get(lang, "")
    if len(text) == 0:
        if auto_translate and lang == "en":
            from app.translation.poc import translate_text
            text = translate_text(text)
        else:
            other_lang = "en" if lang == "he" else "he"
            text = source.text.get(other_lang, "")

    return text


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


def get_by_xml_tag(text, tag_name) -> str:
    match = re.search(fr'<{tag_name}>(.+?)</{tag_name}>', text, re.DOTALL)
    if not match:
        return None
    return match.group(1)

