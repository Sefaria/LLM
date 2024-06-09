from sefaria_llm_interface.topic_prompt import TopicPromptSource
import diff_match_patch
import re
import numpy as np
from typing import Any, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from basic_langchain.schema import SystemMessage, HumanMessage


def get_source_text_with_fallback(source: TopicPromptSource, lang: str, auto_translate=False) -> str:
    text = source.text.get(lang, "")
    other_lang = "en" if lang == "he" else "he"
    other_lang_text = source.text.get(other_lang, "")
    if len(text) == 0:
        if auto_translate and lang == "en":
            from translation.translation import translate_text
            text = translate_text(other_lang_text)
        else:
            text = other_lang_text

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


def get_by_xml_list(text, list_item_tag_name) -> list[str]:
    items = []
    for match in re.finditer(fr"<{list_item_tag_name}>(.*?)</{list_item_tag_name}>", text.replace('\n', ' ')):
        items += [match.group(1)]
    return items


def embedding_distance(embedding1, embedding2):
    # Compute dot product
    dot_product = np.dot(embedding1, embedding2)

    # Compute magnitudes
    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)

    # Compute cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    # Compute cosine distance (1 - cosine_similarity)
    cosine_distance = 1 - cosine_similarity

    return cosine_distance


def run_parallel(items: list[Any], unit_func: Callable, max_workers: int, **tqdm_kwargs) -> list:
    def _pbar_wrapper(pbar, item):
        unit = unit_func(item)
        with pbar.get_lock():
            pbar.update(1)
        return unit


    with tqdm(total=len(items), **tqdm_kwargs) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in items:
                futures.append(executor.submit(_pbar_wrapper, pbar, item))

    output = [future.result() for future in futures if future.result() is not None]
    return output


def summarize_text(text, llm, max_words: int):
    system = SystemMessage(content=f"Given text wrapped in <text> tags, output a summary of text that is no more than "
                                   f"{max_words} words long. Summary should be wrapped in <summary> tags.")
    human = HumanMessage(content=f"<text>{text}</text>")
    response = llm([system, human])
    return get_by_xml_tag(response.content, 'summary')
