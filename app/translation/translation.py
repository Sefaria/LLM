import random
from util.general import get_by_xml_tag

from basic_langchain.chat_models import ChatAnthropic
from basic_langchain.schema import HumanMessage, SystemMessage
from anthropic import BadRequestError

random.seed(26)


def translate_text(text: str, context: str = None):
    context_prompt = "Context is provided in <context> tags. Use context to provide context to <input> " \
                     "text. Don't translate <context>. Only translate <input> text. "
    identity_message = SystemMessage(content="You are a Jewish scholar knowledgeable in all Torah and Jewish texts. Your "
                                            f"task is to translate the Hebrew text wrapped in <input> tags. {context_prompt if context else ''}Output "
                                            "translation to English wrapped in <translation> tags.")
    task_prompt = f"<input>{text}</input>"
    if context:
        task_prompt = f"<context>{context}</context>{task_prompt}"
    task_message = HumanMessage(content=task_prompt)
    llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
    try:
        response_message = llm([identity_message, task_message])
    except BadRequestError:
        print(f"BadRequestError\n{task_message.content}")
        return ""
    translation = get_by_xml_tag(response_message.content, 'translation')
    if translation is None:
        print("TRANSLATION FAILED")
        print(text)
        print(response_message.content)
        return response_message.content
    return translation


def validate_translation(he, en):
    """
    Doesn't actually work on invalid translations
    :param he:
    :param en:
    :return:
    """
    identity_message = SystemMessage(content="Input is Hebrew text (wrapped in <hebrew> tags) with an English "
                                             "translation (wrapped in <english> tags). Output \"yes\" if the "
                                             "translation is an accurate translation of the Hebrew. Output \"no\" "
                                             "if it is not accurate. Translation is inaccurate if the meaning of any "
                                             "Hebrew word is mistranslated. Output should be wrapped in <answer> tags.")
    task_message = HumanMessage(content=f"<hebrew>{he}</hebrew>\n<english>{en}</english>")
    llm = ChatAnthropic(model="claude-2", temperature=0)
    response_message = llm([identity_message, task_message])
    answer = get_by_xml_tag(response_message.content, 'answer')
    if answer is None:
        print("VALIDATION FAILED")
        print(response_message.content)
        return response_message.content
    return answer == "yes"


# def randomly_translate_book(title: str, n: int = 30):
#     segment_orefs = library.get_index(title).all_segment_refs()
#     # random_segment_orefs = random.sample(segment_orefs, n)
#     rows = []
#     for oref in tqdm(segment_orefs[:16], desc='randomly translating'):
#         tref = oref.normal()
#         rows += [{
#             "Ref": tref,
#             "Hebrew": get_normalized_ref_text(oref, 'he'),
#             "English": translate_segment(tref),
#         }]
#     with open('output/random_mb_translations.csv', 'w') as fout:
#         cout = csv.DictWriter(fout, ['Ref', 'Hebrew', 'English'])
#         cout.writeheader()
#         cout.writerows(rows)
