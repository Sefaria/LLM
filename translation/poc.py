import django
django.setup()
import typer
import csv
from tqdm import tqdm
import random
from sefaria.model import *
from util.general import get_normalized_ref_text, get_by_xml_tag

import langchain
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

random.seed(26)


def translate_segment(tref: str, context: str = None):
    oref = Ref(tref)
    text = get_normalized_ref_text(oref, 'he')
    context_prompt = "Context is provided in <context> tags. Use context to provide context to <input> " \
                     "text. Don't translate <context>. Only translate <input> text. "
    identity_message = SystemMessage(content="You are a Jewish scholar knowledgeable in all Torah and Jewish texts. Your "
                                            f"task is to translate the Hebrew text wrapped in <input> tags. {context_prompt if context else ''}Output "
                                            "translation wrapped in <translation> tags.")
    task_prompt = f"<input>{text}</input>"
    if context:
        task_prompt = f"<context>{context}</context>{task_prompt}"
    task_message = HumanMessage(content=task_prompt)
    llm = ChatAnthropic(model="claude-2", temperature=0, max_tokens_to_sample=1000000)
    response_message = llm([identity_message, task_message])
    translation = get_by_xml_tag(response_message.content, 'translation')
    if translation is None:
        print("TRANSLATION FAILED")
        print(tref)
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
    llm = ChatAnthropic(model="claude-2", temperature=0, max_tokens_to_sample=1000000)
    response_message = llm([identity_message, task_message])
    answer = get_by_xml_tag(response_message.content, 'answer')
    if answer is None:
        print("VALIDATION FAILED")
        print(response_message.content)
        return response_message.content
    return answer == "yes"


def randomly_translate_book(title: str, n: int = 30):
    segment_orefs = library.get_index(title).all_segment_refs()
    # random_segment_orefs = random.sample(segment_orefs, n)
    rows = []
    for oref in tqdm(segment_orefs[:16], desc='randomly translating'):
        tref = oref.normal()
        rows += [{
            "Ref": tref,
            "Hebrew": get_normalized_ref_text(oref, 'he'),
            "English": translate_segment(tref),
        }]
    with open('output/random_mb_translations.csv', 'w') as fout:
        cout = csv.DictWriter(fout, ['Ref', 'Hebrew', 'English'])
        cout.writeheader()
        cout.writerows(rows)


if __name__ == '__main__':
    # typer.run(randomly_translate_book)
    sa = """ מי שיצא (אם) מוציא אחרים. ובו ג סעיפים: על כל פירות ושאר דברים חוץ מפת ויין אם היו האוכלים שנים או יותר אחד פוטר את חבירו אפי' בלא הסיבה ומיהו ישיבה מיהא בעי דדוקא פת ויין בעי הסיבה ולדידן הוי ישיבה כמו הסיבה לדידהו ולפי זה לדידן דלית לן הסיבה אין חילוק בין פת ויין לשאר דברים דבישיבה אפילו פת ויין אחד מברך לכולם ושלא בישיבה בשאר דברים נמי כל אחד מברך לעצמו. והא דאמרינן דאחד מברך לכולם בשאר דברים חוץ מן הפת ה"מ בברכה ראשונה אבל בברכה אחרונה צריכין ליחלק וכל אחד מברך לעצמו דאין זימון לפירות: הגה וי"א דבכל הדברים חוץ מפת ויין לא מהני הסיבה וה"ה ישיבה לדידן (ב"י סי' קע"ד בשם הראב"ד) ולכן נהגו עכשיו בפירות שכ"א מברך לעצמו: """
    print(translate_segment("Mishnah Berurah 213:12", context=sa))

