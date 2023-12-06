"""
When generating many prompts for one topic page,
GPT may repeat many phrases
This file tries to differentiate the writing style
"""


from util.general import get_by_xml_tag
import langchain
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def differentiate_writing(sentence, phrase_to_avoid):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_message = SystemMessage(content="Given a target sentence (wrapped in <target> tags) and a phrase to avoid "
                                           "using (wrapped in <phrase_to_avoid> tags), rewrite <target> so it doesn't "
                                           "use <phrase_to_avoid> but maintains the same meaning, tone and terminology."
                                           " Change as few words as possible to achieve this goal."
                                           " <example>\n<target>Abraham, patriarch of the Jewish people, emerged from a world of idolatry and established a covenant with God.</target><phrase_to_avoid>Abraham, patriarch of the Jewish people</phrase_to_avoid><output>The Jewish people's patriarch, Abraham, emerged from a world of idolatry and established a covenant with God.</output></example>"
                                           " Wrap output in <output> tags.")
    human_message = HumanMessage(content=f"<target>{sentence}</target>\n<phrase_to_avoid>{phrase_to_avoid}</phrase_to_avoid>")
    response = llm([system_message, human_message])
    output = get_by_xml_tag(response.content, 'output')
    if not output:
        return response.content
    return output


def repeated_phrase(sentence, comparison_sentences: list[str]):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_message = SystemMessage(content="Given a target sentence (wrapped in <target> tags) and a list of "
                                           "comparison sentences (each wrapped in <comparison> tags):\n"
                                           "1) Decide if there is a phrase shared between the target sentence and at least one comparison sentence\n"
                                           "2) If no phrase is repeated between the target sentence and at least one comparison sentence, output N/A\n"
                                           "3) If a phrase is repeated, output the phrase\n"
                                           "4) The repeated phrase MUST be from the target sentence, not the comparison sentences.\n"
                                           "5) Output should be wrapped in <output> tags.")
    comparison_string = '<comparison>' + '</comparison>\n<comparison>'.join(comparison_sentences) + '</comparison>'
    human_message = HumanMessage(content=f"<target>{sentence}</target>\n{comparison_string}")
    response = llm([system_message, human_message])
    output = get_by_xml_tag(response.content, 'output')
    if not output or output == "N/A" or len(output.split()) < 3:
        return None
    return output


if __name__ == '__main__':
    sentences = [
        # "The Shema is not just a prayer, but a declaration of faith and a guide for daily life, emphasizing the importance of loving God, teaching His laws to the next generation, and keeping His commandments.",
        "The Shema is a central prayer in Judaism, affirming oneness of God and the unique relationship between God and the Jewish people.",
        "The Shema is a central prayer in Judaism, and its precise recitation is of utmost importance.",
        "The Shema is a central prayer in Judaism, and understanding when to recite it is crucial to its observance.",
        "The Shema is a central prayer in Judaism, and understanding its structure and timing can deepen one's connection to this daily declaration of faith.",
        "How can one truly love God?",
    ]
    for target_id in range(len(sentences)):
        target = sentences[target_id]
        comps = [sent for j, sent in enumerate(sentences) if j != target_id]
        print("-----")
        print(target)
        phrase = "The Shema is a central prayer in Judaism"
        print(differentiate_writing(target, phrase))
    # print(repeated_phrase(target, comps))

