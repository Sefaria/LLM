"""
When generating many prompts for one topic page,
GPT may repeat many phrases
This file tries to differentiate the writing style

Currently unused because I couldn't find a way to differentiate the writing without degrading it
"""


from app.util.general import get_by_xml_tag
from app.basic_langchain.schema import HumanMessage, SystemMessage
from app.basic_langchain.chat_models import ChatOpenAI


def differentiate_writing(sentence, phrase_to_avoid):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_message = SystemMessage(content="Given a target sentence (wrapped in <target> tags) and a phrase to avoid "
                                           "using (wrapped in <phrase_to_avoid> tags), rewrite <target> so it doesn't "
                                           "use <phrase_to_avoid> but maintains the same meaning, tone and terminology."
                                           " Change as few words as possible to achieve this goal."
                                           " Keep <target> to one sentence."
                                           " <example>\n<target>Abraham, patriarch of the Jewish people, emerged from a world of idolatry and established a covenant with God.</target><phrase_to_avoid>Abraham, patriarch of the Jewish people</phrase_to_avoid><output>The Jewish people's patriarch, Abraham, emerged from a world of idolatry and established a covenant with God.</output></example>"
                                           " Wrap output in <output> tags.")
    human_message = HumanMessage(content=f"<target>{sentence}</target>\n<phrase_to_avoid>{phrase_to_avoid}</phrase_to_avoid>")
    response = llm([system_message, human_message])
    output = get_by_xml_tag(response.content, 'output')
    if not output:
        return response.content
    return output


def remove_dependent_clause(sentence, phrase_to_avoid):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_message = SystemMessage(content="Given a target sentence (wrapped in <target> tags) and a phrase to avoid "
                                           "using (wrapped in <phrase_to_avoid> tags):\n"
                                           "1) Determine if <phrase_to_avoid> includes an explanation of a term\n"
                                           "2) If yes, rewrite <target> to by deleting this explanation but keeping the term itself\n"
                                           "3) If no, return <target> unchanged.\n"
                                           "4) Wrap rewritten <target> sentence in <output> tags."
                                           "<example>"
                                           "<target>The Shema is a central prayer in Judaism, and its precise recitation is of utmost importance.</target>"
                                           "<phrase_to_avoid>The Shema is a central prayer in Judaism</phrase_to_avoid>"
                                           "<output>The Shema's precise recitation is of utmost importance.</output>"
                                           "</example>"
                                           "<example>"
                                           "<target>Moses, as the primary teacher of the Oral Law, played a crucial role in ensuring its transmission to future generations.</target>"
                                           "<phrase_to_avoid>Moses, as the primary teacher of the Oral Law</phrase_to_avoid>"
                                           "<output>Moses played a crucial role in ensuring the transmission of the Oral Law to future generations.</output>")
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
    return output.strip()


if __name__ == '__main__':
    sentences = [
        # "The Shema is not just a prayer, but a declaration of faith and a guide for daily life, emphasizing the importance of loving God, teaching His laws to the next generation, and keeping His commandments.",
        # "The Shema is a central prayer in Judaism, affirming oneness of God and the unique relationship between God and the Jewish people.",
        # "The Shema is a central prayer in Judaism, and its precise recitation is of utmost importance.",
        # "The Shema is a central prayer in Judaism, and understanding when to recite it is crucial to its observance.",
        # "The Shema is a central prayer in Judaism, and understanding its structure and timing can deepen one's connection to this daily declaration of faith.",
        # "How can one truly love God?",
        "Circumcision is not just a physical act, but a father's religious obligation."
    ]
    for target_id in range(len(sentences)):
        target = sentences[target_id]
        comps = [sent for j, sent in enumerate(sentences) if j != target_id]
        print("-----")
        print(target)
        phrase = "Circumcision is not just a physical act"
        print(remove_dependent_clause(target, phrase))
    # print(repeated_phrase(target, comps))

