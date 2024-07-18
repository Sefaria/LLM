"""
Make prompts conform to the style guide
"""
import csv
from typing import Optional
from basic_langchain.schema import SystemMessage, HumanMessage
from basic_langchain.chat_models import ChatOpenAI
from util.general import get_by_xml_tag
from dataclasses import dataclass


@dataclass
class StyleGuideRule:
    title: str
    example: str


class StyleGuide:
    STYLE_GUIDE_FILE = "topic_prompt/input/Copy of The Sefaria Glossary - Commonly Used Glosses of Works.csv"

    def __init__(self):
        self._rules: list[StyleGuideRule] = self._read_style_guide_file()

    def _read_style_guide_file(self):
        rules = []
        with open(self.STYLE_GUIDE_FILE, "r") as fin:
            cin = csv.DictReader(fin)
            for row in cin:
                rules.append(StyleGuideRule(row["Work"].strip(), row["Example"].strip()))
        return rules

    def _get_all_titles(self) -> list[str]:
        return [r.title for r in self._rules]

    def _get_title_prompt_uses(self, prompt: str) -> Optional[str]:
        system = SystemMessage(content="Given a list of titles of classic Jewish books, output the title that is mentioned in the input string. Titles are wrapped in <titles> tags. Input string is wrapped in <input> tags. Output the title mentioned in <title> tags. If the no title in <titles> is mentioned, output <title>N/A</title>.")
        human = HumanMessage(content=f"<titles>{', '.join(self._get_all_titles())}</titles>\n<input>{prompt}</input>")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm([system, human])
        title = get_by_xml_tag(response.content, 'title')
        if title is None or title == 'N/A':
            return None
        return title

    def _get_example_by_title(self, title: str) -> Optional[str]:
        for rule in self._rules:
            if rule.title == title:
                return rule.example

    def rewrite_prompt(self, prompt: str) -> str:
        title = self._get_title_prompt_uses(prompt)
        if title is None:
            return prompt
        example = self._get_example_by_title(title)
        return self._rewrite_prompt_to_match_example(prompt, title, example)

    @staticmethod
    def _rewrite_prompt_to_match_rule(prompt: str, title: str, gloss: str) -> str:
        system = SystemMessage(content="Goal: Write <input> so that when it discussed <title> it uses <gloss> as a dependent clause to explain what <title> is.\n"
                                       "Input:\n<input>: string that mentions <title>. Remove any gloss for <title> and replace it with <gloss>.\n"
                                       "<title>: Title of work mentioned in <input>."
                                       "<gloss>: Gloss of work mentioned in <input>. Should be added as a dependent clause to explain what <title is.\n"
                                       "Output: Output the rewritten <input> using the <gloss>. Output should be wrapped in <output> tags. Refrain from changing anything else in <input> besides the gloss for <title>.\n"
                                       "Example:\n"
                                       "<input>The city of Jerusalem has a unique role in rendering its inhabitants righteous, establishing a foundation for justice. Bereshit Rabbah, a collection of rabbinic interpretations of the Book of Genesis, discusses the role of Malkitzedek, the king of Salem (Jerusalem), in revealing the laws of the High Priesthood and Torah precepts to Abraham.</input>\n"
                                       "<title>Bereshit Rabbah</title>\n"
                                       "<gloss>a talmudic-era midrashic work on the book of Genesis</gloss>\n"
                                       "<output>The city of Jerusalem has a unique role in rendering its inhabitants righteous, establishing a foundation for justice. Bereshit Rabbah, a talmudic-era midrashic work on the book of Genesis, discusses the role of Malkitzedek, the king of Salem (Jerusalem), in revealing the laws of the High Priesthood and Torah precepts to Abraham.</output>")
        human = HumanMessage(content=f"<input>{prompt}</input>\n<title>{title}</title>\n<gloss>{gloss}</gloss>")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm([system, human])
        return get_by_xml_tag(response.content, "output")

    @staticmethod
    def _rewrite_prompt_to_match_example(prompt: str, title: str, example: str) -> str:
        system = SystemMessage(content="Goal: Write <input> so that when it introduces <title> in the same way that it is introduced in <example>. DON'T use any other content in <example> except the explanation of <title>.\n"
                                       "Input:\n<input>: string that mentions <title>. Remove any gloss for <title> and replace it with the gloss used in <example>.\n"
                                       "<title>: Title of work mentioned in <input>."
                                       "<example>: An example of how to explain <title>.\n"
                                       "Output: Output the rewritten <input> using the explanation of <title> in <example>. Output should be wrapped in <output> tags. Refrain from changing anything else in <input> besides the gloss for <title>.\n"
                                       "Example:\n"
                                       "<input>Bereshit Rabbah, a collection of rabbinic interpretations of the Book of Genesis, discusses the role of Malkitzedek, the king of Salem (Jerusalem), in revealing the laws of the High Priesthood and Torah precepts to Abraham.</input>\n"
                                       "<title>Bereshit Rabbah</title>\n"
                                       "<example>Bereshit Rabbah, a talmudic-era midrashic work on the book of Genesis, expands upon the biblical narrative in which God warns Isaac against leaving the land of Israel.</example>\n"
                                       "<output>Bereshit Rabbah, a talmudic-era midrashic work on the book of Genesis, discusses the role of Malkitzedek, the king of Salem (Jerusalem), in revealing the laws of the High Priesthood and Torah precepts to Abraham.</output>")
        human = HumanMessage(content=f"<input>{prompt}</input>\n<title>{title}</title>\n<example>{example}</example>")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm([system, human])
        return get_by_xml_tag(response.content, "output")


if __name__ == '__main__':
    s = StyleGuide()
    prompt = "The Mishnah Berakhot, a tractate of the Talmud, discusses the importance of intention in the act of sacrifice, and the specific consequences when a sacrifice is made not for its own sake."
    print(s.rewrite_prompt(prompt))
