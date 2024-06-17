"""
Make prompts conform to the style guide
"""
import csv


class StyleGuide:
    STYLE_GUIDE_FILE = "input/The Sefaria Glossary - English + Transliterated Word List.csv"

    def __init__(self):
        self._rules = self._read_style_guide_file()

    def _read_style_guide_file(self):
        rules = []
        with open(self.STYLE_GUIDE_FILE, "r") as fin:
            cin = csv.reader(fin)
            for row in list(cin)[4:]:
                rules.append(row[0].strip())
        return rules


if __name__ == '__main__':
    s = StyleGuide()
    print(s._rules)