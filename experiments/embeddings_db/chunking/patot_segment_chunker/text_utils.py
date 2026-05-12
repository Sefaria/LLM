import re
import unicodedata


HTML_TAG_RE = re.compile(r"<[^>]+>")
FOOTNOTE_START_RE = re.compile(
    r"""<sup[^>]*class="(?:[^"]+ )?footnote-marker(?: [^"]+)?">.*?</sup>\s*<i[^>]*class="(?:[^"]+ )?footnote(?: [^"]+)?"[^>]*>""",
)


def strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


def find_html_footnote_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []

    for match in FOOTNOTE_START_RE.finditer(text):
        start_pos = match.start()
        i_open_end = match.end()
        depth = 1
        i = i_open_end

        while depth > 0:
            next_open = text.find("<i", i)
            next_close = text.find("</i", i)

            next_close_end = -1
            if next_close != -1:
                close_probe = next_close + 3
                if close_probe < len(text) and text[close_probe] == ">":
                    next_close_end = next_close + 4
                elif close_probe + 1 < len(text) and text[close_probe + 1] == ">":
                    next_close_end = next_close + 5

            if next_close == -1 or next_close_end == -1:
                break

            if next_open != -1 and next_open < next_close and next_open + 2 < len(text) and text[next_open + 2] in (" ", ">"):
                depth += 1
                i = next_open + 2
            else:
                depth -= 1
                i = next_close_end

        if depth > 0:
            continue

        end_pos = i
        is_subset = False
        for start, end in spans:
            if start_pos > start and end_pos < end:
                is_subset = True
                break
        if not is_subset:
            spans.append((start_pos, end_pos))

    return spans


def extract_html_footnotes(text: str) -> list[str]:
    return [text[start:end] for start, end in find_html_footnote_spans(text)]


def remove_html_footnotes(text: str, replacement: str = " ") -> str:
    spans = find_html_footnote_spans(text)
    if not spans:
        return text

    chars = list(text)
    for start, end in reversed(spans):
        chars[start:end] = replacement
    return "".join(chars)


def strip_hebrew_niqqud(text: str) -> str:
    result = []
    for char in unicodedata.normalize("NFD", text):
        codepoint = ord(char)
        if 0x0591 <= codepoint <= 0x05C7:
            continue
        result.append(char)
    return unicodedata.normalize("NFC", "".join(result))


def detect_language(text: str) -> str:
    hebrew_count = sum(1 for char in text if "\u0590" <= char <= "\u05FF")
    english_count = sum(1 for char in text if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    return "he" if hebrew_count > english_count else "en"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
