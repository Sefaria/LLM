import re
import unicodedata


HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


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
