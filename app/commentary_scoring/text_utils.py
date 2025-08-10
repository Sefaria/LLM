import re
from typing import Union, List

TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    return TAG_RE.sub("", s)


def flatten_text(x: Union[str, List, tuple]) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(flatten_text(el) for el in x)
    return str(x)


def to_plain_text(raw: Union[str, List, tuple]) -> str:
    """Recursively flatten + remove HTML → clean unicode."""
    flat = flatten_text(raw)
    clean = strip_html(flat)
    return re.sub(r"\s+", " ", clean).strip()