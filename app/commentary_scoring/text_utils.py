import re
from typing import Union, List

# Regular expression to match HTML tags (e.g., <div>, <b>, </p>, etc.)
TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    """
    Remove all HTML tags from a given string.
    """
    return TAG_RE.sub("", s)


def flatten_text(x: Union[str, List, tuple]) -> str:
    """
    Recursively flatten a nested structure of strings, lists, or tuples into a single string.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        # Recursively flatten all elements and join with spaces
        return " ".join(flatten_text(el) for el in x)
    # If it's not a string or list/tuple, convert it to string
    return str(x)


def to_plain_text(raw: Union[str, List, tuple]) -> str:
    """
    Convert raw input (possibly nested and HTML-formatted) to clean plain text.
    """
    flat = flatten_text(raw)  # Step 1: Flatten nested structure
    clean = strip_html(flat)  # Step 2: Remove HTML tags
    return re.sub(r"\s+", " ", clean).strip()  # Step 3: Normalize whitespace
