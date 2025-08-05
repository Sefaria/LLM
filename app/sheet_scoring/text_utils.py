import re
import html
from typing import Dict, List, Tuple, Any

TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def strip_html(raw: str) -> str:
    """Remove tags & entities, collapse whitespace."""
    if not raw:
        return ""
    text = TAG_RE.sub("", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+\n", "\n", text)  # trim spaces before newlines
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse runs of blanks
    return text.strip()


def token_count(text: str) -> int:
    """Approximate word tokens (both English & Hebrew)."""
    return len(TOKEN_RE.findall(text))


def sheet_to_text_views(
    sheet: Dict[str, Any],
    default_lang: str = "en",
) -> Tuple[str, str, str, bool, float]:
    """
    Build three plain‑text snapshots of a Sefaria sheet **and** compute a
    creativity score.

    Returns
    -------
    quotes_only        str   – ref + canonical text blocks
    no_quotes          str   – title & user commentary, refs only for quotes
    with_quotes        str   – full sheet (title, commentary, *and* quotes)
    has_original       bool  – True if any user commentary exists
    creativity_score   float – user_token_count / total_token_count
    """

    quotes: List[str] = []
    no_quotes: List[str] = []
    with_quotes: List[str] = []

    original_tokens = 0
    quoted_tokens = 0
    has_original = False

    title = strip_html(sheet.get("title", "")).strip()
    if title:
        tok = token_count(title)
        original_tokens += tok
        no_quotes.append(title)
        with_quotes.append(title)

    for blk in sheet.get("sources", []):
        # --- outsideText (single‑lang commentary)
        if "outsideText" in blk:
            txt = strip_html(blk["outsideText"]).strip()
            if txt:
                has_original = True
                t = token_count(txt)
                original_tokens += t
                no_quotes.append(txt)
                with_quotes.append(txt)

        if "outsideBiText" in blk:
            for lang in ("en", "he"):
                txt = strip_html(blk["outsideBiText"].get(lang, "")).strip()
                if txt:
                    has_original = True
                    original_tokens += token_count(txt)
                    no_quotes.append(txt)
                    with_quotes.append(txt)

        if "text" in blk:
            ref = blk.get("ref", "").strip()
            canon = strip_html(blk["text"].get(default_lang, "")).strip()

            # show ref label in all views
            if ref:
                no_quotes.append(ref)
                header = f"{ref}:"
            else:
                header = ""

            if canon:
                # quote tokens count toward quoted_tokens
                qtok = token_count(canon)
                quoted_tokens += qtok

                # add to quotes‑only and with_quotes
                if header:
                    quotes.append(header)
                    with_quotes.append(header)
                quotes.append(canon)
                with_quotes.append(canon)

        if "comment" in blk:
            txt = strip_html(blk["comment"]).strip()
            if txt:
                has_original = True
                original_tokens += token_count(txt)
                no_quotes.append(txt)
                with_quotes.append(txt)

    joiner = "\n\n"
    quotes_only = joiner.join(quotes)
    commentary = joiner.join(no_quotes)
    full_sheet = joiner.join(with_quotes)

    total_tokens = original_tokens + quoted_tokens or 1  # avoid div‑by‑zero
    creativity = original_tokens / total_tokens

    return quotes_only, commentary, full_sheet, has_original, creativity