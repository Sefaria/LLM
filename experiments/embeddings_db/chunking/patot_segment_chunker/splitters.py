import re

from semantic_chunkers.splitters.base import BaseSplitter
from semantic_chunkers.splitters.regex import RegexSplitter


class HebrewTokenizerSplitter(BaseSplitter):
    re_paranthesis_open = r"[\(\[\{\'\"`]"
    re_paranthesis_close = r"[\)\]\}\'\"`]"
    re_sentence_separators = r"[\.!?]"
    re_no_space_sequence = r"[^ \t\f\v]+(?:[\n][^ \t\f\v]*)*"
    re_numbering = r"(?:(?:[א-י]|\d+)\.)+"
    re_heb_dot_acronym = r"(?:(?:[א-ת]\.)+[א-ת]+)"
    re_numeric = r"(?:[+-]?(?:[0-9][0-9.,\/\-:]*)?(?:[0-9])%?)"
    re_3dots_style_sequence = re_sentence_separators + "{2,}"
    re_legal_with_separator = r"{0}*{1}{2}*".format(
        re_paranthesis_open,
        "|".join((re_numbering, re_numeric, re_heb_dot_acronym)),
        re_paranthesis_close,
    )
    re_sentence_end = r"(?:{0}{1}\n*)|\n+".format(re_paranthesis_close, re_sentence_separators)

    def __call__(self, doc: str) -> list[str]:
        doc = doc.replace("\r", "").replace("''", '"')
        sentences = []
        current_sentence = []

        for suspect_sequence in re.findall(self.re_no_space_sequence, doc, flags=re.MULTILINE | re.UNICODE):
            current_start = 0
            i = 0
            while i < len(suspect_sequence):
                match_end_sentence = re.match(self.re_sentence_end, suspect_sequence[i:])
                if match_end_sentence:
                    current_sentence.append(suspect_sequence[current_start:i])
                    current_sentence.extend(c for c in suspect_sequence[i : i + match_end_sentence.end()] if c != "\n")
                    sentence = " ".join(part for part in current_sentence if part).strip()
                    if sentence:
                        sentences.append(sentence)
                    current_sentence = []
                    i += match_end_sentence.end()
                    current_start = i
                    continue

                if suspect_sequence[i] in ["!", "?", "."]:
                    match_multiple_seps = re.match(self.re_3dots_style_sequence, suspect_sequence[i:])
                    if match_multiple_seps:
                        current_sentence.append(suspect_sequence[current_start:i])
                        current_sentence.append(suspect_sequence[i : i + match_multiple_seps.end()])
                        i += match_multiple_seps.end()
                        current_start = i
                        continue

                    match_sep_before_closing = re.match(
                        self.re_sentence_separators + self.re_paranthesis_close + "+$",
                        suspect_sequence[i:],
                    )
                    if match_sep_before_closing:
                        current_sentence.append(suspect_sequence[current_start:i])
                        current_sentence.extend(
                            c
                            for c in suspect_sequence[
                                i + match_sep_before_closing.start() : i + match_sep_before_closing.end()
                            ]
                        )
                        i += match_sep_before_closing.end()
                        current_start = i
                    elif suspect_sequence[i] in ["!", "?"] or (
                        suspect_sequence[i] == "." and i == len(suspect_sequence) - 1
                    ):
                        current_sentence.append(suspect_sequence[current_start:i])
                        current_sentence.append(suspect_sequence[i])
                        sentence = " ".join(part for part in current_sentence if part).strip()
                        if sentence:
                            sentences.append(sentence)
                        current_sentence = []
                        i += 1
                        current_start = i
                        continue
                    else:
                        match_legal_token = re.match(self.re_legal_with_separator, suspect_sequence[current_start:])
                        if match_legal_token:
                            current_sentence.append(suspect_sequence[current_start : current_start + match_legal_token.end()])
                            i = current_start + match_legal_token.end()
                            current_start = i
                            continue
                        current_sentence.append(suspect_sequence[current_start:i])
                        current_sentence.append(suspect_sequence[i])
                        sentence = " ".join(part for part in current_sentence if part).strip()
                        if sentence:
                            sentences.append(sentence)
                        current_sentence = []
                        i += 1
                        current_start = i
                        continue
                i += 1

            if current_start <= len(suspect_sequence) - 1:
                current_sentence.append(suspect_sequence[current_start:])

        trailing_sentence = " ".join(part for part in current_sentence if part).strip()
        if trailing_sentence:
            sentences.append(trailing_sentence)

        cleaned_sentences = [sentence for sentence in sentences if sentence]
        if len(cleaned_sentences) <= 1:
            fallback_sentences = fallback_clause_split(doc)
            if len(fallback_sentences) > 1:
                return fallback_sentences
        return cleaned_sentences


def sentence_splitter_for_language(lang: str) -> BaseSplitter:
    if lang == "he":
        return HebrewTokenizerSplitter()
    return RegexSplitter()


class StanzaHebrewSentenceSplitter:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            import stanza

            self._pipeline = stanza.Pipeline(
                lang="he",
                processors="tokenize",
                dir=self.model_dir,
                verbose=False,
                use_gpu=False,
            )
        return self._pipeline

    def __call__(self, text: str) -> list[str]:
        doc = self._get_pipeline()(text)
        return [sentence.text.strip() for sentence in doc.sentences if sentence.text and sentence.text.strip()]


def fallback_clause_split(text: str) -> list[str]:
    parts = re.split(r"([:,])", text)
    chunks = []
    current = ""

    for part in parts:
        if not part:
            continue
        current += part
        if part in {":", ","}:
            cleaned = current.strip()
            if cleaned:
                chunks.append(cleaned)
            current = ""

    cleaned = current.strip()
    if cleaned:
        chunks.append(cleaned)

    return chunks
