# -*- coding: utf-8 -*-
import os
import re
import sys
import textwrap
import unicodedata
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

from matplotlib import pyplot as plt
from bidi.algorithm import get_display
from pydantic.v1 import PrivateAttr
from semantic_chunkers import StatisticalChunker
from semantic_router.encoders.base import BaseEncoder
from semantic_chunkers.splitters.base import BaseSplitter
from semantic_chunkers.splitters.regex import RegexSplitter


# Edit everything here.
TEXT = """
Thou knowest, also, that God said to our teacher Moses, the master of former and later ages, "Because ye have not confided in me, to sanctity me", "because ye rebelled against my order at the waters of Meribah", "because ye did not sanctify me". All this (God said) although the sin of Moses consisted merely in that he departed from the moral mean of patience to the extreme of wrath in so far as he exclaimed, "Hear now ye rebels" etc., yet for this God found fault with him that such a man as he should show anger in the presence of the entire community of Israel, where wrath is unbecoming. This was a profanation of God's name, because men imitated the words and conduct of Moses, hoping thereby to attain temporal and eternal happiness. How could he, then, allow his wrath free play, since it is a pernicious characteristic, arising, as we have shown, from an evil psychic condition? The divine words, "Ye (Israel) have rebelled against me" are, however, to be explained as follows. Moses was not speaking to ignorant and vicious people, but to an assembly, the most insignificant of whose women, as the sages put it, were on a plane with Ezekiel, the son of Buzi. So, when Moses said or did anything, they subjected his words or actions to the most searching examination. Therefore, when they saw that he waxed wrathful, they said, "He has no moral imperfection, and did he not know that God is angry with us for demanding water, and that we have stirred up the wrath of God, he would not have been angry with us". However, we do not find that when God spoke to Moses about this matter He was angry, but on the contrary, said, "Take the staff ... and give drink to the congregation and their cattle". We have, indeed, digressed from the subject of this chapter, but have, I hope, satisfactorily solved one of the most difficult passages of Scripture concerning which there has been much arguing in the attempt to state exactly what the sin was which Moses committed. Let what others have said be compared with our opinion, and the truth will surely prevail.

"""

MODEL = "gemini-embedding-001"
SETUP = "retrieval"
DIM = 1536
SIM = "dot"
DOC = "raw_text"
QUERY = "raw_query"
NORM = True

SCORE_THRESHOLD = None
THRESHOLD_ADJUSTMENT = 0.01
DYNAMIC_THRESHOLD = True
WINDOW_SIZE = 5
MIN_SPLIT_TOKENS = 200
MAX_SPLIT_TOKENS = 400
SPLIT_TOKENS_TOLERANCE = 10
PLOT_CHUNKS = True
ENABLE_STATISTICS = True
OUTPUT_PDF = BASE_DIR / "semantic_chunker_report.pdf"
OUTPUT_PLOT_PNG = BASE_DIR / "semantic_chunker_plot.png"
PLOT_FIGURE_SIZE = (12, 8)
STRIP_HEBREW_NIQQUD = True
REPORT_BODY_FONT_NAME = "ReportBody"
REPORT_TITLE_FONT_NAME = "Helvetica-Bold"
REPORT_UI_FONT_NAME = "Helvetica"
REPORT_BODY_FONT_PATHS = [
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]

PROJECT_ROOT = BASE_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.embeddings_db.embedding_eval.experiment_grid import EMBEDDING_001_TASK_SETUPS
from experiments.embeddings_db.embedding_eval.gemini_embedder import GeminiEmbedder, l2_normalize_vector


class RepoGeminiEncoder(BaseEncoder):
    name: str = "repo-gemini"
    score_threshold: Optional[float] = SCORE_THRESHOLD
    type: str = "gemini"
    _embedder = PrivateAttr()
    _doc_task_type = PrivateAttr()
    _query_task_type = PrivateAttr()

    def __init__(self, api_key: str):
        super().__init__()
        query_task_type, doc_task_type = EMBEDDING_001_TASK_SETUPS[SETUP]
        self._embedder = GeminiEmbedder(api_key=api_key)
        self._doc_task_type = doc_task_type
        self._query_task_type = query_task_type

    def __call__(self, docs: list[str]) -> list[list[float]]:
        vectors = []
        for doc in docs:
            vector = self._embedder.embed_text(
                model=MODEL,
                text=doc,
                output_dimensionality=DIM,
                task_type=self._doc_task_type,
            )
            if NORM:
                vector = l2_normalize_vector(vector)
            vectors.append(vector)
        return vectors

    async def acall(self, docs: list[str]) -> list[list[float]]:
        return self(docs)


def strip_hebrew_niqqud(text: str) -> str:
    result = []
    for char in unicodedata.normalize("NFD", text):
        codepoint = ord(char)
        if 0x0591 <= codepoint <= 0x05C7:
            continue
        result.append(char)
    return unicodedata.normalize("NFC", "".join(result))


def preprocess_text(text: str) -> str:
    if STRIP_HEBREW_NIQQUD:
        text = strip_hebrew_niqqud(text)
    return text


def detect_language(text: str) -> str:
    hebrew_count = sum(1 for char in text if "\u0590" <= char <= "\u05FF")
    english_count = sum(1 for char in text if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    return "he" if hebrew_count > english_count else "en"


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
                        match_legal_token = re.match(
                            self.re_legal_with_separator,
                            suspect_sequence[current_start:],
                        )
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

        return [sentence for sentence in sentences if sentence]


def build_splitter(text: str) -> BaseSplitter:
    language = detect_language(text)
    if language == "he":
        return HebrewTokenizerSplitter()
    return RegexSplitter()


def build_chunker(api_key: str, text: str) -> StatisticalChunker:
    encoder = RepoGeminiEncoder(api_key=api_key)
    return StatisticalChunker(
        encoder=encoder,
        splitter=build_splitter(text),
        threshold_adjustment=THRESHOLD_ADJUSTMENT,
        dynamic_threshold=DYNAMIC_THRESHOLD,
        window_size=WINDOW_SIZE,
        min_split_tokens=MIN_SPLIT_TOKENS,
        max_split_tokens=MAX_SPLIT_TOKENS,
        split_tokens_tolerance=SPLIT_TOKENS_TOLERANCE,
        plot_chunks=False,
        enable_statistics=ENABLE_STATISTICS,
    )


def analyze_text(chunker: StatisticalChunker) -> tuple[list[str], list[float], float, list[int], list]:
    processed_text = preprocess_text(TEXT)
    splits = chunker._split(processed_text)
    encoded_splits = chunker._encode_documents(splits)
    similarities = chunker._calculate_similarity_scores(encoded_splits)
    if DYNAMIC_THRESHOLD and similarities:
        threshold = chunker._find_optimal_threshold(splits, similarities)
    else:
        threshold = chunker.encoder.score_threshold or chunker.DEFAULT_THRESHOLD
    split_indices = chunker._find_split_indices(similarities, threshold)
    chunks = chunker._split_documents(splits, split_indices, similarities)
    return splits, similarities, threshold, split_indices, chunks


def format_config_lines() -> list[str]:
    return [
        "Gemini config:",
        f"model={MODEL}",
        f"setup={SETUP}",
        f"dim={DIM}",
        f"sim={SIM}",
        f"doc={DOC}",
        f"query={QUERY}",
        f"norm={NORM}",
        "",
        "Chunker config:",
        f"score_threshold={SCORE_THRESHOLD}",
        f"threshold_adjustment={THRESHOLD_ADJUSTMENT}",
        f"dynamic_threshold={DYNAMIC_THRESHOLD}",
        f"window_size={WINDOW_SIZE}",
        f"min_split_tokens={MIN_SPLIT_TOKENS}",
        f"max_split_tokens={MAX_SPLIT_TOKENS}",
        f"split_tokens_tolerance={SPLIT_TOKENS_TOLERANCE}",
        f"plot_chunks={PLOT_CHUNKS}",
        f"enable_statistics={ENABLE_STATISTICS}",
    ]


def build_plot_figure(similarities: list[float], threshold: float, split_indices: list[int], chunks: list):
    fig = plt.figure(figsize=PLOT_FIGURE_SIZE)
    ax_top = fig.add_axes([0.08, 0.58, 0.88, 0.32])
    ax_bottom = fig.add_axes([0.08, 0.12, 0.88, 0.28])

    ax_top.plot(similarities, label="Similarity score", marker="o", color="#1f77b4")
    for split_index in split_indices:
        ax_top.axvline(
            x=split_index - 1,
            color="#c0392b",
            linestyle="--",
            linewidth=1.2,
            label="Split point" if split_index == split_indices[0] else "",
        )
    ax_top.axhline(
        y=threshold,
        color="#2e8b57",
        linestyle="-.",
        linewidth=1.2,
        label=f"Threshold {threshold:.4f}",
    )
    for i, score in enumerate(similarities):
        ax_top.annotate(f"{score:.2f}", (i, score), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    if similarities:
        y_min = min(min(similarities), threshold)
        y_max = max(max(similarities), threshold)
        padding = max((y_max - y_min) * 0.15, 0.02)
        ax_top.set_ylim(y_min - padding, y_max + padding)
        ax_top.set_xlim(-0.5, max(len(similarities) - 0.5, 0.5))

    ax_top.set_title("Similarity Scores and Split Decisions")
    ax_top.set_xlabel("Split index")
    ax_top.set_ylabel("Cosine similarity")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    token_counts = [chunk.token_count or 0 for chunk in chunks]
    bars = ax_bottom.bar(range(1, len(token_counts) + 1), token_counts, color="#87ceeb", edgecolor="#4a708b")
    ax_bottom.axhline(MIN_SPLIT_TOKENS, color="#999999", linestyle=":", linewidth=1, label=f"Min target {MIN_SPLIT_TOKENS}")
    ax_bottom.axhline(MAX_SPLIT_TOKENS, color="#555555", linestyle=":", linewidth=1, label=f"Max target {MAX_SPLIT_TOKENS}")
    ax_bottom.set_title("Chunk Token Counts")
    ax_bottom.set_xlabel("Chunk number")
    ax_bottom.set_ylabel("Token count")
    ax_bottom.grid(True, axis="y", alpha=0.3)
    ax_bottom.legend()
    ax_bottom.set_xlim(0.5, max(len(token_counts) + 0.5, 1.5))
    ax_bottom.set_ylim(0, max(token_counts + [MAX_SPLIT_TOKENS]) * 1.15 if token_counts else MAX_SPLIT_TOKENS * 1.15)
    for bar, token_count in zip(bars, token_counts):
        ax_bottom.text(bar.get_x() + bar.get_width() / 2, token_count + 1, str(token_count), ha="center", va="bottom", fontsize=8)

    return fig, ax_top, ax_bottom


def build_chunk_entries(chunks: list) -> list[dict]:
    entries = []
    for i, chunk in enumerate(chunks, start=1):
        entries.append(
            {
                "title": f"Chunk {i}",
                "bookmark": f"chunk_{i}",
                "meta": (
                    f"token_count={chunk.token_count} "
                    f"triggered={chunk.is_triggered} "
                    f"score={chunk.triggered_score}"
                ),
                "body": chunk.content.strip(),
            }
        )
    return entries


def build_split_entries(splits: list[str]) -> list[dict]:
    entries = []
    for i, split in enumerate(splits, start=1):
        entries.append(
            {
                "title": f"Split {i}",
                "bookmark": f"split_{i}",
                "meta": None,
                "body": split.strip(),
            }
        )
    return entries


def build_summary_lines(chunker: StatisticalChunker, splits: list[str], similarities: list[float], threshold: float, split_indices: list[int], chunks: list) -> list[str]:
    stats = getattr(chunker, "statistics", None)
    lines = [
        "Summary",
        f"original_length_chars={len(TEXT)}",
        f"split_count={len(splits)}",
        f"chunk_count={len(chunks)}",
        f"similarity_score_count={len(similarities)}",
        f"calculated_threshold={threshold}",
        f"split_indices={split_indices}",
        "",
    ]
    lines.extend(format_config_lines())
    if stats is not None:
        lines.extend(
            [
                "",
                "Chunk statistics:",
                f"total_documents={stats.total_documents}",
                f"total_chunks={stats.total_chunks}",
                f"chunks_by_threshold={stats.chunks_by_threshold}",
                f"chunks_by_max_chunk_size={stats.chunks_by_max_chunk_size}",
                f"chunks_by_last_split={stats.chunks_by_last_split}",
                f"min_token_size={stats.min_token_size}",
                f"max_token_size={stats.max_token_size}",
                f"chunks_by_similarity_ratio={stats.chunks_by_similarity_ratio}",
            ]
        )
    return lines


def ensure_report_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if REPORT_BODY_FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return

    for path in REPORT_BODY_FONT_PATHS:
        if path.exists():
            pdfmetrics.registerFont(TTFont(REPORT_BODY_FONT_NAME, str(path)))
            return

    raise SystemExit(
        "Could not find a Unicode font for PDF output. "
        f"Tried: {', '.join(str(path) for path in REPORT_BODY_FONT_PATHS)}"
    )


def wrap_text_for_pdf(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    from reportlab.pdfbase import pdfmetrics

    if not text:
        return [""]

    lines = []
    paragraphs = text.splitlines() or [text]
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split(" ")
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)

    return lines


def contains_hebrew(text: str) -> bool:
    return any("\u0590" <= char <= "\u05FF" for char in text)


def to_pdf_display_text(text: str) -> str:
    if contains_hebrew(text):
        return get_display(text)
    return text


def draw_pdf_line(pdf, text: str, x: float, y: float, max_width: float) -> None:
    display_text = to_pdf_display_text(text)
    if contains_hebrew(text):
        pdf.drawRightString(x + max_width, y, display_text)
    else:
        pdf.drawString(x, y, display_text)


def build_annotated_split_rows(splits: list[str], split_indices: list[int], chunks: list) -> list[dict]:
    rows = []
    semantic_boundaries = set(split_indices)
    final_boundaries = []
    split_cursor = 0
    for chunk_number, chunk in enumerate(chunks, start=1):
        split_cursor += len(chunk.splits)
        final_boundaries.append(split_cursor)

    final_boundary_set = set(final_boundaries[:-1])
    split_to_chunk = {}
    split_cursor = 0
    for chunk_number, chunk in enumerate(chunks, start=1):
        for _ in chunk.splits:
            split_cursor += 1
            split_to_chunk[split_cursor] = chunk_number

    for split_number, split_text in enumerate(splits, start=1):
        rows.append(
            {
                "split_number": split_number,
                "chunk_number": split_to_chunk[split_number],
                "text": split_text.strip(),
                "semantic_boundary_after": split_number in semantic_boundaries,
                "final_boundary_after": split_number in final_boundary_set,
            }
        )
    return rows


def write_wrapped_lines(pdf, lines: list[str], x: float, y: float, max_width: float, font_name: str, font_size: int, leading: int) -> float:
    current_y = y
    pdf.setFont(font_name, font_size)
    for line in lines:
        wrapped = wrap_text_for_pdf(line, font_name, font_size, max_width)
        for wrapped_line in wrapped:
            draw_pdf_line(pdf, wrapped_line, x, current_y, max_width)
            current_y -= leading
    return current_y


def start_page(pdf, title: str, page_width: float, page_height: float) -> float:
    pdf.setFont(REPORT_TITLE_FONT_NAME, 16)
    pdf.drawString(40, page_height - 40, title)
    pdf.setFont(REPORT_BODY_FONT_NAME, 10)
    return page_height - 65


def add_summary_pages(pdf, lines: list[str], page_width: float, page_height: float) -> None:
    pdf.setTitle("Semantic Chunker Report")
    y = start_page(pdf, "Semantic Chunker Report", page_width, page_height)
    leading = 12
    max_y = 45
    usable_width = page_width - 80
    for line in lines:
        wrapped = wrap_text_for_pdf(line, REPORT_BODY_FONT_NAME, 10, usable_width)
        if y - leading * len(wrapped) < max_y:
            pdf.showPage()
            y = start_page(pdf, "Semantic Chunker Report", page_width, page_height)
        y = write_wrapped_lines(pdf, wrapped, 40, y, usable_width, REPORT_BODY_FONT_NAME, 10, leading)
    pdf.showPage()


def add_back_to_plot_link(pdf, page_width: float, page_height: float) -> None:
    label = "Back to plot"
    x = page_width - 120
    y = page_height - 42
    pdf.setFont(REPORT_UI_FONT_NAME, 10)
    pdf.drawString(x, y, label)
    pdf.linkAbsolute("", "plot", Rect=(x - 4, y - 4, x + 68, y + 10), thickness=0)


def add_entries_section(pdf, title: str, entries: list[dict], page_width: float, page_height: float) -> None:
    leading = 12
    min_y = 45
    usable_width = page_width - 80
    for entry in entries:
        y = start_page(pdf, title, page_width, page_height)
        add_back_to_plot_link(pdf, page_width, page_height)
        pdf.bookmarkPage(entry["bookmark"], fit="FitH", top=page_height - 50)
        pdf.addOutlineEntry(entry["title"], entry["bookmark"], level=1, closed=False)
        pdf.setFont(REPORT_TITLE_FONT_NAME, 12)
        pdf.drawString(40, y, entry["title"])
        y -= leading + 2
        pdf.setFont(REPORT_BODY_FONT_NAME, 10)
        if entry["meta"]:
            pdf.drawString(40, y, entry["meta"])
            y -= leading

        body_lines = wrap_text_for_pdf(entry["body"], REPORT_BODY_FONT_NAME, 10, usable_width)
        for line in body_lines:
            if y < min_y:
                pdf.showPage()
                y = start_page(pdf, title, page_width, page_height)
                add_back_to_plot_link(pdf, page_width, page_height)
                pdf.setFont(REPORT_BODY_FONT_NAME, 10)
            draw_pdf_line(pdf, line, 40, y, usable_width)
            y -= leading
        pdf.showPage()


def add_annotated_passage_section(pdf, rows: list[dict], page_width: float, page_height: float) -> None:
    from reportlab.lib import colors

    left_margin = 40
    right_margin = 40
    bottom_y = 45
    leading = 12
    row_padding = 4
    chunk_x = left_margin
    split_x = left_margin + 42
    flags_x = left_margin + 78
    text_x = left_margin + 120
    flag_display = {
        "sem": "S",
        "chunk": "C",
        "sem+chunk": "S+C",
        "-": "-",
    }
    chunk_bg_colors = [colors.HexColor("#f4f7fb"), colors.HexColor("#eef8ef")]
    legend_font_size = 8

    def start_annotated_page(first_page: bool) -> float:
        y = start_page(pdf, "Annotated Passage", page_width, page_height)
        add_back_to_plot_link(pdf, page_width, page_height)
        pdf.setFont(REPORT_UI_FONT_NAME, legend_font_size)
        if first_page:
            pdf.drawString(left_margin, y, "Legend:")
            pdf.setFillColor(colors.HexColor("#c0392b"))
            pdf.drawString(left_margin + 40, y, "red line = semantic candidate")
            pdf.setFillColor(colors.HexColor("#1f5aa6"))
            pdf.drawString(left_margin + 190, y, "blue thick line = final chunk boundary")
            pdf.setFillColor(colors.black)
            pdf.drawString(left_margin, y - 12, "alternating row shading = chunk grouping")
            y -= 28
            pdf.drawString(left_margin, y, "Flags: S = semantic candidate, C = final chunk end, S+C = both, - = no boundary")
            y -= 12
        pdf.setFont(REPORT_TITLE_FONT_NAME, 9)
        pdf.drawString(chunk_x, y, "Chunk")
        pdf.drawString(split_x, y, "Split")
        pdf.drawString(flags_x, y, "Flags")
        pdf.drawString(text_x, y, "Text")
        return y - 12

    y = start_annotated_page(first_page=True)
    pdf.bookmarkPage("annotated", fit="Fit")
    pdf.addOutlineEntry("Annotated Passage", "annotated", level=0, closed=False)

    for row in rows:
        wrapped_text = wrap_text_for_pdf(row["text"], REPORT_BODY_FONT_NAME, 10, page_width - text_x - right_margin)
        row_height = row_padding * 2 + leading * len(wrapped_text)
        boundary_extra = 8 if row["semantic_boundary_after"] or row["final_boundary_after"] else 0

        if y - row_height - boundary_extra < bottom_y:
            pdf.showPage()
            y = start_annotated_page(first_page=False)

        row_top = y
        row_bottom = y - row_height

        pdf.setFillColor(chunk_bg_colors[(row["chunk_number"] - 1) % len(chunk_bg_colors)])
        pdf.rect(left_margin - 4, row_bottom, page_width - left_margin - right_margin + 8, row_height, fill=1, stroke=0)

        pdf.setFillColor(colors.black)
        pdf.setFont(REPORT_UI_FONT_NAME, 9)
        pdf.drawString(chunk_x, row_top - 11, f"C{row['chunk_number']}")
        pdf.drawString(split_x, row_top - 11, f"S{row['split_number']}")

        flags = []
        if row["semantic_boundary_after"]:
            flags.append("sem")
        if row["final_boundary_after"]:
            flags.append("chunk")
        raw_flag = "+".join(flags) if flags else "-"
        pdf.drawString(flags_x, row_top - 11, flag_display[raw_flag])

        pdf.setFont(REPORT_BODY_FONT_NAME, 10)
        text_y = row_top - 11
        for wrapped_line in wrapped_text:
            draw_pdf_line(pdf, wrapped_line, text_x, text_y, page_width - text_x - right_margin)
            text_y -= leading

        y = row_bottom - 4

        if row["semantic_boundary_after"]:
            pdf.setStrokeColor(colors.HexColor("#c0392b"))
            pdf.setLineWidth(1)
            pdf.line(left_margin - 4, y, page_width - right_margin + 4, y)
            y -= 3

        if row["final_boundary_after"]:
            pdf.setStrokeColor(colors.HexColor("#1f5aa6"))
            pdf.setLineWidth(2.2)
            pdf.line(left_margin - 4, y, page_width - right_margin + 4, y)
            y -= 5

        pdf.setStrokeColor(colors.black)
        pdf.setLineWidth(1)

    pdf.showPage()


def data_to_pdf_x(value: float, data_min: float, data_max: float, box_x: float, box_width: float) -> float:
    if data_max == data_min:
        return box_x + box_width / 2
    return box_x + ((value - data_min) / (data_max - data_min)) * box_width


def data_to_pdf_y(value: float, data_min: float, data_max: float, box_y: float, box_height: float) -> float:
    if data_max == data_min:
        return box_y + box_height / 2
    return box_y + ((value - data_min) / (data_max - data_min)) * box_height


def add_plot_page(pdf, similarities: list[float], threshold: float, split_indices: list[int], chunks: list, page_width: float, page_height: float) -> None:
    from reportlab.lib.utils import ImageReader

    fig, ax_top, ax_bottom = build_plot_figure(similarities, threshold, split_indices, chunks)
    fig.savefig(OUTPUT_PLOT_PNG, dpi=160)
    plt.close(fig)

    image_x = 30
    image_y = 145
    image_width = page_width - 60
    image_height = image_width * (PLOT_FIGURE_SIZE[1] / PLOT_FIGURE_SIZE[0])

    pdf.setFont(REPORT_TITLE_FONT_NAME, 16)
    pdf.drawString(40, page_height - 40, "Interactive Plot")
    pdf.setFont(REPORT_BODY_FONT_NAME, 10)
    pdf.drawString(40, page_height - 50, "Top plot points jump to splits. Bottom bars jump to chunks.")
    pdf.drawImage(ImageReader(str(OUTPUT_PLOT_PNG)), image_x, image_y, width=image_width, height=image_height)

    top_pos = ax_top.get_position()
    top_box_x = image_x + image_width * top_pos.x0
    top_box_y = image_y + image_height * top_pos.y0
    top_box_width = image_width * top_pos.width
    top_box_height = image_height * top_pos.height
    top_x_min, top_x_max = ax_top.get_xlim()
    top_y_min, top_y_max = ax_top.get_ylim()

    for i, score in enumerate(similarities):
        point_x = data_to_pdf_x(i, top_x_min, top_x_max, top_box_x, top_box_width)
        point_y = data_to_pdf_y(score, top_y_min, top_y_max, top_box_y, top_box_height)
        pdf.linkAbsolute("", "annotated", Rect=(point_x - 8, point_y - 8, point_x + 8, point_y + 8), thickness=0)

    bottom_pos = ax_bottom.get_position()
    bottom_box_x = image_x + image_width * bottom_pos.x0
    bottom_box_y = image_y + image_height * bottom_pos.y0
    bottom_box_width = image_width * bottom_pos.width
    bottom_box_height = image_height * bottom_pos.height
    bottom_x_min, bottom_x_max = ax_bottom.get_xlim()
    bottom_y_min, bottom_y_max = ax_bottom.get_ylim()

    for i, chunk in enumerate(chunks, start=1):
        token_count = chunk.token_count or 0
        bar_center_x = data_to_pdf_x(i, bottom_x_min, bottom_x_max, bottom_box_x, bottom_box_width)
        bar_top_y = data_to_pdf_y(token_count, bottom_y_min, bottom_y_max, bottom_box_y, bottom_box_height)
        half_bar_width = bottom_box_width / max(len(chunks), 1) * 0.35
        pdf.linkAbsolute("", "annotated", Rect=(bar_center_x - half_bar_width, bottom_box_y, bar_center_x + half_bar_width, bar_top_y), thickness=0)

    pdf.showPage()


def write_pdf_report(chunker: StatisticalChunker, splits: list[str], similarities: list[float], threshold: float, split_indices: list[int], chunks: list) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: reportlab\n"
            "Install it in the same environment with:\n"
            "python -m pip install reportlab\n"
            f"Original error: {exc}"
        )

    ensure_report_fonts()
    pdf = canvas.Canvas(str(OUTPUT_PDF), pagesize=A4)
    page_width, page_height = A4
    pdf.bookmarkPage("summary", fit="Fit")
    pdf.addOutlineEntry("Summary", "summary", level=0, closed=False)
    add_summary_pages(pdf, build_summary_lines(chunker, splits, similarities, threshold, split_indices, chunks), page_width, page_height)
    add_annotated_passage_section(pdf, build_annotated_split_rows(splits, split_indices, chunks), page_width, page_height)
    pdf.bookmarkPage("plot", fit="Fit")
    pdf.addOutlineEntry("Interactive Plot", "plot", level=0, closed=False)
    add_plot_page(pdf, similarities, threshold, split_indices, chunks, page_width, page_height)
    pdf.save()


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.")

    mpl_config_dir = BASE_DIR / ".mplconfig"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    chunker = build_chunker(api_key=api_key, text=TEXT)
    splits, similarities, threshold, split_indices, chunks = analyze_text(chunker)
    write_pdf_report(chunker, splits, similarities, threshold, split_indices, chunks)

    print(f"PDF report written to {OUTPUT_PDF}")
    print(f"Plot image written to {OUTPUT_PLOT_PNG}")


if __name__ == "__main__":
    main()
