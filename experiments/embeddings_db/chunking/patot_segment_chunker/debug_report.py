from pathlib import Path

from bidi.algorithm import get_display

from .chunker import PatotChunkResult


TITLE_FONT = "Helvetica-Bold"
UI_FONT = "Helvetica"
BODY_FONT = "DebugReportBody"
BODY_FONT_PATHS = [
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]


def ensure_report_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if BODY_FONT in pdfmetrics.getRegisteredFontNames():
        return

    for path in BODY_FONT_PATHS:
        if path.exists():
            pdfmetrics.registerFont(TTFont(BODY_FONT, str(path)))
            return

    raise RuntimeError(
        "Could not find a Unicode font for PDF output. "
        f"Tried: {', '.join(str(path) for path in BODY_FONT_PATHS)}"
    )


def contains_hebrew(text: str) -> bool:
    return any("\u0590" <= char <= "\u05FF" for char in text)


def to_pdf_display_text(text: str) -> str:
    if contains_hebrew(text):
        return get_display(text)
    return text


def wrap_text_for_pdf(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    from reportlab.pdfbase import pdfmetrics

    if not text:
        return [""]

    lines = []
    for paragraph in text.splitlines() or [text]:
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split(" ")
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if pdfmetrics.stringWidth(to_pdf_display_text(candidate), font_name, font_size) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_pdf_line(pdf, text: str, x: float, y: float, max_width: float) -> None:
    display_text = to_pdf_display_text(text)
    if contains_hebrew(text):
        pdf.drawRightString(x + max_width, y, display_text)
    else:
        pdf.drawString(x, y, display_text)


def start_page(pdf, title: str, page_width: float, page_height: float) -> float:
    pdf.setFont(TITLE_FONT, 16)
    pdf.drawString(40, page_height - 40, title)
    pdf.setFont(BODY_FONT, 10)
    return page_height - 65


def add_summary_page(pdf, result: PatotChunkResult, tref: str, lang: str, config, page_width: float, page_height: float) -> None:
    from reportlab.lib import colors

    y = start_page(pdf, "Patot Segment Chunker Report", page_width, page_height)
    usable_width = page_width - 80
    summary_lines = [
        f"tref={tref}",
        f"lang={lang}",
        f"detected_lang={result.debug_trace.detected_lang}",
        f"input_segment_count={result.input_segment_count}",
        f"final_chunk_count={result.final_chunk_count}",
        f"min_split_tokens={config.min_split_tokens}",
        f"max_split_tokens={config.max_split_tokens}",
        "",
        "How to read the table:",
        "Segment = one original library segment from the source text.",
        "Unit = one smaller piece inside a segment, usually a sentence-like split.",
        "Chunk = one final output piece returned by the algorithm.",
        "A chunk may contain several units. A segment may contain several units.",
        "The table is read row by row from top to bottom.",
        "",
        "Columns:",
        "Seg = which original source segment this row belongs to.",
        "Unit = which sentence-like piece inside that segment this row represents.",
        "Chunk = which final output chunk this row ends up inside.",
    ]

    for line in summary_lines:
        for wrapped in wrap_text_for_pdf(line, BODY_FONT, 10, usable_width):
            draw_pdf_line(pdf, wrapped, 40, y, usable_width)
            y -= 12

    y -= 6
    pdf.setFont(TITLE_FONT, 11)
    pdf.drawString(40, y, "Visual Legend")
    y -= 16

    pdf.setFont(BODY_FONT, 10)
    pdf.drawString(40, y, "Text rows:")
    pdf.setFillColor(colors.HexColor("#f7f9fc"))
    pdf.rect(95, y - 9, 42, 11, fill=1, stroke=0)
    pdf.setFillColor(colors.black)
    pdf.drawString(145, y, "alternating row shading only helps your eye track rows")
    y -= 16

    pdf.drawString(40, y, "Boundaries:")
    pdf.setStrokeColor(colors.HexColor("#888888"))
    pdf.setLineWidth(1)
    pdf.line(98, y - 3, 138, y - 3)
    pdf.setStrokeColor(colors.black)
    pdf.drawString(145, y, "gray line = the source segment ends here")
    y -= 16

    pdf.setStrokeColor(colors.HexColor("#1f5aa6"))
    pdf.setLineWidth(2.2)
    pdf.line(98, y - 3, 138, y - 3)
    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(1)
    pdf.drawString(145, y, "blue thick line = the final chunk ends here")
    y -= 22

    example_lines = [
        "Example:",
        "If three rows all show Chunk C4, those three rows are merged into one final chunk.",
        "If the segment changes from S7 to S8, you crossed an original library segment boundary.",
        "If Unit goes 1, 2, 3 inside the same segment, that segment was internally split into three pieces.",
    ]
    for line in example_lines:
        for wrapped in wrap_text_for_pdf(line, BODY_FONT, 10, usable_width):
            draw_pdf_line(pdf, wrapped, 40, y, usable_width)
            y -= 12

    if result.debug_trace.pass3_adjustments:
        y -= 6
        warning_lines = [
            f"Pass 3 hard-max fallback was used on {len(result.debug_trace.pass3_adjustments)} chunk(s).",
            "Those forced token-window cuts are reflected in the final chunk list, but not drawn exactly inside the table below.",
        ]
        for line in warning_lines:
            for wrapped in wrap_text_for_pdf(line, BODY_FONT, 10, usable_width):
                draw_pdf_line(pdf, wrapped, 40, y, usable_width)
                y -= 12
    pdf.showPage()


def build_table_rows(result: PatotChunkResult) -> list[dict]:
    prepared_segments = result.debug_trace.prepared_segments
    pass2_by_tref = {segment.tref: segment for segment in result.debug_trace.pass2_segments}
    pass1_refs = result.debug_trace.pass1_chunk_segment_refs

    rows = []
    segment_idx = 0
    chunk_idx = 0

    for pass1_group_refs in pass1_refs:
        if len(pass1_group_refs) > 1:
            for local_index, tref in enumerate(pass1_group_refs, start=1):
                segment = prepared_segments[segment_idx]
                segment_idx += 1
                rows.append(
                    {
                        "segment_number": segment_idx,
                        "unit_number": "-",
                        "chunk_number": chunk_idx + 1,
                        "text": segment.processed_text,
                        "segment_boundary_after": True,
                        "final_chunk_boundary_after": local_index == len(pass1_group_refs),
                    }
                )
            chunk_idx += 1
            continue

        segment = prepared_segments[segment_idx]
        segment_idx += 1
        pass2_segment = pass2_by_tref.get(segment.tref)
        if not pass2_segment:
            rows.append(
                {
                    "segment_number": segment_idx,
                    "unit_number": "-",
                    "chunk_number": chunk_idx + 1,
                    "text": segment.processed_text,
                    "segment_boundary_after": True,
                    "final_chunk_boundary_after": True,
                }
            )
            chunk_idx += 1
            continue

        effective_units = pass2_segment.fallback_splits or pass2_segment.initial_splits or [segment.processed_text]
        final_chunk_split_counts = [chunk.split_count for chunk in pass2_segment.final_chunks] or [len(effective_units)]
        chunk_unit_cursor = 0
        current_chunk_end = final_chunk_split_counts[0]
        local_chunk_idx = 0

        for unit_idx, unit_text in enumerate(effective_units, start=1):
            if unit_idx > current_chunk_end and local_chunk_idx + 1 < len(final_chunk_split_counts):
                local_chunk_idx += 1
                current_chunk_end += final_chunk_split_counts[local_chunk_idx]
            is_last_unit_in_chunk = unit_idx == current_chunk_end
            rows.append(
                {
                    "segment_number": segment_idx,
                    "unit_number": unit_idx,
                    "chunk_number": chunk_idx + local_chunk_idx + 1,
                    "text": unit_text,
                    "segment_boundary_after": unit_idx == len(effective_units),
                    "final_chunk_boundary_after": is_last_unit_in_chunk,
                }
            )
            chunk_unit_cursor += 1

        chunk_idx += max(len(final_chunk_split_counts), 1)

    return rows


def add_single_table_section(pdf, rows: list[dict], page_width: float, page_height: float) -> None:
    from reportlab.lib import colors

    left_margin = 40
    right_margin = 40
    bottom_y = 45
    leading = 12
    row_padding = 4
    segment_x = left_margin
    unit_x = left_margin + 40
    chunk_x = left_margin + 78
    text_x = left_margin + 120
    text_width = page_width - text_x - right_margin
    row_colors = [colors.HexColor("#f7f9fc"), colors.HexColor("#eef5fb")]

    def start_table_page(first_page: bool) -> float:
        y = start_page(pdf, "Annotated Structure", page_width, page_height)
        if first_page:
            pdf.bookmarkPage("annotated", fit="Fit")
            pdf.addOutlineEntry("Annotated Structure", "annotated", level=0, closed=False)
        pdf.setFont(TITLE_FONT, 9)
        pdf.drawString(segment_x, y, "Seg")
        pdf.drawString(unit_x, y, "Unit")
        pdf.drawString(chunk_x, y, "Chunk")
        pdf.drawString(text_x, y, "Text")
        return y - 12

    y = start_table_page(first_page=True)

    for row_index, row in enumerate(rows, start=1):
        wrapped_text = wrap_text_for_pdf(row["text"], BODY_FONT, 10, text_width)
        row_height = row_padding * 2 + leading * len(wrapped_text)
        boundary_extra = 8 if row["segment_boundary_after"] or row["final_chunk_boundary_after"] else 0

        if y - row_height - boundary_extra < bottom_y:
            pdf.showPage()
            y = start_table_page(first_page=False)

        row_top = y
        row_bottom = y - row_height
        pdf.setFillColor(row_colors[(row_index - 1) % len(row_colors)])
        pdf.rect(left_margin - 4, row_bottom, page_width - left_margin - right_margin + 8, row_height, fill=1, stroke=0)
        pdf.setFillColor(colors.black)

        pdf.setFont(UI_FONT, 9)
        pdf.drawString(segment_x, row_top - 11, f"S{row['segment_number']}")
        pdf.drawString(unit_x, row_top - 11, str(row["unit_number"]))
        pdf.drawString(chunk_x, row_top - 11, f"C{row['chunk_number']}")

        pdf.setFont(BODY_FONT, 10)
        text_y = row_top - 11
        for wrapped_line in wrapped_text:
            draw_pdf_line(pdf, wrapped_line, text_x, text_y, text_width)
            text_y -= leading

        y = row_bottom - 4

        if row["segment_boundary_after"]:
            pdf.setStrokeColor(colors.HexColor("#888888"))
            pdf.setLineWidth(1)
            pdf.line(left_margin - 4, y, page_width - right_margin + 4, y)
            y -= 3

        if row["final_chunk_boundary_after"]:
            pdf.setStrokeColor(colors.HexColor("#1f5aa6"))
            pdf.setLineWidth(2.2)
            pdf.line(left_margin - 4, y, page_width - right_margin + 4, y)
            y -= 5

        pdf.setStrokeColor(colors.black)
        pdf.setLineWidth(1)

    pdf.showPage()


def add_final_chunks_page(pdf, result: PatotChunkResult, page_width: float, page_height: float) -> None:
    y = start_page(pdf, "Final Chunks", page_width, page_height)
    pdf.bookmarkPage("final_chunks", fit="Fit")
    pdf.addOutlineEntry("Final Chunks", "final_chunks", level=0, closed=False)
    usable_width = page_width - 80

    for i, chunk in enumerate(result.chunks, start=1):
        lines = [
            f"{i}. kind={chunk.kind} pass={chunk.pass_number} refs={chunk.source_segment_refs} token_count={chunk.token_count}",
            chunk.text,
            "",
        ]
        for line in lines:
            wrapped_lines = wrap_text_for_pdf(line, BODY_FONT, 10, usable_width)
            if y - 12 * len(wrapped_lines) < 45:
                pdf.showPage()
                y = start_page(pdf, "Final Chunks", page_width, page_height)
            for wrapped in wrapped_lines:
                draw_pdf_line(pdf, wrapped, 40, y, usable_width)
                y -= 12
    pdf.showPage()


def write_debug_pdf(output_path: Path, result: PatotChunkResult, tref: str, lang: str, config) -> None:
    if not result.debug_trace:
        return

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: reportlab\nInstall it with:\npython -m pip install reportlab"
        ) from exc

    ensure_report_fonts()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    page_width, page_height = A4
    pdf.setTitle("Patot Segment Chunker Report")

    pdf.bookmarkPage("summary", fit="Fit")
    pdf.addOutlineEntry("Summary", "summary", level=0, closed=False)
    add_summary_page(pdf, result, tref, lang, config, page_width, page_height)
    add_single_table_section(pdf, build_table_rows(result), page_width, page_height)
    add_final_chunks_page(pdf, result, page_width, page_height)
    pdf.save()
