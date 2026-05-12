# -*- coding: utf-8 -*-
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.embeddings_db.chunking.patot_segment_chunker.chunker import PatotChunker
from experiments.embeddings_db.chunking.patot_segment_chunker.config import ChunkerConfig, SefariaSourceConfig
from experiments.embeddings_db.chunking.patot_segment_chunker.debug_report import write_debug_pdf
from experiments.embeddings_db.chunking.patot_segment_chunker.sefaria_loader import load_segment_records


SEFARIA_SOURCE = SefariaSourceConfig(
    sefaria_project_path=REPO_ROOT.parent / "Sefaria-Project",
    tref="Talmud_Series;_Shemitah,_Editor's_Introduction.12",
    lang="he",
    version_title=None,
)
DEBUG_OUTPUT_PDF = BASE_DIR / "patot_chunker_debug_report.pdf"

CHUNKER_CONFIG = ChunkerConfig(
    model="gemini-embedding-001",
    setup="retrieval",
    dim=1536,
    sim="dot",
    doc="raw_text",
    query="raw_query",
    norm=True,
    min_split_tokens=200,
    max_split_tokens=500,
    split_tokens_tolerance=10,
    dynamic_threshold=True,
    strip_hebrew_niqqud=True,
    debug=True,
)


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.")

    segments = load_segment_records(
        sefaria_project_path=SEFARIA_SOURCE.sefaria_project_path,
        tref=SEFARIA_SOURCE.tref,
        lang=SEFARIA_SOURCE.lang,
        version_title=SEFARIA_SOURCE.version_title,
    )
    chunker = PatotChunker(api_key=api_key, config=CHUNKER_CONFIG)
    result = chunker.chunk_segments(segments)
    if CHUNKER_CONFIG.debug:
        write_debug_pdf(DEBUG_OUTPUT_PDF, result, SEFARIA_SOURCE.tref, SEFARIA_SOURCE.lang, CHUNKER_CONFIG)

    print("Patot segment chunker")
    print("==============================")
    print(f"tref={SEFARIA_SOURCE.tref}")
    print(f"lang={SEFARIA_SOURCE.lang}")
    print(f"input_segment_count={result.input_segment_count}")
    print(f"pass1_chunk_count={result.pass1_chunk_count}")
    print(f"final_chunk_count={result.final_chunk_count}")
    if CHUNKER_CONFIG.debug:
        print(f"debug_pdf={DEBUG_OUTPUT_PDF}")
    print()

    for i, chunk in enumerate(result.chunks, start=1):
        print(f"Chunk {i}")
        print(
            json.dumps(
                {
                    "kind": chunk.kind,
                    "pass_number": chunk.pass_number,
                    "source_segment_refs": chunk.source_segment_refs,
                    "token_count": chunk.token_count,
                    "triggered": chunk.triggered,
                    "score": chunk.score,
                },
                ensure_ascii=False,
            )
        )
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()
