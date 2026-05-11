# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.embeddings_db.chunking.patot_segment_chunker.chunker import PatotChunker
from experiments.embeddings_db.chunking.patot_segment_chunker.config import ChunkerConfig
from experiments.embeddings_db.chunking.patot_segment_chunker.sefaria_loader import load_segment_records
from experiments.embeddings_db.chunking.patot_segment_chunker.splitters import HebrewTokenizerSplitter


# Edit here.
TREF = "Tur, Orach Chayim 11:1"
LANG = "he"
VERSION_TITLE = None
SEFARIA_PROJECT_PATH = REPO_ROOT.parent / "Sefaria-Project"


def main() -> None:
    rows = load_segment_records(
        sefaria_project_path=SEFARIA_PROJECT_PATH,
        tref=TREF,
        lang=LANG,
        version_title=VERSION_TITLE,
    )
    if not rows:
        raise SystemExit(f"No segment text found for tref={TREF!r} lang={LANG!r}")

    cfg = ChunkerConfig()
    chunker = PatotChunker(api_key=os.getenv("GEMINI_API_KEY", "DUMMY"), config=cfg)
    hebrew_splitter = HebrewTokenizerSplitter()

    for row in rows:
        processed = chunker._preprocess(row.text)
        splits = hebrew_splitter(processed)
        print(f"ref={row.tref}")
        print(f"sentence_count={len(splits)}")
        for i, split in enumerate(splits, start=1):
            print(f"[{i}] {split}")
        print()


if __name__ == "__main__":
    main()
