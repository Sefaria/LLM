# -*- coding: utf-8 -*-
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.embeddings_db.chunking.patot_segment_chunker.chunker import PatotChunker
from experiments.embeddings_db.chunking.patot_segment_chunker.config import ChunkerConfig
from experiments.embeddings_db.chunking.patot_segment_chunker.debug_report import write_debug_pdf
from experiments.embeddings_db.chunking.patot_segment_chunker.sefaria_loader import bootstrap_sefaria, load_segment_records


# Edit here.
SEFARIA_PROJECT_PATH = REPO_ROOT.parent / "Sefaria-Project"
LANG = "he"
VERSION_TITLE = None
NUM_SAMPLES = 5
RANDOM_SEED = 13
USE_TOP_SECTIONS = False
OUTPUT_DIR = BASE_DIR / "sampled_section_reports"
INDEX_TITLE_ALLOWLIST: Optional[list[str]] = None

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
    enforce_hard_max_in_pass3=True,
    debug=True,
)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "sample"


def sample_index_and_section_trefs(
    sefaria_project_path: Path,
    n: int,
    seed: int,
    use_top_sections: bool,
) -> list[dict]:
    bootstrap_sefaria(sefaria_project_path)

    from sefaria.model import library

    rng = random.Random(seed)
    candidate_indexes = []

    for index in library.all_index_records():
        title = index.title
        if INDEX_TITLE_ALLOWLIST is not None and title not in INDEX_TITLE_ALLOWLIST:
            continue
        try:
            refs = index.all_top_section_refs() if use_top_sections else index.all_section_refs()
        except Exception as exc:
            print(f"Skipping index={title!r} because section sampling failed: {exc}")
            continue
        if not refs:
            continue
        candidate_indexes.append(
            {
                "index_title": title,
                "section_trefs": [oref.normal() for oref in refs],
            }
        )

    if n > len(candidate_indexes):
        raise ValueError(f"Requested {n} random indexes, but only found {len(candidate_indexes)} indexes with section refs.")

    sampled_indexes = rng.sample(candidate_indexes, n)
    sampled_sections = []
    for sampled_index in sampled_indexes:
        sampled_sections.append(
            {
                "index_title": sampled_index["index_title"],
                "tref": rng.choice(sampled_index["section_trefs"]),
            }
        )
    return sampled_sections


def build_manifest_entry(index_title: str, tref: str, result) -> dict:
    return {
        "index_title": index_title,
        "tref": tref,
        "input_segment_count": result.input_segment_count,
        "pass1_chunk_count": result.pass1_chunk_count,
        "final_chunk_count": result.final_chunk_count,
        "chunks": [
            {
                "kind": chunk.kind,
                "pass_number": chunk.pass_number,
                "source_segment_refs": chunk.source_segment_refs,
                "token_count": chunk.token_count,
                "text_preview": chunk.text[:180],
            }
            for chunk in result.chunks
        ],
    }


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sampled_sections = sample_index_and_section_trefs(
        sefaria_project_path=SEFARIA_PROJECT_PATH,
        n=NUM_SAMPLES,
        seed=RANDOM_SEED,
        use_top_sections=USE_TOP_SECTIONS,
    )

    chunker = PatotChunker(api_key=api_key, config=CHUNKER_CONFIG)
    manifest = {
        "lang": LANG,
        "version_title": VERSION_TITLE,
        "num_samples": NUM_SAMPLES,
        "random_seed": RANDOM_SEED,
        "use_top_sections": USE_TOP_SECTIONS,
        "index_title_allowlist": INDEX_TITLE_ALLOWLIST,
        "samples": [],
    }

    print("Patot section sampler")
    print("=====================")
    print(f"num_samples={NUM_SAMPLES}")
    print(f"output_dir={OUTPUT_DIR}")
    print()

    for i, sample in enumerate(sampled_sections, start=1):
        index_title = sample["index_title"]
        tref = sample["tref"]
        segments = load_segment_records(
            sefaria_project_path=SEFARIA_PROJECT_PATH,
            tref=tref,
            lang=LANG,
            version_title=VERSION_TITLE,
        )
        result = chunker.chunk_segments(segments)

        pdf_path = OUTPUT_DIR / f"{i:02d}_{slugify(tref)}.pdf"
        write_debug_pdf(pdf_path, result, tref, LANG, CHUNKER_CONFIG)

        entry = build_manifest_entry(index_title, tref, result)
        entry["pdf_path"] = str(pdf_path)
        manifest["samples"].append(entry)

        print(f"[{i}] index_title={index_title}")
        print(f"    tref={tref}")
        print(f"    pdf={pdf_path}")
        print(f"    input_segment_count={result.input_segment_count}")
        print(f"    final_chunk_count={result.final_chunk_count}")

    manifest_path = OUTPUT_DIR / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print()
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
