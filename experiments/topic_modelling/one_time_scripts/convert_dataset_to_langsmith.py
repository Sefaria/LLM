import json
from pathlib import Path
from typing import Dict, Iterable

# --------------------------------------------------------------------------- #
# Config – adjust if you want other filenames
# --------------------------------------------------------------------------- #
SOURCE_PATH = Path("../evaluation_data/revised_gold.jsonl")
DEST_PATH   = Path("../evaluation_data/revised_gold_langsmith.jsonl")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> Iterable[Dict]:
    """Yield JSON objects, one per non-blank line."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line := line.strip():
                yield json.loads(line)


def transform(row: Dict) -> Dict:
    """Wrap into LangSmith's required schema."""
    return {
        "inputs":  {"ref": row["ref"]},
        "outputs": {"slugs": row["slugs"]},
    }


def write_jsonl(rows: Iterable[Dict], path: Path) -> None:
    """Write rows to path as JSON Lines."""
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"Source file {SOURCE_PATH} not found. "
            "Place your raw lines there or edit SOURCE_PATH."
        )

    transformed = (transform(r) for r in read_jsonl(SOURCE_PATH))
    write_jsonl(transformed, DEST_PATH)
    print(f"✅ Wrote LangSmith-formatted lines to {DEST_PATH.resolve()}")