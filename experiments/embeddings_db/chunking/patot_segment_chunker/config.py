from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ChunkerConfig:
    model: str = "gemini-embedding-001"
    setup: str = "retrieval"
    dim: int = 1536
    sim: str = "dot"
    doc: str = "raw_text"
    query: str = "raw_query"
    norm: bool = True
    score_threshold: Optional[float] = None
    threshold_adjustment: float = 0.01
    dynamic_threshold: bool = True
    window_size: int = 5
    min_split_tokens: int = 200
    max_split_tokens: int = 500
    split_tokens_tolerance: int = 10
    tokenizer_model: str = "dicta-il/BEREL_3.0"
    strip_hebrew_niqqud: bool = True
    stanza_model_dir: str = "/Users/yon/stanza_resources"
    enforce_hard_max_in_pass3: bool = True
    debug: bool = True


@dataclass(frozen=True)
class SefariaSourceConfig:
    sefaria_project_path: Path
    tref: str
    lang: str = "he"
    version_title: Optional[str] = None
