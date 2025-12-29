from dataclasses import dataclass
from typing import Dict


@dataclass
class CommentaryScoringInput:
    commentary_text: str
    cited_refs: Dict[str, str]
    commentary_ref: str

