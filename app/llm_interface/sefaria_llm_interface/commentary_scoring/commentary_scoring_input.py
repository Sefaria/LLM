from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CommentaryScoringInput:
    commentary_text: str
    cited_refs: Dict[str, str]
    commentary_ref: str

