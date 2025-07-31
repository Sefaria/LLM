from dataclasses import dataclass
from typing import Dict, Union, List
from datetime import datetime


@dataclass
class SheetScoringOutput:
    sheet_id:str
    processed_at: str
    language: str
    title_interest_level: int
    title_interest_reason: str
    creativity_score:float
    ref_levels: Dict[str, int]
    ref_scores: Dict[str, float]


    def __init__(self,
                 sheet_id: str,
                 ref_scores: Dict[str, float],
                 ref_levels:Dict[str, int],
                 processed_at: Union[str, datetime],
                 language: str,
                 creativity_score: float,
                 title_interest_level: int,
                 title_interest_reason: str):
        self.ref_scores = ref_scores
        self.sheet_id = sheet_id
        self.processed_at = processed_at.isoformat() if isinstance(
            processed_at, datetime
        ) else processed_at
        self.ref_levels = ref_levels
        self.creativity_score = creativity_score
        self.language = language
        self.title_interest_level = title_interest_level
        self.title_interest_reason = title_interest_reason