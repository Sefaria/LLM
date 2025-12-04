from dataclasses import dataclass
from typing import Dict, Union
from datetime import datetime


@dataclass
class SheetScoringOutput:
    sheet_id: str
    processed_datetime: str
    language: str
    title_interest_level: int
    title_interest_reason: str
    creativity_score: float
    ref_levels: Dict[str, int]
    ref_scores: Dict[str, float]
    request_status: int
    request_status_message: str

    def __post_init__(self):
        if isinstance(self.processed_datetime, datetime):
            self.processed_datetime = self.processed_datetime.isoformat()