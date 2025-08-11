from dataclasses import dataclass
from typing import Dict, Union
from datetime import datetime


@dataclass
class CommentaryScoringOutput:
    commentary_ref: str
    ref_scores: Dict[str, int]
    scores_explanation: Dict[str, str]
    processed_datetime: str
    request_status_message: str
    request_status: int

    def __post_init__(self):
        if isinstance(self.processed_datetime, datetime):
            self.processed_datetime = self.processed_datetime.isoformat()