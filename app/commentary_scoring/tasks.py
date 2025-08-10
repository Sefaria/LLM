from celery import shared_task
from commentary_scoring.commentary_scoring import score_one_commentary
from sefaria_llm_interface.commentary_scoring import (
    CommentaryScoringInput
)
from dataclasses import asdict


@shared_task(name='llm.score_commentary')
def score_sheet_task(raw_input: dict) -> dict:
    inp = CommentaryScoringInput(**raw_input)
    out = score_one_commentary(inp)
    return asdict(out)