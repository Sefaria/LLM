from celery import shared_task
from sheet_scoring.sheet_scoring import score_one_sheet
from sefaria_llm_interface.sheet_scoring import (
    SheetScoringInput
)
from dataclasses import asdict


@shared_task(name='llm.score_sheet', ignore_result=True)
def score_sheet_task(raw_input: dict) -> dict:
    # ignore_result only skips storing the return value in the result backend (one Redis SET + expiry DEL per call).
    # The value is still passed to the next task in the chain (web.save_sheets_score), and nothing polls this task's id.
    inp = SheetScoringInput(**raw_input)
    out = score_one_sheet(inp)
    return asdict(out)