from celery import shared_task
from .sheet_scoring import score_one_sheet
from sefaria_llm_interface.scoring_io import (
    SheetScoringInput
)
from dataclasses import asdict


@shared_task(name='llm.score_sheet')
def score_sheet_task(raw_input:dict) -> dict:
    inp = SheetScoringInput(**raw_input)
    out = score_one_sheet(inp)
    return asdict(out)