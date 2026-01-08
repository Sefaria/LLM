from sheet_scoring.openai_sheets_scorer import SheetScorer
import os
from sefaria_llm_interface.sheet_scoring import (
    SheetScoringInput,
    SheetScoringOutput,
)


def score_one_sheet(inp: SheetScoringInput) -> SheetScoringOutput:
    with SheetScorer(api_key=os.getenv("OPENAI_API_KEY")) as scorer:
        return scorer.process_sheet_by_content(sheet_id=inp.sheet_id,
                                               title=inp.title,
                                               sources=inp.sources,
                                               expanded_refs=inp.expanded_refs)