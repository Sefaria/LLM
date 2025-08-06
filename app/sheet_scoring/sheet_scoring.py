from sheet_scoring.openai_sheets_scorer import SheetScorer
import os
from pathlib import Path
from sefaria_llm_interface.scoring_io import (
    SheetScoringInput,
    SheetScoringOutput,
)


def score_one_sheet(inp: SheetScoringInput) -> SheetScoringOutput:
    scorer = SheetScorer(
        api_key=os.getenv("OPENAI_API_KEY"))
    result = scorer.process_sheet_by_content(sheet=inp.sheet_content)
    if not result:
        return SheetScoringOutput(
            sheet_id=result[scorer.SHEET_ID_FIELD],
            ref_scores={},
            ref_levels={},
            title_interest_level=0,
            title_interest_reason="",
            language="",
            creativity_score=0,
            processed_datetime=None,
        )
    return SheetScoringOutput(
        sheet_id=result[scorer.SHEET_ID_FIELD],
        ref_scores=result[scorer.REF_SCORES_FIELD],
        ref_levels=result[scorer.REF_LEVELS_FIELD],
        title_interest_level=result[scorer.TITLE_INTEREST_FIELD],
        title_interest_reason=result[scorer.TITLE_INTEREST_REASON_FIELD],
        language=result[scorer.LANGUAGE_FIELD],
        creativity_score=result[scorer.CREATIVITY_SCORE_FIELD],
        processed_datetime=result["processed_datetime"].isoformat(),
    )
