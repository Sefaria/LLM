from .openai_commentary_scorer import CommentaryScorer
import os
from pathlib import Path
from sefaria_llm_interface.commentary_scoring import (
    CommentaryScoringInput,
    CommentaryScoringOutput,
)


def score_one_commentary(inp: CommentaryScoringInput) -> (
        CommentaryScoringOutput):
    scorer = CommentaryScorer(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    result = (scorer.
    process_commentary_by_content(
        commentary_ref=inp.commentary_ref,
        cited_refs=inp.cited_refs,
        commentary_text=inp.commentary_text
    )
    )
    if not result:
        return CommentaryScoringOutput(
            commentary_ref=inp.commentary_ref,
            ref_scores={},
            scores_explanation="",
            processed_datetime=None,
        )
    return CommentaryScoringOutput(
        commentary_ref=inp.commentary_ref,
        ref_scores=result.get(scorer.REF_SCORE_FIELD),
        scores_explanation=result.get(scorer.EXPLANATION_FIELD),
        processed_datetime=result.get(scorer.PROCESSED_AT_FIELD),
    )
