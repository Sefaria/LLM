from .openai_commentary_scorer import CommentaryScorer
import os
from pathlib import Path
# TODO: change the imports when compile package
from app.llm_interface.sefaria_llm_interface.commentary_scoring import (
    CommentaryScoringInput,
    CommentaryScoringOutput,
)

def score_one_commentary(inp: CommentaryScoringInput) -> (
        CommentaryScoringOutput):
    scorer = CommentaryScorer(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return scorer.process_commentary_by_content(
        commentary_ref=inp.commentary_ref,
        cited_refs=inp.cited_refs,
        commentary_text=inp.commentary_text
    )