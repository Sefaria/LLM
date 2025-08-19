"""Commentary Scorer for evaluating Jewish text Commentaries.

This module provides functionality to score how well commentaries explain
their cited texts using OpenAI's language models.
"""

import json
import logging
import textwrap
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from sefaria_llm_interface.commentary_scoring import (
    CommentaryScoringInput,
    CommentaryScoringOutput,
)
logger = logging.getLogger(__name__)


class ExplainsFlag(IntEnum):
   """Binary flags for whether a commentary explains a cited text."""
   NOT_EXPLAINED = 0  # Commentary doesn't interpret the cited text
   EXPLAINED = 1      # Commentary provides interpretation/explanation


class RequestStatus(IntEnum):
   """Status codes for LLM processing requests."""
   SUCCESS = 1
   FAILURE = 0


class LanguageCode:
   """ISO 639-1 language codes for supported languages."""
   ENGLISH = "en"
   HEBREW = "he"
   DEFAULT = ENGLISH


class CommentaryScorer:
    """Scores how well commentaries explain their cited texts.

    This class uses OpenAI's language models to evaluate the quality of
    explanations provided by Jewish commentaries for their cited texts.

    Attributes:
       model (str): The OpenAI model to use for scoring
       max_prompt_tokens (int): Maximum tokens allowed in prompt
       token_margin (int): Reserved tokens for model response
       llm (ChatOpenAI): Initialized language model client
   """

    # Configuration constants for token management
    DEFAULT_MAX_OUTPUT_TOKENS = 4096  # Reserve for LLM response
    DEFAULT_MAX_INPUT_OUTPUT_TOKENS = 32000  # Total token budget
    DEFAULT_TOKEN_CHAR_RATIO = 3  # Fallback chars-per-token estimate

    # JSON response field names for structured output
    REF_SCORE_FIELD = "ref_scores"  # Binary scores per citation
    EXPLANATION_FIELD = "explanation"  # Rationale strings per citation
    LANGUAGE_FIELD = "language"  # Detected language code
    CITED_REF_FIELD = "cited_ref"  # Citation reference key
    PROCESSED_AT_FIELD = "processed_datetime"  # Processing timestamp

    # Valid explanation levels for score validation
    VALID_LEVELS: Set[int] = \
        {ExplainsFlag.NOT_EXPLAINED, ExplainsFlag.EXPLAINED}

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "gpt-4o-mini",
            max_prompt_tokens: int = DEFAULT_MAX_INPUT_OUTPUT_TOKENS,
            token_margin: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        """Initialize the commentary scorer with OpenAI client.
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var
            model: OpenAI model name (default: gpt-4o-mini for cost efficiency)
            max_prompt_tokens: Maximum tokens for input prompt (includes commentary + citations)
            token_margin: Reserved tokens for model response (ensures budget compliance)
        Raises:
            ValueError: If model is not supported or parameters are invalid
            Exception: If OpenAI client initialization fails
        """

        self.model = model
        self.max_prompt_tokens = max_prompt_tokens
        self.token_margin = token_margin

        try:
            # Initialize OpenAI client with deterministic settings for consistent scoring
            self.llm = ChatOpenAI(
                model_name=model,
                temperature=0,  # Deterministic output for consistent grading
                openai_api_key=api_key,
                model_kwargs={
                    "top_p": 0,  # No nucleus sampling
                    "frequency_penalty": 0,  # No frequency penalties
                    "presence_penalty": 0,  # No presence penalties
                    "seed": 42,  # Fixed seed for reproducibility
                },
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise

        logger.info(f"Initialized CommentaryScorer with model {model}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Failed to get exact token count: {e}")
            # Fallback to character-based estimation
            return max(1, len(text) // self.DEFAULT_TOKEN_CHAR_RATIO)

    def _validate_level(self, score: Any) -> int:
        try:
            score = int(score)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid reference score '{score}', defaulting to 0"
                )
            return ExplainsFlag.NOT_EXPLAINED

        if score not in self.VALID_LEVELS:
            clamped = ExplainsFlag.EXPLAINED if score >= 1 else ExplainsFlag.NOT_EXPLAINED
            logger.warning(
                f"Reference score {score} out of range, clamping to {clamped}"
                )
            return clamped

        return score

    def _invoke_llm(self, prompt: str, function_schema: Dict[str, Any]) \
            -> Dict[str, Any]:
        """Invoke the language model with function calling.
        """
        try:
            response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                functions=[function_schema],
                function_call={"name": function_schema["name"]}
            )
            function_call = getattr(response, "additional_kwargs", {}).get(
                "function_call"
                )
            if not function_call:
                raise ValueError("No function call found in LLM response")

            arguments = function_call.get("arguments", "{}")
            return json.loads(arguments)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response JSON: {e}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    def _build_function_schema(self, cited_keys: List[str]) -> Dict[str, Any]:
        if not cited_keys:
            raise ValueError("cited_keys cannot be empty")

        return {
            "name": "score_multiple_explanations",
            "description": "Binary labeling: does the commentary actually interpret/explain each cited text?",
            "parameters": {
                "type": "object",
                "properties": {
                    self.REF_SCORE_FIELD: {
                        "type": "object",
                        "properties": {
                            key: {"type": "integer","minimum": 0,"maximum": 1}
                            for key in cited_keys
                        },
                        "required": cited_keys,
                        "additionalProperties": False,
                    },
                    self.EXPLANATION_FIELD: {
                        "type": "object",
                        "properties": {
                            key: {
                                "type": "string",
                                "maxLength": 150,
                                "description": (
                                    "Brief rationale. Start with: "
                                    "\"Explained spans: '<phrase1>'; '<phrase2>'\" "
                                    "then 1–2 sentences why 0/1."
                                ),
                            }
                            for key in cited_keys
                        },
                        "required": cited_keys,
                        "additionalProperties": False,
                    },
                },
                "required": [self.REF_SCORE_FIELD,self.EXPLANATION_FIELD],
                "additionalProperties": False,
            },
        }

    def _create_failure_scoring_output(
            self,
            commentary_ref: str,
            processed_datetime: datetime,
            request_status_message: str
    ) -> CommentaryScoringOutput:
        """Create standardized failure output for error cases.
        Returns: CommentaryScoringOutput: Failure result with error details
        """
        logger.warning(request_status_message)
        return CommentaryScoringOutput(
            commentary_ref=commentary_ref,
            ref_scores={},
            scores_explanation={},
            processed_datetime=str(processed_datetime),
            request_status_message=request_status_message,
            request_status=RequestStatus.FAILURE
        )

    def _build_scoring_prompt(self, cited_refs: Dict[str, str], commentary_text: str) -> str:
        """Build the prompt for scoring commentary explanations.
        """
        refs_section = "\n".join(f"- {key}: {text}" for key, text in cited_refs.items())

        return textwrap.dedent(f"""You are labeling whether a commentary ACTUALLY EXPLAINS each cited text.

        COMMENTARY TEXT:
        {commentary_text}

        CITED TEXTS:
        {refs_section}

        TASK (binary per citation):
        Return {ExplainsFlag.EXPLAINED} if the commentary provides any substantive interpretation or explanation
        of ANY PART of the cited text (including methodological interpretation, e.g., reading a word
        as a symbol) — not just quoting or paraphrasing.

        Return {ExplainsFlag.NOT_EXPLAINED} if:
        • The citation is used for another goal (decorative, rhetorical, prooftext with no interpretation).
        • The commentary cites Source A only via Source B, but adds NO new interpretation of A beyond B.
          (Inherited interpretation does NOT count as explanation of A.)
        • It merely references or paraphrases without interpreting.

        Important:
        • If the commentary explains only PARTS of the citation, still return 1.
        • In your explanation, list the exact phrases from the cited text that ARE explained (if any),
          then give a concise rationale for {ExplainsFlag.NOT_EXPLAINED}/{ExplainsFlag.EXPLAINED}.

        RETURN JSON WITH:
        1. {self.REF_SCORE_FIELD}: object of {ExplainsFlag.NOT_EXPLAINED}/{ExplainsFlag.EXPLAINED} per citation key
        2. {self.EXPLANATION_FIELD}: object of brief rationales. Begin each value with:
           Explained spans: '<phrase1>'; '<phrase2>' (or 'None'), then 1–2 sentences of rationale.
        """)

    def process_commentary_by_content(
            self,
            commentary_text: str,
            cited_refs: Dict[str, str],
            commentary_ref: str = ""
    ) -> CommentaryScoringOutput:
        """Score how well a commentary explains its cited texts.
        """
        if not cited_refs:
            return self._create_failure_scoring_output(commentary_ref=commentary_ref,
                                                       processed_datetime=datetime.now(timezone.utc),
                                                       request_status_message=f"Commentary {commentary_ref} doesn't cite anything. ")

        if not commentary_text:
            return self._create_failure_scoring_output(commentary_ref=commentary_ref,
                                                       processed_datetime=datetime.now(timezone.utc),
                                                       request_status_message=f"Commentary {commentary_ref} is empty. ")

        if not commentary_text.strip():

            return self._create_failure_scoring_output(
                commentary_ref=commentary_ref,
                processed_datetime=datetime.now(timezone.utc),
                request_status_message=f"Commentary's {commentary_ref} text is empty "
                )

        token_count = self._count_tokens(commentary_text)
        max_allowed_tokens = self.max_prompt_tokens - self.token_margin

        if token_count > max_allowed_tokens:
            # TODO: add long commentary support
            return self._create_failure_scoring_output(commentary_ref=commentary_ref,
                                                       processed_datetime=datetime.now(timezone.utc),
                                                       request_status_message=(f"{commentary_ref}'s input too long "
                                      f"({token_count} tokens > {max_allowed_tokens} limit). "))

        logger.info(
            f"Processing commentary with {token_count} tokens, "
            f"{len(cited_refs)} citations"
        )
        try:
            # Build prompt and schema
            prompt = self._build_scoring_prompt(cited_refs, commentary_text)
            schema = self._build_function_schema(list(cited_refs.keys()))

            # Get LLM response
            raw_response = self._invoke_llm(prompt, schema)

            # Validate and normalize scores
            raw_scores = raw_response.get(self.REF_SCORE_FIELD, {})
            validated_scores = {
                key: self._validate_level(score)
                for key, score in raw_scores.items()
            }
            # Create structured result
            result = CommentaryScoringOutput(
                commentary_ref=commentary_ref,
                ref_scores=validated_scores,
                scores_explanation=raw_response.get(
                    self.EXPLANATION_FIELD, {}
                    ),
                processed_datetime=str(datetime.now(timezone.utc)),
                request_status_message="",
                request_status=RequestStatus.SUCCESS)

            explained = sum(validated_scores.values())
            total = len(validated_scores)
            logger.info(
                f"Scored commentary {commentary_ref}: explained {explained}/{total} "
                f"({(explained / total * 100 if total else 0):.0f}%)"
            )

            return result

        except Exception as e:
            return self._create_failure_scoring_output(
                commentary_ref=commentary_ref,
                processed_datetime=datetime.now(timezone.utc),
                request_status_message=f"Commentary {commentary_ref} scoring failed: {e}"
            )