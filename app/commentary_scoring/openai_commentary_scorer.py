"""Commentary Scorer for evaluating Jewish text Commentaries.

This module provides functionality to score how well commentaries explain
their cited texts using OpenAI's language models.
"""

import json
import logging
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Union

import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from commentary_scoring.text_utils import to_plain_text

logger = logging.getLogger(__name__)


class ExplanationLevel(IntEnum):
    """Levels of explanation quality for commentary scoring."""

    NO_EXPLANATION = 0
    MINIMAL = 1
    MODERATE = 2
    SIGNIFICANT = 3
    COMPREHENSIVE = 4


class LanguageCode:
    """ISO 639-1 language codes."""

    ENGLISH = "en"
    HEBREW = "he"
    DEFAULT = ENGLISH


class CommentaryScorer:
    """Scores how well commentaries explain their cited texts.

    This class uses OpenAI's language models to evaluate the quality of
    explanations provided by Jewish commentaries for their cited texts.

    Attributes:
        model: The OpenAI model to use for scoring
        max_prompt_tokens: Maximum tokens allowed in prompt
        token_margin: Reserved tokens for model response
    """

    # Configuration constants
    DEFAULT_MAX_OUTPUT_TOKENS = 4096
    DEFAULT_MAX_INPUT_OUTPUT_TOKENS = 32000
    DEFAULT_TOKEN_CHAR_RATIO = 3

    # Response field names
    REF_SCORE_FIELD = "ref_scores"
    EXPLANATION_FIELD = "explanation"
    LANGUAGE_FIELD = "language"
    CITED_REF_FIELD = "cited_ref"
    PROCESSED_AT_FIELD = "processed_datetime"

    # Valid explanation levels
    VALID_LEVELS: Set[int] = {level.value for level in ExplanationLevel}

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "gpt-4o-mini",
            max_prompt_tokens: int = DEFAULT_MAX_INPUT_OUTPUT_TOKENS,
            token_margin: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        """Initialize the commentary scorer.
        Args:
            api_key: OpenAI API key. If None, will use environment variable
            model: OpenAI model name to use
            max_prompt_tokens: Maximum tokens for input prompt
            token_margin: Reserved tokens for model response
        Raises:
            ValueError: If model is not supported or parameters are invalid
        """

        self.model = model
        self.max_prompt_tokens = max_prompt_tokens
        self.token_margin = token_margin

        try:
            self.llm = ChatOpenAI(
                model_name=model,
                temperature=0, #Model temperature (0.0 for deterministic grading)
                openai_api_key=api_key,
                model_kwargs={
                    "top_p": 0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "seed": 42,
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

    def _validate_level(self,score: Any) -> int:
        """Validate and normalize explanation level score.
        """
        try:
            score = int(score)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid reference score '{score}', defaulting to 0"
                )
            return ExplanationLevel.NO_EXPLANATION

        if score not in self.VALID_LEVELS:
            clamped = max(
                ExplanationLevel.NO_EXPLANATION,
                min(score,ExplanationLevel.COMPREHENSIVE)
            )
            logger.warning(
                f"Reference score {score} out of range, clamping to {clamped}"
            )
            return clamped

        return score

    def _invoke_llm(
            self,
            prompt: str,
            function_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
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

    def _build_function_schema(self, cited_keys: List[str]) -> Dict[str,Any]:
        """Build JSON schema for function calling.

        Args:
            cited_keys: List of citation keys to score

        Returns:
            JSON schema for the scoring function
        """
        if not cited_keys:
            raise ValueError("cited_keys cannot be empty")

        return {
            "name": "score_multiple_explanations",
            "description": "Score how well a commentary explains each cited text",
            "parameters": {
                "type": "object",
                "properties": {
                    self.REF_SCORE_FIELD: {
                        "type": "object",
                        "properties": {
                            key: {
                                "type": "integer",
                                "minimum": ExplanationLevel.NO_EXPLANATION,
                                "maximum": ExplanationLevel.COMPREHENSIVE
                            }
                            for key in cited_keys
                        },
                        "required": cited_keys,
                        "additionalProperties": False
                    },
                    self.EXPLANATION_FIELD: {
                        "type": "object",
                        "properties": {
                            key: {
                                "type": "string",
                                "maxLength": 200,
                                "description": f"Explanation for {key} score (1-2 sentences)"
                            }
                            for key in cited_keys
                        },
                        "required": cited_keys,
                        "additionalProperties": False
                    }
                },
                "required": [
                    self.REF_SCORE_FIELD,
                    self.EXPLANATION_FIELD
                ],
                "additionalProperties": False
            }
        }

    def _build_scoring_prompt(
            self,
            cited_refs: Dict[str,str],
            commentary_text: str
    ) -> str:
        """Build the prompt for scoring commentary explanations.

        Args:
            cited_refs: Mapping of reference keys to cited texts
            commentary_text: The commentary text to evaluate

        Returns:
            Formatted prompt string
        """
        refs_section = "\n".join(
            f"- {key}: {text}" for key,text in cited_refs.items()
        )

        return f"""You are an expert evaluator of Jewish commentary quality.

COMMENTARY TEXT:
{commentary_text}

CITED TEXTS TO EVALUATE:
{refs_section}

TASK: For each cited text, score (0-4) how well the commentary explains it:

SCORING SCALE:
{ExplanationLevel.NO_EXPLANATION}: NO EXPLANATION - Citation used for unrelated point
{ExplanationLevel.MINIMAL}: MINIMAL - Text merely paraphrased or mentioned
{ExplanationLevel.MODERATE}: MODERATE - Commentary shares theme but doesn't explain text
{ExplanationLevel.SIGNIFICANT}: SIGNIFICANT - Citation is main focus with meaningful explanation
{ExplanationLevel.COMPREHENSIVE}: COMPREHENSIVE - Deep, thorough explanation that fully illuminates the text

RETURN JSON WITH:
1. {self.REF_SCORE_FIELD}: Object mapping each citation key to score (0-4)
2. {self.EXPLANATION_FIELD}: Object mapping each key to brief explanation (1-2 sentences)

Be precise and consistent in your scoring."""

    def process_commentary_by_content(
            self,
            commentary_text: Union[List[str],str],
            cited_refs: Dict[str,str],
            commentary_ref: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Score how well a commentary explains its cited texts.
        """
        if not cited_refs:
            logger.info(
                f"Commentary {commentary_ref} doesn't cite anything. "
                f"Defaulting to None"
            )
            return None

        if not commentary_text:
            logger.info(
                f"Commentary {commentary_ref} is empty. "
                f"Defaulting to None"
            )
            return None

            # Convert commentary text to string format
        if isinstance(commentary_text,list):
            commentary_text_str = to_plain_text(commentary_text)
        else:
            commentary_text_str = str(commentary_text)

        if not commentary_text_str.strip():
            logger.warning(f"Commentary's {commentary_ref} text is empty "
                           f"after processing")

            return None

        token_count = self._count_tokens(commentary_text_str)
        max_allowed_tokens = self.max_prompt_tokens - self.token_margin

        if token_count > max_allowed_tokens:
            # TODO: add long commentary support
            logger.warning(
                f"{commentary_ref}'s input too long "
                f"({token_count} tokens > {max_allowed_tokens} limit). "
                "Skipping scoring."
            )
            return None

        logger.info(
            f"Processing commentary with {token_count} tokens, "
            f"{len(cited_refs)} citations"
        )

        try:
            # Build prompt and schema
            prompt = self._build_scoring_prompt(cited_refs, commentary_text_str)
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
            result = {
                self.REF_SCORE_FIELD: validated_scores,
                self.EXPLANATION_FIELD: raw_response.get(
                    self.EXPLANATION_FIELD, {}
                    ),
                self.PROCESSED_AT_FIELD: datetime.now(timezone.utc),
            }

            logger.info(
                f"Successfully scored commentary {commentary_ref}. "
                f"Average score: {sum(validated_scores.values()) / len(validated_scores):.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Commentary {commentary_ref} scoring failed: {e}")
            return None
