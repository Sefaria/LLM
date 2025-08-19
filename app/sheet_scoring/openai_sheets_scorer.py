import json
import logging
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
import textwrap
import tiktoken
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from sheet_scoring.text_utils import sheet_to_text_views
from sefaria_llm_interface.sheet_scoring import SheetScoringOutput
# Configure logging
logger = logging.getLogger(__name__)


class IncompleteScoreError(Exception):
    """Raised when LLM JSON is valid but doesn’t cover every reference."""
    pass


class RequestStatusOptions(IntEnum):
    """Enumeration for tracking the status of LLM processing requests."""
    SUCCESS = 1
    FAILURE = 0


class ScoreLevel(IntEnum):
    """Reference discussion and title interest levels."""
    NOT_DISCUSSED = 0
    MINIMAL = 1
    MODERATE = 2
    SIGNIFICANT = 3
    CENTRAL = 4


class LanguageCode:
    """Supported language codes."""
    ENGLISH = 'en'
    HEBREW = 'he'
    DEFAULT = ENGLISH


class SheetScorer:
    """
    Scores Jewish study sheets for reference relevance and title interest using LLMs,
    computes creativity score based on percentage of user generated content.

    This class processes sheets from MongoDB, analyzes their content using OpenAI's GPT models,
    and assigns scores for how well each reference is discussed and how interesting
    the sheet title is to users.
    """

    # Configuration constants -
    # DEFAULT_MAX_INPUT_OUTPUT_TOKENS: total
    # tokens (prompt+response) we’ll send in one API call. Lowering this
    # shrinks your available context; raising it risks exceeding the model’s
    # limit.
    # DEFAULT_MAX_OUTPUT_TOKENS: cap on how many tokens the model
    # may generate. If you set this too low, responses may be cut off; too
    # high wastes quota.
    # DEFAULT_CHUNK_SIZE: how many references to score
    # in each batch. Larger chunks use more context (better global view) but
    # may exceed token budgets.
    # MAX_CHUNK_OVERLAP: how many refs to repeat
    # between chunks. More overlap reduces missing-edge-case errors at the
    # cost of redundant API calls.
    # DEFAULT_MAX_REFS_TO_PROCESS: total refs
    # before falling back to equal-distribution scoring. Hitting this limit
    # skips heavy LLM work to avoid runaway costs. -
    # DEFAULT_TOKEN_CHAR_RATIO: fallback characters‐per‐token estimate when
    # encoding fails. Tweak if you find your actual token counts diverge
    # significantly from this estimate.
    DEFAULT_MAX_OUTPUT_TOKENS = 16384
    DEFAULT_CHUNK_SIZE = 80
    DEFAULT_MAX_INPUT_OUTPUT_TOKENS = 128000
    DEFAULT_MAX_REFS_TO_PROCESS = 800
    DEFAULT_TOKEN_CHAR_RATIO = 3
    MAX_CHUNK_OVERLAP = 10
    # Database field names
    REF_SCORES_FIELD = "ref_scores"
    REF_LEVELS_FIELD = "ref_levels"
    TITLE_INTEREST_FIELD = "title_interest_level"
    LANGUAGE_FIELD = "language"
    TITLE_INTEREST_REASON_FIELD = 'title_interest_reason'
    PROCESSED_DATETIME_FIELD = "processed_datetime"
    CREATIVITY_SCORE_FIELD = 'creativity_score'

    # Valid score levels
    VALID_LEVELS: Set[int] = {level.value for level in ScoreLevel}

    def __init__(
            self,
            api_key: Optional[str],
            model: str = "gpt-4o-mini",
            max_prompt_tokens: int = DEFAULT_MAX_INPUT_OUTPUT_TOKENS,
            token_margin: int = DEFAULT_MAX_OUTPUT_TOKENS,
            max_ref_to_process: int = DEFAULT_MAX_REFS_TO_PROCESS,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.max_prompt_tokens = max_prompt_tokens
        self.token_margin = token_margin
        self.model = model
        self.chunk_size = chunk_size
        self.max_ref_to_process = max_ref_to_process
        self.llm = self._create_json_llm(api_key,model)
        self.summarizer = self._create_text_llm(api_key,model)

    def _create_json_llm(self, api_key: str, model: str) -> ChatOpenAI:
        """Create LLM client for JSON responses."""
        return ChatOpenAI(
            model=model,
            temperature=0,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            seed=42,
            api_key=api_key,
        )

    def _create_text_llm(self, api_key: str, model: str) -> ChatOpenAI:
        """Create LLM client for text responses."""
        return ChatOpenAI(
            model=model,
            temperature=0,
            model_kwargs={"response_format": {"type": "text"}},
            api_key=api_key,
        )

    def _invoke_llm_with_function(self, prompt: str,
                                  function_schema: Dict[str, Any]) -> (
            Dict)[str, Any]:
        """Invoke LLM using function calling instead of JSON mode."""
        response = self.llm.invoke(
            [HumanMessage(content=prompt)],
            functions=[function_schema],
            function_call={"name": function_schema["name"]}
        )

        function_call = getattr(response, "additional_kwargs", {}).get(
            "function_call"
            )
        if function_call:
            return json.loads(function_call["arguments"])

        raise ValueError("No function call in response")

    def _get_reference_scoring_function_schema(self, expanded_refs: List[str]) -> \
            Dict[str, Any]:
        """Create function schema for reference scoring with exact reference
        names."""
        return {
            "name": "score_references",
            "description": "Score how well each reference is "
                           "discussed in the sheet",
            "parameters": {
                "type": "object",
                "properties": {
                    self.REF_LEVELS_FIELD: {
                        "type": "object",
                        "description": "Scores for each reference (0-4 scale)",
                        "properties": {
                            ref_name: {
                                "type": "integer",
                                "description": f"Discussion level for {ref_name}",
                                "minimum": 0,
                                "maximum": 4
                            }
                            for ref_name in expanded_refs
                        },
                        "required": expanded_refs,
                        "additionalProperties": False
                    }
                },
                "required": [self.REF_LEVELS_FIELD],
                "additionalProperties": False
            }
        }

    def _get_title_scoring_schema(self) -> Dict[str, Any]:
        """Create function schema for both reference and title scoring."""
        return {
            "name": "score_title",
            "description": "Score title interest for a Jewish study sheet",
            "parameters": {
                "type": "object",
                "properties": {
                    self.LANGUAGE_FIELD: {
                        "type": "string",
                        "description": "ISO-639-1 title language code",
                    },
                    self.TITLE_INTEREST_FIELD: {
                        "type": "integer",
                        "description": "How interesting the title is to "
                                       "users (0-4 scale)",
                        "minimum": 0,
                        "maximum": 4
                    },
                    self.TITLE_INTEREST_REASON_FIELD: {
                        "type": "string",
                        "description": "Brief explanation of title interest "
                                       "score (max 20 words)",
                        "maxLength": 100
                    }
                },
                "required": [self.LANGUAGE_FIELD, self.TITLE_INTEREST_FIELD,
                             self.TITLE_INTEREST_REASON_FIELD],
                "additionalProperties": False
            }
        }

    def _get_full_scoring_function_schema(self, expanded_refs: List[str]) -> (
            Dict)[str, Any]:
        """Create function schema for both reference and title scoring."""
        return {
            "name": "score_sheet",
            "description": "Score references and title interest for a Jewish "
                           "study sheet",
            "parameters": {
                "type": "object",
                "properties": {
                    self.LANGUAGE_FIELD: {
                        "type": "string",
                        "description": "# ISO‑639‑1 code inferred from "
                                       "*original user‑written* content",
                    },
                    self.REF_LEVELS_FIELD: {
                        "type": "object",
                        "description": "Scores for each reference (0-4 scale)",
                        "properties": {
                            ref_name: {
                                "type": "integer",
                                "description": f"Discussion level for {ref_name}",
                                "minimum": 0,
                                "maximum": 4
                            }
                            for ref_name in expanded_refs
                        },
                        "required": expanded_refs,
                        "additionalProperties": False
                    },
                    self.TITLE_INTEREST_FIELD: {
                        "type": "integer",
                        "description": "How interesting the title is to "
                                       "users (0-4 scale)",
                        "minimum": 0,
                        "maximum": 4
                    },
                    self.TITLE_INTEREST_REASON_FIELD: {
                        "type": "string",
                        "description": "Brief explanation of title interest "
                                       "score (max 20 words)",
                        "maxLength": 100
                    }
                },
                "required": [self.LANGUAGE_FIELD, self.REF_LEVELS_FIELD,
                             self.TITLE_INTEREST_FIELD,
                             self.TITLE_INTEREST_REASON_FIELD],
                "additionalProperties": False
            }
        }

    @staticmethod
    def chunk_list(lst: List[Any], n: int) -> Iterator[List[Any]]:
        """Yield successive n‑sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    def _count_tokens(self, text: str) -> int:
        """Rough token count; if no encoder, fall back to char heuristic."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except (KeyError, ValueError) as e:
            logger.warning(
                f"Could not get encoding for model {self.model}: {e}"
            )
            return len(text) // self.DEFAULT_TOKEN_CHAR_RATIO

    def _invoke_llm(self, prompt: str) -> Dict[str, Any]:
        """Invoke LLM with prompt and parse JSON response."""
        response = self.llm([HumanMessage(content=prompt)])
        return json.loads(response.content)

    def _create_title_only_prompt_function(self, sheet_title: str) -> str:
        return textwrap.dedent(
            f"""You are scoring THE TITLE of a Jewish study sheet for how interesting it would be to users.
    
            SHEET TITLE:
            {sheet_title}
        
            TASK: Return JSON with keys `title_interest_level` (0-4) and `title_interest_reason` ( < 20 words). 
            Choose a higher score when the title:
        
            Title interest level (int 0–4):
              0: Not interesting / off‑topic for users
              1: Slight relevance, low pull
              2: Somewhat interesting; user might skim
              3: Interesting; user likely to open
              4: Very compelling / must‑open
            """)

    def _create_chunk_prompt_for_function(self, sheet_content: str,
                                          expanded_refs: List[str]) -> str:
        """Create prompt for function calling (no JSON format instructions
        needed)."""
        refs_md = "\n".join(f"- {r}" for r in expanded_refs)
        return textwrap.dedent(
            f"""
            You are analyzing a Jewish study sheet. Rate how much each listed reference 
            is discussed or central in the sheet.

            SHEET CONTENT:
            {sheet_content}

            REFERENCES TO EVALUATE:
            {refs_md}

            Scoring Scale (0-4):
              {ScoreLevel.NOT_DISCUSSED}: Quoted only, no discussion
              {ScoreLevel.MINIMAL}: Mentioned only through neighboring verses
              {ScoreLevel.MODERATE}: Moderate discussion (some commentary)
              {ScoreLevel.SIGNIFICANT}: Significant discussion (substantial commentary)
              {ScoreLevel.CENTRAL}: Central focus of sheet

            Score each reference based on how thoroughly it's discussed in the content."""
            )

    def _create_final_chunk_prompt_for_function(self, sheet_content: str,
                                                expanded_refs: List[str],
                                                sheet_title: str) -> str:
        """Create prompt for final chunk with title scoring using function
        calling."""
        sheet_title_clean = sheet_title.strip() or "(untitled)"
        refs_md = "\n".join(f"- {r}" for r in expanded_refs)

        return textwrap.dedent(f"""
            Analyze this Jewish study sheet and provide two types of scores:
            
            SHEET TITLE: {sheet_title_clean}
            
            SHEET CONTENT:
            {sheet_content}
            
            REFERENCES TO EVALUATE:
            {refs_md}
            
            TASKS:
            1. Reference Discussion Scoring (0-4):
              {ScoreLevel.NOT_DISCUSSED}: Quoted only, no discussion
              {ScoreLevel.MINIMAL}: Mentioned only through neighboring verses
              {ScoreLevel.MODERATE}: Moderate discussion (some commentary)
              {ScoreLevel.SIGNIFICANT}: Significant discussion (substantial commentary)
              {ScoreLevel.CENTRAL}: Central focus of sheet
            
            2. Title Interest Scoring (0-4):
               0: Not interesting/off-topic
               1: Slight relevance, low appeal
               2: Somewhat interesting; user might skim
               3: Interesting; user likely to open
               4: Very compelling/must-open
            
            Infer the language from the original user-written content.
            """)

    def _validate_score_level(self, score: Any,
                              field_name: str = "score") -> int:
        """Validate and normalize score to valid range."""
        if score not in self.VALID_LEVELS:
            try:
                score = int(score)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid {field_name}: {score}, defaulting to 0"
                )
                return ScoreLevel.NOT_DISCUSSED

            if score not in self.VALID_LEVELS:
                clamped = max(
                    ScoreLevel.NOT_DISCUSSED,
                    min(ScoreLevel.CENTRAL, score)
                )
                logger.warning(
                    f"{field_name} {score} out of range, clamping to {clamped}"
                )
                return clamped

        return score

    def _sheet_to_text(
        self,
        no_quotes_content: str,
        full_content: str,
        max_tokens: int,
        add_full_commentary: bool
    ) -> str:
        """
        Build a text snapshot of the sheet with an *all‑or‑nothing* rule:
        • Always include every bit of author commentary.
        • Append *all* canonical quotations only if the whole bundle still
          fits into `max_tokens`.
        """
        comm_tokens = self._count_tokens(no_quotes_content)
        # Commentary alone is already bigger than the budget → truncate & quit
        full_tokens = self._count_tokens(full_content)
        if add_full_commentary:
            if full_tokens <= max_tokens:
                logger.info("Sending to LLM sheet with quotations")
                return full_content

        if comm_tokens >= max_tokens:
            logger.info("Truncating user commentaries")
            return self._truncate_to_token_budget(no_quotes_content, max_tokens)
        logger.info("Sending to LLM sheet without quotations text")
        return no_quotes_content

    def _get_title_info(self,sheet_title: str) -> Dict[str, Any]:
        """Obtain title-interest score ONLY (used when no content)."""
        prompt = self._create_title_only_prompt_function(sheet_title)
        try:
            function_schema = self._get_title_scoring_schema()
            data = self._invoke_llm_with_function(prompt, function_schema)
            title_level = self._validate_score_level(
                data.get(self.TITLE_INTEREST_FIELD),
                self.TITLE_INTEREST_FIELD
            )

            return {
                self.TITLE_INTEREST_FIELD:
                    title_level,
                self.TITLE_INTEREST_REASON_FIELD:
                    data.get(self.TITLE_INTEREST_REASON_FIELD, ""),
                self.LANGUAGE_FIELD: data.get(
                    self.LANGUAGE_FIELD, LanguageCode.DEFAULT
                ),
            }
        except Exception as e:
            logger.error(f"Title-only GPT attempt failed: {e}")
            return {
                self.TITLE_INTEREST_FIELD: ScoreLevel.NOT_DISCUSSED,
                self.TITLE_INTEREST_REASON_FIELD: "LLM error",
                self.LANGUAGE_FIELD: LanguageCode.DEFAULT
            }

    def _normalize_scores_to_percentages(
            self,
            sheet_tokens: int,
            score_levels: Dict[str, int],
            beta: float = 1500  # token mass where no penalty
    ) -> Dict[str, float]:
        """Convert reference scores to percentages with size penalty
        for shorter sheets."""

        total_level = sum(score_levels.values()) or 1
        size_factor = min(1.0, sheet_tokens / beta)  # clamp to 1

        # small sheets (few tokens) → size_factor < 1 → percentages shrink
        percentages = {
            ref: round(level * 100 / total_level * size_factor, 2)
            for ref, level in score_levels.items()
        }

        norm = sum(percentages.values()) or 1
        percentages = {r: round(v * 100 / norm, 2) for r, v in
                       percentages.items()}
        return percentages

    def _grade_refs_resilient(
            self,
            content: str,
            refs: List[str],
            *,
            with_title: bool = False,
            sheet_title: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
        """
        Fault-tolerant reference scoring using divide-and-conquer strategy.
        Attempts to score all references at once via LLM. If that fails
        (due to incomplete responses),
        recursively splits the reference list in half and scores each
        subset separately until all references have scores.
        This prevents total failure when the LLM struggles with large
        reference lists or encounters transient errors.

        """
        if not refs:
            return {}, {}

        try:
            if with_title:
                prompt = self._create_final_chunk_prompt_for_function(
                    content, refs, sheet_title
                )
                function_schema = self._get_full_scoring_function_schema(refs)
            else:
                prompt = self._create_chunk_prompt_for_function(content, refs)
                function_schema = self._get_reference_scoring_function_schema(
                    refs
                )
            data, scores = self._get_gpt_ref_scores_function(
                prompt, function_schema, refs
                )
            return data, scores
        except Exception:
            pass

        # fallback branch
        if len(refs) == 1:  # nothing left to split
            return {}, {refs[0]: ScoreLevel.NOT_DISCUSSED}

        mid = len(refs) // 2
        ld, ls = self._grade_refs_resilient(
            content, refs[:mid],
            with_title=with_title,
            sheet_title=sheet_title
        )
        rd, rs = self._grade_refs_resilient(
            content, refs[mid:],
            with_title=with_title,
            sheet_title=sheet_title
        )
        merged_scores = {**ls, **rs}
        merged_data = ld or rd
        return merged_data, merged_scores

    def _get_gpt_ref_scores_function(self, prompt: str, function_schema,
                                     expected_refs: List[str]):
        """Calls the LLM with structured function schema, validates all
        returned scores are in valid range (0-4), handles missing references,
        and ensures exactly the expected references are scored."""
        try:
            data = self._invoke_llm_with_function(prompt, function_schema)
            chunk_scores = data.get(self.REF_LEVELS_FIELD, {})
            validated_scores = {}
            for ref, score in chunk_scores.items():
                validated_scores[ref] = self._validate_score_level(
                    score, f"ref_score[{ref}]"
                )

            # Check for missing references and assign default scores (0)
            missing_refs = set(expected_refs) - set(validated_scores.keys())
            if missing_refs:
                logger.warning(
                    f"GPT didn't return scores for {len(missing_refs)} "
                )
                if len(missing_refs) < 5:
                    logger.warning(f"Defaulting missing scores to zeros")
                    for ref in missing_refs:
                        validated_scores[ref] = ScoreLevel.NOT_DISCUSSED

                else:
                    raise IncompleteScoreError(
                        f"Missing {len(missing_refs)} references"
                    )

            # Ensure we only include expected references (in case GPT
            # returned extras)
            final_scores = {
                ref: validated_scores.get(ref, ScoreLevel.NOT_DISCUSSED) for ref
                in expected_refs}

            data[self.REF_SCORES_FIELD] = final_scores
            return data, final_scores

        except IncompleteScoreError:
            raise

        except Exception as e:
            logger.error(f"Chunk GPT failed: {e}")
            return None

    def _last_regular_start(self, n: int, chunk: int, overlap: int) -> int:
        """
        Return the index where the *final* chunk (with title) should start.
        If the total length fits into one chunk plus the allowed overlap,
        analyse everything together (start = 0).
        """
        if n <= chunk + overlap:
            return 0
        step = chunk - overlap
        return max(0, n - chunk) if step <= 0 else (n - chunk)

    def _process_reference_chunks(
            self,
            content: str,
            expanded_refs: List[str]
    ) -> Optional[Dict[str, int]]:
        """Process reference chunks in batches."""
        ref_scores: Dict[str, int] = {}

        last_chunk_start = self._last_regular_start(
            len(expanded_refs), self.chunk_size, self.MAX_CHUNK_OVERLAP
        )

        for chunk in self.chunk_list(
                expanded_refs[:last_chunk_start], self.chunk_size
        ):
            # prompt = self._create_chunk_prompt(content,chunk)
            _, chunk_scores = self._grade_refs_resilient(
                content=content,
                refs=chunk,
                with_title=False
            )
            if chunk_scores is None:
                return None
            ref_scores.update(chunk_scores)

        return ref_scores

    def _process_final_chunk_with_title(
            self,
            content: str,
            expanded_refs: List[str],
            title: str,
    ) -> Optional[Dict[str, Any]]:
        """Process final chunk and get title scores."""
        start = self._last_regular_start(
            len(expanded_refs), self.chunk_size, self.MAX_CHUNK_OVERLAP
        )
        final_chunk = expanded_refs[start:]

        # prompt = self._create_final_chunk_prompt(content,final_chunk,title)
        result = self._grade_refs_resilient(
            content=content,
            refs=final_chunk,
            with_title=True,
            sheet_title=title
        )

        if result is None:
            return None

        data, _ = result
        return data

    def get_gpt_scores(
            self,
            content: str,
            expanded_refs: List[str],
            title: str,
    ) -> Optional[Dict[str, Any]]:
        """Get GPT scores for references and title."""
        # Process reference chunks
        ref_scores = self._process_reference_chunks(content, expanded_refs)
        if ref_scores is None:
            return None

        # Process final chunk with title
        final_data = self._process_final_chunk_with_title(
            content, expanded_refs, title
        )
        if final_data is None:
            return None

        # Combine scores
        final_chunk_scores = final_data.get(self.REF_SCORES_FIELD, {})
        ref_scores.update(final_chunk_scores)

        # # Normalize to percentages
        score_percentages = self._normalize_scores_to_percentages(
            score_levels=ref_scores,
            sheet_tokens=self._count_tokens(content)
        )

        # Validate title score
        title_level = self._validate_score_level(
            final_data.get(self.TITLE_INTEREST_FIELD),
            self.TITLE_INTEREST_FIELD
        )

        return {
            self.LANGUAGE_FIELD: final_data.get(
                self.LANGUAGE_FIELD, LanguageCode.DEFAULT
            ),
            self.REF_LEVELS_FIELD: ref_scores,
            self.REF_SCORES_FIELD: score_percentages,
            self.TITLE_INTEREST_FIELD: title_level,
            self.TITLE_INTEREST_REASON_FIELD: final_data.get(
                self.TITLE_INTEREST_REASON_FIELD, ""
            ),
        }

    def _truncate_to_token_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget using LLM summarization."""
        if self._count_tokens(text) <= max_tokens:
            return text
        try:
            prompt = f"""
            Compress the following commentary to ≤ {max_tokens} tokens.
            Keep every reference tag like "Genesis 1:1" or "Exodus 2:5".
            Use clear sentences; preserve main ideas.

            {text}
            """
            summary = self.summarizer(
                [HumanMessage(content=prompt)]
            ).content.strip()

            if self._count_tokens(summary) <= max_tokens:
                return summary
            else:
                # Fallback: character-based truncation
                return summary[:max_tokens * self.DEFAULT_TOKEN_CHAR_RATIO]

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: character-based truncation
            return text[:max_tokens * self.DEFAULT_TOKEN_CHAR_RATIO]

    def create_failure_output(self, sheet_id: str, request_status_message: str) -> (
            SheetScoringOutput):
        """Create a standardized failure output when sheet processing cannot
        be completed."""
        return SheetScoringOutput(
            sheet_id=sheet_id,
            processed_datetime=str(datetime.utcnow()),
            language="",
            title_interest_level=0,
            title_interest_reason="",
            creativity_score=0,
            ref_levels={},
            ref_scores={},
            request_status=RequestStatusOptions.FAILURE,
            request_status_message=request_status_message
            )

    def process_sheet_by_content(self,
                                 sheet_id: str,
                                 expanded_refs: List[str],
                                 title: str,
                                 sources: List[Dict[str, Union[str, Dict[str, str]]]],
                                 add_full_commentary=False) -> SheetScoringOutput:
        """Score a single sheet based on its content."""
        if not expanded_refs:
            request_status_message = f"No expanded refs for sheet {sheet_id}, skipping"
            logger.info(request_status_message)
            return self.create_failure_output(sheet_id,
                                              request_status_message=request_status_message)
        text_views = sheet_to_text_views(title=title, sources=sources, default_lang=LanguageCode.DEFAULT)
        no_quotes_content = text_views["no_quotes"]
        full_content = text_views["with_quotes"]
        has_original = text_views["has_original"]
        creativity_score = text_views["creativity_score"]

        # Check for original content and reference limits
        if (not has_original or
                len(expanded_refs) > self.max_ref_to_process):
            logger.info(f"Sheet {sheet_id}: using equal distribution")
            score_percentages = {ref: 0 for ref in expanded_refs}
            title_info = self._get_title_info(title)

            return SheetScoringOutput(sheet_id=sheet_id,
                                      ref_levels=score_percentages,
                                      ref_scores=score_percentages,
                                      processed_datetime=str(datetime.utcnow()),
                                      creativity_score=creativity_score,
                                      title_interest_level=title_info[self.TITLE_INTEREST_FIELD],
                                      title_interest_reason=title_info[self.TITLE_INTEREST_REASON_FIELD],
                                      language=title_info[self.LANGUAGE_FIELD],
                                      request_status=RequestStatusOptions.SUCCESS,
                                      request_status_message="The sheet has no user generated content"
                                      )

        content = self._sheet_to_text(
            no_quotes_content=no_quotes_content,
            full_content=full_content,
            max_tokens=self.max_prompt_tokens-self.token_margin,
            add_full_commentary=add_full_commentary)
        # Process with GPT
        gpt_analysis = self.get_gpt_scores(content, expanded_refs, title)
        if not gpt_analysis:
            request_status_message=f"Failed to get GPT scores for sheet {sheet_id}"
            logger.error(request_status_message)
            return self.create_failure_output(sheet_id=sheet_id,
                                              request_status_message=request_status_message)

        return SheetScoringOutput(
                sheet_id=sheet_id,
                ref_levels=gpt_analysis[self.REF_LEVELS_FIELD],
                ref_scores=gpt_analysis[self.REF_SCORES_FIELD],
                processed_datetime=str(datetime.utcnow()),
                creativity_score=creativity_score,
                title_interest_level=gpt_analysis[self.TITLE_INTEREST_FIELD],
                title_interest_reason=gpt_analysis[self.TITLE_INTEREST_REASON_FIELD],
                language=gpt_analysis[self.LANGUAGE_FIELD],
                request_status=RequestStatusOptions.SUCCESS,
                request_status_message=""
                )
