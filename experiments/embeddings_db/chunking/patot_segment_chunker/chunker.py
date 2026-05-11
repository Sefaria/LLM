import math
from dataclasses import dataclass
from typing import Optional

from pydantic.v1 import PrivateAttr
from semantic_chunkers import StatisticalChunker
from semantic_chunkers.chunkers import statistical as statistical_chunker_module
from semantic_router.encoders.base import BaseEncoder
from transformers import AutoTokenizer

from experiments.embeddings_db.embedding_eval.experiment_grid import EMBEDDING_001_TASK_SETUPS
from experiments.embeddings_db.embedding_eval.gemini_embedder import GeminiEmbedder, l2_normalize_vector

from .config import ChunkerConfig
from .sefaria_loader import SegmentRecord
from .splitters import HebrewTokenizerSplitter, fallback_clause_split, sentence_splitter_for_language
from .text_utils import detect_language, normalize_whitespace, strip_hebrew_niqqud, strip_html


@dataclass(frozen=True)
class PatotChunk:
    text: str
    source_segment_refs: list[str]
    kind: str
    pass_number: int
    token_count: Optional[int]
    triggered: Optional[bool]
    score: Optional[float]


@dataclass(frozen=True)
class DebugPreparedSegment:
    tref: str
    original_text: str
    processed_text: str
    token_count: int


@dataclass(frozen=True)
class DebugStatisticalChunk:
    text: str
    split_count: int
    token_count: Optional[int]
    triggered: Optional[bool]
    score: Optional[float]


@dataclass(frozen=True)
class DebugPass2Segment:
    tref: str
    processed_text: str
    splitter_name: str
    initial_splits: list[str]
    fallback_splits: list[str]
    final_chunks: list[DebugStatisticalChunk]
    returned_single_segment: bool


@dataclass(frozen=True)
class DebugPass3Adjustment:
    original_kind: str
    source_segment_refs: list[str]
    original_token_count: int
    produced_chunks: list[DebugStatisticalChunk]


@dataclass(frozen=True)
class DebugTrace:
    detected_lang: str
    prepared_segments: list[DebugPreparedSegment]
    pass1_units: list[str]
    pass1_chunks: list[DebugStatisticalChunk]
    pass1_chunk_segment_refs: list[list[str]]
    pass2_segments: list[DebugPass2Segment]
    pass3_adjustments: list[DebugPass3Adjustment]


@dataclass(frozen=True)
class PatotChunkResult:
    input_segment_count: int
    pass1_chunk_count: int
    final_chunk_count: int
    chunks: list[PatotChunk]
    debug_trace: Optional[DebugTrace] = None


class GeminiRouterEncoder(BaseEncoder):
    name: str = "repo-gemini"
    score_threshold: Optional[float] = None
    type: str = "gemini"
    _embedder = PrivateAttr()
    _doc_task_type = PrivateAttr()
    _config = PrivateAttr()

    def __init__(self, api_key: str, config: ChunkerConfig):
        super().__init__()
        _, doc_task_type = EMBEDDING_001_TASK_SETUPS[config.setup]
        self._embedder = GeminiEmbedder(api_key=api_key)
        self._doc_task_type = doc_task_type
        self._config = config

    def __call__(self, docs: list[str]) -> list[list[float]]:
        vectors = []
        for doc in docs:
            vector = self._embedder.embed_text(
                model=self._config.model,
                text=doc,
                output_dimensionality=self._config.dim,
                task_type=self._doc_task_type,
            )
            if self._config.norm:
                vector = l2_normalize_vector(vector)
            vectors.append(vector)
        return vectors

    async def acall(self, docs: list[str]) -> list[list[float]]:
        return self(docs)


class PatotChunker:
    def __init__(self, api_key: str, config: ChunkerConfig):
        self.api_key = api_key
        self.config = config
        self.encoder = GeminiRouterEncoder(api_key=api_key, config=config)
        self._tokenizer = None
        self._second_pass_hebrew_splitter = None

    def _debug(self, message: str) -> None:
        if self.config.debug:
            print(message)

    def _debug_block(self, title: str, rows: list[str]) -> None:
        if not self.config.debug:
            return
        print(f"\n[{title}]")
        for row in rows:
            print(row)

    def _debug_units(self, title: str, units: list[str]) -> None:
        if not self.config.debug:
            return
        rows = [f"count={len(units)}"]
        for i, unit in enumerate(units, start=1):
            rows.append(f"{i}. tokens={self._token_length(unit)} text={unit}")
        self._debug_block(title, rows)

    def _debug_statistical_chunks(self, title: str, chunks) -> None:
        if not self.config.debug:
            return
        rows = [f"count={len(chunks)}"]
        for i, chunk in enumerate(chunks, start=1):
            rows.append(
                " ".join(
                    [
                        f"{i}.",
                        f"split_count={len(chunk.splits)}",
                        f"token_count={chunk.token_count}",
                        f"triggered={chunk.is_triggered}",
                        f"score={chunk.triggered_score}",
                        f"text={chunk.content.strip()}",
                    ]
                )
            )
        self._debug_block(title, rows)

    def _build_debug_statistical_chunks(self, chunks) -> list[DebugStatisticalChunk]:
        return [
            DebugStatisticalChunk(
                text=chunk.content.strip(),
                split_count=len(chunk.splits),
                token_count=chunk.token_count,
                triggered=chunk.is_triggered,
                score=chunk.triggered_score,
            )
            for chunk in chunks
        ]

    def _build_debug_chunks_from_final_chunks(self, chunks: list[PatotChunk]) -> list[DebugStatisticalChunk]:
        return [
            DebugStatisticalChunk(
                text=chunk.text,
                split_count=1,
                token_count=chunk.token_count if chunk.token_count is not None else self._token_length(chunk.text),
                triggered=chunk.triggered,
                score=chunk.score,
            )
            for chunk in chunks
        ]

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_model)
        return self._tokenizer

    def _normalize_for_tokenizer(self, text: str) -> str:
        tokenizer = self._get_tokenizer()
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise RuntimeError(f"Tokenizer {self.config.tokenizer_model} does not expose backend_tokenizer")
        normalizer = getattr(backend, "normalizer", None)
        if normalizer is None:
            raise RuntimeError(f"Tokenizer {self.config.tokenizer_model} does not expose a backend normalizer")
        return normalizer.normalize_str(text)

    def _token_length(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        normalized = self._normalize_for_tokenizer(text)
        return len(tokenizer.encode(normalized, add_special_tokens=False))

    def _token_offsets(self, text: str) -> list[tuple[int, int]]:
        tokenizer = self._get_tokenizer()
        normalized = self._normalize_for_tokenizer(text)
        encoded = tokenizer(normalized, add_special_tokens=False, return_offsets_mapping=True)
        return list(encoded["offset_mapping"])

    def _preprocess(self, text: str) -> str:
        # Keep preprocessing centralized so pass 1, pass 2, and helper scripts all see the same text.
        original_text = text
        text = strip_html(text)
        if self.config.strip_hebrew_niqqud:
            text = strip_hebrew_niqqud(text)
        text = self._normalize_for_tokenizer(text)
        text = normalize_whitespace(text)
        if self.config.debug:
            self._debug_block(
                "Preprocess",
                [
                    f"original_tokens={self._token_length(original_text)}",
                    f"normalized_tokens={self._token_length(text)}",
                    f"original_text={original_text}",
                    f"normalized_text={text}",
                ],
            )
        return text

    def _make_statistical_chunker(self, lang: str) -> StatisticalChunker:
        statistical_chunker_module.tiktoken_length = self._token_length
        return StatisticalChunker(
            encoder=self.encoder,
            splitter=sentence_splitter_for_language(lang),
            threshold_adjustment=self.config.threshold_adjustment,
            dynamic_threshold=self.config.dynamic_threshold,
            window_size=self.config.window_size,
            min_split_tokens=self.config.min_split_tokens,
            max_split_tokens=self.config.max_split_tokens,
            split_tokens_tolerance=self.config.split_tokens_tolerance,
            plot_chunks=False,
            enable_statistics=False,
        )

    def _get_second_pass_hebrew_splitter(self):
        if self._second_pass_hebrew_splitter is None:
            self._second_pass_hebrew_splitter = HebrewTokenizerSplitter()
        return self._second_pass_hebrew_splitter

    def _chunk_atomic_units(self, unit_texts: list[str], lang: str):
        # Pass 1 works at the library-segment level: each input unit is one full source segment.
        chunker = self._make_statistical_chunker(lang)
        self._debug_units("Pass 1 Atomic Units", unit_texts)
        chunks = chunker._chunk(unit_texts)
        self._debug_statistical_chunks("Pass 1 Statistical Chunks", chunks)
        return chunks

    def _chunk_segment_internally(self, segment: SegmentRecord, lang: str) -> tuple[list[PatotChunk], DebugPass2Segment]:
        # Pass 2 only runs on singleton pass-1 outputs, so it is allowed to split inside one segment.
        self._debug(
            f"\n[Pass 2 Segment]\nref={segment.tref}\nsegment_tokens={self._token_length(segment.text)}\nsegment_text={segment.text}"
        )
        processed = self._preprocess(segment.text)
        sentence_chunker = self._make_statistical_chunker(lang)
        splitter_name = "HebrewTokenizerSplitter" if lang == "he" else "default_language_splitter"
        if lang == "he":
            self._debug(f"second_pass_splitter={splitter_name}")
            sentence_splits = self._get_second_pass_hebrew_splitter()(processed)
        else:
            self._debug(f"second_pass_splitter={splitter_name}")
            sentence_splits = sentence_chunker._split(processed)
        initial_sentence_splits = list(sentence_splits)
        self._debug_units("Pass 2 Initial Sentence Splits", sentence_splits)
        fallback_splits: list[str] = []
        if lang == "he" and len(sentence_splits) <= 1:
            self._debug("second_pass_fallback=fallback_clause_split")
            fallback_splits = fallback_clause_split(processed)
            sentence_splits = fallback_splits
            self._debug_units("Pass 2 Fallback Clause Splits", fallback_splits)
        if len(sentence_splits) <= 1:
            # If we still have one atomic unit here, the statistical chunker cannot split inside it.
            self._debug("pass_2_result=single_segment_no_internal_split")
            returned_chunk = PatotChunk(
                text=processed,
                source_segment_refs=[segment.tref],
                kind="single_segment",
                pass_number=2,
                token_count=None,
                triggered=None,
                score=None,
            )
            debug_segment = DebugPass2Segment(
                tref=segment.tref,
                processed_text=processed,
                splitter_name=splitter_name,
                initial_splits=initial_sentence_splits,
                fallback_splits=fallback_splits,
                final_chunks=[],
                returned_single_segment=True,
            )
            return [returned_chunk], debug_segment

        chunks = sentence_chunker._chunk(sentence_splits)
        self._debug_statistical_chunks("Pass 2 Statistical Chunks", chunks)
        chunk_list = [
            PatotChunk(
                text=chunk.content.strip(),
                source_segment_refs=[segment.tref],
                kind="intra_segment",
                pass_number=2,
                token_count=chunk.token_count,
                triggered=chunk.is_triggered,
                score=chunk.triggered_score,
            )
            for chunk in chunks
        ]
        debug_segment = DebugPass2Segment(
            tref=segment.tref,
            processed_text=processed,
            splitter_name=splitter_name,
            initial_splits=initial_sentence_splits,
            fallback_splits=fallback_splits,
            final_chunks=self._build_debug_statistical_chunks(chunks),
            returned_single_segment=False,
        )
        return chunk_list, debug_segment

    def _split_text_evenly_by_max_tokens(self, text: str) -> list[str]:
        token_count = self._token_length(text)
        if token_count <= self.config.max_split_tokens:
            return [text]

        offsets = self._token_offsets(text)
        if not offsets:
            return [text]

        chunk_count = math.ceil(token_count / self.config.max_split_tokens)
        tokens_per_chunk = math.ceil(token_count / chunk_count)
        pieces = []

        for chunk_index in range(chunk_count):
            start_token = chunk_index * tokens_per_chunk
            end_token = min((chunk_index + 1) * tokens_per_chunk, len(offsets))
            if start_token >= len(offsets):
                break
            start_char = offsets[start_token][0]
            end_char = offsets[end_token - 1][1]
            piece = text[start_char:end_char].strip()
            if piece:
                pieces.append(piece)

        return pieces or [text]

    def _build_chunk_from_whole_segments(self, segment_refs: list[str], segment_text_by_ref: dict[str, str]) -> PatotChunk:
        text = " ".join(segment_text_by_ref[segment_ref] for segment_ref in segment_refs).strip()
        return PatotChunk(
            text=text,
            source_segment_refs=segment_refs,
            kind="multi_segment_hard_max" if len(segment_refs) > 1 else "single_segment",
            pass_number=3,
            token_count=self._token_length(text),
            triggered=False,
            score=None,
        )

    def _split_multi_segment_chunk_by_segment_boundaries(
        self,
        chunk: PatotChunk,
        segment_text_by_ref: dict[str, str],
    ) -> list[PatotChunk]:
        output_chunks: list[PatotChunk] = []
        current_group_refs: list[str] = []
        current_group_tokens = 0

        for segment_ref in chunk.source_segment_refs:
            segment_text = segment_text_by_ref[segment_ref]
            segment_token_count = self._token_length(segment_text)

            if segment_token_count > self.config.max_split_tokens:
                if current_group_refs:
                    output_chunks.append(self._build_chunk_from_whole_segments(current_group_refs, segment_text_by_ref))
                    current_group_refs = []
                    current_group_tokens = 0

                for piece in self._split_text_evenly_by_max_tokens(segment_text):
                    output_chunks.append(
                        PatotChunk(
                            text=piece,
                            source_segment_refs=[segment_ref],
                            kind="hard_max_split",
                            pass_number=3,
                            token_count=self._token_length(piece),
                            triggered=False,
                            score=None,
                        )
                    )
                continue

            if current_group_refs and current_group_tokens + segment_token_count > self.config.max_split_tokens:
                output_chunks.append(self._build_chunk_from_whole_segments(current_group_refs, segment_text_by_ref))
                current_group_refs = []
                current_group_tokens = 0

            current_group_refs.append(segment_ref)
            current_group_tokens += segment_token_count

        if current_group_refs:
            output_chunks.append(self._build_chunk_from_whole_segments(current_group_refs, segment_text_by_ref))

        return output_chunks

    def _apply_hard_max_pass(
        self,
        chunks: list[PatotChunk],
        segment_text_by_ref: dict[str, str],
    ) -> tuple[list[PatotChunk], list[DebugPass3Adjustment]]:
        if not self.config.enforce_hard_max_in_pass3:
            return chunks, []

        adjusted_chunks: list[PatotChunk] = []
        adjustments: list[DebugPass3Adjustment] = []

        for chunk in chunks:
            actual_token_count = chunk.token_count if chunk.token_count is not None else self._token_length(chunk.text)
            if actual_token_count <= self.config.max_split_tokens:
                adjusted_chunks.append(
                    PatotChunk(
                        text=chunk.text,
                        source_segment_refs=chunk.source_segment_refs,
                        kind=chunk.kind,
                        pass_number=chunk.pass_number,
                        token_count=actual_token_count,
                        triggered=chunk.triggered,
                        score=chunk.score,
                    )
                )
                continue

            # Pass 3 is a hard-cap fallback: once semantic splitting is exhausted, force even token windows.
            self._debug_block(
                "Pass 3 Hard Max Split",
                [
                    f"original_kind={chunk.kind}",
                    f"source_segment_refs={chunk.source_segment_refs}",
                    f"original_token_count={actual_token_count}",
                    f"max_split_tokens={self.config.max_split_tokens}",
                    f"text={chunk.text}",
                ],
            )

            if len(chunk.source_segment_refs) > 1:
                split_chunks = self._split_multi_segment_chunk_by_segment_boundaries(chunk, segment_text_by_ref)
            else:
                split_pieces = self._split_text_evenly_by_max_tokens(chunk.text)
                split_chunks = [
                    PatotChunk(
                        text=piece,
                        source_segment_refs=chunk.source_segment_refs,
                        kind="hard_max_split",
                        pass_number=3,
                        token_count=self._token_length(piece),
                        triggered=False,
                        score=None,
                    )
                    for piece in split_pieces
                ]
            self._debug_units("Pass 3 Produced Chunks", [piece.text for piece in split_chunks])
            adjusted_chunks.extend(split_chunks)
            adjustments.append(
                DebugPass3Adjustment(
                    original_kind=chunk.kind,
                    source_segment_refs=chunk.source_segment_refs,
                    original_token_count=actual_token_count,
                    produced_chunks=self._build_debug_chunks_from_final_chunks(split_chunks),
                )
            )

        return adjusted_chunks, adjustments

    def chunk_segments(self, segments: list[SegmentRecord]) -> PatotChunkResult:
        if not segments:
            return PatotChunkResult(0, 0, 0, [])

        original_text_by_tref: dict[str, str] = {}
        prepared_segments = []
        for segment in segments:
            processed_text = self._preprocess(segment.text)
            if not processed_text:
                continue
            original_text_by_tref[segment.tref] = segment.text
            prepared_segments.append(
                SegmentRecord(
                    tref=segment.tref,
                    text=processed_text,
                    segment_index=segment.segment_index,
                )
            )
        lang = detect_language(" ".join(segment.text for segment in prepared_segments[:5]))
        self._debug_block(
            "Chunk Segments Setup",
            [
                f"detected_lang={lang}",
                f"input_segment_count={len(segments)}",
                f"prepared_segment_count={len(prepared_segments)}",
                f"min_split_tokens={self.config.min_split_tokens}",
                f"max_split_tokens={self.config.max_split_tokens}",
                f"tokenizer_model={self.config.tokenizer_model}",
            ],
        )
        self._debug_units("Prepared Segments", [segment.text for segment in prepared_segments])
        segment_texts = [segment.text for segment in prepared_segments]
        pass1_chunks = self._chunk_atomic_units(segment_texts, lang)
        debug_pass2_segments: list[DebugPass2Segment] = []
        debug_pass1_chunk_segment_refs: list[list[str]] = []
        segment_text_by_ref = {segment.tref: segment.text for segment in prepared_segments}

        final_chunks: list[PatotChunk] = []
        cursor = 0
        for pass1_chunk in pass1_chunks:
            segment_slice = prepared_segments[cursor : cursor + len(pass1_chunk.splits)]
            cursor += len(pass1_chunk.splits)
            debug_pass1_chunk_segment_refs.append([segment.tref for segment in segment_slice])
            self._debug_block(
                "Pass 1 Chunk Routing",
                [
                    f"pass1_chunk_token_count={pass1_chunk.token_count}",
                    f"pass1_chunk_triggered={pass1_chunk.is_triggered}",
                    f"pass1_chunk_score={pass1_chunk.triggered_score}",
                    f"segment_refs={[segment.tref for segment in segment_slice]}",
                    f"segment_count={len(segment_slice)}",
                    f"text={pass1_chunk.content.strip()}",
                ],
            )

            if len(segment_slice) > 1:
                # Multi-segment pass-1 groups are final by construction so they never enter pass 2.
                final_chunks.append(
                    PatotChunk(
                        text=pass1_chunk.content.strip(),
                        source_segment_refs=[segment.tref for segment in segment_slice],
                        kind="multi_segment",
                        pass_number=1,
                        token_count=pass1_chunk.token_count,
                        triggered=pass1_chunk.is_triggered,
                        score=pass1_chunk.triggered_score,
                    )
                )
                continue

            segment_chunks, debug_pass2_segment = self._chunk_segment_internally(segment_slice[0], lang)
            final_chunks.extend(segment_chunks)
            debug_pass2_segments.append(debug_pass2_segment)

        final_chunks, debug_pass3_adjustments = self._apply_hard_max_pass(final_chunks, segment_text_by_ref)

        self._debug_block(
            "Final Chunks",
            [
                f"count={len(final_chunks)}",
                *[
                    " ".join(
                        [
                            f"{i}.",
                            f"kind={chunk.kind}",
                            f"pass_number={chunk.pass_number}",
                            f"token_count={chunk.token_count}",
                            f"triggered={chunk.triggered}",
                            f"score={chunk.score}",
                            f"source_segment_refs={chunk.source_segment_refs}",
                            f"text={chunk.text}",
                        ]
                    )
                    for i, chunk in enumerate(final_chunks, start=1)
                ],
            ],
        )
        debug_trace = None
        if self.config.debug:
            debug_trace = DebugTrace(
                detected_lang=lang,
                prepared_segments=[
                    DebugPreparedSegment(
                        tref=segment.tref,
                        original_text=original_text_by_tref.get(segment.tref, segment.text),
                        processed_text=segment.text,
                        token_count=self._token_length(segment.text),
                    )
                    for segment in prepared_segments
                ],
                pass1_units=segment_texts,
                pass1_chunks=self._build_debug_statistical_chunks(pass1_chunks),
                pass1_chunk_segment_refs=debug_pass1_chunk_segment_refs,
                pass2_segments=debug_pass2_segments,
                pass3_adjustments=debug_pass3_adjustments,
            )
        return PatotChunkResult(
            input_segment_count=len(prepared_segments),
            pass1_chunk_count=len(pass1_chunks),
            final_chunk_count=len(final_chunks),
            chunks=final_chunks,
            debug_trace=debug_trace,
        )
