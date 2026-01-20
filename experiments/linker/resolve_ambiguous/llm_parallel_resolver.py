import html
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure LangSmith integration BEFORE any LangChain imports
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "citation-disambiguator"
# LANGSMITH_API_KEY should be set in your environment

import django
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

django.setup()

try:  # Optional: only if OpenAI is available
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

from sefaria.model.text import AddressType, Ref


# ----------------------------
# Config + lightweight logging
# ----------------------------

@dataclass(frozen=True)
class ResolverConfig:
    dicta_url: str = "https://parallels-3-0a.loadbalancer.dicta.org.il/parallels/api/findincorpus"
    sefaria_search_url: str = "https://www.sefaria.org/api/search/text/_search"

    min_threshold: float = 7.0
    max_distance: float = 4.0
    request_timeout: int = 30
    window_words_per_side: int = 120

    # kept from your signature (even if not currently used everywhere)
    general_min_score: float = 7.0
    tanakh_min_score: float = 1.45
    canonical_min_score: float = 2.35
    min_frequency_to_count_phrase_as_one_word: int = 30


class DebugLogger:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def log(self, msg: str) -> None:
        if self.enabled:
            print(msg)


# ----------------------------
# Networking helpers
# ----------------------------

class HttpClient:
    def __init__(self, timeout: int, debug: DebugLogger):
        self.timeout = timeout
        self.debug = debug

    def post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.debug.log(f"HTTP POST JSON failed: url={url} error={exc}")
            raise

    def post_form(self, url: str, params: Dict[str, Any], data: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            resp = requests.post(url, params=params, data=data, timeout=self.timeout, headers=headers)
            resp.raise_for_status()
            # Dicta sometimes includes BOM or returns non-json content-type
            text = resp.text.lstrip("\ufeff")
            try:
                return resp.json()
            except Exception:
                import json as _json
                return _json.loads(text)
        except Exception as exc:
            self.debug.log(f"HTTP POST FORM failed: url={url} error={exc}")
            raise


# ----------------------------
# Core resolver
# ----------------------------

class LLMParallelResolver:
    """
    Resolve ambiguous non-segment refs by:
    1) Dicta parallels API first
    2) fallback to Sefaria search with LLM-generated lexical queries
    3) LLM confirm final candidate
    """

    def __init__(
        self,
        llm=None,
        dicta_url: Optional[str] = None,
        min_threshold: float = 1.0,
        max_distance: float = 10.0,
        general_min_score: float = 7.0,
        tanakh_min_score: float = 1.45,
        canonical_min_score: float = 2.35,
        min_frequency_to_count_phrase_as_one_word: int = 30,
        request_timeout: int = 30,
        window_words_per_side: int = 120,
        sefaria_search_url: Optional[str] = None,
        debug: bool = False,  # NEW
    ):
        self.debug = DebugLogger(enabled=debug)

        # LLMs
        self.llm = llm or ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            temperature=0,
            max_tokens=256,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

        if not ChatOpenAI:
            raise RuntimeError("OpenAI is required for keyword extraction but langchain_openai is not installed.")

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is required for keyword extraction.")

        keyword_model = os.getenv("LLM_KEYWORD_MODEL")
        self.keyword_llm = ChatOpenAI(
            model=keyword_model or "gpt-4o-mini",
            temperature=0,
            max_tokens=256,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Config
        self.cfg = ResolverConfig(
            dicta_url=dicta_url or os.getenv("DICTA_PARALLELS_URL", ResolverConfig.dicta_url),
            sefaria_search_url=sefaria_search_url or os.getenv("SEFARIA_SEARCH_URL", ResolverConfig.sefaria_search_url),
            min_threshold=min_threshold,
            max_distance=max_distance,
            request_timeout=request_timeout,
            window_words_per_side=window_words_per_side,
            general_min_score=general_min_score,
            tanakh_min_score=tanakh_min_score,
            canonical_min_score=canonical_min_score,
            min_frequency_to_count_phrase_as_one_word=min_frequency_to_count_phrase_as_one_word,
        )

        self.http = HttpClient(timeout=self.cfg.request_timeout, debug=self.debug)
        self._profile: Optional[Dict[str, Any]] = None

    # -------- profiling helpers --------

    def _profile_reset(self, profile: Optional[Dict[str, Any]]) -> None:
        if profile is not None:
            profile.setdefault("dicta_seconds", 0.0)
            profile.setdefault("es_seconds", 0.0)
            profile.setdefault("llm_tokens", 0)
            profile.setdefault("llm_tokens_by_model", {})
        self._profile = profile

    def _profile_add(self, key: str, value: Union[float, int]) -> None:
        if self._profile is None:
            return
        if key == "llm_tokens":
            self._profile[key] = int(self._profile.get(key, 0)) + int(value or 0)
        else:
            self._profile[key] = float(self._profile.get(key, 0.0)) + float(value or 0.0)

    def _extract_llm_tokens(self, resp: Any) -> int:
        if resp is None:
            return 0
        usage = None
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            usage = resp.usage_metadata
        elif hasattr(resp, "response_metadata") and resp.response_metadata:
            usage = resp.response_metadata.get("usage") or resp.response_metadata.get("token_usage")
        if not isinstance(usage, dict):
            return 0
        if "total_tokens" in usage:
            return int(usage.get("total_tokens") or 0)
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        return int(input_tokens) + int(output_tokens)

    def _get_model_name(self, llm: Any) -> str:
        for attr in ("model_name", "model", "model_id"):
            if hasattr(llm, attr):
                val = getattr(llm, attr)
                if isinstance(val, str) and val:
                    return val
        return llm.__class__.__name__

    def _profile_add_tokens(self, llm: Any, resp: Any) -> None:
        tokens = self._extract_llm_tokens(resp)
        if tokens <= 0:
            return
        self._profile_add("llm_tokens", tokens)
        if self._profile is None:
            return
        by_model = self._profile.get("llm_tokens_by_model")
        if not isinstance(by_model, dict):
            by_model = {}
            self._profile["llm_tokens_by_model"] = by_model
        model_name = self._get_model_name(llm)
        by_model[model_name] = int(by_model.get(model_name, 0)) + int(tokens)

    # -------- public API --------

    @traceable(run_type="chain", name="resolve_ambiguous_ref")
    def resolve(self, link: dict, chunk: dict, profile: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Main entry: resolve one ambiguous non-segment reference found in link/chunk.
        Returns updated link/chunk + metadata, or None if no resolution found.
        """
        self._profile_reset(profile)

        # Step 1: Validate inputs and extract basic context
        context = self._validate_and_extract_context(link, chunk)
        if not context:
            return None

        # Step 2: Try early segment-based resolution (1-3 segments) - works for any language
        early_result = self._try_early_segment_resolution(link, chunk, context)
        if early_result:
            return early_result

        # Step 3: For non-Hebrew, we can't use Dicta/search pipeline, so stop here
        lang = context.get("lang")
        if lang and lang != "he":
            return None

        # Step 4: Try Dicta and fallback search pipeline (Hebrew only)
        resolution = self._find_candidate_resolution(chunk, context)
        if not resolution:
            return None

        # Step 5: Confirm candidate with LLM
        return self._confirm_and_build_result(link, chunk, context, resolution)

    def _validate_and_extract_context(self, link: dict, chunk: dict) -> Optional[dict]:
        """Extract and validate all required context for resolution."""
        non_segment_ref = self._find_non_segment_ref(link)
        if not non_segment_ref:
            return None

        citing_ref = chunk.get("ref")
        lang = chunk.get("language")

        citing_text = self._get_ref_text(citing_ref, lang, chunk.get("versionTitle"))
        if not citing_text:
            return None

        span = self._find_span_for_ref(chunk, non_segment_ref)
        citing_window, span_window = self._window_around_span(
            citing_text, span, self.cfg.window_words_per_side
        )
        marked_citing_text = self._mark_citation(citing_window, span_window)
        base_ref, base_text = self._get_commentary_base_context(citing_ref)
        segments = self._get_segments_safely(non_segment_ref)

        return {
            "non_segment_ref": non_segment_ref,
            "citing_ref": citing_ref,
            "lang": lang,
            "vtitle": chunk.get("versionTitle"),
            "citing_text": citing_text,
            "citing_window": citing_window,
            "span": span,
            "span_window": span_window,
            "marked_citing_text": marked_citing_text,
            "base_ref": base_ref,
            "base_text": base_text,
            "segments": segments,
        }

    def _get_segments_safely(self, non_segment_ref: str) -> List:
        """Safely get all segment refs, returning empty list on error."""
        try:
            return Ref(non_segment_ref).all_segment_refs()
        except Exception:
            return []

    @traceable(run_type="chain", name="try_early_segment_resolution")
    def _try_early_segment_resolution(
        self, link: dict, chunk: dict, context: dict
    ) -> Optional[Dict[str, Any]]:
        """Try early resolution for refs with 1-3 segments."""
        segments = context["segments"]
        if not segments:
            return None

        if len(segments) == 1:
            return self._resolve_single_segment(link, chunk, context)

        if len(segments) in {2, 3}:
            return self._resolve_small_segment_range(link, chunk, context)

        return None

    def _resolve_single_segment(
        self, link: dict, chunk: dict, context: dict
    ) -> Dict[str, Any]:
        """Automatically resolve when there's only one segment."""
        segments = context["segments"]
        non_segment_ref = context["non_segment_ref"]
        citing_ref = context["citing_ref"]
        resolved_ref = segments[0].normal()

        self.debug.log(
            f"Auto-resolved single segment: citing_ref={citing_ref} "
            f"target_ref={non_segment_ref} resolved_ref={resolved_ref}"
        )
        self._log_resolution_links(citing_ref, non_segment_ref, resolved_ref, "Auto-resolve")

        return self._build_resolution_result(
            link=link,
            chunk=chunk,
            non_segment_ref=non_segment_ref,
            resolved_ref=resolved_ref,
            selected_segments=[resolved_ref],
            reason="Automatically resolved: only one segment exists",
            llm_reason=None,
            match_source="segment_auto",
        )

    def _resolve_small_segment_range(
        self, link: dict, chunk: dict, context: dict
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to resolve refs with 2-3 segments."""
        segments = context["segments"]
        non_segment_ref = context["non_segment_ref"]
        citing_ref = context["citing_ref"]
        marked_citing_text = context["marked_citing_text"]
        base_ref = context["base_ref"]
        base_text = context["base_text"]

        range_result = self._llm_pick_small_range(
            marked_citing_text, non_segment_ref, segments,
            base_ref=base_ref, base_text=base_text,
        )
        if not range_result:
            return None

        start_idx, end_idx, reason = range_result
        selected = self._extract_segment_range(segments, start_idx, end_idx)
        if not selected:
            return None

        resolved_ref = self._build_resolved_ref_from_segments(selected)

        self.debug.log(
            f"LLM resolved small segment range: citing_ref={citing_ref} "
            f"target_ref={non_segment_ref} resolved_ref={resolved_ref} "
            f"selected_count={len(selected)} reason='{reason}'"
        )
        self._log_resolution_links(citing_ref, non_segment_ref, resolved_ref, "LLM small-range")

        return self._build_resolution_result(
            link=link,
            chunk=chunk,
            non_segment_ref=non_segment_ref,
            resolved_ref=resolved_ref,
            selected_segments=[s.normal() for s in selected],
            reason="LLM resolved small segment range",
            llm_reason=reason,
            match_source="segment_llm_small",
        )

    def _extract_segment_range(self, segments: List, start_idx: int, end_idx: int) -> List:
        """Extract and validate segment range."""
        start_idx = max(1, min(start_idx, len(segments)))
        end_idx = max(start_idx, min(end_idx, len(segments)))
        return segments[start_idx - 1:end_idx]

    def _build_resolved_ref_from_segments(self, selected: List) -> str:
        """Build a resolved ref string from selected segments."""
        if len(selected) == 1:
            return selected[0].normal()
        return selected[0].to(selected[-1]).normal()

    def _find_candidate_resolution(self, chunk: dict, context: dict) -> Optional[Dict[str, Any]]:
        """Try Dicta API first, then fallback to search pipeline."""
        # Try Dicta first
        resolution = self._query_dicta(
            query_text=context["citing_window"],
            target_ref=context["non_segment_ref"],
            citing_ref=context["citing_ref"],
            citing_lang=context["lang"],
            citing_version=context["vtitle"],
            ranking_context=context["marked_citing_text"],
            base_ref=context["base_ref"],
            base_text=context["base_text"],
        )
        if resolution:
            return resolution

        # Fallback to search pipeline
        return self._fallback_search_pipeline(
            marked_citing_text=context["marked_citing_text"],
            citing_text=context["citing_text"],
            span=context["span"],
            non_segment_ref=context["non_segment_ref"],
            citing_ref=context["citing_ref"],
            lang=context["lang"],
            vtitle=context["vtitle"],
            base_ref=context["base_ref"],
            base_text=context["base_text"],
        )

    @traceable(run_type="chain", name="confirm_and_build_result")
    def _confirm_and_build_result(
        self, link: dict, chunk: dict, context: dict, resolution: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Confirm candidate with LLM and build final result."""
        resolved_ref = resolution.get("resolved_ref")
        if not resolved_ref:
            return None

        # Edge case: Metzudat Zion is always approved without LLM confirmation
        citing_ref = context.get("citing_ref", "")
        if citing_ref.startswith("Metzudat Zion"):
            self.debug.log(
                f"Auto-approved Metzudat Zion (skipped LLM): citing_ref={citing_ref} "
                f"target_ref={context['non_segment_ref']} resolved_ref={resolved_ref}"
            )
            source = resolution.get("source", "Dicta")
            return self._build_resolution_result(
                link=link,
                chunk=chunk,
                non_segment_ref=context["non_segment_ref"],
                resolved_ref=resolved_ref,
                selected_segments=[resolved_ref],
                reason=f"{source.title()} hit {resolved_ref}, auto-approved (Metzudat Zion)",
                llm_reason="Metzudat Zion is always approved",
                match_source=source,
                resolution=resolution,
            )

        candidate_text = self._get_ref_text(resolved_ref, context["lang"])
        ok, reason = self._llm_confirm_candidate(
            context["marked_citing_text"],
            resolved_ref,
            candidate_text,
            base_ref=context["base_ref"],
            base_text=context["base_text"],
        )

        if not ok:
            self.debug.log(
                f"LLM rejected: citing_ref={context['citing_ref']} "
                f"target_ref={context['non_segment_ref']} "
                f"candidate_ref={resolved_ref} reason='{reason}'"
            )
            return None

        self.debug.log(
            f"LLM confirmed: citing_ref={context['citing_ref']} "
            f"target_ref={context['non_segment_ref']} "
            f"candidate_ref={resolved_ref} reason='{reason}'"
        )

        source = resolution.get("source", "Dicta")
        return self._build_resolution_result(
            link=link,
            chunk=chunk,
            non_segment_ref=context["non_segment_ref"],
            resolved_ref=resolved_ref,
            selected_segments=[resolved_ref],
            reason=f"{source.title()} hit {resolved_ref} confirmed by LLM",
            llm_reason=reason,
            match_source=source,
            resolution=resolution,
        )

    def _build_resolution_result(
        self,
        link: dict,
        chunk: dict,
        non_segment_ref: str,
        resolved_ref: str,
        selected_segments: List[str],
        reason: str,
        llm_reason: Optional[str],
        match_source: str,
        resolution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the final resolution result dictionary."""
        updated_link = self._replace_ref_in_link(link, non_segment_ref, resolved_ref)
        updated_chunk = self._replace_ref_in_chunk(chunk, non_segment_ref, resolved_ref)

        return {
            "link": updated_link,
            "chunk": updated_chunk,
            "original_ref": non_segment_ref,
            "resolved_ref": resolved_ref,
            "selected_segments": selected_segments,
            "reason": reason,
            "llm_reason": llm_reason,
            "llm_verdict": "YES",
            "dicta_hit": resolution if resolution and resolution.get("source") == "dicta" else None,
            "search_hit": resolution if resolution and resolution.get("source") == "sefaria_search" else None,
            "match_source": match_source,
        }

    def _log_resolution_links(
        self, citing_ref: str, target_ref: str, resolved_ref: str, label: str
    ) -> None:
        """Log resolution links for debugging."""
        citing_link = self._safe_build_link(citing_ref)
        target_link = self._safe_build_link(target_ref)
        resolved_link = self._safe_build_link(resolved_ref)

        if not (citing_link or target_link or resolved_link):
            return

        parts = []
        if citing_link:
            parts.append(f"citing={citing_link}")
        if target_link:
            parts.append(f"target={target_link}")
        if resolved_link:
            parts.append(f"resolved={resolved_link}")

        self.debug.log(f"{label} links:\n  " + "\n  ".join(parts))

    def _safe_build_link(self, tref: Optional[str]) -> Optional[str]:
        """Safely build a Sefaria library link from a ref."""
        if not tref:
            return None
        try:
            return f"https://library-linker2.cauldron.sefaria.org/{Ref(tref).url()}"
        except Exception:
            return None

    # -------- fallback pipeline (clean + dedup) --------

    def _fallback_search_pipeline(
        self,
        marked_citing_text: str,
        citing_text: str,
        span: Optional[dict],
        non_segment_ref: str,
        citing_ref: Optional[str],
        lang: Optional[str],
        vtitle: Optional[str],
        base_ref: Optional[str],
        base_text: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        searched: set[str] = set()
        candidates: List[Dict[str, Any]] = []

        def run_queries(queries: List[str], label: str) -> None:
            for q in queries:
                q = (q or "").strip()
                if not q or q in searched:
                    continue
                searched.add(q)
                hit = self._query_sefaria_search(
                    q,
                    target_ref=non_segment_ref,
                    citing_ref=citing_ref,
                    citing_lang=lang,
                    citing_version=vtitle,
                )
                if hit:
                    self.debug.log(f"Sefaria search {label} succeeded: '{q}' -> {hit.get('resolved_ref')}")
                    candidates.append(hit)
                    continue

                # one retry
                self.debug.log(f"Sefaria search {label} failed: '{q}', retrying once...")
                retry = self._query_sefaria_search(
                    q,
                    target_ref=non_segment_ref,
                    citing_ref=citing_ref,
                    citing_lang=lang,
                    citing_version=vtitle,
                )
                if retry:
                    self.debug.log(f"Sefaria search {label} retry succeeded: '{q}' -> {retry.get('resolved_ref')}")
                    candidates.append(retry)

        # A) normal window queries
        q1 = self._llm_form_search_query(marked_citing_text) or []
        run_queries(q1, label="(text-only)")

        # B) base-text seeded queries
        if base_text:
            q2 = self._llm_form_search_query(marked_citing_text, base_ref=base_ref, base_text=base_text) or []
            run_queries(q2, label="(base-seeded)")

        # C) expanded window queries
        if not candidates:
            expanded_words = max(self.cfg.window_words_per_side * 2, self.cfg.window_words_per_side + 1)
            expanded_window, expanded_span = self._window_around_span(citing_text, span, expanded_words)
            expanded_marked = self._mark_citation(expanded_window, expanded_span)

            q3 = self._llm_form_search_query(expanded_marked) or []
            run_queries(q3, label="(expanded text-only)")

            if base_text:
                q4 = self._llm_form_search_query(expanded_marked, base_ref=base_ref, base_text=base_text) or []
                run_queries(q4, label="(expanded base-seeded)")

        if not candidates:
            return None

        deduped = self._dedupe_candidates_by_ref(candidates)
        if len(deduped) == 1:
            return deduped[0]

        chosen = self._llm_choose_best_candidate(
            marked_citing_text,
            non_segment_ref,
            deduped,
            base_ref=base_ref,
            base_text=base_text,
            lang=lang,
        )
        if chosen:
            self.debug.log(f"Sefaria search had {len(deduped)} candidates; LLM chose {chosen.get('resolved_ref')}")
        return chosen

    # -------- ref + text helpers --------

    def _find_non_segment_ref(self, link: dict) -> Optional[str]:
        for tref in link.get("refs", []):
            try:
                oref = Ref(tref)
            except Exception:
                continue
            if not oref.is_segment_level():
                return oref.normal()
        return None

    def _get_ref_text(self, tref: str, lang: Optional[str] = None, vtitle: Optional[str] = None) -> str:
        if not tref:
            return ""
        vtitle = html.unescape(vtitle) if vtitle else None
        try:
            primary = lang or "en"
            text = Ref(tref).text(primary, vtitle=vtitle).as_string()
            if text:
                return text
            fallback = "he" if primary == "en" else "en"
            return Ref(tref).text(fallback).as_string()
        except Exception:
            return ""

    def _find_span_for_ref(self, chunk: dict, target_ref: str) -> Optional[dict]:
        spans = chunk.get("spans") or []
        for s in spans:
            if s.get("ref") == target_ref and s.get("type") == "citation":
                return s
        for s in spans:
            if s.get("type") == "citation":
                return s
        return None

    def _window_around_span(self, text: str, span: Optional[dict], window_words: int) -> Tuple[str, Optional[dict]]:
        if not text or not span:
            return text, span
        cr = span.get("charRange")
        if not cr or len(cr) != 2:
            return text, span
        start, end = cr
        if start < 0 or end > len(text) or start >= end:
            return text, span

        tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if not tokens:
            return text, span

        first_idx = last_idx = None
        for i, (s, e) in enumerate(tokens):
            if first_idx is None and e > start:
                first_idx = i
            if s < end:
                last_idx = i
            if e >= end and first_idx is not None:
                break

        if first_idx is None or last_idx is None:
            return text, span

        left = max(0, first_idx - window_words)
        right = min(len(tokens) - 1, last_idx + window_words)

        w_start = tokens[left][0]
        w_end = tokens[right][1]

        new_text = text[w_start:w_end]
        new_span = dict(span)
        new_span["charRange"] = [start - w_start, end - w_start]
        return new_text, new_span

    def _mark_citation(self, text: str, span: Optional[dict]) -> str:
        if not text or not span:
            return text
        cr = span.get("charRange")
        ref_attr = span.get("ref")
        if not cr or len(cr) != 2:
            return text
        start, end = cr
        if start < 0 or end > len(text) or start >= end:
            return text
        open_tag = "<citation"
        if ref_attr:
            open_tag += f' ref="{ref_attr}"'
        open_tag += ">"
        return text[:start] + open_tag + text[start:end] + "</citation>" + text[end:]

    def _get_commentary_base_context(self, citing_ref: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not citing_ref:
            return None, None
        try:
            citing_oref = Ref(citing_ref)
            base_titles = getattr(citing_oref.index, "base_text_titles", []) or []
            if not base_titles:
                return None, None

            base_title = base_titles[0]
            section_ref = citing_oref.section_ref()
            for sec, addr_type in zip(section_ref.sections, section_ref.index_node.addressTypes):
                address = AddressType.to_str_by_address_type(addr_type, "en", sec)
                base_title += f" {address}"

            base_ref = Ref(base_title).normal()
            base_text = self._get_ref_text(base_ref, lang="he") or self._get_ref_text(base_ref, lang="en")
            return base_ref, base_text
        except Exception:
            return None, None

    # -------- LLM helpers --------

    @traceable(run_type="llm", name="llm_form_search_query")
    def _llm_form_search_query(
        self, marked_citing_text: str, base_ref: Optional[str] = None, base_text: Optional[str] = None
    ) -> Optional[List[str]]:
        if not marked_citing_text:
            return None

        context_only = re.sub(
            r"<citation[^>]*>.*?</citation>", " [CITATION] ", marked_citing_text, flags=re.DOTALL
        )

        base_block = ""
        if base_ref and base_text:
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text[:3000]}\n\n"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are extracting a concise citation phrase to search for parallels."),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "Context with citation redacted:\n{context}\n\n"
                    "{base_block}"
                    "Return 5-6 short lexical search queries (<=6 words each), taken from surrounding context "
                    "outside the citation span.\n"
                    "- If base text is provided, prefer keywords that appear verbatim in the base text.\n"
                    "- Include at least one 2-3 word query.\n"
                    "- Do NOT copy words that appear inside <citation>...</citation>.\n"
                    "Strict output: one per line, numbered 1) ... through 6) ... or a single line 'NONE'.",
                ),
            ]
        )

        try:
            chain = prompt | self.keyword_llm
            resp = chain.invoke(
                {"citing": marked_citing_text[:6000], "context": context_only[:6000], "base_block": base_block},
                config={
                    "metadata": {
                        "has_base_text": base_ref is not None,
                        "base_ref": base_ref or "none"
                    },
                    "tags": ["search_query", "keyword_extraction", "citation_disambiguation"]
                }
            )
            self._profile_add_tokens(self.keyword_llm, resp)
            content = getattr(resp, "content", "").strip()
            if not content or content.upper() == "NONE":
                return None

            queries: List[str] = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.upper() == "NONE":
                    return None
                line = re.sub(r"^\s*\d+[\).\s]+", "", line).strip()
                if not line:
                    continue
                words = line.split()
                if len(words) > 6:
                    line = " ".join(words[:6])
                if line not in queries:
                    queries.append(line)
                if len(queries) >= 6:
                    break

            if queries:
                self.debug.log(f"LLM formed lexical search queries: {queries}")
            return queries or None
        except Exception as exc:
            self.debug.log(f"LLM lexical search query formation failed: {exc}")
            return None

    @traceable(run_type="llm", name="llm_confirm_candidate")
    def _llm_confirm_candidate(
        self,
        citing_text: str,
        candidate_ref: str,
        candidate_text: str,
        base_ref: Optional[str] = None,
        base_text: Optional[str] = None,
    ) -> Tuple[bool, str]:
        base_block = ""
        if base_ref and base_text:
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text[:3000]}\n\n"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You verify whether one Jewish text is genuinely citing or closely paraphrasing a specific target segment. "
                    "Be strict in your evaluation."
                ),
                (
                    "human",
                    "Citing passage (the citation span is wrapped in <citation ...></citation>):\n"
                    "{citing}\n\n"
                    "{base_block}"
                    "Candidate segment ref (retrieved by similarity):\n{candidate_ref}\n\n"
                    "Candidate segment text:\n{candidate_text}\n\n"
                    "Determine whether the citing passage is actually referring to this candidate segment.\n"
                    "If base text is provided, consider whether the commentary is discussing that base passage.\n\n"
                    "Answer in exactly two lines (no preamble):\n"
                    "Reason: <brief rationale>\n"
                    "Verdict: YES or NO",
                ),
            ]
        )

        chain = prompt | self.llm
        resp = chain.invoke(
            {
                "citing": citing_text[:6000],
                "base_block": base_block,
                "candidate_ref": candidate_ref,
                "candidate_text": (candidate_text or "")[:6000],
            },
            config={
                "metadata": {
                    "candidate_ref": candidate_ref,
                    "has_base_text": base_ref is not None,
                    "base_ref": base_ref or "none"
                },
                "tags": ["confirmation", "llm_verdict", "citation_disambiguation"]
            }
        )
        self._profile_add_tokens(self.llm, resp)
        content = getattr(resp, "content", "").strip()

        verdict = None
        for line in content.splitlines():
            if line.lower().startswith("verdict"):
                for tok in line.split():
                    t = tok.strip().strip(":").upper()
                    if t in {"YES", "NO"}:
                        verdict = t
                        break

        if verdict is None:
            # More robust fallback parsing
            if "verdict:" in content.lower():
                verdict_line = [line for line in content.splitlines() if "verdict:" in line.lower()]
                if verdict_line:
                    verdict = "YES" if "yes" in verdict_line[0].lower() else "NO"
            else:
                # Last resort: check if content starts with y/n
                verdict = "YES" if content.lower().startswith("y") else "NO"

        if verdict is None:
            self.debug.log(f"Could not parse LLM verdict from: {content}")
            verdict = "NO"  # Default to NO if unparseable

        return verdict == "YES", content

    # -------- Dicta + Search query methods --------

    @traceable(run_type="tool", name="query_dicta")
    def _query_dicta(
        self,
        query_text: str,
        target_ref: Optional[str] = None,
        citing_ref: Optional[str] = None,
        citing_lang: Optional[str] = None,
        citing_version: Optional[str] = None,
        ranking_context: Optional[str] = None,
        base_ref: Optional[str] = None,
        base_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query Dicta parallels API."""
        start = time.perf_counter()
        params = {
            "minthreshold": int(self.cfg.min_threshold) if self.cfg.min_threshold is not None else "",
            "maxdistance": int(self.cfg.max_distance) if self.cfg.max_distance is not None else "",
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://parallels.dicta.org.il",
            "Referer": "https://parallels.dicta.org.il/",
        }

        # Debug: Show exact URL and params
        param_str = "&".join([f"{k}={v}" for k, v in params.items() if v != ""])
        full_url = f"{self.cfg.dicta_url}?{param_str}"
        self.debug.log(f"[Base Resolver] Dicta URL: {full_url}")
        self.debug.log(f"[Base Resolver] Dicta text query (first 200 chars): {query_text[:200]}")

        try:
            data = self.http.post_form(
                self.cfg.dicta_url,
                params=params,
                data=f"text={query_text}".encode("utf-8"),
                headers=headers,
            )
        except Exception as exc:
            self.debug.log(f"Dicta request failed: {exc}")
            return None
        finally:
            self._profile_add("dicta_seconds", time.perf_counter() - start)

        candidates = self._collect_dicta_hits(
            data.get("results") or [],
            target_ref=target_ref,
            citing_ref=citing_ref,
            citing_lang=citing_lang,
            citing_version=citing_version,
        )
        if not candidates:
            self.debug.log("Dicta returned no contained matches")
            return None

        deduped = self._dedupe_candidates_by_ref(candidates)
        if len(deduped) == 1:
            self.debug.log(f"Dicta success: {deduped[0].get('resolved_ref')}")
            return deduped[0]

        chosen = self._llm_choose_best_candidate(
            ranking_context or query_text,
            target_ref or "",
            deduped,
            base_ref=base_ref,
            base_text=base_text,
            lang=citing_lang,
        )
        if chosen:
            self.debug.log(f"Dicta multiple hits; LLM chose {chosen.get('resolved_ref')} from {len(deduped)}")
        else:
            self.debug.log("LLM failed to pick among Dicta hits")
        return chosen

    def _collect_dicta_hits(
        self,
        results: List[Dict[str, Any]],
        target_ref: Optional[str] = None,
        citing_ref: Optional[str] = None,
        citing_lang: Optional[str] = None,
        citing_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Collect Dicta hits that match the target ref."""
        target_oref = None
        if target_ref:
            try:
                target_oref = Ref(target_ref)
            except Exception:
                pass

        candidates: List[Dict[str, Any]] = []
        for entry in results:
            for cand in entry.get("data", []):
                url = cand.get("url") or cand.get("compUrl") or ""
                normalized = self._normalize_dicta_url_to_ref(url)
                if not normalized:
                    continue
                try:
                    oref = Ref(normalized)
                    if not oref.is_segment_level():
                        continue
                    if target_oref and target_oref.contains(oref):
                        score = cand.get("score")
                        candidates.append({"resolved_ref": normalized, "raw": cand, "source": "dicta", "score": score})
                except Exception:
                    continue

        return candidates

    @traceable(run_type="tool", name="query_sefaria_search")
    def _query_sefaria_search(
        self,
        query_text: str,
        target_ref: Optional[str] = None,
        citing_ref: Optional[str] = None,
        citing_lang: Optional[str] = None,
        citing_version: Optional[str] = None,
        size: int = 500,
        slop: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Query Sefaria search API."""
        start = time.perf_counter()
        path_regex = self._path_regex_for_ref(target_ref)

        # Build query with proper filter handling
        bool_query = {
            "must": {"match_phrase": {"naive_lemmatizer": {"query": query_text, "slop": slop}}}
        }

        # Only add filter if we have a path_regex
        if path_regex:
            bool_query["filter"] = {"bool": {"should": [{"regexp": {"path": path_regex}}]}}

        payload = {
            "from": 0,
            "size": size,
            "highlight": {"pre_tags": ["<b>"], "post_tags": ["</b>"], "fields": {"naive_lemmatizer": {"fragment_size": 200}}},
            "query": {
                "function_score": {
                    "field_value_factor": {"field": "pagesheetrank", "missing": 0.04},
                    "query": {
                        "bool": bool_query
                    },
                }
            },
        }
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json",
            "Origin": "https://www.sefaria.org",
            "Referer": "https://www.sefaria.org/texts",
        }

        try:
            data = self.http.post_json(self.cfg.sefaria_search_url, payload=payload, headers=headers)
        except Exception as exc:
            self.debug.log(f"Sefaria search-api request failed: {exc}")
            return None
        finally:
            self._profile_add("es_seconds", time.perf_counter() - start)

        target_oref = None
        try:
            target_oref = Ref(target_ref) if target_ref else None
        except Exception:
            pass

        hits = (data.get("hits") or {}).get("hits") or []
        for entry in hits:
            normalized = self._extract_ref_from_search_hit(entry)
            if not normalized:
                continue
            try:
                cand_oref = Ref(normalized)
                if not cand_oref.is_segment_level():
                    continue
                if target_oref and target_oref.contains(cand_oref):
                    self.debug.log(f"Search found: {normalized}")
                    return {"resolved_ref": normalized, "raw": entry, "source": "sefaria_search", "query": query_text}
            except Exception:
                continue

        return None

    def _extract_ref_from_search_hit(self, hit: Dict[str, Any]) -> Optional[str]:
        """Extract ref from search hit."""
        candidates: List[Optional[str]] = []
        for k in ("ref", "he_ref", "sourceRef"):
            if k in hit:
                candidates.append(hit.get(k))
        src = hit.get("_source") or {}
        for k in ("ref", "he_ref", "sourceRef"):
            if k in src:
                candidates.append(src.get(k))

        for c in candidates:
            if not c:
                continue
            try:
                return Ref(c).normal()
            except Exception:
                continue
        return None

    def _path_regex_for_ref(self, target_ref: Optional[str]) -> Optional[str]:
        """Build path regex for search filtering."""
        if not target_ref:
            return None
        try:
            oref = Ref(target_ref)
            idx = oref.index
            parts: List[str] = []
            parts.extend(getattr(idx, "categories", []) or [])
            parts.append(idx.title)
            path = "/".join([p for p in parts if p])
            return path.replace("/", r"\/") + ".*" if path else None
        except Exception:
            return None

    # -------- misc helpers --------

    def _normalize_dicta_url_to_ref(self, url: str) -> Optional[str]:
        """Normalize a Dicta URL to a Sefaria ref."""
        if not url:
            return None
        cleaned = url.lstrip("/")
        if cleaned.startswith("www.sefaria.org/"):
            cleaned = cleaned[len("www.sefaria.org/"):]
        if cleaned.startswith("sefaria.org/"):
            cleaned = cleaned[len("sefaria.org/"):]
        cleaned = cleaned.split("?")[0]
        try:
            from urllib.parse import unquote
            cleaned = unquote(cleaned)
        except Exception:
            pass
        cleaned = cleaned.replace("_", " ")
        try:
            return Ref(cleaned).normal()
        except Exception:
            return None

    def _dedupe_candidates_by_ref(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate candidates by ref, keeping highest score."""
        unique: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            r = c.get("resolved_ref")
            if not r:
                continue
            if r not in unique:
                unique[r] = c
            else:
                ps, ns = unique[r].get("score"), c.get("score")
                if ns is not None and (ps is None or ns > ps):
                    unique[r] = c
        return list(unique.values())

    def _replace_ref_in_link(self, link: dict, old_ref: str, new_ref: str) -> dict:
        """Replace a ref in a link dict."""
        updated = dict(link)
        updated["refs"] = [new_ref if r == old_ref else r for r in link.get("refs", [])]
        return updated

    def _replace_ref_in_chunk(self, chunk: dict, old_ref: str, new_ref: str) -> dict:
        """Replace a ref in a chunk dict."""
        updated = dict(chunk)
        spans = []
        for s in chunk.get("spans") or []:
            if s.get("type") == "citation" and s.get("ref") == old_ref:
                ns = dict(s)
                ns["ref"] = new_ref
                spans.append(ns)
            else:
                spans.append(s)
        updated["spans"] = spans
        return updated

    @traceable(run_type="llm", name="llm_choose_best_candidate")
    def _llm_choose_best_candidate(
        self,
        citing_text: str,
        target_ref: str,
        candidates: List[Dict[str, Any]],
        base_ref: Optional[str] = None,
        base_text: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        unique: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            r = c.get("resolved_ref")
            if not r:
                continue
            if r not in unique:
                unique[r] = c
            else:
                ps, ns = unique[r].get("score"), c.get("score")
                if ns is not None and (ps is None or ns > ps):
                    unique[r] = c

        numbered: List[str] = []
        payloads: List[Tuple[int, Dict[str, Any]]] = []
        for i, (ref, cand) in enumerate(unique.items(), 1):
            txt = self._get_ref_text(ref, lang=lang)
            preview = (txt or "").strip()[:400]
            if txt and len(txt) > 400:
                preview += "..."
            numbered.append(f"{i}) {ref} (score={cand.get('score')})\n{preview}")
            payloads.append((i, cand))

        base_block = ""
        if base_ref and base_text:
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text[:2000]}\n\n"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You choose the single best candidate ref that the citing text most directly quotes/paraphrases. "
                    "You must respond with an actual number, not a placeholder or variable."
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    f"{base_block}"
                    "Candidate refs:\n{candidates}\n\n"
                    "Pick exactly ONE number from the list above (e.g., 1, 2, 3, etc.).\n\n"
                    "Output format (replace with actual content, do NOT use placeholders):\n"
                    "Reason: Your brief explanation here\n"
                    "Choice: 1\n\n"
                    "Now provide your answer:",
                ),
            ]
        )

        chain = prompt | self.llm
        try:
            resp = chain.invoke(
                {"citing": citing_text[:6000], "candidates": "\n\n".join(numbered)},
                config={
                    "metadata": {
                        "target_ref": target_ref,
                        "num_candidates": len(payloads),
                        "has_base_text": base_ref is not None,
                        "base_ref": base_ref or "none"
                    },
                    "tags": ["candidate_selection", "llm_choice", "citation_disambiguation"]
                }
            )
            self._profile_add_tokens(self.llm, resp)
            content = getattr(resp, "content", "")
        except Exception as exc:
            self.debug.log(f"LLM choose-best failed: {exc}")
            return None

        m = re.search(r"choice\s*:\s*(\d+)", content, re.IGNORECASE)
        if not m:
            nums = re.findall(r"\d+", content or "")
            if not nums:
                return None
            choice = int(nums[0])
        else:
            choice = int(m.group(1))

        for idx, cand in payloads:
            if idx == choice:
                cand["llm_choice_reason"] = content
                raw = cand.get("raw", {})
                if isinstance(raw, dict):
                    bm, cm = raw.get("baseMatchedText"), raw.get("compMatchedText")
                    if bm or cm:
                        self.debug.log("Selected candidate matched text:")
                        if bm:
                            self.debug.log(f"  baseMatchedText: {bm}")
                        if cm:
                            self.debug.log(f"  compMatchedText: {cm}")
                return cand

        return None

    def _llm_choose_among_matches(
        self,
        matching_candidates: List[Dict[str, Any]],
        citing_text: str,
        base_ref: Optional[str],
        base_text: Optional[str],
        lang: Optional[str],
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to choose the best candidate when multiple matches are found.

        Args:
            matching_candidates: List of dicts with 'candidate', 'score'/'query', 'raw'
            citing_text: The citing passage text
            base_ref: Base text ref if commentary
            base_text: Base text if commentary
            lang: Language
            source: "Dicta" or "Search" for logging
        """
        # Build options for LLM
        options = []
        for i, match in enumerate(matching_candidates, 1):
            cand_ref = match["candidate"]["ref"]

            # Get score or query info
            if "score" in match:
                info = f"Dicta score: {match['score']}"
            elif "query" in match:
                info = f"Search query: '{match['query']}'"
            else:
                info = "Match found"

            # Get text preview
            cand_text = self.base_resolver._get_ref_text(cand_ref, lang)
            preview = (cand_text or "").strip()[:300]
            if cand_text and len(cand_text) > 300:
                preview += "..."

            options.append(
                f"{i}) {cand_ref}\n"
                f"   {info}\n"
                f"   Text: {preview}"
            )

        # Escape curly braces
        def escape_braces(text: str) -> str:
            if not text:
                return text
            return text.replace("{", "{{").replace("}", "}}")

        citing_escaped = escape_braces(citing_text[:6000])
        options_escaped = escape_braces("\n\n".join(options))

        base_block = ""
        if base_ref and base_text:
            base_text_escaped = escape_braces(base_text[:2000])
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text_escaped}\n\n"

        # LLM prompt to choose
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"You are choosing the best match from multiple {source} results for an ambiguous citation. "
                "You must respond with an actual number, not a placeholder or variable."
            ),
            ("human",
             "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
             "{base_block}"
             "Multiple possible matches found:\n{options}\n\n"
             "Pick exactly ONE number from the list above (e.g., 1, 2, 3, etc.).\n\n"
             "Output format (replace with actual content, do NOT use placeholders):\n"
             "Reason: Your brief explanation here\n"
             "Choice: 1\n\n"
             "Now provide your answer:"
            ),
        ])

        chain = prompt | self.base_resolver.llm
        try:
            resp = chain.invoke(
                {
                    "citing": citing_escaped,
                    "options": options_escaped,
                    "base_block": base_block
                },
                config={
                    "metadata": {
                        "num_options": len(options),
                        "source": source,
                    },
                    "tags": ["ambiguous_resolution", "multiple_matches", "citation_disambiguation"]
                }
            )
            self.base_resolver._profile_add_tokens(self.base_resolver.llm, resp)
            content = getattr(resp, "content", "")
        except Exception as exc:
            self.debug.log(f"LLM choice among matches failed: {exc}")
            return None

        # Parse the choice
        import re
        m = re.search(r"choice\s*:\s*(\d+)", content, re.IGNORECASE)
        if not m:
            nums = re.findall(r"\d+", content or "")
            if not nums:
                self.debug.log(f"Could not parse LLM choice from: {content}")
                return None
            choice = int(nums[0])
        else:
            choice = int(m.group(1))

        if 1 <= choice <= len(matching_candidates):
            match = matching_candidates[choice - 1]
            enriched_candidate = dict(match["candidate"])
            enriched_candidate["resolved_ref"] = match.get("resolved_ref", match["candidate"]["ref"])
            resolved = enriched_candidate["resolved_ref"]
            self.debug.log(f"LLM chose option {choice}: {match['candidate']['ref']} → {resolved}")
            return enriched_candidate
        else:
            self.debug.log(f"LLM choice {choice} out of range (1-{len(matching_candidates)})")
            return None

    def _llm_confirm_match(
        self,
        marked_citing_text: str,
        candidate_ref: str,
        candidate_text: str,
        base_ref: Optional[str],
        base_text: Optional[str],
        citing_ref: str
    ) -> bool:
        """
        Use LLM to confirm if the candidate is a correct match for the citation.
        Returns True if confirmed, False if rejected.
        """
        # Special case: Metzudat Zion is always confirmed
        if citing_ref.startswith("Metzudat Zion"):
            self.debug.log("Auto-confirming Metzudat Zion")
            return True

        # Use the existing LLM confirmation method
        confirmed, reason = self.base_resolver._llm_confirm_candidate(
            citing_text=marked_citing_text,
            candidate_ref=candidate_ref,
            candidate_text=candidate_text,
            base_ref=base_ref,
            base_text=base_text,
        )

        self.debug.log(f"LLM confirmation for {candidate_ref}: {confirmed}, reason: {reason}")
        return confirmed

    def _build_simple_resolution(
        self,
        candidate: Dict[str, Any],
        source: str,
        citing_ref: str,
        lang: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build a resolution result from a confirmed candidate.
        Returns the candidate ref as resolved_ref, with the matched segment as metadata.
        """
        # resolved_ref is the candidate (e.g., "Genesis 7:1-3")
        # matched_segment is the specific segment Dicta/Search found (e.g., "Genesis 7:2")
        matched_segment = candidate.get("resolved_ref")  # The segment-level ref from Dicta/Search
        result = {
            "span": candidate["span"],
            "original_ref": candidate["ref"],
            "resolved_ref": candidate["ref"],  # Return the candidate ref
            "reason": f"Matched via {source} and confirmed by LLM",
            "llm_reason": f"Confirmed as correct match",
            "match_source": source,
        }
        # Add the matched segment as metadata if it exists and differs from candidate
        if matched_segment and matched_segment != candidate["ref"]:
            result["matched_segment"] = matched_segment
        return result

    def _build_debug_url(self, tref: str, lang: Optional[str] = None) -> str:
        """
        Build a Sefaria library URL with debug_mode=linker for easy debugging.
        """
        if not tref:
            return ""
        try:
            from urllib.parse import quote
            oref = Ref(tref)
            url_safe_ref = oref.url()
            base_url = f"https://library-linker2.cauldron.sefaria.org/{url_safe_ref}"
            params = []
            if lang:
                params.append(f"lang={lang}")
            params.append("debug_mode=linker")
            return base_url + "?" + "&".join(params)
        except Exception:
            # Fallback if ref parsing fails
            try:
                from urllib.parse import quote
                url_safe_ref = quote(tref.replace(" ", "_"))
                base_url = f"https://library-linker2.cauldron.sefaria.org/{url_safe_ref}"
                params = []
                if lang:
                    params.append(f"lang={lang}")
                params.append("debug_mode=linker")
                return base_url + "?" + "&".join(params)
            except Exception:
                return ""


class LLMParallelAmbiguousResolver:
    """
    Resolves ambiguous citation spans in linker_output documents.
    Uses Dicta and search to find matches among ambiguous candidates,
    then LLM to confirm correctness.
    """

    def __init__(
        self,
        debug: bool = False,
        window_words_per_side: int = 120,
        min_threshold: float = 1.0,
        max_distance: float = 10.0,
        **kwargs
    ):
        """
        Args:
            debug: Enable debug logging
            window_words_per_side: Words to include on each side of citation
            min_threshold: Minimum threshold for Dicta matching (default: 7.0)
            max_distance: Maximum distance for Dicta matching (default: 4.0)
            **kwargs: Additional config parameters passed to LLMParallelResolver
        """
        self.base_resolver = LLMParallelResolver(
            debug=debug,
            window_words_per_side=window_words_per_side,
            min_threshold=min_threshold,
            max_distance=max_distance,
            **kwargs
        )
        self.debug = self.base_resolver.debug

    @traceable(run_type="chain", name="resolve_ambiguous_linker_output")
    def resolve(
        self,
        linker_output: dict,
        profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve all ambiguous citation spans in a linker_output document.
        """
        self.base_resolver._profile_reset(profile)

        citing_ref = linker_output.get("ref")
        lang = linker_output.get("language")
        vtitle = linker_output.get("versionTitle")
        spans = linker_output.get("spans", [])

        self.debug.log(f"Processing linker_output document: ref={citing_ref}, lang={lang}, total_spans={len(spans)}")

        # Filter to only ambiguous citation spans
        ambiguous_citations = [
            span for span in spans
            if span.get("ambiguous") is True and span.get("type") == "citation"
        ]

        if not ambiguous_citations:
            self.debug.log("No ambiguous citation spans found")
            return {
                "ref": citing_ref,
                "total_groups": 0,
                "resolved_groups": 0,
                "unresolved_groups": 0,
                "total_ambiguous_spans": 0,
                "resolutions": [],
                "unresolved_spans": [],
                "profile": profile or {}
            }

        self.debug.log(f"Found {len(ambiguous_citations)} ambiguous citation spans")

        # Group ambiguous spans by charRange - spans with same charRange are alternative resolutions
        from collections import defaultdict
        char_range_groups = defaultdict(list)
        for span in ambiguous_citations:
            char_range = tuple(span.get('charRange', []))
            if char_range:
                char_range_groups[char_range].append(span)

        self.debug.log(f"Grouped into {len(char_range_groups)} unique citation locations")

        # Get the full text of the citing document once
        citing_text = self.base_resolver._get_ref_text(citing_ref, lang, vtitle)
        if not citing_text:
            self.debug.log(f"Could not retrieve text for citing ref: {citing_ref}")
            total_groups = len(char_range_groups)

            return {
                "ref": citing_ref,
                "total_groups": total_groups,
                "resolved_groups": 0,
                "unresolved_groups": total_groups,
                "total_ambiguous_spans": len(ambiguous_citations),
                "resolutions": [],
                "unresolved_spans": ambiguous_citations,
                "profile": profile or {}
            }

        resolutions = []
        unresolved_spans = []

        # Process each group of ambiguous spans (one group per citation location)
        for group_idx, (char_range, span_group) in enumerate(sorted(char_range_groups.items()), 1):
            self.debug.log(
                f"Processing group {group_idx}/{len(char_range_groups)}: "
                f"charRange={list(char_range)}, {len(span_group)} alternative(s), "
                f"text='{span_group[0].get('text', '')[:30]}...'"
            )

            resolution = self._resolve_span_group(
                span_group=span_group,
                citing_ref=citing_ref,
                citing_text=citing_text,
                lang=lang,
                vtitle=vtitle,
                linker_output=linker_output
            )

            if resolution:
                resolutions.append(resolution)
                self.debug.log(f"✓ Resolved group {group_idx} to: {resolution.get('resolved_ref')}")
            else:
                unresolved_spans.extend(span_group)
                self.debug.log(f"✗ Could not resolve group {group_idx}")

        # Count groups, not individual spans
        total_groups = len(char_range_groups)
        resolved_groups = len(resolutions)
        unresolved_groups = total_groups - resolved_groups

        result = {
            "ref": citing_ref,
            "total_groups": total_groups,
            "resolved_groups": resolved_groups,
            "unresolved_groups": unresolved_groups,
            "total_ambiguous_spans": len(ambiguous_citations),
            "resolutions": resolutions,
            "unresolved_spans": unresolved_spans,
            "profile": profile or {}
        }

        self.debug.log(
            f"Completed: {resolved_groups}/{total_groups} groups resolved "
            f"({unresolved_groups} unresolved groups, {len(unresolved_spans)} unresolved spans)"
        )

        return result

    @traceable(run_type="chain", name="resolve_span_group")
    def _resolve_span_group(
        self,
        span_group: List[dict],
        citing_ref: str,
        citing_text: str,
        lang: Optional[str],
        vtitle: Optional[str],
        linker_output: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a group of ambiguous spans that represent alternative resolutions for the same citation.
        """
        if not span_group:
            return None

        # Check if this is Hebrew (required for Dicta + search pipeline)
        if lang != "he":
            self.debug.log(f"Skipping non-Hebrew span group (lang={lang})")
            return None

        # Use the first span to get the text and charRange (all spans in group have same text/range)
        representative_span = span_group[0]
        citation_text = representative_span.get("text", "")

        # Print citing URL
        citing_url = self._build_debug_url(citing_ref, lang)
        self.debug.log(f"Citing ref: {citing_ref}")
        self.debug.log(f"  URL: {citing_url}")

        self.debug.log(
            f"Resolving group with {len(span_group)} alternatives for '{citation_text}': "
            f"{[s.get('ref') for s in span_group]}"
        )

        # Normalize all candidate refs first
        valid_candidates = []
        for span in span_group:
            candidate_ref = span.get("ref")
            if not candidate_ref:
                continue
            try:
                oref = Ref(candidate_ref)
                valid_candidates.append({
                    "span": span,
                    "ref": oref.normal(),
                    "oref": oref
                })
            except Exception as e:
                self.debug.log(f"Invalid ref in span: {candidate_ref}, error: {e}")
                continue

        if not valid_candidates:
            self.debug.log("No valid candidates found")
            return None

        # Debug: Print all candidates
        self.debug.log(f"Valid candidates ({len(valid_candidates)}):")
        for i, cand in enumerate(valid_candidates, 1):
            cand_url = self._build_debug_url(cand['ref'], lang)
            self.debug.log(f"  {i}. {cand['ref']}")
            self.debug.log(f"     {cand_url}")

        # Get context for matching
        base_ref, base_text = self.base_resolver._get_commentary_base_context(citing_ref)
        citing_window, span_window = self.base_resolver._window_around_span(
            citing_text, representative_span, self.base_resolver.cfg.window_words_per_side
        )
        marked_citing_text = self.base_resolver._mark_citation(citing_window, span_window)

        # Step 1: Try Dicta once to see if it matches any of our candidates
        self.debug.log("Trying Dicta to find match among candidates...")
        dicta_match = self._try_dicta_for_candidates(
            citing_window=citing_window,
            candidates=valid_candidates,
            citing_ref=citing_ref,
            lang=lang,
            vtitle=vtitle,
            base_ref=base_ref,
            base_text=base_text
        )

        if dicta_match:
            # Dicta found a match - verify with LLM
            # Use resolved_ref if available (segment-level), otherwise use original ref
            match_ref = dicta_match.get('resolved_ref', dicta_match['ref'])
            self.debug.log(f"Dicta found match: {dicta_match['ref']} → {match_ref}, verifying with LLM...")
            confirmed = self._llm_confirm_match(
                marked_citing_text=marked_citing_text,
                candidate_ref=match_ref,
                candidate_text=self.base_resolver._get_ref_text(match_ref, lang),
                base_ref=base_ref,
                base_text=base_text,
                citing_ref=citing_ref
            )

            if confirmed:
                self.debug.log(f"✓ LLM confirmed Dicta match: {match_ref}")
                return self._build_simple_resolution(dicta_match, "dicta", citing_ref, lang)
            else:
                self.debug.log(f"✗ LLM rejected Dicta match: {match_ref}")

        # Step 2: Dicta missed or was rejected - try search queries
        self.debug.log("Dicta didn't find valid match, trying search queries...")
        search_match = self._try_search_for_candidates(
            marked_citing_text=marked_citing_text,
            citing_text=citing_text,
            representative_span=representative_span,
            candidates=valid_candidates,
            citing_ref=citing_ref,
            lang=lang,
            vtitle=vtitle,
            base_ref=base_ref,
            base_text=base_text
        )

        if search_match:
            # Search found a match - verify with LLM
            # Use resolved_ref if available (segment-level), otherwise use original ref
            match_ref = search_match.get('resolved_ref', search_match['ref'])
            self.debug.log(f"Search found match: {search_match['ref']} → {match_ref}, verifying with LLM...")
            confirmed = self._llm_confirm_match(
                marked_citing_text=marked_citing_text,
                candidate_ref=match_ref,
                candidate_text=self.base_resolver._get_ref_text(match_ref, lang),
                base_ref=base_ref,
                base_text=base_text,
                citing_ref=citing_ref
            )

            if confirmed:
                self.debug.log(f"✓ LLM confirmed search match: {match_ref}")
                return self._build_simple_resolution(search_match, "search", citing_ref, lang)
            else:
                self.debug.log(f"✗ LLM rejected search match: {match_ref}")

        # Step 3: Nothing worked - return None
        self.debug.log("Could not find valid match among candidates")
        return None

    def _try_dicta_for_candidates(
        self,
        citing_window: str,
        candidates: List[Dict[str, Any]],
        citing_ref: str,
        lang: Optional[str],
        vtitle: Optional[str],
        base_ref: Optional[str],
        base_text: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Query Dicta once with citing text to see if it matches any of the candidates.
        If multiple candidates match, uses LLM to choose the best one.
        Returns the matching candidate, or None if no match.
        """
        # Build a set of candidate refs for quick lookup
        candidate_refs = {c["ref"] for c in candidates}

        # Query Dicta and get raw results
        try:
            start = time.perf_counter()
            params = {
                "minthreshold": int(self.base_resolver.cfg.min_threshold) if self.base_resolver.cfg.min_threshold is not None else "",
                "maxdistance": int(self.base_resolver.cfg.max_distance) if self.base_resolver.cfg.max_distance is not None else "",
            }
            headers = {
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Origin": "https://parallels.dicta.org.il",
                "Referer": "https://parallels.dicta.org.il/",
            }

            # Debug: Show exact URL and params
            param_str = "&".join([f"{k}={v}" for k, v in params.items() if v != ""])
            full_url = f"{self.base_resolver.cfg.dicta_url}?{param_str}"
            self.debug.log(f"Dicta URL: {full_url}")
            self.debug.log(f"Dicta text query (first 200 chars): {citing_window[:200]}")

            data = self.base_resolver.http.post_form(
                self.base_resolver.cfg.dicta_url,
                params=params,
                data=f"text={citing_window}".encode("utf-8"),
                headers=headers,
            )
            self.base_resolver._profile_add("dicta_seconds", time.perf_counter() - start)
        except Exception as exc:
            self.debug.log(f"Dicta request failed: {exc}")
            return None

        # Collect all Dicta results that match our candidates
        matching_candidates = []
        results = data.get("results") or []

        for entry in results:
            for hit in entry.get("data", []):
                url = hit.get("url") or hit.get("compUrl") or ""
                normalized = self.base_resolver._normalize_dicta_url_to_ref(url)
                if not normalized:
                    continue

                try:
                    result_oref = Ref(normalized)

                    # Check if this result is contained within any of our candidates
                    for cand in candidates:
                        try:
                            cand_oref = cand["oref"]
                            # Check containment: candidate contains result
                            # E.g., "Genesis 7" contains "Genesis 7:15"
                            if cand_oref.contains(result_oref):
                                score = hit.get("score")
                                self.debug.log(f"Dicta found matching candidate: {normalized} matches {cand['ref']} (score={score})")
                                matching_candidates.append({
                                    "candidate": cand,
                                    "score": score,
                                    "raw": hit,
                                    "resolved_ref": normalized  # Store the actual segment-level ref
                                })
                                break  # Only match to first candidate that contains it
                        except Exception:
                            continue
                except Exception:
                    continue

        if not matching_candidates:
            self.debug.log(f"Dicta returned no matches from candidates: {candidate_refs}")
            return None

        # Debug: Show summary of all Dicta matches
        self.debug.log(f"Dicta matches summary ({len(matching_candidates)} total):")
        for i, match in enumerate(matching_candidates, 1):
            resolved = match.get('resolved_ref', match['candidate']['ref'])
            self.debug.log(f"  {i}. {match['candidate']['ref']} → {resolved} (score={match['score']})")

        # Deduplicate by segment-level resolved_ref, keeping highest score
        deduped_by_segment = {}
        for match in matching_candidates:
            segment_ref = match.get('resolved_ref', match['candidate']['ref'])
            if segment_ref not in deduped_by_segment:
                deduped_by_segment[segment_ref] = match
            else:
                # Keep the match with higher score
                existing_score = deduped_by_segment[segment_ref].get('score', 0)
                new_score = match.get('score', 0)
                if new_score > existing_score:
                    deduped_by_segment[segment_ref] = match

        deduped_matches = list(deduped_by_segment.values())

        if len(deduped_matches) < len(matching_candidates):
            self.debug.log(f"Deduped {len(matching_candidates)} matches to {len(deduped_matches)} unique segments")

        # If only one unique segment, return it directly
        if len(deduped_matches) == 1:
            match = deduped_matches[0]
            enriched_candidate = dict(match["candidate"])
            enriched_candidate["resolved_ref"] = match.get("resolved_ref", match["candidate"]["ref"])
            self.debug.log(f"Dicta matched single unique segment: {match['candidate']['ref']} → {enriched_candidate['resolved_ref']}")
            return enriched_candidate

        # Multiple unique segments - use LLM to choose the single best segment for confirmation
        self.debug.log(f"Dicta matched {len(deduped_matches)} unique segments, using LLM to choose best segment")
        return self._llm_choose_among_matches(
            deduped_matches,
            citing_window,
            base_ref,
            base_text,
            lang,
            "Dicta"
        )

    def _try_search_for_candidates(
        self,
        marked_citing_text: str,
        citing_text: str,
        representative_span: dict,
        candidates: List[Dict[str, Any]],
        citing_ref: str,
        lang: Optional[str],
        vtitle: Optional[str],
        base_ref: Optional[str],
        base_text: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate search queries and try to match any of the candidates.
        If multiple candidates match, uses LLM to choose the best one.
        Returns the matching candidate, or None if no match.
        """
        # Build a set of candidate refs for quick lookup
        candidate_refs = {c["ref"] for c in candidates}

        # Generate search queries from context
        queries = self.base_resolver._llm_form_search_query(
            marked_citing_text, base_ref=base_ref, base_text=base_text
        )

        if not queries:
            self.debug.log("LLM couldn't generate search queries")
            return None

        self.debug.log(f"Generated {len(queries)} search queries: {queries}")

        # Collect all matching candidates from all queries
        matching_candidates = []
        seen_refs = set()

        # Try each query
        for query in queries:
            search_result = self.base_resolver._query_sefaria_search(
                query_text=query,
                target_ref=None,  # Don't filter, check all results
                citing_ref=citing_ref,
                citing_lang=lang,
                citing_version=vtitle,
            )

            if search_result:
                search_ref = search_result.get("resolved_ref")
                if search_ref not in seen_refs:
                    try:
                        result_oref = Ref(search_ref)
                        if not result_oref.is_segment_level():
                            continue

                        # Check if this result is contained within any of our candidates
                        for cand in candidates:
                            try:
                                cand_oref = cand["oref"]
                                # Check containment: candidate contains result
                                if cand_oref.contains(result_oref):
                                    self.debug.log(f"Search matched candidate: {search_ref} matches {cand['ref']} with query '{query}'")
                                    seen_refs.add(search_ref)
                                    matching_candidates.append({
                                        "candidate": cand,
                                        "query": query,
                                        "raw": search_result,
                                        "resolved_ref": search_ref  # Store the actual segment-level ref
                                    })
                                    break  # Only match to first candidate that contains it
                            except Exception:
                                continue
                    except Exception:
                        continue

        if not matching_candidates:
            self.debug.log("Search didn't match any candidates")
            return None

        # Debug: Show summary of all search matches
        self.debug.log(f"Search matches summary ({len(matching_candidates)} total):")
        for i, match in enumerate(matching_candidates, 1):
            resolved = match.get('resolved_ref', match['candidate']['ref'])
            self.debug.log(f"  {i}. {match['candidate']['ref']} → {resolved} (query='{match['query']}')")

        # Deduplicate by segment-level resolved_ref, keeping first occurrence
        deduped_by_segment = {}
        for match in matching_candidates:
            segment_ref = match.get('resolved_ref', match['candidate']['ref'])
            if segment_ref not in deduped_by_segment:
                deduped_by_segment[segment_ref] = match

        deduped_matches = list(deduped_by_segment.values())

        if len(deduped_matches) < len(matching_candidates):
            self.debug.log(f"Deduped {len(matching_candidates)} matches to {len(deduped_matches)} unique segments")

        # If only one unique segment, return it directly
        if len(deduped_matches) == 1:
            match = deduped_matches[0]
            enriched_candidate = dict(match["candidate"])
            enriched_candidate["resolved_ref"] = match.get("resolved_ref", match["candidate"]["ref"])
            self.debug.log(f"Search matched single unique segment: {match['candidate']['ref']} → {enriched_candidate['resolved_ref']}")
            return enriched_candidate

        # Multiple unique segments - use LLM to choose the single best segment for confirmation
        self.debug.log(f"Search matched {len(deduped_matches)} unique segments, using LLM to choose best segment")
        return self._llm_choose_among_matches(
            deduped_matches,
            marked_citing_text,
            base_ref,
            base_text,
            lang,
            "Search"
        )

    def _llm_choose_among_matches(
        self,
        matching_candidates: List[Dict[str, Any]],
        citing_text: str,
        base_ref: Optional[str],
        base_text: Optional[str],
        lang: Optional[str],
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to choose the best candidate when multiple matches are found.
        """
        # Build options for LLM
        options = []
        for i, match in enumerate(matching_candidates, 1):
            cand_ref = match["candidate"]["ref"]

            # Get score or query info
            if "score" in match:
                info = f"Dicta score: {match['score']}"
            elif "query" in match:
                info = f"Search query: '{match['query']}'"
            else:
                info = "Match found"

            # Get text preview
            cand_text = self.base_resolver._get_ref_text(cand_ref, lang)
            preview = (cand_text or "").strip()[:300]
            if cand_text and len(cand_text) > 300:
                preview += "..."

            options.append(
                f"{i}) {cand_ref}\n"
                f"   {info}\n"
                f"   Text: {preview}"
            )

        # Escape curly braces
        def escape_braces(text: str) -> str:
            if not text:
                return text
            return text.replace("{", "{{").replace("}", "}}")

        citing_escaped = escape_braces(citing_text[:6000])
        options_escaped = escape_braces("\n\n".join(options))

        base_block = ""
        if base_ref and base_text:
            base_text_escaped = escape_braces(base_text[:2000])
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text_escaped}\n\n"

        # LLM prompt to choose
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"You are choosing the best match from multiple {source} results for an ambiguous citation. "
                "You must respond with an actual number, not a placeholder or variable."
            ),
            ("human",
             "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
             "{base_block}"
             "Multiple possible matches found:\n{options}\n\n"
             "Pick exactly ONE number from the list above (e.g., 1, 2, 3, etc.).\n\n"
             "Output format (replace with actual content, do NOT use placeholders):\n"
             "Reason: Your brief explanation here\n"
             "Choice: 1\n\n"
             "Now provide your answer:"
            ),
        ])

        chain = prompt | self.base_resolver.llm
        try:
            resp = chain.invoke(
                {
                    "citing": citing_escaped,
                    "options": options_escaped,
                    "base_block": base_block
                },
                config={
                    "metadata": {
                        "num_options": len(options),
                        "source": source,
                    },
                    "tags": ["ambiguous_resolution", "multiple_matches", "citation_disambiguation"]
                }
            )
            self.base_resolver._profile_add_tokens(self.base_resolver.llm, resp)
            content = getattr(resp, "content", "")
        except Exception as exc:
            self.debug.log(f"LLM choice among matches failed: {exc}")
            return None

        # Parse the choice
        import re
        m = re.search(r"choice\s*:\s*(\d+)", content, re.IGNORECASE)
        if not m:
            nums = re.findall(r"\d+", content or "")
            if not nums:
                self.debug.log(f"Could not parse LLM choice from: {content}")
                return None
            choice = int(nums[0])
        else:
            choice = int(m.group(1))

        if 1 <= choice <= len(matching_candidates):
            match = matching_candidates[choice - 1]
            enriched_candidate = dict(match["candidate"])
            enriched_candidate["resolved_ref"] = match.get("resolved_ref", match["candidate"]["ref"])
            resolved = enriched_candidate["resolved_ref"]
            self.debug.log(f"LLM chose option {choice}: {match['candidate']['ref']} → {resolved}")
            return enriched_candidate
        else:
            self.debug.log(f"LLM choice {choice} out of range (1-{len(matching_candidates)})")
            return None

    def _llm_confirm_match(
        self,
        marked_citing_text: str,
        candidate_ref: str,
        candidate_text: str,
        base_ref: Optional[str],
        base_text: Optional[str],
        citing_ref: str
    ) -> bool:
        """
        Use LLM to confirm if the candidate is a correct match for the citation.
        Returns True if confirmed, False if rejected.
        """
        # Special case: Metzudat Zion is always confirmed
        if citing_ref.startswith("Metzudat Zion"):
            self.debug.log("Auto-confirming Metzudat Zion")
            return True

        # Use the existing LLM confirmation method
        confirmed, reason = self.base_resolver._llm_confirm_candidate(
            citing_text=marked_citing_text,
            candidate_ref=candidate_ref,
            candidate_text=candidate_text,
            base_ref=base_ref,
            base_text=base_text,
        )

        self.debug.log(f"LLM confirmation for {candidate_ref}: {confirmed}, reason: {reason}")
        return confirmed

    def _build_simple_resolution(
        self,
        candidate: Dict[str, Any],
        source: str,
        citing_ref: str,
        lang: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build a resolution result from a confirmed candidate.
        Returns the candidate ref as resolved_ref, with the matched segment as metadata.
        """
        # resolved_ref is the candidate (e.g., "Genesis 7:1-3")
        # matched_segment is the specific segment Dicta/Search found (e.g., "Genesis 7:2")
        matched_segment = candidate.get("resolved_ref")  # The segment-level ref from Dicta/Search
        result = {
            "span": candidate["span"],
            "original_ref": candidate["ref"],
            "resolved_ref": candidate["ref"],  # Return the candidate ref
            "reason": f"Matched via {source} and confirmed by LLM",
            "llm_reason": f"Confirmed as correct match",
            "match_source": source,
        }
        # Add the matched segment as metadata if it exists and differs from candidate
        if matched_segment and matched_segment != candidate["ref"]:
            result["matched_segment"] = matched_segment
        return result

    def _build_debug_url(self, tref: str, lang: Optional[str] = None) -> str:
        """
        Build a Sefaria library URL with debug_mode=linker for easy debugging.
        """
        if not tref:
            return ""
        try:
            from urllib.parse import quote
            oref = Ref(tref)
            url_safe_ref = oref.url()
            base_url = f"https://library-linker2.cauldron.sefaria.org/{url_safe_ref}"
            params = []
            if lang:
                params.append(f"lang={lang}")
            params.append("debug_mode=linker")
            return base_url + "?" + "&".join(params)
        except Exception:
            # Fallback if ref parsing fails
            try:
                from urllib.parse import quote
                url_safe_ref = quote(tref.replace(" ", "_"))
                base_url = f"https://library-linker2.cauldron.sefaria.org/{url_safe_ref}"
                params = []
                if lang:
                    params.append(f"lang={lang}")
                params.append("debug_mode=linker")
                return base_url + "?" + "&".join(params)
            except Exception:
                return ""


if __name__ == "__main__":
    from utils import get_random_ambiguous_linker_outputs

    ambiguous_resolver = LLMParallelAmbiguousResolver(debug=True)
    ambiguous_outputs = get_random_ambiguous_linker_outputs(seed=43, use_cache=True, use_remote=True, n=10, progress=True)

    for i, output in enumerate(ambiguous_outputs, 1):
        print(f"\n=== Ambiguous Output {i} ===")
        try:
            result = ambiguous_resolver.resolve(output)
            print(f"Ref: {result['ref']}")
            print(f"Total groups: {result['total_groups']}")
            print(f"Resolved groups: {result['resolved_groups']}")
            print(f"Unresolved groups: {result['unresolved_groups']}")
            print(f"Total ambiguous spans: {result['total_ambiguous_spans']}")
            for res in result['resolutions']:
                print(f"  - {res['original_ref']} → {res['resolved_ref']} ({res['match_source']})")
        except Exception as exc:
            print(f"Error resolving ambiguous output {i}: {exc}")

