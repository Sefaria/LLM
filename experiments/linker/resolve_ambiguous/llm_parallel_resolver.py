import html
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import django
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

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

    min_threshold: float = 1.0
    max_distance: float = 10.0
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

        # Step 2: Try early segment-based resolution (1-3 segments)
        early_result = self._try_early_segment_resolution(link, chunk, context)
        if early_result:
            return early_result

        # Step 3: Try Dicta and fallback search pipeline
        resolution = self._find_candidate_resolution(chunk, context)
        if not resolution:
            return None

        # Step 4: Confirm candidate with LLM
        return self._confirm_and_build_result(link, chunk, context, resolution)

    def _validate_and_extract_context(self, link: dict, chunk: dict) -> Optional[dict]:
        """Extract and validate all required context for resolution."""
        non_segment_ref = self._find_non_segment_ref(link)
        if not non_segment_ref:
            return None

        citing_ref = chunk.get("ref")
        lang = chunk.get("language")
        if lang and lang != "he":
            return None

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

    def _confirm_and_build_result(
        self, link: dict, chunk: dict, context: dict, resolution: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Confirm candidate with LLM and build final result."""
        resolved_ref = resolution.get("resolved_ref")
        if not resolved_ref:
            return None

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
                {"citing": marked_citing_text[:6000], "context": context_only[:6000], "base_block": base_block}
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
                ("system", "You verify whether one Jewish text is citing or closely paraphrasing a specific segment."),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "{base_block}"
                    "Candidate segment ref: {candidate_ref}\n"
                    "Candidate segment text:\n{candidate_text}\n\n"
                    "Answer in two lines:\nReason: <brief rationale>\nVerdict: YES or NO",
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
            verdict = "YES" if content.lower().startswith("y") else "NO"
        return verdict == "YES", content

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
                ("system", "You choose the single best candidate ref that the citing text most directly quotes/paraphrases."),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    f"{base_block}"
                    "Candidate refs:\n{candidates}\n\n"
                    "Pick exactly one number. Answer in two lines:\nReason: <brief rationale>\nChoice: <number>",
                ),
            ]
        )

        chain = prompt | self.llm
        try:
            resp = chain.invoke({"citing": citing_text[:6000], "candidates": "\n\n".join(numbered)})
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

    def _format_segment_texts(self, segments: List[Ref]) -> List[str]:
        formatted = []
        for i, seg in enumerate(segments, start=1):
            text = ""
            try:
                text = seg.text("en").as_string()
            except Exception:
                pass
            if not text:
                try:
                    text = seg.text("he").as_string()
                except Exception:
                    text = ""
            formatted.append(f"{i}. {seg.normal()} — {text}")
        return formatted

    def _parse_range_response(self, content: str) -> Optional[Tuple[int, int, str]]:
        if not content:
            return None
        start = end = None
        reason = content.strip()
        m = re.search(r"range\s*:\s*([0-9]+)\s*,\s*([0-9]+)", content, re.IGNORECASE)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            reason = re.split(r"range\s*:", content, flags=re.IGNORECASE)[0].replace("Explanation:", "").strip()
        else:
            nums = re.findall(r"\d+", content)
            if nums:
                start = int(nums[0])
                end = int(nums[1]) if len(nums) > 1 else int(nums[0])
        if start is None or end is None:
            return None
        return start, end, reason

    def _llm_pick_small_range(
        self,
        marked_citing_text: str,
        target_ref: str,
        segments: List[Ref],
        base_ref: Optional[str],
        base_text: Optional[str],
    ) -> Optional[Tuple[int, int, str]]:
        numbered_segments = self._format_segment_texts(segments)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You map a citation to a range of numbered segments. Provide a short reason, "
                    "then two integers: start and end numbers.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "Target higher-level ref: {target_ref}\n\n"
                    "{base_block}"
                    "Numbered segment texts:\n{segments}\n\n"
                    "Which two numbers (start and end) best cover the cited text? "
                    "If only one segment is relevant, repeat the same number for start and end. "
                    "Respond in two lines:\n"
                    "Explanation: <brief reason>\n"
                    "Range: <start>,<end>",
                ),
            ]
        )
        base_block = ""
        if base_ref and base_text:
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text}\n\n"

        chain = prompt | self.llm
        resp = chain.invoke(
            {
                "citing": marked_citing_text,
                "target_ref": target_ref,
                "segments": "\n".join(numbered_segments),
                "base_block": base_block,
            }
        )
        self._profile_add_tokens(self.llm, resp)
        return self._parse_range_response(getattr(resp, "content", ""))

    # -------- Dicta --------

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

        try:
            data = self.http.post_form(
                self.cfg.dicta_url,
                params=params,
                data=f"text={query_text}".encode("utf-8"),
                headers=headers,
            )
        except Exception as exc:
            self.debug.log(
                "Dicta request failed.\n"
                f"Payload (first 2000 chars): {query_text[:2000]}{'...' if len(query_text) > 2000 else ''}\n"
                f"Error: {exc}"
            )
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
            self.debug.log(
                "Dicta returned no contained matches. "
                f"Payload length={len(query_text)}"
            )
            return None

        deduped = self._dedupe_candidates_by_ref(candidates)
        if len(deduped) == 1:
            self.debug.log(f"Dicta success: payload_len={len(query_text)}")
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
            self.debug.log("LLM failed to pick among Dicta hits.")
        return chosen

    def _collect_dicta_hits(
        self,
        results: List[Dict[str, Any]],
        target_ref: Optional[str] = None,
        citing_ref: Optional[str] = None,
        citing_lang: Optional[str] = None,
        citing_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        target_oref = None
        if target_ref:
            try:
                target_oref = Ref(target_ref)
            except Exception:
                target_oref = None

        def safe_link(tref: Optional[str]) -> Optional[str]:
            if not tref:
                return None
            try:
                return f"https://library-linker2.cauldron.sefaria.org/{Ref(tref).url()}"
            except Exception:
                return None

        citing_link = safe_link(citing_ref)
        target_link = safe_link(target_ref) if target_oref else None

        debug_hits: List[str] = []
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
                        debug_hits.append(f"hit: citing={citing_link} resolved={safe_link(normalized)} score={score}")
                        candidates.append({"resolved_ref": normalized, "raw": cand, "source": "dicta", "score": score})
                except Exception:
                    continue

        if debug_hits:
            self.debug.log("Dicta hits:\n  " + "\n  ".join(debug_hits))
        elif any([citing_link, target_link, citing_lang, citing_version]):
            miss = []
            if citing_link:
                miss.append(f"citing={citing_link}")
            if target_link:
                miss.append(f"target={target_link}")
            if citing_lang:
                miss.append(f"lang={citing_lang}")
            if citing_version:
                miss.append(f"version={citing_version}")
            self.debug.log("Dicta misses:\n  " + "\n  ".join(miss))

        return candidates

    def _normalize_dicta_url_to_ref(self, url: str) -> Optional[str]:
        if not url:
            return None
        cleaned = url.lstrip("/")
        if cleaned.startswith("www.sefaria.org/"):
            cleaned = cleaned[len("www.sefaria.org/") :]
        if cleaned.startswith("sefaria.org/"):
            cleaned = cleaned[len("sefaria.org/") :]
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

    # -------- Sefaria search --------

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
        start = time.perf_counter()
        path_regex = self._path_regex_for_ref(target_ref)
        payload = {
            "from": 0,
            "size": size,
            "highlight": {"pre_tags": ["<b>"], "post_tags": ["</b>"], "fields": {"naive_lemmatizer": {"fragment_size": 200}}},
            "query": {
                "function_score": {
                    "field_value_factor": {"field": "pagesheetrank", "missing": 0.04},
                    "query": {
                        "bool": {
                            "must": {"match_phrase": {"naive_lemmatizer": {"query": query_text, "slop": slop}}},
                            "filter": ({"bool": {"should": [{"regexp": {"path": path_regex}}]}} if path_regex else {}),
                        }
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
            self.debug.log(f"Sefaria search-api request failed. query='{query_text}' error={exc}")
            return None
        finally:
            self._profile_add("es_seconds", time.perf_counter() - start)

        target_oref = None
        try:
            target_oref = Ref(target_ref) if target_ref else None
        except Exception:
            target_oref = None

        hits = (data.get("hits") or {}).get("hits") or []
        match: Optional[Dict[str, Any]] = None

        def safe_link(tref: Optional[str]) -> Optional[str]:
            if not tref:
                return None
            try:
                return f"https://library-linker2.cauldron.sefaria.org/{Ref(tref).url()}"
            except Exception:
                return None

        citing_link = safe_link(citing_ref)
        target_link = safe_link(target_ref) if target_oref else None

        debug_hits: List[str] = []
        for entry in hits:
            normalized = self._extract_ref_from_search_hit(entry)
            if not normalized:
                continue
            try:
                cand_oref = Ref(normalized)
                if not cand_oref.is_segment_level():
                    continue
                if target_oref and target_oref.contains(cand_oref):
                    debug_hits.append(f"hit: citing={citing_link} resolved={safe_link(normalized)}")
                    match = {"resolved_ref": normalized, "raw": entry, "source": "sefaria_search", "query": query_text}
                    break
            except Exception:
                continue

        if debug_hits:
            self.debug.log("Sefaria search hits:\n  " + "\n  ".join(debug_hits))
        else:
            miss = []
            if citing_link:
                miss.append(f"citing={citing_link}")
            if target_link:
                miss.append(f"target={target_link}")
            if citing_lang:
                miss.append(f"lang={citing_lang}")
            if citing_version:
                miss.append(f"version={citing_version}")
            if path_regex:
                miss.append(f"path_regex={path_regex}")
            self.debug.log("Sefaria search misses:\n  " + "\n  ".join(miss) if miss else "Sefaria search misses.")

        return match

    def _extract_ref_from_search_hit(self, hit: Dict[str, Any]) -> Optional[str]:
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

    # -------- misc --------

    def _dedupe_candidates_by_ref(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        updated = dict(link)
        updated["refs"] = [new_ref if r == old_ref else r for r in link.get("refs", [])]
        return updated

    def _replace_ref_in_chunk(self, chunk: dict, old_ref: str, new_ref: str) -> dict:
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


if __name__ == "__main__":
    from utils import get_random_non_segment_links_with_chunks

    resolver = LLMParallelResolver(window_words_per_side=30, debug=True)  # NEW
    samples = get_random_non_segment_links_with_chunks(n=60, use_remote=True, seed=71, use_cache=True)

    for i, item in enumerate(samples, 1):
        print(f"\n=== Sample {i} ===")
        try:
            print(resolver.resolve(item["link"], item["chunk"]))
        except Exception as exc:  # pragma: no cover
            print(f"Error resolving sample {i}: {exc}")
