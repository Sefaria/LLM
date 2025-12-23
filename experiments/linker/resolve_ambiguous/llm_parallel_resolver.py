import html
import os
from typing import Any, Dict, List, Optional, Tuple
import django
django.setup()
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
try:  # Optional: only if OpenAI is available
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None
from sefaria.model.text import Ref, AddressType


class LLMParallelResolver:
    """
    Resolve ambiguous non-segment refs by:
    1) Asking an LLM if the citing text is explicitly quoting or closely paraphrasing the cited ref.
    2) If yes, querying Dicta's parallels API to locate the precise segment where the parallel appears.
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
    ):
        if llm is not None:
            self.llm = llm
        else:
            # Default to Claude; caller can pass any LangChain chat model.
            self.llm = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
                temperature=0,
                max_tokens=256,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

        keyword_model = os.getenv("LLM_KEYWORD_MODEL")
        if keyword_model:
            if not ChatOpenAI:
                raise RuntimeError("LLM_KEYWORD_MODEL is set but langchain_openai is not installed.")
            self.keyword_llm = ChatOpenAI(
                model=keyword_model,
                temperature=0,
                max_tokens=256,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            # Default keyword extractor uses the main LLM.
            self.keyword_llm = self.llm
        # Dicta API configuration
        self.dicta_url = dicta_url or os.getenv(
            "DICTA_PARALLELS_URL",
            "https://parallels-3-0a.loadbalancer.dicta.org.il/parallels/api/findincorpus",
        )
        self.min_threshold = min_threshold
        self.max_distance = max_distance
        self.general_min_score = general_min_score
        self.tanakh_min_score = tanakh_min_score
        self.canonical_min_score = canonical_min_score
        self.min_frequency_to_count_phrase_as_one_word = min_frequency_to_count_phrase_as_one_word
        self.request_timeout = request_timeout
        self.window_words_per_side = window_words_per_side
        self.sefaria_search_url = sefaria_search_url or os.getenv(
            "SEFARIA_SEARCH_URL", "https://www.sefaria.org/api/search/text/_search"
        )

    def resolve(self, link: dict, chunk: dict) -> Optional[Dict[str, Any]]:
        """
        Attempt to resolve a non-segment ref via Dicta parallels when the citing text appears to
        explicitly quote or paraphrase the cited work.
        """
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
        citing_window, span_window = self._window_around_span(citing_text, span, self.window_words_per_side)
        marked_citing_text = self._mark_citation(citing_window, span_window)
        base_ref_for_prompt, base_text_for_prompt = self._get_commentary_base_context(citing_ref)

        resolution_result = self._query_dicta(
            citing_window,
            target_ref=non_segment_ref,
            citing_ref=citing_ref,
            citing_lang=chunk.get("language"),
            citing_version=chunk.get("versionTitle"),
            ranking_context=marked_citing_text,
            base_ref=base_ref_for_prompt,
            base_text=base_text_for_prompt,
        )
        if not resolution_result:
            # First try queries from the citing text alone
            searched_queries = set()
            search_candidates: List[Dict[str, Any]] = []
            search_queries = self._llm_form_search_query(marked_citing_text) or []
            for search_query in search_queries:
                if not search_query or search_query in searched_queries:
                    continue
                searched_queries.add(search_query)
                search_result = self._query_sefaria_search(
                    search_query,
                    target_ref=non_segment_ref,
                    citing_ref=citing_ref,
                    citing_lang=chunk.get("language"),
                    citing_version=chunk.get("versionTitle"),
                )
                if search_result:
                    print(f"Sefaria search succeeded with query '{search_query}' -> {search_result.get('resolved_ref')}")
                    search_candidates.append(search_result)
                else:
                    print(f"Sefaria search failed for query '{search_query}', retrying once...")
                    retry_result = self._query_sefaria_search(
                        search_query,
                        target_ref=non_segment_ref,
                        citing_ref=citing_ref,
                        citing_lang=chunk.get("language"),
                        citing_version=chunk.get("versionTitle"),
                    )
                    if retry_result:
                        print(
                            f"Sefaria search retry succeeded with query '{search_query}' "
                            f"-> {retry_result.get('resolved_ref')}"
                        )
                        search_candidates.append(retry_result)

            # If no hit, fall back to using base text (for commentaries) to seed queries
            if not resolution_result and base_text_for_prompt:
                base_queries = self._llm_form_search_query(
                    marked_citing_text,
                    base_ref=base_ref_for_prompt,
                    base_text=base_text_for_prompt,
                ) or []
                for search_query in base_queries:
                    if not search_query or search_query in searched_queries:
                        continue
                    searched_queries.add(search_query)
                    search_result = self._query_sefaria_search(
                        search_query,
                        target_ref=non_segment_ref,
                        citing_ref=citing_ref,
                        citing_lang=chunk.get("language"),
                        citing_version=chunk.get("versionTitle"),
                    )
                    if search_result:
                        print(
                            f"Sefaria search (base-text seeded) succeeded with query '{search_query}' "
                            f"-> {search_result.get('resolved_ref')}"
                        )
                        search_candidates.append(search_result)
                    else:
                        print(f"Sefaria search (base-text seeded) failed for query '{search_query}', retrying once...")
                        retry_result = self._query_sefaria_search(
                            search_query,
                            target_ref=non_segment_ref,
                            citing_ref=citing_ref,
                            citing_lang=chunk.get("language"),
                            citing_version=chunk.get("versionTitle"),
                        )
                        if retry_result:
                            print(
                                f"Sefaria search (base-text seeded) retry succeeded with query '{search_query}' "
                                f"-> {retry_result.get('resolved_ref')}"
                            )
                            search_candidates.append(retry_result)

            # If still no hit, expand window (2x current) and retry keyword generation/search
            if not resolution_result:
                expanded_words = max(self.window_words_per_side * 2, self.window_words_per_side + 1)
                if expanded_words > self.window_words_per_side:
                    expanded_window, expanded_span = self._window_around_span(citing_text, span, expanded_words)
                    expanded_marked = self._mark_citation(expanded_window, expanded_span)

                    # text-only queries on expanded context
                    expanded_queries = self._llm_form_search_query(expanded_marked) or []
                    for search_query in expanded_queries:
                        if not search_query or search_query in searched_queries:
                            continue
                        searched_queries.add(search_query)
                        search_result = self._query_sefaria_search(
                            search_query,
                            target_ref=non_segment_ref,
                            citing_ref=citing_ref,
                            citing_lang=chunk.get("language"),
                            citing_version=chunk.get("versionTitle"),
                        )
                        if search_result:
                            print(
                                f"Sefaria search (expanded window) succeeded with query '{search_query}' "
                                f"-> {search_result.get('resolved_ref')}"
                            )
                            search_candidates.append(search_result)
                        else:
                            print(
                                f"Sefaria search (expanded window) failed for query '{search_query}', retrying once..."
                            )
                            retry_result = self._query_sefaria_search(
                                search_query,
                                target_ref=non_segment_ref,
                                citing_ref=citing_ref,
                                citing_lang=chunk.get("language"),
                                citing_version=chunk.get("versionTitle"),
                            )
                            if retry_result:
                                print(
                                    f"Sefaria search (expanded window) retry succeeded with query '{search_query}' "
                                    f"-> {retry_result.get('resolved_ref')}"
                                )
                                search_candidates.append(retry_result)

                    # base-text seeded queries on expanded context
                    if not resolution_result and base_text_for_prompt:
                        base_expanded_queries = self._llm_form_search_query(
                            expanded_marked,
                            base_ref=base_ref_for_prompt,
                            base_text=base_text_for_prompt,
                        ) or []
                        for search_query in base_expanded_queries:
                            if not search_query or search_query in searched_queries:
                                continue
                            searched_queries.add(search_query)
                            search_result = self._query_sefaria_search(
                                search_query,
                                target_ref=non_segment_ref,
                                citing_ref=citing_ref,
                                citing_lang=chunk.get("language"),
                                citing_version=chunk.get("versionTitle"),
                            )
                            if search_result:
                                print(
                                    f"Sefaria search (expanded window + base text) succeeded with query '{search_query}' "
                                    f"-> {search_result.get('resolved_ref')}"
                                )
                                search_candidates.append(search_result)
                            else:
                                print(
                                    f"Sefaria search (expanded window + base text) failed for query '{search_query}', retrying once..."
                                )
                                retry_result = self._query_sefaria_search(
                                    search_query,
                                    target_ref=non_segment_ref,
                                    citing_ref=citing_ref,
                                    citing_lang=chunk.get("language"),
                                    citing_version=chunk.get("versionTitle"),
                                )
                                if retry_result:
                                    print(
                                        f"Sefaria search (expanded window + base text) retry succeeded with query '{search_query}' "
                                        f"-> {retry_result.get('resolved_ref')}"
                                    )
                                    search_candidates.append(retry_result)

            # If we accumulated candidates from search, pick best via LLM if more than one
            if search_candidates:
                deduped_search = self._dedupe_candidates_by_ref(search_candidates)
                if len(deduped_search) == 1:
                    resolution_result = deduped_search[0]
                else:
                    chosen = self._llm_choose_best_candidate(
                        marked_citing_text,
                        non_segment_ref,
                        deduped_search,
                        base_ref=base_ref_for_prompt,
                        base_text=base_text_for_prompt,
                        lang=chunk.get("language"),
                    )
                    if chosen:
                        print(
                            f"Sefaria search had {len(deduped_search)} candidates; LLM chose {chosen.get('resolved_ref')}"
                        )
                        resolution_result = chosen
        if not resolution_result:
            return None

        resolved_ref = resolution_result.get("resolved_ref")
        if not resolved_ref:
            return None

        candidate_text = self._get_ref_text(resolved_ref, chunk.get("language"))
        llm_ok, llm_reason = self._llm_confirm_candidate(
            marked_citing_text,
            non_segment_ref,
            resolved_ref,
            candidate_text,
            base_ref=base_ref_for_prompt,
            base_text=base_text_for_prompt,
        )
        if not llm_ok:
            print(
                f"LLM rejected Dicta pick: citing_ref={citing_ref} target_ref={non_segment_ref} "
                f"candidate_ref={resolved_ref} reason='{llm_reason}'"
            )
            return None
        else:
            print(
                f"LLM confirmed Dicta pick: citing_ref={citing_ref} target_ref={non_segment_ref} "
                f"candidate_ref={resolved_ref} reason='{llm_reason}'"
            )

        updated_link = self._replace_ref_in_link(link, non_segment_ref, resolved_ref)
        updated_chunk = self._replace_ref_in_chunk(chunk, non_segment_ref, resolved_ref)

        return {
            "link": updated_link,
            "chunk": updated_chunk,
            "original_ref": non_segment_ref,
            "resolved_ref": resolved_ref,
            "selected_segments": [resolved_ref],
            "reason": f"{resolution_result.get('source', 'Dicta').title()} hit {resolved_ref} confirmed by LLM",
            "llm_reason": llm_reason,
            "llm_verdict": "YES",
            "dicta_hit": resolution_result if resolution_result.get("source") == "dicta" else None,
            "search_hit": resolution_result if resolution_result.get("source") == "sefaria_search" else None,
            "match_source": resolution_result.get("source"),
        }

    def _find_non_segment_ref(self, link: dict) -> Optional[str]:
        """Return the first non-segment ref in the link, if any."""
        for tref in link.get("refs", []):
            try:
                oref = Ref(tref)
            except Exception:
                continue
            if not oref.is_segment_level():
                return oref.normal()
        return None

    def _get_ref_text(self, tref: str, lang: Optional[str] = None, vtitle: Optional[str] = None) -> str:
        """Fetch text for a ref with optional language/version preference."""
        if not tref:
            return ""
        vtitle = html.unescape(vtitle) if vtitle else None
        try:
            primary_lang = lang or "en"
            text = Ref(tref).text(primary_lang, vtitle=vtitle).as_string()
            if text:
                return text
            fallback_lang = "he" if primary_lang == "en" else "en"
            return Ref(tref).text(fallback_lang).as_string()
        except Exception:
            return ""

    def _window_around_span(self, text: str, span: Optional[dict], window_words: int) -> Tuple[str, Optional[dict]]:
        """
        Return a substring around the citation span, limited to window_words on each side.
        Adjust charRange accordingly for the returned substring.
        """
        if not text or not span:
            return text, span
        char_range = span.get("charRange")
        if not char_range or len(char_range) != 2:
            return text, span
        start, end = char_range
        if start < 0 or end > len(text) or start >= end:
            return text, span

        import re

        tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if not tokens:
            return text, span

        first_idx = None
        last_idx = None
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

        window_start = tokens[left][0]
        window_end = tokens[right][1]

        new_text = text[window_start:window_end]
        new_span = dict(span)
        new_span["charRange"] = [start - window_start, end - window_start]
        return new_text, new_span

    def _find_span_for_ref(self, chunk: dict, target_ref: str) -> Optional[dict]:
        spans = chunk.get("spans") or []
        for span in spans:
            if span.get("ref") == target_ref and span.get("type") == "citation":
                return span
        for span in spans:
            if span.get("type") == "citation":
                return span
        return None

    def _mark_citation(self, text: str, span: Optional[dict]) -> str:
        """Wrap the cited substring with <citation> tags to give the LLM explicit focus."""
        if not text or not span:
            return text
        char_range = span.get("charRange")
        ref_attr = span.get("ref")
        if not char_range or len(char_range) != 2:
            return text
        start, end = char_range
        if start < 0 or end > len(text) or start >= end:
            return text
        open_tag = "<citation"
        if ref_attr:
            open_tag += f' ref="{ref_attr}"'
        open_tag += ">"
        close_tag = "</citation>"
        return text[:start] + open_tag + text[start:end] + close_tag + text[end:]

    def _llm_form_search_query(
        self, marked_citing_text: str, base_ref: Optional[str] = None, base_text: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Ask the LLM to form short lexical search queries from the surrounding context (not the citation itself).
        """
        if not marked_citing_text:
            return None
        import re

        context_only = re.sub(
            r"<citation[^>]*>.*?</citation>", " [CITATION] ", marked_citing_text, flags=re.DOTALL
        )

        base_block = ""
        if base_ref and base_text:
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text[:3000]}\n\n"
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting a concise citation phrase to search for parallels.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "Context with citation redacted:\n{context}\n\n"
                    "{base_block}"
                    "Return 5-6 short lexical search queries (if fewer are possible, give as many as you can), "
                    "each <=6 words, taken from the surrounding context outside the citation span that best "
                    "describes what the citation is referring to. "
                    "If this is a commentary and base text is provided, prefer keywords that appear verbatim in the base text when relevant. "
                    "Include at least one very short keyword-only query (2-3 important words). "
                    "Prefer lexical forms likely to appear verbatim in the target text; avoid loose summaries. "
                    "Make each query as useful and distinguishing as possible without being verbose. "
                    "Do not copy words that appear inside the <citation>...</citation> span. "
                    "Do not include the placeholder [CITATION].\n\n"
                    "Output format (strict): one query per line, numbered like:\n"
                    "1) <query one>\n"
                    "2) <query two>\n"
                    "3) <query three>\n"
                    "4) <query four>\n"
                    "5) <query five>\n"
                    "6) <query six>\n"
                    "If no clear query is available, respond with a single line: NONE",
                ),
            ]
        )
        try:
            chain = prompt | self.keyword_llm
            resp = chain.invoke(
                {
                    "citing": marked_citing_text[:6000],
                    "context": context_only[:6000],
                    "base_block": base_block,
                }
            )
            content = getattr(resp, "content", "").strip()
            if not content or content.upper() == "NONE":
                return None
            queries: List[str] = []
            for line in content.splitlines():
                phrase = line.strip()
                if not phrase:
                    continue
                if phrase.upper() == "NONE":
                    queries = []
                    break
                # Strip numbering patterns like "1) text" or "1. text"
                import re

                phrase = re.sub(r"^\s*\d+[\).\s]+", "", phrase).strip()
                if not phrase:
                    continue
                if len(phrase.split()) > 6:
                    phrase = " ".join(phrase.split()[:6])
                if phrase and phrase not in queries:
                    queries.append(phrase)
                if len(queries) >= 6:
                    break
            if not queries:
                return None
            print(f"LLM formed lexical search queries: {queries}")
            return queries
        except Exception as exc:
            print(f"LLM lexical search query formation failed: {exc}")
            return None

    def _llm_confirm_candidate(
        self,
        citing_text: str,
        target_ref: str,
        candidate_ref: str,
        candidate_text: str,
        base_ref: Optional[str] = None,
        base_text: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Ask the LLM to confirm whether the Dicta-picked candidate is a likely direct/close paraphrase.
        """
        base_block = ""
        if base_ref and base_text:
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text[:3000]}\n\n"
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You verify whether a citing passage points to a specific target segment.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "{base_block}"
                    "Target ref: {target_ref}\n"
                    "Candidate segment ref: {candidate_ref}\n"
                    "Candidate segment text:\n{candidate_text}\n\n"
                    "Answer in two lines:\n"
                    "Reason: <brief rationale>\n"
                    "Verdict: YES or NO (YES if notable wording/phrase overlap is likely; NO if the link is purely thematic/indirect).",
                ),
            ]
        )
        chain = prompt | self.llm
        resp = chain.invoke(
            {
                "citing": citing_text[:6000],
                "base_block": base_block,
                "target_ref": target_ref,
                "candidate_ref": candidate_ref,
                "candidate_text": candidate_text[:6000],
            }
        )
        content = getattr(resp, "content", "").strip()
        verdict = None
        for line in content.splitlines():
            if line.lower().startswith("verdict"):
                for token in line.split():
                    t = token.strip().strip(":").strip().upper()
                    if t in {"YES", "NO"}:
                        verdict = t
                        break
        if verdict is None:
            verdict = "YES" if content.lower().startswith("y") else "NO"
        return verdict == "YES", content

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
        """
        Call Dicta's parallels API with the provided text and return the best hit (if any).
        """
        params = {
            "minthreshold": int(self.min_threshold) if self.min_threshold is not None else "",
            "maxdistance": int(self.max_distance) if self.max_distance is not None else "",
        }

        # Single payload: mirror the working browser request (form-encoded text).
        headers_form = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://parallels.dicta.org.il",
            "Referer": "https://parallels.dicta.org.il/",
        }
        kwargs = {
            "params": params,
            "timeout": self.request_timeout,
            "data": f"text={query_text}".encode("utf-8"),
            "headers": headers_form,
        }

        try:
            response = requests.post(self.dicta_url, **kwargs)
            response.raise_for_status()
            text = response.text.lstrip("\ufeff")
            try:
                data = response.json()
            except Exception:
                import json as _json

                data = _json.loads(text)
            candidates = self._collect_dicta_hits(
                data.get("results") or [],
                target_ref=target_ref,
                citing_ref=citing_ref,
                citing_lang=citing_lang,
                citing_version=citing_version,
            )
            if not candidates:
                print(
                    "Dicta returned no contained matches. Payload sent:\n"
                    f"{query_text[:5000]}{'...' if len(query_text) > 5000 else ''}"
                )
                return None
            deduped_candidates = self._dedupe_candidates_by_ref(candidates)
            if len(deduped_candidates) == 1:
                print(f"Dicta success with payload=text=<len {len(query_text)}> mode=form-string")
                return deduped_candidates[0]
            # Multiple candidates: let LLM choose best
            chosen = self._llm_choose_best_candidate(
                ranking_context or query_text,
                target_ref,
                deduped_candidates,
                base_ref=base_ref,
                base_text=base_text,
                lang=citing_lang,
            )
            if chosen:
                print(
                    "Dicta multiple hits; LLM chose "
                    f"{chosen.get('resolved_ref')} from {len(deduped_candidates)} candidates"
                )
                return chosen
            else:
                print("LLM failed to pick among Dicta hits.")
                return None
        except Exception as exc:
            print(
                "Dicta request failed. Payload sent:\n"
                f"{query_text[:5000]}{'...' if len(query_text) > 5000 else ''}\n"
                f"Error: {exc}"
            )
            return None

    def _collect_dicta_hits(
        self,
        results: List[Dict[str, Any]],
        target_ref: Optional[str] = None,
        citing_ref: Optional[str] = None,
        citing_lang: Optional[str] = None,
        citing_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Given Dicta results, return the best candidate (highest score) whose ref is contained within the target_ref.
        Prints all normalized hits and the selected one.
        """
        target_oref = None
        if target_ref:
            try:
                target_oref = Ref(target_ref)
            except Exception:
                target_oref = None

        debug_hits: List[str] = []
        candidates: List[Dict[str, Any]] = []
        citing_link = None
        try:
            if citing_ref:
                citing_link = f"https://library-linker2.cauldron.sefaria.org/{Ref(citing_ref).url()}"
        except Exception:
            citing_link = None
        target_link = None
        try:
            if target_oref:
                target_link = f"https://library-linker2.cauldron.sefaria.org/{target_oref.url()}"
        except Exception:
            target_link = None

        for entry in results:
            for candidate in entry.get("data", []):
                url = candidate.get("url") or candidate.get("compUrl") or ""
                normalized = self._normalize_dicta_url_to_ref(url)
                if not normalized:
                    continue

                try:
                    cand_oref = Ref(normalized)
                    cand_link = f"https://library-linker2.cauldron.sefaria.org/{cand_oref.url()}"
                    if target_oref and target_oref.contains(cand_oref):
                        score = candidate.get("score")
                        debug_hits.append(f"hit: citing={citing_link} resolved={cand_link} score={score}")
                        candidates.append(
                            {"resolved_ref": normalized, "raw": candidate, "source": "dicta", "score": score}
                        )
                except Exception:
                    continue

        if debug_hits:
            print("Dicta hits:\n  " + "\n  ".join(debug_hits))

        if not candidates:
            # Print misses with additional context if available
            miss_info = []
            if citing_link:
                miss_info.append(f"citing={citing_link}")
            if target_link:
                miss_info.append(f"target={target_link}")
            if citing_lang:
                miss_info.append(f"lang={citing_lang}")
            if citing_version:
                miss_info.append(f"version={citing_version}")
            if miss_info:
                print("Dicta misses:\n  " + "\n  ".join(miss_info))

        return candidates

    def _llm_choose_best_candidate(
        self,
        citing_text: str,
        target_ref: str,
        candidates: List[Dict[str, Any]],
        base_ref: Optional[str] = None,
        base_text: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ask the LLM to pick the best candidate ref among multiple hits.
        """
        if not candidates:
            return None

        # Deduplicate by resolved_ref (keep highest score version if duplicates)
        unique_candidates: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            ref = cand.get("resolved_ref")
            if not ref:
                continue
            if ref not in unique_candidates:
                unique_candidates[ref] = cand
            else:
                prev_score = unique_candidates[ref].get("score")
                new_score = cand.get("score")
                if new_score is not None and (prev_score is None or new_score > prev_score):
                    unique_candidates[ref] = cand

        # Load candidate texts for ranking
        numbered = []
        candidate_payloads = []
        for idx, (ref, cand) in enumerate(unique_candidates.items(), 1):
            cand_text = self._get_ref_text(ref, lang=lang) if ref else ""
            candidate_payloads.append((idx, cand, cand_text))
            preview = (cand_text or "").strip()
            preview = preview[:400] + ("..." if len(preview) > 400 else "")
            score = cand.get("score")
            numbered.append(f"{idx}) {ref} (score={score})\n{preview}")

        base_block = ""
        if base_ref and base_text:
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text[:2000]}\n\n"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You choose the single best candidate ref that the citing text most directly quotes or paraphrases.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    f"{base_block}"
                    "Candidate refs:\n{candidates}\n\n"
                    "Pick exactly one number. Answer in two lines:\n"
                    "Reason: <brief rationale>\n"
                    "Choice: <number>",
                ),
            ]
        )

        chain = prompt | self.llm
        try:
            resp = chain.invoke({"citing": citing_text[:6000], "candidates": "\n\n".join(numbered)})
            content = getattr(resp, "content", "")
        except Exception:
            return None

        choice = None
        import re

        match = re.search(r"choice\s*:\s*([0-9]+)", content, re.IGNORECASE)
        if match:
            choice = int(match.group(1))
        else:
            nums = re.findall(r"\d+", content)
            if nums:
                choice = int(nums[0])

        if choice is None:
            return None
        for idx, cand, _text in candidate_payloads:
            if idx == choice:
                raw = cand.get("raw", {})
                base_match = raw.get("baseMatchedText") if isinstance(raw, dict) else None
                comp_match = raw.get("compMatchedText") if isinstance(raw, dict) else None
                if base_match or comp_match:
                    print("Selected candidate matched text:")
                    if base_match:
                        print(f"  baseMatchedText: {base_match}")
                    if comp_match:
                        print(f"  compMatchedText: {comp_match}")
                cand["llm_choice_reason"] = content
                return cand
        return None

    def _dedupe_candidates_by_ref(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate candidates by resolved_ref, keeping highest score if duplicates."""
        unique: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            ref = cand.get("resolved_ref")
            if not ref:
                continue
            if ref not in unique:
                unique[ref] = cand
            else:
                prev_score = unique[ref].get("score")
                new_score = cand.get("score")
                if new_score is not None and (prev_score is None or new_score > prev_score):
                    unique[ref] = cand
        return list(unique.values())

    def _normalize_dicta_url_to_ref(self, url: str) -> Optional[str]:
        """Convert Dicta's URL (often starting with //www.sefaria.org/) into a normalized ref string."""
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
        """
        Call Sefaria's ElasticSearch proxy (text index) with a lexical query and return the first hit contained in target_ref.
        """
        path_regex = self._path_regex_for_ref(target_ref)
        payload = {
            "from": 0,
            "size": size,
            "highlight": {
                "pre_tags": ["<b>"],
                "post_tags": ["</b>"],
                "fields": {"naive_lemmatizer": {"fragment_size": 200}},
            },
            "query": {
                "function_score": {
                    "field_value_factor": {"field": "pagesheetrank", "missing": 0.04},
                    "query": {
                        "bool": {
                            "must": {
                                "match_phrase": {"naive_lemmatizer": {"query": query_text, "slop": slop}}
                            },
                            "filter": (
                                {
                                    "bool": {
                                        "should": [{"regexp": {"path": path_regex}}],
                                    }
                                }
                                if path_regex
                                else {}
                            ),
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
            resp = requests.post(
                self.sefaria_search_url,
                json=payload,
                timeout=self.request_timeout,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(
                "Sefaria search-api request failed.\n"
                f"Query: {query_text}\n"
                f"Error: {exc}"
            )
            return None

        target_oref = None
        try:
            target_oref = Ref(target_ref) if target_ref else None
        except Exception:
            target_oref = None

        hits = data.get("hits", {})
        hit_list = hits.get("hits") if isinstance(hits, dict) else hits or []

        debug_hits: List[str] = []
        match: Optional[Dict[str, Any]] = None
        citing_link = None
        try:
            if citing_ref:
                citing_link = f"https://library-linker2.cauldron.sefaria.org/{Ref(citing_ref).url()}"
        except Exception:
            citing_link = None
        target_link = None
        try:
            if target_oref:
                target_link = f"https://library-linker2.cauldron.sefaria.org/{target_oref.url()}"
        except Exception:
            target_link = None

        for entry in hit_list:
            normalized = self._extract_ref_from_search_hit(entry)
            if not normalized:
                continue
            try:
                cand_oref = Ref(normalized)
                cand_link = f"https://library-linker2.cauldron.sefaria.org/{cand_oref.url()}"
                if target_oref and target_oref.contains(cand_oref):
                    debug_hits.append(f"hit: citing={citing_link} resolved={cand_link}")
                    match = {"resolved_ref": normalized, "raw": entry, "source": "sefaria_search", "query": query_text}
                    break
            except Exception:
                continue

        if debug_hits:
            print("Sefaria search hits:\n  " + "\n  ".join(debug_hits))
        else:
            miss_info = []
            if citing_link:
                miss_info.append(f"citing={citing_link}")
            if target_link:
                miss_info.append(f"target={target_link}")
            if citing_lang:
                miss_info.append(f"lang={citing_lang}")
            if citing_version:
                miss_info.append(f"version={citing_version}")
            if path_regex:
                miss_info.append(f"path_regex={path_regex}")
            print("Sefaria search misses:\n  " + "\n  ".join(miss_info) if miss_info else "Sefaria search misses.")

        return match

    def _extract_ref_from_search_hit(self, hit: Dict[str, Any]) -> Optional[str]:
        """Try to pull a ref string from a Sefaria search hit."""
        candidates = []
        for key in ("ref", "he_ref", "sourceRef"):
            if key in hit:
                candidates.append(hit.get(key))
        src = hit.get("_source") or {}
        for key in ("ref", "he_ref", "sourceRef"):
            if key in src:
                candidates.append(src.get(key))
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return Ref(candidate).normal()
            except Exception:
                continue
        return None

    def _path_regex_for_ref(self, target_ref: Optional[str]) -> Optional[str]:
        """
        Build a regex for the `path` field to constrain search to the relevant book/category tree.
        Example: "Mishnah/Seder Zeraim/Mishnah Kilayim.*"
        """
        if not target_ref:
            return None
        try:
            oref = Ref(target_ref)
            idx = oref.index
            parts = []
            try:
                parts.extend(idx.categories or [])
            except Exception:
                pass
            try:
                parts.append(idx.title)
            except Exception:
                pass
            if not parts:
                return None
            path = "/".join(parts)
            return path.replace("/", r"\/") + ".*"
        except Exception:
            return None

    def _get_commentary_base_context(self, citing_ref: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        For commentary refs, fetch the aligned base ref and text for prompt context.
        Returns (base_ref, base_text).
        """
        if not citing_ref:
            return None, None
        try:
            citing_oref = Ref(citing_ref)
            base_titles = getattr(citing_oref.index, "base_text_titles", []) or []
            if not base_titles:
                return None, None
            base_title = base_titles[0]
            section_ref = citing_oref.section_ref()
            sec_parts = section_ref.sections
            addr_types = section_ref.index_node.addressTypes
            for sec, addr_type in zip(sec_parts, addr_types):
                address = AddressType.to_str_by_address_type(addr_type, "en", sec)
                base_title += f" {address}"
            base_ref = Ref(base_title)
            base_ref_norm = base_ref.normal()
            base_text = self._get_ref_text(base_ref_norm, lang="he") or self._get_ref_text(base_ref_norm, lang="en")
            return base_ref_norm, base_text
        except Exception:
            return None, None

    def _replace_ref_in_link(self, link: dict, old_ref: str, new_ref: str) -> dict:
        updated = dict(link)
        updated_refs = []
        for tref in link.get("refs", []):
            updated_refs.append(new_ref if tref == old_ref else tref)
        updated["refs"] = updated_refs
        return updated

    def _replace_ref_in_chunk(self, chunk: dict, old_ref: str, new_ref: str) -> dict:
        """Replace old_ref with new_ref in all citation spans within the chunk."""
        updated = dict(chunk)
        updated_spans = []
        for span in chunk.get("spans") or []:
            if span.get("type") == "citation" and span.get("ref") == old_ref:
                new_span = dict(span)
                new_span["ref"] = new_ref
                updated_spans.append(new_span)
            else:
                updated_spans.append(span)
        updated["spans"] = updated_spans
        return updated


if __name__ == "__main__":
    # Manual smoke test: sample real data from remote using the shared sampler.
    # Requires Sefaria + Anthropic + Dicta credentials and remote access configured.
    from utils import get_random_non_segment_links_with_chunks

    resolver = LLMParallelResolver(window_words_per_side=30)
    samples = get_random_non_segment_links_with_chunks(n=60, use_remote=True, seed=68, use_cache=True)
    for i, item in enumerate(samples, 1):
        print(f"\n=== Sample {i} ===")
        link = item["link"]
        chunk = item["chunk"]
        try:
            result = resolver.resolve(link, chunk)
            print(result)
        except Exception as exc:  # pragma: no cover - manual smoke guard
            print(f"Error resolving sample {i}: {exc}")
