import html
import os
from typing import Any, Dict, List, Optional, Tuple
import django
django.setup()
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from sefaria.model.text import Ref


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
        min_threshold: float = 7.0,
        max_distance: float = 4.0,
        general_min_score: float = 7.0,
        tanakh_min_score: float = 1.45,
        canonical_min_score: float = 2.35,
        min_frequency_to_count_phrase_as_one_word: int = 30,
        request_timeout: int = 30,
    ):
        # Default to Claude; caller can pass any LangChain chat model.
        self.llm = llm or ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            temperature=0,
            max_tokens=256,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
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

    def resolve(self, link: dict, chunk: dict) -> Optional[Dict[str, Any]]:
        """
        Attempt to resolve a non-segment ref via Dicta parallels when the citing text appears to
        explicitly quote or paraphrase the cited work.
        """
        non_segment_ref = self._find_non_segment_ref(link)
        if not non_segment_ref:
            return None

        citing_ref = chunk.get("ref")
        citing_text = self._get_ref_text(citing_ref, chunk.get("language"), chunk.get("versionTitle"))
        if not citing_text:
            return None

        span = self._find_span_for_ref(chunk, non_segment_ref)
        marked_citing_text = self._mark_citation(citing_text, span)

        dicta_result = self._query_dicta(
            citing_text,
            target_ref=non_segment_ref,
            citing_ref=citing_ref,
            citing_lang=chunk.get("language"),
            citing_version=chunk.get("versionTitle"),
        )
        if not dicta_result:
            return None

        resolved_ref = dicta_result.get("resolved_ref")
        if not resolved_ref:
            return None

        candidate_text = self._get_ref_text(resolved_ref, chunk.get("language"))
        llm_ok, llm_reason = self._llm_confirm_candidate(
            marked_citing_text, non_segment_ref, resolved_ref, candidate_text
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
            "reason": f"Dicta hit {resolved_ref} confirmed by LLM",
            "llm_reason": llm_reason,
            "llm_verdict": "YES",
            "dicta_hit": dicta_result,
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

    def _llm_confirm_candidate(
        self, citing_text: str, target_ref: str, candidate_ref: str, candidate_text: str
    ) -> Tuple[bool, str]:
        """
        Ask the LLM to confirm whether the Dicta-picked candidate is a likely direct/close paraphrase.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You verify whether a citing passage points to a specific target segment.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
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

        response = requests.post(self.dicta_url, **kwargs)
        response.raise_for_status()
        text = response.text.lstrip("\ufeff")
        try:
            data = response.json()
        except Exception:
            import json as _json

            data = _json.loads(text)
        best_hit = self._pick_best_dicta_hit(
            data.get("results") or [],
            target_ref=target_ref,
            citing_ref=citing_ref,
            citing_lang=citing_lang,
            citing_version=citing_version,
        )
        if best_hit:
            print(f"Dicta success with payload=text=<len {len(query_text)}> mode=form-string")
            return best_hit

        return None

    def _pick_best_dicta_hit(
        self,
        results: List[Dict[str, Any]],
        target_ref: Optional[str] = None,
        citing_ref: Optional[str] = None,
        citing_lang: Optional[str] = None,
        citing_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Given Dicta results, return the first candidate whose ref is contained within the target_ref.
        Prints all normalized hits for debugging.
        """
        target_oref = None
        if target_ref:
            try:
                target_oref = Ref(target_ref)
            except Exception:
                target_oref = None

        debug_hits: List[str] = []
        match: Optional[Dict[str, Any]] = None
        citing_link = None
        try:
            if citing_ref:
                citing_link = f"https://library-linker.cauldron.sefaria.org/{Ref(citing_ref).url()}"
        except Exception:
            citing_link = None
        target_link = None
        try:
            if target_oref:
                target_link = f"https://library-linker.cauldron.sefaria.org/{target_oref.url()}"
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
                    cand_link = f"https://library-linker.cauldron.sefaria.org/{cand_oref.url()}"
                    if target_oref and target_oref.contains(cand_oref):
                        debug_hits.append(f"hit: citing={citing_link} resolved={cand_link}")
                        match = {"resolved_ref": normalized, "raw": candidate}
                        break
                except Exception:
                    continue

            if match:
                break

        if debug_hits:
            print("Dicta hits:", "; ".join(debug_hits))

        if not match:
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
                print("Dicta misses:", "; ".join(miss_info))

        return match

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

    resolver = LLMParallelResolver()
    samples = get_random_non_segment_links_with_chunks(n=30, use_remote=True, seed=60, use_cache=True)
    for i, item in enumerate(samples, 1):
        print(f"\n=== Sample {i} ===")
        link = item["link"]
        chunk = item["chunk"]
        try:
            result = resolver.resolve(link, chunk)
            print(result)
        except Exception as exc:  # pragma: no cover - manual smoke guard
            print(f"Error resolving sample {i}: {exc}")
