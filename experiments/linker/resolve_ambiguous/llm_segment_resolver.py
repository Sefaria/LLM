import os
import re
import html
from typing import Optional, Tuple, List, Dict, Any

import django
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

django.setup()

from sefaria.model.text import Ref
from utils import get_random_non_segment_links_with_chunks


class LLMSegmentResolver:
    """
    Resolve non-segment-level link refs down to segment-level refs using LLM guidance.
    """

    def __init__(self, llm=None):
        # Default to Claude; caller can pass any LangChain chat model.
        self.llm = llm or ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0,
            max_tokens=512,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def resolve(self, link: dict, chunk: dict) -> Optional[Dict[str, Any]]:
        """
        Given a link + associated marked_up_text_chunk (non-segment ref), attempt to resolve
        the non-segment ref to a specific segment range. Returns updated link/chunk/resolution,
        or None if no resolution was found.
        """
        non_segment_ref = self._find_non_segment_ref(link)
        if not non_segment_ref:
            return None

        span = self._find_span_for_ref(chunk, non_segment_ref)
        if not span:
            return None

        citing_ref = chunk.get("ref")
        citing_oref = None
        is_commentary = False
        try:
            citing_oref = Ref(citing_ref)
            base_titles = getattr(citing_oref.index, "base_text_titles", []) or []
            is_commentary = bool(base_titles)
        except Exception:
            pass

        base_ref_for_prompt = None
        base_text_for_prompt = ""
        if is_commentary:
            base_ref_for_prompt = self._get_base_ref_from_index(citing_oref) or self._get_base_ref_from_link(
                citing_ref, link, expected_base_title=self._get_base_title(citing_oref)
            )
            if base_ref_for_prompt:
                base_text_for_prompt = self._get_ref_text(
                    base_ref_for_prompt
                )

        citing_text = self._get_ref_text(citing_ref, chunk.get("language"), chunk.get("versionTitle"))
        marked_citing_text = self._mark_citation(citing_text, span)
        if base_text_for_prompt:
            marked_citing_text = f"{marked_citing_text}\n\nBase text for commentary target ({base_ref_for_prompt}):\n{base_text_for_prompt}"

        segments = Ref(non_segment_ref).all_segment_refs()
        if not segments:
            return None
        numbered_segments = self._format_segment_texts(segments)

        if not self._llm_contains_reference(marked_citing_text, non_segment_ref, numbered_segments):
            return None

        start_idx, end_idx = self._llm_pick_range(
            marked_citing_text, non_segment_ref, numbered_segments, base_ref_for_prompt, base_text_for_prompt
        )
        if start_idx is None or end_idx is None:
            return None

        start_idx = max(1, min(start_idx, len(segments)))
        end_idx = max(start_idx, min(end_idx, len(segments)))
        selected = segments[start_idx - 1:end_idx]
        if not selected:
            return None
        resolved_ref = selected[0].normal() if len(selected) == 1 else selected[0].to(selected[-1]).normal()

        updated_link = self._replace_ref_in_link(link, non_segment_ref, resolved_ref)
        updated_chunk = self._replace_ref_in_chunk(chunk, non_segment_ref, resolved_ref)

        return {
            "link": updated_link,
            "chunk": updated_chunk,
            "resolved_ref": resolved_ref,
            "selected_segments": [r.normal() for r in selected],
        }

    def _find_non_segment_ref(self, link: dict) -> Optional[str]:
        for tref in link.get("refs", []):
            try:
                oref = Ref(tref)
            except Exception:
                continue
            if not oref.is_segment_level():
                return oref.normal()
        return None

    def _find_span_for_ref(self, chunk: dict, target_ref: str) -> Optional[dict]:
        spans = chunk.get("spans") or []
        for span in spans:
            if span.get("ref") == target_ref and span.get("type") == "citation":
                return span
        # Fallback: first citation span
        for span in spans:
            if span.get("type") == "citation":
                return span
        return None

    def _get_ref_text(self, tref: str, lang: Optional[str]=None, vtitle: Optional[str]=None) -> str:
        if not tref:
            return ""
        vtitle = html.unescape(vtitle) if vtitle else None
        try:
            return Ref(tref).text(lang or "en", vtitle=vtitle).as_string()
        except Exception:
            return ""

    def _mark_citation(self, text: str, span: Optional[dict]) -> str:
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

    def _llm_contains_reference(
        self, marked_citing_text: str, target_ref: str, numbered_segments: List[str]
    ) -> bool:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You verify whether a citation points to any text in a list of segment-level texts.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "Target higher-level ref: {target_ref}\n\n"
                    "Candidate segment texts:\n{segments}\n\n"
                    "Answer only YES or NO: Does the cited text appear within any of the candidate segments?",
                ),
            ]
        )
        chain = prompt | self.llm
        resp = chain.invoke(
            {
                "citing": marked_citing_text,
                "target_ref": target_ref,
                "segments": "\n".join(numbered_segments),
            }
        )
        content = getattr(resp, "content", "").strip().lower()
        return content.startswith("y")

    def _llm_pick_range(
        self,
        marked_citing_text: str,
        target_ref: str,
        numbered_segments: List[str],
        base_ref: Optional[str],
        base_text: str,
    ) -> Tuple[Optional[int], Optional[int]]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You map a citation to a range of numbered segments. Respond with two integers: start and end numbers.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "Target higher-level ref: {target_ref}\n\n"
                    "{base_block}"
                    "Numbered segment texts:\n{segments}\n\n"
                    "Which two numbers (start and end) best cover the cited text? "
                    "Both start and end must be inclusive (i.e., both indices include relevant material). "
                    "If only one segment is relevant, repeat the same number for start and end. "
                    "Respond as 'start,end' with numbers only.",
                ),
            ]
        )
        chain = prompt | self.llm
        base_block = ""
        if base_ref and base_text:
            base_block = f"Base text of commentary target ({base_ref}):\n{base_text}\n\n"
        resp = chain.invoke(
            {
                "citing": marked_citing_text,
                "target_ref": target_ref,
                "segments": "\n".join(numbered_segments),
                "base_block": base_block,
            }
        )
        content = getattr(resp, "content", "")
        nums = re.findall(r"\d+", content)
        if len(nums) < 1:
            return None, None
        start = int(nums[0])
        end = int(nums[1]) if len(nums) > 1 else start
        return start, end

    def _replace_ref_in_link(self, link: dict, old_ref: str, new_ref: str) -> dict:
        updated = dict(link)
        updated_refs = []
        for tref in link.get("refs", []):
            updated_refs.append(new_ref if tref == old_ref else tref)
        updated["refs"] = updated_refs
        return updated

    def _replace_ref_in_chunk(self, chunk: dict, old_ref: str, new_ref: str) -> dict:
        updated = dict(chunk)
        updated_spans = []
        for span in chunk.get("spans") or []:
            if span.get("ref") == old_ref and span.get("type") == "citation":
                new_span = dict(span)
                new_span["ref"] = new_ref
                updated_spans.append(new_span)
            else:
                updated_spans.append(span)
        updated["spans"] = updated_spans
        return updated

    def _get_base_ref_from_link(self, citing_ref: str, link: dict, expected_base_title: Optional[str]) -> Optional[str]:
        for tref in link.get("refs", []):
            if tref == citing_ref:
                continue
            if expected_base_title:
                try:
                    if Ref(tref).index.title != expected_base_title:
                        continue
                except Exception:
                    continue
            return tref
        return None

    def _get_base_ref_from_index(self, citing_oref: Ref) -> Optional[str]:
        try:
            base_title = self._get_base_title(citing_oref)
            if not base_title:
                return None
            # Use the section structure of the commentary ref to align to base text.
            section_refs = citing_oref.section_ref()
            sec_parts = section_refs.sections
            sec_str = " ".join(str(x) for x in sec_parts) if sec_parts else ""
            if sec_str:
                return f"{base_title} {sec_str}"
            return base_title
        except Exception:
            return None

    def _get_base_title(self, citing_oref: Optional[Ref]) -> Optional[str]:
        try:
            base_titles = getattr(citing_oref.index, "base_text_titles", []) or []
            return base_titles[0] if base_titles else None
        except Exception:
            return None


if __name__ == "__main__":
    resolver = LLMSegmentResolver()
    samples = get_random_non_segment_links_with_chunks(n=5, use_remote=True, seed=615, use_cache=True)
    for item in samples:
        link = item["link"]
        chunk = item["chunk"]
        result = resolver.resolve(link, chunk)
        print("Original link refs:", link.get("refs"))
        if result:
            print("Resolved:", result["resolved_ref"])
            print("Updated link refs:", result["link"].get("refs"))
        else:
            print("No resolution found")
        print("-" * 40)
