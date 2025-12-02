import os
import re
import html
from typing import Optional, Tuple, List, Dict, Any

# LangChain cache setup
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Use persistent cache file, configurable via env
cache_path = os.getenv("LLM_CACHE_PATH", "llm_cache.sqlite")
set_llm_cache(SQLiteCache(database_path=cache_path))

import django
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

django.setup()

from sefaria.model.text import Ref, AddressType
from utils import get_random_non_segment_links_with_chunks



class LLMSegmentResolver:
    """
    Resolve non-segment-level link refs down to segment-level refs using LLM guidance.
    """

    def __init__(self, llm=None):
        # Default to Claude; caller can pass any LangChain chat model.
        self.llm = llm or ChatAnthropic(
            # model="claude-sonnet-4-5-20250929",
            # model="claude-3-opus-20240229",
            model="claude-3-5-haiku-20241022",
            temperature=0,
            max_tokens=512,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def resolve(self, link: dict, chunk: dict) -> Optional[Dict[str, Any]]:
        """
        Given a link + associated marked_up_text_chunk, attempt to resolve
        non-segment refs to specific segment ranges. Handles multiple non-segment refs.
        Returns updated link/chunk/resolutions, or None if no resolutions were found.
        """
        non_segment_refs = self._find_all_non_segment_refs(link)
        if not non_segment_refs:
            return None
        if non_segment_refs[0] == "Tosafot on Rosh Hashanah 26a":
            halt = True
        resolutions = []
        updated_link = link
        updated_chunk = chunk

        for non_segment_ref in non_segment_refs:
            resolution = self._resolve_single_ref(non_segment_ref, updated_link, updated_chunk)
            if resolution:
                resolutions.append(resolution)
                # Update link and chunk for next iteration
                updated_link = resolution["link"]
                updated_chunk = resolution["chunk"]

        if not resolutions:
            return None

        # Return combined result
        all_resolved_refs = [r["resolved_ref"] for r in resolutions if r.get("resolved_ref")]
        all_selected_segments = []
        all_reasons = []
        for r in resolutions:
            all_selected_segments.extend(r.get("selected_segments", []))
            if r.get("reason"):
                all_reasons.append(f"{r.get('original_ref', '')}: {r['reason']}")

        return {
            "link": updated_link,
            "chunk": updated_chunk,
            "resolved_refs": all_resolved_refs,
            "resolved_ref": all_resolved_refs[0] if len(all_resolved_refs) == 1 else None,  # For backward compatibility
            "selected_segments": all_selected_segments,
            "reason": "; ".join(all_reasons) if all_reasons else None,
            "resolutions": resolutions,  # Detailed info for each resolution
        }

    def _resolve_single_ref(self, non_segment_ref: str, link: dict, chunk: dict) -> Optional[Dict[str, Any]]:
        """Resolve a single non-segment reference."""
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
            base_ref_for_prompt = self._get_base_ref_from_index(citing_oref)
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

        # If there's only one segment, automatically resolve to it without calling LLM
        if len(segments) == 1:
            resolved_ref = segments[0].normal()
            updated_link = self._replace_ref_in_link(link, non_segment_ref, resolved_ref)
            updated_chunk = self._replace_ref_in_chunk(chunk, non_segment_ref, resolved_ref)
            return {
                "link": updated_link,
                "chunk": updated_chunk,
                "original_ref": non_segment_ref,
                "resolved_ref": resolved_ref,
                "selected_segments": [resolved_ref],
                "reason": "Automatically resolved: only one segment exists",
                "citing_text": citing_text,
                "citation_span": span,
            }

        numbered_segments = self._format_segment_texts(segments)

        if not self._llm_contains_reference(marked_citing_text, non_segment_ref, numbered_segments):
            second_note = (
                "Note: Some references are 'see' / 'עיין' style, where the cited text discusses a similar theme "
                "without being explicitly quoted."
            )
            if not self._llm_contains_reference(marked_citing_text, non_segment_ref, numbered_segments, extra_note=second_note):
                return None

        range_result = self._llm_pick_range(
            marked_citing_text, non_segment_ref, numbered_segments, base_ref_for_prompt, base_text_for_prompt
        )
        if not range_result:
            return None
        start_idx, end_idx, reason = range_result

        start_idx = max(1, min(start_idx, len(segments)))
        end_idx = max(start_idx, min(end_idx, len(segments)))
        selected = segments[start_idx - 1:end_idx]
        if not selected:
            return None

        # If more than 7 segments were selected, try to narrow down the range
        if len(selected) > 7:
            refined_result = self._llm_refine_range(
                marked_citing_text, non_segment_ref, selected, base_ref_for_prompt, base_text_for_prompt
            )
            if refined_result:
                refined_start, refined_end, refined_reason = refined_result
                # Adjust indices relative to the selected segments
                refined_start = max(1, min(refined_start, len(selected)))
                refined_end = max(refined_start, min(refined_end, len(selected)))
                selected = selected[refined_start - 1:refined_end]
                reason = f"{reason} (refined: {refined_reason})"

        # For dibur hamatchil-style indexes, pick a single segment within the chosen range
        if self._is_dibur_hamatchil_ref(non_segment_ref) and len(selected) > 1:
            dh_pick = self._llm_pick_single_segment_dh(
                marked_citing_text,
                non_segment_ref,
                selected,
                base_ref_for_prompt,
                base_text_for_prompt,
            )
            if dh_pick:
                dh_idx, dh_reason = dh_pick
                dh_idx = max(1, min(dh_idx, len(selected)))
                selected = [selected[dh_idx - 1]]
                reason = f"{reason} (DH narrowed: {dh_reason})" if reason else f"DH narrowed: {dh_reason}"

        # If LLM selected all segments in a multi-segment ref, keep it non-segment-level
        selected_all_segments = len(selected) == len(segments)
        has_multiple_segments = len(segments) >= 4
        if has_multiple_segments and selected_all_segments:
            return {
                "link": link,
                "chunk": chunk,
                "original_ref": non_segment_ref,
                "resolved_ref": None,
                "selected_segments": [],
                "reason": "LLM selected all segments; leaving as non-segment-level.",
                "citing_text": citing_text,
                "citation_span": span,
            }
        resolved_ref = selected[0].normal() if len(selected) == 1 else selected[0].to(selected[-1]).normal()

        updated_link = self._replace_ref_in_link(link, non_segment_ref, resolved_ref)
        updated_chunk = self._replace_ref_in_chunk(chunk, non_segment_ref, resolved_ref)

        return {
            "link": updated_link,
            "chunk": updated_chunk,
            "original_ref": non_segment_ref,
            "resolved_ref": resolved_ref,
            "selected_segments": [r.normal() for r in selected],
            "reason": reason,
            "citing_text": citing_text,
            "citation_span": span,
        }

    def _find_all_non_segment_refs(self, link: dict) -> List[str]:
        """Find all non-segment-level references in a link."""
        non_segment_refs = []
        for tref in link.get("refs", []):
            try:
                oref = Ref(tref)
                if not oref.is_segment_level():
                    non_segment_refs.append(oref.normal())
            except Exception:
                continue
        return non_segment_refs

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
            primary_lang = lang or "en"
            text = Ref(tref).text(primary_lang, vtitle=vtitle).as_string()
            if text:
                return text
            fallback_lang = "he" if primary_lang == "en" else "en"
            return Ref(tref).text(fallback_lang).as_string()
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
        self, marked_citing_text: str, target_ref: str, numbered_segments: List[str], extra_note: Optional[str] = None
    ) -> bool:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You verify whether a citing passage points to (or thematically references) any text "
                    "in a list of segment-level texts. The citation might be explicit or implicit.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "Target higher-level ref: {target_ref}\n\n"
                    "{note_block}"
                    "Candidate segment texts:\n{segments}\n\n"
                    "The citing passage may reference the target text either directly or indirectly (e.g., via a theme). "
                    "Answer only YES or NO: Does the cited/related text appear within any of the candidate segments?",
                ),
            ]
        )
        chain = prompt | self.llm
        note_block = f"{extra_note}\n\n" if extra_note else ""
        resp = chain.invoke(
            {
                "citing": marked_citing_text,
                "target_ref": target_ref,
                "note_block": note_block,
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
    ) -> Tuple[Optional[int], Optional[int], Optional[str]]:
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
                    "Both start and end must be inclusive (i.e., both indices include relevant material). "
                    "If only one segment is relevant, repeat the same number for start and end. "
                    "Respond in two lines:\n"
                    "Explanation: <brief reason>\n"
                    "Range: <start>,<end>",
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
        reason = None
        start = end = None
        # Try to parse structured lines first.
        range_match = re.search(r"range\s*:\s*([0-9]+)\s*,\s*([0-9]+)", content, re.IGNORECASE)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            # Extract reason if present before Range:
            parts = re.split(r"range\s*:", content, flags=re.IGNORECASE)
            if parts:
                reason = parts[0].replace("Explanation:", "").strip()
        else:
            nums = re.findall(r"\d+", content)
            if len(nums) >= 1:
                start = int(nums[0])
                end = int(nums[1]) if len(nums) > 1 else start
        if start is None or end is None:
            return None
        reason = reason or content.strip()
        return start, end, reason

    def _llm_refine_range(
        self,
        marked_citing_text: str,
        target_ref: str,
        selected_segments: List[Ref],
        base_ref: Optional[str],
        base_text: str,
    ) -> Optional[Tuple[int, int, str]]:
        """
        When the initial range selection is too broad (>7 segments), attempt to narrow it down
        by asking the LLM to be more specific about which parts are actually relevant.
        """
        numbered_segments = self._format_segment_texts(selected_segments)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You refine an overly broad citation range to a more precise subset. "
                    "The user has already identified a broad range, but it's too large. "
                    "Your job is to identify the most relevant portion within that range.",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "Target ref: {target_ref}\n\n"
                    "{base_block}"
                    "Previously selected range (too broad - {count} segments):\n{segments}\n\n"
                    "The initial selection included {count} segments, which is too many. "
                    "Please identify a more precise range within these segments that best matches the actual citation. "
                    "Focus on the segments that are directly quoted or most closely discussed. "
                    "If the citation truly spans all segments, you may keep the full range, but try to be more specific.\n\n"
                    "Respond in two lines:\n"
                    "Explanation: <brief reason for the refined selection>\n"
                    "Range: <start>,<end>",
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
                "count": len(selected_segments),
            }
        )
        content = getattr(resp, "content", "")
        reason = None
        start = end = None

        # Try to parse structured lines first
        range_match = re.search(r"range\s*:\s*([0-9]+)\s*,\s*([0-9]+)", content, re.IGNORECASE)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            # Extract reason if present before Range:
            parts = re.split(r"range\s*:", content, flags=re.IGNORECASE)
            if parts:
                reason = parts[0].replace("Explanation:", "").strip()
        else:
            nums = re.findall(r"\d+", content)
            if len(nums) >= 1:
                start = int(nums[0])
                end = int(nums[1]) if len(nums) > 1 else start

        if start is None or end is None:
            return None

        # If the refined range is still very broad or identical to original, return None
        # to keep the original selection
        if end - start + 1 >= len(selected_segments):
            return None

        reason = reason or content.strip()
        return start, end, reason

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


    def _get_base_ref_from_index(self, citing_oref: Ref) -> Optional[str]:
        try:
            base_title = self._get_base_title(citing_oref)
            if not base_title:
                return None
            # Use the section structure of the commentary ref to align to base text.
            section_ref = citing_oref.section_ref()
            sec_parts = section_ref.sections
            addr_types = section_ref.index_node.addressTypes
            for sec, addr_type in zip(sec_parts, addr_types):
                address = AddressType.to_str_by_address_type(addr_type, "en", sec)
                base_title += f" {address}"
            base_ref = Ref(base_title)

            # If the base_ref is not segment-level, query for commentary links to infer exact segment
            if not base_ref.is_segment_level():
                inferred_ref = self._infer_segment_from_commentary_links(citing_oref, base_ref)
                if inferred_ref:
                    return inferred_ref

            return base_ref.normal()
        except Exception:
            return None

    def _infer_segment_from_commentary_links(self, citing_oref: Ref, base_ref: Ref) -> Optional[str]:
        """
        When base_ref is not segment-level, query for commentary links from the citing segment
        to infer the exact segment of the base text being commented on.
        """
        try:
            from sefaria.model.link import LinkSet

            # Get the segment-level ref of the commentary
            if not citing_oref.is_segment_level():
                citing_oref = citing_oref.section_ref()

            # Query for links from this commentary segment
            links = LinkSet(citing_oref)

            # Look for links to the base text that are segment-level
            for link in links:
                opposite_ref = link.ref_opposite(citing_oref)
                if not opposite_ref:
                    continue

                # Check if this link points to the same base text range
                try:
                    # If the opposite ref is within the base_ref range and is segment-level
                    if opposite_ref.is_segment_level() and base_ref.contains(opposite_ref):
                        return opposite_ref.normal()
                except Exception:
                    continue

            return None
        except Exception:
            return None

    def _get_base_title(self, citing_oref: Optional[Ref]) -> Optional[str]:
        try:
            base_titles = getattr(citing_oref.index, "base_text_titles", []) or []
            return base_titles[0] if base_titles else None
        except Exception:
            return None

    def _is_dibur_hamatchil_ref(self, tref: str) -> bool:
        """Check if a ref belongs to a dibur hamatchil-structured index."""
        try:
            oref = Ref(tref)
            return oref.index_node.is_segment_level_dibur_hamatchil()
        except Exception:
            return False

    def _llm_pick_single_segment_dh(
        self,
        marked_citing_text: str,
        target_ref: str,
        candidate_segments: List[Ref],
        base_ref: Optional[str],
        base_text: str,
    ) -> Optional[Tuple[int, str]]:
        """
        For dibur-hamatchil style works, choose a single segment within a previously-selected range.
        Returns (index, reason) where index is 1-based within candidate_segments.
        """
        numbered_segments = self._format_segment_texts(candidate_segments)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are disambiguating a dibur hamatchil-style commentary. Each citation should resolve to a single segment (one DH).",
                ),
                (
                    "human",
                    "Citing passage (citation wrapped in <citation ...></citation>):\n{citing}\n\n"
                    "Target ref: {target_ref}\n\n"
                    "{base_block}"
                    "Candidate segments (choose exactly one):\n{segments}\n\n"
                    "Pick the single number whose text most directly matches or is quoted by the citing passage."
                    " Respond in two lines:\n"
                    "Explanation: <brief reason>\n"
                    "Choice: <number>",
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
        content = getattr(resp, "content", "")
        choice = None
        choice_match = re.search(r"choice\s*:\s*([0-9]+)", content, re.IGNORECASE)
        if choice_match:
            choice = int(choice_match.group(1))
            reason = re.split(r"choice\s*:", content, flags=re.IGNORECASE)[0].replace("Explanation:", "").strip()
        else:
            nums = re.findall(r"\d+", content)
            if nums:
                choice = int(nums[0])
                reason = content.strip()
            else:
                reason = None

        if choice is None:
            return None
        return choice, reason or "Picked single segment"


if __name__ == "__main__":
    resolver = LLMSegmentResolver()
    samples = get_random_non_segment_links_with_chunks(n=5, use_remote=True, seed=102, use_cache=True)
    for i, item in enumerate(samples):
        link = item["link"]
        chunk = item["chunk"]
        result = resolver.resolve(link, chunk)
        print("Original link refs:", link.get("refs"))
        if result:
            # Handle both single and multiple resolutions
            if result.get("resolved_ref"):
                print("Resolved:", result["resolved_ref"])
            elif result.get("resolved_refs"):
                print(f"Resolved {len(result['resolved_refs'])} refs:", result["resolved_refs"])

            print("Updated link refs:", result["link"].get("refs"))

            # Show detailed info for each resolution if multiple
            if result.get("resolutions") and len(result["resolutions"]) > 1:
                print("\nDetailed resolutions:")
                for j, res in enumerate(result["resolutions"], 1):
                    print(f"  {j}. {res.get('original_ref')} → {res.get('resolved_ref')}")
                    if res.get("reason"):
                        print(f"     Reason: {res['reason']}")
            elif result.get("reason"):
                print("LLM reason:", result["reason"])
        else:
            print("No resolution found")
        print("-" * 40)
