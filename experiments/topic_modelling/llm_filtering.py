# simple_ref_topic_filter.py
from __future__ import annotations
import json, textwrap
from dataclasses import dataclass, field
from typing import List
from langchain_openai import ChatOpenAI        # lightweight OpenAI chat wrapper

# -------------- Django / Sefaria boot-strap ----------------------------------
import django
django.setup()
from sefaria.model import Ref, TopicSet

from experiments.topic_modelling.utils import DataHandler
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from util.sefaria_specific import get_ref_text_with_fallback

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# ----------------------------- FILTERER --------------------------------------
@dataclass
class SequentialRefTopicFilter:
    """
    • First pass – regular filtering: ask the LLM to pick the most relevant
      slugs out of the candidate list.

    • Second pass – for any slug in `overly_general_slugs` that survived the
      first pass, prompt the LLM with a *yes/no* question: “Is this slug truly
      **specifically** relevant to the passage, or is it merely a very general
      tag that could apply to almost anything?”  Only keep the slug if the
      model answers `true`.
    """

    llm: ChatOpenAI
    max_topics: int = 25
    overly_general_slugs: List[str] = field(
        default_factory=lambda: [
            # ✱ customise as you wish
            "avodat-hashem", "beliefs", "creation", "faith", "food", "god", "halakhah", "halakhic-principles",
            "history", "israel", "jewish-people", "laws", "laws-of-the-calendar", "learning", "life", "mitzvot",
            "sacrifices", "social-issues", "stories", "talmudic-figures", "teshuvah", "values", "women"
        ]
    )
    debug: bool = False  # print prompt / reply for every call

    # ---- first-pass prompt ---------------------------------------------------
    def _build_prompt(self, context: str, slugs: List[str]) -> str:
        topics = TopicSet({"slug": {"$in": slugs}}).array()
        lines = [
            f"{topic.slug} — {topic.get_primary_title('en') or ''}"
            for topic in topics
        ] or [""]

        return textwrap.dedent(f"""
            SYSTEM: You are an expert Judaic librarian who tags texts with topical slugs.

            TEXT (reference or excerpt):
            \"\"\"{context}\"\"\"

            CANDIDATE TOPICS (slug — title):
            {chr(10).join(sorted(lines))}

            TASK ▾
            • Return **up to {self.max_topics}** *slugs* (the identifiers before the “—”),
              in descending order of relevance. Choose only the most relevant slugs.
              Think about a user who is looking for a text on a specific topic—would
              they want to see this slug?
            • Reply *only* with a JSON array of strings, no prose. Example:
              ["laws-of-prayer", "shema"]
        """).strip()

    # ---- second-pass prompt --------------------------------------------------
    def _build_confirm_prompt(self, context: str, slug: str) -> str:
        return textwrap.dedent(f"""
            SYSTEM: You are an expert Judaic librarian.

            TEXT (reference or excerpt):
            \"\"\"{context}\"\"\"

            CANDIDATE SLUG:
            {slug}

            QUESTION ▾
            The slug above is very broad and could apply to many passages.
            Considering the *specific* content of this passage,
            **should we still tag it with this slug?**
            • Answer with only one word `true` or `false`, no prose.
        """).strip()

    # ---- utilities -----------------------------------------------------------
    def _extract_json(self, raw: str):
        """
        Return the first JSON value (array or bool) found in `raw`.
        """
        try:
            start = raw.index("[") if "[" in raw else raw.index("t")  # 't' or 'f'
        except ValueError:
            raise ValueError("No JSON value found")

        if raw[start] == "[":
            end = raw.index("]", start) + 1
        else:  # boolean
            end = start + 4 if raw[start:start + 4].lower() == "true" else start + 5

        return json.loads(raw[start:end])

    def _call_llm(self, prompt: str):
        resp = self.llm.invoke(prompt)
        if self.debug:
            print("\n" + "=" * 70)
            print("PROMPT\n" + "-" * 70 + "\n" + prompt)
            print("RAW REPLY\n" + "-" * 70 + f"\n{resp.content}")
        return resp.content

    # ---- core public methods -------------------------------------------------
    def filter_ref(self, lr: "LabelledRef") -> List[str]:
        """Filter one labelled ref, applying the two-step logic."""
        passage_text = get_ref_text_with_fallback(lr.ref, "en")
        context = f"{lr.ref}: {passage_text}"

        # -- 1️⃣  first pass ----------------------------------------------------
        first_prompt = self._build_prompt(context, lr.slugs)
        try:
            raw_first = self._call_llm(first_prompt)
            chosen_slugs: List[str] = self._extract_json(raw_first)
        except Exception as e:
            print("⚠️  first-pass parse error:", e)
            return []

        # -- 2️⃣  second pass for overly general slugs -------------------------
        final_slugs: List[str] = []
        for slug in chosen_slugs:
            if slug not in self.overly_general_slugs:
                final_slugs.append(slug)
                continue

            confirm_prompt = self._build_confirm_prompt(context, slug)
            try:
                raw_second = self._call_llm(confirm_prompt)
                keep = True if raw_second == 'true' else False
                if keep:
                    print(f"✅  Keeping slug: {slug}")
            except Exception as e:
                print(f"⚠️  confirm-pass parse error for {slug}:", e)
                keep = False

            if keep:
                final_slugs.append(slug)

        return final_slugs

    def filter_refs(self, lrs: List["LabelledRef"]) -> dict[str, List[str]]:
        """Sequentially filter a list of LabelledRefs. Returns {ref_str: kept_slugs}."""
        return {lr.ref: self.filter_ref(lr) for lr in lrs}

# ------------------------------ CLI DEMO -------------------------------------
if __name__ == "__main__":
    # 1. Read predictions
    dh = DataHandler(
        "evaluation_data/gold.jsonl",
        "evaluation_data/predictions.jsonl",
        "evaluation_data/all_slugs.csv",
    )
    predicted = dh.get_predicted()  # List[LabelledRef]

    # 2. Init chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3. Create filterer (debug=True ↔︎ verbose logging)
    filterer = SequentialRefTopicFilter(llm, max_topics=25, debug=True)

    # 4. Run for a few refs
    kept = filterer.filter_refs(predicted[50:52])
    for ref, slugs in kept.items():
        print(f"{ref} → {slugs}")
