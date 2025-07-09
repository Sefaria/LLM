# simple_ref_topic_filter.py
from __future__ import annotations
import json, textwrap
from dataclasses import dataclass
from typing import List, Callable
from langchain_openai import ChatOpenAI      # lightweight OpenAI chat wrapper
import django
django.setup()
from sefaria.model import Ref
from experiments.topic_modelling.utils import DataHandler
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from util.sefaria_specific import get_ref_text_with_fallback, get_passage_refs
from sefaria.model import Topic, TopicSet

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# only needed if you want full text

# --------------------------- OBJECT ---------------------------------- #
@dataclass
class SequentialRefTopicFilter:

    llm: ChatOpenAI
    max_topics: int = 25
    debug: bool = False     # print prompt and raw reply for every call

    # -- private helpers ------------------------------------------------
    def _build_prompt(self, context: str, slugs: List[str]) -> str:
        """Return the plain prompt string sent to the LLM."""
        # Look up the Topic objects
        topics = TopicSet({"slug": {"$in": slugs}}).array()

        # Build “slug — title” strings for the prompt
        lines = [
            f"{topic.slug} — {topic.get_primary_title('en') or ''}"
            for topic in topics
        ]
        if not lines:  # just in case
            lines.append("")

        # Craft the prompt
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

    def _extract_json_array(self, raw: str) -> list:
        """
        Return the first JSON array found in `raw`.
        Raises ValueError if none present or if it isn't valid JSON.
        """
        try:
            start = raw.index("[")
            # naive but works because the model never uses nested ] in this task
            end = raw.index("]", start) + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Could not parse JSON array from model output") from exc
    def _call_llm(self, prompt: str) -> List[str]:
        """Send prompt → model, parse JSON list, return Python list."""
        response = self.llm.invoke(prompt)

        if self.debug:
            print("\n" + "=" * 70)
            print("PROMPT\n" + "-" * 70 + "\n" + prompt)
            print("RAW REPLY\n" + "-" * 70 + f"\n{response.content}")

        try:
            data = self._extract_json_array(response.content)  # extract JSON array from the response
            if not isinstance(data, list):
                raise ValueError("Model did not return a JSON array")
            return data
        except Exception as err:
            print("⚠️  JSON parse error:", err)
            return []

    # -- public API -----------------------------------------------------
    def filter_ref(
        self,
        lr: "LabelledRef",
    ) -> List[str]:
        content_text = get_ref_text_with_fallback(lr.ref, 'en')
        # if len(content_text.split()) <= 5:
        #     # If the text is too short, return empty list
        #     return []
        context = (
            f"Ref: {lr.ref}: {content_text}"
        )
        prompt = self._build_prompt(context, lr.slugs)
        return self._call_llm(prompt)

    def filter_refs(
        self,
        lrs: List["LabelledRef"],
    ) -> dict[str, List[str]]:
        """
        Sequentially filter many LabelledRefs.
        Returns {ref_str: kept_slugs}.
        """
        return {lr.ref: self.filter_ref(lr) for lr in lrs}

if __name__ == "__main__":
    # 1. Read predictions
    dh = DataHandler("evaluation_data/gold.jsonl", "evaluation_data/predictions.jsonl", "evaluation_data/all_slugs.csv")
    predicted = dh.get_predicted()  # List[LabelledRef]

    # 2. Init the chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3. Create the sequential filter (turn on debug if you want full logs)
    filterer = SequentialRefTopicFilter(llm, max_topics=25, debug=True)


    # 4. Optional full-text lookup
    def sefaria_en(ref_str: str) -> str:
        return Ref(ref_str).text().text


    # 5. Run (still sequential, just wrapped)
    kept = filterer.filter_refs(predicted[50:52])

    for ref, slugs in kept.items():
        print(f"{ref} → {slugs}")
