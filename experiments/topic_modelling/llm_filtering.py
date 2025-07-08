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
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# only needed if you want full text

# --------------------------- OBJECT ---------------------------------- #
@dataclass
class SequentialRefTopicFilter:
    """
    Sequentially keep ≤ `max_topics` relevant slugs out of each LabelledRef.
    The object’s only dependencies are ChatOpenAI and json.
    """
    llm: ChatOpenAI
    max_topics: int = 25
    debug: bool = False     # print prompt and raw reply for every call

    # -- private helpers ------------------------------------------------
    def _build_prompt(self, context: str, slugs: List[str]) -> str:
        """Return the plain prompt string sent to the LLM."""
        return textwrap.dedent(f"""
        SYSTEM: You are an expert Judaic librarian who tags texts with topical slugs.

        TEXT (reference or excerpt):
        \"\"\"{context}\"\"\"

        CANDIDATE SLUGS:
        {', '.join(sorted(slugs))}

        TASK ▾
        • Return **up to {self.max_topics}** slugs that are actually relevant,
          in descending order of relevance.
        • Reply *only* with a JSON array of strings, no prose.
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
        *,
        text_lookup: Callable[[str], str] | None = None,
    ) -> List[str]:
        """
        Filter slugs for a single LabelledRef.

        Parameters
        ----------
        lr : LabelledRef
        text_lookup : optional callable that maps a ref string → full text.
                      Pass None to fall back to the ref string itself.
        """
        context = (
            text_lookup(lr.ref) if text_lookup else lr.ref
        )
        prompt = self._build_prompt(context, lr.slugs)
        return self._call_llm(prompt)

    def filter_refs(
        self,
        lrs: List["LabelledRef"],
        *,
        text_lookup: Callable[[str], str] | None = None,
    ) -> dict[str, List[str]]:
        """
        Sequentially filter many LabelledRefs.
        Returns {ref_str: kept_slugs}.
        """
        return {lr.ref: self.filter_ref(lr, text_lookup=text_lookup) for lr in lrs}

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
    kept = filterer.filter_refs(predicted[50:52], text_lookup=sefaria_en)

    for ref, slugs in kept.items():
        print(f"{ref} → {slugs}")
