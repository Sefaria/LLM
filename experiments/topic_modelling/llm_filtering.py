# ref_topic_filter_debug.py
from __future__ import annotations
import django
django.setup()

from sefaria.model import Ref                        # noqa: E402

from dataclasses import dataclass
from typing import List, Dict, Callable
import json, textwrap, os

# ── LangChain ────────────────────────────────────────────────
from langchain_openai import ChatOpenAI                   # pip install langchain-openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence

# ── Your helpers ────────────────────────────────────────────
from utils import DataHandler, LabelledRef                 # keep as-is


@dataclass
class RefTopicFilter:
    """
    Keep ≤25 relevant slugs from the predicted list – now with step-by-step debugging.
    """
    llm: ChatOpenAI
    max_topics: int = 25
    verbose: bool = False        # print LangChain internals
    debug: bool = False          # print prompt & raw LLM reply

    # ------------------------------------------------------------------ #
    # PROMPT TEMPLATE (shared across calls)
    # ------------------------------------------------------------------ #
    def _prompt(self) -> ChatPromptTemplate:
        template = textwrap.dedent("""
        SYSTEM: You are an expert Judaic librarian who tags texts with topical slugs.

        TEXT (reference or excerpt):
        \"\"\"{context}\"\"\"

        CANDIDATE SLUGS:
        {candidates}

        TASK ▾
        • Return **up to {k}** slugs that are actually relevant,
          in descending order of relevance.
        • Reply *only* with a JSON array of strings, no prose.
        """)
        return ChatPromptTemplate.from_template(template)

    # ------------------------------------------------------------------ #
    # ONE-OFF CHAIN (prompt → llm → parser)
    # ------------------------------------------------------------------ #
    def _chain(self) -> RunnableSequence:
        # propagate the LangChain verbosity flag if requested
        if self.verbose:
            import langchain
            langchain.verbose = True                # global switch :contentReference[oaicite:1]{index=1}
        return self._prompt() | self.llm | JsonOutputParser()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _run_single(self, inp: Dict) -> List[str]:
        """
        Internal helper that prints prompt / response when self.debug is on.
        """
        # Pretty-print the prompt before the call
        if self.debug:
            prompt_text = self._prompt().format(**inp)
            print("\n" + "="*60)
            print("PROMPT\n" + "-"*60 + "\n" + prompt_text)

        # Call the LLM chain
        raw_output = self._chain().invoke(inp)  # already parsed by JsonOutputParser

        if self.debug:
            # We need the *raw* model text to show; easiest is to call model directly
            raw_text = self.llm.invoke(self._prompt().format(**inp))
            print("\nRAW MODEL TEXT\n" + "-"*60 + f"\n{raw_text.content}")
            print("PARSED JSON\n" + "-"*60 + f"\n{raw_output}")

        return raw_output

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def filter_ref(
        self,
        ref: LabelledRef,
        *,
        context_fetcher: Callable[[str], str] | None = None,
    ) -> List[str]:
        """
        Filter slugs for a single reference.
        """
        context = (context_fetcher(ref.ref) if context_fetcher else ref.ref)
        candidates = ", ".join(sorted(ref.slugs)) if ref.slugs else "NONE"
        inp = {"context": context, "candidates": candidates, "k": self.max_topics}
        return self._run_single(inp)

    def filter_refs_sequential(
        self,
        refs: List[LabelledRef],
        *,
        context_fetcher: Callable[[str], str] | None = None,
    ) -> Dict[str, List[str]]:
        """
        Iterate one-by-one (easy to step through a debugger).
        """
        out: Dict[str, List[str]] = {}
        for r in refs:
            out[r.ref] = self.filter_ref(r, context_fetcher=context_fetcher)
        return out


if __name__ == "__main__":
    # 1. Load predicted refs & slugs
    dh = DataHandler(
        "evaluation_data/gold.jsonl",
        "evaluation_data/inference.jsonl",
        "evaluation_data/all_slugs_in_training_set.csv"
    )
    predicted = dh.get_predicted()

    # 2. Chat model (set temperature=0 for stable ranking)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3. Create the filter (turn on verbose & debug for full trace)
    filterer = RefTopicFilter(llm, max_topics=25, verbose=True, debug=True)

    # 4. Fetch full English text for each ref via Sefaria
    def sefaria_text_lookup(ref_str: str) -> str:
        return Ref(ref_str).text("en")

    # 5. Run sequentially so we can inspect each step
    result = filterer.filter_refs_sequential(
        predicted[20:25],                      # only first few for a quick test
        context_fetcher=sefaria_text_lookup
    )
    print("\nFINAL RESULT\n" + "="*60)
    for k, v in result.items():
        print(f"{k} → {v}")
