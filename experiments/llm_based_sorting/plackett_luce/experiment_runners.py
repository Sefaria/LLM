from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Protocol, Sequence

from anthropic import Anthropic


ItemId = Hashable
ItemRenderer = Callable[[ItemId], str]


class RankingExperimentRunner(Protocol):
    def __call__(self, items: list[ItemId]) -> list[ItemId]:
        ...


@dataclass
class DeterministicScoreExperimentRunner:
    score_lookup: Mapping[ItemId, float]

    def __call__(self, items: list[ItemId]) -> list[ItemId]:
        return sorted(items, key=lambda item: self.score_lookup[item], reverse=True)


class ClaudeListwiseRankingExperimentRunner:
    """
    Injected LLM-based ranking runner.

    The active-learning algorithm should call this object as a black box:
        ranking = runner(items_subset)

    The runner shows Claude the selected items, asks it to rank them by local
    indices 0..K-1, validates the response, and maps the local ranking back to
    the original item ids.
    """

    def __init__(
        self,
        item_renderer: ItemRenderer,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        ranking_instruction: str | None = None,
        client: Anthropic | None = None,
    ):
        self.item_renderer = item_renderer
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a careful ranking judge. Rank the provided items from best to worst "
            "according to the user instruction. Return only valid JSON."
        )
        self.ranking_instruction = ranking_instruction or (
            "Return JSON of the form {\"ranking\": [0, 2, 1]} where each number is the local "
            "index of an item in best-to-worst order. Use each local index exactly once."
        )
        self.client = client or Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def __call__(self, items: list[ItemId]) -> list[ItemId]:
        prompt = self._build_prompt(items)
        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        content = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        local_ranking = self.parse_local_ranking(content, expected_length=len(items))
        return [items[index] for index in local_ranking]

    def _build_prompt(self, items: Sequence[ItemId]) -> str:
        rendered_items = []
        for local_index, item in enumerate(items):
            rendered_items.append(
                "\n".join(
                    [
                        f"Local index: {local_index}",
                        f"Item id: {item}",
                        "Content:",
                        self.item_renderer(item).strip(),
                    ]
                )
            )

        item_block = "\n\n---\n\n".join(rendered_items)
        return (
            "Rank the following items from best to worst according to the task-specific criterion.\n\n"
            f"{self.ranking_instruction}\n\n"
            "Items:\n\n"
            f"{item_block}\n"
        )

    @staticmethod
    def parse_local_ranking(response_text: str, *, expected_length: int) -> list[int]:
        payload = ClaudeListwiseRankingExperimentRunner._extract_json_payload(response_text)
        ranking = payload["ranking"]
        if not isinstance(ranking, list):
            raise ValueError("Claude response must contain a list under the 'ranking' key.")
        if len(ranking) != expected_length:
            raise ValueError(
                f"Claude response returned {len(ranking)} ranked items; expected {expected_length}."
            )
        if sorted(ranking) != list(range(expected_length)):
            raise ValueError(
                "Claude response must be a permutation of local indices 0..K-1 exactly once."
            )
        return [int(index) for index in ranking]

    @staticmethod
    def _extract_json_payload(response_text: str) -> dict[str, Any]:
        text = response_text.strip()

        # Accept either a bare JSON list or an object with {"ranking": [...]}
        if text.startswith("["):
            ranking = json.loads(text)
            return {"ranking": ranking}
        if text.startswith("{"):
            return json.loads(text)

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
        if fenced_match:
            payload = fenced_match.group(1)
            if payload.startswith("["):
                return {"ranking": json.loads(payload)}
            return json.loads(payload)

        object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if object_match:
            return json.loads(object_match.group(1))

        list_match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
        if list_match:
            return {"ranking": json.loads(list_match.group(1))}

        raise ValueError("Could not parse JSON ranking from Claude response.")


def make_item_renderer_from_lookup(item_lookup: Mapping[ItemId, str]) -> ItemRenderer:
    def render(item: ItemId) -> str:
        return item_lookup[item]

    return render
