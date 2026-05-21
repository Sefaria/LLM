from __future__ import annotations

import pytest

from experiments.llm_based_sorting.plackett_luce.experiment_runners import (
    ClaudeListwiseRankingExperimentRunner,
    DeterministicScoreExperimentRunner,
)


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeMessagesAPI:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def create(self, **kwargs):
        return _FakeResponse(self.response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text: str):
        self.messages = _FakeMessagesAPI(response_text)


def test_deterministic_score_runner_orders_by_score_descending() -> None:
    runner = DeterministicScoreExperimentRunner({10: 0.2, 11: 1.4, 12: 0.7})
    assert runner([10, 12, 11]) == [11, 12, 10]


def test_claude_runner_maps_local_indices_back_to_original_items() -> None:
    runner = ClaudeListwiseRankingExperimentRunner(
        item_renderer=lambda item: f"Item text for {item}",
        client=_FakeAnthropicClient('{"ranking": [2, 0, 1]}'),
    )

    ranking = runner([100, 200, 300])

    assert ranking == [300, 100, 200]


def test_claude_runner_parses_fenced_json_list() -> None:
    parsed = ClaudeListwiseRankingExperimentRunner.parse_local_ranking(
        "```json\n[1, 0, 2]\n```",
        expected_length=3,
    )
    assert parsed == [1, 0, 2]


def test_claude_runner_rejects_non_permutations() -> None:
    with pytest.raises(ValueError, match="permutation"):
        ClaudeListwiseRankingExperimentRunner.parse_local_ranking(
            '{"ranking": [0, 0, 1]}',
            expected_length=3,
        )
