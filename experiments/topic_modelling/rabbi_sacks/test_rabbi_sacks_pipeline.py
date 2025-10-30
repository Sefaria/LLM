"""
Lightweight integration helper for the Rabbi Sacks pipeline.

Run with:
    pytest experiments/topic_modelling/rabbi_sacks/test_rabbi_sacks_pipeline.py
or execute the module directly for a quick manual smoke test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

from experiments.topic_modelling.rabbi_sacks.run_rabbi_sacks_pipeline import (
    PipelineConfig,
    run_pipeline,
)


def _missing_api_keys() -> list[str]:
    required = ("OPENAI_API_KEY",)
    return [env for env in required if not os.getenv(env)]


@pytest.mark.integration
def test_small_pipeline(tmp_path: Path) -> None:
    """
    Run both stages on two refs to verify end-to-end wiring.
    Skips automatically when OpenAI credentials are absent.
    """
    missing = _missing_api_keys()
    if missing:
        pytest.skip(f"Missing environment variables: {', '.join(missing)}")

    cfg = PipelineConfig(
        output_dir=tmp_path,
        limit=2,
        max_topics=5,
        debug=False,
    )
    result = run_pipeline(cfg)

    assert result.refs_processed == 2
    assert result.stage1_path.is_file()
    assert result.stage2_path.is_file()


def main(limit: Optional[int] = 3) -> None:
    """
    Convenience entry point for manual debugging without pytest.
    """
    missing = _missing_api_keys()
    if missing:
        print(f"Skipping run, missing environment variables: {', '.join(missing)}")
        return

    output_dir = Path(__file__).resolve().parent / "runs"
    cfg = PipelineConfig(output_dir=output_dir, limit=limit, max_topics=5, debug=True)
    result = run_pipeline(cfg)
    print(f"Stage 1 → {result.stage1_path}")
    print(f"Stage 2 → {result.stage2_path}")
    print(f"Processed {result.refs_processed} refs.")


if __name__ == "__main__":
    main()
