from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from assumption_agent.benchmarks.v320_train_candidate_material_v2 import (
    V320_SOURCE_RELATIVE_ROOT,
    V320TrainCandidateMaterialError,
    load_v320_train_candidate_material_v2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / V320_SOURCE_RELATIVE_ROOT


@pytest.mark.skipif(
    not SOURCE_ROOT.is_dir(),
    reason="historical v3.20 development source is not installed",
)
def test_restores_exact_fourteen_train_only_candidate_subsets() -> None:
    material = load_v320_train_candidate_material_v2(
        project_root=PROJECT_ROOT,
    )

    material.verify()
    payload = material.receipt.safe_payload()
    assert payload["program_count"] == 6
    assert payload["candidate_subset_count"] == 14
    assert payload["expected_full_outcome_count"] == 532
    assert payload["expected_active_route_count"] == 56
    assert payload["expected_inactive_replay_count"] == 476
    assert payload["validation_or_test_content_accessed"] is False
    assert payload["model_calls"] == 0
    assert payload["evaluator_calls"] == 0
    assert sum(row.selected for row in material.subsets) == 2
    assert {len(row.program_ids) for row in material.subsets} == {1, 2, 3}

    with pytest.raises(
        V320TrainCandidateMaterialError,
        match="source receipt drifted",
    ):
        replace(
            material.receipt,
            expected_active_route_count=55,
        ).verify()

    with pytest.raises(
        V320TrainCandidateMaterialError,
        match="subset receipt drifted",
    ):
        replace(
            material.subsets[0],
            expected_active_item_count=0,
        ).verify()
