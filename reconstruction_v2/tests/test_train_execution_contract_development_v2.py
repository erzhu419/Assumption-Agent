from __future__ import annotations

from pathlib import Path

import pytest

from assumption_agent.benchmarks.train_execution_contract_development_v2 import (
    compile_v320_train_execution_contract_candidates_v2,
)
from assumption_agent.benchmarks.v320_train_candidate_material_v2 import (
    V320_SOURCE_RELATIVE_ROOT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / V320_SOURCE_RELATIVE_ROOT


@pytest.mark.skipif(
    not SOURCE_ROOT.is_dir(),
    reason="historical v3.20 development source is not installed",
)
def test_compiles_full_sparse_candidate_grid_without_scoring(
    tmp_path: Path,
) -> None:
    result = compile_v320_train_execution_contract_candidates_v2(
        project_root=PROJECT_ROOT,
        output_root=tmp_path / "integration",
    )

    result.verify()
    assert len(result.candidates) == 14
    assert len(result.candidate_specs) == 14
    assert len(result.candidate_bundles_by_hash) == 14
    assert result.report["full_outcome_count"] == 532
    assert result.report["active_execution_count"] == 56
    assert result.report["inactive_raw_replay_count"] == 476
    assert result.report["scoring_performed"] is False
    assert result.report["freeze_or_promotion_authorized"] is False
    assert result.report["validation_or_test_content_accessed"] is False
    assert result.report["model_calls"] == 0
    assert result.report["evaluator_calls"] == 0
    assert result.report["online_judge_calls"] == 0
