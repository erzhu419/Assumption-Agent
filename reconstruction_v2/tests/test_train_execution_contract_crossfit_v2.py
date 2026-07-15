from __future__ import annotations

from pathlib import Path

import pytest

from assumption_agent.benchmarks.train_execution_contract_crossfit_v2 import (
    EXPECTED_CANDIDATE_HASH,
    EXPECTED_CONTRACT_HASH,
    EXPECTED_FOLD_RECEIPT_HASH,
    EXPECTED_WORK_UNIT_HASH,
    GRAPH_SOURCE_ITEM_IDS,
    SOURCE_RANKING_REPORT_RELATIVE_PATH,
    compile_v320_train_item_out_crossfit_v2,
)
from assumption_agent.benchmarks.v320_train_candidate_material_v2 import (
    V320_SOURCE_RELATIVE_ROOT,
)
from assumption_agent.models import stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / V320_SOURCE_RELATIVE_ROOT
SOURCE_RANKING_REPORT = PROJECT_ROOT / SOURCE_RANKING_REPORT_RELATIVE_PATH


@pytest.mark.skipif(
    not SOURCE_ROOT.is_dir() or not SOURCE_RANKING_REPORT.is_file(),
    reason="historical v3.20 source ranking is not installed",
)
def test_compiles_exact_single_train_item_out_route_without_scoring(
    tmp_path: Path,
) -> None:
    result = compile_v320_train_item_out_crossfit_v2(
        project_root=PROJECT_ROOT,
        output_root=tmp_path / "crossfit",
    )

    result.verify()
    assert result.report["fold_receipt_hash"] == EXPECTED_FOLD_RECEIPT_HASH
    assert result.contract.contract_hash == EXPECTED_CONTRACT_HASH
    assert result.candidate.candidate_hash == EXPECTED_CANDIDATE_HASH
    assert result.work.work_unit_hash == EXPECTED_WORK_UNIT_HASH
    assert result.report["expected_active_execution_count"] == 1
    assert result.report["expected_inactive_raw_replay_count"] == 37
    assert result.report["contract_source_task_id_hashes"] == sorted(
        stable_hash({"task_id": value}) for value in GRAPH_SOURCE_ITEM_IDS
    )
    assert result.report["heldout_excluded_from_graph"] is True
    assert result.report["heldout_excluded_from_contract"] is True
    assert result.report["fold_and_workflow_selected_post_source_ranking"] is True
    assert result.report["targeted_item_out_refit_falsification"] is True
    assert result.report["unbiased_crossfit"] is False
    assert result.report["workflow_reselected_without_heldout"] is False
    assert result.report["compile_is_non_scoring_diagnostic"] is True
    assert result.report["freeze_or_promotion_authorized"] is False
    assert result.report["validation_or_test_content_accessed"] is False
    assert result.report["model_calls"] == 0
    assert result.report["evaluator_calls"] == 0
    assert result.report["online_judge_calls"] == 0
