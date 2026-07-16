from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks.archive_epoch_retention_integration_v1 import (
    _restore_archive,
    run_integration,
)
from assumption_agent.models import stable_hash


def test_archive_epoch_retention_integration_is_exact_and_offline(tmp_path) -> None:
    report = run_integration(tmp_path)

    assert report["integration_passed"] is True
    assert report["diagnostic_only"] is True
    assert report["performance_gate"] is False
    assert report["generation_one"]["promotion_allowed"] is True
    retention = report["generation_two_retention"]
    assert retention["repair_participated"] is False
    assert retention["estimand"] == "Y(P_plus_Q)-Y(Q)"
    assert retention["retention_effect_count"] == 4
    assert retention["arm_counts"]["Q"]["success_count"] == 0
    assert retention["arm_counts"]["P_plus_Q"]["success_count"] == 4
    epoch = report["evaluator_epoch"]
    assert epoch["transition"]["promoted"] is True
    assert epoch["anchor"]["scores_computed_by_evaluator_implementations"] is True
    assert epoch["anchor"]["incumbent_successes"] == 4
    assert epoch["anchor"]["challenger_successes"] == 8
    assert epoch["ready_before_reevaluation"] is False
    assert epoch["ready_after_reevaluation"] is True
    assert epoch["old_incumbent_node_id"] != epoch["rebased_incumbent_node_id"]
    assert report["invariants"]["independent_objective_preserved"] is True
    assert report["invariants"]["incumbent_rebased_to_new_epoch_node"] is True
    assert (
        report["invariants"]["incumbent_behavior_retained_after_revalidation"]
        is True
    )
    assert report["persistence"]["archive_reload_exact"] is True
    assert report["model_calls"] == 0
    assert report["task_backend_calls"] == 0
    assert report["online_evaluator_calls"] == 0
    assert report["sealed_or_test_content_accessed"] is False

    assert json.loads((tmp_path / "report.json").read_text()) == report
    archive = json.loads((tmp_path / "archive.json").read_text())
    assert stable_hash(_restore_archive(archive).to_dict()) == stable_hash(archive)
    incumbent = archive["nodes"][archive["incumbent_id"]]
    assert incumbent["id"] == epoch["rebased_incumbent_node_id"]
    assert incumbent["evaluator_epoch_id"] == epoch["transition"]["next_epoch"]["id"]
    assert archive["score_records"][epoch["reevaluation_score_record_id"]][
        "archive_node_id"
    ] == incumbent["id"]


def test_archive_checkpoint_restore_fails_closed_on_tamper(tmp_path) -> None:
    run_integration(tmp_path)
    archive = json.loads((tmp_path / "archive.json").read_text())
    archive["score_records"][
        next(iter(archive["score_records"]))
    ]["successes"] += 1

    with pytest.raises(ValueError, match="round trip drifted"):
        _restore_archive(archive)


def test_public_integration_result_binds_rebased_epoch_artifacts() -> None:
    project = Path(__file__).parents[1]
    path = project / "manifests" / "archive_epoch_retention_integration_result_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = payload.pop("result_hash")
    assert stable_hash(payload) == declared
    assert payload["evaluator_anchor_scores_computed"] is True
    assert payload["evaluator_incumbent_anchor_successes"] == 4
    assert payload["evaluator_challenger_anchor_successes"] == 8
    assert payload["incumbent_rebased_to_new_epoch_node"] is True
    assert payload["incumbent_behavior_retained_after_revalidation"] is True
    assert payload["invariant_count"] == 13
    artifact_root = project / "artifacts" / "archive_epoch_retention_integration_v1"
    assert hashlib.sha256((artifact_root / "report.json").read_bytes()).hexdigest() == payload[
        "report_file_sha256"
    ]
    assert hashlib.sha256((artifact_root / "archive.json").read_bytes()).hexdigest() == payload[
        "archive_file_sha256"
    ]
