from __future__ import annotations

import copy
from pathlib import Path

import pytest

from replication_runtime.financial_sec13f_contract_v2 import (
    replication_b_freeze as freeze,
)
from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    read_json,
    verify_measurement_view,
)


PROJECT = Path(__file__).resolve().parents[1]


def _fixed_evidence() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    preregistration = read_json(
        PROJECT / freeze.PREREGISTRATION_RELATIVE_PATH
    )
    acquisition = read_json(PROJECT / freeze.ACQUISITION_RELATIVE_PATH)
    view = verify_measurement_view(
        read_json(PROJECT / freeze.MEASUREMENT_VIEW_RELATIVE_PATH)
    )
    formation = read_json(PROJECT / freeze.FORMATION_RELATIVE_PATH)
    return preregistration, acquisition, view, formation


def test_fixed_replication_b_evidence_is_exact_and_committed() -> None:
    preregistration, acquisition, view, formation = _fixed_evidence()

    prereg_binding, bound_preregistration = freeze._fixed_committed_json(
        PROJECT,
        relative_path=freeze.PREREGISTRATION_RELATIVE_PATH,
        expected_file_sha256=freeze.PREREGISTRATION_FILE_SHA256,
        label="replication-B preregistration",
    )
    assert bound_preregistration == preregistration
    assert prereg_binding["file_sha256"] == (
        freeze.PREREGISTRATION_FILE_SHA256
    )
    assert freeze._validate_replication_b_preregistration(
        preregistration
    ) == freeze.PREREGISTRATION_MANIFEST_HASH
    assert freeze._validate_inherited_acquisition(
        acquisition,
        preregistration=preregistration,
    ) == freeze.ACQUISITION_RECEIPT_HASH
    freeze._validate_view_identity(view)
    assert freeze._validate_replication_b_formation(
        formation,
        preregistration=preregistration,
        acquisition_receipt_hash=freeze.ACQUISITION_RECEIPT_HASH,
        measurement_view=view,
    ) == freeze.FORMATION_RECEIPT_HASH


def test_preregistration_and_formation_tampering_fail_closed() -> None:
    preregistration, acquisition, view, formation = _fixed_evidence()

    changed_preregistration = copy.deepcopy(preregistration)
    changed_preregistration["analysis_policy"][
        "performance_gate_bound"
    ] = True
    prereg_body = dict(changed_preregistration)
    prereg_body.pop("manifest_hash")
    changed_preregistration["manifest_hash"] = payload_hash(prereg_body)
    with pytest.raises(
        freeze.ContractFreezeError,
        match="preregistration drifted",
    ):
        freeze._validate_replication_b_preregistration(
            changed_preregistration
        )

    # Even after recomputing every local self hash, the fixed formation hash
    # and zero-collision semantics reject a substituted measurement set.
    changed_formation = copy.deepcopy(formation)
    audit = changed_formation["exclusion_collision_audits"][0]
    collision = audit["collision_audit"]
    collision["query_collision_count"] = 1
    collision_body = dict(collision)
    collision_body.pop("audit_hash")
    collision["audit_hash"] = payload_hash(collision_body)
    audit_body = dict(audit)
    audit_body.pop("binding_hash")
    audit["binding_hash"] = payload_hash(audit_body)
    changed_formation["exclusion_collision_audit_set_hash"] = payload_hash(
        changed_formation["exclusion_collision_audits"]
    )
    formation_body = dict(changed_formation)
    formation_body.pop("receipt_hash")
    changed_formation["receipt_hash"] = payload_hash(formation_body)
    with pytest.raises(freeze.ContractFreezeError):
        freeze._validate_replication_b_formation(
            changed_formation,
            preregistration=preregistration,
            acquisition_receipt_hash=freeze._validate_inherited_acquisition(
                acquisition,
                preregistration=preregistration,
            ),
            measurement_view=view,
        )


def _safe_plan() -> dict[str, object]:
    work_units: list[dict[str, object]] = []
    for index in range(8):
        for arm in ("candidate", "raw"):
            work_units.append(
                {
                    "pair_id": f"pair-{index}",
                    "arm": arm,
                    "candidate_source_required": arm == "candidate",
                    "retry_count": 0,
                    "raw_content_persisted": False,
                }
            )
    return {
        "physical_work_unit_count": 16,
        "measurement_pair_count": 8,
        "raw_execution_count": 8,
        "candidate_execution_count": 8,
        "official_hipporag": False,
        "official_hipporag_execution_count": 0,
        "projection_count": 0,
        "maximum_workers": 16,
        "retry_count": 0,
        "retry_policy": "none",
        "descriptive_only": True,
        "performance_gate_bound": False,
        "promotion_authorized": False,
        "work_units": work_units,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_workers", 8),
        ("retry_count", 1),
        ("performance_gate_bound", True),
        ("promotion_authorized", True),
    ),
)
def test_exact_plan_rejects_non_preregistered_grid(
    field: str,
    value: object,
) -> None:
    safe = _safe_plan()
    freeze._assert_exact_replication_b_plan(safe)
    safe[field] = value
    with pytest.raises(freeze.ContractFreezeError, match="plan"):
        freeze._assert_exact_replication_b_plan(safe)


def test_source_closure_must_contain_the_preregistered_runner_fix() -> None:
    preregistration, _, _, _ = _fixed_evidence()
    closure = {
        "files": [
            {
                "relative_path": (
                    "replication_runtime/financial_sec13f_contract_v2/runner.py"
                ),
                "file_sha256": freeze.FIXED_RUNNER_SHA256,
            }
        ]
    }
    freeze._validate_infrastructure_fix_in_source_closure(
        closure,
        preregistration,
    )
    closure["files"][0]["file_sha256"] = "0" * 64
    with pytest.raises(freeze.ContractFreezeError, match="fix"):
        freeze._validate_infrastructure_fix_in_source_closure(
            closure,
            preregistration,
        )


def test_cli_has_no_substitutable_study_evidence_paths() -> None:
    parser = freeze._parser()
    args = parser.parse_args(
        [
            "execution-freeze",
            "--project-root",
            str(PROJECT),
            "--provider-env-file",
            str(PROJECT.parent / ".env"),
            "--output",
            str(PROJECT / "unused.json"),
        ]
    )
    assert args.command == "execution-freeze"
    assert not hasattr(args, "preregistration")
    assert not hasattr(args, "measurement_view")
    assert not hasattr(args, "formation_receipt")
