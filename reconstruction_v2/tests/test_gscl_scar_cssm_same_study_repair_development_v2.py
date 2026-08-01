"""Source-free integrity tests for the same-study SCAR repair runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import math
from pathlib import Path
import stat
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent import gscl_scar_cssm_repair_contract_v2 as contract
from assumption_agent.benchmarks import (
    gscl_scar_cssm_same_study_repair_development_v2 as subject,
)


def _fake_roots() -> dict[str, str]:
    return {
        "binding_file_sha256": "1" * 64,
        "binding_self_sha256": "2" * 64,
        "formal_result_self_sha256": "3" * 64,
    }


def _fake_closure() -> dict[str, Any]:
    return {
        "implementation_closure_sha256": "4" * 64,
        "input_archive_set_commitment_sha256": "5" * 64,
        "runner": {"file_sha256": "6" * 64},
    }


def _assert_intent(path: Path) -> bytes:
    intent_path = path / subject.ATTEMPT_INTENT_FILENAME
    raw = intent_path.read_bytes()
    parsed = json.loads(raw.decode("ascii"))
    assert raw == subject._canonical_bytes(parsed)
    assert (
        contract.validate_self_seal(
            parsed, expected_schema=subject.ATTEMPT_INTENT_SCHEMA
        )
        is not None
    )
    assert parsed["content_free_attempt_evidence"] is True
    assert parsed["private_input_access_counts_at_claim"] == {
        "label_pack": 0,
        "prediction_pack": 0,
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE(intent_path.stat().st_mode) == 0o600
    assert {row.name for row in path.iterdir()} == {
        subject.ATTEMPT_INTENT_FILENAME
    }
    return raw


def test_atomic_claim_has_exact_permissions_and_double_claim_fails_closed(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "one_attempt"
    binding = subject._claim_single_attempt(
        output_root,
        roots=_fake_roots(),
        implementation_closure=_fake_closure(),
    )
    raw = _assert_intent(output_root)
    assert binding == {
        "file_sha256": subject._file_sha256(raw),
        "filename": subject.ATTEMPT_INTENT_FILENAME,
        "self_sha256": json.loads(raw.decode("ascii"))["self_sha256"],
    }

    with pytest.raises(subject.SameStudyRepairDevelopmentError) as caught:
        subject._claim_single_attempt(
            output_root,
            roots=_fake_roots(),
            implementation_closure=_fake_closure(),
        )
    assert caught.value.issue_id == "SCAR_REPAIR_OUTPUT_ROOT_ALREADY_CLAIMED"
    assert _assert_intent(output_root) == raw


def test_concurrent_claim_allows_exactly_one_attempt(tmp_path: Path) -> None:
    output_root = tmp_path / "concurrent_attempt"
    barrier = threading.Barrier(2)

    def claim() -> tuple[str, Any]:
        barrier.wait()
        try:
            return (
                "ok",
                subject._claim_single_attempt(
                    output_root,
                    roots=_fake_roots(),
                    implementation_closure=_fake_closure(),
                ),
            )
        except subject.SameStudyRepairDevelopmentError as exc:
            return "error", exc.issue_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(lambda _: claim(), range(2)))
    assert sorted(row[0] for row in outcomes) == ["error", "ok"]
    assert [row[1] for row in outcomes if row[0] == "error"] == [
        "SCAR_REPAIR_OUTPUT_ROOT_ALREADY_CLAIMED"
    ]
    _assert_intent(output_root)


def test_private_read_failure_is_retained_and_second_run_reads_no_private_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "failed_attempt"
    archive_root = tmp_path / "old_archive"
    prediction_path = archive_root / "control" / "prediction_pack.private.json"
    label_path = tmp_path / "label_pack.private.json"
    binding = {
        "append_only_runtime_contract": {
            "append_only_output_root": str(output_root)
        },
        "old_remote_roots_read_only": {
            "prepared_label_pack": str(label_path),
            "private_result_archive_root": str(archive_root),
        },
    }
    monkeypatch.setattr(
        subject,
        "_validate_frozen_manifests",
        lambda **_: (binding, _fake_roots()),
    )
    monkeypatch.setattr(
        subject,
        "_validate_static_implementation_closure",
        lambda *_, **__: _fake_closure(),
    )
    private_reads: list[Path] = []

    def fail_private_read(path: Path, *, issue_id: str) -> tuple[dict[str, Any], str]:
        private_reads.append(path)
        raise subject.SameStudyRepairDevelopmentError(issue_id)

    monkeypatch.setattr(subject, "_read_json_once", fail_private_read)
    arguments = {
        "prediction_pack_path": prediction_path,
        "label_pack_path": label_path,
        "formal_result_path": tmp_path / "formal.json",
        "arm_spec_path": tmp_path / "arm.json",
        "analysis_spec_path": tmp_path / "analysis.json",
        "oracle_spec_path": tmp_path / "oracle.json",
        "binding_path": tmp_path / "binding.json",
    }

    with pytest.raises(subject.SameStudyRepairDevelopmentError) as first:
        subject.run_same_study_repair_development_v2(**arguments)
    assert first.value.issue_id == "SCAR_REPAIR_PREDICTION_PACK_INVALID"
    assert private_reads == [prediction_path]
    retained = _assert_intent(output_root)

    private_reads.clear()
    with pytest.raises(subject.SameStudyRepairDevelopmentError) as second:
        subject.run_same_study_repair_development_v2(**arguments)
    assert second.value.issue_id == "SCAR_REPAIR_OUTPUT_ROOT_ALREADY_CLAIMED"
    assert private_reads == []
    assert _assert_intent(output_root) == retained


def _exact_static_closure(
    *, prediction_path: Path, label_path: Path, launch_unit: Path
) -> dict[str, Any]:
    issue = "SCAR_REPAIR_IMPLEMENTATION_CLOSURE_INVALID"
    dependencies = subject._runtime_dependency_snapshot(issue)
    input_archives = {
        "label_pack": {
            "absolute_path": str(label_path),
            "file_sha256": "7" * 64,
            "self_sha256": "8" * 64,
        },
        "prediction_pack": {
            "absolute_path": str(prediction_path),
            "file_sha256": "9" * 64,
            "self_sha256": "a" * 64,
        },
    }

    def module_row(relative_path: str, path: Path) -> dict[str, str]:
        return {
            "file_sha256": subject._hash_runtime_file(path, issue),
            "relative_path": relative_path,
        }

    body: dict[str, Any] = {
        "contract_module": module_row(
            "assumption_agent/gscl_scar_cssm_repair_contract_v2.py",
            Path(str(contract.__file__)),
        ),
        "execution_authorized": True,
        "input_archive_set_commitment_sha256": contract.content_hash(
            input_archives
        ),
        "input_archives": input_archives,
        "launch_unit": {
            "absolute_path": str(launch_unit),
            "file_sha256": subject._hash_runtime_file(launch_unit, issue),
        },
        "mechanisms_module": module_row(
            "assumption_agent/gscl_scar_cssm_repair_mechanisms_v2.py",
            Path(str(subject.mechanisms.__file__)),
        ),
        "python_executable": {
            "absolute_path": sys.executable,
            "file_sha256": subject._hash_runtime_file(Path(sys.executable), issue),
        },
        "runner": module_row(
            "assumption_agent/benchmarks/gscl_scar_cssm_same_study_repair_development_v2.py",
            Path(subject.__file__),
        ),
        "runtime_dependencies": dependencies,
        "runtime_dependency_binding_sha256": contract.content_hash(dependencies),
        "runtime_environment": dict(subject._REQUIRED_RUNTIME_ENVIRONMENT),
        "runtime_environment_binding_sha256": contract.content_hash(
            subject._REQUIRED_RUNTIME_ENVIRONMENT
        ),
        "status": "FROZEN_EXACT_IMPLEMENTATION_RUNTIME_AND_INPUT_CLOSURE",
    }
    return {
        **body,
        "implementation_closure_sha256": contract.content_hash(body),
    }


def test_static_closure_binds_environment_numpy_svd_blas_and_launch_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in subject._REQUIRED_RUNTIME_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    launch_unit = tmp_path / "formal.service"
    launch_unit.write_text("[Service]\nExecStart=/usr/bin/false\n", encoding="utf-8")
    monkeypatch.setattr(subject, "_EXPECTED_LAUNCH_UNIT_PATH", str(launch_unit))
    prediction_path = tmp_path / "prediction.private.json"
    label_path = tmp_path / "label.private.json"
    closure = _exact_static_closure(
        prediction_path=prediction_path,
        label_path=label_path,
        launch_unit=launch_unit,
    )
    validated = subject._validate_static_implementation_closure(
        {"implementation_closure_binding": closure},
        prediction_pack_path=prediction_path,
        label_pack_path=label_path,
    )
    assert validated == closure
    dependencies = closure["runtime_dependencies"]
    assert set(dependencies) == {
        "loaded_blas_shared_libraries",
        "numpy_lapack_lite",
        "numpy_multiarray_umath",
        "numpy_package_init",
        "numpy_umath_linalg",
        "numpy_version",
    }
    assert dependencies["loaded_blas_shared_libraries"]

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(subject.SameStudyRepairDevelopmentError) as env_error:
        subject._validate_static_implementation_closure(
            {"implementation_closure_binding": closure},
            prediction_pack_path=prediction_path,
            label_pack_path=label_path,
        )
    assert env_error.value.issue_id == "SCAR_REPAIR_IMPLEMENTATION_CLOSURE_INVALID"

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    launch_unit.write_text("[Service]\nExecStart=/usr/bin/true\n", encoding="utf-8")
    with pytest.raises(subject.SameStudyRepairDevelopmentError) as unit_error:
        subject._validate_static_implementation_closure(
            {"implementation_closure_binding": closure},
            prediction_pack_path=prediction_path,
            label_pack_path=label_path,
        )
    assert unit_error.value.issue_id == "SCAR_REPAIR_IMPLEMENTATION_CLOSURE_INVALID"


def test_full_data_artifact_failure_is_all_noop_and_not_a_primary_count() -> None:
    artifact = subject._full_data_artifact(
        (), [None] * subject.FOLD_COUNT, feature_mode="U1"
    )
    assert artifact == {
        "failure_issue_id": "SCAR_REPAIR_OUTER_FIT_FAILURE_PROPAGATED",
        "feature_mode": "U1",
        "status": "ALL_NOOP",
        "threshold": {"kind": "ALL_NOOP"},
    }
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "failure_count += full_failure_count" not in source
    assert "u0_failure_count += u0_full_failure_count" not in source


def _synthetic_item(fold: int) -> subject.PrimaryItem:
    baseline = ((f"left-{fold}", f"right-{fold}"),)
    baseline_swap = ((f"right-{fold}", f"left-{fold}"),)
    alternative = subject.Candidate(
        proposal_hash=f"{fold + 10:064x}",
        mapping=((f"left-{fold}", f"other-{fold}"),),
        semantic_score=10,
        features=tuple(float(index) for index in range(16)),
        null_features=tuple(float(index + 100) for index in range(16)),
        target_delta=Fraction(-1),
        exact_against_gold=False,
    )
    return subject.PrimaryItem(
        item_token=f"scar-item-v1-{fold:064x}",
        arity=1,
        domain_relation="synthetic",
        stratum="synthetic::ARITY_1",
        fold=fold,
        baseline=baseline,
        baseline_swap=baseline_swap,
        gold_base=baseline,
        gold_swap=baseline_swap,
        baseline_f1=Fraction(1),
        baseline_exact=True,
        candidates=(alternative,),
        common_v1_base=baseline,
        common_v1_swap=baseline_swap,
        common_v1_answered_base=True,
        common_v1_answered_swap=True,
        common_v1_f1=Fraction(1),
    )


def test_u0_is_exactly_the_two_semantic_features() -> None:
    candidate = _synthetic_item(0).candidates[0]
    assert subject._candidate_features(candidate, "U0") == candidate.features[1:3]


def test_nested_crossfit_excludes_outer_and_inner_and_noop_is_exact_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = tuple(_synthetic_item(fold) for fold in range(subject.FOLD_COUNT))
    fit_scopes: list[tuple[int, ...]] = []

    def fake_fit(
        rows: Any, *, feature_mode: str
    ) -> SimpleNamespace:
        scope = tuple(sorted(row.fold for row in rows))
        fit_scopes.append(scope)
        return SimpleNamespace(commitment=contract.content_hash({"scope": list(scope)}))

    monkeypatch.setattr(subject, "_fit", fake_fit)
    monkeypatch.setattr(
        subject,
        "_best_candidate",
        lambda item, model, *, feature_mode: (item.candidates[0], 0.5),
    )
    monkeypatch.setattr(subject, "_select_threshold", lambda _: math.inf)

    applied, receipts, outer_models, failure_count = subject._nested_crossfit(
        items, feature_mode="U1"
    )
    assert failure_count == 0
    assert len(receipts) == subject.FOLD_COUNT
    assert len(outer_models) == subject.FOLD_COUNT
    cursor = 0
    all_folds = set(range(subject.FOLD_COUNT))
    for outer_fold in range(subject.FOLD_COUNT):
        for inner_fold in range(subject.FOLD_COUNT):
            if inner_fold == outer_fold:
                continue
            assert set(fit_scopes[cursor]) == all_folds - {
                outer_fold,
                inner_fold,
            }
            cursor += 1
        assert set(fit_scopes[cursor]) == all_folds - {outer_fold}
        cursor += 1
    assert cursor == len(fit_scopes) == subject.FOLD_COUNT**2
    by_token = {row.item.item_token: row for row in applied}
    for item in items:
        row = by_token[item.item_token]
        assert row.selected is None
        assert row.output_base is item.baseline
        assert row.output_swap is item.baseline_swap
