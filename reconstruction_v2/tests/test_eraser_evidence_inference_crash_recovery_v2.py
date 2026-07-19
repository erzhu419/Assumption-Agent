from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    eraser_evidence_inference_crash_recovery_v2 as subject,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_direct_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_formal_controller_v1 as formal_controller,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_local_runtime_v1 as local_runtime,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_source_qualification_v1 as source_qualification,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_three_arm_scheduler_v1 as scheduler,
)


@pytest.fixture
def private_tmp_path() -> Any:
    """Use WSL's native filesystem; DrvFS does not preserve private modes."""

    with tempfile.TemporaryDirectory(
        prefix="eraser-recovery-v2-", dir="/tmp"
    ) as directory:
        yield Path(directory)


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _self_hashed(
    body: dict[str, Any], field: str, *, canonical=subject.stable_hash
) -> dict[str, Any]:
    return {**body, field: canonical(body)}


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    path.chmod(0o700)


def _write_private(path: Path, raw: bytes) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, mode=0o700)
        path.parent.chmod(0o700)
    path.write_bytes(raw)
    path.chmod(0o600)


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    _write_private(path, subject.canonical_bytes(payload))


def test_tree_snapshot_uses_exact_metadata_rows_and_rejects_symlink(
    private_tmp_path: Path,
) -> None:
    root = private_tmp_path / "private"
    _mkdir_private(root)
    directory = root / "nested"
    _mkdir_private(directory)
    file_a = root / "a.bin"
    file_b = directory / "b.bin"
    _write_private(file_a, b"alpha")
    _write_private(file_b, b"beta")

    rows = []
    for path in (file_a, directory, file_b):
        metadata = path.lstat()
        is_directory = path == directory
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "type": "directory" if is_directory else "regular_file",
                "mode": stat.S_IMODE(metadata.st_mode),
                "size": metadata.st_size,
                "file_sha256": None if is_directory else _sha(path.read_bytes()),
            }
        )
    rows.sort(key=lambda row: row["relative_path"].encode("utf-8"))
    expected_raw = subject.canonical_bytes(rows)

    snapshot = subject.snapshot_private_tree(root)
    assert snapshot.tree_sha256 == hashlib.sha256(expected_raw).hexdigest()
    assert snapshot.canonical_json_byte_count == len(expected_raw)
    assert snapshot.descendant_entry_count == 3
    assert snapshot.descendant_regular_file_count == 2
    assert snapshot.descendant_directory_count == 1
    assert snapshot.descendant_regular_file_total_bytes == 9

    link = root / "forbidden-link"
    link.symlink_to(file_a)
    with pytest.raises(subject.EraserEvidenceInferenceCrashRecoveryError):
        subject.snapshot_private_tree(root)


def test_recovery_implementation_freeze_binds_exact_two_new_files(
    private_tmp_path: Path,
) -> None:
    tmp_path = private_tmp_path
    rows = []
    for role, relative in subject.RECOVERY_IMPLEMENTATION_ROLE_PATHS.items():
        path = tmp_path / relative
        raw = f"synthetic:{role}\n".encode("ascii")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        rows.append(
            {
                "role": role,
                "relative_path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    body = {
        "schema": f"{subject.VERSION}_implementation_freeze",
        "version": "v2",
        "status": "frozen_before_recovery_marker_or_archive_transition",
        "recovery_design_sha256": subject.EXPECTED_RECOVERY_DESIGN_SHA256,
        "recovery_design_file_sha256": (
            subject.EXPECTED_RECOVERY_DESIGN_FILE_SHA256
        ),
        "incident_sha256": subject.EXPECTED_INCIDENT_SHA256,
        "incident_file_sha256": subject.EXPECTED_INCIDENT_FILE_SHA256,
        "base_design_sha256": subject.EXPECTED_BASE_DESIGN_SHA256,
        "base_design_file_sha256": subject.EXPECTED_BASE_DESIGN_FILE_SHA256,
        "base_implementation_freeze_sha256": (
            subject.EXPECTED_BASE_IMPLEMENTATION_FREEZE_SHA256
        ),
        "base_implementation_freeze_file_sha256": (
            subject.EXPECTED_BASE_IMPLEMENTATION_FREEZE_FILE_SHA256
        ),
        "required_role_registry": list(
            subject.RECOVERY_IMPLEMENTATION_ROLE_PATHS
        ),
        "implementation_binding": {"files": rows},
        "synthetic_test_receipt": {
            "collected_case_count": 5,
            "passed_case_count": 5,
            "real_source_or_benchmark_item_read": False,
            "model_inference_calls": 0,
            "online_or_network_calls": 0,
        },
    }
    payload = _self_hashed(body, "implementation_freeze_sha256")
    freeze_path = tmp_path / subject.RECOVERY_IMPLEMENTATION_FREEZE_RELATIVE
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_bytes(subject.canonical_bytes(payload))

    verified = subject.verify_recovery_implementation_freeze(
        project=tmp_path, freeze_path=freeze_path
    )
    assert verified["implementation_freeze_sha256"] == payload[
        "implementation_freeze_sha256"
    ]

    controller_path = (
        tmp_path
        / subject.RECOVERY_IMPLEMENTATION_ROLE_PATHS[
            "crash_recovery_controller"
        ]
    )
    controller_path.write_text("tampered\n", encoding="ascii")
    with pytest.raises(subject.EraserEvidenceInferenceCrashRecoveryError):
        subject.verify_recovery_implementation_freeze(
            project=tmp_path, freeze_path=freeze_path
        )


def _qualification_payload() -> dict[str, Any]:
    body = {
        "schema": source_qualification.SCHEMA,
        "version": source_qualification.VERSION,
        "status": "passed_source_qualification_no_selection",
        "source_or_item_content_persisted": False,
    }
    return _self_hashed(
        body,
        "qualification_sha256",
        canonical=formal_controller.stable_hash,
    )


def _build_interrupted_root(project: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    formal_root = project / subject.CANONICAL_FORMAL_ROOT_RELATIVE
    _mkdir_private(formal_root)
    controller = formal_root / formal_controller.CONTROLLER_DIRECTORY
    acquisition_root = formal_root / formal_controller.ACQUISITION_DIRECTORY
    _mkdir_private(controller)
    _mkdir_private(acquisition_root)
    views = acquisition_root / "views"
    _mkdir_private(views)

    qualification = _qualification_payload()
    _write_private_json(
        controller / formal_controller.QUALIFICATION_FILENAME,
        qualification,
    )
    clone_bytes = {
        "acquisition.marker.private.json": b"marker-secret-custody-bytes",
        "assignment.private.json": b"fixed-assignment-bytes",
        "acquisition.receipt.json": b"aggregate-public-receipt-bytes",
        "views/A_form.private.json": b"A-form-private-item-view-bytes",
        "views/F_search.private.json": b"F-search-private-item-view-bytes",
    }
    for relative, raw in clone_bytes.items():
        _write_private(acquisition_root / relative, raw)

    stage = formal_root / "official_hipporag_item_stage_parent"
    _mkdir_private(stage)
    a_form = stage / "A_form"
    f_search = stage / "F_search"
    _mkdir_private(a_form)
    _mkdir_private(f_search)
    work = a_form / f"{_sha('partial-item')}.00000001.work"
    _mkdir_private(work)
    cache = work / "cache"
    _mkdir_private(cache)
    _write_private(work / "single_item.input.json", b"private-partial-input")
    _write_private(cache / "embeddings.parquet", b"partial-cache-must-not-reuse")
    return clone_bytes, qualification


def _write_marker_binding_files(project: Path) -> None:
    for relative, raw in (
        (subject.BASE_IMPLEMENTATION_FREEZE_RELATIVE, b"base-freeze"),
        (subject.RECOVERY_IMPLEMENTATION_FREEZE_RELATIVE, b"recovery-freeze"),
        (subject.BASE_DESIGN_RELATIVE, b"base-design"),
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def _synthetic_prerequisites(
    *, project: Path, qualification: dict[str, Any]
) -> subject.RecoveryPrerequisites:
    crash_tree = subject.snapshot_private_tree(
        project / subject.CANONICAL_FORMAL_ROOT_RELATIVE
    )
    qualification_raw = subject.canonical_bytes(qualification)
    incident = {
        "binding": {
            "qualification_sha256": qualification["qualification_sha256"],
            "qualification_file_sha256": hashlib.sha256(
                qualification_raw
            ).hexdigest(),
        }
    }
    design = {
        "base_acquisition_custody": {
            "same_private_assignment_sha256": _sha("assignment"),
            "same_public_receipt_sha256": _sha("public"),
            "same_selection_secret_commitment_sha256": _sha("secret"),
        }
    }
    return subject.RecoveryPrerequisites(
        base_freeze={"implementation_freeze_sha256": _sha("base-freeze")},
        base_freeze_file_sha256=_sha(b"base-freeze"),
        recovery_freeze={
            "implementation_freeze_sha256": _sha("recovery-freeze")
        },
        recovery_freeze_file_sha256=_sha(b"recovery-freeze"),
        incident=incident,
        design=design,
        crash_tree=crash_tree,
    )


def test_atomic_archive_moves_whole_root_and_clone_is_exact_five_bytes(
    private_tmp_path: Path,
) -> None:
    project = private_tmp_path / "project"
    project.mkdir()
    clone_bytes, qualification = _build_interrupted_root(project)
    _write_marker_binding_files(project)
    prerequisites = _synthetic_prerequisites(
        project=project, qualification=qualification
    )
    recovery_root = project / subject.RECOVERY_ROOT_RELATIVE
    recovery_root.parent.mkdir(parents=True, exist_ok=True)
    _mkdir_private(recovery_root)

    receipt = subject._atomic_archive_interrupted_root(
        project=project,
        recovery_root=recovery_root,
        snapshot=prerequisites.crash_tree,
    )
    assert receipt["same_filesystem_atomic_os_rename_count"] == 1
    assert not (project / subject.CANONICAL_FORMAL_ROOT_RELATIVE).exists()
    archived = project / subject.ARCHIVE_DESTINATION_RELATIVE
    assert archived.exists()
    assert (
        archived
        / "official_hipporag_item_stage_parent"
        / "A_form"
    ).exists()

    acquisition_root, clone_receipt = (
        subject._recreate_and_clone_base_acquisition(
            project=project, recovery_root=recovery_root
        )
    )
    assert clone_receipt["exact_clone_file_count"] == 5
    assert [row["relative_path"] for row in clone_receipt["files"]] == list(
        subject.CLONED_ACQUISITION_RELATIVE_PATHS
    )
    for relative, expected in clone_bytes.items():
        clone = acquisition_root / relative
        assert clone.read_bytes() == expected
        assert stat.S_IMODE(clone.stat().st_mode) == 0o600
    assert not (
        project
        / subject.CANONICAL_FORMAL_ROOT_RELATIVE
        / "official_hipporag_item_stage_parent"
    ).exists()
    assert subject.snapshot_private_tree(archived) == prerequisites.crash_tree


def _install_synthetic_run_mocks(
    *,
    monkeypatch: pytest.MonkeyPatch,
    project: Path,
    prerequisites: subject.RecoveryPrerequisites,
    fail_inner: bool,
) -> tuple[dict[str, int], Any, Any]:
    original_counts = {
        "qualification": 0,
        "acquire": 0,
        "verify": 0,
        "inner": 0,
    }

    def forbidden_original_qualification(_project: Path) -> dict[str, Any]:
        original_counts["qualification"] += 1
        pytest.fail("original qualifier body must not execute")

    def forbidden_original_acquire(**_kwargs: Any) -> dict[str, Any]:
        original_counts["acquire"] += 1
        pytest.fail("original acquire_once body must not execute")

    monkeypatch.setattr(
        source_qualification,
        "build_formal_qualification",
        forbidden_original_qualification,
    )
    monkeypatch.setattr(acquisition, "acquire_once", forbidden_original_acquire)

    def fake_verify_prerequisites(*, project: Path) -> subject.RecoveryPrerequisites:
        if os.path.lexists(project / subject.RECOVERY_ROOT_RELATIVE):
            raise subject.EraserEvidenceInferenceCrashRecoveryError(
                "recovery root already exists; a second attempt is forbidden"
            )
        return prerequisites

    monkeypatch.setattr(
        subject, "verify_recovery_prerequisites", fake_verify_prerequisites
    )

    runtime_config = SimpleNamespace(
        project=project,
        hippo_stage_parent_root=(
            project
            / subject.CANONICAL_FORMAL_ROOT_RELATIVE
            / "official_hipporag_item_stage_parent"
        ),
    )

    def fake_default_config(call_project: Path) -> SimpleNamespace:
        assert call_project == project
        return runtime_config

    def fake_preflight(config: SimpleNamespace) -> dict[str, Any]:
        assert config is runtime_config
        assert not config.hippo_stage_parent_root.exists()
        return {
            "schema": local_runtime.PREFLIGHT_SCHEMA,
            "version": local_runtime.VERSION,
            "model_inference_calls": 0,
            "benchmark_source_or_private_pack_reads": 0,
            "external_network_calls": 0,
        }

    monkeypatch.setattr(
        local_runtime, "default_formal_runtime_config", fake_default_config
    )
    monkeypatch.setattr(
        local_runtime, "preflight_formal_runtime_config", fake_preflight
    )

    public = {
        "private_assignment_sha256": prerequisites.design[
            "base_acquisition_custody"
        ]["same_private_assignment_sha256"],
        "public_receipt_sha256": prerequisites.design[
            "base_acquisition_custody"
        ]["same_public_receipt_sha256"],
    }

    def fake_verify_acquisition_state(**kwargs: Any) -> dict[str, Any]:
        original_counts["verify"] += 1
        assert kwargs["acquisition_root"] == (
            project
            / subject.CANONICAL_FORMAL_ROOT_RELATIVE
            / formal_controller.ACQUISITION_DIRECTORY
        )
        assert kwargs["enforce_formal_design_identity"] is True
        return dict(public)

    monkeypatch.setattr(
        acquisition, "verify_acquisition_state", fake_verify_acquisition_state
    )

    def fake_inner(**kwargs: Any) -> dict[str, Any]:
        original_counts["inner"] += 1
        assert original_counts["inner"] == 1
        assert scheduler.HIPPORAG_WORKER_CAP == subject.RECOVERY_HIPPORAG_WORKER_CAP
        controller_root = kwargs["controller_root"]
        acquisition_root = kwargs["acquisition_root"]
        call_project = kwargs["project"]
        qualification = source_qualification.build_formal_qualification(
            call_project
        )
        formal_controller._persist_typed_artifact(
            path=controller_root / formal_controller.QUALIFICATION_FILENAME,
            payload=qualification,
            schema=source_qualification.SCHEMA,
            field="qualification_sha256",
            expected_sha256=qualification["qualification_sha256"],
        )
        observed_public = acquisition.acquire_once(
            archive_path=call_project / formal_controller.ARCHIVE_RELATIVE,
            prompt_sidecar_path=(
                call_project / formal_controller.PROMPT_SIDECAR_RELATIVE
            ),
            qualification_receipt_path=(
                controller_root / formal_controller.QUALIFICATION_FILENAME
            ),
            design_path=call_project / subject.BASE_DESIGN_RELATIVE,
            implementation_freeze_path=(
                call_project / subject.BASE_IMPLEMENTATION_FREEZE_RELATIVE
            ),
            project_root=call_project,
            acquisition_root=acquisition_root,
            selection_secret=None,
            enforce_formal_design_identity=True,
        )
        assert observed_public == public
        assert not (
            acquisition_root / "views" / "M_search.private.json"
        ).exists()
        if fail_inner:
            raise RuntimeError("synthetic guarded replay failure")
        claims = {
            "A_hold_real_domain_primary_passed": False,
            "A_hold_RAW_block_passed": False,
            "evaluator_promoted": False,
            "M_L5_passed": None,
            "cross_relation_stability_passed": None,
            "RAW_advantage_overcome": None,
            "total_goal_evidence_passed": False,
        }
        body = {
            "schema": f"{formal_controller.VERSION}_terminal_result",
            "version": formal_controller.VERSION,
            "status": "complete_nonpromotion_M_search_unopened",
            "claims": claims,
            "M_search_label_free_view_sha256": None,
            "M_search_schedule_receipt_sha256": None,
            "M_search_label_capability_sha256": None,
            "M_search_label_pack_sha256": None,
            "M_search_score_receipt_sha256": None,
            "M_search_score_receipt": None,
            "M_search_opened_without_promotion": False,
        }
        inner = _self_hashed(
            body,
            "terminal_result_sha256",
            canonical=formal_controller.stable_hash,
        )
        formal_controller._persist_typed_artifact(
            path=controller_root / formal_controller.RESULT_FILENAME,
            payload=inner,
            schema=f"{formal_controller.VERSION}_terminal_result",
            field="terminal_result_sha256",
            expected_sha256=inner["terminal_result_sha256"],
        )
        return inner

    monkeypatch.setattr(
        formal_controller, "_run_started_lifecycle", fake_inner
    )
    return (
        original_counts,
        forbidden_original_qualification,
        forbidden_original_acquire,
    )


def _prepare_synthetic_run(
    tmp_path: Path,
) -> tuple[Path, dict[str, bytes], subject.RecoveryPrerequisites]:
    project = tmp_path / "project"
    project.mkdir()
    clone_bytes, qualification = _build_interrupted_root(project)
    _write_marker_binding_files(project)
    prerequisites = _synthetic_prerequisites(
        project=project, qualification=qualification
    )
    return project, clone_bytes, prerequisites


def test_guarded_replay_uses_no_original_qualifier_or_acquire_and_restores_symbols(
    private_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, clone_bytes, prerequisites = _prepare_synthetic_run(private_tmp_path)
    counts, original_qualification, original_acquire = _install_synthetic_run_mocks(
        monkeypatch=monkeypatch,
        project=project,
        prerequisites=prerequisites,
        fail_inner=False,
    )

    result = subject.run_crash_recovery(project_root=project)

    assert result["status"] == "complete_crash_replay_nonpromotion_M_search_unopened"
    assert result["pristine_v1_completion_claim_allowed"] is False
    assert result["M_search_opened"] is False
    assert result["M_search_opened_without_promotion"] is False
    assert result["qualification_original_body_call_count"] == 0
    assert result["acquire_once_original_body_call_count"] == 0
    assert result["substituted_symbols_and_worker_cap_restored"] is True
    assert counts == {
        "qualification": 0,
        "acquire": 0,
        "verify": 1,
        "inner": 1,
    }
    assert source_qualification.build_formal_qualification is original_qualification
    assert acquisition.acquire_once is original_acquire
    assert scheduler.HIPPORAG_WORKER_CAP == subject.BASE_HIPPORAG_WORKER_CAP

    archived_stage = (
        project
        / subject.ARCHIVE_DESTINATION_RELATIVE
        / "official_hipporag_item_stage_parent"
    )
    assert archived_stage.exists()
    new_root = project / subject.CANONICAL_FORMAL_ROOT_RELATIVE
    for relative, expected in clone_bytes.items():
        assert (
            new_root / formal_controller.ACQUISITION_DIRECTORY / relative
        ).read_bytes() == expected
    assert not (
        new_root
        / formal_controller.ACQUISITION_DIRECTORY
        / "views"
        / "M_search.private.json"
    ).exists()
    controller = project / subject.RECOVERY_ROOT_RELATIVE / "controller"
    assert (controller / formal_controller.RESULT_FILENAME).exists()
    assert (controller / subject.RESULT_FILENAME).exists()
    assert not (controller / subject.FAILURE_FILENAME).exists()

    with pytest.raises(subject.EraserEvidenceInferenceCrashRecoveryError):
        subject.run_crash_recovery(project_root=project)


def test_failure_after_v2_marker_is_terminal_and_always_restores_substitutions(
    private_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _clone_bytes, prerequisites = _prepare_synthetic_run(private_tmp_path)
    counts, original_qualification, original_acquire = _install_synthetic_run_mocks(
        monkeypatch=monkeypatch,
        project=project,
        prerequisites=prerequisites,
        fail_inner=True,
    )

    with pytest.raises(subject.EraserEvidenceInferenceCrashRecoveryError):
        subject.run_crash_recovery(project_root=project)

    assert counts == {
        "qualification": 0,
        "acquire": 0,
        "verify": 1,
        "inner": 1,
    }
    assert source_qualification.build_formal_qualification is original_qualification
    assert acquisition.acquire_once is original_acquire
    assert scheduler.HIPPORAG_WORKER_CAP == subject.BASE_HIPPORAG_WORKER_CAP
    controller = project / subject.RECOVERY_ROOT_RELATIVE / "controller"
    assert (controller / subject.MARKER_FILENAME).exists()
    failure_path = controller / subject.FAILURE_FILENAME
    assert failure_path.exists()
    failure = json.loads(failure_path.read_text(encoding="ascii"))
    assert failure["status"] == "terminal_crash_recovery_failure_no_further_attempt"
    assert failure["failed_stage"] == "guarded_frozen_lifecycle_replay"
    assert failure["exception_message_persisted"] is False
    assert not (controller / subject.RESULT_FILENAME).exists()

    with pytest.raises(subject.EraserEvidenceInferenceCrashRecoveryError):
        subject.run_crash_recovery(project_root=project)
