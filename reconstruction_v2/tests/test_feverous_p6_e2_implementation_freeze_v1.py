from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from assumption_agent.benchmarks import (
    feverous_p6_e2_implementation_freeze_v1 as freeze,
)


QUALIFICATION_SHA = "1" * 64
ROLLOVER_SHA = "3" * 64
TRAIN_LOADER_QUALIFICATION_SHA = "4" * 64
PREFLIGHT_SHA = "2" * 64
TEST_RECEIPT = {"status": "passed", "test_count": 7, "suite": "synthetic"}


def _git(project: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _synthetic_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qualification_state: dict[str, str] | None = None,
) -> Path:
    project = tmp_path / "project"
    (project / "roles").mkdir(parents=True)
    (project / "manifests").mkdir()
    (project / "roles" / "role.py").write_text("VALUE = 1\n", encoding="utf-8")
    design_body = {"schema": "synthetic_design_v1", "status": "frozen"}
    design_sha = freeze.stable_hash(design_body)
    (project / "manifests" / "design.json").write_text(
        json.dumps({**design_body, "design_sha256": design_sha}) + "\n",
        encoding="utf-8",
    )
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "synthetic@example.invalid")
    _git(project, "config", "user.name", "Synthetic Freeze Test")
    _git(project, "add", "roles/role.py", "manifests/design.json")
    _git(project, "commit", "-q", "-m", "synthetic implementation")

    monkeypatch.setattr(freeze, "BOUND_PATHS", {"role": "roles/role.py"})
    monkeypatch.setattr(
        freeze, "MANIFEST_RELATIVE", Path("manifests/freeze.json")
    )
    monkeypatch.setattr(
        freeze, "DESIGN_RELATIVE", Path("manifests/design.json")
    )
    monkeypatch.setattr(freeze, "DESIGN_SHA256", design_sha)
    state = qualification_state or {"sha": QUALIFICATION_SHA}
    monkeypatch.setattr(
        freeze, "_qualification_sha256", lambda _project: state["sha"]
    )
    monkeypatch.setattr(
        freeze,
        "_source_epoch_rollover_bindings",
        lambda _project, *, require_successor_absent: (
            ROLLOVER_SHA,
            TRAIN_LOADER_QUALIFICATION_SHA,
        ),
    )
    return project


def _synthetic_subdirectory_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repository = tmp_path / "repository"
    project = repository / "reconstruction_v2"
    (project / "roles").mkdir(parents=True)
    (project / "manifests").mkdir()
    (project / "roles" / "role.py").write_text("VALUE = 1\n", encoding="utf-8")
    design_body = {"schema": "synthetic_design_v1", "status": "frozen"}
    design_sha = freeze.stable_hash(design_body)
    (project / "manifests" / "design.json").write_text(
        json.dumps({**design_body, "design_sha256": design_sha}) + "\n",
        encoding="utf-8",
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "synthetic@example.invalid")
    _git(repository, "config", "user.name", "Synthetic Freeze Test")
    _git(repository, "add", "reconstruction_v2/roles/role.py")
    _git(repository, "add", "reconstruction_v2/manifests/design.json")
    _git(repository, "commit", "-q", "-m", "synthetic nested implementation")

    monkeypatch.setattr(freeze, "BOUND_PATHS", {"role": "roles/role.py"})
    monkeypatch.setattr(
        freeze, "MANIFEST_RELATIVE", Path("manifests/freeze.json")
    )
    monkeypatch.setattr(
        freeze, "DESIGN_RELATIVE", Path("manifests/design.json")
    )
    monkeypatch.setattr(freeze, "DESIGN_SHA256", design_sha)
    monkeypatch.setattr(
        freeze, "_qualification_sha256", lambda _project: QUALIFICATION_SHA
    )
    monkeypatch.setattr(
        freeze,
        "_source_epoch_rollover_bindings",
        lambda _project, *, require_successor_absent: (
            ROLLOVER_SHA,
            TRAIN_LOADER_QUALIFICATION_SHA,
        ),
    )
    return project


def _commit_freeze(
    project: Path,
    *,
    mutate: object | None = None,
) -> dict[str, object]:
    manifest = freeze.form_implementation_freeze(
        project=project,
        test_receipt=TEST_RECEIPT,
        runtime_preflight_sha256=PREFLIGHT_SHA,
    )
    if callable(mutate):
        mutate(manifest)
    path = project / freeze.MANIFEST_RELATIVE
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    _git(project, "add", freeze.MANIFEST_RELATIVE.as_posix())
    _git(project, "commit", "-q", "-m", "synthetic implementation freeze")
    return manifest


def _replace_manifest_body_field(
    manifest: dict[str, object], field: str, value: object
) -> None:
    manifest[field] = value
    body = dict(manifest)
    body.pop("implementation_freeze_sha256")
    manifest["implementation_freeze_sha256"] = freeze.stable_hash(body)


def test_production_registry_closes_over_verifier_tests_and_parallel_evidence() -> None:
    expected = {
        "wikipedia_source_qualification": (
            "assumption_agent/benchmarks/"
            "feverous_wikipedia_source_qualification_v1.py"
        ),
        "implementation_freeze_verifier": (
            "assumption_agent/benchmarks/"
            "feverous_p6_e2_implementation_freeze_v1.py"
        ),
        "test_implementation_freeze": (
            "tests/test_feverous_p6_e2_implementation_freeze_v1.py"
        ),
        "formal_acquisition_v2": (
            "assumption_agent/benchmarks/"
            "feverous_p6_e2_formal_acquisition_v2.py"
        ),
        "formal_acquisition_entrypoint_v2": (
            "assumption_agent/benchmarks/"
            "feverous_p6_e2_formal_acquisition_entrypoint_v2.py"
        ),
        "test_formal_acquisition_entrypoint_v2": (
            "tests/test_feverous_p6_e2_formal_acquisition_entrypoint_v2.py"
        ),
        "source_epoch_rollover_v2": (
            "assumption_agent/benchmarks/"
            "feverous_p6_e2_source_epoch_rollover_v2.py"
        ),
        "source_epoch_rollover_v2_manifest": (
            "manifests/feverous_p6_e2_source_epoch_rollover_v2.json"
        ),
        "train_loader_qualification_v2": (
            "assumption_agent/benchmarks/"
            "feverous_p6_e2_train_loader_qualification_v2.py"
        ),
        "train_loader_qualification_v2_manifest": (
            "manifests/feverous_p6_e2_train_loader_qualification_v2.json"
        ),
        "identity_parallel_performance_diagnostic": (
            "assumption_agent/benchmarks/"
            "feverous_p6_e2_identity_parallel_performance_diagnostic_v1.py"
        ),
        "test_identity_parallel_performance_diagnostic": (
            "tests/test_feverous_p6_e2_identity_parallel_performance_diagnostic_v1.py"
        ),
        "identity_parallel_performance_diagnostic_receipt": (
            "manifests/"
            "feverous_p6_e2_identity_parallel_performance_diagnostic_v1.json"
        ),
        "parallel_identity_selection": (
            "assumption_agent/benchmarks/"
            "feverous_p6_e2_parallel_identity_selection_v1.py"
        ),
        "test_parallel_identity_selection": (
            "tests/test_feverous_p6_e2_parallel_identity_selection_v1.py"
        ),
    }
    assert all(freeze.BOUND_PATHS.get(role) == path for role, path in expected.items())


def test_forms_and_verifies_clean_committed_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _synthetic_project(tmp_path, monkeypatch)
    formed = _commit_freeze(project)
    verified = freeze.verify_committed_implementation_freeze(project)
    assert verified == formed
    assert verified["identity_compiler_qualification_sha256"] == QUALIFICATION_SHA
    assert verified["source_epoch_rollover_sha256"] == ROLLOVER_SHA
    assert (
        verified["train_loader_qualification_sha256"]
        == TRAIN_LOADER_QUALIFICATION_SHA
    )
    assert verified["predecessor_v1_terminal_failure_bound"] is True
    assert verified["successor_v2_selection_secret_generated"] is False
    assert verified["successor_v2_cohort_acquisition_started"] is False


def test_subdirectory_project_forms_commits_and_verifies_in_repo_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_project = _synthetic_project(tmp_path / "root", monkeypatch)
    root_formed = _commit_freeze(root_project)
    assert freeze.verify_committed_implementation_freeze(root_project) == root_formed

    project = _synthetic_subdirectory_project(tmp_path / "nested", monkeypatch)
    formed = _commit_freeze(project)
    verified = freeze.verify_committed_implementation_freeze(project)

    assert verified == formed
    assert verified["bound_files"] == root_formed["bound_files"]
    assert verified["bound_file_set_sha256"] == root_formed[
        "bound_file_set_sha256"
    ]


def test_rejects_dirty_untracked_bound_path_and_uncommitted_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirty = _synthetic_project(tmp_path / "dirty", monkeypatch)
    (dirty / "roles" / "role.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(
        freeze.FeverousImplementationFreezeError,
        match="paths are dirty",
    ):
        freeze.form_implementation_freeze(
            project=dirty,
            test_receipt=TEST_RECEIPT,
            runtime_preflight_sha256=PREFLIGHT_SHA,
        )

    monkeypatch.undo()
    untracked = _synthetic_project(tmp_path / "untracked", monkeypatch)
    (untracked / "roles" / "new.py").write_text("VALUE = 9\n", encoding="utf-8")
    monkeypatch.setattr(
        freeze,
        "BOUND_PATHS",
        {"role": "roles/role.py", "untracked": "roles/new.py"},
    )
    with pytest.raises(
        freeze.FeverousImplementationFreezeError,
        match="paths are dirty",
    ):
        freeze.form_implementation_freeze(
            project=untracked,
            test_receipt=TEST_RECEIPT,
            runtime_preflight_sha256=PREFLIGHT_SHA,
        )

    monkeypatch.undo()
    manifest_dirty = _synthetic_project(tmp_path / "manifest", monkeypatch)
    _commit_freeze(manifest_dirty)
    path = manifest_dirty / freeze.MANIFEST_RELATIVE
    path.write_text(path.read_text(encoding="ascii") + "\n", encoding="ascii")
    with pytest.raises(
        freeze.FeverousImplementationFreezeError,
        match="formal implementation paths are dirty",
    ):
        freeze.verify_committed_implementation_freeze(manifest_dirty)


def test_rejects_clean_committed_blob_drift_after_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _synthetic_project(tmp_path, monkeypatch)
    _commit_freeze(project)
    (project / "roles" / "role.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(project, "add", "roles/role.py")
    _git(project, "commit", "-q", "-m", "post-freeze blob drift")
    with pytest.raises(
        freeze.FeverousImplementationFreezeError,
        match="differs from implementation commit",
    ):
        freeze.verify_committed_implementation_freeze(project)


def test_rejects_qualification_drift_even_when_git_bytes_are_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"sha": QUALIFICATION_SHA}
    project = _synthetic_project(
        tmp_path, monkeypatch, qualification_state=state
    )
    _commit_freeze(project)
    state["sha"] = "9" * 64
    with pytest.raises(
        freeze.FeverousImplementationFreezeError,
        match="semantics drifted",
    ):
        freeze.verify_committed_implementation_freeze(project)


def test_rejects_manifest_implementation_commit_that_is_not_an_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _synthetic_project(tmp_path, monkeypatch)
    main_branch = _git(project, "rev-parse", "--abbrev-ref", "HEAD")
    _git(project, "switch", "-q", "-c", "sibling")
    _git(project, "commit", "-q", "--allow-empty", "-m", "sibling commit")
    sibling = _git(project, "rev-parse", "HEAD")
    _git(project, "switch", "-q", main_branch)

    def point_to_sibling(manifest: dict[str, object]) -> None:
        manifest["implementation_git_commit"] = sibling
        body = dict(manifest)
        body.pop("implementation_freeze_sha256")
        manifest["implementation_freeze_sha256"] = freeze.stable_hash(body)

    _commit_freeze(project, mutate=point_to_sibling)
    with pytest.raises(
        freeze.FeverousImplementationFreezeError,
        match="not an ancestor",
    ):
        freeze.verify_committed_implementation_freeze(project)


@pytest.mark.parametrize(
    "receipt",
    (
        {"status": "failed", "test_count": 7},
        {"status": "passed", "test_count": 0},
        {"status": "passed", "test_count": -1},
        {"status": "passed", "test_count": True},
        {"status": "passed", "test_count": 1.0},
    ),
)
def test_verify_rejects_rehashed_invalid_test_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: dict[str, object],
) -> None:
    project = _synthetic_project(tmp_path, monkeypatch)

    def mutate(manifest: dict[str, object]) -> None:
        _replace_manifest_body_field(manifest, "test_receipt", receipt)

    _commit_freeze(project, mutate=mutate)
    with pytest.raises(
        freeze.FeverousImplementationFreezeError,
        match="semantics drifted",
    ):
        freeze.verify_committed_implementation_freeze(project)


@pytest.mark.parametrize(
    "preflight_sha",
    (None, "", "2" * 63, "2" * 65, "G" * 64, 2),
)
def test_verify_rejects_rehashed_invalid_runtime_preflight_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preflight_sha: object,
) -> None:
    project = _synthetic_project(tmp_path, monkeypatch)

    def mutate(manifest: dict[str, object]) -> None:
        _replace_manifest_body_field(
            manifest, "runtime_preflight_sha256", preflight_sha
        )

    _commit_freeze(project, mutate=mutate)
    with pytest.raises(
        freeze.FeverousImplementationFreezeError,
        match="semantics drifted",
    ):
        freeze.verify_committed_implementation_freeze(project)
