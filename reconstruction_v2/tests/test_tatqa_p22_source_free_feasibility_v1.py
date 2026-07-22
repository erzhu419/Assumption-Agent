from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Any

import pytest

from assumption_agent.benchmarks import tatqa_p22_source_free_feasibility_v1 as feasibility


REAL_OUTER_UNIT_RECEIPT = feasibility._outer_unit_receipt


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()


@pytest.fixture
def native_tmp_path() -> Path:
    path = Path(tempfile.mkdtemp(prefix="tatqa-p22-feasibility-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def fixed_host_layout(
    native_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = native_tmp_path / "fixed-host-root"
    project = host / "runtime" / "reconstruction_v2"
    attempt = host / "attempt"
    monkeypatch.setattr(feasibility, "EXPECTED_HOST_ROOT", host)
    monkeypatch.setattr(feasibility, "EXPECTED_PROJECT_ROOT", project)
    monkeypatch.setattr(feasibility, "EXPECTED_FEASIBILITY_ROOT", attempt)
    monkeypatch.setattr(feasibility, "EXPECTED_WORK_ROOT", attempt / "work")
    monkeypatch.setattr(
        feasibility, "EXPECTED_PUBLIC_CANARY_OUTPUT", attempt / "public-canary.json"
    )
    local_git = Path("/usr/bin/git")
    local_git_raw = local_git.read_bytes()
    local_git_version = subprocess.run(
        [str(local_git), "--version"], check=True, capture_output=True
    ).stdout
    monkeypatch.setattr(feasibility, "GIT_EXECUTABLE", local_git)
    monkeypatch.setattr(
        feasibility,
        "GIT_EXECUTABLE_SHA256",
        hashlib.sha256(local_git_raw).hexdigest(),
    )
    monkeypatch.setattr(feasibility, "GIT_VERSION_STDOUT", local_git_version)
    outer_body = {
        "schema": f"{feasibility.VERSION}_outer_unit_v1",
        "unit_name": feasibility.EXPECTED_OUTER_UNIT,
        "cgroup_file_sha256": "1" * 64,
        "matched_line_sha256": "2" * 64,
    }
    outer = {**outer_body, "self_sha256": _semantic_hash(outer_body)}
    monkeypatch.setattr(feasibility, "_outer_unit_receipt", lambda: outer)
    loaded_body = {
        "schema": f"{feasibility.VERSION}_loaded_project_modules_v1",
        "module_count": 1,
        "modules": {
            "synthetic": {
                "relative_path": next(iter(feasibility.REQUIRED_IMPLEMENTATION_PATHS)),
                "file_sha256": "3" * 64,
            }
        },
    }
    loaded = {**loaded_body, "self_sha256": _semantic_hash(loaded_body)}
    monkeypatch.setattr(
        feasibility, "_loaded_project_module_binding", lambda _project: loaded
    )


def _git_environment() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-07-23T00:00:00+08:00",
        "GIT_AUTHOR_EMAIL": "p22@example.invalid",
        "GIT_AUTHOR_NAME": "P22 Source-Free Snapshot",
        "GIT_COMMITTER_DATE": "2026-07-23T00:00:00+08:00",
        "GIT_COMMITTER_EMAIL": "p22@example.invalid",
        "GIT_COMMITTER_NAME": "P22 Source-Free Snapshot",
    }


def _git(project: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=project,
        check=True,
        capture_output=True,
        env=_git_environment(),
    )
    return completed.stdout


def _snapshot_commit(project: Path, message: str) -> str:
    _git(project, "add", "--all")
    _git(project, "commit", "--quiet", "-m", message)
    return _git(project, "rev-parse", "HEAD").decode("ascii").strip()


def _phase(phase: str) -> dict[str, object]:
    body = {"schema": "p21-phase", "phase": phase}
    return {**body, "self_sha256": _semantic_hash(body)}


def _arguments(tmp_path: Path) -> dict[str, object]:
    project = feasibility.EXPECTED_PROJECT_ROOT
    project.mkdir(parents=True)
    for relative in sorted(feasibility.REQUIRED_SNAPSHOT_PATHS):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in feasibility.SOURCE_ISOLATION_SENTINEL_PATHS:
            path.write_bytes(feasibility.SOURCE_ISOLATION_SENTINEL_BYTES)
        elif relative.endswith("tatqa_p21_composite_runtime_fingerprint_v1.json"):
            path.write_bytes(b"placeholder\n")
        else:
            path.write_text(f"synthetic snapshot member: {relative}\n", "utf-8")
    _git(project, "init", "--quiet")
    commit = _snapshot_commit(project, "Synthetic P22 source-free snapshot")
    return {
        "project_root": project,
        "typed_runtime_python": tmp_path / "typed" / "bin" / "python",
        "hippo_runtime_python": tmp_path / "hippo" / "bin" / "python",
        "qwen_model": tmp_path / "assets" / "qwen",
        "minilm_asset_manifest": tmp_path / "assets" / "minilm.json",
        "minilm_model": tmp_path / "assets" / "minilm",
        "hippo_llm_model": tmp_path / "assets" / "hippo-llm",
        "hippo_embedding_model": tmp_path / "assets" / "hippo-embedding",
        "hipporag_source": tmp_path / "assets" / "hipporag",
        "hippo_attestation": project
        / "manifests/tatqa_p19_hipporag_runtime_attestation_v1.json",
        "p21_runtime_fingerprint": project
        / "manifests/tatqa_p21_composite_runtime_fingerprint_v1.json",
        "diagnostic_snapshot_commit": commit,
    }


def _commit_changed_snapshot(arguments: dict[str, object], message: str) -> None:
    project = Path(arguments["project_root"])
    arguments["diagnostic_snapshot_commit"] = _snapshot_commit(project, message)


def _portable_canary() -> dict[str, object]:
    return {
        "all_values_finite": True,
        "at_least_two_distinct_vectors": True,
        "embedding_dtype": "float32",
        "embedding_shape": [256, 384],
        "external_network_calls": 0,
        "formal_QASPER_source_or_rows_accessed": False,
        "formal_TAT_QA_source_or_rows_accessed": False,
        "maximum_observed_row_l2_norm_error": 1e-7,
        "observed_output_hashes": {
            "compared_to_expected_or_allowlist": False,
            "float32_little_endian_c_order_sha256": "4" * 64,
            "normative_acceptance": False,
            "quantized_embedding_matrix_sha256": "5" * 64,
        },
        "per_row_l2_norm_maximum_error": 1e-5,
        "public_text_vector_identity_exact": True,
        "public_text_vector_sha256": (
            "c122a1e09d2f84ad00a4c0b30abb979e13facdb8c1a5b3b15cb952b51b173249"
        ),
        "qasper_rows_or_archives_accessed_by_canary": False,
        "repeat_byte_exact": True,
        "repeat_count": 2,
        "repeat_elementwise_exact": True,
        "schema": "qasper_minilm_portable_startup_canary_v2",
        "sentence_count": 256,
        "status": "passed_portable_public_synthetic_structural_canary",
        "tatqa_rows_or_archives_accessed_by_canary": False,
    }


def _portable_runtime(asset_manifest: Path, model_root: Path) -> dict[str, object]:
    return {
        "asset_file_sha256": feasibility.frozen_minilm.ASSET_FILE_SHA256,
        "asset_manifest_path": str(asset_manifest.resolve(strict=True)),
        "asset_sha256": feasibility.frozen_minilm.ASSET_SELF_SHA256,
        "embedding_dimension": feasibility.frozen_minilm.EMBEDDING_DIMENSION,
        "maximum_sequence_length": feasibility.frozen_minilm.MAXIMUM_SEQUENCE_LENGTH,
        "model_root": str(model_root.resolve(strict=True)),
        "model_tree_sha256": feasibility.frozen_minilm.MODEL_TREE_SHA256,
        "runtime_versions": dict(feasibility.frozen_minilm.EXPECTED_RUNTIME_VERSIONS),
        "status": "verified_offline_immutable_qasper_minilm_runtime",
        "weights_sha256": feasibility.frozen_minilm.WEIGHTS_SHA256,
    }


def _install_common_runtime_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phases: dict[str, dict[str, object]],
    inventory: dict[str, object],
    fingerprint: dict[str, object],
) -> None:
    monkeypatch.setattr(
        feasibility.formal_runtime,
        "user_systemd_launcher_phase_receipt",
        lambda *, phase: phases[phase],
    )
    monkeypatch.setattr(
        feasibility.formal_runtime,
        "runtime_inventory_snapshot",
        lambda **_kwargs: inventory,
    )
    monkeypatch.setattr(
        feasibility.formal_runtime,
        "systemd_network_preflight",
        lambda: {"status": "passed"},
    )
    monkeypatch.setattr(
        feasibility.formal_runtime,
        "verify_runtime_fingerprint",
        lambda paths: fingerprint,
    )


def test_success_uses_real_git_snapshot_fixed_paths_and_source_free_workers(
    native_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(native_tmp_path)
    observed: dict[str, Any] = {}
    phase_order: list[str] = []
    phases = {
        name: _phase(name)
        for name in ("entry", "post_runtime_inventory", "post_minilm")
    }
    inventory = {"split_runtime": "P21-exact"}
    fingerprint_body = {
        "schema": "tatqa_p21_composite_runtime_fingerprint_v1",
        "runtime_inventory": inventory,
        "safe_user_systemd_launch_envelope": {
            "phase_receipts": {
                "entry": phases["entry"],
                "post_runtime_inventory": phases["post_runtime_inventory"],
            }
        },
        "systemd_network_preflight": {"status": "passed"},
    }
    fingerprint = {**fingerprint_body, "self_sha256": _semantic_hash(fingerprint_body)}
    Path(arguments["p21_runtime_fingerprint"]).write_bytes(_canonical(fingerprint))
    _commit_changed_snapshot(arguments, "Bind synthetic P21 fingerprint")
    Path(arguments["minilm_asset_manifest"]).parent.mkdir(parents=True)
    Path(arguments["minilm_asset_manifest"]).write_bytes(b"synthetic manifest\n")
    Path(arguments["minilm_model"]).mkdir(parents=True)

    class FakePortable:
        def __init__(self, **kwargs: object) -> None:
            observed["portable_args"] = kwargs
            self.runtime_receipt = _portable_runtime(
                Path(kwargs["asset_manifest_path"]), Path(kwargs["model_root"])
            )
            self.canary_receipt = _portable_canary()

    class FakeTyped:
        def __init__(self, paths: object) -> None:
            self.paths = paths
            observed["typed_paths"] = paths

        def abort_all_workers(self) -> tuple[dict[str, int], ...]:
            observed["typed_abort"] = True
            return ({"closed": 1}, {"closed": 2})

        def verify_all_workers_closed(self) -> tuple[dict[str, int], ...]:
            observed["typed_verify"] = True
            return ({"closed": 1}, {"closed": 2})

    class FakeHippo:
        def __init__(self, paths: object) -> None:
            self.paths = paths
            observed["hippo_paths"] = paths

        def abort_all_workers(self) -> tuple[dict[str, int], ...]:
            observed["hippo_abort"] = True
            return ({"closed": 3},)

        def verify_all_workers_closed(self) -> tuple[dict[str, int], ...]:
            observed["hippo_verify"] = True
            return ({"closed": 3},)

    def fake_canary(**kwargs: object) -> dict[str, object]:
        observed["canary_args"] = kwargs
        typed = kwargs["typed_plan_runner"]
        hippo = kwargs["hippo_runner"]
        feasibility.formal_runtime._worker_inaccessible_paths(typed.paths)
        feasibility.formal_runtime._worker_inaccessible_paths(typed.paths)
        feasibility.formal_runtime._worker_inaccessible_paths(hippo.paths)
        body = {
            "schema": "tatqa_p21_public_synthetic_production_canary_v1",
            "hippo_canary_ran": True,
            "typed_plan_worker_receipt_source": "capability_receipt_snapshot",
            "minilm_worker_receipt_source": "explicit_formal_receipt",
            "hippo_worker_receipt_source": "capability_receipt_snapshot",
            "minilm_worker_receipt_snapshot": kwargs["minilm_worker_receipt"],
            "formal_source_opened": False,
            "external_network_calls": 0,
            "api_or_online_evaluator_calls": 0,
        }
        receipt = {**body, "self_sha256": _semantic_hash(body)}
        Path(kwargs["output_path"]).write_bytes(_canonical(receipt))
        return receipt

    _install_common_runtime_mocks(
        monkeypatch, phases=phases, inventory=inventory, fingerprint=fingerprint
    )
    monkeypatch.setattr(
        feasibility.formal_runtime,
        "user_systemd_launcher_phase_receipt",
        lambda *, phase: phase_order.append(phase) or phases[phase],
    )
    monkeypatch.setattr(feasibility, "PortableOfflineMiniLMEncoder", FakePortable)
    monkeypatch.setattr(
        feasibility.formal_runtime, "SystemdTypedPlanBatchRunner", FakeTyped
    )
    monkeypatch.setattr(feasibility.formal_runtime, "SystemdHippoByteRunner", FakeHippo)
    monkeypatch.setattr(feasibility.canary, "run_public_production_canary", fake_canary)

    normalizer_body = {
        "schema": "synthetic-post-minilm-normalizer-v1",
        "status": "normalized",
    }
    normalizer_receipt = {
        **normalizer_body,
        "self_sha256": _semantic_hash(normalizer_body),
    }
    normalizer_binding = {
        "schema": "synthetic-normalizer-binding-v1",
        "self_sha256": "6" * 64,
    }

    def normalize() -> dict[str, object]:
        phase_order.append("normalize")
        return normalizer_receipt

    monkeypatch.setattr(
        feasibility,
        "_post_minilm_normalizer_binding",
        lambda _project, _normalizer: normalizer_binding,
    )

    terminal = feasibility.run_source_free_feasibility(
        **arguments, _post_minilm_environment_normalizer=normalize
    )

    portable = observed["canary_args"]["minilm_worker_receipt"]
    assert portable["schema"] == feasibility.PORTABLE_CAPABILITY_SCHEMA
    assert portable["portable_startup_canary_receipt"] == _portable_canary()
    assert terminal["status"] == "passed_source_free_feasibility_only"
    assert terminal["P21_qualification_claimed"] is False
    assert terminal["efficacy_claimed"] is False
    assert terminal["formal_TAT_QA_source_opened"] is False
    assert terminal["external_network_calls"] == 0
    assert terminal["api_or_online_evaluator_calls"] == 0
    assert terminal["retry_replay_resample_provider_switch"] == 0
    assert terminal["diagnostic_snapshot_binding"]["diagnostic_snapshot_commit"] == (
        arguments["diagnostic_snapshot_commit"]
    )
    assert terminal["diagnostic_snapshot_binding"]["minimal_tracked_registry_exact"]
    assert terminal["source_isolation"]["formal_TAT_QA_source_or_rows_present"] is False
    assert terminal["source_free_worker_isolation"]["launch_count"] == 3
    assert terminal["source_free_worker_isolation"]["hook_restored_exact"] is True
    assert terminal["post_minilm_environment_normalization_receipt"] == (
        normalizer_receipt
    )
    assert terminal["post_minilm_environment_normalizer_binding"] == (
        normalizer_binding
    )
    assert phase_order == ["entry", "post_runtime_inventory", "normalize", "post_minilm"]
    assert observed["typed_abort"] is observed["typed_verify"] is True
    assert observed["hippo_abort"] is observed["hippo_verify"] is True
    root = feasibility.EXPECTED_FEASIBILITY_ROOT
    assert (root / feasibility.MARKER_FILENAME).is_file()
    success = root / feasibility.SUCCESS_FILENAME
    assert json.loads(success.read_text("ascii")) == terminal
    assert stat.S_IMODE(success.stat().st_mode) == 0o600
    assert not (root / feasibility.FAILURE_FILENAME).exists()


def test_failure_is_terminal_and_restores_worker_isolation(
    native_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(native_tmp_path)
    phases = {
        name: _phase(name)
        for name in ("entry", "post_runtime_inventory", "post_minilm")
    }
    inventory = {"runtime": "exact"}
    fingerprint = {
        "runtime_inventory": inventory,
        "safe_user_systemd_launch_envelope": {
            "phase_receipts": {
                "entry": phases["entry"],
                "post_runtime_inventory": phases["post_runtime_inventory"],
            }
        },
        "systemd_network_preflight": {"status": "passed"},
        "self_sha256": "a" * 64,
    }
    Path(arguments["p21_runtime_fingerprint"]).write_bytes(_canonical(fingerprint))
    _commit_changed_snapshot(arguments, "Bind failure fingerprint")
    Path(arguments["minilm_asset_manifest"]).parent.mkdir(parents=True)
    Path(arguments["minilm_asset_manifest"]).write_bytes(b"synthetic manifest\n")
    Path(arguments["minilm_model"]).mkdir(parents=True)
    original_hook = feasibility.formal_runtime._worker_inaccessible_paths
    closed: list[str] = []

    class FakeTyped:
        def __init__(self, paths: object) -> None:
            self.paths = paths

        def abort_all_workers(self) -> tuple[()]:
            closed.append("abort")
            return ()

        def verify_all_workers_closed(self) -> tuple[()]:
            closed.append("verify")
            return ()

    class FailingHippo:
        def __init__(self, _paths: object) -> None:
            raise RuntimeError("synthetic construction failure")

    class FakePortable:
        def __init__(self, **kwargs: object) -> None:
            self.runtime_receipt = _portable_runtime(
                Path(kwargs["asset_manifest_path"]), Path(kwargs["model_root"])
            )
            self.canary_receipt = _portable_canary()

    _install_common_runtime_mocks(
        monkeypatch, phases=phases, inventory=inventory, fingerprint=fingerprint
    )
    monkeypatch.setattr(feasibility, "PortableOfflineMiniLMEncoder", FakePortable)
    monkeypatch.setattr(
        feasibility.formal_runtime, "SystemdTypedPlanBatchRunner", FakeTyped
    )
    monkeypatch.setattr(
        feasibility.formal_runtime, "SystemdHippoByteRunner", FailingHippo
    )

    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="failed terminally",
    ):
        feasibility.run_source_free_feasibility(**arguments)

    assert closed == ["abort", "verify"]
    assert feasibility.formal_runtime._worker_inaccessible_paths is original_hook
    failure = json.loads(
        (feasibility.EXPECTED_FEASIBILITY_ROOT / feasibility.FAILURE_FILENAME).read_text(
            "ascii"
        )
    )
    assert failure["failure_stage"] == "p21_public_synthetic_production_path"
    assert failure["status"] == "terminal_no_retry_nonqualification_non_efficacy"
    assert failure["formal_TAT_QA_source_opened"] is False
    assert not (
        feasibility.EXPECTED_FEASIBILITY_ROOT / feasibility.SUCCESS_FILENAME
    ).exists()


@pytest.mark.parametrize("consumed", ("root", "work", "output"))
def test_fixed_attempt_paths_cannot_be_replaced_or_reused(
    native_tmp_path: Path,
    consumed: str,
) -> None:
    arguments = _arguments(native_tmp_path)
    target = {
        "root": feasibility.EXPECTED_FEASIBILITY_ROOT,
        "work": feasibility.EXPECTED_WORK_ROOT,
        "output": feasibility.EXPECTED_PUBLIC_CANARY_OUTPUT,
    }[consumed]
    if consumed in {"root", "work"}:
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True)
        target.write_bytes(b"preserve")

    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="already consumed",
    ):
        feasibility.run_source_free_feasibility(**arguments)


def test_fake_commit_and_dirty_or_extra_snapshot_are_rejected(
    native_tmp_path: Path,
) -> None:
    arguments = _arguments(native_tmp_path)
    real_commit = arguments["diagnostic_snapshot_commit"]
    arguments["diagnostic_snapshot_commit"] = "b" * 40
    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="Git verification failed",
    ):
        feasibility.run_source_free_feasibility(**arguments)
    arguments["diagnostic_snapshot_commit"] = real_commit
    member = Path(arguments["project_root"]) / next(
        iter(feasibility.REQUIRED_IMPLEMENTATION_PATHS)
    )
    member.write_text("dirty\n", "utf-8")
    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="worktree is not exactly clean",
    ):
        feasibility.run_source_free_feasibility(**arguments)


def test_ignored_file_injection_is_rejected_by_real_filesystem_closure(
    native_tmp_path: Path,
) -> None:
    arguments = _arguments(native_tmp_path)
    project = Path(arguments["project_root"])
    (project / ".git/info/exclude").write_text("sitecustomize.py\n", "utf-8")
    (project / "sitecustomize.py").write_text("raise SystemExit(99)\n", "utf-8")
    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="filesystem closure drifted",
    ):
        feasibility._diagnostic_snapshot_binding(
            project, str(arguments["diagnostic_snapshot_commit"])
        )


def test_git_replace_ref_cannot_replace_the_frozen_snapshot(
    native_tmp_path: Path,
) -> None:
    arguments = _arguments(native_tmp_path)
    project = Path(arguments["project_root"])
    original = str(arguments["diagnostic_snapshot_commit"])
    member = project / next(iter(feasibility.REQUIRED_IMPLEMENTATION_PATHS))
    member.write_text("replacement bytes\n", "utf-8")
    replacement = _snapshot_commit(project, "Replacement commit")
    _git(project, "replace", original, replacement)
    _git(project, "--no-replace-objects", "checkout", "--quiet", "--detach", original)
    binding = feasibility._diagnostic_snapshot_binding(project, original)
    assert binding["diagnostic_snapshot_commit"] == original


def test_source_isolation_rejects_any_payload_beside_sentinel(
    native_tmp_path: Path,
) -> None:
    arguments = _arguments(native_tmp_path)
    root = Path(arguments["project_root"]) / feasibility.SOURCE_ISOLATION_ROOTS[0]
    (root / "tatqa_dataset_train.json").write_text("[]\n", "utf-8")
    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="not an empty isolation sentinel",
    ):
        feasibility._source_isolation_receipt(Path(arguments["project_root"]))


@pytest.mark.parametrize(
    "field",
    (
        "formal_QASPER_source_or_rows_accessed",
        "qasper_rows_or_archives_accessed_by_canary",
        "maximum_observed_row_l2_norm_error",
        "public_text_vector_identity_exact",
        "public_text_vector_sha256",
    ),
)
def test_portable_receipt_validator_requires_complete_structural_contract(
    native_tmp_path: Path,
    field: str,
) -> None:
    manifest = native_tmp_path / "minilm.json"
    manifest.write_bytes(b"manifest\n")
    model = native_tmp_path / "model"
    model.mkdir()

    class Encoder:
        runtime_receipt = _portable_runtime(manifest, model)
        canary_receipt = _portable_canary()

    Encoder.canary_receipt = dict(Encoder.canary_receipt)
    del Encoder.canary_receipt[field]
    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="structural canary drifted",
    ):
        feasibility._portable_capability_receipt(
            Encoder(), expected_asset_manifest=manifest, expected_model_root=model
        )


@pytest.mark.parametrize(
    "field",
    (
        "asset_file_sha256",
        "asset_sha256",
        "embedding_dimension",
        "maximum_sequence_length",
        "model_tree_sha256",
        "runtime_versions",
        "status",
        "weights_sha256",
    ),
)
def test_portable_receipt_validator_requires_exact_runtime_contract(
    native_tmp_path: Path,
    field: str,
) -> None:
    manifest = native_tmp_path / "minilm.json"
    manifest.write_bytes(b"manifest\n")
    model = native_tmp_path / "model"
    model.mkdir()

    class Encoder:
        runtime_receipt = _portable_runtime(manifest, model)
        canary_receipt = _portable_canary()

    Encoder.runtime_receipt = dict(Encoder.runtime_receipt)
    del Encoder.runtime_receipt[field]
    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="immutable runtime receipt drifted",
    ):
        feasibility._portable_capability_receipt(
            Encoder(), expected_asset_manifest=manifest, expected_model_root=model
        )


def test_outer_unit_receipt_requires_exact_preregistered_service(
    native_tmp_path: Path,
) -> None:
    cgroup = native_tmp_path / "cgroup"
    cgroup.write_text(
        "0::/user.slice/user-1001.slice/user@1001.service/app.slice/"
        f"{feasibility.EXPECTED_OUTER_UNIT}\n",
        "utf-8",
    )
    receipt = REAL_OUTER_UNIT_RECEIPT(cgroup)
    assert receipt["unit_name"] == feasibility.EXPECTED_OUTER_UNIT
    cgroup.write_text("0::/wrong.service\n", "utf-8")
    with pytest.raises(
        feasibility.TatqaP22SourceFreeFeasibilityError,
        match="unit identity drifted",
    ):
        REAL_OUTER_UNIT_RECEIPT(cgroup)


def test_cli_has_no_attempt_path_or_formal_source_override() -> None:
    options = {
        action.dest for action in feasibility._parser()._actions if action.dest != "help"
    }
    assert options == {
        "project_root",
        "typed_runtime_python",
        "hippo_runtime_python",
        "qwen_model",
        "minilm_asset_manifest",
        "minilm_model",
        "hippo_llm_model",
        "hippo_embedding_model",
        "hipporag_source",
        "hippo_attestation",
        "p21_runtime_fingerprint",
        "diagnostic_snapshot_commit",
    }
    assert not any("source" in option and option != "hipporag_source" for option in options)
