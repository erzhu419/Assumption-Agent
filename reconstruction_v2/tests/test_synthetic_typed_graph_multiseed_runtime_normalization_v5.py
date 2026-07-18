from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import (
    synthetic_typed_graph_multiseed_runtime_normalization_v5 as repair,
)


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_private(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    path.chmod(0o600)


def _completed(
    argv: list[str], *, stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        argv, returncode=returncode, stdout=stdout, stderr=""
    )


def _runtime_symlink(root: Path) -> Path:
    target = root / "base/python-target"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"#!/bin/sh\nexit 0\n")
    target.chmod(0o755)
    lexical = root / "runtime/venv/bin/python"
    lexical.parent.mkdir(parents=True, exist_ok=True)
    lexical.symlink_to(target)
    return lexical


def _models(root: Path) -> tuple[Path, Path]:
    llm = root / "models/llm"
    embedding = root / "models/embedding"
    llm.mkdir(parents=True)
    embedding.mkdir(parents=True)
    return llm, embedding


def _preflight() -> dict[str, Any]:
    official = {
        "attestation_receipt_sha256": "7" * 64,
        "attestation_receipt_file_sha256": "8" * 64,
    }
    body = {
        "schema": "synthetic_typed_graph_multiseed_v5_path_free_preflight",
        "official_hipporag_runtime_binding": official,
        "official_runtime_binding_sha256": "6" * 64,
        "action_label_or_compiled_pack_open_calls": 0,
        "performance_signal_or_gate_computed": False,
        "runtime_asset_paths_persisted": False,
    }
    return {**body, "preflight_sha256": _semantic_hash(body)}


def _terminal_systemd_state(invocation_id: str = "3" * 32) -> dict[str, str]:
    return {
        "LoadState": "loaded",
        "ActiveState": "failed",
        "SubState": "failed",
        "MainPID": "0",
        "ControlGroup": "/user.slice/test.service",
        "Result": "exit-code",
        "ExecMainCode": "1",
        "ExecMainStatus": "1",
        "ExecMainStartTimestamp": "Sat 2026-07-18 09:00:00 CST",
        "ExecMainExitTimestamp": "Sat 2026-07-18 09:00:01 CST",
        "InvocationID": invocation_id,
    }


def _never_started_systemd_state() -> dict[str, str]:
    return {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "ControlGroup": "",
        "Result": "success",
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "ExecMainStartTimestamp": "",
        "ExecMainExitTimestamp": "",
        "InvocationID": "",
    }


def _systemd_show_output(state: Mapping[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in state.items())


def _write_valid_action_seal_and_kernel_receipt(
    root: Path,
    *,
    acquisition: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    commitments = acquisition["commitments"]
    rows = []
    for ordinal in range(512):
        rows.append(
            {
                "global_ordinal": ordinal,
                "action_item_sha256": hashlib.sha256(
                    f"action:{ordinal}".encode("ascii")
                ).hexdigest(),
                "RAW_top5": [0, 1, 2, 3, 4],
                "official_HippoRAG_top5": [1, 2, 3, 4, 5],
                "Agent_R1_top5": [2, 3, 4, 5, 6],
                "common_scan_sha256": hashlib.sha256(
                    f"scan:{ordinal}".encode("ascii")
                ).hexdigest(),
                "local_tensor_sha256": hashlib.sha256(
                    f"tensor:{ordinal}".encode("ascii")
                ).hexdigest(),
            }
        )
    action_table_sha256 = _semantic_hash(
        [
            [
                row["RAW_top5"],
                row["official_HippoRAG_top5"],
                row["Agent_R1_top5"],
            ]
            for row in rows
        ]
    )
    runtime_binding = str(preflight["official_runtime_binding_sha256"])
    action_pack_sha256 = "a" * 64
    label_pack_sha256 = "b" * 64
    seal_body = {
        "schema": f"{repair.kernel_v2.VERSION}_private_action_seal",
        "version": repair.kernel_v2.VERSION,
        "status": "all_1536_actions_joined_official_postflight_terminal",
        "purpose": "fresh_formal_replication",
        "block": repair.kernel_v2.BLOCK,
        "recipe_id": repair.kernel_v2.RECIPE_ID,
        "item_count": 512,
        "action_work_unit_count": 1536,
        "submitted_action_work_unit_count": 1536,
        "terminal_action_work_unit_count": 1536,
        "official_retrieve_action_count": 512,
        "official_call_count": 512,
        "RAW_action_count": 512,
        "Agent_R1_action_count": 512,
        "official_concurrency_cap": 8,
        "local_concurrency_cap": 64,
        "official_peak_concurrency_count": 1,
        "local_peak_concurrency_count": 1,
        "chunk_schedule_sha256": repair.kernel_v2.CHUNK_SCHEDULE_SHA256,
        "observed_encoder_input_row_counts": [8448, 8448],
        "observed_encoder_output_row_counts": [8448, 8448],
        "action_pack_file_sha256": commitments["action_pack_file_sha256"],
        "action_pack_sha256": action_pack_sha256,
        "action_item_commitment_set_sha256": commitments[
            "action_item_commitment_set_sha256"
        ],
        "runtime_binding_sha256": runtime_binding,
        "official_postflight_receipt_sha256": runtime_binding,
        "action_table_sha256": action_table_sha256,
        "action_rows": rows,
        "labels_opened_before_action_seal": False,
        "labels_opened_before_seal": False,
        "scores_computed_before_action_seal": False,
    }
    seal = {
        **seal_body,
        "action_seal_sha256": _semantic_hash(seal_body),
    }
    seal_path = root / repair.FORMAL_ACTION_SEAL_RELATIVE_PATH
    _write_private(seal_path, seal)
    seal_file_sha256 = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    body = {
        "schema": repair.kernel_v2.RESULT_SCHEMA,
        "version": repair.kernel_v2.DESIGN_VERSION,
        "status": repair.kernel_v2.SUCCESS_RESULT_STATUS,
        "design_sha256": repair.kernel_v2.DESIGN_SHA256,
        "design_file_sha256": repair.kernel_v2.DESIGN_FILE_SHA256,
        "block": repair.kernel_v2.BLOCK,
        "recipe_id": repair.kernel_v2.RECIPE_ID,
        "seed_count": 8,
        "item_count_per_seed": 64,
        "total_item_count": 512,
        "arms": list(repair.kernel_v2.ARM_IDS),
        "action_work_unit_count": 1536,
        "official_retrieve_action_count": 512,
        "official_concurrency_cap": 8,
        "local_concurrency_cap": 64,
        "official_peak_concurrency_count": 1,
        "local_peak_concurrency_count": 1,
        "chunk_schedule_sha256": repair.kernel_v2.CHUNK_SCHEDULE_SHA256,
        "observed_encoder_input_row_counts": [8448, 8448],
        "observed_encoder_output_row_counts": [8448, 8448],
        "action_pack_file_sha256": commitments["action_pack_file_sha256"],
        "action_pack_sha256": action_pack_sha256,
        "action_item_commitment_set_sha256": commitments[
            "action_item_commitment_set_sha256"
        ],
        "label_pack_file_sha256": commitments["label_pack_file_sha256"],
        "label_pack_sha256": label_pack_sha256,
        "label_item_commitment_set_sha256": commitments[
            "label_item_commitment_set_sha256"
        ],
        "runtime_binding_sha256": runtime_binding,
        "official_postflight_receipt_sha256": runtime_binding,
        "action_table_sha256": action_table_sha256,
        "action_seal_sha256": seal["action_seal_sha256"],
        "action_seal_file_sha256": seal_file_sha256,
        "aggregates": {arm: {} for arm in repair.kernel_v2.ARM_IDS},
        "cluster_differences": {
            "Agent_R1_minus_official_HippoRAG": {},
            "Agent_R1_minus_RAW": {},
        },
        "interpretation": "descriptive_fixed_cohort_replication_only",
        "seeds_or_item_rows_disclosed": False,
    }
    return {**body, "receipt_sha256": _semantic_hash(body)}


def _patch_predecessors(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = {"implementation_freeze_sha256": "f" * 64}
    closure = {
        "closure_receipt_sha256": repair.V4_CLOSURE_RECEIPT_SHA256,
    }
    acquisition = {
        "receipt_sha256": "c" * 64,
        "generated_item_commitment_set_sha256": "d" * 64,
        "commitments": {
            "action_pack_file_sha256": "1" * 64,
            "action_item_commitment_set_sha256": "2" * 64,
            "label_pack_file_sha256": "3" * 64,
            "label_item_commitment_set_sha256": "4" * 64,
        },
    }
    acquisition_path = root / repair.v3.ACQUISITION_RECEIPT_RELATIVE_PATH
    acquisition_path.parent.mkdir(parents=True, exist_ok=True)
    acquisition_path.write_bytes(b"{}\n")
    acquisition_path.chmod(0o644)
    monkeypatch.setattr(
        repair,
        "verify_implementation_freeze",
        lambda _root: (freeze, "a" * 40),
    )
    monkeypatch.setattr(
        repair, "verify_v4_prefreeze_closure", lambda _root: closure
    )
    monkeypatch.setattr(
        repair.v3,
        "load_committed_acquisition_receipt",
        lambda _root, **kwargs: (
            acquisition
            if kwargs.get("verify_private_packs") is False
            else pytest.fail("private packs must remain closed")
        ),
    )
    return freeze, closure, acquisition


def test_frozen_design_and_repair_scope_are_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / repair.DESIGN_RELATIVE_PATH
    design = _read_json(path)
    body = dict(design)
    declared = body.pop("design_sha256")

    assert declared == repair.DESIGN_SHA256
    assert _semantic_hash(body) == declared
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        repair.DESIGN_FILE_SHA256
    )
    assert design["repair_scope"]["new_feasibility_or_performance_gate_authorized"] is False
    assert design["repair_scope"]["allowed_semantic_changes"] == list(
        repair.ALLOWED_SEMANTIC_CHANGES
    )
    assert design["cohort_reuse_contract"]["source_cohort_version"] == (
        repair.SOURCE_COHORT_VERSION
    )
    source = (root / repair.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "runtime_python.resolve(" not in source
    assert "local_llm_model.expanduser().resolve(" not in source


def test_lexical_runtime_python_preserves_real_symlink_and_rejects_target(
    tmp_path: Path,
) -> None:
    lexical = _runtime_symlink(tmp_path)
    observed = repair._lexical_runtime_python(lexical)

    assert observed == lexical.absolute()
    assert observed.is_symlink()
    with pytest.raises(repair.SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error):
        repair._lexical_runtime_python(observed.resolve())


def test_systemd_argv_keeps_lexical_runtime_token(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    lexical = _runtime_symlink(root)
    llm, embedding = _models(root)

    child = repair._systemd_child_argv(
        root,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    argv = repair._systemd_run_argv(root, child)

    runtime_index = argv.index("--runtime-python") + 1
    assert argv[runtime_index] == str(lexical)
    assert argv[runtime_index] != str(lexical.resolve())
    assert argv == [
        "systemd-run",
        "--user",
        f"--unit={repair.FORMAL_SYSTEMD_UNIT}",
        "--service-type=exec",
        "--remain-after-exit",
        f"--working-directory={root}",
        "--property=StandardOutput=journal",
        "--property=StandardError=journal",
        "--property=KillMode=control-group",
        "--property=Restart=no",
        "--property=UMask=0077",
        "--property=TimeoutStopSec=60s",
        "--setenv=TMPDIR=/tmp",
        "--setenv=HF_HUB_OFFLINE=1",
        "--setenv=TRANSFORMERS_OFFLINE=1",
        str(Path(sys.executable).resolve()),
        "-u",
        "-m",
        "assumption_agent.benchmarks."
        "synthetic_typed_graph_multiseed_runtime_normalization_v5",
        "formal-child",
        "--project-root",
        str(root),
        "--runtime-python",
        str(lexical),
        "--local-llm-model",
        str(llm),
        "--local-embedding-model",
        str(embedding),
    ]


def test_resource_preflight_calls_v3_binding_and_does_not_resolve_llm_root_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    lexical = _runtime_symlink(root)
    llm_target = root / "models/llm-target"
    llm_target.mkdir(parents=True)
    llm = root / "models/llm-link"
    llm.symlink_to(llm_target, target_is_directory=True)
    embedding = root / "models/embedding"
    embedding.mkdir()
    fake_runtime = object()
    observed: dict[str, Any] = {}

    def encoder_factory(**kwargs: Any) -> SimpleNamespace:
        observed["encoder_kwargs"] = kwargs
        return SimpleNamespace(
            runtime_receipt={
                "asset_manifest_path": str(
                    root / repair.kernel_v2.MINILM_MANIFEST_RELATIVE_PATH
                ),
                "model_root": str(
                    root / repair.kernel_v2.MINILM_MODEL_ROOT_RELATIVE_PATH
                ),
            }
        )

    def prepare_v3(**kwargs: Any) -> object:
        observed["runtime_kwargs"] = kwargs
        return fake_runtime

    monkeypatch.setattr(repair, "OfflineMiniLMEncoder", encoder_factory)
    monkeypatch.setattr(repair, "prepare_formal_runtime_v3", prepare_v3)
    monkeypatch.setattr(
        repair,
        "_path_free_preflight_receipt",
        lambda encoder, runtime: (
            _preflight()
            if runtime is fake_runtime
            else pytest.fail("wrong prepared runtime")
        ),
    )
    monkeypatch.setattr(
        repair.kernel_v2,
        "_prepare_formal_resources",
        lambda **_kwargs: pytest.fail("v2 resource factory must not be called"),
    )

    _encoder, runtime, preflight = repair._prepare_formal_resources(
        project_root=root,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )

    assert runtime is fake_runtime
    assert preflight == _preflight()
    assert observed["runtime_kwargs"] == {
        "project_root": root,
        "attestation_receipt_path": root / repair.OFFICIAL_ATTESTATION_V3_RELATIVE_PATH,
        "base_binding_receipt_path": (
            root / repair.kernel_v2.OFFICIAL_BASE_RECEIPT_RELATIVE_PATH
        ),
        "runtime_python": lexical,
        "local_llm_model": llm,
        "local_embedding_model": embedding,
    }
    assert llm.is_symlink()


def test_launcher_preflight_precedes_marker_and_never_opens_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    lexical = _runtime_symlink(root)
    llm, embedding = _models(root)
    _patch_predecessors(root, monkeypatch)
    events: list[str] = []
    preflight = _preflight()

    prohibited = lambda *_args, **_kwargs: pytest.fail(
        "launcher must not open action, label, or compiled cohort packs"
    )
    monkeypatch.setattr(repair.kernel_v2, "load_action_pack", prohibited)
    monkeypatch.setattr(repair.kernel_v2, "load_label_pack", prohibited)
    monkeypatch.setattr(repair.v3, "_verify_compiled_cohort_pack", prohibited)

    def prepare(**kwargs: object):
        assert kwargs["runtime_python"] == lexical
        assert lexical.is_symlink()
        assert not (root / repair.FORMAL_LAUNCH_MARKER_RELATIVE_PATH).exists()
        events.append("preflight")
        return object(), object(), preflight

    monkeypatch.setattr(repair, "_prepare_formal_resources", prepare)
    observed: list[list[str]] = []

    def fake_run(argv: list[object], **_kwargs: object):
        marker_path = root / repair.FORMAL_LAUNCH_MARKER_RELATIVE_PATH
        assert marker_path.is_file()
        assert stat.S_IMODE(marker_path.stat().st_mode) == 0o600
        assert events == ["preflight"]
        events.append("launch")
        command = [str(value) for value in argv]
        observed.append(command)
        return _completed(command, stdout="Running as unit\n")

    marker = repair.launch_formal(
        root,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
        run=fake_run,
    )

    assert events == ["preflight", "launch"]
    assert len(observed) == 1
    assert marker["runtime_python"] == str(lexical)
    assert marker["path_free_preflight"] == preflight
    assert marker["private_packs_opened_before_marker"] is False


def test_preflight_failure_does_not_consume_formal_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    lexical = _runtime_symlink(root)
    llm, embedding = _models(root)
    _patch_predecessors(root, monkeypatch)
    calls = 0

    def fail(**_kwargs: object):
        nonlocal calls
        calls += 1
        raise RuntimeError("installation invalid")

    monkeypatch.setattr(repair, "_prepare_formal_resources", fail)
    with pytest.raises(RuntimeError, match="installation invalid"):
        repair.launch_formal(
            root,
            runtime_python=lexical,
            local_llm_model=llm,
            local_embedding_model=embedding,
            run=lambda *_args, **_kwargs: pytest.fail("must not launch"),
        )

    assert calls == 1
    assert not (root / repair.FORMAL_LAUNCH_MARKER_RELATIVE_PATH).exists()
    assert not (root / repair.RESULT_RELATIVE_PATH).exists()


@pytest.mark.parametrize(
    ("state", "state_returncode", "expected_open_state", "expected_evidence"),
    [
        (
            _never_started_systemd_state(),
            4,
            "unopened",
            "systemd_launch_failure_before_child_start",
        ),
        (
            _terminal_systemd_state(),
            0,
            "unknown",
            "launcher_nonzero_with_verified_terminal_unit",
        ),
    ],
)
def test_nonzero_systemd_run_uses_only_positive_state_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, str],
    state_returncode: int,
    expected_open_state: str,
    expected_evidence: str,
) -> None:
    root = tmp_path.resolve()
    lexical = _runtime_symlink(root)
    llm, embedding = _models(root)
    _patch_predecessors(root, monkeypatch)
    monkeypatch.setattr(
        repair,
        "_prepare_formal_resources",
        lambda **_kwargs: (object(), object(), _preflight()),
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        repair,
        "_persist_formal_failure",
        lambda **kwargs: captured.update(kwargs) or {"status": "captured"},
    )

    def run(argv: list[object], **_kwargs: object):
        command = [str(value) for value in argv]
        if command[0] == "systemd-run":
            return _completed(command, returncode=1)
        return _completed(
            command,
            stdout=_systemd_show_output(state),
            returncode=state_returncode,
        )

    result = repair.launch_formal(
        root,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
        run=run,
    )

    assert result == {"status": "captured"}
    assert captured["pack_label_open_state"] == expected_open_state
    assert captured["open_state_evidence"] == expected_evidence


def test_nonzero_systemd_run_with_possibly_running_child_keeps_marker_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    lexical = _runtime_symlink(root)
    llm, embedding = _models(root)
    _patch_predecessors(root, monkeypatch)
    monkeypatch.setattr(
        repair,
        "_prepare_formal_resources",
        lambda **_kwargs: (object(), object(), _preflight()),
    )
    running = {
        **_terminal_systemd_state(),
        "ActiveState": "active",
        "SubState": "running",
        "MainPID": "123",
        "Result": "success",
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "ExecMainExitTimestamp": "",
    }
    monkeypatch.setattr(
        repair,
        "_persist_formal_failure",
        lambda **_kwargs: pytest.fail("possibly running child must stay pending"),
    )

    def run(argv: list[object], **_kwargs: object):
        command = [str(value) for value in argv]
        if command[0] == "systemd-run":
            return _completed(command, returncode=1)
        return _completed(command, stdout=_systemd_show_output(running))

    with pytest.raises(
        repair.SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error,
        match="consumed marker remains pending",
    ):
        repair.launch_formal(
            root,
            runtime_python=lexical,
            local_llm_model=llm,
            local_embedding_model=embedding,
            run=run,
        )

    assert (root / repair.FORMAL_LAUNCH_MARKER_RELATIVE_PATH).is_file()
    assert not (root / repair.RESULT_RELATIVE_PATH).exists()


@pytest.mark.parametrize("relative", repair.V4_REQUIRED_ABSENT_PATHS)
def test_any_closed_v4_path_invalidates_v5(
    tmp_path: Path, relative: Path
) -> None:
    root = tmp_path.resolve()
    occupied = root / relative
    if relative.suffix:
        occupied.parent.mkdir(parents=True, exist_ok=True)
        occupied.write_bytes(b"occupied")
    else:
        occupied.mkdir(parents=True)

    with pytest.raises(
        repair.SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error,
        match="closed v4 output unexpectedly exists",
    ):
        repair.verify_v4_prefreeze_closure(root)


def test_formal_child_reuses_v2_kernel_and_opens_labels_once_after_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    monkeypatch.chdir(root)
    for key, value in repair.SYSTEMD_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("INVOCATION_ID", "2" * 32)
    lexical = _runtime_symlink(root)
    llm, embedding = _models(root)
    freeze, closure, acquisition = _patch_predecessors(root, monkeypatch)
    acquisition_file_sha256 = hashlib.sha256(
        (root / repair.v3.ACQUISITION_RECEIPT_RELATIVE_PATH).read_bytes()
    ).hexdigest()
    preflight = _preflight()
    marker = {
        "actual_HEAD": "a" * 40,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "v4_closure_receipt_sha256": closure["closure_receipt_sha256"],
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "acquisition_receipt_file_sha256": acquisition_file_sha256,
        "marker_sha256": "b" * 64,
        "systemd_contract_sha256": "e" * 64,
        "runtime_python": str(lexical),
        "local_llm_model": str(llm),
        "local_embedding_model": str(embedding),
        "path_free_preflight": preflight,
        "official_attestation_v3_receipt_sha256": preflight[
            "official_hipporag_runtime_binding"
        ]["attestation_receipt_sha256"],
        "official_attestation_v3_receipt_file_sha256": preflight[
            "official_hipporag_runtime_binding"
        ]["attestation_receipt_file_sha256"],
        "generated_item_commitment_set_sha256": acquisition[
            "generated_item_commitment_set_sha256"
        ],
    }
    monkeypatch.setattr(
        repair, "_load_formal_marker", lambda _root: (marker, "9" * 64)
    )

    encoder = object()
    runtime = object()
    monkeypatch.setattr(
        repair,
        "_prepare_formal_resources",
        lambda **_kwargs: (encoder, runtime, preflight),
    )
    events: list[str] = []
    actions = SimpleNamespace(
        file_sha256=acquisition["commitments"]["action_pack_file_sha256"],
        item_commitment_set_sha256=acquisition["commitments"][
            "action_item_commitment_set_sha256"
        ],
    )
    labels = SimpleNamespace(
        file_sha256=acquisition["commitments"]["label_pack_file_sha256"],
        item_commitment_set_sha256=acquisition["commitments"][
            "label_item_commitment_set_sha256"
        ],
    )
    receipt_holder: dict[str, dict[str, Any]] = {}

    def load_actions(path: Path):
        assert path == root / repair.v3.ACTION_PACK_RELATIVE_PATH
        events.append("actions_opened")
        return actions

    def load_labels(path: Path):
        assert path == root / repair.v3.LABEL_PACK_RELATIVE_PATH
        assert (root / repair.FORMAL_ACTION_SEAL_RELATIVE_PATH).is_file()
        events.append("labels_opened")
        return labels

    def run_kernel(
        action_pack: object,
        *,
        label_loader: Any,
        encoder: object,
        runtime: object,
        work_root: Path,
        action_seal_path: Path,
    ):
        assert action_pack is actions
        assert work_root == root / repair.FORMAL_WORK_RELATIVE_PATH
        assert action_seal_path == root / repair.FORMAL_ACTION_SEAL_RELATIVE_PATH
        receipt_holder["value"] = _write_valid_action_seal_and_kernel_receipt(
            root,
            acquisition=acquisition,
            preflight=preflight,
        )
        events.append("actions_postflight_and_sealed")
        assert label_loader() is labels
        return SimpleNamespace(done=True)

    monkeypatch.setattr(repair.kernel_v2, "load_action_pack", load_actions)
    monkeypatch.setattr(repair.kernel_v2, "load_label_pack", load_labels)
    monkeypatch.setattr(repair.kernel_v2, "run_multiseed_replication", run_kernel)
    monkeypatch.setattr(
        repair.kernel_v2,
        "multiseed_public_result",
        lambda _outcome: receipt_holder["value"],
    )

    result = repair.run_formal_child(
        root,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )

    assert result["status"] == repair.SUCCESS_RESULT_STATUS
    assert result["source_cohort_version"] == repair.SOURCE_COHORT_VERSION
    assert result["execution_kernel_version"] == repair.kernel_v2.VERSION
    assert events == [
        "actions_opened",
        "actions_postflight_and_sealed",
        "labels_opened",
    ]


def test_strict_v2_kernel_receipt_validates_exact_seal_and_rejects_minimal_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    _freeze, _closure, acquisition = _patch_predecessors(root, monkeypatch)
    preflight = _preflight()
    marker = {"path_free_preflight": preflight}
    receipt = _write_valid_action_seal_and_kernel_receipt(
        root,
        acquisition=acquisition,
        preflight=preflight,
    )

    assert repair._validate_v2_kernel_receipt(
        root,
        receipt,
        acquisition=acquisition,
        marker=marker,
    ) == receipt
    with pytest.raises(
        repair.SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error,
        match="key set",
    ):
        repair._validate_v2_kernel_receipt(
            root,
            {"status": repair.kernel_v2.SUCCESS_RESULT_STATUS},
            acquisition=acquisition,
            marker=marker,
        )

    import inspect

    assert "_validate_v2_kernel_receipt" in inspect.getsource(
        repair._load_terminal_result_local
    )


def test_freeze_scope_binds_v5_v4_closure_runtime_v3_and_exact_v2_kernel() -> None:
    required = set(repair.REQUIRED_FREEZE_PATHS)
    assert {
        repair.DESIGN_RELATIVE_PATH.as_posix(),
        repair.MODULE_RELATIVE_PATH.as_posix(),
        repair.TEST_RELATIVE_PATH.as_posix(),
        repair.V4_CLOSURE_RECEIPT_RELATIVE_PATH.as_posix(),
        repair.V3_CLOSURE_RECEIPT_RELATIVE_PATH.as_posix(),
        repair.v3.IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix(),
        repair.v3.SEED_CUSTODY_RELATIVE_PATH.as_posix(),
        repair.v3.ACQUISITION_RECEIPT_RELATIVE_PATH.as_posix(),
        "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v2.py",
        "tests/test_synthetic_typed_graph_multiseed_runner_v2.py",
        "replication_runtime/qasper_minilm_v1/binding.py",
        "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
        "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v3.py",
        "replication_runtime/musique_official_hipporag_v1/adapter_v3.py",
        "assumption_agent/benchmarks/musique_formal_runtime_binding_v3.py",
        "tests/test_musique_runtime_attestation_v3.py",
        repair.OFFICIAL_ATTESTATION_V3_RELATIVE_PATH.as_posix(),
    } <= required
    assert tuple(relative for relative, _key in repair.PRIVATE_PACK_BINDINGS) == (
        repair.v3.ACTION_PACK_RELATIVE_PATH,
        repair.v3.LABEL_PACK_RELATIVE_PATH,
        repair.v3.COMPILED_COHORT_PACK_RELATIVE_PATH,
    )


def test_consumed_marker_prevents_relaunch_before_second_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    lexical = _runtime_symlink(root)
    llm, embedding = _models(root)
    _patch_predecessors(root, monkeypatch)
    calls = 0

    def prepare(**_kwargs: object):
        nonlocal calls
        calls += 1
        return object(), object(), _preflight()

    monkeypatch.setattr(repair, "_prepare_formal_resources", prepare)
    repair.launch_formal(
        root,
        runtime_python=lexical,
        local_llm_model=llm,
        local_embedding_model=embedding,
        run=lambda argv, **_kwargs: _completed([str(value) for value in argv]),
    )
    with pytest.raises(repair.SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error):
        repair.launch_formal(
            root,
            runtime_python=lexical,
            local_llm_model=llm,
            local_embedding_model=embedding,
            run=lambda *_args, **_kwargs: pytest.fail("must not relaunch"),
        )

    assert calls == 1


def test_finalizer_uses_only_verified_target_unit_invocation_and_claims_unknown_open_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    monkeypatch.setenv("INVOCATION_ID", "9" * 32)
    freeze, _closure, acquisition = _patch_predecessors(root, monkeypatch)
    preflight = _preflight()
    marker = {
        "actual_HEAD": "a" * 40,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "v4_closure_receipt_sha256": repair.V4_CLOSURE_RECEIPT_SHA256,
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "acquisition_receipt_file_sha256": hashlib.sha256(
            (root / repair.v3.ACQUISITION_RECEIPT_RELATIVE_PATH).read_bytes()
        ).hexdigest(),
        "generated_item_commitment_set_sha256": acquisition[
            "generated_item_commitment_set_sha256"
        ],
        "marker_sha256": "b" * 64,
        "systemd_contract_sha256": "e" * 64,
        "path_free_preflight": preflight,
        "official_attestation_v3_receipt_sha256": preflight[
            "official_hipporag_runtime_binding"
        ]["attestation_receipt_sha256"],
        "official_attestation_v3_receipt_file_sha256": preflight[
            "official_hipporag_runtime_binding"
        ]["attestation_receipt_file_sha256"],
    }
    monkeypatch.setattr(
        repair, "_load_formal_marker", lambda _root: (marker, "4" * 64)
    )
    state = _terminal_systemd_state()
    monkeypatch.setattr(repair, "_read_systemd_state", lambda *_args, **_kwargs: (state, 0))
    captured: dict[str, Any] = {}

    def persist(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr(repair, "_persist_formal_failure", persist)

    result = repair.finalize_formal(
        root, run=lambda *_args, **_kwargs: pytest.fail("state read is patched")
    )

    assert result == {"status": "captured"}
    assert captured["systemd_invocation_id"] == "3" * 32
    assert captured["systemd_invocation_id"] != os.environ["INVOCATION_ID"]
    assert captured["pack_label_open_state"] == "unknown"
    assert captured["open_state_evidence"] == (
        "administrative_finalizer_without_durable_child_evidence"
    )
    assert captured["administrative"] is True


@pytest.mark.parametrize(
    ("state", "returncode"),
    [
        ({}, 1),
        ({"LoadState": "loaded", "ActiveState": "inactive"}, 0),
        (
            {
                **_terminal_systemd_state(),
                "ActiveState": "deactivating",
                "SubState": "stop-sigterm",
            },
            0,
        ),
        (
            {
                **_terminal_systemd_state(),
                "ActiveState": "maintenance",
                "SubState": "unknown",
            },
            0,
        ),
        ({**_terminal_systemd_state(), "InvocationID": "invalid"}, 0),
        (_terminal_systemd_state(), 1),
    ],
)
def test_finalizer_waits_on_empty_partial_deactivating_or_unverified_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, str],
    returncode: int,
) -> None:
    root = tmp_path.resolve()
    _patch_predecessors(root, monkeypatch)
    monkeypatch.setattr(repair, "_load_formal_marker", lambda _root: ({}, "4" * 64))
    monkeypatch.setattr(
        repair,
        "_read_systemd_state",
        lambda *_args, **_kwargs: (state, returncode),
    )
    monkeypatch.setattr(
        repair,
        "_persist_formal_failure",
        lambda **_kwargs: pytest.fail("uncertain state must not create a result"),
    )

    with pytest.raises(
        repair.SyntheticTypedGraphMultiseedRuntimeNormalizationV5Error,
        match="complete positive terminal evidence",
    ):
        repair.finalize_formal(root)

    assert not (root / repair.RESULT_RELATIVE_PATH).exists()


def test_failure_body_never_reads_current_process_invocation_environment() -> None:
    import inspect

    source = inspect.getsource(repair._formal_failure_body)
    assert "INVOCATION_ID" not in source
    assert "os.environ" not in source
