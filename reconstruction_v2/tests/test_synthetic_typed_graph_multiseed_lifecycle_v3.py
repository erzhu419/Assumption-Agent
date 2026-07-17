from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import (
    synthetic_typed_graph_multiseed_lifecycle_v3 as lifecycle,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _semantic_hash(value: object) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


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


def _seed_batch() -> bytes:
    return b"".join(
        bytes([32 + index]) * lifecycle.SEED_BYTES
        for index in range(lifecycle.SEED_COUNT)
    )


def _completed(
    argv: list[str], *, stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        argv,
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def test_frozen_design_is_self_hashed() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / lifecycle.DESIGN_RELATIVE_PATH
    design = _read_json(path)
    body = dict(design)
    declared = body.pop("design_sha256")

    assert declared == lifecycle.DESIGN_SHA256
    assert _semantic_hash(body) == declared
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        lifecycle.DESIGN_FILE_SHA256
    )
    assert design["cohort_contract"]["fresh_seed_count"] == 8
    assert design["cohort_contract"]["items_per_seed"] == 64
    assert design["cohort_contract"]["total_items"] == 512
    assert design["execution_kernel_contract"]["action_work_units"] == 1536
    assert design["execution_kernel_contract"][
        "late_labels_opened_exactly_once_only_after_all_actions_postflight_and_private_seal"
    ] is True
    assert design["smoke_contract"]["child_sleep_seconds"] == 10
    assert design["path_contract"]["smoke_receipt"] == str(
        lifecycle.SMOKE_RECEIPT_RELATIVE_PATH
    )
    assert design["systemd_contract"]["smoke_unit"] == (
        lifecycle.SMOKE_SYSTEMD_UNIT
    )
    assert design["systemd_contract"]["formal_unit"] == (
        lifecycle.FORMAL_SYSTEMD_UNIT
    )
    assert list(lifecycle.SMOKE_BINDING_PATHS) == design["smoke_contract"][
        "successful_receipt_must_bind_exact_code_and_test_tuples"
    ]
    for binding_name in (
        "v2_acquisition_kernel",
        "v2_runner_kernel",
        "v2_acquisition_test",
        "v2_runner_test",
    ):
        binding = design["artifact_bindings"][binding_name]
        assert hashlib.sha256((root / binding["relative_path"]).read_bytes()).hexdigest() == (
            binding["file_sha256"]
        )
    assert design["test_contract"]["existing_v1_v2_test_count"] == 57
    assert design["test_contract"]["required_result"] == (
        "exit_zero_with_57_passed_before_the_v3_smoke"
    )


def test_smoke_is_exactly_ten_seconds_data_free_and_marker_precedes_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    for index, relative in enumerate(lifecycle.SMOKE_BINDING_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"smoke-bound:{index}:{relative}".encode("utf-8"))
    monkeypatch.setattr(lifecycle, "verify_frozen_design", lambda _root: {})
    monkeypatch.setattr(
        lifecycle, "_validate_v2_interruption_and_absence", lambda _root: {}
    )
    monkeypatch.setattr(
        lifecycle, "_committed_head", lambda _root: "a" * 40, raising=False
    )
    monkeypatch.setattr(
        lifecycle, "_git_project_prefix", lambda _root: "", raising=False
    )
    monkeypatch.setattr(
        lifecycle,
        "_committed_bytes",
        lambda project_root, relative: (project_root / relative).read_bytes(),
    )
    monkeypatch.chdir(root)
    for key, value in lifecycle.SYSTEMD_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("INVOCATION_ID", "1" * 32)

    prohibited = lambda *_args, **_kwargs: pytest.fail(
        "the pre-seed smoke must not open data, models, actions, or labels"
    )
    monkeypatch.setattr(lifecycle.grammar, "generate_block", prohibited)
    monkeypatch.setattr(lifecycle.kernel_v2, "load_action_pack", prohibited)
    monkeypatch.setattr(lifecycle.kernel_v2, "load_label_pack", prohibited)
    monkeypatch.setattr(lifecycle.kernel_v2, "run_multiseed_replication", prohibited)
    monkeypatch.setattr(lifecycle, "OfflineMiniLMEncoder", prohibited)
    monkeypatch.setattr(lifecycle, "PreparedFormalRuntimeV2", prohibited)

    child_sleeps: list[float] = []
    monkeypatch.setattr(
        lifecycle.time,
        "sleep",
        lambda seconds: child_sleeps.append(seconds),
    )
    launches: list[list[str]] = []

    def fake_run(argv: list[object], **_kwargs: object):
        command = [str(value) for value in argv]
        if command[0] == "systemd-run":
            marker = root / lifecycle.SMOKE_LAUNCH_MARKER_RELATIVE_PATH
            assert marker.is_file()
            assert stat.S_IMODE(marker.stat().st_mode) == lifecycle.PRIVATE_MODE
            launches.append(command)
            lifecycle.run_smoke_child(root)
            return _completed(command)
        if command[:3] == ["systemctl", "--user", "show"]:
            return _completed(
                command,
                stdout=(
                    "LoadState=loaded\nActiveState=active\nSubState=exited\n"
                    "Result=success\nExecMainCode=1\nExecMainStatus=0\n"
                ),
            )
        pytest.fail(f"unexpected command: {command!r}")

    ticks = itertools.count(0.0, 10.0)
    receipt = lifecycle.run_systemd_smoke(
        root,
        run=fake_run,
        monotonic=lambda: next(ticks),
        sleep=lambda _seconds: None,
    )

    assert child_sleeps == [10]
    assert receipt["status"] == (
        "detached_preseed_process_custody_verified_no_data_or_models_opened"
    )
    assert receipt["child_sleep_seconds"] == 10
    assert receipt["systemd_invocation_id"] == "1" * 32
    assert [row["relative_path"] for row in receipt["bindings"]] == list(
        lifecycle.SMOKE_BINDING_PATHS
    )
    assert len(launches) == 1
    launch = launches[0]
    assert "--wait" not in launch and "--collect" not in launch
    assert launch == [
        "systemd-run",
        "--user",
        f"--unit={lifecycle.SMOKE_SYSTEMD_UNIT}",
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
        "assumption_agent.benchmarks.synthetic_typed_graph_multiseed_lifecycle_v3",
        "smoke-child",
        "--project-root",
        str(root),
    ]


def test_formal_systemd_argv_is_the_frozen_ordered_contract(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    runtime_python = root / "runtime/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_bytes(b"python")
    llm = root / "models/llm"
    embedding = root / "models/embedding"
    llm.mkdir(parents=True)
    embedding.mkdir(parents=True)

    child = lifecycle._systemd_child_argv(
        root,
        "formal-child",
        runtime_python=runtime_python,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    argv = lifecycle._systemd_run_argv(root, lifecycle.FORMAL_SYSTEMD_UNIT, child)
    assert argv == [
        "systemd-run",
        "--user",
        f"--unit={lifecycle.FORMAL_SYSTEMD_UNIT}",
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
        "assumption_agent.benchmarks.synthetic_typed_graph_multiseed_lifecycle_v3",
        "formal-child",
        "--project-root",
        str(root),
        "--runtime-python",
        str(runtime_python),
        "--local-llm-model",
        str(llm),
        "--local-embedding-model",
        str(embedding),
    ]
    assert "--wait" not in argv and "--collect" not in argv


def test_freeze_binds_committed_smoke_exact_v2_kernel_and_all_v3_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = {Path(relative) for relative in lifecycle.REQUIRED_FREEZE_PATHS}
    exact_core = {
        lifecycle.DESIGN_RELATIVE_PATH,
        lifecycle.SMOKE_RECEIPT_RELATIVE_PATH,
        Path(
            "assumption_agent/benchmarks/"
            "synthetic_typed_graph_multiseed_lifecycle_v3.py"
        ),
        Path("tests/test_synthetic_typed_graph_multiseed_lifecycle_v3.py"),
        Path(
            "assumption_agent/benchmarks/"
            "synthetic_typed_graph_multiseed_acquisition_v2.py"
        ),
        Path(
            "assumption_agent/benchmarks/"
            "synthetic_typed_graph_multiseed_runner_v2.py"
        ),
        Path("tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py"),
        Path("tests/test_synthetic_typed_graph_multiseed_runner_v2.py"),
    }
    assert exact_core <= required

    for index, relative in enumerate(sorted(required, key=str)):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bound-v3:{index}:{relative}".encode("utf-8"))

    smoke_bindings = [
        {
            "relative_path": str(relative),
            "file_sha256": hashlib.sha256(
                (tmp_path / relative).read_bytes()
            ).hexdigest(),
            "git_blob_sha1": lifecycle._git_blob_sha1(
                (tmp_path / relative).read_bytes()
            ),
        }
        for relative in lifecycle.SMOKE_BINDING_PATHS
    ]
    smoke = {
        "status": "detached_preseed_process_custody_verified_no_data_or_models_opened",
        "receipt_sha256": "5" * 64,
        "bindings": smoke_bindings,
    }
    monkeypatch.setattr(lifecycle, "verify_frozen_design", lambda _root: {})
    monkeypatch.setattr(
        lifecycle, "_validate_v2_interruption_and_absence", lambda _root: {}
    )
    monkeypatch.setattr(
        lifecycle,
        "load_committed_smoke_receipt",
        lambda _root: smoke,
    )
    monkeypatch.setattr(
        lifecycle, "_git_project_prefix", lambda _root: "", raising=False
    )
    monkeypatch.setattr(
        lifecycle,
        "_committed_bytes",
        lambda root, relative: (root / relative).read_bytes(),
    )
    monkeypatch.setattr(
        lifecycle,
        "_git",
        lambda _root, *arguments: (
            b"a" * 40 + b"\n"
            if arguments == ("rev-parse", "HEAD")
            else pytest.fail(f"unexpected Git call: {arguments!r}")
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "V2_ACQUISITION_KERNEL_FILE_SHA256",
        hashlib.sha256(
            (
                tmp_path
                / "assumption_agent/benchmarks/"
                "synthetic_typed_graph_multiseed_acquisition_v2.py"
            ).read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        lifecycle,
        "V2_RUNNER_KERNEL_FILE_SHA256",
        hashlib.sha256(
            (
                tmp_path
                / "assumption_agent/benchmarks/"
                "synthetic_typed_graph_multiseed_runner_v2.py"
            ).read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        lifecycle,
        "V2_ACQUISITION_TEST_FILE_SHA256",
        hashlib.sha256(
            (tmp_path / "tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py").read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        lifecycle,
        "V2_RUNNER_TEST_FILE_SHA256",
        hashlib.sha256(
            (tmp_path / "tests/test_synthetic_typed_graph_multiseed_runner_v2.py").read_bytes()
        ).hexdigest(),
    )

    freeze = lifecycle.create_implementation_freeze(tmp_path)
    observed = {
        Path(row["relative_path"]): (
            row["file_sha256"],
            row["git_blob_sha1"],
        )
        for row in freeze["bindings"]
    }
    for relative in required:
        raw = (tmp_path / relative).read_bytes()
        assert observed[relative] == (
            hashlib.sha256(raw).hexdigest(),
            lifecycle._git_blob_sha1(raw),
        )
    by_relative = {row["relative_path"]: row for row in freeze["bindings"]}
    assert [
        by_relative[relative] for relative in lifecycle.SMOKE_BINDING_PATHS
    ] == smoke_bindings
    assert freeze["smoke_receipt_sha256"] == smoke["receipt_sha256"]
    assert not any(
        (tmp_path / relative).exists()
        for relative in (
            lifecycle.SEED_MARKER_RELATIVE_PATH,
            lifecycle.SEED_BATCH_RELATIVE_PATH,
            lifecycle.SEED_CUSTODY_RELATIVE_PATH,
            lifecycle.ACQUISITION_MARKER_RELATIVE_PATH,
            lifecycle.ACTION_PACK_RELATIVE_PATH,
            lifecycle.LABEL_PACK_RELATIVE_PATH,
            lifecycle.RESULT_RELATIVE_PATH,
        )
    )


def _patch_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "verify_frozen_design", lambda _root: {})
    monkeypatch.setattr(
        lifecycle,
        "verify_implementation_freeze",
        lambda _root: ({"implementation_freeze_sha256": "f" * 64}, "a" * 40),
    )


def test_seed_custody_is_one_marked_urandom256_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_frozen(monkeypatch)
    batch = _seed_batch()
    calls: list[int] = []

    def entropy(size: int) -> bytes:
        calls.append(size)
        marker = tmp_path / lifecycle.SEED_MARKER_RELATIVE_PATH
        assert marker.is_file()
        assert stat.S_IMODE(marker.stat().st_mode) == lifecycle.PRIVATE_MODE
        return batch

    monkeypatch.setattr(lifecycle.os, "urandom", entropy)
    custody = lifecycle.create_seed_custody(tmp_path)

    assert calls == [256]
    seed_path = tmp_path / lifecycle.SEED_BATCH_RELATIVE_PATH
    assert seed_path.read_bytes() == batch
    assert stat.S_IMODE(seed_path.stat().st_mode) == lifecycle.PRIVATE_MODE
    assert custody["seed_batch_commitment_sha256"] == hashlib.sha256(batch).hexdigest()
    assert len(custody["ordered_seed_commitments_sha256"]) == 8
    assert custody["fresh_seeds_disjoint_from_original_and_v1"] is True


@pytest.mark.parametrize("collision", ["duplicate", "original", "v1"])
def test_seed_collision_is_terminal_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    _patch_frozen(monkeypatch)
    batch = b"z" * 256 if collision == "duplicate" else _seed_batch()
    first = hashlib.sha256(batch[: lifecycle.SEED_BYTES]).hexdigest()
    if collision == "original":
        monkeypatch.setattr(
            lifecycle.acquisition_v2,
            "ORIGINAL_SEED_COMMITMENT_SHA256",
            first,
        )
    if collision == "v1":
        monkeypatch.setattr(
            lifecycle.acquisition_v2,
            "V1_ORDERED_SEED_COMMITMENTS",
            (first, *("1" * 64 for _ in range(7))),
        )
    calls: list[int] = []

    def entropy(size: int) -> bytes:
        calls.append(size)
        assert (tmp_path / lifecycle.SEED_MARKER_RELATIVE_PATH).is_file()
        return batch

    monkeypatch.setattr(lifecycle.os, "urandom", entropy)
    with pytest.raises(lifecycle.SyntheticTypedGraphMultiseedLifecycleV3Error):
        lifecycle.create_seed_custody(tmp_path)

    assert calls == [256]
    assert not (tmp_path / lifecycle.SEED_BATCH_RELATIVE_PATH).exists()
    assert not (tmp_path / lifecycle.SEED_CUSTODY_RELATIVE_PATH).exists()
    failure = _read_json(tmp_path / lifecycle.SEED_FAILURE_RELATIVE_PATH)
    assert failure["retry_replacement_or_seed_count_change_authorized"] is False


def test_acquisition_is_exactly_eight_by_sixty_four_private_and_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_frozen(monkeypatch)
    batch = _seed_batch()
    monkeypatch.setattr(lifecycle.os, "urandom", lambda size: batch)
    custody = lifecycle.create_seed_custody(tmp_path)
    monkeypatch.setattr(
        lifecycle, "load_committed_seed_custody", lambda _root: custody
    )

    def load_prior_after_marker(root: Path) -> frozenset[str]:
        marker = root / lifecycle.ACQUISITION_MARKER_RELATIVE_PATH
        assert marker.is_file()
        assert stat.S_IMODE(marker.stat().st_mode) == lifecycle.PRIVATE_MODE
        return frozenset()

    monkeypatch.setattr(
        lifecycle,
        "_load_prior_item_commitments_after_marker",
        load_prior_after_marker,
    )
    original_generate = lifecycle.grammar.generate_block
    calls: list[tuple[bytes, str]] = []

    def generate(seed: bytes, block: str):
        calls.append((seed, block))
        return original_generate(seed, block)

    monkeypatch.setattr(lifecycle.grammar, "generate_block", generate)
    receipt = lifecycle.acquire_formal_cohort(tmp_path)

    assert len(calls) == 8
    assert [block for _seed, block in calls] == ["A_hold"] * 8
    assert [hashlib.sha256(seed).hexdigest() for seed, _block in calls] == custody[
        "ordered_seed_commitments_sha256"
    ]
    assert receipt["seed_count"] == 8
    assert receipt["item_count_per_seed"] == 64
    assert receipt["total_item_count"] == 512
    assert receipt["grammar_generate_block_call_count"] == 8
    assert receipt["new_original_and_v1_item_commitments_pairwise_disjoint"] is True
    assert receipt["formation_candidate_pool_filter_or_recipe_search_used"] is False

    for relative in (
        lifecycle.ACTION_PACK_RELATIVE_PATH,
        lifecycle.LABEL_PACK_RELATIVE_PATH,
        lifecycle.COMPILED_COHORT_PACK_RELATIVE_PATH,
    ):
        assert stat.S_IMODE((tmp_path / relative).stat().st_mode) == 0o600
    action = _read_json(tmp_path / lifecycle.ACTION_PACK_RELATIVE_PATH)
    labels = _read_json(tmp_path / lifecycle.LABEL_PACK_RELATIVE_PATH)
    compiled = _read_json(tmp_path / lifecycle.COMPILED_COHORT_PACK_RELATIVE_PATH)
    assert len(action["items"]) == len(labels["items"]) == len(compiled["items"]) == 512
    assert action["labels_included"] is False
    assert not {
        "gold_node_indices",
        "family_id",
        "family_role",
        "polarity",
        "edge_family",
        "item_commitment_sha256",
    }.intersection(action["items"][0])
    assert labels["items"][0]["action_item_sha256"] == action["items"][0][
        "action_item_sha256"
    ]


def test_acquisition_overlap_is_terminal_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_frozen(monkeypatch)
    batch = _seed_batch()
    monkeypatch.setattr(lifecycle.os, "urandom", lambda _size: batch)
    custody = lifecycle.create_seed_custody(tmp_path)
    monkeypatch.setattr(
        lifecycle, "load_committed_seed_custody", lambda _root: custody
    )
    prior = hashlib.sha256(b"prior-original-or-v1").hexdigest()
    monkeypatch.setattr(
        lifecycle,
        "_load_prior_item_commitments_after_marker",
        lambda root: (
            frozenset({prior})
            if (root / lifecycle.ACQUISITION_MARKER_RELATIVE_PATH).is_file()
            else pytest.fail("prior commitments opened before acquisition marker")
        ),
    )
    call_count = 0

    def generate(_seed: bytes, block: str):
        nonlocal call_count
        assert block == "A_hold"
        seed_index = call_count
        call_count += 1
        return tuple(
            SimpleNamespace(
                item_commitment_sha256=(
                    prior
                    if seed_index == 0 and ordinal == 0
                    else hashlib.sha256(
                        f"fresh:{seed_index}:{ordinal}".encode("ascii")
                    ).hexdigest()
                )
            )
            for ordinal in range(64)
        )

    monkeypatch.setattr(lifecycle.grammar, "generate_block", generate)
    monkeypatch.setattr(
        lifecycle.acquisition_v2,
        "_validate_compiled_item",
        lambda _item, _ordinal: None,
    )
    with pytest.raises(lifecycle.SyntheticTypedGraphMultiseedLifecycleV3Error):
        lifecycle.acquire_formal_cohort(tmp_path)

    assert call_count == 8
    assert not (tmp_path / lifecycle.ACTION_PACK_RELATIVE_PATH).exists()
    assert not (tmp_path / lifecycle.LABEL_PACK_RELATIVE_PATH).exists()
    assert not (tmp_path / lifecycle.COMPILED_COHORT_PACK_RELATIVE_PATH).exists()
    failure = _read_json(tmp_path / lifecycle.ACQUISITION_RECEIPT_RELATIVE_PATH)
    assert failure["retry_replacement_or_smaller_cohort_authorized"] is False


def test_formal_launcher_consumes_marker_before_one_fixed_systemd_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_frozen(monkeypatch)
    acquisition = {
        "receipt_sha256": "c" * 64,
        "generated_item_commitment_set_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        lifecycle,
        "load_committed_acquisition_receipt",
        lambda _root, **_kwargs: acquisition,
    )
    acquisition_path = tmp_path / lifecycle.ACQUISITION_RECEIPT_RELATIVE_PATH
    acquisition_path.parent.mkdir(parents=True, exist_ok=True)
    acquisition_path.write_bytes(b"{}\n")
    acquisition_path.chmod(0o644)
    prohibited = lambda *_args, **_kwargs: pytest.fail(
        "formal launcher must not open actions or labels"
    )
    monkeypatch.setattr(lifecycle.kernel_v2, "load_action_pack", prohibited)
    monkeypatch.setattr(lifecycle.kernel_v2, "load_label_pack", prohibited)

    runtime_python = tmp_path / "runtime/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_bytes(b"python")
    llm = tmp_path / "models/llm"
    embedding = tmp_path / "models/embedding"
    llm.mkdir(parents=True)
    embedding.mkdir(parents=True)
    observed: list[list[str]] = []

    def fake_run(argv: list[object], **_kwargs: object):
        command = [str(value) for value in argv]
        marker = tmp_path / lifecycle.FORMAL_LAUNCH_MARKER_RELATIVE_PATH
        assert marker.is_file()
        assert stat.S_IMODE(marker.stat().st_mode) == lifecycle.PRIVATE_MODE
        observed.append(command)
        return _completed(command, stdout="Running as unit: formal.service\n")

    lifecycle.launch_formal(
        tmp_path,
        runtime_python=runtime_python,
        local_llm_model=llm,
        local_embedding_model=embedding,
        run=fake_run,
    )

    assert len(observed) == 1
    expected_child = lifecycle._systemd_child_argv(
        tmp_path.resolve(),
        "formal-child",
        runtime_python=runtime_python,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )
    assert observed[0] == lifecycle._systemd_run_argv(
        tmp_path.resolve(), lifecycle.FORMAL_SYSTEMD_UNIT, expected_child
    )
    persisted_marker = _read_json(
        tmp_path / lifecycle.FORMAL_LAUNCH_MARKER_RELATIVE_PATH
    )
    assert persisted_marker["attempt_count"] == 1
    assert persisted_marker["unit"] == lifecycle.FORMAL_SYSTEMD_UNIT
    assert not (tmp_path / lifecycle.RESULT_RELATIVE_PATH).exists()


def test_formal_child_reuses_v2_kernel_and_opens_labels_once_after_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    monkeypatch.chdir(root)
    for key, value in lifecycle.SYSTEMD_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("INVOCATION_ID", "2" * 32)
    runtime_python = root / "runtime/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_bytes(b"python")
    llm = root / "models/llm"
    embedding = root / "models/embedding"
    llm.mkdir(parents=True)
    embedding.mkdir(parents=True)

    acquisition_path = root / lifecycle.ACQUISITION_RECEIPT_RELATIVE_PATH
    acquisition_path.parent.mkdir(parents=True, exist_ok=True)
    acquisition_path.write_bytes(b"{}\n")
    acquisition_path.chmod(0o644)
    acquisition_file_sha256 = hashlib.sha256(acquisition_path.read_bytes()).hexdigest()
    freeze = {"implementation_freeze_sha256": "f" * 64}
    commitments = {
        "action_pack_file_sha256": "1" * 64,
        "action_item_commitment_set_sha256": "2" * 64,
        "label_pack_file_sha256": "3" * 64,
        "label_item_commitment_set_sha256": "4" * 64,
    }
    acquisition = {
        "receipt_sha256": "c" * 64,
        "generated_item_commitment_set_sha256": "d" * 64,
        "commitments": commitments,
    }
    marker = {
        "actual_HEAD": "a" * 40,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "acquisition_receipt_file_sha256": acquisition_file_sha256,
        "marker_sha256": "b" * 64,
        "systemd_contract_sha256": "e" * 64,
        "runtime_python": str(runtime_python),
        "local_llm_model": str(llm),
        "local_embedding_model": str(embedding),
    }
    monkeypatch.setattr(
        lifecycle, "_load_formal_marker", lambda _root: (marker, "9" * 64)
    )
    monkeypatch.setattr(
        lifecycle,
        "verify_implementation_freeze",
        lambda _root: (freeze, "a" * 40),
    )
    monkeypatch.setattr(
        lifecycle,
        "load_committed_acquisition_receipt",
        lambda _root, **_kwargs: acquisition,
    )

    class FakeEncoder:
        pass

    class FakeRuntime:
        pass

    encoder = FakeEncoder()
    runtime = FakeRuntime()
    monkeypatch.setattr(lifecycle, "OfflineMiniLMEncoder", FakeEncoder)
    monkeypatch.setattr(lifecycle, "PreparedFormalRuntimeV2", FakeRuntime)
    monkeypatch.setattr(
        lifecycle.kernel_v2,
        "_prepare_formal_resources",
        lambda **_kwargs: (encoder, runtime),
    )
    events: list[str] = []
    actions = SimpleNamespace(
        file_sha256=commitments["action_pack_file_sha256"],
        item_commitment_set_sha256=commitments[
            "action_item_commitment_set_sha256"
        ],
    )
    labels = SimpleNamespace(
        file_sha256=commitments["label_pack_file_sha256"],
        item_commitment_set_sha256=commitments[
            "label_item_commitment_set_sha256"
        ],
    )

    def load_actions(_path: Path):
        events.append("actions_opened")
        return actions

    def load_labels(_path: Path):
        assert (root / lifecycle.FORMAL_ACTION_SEAL_RELATIVE_PATH).is_file()
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
        assert encoder is globals_encoder
        assert runtime is globals_runtime
        assert work_root == root / lifecycle.FORMAL_WORK_RELATIVE_PATH
        _write_private(action_seal_path, {"sealed": True})
        events.append("actions_postflight_and_sealed")
        assert label_loader() is labels
        return SimpleNamespace(done=True)

    globals_encoder = encoder
    globals_runtime = runtime
    monkeypatch.setattr(lifecycle.kernel_v2, "load_action_pack", load_actions)
    monkeypatch.setattr(lifecycle.kernel_v2, "load_label_pack", load_labels)
    monkeypatch.setattr(lifecycle.kernel_v2, "run_multiseed_replication", run_kernel)
    kernel_body = {"status": lifecycle.kernel_v2.SUCCESS_RESULT_STATUS}
    kernel_receipt = {
        **kernel_body,
        "receipt_sha256": _semantic_hash(kernel_body),
    }
    monkeypatch.setattr(
        lifecycle.kernel_v2,
        "multiseed_public_result",
        lambda _outcome: kernel_receipt,
    )

    result = lifecycle.run_formal_child(
        root,
        runtime_python=runtime_python,
        local_llm_model=llm,
        local_embedding_model=embedding,
    )

    assert result["status"] == lifecycle.SUCCESS_RESULT_STATUS
    assert result["systemd_invocation_id"] == "2" * 32
    assert result["execution_kernel_version"] == lifecycle.kernel_v2.VERSION
    assert events == [
        "actions_opened",
        "actions_postflight_and_sealed",
        "labels_opened",
    ]


def test_finalize_rejects_active_then_closes_terminal_no_result_without_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    freeze = {"implementation_freeze_sha256": "f" * 64}
    acquisition = {
        "receipt_sha256": "c" * 64,
        "generated_item_commitment_set_sha256": "d" * 64,
    }
    acquisition_path = root / lifecycle.ACQUISITION_RECEIPT_RELATIVE_PATH
    acquisition_path.parent.mkdir(parents=True, exist_ok=True)
    acquisition_path.write_bytes(b"{}\n")
    acquisition_path.chmod(0o644)
    marker = {
        "actual_HEAD": "a" * 40,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "marker_sha256": "b" * 64,
        "systemd_contract_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        lifecycle, "_load_formal_marker", lambda _root: (marker, "9" * 64)
    )
    monkeypatch.setattr(
        lifecycle,
        "verify_implementation_freeze",
        lambda _root: (freeze, "a" * 40),
    )
    monkeypatch.setattr(
        lifecycle,
        "load_committed_acquisition_receipt",
        lambda _root, **_kwargs: acquisition,
    )
    prohibited = lambda *_args, **_kwargs: pytest.fail(
        "administrative finalization must not open private actions or labels"
    )
    monkeypatch.setattr(lifecycle.kernel_v2, "load_action_pack", prohibited)
    monkeypatch.setattr(lifecycle.kernel_v2, "load_label_pack", prohibited)
    commands: list[list[str]] = []

    def state_run(active: str, sub: str, result: str, status: str):
        def fake(argv: list[object], **_kwargs: object):
            command = [str(value) for value in argv]
            commands.append(command)
            assert command[:3] == ["systemctl", "--user", "show"]
            assert all("systemd-run" not in value for value in command)
            return _completed(
                command,
                stdout=(
                    "LoadState=loaded\n"
                    f"ActiveState={active}\nSubState={sub}\n"
                    f"Result={result}\nExecMainCode=1\nExecMainStatus={status}\n"
                ),
            )

        return fake

    with pytest.raises(lifecycle.SyntheticTypedGraphMultiseedLifecycleV3Error):
        lifecycle.finalize_formal(
            root,
            run=state_run("active", "running", "success", "0"),
        )
    assert not (root / lifecycle.RESULT_RELATIVE_PATH).exists()

    result = lifecycle.finalize_formal(
        root,
        run=state_run("failed", "failed", "exit-code", "7"),
    )
    assert len(commands) == 2
    assert result["status"] == lifecycle.FAILURE_RESULT_STATUS
    assert result["failure_class"] == (
        "DetachedSystemdServiceTerminalWithoutCanonicalResult"
    )
    assert result[
        "administrative_finalization_without_private_pack_or_label_open"
    ] is True
    assert result["retry_replacement_or_relaunch_authorized"] is False


def test_terminal_publication_uses_stored_rows_and_never_regenerates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    freeze = {"implementation_freeze_sha256": "f" * 64}
    seeds = tuple(
        _seed_batch()[index * lifecycle.SEED_BYTES : (index + 1) * lifecycle.SEED_BYTES]
        for index in range(lifecycle.SEED_COUNT)
    )
    stored_rows = [
        {
            "global_ordinal": ordinal,
            "seed_index": ordinal // 64,
            "seed_ordinal": ordinal % 64,
            "item_commitment_sha256": hashlib.sha256(
                f"item:{ordinal}".encode("ascii")
            ).hexdigest(),
            "compiled_row_sha256": hashlib.sha256(
                f"row:{ordinal}".encode("ascii")
            ).hexdigest(),
        }
        for ordinal in range(512)
    ]
    item_set = lifecycle.stable_hash(
        [row["item_commitment_sha256"] for row in stored_rows]
    )
    acquisition = {
        "receipt_sha256": "c" * 64,
        "generated_item_commitment_set_sha256": item_set,
        "commitments": {
            "compiled_cohort_pack_file_sha256": "1" * 64,
            "compiled_row_commitment_set_sha256": "2" * 64,
        },
    }
    custody = {
        "seed_batch_commitment_sha256": hashlib.sha256(b"".join(seeds)).hexdigest()
    }
    result = {
        "status": lifecycle.FAILURE_RESULT_STATUS,
        "receipt_sha256": "e" * 64,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "generated_item_commitment_set_sha256": item_set,
    }
    result_path = root / lifecycle.RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(b"{}\n")
    result_path.chmod(0o644)
    monkeypatch.setattr(
        lifecycle,
        "verify_implementation_freeze",
        lambda _root: (freeze, "a" * 40),
    )
    monkeypatch.setattr(
        lifecycle,
        "load_committed_acquisition_receipt",
        lambda _root, **_kwargs: acquisition,
    )
    monkeypatch.setattr(
        lifecycle, "load_committed_seed_custody", lambda _root: custody
    )
    monkeypatch.setattr(
        lifecycle, "load_committed_terminal_result", lambda _root: result
    )

    def read_seed_batch(_path: Path, _custody: Mapping[str, object]):
        marker = root / lifecycle.PUBLICATION_MARKER_RELATIVE_PATH
        assert marker.is_file()
        assert stat.S_IMODE(marker.stat().st_mode) == lifecycle.PRIVATE_MODE
        return seeds

    monkeypatch.setattr(lifecycle, "_read_seed_batch", read_seed_batch)
    monkeypatch.setattr(
        lifecycle,
        "_verify_compiled_cohort_pack",
        lambda *_args, **_kwargs: {"items": stored_rows},
    )
    monkeypatch.setattr(
        lifecycle.grammar,
        "generate_block",
        lambda *_args, **_kwargs: pytest.fail(
            "terminal publication must not regenerate grammar"
        ),
    )

    artifact = lifecycle.publish_terminal(root)
    assert artifact["terminal_result_status"] == lifecycle.FAILURE_RESULT_STATUS
    assert artifact["formal_seed_hexes"] == [seed.hex() for seed in seeds]
    assert len(artifact["items"]) == 512
    assert all("compiled_row_sha256" not in row for row in artifact["items"])
    assert artifact["cohort_regenerated_during_publication"] is False
    assert artifact["grammar_generate_block_call_count_during_publication"] == 0
    assert artifact["retrieval_actions_model_outputs_or_scores_included"] is False
