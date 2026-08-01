from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

from assumption_agent.benchmarks import gscl_arn_formal_supervisor_v1 as supervisor
from assumption_agent.benchmarks import gscl_arn_intrinsic_protocol_v1 as protocol
from assumption_agent.gscl_arn_raw_adapter_v1 import ArnTopology


RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = RECONSTRUCTION_ROOT / "tests" / "fixtures"
CHILD = FIXTURE_ROOT / "gscl_formal_supervisor_child_v1.py"
CLOSURE_ENTRY = FIXTURE_ROOT / "gscl_closure_fixture_v1.py"
CLOSURE_TEST = FIXTURE_ROOT / "test_gscl_closure_fixture_v1.py"


@pytest.fixture
def secure_tmp_path() -> Path:
    root = Path(tempfile.mkdtemp(prefix="gscl-supervisor-", dir="/var/tmp"))
    root.chmod(0o700)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _csv() -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, dialect="excel", lineterminator="\r\n")
    writer.writerow(protocol.OFFICIAL_HEADER)
    writer.writerows(
        [
            [
                "1",
                "A source-free proverb.",
                "A source-free query.",
                "The first candidate.",
                "The second candidate.",
                "high",
                "far",
                "A",
            ],
            [
                "3",
                "Another source-free proverb.",
                "Another source-free query.",
                "One candidate.",
                "Two candidate.",
                "low",
                "near",
                "Two candidate.",
            ],
        ]
    )
    return buffer.getvalue().encode("utf-8")


def _topology() -> ArnTopology:
    return ArnTopology(
        row_count=2,
        id_minimum=1,
        id_maximum=3,
        missing_ids=(2,),
        cell_counts={
            "far_high": 1,
            "far_low": 0,
            "near_high": 0,
            "near_low": 1,
        },
    )


def _closure(tmp_path: Path) -> supervisor.RuntimeClosure:
    config = tmp_path / "frozen_config.json"
    config.write_text('{"fixture":true}\n', encoding="ascii")
    asset_root = tmp_path / "model"
    asset_root.mkdir(mode=0o700)
    (asset_root / "weights.bin").write_bytes(b"synthetic weights only")
    attestation = supervisor.run_source_free_tests(
        code_root=FIXTURE_ROOT,
        test_files=(CLOSURE_TEST,),
    )
    closure = supervisor.attest_runtime_closure(
        code_roots=(FIXTURE_ROOT,),
        entry_files=(CLOSURE_ENTRY, CHILD),
        config_files=(config,),
        asset_roots=(asset_root,),
        test_attestation=attestation,
    )
    assert closure.manifest["source_content_supplied"] is False
    assert closure.manifest["formal_measurement_run"] is False
    assert closure.manifest["test_attestation"]["self_hash"] == (
        attestation.receipt["self_hash"]
    )
    assert str(config) in closure.manifest["config_sha256s"]
    assert str(asset_root / "weights.bin") in closure.manifest[
        "asset_sha256s"
    ]
    assert {
        str(Path(value).resolve())
        for value in (
            sys.prefix,
            sys.exec_prefix,
            sys.base_prefix,
            sys.base_exec_prefix,
            Path(sys.executable).resolve().parent.parent,
        )
    }.issubset(set(closure.manifest["runtime_roots"]))
    return closure


def _action(
    runtime: supervisor.FormalSupervisor,
    closure: supervisor.RuntimeClosure,
    raw: bytes,
    tmp_path: Path,
) -> supervisor.FrozenAction:
    implementation_hash = hashlib.sha256(CHILD.read_bytes()).hexdigest()
    asset_root = tmp_path / "model"
    commands = tuple(
        supervisor.ArmCommand(
            arm_id=arm_id,
            command_template=(
                str(Path(sys.executable).resolve()),
                str(CHILD),
                "arm",
                "{input}",
                "{output}",
            ),
            code_roots=(FIXTURE_ROOT,),
            model_roots=(asset_root,),
            implementation_path=CHILD,
            implementation_sha256=implementation_hash,
        )
        for arm_id in protocol.ARM_IDS
    )
    return runtime.freeze_action_once(
        closure=closure,
        arm_commands=commands,
        scorer_command=supervisor.ScorerCommand(
            command_template=(
                str(Path(sys.executable).resolve()),
                str(CHILD),
                "score",
                "{labels}",
                "{predictions}",
                "{output}",
            ),
            implementation_path=CHILD,
            implementation_sha256=implementation_hash,
        ),
        freeze_commitments={
            "extractor": hashlib.sha256(b"extractor").hexdigest(),
            "binder": hashlib.sha256(b"binder").hexdigest(),
            "four_arm_contract": hashlib.sha256(
                b"four-arm-contract"
            ).hexdigest(),
        },
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def test_secure_directory_rejects_symlink_components_and_is_exclusive(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    root = tmp_path / "secure"
    root.mkdir(mode=0o700)
    with supervisor.SecureDirectory(root) as store:
        assert store.write_exclusive("a/b.bin", b"one") == hashlib.sha256(
            b"one"
        ).hexdigest()
        assert store.read_bytes("a/b.bin") == b"one"
        with pytest.raises(
            supervisor.FormalSupervisorError,
            match="secure_output_already_exists",
        ):
            store.write_exclusive("a/b.bin", b"two")
        target = tmp_path / "target"
        target.mkdir()
        os.symlink(target, root / "link")
        with pytest.raises(
            supervisor.FormalSupervisorError,
            match="secure_parent_open_failed",
        ):
            store.write_exclusive("link/escape.bin", b"escape")


def test_formal_constructor_cannot_accept_a_caller_root(
    secure_tmp_path: Path,
) -> None:
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="caller_selected_root_forbidden",
    ):
        supervisor.FormalSupervisor(_root=secure_tmp_path / "forbidden")


def test_closure_revalidation_detects_config_drift(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    closure = _closure(tmp_path)
    config = Path(next(iter(closure.manifest["config_sha256s"])))
    config.write_text('{"fixture":false}\n', encoding="ascii")
    root = tmp_path / "formal"
    with supervisor.FormalSupervisor._source_free_qualification(
        root
    ) as runtime:
        with pytest.raises(
            supervisor.FormalSupervisorError,
            match="runtime_closure_changed",
        ):
            _action(runtime, closure, _csv(), tmp_path)


def test_closure_binds_every_concrete_ancestor_package_initializer(
    secure_tmp_path: Path,
) -> None:
    code_root = secure_tmp_path / "code"
    package = code_root / "pkg"
    subpackage = package / "sub"
    subpackage.mkdir(parents=True)
    package_init = package / "__init__.py"
    subpackage_init = subpackage / "__init__.py"
    module = subpackage / "mod.py"
    test_file = code_root / "test_source_free.py"
    package_init.write_text("ROOT = 'bound'\n", encoding="ascii")
    subpackage_init.write_text(
        "from .mod import VALUE\n", encoding="ascii"
    )
    module.write_text("VALUE = 7\n", encoding="ascii")
    test_file.write_text(
        "def test_source_free():\n    assert True\n",
        encoding="ascii",
    )
    attestation = supervisor.run_source_free_tests(
        code_root=code_root,
        test_files=(test_file,),
    )
    closure = supervisor.attest_runtime_closure(
        code_roots=(code_root,),
        entry_files=(module,),
        config_files=(),
        asset_roots=(),
        test_attestation=attestation,
    )
    origins = closure.manifest["python_origins"]
    assert origins["pkg"]["origin"] == str(package_init)
    assert origins["pkg.sub"]["origin"] == str(subpackage_init)
    assert origins["pkg.sub.mod"]["origin"] == str(module)
    for path in (package_init, subpackage_init, module):
        assert closure.file_hashes[str(path)] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_closure_binds_explicit_dynamic_support_without_static_import(
    secure_tmp_path: Path,
) -> None:
    code_root = secure_tmp_path / "code"
    support_root = secure_tmp_path / "support"
    package = support_root / "dynamic_support"
    code_root.mkdir()
    package.mkdir(parents=True)
    entry = code_root / "entry.py"
    test_file = code_root / "test_source_free.py"
    package_init = package / "__init__.py"
    helper = package / "helper.py"
    entry.write_text("VALUE = 1\n", encoding="ascii")
    test_file.write_text(
        "def test_source_free():\n    assert True\n",
        encoding="ascii",
    )
    package_init.write_text("NAME = 'bound'\n", encoding="ascii")
    helper.write_text("VALUE = 7\n", encoding="ascii")
    attestation = supervisor.run_source_free_tests(
        code_root=code_root,
        test_files=(test_file,),
    )
    closure = supervisor.attest_runtime_closure(
        code_roots=(code_root,),
        entry_files=(entry,),
        config_files=(),
        asset_roots=(),
        test_attestation=attestation,
        support_module_files={
            "dynamic_support": package_init,
            "dynamic_support.helper": helper,
        },
    )
    assert closure.manifest["support_roots"] == [str(package)]
    for module_name, path in {
        "dynamic_support": package_init,
        "dynamic_support.helper": helper,
    }.items():
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert closure.manifest["support_module_files"][module_name] == {
            "path": str(path),
            "sha256": expected,
        }
        assert closure.file_hashes[str(path)] == expected


def test_import_parser_keeps_absolute_from_import_at_top_level(
    secure_tmp_path: Path,
) -> None:
    module = secure_tmp_path / "module.py"
    module.write_text(
        "from json import dumps\nfrom pathlib import Path\n",
        encoding="ascii",
    )
    imports = supervisor._imports_from_source(  # noqa: SLF001
        module, "pkg.module"
    )
    assert {"json", "json.dumps", "pathlib", "pathlib.Path"}.issubset(
        imports
    )
    assert not any(name.startswith("pkg.") for name in imports)


def test_import_parser_preserves_relative_from_import_package(
    secure_tmp_path: Path,
) -> None:
    module = secure_tmp_path / "module.py"
    module.write_text(
        "from .helper import VALUE\n",
        encoding="ascii",
    )
    imports = supervisor._imports_from_source(  # noqa: SLF001
        module, "pkg.module"
    )
    assert imports == {"pkg.helper", "pkg.helper.VALUE"}


def test_same_harness_executes_real_landlock_four_arm_barrier_and_custodian(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    raw = _csv()
    closure = _closure(tmp_path)
    root = tmp_path / "formal"
    with supervisor.FormalSupervisor._source_free_qualification(
        root
    ) as runtime:
        action = _action(runtime, closure, raw, tmp_path)
        assert (
            action.receipt["legacy_freeze_ready_receipt_is_authority"]
            is False
        )
        invocation = runtime.begin_once(action)
        with pytest.raises(
            supervisor.FormalSupervisorError,
            match="secure_output_already_exists",
        ):
            runtime.begin_once(action)
        packs = runtime.materialize_synthetic_packs_once(
            invocation,
            raw=raw,
            expected_topology=_topology(),
        )
        assert packs["label_opened_by_arm"] is False
        assert packs["linkage_opened_by_arm"] is False

        arm_receipts = [
            runtime.run_arm_once(invocation, arm_id=arm_id)
            for arm_id in protocol.ARM_IDS
        ]
        assert {receipt["arm_id"] for receipt in arm_receipts} == set(
            protocol.ARM_IDS
        )
        for arm_id in protocol.ARM_IDS:
            sandbox_spec = runtime.store.read_json(
                "work/arms/"
                f"{invocation.receipt['one_shot_key']}.{arm_id}/"
                "sandbox.spec.json"
            )
            assert set(closure.manifest["runtime_roots"]).issubset(
                set(sandbox_spec["code_roots"])
            )
            sandbox_receipt = runtime.store.read_json(
                "work/arms/"
                f"{invocation.receipt['one_shot_key']}.{arm_id}/"
                "sandbox.safe.json"
            )
            assert sandbox_receipt["label_denial_errno"] in {
                1,
                13,
            }
            assert sandbox_receipt["linkage_denial_errno"] in {
                1,
                13,
            }
            assert sandbox_receipt["landlock_abi"] >= 3

        barrier = runtime.seal_four_arm_barrier_once(invocation)
        assert barrier["status"] == (
            "ALL_FOUR_ARMS_SEALED_BEFORE_LABEL_OPEN"
        )
        assert barrier["label_opened"] is False
        score = runtime.run_fixed_scorer_once(invocation)
        assert score["status"] == "FIXED_OFFLINE_SCORER_COMPLETED"
        assert score["aggregate_result"]["status"] == (
            "SYNTHETIC_AGGREGATE_ONLY"
        )
        assert score["item_content_emitted"] is False


def test_barrier_refuses_missing_arm_before_label_open(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    raw = _csv()
    closure = _closure(tmp_path)
    with supervisor.FormalSupervisor._source_free_qualification(
        tmp_path / "formal"
    ) as runtime:
        action = _action(runtime, closure, raw, tmp_path)
        invocation = runtime.begin_once(action)
        runtime.materialize_synthetic_packs_once(
            invocation, raw=raw, expected_topology=_topology()
        )
        runtime.run_arm_once(invocation, arm_id=protocol.ARM_IDS[0])
        with pytest.raises(supervisor.FormalSupervisorError):
            runtime.seal_four_arm_barrier_once(invocation)
        key = invocation.receipt["one_shot_key"]
        assert not runtime.store.exists(
            f"state/attempts/{key}.score.safe.json"
        )


def test_formal_official_path_is_unreachable_from_qualification(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    raw = _csv()
    closure = _closure(tmp_path)
    with supervisor.FormalSupervisor._source_free_qualification(
        tmp_path / "formal"
    ) as runtime:
        action = _action(runtime, closure, raw, tmp_path)
        invocation = runtime.begin_once(action)
        with pytest.raises(
            supervisor.FormalSupervisorError,
            match="official_source_forbidden_in_qualification",
        ):
            runtime.materialize_official_packs_once(invocation)


def test_no_public_prediction_constructor_or_label_loader_hook() -> None:
    assert not hasattr(supervisor, "make_prediction_pack")
    assert "label_loader" not in Path(supervisor.__file__).read_text(
        encoding="utf-8"
    )
    assert supervisor.FORMAL_ROOT == Path(
        "/var/tmp/gscl_arn_intrinsic_formal_v1"
    )
    formal_parameters = set(
        inspect.signature(
            supervisor.FormalSupervisor.freeze_internal_factory_action_once
        ).parameters
    )
    assert not formal_parameters.intersection(
        {
            "arm_commands",
            "scorer_command",
            "predictions",
            "prepared_results",
            "freeze_ready",
        }
    )
    execution_parameters = set(
        inspect.signature(
            supervisor.FormalSupervisor.run_internal_factory_once
        ).parameters
    )
    assert execution_parameters == {"self", "invocation"}
    execution_source = inspect.getsource(
        supervisor.FormalSupervisor.run_internal_factory_once
    )
    assert "admit_story_only_pack_qualification_only" not in execution_source
    assert "load_trusted_story_only_input_pack" in execution_source
    assert "_run_internal_sandbox_command" in execution_source
    assert "ThreadPoolExecutor" in execution_source
    assert "closed_choice_multi_pack_worker" in execution_source
    factory_command_source = execution_source.split(
        "factory_command = (", 1
    )[1].split("factory_raw, factory_sandbox", 1)[0]
    assert 'action["minilm_asset_manifest"]' in factory_command_source
    assert "minilm_manifest_relative" not in factory_command_source
    factory_sandbox_source = execution_source.split(
        "factory_raw, factory_sandbox", 1
    )[1].split("factory_output =", 1)[0]
    assert 'Path(action["minilm_asset_manifest"])' in (
        factory_sandbox_source
    )
    assert "qwen_actual_canary_lineage_terminal" in formal_parameters
    assert "qwen_runtime_qualification_receipts" in formal_parameters
    assert (
        "internal_factory_qualification_receipt"
        in formal_parameters
    )
    assert "minilm_target_manifest" in formal_parameters
    scorer_source = inspect.getsource(
        supervisor.FormalSupervisor._run_internal_fixed_scorer_once
    )
    assert "protocol._score_aggregates" in scorer_source
    assert "labels_open.claim.json" in scorer_source
    assert "online_or_api_evaluator_used" in scorer_source
    for component, expected in supervisor._STABLE_EXTRACTOR_SHA256S.items():
        assert (
            hashlib.sha256(
                supervisor._INTERNAL_FORMAL_IMPLEMENTATION_PATHS[
                    component
                ].read_bytes()
            ).hexdigest()
            == expected
        )
    assert set(
        supervisor._STABLE_QUALIFICATION_TEST_SHA256S  # noqa: SLF001
    ) == set(
        supervisor._INTERNAL_QUALIFICATION_TEST_PATHS  # noqa: SLF001
    )
    for component, expected in (
        supervisor._STABLE_QUALIFICATION_TEST_SHA256S.items()  # noqa: SLF001
    ):
        path = supervisor._INTERNAL_QUALIFICATION_TEST_PATHS[  # noqa: SLF001
            component
        ]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    for module_name, expected in (
        supervisor._STABLE_SUPPORT_MODULE_SHA256S.items()  # noqa: SLF001
    ):
        path = supervisor._INTERNAL_SUPPORT_MODULE_PATHS[  # noqa: SLF001
            module_name
        ]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_gpu_allowlist_is_exact_and_never_admits_dev_root() -> None:
    candidates = supervisor._gpu_device_candidates(0)
    assert set(candidates).issuperset(
        {
            Path("/dev/nvidia0"),
            Path("/dev/nvidia1"),
            Path("/dev/nvidiactl"),
            Path("/dev/nvidia-uvm"),
        }
    )
    assert Path("/dev") not in candidates
    assert supervisor.QWEN_CUDA_VISIBLE_DEVICES == ("0", "1")
    assert supervisor._validate_internal_environment_overrides(
        {"CUDA_VISIBLE_DEVICES": "0"}
    ) == {"CUDA_VISIBLE_DEVICES": "0"}
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="internal_environment_override_invalid",
    ):
        supervisor._validate_internal_environment_overrides(
            {"CUDA_VISIBLE_DEVICES": "0,1"}
        )


def test_gpu_landlock_uses_numeric_same_process_task_path(
    monkeypatch: pytest.MonkeyPatch,
    secure_tmp_path: Path,
) -> None:
    recorded_rules: list[tuple[Path, int]] = []
    syscall_results = iter((6, 91, 0))

    class FakeLibc:
        @staticmethod
        def prctl(*_: object) -> int:
            return 0

    monkeypatch.setattr(
        supervisor.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: FakeLibc(),
    )
    monkeypatch.setattr(
        supervisor,
        "_landlock_syscall",
        lambda *_args, **_kwargs: next(syscall_results),
    )
    monkeypatch.setattr(
        supervisor,
        "_add_landlock_rule",
        lambda *, path, allowed_access, **_kwargs: (
            recorded_rules.append((path, allowed_access))
        ),
    )
    monkeypatch.setattr(supervisor.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        supervisor,
        "_validated_gpu_device_rows",
        lambda _index: [
            {
                "gid": 0,
                "major": 195,
                "minor": 0,
                "mode": 0o666,
                "path": "/dev/nvidia0",
                "uid": 0,
            }
        ],
    )
    receipt = supervisor._apply_landlock(  # noqa: SLF001
        read_execute_roots=(secure_tmp_path,),
        work_root=secure_tmp_path,
        denial_probes={},
        gpu_device_index=0,
    )
    numeric_task_root = Path(f"/proc/{os.getpid()}/task")
    recorded_paths = [path for path, _ in recorded_rules]
    assert numeric_task_root in recorded_paths
    assert Path("/proc/self/task") not in recorded_paths
    task_access = dict(recorded_rules)[numeric_task_root]
    assert task_access & supervisor.LANDLOCK_ACCESS_FS_WRITE_FILE
    assert task_access & supervisor.LANDLOCK_ACCESS_FS_TRUNCATE
    assert (
        supervisor._safe_absolute_path(  # noqa: SLF001
            numeric_task_root, allow_file=False
        )
        == numeric_task_root
    )
    current = Path("/")
    for component in numeric_task_root.parts[1:]:
        current /= component
        assert not current.is_symlink()
    assert receipt["gpu_proc_self_task_write_allowed"] is True


def test_balanced_triplet_plan_is_round_robin_bounded_and_deterministic() -> None:
    stories = tuple(
        (f"opaque-{item}", role, f"story-{item}-{role}")
        for item in range(44)
        for role in ("query", "first_choice", "second_choice")
    )
    first = supervisor._balanced_triplet_batch_plan(  # noqa: SLF001
        stories,
        shard_count=2,
        maximum_story_count=63,
    )
    second = supervisor._balanced_triplet_batch_plan(  # noqa: SLF001
        stories,
        shard_count=2,
        maximum_story_count=63,
    )
    assert first == second
    assert all(
        0 < len(row["stories"]) <= 63
        and len(row["stories"]) == 3 * len(row["item_indices"])
        for row in first
    )
    by_shard = {
        shard: tuple(
            item
            for row in first
            if row["shard_index"] == shard
            for item in row["item_indices"]
        )
        for shard in (0, 1)
    }
    assert by_shard[0] == tuple(range(0, 44, 2))
    assert by_shard[1] == tuple(range(1, 44, 2))
    assert abs(len(by_shard[0]) - len(by_shard[1])) <= 1
    two_items = supervisor._balanced_triplet_batch_plan(  # noqa: SLF001
        stories[:6], shard_count=2
    )
    assert tuple(
        (row["shard_index"], row["item_indices"])
        for row in two_items
    ) == ((0, (0,)), (1, (1,)))
    malformed = list(stories[:6])
    malformed[1] = (
        malformed[1][0],
        "second_choice",
        malformed[1][2],
    )
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="extractor_triplet_plan_invalid",
    ):
        supervisor._balanced_triplet_batch_plan(  # noqa: SLF001
            tuple(malformed), shard_count=2
        )


def test_qwen_batch_safe_receipts_are_strict_and_actual_bound() -> None:
    row = {
        "batch_id": "arn-0000",
        "decision_elapsed_ns": 1,
        "decision_invalid_count": 1,
        "decision_valid_count": 2,
        "input_file_sha256": "1" * 64,
        "input_pack_commitment": "2" * 64,
        "output_file_sha256": "3" * 64,
        "selection_receipt_commitment": "4" * 64,
        "selection_receipt_count": 2,
        "sequence": 0,
        "story_count": 3,
        "valid_wire_completion_token_count_maximum": 12,
        "valid_wire_completion_token_count_sum": 20,
    }
    assert supervisor._validated_qwen_batch_rows(
        {"batch_count": 1, "batches": [row]}
    ) == {0: row}
    invalid = {**row, "decision_valid_count": 3}
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="qwen_runtime_batch_receipts_invalid",
    ):
        supervisor._validated_qwen_batch_rows(
            {"batch_count": 1, "batches": [invalid]}
        )
    extra = {**row, "untrusted": "field"}
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="qwen_runtime_batch_receipts_invalid",
    ):
        supervisor._validated_qwen_batch_rows(
            {"batch_count": 1, "batches": [extra]}
        )
    execution_source = inspect.getsource(
        supervisor.FormalSupervisor.run_internal_factory_once
    )
    assert "qwen_runtime_batch_binding_invalid" in execution_source
    assert (
        'decoded_output.get("execution_closure")'
        in execution_source
    )
    validator_source = inspect.getsource(
        supervisor._validate_qwen_runtime_safe_receipt
    )
    assert (
        "hashlib.sha256(\n"
        "            _canonical_bytes(runtime_receipt)\n"
        "        ).hexdigest()"
        in validator_source
    )
    assert "_content_hash(runtime_receipt)" not in validator_source


def test_closed_choice_actual_canary_lineage_is_exact_bound() -> None:
    path = (
        supervisor._RECONSTRUCTION_ROOT  # noqa: SLF001
        / (
            "manifests/"
            "gscl_closed_choice_actual_canary_"
            "lineage_terminal_ext4_20260730.json"
        )
    )
    raw = path.read_bytes()
    binding = (
        supervisor._validate_closed_choice_actual_canary_lineage(  # noqa: SLF001
            raw,
            expected_model_manifest_sha256=(
                "970fd38542fc3e00f9c98e2efda0bcb4"
                "e9355e0974f0a9cd5ae38cc57a82e658"
            ),
        )
    )
    assert binding["file_sha256"] == (
        supervisor.CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256
    )
    assert binding["self_sha256"] == (
        supervisor.CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_SELF_SHA256
    )
    assert binding["worker_sha256"] == (
        supervisor.CLOSED_CHOICE_WORKER_SHA256
    )
    assert binding["lineage_model_weight_load_count"] == 3
    assert (
        binding["successful_teacher_forced_qualification_run_count"]
        == 2
    )
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="closed_choice_actual_canary_lineage_hash_invalid",
    ):
        supervisor._validate_closed_choice_actual_canary_lineage(  # noqa: SLF001
            raw + b" ",
            expected_model_manifest_sha256=(
                "970fd38542fc3e00f9c98e2efda0bcb4"
                "e9355e0974f0a9cd5ae38cc57a82e658"
            ),
        )


def test_closed_choice_safe_receipt_binds_byte_preimages_and_selection_rows() -> None:
    model_manifest = "a" * 64
    runtime = {
        "schema": (
            "gscl_narrative_closed_choice_qwen_runtime_v1."
            "runtime_receipt.v1"
        ),
        "free_form_generation_count": 0,
        "score_operation": (
            "teacher_forced_forward_log_softmax"
        ),
    }
    runtime_sha = hashlib.sha256(
        supervisor._canonical_bytes(runtime)  # noqa: SLF001
    ).hexdigest()
    double = {
        "schema": (
            "gscl_narrative_closed_choice_qwen_runtime_v1."
            "double_run_receipt.v1"
        ),
        "free_form_generation_count": 0,
        "repeat_count": 2,
        "repeat_exact": True,
        "runtime_receipt_sha256": runtime_sha,
    }
    double_sha = hashlib.sha256(
        supervisor._canonical_bytes(double)  # noqa: SLF001
    ).hexdigest()
    teacher = "b" * 64
    selection = "c" * 64
    batch = {
        "batch_id": "arn-0000",
        "decision_elapsed_ns": 1,
        "decision_invalid_count": 0,
        "decision_valid_count": 1,
        "input_file_sha256": "d" * 64,
        "input_pack_commitment": "e" * 64,
        "output_file_sha256": "f" * 64,
        "selection_receipt_commitment": selection,
        "selection_receipt_count": 1,
        "sequence": 0,
        "story_count": 1,
        "valid_wire_completion_token_count_maximum": 12,
        "valid_wire_completion_token_count_sum": 12,
    }
    execution = {
        "model_asset_manifest_sha256": model_manifest,
        "model_runtime_closure_sha256": (
            supervisor._content_hash(  # noqa: SLF001
                {
                    "double_run_receipt_sha256": double_sha,
                    "model_asset_manifest_sha256": model_manifest,
                    "runtime_receipt_sha256": runtime_sha,
                    "teacher_forced_backend_commitment": teacher,
                }
            )
        ),
        "parser_closure_sha256": (
            supervisor.CLOSED_CHOICE_PARSER_SHA256
        ),
        "prompt_sha256": (
            supervisor.CLOSED_CHOICE_PROMPT_SHA256
        ),
        "target_double_run_receipt_sha256": double_sha,
    }
    distributions = [
        {
            "closure_sha256": hashlib.sha256(
                name.encode("ascii")
            ).hexdigest(),
            "critical": True,
            "distribution": name,
            "loaded_top_level_modules": [],
            "version": "1",
        }
        for name in sorted(
            supervisor._QWEN_CRITICAL_DISTRIBUTIONS  # noqa: SLF001
        )
    ]
    body = {
        "batch_count": 1,
        "batches": [batch],
        "claim_scope": (
            "untrusted_grounded_closed_choice_proposal_only"
        ),
        "execution_closure": execution,
        "free_form_generation_count": 0,
        "input_manifest_file_sha256": "1" * 64,
        "lineage": "source_free_qualification",
        "loaded_distribution_closure_sha256": (
            supervisor._content_hash(distributions)  # noqa: SLF001
        ),
        "loaded_distributions": distributions,
        "logical_gpu_binding": {
            "cuda_visible_devices": "0",
            "logical_compute_capability": [8, 6],
            "logical_device_count": 1,
            "logical_device_index": 0,
            "logical_device_name": "Synthetic GPU",
            "logical_device_uuid": "GPU-synthetic-0",
            "model_parameter_logical_device_indices": [0],
        },
        "model_asset_manifest_file_sha256": model_manifest,
        "runtime_receipt": runtime,
        "runtime_receipt_sha256": runtime_sha,
        "schema": supervisor.QWEN_MULTI_SAFE_RECEIPT_SCHEMA,
        "score_operation": (
            "teacher_forced_forward_log_softmax"
        ),
        "selection_receipt_commitments_sha256": (
            supervisor._content_hash(  # noqa: SLF001
                [
                    {
                        "selection_receipt_commitment": selection,
                        "selection_receipt_count": 1,
                        "sequence": 0,
                    }
                ]
            )
        ),
        "selection_receipt_count": 1,
        "single_model_load_count": 1,
        "source_content_supplied": False,
        "target_double_run_receipt": double,
        "target_double_run_receipt_sha256": double_sha,
        "teacher_forced_backend_commitment": teacher,
        "worker_version": (
            "gscl_narrative_closed_choice_multi_pack_worker_v1"
        ),
    }
    receipt = {
        **body,
        "self_sha256": supervisor._content_hash(body),  # noqa: SLF001
    }
    binding = supervisor._validate_qwen_runtime_safe_receipt(  # noqa: SLF001
        receipt,
        expected_model_manifest_sha256=model_manifest,
        expected_visible_device="0",
        expected_lineage="source_free_qualification",
    )
    assert binding["runtime_receipt_sha256"] == runtime_sha
    bad_body = {
        **body,
        "runtime_receipt_sha256": (
            supervisor._content_hash(runtime)  # noqa: SLF001
        ),
    }
    bad = {
        **bad_body,
        "self_sha256": supervisor._content_hash(  # noqa: SLF001
            bad_body
        ),
    }
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="qwen_runtime_receipt_preimage_invalid",
    ):
        supervisor._validate_qwen_runtime_safe_receipt(  # noqa: SLF001
            bad,
            expected_model_manifest_sha256=model_manifest,
            expected_visible_device="0",
            expected_lineage="source_free_qualification",
        )


def test_factory_output_receipt_is_complete_and_batch_bound() -> None:
    opaque = "a" * 64
    error_prediction = {
        "opaque_item_id": opaque,
        "disposition": "ERROR",
        "selected_choice": None,
        "error_code": "ARM_RUNTIME_ERROR",
    }
    batch_receipt = {
        "batch_id": "arn-0000",
        "execution_closure_commitment": "1" * 64,
        "input_file_sha256": "2" * 64,
        "input_pack_commitment": "3" * 64,
        "output_file_sha256": "4" * 64,
        "sequence": 0,
        "story_count": 3,
    }
    body = {
        "schema": "fixture.private_output.v1",
        "status": "PRIVATE_ALL_FOUR_ITEM_RESULTS_RECOMPUTED",
        "lineage": "formal_frozen_assets",
        "predictor_pack_file_sha256": "5" * 64,
        "extractor_batch_receipts": [batch_receipt],
        "factory_receipt_self_hash": None,
        "encoder_binding": {"fixture": True},
        "by_arm": {
            arm_id: [dict(error_prediction)]
            for arm_id in protocol.ARM_IDS
        },
        "private_item_recomputation_receipts": [],
        "item_count": 1,
        "error_item_count": 1,
        "caller_predictions_accepted": False,
        "caller_commitments_accepted": False,
        "item_content_emitted": False,
    }
    output = {
        **body,
        "self_hash": supervisor._content_hash(body),
    }
    normalized = supervisor._validate_factory_output_receipt(
        output,
        expected_schema="fixture.private_output.v1",
        expected_lineage="formal_frozen_assets",
        expected_predictor_file_sha256="5" * 64,
        expected_batch_receipts=[batch_receipt],
        expected_item_ids={opaque},
    )
    assert set(normalized) == set(protocol.ARM_IDS)
    tampered = {
        **output,
        "extractor_batch_receipts": [
            {**batch_receipt, "output_file_sha256": "6" * 64}
        ],
    }
    tampered_body = dict(tampered)
    tampered_body.pop("self_hash")
    tampered["self_hash"] = supervisor._content_hash(
        tampered_body
    )
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="formal_factory_output_invalid",
    ):
        supervisor._validate_factory_output_receipt(
            tampered,
            expected_schema="fixture.private_output.v1",
            expected_lineage="formal_frozen_assets",
            expected_predictor_file_sha256="5" * 64,
            expected_batch_receipts=[batch_receipt],
            expected_item_ids={opaque},
        )
    impossible = {
        **output,
        "by_arm": {
            **output["by_arm"],
            protocol.ARM_IDS[0]: [
                {
                    **error_prediction,
                    "error_code": "NO_VALID_PREDICTION",
                }
            ],
        },
    }
    impossible_body = dict(impossible)
    impossible_body.pop("self_hash")
    impossible["self_hash"] = supervisor._content_hash(
        impossible_body
    )
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="formal_factory_error_set_invalid",
    ):
        supervisor._validate_factory_output_receipt(
            impossible,
            expected_schema="fixture.private_output.v1",
            expected_lineage="formal_frozen_assets",
            expected_predictor_file_sha256="5" * 64,
            expected_batch_receipts=[batch_receipt],
            expected_item_ids={opaque},
        )


def _synthetic_outer_systemd_attestation(
    writable_root: Path,
) -> dict[str, object]:
    full_contract = supervisor._outer_systemd_full_contract(
        writable_root
    )
    filesystem_probe = (
        supervisor._outer_filesystem_namespace_probe(
            writable_root
        )
    )
    properties = {
        **full_contract,
        "ActiveState": "active",
        "ControlGroup": (
            "/user.slice/user-1001.slice/user@1001.service/"
            "app.slice/gscl-qualification.service"
        ),
        "FragmentPath": (
            "/home/erzhu419/.config/systemd/user/"
            "gscl-qualification.service"
        ),
        "Id": "gscl-qualification.service",
        "InvocationID": "a" * 32,
        "MainPID": "1234",
        "NRestarts": "0",
        "SubState": "running",
    }
    probe = {
        "AF_INET_socket_creation_denied": True,
        "AF_INET_socket_denial_errno": 97,
        "AF_INET6_socket_creation_denied": True,
        "AF_INET6_socket_denial_errno": 97,
        "AF_UNIX_socket_creation_allowed": True,
        "network_endpoint_contacted": False,
    }
    stable = {
        "contract_self_hash": supervisor._content_hash(
            full_contract
        ),
        "common_contract_self_hash": supervisor._content_hash(
            supervisor.OUTER_SYSTEMD_CONTRACT
        ),
        "control_group_sha256": "1" * 64,
        "filesystem_namespace_probe": filesystem_probe,
        "fragment_source_file_sha256": "2" * 64,
        "invocation_id": "a" * 32,
        "main_pid": 1234,
        "network_family_probe": probe,
        "nrestarts": 0,
        "systemctl_file_sha256": "3" * 64,
        "unit_id": "gscl-qualification.service",
        "writable_root": str(writable_root),
    }
    body = {
        "schema": supervisor.OUTER_SYSTEMD_ATTESTATION_SCHEMA,
        "active_state": "active",
        "sub_state": "running",
        "common_contract": dict(supervisor.OUTER_SYSTEMD_CONTRACT),
        "common_contract_self_hash": stable[
            "common_contract_self_hash"
        ],
        "contract": full_contract,
        "contract_self_hash": stable["contract_self_hash"],
        "writable_root": str(writable_root),
        "private_tmp_tradeoff": supervisor._OUTER_PRIVATE_TMP_TRADEOFF,
        "filesystem_namespace_probe": filesystem_probe,
        "properties": properties,
        "unit_id": "gscl-qualification.service",
        "invocation_id": "a" * 32,
        "main_pid": 1234,
        "nrestarts": 0,
        "control_group_sha256": "1" * 64,
        "fragment_source_file_sha256": "2" * 64,
        "systemctl_file_sha256": "3" * 64,
        "systemd_show_stdout_sha256": "4" * 64,
        "network_family_probe": probe,
        "stable_binding_sha256": supervisor._content_hash(stable),
    }
    return {**body, "self_hash": supervisor._content_hash(body)}


def test_outer_systemd_contract_is_exact_and_tamper_evident(
    secure_tmp_path: Path,
) -> None:
    receipt = _synthetic_outer_systemd_attestation(
        secure_tmp_path
    )
    assert (
        supervisor._validate_outer_systemd_attestation(
            receipt,
            expected_writable_root=secure_tmp_path,
        )
        == receipt
    )
    assert receipt["common_contract"] == {
        "CPUQuotaPerSecUSec": "4s",
        "CPUWeight": "25",
        "IPAddressDeny": "::/0 0.0.0.0/0",
        "IOSchedulingClass": "3",
        "IOWeight": "25",
        "KillMode": "control-group",
        "MemoryHigh": "25769803776",
        "MemoryMax": "34359738368",
        "MemorySwapMax": "0",
        "Nice": "10",
        "NoNewPrivileges": "yes",
        "PrivateDevices": "no",
        "PrivateTmp": "no",
        "ProtectSystem": "no",
        "ReadOnlyPaths": "",
        "ReadWritePaths": "",
        "Restart": "no",
        "RestrictAddressFamilies": "AF_UNIX",
        "RuntimeMaxUSec": "infinity",
        "TasksMax": "96",
        "TimeoutStartUSec": "infinity",
        "Type": "oneshot",
        "UMask": "0077",
    }
    assert receipt["writable_root"] == str(secure_tmp_path)
    assert receipt["filesystem_namespace_probe"][
        "root_directory_open_allowed"
    ] is True
    tampered = dict(receipt)
    tampered["properties"] = {
        **receipt["properties"],
        "IPAddressDeny": "no",
    }
    tampered_body = dict(tampered)
    tampered_body.pop("self_hash")
    tampered["self_hash"] = supervisor._content_hash(tampered_body)
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="outer_systemd_attestation_invalid",
    ):
        supervisor._validate_outer_systemd_attestation(
            tampered,
            expected_writable_root=secure_tmp_path,
        )


@pytest.mark.parametrize(
    "reported",
    ("0.0.0.0/0 ::/0", "::/0 0.0.0.0/0"),
)
def test_systemd_show_normalises_unordered_ip_deny_set(
    reported: str,
) -> None:
    properties = {
        **supervisor.OUTER_SYSTEMD_CONTRACT,
        "ActiveState": "activating",
        "ControlGroup": (
            "/user.slice/user-1001.slice/user@1001.service/"
            "app.slice/gscl-qualification.service"
        ),
        "FragmentPath": (
            "/home/erzhu419/.config/systemd/user/"
            "gscl-qualification.service"
        ),
        "Id": "gscl-qualification.service",
        "InvocationID": "a" * 32,
        "MainPID": "1234",
        "NRestarts": "0",
        "SubState": "start",
        "IPAddressDeny": reported,
    }
    raw = (
        "".join(
            f"{key}={properties[key]}\n"
            for key in supervisor._OUTER_SYSTEMD_LIVE_PROPERTIES  # noqa: SLF001
        )
    ).encode("ascii")
    parsed = supervisor._parse_systemd_show(raw)  # noqa: SLF001
    assert parsed["IPAddressDeny"] == (
        supervisor.OUTER_SYSTEMD_CONTRACT["IPAddressDeny"]
    )


def test_internal_sandbox_pythonpath_binds_workspace_and_reconstruction() -> None:
    expected_roots = (
        supervisor._WORKSPACE_ROOT,  # noqa: SLF001
        supervisor._RECONSTRUCTION_ROOT,  # noqa: SLF001
    )
    expected = os.pathsep.join(map(str, expected_roots))
    assert supervisor._LOCAL_PYTHONPATH_ROOTS == expected_roots  # noqa: SLF001
    assert supervisor._LOCAL_PYTHONPATH == expected  # noqa: SLF001
    assert str(
        supervisor._WORKSPACE_ROOT / "assumption_os"  # noqa: SLF001
    ) in {
        str(path.parent)
        for path in (
            supervisor._INTERNAL_SUPPORT_MODULE_PATHS.values()  # noqa: SLF001
        )
    }
    for function in (
            supervisor.FormalSupervisor._run_internal_sandbox_command,
            supervisor.FormalSupervisor.run_arm_once,
            supervisor._sandbox_child,  # noqa: SLF001
        ):
        assert "_LOCAL_PYTHONPATH" in inspect.getsource(function)


def test_factory_encoder_binding_accepts_frozen_finite_canary_diagnostics() -> None:
    runtime = {
        "schema": supervisor.GSCL_MINILM_RUNTIME_SCHEMA,
        "status": "verified_exact_gscl_target_local_minilm_runtime",
        "formal_source_or_rows_accessed": False,
        "labels_accessed": False,
        "network_calls": 0,
    }
    canary = {
        "schema": supervisor.GSCL_MINILM_CANARY_SCHEMA,
        "status": "passed_target_local_repeat_exact_canary",
        "repeat_count": 2,
        "repeat_byte_exact": True,
        "repeat_elementwise_exact": True,
        "cross_hardware_byte_identity_claimed": False,
        "formal_source_or_rows_accessed": False,
        "labels_accessed": False,
        "network_calls": 0,
        "portable_canary_receipt": {
            "maximum_observed_row_l2_norm_error": 0.0,
            "per_row_l2_norm_maximum_error": 1e-07,
        },
    }

    def encoded(value: object) -> str:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    runtime_json = encoded(runtime)
    canary_json = encoded(canary)
    binding = {
        "encoder_exact_type": (
            "replication_runtime.gscl_minilm_portable_v1.binding."
            "GSCLPortableOfflineMiniLMEncoder"
        ),
        "encoder_runtime_receipt_json": runtime_json,
        "encoder_runtime_receipt_sha256": hashlib.sha256(
            runtime_json.encode("ascii")
        ).hexdigest(),
        "encoder_canary_receipt_json": canary_json,
        "encoder_canary_receipt_sha256": hashlib.sha256(
            canary_json.encode("ascii")
        ).hexdigest(),
    }
    assert supervisor._validate_factory_encoder_binding(  # noqa: SLF001
        binding
    ) == binding


def test_factory_encoder_binding_compares_content_not_custody_path() -> None:
    target_file = "a" * 64
    target_self = "b" * 64

    def binding(path: str, *, runtime_marker: str = "same") -> dict[str, str]:
        runtime = {
            "schema": supervisor.GSCL_MINILM_RUNTIME_SCHEMA,
            "status": "verified_exact_gscl_target_local_minilm_runtime",
            "formal_source_or_rows_accessed": False,
            "labels_accessed": False,
            "network_calls": 0,
            "target_manifest_path": path,
            "target_manifest_file_sha256": target_file,
            "target_manifest_self_sha256": target_self,
            "portable_runtime_receipt": {
                "runtime_marker": runtime_marker,
            },
        }
        canary = {
            "schema": supervisor.GSCL_MINILM_CANARY_SCHEMA,
            "status": "passed_target_local_repeat_exact_canary",
            "repeat_count": 2,
            "repeat_byte_exact": True,
            "repeat_elementwise_exact": True,
            "cross_hardware_byte_identity_claimed": False,
            "formal_source_or_rows_accessed": False,
            "labels_accessed": False,
            "network_calls": 0,
            "target_manifest_self_sha256": target_self,
        }
        runtime_json = json.dumps(
            runtime,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        canary_json = json.dumps(
            canary,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return {
            "encoder_exact_type": (
                "replication_runtime.gscl_minilm_portable_v1.binding."
                "GSCLPortableOfflineMiniLMEncoder"
            ),
            "encoder_runtime_receipt_json": runtime_json,
            "encoder_runtime_receipt_sha256": hashlib.sha256(
                runtime_json.encode("ascii")
            ).hexdigest(),
            "encoder_canary_receipt_json": canary_json,
            "encoder_canary_receipt_sha256": hashlib.sha256(
                canary_json.encode("ascii")
            ).hexdigest(),
        }

    qualification = binding("/qualification/minilm.target.json")
    formal_path = Path("/formal/item_factory/minilm.target.json")
    observed = binding(str(formal_path))
    assert supervisor._factory_encoder_bindings_content_equivalent(  # noqa: SLF001
        qualification,
        observed,
        expected_observed_target_manifest_path=formal_path,
        expected_target_file_sha256=target_file,
        expected_target_self_sha256=target_self,
    )
    assert not supervisor._factory_encoder_bindings_content_equivalent(  # noqa: SLF001
        qualification,
        binding(str(formal_path), runtime_marker="changed"),
        expected_observed_target_manifest_path=formal_path,
        expected_target_file_sha256=target_file,
        expected_target_self_sha256=target_self,
    )
    assert not supervisor._factory_encoder_bindings_content_equivalent(  # noqa: SLF001
        qualification,
        observed,
        expected_observed_target_manifest_path=Path(
            "/formal/other/minilm.target.json"
        ),
        expected_target_file_sha256=target_file,
        expected_target_self_sha256=target_self,
    )


def test_landlock_allowlist_contains_every_python_import_root() -> None:
    run_arm_source = inspect.getsource(
        supervisor.FormalSupervisor.run_arm_once
    )
    run_factory_source = inspect.getsource(
        supervisor.FormalSupervisor.run_internal_factory_once
    )
    for source in (run_arm_source, run_factory_source):
        assert "_LOCAL_PYTHONPATH_ROOTS" in source


def test_landlock_child_injects_direct_parent_authority() -> None:
    source = inspect.getsource(supervisor._sandbox_child)  # noqa: SLF001
    assert (
        "SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY"
        in source
    )
    assert (
        "SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY"
        in source
    )
    assert (
        supervisor.SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY
        not in supervisor.OFFLINE_ENVIRONMENT
    )
    assert (
        len(supervisor.SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY)
        == 64
    )


def test_landlock_work_root_supports_private_tempfile_lifecycle() -> None:
    assert (
        supervisor.LANDLOCK_WORK_ACCESS
        & supervisor.LANDLOCK_ACCESS_FS_REMOVE_FILE
    )
    assert (
        supervisor.LANDLOCK_WORK_ACCESS
        & supervisor.LANDLOCK_ACCESS_FS_REMOVE_DIR
    )
    assert (
        supervisor.LANDLOCK_WORK_ACCESS
        & supervisor.LANDLOCK_ACCESS_FS_TRUNCATE
    )
    assert not (
        supervisor.LANDLOCK_READ_EXECUTE_ACCESS
        & supervisor.LANDLOCK_ACCESS_FS_REMOVE_FILE
    )
    assert not (
        supervisor.LANDLOCK_READ_EXECUTE_ACCESS
        & supervisor.LANDLOCK_ACCESS_FS_REMOVE_DIR
    )
    assert not (
        supervisor.LANDLOCK_READ_EXECUTE_ACCESS
        & supervisor.LANDLOCK_ACCESS_FS_TRUNCATE
    )


def test_outer_systemd_unit_selects_leaf_below_user_manager() -> None:
    realistic = (
        "/user.slice/user-1001.slice/user@1001.service/"
        "app.slice/gscl-qualification.service"
    )
    assert (
        supervisor._outer_service_unit_from_cgroup(realistic)
        == "gscl-qualification.service"
    )
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="outer_systemd_service_unit_invalid",
    ):
        supervisor._outer_service_unit_from_cgroup(
            "/user.slice/user-1001.slice/user@1001.service"
        )
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="outer_systemd_service_unit_invalid",
    ):
        supervisor._outer_service_unit_from_cgroup(
            realistic + "/nested.scope"
        )


def test_outer_systemd_attestation_binds_controller_as_main_pid() -> None:
    source = inspect.getsource(
        supervisor._attest_current_outer_systemd_service
    )
    assert (
        'int(properties["MainPID"]) != os.getpid()'
        in source
    )


def test_internal_factory_qualification_accepts_no_claimed_runtime_result() -> None:
    parameters = set(
        inspect.signature(
            supervisor.FormalSupervisor
            .freeze_internal_factory_qualification_action_once
        ).parameters
    )
    assert parameters == {
        "self",
        "closure",
        "freeze_commitments",
        "qwen_model_root",
        "qwen_model_manifest",
        "qwen_actual_canary_lineage_terminal",
        "minilm_model_root",
        "minilm_asset_manifest",
        "minilm_target_manifest",
        "source_sha256",
    }
    assert not parameters.intersection(
        {
            "arm_commands",
            "predictions",
            "prepared_results",
            "runtime_receipts",
            "scorer_command",
        }
    )
    source = inspect.getsource(
        supervisor.FormalSupervisor.seal_four_arm_barrier_once
    )
    assert '"official_source_content_supplied_to_model": False' in source
    assert '"public_synthetic_content_supplied_to_model": True' in source
    assert "source_content_supplied_to_qwen" not in source
