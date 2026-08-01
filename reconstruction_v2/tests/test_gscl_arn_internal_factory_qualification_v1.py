from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    gscl_arn_formal_supervisor_v1 as supervisor,
)
from assumption_agent.benchmarks import (
    gscl_arn_internal_factory_qualification_v1 as runner,
)
from assumption_agent.gscl_arn_raw_adapter_v1 import (
    parse_arn_csv_bytes,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_worker,
)


ROOT = Path(__file__).resolve().parents[1]
SERVICE = (
    ROOT
    / "manifests/gscl_arn_internal_factory_qualification_v1.service"
)


def test_public_fixture_is_fixed_parseable_and_closed_choice_feasible() -> None:
    assert hashlib.sha256(runner.PUBLIC_FIXTURE_BYTES).hexdigest() == (
        runner.PUBLIC_FIXTURE_SHA256
    )
    rows = parse_arn_csv_bytes(
        runner.PUBLIC_FIXTURE_BYTES,
        expected_topology=runner.PUBLIC_TOPOLOGY,
    )
    assert len(rows) == 2
    for row in rows:
        for story in (
            row.query_narrative,
            row.first_choice,
            row.second_choice,
        ):
            spans = closed_choice_worker._catalog_spans(story)  # noqa: SLF001
            assert closed_choice_worker._feasible_anchors(spans)  # noqa: SLF001


def test_runner_api_accepts_assets_but_no_source_result_or_score() -> None:
    assert set(inspect.signature(runner.run_qualification).parameters) == {
        "root",
        "qwen_model_root",
        "qwen_model_manifest",
        "qwen_actual_canary_lineage_terminal",
        "minilm_model_root",
        "minilm_asset_manifest",
    }
    source = inspect.getsource(runner.run_qualification)
    assert "PUBLIC_FIXTURE_BYTES" in source
    assert "seal_four_arm_barrier_once" in source
    assert "run_fixed_scorer_once" not in source
    assert "materialize_official_packs_once" not in source
    assert "run_source_free_tests" in source
    assert "attest_runtime_closure" in source
    assert "SOURCE_FREE_DESELECTED_TEST_NODES" in source
    assert "FROZEN_TEST_PYTHON" in source
    assert "FROZEN_MINILM_TARGET_MANIFEST" in source
    assert "_preflight_frozen_main_runtime" in source
    assert "_preflight_frozen_runtime_binding_manifest" in source


def test_runner_fixes_official_source_deselection_and_absent_source_tree() -> None:
    assert runner.SOURCE_FREE_DESELECTED_TEST_NODES == (
        (
            "tests/test_gscl_arn_intrinsic_protocol_v1.py::"
            "test_official_source_exact_hash_doi_license_and_header_verify"
        ),
    )
    source = inspect.getsource(
        runner._preflight_fixed_source_free_test_runtime  # noqa: SLF001
    )
    assert '"reference"' in source
    assert '"arn.csv"' in source
    validator = inspect.getsource(
        runner._validate_fixed_test_attestation  # noqa: SLF001
    )
    assert '"official_source_access_count"' in validator
    assert '"deselected_test_nodes"' in validator
    assert runner.FROZEN_TEST_PYTHON_LINK_TARGET == (
        "/var/tmp/gscl_unified_nonscoring_harness_20260730/"
        "assets/gscl_runtime_ext4_v1/typed_venv/bin/python"
    )
    assert runner.FROZEN_MAIN_PYTHON == Path(
        runner.FROZEN_TEST_PYTHON_LINK_TARGET
    )
    assert runner.FROZEN_MAIN_PYTHON_SHA256 == (
        "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
    )
    assert runner.FROZEN_MAIN_PYVENV_CONFIG_SHA256 == (
        "da6c0ab165bd098b86649d2af4da536e7c91ee921c20b4f03d5d631f7172a503"
    )
    assert runner.FROZEN_RUNTIME_BINDING_FILE_SHA256 == (
        "4ee4c6be40af92b0c24e540735621576502e6d5a097bf265d90071386405f1a3"
    )
    assert runner.FROZEN_RUNTIME_BINDING_SELF_SHA256 == (
        "8929d1d96581373b1c2a13c1c2330fceb56c26283eab415534c2e3543217c356"
    )
    assert runner.FROZEN_TEST_PYTHON_RESOLVED_SHA256 == (
        "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
    )
    assert runner.FROZEN_TEST_PYVENV_CONFIG_SHA256 == (
        "71f98c0e02f9dbb439cbda7b1ffc40999bd963123abf4934a1266988cacb71a0"
    )
    assert hashlib.sha256(
        runner._EXPECTED_TEST_PTH_BYTES[  # noqa: SLF001
            runner.FROZEN_TEST_PARENT_PTH
        ]
    ).hexdigest() == (
        "aaab17b51d56bf56c30292613d88ce3a70be1095879b5f899e432ff794186867"
    )
    assert hashlib.sha256(
        runner._EXPECTED_TEST_PTH_BYTES[  # noqa: SLF001
            runner.FROZEN_TEST_CODE_PTH
        ]
    ).hexdigest() == (
        "caa161defc3f53b0b0f9f06d6327577dd926bcad3ae83e00cbc812aea1db4503"
    )


def test_fixed_test_python_accepts_only_the_exact_leaf_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deployment = tmp_path / "deployment"
    workspace_root = deployment / "code"
    code_root = workspace_root / "reconstruction_v2"
    code_root.mkdir(parents=True)
    runtime_bin = tmp_path / "runtime/bin"
    runtime_bin.mkdir(parents=True)
    resolved_python = runtime_bin / "python"
    resolved_python.write_bytes(b"fixed synthetic python binary")
    venv_root = deployment / "assets/test-venv"
    (venv_root / "bin").mkdir(parents=True)
    invocation = venv_root / "bin/python"
    invocation.symlink_to(resolved_python)
    pyvenv = venv_root / "pyvenv.cfg"
    pyvenv.write_text(
        "include-system-site-packages = false\n",
        encoding="ascii",
    )
    wheel_manifest = deployment / "assets/wheels.json"
    wheel_manifest.write_text("{}\n", encoding="ascii")
    site_packages = venv_root / "lib/python3.10/site-packages"
    site_packages.mkdir(parents=True)
    parent_pth = site_packages / "parent.pth"
    code_pth = site_packages / "code.pth"
    parent_raw = f"{tmp_path / 'parent-site'}\n".encode("ascii")
    code_raw = (
        f"{workspace_root}\n{code_root}\n".encode("ascii")
    )
    parent_pth.write_bytes(parent_raw)
    code_pth.write_bytes(code_raw)

    monkeypatch.setattr(
        supervisor, "_RECONSTRUCTION_ROOT", code_root
    )
    monkeypatch.setattr(
        supervisor, "_WORKSPACE_ROOT", workspace_root
    )
    monkeypatch.setattr(
        runner, "FROZEN_WORKSPACE_CODE_ROOT", workspace_root
    )
    monkeypatch.setattr(runner, "FROZEN_CODE_ROOT", code_root)
    monkeypatch.setattr(runner, "FROZEN_TEST_PYTHON", invocation)
    monkeypatch.setattr(
        runner,
        "FROZEN_TEST_PYTHON_LINK_TARGET",
        str(resolved_python),
    )
    monkeypatch.setattr(
        runner, "FROZEN_TEST_PYTHON_RESOLVED", resolved_python
    )
    monkeypatch.setattr(
        runner,
        "FROZEN_TEST_PYTHON_RESOLVED_SHA256",
        hashlib.sha256(resolved_python.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner, "FROZEN_TEST_PYVENV_CONFIG", pyvenv
    )
    monkeypatch.setattr(
        runner,
        "FROZEN_TEST_PYVENV_CONFIG_SHA256",
        hashlib.sha256(pyvenv.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST",
        wheel_manifest,
    )
    monkeypatch.setattr(
        runner,
        "_EXPECTED_TEST_PTH_BYTES",
        {
            parent_pth: parent_raw,
            code_pth: code_raw,
        },
    )
    runner._preflight_fixed_source_free_test_runtime()  # noqa: SLF001

    linked_venv = deployment / "assets/linked-test-venv"
    linked_venv.symlink_to(venv_root, target_is_directory=True)
    monkeypatch.setattr(
        runner,
        "FROZEN_TEST_PYTHON",
        linked_venv / "bin/python",
    )
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="closure_path_symlink",
    ):
        runner._preflight_fixed_source_free_test_runtime()  # noqa: SLF001


def test_inner_and_outer_terminals_remain_aggregate_only() -> None:
    canary = {
        "file_sha256": "1" * 64,
        "lineage_model_weight_load_count": 2,
        "repaired_actual_file_sha256": "2" * 64,
        "repaired_actual_self_sha256": "3" * 64,
        "self_sha256": "4" * 64,
        "successful_teacher_forced_qualification_run_count": 1,
        "worker_sha256": "5" * 64,
        "path": "/private/canary.json",
    }
    action = SimpleNamespace(
        receipt={
            "self_hash": "6" * 64,
            "qwen_actual_canary_lineage_terminal": canary,
        },
        closure=SimpleNamespace(manifest={"self_hash": "7" * 64}),
    )
    invocation = SimpleNamespace(
        receipt={"one_shot_key": "8" * 64}
    )
    barrier = {"self_hash": "9" * 64}
    body = {
        "schema": supervisor.INTERNAL_FACTORY_QUALIFICATION_SCHEMA,
        "status": "PASS_SOURCE_FREE_EXACT_INTERNAL_FACTORY_QUALIFICATION",
        "one_shot_key": "8" * 64,
        "qualification_action_self_hash": "6" * 64,
        "qualification_runtime_closure_self_hash": "7" * 64,
        "four_arm_barrier_self_hash": "9" * 64,
        "synthetic_source_sha256": runner.PUBLIC_FIXTURE_SHA256,
        "common_item_count": 2,
        "closed_choice_selection_count": 6,
        "free_form_generation_count": 0,
        "score_operation": "teacher_forced_forward_log_softmax",
        "qwen_actual_canary_lineage_binding": {
            key: value for key, value in canary.items() if key != "path"
        },
        "qwen_model_manifest_sha256": "a" * 64,
        "minilm_asset_manifest_sha256": "b" * 64,
        "minilm_target_manifest_file_sha256": "c" * 64,
        "minilm_target_manifest_self_sha256": "d" * 64,
        "outer_systemd_attestation_self_hash": "e" * 64,
        "outer_systemd_stable_binding_sha256": "f" * 64,
        "official_source_content_supplied_to_model": False,
        "public_synthetic_content_supplied_to_model": True,
        "official_source_access_count": 0,
        "label_open_count": 0,
        "online_or_api_evaluation_count": 0,
        "formal_measurement_authorized": False,
        "formal_root_used": False,
        "formal_result": False,
        "efficacy_evidence": False,
        "effect_gate_added": False,
        "item_content_emitted": False,
    }
    inner = {**body, "self_hash": runner._content_hash(body)}  # noqa: SLF001
    assert (
        runner._validate_inner_terminal(  # noqa: SLF001
            inner,
            action=action,
            invocation=invocation,
            barrier=barrier,
        )
        == inner
    )
    terminal = runner._outer_terminal(  # noqa: SLF001
        action=action,
        invocation=invocation,
        source_receipt={"self_hash": "0" * 64},
        execution_receipt={"self_hash": "1" * 64},
        barrier=barrier,
        inner_terminal=inner,
        test_attestation=SimpleNamespace(
            receipt={"self_hash": "2" * 64}
        ),
    )
    assert terminal["offline_scorer_call_count"] == 0
    assert terminal["formal_measurement"] is False
    assert terminal["efficacy_evidence"] is False
    assert terminal["item_content_emitted"] is False
    serialized = json.dumps(terminal, sort_keys=True)
    for forbidden in (
        '"opaque_item_id":',
        '"query_narrative":',
        '"first_choice":',
        '"second_choice":',
        '"correct_answer":',
        '"predictions":',
    ):
        assert forbidden not in serialized


def test_explicit_test_interpreter_isolated_and_hash_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv_root = tmp_path / "test-venv"
    (venv_root / "bin").mkdir(parents=True)
    python_target = Path("/usr/bin/python3").resolve()
    python = venv_root / "bin/python"
    python.symlink_to(python_target)
    pyvenv = venv_root / "pyvenv.cfg"
    pyvenv.write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="ascii",
    )
    site_packages = (
        venv_root / "lib/python3.10/site-packages"
    )
    site_packages.mkdir(parents=True)
    pth = site_packages / "fixture_parent.pth"
    pth.write_text(f"{ROOT}\n", encoding="ascii")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def fake_run(
        command: tuple[str, ...] | list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        normalized = tuple(command)
        environment = dict(kwargs["env"])  # type: ignore[arg-type]
        calls.append((normalized, environment))
        if "-c" in normalized:
            distributions = []
            for module_name, distribution_name in (
                supervisor._PYTEST_RUNTIME_DISTRIBUTIONS  # noqa: SLF001
            ):
                origin = (
                    site_packages
                    / f"{distribution_name.replace('-', '_')}.py"
                )
                origin.write_text(
                    f"NAME = {distribution_name!r}\n",
                    encoding="ascii",
                )
                file_row = {
                    "declared_path": origin.name,
                    "path": str(origin),
                    "present": True,
                    "sha256": hashlib.sha256(
                        origin.read_bytes()
                    ).hexdigest(),
                    "size": origin.stat().st_size,
                }
                file_rows = [file_row]
                if distribution_name == "pytest":
                    file_rows.extend(
                        {
                            "declared_path": declared_path,
                            "path": str(
                                venv_root
                                / "lib/bin"
                                / Path(declared_path).name
                            ),
                            "present": False,
                            "sha256": None,
                            "size": None,
                        }
                        for declared_path in (
                            supervisor._PYTEST_KNOWN_ABSENT_RECORD_ENTRIES  # noqa: SLF001
                        )
                    )
                file_rows.sort(key=lambda row: row["declared_path"])
                distributions.append(
                    {
                        "distribution": distribution_name,
                        "distribution_content_sha256": (
                            supervisor._content_hash(  # noqa: SLF001
                                file_rows
                            )
                        ),
                        "files": file_rows,
                        "module": module_name,
                        "origin": str(origin),
                        "origin_sha256": file_row["sha256"],
                        "version": (
                            supervisor._PYTEST_WHEEL_DISTRIBUTION_VERSIONS.get(  # noqa: SLF001
                                distribution_name,
                                "2.2.6",
                            )
                        ),
                    }
                )
            payload = {
                "base_prefix": str(python_target.parent.parent),
                "distributions": distributions,
                "executable": str(python_target),
                "pth_files": [
                    {
                        "path": str(pth),
                        "sha256": hashlib.sha256(
                            pth.read_bytes()
                        ).hexdigest(),
                        "size": pth.stat().st_size,
                    }
                ],
                "prefix": str(venv_root),
            }
            return subprocess.CompletedProcess(
                normalized,
                0,
                stdout=(
                    json.dumps(
                        payload,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("ascii"),
                stderr=b"",
            )
        return subprocess.CompletedProcess(
            normalized, 0, stdout=b"1 passed\n", stderr=b""
        )

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    test_file = (
        ROOT / "tests/fixtures/test_gscl_closure_fixture_v1.py"
    )
    receipt = supervisor.run_source_free_tests(
        code_root=test_file.parent,
        test_files=(test_file,),
        deselected_test_nodes=(
            (
                "test_gscl_closure_fixture_v1.py::"
                "test_fixture_hash_is_stable"
            ),
        ),
        test_python=python,
        pytest_wheel_bundle_manifest=(
            ROOT / "manifests/gscl_pytest_wheel_bundle_v1.json"
        ),
    ).receipt
    assert receipt["test_runner"]["explicit_frozen_interpreter"] is True
    assert receipt["test_runner"]["isolated_mode"] is True
    assert receipt["test_runner"]["pythonpath_injected"] is False
    assert receipt["test_runner"]["interpreter_sha256"] == (
        hashlib.sha256(python_target.read_bytes()).hexdigest()
    )
    assert receipt["test_runner"]["pyvenv_config_sha256"] == (
        hashlib.sha256(pyvenv.read_bytes()).hexdigest()
    )
    assert len(
        receipt["test_runner"]["distribution_closures"]
    ) == len(supervisor._PYTEST_RUNTIME_DISTRIBUTIONS)  # noqa: SLF001
    pytest_closure = next(
        row
        for row in receipt["test_runner"]["distribution_closures"]
        if row["distribution"] == "pytest"
    )
    assert pytest_closure["absent_entries"] == [
        {
            "declared_path": "../../bin/py.test",
            "path": str(venv_root / "lib/bin/py.test"),
        },
        {
            "declared_path": "../../bin/pytest",
            "path": str(venv_root / "lib/bin/pytest"),
        },
    ]
    assert pytest_closure["declared_entry_count"] == (
        pytest_closure["present_file_count"] + 2
    )
    supervisor._validate_test_runner_closure(  # noqa: SLF001
        receipt["test_runner"],
        require_frozen=True,
    )
    unexpectedly_present = venv_root / "lib/bin/py.test"
    unexpectedly_present.parent.mkdir(parents=True)
    unexpectedly_present.write_text("#!/bin/sh\n", encoding="ascii")
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="closure_absent_path_present",
    ):
        supervisor._validate_test_runner_closure(  # noqa: SLF001
            receipt["test_runner"],
            require_frozen=True,
        )
    unexpectedly_present.unlink()
    invalid_runner = json.loads(
        json.dumps(receipt["test_runner"])
    )
    invalid_pytest_closure = next(
        row
        for row in invalid_runner["distribution_closures"]
        if row["distribution"] == "pytest"
    )
    invalid_pytest_closure["absent_entries"].append(
        {
            "declared_path": "../../bin/not-declared-by-pytest",
            "path": str(
                venv_root / "lib/bin/not-declared-by-pytest"
            ),
        }
    )
    invalid_pytest_closure["declared_entry_count"] += 1
    with pytest.raises(
        supervisor.FormalSupervisorError,
        match="qualification_test_distribution_changed",
    ):
        supervisor._validate_test_runner_closure(  # noqa: SLF001
            invalid_runner,
            require_frozen=True,
        )
    assert receipt["test_runner"]["cuda_visible_devices"] == ""
    assert (
        receipt["test_runner"][
            "bytecode_writes_disabled_by_cli"
        ]
        is True
    )
    assert receipt["test_runner"]["pytest_config_file"] == "/dev/null"
    assert receipt["test_runner"]["pytest_rootdir"] == str(
        test_file.parent
    )
    assert receipt["deselected_test_nodes"] == [
        (
            "test_gscl_closure_fixture_v1.py::"
            "test_fixture_hash_is_stable"
        )
    ]
    assert receipt["official_source_access_count"] == 0
    assert len(calls) == 2
    assert all("-I" in command for command, _ in calls)
    assert all("-B" in command for command, _ in calls)
    assert all("PYTHONPATH" not in environment for _, environment in calls)
    assert all(
        environment["CUDA_VISIBLE_DEVICES"] == ""
        for _, environment in calls
    )
    assert any(
        (
            "--deselect=test_gscl_closure_fixture_v1.py::"
            "test_fixture_hash_is_stable"
        )
        in command
        for command, _ in calls
    )
    assert any(
        ("-p", "no:cacheprovider")
        == command[
            command.index("-p"):command.index("-p") + 2
        ]
        for command, _ in calls
        if "-p" in command
    )
    assert any(
        ("-c", "/dev/null")
        == command[
            command.index("-c"):command.index("-c") + 2
        ]
        for command, _ in calls
        if "-c" in command
    )


def test_generated_inventory_probe_executes_known_absent_entries(
    tmp_path: Path,
) -> None:
    venv_root = tmp_path / "inventory-venv"
    (venv_root / "bin").mkdir(parents=True)
    python_target = Path(sys.executable).resolve()
    python = venv_root / "bin/python"
    python.symlink_to(python_target)
    (venv_root / "pyvenv.cfg").write_text(
        (
            f"home = {python_target.parent}\n"
            "include-system-site-packages = false\n"
            f"version = {sys.version_info.major}.{sys.version_info.minor}\n"
        ),
        encoding="ascii",
    )
    site_packages = (
        venv_root
        / (
            f"lib/python{sys.version_info.major}."
            f"{sys.version_info.minor}/site-packages"
        )
    )
    site_packages.mkdir(parents=True)
    versions = {
        **supervisor._PYTEST_WHEEL_DISTRIBUTION_VERSIONS,  # noqa: SLF001
        "numpy": "2.2.6",
    }
    for module_name, distribution_name in (
        supervisor._PYTEST_RUNTIME_DISTRIBUTIONS  # noqa: SLF001
    ):
        module_path = site_packages / f"{module_name}.py"
        module_path.write_text(
            f"NAME = {distribution_name!r}\n",
            encoding="ascii",
        )
        dist_info = (
            site_packages
            / (
                f"{distribution_name.replace('-', '_')}-"
                f"{versions[distribution_name]}.dist-info"
            )
        )
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            (
                "Metadata-Version: 2.1\n"
                f"Name: {distribution_name}\n"
                f"Version: {versions[distribution_name]}\n"
            ),
            encoding="ascii",
        )
        record_lines = [f"{module_path.name},,"]
        if distribution_name == "pytest":
            record_lines.extend(
                f"{declared_path},,"
                for declared_path in (
                    supervisor._PYTEST_KNOWN_ABSENT_RECORD_ENTRIES  # noqa: SLF001
                )
            )
        (dist_info / "RECORD").write_text(
            "\n".join(record_lines) + "\n",
            encoding="ascii",
        )
    completed = subprocess.run(
        (
            str(python),
            "-B",
            "-I",
            "-c",
            supervisor._pytest_inventory_probe_code(  # noqa: SLF001
                explicit_frozen_interpreter=True
            ),
        ),
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert b"NameError" not in completed.stderr
    payload = json.loads(completed.stdout.decode("ascii"))
    assert payload["prefix"] == str(venv_root)
    pytest_inventory = next(
        row
        for row in payload["distributions"]
        if row["distribution"] == "pytest"
    )
    absent = [
        {
            "declared_path": row["declared_path"],
            "path": row["path"],
        }
        for row in pytest_inventory["files"]
        if row["present"] is False
    ]
    assert absent == [
        {
            "declared_path": "../../bin/py.test",
            "path": str(venv_root / "lib/bin/py.test"),
        },
        {
            "declared_path": "../../bin/pytest",
            "path": str(venv_root / "lib/bin/pytest"),
        },
    ]


def test_nested_nonfrozen_inventory_binds_remote_shape_missing_entries(
    tmp_path: Path,
) -> None:
    venv_root = tmp_path / "nested-inventory-venv"
    (venv_root / "bin").mkdir(parents=True)
    python_target = Path(sys.executable).resolve()
    python = venv_root / "bin/python"
    python.symlink_to(python_target)
    (venv_root / "pyvenv.cfg").write_text(
        (
            f"home = {python_target.parent}\n"
            "include-system-site-packages = false\n"
            f"version = {sys.version_info.major}.{sys.version_info.minor}\n"
        ),
        encoding="ascii",
    )
    site_packages = (
        venv_root
        / (
            f"lib/python{sys.version_info.major}."
            f"{sys.version_info.minor}/site-packages"
        )
    )
    site_packages.mkdir(parents=True)
    runtime_import_roots: list[Path] = []
    module_import_roots: dict[str, Path] = {}
    for module_name, _ in (
        supervisor._PYTEST_RUNTIME_DISTRIBUTIONS  # noqa: SLF001
    ):
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None
        assert module_spec.origin is not None
        module_origin = Path(module_spec.origin).resolve()
        module_root = (
            module_origin.parent.parent
            if module_origin.name == "__init__.py"
            else module_origin.parent
        )
        module_import_roots[module_name] = module_root
        if module_root not in runtime_import_roots:
            runtime_import_roots.append(module_root)
    numpy_spec = importlib.util.find_spec("numpy")
    assert numpy_spec is not None
    assert numpy_spec.origin is not None
    numpy_origin = Path(numpy_spec.origin).resolve()
    expected_numpy_root = (
        numpy_origin.parent.parent
        if numpy_origin.name == "__init__.py"
        else numpy_origin.parent
    )
    assert module_import_roots["numpy"] == expected_numpy_root
    assert expected_numpy_root in runtime_import_roots
    source_runtime_site = module_import_roots["pytest"]
    source_pytest_package = source_runtime_site / "pytest"
    local_pytest_package = site_packages / "pytest"
    shutil.copytree(
        source_pytest_package,
        local_pytest_package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    pytest_distribution = importlib.metadata.distribution("pytest")
    pytest_version = pytest_distribution.version
    dist_info = site_packages / f"pytest-{pytest_version}.dist-info"
    dist_info.mkdir()
    metadata_path = dist_info / "METADATA"
    metadata_path.write_text(
        (
            "Metadata-Version: 2.1\n"
            "Name: pytest\n"
            f"Version: {pytest_version}\n"
        ),
        encoding="ascii",
    )
    record_path = dist_info / "RECORD"
    present_rows = sorted(
        str(path.relative_to(site_packages))
        for path in local_pytest_package.rglob("*")
        if path.is_file()
    )
    record_path.write_text(
        (
            "\n".join(
                (
                    *present_rows,
                    str(metadata_path.relative_to(site_packages)),
                    str(record_path.relative_to(site_packages)),
                    "../../bin/py.test",
                    "../../bin/pytest",
                    "missing/non_console_entry.py",
                )
            )
            + "\n"
        ).replace("\n", ",,\n"),
        encoding="ascii",
    )
    pth = site_packages / "nested_remote_shape.pth"
    pth_roots: list[Path] = []
    for root in (*runtime_import_roots, ROOT.parent, ROOT):
        if root not in pth_roots:
            pth_roots.append(root)
    pth.write_text(
        "".join(f"{root}\n" for root in pth_roots),
        encoding="ascii",
    )
    code_root = tmp_path / "nested-code"
    code_root.mkdir()
    nested_test = code_root / "test_nested_remote_shape.py"
    nested_test.write_text(
        "def test_nested_remote_shape():\n    assert True\n",
        encoding="ascii",
    )
    driver = tmp_path / "nested_driver.py"
    driver.write_text(
        (
            "import json\n"
            "from pathlib import Path\n"
            "from assumption_agent.benchmarks import "
            "gscl_arn_formal_supervisor_v1 as supervisor\n"
            f"test_path = Path({str(nested_test)!r})\n"
            "receipt = supervisor.run_source_free_tests(\n"
            "    code_root=test_path.parent,\n"
            "    test_files=(test_path,),\n"
            ").receipt\n"
            "print(json.dumps(receipt, sort_keys=True))\n"
        ),
        encoding="ascii",
    )
    completed = subprocess.run(
        (str(python), "-B", "-I", str(driver)),
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    receipt = json.loads(completed.stdout.decode("ascii"))
    assert (
        receipt["test_runner"]["explicit_frozen_interpreter"]
        is False
    )
    assert receipt["test_runner"]["pytest_origin"] == str(
        local_pytest_package / "__init__.py"
    )
    pytest_closure = next(
        row
        for row in receipt["test_runner"]["distribution_closures"]
        if row["distribution"] == "pytest"
    )
    assert pytest_closure["absent_entries"] == [
        {
            "declared_path": "../../bin/py.test",
            "path": str(venv_root / "lib/bin/py.test"),
        },
        {
            "declared_path": "../../bin/pytest",
            "path": str(venv_root / "lib/bin/pytest"),
        },
        {
            "declared_path": "missing/non_console_entry.py",
            "path": str(
                site_packages / "missing/non_console_entry.py"
            ),
        },
    ]


def test_main_returns_75_for_pre_attempt_gpu_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deferred(**_: object) -> None:
        raise runner.QualificationDeferred(
            "gpu_compute_process_present"
        )

    monkeypatch.setattr(runner, "run_qualification", deferred)
    arguments = [
        "--root",
        "/var/tmp/gscl-qualification-test",
        "--qwen-model-root",
        "/models/qwen",
        "--qwen-model-manifest",
        "/models/qwen.json",
        "--qwen-actual-canary-lineage-terminal",
        "/models/canary.json",
        "--minilm-model-root",
        "/models/minilm",
        "--minilm-asset-manifest",
        "/models/minilm.json",
    ]
    assert runner._main(arguments) == runner.DEFERRED_EXIT_CODE  # noqa: SLF001


def test_service_template_matches_normalized_runtime_contract() -> None:
    text = SERVICE.read_text(encoding="ascii")
    assert "IPAddressDeny=any" in text
    assert (
        supervisor.OUTER_SYSTEMD_CONTRACT["IPAddressDeny"]
        == "::/0 0.0.0.0/0"
    )
    assert "--test-python" not in text
    assert "--pytest-wheel-bundle-manifest" not in text
    assert "--minilm-target-manifest" not in text
    assert "Environment=CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert (
        "Environment=PYTHONPATH="
        "/var/tmp/gscl_unified_nonscoring_harness_20260730/code:"
        "/var/tmp/gscl_unified_nonscoring_harness_20260730/"
        "code/reconstruction_v2"
    ) in text
    assert "/bin/sh" not in text
    expected = {
        "CPUQuota=400%",
        "CPUWeight=25",
        "IOWeight=25",
        "IOSchedulingClass=idle",
        "MemoryHigh=24G",
        "MemoryMax=32G",
        "MemorySwapMax=0",
        "TasksMax=96",
        "Nice=10",
        "NoNewPrivileges=yes",
        "PrivateDevices=no",
        "PrivateTmp=no",
        "ProtectSystem=no",
        "ReadOnlyPaths=",
        "ReadWritePaths=",
        "RestrictAddressFamilies=AF_UNIX",
        "KillMode=control-group",
        "Restart=no",
        "RuntimeMaxSec=infinity",
        "TimeoutStartSec=infinity",
        "UMask=0077",
    }
    assert expected.issubset(set(text.splitlines()))
