from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools/phase3_q0_dual_qualification_v1.py"
ENTRYPOINT_PATH = PROJECT_ROOT / "tools/phase3_q0_python_oracle_entrypoint_v1.py"
SPEC = importlib.util.spec_from_file_location("phase3_q0_dual_qualification_v1", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


@pytest.fixture(scope="module")
def python_report() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(ENTRYPOINT_PATH)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    assert completed.stderr == b""
    value = json.loads(completed.stdout)
    assert type(value) is dict
    return value


def _rust_report(python: dict[str, object]) -> dict[str, object]:
    report = json.loads(
        json.dumps(
            {
                key: value
                for key, value in python.items()
                if key not in {"python_source_root", "endpoint_state_cbor_hex"}
            }
        )
    )
    report["schema_version"] = tool.RUST_SCHEMA
    report["implementation_id"] = tool.RUST_IMPLEMENTATION_ID
    report["rust_source_root"] = "sha256:" + "5a" * 32
    report["direct_rounds"][0]["frontier_mutation_count"] = 37
    report["direct_rounds"][0]["cohort_bank_mutation_count"] = 68
    report["direct_rounds"][1]["frontier_mutation_count"] = 70
    report["direct_rounds"][1]["cohort_bank_mutation_count"] = 182
    return report


@pytest.fixture(scope="module")
def endpoints(
    python_report: dict[str, object],
) -> tuple[object, object]:
    python = tool.validate_endpoint_v1(python_report, "python")
    rust = tool.validate_endpoint_v1(_rust_report(python_report), "rust")
    return python, rust


@pytest.fixture(scope="module")
def host_replay() -> object:
    return tool.host_local_replay_v1()


@pytest.fixture(scope="module")
def pre_dual_gates() -> object:
    return tool.qualify_pre_dual_gate_evidence_v1(PROJECT_ROOT)


@pytest.fixture(scope="module")
def committed_project() -> tuple[Path, str]:
    python = tool.python_source_manifest_v1(PROJECT_ROOT)
    rust = tool.rust_source_manifest_v1(PROJECT_ROOT)
    rows = {row.path: row for row in (*python.files, *rust.files)}
    for relative in (
        "tools/phase3_q0_dual_qualification_v1.py",
        "config/phase3_q0_dual_isolation_v1.json",
    ):
        rows[relative] = tool._source_row(PROJECT_ROOT, relative)
    with tool.tempfile.TemporaryDirectory(
        dir=tool.linux_temp_root_v1()
    ) as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for relative, row in sorted(rows.items()):
            source = PROJECT_ROOT / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            destination.chmod(0o755 if row.mode == 0o100755 else 0o644)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "add", "--all"], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Q0 Test",
                "-c",
                "user.email=q0@example.invalid",
                "commit",
                "-q",
                "-m",
                "Q0 isolated source fixture",
            ],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        yield root, commit


@pytest.fixture(scope="module")
def qualified_material(
    endpoints: tuple[object, object],
    host_replay: object,
    pre_dual_gates: object,
    committed_project: tuple[Path, str],
) -> tuple[object, object, object, object, object]:
    python, rust = endpoints
    project_root, commit = committed_project
    python_manifest = tool.python_source_manifest_v1(project_root)
    rust_manifest = tool.rust_source_manifest_v1(project_root)
    host_manifest = tool.host_replay_source_manifest_v1(PROJECT_ROOT, host_replay)
    with tool.tempfile.TemporaryDirectory(
        dir=tool.linux_temp_root_v1()
    ) as temporary:
        control = Path(temporary)
        python_snapshot = control / "python"
        rust_snapshot = control / "rust"
        cargo_snapshot = control / "cargo"
        tool.materialize_source_snapshot_v1(
            project_root, python_manifest, python_snapshot, commit
        )
        tool.materialize_source_snapshot_v1(
            project_root, rust_manifest, rust_snapshot, commit
        )
        tool.materialize_cargo_home_snapshot_v1(
            tool.DEFAULT_CARGO_CACHE, rust_manifest, cargo_snapshot
        )
        isolation = tool.isolation_prerequisite_evidence_v1(
            project_root,
            commit,
            tool.python_endpoint_command(python_snapshot),
            tool.rust_endpoint_command(rust_snapshot, cargo_snapshot),
            tool.PYTHON_IMAGE.split("@", 1)[1],
            tool.RUST_IMAGE.split("@", 1)[1],
            python_snapshot,
            rust_snapshot,
            cargo_snapshot,
            python_manifest,
            rust_manifest,
        )
        pre_receipt = tool.finalize_gate_evidence_v1(
            pre_dual_gates,
            python,
            rust,
            host_replay,
            python_manifest,
            rust_manifest,
            host_manifest,
            isolation,
            tool.load_isolation_config(PROJECT_ROOT)["downstream_state"],
        )
    candidate = tool.build_saturation_receipt_v1(
        python,
        rust,
        host_replay,
        python_manifest,
        rust_manifest,
        pre_receipt,
    )
    final_gates = tool.finalize_candidate_receipt_v1(pre_receipt, candidate)
    return python_manifest, rust_manifest, pre_receipt, candidate, final_gates


def test_isolation_config_is_exact_and_downstream_remains_not_run() -> None:
    config = tool.load_isolation_config(PROJECT_ROOT)
    assert config["images"] == {
        "python": tool.PYTHON_IMAGE,
        "rust": tool.RUST_IMAGE,
    }
    assert config["readiness_gates"] == [list(row) for row in tool.READINESS_GATES]
    assert config["isolation_prerequisites"] == [
        list(row) for row in tool.ISOLATION_PREREQUISITES
    ]
    assert config["host_role"] == {
        "trusted_issuer": True,
        "third_independent_endpoint": False,
        "filesystem_hard_isolation": False,
        "target_blind_import_manifest_required": True,
    }
    assert config["downstream_state"] == {
        "q1_status_id": 0,
        "q1_output_root": None,
        "q2_status_id": 0,
        "role_evaluation_performed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
    }


def test_hardened_commands_are_offline_read_only_and_resource_bounded(
    tmp_path: Path,
) -> None:
    python_snapshot = tmp_path / "python"
    rust_snapshot = tmp_path / "rust"
    cache = tmp_path / "cargo"
    for path in (python_snapshot, rust_snapshot, cache):
        path.mkdir()
    python = tool.python_endpoint_command(python_snapshot)
    rust = tool.rust_endpoint_command(rust_snapshot, cache)
    for command in (python, rust):
        assert command[:3] == [
            tool.DOCKER_EXECUTABLE,
            f"--host={tool.DOCKER_HOST}",
            "run",
        ]
        for option in (
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory=512m",
            "--memory-swap=512m",
            "--pids-limit=64",
            "--ulimit=nofile=128:128",
            "--tmpfs=/tmp:rw,exec,nosuid,nodev,size=512m,mode=1777",
        ):
            assert option in command
        assert any(argument.endswith(":/workspace:ro") for argument in command)
        assert not any(
            "truth" in argument.lower()
            or "split" in argument.lower()
            or "phase3_dsl_v1" in argument.lower()
            for argument in command
        )
    assert tool.PYTHON_IMAGE in python
    assert tool.RUST_IMAGE in rust
    assert f"{cache.resolve()}:/cargo-home:ro" in rust
    assert not any("cargo-home" in argument for argument in python)
    assert "CARGO_NET_OFFLINE=true" in rust
    assert "CARGO_BUILD_JOBS=1" in rust
    assert rust[-2:] == ["--target", "x86_64-unknown-linux-gnu"]
    assert ["-j", "1"] == rust[rust.index("-j") : rust.index("-j") + 2]


def test_complete_source_manifests_are_target_blind_and_domain_separated(
    host_replay: object,
) -> None:
    python = tool.python_source_manifest_v1(PROJECT_ROOT)
    rust = tool.rust_source_manifest_v1(PROJECT_ROOT)
    host = tool.host_replay_source_manifest_v1(PROJECT_ROOT, host_replay)
    python_paths = {row.path for row in python.files}
    rust_paths = {row.path for row in rust.files}
    assert "tools/phase3_q0_python_oracle_entrypoint_v1.py" in python_paths
    assert "src/hegel_machine/phase3_q0_gate_qualification_v1.py" in python_paths
    assert "config/phase3_q0_quotient_freeze_v1.json" in python_paths
    assert "src/hegel_machine/phase3_q0_quotient_oracle_v1.py" in python_paths
    assert "src/hegel_machine/strict_ast_shrink6_v1.py" in python_paths
    assert "src/hegel_machine/__init__.py" not in python_paths
    assert "rust/q0_quotient_oracle/Cargo.lock" in rust_paths
    assert "rust/q0_quotient_oracle/src/lib.rs" in rust_paths
    assert "rust/strict_canonicalizer_shrink6/src/lib.rs" in rust_paths
    assert rust.target_triple == "x86_64-unknown-linux-gnu"
    assert len(rust.registry_dependencies) == 21
    assert len(rust.dependency_files) > 600
    assert not any(
        token in path.lower()
        for path in python_paths | rust_paths
        for token in tool.FORBIDDEN_SOURCE_TOKENS
    )
    host_paths = {row.path for row in host.files}
    assert "tools/phase3_q0_dual_qualification_v1.py" in host_paths
    assert all(
        f"src/hegel_machine/{name.rsplit('.', 1)[-1]}.py" in host_paths
        for name in host_replay.loaded_modules
    )
    assert len(python.root) == len(rust.root) == len(host.root) == 32
    assert len({python.root, rust.root, host.root}) == 3
    changed = replace(
        python,
        files=(
            replace(python.files[0], digest=b"\xff" * 32),
            *python.files[1:],
        ),
    )
    assert changed.root != python.root
    changed_runtime = replace(
        python,
        runtime_identity=(*python.runtime_identity[:-1], "/changed-entrypoint"),
    )
    assert changed_runtime.root != python.root


def test_source_snapshots_contain_exactly_one_implementation() -> None:
    python = tool.python_source_manifest_v1(PROJECT_ROOT)
    with tool.tempfile.TemporaryDirectory(
        dir=tool.linux_temp_root_v1()
    ) as temporary:
        destination = Path(temporary) / "python-snapshot"
        tool.materialize_source_snapshot_v1(PROJECT_ROOT, python, destination)
        observed = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        assert observed == {row.path for row in python.files}
        assert not (destination / "src/hegel_machine/__init__.py").exists()
        assert not (destination / "rust").exists()
        assert tool._sealed_snapshot_file_rows_v1(destination, "source") == (
            python.files
        )


def test_cargo_home_snapshot_is_complete_checksum_bound_and_read_only() -> None:
    manifest = tool.rust_source_manifest_v1(
        PROJECT_ROOT, tool.DEFAULT_CARGO_CACHE
    )
    with tool.tempfile.TemporaryDirectory(
        dir=tool.linux_temp_root_v1()
    ) as temporary:
        destination = Path(temporary) / "cargo-home"
        tool.materialize_cargo_home_snapshot_v1(
            tool.DEFAULT_CARGO_CACHE, manifest, destination
        )
        assert tool._cargo_home_file_rows(destination) == manifest.dependency_files
        assert all(
            dependency.archive_digest == dependency.lock_checksum
            for dependency in manifest.registry_dependencies
        )


def test_linux_temp_root_ignores_drvfs_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, "/mnt/c/forbidden-q0-temp")
    root = tool.linux_temp_root_v1()
    assert root == Path("/tmp").resolve()
    assert root != Path("/mnt") and Path("/mnt") not in root.parents


def test_python_entrypoint_never_executes_package_initializer_or_target_modules() -> None:
    script = f"""
import contextlib,io,json,runpy,sys
ns=runpy.run_path({str(ENTRYPOINT_PATH)!r},run_name='q0_entrypoint_test')
for name in sys.modules:
    low=name.lower()
    assert all(token not in low for token in ('phase3_dsl_v1','target','truth','split')), name
oracle=ns['oracle']
contract=ns['contract']
def boom():
    raise oracle.Q0OracleError(
        'INCONCLUSIVE_RESOURCE_LIMIT',
        'synthetic',
        guard_id=contract.Q0ResourceGuardId.OUTPUT_BYTES,
    )
ns['main'].__globals__['endpoint_object']=boom
stream=io.StringIO()
with contextlib.redirect_stdout(stream):
    code=ns['main']()
assert code == 1
payload=json.loads(stream.getvalue())
assert payload['resource_guard_id'] == int(contract.Q0ResourceGuardId.OUTPUT_BYTES)
assert payload['authority_claimed'] is False
print(json.dumps(payload,sort_keys=True,separators=(',',':')))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert completed.stderr == b""
    assert json.loads(completed.stdout)["resource_guard_id"] == 9


def test_endpoint_json_parser_rejects_duplicate_nan_and_noncanonical_bytes() -> None:
    assert tool._strict_json_object_v1(b'{"a":1,"b":2}\n', "test") == {
        "a": 1,
        "b": 2,
    }
    with pytest.raises(tool.SupervisorError, match="duplicate JSON key"):
        tool._strict_json_object_v1(b'{"a":1,"a":2}\n', "test")
    with pytest.raises(tool.SupervisorError, match="non-finite JSON"):
        tool._strict_json_object_v1(b'{"a":NaN}\n', "test")
    with pytest.raises(tool.SupervisorError, match="canonical JSON bytes"):
        tool._strict_json_object_v1(b'{"b":2,"a":1}\n', "test")


def test_strict_43_field_endpoint_replay_and_dual_agreement(
    endpoints: tuple[object, object],
) -> None:
    python, rust = endpoints
    assert len(python.canonical_state) == len(rust.canonical_state) == 43
    assert tool._cbor.canonical_cbor_decode(python.canonical_bytes) == python.canonical_state
    assert python.canonical_bytes == rust.canonical_bytes
    assert python.endpoint_root == rust.endpoint_root
    assert len(python.syntax_preimage.canonical_bytes) == 127_439
    assert len(python.direct_preimage.canonical_bytes) == 125_153
    assert (
        python.syntax_preimage.canonical_bytes
        == rust.syntax_preimage.canonical_bytes
    )
    assert (
        python.direct_preimage.canonical_bytes
        == rust.direct_preimage.canonical_bytes
    )
    tool.compare_endpoints_v1(python, rust)


def test_endpoint_rejects_noncanonical_cbor_coverage_and_rounds(
    python_report: dict[str, object],
) -> None:
    cbor_tamper = dict(python_report)
    cbor_tamper["endpoint_state_cbor_hex"] = "80"
    with pytest.raises(tool.SupervisorError, match="CBOR differs"):
        tool.validate_endpoint_v1(cbor_tamper, "python")

    odd_cbor = dict(python_report)
    odd_cbor["endpoint_state_cbor_hex"] = "0"
    with pytest.raises(tool.SupervisorError, match="CBOR hex"):
        tool.validate_endpoint_v1(odd_cbor, "python")

    preimage_tamper = dict(python_report)
    preimage_tamper["syntax_saturation_state_preimage_cbor_hex"] = "80"
    with pytest.raises(tool.SupervisorError, match="five-tuple"):
        tool.validate_endpoint_v1(preimage_tamper, "python")

    coverage_tamper = json.loads(json.dumps(python_report))
    coverage_tamper["syntax_coverage"][0]["eligible_raw"] += 1
    with pytest.raises(tool.SupervisorError, match="coverage"):
        tool.validate_endpoint_v1(coverage_tamper, "python")

    round_tamper = json.loads(json.dumps(python_report))
    round_tamper["direct_rounds"][0]["round_index"] = 0
    with pytest.raises(tool.SupervisorError, match="round indices"):
        tool.validate_endpoint_v1(round_tamper, "python")

    final_round_tamper = json.loads(json.dumps(python_report))
    final_round_tamper["direct_rounds"][-1]["new_canonical_program_count"] = 1
    with pytest.raises(tool.SupervisorError, match="zero-delta"):
        tool.validate_endpoint_v1(final_round_tamper, "python")


def test_dual_comparison_allows_only_intermediate_round_history_drift(
    python_report: dict[str, object],
) -> None:
    python = tool.validate_endpoint_v1(python_report, "python")
    rust_report = json.loads(json.dumps(_rust_report(python_report)))
    rust_report["direct_rounds"][0]["queued_application_count"] += 1
    rust = tool.validate_endpoint_v1(rust_report, "rust")
    tool.compare_endpoints_v1(python, rust)
    assert python.report["direct_rounds"] != rust.report["direct_rounds"]


def test_host_local_replay_is_target_blind_and_matches_both_endpoints(
    host_replay: object,
    endpoints: tuple[object, object],
) -> None:
    python, rust = endpoints
    tool.compare_host_replay_v1(host_replay, python, rust)
    assert host_replay.syntax_preimage_bytes == python.syntax_preimage.canonical_bytes
    assert host_replay.direct_preimage_bytes == python.direct_preimage.canonical_bytes
    assert not any(
        token in name.lower()
        for name in host_replay.loaded_modules
        for token in tool.FORBIDDEN_SOURCE_TOKENS
    )


def test_host_manifest_rejects_same_named_module_outside_project(
    host_replay: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = host_replay.loaded_modules[0]
    module = sys.modules[name]
    impostor = tmp_path / f"{name.rsplit('.', 1)[-1]}.py"
    impostor.write_text("# same leaf, wrong source\n", encoding="ascii")
    monkeypatch.setattr(module, "__file__", str(impostor))
    with pytest.raises(tool.SupervisorError, match="module file differs"):
        tool.host_replay_source_manifest_v1(PROJECT_ROOT, host_replay)


def test_host_only_receipt_is_exact_40_fields_and_keeps_downstream_closed(
    qualified_material: tuple[object, object, object, object, object],
) -> None:
    python_manifest, rust_manifest, pre_receipt, receipt, final_gates = (
        qualified_material
    )
    decoded = tool._cbor.canonical_cbor_decode(receipt.canonical_bytes)
    assert len(decoded) == 40
    assert decoded[9] == 2
    assert decoded[27] == python_manifest.root
    assert decoded[28] == rust_manifest.root
    assert decoded[32:40] == (14, 0x3FFF, 0, None, 0, False, None, False)
    assert len(receipt.receipt_root) == 32
    assert [row["passed"] for row in pre_receipt.gates] == [True] * 13 + [False]
    assert pre_receipt.gates[13]["pending_dual"] is True
    assert [row["passed"] for row in final_gates.gates] == [True] * 14
    assert all(row["pending_dual"] is False for row in final_gates.gates)


def test_receipt_issuer_rejects_missing_false_or_stale_gate_evidence(
    endpoints: tuple[object, object],
    host_replay: object,
    pre_dual_gates: object,
    qualified_material: tuple[object, object, object, object, object],
) -> None:
    python, rust = endpoints
    python_manifest, rust_manifest, pre_receipt, receipt, _ = qualified_material
    with pytest.raises(tool.SupervisorError, match="pre-receipt token"):
        tool.build_saturation_receipt_v1(
            python,
            rust,
            host_replay,
            python_manifest,
            rust_manifest,
            None,
        )

    tampered_payload = pre_dual_gates.payload
    first_predicate = next(iter(tampered_payload["gates"][0]["predicates"]))
    tampered_payload["gates"][0]["predicates"][first_predicate] = False
    tampered_payload["gates"][0]["passed"] = False
    canonical = tool._canonical_json_bytes(tampered_payload)
    evidence_root = tool.sha256(
        tool._PRE_DUAL_GATE_EVIDENCE_ROOT_DOMAIN + b"\x00" + canonical
    ).digest()
    tampered_pre_dual = tool.ValidatedPreDualGateEvidenceV1(
        canonical, evidence_root, tool._GATE_ISSUER_TOKEN
    )
    with pytest.raises(tool.SupervisorError, match="readiness gate 1"):
        tool.finalize_gate_evidence_v1(
            tampered_pre_dual,
            python,
            rust,
            host_replay,
            python_manifest,
            rust_manifest,
            tool.host_replay_source_manifest_v1(PROJECT_ROOT, host_replay),
            None,
            tool.load_isolation_config(PROJECT_ROOT)["downstream_state"],
        )

    stale_receipt = replace(receipt, python_implementation_root=b"\x00" * 32)
    with pytest.raises(tool.SupervisorError, match="bindings differ"):
        tool.finalize_candidate_receipt_v1(pre_receipt, stale_receipt)


def test_parallel_runner_uses_two_concurrent_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    barrier = threading.Barrier(2)
    threads: set[int] = set()

    def fake_run(
        implementation: str,
        command: object,
        environment: object,
    ) -> object:
        threads.add(threading.get_ident())
        barrier.wait(timeout=2)
        return tool.EndpointRunV1(implementation, b"{}\n", {})

    monkeypatch.setattr(tool, "_run_endpoint", fake_run)
    python, rust = tool.run_endpoints_parallel_v1(["python"], ["rust"], {})
    assert (python.implementation, rust.implementation) == ("python", "rust")
    assert len(threads) == 2


def test_dry_run_is_nonmutating_and_never_constructs_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "must-not-exist.json"

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run must not execute Docker or create a receipt")

    monkeypatch.setattr(tool, "run_endpoints_parallel_v1", forbidden)
    monkeypatch.setattr(tool, "build_saturation_receipt_v1", forbidden)
    plan = tool.dry_run_plan_v1(
        PROJECT_ROOT, tool.DEFAULT_CARGO_CACHE, artifact
    )
    assert plan["execution"] == "DRY_RUN"
    assert plan["receipt_created"] is False
    assert plan["artifact_written"] is False
    assert plan["q0_state"] == "NOT_RUN"
    assert plan["readiness_gate_total"] == 14
    assert plan["readiness_gates_passed"] == 0
    assert plan["readiness_gate_mask"] == 0
    assert plan["q1_status_id"] == plan["q2_status_id"] == 0
    assert plan["m3_formal_roots"] is None
    assert not artifact.exists()


def test_actual_mode_requires_full_commit_and_clean_worktree() -> None:
    with pytest.raises(tool.SupervisorError, match="full lowercase commit"):
        tool.verify_source_commit_v1(PROJECT_ROOT, "HEAD")


def test_artifact_writer_refuses_overwrite(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.json"
    artifact.write_text("owner data", encoding="utf-8")
    with pytest.raises(tool.SupervisorError, match="already exists"):
        tool._write_artifact(artifact, {"replacement": True})
    assert artifact.read_text(encoding="utf-8") == "owner data"
