from __future__ import annotations

from hashlib import sha256
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "tools/phase3_q1_python_capacity_preflight_entrypoint_v1.py"


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        assert key not in value
        value[key] = item
    return value


def test_import_isolated_subset_endpoint_is_non_authoritative() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(ENTRYPOINT),
            "--local-subset-node-count",
            "3",
        ],
        cwd=ROOT,
        env={
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        capture_output=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stderr == b""
    assert completed.stdout.count(b"\n") == 1
    assert completed.stdout.endswith(b"\n")
    payload = json.loads(
        completed.stdout,
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )
    canonical = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert completed.stdout == canonical
    assert payload["endpoint_schema_version"] == (
        "hegel-phase3a-q1-python-capacity-preflight-endpoint/1"
    )
    assert payload["preregistered_full_limits_used"] is False
    assert payload["dual_agreement_claimed"] is False
    assert payload["source_commit_bound"] is False
    assert payload["normal_package_initializer_executed"] is False
    assert payload["import_allowlist_isolation_passed"] is True
    assert payload["source_snapshot_filesystem_isolated"] is False
    assert payload["q1_state"] == "NOT_RUN"
    assert payload["q1_gate_count"] == 0
    assert payload["q1_gate_mask"] == 0
    assert payload["q1_receipt"] is None
    assert payload["q2_state"] == "NOT_RUN"
    assert payload["m3_formal_roots"] is None
    assert payload["target_truth_accessed"] is False
    assert payload["split_accessed"] is False
    assert payload["role_evaluation_performed"] is False
    assert payload["outside_certificate_issued"] is False
    assert payload["active_transition_allowed"] is False

    engine = payload["engine_diagnostic"]
    assert engine["terminal_status"] == "LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED"
    assert [row["input_signature_id"] for row in engine["partitions"]] == [1, 2]
    assert [row["raw_operator_application_count"] for row in engine["partitions"]] == [1048, 1101]
    assert all(row["depth_barriers"][-1]["barrier_kind"] == "STRUCTURAL_BOUNDARY" for row in engine["partitions"])
    assert all(row["resource_guard_id"] is None for row in engine["partitions"])

    engine_bytes = (
        json.dumps(
            engine,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert payload["engine_diagnostic_json_sha256"] == (
        "sha256:" + sha256(engine_bytes).hexdigest()
    )
    loaded = payload["loaded_project_modules"]
    assert len(loaded) == 27
    assert "hegel_machine.phase3_dsl_v1" not in loaded
    assert "hegel_machine.phase3_q0_quotient_contract_v1" not in loaded
    assert "src/hegel_machine/__init__.py" not in payload["implementation_source_paths"]


def test_default_full_preflight_is_fail_closed_before_execution() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(ENTRYPOINT)],
        cwd=ROOT,
        env={
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 1
    assert completed.stderr == b""
    assert completed.stdout.count(b"\n") == 1
    payload = json.loads(completed.stdout, object_pairs_hook=_strict_object)
    assert payload["error_code"] == "Q1_FULL_PREFLIGHT_NOT_ADMITTED"
    assert payload["q1_state"] == "NOT_RUN"
    assert payload["q1_gate_count"] == 0
    assert payload["source_snapshot_filesystem_isolated"] is False


def test_subset_runs_from_tree_containing_only_allowlisted_sources() -> None:
    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    module_names = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "MODULE_NAMES":
                module_names = ast.literal_eval(node.value)
                break
    assert isinstance(module_names, tuple) and len(module_names) == 27
    paths = {
        *(f"src/hegel_machine/{name}.py" for name in module_names),
        "config/phase3_q1_capacity_preflight_v1.json",
        "tools/phase3_q1_python_capacity_preflight_entrypoint_v1.py",
    }
    with tempfile.TemporaryDirectory(prefix="hegel-q1-allowlist-") as directory:
        snapshot = Path(directory)
        for relative in sorted(paths):
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        actual = {
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
        }
        assert actual == paths
        assert not any(path.is_symlink() for path in snapshot.rglob("*"))
        assert not (snapshot / "src/hegel_machine/__init__.py").exists()
        assert not (snapshot / "src/hegel_machine/phase3_dsl_v1.py").exists()
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(snapshot / "tools/phase3_q1_python_capacity_preflight_entrypoint_v1.py"),
                "--local-subset-node-count",
                "3",
            ],
            cwd=snapshot,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
            capture_output=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
        payload = json.loads(completed.stdout, object_pairs_hook=_strict_object)
        assert len(payload["loaded_project_modules"]) == 27
        assert payload["normal_package_initializer_executed"] is False
        assert payload["target_truth_accessed"] is False


def test_entrypoint_source_contains_empty_package_bootstrap_and_no_target_import() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'ModuleType("hegel_machine")' in source
    assert "phase3_dsl_v1 as" not in source
    assert "phase3_q0_quotient_contract_v1 as" not in source
    assert "python -I -S -B" in source
