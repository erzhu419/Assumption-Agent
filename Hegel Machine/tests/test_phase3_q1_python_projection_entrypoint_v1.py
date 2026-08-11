from __future__ import annotations

import ast
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
qualification = importlib.import_module("hegel_machine.phase3_q1_qualification_wire_v1")
ENTRYPOINT = ROOT / "tools/phase3_q1_python_projection_entrypoint_v1.py"
ACTION = "bounded-node3-golden-v1"
MAXIMUM_STDOUT_BYTES = 1024 * 1024


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        assert key not in value
        value[key] = item
    return value


def _strict_load(payload: bytes):
    return json.loads(
        payload,
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@pytest.fixture(scope="module")
def actor_result(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("q05b-python-output")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(ENTRYPOINT),
            "--action",
            ACTION,
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        timeout=420,
    )
    assert completed.returncode == 0, completed.stdout[-4096:].decode(
        "utf-8", "replace"
    )
    assert completed.stderr == b""
    assert completed.stdout.endswith(b"\n")
    assert completed.stdout.count(b"\n") == 1
    assert len(completed.stdout) < MAXIMUM_STDOUT_BYTES
    payload = _strict_load(completed.stdout)
    assert completed.stdout == _canonical_json_bytes(payload)
    return completed, payload, output_dir


def _module_names_from_source() -> tuple[str, ...]:
    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "MODULE_NAMES"
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple)
            return value
    raise AssertionError("MODULE_NAMES not found")


def _copy_allowlisted_tree(destination: Path) -> set[str]:
    module_names = _module_names_from_source()
    paths = {
        *(f"src/hegel_machine/{name}.py" for name in module_names),
        "config/phase3_q05b_node3_dual_projection_qualification_v1.json",
        "config/phase3_q1_archive_projection_freeze_v1.json",
        "docs/Hegel_Machine_Phase3A_Q05a_Q1_Archive_Projection_Engineering_Freeze_v1.md",
        "docs/Hegel_Machine_Phase3A_Q05b_Node3_Dual_Projection_Qualification_Engineering_v1.md",
        "tools/phase3_q1_python_projection_entrypoint_v1.py",
    }
    for relative in sorted(paths):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return paths


def _run_error(entrypoint: Path, *arguments: str) -> tuple[subprocess.CompletedProcess, dict]:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(entrypoint), *arguments],
        cwd=entrypoint.parents[1],
        env=_environment(),
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 1
    assert completed.stderr == b""
    assert completed.stdout.count(b"\n") == 1
    payload = _strict_load(completed.stdout)
    assert completed.stdout == _canonical_json_bytes(payload)
    return completed, payload


def test_actor_is_explicit_bounded_node3_and_keeps_all_authority_null(actor_result) -> None:
    _completed, payload, _output_dir = actor_result
    assert tuple(sorted(payload)) == tuple(sorted(qualification.ACTOR_STDOUT_REQUIRED_FIELDS))
    assert qualification.validate_actor_stdout_envelope_v1(
        _canonical_json_bytes(payload)
    ) == payload
    assert payload["actor_id"] == "PYTHON_ENDPOINT"
    assert payload["implementation_id"] == (
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1"
    )
    assert payload["schema_version"] == "hegel-q05b-actor-envelope/1"
    assert payload["action_id"] == ACTION
    assert payload["status"] == "BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED"
    assert payload["file_count"] == 5
    assert payload["q1_state"] == "NOT_RUN"
    assert payload["q1_gate_count"] == 0
    assert payload["q1_gate_mask"] == 0
    assert payload["q1_formal_roots"] is None
    assert payload["q1_output_slots"] == [None] * 8
    for name in (
        "neutral_manifest_raw_sha256",
        "neutral_manifest_root",
        "runtime_identity_sha256",
        "sidecar_manifest_raw_sha256",
        "sidecar_manifest_root",
        "source_identity_sha256",
    ):
        assert len(payload[name]) == 64
        assert bytes.fromhex(payload[name]).hex() == payload[name]


def test_actor_binds_exact_odd_sink_counts_and_complete_preimages(actor_result) -> None:
    _completed, payload, output_dir = actor_result
    expected_paths = tuple(
        value.decode("ascii") for value in qualification.ORDERED_OUTPUT_RELATIVE_PATHS
    )
    actual_paths = tuple(
        sorted(
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        )
    )
    assert actual_paths == tuple(sorted(expected_paths))
    assert all(
        ((output_dir / relative).stat().st_mode & 0o777) == 0o444
        for relative in expected_paths
    )
    files = {relative: (output_dir / relative).read_bytes() for relative in expected_paths}
    neutral = files[payload["neutral_manifest_relative_path"]]
    sidecar = files[payload["sidecar_manifest_relative_path"]]
    assert len(neutral) == payload["neutral_manifest_length"]
    assert sha256(neutral).hexdigest() == payload["neutral_manifest_raw_sha256"]
    assert len(sidecar) == payload["sidecar_manifest_length"]
    assert sha256(sidecar).hexdigest() == payload["sidecar_manifest_raw_sha256"]

    preimages = tuple(files[path] for path in expected_paths[:3])
    leaf = qualification.decode_full_v16_leaf_manifest_v1(preimages[0])
    assert len(leaf.rows) == 810
    assert len(preimages[0]) == 70244
    assert leaf.manifest_root.hex() == (
        "3fefacd3db59294f2b6d44a5d0b813e73af3ec84742a24ab846bbdacae6c1f1b"
    )
    odd = qualification.decode_node3_partition_evidence_v1(preimages[1])
    sink = qualification.decode_node3_partition_evidence_v1(preimages[2])
    assert (odd.input_signature_id, sink.input_signature_id) == (1, 2)
    assert (len(odd.coverage_rows), len(sink.coverage_rows)) == (846, 846)
    assert (len(odd.stream_rows), len(sink.stream_rows)) == (4, 4)
    for evidence in (odd, sink):
        for stream_row in evidence.stream_rows:
            assert len(stream_row) == 5
            projected = stream_row[1]
            trace = stream_row[3]
            counting = stream_row[4]
            assert len(trace) == 6
            assert trace[2] == projected[7]
            assert len(trace[3]) == projected[5][3]
            assert trace[4]
            assert trace[5]
            assert len(counting) == 15
            assert counting[5] == projected[5][3]
            assert counting[7] == projected[5][5]
            assert counting[8] == projected[5][6]
            assert counting[9:13] == projected[5:9]
            assert counting[13:] == (0, 0)
    assert [len(odd.record_set_object[index]) for index in (4, 5, 6)] == [110, 86, 40]
    assert [len(sink.record_set_object[index]) for index in (4, 5, 6)] == [144, 112, 28]
    assert sum(row[0][7] for row in odd.coverage_rows) == 1048
    assert sum(row[0][7] for row in sink.coverage_rows) == 1101
    replayed_sidecar = qualification.replay_sidecar_manifest_v1(sidecar, preimages)
    assert replayed_sidecar.manifest_root.hex() == payload["sidecar_manifest_root"]
    golden = qualification.decode_node3_golden_manifest_v1(neutral)
    assert golden.manifest_root.hex() == payload["neutral_manifest_root"]
    assert tuple(row[0] for row in golden.partition_summaries) == (1, 2)
    assert tuple(
        (row[7], row[10], row[13], row[12])
        for row in golden.partition_summaries
    ) == ((1048, 40, 59, 110), (1101, 28, 84, 144))


def test_neutral_bytes_exclude_actor_source_and_runtime_identity(actor_result) -> None:
    completed, payload, output_dir = actor_result
    neutral = (
        output_dir / qualification.NODE3_GOLDEN_MANIFEST_RELATIVE_PATH.decode("ascii")
    ).read_bytes()
    assert len(completed.stdout) < 4096
    assert b"canonical_cbor_hex" not in completed.stdout
    assert str(output_dir).encode("utf-8") not in completed.stdout
    for value in (
        payload["actor_id"],
        payload["implementation_id"],
        payload["source_identity_sha256"],
        payload["runtime_identity_sha256"],
    ):
        assert value.encode("ascii") not in neutral


def test_empty_package_recursive_allowlist_is_exact_and_target_modules_absent(
    actor_result,
) -> None:
    _completed, _payload, _output_dir = actor_result
    module_names = _module_names_from_source()
    assert len(module_names) == 35
    forbidden = {
        "hegel_machine.phase3_dsl_v1",
        "hegel_machine.phase3_m25_rows_v1",
        "hegel_machine.phase3_m25_split_v1",
        "hegel_machine.phase3_m25_formal_static_basis_v1",
    }
    assert forbidden.isdisjoint(f"hegel_machine.{name}" for name in module_names)

    with tempfile.TemporaryDirectory(prefix="hegel-q05b-python-allowlist-") as directory:
        snapshot = Path(directory)
        paths = _copy_allowlisted_tree(snapshot)
        actual = {
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
        }
        assert actual == paths
        assert not any(path.is_symlink() for path in snapshot.rglob("*"))
        assert not (snapshot / "src/hegel_machine/__init__.py").exists()
        assert not (snapshot / "src/hegel_machine/phase3_dsl_v1.py").exists()
        _completed, error = _run_error(
            snapshot / "tools/phase3_q1_python_projection_entrypoint_v1.py"
        )
        assert error["error_code"] == "Q1_PROJECTION_ACTION_NOT_ADMITTED"


def test_default_and_any_non_node3_action_fail_before_engine_execution() -> None:
    _completed, default = _run_error(ENTRYPOINT)
    assert default["error_code"] == "Q1_PROJECTION_ACTION_NOT_ADMITTED"
    assert default["full_node6_executed"] is False
    _completed, node6 = _run_error(
        ENTRYPOINT,
        "--action",
        "bounded-node6-golden-v1",
    )
    assert node6["error_code"] == "Q1_PROJECTION_ACTION_NOT_ADMITTED"
    assert node6["q1_state"] == "NOT_RUN"


def test_output_directory_must_be_explicit_absolute_empty_and_nonsymlink() -> None:
    _completed, relative = _run_error(
        ENTRYPOINT,
        "--action",
        ACTION,
        "--output-dir",
        "relative-output",
    )
    assert relative["error_code"] == "FAIL_Q1_PROJECTION_OUTPUT_DIR"
    with tempfile.TemporaryDirectory(prefix="hegel-q05b-output-policy-") as directory:
        root = Path(directory)
        nonempty = root / "nonempty"
        nonempty.mkdir()
        (nonempty / "untrusted").write_bytes(b"x")
        _completed, occupied = _run_error(
            ENTRYPOINT,
            "--action",
            ACTION,
            "--output-dir",
            str(nonempty),
        )
        assert occupied["error_code"] == "FAIL_Q1_PROJECTION_OUTPUT_DIR"
        real = root / "real"
        real.mkdir()
        link = root / "link"
        link.symlink_to(real, target_is_directory=True)
        _completed, symlink = _run_error(
            ENTRYPOINT,
            "--action",
            ACTION,
            "--output-dir",
            str(link),
        )
        assert symlink["error_code"] == "FAIL_Q1_PROJECTION_OUTPUT_DIR"


@pytest.mark.parametrize(
    ("config_relative", "mutation"),
    [
        (
            "config/phase3_q05b_node3_dual_projection_qualification_v1.json",
            "duplicate",
        ),
        (
            "config/phase3_q05b_node3_dual_projection_qualification_v1.json",
            "nan",
        ),
        ("config/phase3_q1_archive_projection_freeze_v1.json", "duplicate"),
        ("config/phase3_q1_archive_projection_freeze_v1.json", "nan"),
    ],
)
def test_config_json_duplicate_and_nonfinite_tokens_fail_closed(
    config_relative: str,
    mutation: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="hegel-q05b-json-") as directory:
        snapshot = Path(directory)
        _copy_allowlisted_tree(snapshot)
        config_path = snapshot / config_relative
        payload = config_path.read_text(encoding="utf-8")
        if mutation == "duplicate":
            needle = next(
                line
                for line in payload.splitlines()
                if line.startswith('  "freeze_id":')
            )
            payload = payload.replace(needle, needle + "\n" + needle, 1)
        else:
            needle = '    "q1_gate_count": 0,'
            payload = payload.replace(needle, '    "q1_gate_count": NaN,', 1)
        config_path.write_text(payload, encoding="utf-8")
        output_dir = snapshot / "output"
        output_dir.mkdir()
        _completed, error = _run_error(
            snapshot / "tools/phase3_q1_python_projection_entrypoint_v1.py",
            "--action",
            ACTION,
            "--output-dir",
            str(output_dir),
        )
        assert error["error_code"] == "FAIL_Q1_PROJECTION_CONFIG_WIRE"
        assert error["q1_state"] == "NOT_RUN"
        assert error["q1_formal_roots"] is None


def test_actual_precondition_nested_bool_integer_alias_fails_closed() -> None:
    with tempfile.TemporaryDirectory(prefix="hegel-q05b-config-type-") as directory:
        snapshot = Path(directory)
        _copy_allowlisted_tree(snapshot)
        config_path = (
            snapshot
            / "config/phase3_q05b_node3_dual_projection_qualification_v1.json"
        )
        config = _strict_load(config_path.read_bytes())
        config["actual_preconditions"][
            "attempt_unique_docker_execution_authority_required"
        ] = 1
        config_path.write_bytes(_canonical_json_bytes(config))
        output_dir = snapshot / "output"
        output_dir.mkdir()
        _completed, error = _run_error(
            snapshot / "tools/phase3_q1_python_projection_entrypoint_v1.py",
            "--action",
            ACTION,
            "--output-dir",
            str(output_dir),
        )
        assert error["error_code"] == "FAIL_Q1_PROJECTION_CONFIG_BINDING"
        assert error["detail"].startswith("primary_q05b:actual_preconditions differs")
        assert error["q1_state"] == "NOT_RUN"
        assert error["q1_gate_count"] == 0
        assert error["q1_gate_mask"] == 0
        assert error["q1_formal_roots"] is None
        assert error["q1_output_slots"] == [None] * 8


def test_source_is_direct_empty_package_endpoint_without_target_imports() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'ModuleType("hegel_machine")' in source
    assert "python -I -S -B" in source
    assert "phase3_dsl_v1 as" not in source
    assert "phase3_m25_rows_v1 as" not in source
    assert "phase3_m25_split_v1 as" not in source
    assert "phase3_m25_formal_static_basis_v1 as" not in source
    assert "phase3_q1_qualification_wire_v1 as qualification" in source
    assert "--output-dir" in source


def test_primary_actual_preconditions_use_shared_wire_authority() -> None:
    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_primary_q05b_config_static"
    )
    matches = []
    for node in ast.walk(validator):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        try:
            path = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if path == ("actual_preconditions",):
            matches.append(node)
    assert len(matches) == 1
    assert ast.unparse(matches[0].args[1]) == (
        "qualification.COMMIT_A_ACTUAL_PRECONDITIONS_V1"
    )

    primary = _strict_load(
        (
            ROOT
            / "config/phase3_q05b_node3_dual_projection_qualification_v1.json"
        ).read_bytes()
    )
    assert primary["actual_preconditions"] == (
        qualification.COMMIT_A_ACTUAL_PRECONDITIONS_V1
    )
