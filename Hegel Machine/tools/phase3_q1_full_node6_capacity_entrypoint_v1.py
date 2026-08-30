#!/usr/bin/env python3
"""Isolated Python endpoint for the admitted full node-6 diagnostic.

The only production action consumes an exact admission JSON plus duplicate
capsule/source identities on the command line.  It measures Odd then Sink once
each, projects the returned same-run snapshots, writes five small canonical
JSON sidecars, and emits one canonical JSON line.  It cannot run a bounded CLI
action, start Q1, or read target/split/role inputs.

Execute only with pinned CPython 3.11.15 as
``/usr/local/bin/python3 -I -S -B <this-file> --action
full-node6-capacity-preflight-v1 --admission-json ABSOLUTE_FILE
--capsule-digest sha256:HEX64 --source-commit HEX40 --output-dir
ABSOLUTE_EMPTY_DIR --scratch-dir ABSOLUTE_EMPTY_WRITABLE_DIR``.
"""

from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import ModuleType
from typing import Callable, Final, NoReturn


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = PROJECT_ROOT / "src"
PACKAGE_ROOT: Final = SOURCE_ROOT / "hegel_machine"
ACTION_ID: Final = "full-node6-capacity-preflight-v1"
SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-capacity-diagnostic/1"
ENDPOINT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-python-full-node6-capacity-endpoint/1"
)
ERROR_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-python-full-node6-capacity-endpoint-error/1"
)
ACTOR_ID: Final = "PYTHON_ENDPOINT"
IMPLEMENTATION_ID: Final = "hegel-python-q1-full-capacity-preflight-engine-v1"
MAXIMUM_STDOUT_BYTES: Final = 1024 * 1024
MAXIMUM_SIDECAR_BYTES: Final = 16 * 1024 * 1024

ROOT_MODULE_NAMES: Final = (
    "phase3_q1_full_node6_capacity_preflight_v1",
    "phase3_q1_full_node6_projection_v1",
)
MODULE_NAMES: Final = (
    "phase3_m3_bounded_enumerator_shrink2_v1",
    "phase3_m3_bounded_enumerator_shrink3_v1",
    "phase3_m3_bounded_enumerator_shrink4_v1",
    "phase3_m3_bounded_enumerator_shrink5_v1",
    "phase3_m3_bounded_enumerator_shrink6_v1",
    "phase3_m3_bounded_enumerator_v1",
    "phase3_m3_dsl_core_v1",
    "phase3_m3_record_wire_v1",
    "phase3_m3_shrink1_core_v1",
    "phase3_m3_shrink2_core_v1",
    "phase3_m3_shrink3_core_v1",
    "phase3_m3_shrink4_core_v1",
    "phase3_m3_shrink5_core_v1",
    "phase3_m3_shrink6_core_v1",
    "phase3_q0_evaluator_v1",
    "phase3_q0_input_adapter_v1",
    "phase3_q1_archive_projection_v1",
    "phase3_q1_capacity_preflight_v1",
    "phase3_q1_external_sort_profile_v1",
    "phase3_q1_formal_archive_contract_v1",
    "phase3_q1_full_node6_capacity_preflight_v1",
    "phase3_q1_full_node6_projection_v1",
    "phase3_q1_partition_snapshot_v1",
    "phase3_q1_quotient_contract_v1",
    "phase3_q1_universe_v1",
    "strict_ast_shrink1_v1",
    "strict_ast_shrink2_v1",
    "strict_ast_shrink3_v1",
    "strict_ast_shrink4_v1",
    "strict_ast_shrink5_v1",
    "strict_ast_shrink6_v1",
    "strict_ast_v1",
    "strict_cbor_v1",
)
MODULE_SOURCE_PATHS: Final = tuple(
    f"src/hegel_machine/{name}.py" for name in MODULE_NAMES
)
FORBIDDEN_PROJECT_MODULES: Final = (
    "hegel_machine.__init__",
    "hegel_machine.phase3_dsl_v1",
    "hegel_machine.phase3_m25_rows_v1",
    "hegel_machine.phase3_m25_split_v1",
    "hegel_machine.phase3_m25_formal_static_basis_v1",
)
FORBIDDEN_NETWORK_IMPORT_ROOTS: Final = frozenset(
    ("aiohttp", "ftplib", "http", "requests", "socket", "urllib")
)


class EndpointError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise EndpointError(code, detail)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            _fail("FAIL_FULL_NODE6_ADMISSION_JSON", f"duplicate key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> NoReturn:
    _fail("FAIL_FULL_NODE6_ADMISSION_JSON", f"non-finite token {value!r}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        _fail("FAIL_FULL_NODE6_ENDPOINT_JSON", str(error))


def _canonical_existing_file(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        _fail("REJECT_FULL_NODE6_ADMISSION_PATH", "admission path must be absolute")
    try:
        resolved = path.resolve(strict=True)
        status = path.stat(follow_symlinks=False)
    except OSError as error:
        _fail("REJECT_FULL_NODE6_ADMISSION_PATH", str(error))
    if (
        resolved != path
        or path.is_symlink()
        or not stat.S_ISREG(status.st_mode)
    ):
        _fail(
            "REJECT_FULL_NODE6_ADMISSION_PATH",
            "admission must be one canonical nonsymlink regular file",
        )
    return path


def _canonical_empty_output_dir(path_text: str) -> tuple[Path, tuple[int, int]]:
    path = Path(path_text)
    if not path.is_absolute():
        _fail("REJECT_FULL_NODE6_OUTPUT_DIR", "output directory must be absolute")
    try:
        resolved = path.resolve(strict=True)
        status = path.stat(follow_symlinks=False)
        nonempty = any(path.iterdir())
    except OSError as error:
        _fail("REJECT_FULL_NODE6_OUTPUT_DIR", str(error))
    if (
        resolved != path
        or path.is_symlink()
        or not stat.S_ISDIR(status.st_mode)
        or nonempty
    ):
        _fail(
            "REJECT_FULL_NODE6_OUTPUT_DIR",
            "output must be one canonical nonsymlink empty directory",
        )
    return path, (status.st_dev, status.st_ino)


def _canonical_empty_scratch_dir(path_text: str) -> tuple[Path, tuple[int, int]]:
    path = Path(path_text)
    if not path.is_absolute():
        _fail("REJECT_FULL_NODE6_SCRATCH_DIR", "scratch directory must be absolute")
    try:
        resolved = path.resolve(strict=True)
        status = path.stat(follow_symlinks=False)
        nonempty = any(path.iterdir())
    except OSError as error:
        _fail("REJECT_FULL_NODE6_SCRATCH_DIR", str(error))
    if (
        resolved != path
        or path.is_symlink()
        or not stat.S_ISDIR(status.st_mode)
        or nonempty
    ):
        _fail(
            "REJECT_FULL_NODE6_SCRATCH_DIR",
            "scratch must be one canonical nonsymlink empty directory",
        )
    identity = (status.st_dev, status.st_ino)
    probe_name = ".hegel-python-full-node6-scratch-probe"
    directory_fd: int | None = None
    probe_fd: int | None = None
    try:
        directory_fd = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        probe_fd = os.open(
            probe_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fsync(probe_fd)
        os.close(probe_fd)
        probe_fd = None
        os.unlink(probe_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as error:
        _fail("REJECT_FULL_NODE6_SCRATCH_DIR", f"writable probe: {error}")
    finally:
        if probe_fd is not None:
            os.close(probe_fd)
        if directory_fd is not None:
            os.close(directory_fd)
    _prove_scratch_empty(path, identity)
    return path, identity


def _prove_scratch_empty(path: Path, identity: tuple[int, int]) -> None:
    try:
        status = path.stat(follow_symlinks=False)
        nonempty = any(path.iterdir())
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SCRATCH_CLEANUP", str(error))
    if (
        path.is_symlink()
        or not stat.S_ISDIR(status.st_mode)
        or (status.st_dev, status.st_ino) != identity
        or nonempty
    ):
        _fail(
            "FAIL_FULL_NODE6_SCRATCH_CLEANUP",
            "scratch identity changed or transient paths remain",
        )


def _parse_invocation(
    argv: list[str],
) -> tuple[
    Path,
    str,
    str,
    Path,
    tuple[int, int],
    Path,
    tuple[int, int],
]:
    if (
        len(argv) != 12
        or argv[0:2] != ["--action", ACTION_ID]
        or argv[2] != "--admission-json"
        or argv[4] != "--capsule-digest"
        or argv[6] != "--source-commit"
        or argv[8] != "--output-dir"
        or argv[10] != "--scratch-dir"
    ):
        _fail(
            "Q1_FULL_CAPACITY_ACTION_NOT_ADMITTED",
            "the only admitted invocation is the exact full-node6 action with admission JSON, capsule digest, source commit, output directory, and scratch directory",
        )
    admission = _canonical_existing_file(argv[3])
    output, identity = _canonical_empty_output_dir(argv[9])
    scratch, scratch_identity = _canonical_empty_scratch_dir(argv[11])
    return (
        admission,
        argv[5],
        argv[7],
        output,
        identity,
        scratch,
        scratch_identity,
    )


def _relative_dependencies(module_name: str) -> tuple[str, ...]:
    path = PACKAGE_ROOT / f"{module_name}.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        _fail("FAIL_FULL_NODE6_SOURCE_CLOSURE", f"{module_name}: {error}")
    output: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            forbidden = sorted(roots & FORBIDDEN_NETWORK_IMPORT_ROOTS)
            if forbidden:
                _fail(
                    "FAIL_FULL_NODE6_OFFLINE_SOURCE",
                    f"{module_name} imports network module {forbidden}",
                )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_NETWORK_IMPORT_ROOTS:
                _fail(
                    "FAIL_FULL_NODE6_OFFLINE_SOURCE",
                    f"{module_name} imports network module {root}",
                )
            continue
        if node.level != 1:
            _fail(
                "FAIL_FULL_NODE6_SOURCE_CLOSURE",
                f"{module_name} uses unsupported relative import level",
            )
        if node.module is not None:
            output.add(node.module.split(".", 1)[0])
        else:
            output.update(
                alias.name.split(".", 1)[0]
                for alias in node.names
                if alias.name != "*"
            )
    return tuple(sorted(output))


def _recursive_allowlist_replay() -> tuple[str, ...]:
    allowed = set(MODULE_NAMES)
    pending = list(ROOT_MODULE_NAMES)
    observed: set[str] = set()
    while pending:
        name = pending.pop()
        if name in observed:
            continue
        if name not in allowed:
            _fail("FAIL_FULL_NODE6_SOURCE_CLOSURE", f"unallowlisted module {name}")
        observed.add(name)
        for dependency in _relative_dependencies(name):
            if dependency not in allowed:
                _fail(
                    "FAIL_FULL_NODE6_SOURCE_CLOSURE",
                    f"{name} imports unallowlisted {dependency}",
                )
            pending.append(dependency)
    if observed != allowed:
        _fail(
            "FAIL_FULL_NODE6_SOURCE_CLOSURE",
            f"unreachable allowlist members {sorted(allowed - observed)}",
        )
    return tuple(sorted(observed))


_recursive_allowlist_replay()
sys.path.insert(0, str(SOURCE_ROOT))
if "hegel_machine" in sys.modules:
    if __name__ == "__main__":
        _fail("FAIL_FULL_NODE6_PACKAGE_ISOLATION", "package existed before bootstrap")
else:
    package = ModuleType("hegel_machine")
    package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
    package.__package__ = "hegel_machine"
    sys.modules["hegel_machine"] = package

from hegel_machine import phase3_q1_full_node6_capacity_preflight_v1 as full  # noqa: E402
from hegel_machine import phase3_q1_full_node6_projection_v1 as projection  # noqa: E402


def _loaded_project_modules() -> tuple[str, ...]:
    expected = {f"hegel_machine.{name}" for name in MODULE_NAMES}
    loaded: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("hegel_machine."):
            continue
        source = getattr(module, "__file__", None)
        if type(source) is not str:
            _fail("FAIL_FULL_NODE6_PACKAGE_ISOLATION", f"{name} has no source path")
        try:
            relative = Path(source).resolve().relative_to(PROJECT_ROOT).as_posix()
        except (OSError, ValueError):
            _fail("FAIL_FULL_NODE6_PACKAGE_ISOLATION", f"{name} escaped project")
        if name not in expected or relative not in MODULE_SOURCE_PATHS:
            _fail(
                "FAIL_FULL_NODE6_PACKAGE_ISOLATION",
                f"unallowlisted module loaded: {name} ({relative})",
            )
        loaded.add(name)
    if loaded != expected:
        _fail(
            "FAIL_FULL_NODE6_PACKAGE_ISOLATION",
            f"allowlist load differs; missing={sorted(expected - loaded)}",
        )
    forbidden = sorted(name for name in FORBIDDEN_PROJECT_MODULES if name in sys.modules)
    if forbidden:
        _fail("FAIL_FULL_NODE6_PACKAGE_ISOLATION", f"forbidden loaded: {forbidden}")
    return tuple(sorted(loaded))


def _load_admission(path: Path, capsule_digest: str, source_commit: str):
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except EndpointError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_FULL_NODE6_ADMISSION_JSON", str(error))
    fields = tuple(full.FullNode6AdmissionV1.__dataclass_fields__)
    if type(value) is not dict or tuple(sorted(value)) != tuple(sorted(fields)):
        _fail("REJECT_FULL_NODE6_ADMISSION", "admission key registry differs")
    if type(value.get("q1_output_slots")) is not list:
        _fail("REJECT_FULL_NODE6_ADMISSION", "q1_output_slots must be JSON array")
    value["q1_output_slots"] = tuple(value["q1_output_slots"])
    try:
        admission = full.FullNode6AdmissionV1(**value)
    except (TypeError, full.FullNode6CapacityPreflightError) as error:
        _fail("REJECT_FULL_NODE6_ADMISSION", str(error))
    if (
        admission.capsule_digest != capsule_digest
        or admission.source_commit != source_commit
    ):
        _fail("REJECT_FULL_NODE6_ADMISSION", "CLI/admission identity differs")
    return admission


def _limits_json() -> dict[str, int]:
    limits = full._capacity.PreflightLimitsV1()
    return {
        name: getattr(limits, name)
        for name in limits.__dataclass_fields__
    }


def _barrier_json(row: tuple[object, ...]) -> dict[str, object]:
    names = (
        "depth",
        "barrier_kind",
        "eligible_raw_application_count",
        "strict_admitted_application_count",
        "rewrite_collapse_count",
        "new_behavior_class_count",
        "new_signature_cohort_count",
        "continuation_bank_mutation_count",
        "behavior_class_count_after_barrier",
        "signature_cohort_count_after_barrier",
        "continuation_bank_point_count_after_barrier",
        "visible_frontier_point_count_after_barrier",
    )
    if type(row) is not tuple or len(row) != len(names):
        _fail("REJECT_FULL_NODE6_RESULT", "depth barrier wire differs")
    return dict(zip(names, row, strict=True))


def _fixed_point_json(row: object) -> dict[str, object]:
    barriers = row.depth_barriers  # type: ignore[attr-defined]
    structural = barriers[-1] if barriers else None
    zero = bool(
        structural is not None
        and structural[1] == "STRUCTURAL_BOUNDARY"
        and structural[2] == 0
        and structural[5] == 0
        and structural[7] == 0
    )
    traversal_closed = row.traversal_closed  # type: ignore[attr-defined]
    saturated = row.terminal_status == full.TERMINAL_SATURATED  # type: ignore[attr-defined]
    return {
        "all_eligible_operator_and_bank_tuple_expansions_covered": traversal_closed,
        "final_continuation_bank_delta": structural[7] if structural is not None else 0,
        "final_new_behavior_class_delta": structural[5] if structural is not None else 0,
        "final_visible_frontier_delta": 0,
        "no_resource_guard_hit": row.resource_guard_id is None,  # type: ignore[attr-defined]
        "saturated_diagnostic_only": traversal_closed and saturated,
        "work_queue_empty": traversal_closed,
        "zero_delta_structural_boundary": zero,
    }


def _partition_json(row: object, projected: object | None) -> dict[str, object]:
    direct_fields = (
        "input_signature_id",
        "universe_root",
        "universe_row_count",
        "full_v16_structural_limits_applied",
        "terminal_status",
        "resource_guard_id",
        "resource_guard_name",
        "traversal_closed",
        "frozen_leaf_count",
        "raw_operator_application_count",
        "strict_admitted_application_count",
        "rewrite_collapse_count",
        "behavior_class_count",
        "signature_cohort_count",
        "continuation_bank_point_count",
        "visible_frontier_point_count",
        "maximum_bank_points_per_class",
        "maximum_frontier_points_per_class",
        "peak_raw_operator_application_count",
        "peak_behavior_class_count",
        "peak_visible_frontier_point_count",
        "peak_visible_frontier_points_per_class",
        "peak_continuation_bank_point_count",
        "peak_continuation_bank_points_per_class",
        "peak_work_queue_points",
        "peak_saturation_round_count",
        "vector_cache_entry_count",
        "vector_cache_hit_count",
        "vector_cache_miss_count",
    )
    value = {name: getattr(row, name) for name in direct_fields}
    signature_id = row.input_signature_id
    value.update(
        {
            "coverage": (
                projected.coverage.json_object() if projected is not None else None
            ),
            "diagnostic_state_projection": (
                projected.capacity_state_projection.json_object()
                if projected is not None
                else None
            ),
            "formal_wire_projection": (
                projection.formal_wire_projection_json_v1(projected)
                if projected is not None
                else None
            ),
            "complete_claim_allowed": False,
            "depth_barriers": [_barrier_json(item) for item in row.depth_barriers],
            "diagnostic_only": True,
            "elapsed_monotonic_ns": 0,
            "fixed_point": (
                _fixed_point_json(row) if projected is not None else None
            ),
            "formal_roots": None,
            "formal_roots_generated": False,
            "input_signature_name": "OddInputV1" if signature_id == 1 else "SinkInputV1",
            "limits": _limits_json(),
            "role_evaluation_performed": False,
            "same_run_record_set_root": (
                "sha256:" + projected.same_run_record_set_root.hex()
                if projected is not None
                else None
            ),
            "split_accessed": False,
            "target_truth_accessed": False,
        }
    )
    return value


def _stdout_object(
    diagnostic: object,
    projected_partitions: tuple[object | None, ...],
    *,
    formal_projection_result: object | None,
    actual_preflight_peak_scratch_bytes: int,
    actual_preflight_terminal_live_scratch_bytes: int,
) -> dict[str, object]:
    formal = (
        formal_projection_result.archive_projection_result
        if formal_projection_result is not None
        else None
    )
    return {
        "action_id": ACTION_ID,
        "active_transition_allowed": False,
        "actor_id": ACTOR_ID,
        "actual_preflight_peak_scratch_bytes": (
            actual_preflight_peak_scratch_bytes
        ),
        "actual_preflight_terminal_live_scratch_bytes": (
            actual_preflight_terminal_live_scratch_bytes
        ),
        "capsule_digest": diagnostic.capsule_digest,
        "closure_semantics_version": full._capacity.CLOSURE_SEMANTICS_VERSION,
        "diagnostic_only": True,
        "dsl_version": full._capacity.DSL_VERSION,
        "endpoint_schema_version": ENDPOINT_SCHEMA_VERSION,
        "implementation_id": IMPLEMENTATION_ID,
        "partitions": [
            _partition_json(row, projection_row)
            for row, projection_row in zip(
                diagnostic.partitions,
                projected_partitions,
                strict=True,
            )
        ],
        "q1_execution_started": False,
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q1_state": "NOT_RUN",
        "q2_state": "NOT_RUN",
        "projected_closure_bundle_frame_bytes": (
            formal.projected_closure_bundle_frame_bytes
            if formal is not None
            else None
        ),
        "projected_formal_archive_payload_bytes": (
            formal.projected_archive_payload_bytes_per_endpoint
            if formal is not None
            else None
        ),
        "projected_formal_total_output_bytes": (
            formal.projected_endpoint_total_output_bytes
            if formal is not None
            else None
        ),
        "projected_peak_formal_scratch_bytes": (
            formal.projected_endpoint_peak_scratch_bytes
            if formal is not None
            else None
        ),
        "role_evaluation_performed": False,
        "schema_version": SCHEMA_VERSION,
        "source_commit": diagnostic.source_commit,
        "split_accessed": False,
        "target_truth_accessed": False,
        "terminal_status": diagnostic.terminal_status,
    }


def _endpoint_sidecars(
    diagnostic: object,
    projected_partitions: tuple[object | None, ...],
    stdout: bytes,
) -> tuple[tuple[str, bytes], ...]:
    def common(schema_version: str, partitions: list[dict[str, object]]) -> dict[str, object]:
        return {
            "actor_id": ACTOR_ID,
            "capsule_digest": diagnostic.capsule_digest,
            "diagnostic_only": True,
            "implementation_id": IMPLEMENTATION_ID,
            "partitions": partitions,
            "q1_formal_roots": None,
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_state": "NOT_RUN",
            "schema_version": schema_version,
            "source_commit": diagnostic.source_commit,
        }

    projection_rows = [
        {
            "diagnostic_state_projection": (
                projection_row.capacity_state_projection.json_object()
                if projection_row is not None
                else None
            ),
            "formal_wire_projection": (
                projection.formal_wire_projection_json_v1(projection_row)
                if projection_row is not None
                else None
            ),
            "input_signature_id": diagnostic_row.input_signature_id,
            "input_signature_name": (
                "OddInputV1"
                if diagnostic_row.input_signature_id == 1
                else "SinkInputV1"
            ),
            "universe_root": diagnostic_row.universe_root,
        }
        for diagnostic_row, projection_row in zip(
            diagnostic.partitions,
            projected_partitions,
            strict=True,
        )
    ]
    coverage_rows = [
        {
            "coverage": (
                projection_row.coverage.json_object()
                if projection_row is not None
                else None
            ),
            "input_signature_id": diagnostic_row.input_signature_id,
            "input_signature_name": (
                "OddInputV1"
                if diagnostic_row.input_signature_id == 1
                else "SinkInputV1"
            ),
            "universe_root": diagnostic_row.universe_root,
        }
        for diagnostic_row, projection_row in zip(
            diagnostic.partitions,
            projected_partitions,
            strict=True,
        )
    ]
    fixed_rows = [
        {
            "fixed_point": (
                _fixed_point_json(diagnostic_row)
                if projection_row is not None
                else None
            ),
            "input_signature_id": diagnostic_row.input_signature_id,
            "input_signature_name": (
                "OddInputV1"
                if diagnostic_row.input_signature_id == 1
                else "SinkInputV1"
            ),
            "same_run_record_set_root": (
                "sha256:" + projection_row.same_run_record_set_root.hex()
                if projection_row is not None
                else None
            ),
            "universe_root": diagnostic_row.universe_root,
        }
        for diagnostic_row, projection_row in zip(
            diagnostic.partitions,
            projected_partitions,
            strict=True,
        )
    ]
    data_files = (
        ("diagnostic-summary-v1.json", stdout),
        (
            "projection-summary-v1.json",
            _canonical_json_bytes(
                common(
                    "hegel-phase3a-q1-full-node6-projection-summary/1",
                    projection_rows,
                )
            ),
        ),
        (
            "coverage-summary-v1.json",
            _canonical_json_bytes(
                common(
                    "hegel-phase3a-q1-full-node6-coverage-summary/1",
                    coverage_rows,
                )
            ),
        ),
        (
            "fixed-point-summary-v1.json",
            _canonical_json_bytes(
                common(
                    "hegel-phase3a-q1-full-node6-fixed-point-summary/1",
                    fixed_rows,
                )
            ),
        ),
    )
    rows = [
        {
            "byte_count": len(payload),
            "filename": filename,
            "sha256": sha256(payload).hexdigest(),
        }
        for filename, payload in sorted(data_files, key=lambda row: row[0])
    ]
    manifest = _canonical_json_bytes(
        {
            "actor_id": ACTOR_ID,
            "capsule_digest": diagnostic.capsule_digest,
            "diagnostic_only": True,
            "implementation_id": IMPLEMENTATION_ID,
            "q1_formal_roots": None,
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_state": "NOT_RUN",
            "rows": rows,
            "schema_version": (
                "hegel-phase3a-q1-full-node6-endpoint-sidecar-manifest/1"
            ),
            "source_commit": diagnostic.source_commit,
        }
    )
    return data_files + (("endpoint-manifest-v1.json", manifest),)


def _publish_sidecars(
    output_dir: Path,
    output_identity: tuple[int, int],
    files: tuple[tuple[str, bytes], ...],
) -> None:
    if tuple(path for path, _payload in files) != projection.SIDECAR_PATHS:
        _fail("FAIL_FULL_NODE6_SIDECAR_SET", "sidecar registry differs")
    if any(type(payload) is not bytes or not payload for _path, payload in files):
        _fail("FAIL_FULL_NODE6_SIDECAR_SET", "sidecar payload differs")
    if sum(len(payload) for _path, payload in files) > MAXIMUM_SIDECAR_BYTES:
        _fail("INCONCLUSIVE_FULL_NODE6_OUTPUT_LIMIT", "sidecars exceed byte limit")
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(output_dir, flags)
        status = os.fstat(descriptor)
        if (status.st_dev, status.st_ino) != output_identity or os.listdir(descriptor):
            _fail("REJECT_FULL_NODE6_OUTPUT_DIR", "output directory identity changed")
        for filename, payload in files:
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_CLOEXEC"):
                file_flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                file_flags |= os.O_NOFOLLOW
            child = os.open(filename, file_flags, 0o600, dir_fd=descriptor)
            try:
                view = memoryview(payload)
                offset = 0
                while offset < len(view):
                    written = os.write(child, view[offset:])
                    if written <= 0:
                        _fail("FAIL_FULL_NODE6_SIDECAR_WRITE", "short sidecar write")
                    offset += written
                os.fchmod(child, 0o444)
                os.fsync(child)
            finally:
                os.close(child)
        os.fsync(descriptor)
    except EndpointError:
        raise
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SIDECAR_WRITE", str(error))
    finally:
        if descriptor is not None:
            os.close(descriptor)


def run_endpoint_v1(
    argv: list[str],
    *,
    capacity_runner: Callable[[object], tuple[object, object]] = (
        full.run_full_node6_capacity_preflight_v1
    ),
    projection_runner: Callable[..., object] = (
        projection.project_full_node6_same_run_v1
    ),
    module_verifier: Callable[[], object] = _loaded_project_modules,
) -> bytes:
    (
        admission_path,
        capsule,
        commit,
        output_dir,
        output_identity,
        scratch_dir,
        scratch_identity,
    ) = _parse_invocation(argv)
    admission = _load_admission(admission_path, capsule, commit)
    if admission.projection_profile_root != projection.FROZEN_PROJECTION_PROFILE_ROOT_ID:
        _fail(
            "REJECT_FULL_NODE6_ADMISSION",
            "projection profile is not the frozen full-node6 profile",
        )
    try:
        diagnostic, snapshots = capacity_runner(admission)
        if type(diagnostic) is not full.FullNode6CapacityDiagnosticV1:
            _fail("REJECT_FULL_NODE6_RESULT", "capacity result exact type differs")
        if diagnostic.terminal_status == full.TERMINAL_SATURATED:
            with tempfile.TemporaryDirectory(
                prefix="hegel-python-full-node6-run-",
                dir=scratch_dir,
            ) as transient:
                projected = projection_runner(
                    diagnostic,
                    snapshots,
                    scratch_root=Path(transient),
                )
            if type(projected) is not projection.FullNode6ProjectionResultV1:
                _fail("REJECT_FULL_NODE6_RESULT", "projection result exact type differs")
            if projected.actual_preflight_scratch_live_bytes_at_terminal != 0:
                _fail(
                    "REJECT_FULL_NODE6_RESULT",
                    "projection scratch ledger remained live",
                )
            projected_partitions: tuple[object | None, ...] = projected.partitions
            actual_peak_scratch_bytes = (
                projected.actual_preflight_peak_scratch_bytes
            )
            formal_projection_result: object | None = projected
        else:
            # Any aggregate guard makes the complete prefix terminal.  Even a
            # preceding saturated Odd row remains counts/barriers only.
            projected_partitions = (None,) * len(diagnostic.partitions)
            actual_peak_scratch_bytes = 0
            formal_projection_result = None
        module_verifier()
        stdout = _canonical_json_bytes(
            _stdout_object(
                diagnostic,
                projected_partitions,
                formal_projection_result=formal_projection_result,
                actual_preflight_peak_scratch_bytes=actual_peak_scratch_bytes,
                actual_preflight_terminal_live_scratch_bytes=0,
            )
        )
        if len(stdout) > MAXIMUM_STDOUT_BYTES:
            _fail("INCONCLUSIVE_FULL_NODE6_OUTPUT_LIMIT", "stdout exceeds byte limit")
        _publish_sidecars(
            output_dir,
            output_identity,
            _endpoint_sidecars(diagnostic, projected_partitions, stdout),
        )
        return stdout
    finally:
        _prove_scratch_empty(scratch_dir, scratch_identity)


def _error_object(error: BaseException) -> dict[str, object]:
    code = getattr(error, "code", "FAIL_FULL_NODE6_ENDPOINT")
    detail = getattr(error, "detail", str(error))
    return {
        "active_transition_allowed": False,
        "error_code": code,
        "error_detail": detail,
        "q1_execution_started": False,
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q1_state": "NOT_RUN",
        "schema_version": ERROR_SCHEMA_VERSION,
        "split_accessed": False,
        "target_truth_accessed": False,
        "role_evaluation_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_endpoint_v1(sys.argv[1:] if argv is None else argv)
        code = 0
    except BaseException as error:  # direct actor boundary is fail closed
        payload = _canonical_json_bytes(_error_object(error))
        code = 1
    try:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    except OSError:
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
