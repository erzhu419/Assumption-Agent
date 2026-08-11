#!/usr/bin/env python3
"""Isolated bounded-node3 Python projection actor for Phase-3A-Q0.5b.

This direct-file endpoint is diagnostic-only.  It uses an empty
``hegel_machine`` package, admits exactly one explicit bounded-node3 action,
and writes the complete locally materialized snapshot-to-record, coverage,
chunk, and external-sort projection evidence to the exact neutral CBOR
sidecars in an explicit empty output directory.  Stdout is only a small
actor-specific canonical-JSON envelope.  It cannot start Q1, run node6,
create a formal output root, pass a gate, inspect target truth/splits/roles,
or issue a receipt/certificate.

Execute only as ``python -I -S -B <this-file> --action
bounded-node3-golden-v1 --output-dir /absolute/empty/directory``.
"""

from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Final, NoReturn


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = PROJECT_ROOT / "src"
PACKAGE_ROOT: Final = SOURCE_ROOT / "hegel_machine"
PRIMARY_CONFIG_RELATIVE_PATH: Final = (
    "config/phase3_q05b_node3_dual_projection_qualification_v1.json"
)
REFERENCED_Q05A_CONFIG_RELATIVE_PATH: Final = (
    "config/phase3_q1_archive_projection_freeze_v1.json"
)
PRIMARY_DOC_RELATIVE_PATH: Final = (
    "docs/Hegel_Machine_Phase3A_Q05b_Node3_Dual_Projection_Qualification_Engineering_v1.md"
)
REFERENCED_Q05A_DOC_RELATIVE_PATH: Final = (
    "docs/Hegel_Machine_Phase3A_Q05a_Q1_Archive_Projection_Engineering_Freeze_v1.md"
)
PRIMARY_CONFIG_PATH: Final = PROJECT_ROOT / PRIMARY_CONFIG_RELATIVE_PATH
REFERENCED_Q05A_CONFIG_PATH: Final = (
    PROJECT_ROOT / REFERENCED_Q05A_CONFIG_RELATIVE_PATH
)

ERROR_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-python-projection-actor-error/1"
)
IMPLEMENTATION_ID: Final = "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1"
ACTION_ID: Final = "bounded-node3-golden-v1"
MAXIMUM_STDOUT_BYTES: Final = 1024 * 1024
MAXIMUM_SIDECAR_BYTES: Final = 64 * 1024 * 1024
SIDECAR_PATHS: Final = (
    "preimages/000-full-v16-leaf-manifest-v1.cbor",
    "preimages/001-odd-node3-partition-evidence-v1.cbor",
    "preimages/002-sink-node3-partition-evidence-v1.cbor",
    "neutral/q05b-node3-sidecar-manifest-v1.cbor",
    "neutral/q05b-node3-golden-manifest-v1.cbor",
)

# This is the exact recursive relative-import closure of the three target-blind
# projection roots.  Runtime AST replay below independently proves that no
# relative dependency was omitted or added.
ROOT_MODULE_NAMES: Final = (
    "phase3_q1_archive_projection_v1",
    "phase3_q1_partition_snapshot_v1",
    "phase3_q1_qualification_wire_v1",
    "phase3_q1_semantic_coverage_v1",
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
    "phase3_q05b_wire_qualification_contract_v1",
    "phase3_q0_evaluator_v1",
    "phase3_q0_input_adapter_v1",
    "phase3_q0_quotient_contract_v1",
    "phase3_q1_archive_projection_v1",
    "phase3_q1_capacity_preflight_v1",
    "phase3_q1_external_sort_profile_v1",
    "phase3_q1_formal_archive_contract_v1",
    "phase3_q1_partition_snapshot_v1",
    "phase3_q1_qualification_wire_v1",
    "phase3_q1_quotient_contract_v1",
    "phase3_q1_semantic_coverage_v1",
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
IMPLEMENTATION_SOURCE_PATHS: Final = tuple(
    sorted(
        MODULE_SOURCE_PATHS
        + (
            PRIMARY_CONFIG_RELATIVE_PATH,
            REFERENCED_Q05A_CONFIG_RELATIVE_PATH,
            PRIMARY_DOC_RELATIVE_PATH,
            REFERENCED_Q05A_DOC_RELATIVE_PATH,
            "tools/phase3_q1_python_projection_entrypoint_v1.py",
        )
    )
)
FORBIDDEN_MODULES: Final = (
    "hegel_machine.__init__",
    "hegel_machine.phase3_dsl_v1",
    "hegel_machine.phase3_m25_rows_v1",
    "hegel_machine.phase3_m25_split_v1",
    "hegel_machine.phase3_m25_formal_static_basis_v1",
)


class EndpointError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise EndpointError(code, detail)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail("FAIL_Q1_PROJECTION_CONFIG_WIRE", f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> NoReturn:
    _fail("FAIL_Q1_PROJECTION_CONFIG_WIRE", f"non-finite JSON token {value!r}")


def _load_config(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        _fail("FAIL_Q1_PROJECTION_CONFIG_WIRE", f"{label}: {error}")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_Q1_PROJECTION_CONFIG_WIRE", f"{label}: {error}")
    if type(value) is not dict:
        _fail(
            "FAIL_Q1_PROJECTION_CONFIG_WIRE",
            f"{label}: configuration must be one object",
        )
    return value, payload


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
        _fail("FAIL_Q1_PROJECTION_OUTPUT_WIRE", str(error))


def _parse_invocation(argv: list[str]) -> tuple[str, Path, tuple[int, int]]:
    if len(argv) != 4 or argv[:2] != ["--action", ACTION_ID] or argv[2] != "--output-dir":
        _fail(
            "Q1_PROJECTION_ACTION_NOT_ADMITTED",
            f"the only admitted invocation is --action {ACTION_ID} --output-dir ABSOLUTE_EMPTY_TARGET",
        )
    path = Path(argv[3])
    if not path.is_absolute():
        _fail("FAIL_Q1_PROJECTION_OUTPUT_DIR", "output directory must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        _fail("FAIL_Q1_PROJECTION_OUTPUT_DIR", str(error))
    if resolved != path or path.is_symlink() or not path.is_dir():
        _fail(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output directory must be one existing canonical nonsymlink directory",
        )
    try:
        if any(path.iterdir()):
            _fail("FAIL_Q1_PROJECTION_OUTPUT_DIR", "output directory must be empty")
        status = path.stat(follow_symlinks=False)
    except OSError as error:
        _fail("FAIL_Q1_PROJECTION_OUTPUT_DIR", str(error))
    return ACTION_ID, path, (status.st_dev, status.st_ino)


def _relative_dependencies(module_name: str) -> tuple[str, ...]:
    source_path = PACKAGE_ROOT / f"{module_name}.py"
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError, UnicodeError) as error:
        _fail("FAIL_Q1_PROJECTION_SOURCE_CLOSURE", f"{module_name}: {error}")
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module is not None:
            dependencies.add(node.module.split(".", 1)[0])
            continue
        for alias in node.names:
            if alias.name != "*":
                dependencies.add(alias.name.split(".", 1)[0])
    return tuple(sorted(dependencies))


def _recursive_allowlist_replay() -> tuple[str, ...]:
    allowed = set(MODULE_NAMES)
    pending = list(ROOT_MODULE_NAMES)
    observed: set[str] = set()
    while pending:
        module_name = pending.pop()
        if module_name in observed:
            continue
        if module_name not in allowed:
            _fail(
                "FAIL_Q1_PROJECTION_SOURCE_CLOSURE",
                f"unallowlisted recursive dependency {module_name}",
            )
        observed.add(module_name)
        for dependency in _relative_dependencies(module_name):
            if dependency not in allowed:
                _fail(
                    "FAIL_Q1_PROJECTION_SOURCE_CLOSURE",
                    f"{module_name} imports unallowlisted {dependency}",
                )
            pending.append(dependency)
    if observed != allowed:
        _fail(
            "FAIL_Q1_PROJECTION_SOURCE_CLOSURE",
            f"allowlist has unreachable members {sorted(allowed - observed)}",
        )
    return tuple(sorted(observed))


# Never execute the historical package initializer: it exports target/control
# objects.  Only the hard-coded recursive target-blind closure above is
# admitted into this private empty package namespace.
sys.path.insert(0, str(SOURCE_ROOT))
if "hegel_machine" in sys.modules:
    _fail("FAIL_Q1_PROJECTION_PACKAGE_ISOLATION", "package existed before bootstrap")
package = ModuleType("hegel_machine")
package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package

from hegel_machine import phase3_q1_archive_projection_v1 as projection  # noqa: E402
from hegel_machine import phase3_q1_capacity_preflight_v1 as capacity  # noqa: E402
from hegel_machine import phase3_q1_partition_snapshot_v1 as snapshots  # noqa: E402
from hegel_machine import phase3_q1_qualification_wire_v1 as qualification  # noqa: E402
from hegel_machine import phase3_q1_semantic_coverage_v1 as coverage  # noqa: E402
from hegel_machine import phase3_q1_universe_v1 as universes  # noqa: E402


def _source_file_set_digest() -> str:
    digest = sha256()
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        payload = (PROJECT_ROOT / relative).read_bytes()
        path_bytes = relative.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(4, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _runtime_identity_digest() -> str:
    digest = sha256(b"HEGEL/Q05B/PYTHON_RUNTIME_IDENTITY/V1\x00")
    executable = Path(sys.executable).resolve(strict=True)
    path_bytes = executable.as_posix().encode("utf-8")
    digest.update(len(path_bytes).to_bytes(4, "big"))
    digest.update(path_bytes)
    with executable.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    version = sys.version.encode("utf-8")
    digest.update(len(version).to_bytes(4, "big"))
    digest.update(version)
    return digest.hexdigest()


def _publish_sidecars(
    output_dir: Path,
    output_dir_identity: tuple[int, int],
    files: tuple[tuple[str, bytes], ...],
) -> list[dict[str, object]]:
    if tuple(path for path, _payload in files) != SIDECAR_PATHS:
        _fail("FAIL_Q1_PROJECTION_SIDECAR_SET", "sidecar path/order differs")
    if any(type(payload) is not bytes or not payload for _path, payload in files):
        _fail("FAIL_Q1_PROJECTION_SIDECAR_SET", "sidecar payload is empty or non-bytes")
    total = sum(len(payload) for _path, payload in files)
    if total > MAXIMUM_SIDECAR_BYTES:
        _fail(
            "INCONCLUSIVE_Q1_PROJECTION_OUTPUT_LIMIT",
            f"sidecar bytes {total} exceed {MAXIMUM_SIDECAR_BYTES}",
        )
    root_descriptor: int | None = None
    child_descriptors: dict[str, int] = {}
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        root_descriptor = os.open(output_dir, directory_flags)
        status = os.fstat(root_descriptor)
        if (status.st_dev, status.st_ino) != output_dir_identity:
            _fail("FAIL_Q1_PROJECTION_OUTPUT_DIR", "output directory identity changed")
        if os.listdir(root_descriptor):
            _fail("FAIL_Q1_PROJECTION_OUTPUT_DIR", "output directory changed or is nonempty")
        for directory in ("preimages", "neutral"):
            os.mkdir(directory, mode=0o700, dir_fd=root_descriptor)
            child_descriptors[directory] = os.open(
                directory,
                directory_flags,
                dir_fd=root_descriptor,
            )
        for relative, payload in files:
            directory, filename = relative.split("/", 1)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(
                filename,
                flags,
                0o600,
                dir_fd=child_descriptors[directory],
            )
            try:
                view = memoryview(payload)
                written = 0
                while written < len(view):
                    count = os.write(descriptor, view[written:])
                    if count <= 0:
                        _fail("FAIL_Q1_PROJECTION_SIDECAR_WRITE", f"short write: {relative}")
                    written += count
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for descriptor in child_descriptors.values():
            os.fsync(descriptor)
        os.fsync(root_descriptor)
    except EndpointError:
        raise
    except OSError as error:
        _fail("FAIL_Q1_PROJECTION_SIDECAR_WRITE", str(error))
    finally:
        for descriptor in child_descriptors.values():
            os.close(descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
    return [
        {
            "path": relative,
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
        for relative, payload in files
    ]


def _loaded_project_modules() -> tuple[str, ...]:
    expected = {f"hegel_machine.{name}" for name in MODULE_NAMES}
    loaded: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("hegel_machine."):
            continue
        path_value = getattr(module, "__file__", None)
        if type(path_value) is not str:
            _fail("FAIL_Q1_PROJECTION_PACKAGE_ISOLATION", f"{name} has no source path")
        path = Path(path_value).resolve()
        try:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            _fail("FAIL_Q1_PROJECTION_PACKAGE_ISOLATION", f"{name} escaped project")
        if name not in expected or relative not in MODULE_SOURCE_PATHS:
            _fail(
                "FAIL_Q1_PROJECTION_PACKAGE_ISOLATION",
                f"unallowlisted module loaded: {name} ({relative})",
            )
        loaded.add(name)
    if loaded != expected:
        _fail(
            "FAIL_Q1_PROJECTION_PACKAGE_ISOLATION",
            f"allowlist load differs; missing={sorted(expected - loaded)}",
        )
    forbidden = sorted(name for name in FORBIDDEN_MODULES if name in sys.modules)
    if forbidden:
        _fail("FAIL_Q1_PROJECTION_PACKAGE_ISOLATION", f"forbidden loaded: {forbidden}")
    return tuple(sorted(loaded))


def _exact_config_value(
    config: dict[str, object],
    path: tuple[str, ...],
    expected: object,
    *,
    label: str,
) -> None:
    value: object = config
    for key in path:
        if type(value) is not dict or key not in value:
            _fail(
                "FAIL_Q1_PROJECTION_CONFIG_BINDING",
                f"{label}:{'.'.join(path)} absent",
        )
        value = value[key]
    if not _json_value_type_exact_v1(value, expected):
        _fail(
            "FAIL_Q1_PROJECTION_CONFIG_BINDING",
            f"{label}:{'.'.join(path)} differs: {value!r}",
        )


def _json_value_type_exact_v1(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        assert type(value) is dict
        return set(value) == set(expected) and all(
            _json_value_type_exact_v1(value[key], expected[key])
            for key in expected
        )
    if type(expected) is list:
        assert type(value) is list
        return len(value) == len(expected) and all(
            _json_value_type_exact_v1(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def _exact_config_keys(
    config: dict[str, object],
    expected: tuple[str, ...],
    *,
    label: str,
) -> None:
    if tuple(sorted(config)) != tuple(sorted(expected)):
        _fail(
            "FAIL_Q1_PROJECTION_CONFIG_BINDING",
            f"{label}: top-level key registry differs",
        )


def _validate_referenced_q05a_config(config: dict[str, object]) -> None:
    def exact(path: tuple[str, ...], expected: object) -> None:
        _exact_config_value(config, path, expected, label="referenced_q05a")

    exact(("schema_version",), "hegel-phase3a-q05a-q1-archive-projection-freeze/1")
    exact(("freeze_id",), "hegel-phase3a-q05a-q1-archive-projection-freeze-v1")
    exact(("dsl_version",), capacity.DSL_VERSION)
    exact(("closure_semantics_version",), capacity.CLOSURE_SEMANTICS_VERSION)
    exact(("authority", "q1_state"), "NOT_RUN")
    exact(("authority", "q1_execution_started"), False)
    exact(("authority", "q1_gate_count"), 0)
    exact(("authority", "q1_gate_mask"), 0)
    exact(("authority", "q1_gate_total"), 20)
    exact(("authority", "q1_formal_roots"), None)
    exact(("authority", "q1_receipt"), None)
    exact(("authority", "q2_state"), "NOT_RUN")
    exact(("authority", "target_truth_accessed"), False)
    exact(("authority", "split_accessed"), False)
    exact(("authority", "role_evaluation_performed"), False)
    exact(("authority", "m3_formal_roots"), None)
    exact(("authority", "outside_certificate_issued"), False)
    exact(("authority", "active_transition_allowed"), False)
    exact(("claim_boundary", "formal_archive_materialization_allowed_now"), False)
    exact(("claim_boundary", "full_node6_capacity_preflight_allowed_now"), False)
    exact(("claim_boundary", "gate10_passed_now"), False)
    exact(("source_freeze_requirements", "local_python_projection_prototype_present"), True)
    exact(
        ("source_freeze_requirements", "local_python_prototype_scope"),
        "BOUNDED_NODE3_MATERIALIZED_GOLDEN_AND_TAMPER_REPLAY_ONLY",
    )
    exact(("source_freeze_requirements", "qualified_isolated_python_source_available"), False)
    exact(("golden_vector_plan", "goldens_qualified"), False)
    exact(("gate10_qualification", "passed_predicate_count"), 0)
    exact(("gate10_qualification", "predicate_mask"), 0)
    exact(
        ("ordered_q1_output_slots",),
        [
            [1, "odd_signature_archive_manifest_root", None],
            [2, "odd_signature_saturation_state_root", None],
            [3, "sink_signature_archive_manifest_root", None],
            [4, "sink_signature_saturation_state_root", None],
            [5, "q1_closure_bundle_root", None],
            [6, "q1_dual_replay_agreement_root", None],
            [7, "q1_target_blind_access_ledger_root", None],
            [8, "q1_completion_receipt_root", None],
        ],
    )
    expected_registry = [
        [guard_id, name] for guard_id, name in capacity.RESOURCE_GUARD_REGISTRY
    ]
    exact(("resource_guard_registry",), expected_registry)
    known = config.get("golden_vector_plan")
    if type(known) is not dict or type(known.get("known_constants")) is not dict:
        _fail("FAIL_Q1_PROJECTION_CONFIG_BINDING", "referenced_q05a: golden constants absent")
    constants = known["known_constants"]
    assert isinstance(constants, dict)
    expected_counts = {
        "node3_odd_counts_raw_class_frontier_bank": [1048, 40, 59, 110],
        "node3_sink_counts_raw_class_frontier_bank": [1101, 28, 84, 144],
    }
    for key, expected in expected_counts.items():
        if constants.get(key) != expected:
            _fail(
                "FAIL_Q1_PROJECTION_CONFIG_BINDING",
                f"referenced_q05a:{key} differs",
            )
    generated = universes.all_production_universes_v1()
    for key, item in zip(
        ("odd_universe_root", "sink_universe_root"), generated, strict=True
    ):
        if constants.get(key) != item.universe_root.hex():
            _fail(
                "FAIL_Q1_PROJECTION_CONFIG_BINDING",
                f"referenced_q05a:{key} differs",
            )


def _validate_primary_q05b_config_static(
    config: dict[str, object],
    full_leaf: object,
) -> None:
    def exact(path: tuple[str, ...], expected: object) -> None:
        _exact_config_value(config, path, expected, label="primary_q05b")

    _exact_config_keys(
        config,
        (
            "schema_version",
            "freeze_id",
            "phase_position",
            "engineering_status",
            "claim_boundary",
            "authority",
            "version_bindings",
            "qualification_numeric_tag_registry",
            "formal_q1_tag_registry_remains_separate",
            "schema_registry",
            "strict_type_and_nullability",
            "external_sort_trace_wire",
            "counting_discard_wire",
            "registry_and_profile_roots",
            "semantic_source_bindings",
            "full_v16_leaf_manifest",
            "q1_formal_input_roots",
            "bounded_node3_contract",
            "sidecar_layout",
            "actor_stdout_envelope",
            "chunk_boundary",
            "qualification_receipt_protocol",
            "qualification_predicate_registry",
            "failure_code_registry",
            "actual_preconditions",
            "two_commit_protocol",
        ),
        label="primary_q05b",
    )
    exact(
        ("schema_version",),
        "hegel-phase3a-q05b-node3-dual-projection-qualification/1",
    )
    exact(
        ("freeze_id",),
        "hegel-phase3a-q05b-node3-dual-projection-qualification-v1",
    )
    exact(("phase_position",), "PHASE3A_Q0_5B_BEFORE_Q1_EXECUTION")
    exact(
        ("engineering_status",),
        qualification.QUALIFICATION_ENGINEERING_STATUS,
    )
    exact(
        ("claim_boundary",),
        {
            "allowed_claim": "TARGET_BLIND_BOUNDED_NODE3_QUALIFICATION_WIRE_SOURCE_FROZEN",
            "full_node6_executed": False,
            "formal_q1_fixed_point_claimed": False,
            "formal_q1_output_root_generated": False,
            "q1_gate_pass_claimed": False,
            "q1_complete_claim_allowed": False,
            "target_truth_accessed": False,
            "split_accessed": False,
            "role_evaluation_performed": False,
        },
    )
    exact(
        ("authority",),
        {
            "qualification_state": qualification.QUALIFICATION_ENGINEERING_STATUS,
            "qualification_predicate_count": 0,
            "qualification_predicate_mask": 0,
            "qualification_predicate_total": 20,
            "qualification_candidate_receipt": None,
            "qualification_final_receipt": None,
            "q1_state": "NOT_RUN",
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_gate_total": 20,
            "q1_formal_output_roots": [None] * 8,
            "q1_receipt": None,
            "q2_state": "NOT_RUN",
            "m3_formal_roots": None,
            "outside_certificate_issued": False,
            "active_transition_allowed": False,
        },
    )
    version_bindings = {
        name.decode("ascii"): value.decode("ascii")
        for _index, name, value in qualification.VERSION_BINDING_ROWS
    }
    exact(("version_bindings",), version_bindings)
    exact(
        ("qualification_numeric_tag_registry",),
        [
            [tag, name.decode("ascii"), f"0x{tag:04X}"]
            for tag, name in qualification.Q05B_QUALIFICATION_TAG_REGISTRY
        ],
    )
    exact(
        ("formal_q1_tag_registry_remains_separate",),
        {
            "first": 0x3700,
            "last": 0x370C,
            "hex_range": "0x3700..0x370C",
            "qualification_tags_must_not_be_added": True,
        },
    )
    profile = qualification.qualification_wire_profile_object_v1()
    exact(
        ("schema_registry",),
        [
            [tag, schema.decode("ascii"), field_count]
            for tag, schema, field_count in profile[5]
        ],
    )
    exact(
        ("strict_type_and_nullability",),
        {
            "cbor_container": "ARRAY_ONLY_NO_MAP_TEXT_FLOAT_TAG_OR_INDEFINITE",
            "numeric_slots": "EXACT_INT_BOOL_ALIAS_FORBIDDEN",
            "boolean_slots": "EXACT_CBOR_BOOL_INT_ALIAS_FORBIDDEN",
            "identifier_names_and_schema_ids": "CBOR_BYTE_STRING",
            "roots": "EXACT_32_BYTE_STRING",
            "source_commit": "EXACT_RAW_20_BYTE_GIT_SHA1",
            "nullable_slots": [
                "q1_receipt",
                *[f"q1_output_slot_{index}" for index in range(1, 9)],
                "m3_formal_roots",
                "formal_fixed_point_tag",
            ],
            "all_other_machine_slots_nullable": False,
        },
    )
    trace_schema, trace_fields, trace_order = profile[15]
    exact(
        ("external_sort_trace_wire",),
        {
            "schema_id_ascii": trace_schema.decode("ascii"),
            "field_count": trace_fields,
            "field_order": [name.decode("ascii") for name in trace_order],
        },
    )
    counting_schema, counting_fields, counting_order, counting_rules, capability = (
        profile[16]
    )
    exact(
        ("counting_discard_wire",),
        {
            "schema_id_ascii": counting_schema.decode("ascii"),
            "field_count": counting_fields,
            "field_order": [name.decode("ascii") for name in counting_order],
            "equality_rules": [rule.decode("ascii") for rule in counting_rules],
            "predicate14_source_capability_frozen": capability,
            "predicate14_actual_qualification_passed": False,
        },
    )
    exact(
        ("registry_and_profile_roots", "qualification_tag_registry_root_hex"),
        qualification.QUALIFICATION_TAG_REGISTRY_ROOT.hex(),
    )
    exact(
        (
            "registry_and_profile_roots",
            "qualification_predicate_registry_root_hex",
        ),
        qualification.QUALIFICATION_PREDICATE_REGISTRY_ROOT.hex(),
    )
    exact(
        ("registry_and_profile_roots", "qualification_wire_profile_root_hex"),
        qualification.qualification_wire_profile_root_v1().hex(),
    )
    exact(
        ("registry_and_profile_roots",),
        {
            "qualification_tag_registry_root_domain": (
                "HEGEL/Q05B/QUALIFICATION/TAG_REGISTRY/V1"
            ),
            "qualification_tag_registry_root_hex": (
                qualification.QUALIFICATION_TAG_REGISTRY_ROOT.hex()
            ),
            "qualification_predicate_registry_root_domain": (
                "HEGEL/Q05B/QUALIFICATION/PREDICATE_REGISTRY/V1"
            ),
            "qualification_predicate_registry_root_hex": (
                qualification.QUALIFICATION_PREDICATE_REGISTRY_ROOT.hex()
            ),
            "qualification_wire_profile_root_domain": (
                "HEGEL/Q05B/QUALIFICATION/WIRE_PROFILE/V1"
            ),
            "qualification_wire_profile_root_hex": (
                qualification.qualification_wire_profile_root_v1().hex()
            ),
        },
    )
    semantic_manifest = qualification.q1_semantic_binding_manifest_v1(full_leaf)
    exact(
        ("semantic_source_bindings",),
        {
            "child_dsl_spec_root": semantic_manifest.child_dsl_root.hex(),
            "operator_semantics_root": semantic_manifest.operator_semantics_root.hex(),
            "identifier_registry_root": semantic_manifest.identifier_registry_root.hex(),
            "canonical_ast_schema_root": semantic_manifest.canonical_ast_root.hex(),
            "canonical_cbor_profile_root": semantic_manifest.canonical_cbor_root.hex(),
            "q0_semantic_binding_root": qualification.SEMANTIC_SOURCE_ROOTS[5].hex(),
            "q0_saturation_receipt_root": semantic_manifest.q0_receipt_root.hex(),
            "q1_preregistration_document_sha256": (
                semantic_manifest.preregistration_document_sha256.hex()
            ),
            "post_shrink6_normative_document_sha256": (
                semantic_manifest.post_shrink6_document_sha256.hex()
            ),
            "mdl_profile_id": semantic_manifest.mdl_profile_id.decode("ascii"),
        },
    )
    exact(
        ("full_v16_leaf_manifest",),
        {
            "leaf_count": len(full_leaf.rows),
            "row_tag": 0x3A00,
            "row_order": [
                "output_sort_id",
                "root_operator_id",
                "canonical_ast_cbor",
            ],
            "root_derivation": (
                "RFC6962_ROOT(ordered_810_Q05BFullLeafManifestRowV1_canonical_objects)"
            ),
            "root_hex": full_leaf.manifest_root.hex(),
            "q0_roots_or_receipt_in_leaf_root": False,
            "leaf_root_is_input_to_formal_0x3700": True,
            "sidecar_canonical_cbor_bytes": len(full_leaf.canonical_bytes),
        },
    )
    exact(("full_v16_leaf_manifest", "leaf_count"), len(full_leaf.rows))
    exact(("full_v16_leaf_manifest", "row_tag"), 0x3A00)
    exact(("full_v16_leaf_manifest", "root_hex"), full_leaf.manifest_root.hex())
    exact(
        ("full_v16_leaf_manifest", "sidecar_canonical_cbor_bytes"),
        len(full_leaf.canonical_bytes),
    )
    semantic_root, projection_root = qualification.q1_semantic_and_projection_roots_v1(
        full_leaf
    )
    exact(
        ("q1_formal_input_roots", "q1_semantic_binding_root_hex"),
        semantic_root.hex(),
    )
    exact(
        ("q1_formal_input_roots", "q1_projection_profile_root_hex"),
        projection_root.hex(),
    )
    exact(("q1_formal_input_roots", "q1_execution_started"), False)
    exact(
        ("q1_formal_input_roots",),
        {
            "q1_semantic_binding_tag": 0x3700,
            "q1_semantic_binding_root_hex": semantic_root.hex(),
            "q1_projection_profile_root_hex": projection_root.hex(),
            "these_are_q1_run_inputs_not_output_slots": True,
            "q1_execution_started": False,
        },
    )
    exact(
        ("bounded_node3_contract", "qualification_scope_id"),
        "BOUNDED_NODE3_SOURCE_AND_WIRE_QUALIFICATION",
    )
    exact(("bounded_node3_contract", "maximum_ast_depth"), 3)
    exact(("bounded_node3_contract", "maximum_ast_node_count"), 3)
    exact(("bounded_node3_contract", "structural_boundary_depth"), 4)
    exact(
        ("bounded_node3_contract", "terminal_status"),
        capacity.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED,
    )
    for name in (
        "work_queue_empty",
        "depth3_zero_delta",
        "structural_boundary_zero_delta",
        "all_846_coverage_rows_replayed",
        "eligible_equals_processed",
    ):
        exact(("bounded_node3_contract", name), True)
    exact(("bounded_node3_contract", "formal_fixed_point_claimed"), False)
    exact(("bounded_node3_contract", "formal_fixed_point_tag"), None)
    exact(("bounded_node3_contract", "formal_0x3707_used"), False)
    exact(
        ("bounded_node3_contract", "primary_count_rows"),
        [list(row) for row in qualification.FROZEN_NODE3_PRIMARY_COUNTS],
    )
    exact(("sidecar_layout", "file_mode_decimal"), qualification.OUTPUT_FILE_MODE)
    exact(("sidecar_layout", "file_mode_octal"), "0444")
    exact(
        ("sidecar_layout", "preimage_file_rows"),
        [
            [index, path.decode("ascii"), index + 1]
            for index, path in enumerate(qualification.ORDERED_PREIMAGE_RELATIVE_PATHS)
        ],
    )
    exact(
        ("sidecar_layout", "sidecar_manifest_relative_path"),
        qualification.SIDECAR_MANIFEST_RELATIVE_PATH.decode("ascii"),
    )
    exact(
        ("sidecar_layout", "neutral_manifest_relative_path"),
        qualification.NODE3_GOLDEN_MANIFEST_RELATIVE_PATH.decode("ascii"),
    )
    exact(
        ("sidecar_layout", "output_file_count"),
        len(qualification.ORDERED_OUTPUT_RELATIVE_PATHS),
    )
    exact(
        ("sidecar_layout",),
        {
            "file_mode_octal": "0444",
            "file_mode_decimal": qualification.OUTPUT_FILE_MODE,
            "preimage_file_rows": [
                [index, path.decode("ascii"), index + 1]
                for index, path in enumerate(
                    qualification.ORDERED_PREIMAGE_RELATIVE_PATHS
                )
            ],
            "sidecar_manifest_relative_path": (
                qualification.SIDECAR_MANIFEST_RELATIVE_PATH.decode("ascii")
            ),
            "neutral_manifest_relative_path": (
                qualification.NODE3_GOLDEN_MANIFEST_RELATIVE_PATH.decode("ascii")
            ),
            "output_file_count": len(qualification.ORDERED_OUTPUT_RELATIVE_PATHS),
            "sidecar_manifest_excludes_itself": True,
            "neutral_manifest_excluded_from_sidecar_manifest_to_avoid_cycle": True,
        },
    )
    exact(
        ("actor_stdout_envelope", "schema_version"),
        qualification.ACTOR_ENVELOPE_SCHEMA_VERSION,
    )
    exact(
        ("actor_stdout_envelope", "action_id"),
        qualification.ACTOR_ACTION_ID,
    )
    exact(
        ("actor_stdout_envelope", "status"),
        qualification.ACTOR_CANDIDATE_STATUS,
    )
    exact(
        ("actor_stdout_envelope", "actor_implementation_registry"),
        [list(row) for row in qualification.ACTOR_IMPLEMENTATION_ID_REGISTRY],
    )
    exact(
        ("actor_stdout_envelope",),
        {
            "schema_version": qualification.ACTOR_ENVELOPE_SCHEMA_VERSION,
            "action_id": qualification.ACTOR_ACTION_ID,
            "status": qualification.ACTOR_CANDIDATE_STATUS,
            "stdout_wire": "ONE_CANONICAL_JSON_LINE_PLUS_LF",
            "actor_implementation_registry": [
                list(row) for row in qualification.ACTOR_IMPLEMENTATION_ID_REGISTRY
            ],
            "actor_source_runtime_identities_are_outside_neutral_cbor": True,
            "neutral_manifest_must_be_python_rust_host_byte_equal": True,
        },
    )
    exact(
        ("chunk_boundary",),
        {
            "maximum_chunk_framed_bytes": qualification.MAX_CHUNK_FRAMED_BYTES,
            "accepted_raw_cbor_bstr_payload_bytes": (
                qualification.MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES
            ),
            "accepted_cbor_bytes": qualification.cbor_bstr_encoded_length_v1(
                qualification.MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES
            ),
            "accepted_u32_frame_bytes": qualification.framed_bstr_record_length_v1(
                qualification.MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES
            ),
            "first_rejected_raw_payload_bytes": (
                qualification.MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES + 1
            ),
            "first_rejected_u32_frame_bytes": qualification.framed_bstr_record_length_v1(
                qualification.MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES + 1
            ),
            "action_for_oversize_single_record": "REJECT_OR_CLOSE_BEFORE_RECORD_NO_SPLIT",
        },
    )
    exact(
        ("qualification_predicate_registry",),
        [
            [predicate_id, name.decode("ascii")]
            for predicate_id, name in qualification.QUALIFICATION_PREDICATE_REGISTRY
        ],
    )
    exact(
        ("failure_code_registry",),
        [code.decode("ascii") for code in qualification.FAILURE_CODE_REGISTRY],
    )
    exact(
        ("qualification_receipt_protocol",),
        {
            "candidate_tag": 0x3A05,
            "candidate_predicate_count": 19,
            "candidate_predicate_mask_hex": "0x7FFFF",
            "candidate_predicate_row": [
                "predicate_id",
                "predicate_name",
                "true",
                "evidence_root",
            ],
            "candidate_pre_receipt_evidence_domain": (
                "HEGEL/Q05B/QUALIFICATION/PRE_RECEIPT_EVIDENCE/V1"
            ),
            "final_tag": 0x3A06,
            "final_predicate_count": 20,
            "final_predicate_mask_hex": "0xFFFFF",
            "predicate20_evidence_domain": (
                "HEGEL/Q05B/QUALIFICATION/PREDICATE20_EVIDENCE/V1"
            ),
            "candidate_and_final_are_non_q1_receipts": True,
            "qualification_20_of_20_does_not_increment_q1_gate_count": True,
        },
    )
    exact(
        ("actual_preconditions",),
        qualification.COMMIT_A_ACTUAL_PRECONDITIONS_V1,
    )
    exact(
        ("two_commit_protocol",),
        {
            "commit_1": "SOURCE_FREEZE_CONTRACT_TESTS_DOC_CONFIG_NO_QUALIFICATION_CLAIM",
            "commit_2": (
                "ACTUAL_OFFLINE_PYTHON_RUST_HOST_EVIDENCE_AND_NON_Q1_QUALIFICATION_RECEIPT"
            ),
            "commit_2_must_bind_commit_1_git_sha1_raw20": True,
            "commit_2_may_be_created_only_after_20_of_20_qualification_predicates": True,
            "q1_remains_not_run_after_commit_2": True,
            "full_node6_allowed_by_this_protocol": False,
        },
    )


def _validate_primary_q05b_config_generated(
    config: dict[str, object],
    partition_snapshots: tuple[object, object],
    partition_evidence: tuple[object, object],
) -> None:
    def exact(path: tuple[str, ...], expected: object) -> None:
        _exact_config_value(config, path, expected, label="primary_q05b")

    bounded_states = tuple(
        qualification.bounded_node3_state_v1(snapshot, evidence)
        for snapshot, evidence in zip(
            partition_snapshots, partition_evidence, strict=True
        )
    )
    expected_resource_rows = [
        [
            snapshot.input_signature_id,
            snapshot.maximum_bank_points_per_class,
            snapshot.maximum_frontier_points_per_class,
            snapshot.peak_work_queue_points,
            snapshot.peak_saturation_round_count,
        ]
        for snapshot in partition_snapshots
    ]
    expected_coverage_roots = [
        [state.input_signature_id, state.coverage_record_root.hex()]
        for state in bounded_states
    ]
    expected_evidence_roots = [
        [evidence.input_signature_id, evidence.evidence_root.hex()]
        for evidence in partition_evidence
    ]
    expected_state_roots = [
        [state.input_signature_id, state.state_root.hex()]
        for state in bounded_states
    ]
    exact(
        ("bounded_node3_contract",),
        {
            "qualification_scope_id": "BOUNDED_NODE3_SOURCE_AND_WIRE_QUALIFICATION",
            "maximum_ast_depth": 3,
            "maximum_ast_node_count": 3,
            "structural_boundary_depth": 4,
            "terminal_status": capacity.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED,
            "work_queue_empty": True,
            "depth3_zero_delta": True,
            "structural_boundary_zero_delta": True,
            "all_846_coverage_rows_replayed": True,
            "eligible_equals_processed": True,
            "formal_fixed_point_claimed": False,
            "formal_fixed_point_tag": None,
            "formal_0x3707_used": False,
            "primary_count_rows": [
                list(row) for row in qualification.FROZEN_NODE3_PRIMARY_COUNTS
            ],
            "resource_rows_max_bank_max_frontier_peak_work_peak_round": (
                expected_resource_rows
            ),
            "coverage_record_roots": expected_coverage_roots,
            "partition_evidence_roots": expected_evidence_roots,
            "bounded_state_roots": expected_state_roots,
        },
    )
    exact(
        (
            "bounded_node3_contract",
            "resource_rows_max_bank_max_frontier_peak_work_peak_round",
        ),
        expected_resource_rows,
    )
    exact(
        ("bounded_node3_contract", "coverage_record_roots"),
        expected_coverage_roots,
    )
    exact(
        ("bounded_node3_contract", "partition_evidence_roots"),
        expected_evidence_roots,
    )
    exact(
        ("bounded_node3_contract", "bounded_state_roots"),
        expected_state_roots,
    )


def endpoint_object(
    action: str,
) -> tuple[dict[str, object], tuple[tuple[str, bytes], ...]]:
    primary_config, _primary_config_bytes = _load_config(
        PRIMARY_CONFIG_PATH,
        "primary_q05b",
    )
    referenced_q05a_config, _referenced_q05a_config_bytes = _load_config(
        REFERENCED_Q05A_CONFIG_PATH,
        "referenced_q05a",
    )
    if action != ACTION_ID:
        _fail("Q1_PROJECTION_ACTION_NOT_ADMITTED", "action identity differs")
    _recursive_allowlist_replay()
    source_identity = _source_file_set_digest()
    runtime_identity = _runtime_identity_digest()
    full_leaf = qualification.full_v16_leaf_manifest_v1()
    _validate_primary_q05b_config_static(primary_config, full_leaf)
    _validate_referenced_q05a_config(referenced_q05a_config)
    limits = capacity.PreflightLimitsV1(maximum_ast_node_count=3)
    partition_snapshots = tuple(
        snapshots.build_q1_partition_snapshot_v1(signature_id, limits=limits)
        for signature_id in (1, 2)
    )
    # Both independent consumers below call the public full-replay validator;
    # avoid a third identical capacity-engine replay in the actor wrapper.
    coverage_archives = tuple(
        coverage.build_q1_semantic_coverage_v1(item)
        for item in partition_snapshots
    )
    record_sets = tuple(
        projection.records_from_partition_snapshot_v1(item)
        for item in partition_snapshots
    )
    partition_evidence = tuple(
        qualification.node3_partition_evidence_v1(item, record_set, archive)
        for item, record_set, archive in zip(
            partition_snapshots,
            record_sets,
            coverage_archives,
            strict=True,
        )
    )
    _validate_primary_q05b_config_generated(
        primary_config,
        partition_snapshots,
        partition_evidence,
    )
    odd_evidence, sink_evidence = partition_evidence
    sidecar_manifest = qualification.sidecar_manifest_v1(
        full_leaf,
        odd_evidence,
        sink_evidence,
    )
    neutral_manifest = qualification.node3_golden_manifest_v1(
        full_leaf,
        partition_snapshots[0],
        odd_evidence,
        partition_snapshots[1],
        sink_evidence,
        sidecar_manifest,
    )
    preimages = (
        full_leaf.canonical_bytes,
        odd_evidence.canonical_bytes,
        sink_evidence.canonical_bytes,
    )
    qualification.decode_full_v16_leaf_manifest_v1(preimages[0])
    qualification.decode_node3_partition_evidence_v1(preimages[1])
    qualification.decode_node3_partition_evidence_v1(preimages[2])
    qualification.replay_sidecar_manifest_v1(
        sidecar_manifest.canonical_bytes,
        preimages,
    )
    qualification.decode_node3_golden_manifest_v1(neutral_manifest.canonical_bytes)
    files = tuple(
        (relative.decode("ascii"), payload)
        for relative, payload in zip(
            qualification.ORDERED_OUTPUT_RELATIVE_PATHS,
            preimages
            + (
                sidecar_manifest.canonical_bytes,
                neutral_manifest.canonical_bytes,
            ),
            strict=True,
        )
    )
    if tuple(path for path, _payload in files) != SIDECAR_PATHS:
        _fail("FAIL_Q1_PROJECTION_SIDECAR_SET", "contract/actor sidecar paths differ")
    _loaded_project_modules()
    if (
        _source_file_set_digest() != source_identity
        or _runtime_identity_digest() != runtime_identity
    ):
        _fail(
            "FAIL_Q1_PROJECTION_IDENTITY_CHANGED",
            "source or runtime identity changed during actor execution",
        )
    payload = {
        "action_id": ACTION_ID,
        "actor_id": "PYTHON_ENDPOINT",
        "file_count": len(files),
        "implementation_id": IMPLEMENTATION_ID,
        "neutral_manifest_length": len(neutral_manifest.canonical_bytes),
        "neutral_manifest_raw_sha256": sha256(
            neutral_manifest.canonical_bytes
        ).hexdigest(),
        "neutral_manifest_relative_path": (
            qualification.NODE3_GOLDEN_MANIFEST_RELATIVE_PATH.decode("ascii")
        ),
        "neutral_manifest_root": neutral_manifest.manifest_root.hex(),
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "runtime_identity_sha256": runtime_identity,
        "schema_version": "hegel-q05b-actor-envelope/1",
        "sidecar_manifest_length": len(sidecar_manifest.canonical_bytes),
        "sidecar_manifest_raw_sha256": sha256(
            sidecar_manifest.canonical_bytes
        ).hexdigest(),
        "sidecar_manifest_relative_path": (
            qualification.SIDECAR_MANIFEST_RELATIVE_PATH.decode("ascii")
        ),
        "sidecar_manifest_root": sidecar_manifest.manifest_root.hex(),
        "source_identity_sha256": source_identity,
        "status": "BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED",
    }
    return payload, files


def _error_object(error: EndpointError) -> dict[str, object]:
    return {
        "active_transition_allowed": False,
        "authority_claimed": False,
        "detail": error.detail,
        "error_code": error.code,
        "formal_roots_generated": False,
        "full_node6_executed": False,
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q1_state": "NOT_RUN",
        "q2_state": "NOT_RUN",
        "schema_version": ERROR_SCHEMA_VERSION,
        "sidecar_set_complete": False,
    }


def main() -> int:
    try:
        action, output_dir, output_dir_identity = _parse_invocation(sys.argv[1:])
        payload, files = endpoint_object(action)
        output = _canonical_json_bytes(payload)
        if len(output) > MAXIMUM_STDOUT_BYTES:
            _fail(
                "INCONCLUSIVE_Q1_PROJECTION_OUTPUT_LIMIT",
                f"output bytes {len(output)} exceed {MAXIMUM_STDOUT_BYTES}",
            )
        qualification.validate_actor_stdout_envelope_v1(output)
        _publish_sidecars(output_dir, output_dir_identity, files)
    except EndpointError as error:
        output = _canonical_json_bytes(_error_object(error))
        if len(output) > MAXIMUM_STDOUT_BYTES:
            output = b'{"error_code":"FAIL_Q1_PROJECTION_ERROR_OUTPUT_LIMIT"}\n'
        sys.stdout.buffer.write(output)
        return 1
    except Exception as error:  # fail closed without leaking a traceback to stderr
        output = _canonical_json_bytes(
            _error_object(
                EndpointError(
                    "FAIL_Q1_PROJECTION_UNHANDLED",
                    f"{type(error).__name__}: {error}",
                )
            )
        )
        sys.stdout.buffer.write(output)
        return 1
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
