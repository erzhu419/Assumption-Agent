"""Committed, public and deterministic M2.5 formal-static basis.

This module closes the static-preimage gap without crossing the custody
boundary.  It constructs every Gate-19 root from exact formal objects or
ordered formal record bytes, binds all ordinary digests to an inspectable
preimage, and can replay the resulting public plan through the existing Rust
``formal_bridge_m25`` binary inside the locally pinned, network-disabled OCI
image.

Nothing here accepts or creates a split seed, private key, signature, opaque
run ID, ledger ID, marker, or M3 state transition.  A successful dual replay
qualifies only the public static roots; it does not by itself satisfy Gates
15--18 or 20--24 and never starts M3.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from ._vendor import tomli as tomllib
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .hashing import canonical_json
from .phase3_local_runtime_v1 import (
    DEFAULT_DOCKER_EXECUTABLE,
    LOCAL_DOCKER_HOST,
    LocalDockerControlPlaneV1,
)
from .phase3_dsl_v1 import (
    AGGREGATE_CATALOG,
    ALL_EXPRESSIONS,
    BINARY_OPERATORS,
    BOOLEAN_COMPOSITION,
    CONTEXT_IDS,
    ENTITY_SLOTS,
    LEAF_EXPRESSIONS,
    OBSERVED_OMITTED_SINK_CONTROL,
    ODD_REDUCTION_TARGET,
    QUANTITY_IDS,
    ROLE_IDS,
    SCALE_IDS,
    SCOPE_CATALOG,
    TASK_IDS,
    TERNARY_OPERATORS,
    TRANSFORM_CATALOG,
    UNARY_OPERATORS,
)
from .phase3_m25_rows_v1 import generate_odd_role_rows_v1, generate_sink_role_rows_v1
from .phase3_m25_wire_v1 import (
    FORMAL_SCHEMA_REGISTRY,
    LEGAL_M3_TRANSITIONS,
    NUMERIC_ENUM_REGISTRIES,
    OBJECT_TAGS,
    build_formal_object,
    candidate_content_root,
    candidate_record_tree_root,
    git_sha1_commit_id,
    id_digest_v1,
)
from .phase3_shrink1_registry_v1 import (
    AGGREGATE_REGISTRY,
    AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
    OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID,
    SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID,
    aggregate_registry_object,
    operator_admission_semantics_object,
    shrunk_dsl_surface_object,
)
from .strict_ast_shrink1_v1 import canonicalize_shrink1_source_ast
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode, content_hash, rfc6962_root


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
PROFILE_PATH: Final = PROJECT_ROOT / "config/phase3_container_actor_profile_v1.json"
SECCOMP_PATH: Final = PROJECT_ROOT / "config/phase3_internal_actor_seccomp_v1.json"
DEFAULT_RUST_BINARY: Final = (
    PROJECT_ROOT / "rust/formal_bridge_m25/target/debug/hegel-formal-bridge-m25"
)
FORMAL_GIT_EXECUTABLE: Final = Path("/usr/bin/git")
FORMAL_GIT_ENVIRONMENT_V1: Final = MappingProxyType({
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "false",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
})

SCHEMA: Final = "hegel-phase3-m25-formal-static-basis/1"
RECEIPT_SCHEMA: Final = "hegel-phase3-m25-formal-static-replay-receipt/1"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
PARENT_DSL_ID: Final = "hegel-old-dsl-v1.0.0"
CHILD_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
PARENT_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.0.2"
SPLIT_ALGORITHM_ID: Final = "hegel-split-algorithm-hkdf-hmac-sha256-rank-v1"
ENGINEERING_FREEZE_PATH: Final = (
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Formal_Static_Basis_Engineering_Freeze_v1.md"
)
GENERATOR_PATH: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m25_formal_static_basis_v1.py"
)
PARENT_NORMATIVE_DECISION_PATH: Final = (
    "Hegel Machine/docs/Hegel_Machine_Phase3_Shrink_Step1_Freeze_Decisions.md"
)
PARENT_EXECUTION_EVIDENCE_PATH: Final = (
    "Hegel Machine/artifacts/phase3_dual_strict_capacity_replay_v1.json"
)
SHRINK1_SUBSET_REPLAY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_shrink1_dual_capacity_replay_v1.json"
)
CONTAINER_PROFILE_PATH: Final = (
    "Hegel Machine/config/phase3_container_actor_profile_v1.json"
)
CONTAINER_SECCOMP_PATH: Final = (
    "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json"
)

GATE19_ROOT_NAMES: Final = (
    "child_dsl_spec_root",
    "child_freeze_root",
    "operator_semantics_root",
    "identifier_registry_root",
    "canonical_ast_schema_root",
    "canonical_cbor_profile_root",
)

FAIL_BASIS_COMMIT: Final = "FAIL_M25_STATIC_BASIS_COMMIT"
FAIL_BASIS_PREIMAGE: Final = "FAIL_M25_STATIC_BASIS_PREIMAGE"
FAIL_BASIS_MAPPING: Final = "FAIL_M25_STATIC_BASIS_MAPPING"
FAIL_RUST_REPLAY_POLICY: Final = "FAIL_M25_STATIC_RUST_REPLAY_POLICY"
FAIL_RUST_REPLAY: Final = "FAIL_M25_STATIC_RUST_REPLAY"
FAIL_DUAL_RECEIPT: Final = "FAIL_M25_STATIC_DUAL_RECEIPT"


class FormalStaticBasisError(RuntimeError):
    """Stable fail-closed error for public static-basis construction."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise FormalStaticBasisError(code, detail)


def _require_commit(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        _fail(FAIL_BASIS_COMMIT, "basis_commit must be lowercase SHA-1 hex")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def formal_git_environment_v1() -> dict[str, str]:
    """Return the complete non-inheriting environment for formal Git reads."""

    return dict(FORMAL_GIT_ENVIRONMENT_V1)


def _git(
    repository_root: Path, args: Sequence[str], *, binary: bool = True
) -> bytes | str:
    if (
        not args
        or any(type(value) is not str or not value or "\0" in value for value in args)
        or not FORMAL_GIT_EXECUTABLE.is_file()
        or FORMAL_GIT_EXECUTABLE.resolve(strict=True) != FORMAL_GIT_EXECUTABLE
    ):
        _fail(FAIL_BASIS_COMMIT, "formal Git executable or argument vector differs")
    try:
        completed = subprocess.run(
            [str(FORMAL_GIT_EXECUTABLE), *args],
            cwd=repository_root.resolve(strict=True),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
            env=formal_git_environment_v1(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_BASIS_COMMIT, f"formal Git read failed to execute: {exc}")
    if completed.returncode != 0:
        _fail(
            FAIL_BASIS_COMMIT,
            completed.stderr.decode("utf-8", "replace")[-1000:],
        )
    if binary:
        return completed.stdout
    return completed.stdout.decode("ascii", "strict").strip()


def _git_blob(repository_root: Path, commit: str, relative_path: str) -> bytes:
    value = _git(repository_root, ["show", f"{commit}:{relative_path}"], binary=True)
    assert isinstance(value, bytes)
    return value


def _git_commit_unix_seconds(repository_root: Path, commit: str) -> int:
    """Return the committed recording instant used by all Commit-A objects."""

    value = _git(
        repository_root,
        ["show", "-s", "--format=%ct", commit],
        binary=False,
    )
    assert isinstance(value, str)
    if re.fullmatch(r"[0-9]+", value) is None:
        _fail(FAIL_BASIS_COMMIT, "Git commit timestamp is not an unsigned integer")
    timestamp = int(value)
    if timestamp <= 0:
        _fail(FAIL_BASIS_COMMIT, "Git commit timestamp must be positive")
    return timestamp


def _git_mode(repository_root: Path, commit: str, relative_path: str) -> int:
    value = _git(
        repository_root,
        ["ls-tree", commit, "--", relative_path],
        binary=False,
    )
    assert isinstance(value, str)
    if not value:
        _fail(FAIL_BASIS_COMMIT, f"missing committed path {relative_path}")
    mode = value.split(None, 1)[0]
    try:
        return int(mode, 8)
    except ValueError:
        _fail(FAIL_BASIS_COMMIT, f"invalid Git mode for {relative_path}")


def _git_blob_sha1(payload: bytes) -> bytes:
    return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload).digest()


def _path_alias(relative_path: str) -> bytes:
    return id_digest_v1(
        "repo-path-sha256:" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
    )


def _descriptor_preimage(kind: str, numeric_id: int, values: Sequence[object]) -> bytes:
    """Exact ordinary-SHA preimage; this is not a new ContentHash domain."""

    def formalize(value: object) -> object:
        if isinstance(value, str):
            return value.encode("utf-8")
        if value is None or type(value) in {bool, int, bytes}:
            return value
        if isinstance(value, (tuple, list)):
            return tuple(formalize(item) for item in value)
        _fail(
            FAIL_BASIS_PREIMAGE,
            f"descriptor contains unsupported {type(value).__name__}",
        )

    return canonical_cbor_encode(
        (
            1,
            b"hegel-formal-static-descriptor/1",
            kind.encode("ascii"),
            numeric_id,
            tuple(formalize(item) for item in values),
        )
    )


def _descriptor_digest(kind: str, numeric_id: int, values: Sequence[object]) -> tuple[bytes, bytes]:
    preimage = _descriptor_preimage(kind, numeric_id, values)
    return hashlib.sha256(preimage).digest(), preimage


def _diagnostic_preimage(value: object) -> bytes:
    return canonical_json(value).encode("utf-8")


def _diagnostic_id_digest(identifier: str) -> bytes:
    suffix = identifier.rsplit("_", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{64}", suffix) is None:
        _fail(FAIL_BASIS_PREIMAGE, f"diagnostic ID has no SHA-256 suffix: {identifier}")
    return bytes.fromhex(suffix)


def _formal_transport(value: object) -> object:
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, (tuple, list)):
        return [_formal_transport(item) for item in value]
    raise TypeError(f"unsupported formal transport type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class RootReplayPlanEntryV1:
    root_name: str
    operation: str
    schema_name: str
    domain_or_null: str | None
    preimage_cbor: tuple[bytes, ...]
    expected_root: bytes

    def request(self) -> dict[str, object]:
        if self.operation == "content_hash":
            value = canonical_cbor_decode(self.preimage_cbor[0])
            return {
                "op": "content_hash",
                "domain": self.domain_or_null,
                "value": _formal_transport(value),
            }
        if self.operation == "rfc6962_root":
            return {
                "op": "rfc6962_root",
                "leaves_hex": [item.hex() for item in self.preimage_cbor],
            }
        raise AssertionError(f"unsupported replay operation {self.operation}")


@dataclass(frozen=True, slots=True)
class FormalStaticBasisV1:
    basis_commit: str
    repository_root: Path
    objects: Mapping[str, Mapping[str, object]]
    record_sets: Mapping[str, tuple[Mapping[str, object], ...]]
    roots: Mapping[str, bytes]
    ordinary_digest_preimages: Mapping[str, bytes]
    diagnostic_preimages: Mapping[str, bytes]
    gate19_plan: tuple[RootReplayPlanEntryV1, ...]
    m3_candidate_static_fields: Mapping[str, object]
    preseed_manifest_static_fields: Mapping[str, Mapping[str, object]]
    preseed_manifest_required_dynamic_fields: Mapping[str, tuple[str, ...]]
    implementation_inputs: Mapping[str, object]
    blocking_gaps: tuple[str, ...]

    def public_transport(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "basis_commit": self.basis_commit,
            "root_hex": {name: value.hex() for name, value in sorted(self.roots.items())},
            "record_counts": {
                name: len(rows) for name, rows in sorted(self.record_sets.items())
            },
            "ordinary_preimage_sha256": {
                name: hashlib.sha256(value).hexdigest()
                for name, value in sorted(self.ordinary_digest_preimages.items())
            },
            "diagnostic_preimage_sha256": {
                name: hashlib.sha256(value).hexdigest()
                for name, value in sorted(self.diagnostic_preimages.items())
            },
            "gate19_root_names": [entry.root_name for entry in self.gate19_plan],
            "preseed_manifest_static_field_names": {
                name: sorted(fields)
                for name, fields in sorted(self.preseed_manifest_static_fields.items())
            },
            "preseed_manifest_required_dynamic_fields": {
                name: list(fields)
                for name, fields in sorted(
                    self.preseed_manifest_required_dynamic_fields.items()
                )
            },
            "seed_key_signature_or_state_created": False,
            "gate_claim": "STATIC_DUAL_REPLAY_INPUT_ONLY_NOT_GATE_PASS",
            "blocking_gaps": list(self.blocking_gaps),
        }


_SORT_IDS: Final = MappingProxyType(
    {
        "Bool": 1,
        "Bit": 2,
        "Sign": 3,
        "BoundedInt": 4,
        "RationalValue": 5,
        "RationalParameter": 6,
        "Tolerance": 7,
        "ClosedInterval": 8,
        "EntitySlot": 9,
        "Index": 10,
        "QuantityId": 11,
        "ContextId": 12,
        "RoleId": 13,
        "ScaleId": 14,
        "TaskId": 15,
        "EntitySet": 16,
        "ScopeId": 17,
        "AggregateMapId": 18,
        "TransformId": 19,
    }
)


def _identifier_catalog(parent: bool) -> tuple[tuple[int, int, str, int, Sequence[object]], ...]:
    rows: list[tuple[int, int, str, int, Sequence[object]]] = []
    simple = (
        (1, ENTITY_SLOTS),
        (2, QUANTITY_IDS),
        (3, CONTEXT_IDS),
        (4, ROLE_IDS),
        (5, SCALE_IDS),
        (6, TASK_IDS),
    )
    for kind, names in simple:
        for numeric_id, name in enumerate(names):
            rows.append((kind, numeric_id, name, 1, (name,)))
    for numeric_id, spec in enumerate(SCOPE_CATALOG):
        rows.append(
            (
                7,
                numeric_id,
                spec.scope_id,
                1,
                (
                    spec.semantic_rule,
                    spec.include_auxiliary,
                    spec.context_clause_limit,
                    spec.context_registry,
                ),
            )
        )
    child_state = {entry.numeric_id: entry.state for entry in AGGREGATE_REGISTRY}
    for numeric_id, spec in enumerate(AGGREGATE_CATALOG):
        state = 1 if parent or child_state[numeric_id] == "ACTIVE" else 2
        rows.append(
            (
                8,
                numeric_id,
                spec.map_id,
                state,
                (
                    spec.input_sorts,
                    spec.output_sort,
                    spec.semantic_rule,
                    spec.undefined_conditions,
                ),
            )
        )
    for numeric_id, spec in enumerate(TRANSFORM_CATALOG):
        rows.append(
            (
                9,
                numeric_id,
                spec.transform_id,
                1,
                (spec.semantic_rule, spec.adapter_only, spec.old_dsl_composable),
            )
        )
    for numeric_id, spec in enumerate(ALL_EXPRESSIONS):
        rows.append(
            (
                10,
                numeric_id,
                spec.expression_id,
                1,
                (
                    spec.expression_class,
                    spec.input_sorts,
                    spec.output_sorts,
                    spec.accepted_arities,
                    spec.semantic_rule,
                    spec.canonical_child_sort_groups,
                ),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row[0], row[1])))


def build_identifier_registry_rows_v1(
    *, parent: bool = False
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, bytes]]:
    """Build the exact full identifier registry (55 rows)."""

    introduced = id_digest_v1(PARENT_DSL_ID)
    removed = id_digest_v1(CHILD_DSL_ID)
    rows: list[Mapping[str, object]] = []
    preimages: dict[str, bytes] = {}
    for kind, numeric_id, name, state, semantic_values in _identifier_catalog(parent):
        digest, preimage = _descriptor_digest(
            f"identifier-kind-{kind}", numeric_id, (name, *semantic_values)
        )
        key = f"identifier/{kind}/{numeric_id}/{name}"
        preimages[key] = preimage
        row = {
            "registry_kind_id": kind,
            "numeric_id": numeric_id,
            "entry_state_id": state,
            "canonical_name_digest": id_digest_v1(f"identifier:{kind}:{name}"),
            "semantics_digest_or_null": digest,
            "introduced_dsl_version_digest": introduced,
            "removed_dsl_version_digest_or_null": (
                removed if state == 2 and not parent else None
            ),
        }
        build_formal_object("IdentifierRegistryEntryV1", row)
        rows.append(MappingProxyType(row))
    if len(rows) != 55:
        _fail(FAIL_BASIS_MAPPING, f"identifier registry must contain 55 rows, got {len(rows)}")
    return tuple(rows), MappingProxyType(preimages)


def _operator_catalog(parent: bool) -> tuple[tuple[int, int, int, Sequence[int], int, int, str, Sequence[object]], ...]:
    rows: list[tuple[int, int, int, Sequence[int], int, int, str, Sequence[object]]] = []

    # ``aggregate`` is an AST dispatcher; its six typed cases are represented
    # by OperatorClassId.AGGREGATE_MAP below, avoiding a false single output sort.
    leaf_undefined = {"scalar_const": 1, "bit_at": 3, "set_size": 1, "context_flag": 1, "task_flag": 1}
    for operator_id, spec in enumerate(LEAF_EXPRESSIONS):
        if spec.expression_id == "aggregate":
            continue
        rows.append((1, operator_id, 1, tuple(_SORT_IDS[x] for x in spec.input_sorts), _SORT_IDS[spec.output_sorts[0]], leaf_undefined[spec.expression_id], spec.expression_id, (spec.semantic_rule, spec.accepted_arities, spec.canonical_child_sort_groups)))
    for operator_id, spec in enumerate(UNARY_OPERATORS):
        rows.append((2, operator_id, 1, tuple(_SORT_IDS[x] for x in spec.input_sorts), _SORT_IDS[spec.output_sorts[0]], 2, spec.expression_id, (spec.semantic_rule, spec.accepted_arities, spec.canonical_child_sort_groups)))
    for operator_id, spec in enumerate(BINARY_OPERATORS):
        undefined = 6 if spec.expression_id in {"add", "difference"} else 2
        rows.append((3, operator_id, 1, tuple(_SORT_IDS[x] for x in spec.input_sorts), _SORT_IDS[spec.output_sorts[0]], undefined, spec.expression_id, (spec.semantic_rule, spec.accepted_arities, spec.canonical_child_sort_groups)))
    for operator_id, spec in enumerate(TERNARY_OPERATORS):
        rows.append((4, operator_id, 1, tuple(_SORT_IDS[x] for x in spec.input_sorts), _SORT_IDS[spec.output_sorts[0]], 2, spec.expression_id, (spec.semantic_rule, spec.accepted_arities, spec.canonical_child_sort_groups)))
    for operator_id, spec in enumerate(BOOLEAN_COMPOSITION):
        rows.append((5, operator_id, 1, tuple(_SORT_IDS[x] for x in spec.input_sorts), _SORT_IDS[spec.output_sorts[0]], 2, spec.expression_id, (spec.semantic_rule, spec.accepted_arities, spec.canonical_child_sort_groups)))
    child_state = {entry.numeric_id: entry.state for entry in AGGREGATE_REGISTRY}
    for operator_id, spec in enumerate(AGGREGATE_CATALOG):
        admission = 1 if parent or child_state[operator_id] == "ACTIVE" else 2
        undefined = 4 if operator_id in {2, 3, 4} else 5
        rows.append((6, operator_id, admission, (16, 11), _SORT_IDS[spec.output_sort], undefined, spec.map_id, (spec.semantic_rule, spec.undefined_conditions)))
    for operator_id, spec in enumerate(TRANSFORM_CATALOG):
        undefined = 6 if spec.transform_id in {"scale_by_2_v1", "scale_by_half_v1"} else 2
        rows.append((7, operator_id, 3, (5,), 5, undefined, spec.transform_id, (spec.semantic_rule, spec.adapter_only, spec.old_dsl_composable)))
    return tuple(sorted(rows, key=lambda row: (row[0], row[1])))


def build_operator_semantics_rows_v1(
    *, executable_semantics_root: bytes, parent: bool = False
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, bytes]]:
    """Build the 28 exact typed operator/map/adapter rows."""

    if type(executable_semantics_root) is not bytes or len(executable_semantics_root) != 32:
        _fail(FAIL_BASIS_PREIMAGE, "executable semantics source root must be 32 bytes")
    rows: list[Mapping[str, object]] = []
    preimages: dict[str, bytes] = {}
    for operator_class, operator_id, admission, inputs, output, undefined, name, semantics in _operator_catalog(parent):
        digest, preimage = _descriptor_digest(
            f"operator-class-{operator_class}", operator_id, (name, admission, tuple(inputs), output, undefined, *semantics)
        )
        preimages[f"operator/{operator_class}/{operator_id}/{name}"] = preimage
        # The executable source object root binds the complete committed
        # evaluator/canonicalizer source.  The per-row descriptor digest above
        # binds the exact row-to-source semantic mapping.
        row = {
            "operator_class_id": operator_class,
            "operator_id": operator_id,
            "admission_state_id": admission,
            "input_sort_ids": tuple(inputs),
            "output_sort_id": output,
            "undefined_semantics_id": undefined,
            "normalization_rule_root_or_null": None,
            "executable_semantics_root": executable_semantics_root,
        }
        build_formal_object("OperatorSemanticsEntryV1", row)
        rows.append(MappingProxyType(row))
        preimages[f"operator-row-descriptor/{operator_class}/{operator_id}"] = preimage
        # Keep the descriptor reachable under its digest even though the row's
        # executable root is the exact committed source-object root.
        if hashlib.sha256(preimage).digest() != digest:
            raise AssertionError("descriptor SHA-256 drift")
    if len(rows) != 28:
        _fail(FAIL_BASIS_MAPPING, f"operator semantics must contain 28 rows, got {len(rows)}")
    return tuple(rows), MappingProxyType(preimages)


def _document_fields(commit_wire: tuple[int, bytes], path: str, payload: bytes) -> dict[str, object]:
    return {
        "repository_relative_path_id_digest": _path_alias(path),
        "raw_git_blob_bytes": payload,
        "repository_commit_id": commit_wire,
    }


def _profile_fields(
    *, identity_field: str, identity: str, governing_root: bytes, source_blob: bytes, source_path: str, commit_wire: tuple[int, bytes]
) -> dict[str, object]:
    return {
        identity_field: id_digest_v1(identity),
        "governing_normative_document_root": governing_root,
        "section_selector_id_digest": id_digest_v1(
            "section:entire-document:" + Path(source_path).name
        ),
        "section_blob_sha256": hashlib.sha256(source_blob).digest(),
        "section_byte_length": len(source_blob),
        "repository_commit_id": commit_wire,
    }


def _source_file_rows(
    repository_root: Path, commit: str, paths: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for path in sorted(set(paths), key=lambda item: item.encode("utf-8")):
        payload = _git_blob(repository_root, commit, path)
        row = {
            "path_alias_id_digest": _path_alias(path),
            "raw_path_bytes": path.encode("utf-8"),
            "git_blob_algorithm_id": 1,
            "git_blob_digest": _git_blob_sha1(payload),
            "file_mode": _git_mode(repository_root, commit, path),
            "byte_length": len(payload),
        }
        build_formal_object("SourceFileRecordV1", row)
        rows.append(MappingProxyType(row))
    return tuple(rows)


def _dependency_lock_rows(cargo_lock: bytes) -> tuple[Mapping[str, object], ...]:
    parsed = tomllib.loads(cargo_lock.decode("utf-8"))
    raw_packages = parsed.get("package", [])
    if not isinstance(raw_packages, list):
        _fail(FAIL_BASIS_PREIMAGE, "Cargo.lock package set is invalid")
    rows: list[Mapping[str, object]] = []
    for package in raw_packages:
        if not isinstance(package, dict):
            _fail(FAIL_BASIS_PREIMAGE, "Cargo.lock package row is invalid")
        name = str(package["name"])
        version = str(package["version"])
        source = str(package.get("source", "local-workspace"))
        exact_entry = _canonical_json_bytes(package)
        row = {
            "ecosystem_id": 2,
            "package_name_id_digest": id_digest_v1("cargo-package:" + name),
            "version_id_digest": id_digest_v1("cargo-version:" + version),
            "source_id_digest": id_digest_v1(
                "cargo-source-sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
            ),
            "lock_entry_digest": hashlib.sha256(exact_entry).digest(),
        }
        build_formal_object("DependencyLockRecordV1", row)
        rows.append(MappingProxyType(row))
    rows.sort(key=lambda row: (row["ecosystem_id"], row["package_name_id_digest"], row["version_id_digest"]))
    return tuple(rows)


def _enum_registry_rows(enum_name: str, kind_prefix: str) -> tuple[Mapping[str, object], ...]:
    registry = NUMERIC_ENUM_REGISTRIES[enum_name]
    rows: list[Mapping[str, object]] = []
    for numeric_id, name in sorted(registry.entries.items()):
        digest, _ = _descriptor_digest(kind_prefix, numeric_id, (name,))
        row = {
            "registry_kind_id": 11,
            "numeric_id": numeric_id,
            "entry_state_id": 1,
            "canonical_name_digest": id_digest_v1(f"enum:{enum_name}:{name}"),
            "semantics_digest_or_null": digest,
            "introduced_dsl_version_digest": id_digest_v1(CHILD_DSL_ID),
            "removed_dsl_version_digest_or_null": None,
        }
        build_formal_object("IdentifierRegistryEntryV1", row)
        rows.append(MappingProxyType(row))
    return tuple(rows)


def _state_contract_fields(roots: Mapping[str, bytes]) -> dict[str, object]:
    reason_by_transition = {
        (0, 0, 1, 1): (1,),
        (1, 1, 1, 2): (2,),
        (1, 1, 3, 0): (3,),
        (1, 1, 4, 0): (4, 5),
        (1, 1, 5, 0): (6,),
        (1, 1, 6, 0): (7,),
        (1, 2, 2, 0): (8,),
        (1, 2, 5, 0): (6,),
        (1, 2, 6, 0): (7,),
    }
    legal_rows: list[tuple[object, ...]] = []
    for transition in sorted(tuple(int(item) for item in row) for row in LEGAL_M3_TRANSITIONS):
        fields = {
            "from_state_id": transition[0],
            "from_phase_id": transition[1],
            "to_state_id": transition[2],
            "to_phase_id": transition[3],
            "allowed_reason_ids": reason_by_transition[transition],
        }
        legal_rows.append(build_formal_object("LegalTransitionRowV1", fields))
    return {
        "m3_state_registry_root": roots["m3_state_registry_root"],
        "m3_phase_registry_root": roots["m3_phase_registry_root"],
        "m3_transition_reason_registry_root": roots["m3_transition_reason_registry_root"],
        "legal_transition_table": tuple(legal_rows),
        "terminal_state_ids": (2, 3, 4, 5, 6),
        "reopen_allowed": False,
    }


def _root_plan_content(name: str, root_name: str, fields: Mapping[str, object]) -> RootReplayPlanEntryV1:
    schema = FORMAL_SCHEMA_REGISTRY[name]
    if schema.hash_domain is None:
        raise AssertionError(f"{name} has no content domain")
    value = build_formal_object(name, fields)
    return RootReplayPlanEntryV1(
        root_name=root_name,
        operation="content_hash",
        schema_name=name,
        domain_or_null=schema.hash_domain,
        preimage_cbor=(canonical_cbor_encode(value),),
        expected_root=candidate_content_root(name, fields),
    )


def _root_plan_records(name: str, root_name: str, rows: Sequence[Mapping[str, object]]) -> RootReplayPlanEntryV1:
    return RootReplayPlanEntryV1(
        root_name=root_name,
        operation="rfc6962_root",
        schema_name=name,
        domain_or_null=None,
        preimage_cbor=tuple(canonical_cbor_encode(build_formal_object(name, row)) for row in rows),
        expected_root=candidate_record_tree_root(name, rows),
    )


def build_formal_static_basis_v1(
    basis_commit: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    rust_binary_path: Path = DEFAULT_RUST_BINARY,
    python_binary_path: Path = Path(sys.executable),
    require_generator_committed: bool = True,
) -> FormalStaticBasisV1:
    """Construct the complete public root DAG from one exact Git commit."""

    commit = _require_commit(basis_commit)
    repository_root = repository_root.resolve()
    resolved = _git(repository_root, ["rev-parse", f"{commit}^{{commit}}"], binary=False)
    if resolved != commit:
        _fail(FAIL_BASIS_COMMIT, "basis_commit does not resolve exactly")
    required_new = (GENERATOR_PATH, ENGINEERING_FREEZE_PATH)
    if require_generator_committed:
        for path in required_new:
            _git_blob(repository_root, commit, path)

    commit_wire = git_sha1_commit_id(bytes.fromhex(commit))
    recorded_at_unix_seconds = _git_commit_unix_seconds(repository_root, commit)
    document_paths = {
        1: "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md",
        2: "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md",
        3: "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md",
    }
    document_blobs = {role: _git_blob(repository_root, commit, path) for role, path in document_paths.items()}
    objects: dict[str, Mapping[str, object]] = {}
    record_sets: dict[str, tuple[Mapping[str, object], ...]] = {}
    roots: dict[str, bytes] = {}
    ordinary: dict[str, bytes] = {}
    diagnostic: dict[str, bytes] = {}

    document_entries: list[tuple[int, bytes]] = []
    for role in (1, 2, 3):
        fields = _document_fields(commit_wire, document_paths[role], document_blobs[role])
        name = f"normative_document_{role}"
        objects[name] = MappingProxyType(fields)
        roots[name + "_root"] = candidate_content_root("NormativeDocumentBlobV1", fields)
        document_entries.append((role, roots[name + "_root"]))
    bundle_fields = {
        "bundle_id_digest": id_digest_v1("hegel-m25-normative-document-bundle-v1.1.2"),
        "document_entries": tuple(document_entries),
        "repository_commit_id": commit_wire,
    }
    objects["normative_document_bundle"] = MappingProxyType(bundle_fields)
    roots["normative_document_bundle_root"] = candidate_content_root("NormativeDocumentBundleV1", bundle_fields)
    roots["amendment_document_root"] = roots["normative_document_bundle_root"]

    # These three roots are exact committed byte-bearing preimages.  They are
    # deliberately not random roots copied from historical diagnostic JSON.
    for key, path in (
        ("parent_normative_decision", PARENT_NORMATIVE_DECISION_PATH),
        ("parent_execution_evidence", PARENT_EXECUTION_EVIDENCE_PATH),
        ("shrink1_subset_replay", SHRINK1_SUBSET_REPLAY_PATH),
    ):
        fields = _document_fields(commit_wire, path, _git_blob(repository_root, commit, path))
        objects[key] = MappingProxyType(fields)
        roots[key + "_root"] = candidate_content_root("NormativeDocumentBlobV1", fields)

    profile_specs = (
        ("canonical_ast_profile", "CanonicalAstProfileSpecV1", "profile_id_digest", "hegel-canonical-ast-v1"),
        ("canonical_cbor_profile", "CanonicalCborProfileSpecV1", "profile_id_digest", "hegel-cbor-det-v1"),
        ("phase2b_contract", "Phase2BContractSpecV1", "contract_id_digest", "hegel-phase2b-contract-v1.1.2"),
        ("mdl_code_table", "MdlCodeTableSpecV1", "table_id_digest", "hegel-mdl-prefix-v1.0.0"),
        ("hidden_artifact_scope", "HiddenArtifactScopeV1", "policy_id_digest", "hegel-hidden-artifact-scope-v1.1.2"),
    )
    for key, schema_name, identity_field, identity in profile_specs:
        fields = _profile_fields(
            identity_field=identity_field,
            identity=identity,
            governing_root=roots["normative_document_bundle_root"],
            source_blob=document_blobs[1],
            source_path=document_paths[1],
            commit_wire=commit_wire,
        )
        objects[key] = MappingProxyType(fields)
        roots[key + "_root"] = candidate_content_root(schema_name, fields)
    roots["canonical_ast_schema_root"] = roots["canonical_ast_profile_root"]

    dsl_source_path = "Hegel Machine/src/hegel_machine/phase3_dsl_v1.py"
    shrink_source_path = "Hegel Machine/src/hegel_machine/phase3_shrink1_registry_v1.py"
    dsl_source = _git_blob(repository_root, commit, dsl_source_path)
    shrink_source = _git_blob(repository_root, commit, shrink_source_path)
    semantics_source_fields = _document_fields(
        commit_wire,
        dsl_source_path,
        dsl_source,
    )
    objects["legacy_parent_dsl_source"] = MappingProxyType(semantics_source_fields)
    roots["legacy_parent_dsl_source_root"] = candidate_content_root("NormativeDocumentBlobV1", semantics_source_fields)
    child_source_fields = _document_fields(commit_wire, shrink_source_path, shrink_source)
    objects["child_registry_source"] = MappingProxyType(child_source_fields)
    roots["child_registry_source_root"] = candidate_content_root("NormativeDocumentBlobV1", child_source_fields)

    parent_identifiers, parent_identifier_preimages = build_identifier_registry_rows_v1(parent=True)
    child_identifiers, child_identifier_preimages = build_identifier_registry_rows_v1(parent=False)
    parent_operators, parent_operator_preimages = build_operator_semantics_rows_v1(
        executable_semantics_root=roots["legacy_parent_dsl_source_root"], parent=True
    )
    child_operators, child_operator_preimages = build_operator_semantics_rows_v1(
        executable_semantics_root=roots["legacy_parent_dsl_source_root"], parent=False
    )
    record_sets["parent_identifier_registry"] = parent_identifiers
    record_sets["identifier_registry"] = child_identifiers
    record_sets["parent_operator_semantics"] = parent_operators
    record_sets["operator_semantics"] = child_operators
    roots["parent_identifier_registry_root"] = candidate_record_tree_root("IdentifierRegistryEntryV1", parent_identifiers)
    roots["identifier_registry_root"] = candidate_record_tree_root("IdentifierRegistryEntryV1", child_identifiers)
    roots["parent_operator_semantics_root"] = candidate_record_tree_root("OperatorSemanticsEntryV1", parent_operators)
    roots["operator_semantics_root"] = candidate_record_tree_root("OperatorSemanticsEntryV1", child_operators)
    tombstones = tuple(row for row in child_identifiers if row["entry_state_id"] == 2)
    survivors = tuple(row for row in child_identifiers if row["entry_state_id"] == 1)
    record_sets["removed_identifier_registry"] = tombstones
    record_sets["surviving_identifier_registry"] = survivors
    roots["removed_registry_entry_root"] = candidate_record_tree_root("IdentifierRegistryEntryV1", tombstones)
    roots["surviving_registry_entry_root"] = candidate_record_tree_root("IdentifierRegistryEntryV1", survivors)
    ordinary.update({"parent/" + key: value for key, value in parent_identifier_preimages.items()})
    ordinary.update({"child/" + key: value for key, value in child_identifier_preimages.items()})
    ordinary.update({"parent/" + key: value for key, value in parent_operator_preimages.items()})
    ordinary.update({"child/" + key: value for key, value in child_operator_preimages.items()})

    parent_dsl = {
        "dsl_version_id_digest": id_digest_v1(PARENT_DSL_ID),
        "parent_dsl_spec_root": roots["legacy_parent_dsl_source_root"],
        "canonical_ast_schema_root": roots["canonical_ast_schema_root"],
        "canonical_cbor_profile_root": roots["canonical_cbor_profile_root"],
        "identifier_registry_root": roots["parent_identifier_registry_root"],
        "operator_semantics_root": roots["parent_operator_semantics_root"],
        "equivalence_mode_id": 1,
        "max_ast_depth": 4,
        "max_ast_node_count": 7,
        "max_top_level_clauses": 3,
        "max_distinct_bit_slots": 4,
        "max_aggregate_leaves": 1,
        "max_scope_clauses": 2,
        "max_composition_depth": 2,
        "max_fitted_parameters": 3,
        "max_entity_set_size": 8,
        "canonical_program_budget": 50_000,
        "raw_operator_application_cap": 5_000_000,
        "shrink_step_id_digest": id_digest_v1("SHRINK_STEP_0_FROZEN_PARENT"),
    }
    objects["parent_dsl_spec"] = MappingProxyType(parent_dsl)
    roots["parent_dsl_spec_root"] = candidate_content_root("DslSpecV1", parent_dsl)
    child_dsl = dict(parent_dsl)
    child_dsl.update(
        {
            "dsl_version_id_digest": id_digest_v1(CHILD_DSL_ID),
            "parent_dsl_spec_root": roots["parent_dsl_spec_root"],
            "identifier_registry_root": roots["identifier_registry_root"],
            "operator_semantics_root": roots["operator_semantics_root"],
            "shrink_step_id_digest": id_digest_v1("SHRINK_STEP_1_REMOVE_MEAN_MIN_MAX"),
        }
    )
    objects["child_dsl_spec"] = MappingProxyType(child_dsl)
    roots["child_dsl_spec_root"] = candidate_content_root("DslSpecV1", child_dsl)

    parent_freeze = {
        "freeze_version_id_digest": id_digest_v1(PARENT_FREEZE_ID),
        "parent_freeze_root_or_null": None,
        "child_dsl_spec_root": roots["parent_dsl_spec_root"],
        "phase2b_contract_root": roots["phase2b_contract_root"],
        "canonical_ast_schema_root": roots["canonical_ast_schema_root"],
        "canonical_cbor_profile_root": roots["canonical_cbor_profile_root"],
        "mdl_code_table_root": roots["mdl_code_table_root"],
        "amendment_document_root": roots["normative_document_1_root"],
        "effective_repository_commit_id": commit_wire,
    }
    objects["parent_freeze"] = MappingProxyType(parent_freeze)
    roots["parent_freeze_root"] = candidate_content_root("FreezeSpecV1", parent_freeze)
    child_freeze = dict(parent_freeze)
    child_freeze.update(
        {
            "freeze_version_id_digest": id_digest_v1(CHILD_FREEZE_ID),
            "parent_freeze_root_or_null": roots["parent_freeze_root"],
            "child_dsl_spec_root": roots["child_dsl_spec_root"],
            "amendment_document_root": roots["normative_document_bundle_root"],
        }
    )
    objects["child_freeze"] = MappingProxyType(child_freeze)
    roots["child_freeze_root"] = candidate_content_root("FreezeSpecV1", child_freeze)

    tombstone_policy = {
        "registry_namespace_id_digest": id_digest_v1("AggregateMapId/v1"),
        "id_reuse_allowed": False,
        "removed_source_name_error_id_digest": id_digest_v1("REJECT_REMOVED_AGGREGATE_MAP"),
        "removed_numeric_id_error_id_digest": id_digest_v1("REJECT_REMOVED_AGGREGATE_MAP"),
        "unknown_numeric_id_error_id_digest": id_digest_v1("REJECT_REGISTRY_INDEX_OUT_OF_RANGE"),
    }
    objects["tombstone_policy"] = MappingProxyType(tombstone_policy)
    roots["tombstone_policy_root"] = candidate_content_root("TombstonePolicyV1", tombstone_policy)
    cross_policy = {
        "ast_hash_domain_id_digest": id_digest_v1("HEGEL/AST/V1"),
        "surviving_ast_bytes_stable": True,
        "surviving_ast_hash_stable": True,
        "semantic_identity_domain_id_digest": id_digest_v1("HEGEL/PROGRAM_SEMANTIC_IDENTITY/V1"),
        "required_binding_root_role_ids": (7, 8, 9),
        "cross_version_archive_reuse_allowed": False,
        "cross_version_receipt_reuse_allowed": False,
        "cross_version_certificate_reuse_allowed": False,
    }
    objects["cross_dsl_hash_policy"] = MappingProxyType(cross_policy)
    roots["cross_dsl_hash_policy_root"] = candidate_content_root("CrossDslHashPolicyV1", cross_policy)

    traversal = {
        "bucket_key_field_ids": (1, 2, 3),
        "canonical_sort_key_field_ids": (1, 2, 3, 4, 5),
        "commutative_child_ordering_rule_id_digest": id_digest_v1("rule:child-canonical-hash-ascending-v1"),
        "maximum_canonical_programs": 50_000,
        "maximum_raw_operator_applications": 5_000_000,
        "frontier_exhaustion_definition_id_digest": id_digest_v1("rule:all-type-buckets-closed-frontier-exhausted-v1"),
    }
    bucket = {
        "bucket_key_field_ids": (1, 2, 3),
        "required_counter_field_ids": (1, 2, 3, 4, 5, 6),
        "bucket_ordering_rule_id_digest": id_digest_v1("rule:output-sort-depth-node-count-ascending-v1"),
        "zero_count_bucket_emission_required": True,
        "accounting_sum_invariants": (1, 2, 3),
    }
    program_archive = {
        "program_record_schema_tag": OBJECT_TAGS["CanonicalProgramRecordV2"],
        "program_ordering_rule_id_digest": id_digest_v1("rule:canonical-program-index-ascending-v1"),
        "records_per_chunk": 4096,
        "chunk_blob_codec_id": 0,
        "chunk_blob_framing_rule_id_digest": id_digest_v1("codec:identity-uint32be-length-framed-v1"),
        "rfc6962_profile_id_digest": id_digest_v1("profile:hegel-rfc6962-v1"),
        "target_independent": True,
    }
    output_archive = {
        "output_record_schema_tag": OBJECT_TAGS["ProgramOutputRecordV2"],
        "output_ordering_rule_id_digest": id_digest_v1("rule:role-program-index-ascending-v1"),
        "records_per_chunk": 4096,
        "chunk_blob_codec_id": 0,
        "chunk_blob_framing_rule_id_digest": id_digest_v1("codec:identity-uint32be-length-framed-v1"),
        "undefined_bitmap_profile_id_digest": id_digest_v1("profile:hegel-undefined-bitmap-v1"),
        "role_specific": True,
    }
    for key, schema_name, fields in (
        ("traversal_contract", "TraversalContractV1", traversal),
        ("bucket_accounting_contract", "BucketAccountingContractV1", bucket),
        ("program_archive_contract", "ProgramArchiveContractV1", program_archive),
        ("output_archive_contract", "OutputArchiveContractV1", output_archive),
    ):
        objects[key] = MappingProxyType(fields)
        roots[key + "_root"] = candidate_content_root(schema_name, fields)

    for key, enum_name, prefix in (
        ("m3_state_registry", "M3StateId", "m3-state"),
        ("m3_phase_registry", "M3RunningPhaseId", "m3-phase"),
        ("m3_transition_reason_registry", "M3TransitionReasonId", "m3-reason"),
    ):
        rows = _enum_registry_rows(enum_name, prefix)
        record_sets[key] = rows
        roots[key + "_root"] = candidate_record_tree_root("IdentifierRegistryEntryV1", rows)
    state_contract = _state_contract_fields(roots)
    objects["state_machine_contract"] = MappingProxyType(state_contract)
    roots["state_machine_contract_root"] = candidate_content_root("StateMachineContractV1", state_contract)

    odd_rows = generate_odd_role_rows_v1()
    sink_rows = generate_sink_role_rows_v1()
    roots.update(
        {
            "outside_target_universe_root": odd_rows.universe_root,
            "outside_target_truth_root": odd_rows.truth_root,
            "null_control_universe_root": sink_rows.universe_root,
            "null_control_truth_root": sink_rows.truth_root,
        }
    )
    witness = canonicalize_shrink1_source_ast(
        [
            "equal_exact",
            ["aggregate", "signed_balance_v1", "control_volume_all_observed_v1", "q0", []],
            ["scalar_const", 0, 1],
        ]
    )
    witness_hash = bytes.fromhex(witness.hash_id.removeprefix("sha256:"))
    metadata_odd = {
        "input_signature_id": 1,
        "role_ids": (),
        "quantity_ids": (),
        "scope_ids": (),
        "signed_orientations": (),
        "metadata_rule_id_digest": id_digest_v1("rule:odd-no-static-role-metadata-v1"),
    }
    metadata_sink = {
        "input_signature_id": 2,
        "role_ids": (0, 1, 2, 3),
        "quantity_ids": (0,),
        "scope_ids": (3,),
        "signed_orientations": (1, 1, -1, -1),
        "metadata_rule_id_digest": id_digest_v1("rule:sink-signed-balance-static-role-metadata-v1"),
    }
    odd_signature = {
        "input_signature_id": 1,
        "input_object_tag": OBJECT_TAGS["OddInputV1"],
        "field_sort_ids": (16,),
        "static_role_metadata": build_formal_object("StaticRoleMetadataV1", metadata_odd),
        "canonical_ordering_rule_id_digest": id_digest_v1("rule:odd-set-size-then-bitstring-v1"),
    }
    sink_signature = {
        "input_signature_id": 2,
        "input_object_tag": OBJECT_TAGS["SinkInputV1"],
        "field_sort_ids": (4, 4, 4, 4),
        "static_role_metadata": build_formal_object("StaticRoleMetadataV1", metadata_sink),
        "canonical_ordering_rule_id_digest": id_digest_v1("rule:sink-a-b-c-d-lexicographic-v1"),
    }
    for key, fields in (("odd_input_signature", odd_signature), ("sink_input_signature", sink_signature)):
        objects[key] = MappingProxyType(fields)
        roots[key + "_root"] = candidate_content_root("InputSignatureSpecV1", fields)
    odd_target = {
        "role_id": 1,
        "target_machine_id_digest": id_digest_v1(ODD_REDUCTION_TARGET.target_id),
        "input_signature_spec_root": roots["odd_input_signature_root"],
        "output_sort_id": 2,
        "target_rule_id_digest": id_digest_v1("rule:generic-odd-cardinality-reduction-v1"),
        "universe_row_count": 480,
        "target_output_cardinality": 2,
        "required_witness_ast_hash_or_null": None,
        "claim_level_id": 3,
    }
    sink_target = {
        "role_id": 2,
        "target_machine_id_digest": id_digest_v1(OBSERVED_OMITTED_SINK_CONTROL.control_id),
        "input_signature_spec_root": roots["sink_input_signature_root"],
        "output_sort_id": 2,
        "target_rule_id_digest": id_digest_v1("rule:observed-omitted-sink-balance-v1"),
        "universe_row_count": 85,
        "target_output_cardinality": 1,
        "required_witness_ast_hash_or_null": witness_hash,
        "claim_level_id": 1,
    }
    for key, fields in (("outside_target_spec", odd_target), ("null_control_spec", sink_target)):
        objects[key] = MappingProxyType(fields)
        roots[key + "_root"] = candidate_content_root("TargetSpecFormalV1", fields)

    split_algorithm = {
        "os_csprng_profile_id_digest": id_digest_v1("profile:os-csprng-32-byte-v1"),
        "hkdf_profile_id_digest": id_digest_v1("profile:hkdf-sha256-role-key-v1"),
        "rank_hmac_profile_id_digest": id_digest_v1("profile:hmac-sha256-split-rank-v1"),
        "rank_tie_break_rule_id_digest": id_digest_v1("rule:rank-digest-then-canonical-input-hash-v1"),
        "exhaustive_partition_required": True,
        "assignment_row_schema_tag": OBJECT_TAGS["SplitAssignmentRowV1"],
    }
    objects["split_algorithm"] = MappingProxyType(split_algorithm)
    roots["split_algorithm_spec_root"] = candidate_content_root("SplitAlgorithmSpecV1", split_algorithm)
    split_contract = {
        "split_contract_version_id_digest": id_digest_v1("hegel-split-contract-p3a-v1.1.2"),
        "split_algorithm_spec_root": roots["split_algorithm_spec_root"],
        "hkdf_profile_id_digest": split_algorithm["hkdf_profile_id_digest"],
        "rank_hmac_profile_id_digest": split_algorithm["rank_hmac_profile_id_digest"],
        "exhaustive_partition_required": True,
        "odd_stratum_quota_table": (
            (1, 16, 6, 3, 7), (2, 16, 6, 3, 7), (3, 32, 13, 6, 13), (4, 32, 13, 6, 13),
            (5, 64, 26, 13, 25), (6, 64, 26, 13, 25), (7, 128, 51, 26, 51), (8, 128, 51, 26, 51),
        ),
        "sink_stratum_quota_table": ((9, 15, 7, 4, 4), (10, 18, 8, 4, 6), (11, 19, 9, 4, 6), (12, 18, 8, 4, 6), (13, 15, 7, 4, 4)),
        "assignment_ordering_rule_id": 1,
        "fallback_split_policy_id": 1,
        "hidden_artifact_scope_root": roots["hidden_artifact_scope_root"],
    }
    objects["split_contract"] = MappingProxyType(split_contract)
    roots["split_contract_root"] = candidate_content_root("SplitContractV1", split_contract)

    fallback = {
        "fallback_entries": (
            (1, id_digest_v1("TARGET_P3A_GENERIC_COUNT_MOD_3_EQ_1_V1"), None),
            (2, id_digest_v1("TARGET_P3A_GENERIC_PRIME_COUNT_V1"), None),
        ),
        "selection_rule_id_digest": id_digest_v1("rule:first-preregistered-outside-target-v1"),
        "requires_new_target_version": True,
        "requires_new_split_first_instantiation": True,
    }
    objects["fallback_registry"] = MappingProxyType(fallback)
    roots["fallback_registry_root"] = candidate_content_root("FallbackRegistryV1", fallback)
    target_bundle = {
        "outside_target_spec_root": roots["outside_target_spec_root"],
        "outside_target_universe_root": roots["outside_target_universe_root"],
        "outside_target_truth_root": roots["outside_target_truth_root"],
        "null_control_spec_root": roots["null_control_spec_root"],
        "null_control_universe_root": roots["null_control_universe_root"],
        "null_control_truth_root": roots["null_control_truth_root"],
        "fallback_registry_root": roots["fallback_registry_root"],
        "null_control_required_witness_ast_hash_or_null": witness_hash,
        "null_control_claim_level_id": 1,
    }
    objects["target_bundle"] = MappingProxyType(target_bundle)
    roots["target_bundle_root"] = candidate_content_root("TargetBundleV1", target_bundle)

    approval_evidence = {
        "amendment_document_root": roots["amendment_document_root"],
        "approving_actor_id_digest": id_digest_v1("project-owner:erzhu419"),
        "approval_statement_id_digest": id_digest_v1(
            "approve:hegel-freeze-p2b-p3-v1.1.2"
        ),
        "parent_normative_decision_root": roots["parent_normative_decision_root"],
        "approval_method_id": 1,
        "approval_recorded_at_unix_seconds": recorded_at_unix_seconds,
    }
    objects["approval_evidence"] = MappingProxyType(approval_evidence)
    roots["approval_evidence_root"] = candidate_content_root(
        "ApprovalEvidenceBundleV1", approval_evidence
    )

    normative_approval = {
        "amendment_document_root": roots["amendment_document_root"],
        "parent_freeze_root": roots["parent_freeze_root"],
        "child_freeze_root": roots["child_freeze_root"],
        "child_dsl_spec_root_or_null": roots["child_dsl_spec_root"],
        "approval_status_id": 1,
        "approval_method_id": 1,
        "approval_evidence_root": roots["approval_evidence_root"],
        "approving_actor_id_digest": approval_evidence["approving_actor_id_digest"],
        "recorded_at_unix_seconds": recorded_at_unix_seconds,
        "repository_commit_id": commit_wire,
    }
    objects["normative_approval_manifest"] = MappingProxyType(normative_approval)
    roots["normative_approval_manifest_root"] = candidate_content_root(
        "NormativeApprovalManifestV1", normative_approval
    )
    roots["approval_manifest_root"] = roots["normative_approval_manifest_root"]

    replacement_policy = {
        "key_rotation_threshold": 2,
        "key_revocation_threshold": 2,
        "custodian_replacement_requires_new_seed_version": True,
        "actor_key_reuse_across_purposes_allowed": False,
        "secret_material_export_allowed": False,
    }
    objects["replacement_policy"] = MappingProxyType(replacement_policy)
    roots["replacement_policy_root"] = candidate_content_root(
        "ReplacementPolicyV1", replacement_policy
    )

    split_spec_freeze = {
        "split_contract_root": roots["split_contract_root"],
        "target_bundle_root": roots["target_bundle_root"],
        "child_freeze_root": roots["child_freeze_root"],
        "amendment_document_root": roots["amendment_document_root"],
        "seed_state_id": 1,
        "frozen_at_unix_seconds": recorded_at_unix_seconds,
        "repository_commit_id": commit_wire,
    }
    objects["split_spec_freeze"] = MappingProxyType(split_spec_freeze)
    roots["split_spec_freeze_root"] = candidate_content_root(
        "SplitSpecFreezeV1", split_spec_freeze
    )

    # Exact legacy diagnostic byte preimages and their formal bridge records.
    diagnostic_sources: list[
        tuple[int, int, object, bytes, int, int, bytes, int | None]
    ] = []
    def add_diag(role: int, namespace: int, value: object, formal_root: bytes, formal_kind: int, tag: int, count: int | None) -> None:
        preimage = _diagnostic_preimage(value)
        diagnostic[f"role-{role}"] = preimage
        diagnostic_sources.append(
            (role, namespace, value, formal_root, formal_kind, tag, preimage, count)
        )

    add_diag(1, 1, ODD_REDUCTION_TARGET, roots["outside_target_spec_root"], 1, OBJECT_TAGS["TargetSpecFormalV1"], None)
    add_diag(2, 3, tuple((r.universe_index, r.set_size, r.bits) for r in __import__("hegel_machine.phase3_dsl_v1", fromlist=["ODD_REDUCTION_UNIVERSE"]).ODD_REDUCTION_UNIVERSE), roots["outside_target_universe_root"], 2, OBJECT_TAGS["BoundedUniverseRowV1"], 480)
    add_diag(3, 4, tuple((r.universe_index, r.bits, r.target_output) for r in __import__("hegel_machine.phase3_dsl_v1", fromlist=["ODD_REDUCTION_UNIVERSE"]).ODD_REDUCTION_UNIVERSE), roots["outside_target_truth_root"], 2, OBJECT_TAGS["TargetTruthRowV1"], 480)
    add_diag(4, 2, OBSERVED_OMITTED_SINK_CONTROL, roots["null_control_spec_root"], 1, OBJECT_TAGS["TargetSpecFormalV1"], None)
    sink_universe = __import__("hegel_machine.phase3_dsl_v1", fromlist=["OMITTED_SINK_UNIVERSE"]).OMITTED_SINK_UNIVERSE
    add_diag(5, 3, tuple((r.universe_index, r.inflow_a, r.inflow_b, r.primary_outflow, r.auxiliary_outflow) for r in sink_universe), roots["null_control_universe_root"], 2, OBJECT_TAGS["BoundedUniverseRowV1"], 85)
    add_diag(6, 4, tuple((r.universe_index, int(r.full_balance_residual == 0)) for r in sink_universe), roots["null_control_truth_root"], 2, OBJECT_TAGS["TargetTruthRowV1"], 85)
    add_diag(7, 5, shrunk_dsl_surface_object(), roots["child_dsl_spec_root"], 1, OBJECT_TAGS["DslSpecV1"], None)
    add_diag(8, 6, operator_admission_semantics_object(), roots["operator_semantics_root"], 2, OBJECT_TAGS["OperatorSemanticsEntryV1"], 28)
    add_diag(9, 7, {"schema_version": "hegel-formal-identifier-registry-diagnostic/1", "rows": [{k: (v.hex() if type(v) is bytes else v) for k, v in row.items()} for row in child_identifiers]}, roots["identifier_registry_root"], 2, OBJECT_TAGS["IdentifierRegistryEntryV1"], 55)
    add_diag(10, 8, {"profile_id": "hegel-canonical-ast-v1", "source_sha256": hashlib.sha256(document_blobs[1]).hexdigest()}, roots["canonical_ast_schema_root"], 1, OBJECT_TAGS["CanonicalAstProfileSpecV1"], None)
    add_diag(11, 9, {"profile_id": "hegel-cbor-det-v1", "source_sha256": hashlib.sha256(document_blobs[1]).hexdigest()}, roots["canonical_cbor_profile_root"], 1, OBJECT_TAGS["CanonicalCborProfileSpecV1"], None)
    add_diag(12, 10, {"schema_version": "hegel-split-contract-diagnostic/1", "odd_quotas": split_contract["odd_stratum_quota_table"], "sink_quotas": split_contract["sink_stratum_quota_table"]}, roots["split_contract_root"], 1, OBJECT_TAGS["SplitContractV1"], None)

    bridge_rows: list[Mapping[str, object]] = []
    for (
        role,
        namespace,
        _,
        formal_root,
        formal_kind,
        target_tag,
        preimage,
        count,
    ) in diagnostic_sources:
        formal_profile = 2 if formal_kind == 2 else 1
        transform = {
            "source_diagnostic_profile_id": 1,
            "source_namespace_id": namespace,
            "target_formal_profile_id": formal_profile,
            "target_object_tag": target_tag,
            "transform_rule_id_digest": id_digest_v1(f"rule:legacy-json-to-formal-role-{role}-v1"),
            "ordering_rule_id_digest": id_digest_v1("rule:formal-schema-order-v1" if count is None else "rule:formal-record-order-v1"),
            "expected_row_count_or_null": count,
        }
        transform_root = candidate_content_root("RowTransformSpecV1", transform)
        objects[f"row_transform_role_{role}"] = MappingProxyType(transform)
        roots[f"row_transform_role_{role}_root"] = transform_root
        diagnostic_digest = hashlib.sha256(preimage).digest()
        row = {
            "artifact_role_id": role,
            "diagnostic_namespace_id": namespace,
            "diagnostic_digest": diagnostic_digest,
            "formal_object_kind_id": formal_kind,
            "formal_digest_or_root": formal_root,
            "row_count_or_null": count,
            "diagnostic_profile_id_digest": id_digest_v1("profile:HEGEL_LEGACY_STABLE_JSON_V1"),
            "formal_profile_id_digest": id_digest_v1("profile:HEGEL_CBOR_CONTENT_HASH_V1" if formal_kind == 1 else "profile:HEGEL_RFC6962_ROW_TREE_V1"),
            "row_transform_spec_root": transform_root,
            "source_artifact_digest": diagnostic_digest,
            "repository_commit_id": commit_wire,
        }
        build_formal_object("DiagnosticFormalBridgeRecordV1", row)
        bridge_rows.append(MappingProxyType(row))
    bridge_rows.sort(key=lambda row: (row["artifact_role_id"], row["diagnostic_namespace_id"], row["diagnostic_digest"]))
    record_sets["diagnostic_formal_bridge"] = tuple(bridge_rows)
    roots["diagnostic_formal_bridge_root"] = candidate_record_tree_root("DiagnosticFormalBridgeRecordV1", bridge_rows)

    python_sources = (
        CONTAINER_PROFILE_PATH,
        CONTAINER_SECCOMP_PATH,
        "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_rows_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_dsl_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_shrink1_registry_v1.py",
        "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
        "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
        GENERATOR_PATH,
        ENGINEERING_FREEZE_PATH,
    )
    rust_sources = (
        CONTAINER_PROFILE_PATH,
        CONTAINER_SECCOMP_PATH,
        "Hegel Machine/rust/formal_bridge_m25/Cargo.toml",
        "Hegel Machine/rust/formal_bridge_m25/Cargo.lock",
        "Hegel Machine/rust/formal_bridge_m25/src/lib.rs",
        "Hegel Machine/rust/formal_bridge_m25/src/main.rs",
        ENGINEERING_FREEZE_PATH,
    )
    if not require_generator_committed:
        python_sources = tuple(path for path in python_sources if path not in required_new)
        rust_sources = tuple(path for path in rust_sources if path not in required_new)
    py_source_rows = _source_file_rows(repository_root, commit, python_sources)
    rs_source_rows = _source_file_rows(repository_root, commit, rust_sources)
    lock_rows = _dependency_lock_rows(_git_blob(repository_root, commit, "Hegel Machine/rust/formal_bridge_m25/Cargo.lock"))
    record_sets["python_implementation_sources"] = py_source_rows
    record_sets["rust_implementation_sources"] = rs_source_rows
    record_sets["rust_dependency_lock"] = lock_rows
    roots["python_source_root"] = candidate_record_tree_root("SourceFileRecordV1", py_source_rows)
    roots["rust_source_root"] = candidate_record_tree_root("SourceFileRecordV1", rs_source_rows)
    roots["python_dependency_lock_root"] = candidate_record_tree_root("DependencyLockRecordV1", ())
    roots["rust_dependency_lock_root"] = candidate_record_tree_root("DependencyLockRecordV1", lock_rows)

    try:
        profile = json.loads(_git_blob(repository_root, commit, CONTAINER_PROFILE_PATH))
        rust_image = profile["images"]["rust_attester"]
        python_image = profile["images"]["python_attester"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        _fail(FAIL_BASIS_PREIMAGE, f"container profile is invalid: {exc}")
    def image_digest(image: str) -> bytes:
        if type(image) is not str or re.fullmatch(r"[a-z0-9._/-]+@sha256:[0-9a-f]{64}", image) is None:
            _fail(FAIL_BASIS_PREIMAGE, "container image is not digest pinned")
        return bytes.fromhex(image.rsplit(":", 1)[1])
    for impl, runtime, image, dep_root in (
        ("python", "python", python_image, roots["python_dependency_lock_root"]),
        ("rust", "rust", rust_image, roots["rust_dependency_lock_root"]),
    ):
        env = {
            "os_id_digest": id_digest_v1("os:linux-oci"),
            "architecture_id_digest": id_digest_v1("architecture:x86_64"),
            "runtime_id_digest": id_digest_v1("runtime:" + runtime),
            "runtime_version_id_digest": id_digest_v1("oci-manifest:" + image.rsplit(":", 1)[1]),
            "dependency_lock_root": dep_root,
            "locale_id_digest": id_digest_v1("locale:C.UTF-8"),
            "timezone_id_digest": id_digest_v1("timezone:UTC"),
            "container_or_host_profile_id_digest": id_digest_v1("hegel-owner-accepted-container-technical-actors-v1"),
            "oci_manifest_digest_or_null": image_digest(image),
        }
        objects[f"{impl}_execution_environment"] = MappingProxyType(env)
        roots[f"{impl}_execution_environment_root"] = candidate_content_root("ExecutionEnvironmentSpecV1", env)

    def exact_binary(path: Path, label: str) -> bytes:
        try:
            resolved_path = path.resolve(strict=True)
            payload = resolved_path.read_bytes()
        except OSError as exc:
            _fail(FAIL_BASIS_PREIMAGE, f"cannot read {label} binary: {exc}")
        if not payload:
            _fail(FAIL_BASIS_PREIMAGE, f"{label} binary is empty")
        ordinary[f"binary/{label}"] = payload
        return hashlib.sha256(payload).digest()
    python_binary_digest = exact_binary(python_binary_path, "python")
    rust_binary_digest = exact_binary(rust_binary_path, "rust")
    committed_seccomp = _git_blob(repository_root, commit, CONTAINER_SECCOMP_PATH)
    ordinary["container/seccomp"] = committed_seccomp
    golden_root = roots.get("normative_document_bundle_root")
    assert golden_root is not None
    for impl, impl_id, source_root, binary_digest, dep_root, image in (
        ("python", 1, roots["python_source_root"], python_binary_digest, roots["python_dependency_lock_root"], python_image),
        ("rust", 2, roots["rust_source_root"], rust_binary_digest, roots["rust_dependency_lock_root"], rust_image),
    ):
        binding = {
            "implementation_id": impl_id,
            "source_root": source_root,
            "binary_digest": binary_digest,
            "execution_environment_spec_root": roots[f"{impl}_execution_environment_root"],
            "compiler_or_interpreter_id_digest": id_digest_v1("runtime:" + impl),
            "compiler_or_interpreter_version_digest": id_digest_v1("oci-manifest:" + image.rsplit(":", 1)[1]),
            "dependency_lock_root": dep_root,
            "build_profile_id_digest": id_digest_v1("build:offline-locked-release-v1" if impl == "rust" else "build:python-isolated-byte-exact-v1"),
            "entrypoint_id_digest": id_digest_v1("entrypoint:formal-static-basis-python-v1" if impl == "python" else "entrypoint:formal-bridge-m25-v1"),
            "golden_vector_root": golden_root,
            "repository_commit_id": commit_wire,
        }
        objects[f"{impl}_static_replay_implementation_binding"] = MappingProxyType(binding)
        roots[f"{impl}_static_replay_implementation_binding_root"] = candidate_content_root("ImplementationBindingV1", binding)

    gate19_plan = (
        _root_plan_content("DslSpecV1", "child_dsl_spec_root", child_dsl),
        _root_plan_content("FreezeSpecV1", "child_freeze_root", child_freeze),
        _root_plan_records("OperatorSemanticsEntryV1", "operator_semantics_root", child_operators),
        _root_plan_records("IdentifierRegistryEntryV1", "identifier_registry_root", child_identifiers),
        _root_plan_content("CanonicalAstProfileSpecV1", "canonical_ast_schema_root", objects["canonical_ast_profile"]),
        _root_plan_content("CanonicalCborProfileSpecV1", "canonical_cbor_profile_root", objects["canonical_cbor_profile"]),
    )
    if tuple(entry.root_name for entry in gate19_plan) != GATE19_ROOT_NAMES:
        raise AssertionError("Gate-19 replay plan order drift")
    if any(roots[entry.root_name] != entry.expected_root for entry in gate19_plan):
        raise AssertionError("Gate-19 plan does not match root DAG")

    m3_fields: dict[str, object] = {
        "child_dsl_spec_root": roots["child_dsl_spec_root"],
        "child_freeze_root": roots["child_freeze_root"],
        "approval_manifest_root": roots["approval_manifest_root"],
        "operator_semantics_root": roots["operator_semantics_root"],
        "identifier_registry_root": roots["identifier_registry_root"],
        "canonical_ast_schema_root": roots["canonical_ast_schema_root"],
        "canonical_cbor_profile_root": roots["canonical_cbor_profile_root"],
        "diagnostic_formal_bridge_root": roots["diagnostic_formal_bridge_root"],
        "outside_target_universe_root": roots["outside_target_universe_root"],
        "outside_target_truth_root": roots["outside_target_truth_root"],
        "null_control_universe_root": roots["null_control_universe_root"],
        "null_control_truth_root": roots["null_control_truth_root"],
        "canonical_program_budget": 50_000,
        "raw_operator_application_cap": 5_000_000,
        "records_per_chunk": 4096,
        "equivalence_mode_id": 1,
        "traversal_contract_root": roots["traversal_contract_root"],
        "bucket_accounting_contract_root": roots["bucket_accounting_contract_root"],
        "program_archive_contract_root": roots["program_archive_contract_root"],
        "output_archive_contract_root": roots["output_archive_contract_root"],
        "state_machine_contract_root": roots["state_machine_contract_root"],
        "repository_commit_id": commit_wire,
    }
    preseed_static = {
        "NormativeApprovalManifestV1": MappingProxyType(dict(normative_approval)),
        "SplitSpecFreezeV1": MappingProxyType(dict(split_spec_freeze)),
        "SplitBindingManifestV1": MappingProxyType(
            {
                "split_contract_root": roots["split_contract_root"],
                "split_algorithm_id_digest": id_digest_v1(SPLIT_ALGORITHM_ID),
                "split_instantiation_status_id": 1,
                "repository_commit_id": commit_wire,
            }
        ),
        "DslRoleBindingManifestV1/OUTSIDE_TARGET": MappingProxyType(
            {
                "role_id": 1,
                "child_dsl_spec_root": roots["child_dsl_spec_root"],
                "child_freeze_root": roots["child_freeze_root"],
                "operator_semantics_root": roots["operator_semantics_root"],
                "identifier_registry_root": roots["identifier_registry_root"],
                "canonical_ast_schema_root": roots["canonical_ast_schema_root"],
                "canonical_cbor_profile_root": roots["canonical_cbor_profile_root"],
                "semantic_spec_diagnostic_id_digest": id_digest_v1(
                    ODD_REDUCTION_TARGET.content_id
                ),
                "semantic_spec_formal_root": roots["outside_target_spec_root"],
                "universe_diagnostic_id_digest": id_digest_v1(
                    ODD_REDUCTION_TARGET.diagnostic_universe_content_id
                ),
                "truth_diagnostic_id_digest": id_digest_v1(
                    ODD_REDUCTION_TARGET.diagnostic_target_table_content_id
                ),
                "formal_universe_root": roots["outside_target_universe_root"],
                "formal_truth_root": roots["outside_target_truth_root"],
                "parent_binding_manifest_root_or_null": None,
                "legacy_parent_payload_source_id_digest_or_null": id_digest_v1(
                    ODD_REDUCTION_TARGET.content_id
                ),
                "fallback_registry_root_or_null": roots["fallback_registry_root"],
                "repository_commit_id": commit_wire,
            }
        ),
        "DslRoleBindingManifestV1/IN_LANGUAGE_NULL": MappingProxyType(
            {
                "role_id": 2,
                "child_dsl_spec_root": roots["child_dsl_spec_root"],
                "child_freeze_root": roots["child_freeze_root"],
                "operator_semantics_root": roots["operator_semantics_root"],
                "identifier_registry_root": roots["identifier_registry_root"],
                "canonical_ast_schema_root": roots["canonical_ast_schema_root"],
                "canonical_cbor_profile_root": roots["canonical_cbor_profile_root"],
                "semantic_spec_diagnostic_id_digest": id_digest_v1(
                    OBSERVED_OMITTED_SINK_CONTROL.content_id
                ),
                "semantic_spec_formal_root": roots["null_control_spec_root"],
                "universe_diagnostic_id_digest": id_digest_v1(
                    OBSERVED_OMITTED_SINK_CONTROL.diagnostic_universe_content_id
                ),
                "truth_diagnostic_id_digest": id_digest_v1(
                    OBSERVED_OMITTED_SINK_CONTROL.diagnostic_target_table_content_id
                ),
                "formal_universe_root": roots["null_control_universe_root"],
                "formal_truth_root": roots["null_control_truth_root"],
                "parent_binding_manifest_root_or_null": None,
                "legacy_parent_payload_source_id_digest_or_null": id_digest_v1(
                    OBSERVED_OMITTED_SINK_CONTROL.content_id
                ),
                "fallback_registry_root_or_null": None,
                "repository_commit_id": commit_wire,
            }
        ),
        "SeedContinuityManifestV1": MappingProxyType(
            {
                "continuity_status_id": 1,
                "split_spec_freeze_root": roots["split_spec_freeze_root"],
                "parent_seed_commitment_manifest_root_or_null": None,
                "repository_commit_id": commit_wire,
            }
        ),
        "DslShrinkTransitionFormalV1": MappingProxyType(
            {
                "parent_dsl_spec_root": roots["parent_dsl_spec_root"],
                "child_dsl_spec_root": roots["child_dsl_spec_root"],
                "parent_freeze_root": roots["parent_freeze_root"],
                "child_freeze_root": roots["child_freeze_root"],
                "parent_execution_evidence_root": roots[
                    "parent_execution_evidence_root"
                ],
                "parent_status_id": 2,
                "shrink_step_id_digest": id_digest_v1(
                    "SHRINK_STEP_1_REMOVE_MEAN_MIN_MAX"
                ),
                "removed_registry_entry_root": roots["removed_registry_entry_root"],
                "surviving_registry_entry_root": roots[
                    "surviving_registry_entry_root"
                ],
                "tombstone_policy_root": roots["tombstone_policy_root"],
                "cross_dsl_hash_policy_root": roots["cross_dsl_hash_policy_root"],
                "approval_manifest_root": roots["approval_manifest_root"],
                "shrink1_subset_replay_root": roots["shrink1_subset_replay_root"],
                "child_initial_state_id": 1,
                "repository_commit_id": commit_wire,
            }
        ),
    }
    preseed_dynamic = {
        "NormativeApprovalManifestV1": (),
        "SplitSpecFreezeV1": (),
        "SplitBindingManifestV1": (
            "split_seed_commitment_manifest_root",
            "seed_continuity_manifest_root",
            "outside_target_discovery_root",
            "outside_target_validation_root",
            "outside_target_sealed_root",
            "null_control_discovery_root",
            "null_control_validation_root",
            "null_control_sealed_root",
            "hidden_access_ledger_genesis_root",
            "hidden_access_ledger_head_root",
            "created_at_unix_seconds",
        ),
        "DslRoleBindingManifestV1/OUTSIDE_TARGET": (
            "split_binding_manifest_root",
            "custodian_binding_manifest_root",
            "seed_continuity_manifest_root",
            "parent_manifest_absence_attestation_root_or_null",
            "created_at_unix_seconds",
        ),
        "DslRoleBindingManifestV1/IN_LANGUAGE_NULL": (
            "split_binding_manifest_root",
            "custodian_binding_manifest_root",
            "seed_continuity_manifest_root",
            "parent_manifest_absence_attestation_root_or_null",
            "created_at_unix_seconds",
        ),
        "SeedContinuityManifestV1": (
            "current_seed_commitment_manifest_root",
            "parent_manifest_absence_attestation_root",
            "hidden_access_ledger_genesis_root",
            "custodian_binding_core_root",
            "instantiated_at_unix_seconds",
        ),
        "DslShrinkTransitionFormalV1": (
            "outside_target_binding_manifest_root",
            "null_control_binding_manifest_root",
            "split_binding_manifest_root",
            "custodian_binding_manifest_root",
            "seed_continuity_manifest_root",
            "created_at_unix_seconds",
        ),
    }
    implementation_inputs = {
        "python_binary_path": str(python_binary_path.resolve()),
        "python_binary_sha256": python_binary_digest,
        "rust_binary_path": str(rust_binary_path.resolve()),
        "rust_binary_sha256": rust_binary_digest,
        "python_image_ref": python_image,
        "rust_image_ref": rust_image,
        "seccomp_path": str(SECCOMP_PATH.resolve()),
        "seccomp_sha256": hashlib.sha256(committed_seccomp).digest(),
        "m3_execution_implementation_bindings_ready": False,
        "m3_execution_implementation_binding_roots": None,
    }
    return FormalStaticBasisV1(
        basis_commit=commit,
        repository_root=repository_root,
        objects=MappingProxyType(objects),
        record_sets=MappingProxyType(record_sets),
        roots=MappingProxyType(roots),
        ordinary_digest_preimages=MappingProxyType(ordinary),
        diagnostic_preimages=MappingProxyType(diagnostic),
        gate19_plan=gate19_plan,
        m3_candidate_static_fields=MappingProxyType(m3_fields),
        preseed_manifest_static_fields=MappingProxyType(preseed_static),
        preseed_manifest_required_dynamic_fields=MappingProxyType(preseed_dynamic),
        implementation_inputs=MappingProxyType(implementation_inputs),
        blocking_gaps=("M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY",),
    )


def _entry_receipt(entry: RootReplayPlanEntryV1, response: Mapping[str, object], request: Mapping[str, object]) -> dict[str, object]:
    request_bytes = _canonical_json_bytes(request)
    return {
        "root_name": entry.root_name,
        "operation": entry.operation,
        "schema_name": entry.schema_name,
        "domain_or_null": entry.domain_or_null,
        "preimage_cbor_hex": [item.hex() for item in entry.preimage_cbor],
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "response": dict(response),
        "expected_root_hex": entry.expected_root.hex(),
    }


def build_python_static_replay_receipt_v1(basis: FormalStaticBasisV1) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for entry in basis.gate19_plan:
        if entry.operation == "content_hash":
            value = canonical_cbor_decode(entry.preimage_cbor[0])
            root = content_hash(entry.domain_or_null or "", value)
            response = {"ok": True, "op": "content_hash", "cbor_hex": canonical_cbor_encode(value).hex(), "digest_hex": root.hex()}
        else:
            values = [canonical_cbor_decode(item) for item in entry.preimage_cbor]
            root = rfc6962_root(values)
            response = {"ok": True, "op": "rfc6962_root", "leaf_count": len(values), "root_hex": root.hex()}
        entries.append(_entry_receipt(entry, response, entry.request()))
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "endpoint_id": "PYTHON_FORMAL_STATIC_GENERATOR_V1",
        "basis_commit": basis.basis_commit,
        "execution_mode": "COMMITTED_HOST_PYTHON_PREIMAGE_GENERATOR",
        "binary_sha256": basis.implementation_inputs["python_binary_sha256"].hex(),
        "container_image_ref_or_null": None,
        "network_mode_none": False,
        "pull_policy_never": False,
        "generator_performs_network_io": False,
        "entries": entries,
        "seed_key_signature_or_state_created": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()
    return receipt


def _rust_docker_command(
    *,
    image_ref: str,
    binary_path: Path,
    control_plane: LocalDockerControlPlaneV1 | None = None,
) -> tuple[str, ...]:
    prefix = (
        tuple(control_plane.command())
        if control_plane is not None
        else (DEFAULT_DOCKER_EXECUTABLE.as_posix(), f"--host={LOCAL_DOCKER_HOST}")
    )
    return (
        *prefix, "run", "--rm", "-i", "--pull=never", "--network=none", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges",
        f"--security-opt=seccomp={SECCOMP_PATH.resolve()}", "--user=65534:65534",
        "--pids-limit=64", "--memory=512m", "--memory-swap=512m", "--ulimit=nofile=64:64",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m,uid=65534,gid=65534,mode=0700",
        f"--mount=type=bind,src={binary_path.resolve()},dst=/input/formal_bridge,readonly",
        "--entrypoint", "/input/formal_bridge", image_ref,
    )


def _docker_rust_call(
    *, image_ref: str, binary_path: Path, request: Mapping[str, object],
    control_plane: LocalDockerControlPlaneV1 | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if not binary_path.is_file():
        _fail(FAIL_RUST_REPLAY_POLICY, "Rust formal bridge binary is missing")
    command = _rust_docker_command(
        image_ref=image_ref,
        binary_path=binary_path,
        control_plane=control_plane,
    )
    if "--pull=never" not in command or "--network=none" not in command:
        raise AssertionError("offline Docker policy drift")
    completed = subprocess.run(
        command,
        input=_canonical_json_bytes(request),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env=(None if control_plane is None else dict(control_plane.environment)),
    )
    if completed.returncode != 0 or completed.stderr:
        _fail(
            FAIL_RUST_REPLAY,
            f"Rust endpoint failed rc={completed.returncode}; "
            f"stdout={completed.stdout.decode('utf-8', 'replace')[-1000:]}; "
            f"stderr={completed.stderr.decode('utf-8', 'replace')[-1000:]}",
        )
    try:
        response = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(FAIL_RUST_REPLAY, f"Rust endpoint returned invalid JSON: {exc}")
    if type(response) is not dict or response.get("ok") is not True:
        _fail(FAIL_RUST_REPLAY, "Rust endpoint did not return ok=true")
    execution = {
        "normalized_command": list(command),
        "exit_code": completed.returncode,
        "stdout_hex": completed.stdout.hex(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_hex": completed.stderr.hex(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }
    return response, execution


def run_rust_static_replay_receipt_v1(
    basis: FormalStaticBasisV1,
    *,
    control_plane: LocalDockerControlPlaneV1,
    daemon_receipt_binding: bytes,
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> dict[str, object]:
    if type(control_plane) is not LocalDockerControlPlaneV1:
        _fail(FAIL_RUST_REPLAY_POLICY, "Rust replay requires the sealed local Docker control plane")
    if type(daemon_receipt_binding) is not bytes or len(daemon_receipt_binding) != 32:
        _fail(FAIL_RUST_REPLAY_POLICY, "Rust replay daemon receipt binding is invalid")
    bound_binary = Path(str(basis.implementation_inputs["rust_binary_path"]))
    if rust_binary.resolve() != bound_binary.resolve():
        _fail(FAIL_RUST_REPLAY_POLICY, "Rust binary path differs from implementation binding")
    expected_digest = basis.implementation_inputs["rust_binary_sha256"]
    actual_digest = hashlib.sha256(rust_binary.read_bytes()).digest()
    if actual_digest != expected_digest:
        _fail(FAIL_RUST_REPLAY_POLICY, "Rust binary differs from implementation binding")
    image_ref = basis.implementation_inputs["rust_image_ref"]
    if type(image_ref) is not str:
        _fail(FAIL_RUST_REPLAY_POLICY, "Rust image reference is invalid")
    seccomp_path = Path(str(basis.implementation_inputs["seccomp_path"]))
    try:
        seccomp_digest = hashlib.sha256(seccomp_path.read_bytes()).digest()
    except OSError as exc:
        _fail(FAIL_RUST_REPLAY_POLICY, f"cannot read committed seccomp profile: {exc}")
    if seccomp_digest != basis.implementation_inputs["seccomp_sha256"]:
        _fail(FAIL_RUST_REPLAY_POLICY, "seccomp profile differs from committed basis")
    entries: list[dict[str, object]] = []
    executions: list[dict[str, object]] = []
    for entry in basis.gate19_plan:
        request = entry.request()
        response, execution = _docker_rust_call(
            image_ref=image_ref,
            binary_path=rust_binary,
            request=request,
            control_plane=control_plane,
        )
        entries.append(_entry_receipt(entry, response, request))
        executions.append(execution)
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "endpoint_id": "RUST_FORMAL_BRIDGE_M25_OFFLINE_CONTAINER_V1",
        "basis_commit": basis.basis_commit,
        "execution_mode": "DIGEST_PINNED_OCI_PUBLIC_STDIN_REPLAY",
        "binary_sha256": actual_digest.hex(),
        "container_image_ref_or_null": image_ref,
        "network_mode_none": True,
        "pull_policy_never": True,
        "docker_control_plane_binding": dict(control_plane.binding),
        "docker_daemon_receipt_binding_hex": daemon_receipt_binding.hex(),
        "entries": entries,
        "executions": executions,
        "seed_key_signature_or_state_created": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()
    return receipt


def _validate_receipt(
    basis: FormalStaticBasisV1, receipt: Mapping[str, object], endpoint_id: str
) -> dict[str, bytes]:
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("endpoint_id") != endpoint_id:
        _fail(FAIL_DUAL_RECEIPT, "receipt schema or endpoint ID differs")
    if receipt.get("basis_commit") != basis.basis_commit:
        _fail(FAIL_DUAL_RECEIPT, "receipt binds another commit")
    if receipt.get("seed_key_signature_or_state_created") is not False:
        _fail(FAIL_DUAL_RECEIPT, "static replay reports a forbidden side effect")
    is_python = endpoint_id == "PYTHON_FORMAL_STATIC_GENERATOR_V1"
    if is_python:
        if (
            receipt.get("execution_mode")
            != "COMMITTED_HOST_PYTHON_PREIMAGE_GENERATOR"
            or receipt.get("binary_sha256")
            != basis.implementation_inputs["python_binary_sha256"].hex()
            or receipt.get("container_image_ref_or_null") is not None
            or receipt.get("network_mode_none") is not False
            or receipt.get("pull_policy_never") is not False
            or receipt.get("generator_performs_network_io") is not False
            or "executions" in receipt
        ):
            _fail(FAIL_DUAL_RECEIPT, "Python generator execution binding differs")
    else:
        control_binding = receipt.get("docker_control_plane_binding")
        daemon_binding_hex = receipt.get("docker_daemon_receipt_binding_hex")
        if (
            receipt.get("execution_mode")
            != "DIGEST_PINNED_OCI_PUBLIC_STDIN_REPLAY"
            or receipt.get("binary_sha256")
            != basis.implementation_inputs["rust_binary_sha256"].hex()
            or receipt.get("container_image_ref_or_null")
            != basis.implementation_inputs["rust_image_ref"]
            or receipt.get("network_mode_none") is not True
            or receipt.get("pull_policy_never") is not True
            or type(control_binding) is not dict
            or control_binding.get("docker_executable")
            != DEFAULT_DOCKER_EXECUTABLE.as_posix()
            or control_binding.get("docker_host") != LOCAL_DOCKER_HOST
            or control_binding.get("network_endpoint_kind") != "LOCAL_UNIX_SOCKET"
            or control_binding.get("proxy_environment_keys") != []
            or type(control_binding.get("environment_keys")) is not list
            or set(control_binding.get("environment_keys", ()))
            != {"DOCKER_CONFIG", "DOCKER_HOST", "HOME", "LANG", "LC_ALL", "PATH"}
            or type(daemon_binding_hex) is not str
            or re.fullmatch(r"[0-9a-f]{64}", daemon_binding_hex) is None
        ):
            _fail(FAIL_DUAL_RECEIPT, "Rust replay execution binding differs")
    expected_receipt = dict(receipt)
    digest = expected_receipt.pop("receipt_sha256", None)
    if digest != hashlib.sha256(_canonical_json_bytes(expected_receipt)).hexdigest():
        _fail(FAIL_DUAL_RECEIPT, "receipt digest differs")
    entries = receipt.get("entries")
    if not isinstance(entries, list) or len(entries) != len(basis.gate19_plan):
        _fail(FAIL_DUAL_RECEIPT, "receipt entry set differs")
    executions = receipt.get("executions")
    if not is_python and (
        not isinstance(executions, list)
        or len(executions) != len(basis.gate19_plan)
    ):
        _fail(FAIL_DUAL_RECEIPT, "Rust execution evidence set differs")
    roots: dict[str, bytes] = {}
    for index, (expected, raw) in enumerate(
        zip(basis.gate19_plan, entries, strict=True)
    ):
        if not isinstance(raw, dict):
            _fail(FAIL_DUAL_RECEIPT, "receipt entry is not an object")
        if raw.get("root_name") != expected.root_name or raw.get("operation") != expected.operation or raw.get("schema_name") != expected.schema_name or raw.get("domain_or_null") != expected.domain_or_null:
            _fail(FAIL_DUAL_RECEIPT, f"receipt metadata differs for {expected.root_name}")
        hexes = raw.get("preimage_cbor_hex")
        if not isinstance(hexes, list):
            _fail(FAIL_DUAL_RECEIPT, "receipt preimage list is missing")
        try:
            preimages = tuple(bytes.fromhex(item) for item in hexes)
        except (TypeError, ValueError):
            _fail(FAIL_DUAL_RECEIPT, "receipt preimage hex is invalid")
        if preimages != expected.preimage_cbor:
            _fail(FAIL_DUAL_RECEIPT, f"receipt preimage differs for {expected.root_name}")
        request = expected.request()
        if raw.get("request_sha256") != hashlib.sha256(_canonical_json_bytes(request)).hexdigest():
            _fail(FAIL_DUAL_RECEIPT, f"request digest differs for {expected.root_name}")
        response = raw.get("response")
        if not isinstance(response, dict) or response.get("ok") is not True:
            _fail(FAIL_DUAL_RECEIPT, f"endpoint response failed for {expected.root_name}")
        if not is_python:
            assert isinstance(executions, list)
            execution = executions[index]
            if not isinstance(execution, dict):
                _fail(FAIL_DUAL_RECEIPT, "Rust execution evidence is not an object")
            expected_command = list(
                _rust_docker_command(
                    image_ref=str(basis.implementation_inputs["rust_image_ref"]),
                    binary_path=Path(str(basis.implementation_inputs["rust_binary_path"])),
                )
            )
            if execution.get("normalized_command") != expected_command:
                _fail(FAIL_DUAL_RECEIPT, "Rust Docker command differs from policy")
            try:
                stdout = bytes.fromhex(execution.get("stdout_hex"))
                stderr = bytes.fromhex(execution.get("stderr_hex"))
            except (TypeError, ValueError):
                _fail(FAIL_DUAL_RECEIPT, "Rust execution byte evidence is invalid")
            if (
                execution.get("exit_code") != 0
                or stderr != b""
                or execution.get("stdout_sha256") != hashlib.sha256(stdout).hexdigest()
                or execution.get("stderr_sha256") != hashlib.sha256(stderr).hexdigest()
            ):
                _fail(FAIL_DUAL_RECEIPT, "Rust execution byte evidence differs")
            try:
                stdout_response = json.loads(stdout)
            except (UnicodeDecodeError, json.JSONDecodeError):
                _fail(FAIL_DUAL_RECEIPT, "Rust stdout is not valid JSON")
            if stdout_response != response:
                _fail(FAIL_DUAL_RECEIPT, "Rust stdout and response object differ")
        if expected.operation == "content_hash":
            value = canonical_cbor_decode(preimages[0])
            rebuilt = content_hash(expected.domain_or_null or "", value)
            if response.get("cbor_hex") != preimages[0].hex() or response.get("digest_hex") != rebuilt.hex():
                _fail(FAIL_DUAL_RECEIPT, f"content replay differs for {expected.root_name}")
        else:
            values = [canonical_cbor_decode(item) for item in preimages]
            rebuilt = rfc6962_root(values)
            if response.get("leaf_count") != len(values) or response.get("root_hex") != rebuilt.hex():
                _fail(FAIL_DUAL_RECEIPT, f"record replay differs for {expected.root_name}")
        if rebuilt != expected.expected_root or raw.get("expected_root_hex") != rebuilt.hex():
            _fail(FAIL_DUAL_RECEIPT, f"expected root differs for {expected.root_name}")
        roots[expected.root_name] = rebuilt
    return roots


def validate_dual_static_replay_receipts_v1(
    basis: FormalStaticBasisV1,
    python_receipt: Mapping[str, object],
    rust_receipt: Mapping[str, object],
) -> Mapping[str, bytes]:
    """Validate both exact preimage receipts and return Gate-19 roots."""

    python_roots = _validate_receipt(basis, python_receipt, "PYTHON_FORMAL_STATIC_GENERATOR_V1")
    rust_roots = _validate_receipt(basis, rust_receipt, "RUST_FORMAL_BRIDGE_M25_OFFLINE_CONTAINER_V1")
    if python_roots != rust_roots or tuple(python_roots) != GATE19_ROOT_NAMES:
        _fail(FAIL_DUAL_RECEIPT, "Python/Rust root maps differ")
    return MappingProxyType(python_roots)


def replay_content_hash_cbor_with_rust_container_v1(
    *, schema_name: str, cbor_bytes: bytes, rust_binary: Path = DEFAULT_RUST_BINARY
) -> dict[str, object]:
    """Strict-decode then hash one public formal object with the Rust endpoint.

    This is the purpose-3 bridge helper.  The caller must additionally use
    ``decode_formal_object`` to enforce the named schema's exact field guards.
    """

    schema = FORMAL_SCHEMA_REGISTRY.get(schema_name)
    if schema is None or schema.hash_domain is None:
        _fail(FAIL_RUST_REPLAY_POLICY, "schema has no frozen ContentHash domain")
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    image_ref = profile["images"]["rust_attester"]
    decoded, decode_execution = _docker_rust_call(
        image_ref=image_ref,
        binary_path=rust_binary,
        request={"op": "decode", "cbor_hex": cbor_bytes.hex()},
    )
    if decoded.get("canonical_cbor_hex") != cbor_bytes.hex():
        _fail(FAIL_RUST_REPLAY, "Rust strict decode did not preserve canonical bytes")
    hashed, hash_execution = _docker_rust_call(
        image_ref=image_ref,
        binary_path=rust_binary,
        request={"op": "content_hash", "domain": schema.hash_domain, "value": decoded["value"]},
    )
    if hashed.get("cbor_hex") != cbor_bytes.hex():
        _fail(FAIL_RUST_REPLAY, "Rust ContentHash re-encoding differs")
    return {
        "schema": "hegel-phase3-m25-rust-content-replay/1",
        "schema_name": schema_name,
        "object_tag": schema.tag,
        "hash_domain": schema.hash_domain,
        "cbor_sha256": hashlib.sha256(cbor_bytes).hexdigest(),
        "root_hex": hashed["digest_hex"],
        "binary_sha256": hashlib.sha256(rust_binary.read_bytes()).hexdigest(),
        "container_image_ref": image_ref,
        "network_mode_none": True,
        "pull_policy_never": True,
        "executions": [decode_execution, hash_execution],
    }


__all__ = [
    "CHILD_DSL_ID",
    "CHILD_FREEZE_ID",
    "DEFAULT_RUST_BINARY",
    "FAIL_DUAL_RECEIPT",
    "FormalStaticBasisError",
    "FormalStaticBasisV1",
    "GATE19_ROOT_NAMES",
    "RootReplayPlanEntryV1",
    "build_formal_static_basis_v1",
    "build_identifier_registry_rows_v1",
    "build_operator_semantics_rows_v1",
    "build_python_static_replay_receipt_v1",
    "replay_content_hash_cbor_with_rust_container_v1",
    "run_rust_static_replay_receipt_v1",
    "validate_dual_static_replay_receipts_v1",
]
