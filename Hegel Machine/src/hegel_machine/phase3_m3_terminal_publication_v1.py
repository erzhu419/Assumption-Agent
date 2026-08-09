"""Static, fail-closed validator for the formal M3 terminal publication.

This module validates the exact archived start state, exact-once start receipt,
and public terminal outcome.  It does not import the M3 executor, open the
canonical run directory, inspect Docker state, or read seed, key,
split-assignment, or target-role inputs.  The accepted claim is correspondingly
narrow: the frozen child DSL reached the 50,000-program bound and produced the
50,001st witness, so its bounded result is ``DSL_TOO_LARGE``.  This is not
COMPLETE closure, an outside-language certificate, Phase-3 exit, or ACTIVE
promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Final, Mapping, NoReturn


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
ARTIFACT_REPOSITORY_PATH: Final = (
    "artifacts/phase3_m3_runtime/formal_m3_terminal_outcome_v1.json"
)
ARTIFACT_PATH: Final = PROJECT_ROOT / ARTIFACT_REPOSITORY_PATH
START_STATE_REPOSITORY_PATH: Final = (
    "artifacts/phase3_m3_runtime/formal_m3_start_state_v1.json"
)
START_PUBLICATION_RECEIPT_REPOSITORY_PATH: Final = (
    "artifacts/phase3_m3_runtime/formal_m3_start_publication_receipt_v1.json"
)
START_STATE_PATH: Final = PROJECT_ROOT / START_STATE_REPOSITORY_PATH
START_PUBLICATION_RECEIPT_PATH: Final = (
    PROJECT_ROOT / START_PUBLICATION_RECEIPT_REPOSITORY_PATH
)

EXPECTED_START_STATE_BYTE_LENGTH: Final = 1_525
EXPECTED_START_STATE_FILE_SHA256: Final = (
    "9f07564d4f859e082288ddf971c336a03b490062c65bce7eb81ddcfa64ea4053"
)
EXPECTED_START_RECEIPT_BYTE_LENGTH: Final = 26_879
EXPECTED_START_RECEIPT_FILE_SHA256: Final = (
    "dede9fb1bf1febe4ec6646f00be456c94ff181fa91e23a23d8392c7596a70df3"
)

EXPECTED_ARTIFACT_BYTE_LENGTH: Final = 62_942
EXPECTED_ARTIFACT_FILE_SHA256: Final = (
    "4f631224383297f6f30d70dbcefc15ed1c1296ba634a604e5a59562d11e67aed"
)
EXPECTED_OUTCOME_SELF_SHA256: Final = (
    "973214b278e0bd3af474fa0b095e518e1ea8323917845b856e8b5de72913c67c"
)
EXPECTED_SCHEMA: Final = "hegel-phase3-m3-formal-enumeration-outcome/1"
EXPECTED_ARTIFACT_KIND: Final = "FORMAL_M3_DUAL_ENUMERATION_TERMINAL"
EXPECTED_ATTEMPT_ID: Final = "attempt-1"
EXPECTED_BASIS_COMMIT_A: Final = "0af65964235390ce2bebefea7379eaa9c50eda24"
EXPECTED_PUBLICATION_COMMIT_B: Final = (
    "78d5c77994ad9088c082c32a948b5a2b40407966"
)
EXPECTED_RUNTIME_COMMIT_C: Final = "7636aba6e07f565f673e2f3cdf39a1c5dc143d9e"
EXPECTED_ADMISSION_COMMIT_D: Final = "1a434ba1236ae6481ba1cb93f85b1d8886d37243"
EXPECTED_RUN_ID_HEX: Final = "e4af9f57c38fb298462ec628c4ed8a03"
EXPECTED_EXECUTION_MANIFEST_ROOT_HEX: Final = (
    "fd84e901e2259943ebf981eeaee8d6dd807c6ca82ae0f89315c57a4808659453"
)
EXPECTED_START_STATE_RECORD_ROOT_HEX: Final = (
    "daa8341296a6fc075346a0bb6df95667eb726dd119f212437c2b2e645e0d91e0"
)
EXPECTED_START_STATE_ARTIFACT_SHA256: Final = (
    "0dc16c47a67fc8fc3a4e2be2af87c4ff2876063e79be7f617dd4c26407cdbdf0"
)
EXPECTED_CLOSURE_STATUS: Final = "DSL_TOO_LARGE"
EXPECTED_CANONICAL_PROGRAM_COUNT: Final = 50_000
EXPECTED_RAW_OPERATOR_APPLICATION_COUNT: Final = 3_292_439
EXPECTED_FIRST_OUT_OF_BUDGET_WITNESS_HEX: Final = (
    "96200a6a131204315ffcd1efd0aa2dcfe2ce665a2c06516461772c9812f0ec71"
)
NEXT_ACTION: Final = (
    "SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1"
)

EXPECTED_FORMAL_OBJECTS: Final = {
    "python_enumeration_receipt": (
        "M3ImplementationEnumerationReceiptV1",
        "c385fded7980e146c8a3090b6adb061e6774d69f6d0b705c932a4e22bda30752",
    ),
    "rust_enumeration_receipt": (
        "M3ImplementationEnumerationReceiptV1",
        "583d77bf2596c0f1ed31dd6de00b5fa36036576f58517edfa6b366be7011c1d1",
    ),
    "dual_replay_agreement": (
        "M3DualReplayAgreementV1",
        "48454aef57c3b560ee2f05e46b1dae1f4cd3e9fd0de52b392083a7c8b2359d83",
    ),
    "terminal_state_record": (
        "M3RunStateRecordV1",
        "1f54925f8187c955ae4e7b2c9bb83144ce6627468153d8e1696b27c4705d23dd",
    ),
}

CORE_ARCHIVE_ROWS: Final = {
    "archive/bucket_accounting_records.cborframed": (
        9_872,
        "03e039144d1d3e88054e85b83141c4764916cdafe4583d4e4f9806f0c03deb62",
    ),
    "archive/canonical_program_records.cborframed": (
        10_913_073,
        "8ffdcf1e64d1d1934404c6fc98f8340662bd9194852a2b558d84e7a533d32e21",
    ),
    "archive/program_chunk_manifests.cborframed": (
        1_610,
        "571b7eddeba06e555b35de9be8ed808b7369da2ff6b898ff25bbb6fbbc56885c",
    ),
}

EXPECTED_ARCHIVE_PATHS: Final = frozenset(
    {
        *CORE_ARCHIVE_ROWS,
        "archive/report.json",
        "execution-stderr.bin",
        "execution-stdout.json",
        "process-completion.json",
    }
)

EXPECTED_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "archive_files",
        "artifact_kind",
        "attempt_id",
        "attempt_root_path_sha256",
        "basis_commit_a",
        "canonical_program_count",
        "closure_status",
        "contains_private_key",
        "contains_raw_split_seed",
        "enumerator_container_split_inputs_accessed",
        "enumerator_container_target_inputs_accessed",
        "execution_manifest_root_hex",
        "first_out_of_budget_program_hash_hex_or_null",
        "formal_objects",
        "host_frozen_public_target_definitions_loaded",
        "host_public_split_commitments_and_roots_replayed",
        "local_admission_artifact",
        "local_admission_receipt",
        "offline_runtime_preflight",
        "outcome_artifact_sha256",
        "publication_commit_b",
        "raw_operator_application_count",
        "raw_split_seed_accessed",
        "role_evaluation_started",
        "run_id_hex",
        "runner_evidence_files",
        "runtime_source_manifest",
        "schema",
        "split_assignment_rows_accessed",
        "start_publication_receipt",
        "start_state_artifact_sha256",
        "start_state_record_root_hex",
    }
)

_LOWER_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_MAX_ARTIFACT_BYTES: Final = 1_048_576

FAIL_FILE: Final = "FAIL_M3_TERMINAL_PUBLICATION_FILE"
FAIL_JSON: Final = "FAIL_M3_TERMINAL_PUBLICATION_JSON"
FAIL_CANONICAL: Final = "FAIL_M3_TERMINAL_PUBLICATION_CANONICAL_JSON"
FAIL_FILE_HASH: Final = "FAIL_M3_TERMINAL_PUBLICATION_FILE_HASH"
FAIL_SELF_HASH: Final = "FAIL_M3_TERMINAL_PUBLICATION_SELF_HASH"
FAIL_IDENTITY: Final = "FAIL_M3_TERMINAL_PUBLICATION_IDENTITY"
FAIL_ACCESS: Final = "FAIL_M3_TERMINAL_PUBLICATION_ACCESS_BOUNDARY"
FAIL_ARCHIVE: Final = "FAIL_M3_TERMINAL_PUBLICATION_ARCHIVE"
FAIL_FORMAL_ROOT: Final = "FAIL_M3_TERMINAL_PUBLICATION_FORMAL_ROOT"


class M3TerminalPublicationError(RuntimeError):
    """Stable fail-closed error for the static terminal publication boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3TerminalPublicationError(code, detail)


class _Pairs(tuple):
    pass


def _canonical_json_v1(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        _fail(FAIL_JSON, f"publication is not a finite JSON value: {type(exc).__name__}")


def _strict_json_loads_v1(payload: bytes) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> _Pairs:
        keys = [key for key, _value in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate object key")
        return _Pairs(pairs)

    def reject_number(token: str) -> NoReturn:
        raise ValueError(f"unsupported JSON number {token}")

    try:
        decoded = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=pairs_hook,
            parse_float=reject_number,
            parse_constant=reject_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        _fail(FAIL_JSON, f"artifact is not strict duplicate-free JSON: {type(exc).__name__}")

    def plain(value: object) -> object:
        if isinstance(value, _Pairs):
            return {key: plain(item) for key, item in value}
        if type(value) is list:
            return [plain(item) for item in value]
        if value is None or type(value) in {bool, int, str}:
            return value
        _fail(FAIL_JSON, "artifact contains a non-JSON value")

    result = plain(decoded)
    if type(result) is not dict:
        _fail(FAIL_JSON, "artifact must be a JSON object")
    return result


def _read_regular_file_v1(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        lexical = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino)
            or before.st_size > _MAX_ARTIFACT_BYTES
        ):
            _fail(FAIL_FILE, "artifact is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                _fail(FAIL_FILE, "artifact ended before its recorded size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_FILE, "artifact grew while being read")
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        ):
            _fail(FAIL_FILE, "artifact identity changed while being read")
        return b"".join(chunks)
    except M3TerminalPublicationError:
        raise
    except OSError as exc:
        _fail(FAIL_FILE, f"artifact cannot be read: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_dict(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(FAIL_IDENTITY, f"{label} must be a JSON object")
    return value


def _validate_self_hash_v1(document: dict[str, object]) -> None:
    claimed = document.get("outcome_artifact_sha256")
    if type(claimed) is not str or _LOWER_SHA256.fullmatch(claimed) is None:
        _fail(FAIL_SELF_HASH, "outcome self-hash is malformed")
    body = dict(document)
    body.pop("outcome_artifact_sha256", None)
    observed = hashlib.sha256(_canonical_json_v1(body)).hexdigest()
    if observed != claimed:
        _fail(FAIL_SELF_HASH, "outcome self-hash replay differs")


def _validate_fixed_identity_v1(document: dict[str, object]) -> None:
    expected: Mapping[str, object] = {
        "schema": EXPECTED_SCHEMA,
        "artifact_kind": EXPECTED_ARTIFACT_KIND,
        "attempt_id": EXPECTED_ATTEMPT_ID,
        "basis_commit_a": EXPECTED_BASIS_COMMIT_A,
        "publication_commit_b": EXPECTED_PUBLICATION_COMMIT_B,
        "run_id_hex": EXPECTED_RUN_ID_HEX,
        "execution_manifest_root_hex": EXPECTED_EXECUTION_MANIFEST_ROOT_HEX,
        "start_state_record_root_hex": EXPECTED_START_STATE_RECORD_ROOT_HEX,
        "start_state_artifact_sha256": EXPECTED_START_STATE_ARTIFACT_SHA256,
        "closure_status": EXPECTED_CLOSURE_STATUS,
        "canonical_program_count": EXPECTED_CANONICAL_PROGRAM_COUNT,
        "raw_operator_application_count": EXPECTED_RAW_OPERATOR_APPLICATION_COUNT,
        "first_out_of_budget_program_hash_hex_or_null": (
            EXPECTED_FIRST_OUT_OF_BUDGET_WITNESS_HEX
        ),
    }
    if set(document) != EXPECTED_TOP_LEVEL_FIELDS:
        _fail(FAIL_IDENTITY, "terminal outcome top-level field set differs")
    for field, value in expected.items():
        if document.get(field) != value or type(document.get(field)) is not type(value):
            _fail(FAIL_IDENTITY, f"terminal outcome identity differs: {field}")

    admission = _require_dict(
        document.get("local_admission_artifact"), label="local admission artifact"
    )
    receipt = _require_dict(
        document.get("local_admission_receipt"), label="local admission receipt"
    )
    if (
        admission.get("basis_commit_a") != EXPECTED_BASIS_COMMIT_A
        or admission.get("publication_commit_b") != EXPECTED_PUBLICATION_COMMIT_B
        or admission.get("runtime_commit_c") != EXPECTED_RUNTIME_COMMIT_C
        or admission.get("runtime_parent_commit_b") != EXPECTED_PUBLICATION_COMMIT_B
        or admission.get("formal_run_id_hex") != EXPECTED_RUN_ID_HEX
        or admission.get("execution_manifest_root_hex")
        != EXPECTED_EXECUTION_MANIFEST_ROOT_HEX
        or receipt.get("runtime_commit_c") != EXPECTED_RUNTIME_COMMIT_C
        or receipt.get("approval_commit_d") != EXPECTED_ADMISSION_COMMIT_D
        or receipt.get("formal_run_id_hex") != EXPECTED_RUN_ID_HEX
        or receipt.get("execution_manifest_root_hex")
        != EXPECTED_EXECUTION_MANIFEST_ROOT_HEX
    ):
        _fail(FAIL_IDENTITY, "A/B/C/D admission chain or run binding differs")


def _validate_access_boundary_v1(document: dict[str, object]) -> None:
    false_fields = (
        "role_evaluation_started",
        "enumerator_container_split_inputs_accessed",
        "raw_split_seed_accessed",
        "split_assignment_rows_accessed",
        "enumerator_container_target_inputs_accessed",
        "contains_private_key",
        "contains_raw_split_seed",
    )
    if any(document.get(field) is not False for field in false_fields):
        _fail(FAIL_ACCESS, "seed/key/split/target-role access boundary differs")
    if (
        document.get("host_public_split_commitments_and_roots_replayed") is not True
        or document.get("host_frozen_public_target_definitions_loaded") is not True
    ):
        _fail(FAIL_ACCESS, "public host-only commitment replay boundary differs")

    admission = _require_dict(
        document.get("local_admission_artifact"), label="local admission artifact"
    )
    if (
        admission.get("role_evaluation_allowed") is not False
        or admission.get("active_promotion_allowed") is not False
        or admission.get("external_actor_attestation") is not False
        or admission.get("external_signatures") != []
        or admission.get("network_fetch_allowed") is not False
        or admission.get("docker_pull_allowed") is not False
    ):
        _fail(FAIL_ACCESS, "local admission access or promotion boundary differs")
    preflight = _require_dict(
        document.get("offline_runtime_preflight"), label="offline runtime preflight"
    )
    if preflight.get("network_mode") != "none":
        _fail(FAIL_ACCESS, "enumeration runtime was not network-isolated")


def _validate_formal_roots_v1(document: dict[str, object]) -> dict[str, str]:
    formal = _require_dict(document.get("formal_objects"), label="formal objects")
    if set(formal) != set(EXPECTED_FORMAL_OBJECTS):
        _fail(FAIL_FORMAL_ROOT, "formal object set differs")
    roots: dict[str, str] = {}
    for name, (schema_name, root_hex) in EXPECTED_FORMAL_OBJECTS.items():
        row = _require_dict(formal.get(name), label=f"formal object {name}")
        cbor_hex = row.get("cbor_hex")
        if (
            set(row) != {"schema_name", "cbor_hex", "content_root_hex"}
            or row.get("schema_name") != schema_name
            or row.get("content_root_hex") != root_hex
            or type(cbor_hex) is not str
        ):
            _fail(FAIL_FORMAL_ROOT, f"formal object identity differs: {name}")
        try:
            cbor = bytes.fromhex(cbor_hex)
        except ValueError:
            _fail(FAIL_FORMAL_ROOT, f"formal object CBOR hex is malformed: {name}")
        if not cbor or cbor.hex() != cbor_hex:
            _fail(FAIL_FORMAL_ROOT, f"formal object CBOR hex is noncanonical: {name}")
        roots[name] = root_hex
    return roots


def _archive_rows_by_path_v1(value: object, *, implementation: str) -> dict[str, tuple[int, str]]:
    if type(value) is not list or len(value) != len(EXPECTED_ARCHIVE_PATHS):
        _fail(FAIL_ARCHIVE, f"{implementation} archive row count differs")
    rows: dict[str, tuple[int, str]] = {}
    observed_order: list[str] = []
    for item in value:
        row = _require_dict(item, label=f"{implementation} archive row")
        path = row.get("relative_path")
        byte_length = row.get("byte_length")
        digest = row.get("sha256")
        if (
            set(row) != {"relative_path", "byte_length", "sha256"}
            or type(path) is not str
            or type(byte_length) is not int
            or byte_length < 0
            or type(digest) is not str
            or _LOWER_SHA256.fullmatch(digest) is None
            or path in rows
        ):
            _fail(FAIL_ARCHIVE, f"{implementation} archive row differs")
        observed_order.append(path)
        rows[path] = (byte_length, digest)
    if set(rows) != EXPECTED_ARCHIVE_PATHS or observed_order != sorted(rows):
        _fail(FAIL_ARCHIVE, f"{implementation} archive path set/order differs")
    return rows


def _validate_core_archives_v1(document: dict[str, object]) -> dict[str, object]:
    archives = _require_dict(document.get("archive_files"), label="archive files")
    if set(archives) != {"python", "rust"}:
        _fail(FAIL_ARCHIVE, "archive implementation set differs")
    python_rows = _archive_rows_by_path_v1(archives["python"], implementation="python")
    rust_rows = _archive_rows_by_path_v1(archives["rust"], implementation="rust")
    core_summary: dict[str, object] = {}
    for path, expected in CORE_ARCHIVE_ROWS.items():
        if python_rows.get(path) != expected or rust_rows.get(path) != expected:
            _fail(FAIL_ARCHIVE, f"dual core archive identity differs: {path}")
        core_summary[path] = {
            "byte_length": expected[0],
            "sha256": expected[1],
            "python_rust_equal": True,
        }
    return core_summary


def build_publication_summary(document: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact frozen mapping and return its narrow claim summary."""

    if type(document) is not dict:
        _fail(FAIL_JSON, "terminal publication mapping must be a plain JSON object")
    mutable = document
    _validate_self_hash_v1(mutable)
    _validate_fixed_identity_v1(mutable)
    _validate_access_boundary_v1(mutable)
    formal_roots = _validate_formal_roots_v1(mutable)
    core_archives = _validate_core_archives_v1(mutable)

    canonical_payload = _canonical_json_v1(mutable)
    if (
        len(canonical_payload) != EXPECTED_ARTIFACT_BYTE_LENGTH
        or hashlib.sha256(canonical_payload).hexdigest()
        != EXPECTED_ARTIFACT_FILE_SHA256
    ):
        _fail(FAIL_FILE_HASH, "terminal publication exact mapping hash differs")

    return {
        "schema": "hegel-phase3-m3-terminal-publication-summary/1",
        "artifact_repository_path": ARTIFACT_REPOSITORY_PATH,
        "artifact_file_byte_length": EXPECTED_ARTIFACT_BYTE_LENGTH,
        "artifact_file_sha256": EXPECTED_ARTIFACT_FILE_SHA256,
        "outcome_self_sha256": EXPECTED_OUTCOME_SELF_SHA256,
        "run_id_hex": EXPECTED_RUN_ID_HEX,
        "execution_manifest_root_hex": EXPECTED_EXECUTION_MANIFEST_ROOT_HEX,
        "closure_status": EXPECTED_CLOSURE_STATUS,
        "canonical_program_count": EXPECTED_CANONICAL_PROGRAM_COUNT,
        "raw_operator_application_count": EXPECTED_RAW_OPERATOR_APPLICATION_COUNT,
        "first_out_of_budget_program_hash_hex": (
            EXPECTED_FIRST_OUT_OF_BUDGET_WITNESS_HEX
        ),
        "formal_object_roots": formal_roots,
        "dual_core_archives": core_archives,
        "access_boundary": {
            "role_evaluation_started": False,
            "enumerator_split_inputs_accessed": False,
            "raw_split_seed_accessed": False,
            "split_assignment_rows_accessed": False,
            "enumerator_target_role_inputs_accessed": False,
            "contains_private_key": False,
            "contains_raw_split_seed": False,
        },
        "claim_boundary": {
            "bounded_child_dsl_dsl_too_large": True,
            "complete_closure": False,
            "outside_frozen_closure_certificate": False,
            "phase3_exit": False,
            "active_promotion": False,
        },
        "next_action": NEXT_ACTION,
    }


def validate(path: Path = ARTIFACT_PATH) -> dict[str, object]:
    """Read and validate the exact canonical terminal publication bytes."""

    payload = _read_regular_file_v1(Path(path))
    document = _strict_json_loads_v1(payload)
    if _canonical_json_v1(document) != payload:
        _fail(FAIL_CANONICAL, "artifact is not compact sorted JSON plus one newline")
    if (
        len(payload) != EXPECTED_ARTIFACT_BYTE_LENGTH
        or hashlib.sha256(payload).hexdigest() != EXPECTED_ARTIFACT_FILE_SHA256
    ):
        _fail(FAIL_FILE_HASH, "terminal publication file bytes differ")
    return build_publication_summary(document)


def _read_exact_canonical_mapping_v1(
    path: Path,
    *,
    expected_byte_length: int,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    payload = _read_regular_file_v1(path)
    document = _strict_json_loads_v1(payload)
    if _canonical_json_v1(document) != payload:
        _fail(FAIL_CANONICAL, f"{label} is not compact sorted JSON plus one newline")
    if (
        len(payload) != expected_byte_length
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        _fail(FAIL_FILE_HASH, f"{label} exact file bytes differ")
    return document


def validate_publication_carrier_v1(
    artifact_directory: Path = ARTIFACT_PATH.parent,
) -> dict[str, object]:
    """Validate the three exact committed carrier files as one publication."""

    directory = Path(artifact_directory)
    start_path = directory / Path(START_STATE_REPOSITORY_PATH).name
    receipt_path = directory / Path(
        START_PUBLICATION_RECEIPT_REPOSITORY_PATH
    ).name
    terminal_path = directory / Path(ARTIFACT_REPOSITORY_PATH).name

    start = _read_exact_canonical_mapping_v1(
        start_path,
        expected_byte_length=EXPECTED_START_STATE_BYTE_LENGTH,
        expected_sha256=EXPECTED_START_STATE_FILE_SHA256,
        label="formal M3 start state",
    )
    receipt = _read_exact_canonical_mapping_v1(
        receipt_path,
        expected_byte_length=EXPECTED_START_RECEIPT_BYTE_LENGTH,
        expected_sha256=EXPECTED_START_RECEIPT_FILE_SHA256,
        label="formal M3 start publication receipt",
    )
    terminal = validate(terminal_path)

    expected_start: Mapping[str, object] = {
        "schema": "hegel-phase3-m3-start-state/1",
        "artifact_kind": "FORMAL_M3_START_STATE_RECORD",
        "action_id": "phase3-m3-start",
        "basis_commit": EXPECTED_BASIS_COMMIT_A,
        "publication_commit": EXPECTED_PUBLICATION_COMMIT_B,
        "run_id_hex": EXPECTED_RUN_ID_HEX,
        "execution_manifest_root_hex": EXPECTED_EXECUTION_MANIFEST_ROOT_HEX,
        "state_record_root_hex": EXPECTED_START_STATE_RECORD_ROOT_HEX,
        "state_artifact_sha256": EXPECTED_START_STATE_ARTIFACT_SHA256,
        "formal_gate_count": 24,
        "child_state_before": "NOT_RUN",
        "child_state_after": "RUNNING",
        "running_phase_after": "CANONICAL_ENUMERATION",
        "closure_invoked": False,
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "contains_split_assignment_rows": False,
    }
    expected_receipt: Mapping[str, object] = {
        "schema": "hegel-phase3-m3-start-publication-receipt/2",
        "artifact_kind": "FORMAL_M3_EXPLICIT_START_PUBLICATION_RECEIPT",
        "action_id": "phase3-m3-start",
        "basis_commit": EXPECTED_BASIS_COMMIT_A,
        "publication_commit": EXPECTED_PUBLICATION_COMMIT_B,
        "run_id_hex": EXPECTED_RUN_ID_HEX,
        "state_record_root_hex": EXPECTED_START_STATE_RECORD_ROOT_HEX,
        "state_artifact_sha256": EXPECTED_START_STATE_ARTIFACT_SHA256,
        "state_file_byte_length": EXPECTED_START_STATE_BYTE_LENGTH,
        "state_file_sha256": EXPECTED_START_STATE_FILE_SHA256,
        "state_relative_path": "m3-start-state.json",
        "exact_once_publication_profile": "openat-linkat-no-replace-v1",
        "prepared_output_redirect_accepted": False,
    }
    for field, expected in expected_start.items():
        if start.get(field) != expected or type(start.get(field)) is not type(expected):
            _fail(FAIL_IDENTITY, f"start carrier identity differs: {field}")
    for field, expected in expected_receipt.items():
        if receipt.get(field) != expected or type(receipt.get(field)) is not type(expected):
            _fail(FAIL_IDENTITY, f"start receipt carrier identity differs: {field}")

    return {
        **terminal,
        "schema": "hegel-phase3-m3-terminal-publication-carrier/1",
        "carrier_files": {
            START_STATE_REPOSITORY_PATH: {
                "byte_length": EXPECTED_START_STATE_BYTE_LENGTH,
                "sha256": EXPECTED_START_STATE_FILE_SHA256,
            },
            START_PUBLICATION_RECEIPT_REPOSITORY_PATH: {
                "byte_length": EXPECTED_START_RECEIPT_BYTE_LENGTH,
                "sha256": EXPECTED_START_RECEIPT_FILE_SHA256,
            },
            ARTIFACT_REPOSITORY_PATH: {
                "byte_length": EXPECTED_ARTIFACT_BYTE_LENGTH,
                "sha256": EXPECTED_ARTIFACT_FILE_SHA256,
            },
        },
        "start_state_record_root_hex": EXPECTED_START_STATE_RECORD_ROOT_HEX,
        "start_state_artifact_sha256": EXPECTED_START_STATE_ARTIFACT_SHA256,
    }


# Versioned aliases for callers that use the repository's prevailing naming style.
build_publication_summary_v1 = build_publication_summary
validate_terminal_publication_v1 = validate


__all__ = [
    "ARTIFACT_PATH",
    "ARTIFACT_REPOSITORY_PATH",
    "START_PUBLICATION_RECEIPT_PATH",
    "START_PUBLICATION_RECEIPT_REPOSITORY_PATH",
    "START_STATE_PATH",
    "START_STATE_REPOSITORY_PATH",
    "EXPECTED_ARTIFACT_FILE_SHA256",
    "EXPECTED_OUTCOME_SELF_SHA256",
    "M3TerminalPublicationError",
    "NEXT_ACTION",
    "build_publication_summary",
    "build_publication_summary_v1",
    "validate",
    "validate_publication_carrier_v1",
    "validate_terminal_publication_v1",
]
