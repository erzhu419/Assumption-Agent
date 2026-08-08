"""Exact-once formal M3 start boundary.

This module may create only the index-zero ``M3RunStateRecordV1``.  It
replays the committed Gate 15--24 public evidence, verifies the still-NOT_RUN
promotion and all bound execution identities, and never imports or invokes a
closure enumerator.  The default CLI operation lives in the sibling module
and is side-effect-free preparation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from . import phase3_container_actor_runtime_v1 as _actor_runtime
from . import phase3_m3_local_admission_v1 as _local_admission
from .phase3_m25_commit_b_publication_audit_v1 import (
    _validate_actor_report_public_only_v1,
    _validate_errata_report_public_only_v1,
    canonical_json_v1,
)
from .phase3_m25_container_ceremony_v1 import (
    GateEvidenceInputsV1,
    QualifiedGateEvidenceV1,
    _evaluate_gates_15_24_with_prevalidated_report_basis_v1,
    promote_gate_evidence_v1,
)
from .phase3_m25_external_v1 import assert_public_payload_contains_no_secret_fields
from .phase3_m25_formal_container_executor_v1 import load_gate_evidence_inputs_v1
from .phase3_m25_wire_v1 import (
    M3_RUN_OUTPUT_ROOTS,
    build_formal_object,
    candidate_content_root,
    decode_formal_object,
    encode_formal_object,
    validate_execution_identity_linkage_v1,
    validate_m3_output_roots_null,
    validate_opaque_id_registry_append_v1,
    validate_timestamp_ordering_v1,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
# This is a formal identity boundary, not a convenience path.  Deriving it
# from HOME/Path.home() would permit the same frozen run ID to acquire multiple
# independent "canonical" state files under different caller environments.
# The single host custody root is therefore explicit in the qualified source.
FORMAL_RUN_PARENT: Path = Path("/home/erzhu419/.local/state/hegel-machine")
EVIDENCE_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/formal_genesis_v2/"
    "phase3_m25_formal_gate_evidence_v1.json"
)
PROMOTION_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/formal_genesis_v2/"
    "phase3_m25_gate_promotion_v1.json"
)
STATE_SCHEMA: Final = "hegel-phase3-m3-start-state/1"
STATE_ARTIFACT_KIND: Final = "FORMAL_M3_START_STATE_RECORD"
START_PUBLICATION_RECEIPT_SCHEMA: Final = (
    "hegel-phase3-m3-start-publication-receipt/2"
)
START_PUBLICATION_RECEIPT_ARTIFACT_KIND: Final = (
    "FORMAL_M3_EXPLICIT_START_PUBLICATION_RECEIPT"
)
START_PUBLICATION_RECEIPT_FILENAME: Final = (
    "m3-start-publication-receipt.json"
)
ACTION_ID: Final = "phase3-m3-start"
MAX_PUBLIC_BLOB_BYTES: Final = 32 * 1024 * 1024
MAX_STATE_BYTES: Final = 4 * 1024 * 1024
PUBLICATION_COMMIT_B: Final = "78d5c77994ad9088c082c32a948b5a2b40407966"
FORMAL_RUN_ID_HEX: Final = "e4af9f57c38fb298462ec628c4ed8a03"

FAIL_JSON = "FAIL_M3_START_STRICT_JSON"
FAIL_PUBLICATION = "FAIL_M3_START_PUBLICATION_BINDING"
FAIL_GATE_REPLAY = "FAIL_M3_START_GATE_REPLAY"
FAIL_IDENTITY = "FAIL_M3_START_EXECUTION_IDENTITY"
FAIL_STATE_RECORD = "FAIL_M3_START_STATE_RECORD"
FAIL_ALREADY_STARTED = "FAIL_M3_START_ALREADY_STARTED"
FAIL_STATE_IO = "FAIL_M3_START_STATE_IO"

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class M3StartError(RuntimeError):
    """Stable fail-closed error at the formal M3 start boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3StartError(code, detail)


@dataclass(frozen=True, slots=True)
class ReplayedGatePublicationV1:
    """Typed result of one complete immutable Commit-B public replay."""

    publication_commit: str
    evidence: Mapping[str, object]
    promotion: Mapping[str, object]
    gate_inputs: GateEvidenceInputsV1
    qualified_gate_evidence: QualifiedGateEvidenceV1


_START_PREPARATION_SEAL = object()


@dataclass(frozen=True, slots=True)
class PreparedM3StartV1:
    """Capability produced only by a complete authoritative start replay."""

    document: Mapping[str, object]
    _seal: object


class _Pairs(tuple):
    pass


def strict_json_loads_v1(payload: bytes, *, label: str) -> dict[str, object]:
    """Decode one duplicate-free, finite, object-valued JSON document."""

    def pairs_hook(pairs: list[tuple[str, object]]) -> _Pairs:
        keys = [key for key, _value in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate object key")
        return _Pairs(pairs)

    def reject_number(token: str) -> NoReturn:
        raise ValueError(f"non-integer JSON number {token}")

    try:
        decoded = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=pairs_hook,
            parse_float=reject_number,
            parse_constant=reject_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        _fail(FAIL_JSON, f"{label} is not strict JSON: {type(exc).__name__}")

    def plain(value: object) -> object:
        if isinstance(value, _Pairs):
            return {key: plain(item) for key, item in value}
        if type(value) is list:
            return [plain(item) for item in value]
        if value is None or type(value) in {bool, int, str}:
            return value
        _fail(FAIL_JSON, f"{label} contains an unsupported JSON value")

    result = plain(decoded)
    if type(result) is not dict:
        _fail(FAIL_JSON, f"{label} must be a JSON object")
    return result


def _git(repository: Path, arguments: tuple[str, ...]) -> bytes:
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repository,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_PUBLICATION, f"Git object read failed: {type(exc).__name__}")
    if completed.returncode != 0 or completed.stderr:
        _fail(FAIL_PUBLICATION, "Git object read rejected the publication identity")
    return completed.stdout


def _require_repository(repository: Path) -> Path:
    try:
        resolved = repository.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_PUBLICATION, f"repository is unavailable: {type(exc).__name__}")
    if not resolved.is_dir():
        _fail(FAIL_PUBLICATION, "repository is not a directory")
    if _git(resolved, ("rev-parse", "--show-toplevel")).decode("utf-8", "strict").strip() != resolved.as_posix():
        _fail(FAIL_PUBLICATION, "repository is not the exact Git top level")
    return resolved


def _resolve_commit(repository: Path, revision: str) -> str:
    if type(revision) is not str or not revision or "\x00" in revision:
        _fail(FAIL_PUBLICATION, "publication revision is malformed")
    value = _git(repository, ("rev-parse", "--verify", f"{revision}^{{commit}}"))
    try:
        commit = value.decode("ascii", "strict").strip()
    except UnicodeDecodeError:
        _fail(FAIL_PUBLICATION, "publication commit is not ASCII")
    if _SHA1_RE.fullmatch(commit) is None:
        _fail(FAIL_PUBLICATION, "publication commit is not a SHA-1 object ID")
    return commit


def _commit_blob(repository: Path, commit: str, repository_path: str) -> bytes:
    row = _git(repository, ("ls-tree", commit, "--", repository_path)).rstrip(b"\n")
    try:
        metadata, raw_path = row.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii", "strict").split(" ")
        observed_path = raw_path.decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError):
        _fail(FAIL_PUBLICATION, f"publication path is absent: {repository_path}")
    if (
        observed_path != repository_path
        or mode != "100644"
        or kind != "blob"
        or _SHA1_RE.fullmatch(object_id) is None
    ):
        _fail(FAIL_PUBLICATION, f"publication path has invalid Git identity: {repository_path}")
    payload = _git(repository, ("cat-file", "blob", object_id))
    if not payload or len(payload) > MAX_PUBLIC_BLOB_BYTES:
        _fail(FAIL_PUBLICATION, f"publication blob has invalid size: {repository_path}")
    return payload


def load_publication_blobs_v1(
    *,
    repository: Path = REPOSITORY_ROOT,
    publication_revision: str = PUBLICATION_COMMIT_B,
) -> tuple[str, bytes, bytes]:
    """Load only the two allowlisted public inputs from immutable Git blobs."""

    root = _require_repository(repository)
    commit = _resolve_commit(root, publication_revision)
    evidence = _commit_blob(root, commit, EVIDENCE_REPOSITORY_PATH)
    promotion = _commit_blob(root, commit, PROMOTION_REPOSITORY_PATH)
    return commit, evidence, promotion


def canonical_run_root_v1(run_id_hex: str) -> Path:
    if type(run_id_hex) is not str or re.fullmatch(r"[0-9a-f]{32}", run_id_hex) is None:
        _fail(FAIL_STATE_IO, "run ID must be lowercase 16-byte hex")
    if run_id_hex != FORMAL_RUN_ID_HEX:
        _fail(FAIL_STATE_IO, "run ID is not the frozen formal M3 execution ID")
    return FORMAL_RUN_PARENT / f"phase3-m3-{run_id_hex}"


def canonical_start_state_path_v1(run_id_hex: str) -> Path:
    return canonical_run_root_v1(run_id_hex) / "m3-start-state.json"


def canonical_terminal_outcome_path_v1(run_id_hex: str) -> Path:
    return canonical_run_root_v1(run_id_hex) / "m3-terminal-outcome.json"


def canonical_start_publication_receipt_path_v1(run_id_hex: str) -> Path:
    return canonical_run_root_v1(run_id_hex) / START_PUBLICATION_RECEIPT_FILENAME


def ensure_canonical_run_root_v1(run_id_hex: str) -> Path:
    """Create or validate the sole persistent directory for one formal run."""

    parent = FORMAL_RUN_PARENT
    try:
        parent_metadata = parent.lstat()
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_STATE_IO, f"formal run parent is unavailable: {type(exc).__name__}")
    if (
        stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or resolved_parent != parent
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
    ):
        _fail(FAIL_STATE_IO, "formal run parent must be real caller-owned mode 0700")
    root = canonical_run_root_v1(run_id_hex)
    try:
        root.mkdir(mode=0o700, exist_ok=True)
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_STATE_IO, f"formal run root is unavailable: {type(exc).__name__}")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != root
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        _fail(FAIL_STATE_IO, "formal run root must be real caller-owned mode 0700")
    return root


def require_canonical_start_state_path_v1(path: Path, run_id_hex: str) -> Path:
    expected = canonical_start_state_path_v1(run_id_hex)
    if not path.is_absolute() or path != expected:
        _fail(FAIL_STATE_IO, "state path is not the unique canonical path for this run")
    return expected


def _retarget_actor_path_digest_v1(
    report: Mapping[str, object],
    *,
    source_digest: str,
    destination_by_purpose: Mapping[int, str],
) -> dict[str, object]:
    if (
        type(source_digest) is not str
        or _SHA256_RE.fullmatch(source_digest) is None
        or set(destination_by_purpose) != {1, 2, 3, 4}
    ):
        _fail(FAIL_GATE_REPLAY, "actor path projection identity is malformed")
    projected = deepcopy(dict(report))
    actors = projected.get("actor_reports")
    if type(actors) is not list or len(actors) != 4:
        _fail(FAIL_GATE_REPLAY, "actor report set is malformed")
    for purpose, actor in enumerate(actors, start=1):
        if type(actor) is not dict or actor.get("purpose_id") != purpose:
            _fail(FAIL_GATE_REPLAY, "actor report purpose order differs")
        probe = actor.get("live_probe")
        environment = probe.get("environment") if type(probe) is dict else None
        output = actor.get("output_binding")
        destination = destination_by_purpose[purpose]
        if (
            type(environment) is not dict
            or environment.get("HEGEL_HOST_REPOSITORY_PATH_SHA256") != source_digest
            or type(output) is not dict
            or type(destination) is not str
            or _SHA256_RE.fullmatch(destination) is None
        ):
            _fail(FAIL_GATE_REPLAY, "actor host-path projection precondition differs")
        environment["HEGEL_HOST_REPOSITORY_PATH_SHA256"] = destination
        probe_bytes = canonical_json_v1(probe)
        output["byte_length"] = len(probe_bytes)
        output["sha256"] = hashlib.sha256(probe_bytes).hexdigest()
    try:
        projected["cross_actor_checks"] = _actor_runtime._validate_cross_actor(actors)
        body = dict(projected)
        body.pop("qualification_payload_sha256")
        projected["qualification_payload_sha256"] = _actor_runtime._canonical_sha256(body)
    except Exception as exc:
        _fail(FAIL_GATE_REPLAY, f"actor path projection failed: {type(exc).__name__}")
    return projected


def _prevalidate_reports_v1(inputs: object, raw_inputs: Mapping[str, object], repository: Path) -> tuple[object, Mapping[str, object], Mapping[str, object]]:
    actor = raw_inputs.get("actor_qualification_report")
    errata = raw_inputs.get("errata_qualification_report")
    if type(actor) is not dict or type(errata) is not dict:
        _fail(FAIL_GATE_REPLAY, "embedded strict-JSON reports are absent")
    inputs = replace(
        inputs,
        actor_qualification_report=actor,
        errata_qualification_report=errata,
    )
    actors = actor.get("actor_reports")
    try:
        source_digest = actors[0]["live_probe"]["environment"][
            "HEGEL_HOST_REPOSITORY_PATH_SHA256"
        ]
    except (KeyError, IndexError, TypeError):
        _fail(FAIL_GATE_REPLAY, "archived actor host-path digest is absent")
    current = {
        purpose: _actor_runtime._actor_environment(purpose)[
            "HEGEL_HOST_REPOSITORY_PATH_SHA256"
        ]
        for purpose in (1, 2, 3, 4)
    }
    projected = _retarget_actor_path_digest_v1(
        actor, source_digest=source_digest, destination_by_purpose=current
    )
    try:
        checked_projected = _validate_actor_report_public_only_v1(
            projected, repository=repository, basis_commit=inputs.basis_commit
        )
        checked_errata = _validate_errata_report_public_only_v1(
            errata, repository=repository, basis_commit=inputs.basis_commit
        )
    except Exception as exc:
        _fail(FAIL_GATE_REPLAY, f"public report-basis replay failed: {type(exc).__name__}")
    checked_actor = _retarget_actor_path_digest_v1(
        checked_projected,
        source_digest=current[1],
        destination_by_purpose={purpose: source_digest for purpose in (1, 2, 3, 4)},
    )
    if checked_actor != actor or checked_errata != errata:
        _fail(FAIL_GATE_REPLAY, "public report projection is not exactly reversible")
    return inputs, checked_actor, checked_errata


def _require_exact_gate_report(promotion: Mapping[str, object]) -> None:
    report = promotion.get("gate_report")
    gates = report.get("gates") if type(report) is dict else None
    if (
        promotion.get("m3_entry_qualified") is not True
        or promotion.get("child_state") != "NOT_RUN"
        or promotion.get("m3_run_started") is not False
        or promotion.get("phase3_m3_start_required_separately") is not True
        or type(report) is not dict
        or report.get("gates_before") != 14
        or report.get("gates_after") != 24
        or report.get("all_gates_15_24_passed") is not True
        or report.get("m3_entry_qualified") is not True
        or report.get("child_state") != "NOT_RUN"
        or report.get("m3_run_started") is not False
        or report.get("output_slot_count") != 15
        or report.get("all_output_slots_null") is not True
        or type(gates) is not list
        or len(gates) != 10
        or [row.get("gate_number") for row in gates if type(row) is dict]
        != list(range(15, 25))
        or any(type(row) is not dict or row.get("passed") is not True for row in gates)
    ):
        _fail(FAIL_GATE_REPLAY, "promotion is not exact 24/24 / NOT_RUN evidence")


def replay_gate_publication_v1(
    evidence_bytes: bytes,
    promotion_bytes: bytes,
    *,
    repository: Path = REPOSITORY_ROOT,
    publication_commit: str,
) -> ReplayedGatePublicationV1:
    """Replay the exact Commit-B gate publication and preserve typed inputs."""

    if type(publication_commit) is not str or _SHA1_RE.fullmatch(publication_commit) is None:
        _fail(FAIL_PUBLICATION, "publication commit must be a resolved SHA-1")
    root = _require_repository(repository)
    resolved_commit = _resolve_commit(root, publication_commit)
    if resolved_commit != publication_commit or resolved_commit != PUBLICATION_COMMIT_B:
        _fail(FAIL_PUBLICATION, "publication is not the frozen authoritative Commit-B")
    evidence = strict_json_loads_v1(evidence_bytes, label="formal gate evidence")
    promotion = strict_json_loads_v1(promotion_bytes, label="formal gate promotion")
    if canonical_json_v1(evidence) != evidence_bytes or canonical_json_v1(promotion) != promotion_bytes:
        _fail(FAIL_JSON, "public evidence and promotion must be canonical JSON bytes")
    assert_public_payload_contains_no_secret_fields(evidence)
    assert_public_payload_contains_no_secret_fields(promotion)
    try:
        typed_inputs = load_gate_evidence_inputs_v1(evidence)
        transported = evidence.get("gate_evidence_inputs")
        if type(transported) is not dict:
            _fail(FAIL_GATE_REPLAY, "gate evidence input transport is absent")
        typed_inputs, actor, errata = _prevalidate_reports_v1(
            typed_inputs, transported, root
        )
        qualified = _evaluate_gates_15_24_with_prevalidated_report_basis_v1(
            typed_inputs,
            actor_report=actor,
            errata_report=errata,
        )
        replayed = promote_gate_evidence_v1(qualified)
    except M3StartError:
        raise
    except Exception as exc:
        _fail(FAIL_GATE_REPLAY, f"complete Gate 15--24 replay failed: {type(exc).__name__}")
    if canonical_json_v1(replayed) != promotion_bytes or replayed != promotion:
        _fail(FAIL_GATE_REPLAY, "committed promotion differs from complete replay")
    _require_exact_gate_report(promotion)

    parents = _git(root, ("show", "-s", "--format=%P", resolved_commit)).decode(
        "ascii", "strict"
    ).strip().split()
    if (
        parents != [typed_inputs.basis_commit]
        or promotion.get("basis_commit") != typed_inputs.basis_commit
    ):
        _fail(
            FAIL_PUBLICATION,
            "publication commit is not the sole child of the evidence basis",
        )
    return ReplayedGatePublicationV1(
        publication_commit=resolved_commit,
        evidence=evidence,
        promotion=promotion,
        gate_inputs=typed_inputs,
        qualified_gate_evidence=qualified,
    )


def prepare_m3_start_v1(
    evidence_bytes: bytes,
    promotion_bytes: bytes,
    *,
    repository: Path = REPOSITORY_ROOT,
    publication_commit: str,
    recorded_at_unix_seconds: int,
) -> dict[str, object]:
    """Replay public evidence and construct, but do not persist, the start."""

    if type(recorded_at_unix_seconds) is not int or recorded_at_unix_seconds < 0:
        _fail(FAIL_STATE_RECORD, "recorded_at_unix_seconds must be uint64")
    replay = replay_gate_publication_v1(
        evidence_bytes,
        promotion_bytes,
        repository=repository,
        publication_commit=publication_commit,
    )
    publication_commit = replay.publication_commit
    evidence = replay.evidence
    promotion = replay.promotion
    typed_inputs = replay.gate_inputs

    candidate_fields = typed_inputs.execution_candidate_fields
    manifest_fields = typed_inputs.execution_manifest_fields
    genesis_fields = typed_inputs.run_genesis_fields
    run_id = candidate_fields.get("run_id")
    if type(run_id) is not bytes or len(run_id) != 16:
        _fail(FAIL_IDENTITY, "run_id is not exactly 16 bytes")
    try:
        _candidate_root, _statement_root, manifest_root = (
            validate_execution_identity_linkage_v1(
                candidate_fields,
                typed_inputs.bridge_statement_fields,
                typed_inputs.bridge_bundle_fields,
                manifest_fields,
                genesis_fields,
            )
        )
        final_snapshot_root = validate_opaque_id_registry_append_v1(
            typed_inputs.opaque_registration_intents,
            typed_inputs.opaque_registry_records,
            typed_inputs.opaque_registry_snapshots[1],
            previous_snapshot_fields=typed_inputs.opaque_registry_snapshots[0],
        )
        genesis_root = candidate_content_root("M3RunGenesisV1", genesis_fields)
        outputs = {
            name: genesis_fields[f"{name}_or_null"] for name in M3_RUN_OUTPUT_ROOTS
        }
        validate_m3_output_roots_null(outputs)
        validate_timestamp_ordering_v1(
            genesis_fields["created_at_unix_seconds"], recorded_at_unix_seconds
        )
    except Exception as exc:
        _fail(FAIL_IDENTITY, f"execution identity replay failed: {type(exc).__name__}")
    snapshot_bound = candidate_fields.get("opaque_id_registry_snapshot_root")
    formal_roots = promotion.get("formal_roots")
    if (
        len(M3_RUN_OUTPUT_ROOTS) != 15
        or genesis_fields.get("initial_state_id") != 0
        or genesis_fields.get("run_id") != run_id
        or manifest_fields.get("run_id") != run_id
        or typed_inputs.bridge_statement_fields.get("run_id") != run_id
        or typed_inputs.opaque_registration_intents[0].get("opaque_id_16_bytes") != run_id
        or snapshot_bound != final_snapshot_root
        or manifest_fields.get("opaque_id_registry_snapshot_root") != final_snapshot_root
        or typed_inputs.bridge_statement_fields.get("opaque_id_registry_snapshot_root") != final_snapshot_root
        or genesis_fields.get("execution_manifest_root") != manifest_root
        or type(formal_roots) is not dict
        or formal_roots.get("m3_execution_manifest_root") != manifest_root.hex()
        or formal_roots.get("m3_run_genesis_root") != genesis_root.hex()
    ):
        _fail(FAIL_IDENTITY, "run/registry/manifest/genesis identity differs")

    state_fields: dict[str, object] = {
        "run_id": run_id,
        "transition_index": 0,
        "previous_state_record_root_or_null": None,
        "from_state_id": 0,
        "from_phase_id": 0,
        "to_state_id": 1,
        "to_phase_id": 1,
        "transition_reason_id": 1,
        "execution_manifest_root": manifest_root,
        "triggering_receipt_root_or_null": None,
        "recorded_at_unix_seconds": recorded_at_unix_seconds,
    }
    try:
        build_formal_object("M3RunStateRecordV1", state_fields)
        state_cbor = encode_formal_object("M3RunStateRecordV1", state_fields)
        decoded = decode_formal_object(state_cbor, expected_name="M3RunStateRecordV1")
        if encode_formal_object("M3RunStateRecordV1", decoded.fields) != state_cbor:
            _fail(FAIL_STATE_RECORD, "state CBOR round trip differs")
        state_root = candidate_content_root("M3RunStateRecordV1", state_fields)
    except M3StartError:
        raise
    except Exception as exc:
        _fail(FAIL_STATE_RECORD, f"formal start record is invalid: {type(exc).__name__}")

    document: dict[str, object] = {
        "schema": STATE_SCHEMA,
        "artifact_kind": STATE_ARTIFACT_KIND,
        "action_id": ACTION_ID,
        "publication_commit": publication_commit,
        "basis_commit": typed_inputs.basis_commit,
        "evidence_blob_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "promotion_blob_sha256": hashlib.sha256(promotion_bytes).hexdigest(),
        "promotion_public_artifact_sha256": promotion["public_artifact_sha256"],
        "formal_gate_count": 24,
        "run_id_hex": run_id.hex(),
        "opaque_id_registry_snapshot_root_hex": final_snapshot_root.hex(),
        "execution_manifest_root_hex": manifest_root.hex(),
        "run_genesis_root_hex": genesis_root.hex(),
        "state_record_cbor_hex": state_cbor.hex(),
        "state_record_root_hex": state_root.hex(),
        "child_state_before": "NOT_RUN",
        "child_state_after": "RUNNING",
        "running_phase_after": "CANONICAL_ENUMERATION",
        "closure_invoked": False,
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "contains_split_assignment_rows": False,
    }
    assert_public_payload_contains_no_secret_fields(document)
    document["state_artifact_sha256"] = hashlib.sha256(canonical_json_v1(document)).hexdigest()
    return document


def prepare_authoritative_m3_start_v1(
    evidence_bytes: bytes,
    promotion_bytes: bytes,
    *,
    repository: Path = REPOSITORY_ROOT,
    publication_commit: str,
    recorded_at_unix_seconds: int,
) -> PreparedM3StartV1:
    document = prepare_m3_start_v1(
        evidence_bytes,
        promotion_bytes,
        repository=repository,
        publication_commit=publication_commit,
        recorded_at_unix_seconds=recorded_at_unix_seconds,
    )
    return PreparedM3StartV1(
        document=MappingProxyType(document),
        _seal=_START_PREPARATION_SEAL,
    )


def validate_state_document_v1(document: Mapping[str, object]) -> None:
    expected = {
        "schema", "artifact_kind", "action_id", "publication_commit", "basis_commit",
        "evidence_blob_sha256", "promotion_blob_sha256", "promotion_public_artifact_sha256",
        "formal_gate_count", "run_id_hex", "opaque_id_registry_snapshot_root_hex",
        "execution_manifest_root_hex", "run_genesis_root_hex", "state_record_cbor_hex",
        "state_record_root_hex", "child_state_before", "child_state_after",
        "running_phase_after", "closure_invoked", "contains_private_key",
        "contains_raw_split_seed", "contains_split_assignment_rows", "state_artifact_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        _fail(FAIL_STATE_RECORD, "state artifact field set differs")
    body = dict(document)
    claimed = body.pop("state_artifact_sha256")
    if (
        document.get("schema") != STATE_SCHEMA
        or document.get("artifact_kind") != STATE_ARTIFACT_KIND
        or document.get("action_id") != ACTION_ID
        or document.get("formal_gate_count") != 24
        or document.get("child_state_before") != "NOT_RUN"
        or document.get("child_state_after") != "RUNNING"
        or document.get("running_phase_after") != "CANONICAL_ENUMERATION"
        or document.get("closure_invoked") is not False
        or any(document.get(name) is not False for name in (
            "contains_private_key", "contains_raw_split_seed", "contains_split_assignment_rows"
        ))
        or type(claimed) is not str
        or claimed != hashlib.sha256(canonical_json_v1(body)).hexdigest()
    ):
        _fail(FAIL_STATE_RECORD, "state artifact identity or self-hash differs")
    if (
        type(document.get("publication_commit")) is not str
        or _SHA1_RE.fullmatch(document["publication_commit"]) is None
        or type(document.get("basis_commit")) is not str
        or _SHA1_RE.fullmatch(document["basis_commit"]) is None
        or type(document.get("run_id_hex")) is not str
        or re.fullmatch(r"[0-9a-f]{32}", document["run_id_hex"]) is None
    ):
        _fail(FAIL_STATE_RECORD, "state artifact commit or run identity is malformed")
    for name in (
        "evidence_blob_sha256", "promotion_blob_sha256", "promotion_public_artifact_sha256",
        "opaque_id_registry_snapshot_root_hex", "execution_manifest_root_hex",
        "run_genesis_root_hex", "state_record_root_hex",
    ):
        if type(document.get(name)) is not str or _SHA256_RE.fullmatch(document[name]) is None:
            _fail(FAIL_STATE_RECORD, f"state artifact digest is malformed: {name}")
    try:
        cbor_hex = document["state_record_cbor_hex"]
        if type(cbor_hex) is not str or len(cbor_hex) % 2:
            raise ValueError("bad CBOR hex")
        cbor = bytes.fromhex(cbor_hex)
        decoded = decode_formal_object(cbor, expected_name="M3RunStateRecordV1")
        if encode_formal_object("M3RunStateRecordV1", decoded.fields) != cbor:
            raise ValueError("noncanonical CBOR")
        if candidate_content_root("M3RunStateRecordV1", decoded.fields).hex() != document["state_record_root_hex"]:
            raise ValueError("state root differs")
        fields = decoded.fields
    except Exception as exc:
        _fail(FAIL_STATE_RECORD, f"state formal record differs: {type(exc).__name__}")
    exact = {
        "transition_index": 0,
        "previous_state_record_root_or_null": None,
        "from_state_id": 0,
        "from_phase_id": 0,
        "to_state_id": 1,
        "to_phase_id": 1,
        "transition_reason_id": 1,
        "triggering_receipt_root_or_null": None,
    }
    if (
        any(fields[name] != value for name, value in exact.items())
        or fields["run_id"].hex() != document.get("run_id_hex")
        or fields["execution_manifest_root"].hex() != document.get("execution_manifest_root_hex")
    ):
        _fail(FAIL_STATE_RECORD, "state record is not the unique index-zero transition")
    assert_public_payload_contains_no_secret_fields(document)


def verify_m3_start_v1(
    state_bytes: bytes,
    evidence_bytes: bytes,
    promotion_bytes: bytes,
    *,
    repository: Path = REPOSITORY_ROOT,
    publication_commit: str,
) -> dict[str, object]:
    """Replay the committed inputs and require exact equality to one state."""

    state = strict_json_loads_v1(state_bytes, label="M3 start state")
    if canonical_json_v1(state) != state_bytes:
        _fail(FAIL_JSON, "M3 start state is not canonical JSON bytes")
    validate_state_document_v1(state)
    decoded = decode_formal_object(
        bytes.fromhex(state["state_record_cbor_hex"]),
        expected_name="M3RunStateRecordV1",
    )
    expected = prepare_m3_start_v1(
        evidence_bytes,
        promotion_bytes,
        repository=repository,
        publication_commit=publication_commit,
        recorded_at_unix_seconds=decoded.fields["recorded_at_unix_seconds"],
    )
    if state != expected:
        _fail(FAIL_STATE_RECORD, "persisted start state differs from exact replay")
    return state


def build_start_publication_receipt_v1(
    state_path: Path,
    state_document: Mapping[str, object],
    local_admission: _local_admission.LocalTwoCommitAdmissionResultV1,
) -> dict[str, object]:
    """Bind the explicit exact-once action to its persisted state bytes."""

    validate_state_document_v1(state_document)
    run_id_hex = state_document.get("run_id_hex")
    if type(run_id_hex) is not str:
        _fail(FAIL_STATE_IO, "start receipt run ID is absent")
    require_canonical_start_state_path_v1(state_path, run_id_hex)
    payload = canonical_json_v1(dict(state_document))
    if not isinstance(
        local_admission,
        _local_admission.LocalTwoCommitAdmissionResultV1,
    ):
        _fail(FAIL_STATE_IO, "start publication lacks local admission evidence")

    def plain(value: Mapping[str, object], *, label: str) -> dict[str, object]:
        result = strict_json_loads_v1(
            _local_admission.canonical_json_v1(value),
            label=label,
        )
        return result

    admission_artifact = plain(
        local_admission.artifact_fields,
        label="local admission artifact",
    )
    admission_manifest = plain(
        local_admission.manifest_fields,
        label="local admission runtime manifest",
    )
    admission_receipt = plain(
        local_admission.receipt_fields,
        label="local admission receipt",
    )
    try:
        _local_admission.validate_local_admission_receipt_v1(
            admission_receipt,
            artifact_fields=admission_artifact,
            manifest_fields=admission_manifest,
        )
    except Exception as exc:
        _fail(FAIL_STATE_IO, f"local admission evidence failed replay: {exc}")
    if (
        local_admission.runtime_commit_c
        != admission_receipt.get("runtime_commit_c")
        or local_admission.approval_commit_d
        != admission_receipt.get("approval_commit_d")
    ):
        _fail(FAIL_STATE_IO, "local admission result identity differs")
    receipt: dict[str, object] = {
        "schema": START_PUBLICATION_RECEIPT_SCHEMA,
        "artifact_kind": START_PUBLICATION_RECEIPT_ARTIFACT_KIND,
        "action_id": ACTION_ID,
        "run_id_hex": run_id_hex,
        "state_relative_path": state_path.name,
        "state_path_sha256": hashlib.sha256(
            state_path.as_posix().encode("utf-8")
        ).hexdigest(),
        "state_file_byte_length": len(payload),
        "state_file_sha256": hashlib.sha256(payload).hexdigest(),
        "state_artifact_sha256": state_document["state_artifact_sha256"],
        "state_record_root_hex": state_document["state_record_root_hex"],
        "publication_commit": state_document["publication_commit"],
        "basis_commit": state_document["basis_commit"],
        "local_admission_artifact": admission_artifact,
        "local_admission_runtime_manifest": admission_manifest,
        "local_admission_receipt": admission_receipt,
        "exact_once_publication_profile": "openat-linkat-no-replace-v1",
        "prepared_output_redirect_accepted": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_v1(receipt)
    ).hexdigest()
    validate_start_publication_receipt_v1(
        receipt,
        state_path=state_path,
        state_document=state_document,
    )
    return receipt


def validate_start_publication_receipt_v1(
    receipt: Mapping[str, object],
    *,
    state_path: Path,
    state_document: Mapping[str, object],
) -> None:
    expected_fields = {
        "schema",
        "artifact_kind",
        "action_id",
        "run_id_hex",
        "state_relative_path",
        "state_path_sha256",
        "state_file_byte_length",
        "state_file_sha256",
        "state_artifact_sha256",
        "state_record_root_hex",
        "publication_commit",
        "basis_commit",
        "local_admission_artifact",
        "local_admission_runtime_manifest",
        "local_admission_receipt",
        "exact_once_publication_profile",
        "prepared_output_redirect_accepted",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
        _fail(FAIL_STATE_RECORD, "start publication receipt field set differs")
    validate_state_document_v1(state_document)
    payload = canonical_json_v1(dict(state_document))
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    run_id_hex = state_document["run_id_hex"]
    expected_path = canonical_start_state_path_v1(run_id_hex)
    admission_artifact = receipt.get("local_admission_artifact")
    admission_manifest = receipt.get("local_admission_runtime_manifest")
    admission_receipt = receipt.get("local_admission_receipt")
    try:
        if not all(
            isinstance(value, Mapping)
            for value in (
                admission_artifact,
                admission_manifest,
                admission_receipt,
            )
        ):
            raise ValueError("local admission evidence is absent")
        _local_admission.validate_local_admission_receipt_v1(
            admission_receipt,
            artifact_fields=admission_artifact,
            manifest_fields=admission_manifest,
        )
    except Exception as exc:
        _fail(FAIL_STATE_RECORD, f"start local admission binding differs: {exc}")
    if (
        state_path != expected_path
        or receipt.get("schema") != START_PUBLICATION_RECEIPT_SCHEMA
        or receipt.get("artifact_kind")
        != START_PUBLICATION_RECEIPT_ARTIFACT_KIND
        or receipt.get("action_id") != ACTION_ID
        or receipt.get("run_id_hex") != run_id_hex
        or receipt.get("state_relative_path") != state_path.name
        or receipt.get("state_path_sha256")
        != hashlib.sha256(state_path.as_posix().encode("utf-8")).hexdigest()
        or receipt.get("state_file_byte_length") != len(payload)
        or receipt.get("state_file_sha256") != hashlib.sha256(payload).hexdigest()
        or receipt.get("state_artifact_sha256")
        != state_document["state_artifact_sha256"]
        or receipt.get("state_record_root_hex")
        != state_document["state_record_root_hex"]
        or receipt.get("publication_commit")
        != state_document["publication_commit"]
        or receipt.get("basis_commit") != state_document["basis_commit"]
        or admission_receipt.get("formal_run_id_hex") != run_id_hex
        or admission_receipt.get("execution_manifest_root_hex")
        != state_document["execution_manifest_root_hex"]
        or admission_artifact.get("publication_commit_b")
        != state_document["publication_commit"]
        or admission_artifact.get("basis_commit_a")
        != state_document["basis_commit"]
        or receipt.get("exact_once_publication_profile")
        != "openat-linkat-no-replace-v1"
        or receipt.get("prepared_output_redirect_accepted") is not False
        or type(claimed) is not str
        or _SHA256_RE.fullmatch(claimed) is None
        or claimed != hashlib.sha256(canonical_json_v1(body)).hexdigest()
    ):
        _fail(FAIL_STATE_RECORD, "start publication receipt identity differs")


def read_start_publication_receipt_v1(
    state_path: Path,
    state_document: Mapping[str, object],
) -> dict[str, object]:
    run_id_hex = state_document.get("run_id_hex")
    if type(run_id_hex) is not str:
        _fail(FAIL_STATE_IO, "start receipt run ID is absent")
    receipt_path = canonical_start_publication_receipt_path_v1(run_id_hex)
    if receipt_path.parent != state_path.parent:
        _fail(FAIL_STATE_IO, "start receipt path is not paired with state path")
    payload = read_state_file_v1(receipt_path)
    receipt = strict_json_loads_v1(payload, label="M3 start publication receipt")
    if canonical_json_v1(receipt) != payload:
        _fail(FAIL_JSON, "M3 start publication receipt is not canonical JSON")
    validate_start_publication_receipt_v1(
        receipt,
        state_path=state_path,
        state_document=state_document,
    )
    return receipt


def _write_start_publication_receipt_exact_once_v1(
    state_path: Path,
    state_document: Mapping[str, object],
    local_admission: _local_admission.LocalTwoCommitAdmissionResultV1,
    *,
    directory_descriptor: int,
) -> None:
    receipt = build_start_publication_receipt_v1(
        state_path,
        state_document,
        local_admission,
    )
    payload = canonical_json_v1(receipt)
    receipt_path = canonical_start_publication_receipt_path_v1(
        state_document["run_id_hex"]
    )
    descriptor: int | None = None
    pending_present = False
    pending_name = (
        f".{receipt_path.name}.{os.getpid()}.{secrets.token_hex(16)}.pending"
    )

    def assert_pinned_directory() -> None:
        directory_metadata = os.fstat(directory_descriptor)
        lexical = receipt_path.parent.lstat()
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(directory_metadata.st_mode) != 0o700
            or (directory_metadata.st_dev, directory_metadata.st_ino)
            != (lexical.st_dev, lexical.st_ino)
        ):
            _fail(FAIL_STATE_IO, "start publication receipt directory differs")

    def read_at(
        name: str,
        *,
        missing_ok: bool = False,
        expected_nlink: int = 1,
    ) -> bytes | None:
        local_descriptor: int | None = None
        try:
            local_descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            _fail(FAIL_STATE_IO, "start publication receipt is absent")
        except OSError as exc:
            _fail(
                FAIL_STATE_IO,
                f"start publication receipt cannot be opened: {type(exc).__name__}",
            )
        try:
            before = os.fstat(local_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_nlink != expected_nlink
                or before.st_size > MAX_STATE_BYTES
            ):
                _fail(FAIL_STATE_IO, "start publication receipt metadata differs")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(local_descriptor, min(remaining, 65_536))
                if not chunk:
                    _fail(FAIL_STATE_IO, "start publication receipt is truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(local_descriptor, 1):
                _fail(FAIL_STATE_IO, "start publication receipt grew while read")
            after = os.fstat(local_descriptor)
            namespace = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or (
                namespace.st_dev,
                namespace.st_ino,
                namespace.st_mode,
                namespace.st_uid,
                namespace.st_nlink,
                namespace.st_size,
                namespace.st_mtime_ns,
                namespace.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                _fail(FAIL_STATE_IO, "start publication receipt changed while read")
            return b"".join(chunks)
        finally:
            if local_descriptor is not None:
                os.close(local_descriptor)

    def recover_published_receipt_hardlink() -> None:
        """Remove only this action's verified post-link crash orphan.

        ``linkat`` publishes the canonical receipt before the temporary name is
        removed.  A process crash in that short interval leaves the same inode
        reachable through two names.  Recovery is permitted only when the
        canonical target has exactly one additional link, that link has this
        receipt writer's exact pending-name grammar, and both names stably read
        as the expected owned mode-0600 bytes.  Unrelated files are never
        removed.
        """

        try:
            target = os.stat(
                receipt_path.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            _fail(
                FAIL_STATE_IO,
                "start publication receipt recovery target cannot be inspected: "
                f"{type(exc).__name__}",
            )
        if target.st_nlink == 1:
            return
        if (
            not stat.S_ISREG(target.st_mode)
            or target.st_uid != os.geteuid()
            or stat.S_IMODE(target.st_mode) != 0o600
            or target.st_nlink != 2
        ):
            _fail(
                FAIL_STATE_IO,
                "start publication receipt has an unrecognized hardlink state",
            )

        pending_pattern = re.compile(
            rf"\.{re.escape(receipt_path.name)}\.[0-9]+\.[0-9a-f]{{32}}\.pending"
        )
        linked_candidates: list[tuple[str, os.stat_result]] = []
        try:
            names = tuple(os.listdir(directory_descriptor))
        except OSError as exc:
            _fail(
                FAIL_STATE_IO,
                "start publication receipt recovery directory cannot be listed: "
                f"{type(exc).__name__}",
            )
        for name in names:
            if pending_pattern.fullmatch(name) is None:
                continue
            try:
                candidate = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except OSError as exc:
                _fail(
                    FAIL_STATE_IO,
                    "start publication receipt pending link cannot be inspected: "
                    f"{type(exc).__name__}",
                )
            if (candidate.st_dev, candidate.st_ino) == (
                target.st_dev,
                target.st_ino,
            ):
                linked_candidates.append((name, candidate))
        if len(linked_candidates) != 1:
            _fail(
                FAIL_STATE_IO,
                "start publication receipt hardlink is not one recognized pending name",
            )
        orphan_name, orphan = linked_candidates[0]
        if (
            not stat.S_ISREG(orphan.st_mode)
            or orphan.st_uid != os.geteuid()
            or stat.S_IMODE(orphan.st_mode) != 0o600
            or orphan.st_nlink != 2
            or read_at(receipt_path.name, expected_nlink=2) != payload
            or read_at(orphan_name, expected_nlink=2) != payload
        ):
            _fail(
                FAIL_STATE_IO,
                "start publication receipt hardlink recovery identity differs",
            )
        target_after = os.stat(
            receipt_path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        orphan_after = os.stat(
            orphan_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )

        def identity(metadata: os.stat_result) -> tuple[int, ...]:
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )

        if (
            identity(target_after) != identity(target)
            or identity(orphan_after) != identity(orphan)
            or (target_after.st_dev, target_after.st_ino)
            != (orphan_after.st_dev, orphan_after.st_ino)
        ):
            _fail(
                FAIL_STATE_IO,
                "start publication receipt hardlink changed during recovery",
            )
        try:
            os.unlink(orphan_name, dir_fd=directory_descriptor)
            os.fsync(directory_descriptor)
        except OSError as exc:
            _fail(
                FAIL_STATE_IO,
                "start publication receipt hardlink recovery failed: "
                f"{type(exc).__name__}",
            )
        if read_at(receipt_path.name) != payload:
            _fail(
                FAIL_STATE_IO,
                "recovered start publication receipt differs",
            )

    try:
        assert_pinned_directory()
        recover_published_receipt_hardlink()
        existing = read_at(receipt_path.name, missing_ok=True)
        if existing is not None:
            if existing != payload:
                _fail(FAIL_STATE_IO, "a different start publication receipt exists")
            assert_pinned_directory()
            return
        descriptor = os.open(
            pending_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        pending_present = True
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(FAIL_STATE_IO, "short start publication receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                pending_name,
                receipt_path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            existing = read_at(receipt_path.name)
            if existing != payload:
                _fail(FAIL_STATE_IO, "a racing start publication receipt differs")
        os.unlink(pending_name, dir_fd=directory_descriptor)
        pending_present = False
        os.fsync(directory_descriptor)
        published = read_at(receipt_path.name)
        if published != payload:
            _fail(FAIL_STATE_IO, "published start publication receipt differs")
        assert_pinned_directory()
        os.fsync(directory_descriptor)
    except M3StartError:
        raise
    except OSError as exc:
        _fail(
            FAIL_STATE_IO,
            f"start publication receipt write failed: {type(exc).__name__}",
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if pending_present:
            try:
                os.unlink(pending_name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
            except FileNotFoundError:
                pass


def read_state_file_v1(path: Path) -> bytes:
    """Read one owned immutable-state candidate through a pinned directory."""

    if not path.is_absolute():
        _fail(FAIL_STATE_IO, "state path must be absolute")
    parent = path.parent
    directory_descriptor: int | None = None
    descriptor: int | None = None
    try:
        lexical_parent = parent.lstat()
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        pinned_parent = os.fstat(directory_descriptor)
        if (
            stat.S_ISLNK(lexical_parent.st_mode)
            or not stat.S_ISDIR(pinned_parent.st_mode)
            or pinned_parent.st_uid != os.geteuid()
            or stat.S_IMODE(pinned_parent.st_mode) != 0o700
            or (lexical_parent.st_dev, lexical_parent.st_ino)
            != (pinned_parent.st_dev, pinned_parent.st_ino)
        ):
            _fail(FAIL_STATE_IO, "state directory identity differs")
        lexical = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size > MAX_STATE_BYTES
        ):
            _fail(FAIL_STATE_IO, "state must be an owned mode-0600 bounded file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(FAIL_STATE_IO, "state file ended before its recorded size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_STATE_IO, "state file grew while being read")
        after = os.fstat(descriptor)
        namespace = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        lexical_parent_after = parent.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (
            namespace.st_dev,
            namespace.st_ino,
            namespace.st_mode,
            namespace.st_uid,
            namespace.st_nlink,
            namespace.st_size,
            namespace.st_mtime_ns,
            namespace.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (lexical_parent_after.st_dev, lexical_parent_after.st_ino) != (
            pinned_parent.st_dev,
            pinned_parent.st_ino,
        ):
            _fail(FAIL_STATE_IO, "state file changed while being read")
        return b"".join(chunks)
    except M3StartError:
        raise
    except OSError as exc:
        _fail(FAIL_STATE_IO, f"state cannot be read safely: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def write_state_exact_once_v1(
    path: Path,
    prepared: PreparedM3StartV1,
    *,
    local_admission: _local_admission.LocalTwoCommitAdmissionResultV1,
) -> str:
    """Atomically publish one canonical state; never overwrite or re-start."""

    if (
        not isinstance(prepared, PreparedM3StartV1)
        or prepared._seal is not _START_PREPARATION_SEAL
    ):
        _fail(FAIL_STATE_IO, "state publication lacks an authoritative replay capability")
    document = dict(prepared.document)
    validate_state_document_v1(document)
    payload = canonical_json_v1(document)
    run_id_hex = document.get("run_id_hex")
    if type(run_id_hex) is not str:
        _fail(FAIL_STATE_IO, "state document run ID is absent")
    require_canonical_start_state_path_v1(path, run_id_hex)
    ensure_canonical_run_root_v1(run_id_hex)
    parent = path.parent
    directory_fd: int | None = None

    def read_regular_exact(
        name: str,
        *,
        missing_ok: bool = False,
        expected_nlink: int = 1,
    ) -> bytes | None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            if missing_ok:
                return None
            _fail(FAIL_ALREADY_STARTED, "existing state is absent")
        except OSError as exc:
            _fail(
                FAIL_ALREADY_STARTED,
                f"existing state cannot be opened: {type(exc).__name__}",
            )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_nlink != expected_nlink
                or before.st_size > MAX_STATE_BYTES
            ):
                _fail(FAIL_ALREADY_STARTED, "existing state metadata differs")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_STATE_BYTES:
                    _fail(FAIL_ALREADY_STARTED, "existing state is oversized")
                chunks.append(chunk)
            after = os.fstat(descriptor)
            namespace = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                before.st_dev,
                before.st_ino,
                before.st_uid,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_mode,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_uid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_mode,
            ) or (
                namespace.st_dev,
                namespace.st_ino,
                namespace.st_uid,
                namespace.st_nlink,
                namespace.st_size,
                namespace.st_mtime_ns,
                namespace.st_ctime_ns,
                namespace.st_mode,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_uid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_mode,
            ):
                _fail(FAIL_ALREADY_STARTED, "existing state changed while read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def existing_status() -> str:
        raw = read_regular_exact(path.name)
        if raw == payload:
            loaded = strict_json_loads_v1(raw, label="existing M3 start state")
            validate_state_document_v1(loaded)
            return "ALREADY_STARTED_IDENTICAL"
        _fail(FAIL_ALREADY_STARTED, "a different state already occupies the exact path")

    pending_name = f".{path.name}.{document['state_record_root_hex']}.pending"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    pending_present = False

    def recover_published_state_hardlink() -> None:
        """Remove only the verified pending alias left by this start action."""

        try:
            target = os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            _fail(
                FAIL_STATE_IO,
                f"state recovery target cannot be inspected: {type(exc).__name__}",
            )
        if target.st_nlink == 1:
            return
        if (
            not stat.S_ISREG(target.st_mode)
            or target.st_uid != os.geteuid()
            or stat.S_IMODE(target.st_mode) != 0o600
            or target.st_nlink != 2
        ):
            _fail(FAIL_STATE_IO, "state has an unrecognized hardlink state")
        try:
            orphan = os.stat(
                pending_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            _fail(
                FAIL_STATE_IO,
                "state hardlink is not the pending name for this start action",
            )
        except OSError as exc:
            _fail(
                FAIL_STATE_IO,
                f"state pending link cannot be inspected: {type(exc).__name__}",
            )
        if (
            not stat.S_ISREG(orphan.st_mode)
            or orphan.st_uid != os.geteuid()
            or stat.S_IMODE(orphan.st_mode) != 0o600
            or orphan.st_nlink != 2
            or (orphan.st_dev, orphan.st_ino) != (target.st_dev, target.st_ino)
            or read_regular_exact(path.name, expected_nlink=2) != payload
            or read_regular_exact(pending_name, expected_nlink=2) != payload
        ):
            _fail(FAIL_STATE_IO, "state hardlink recovery identity differs")
        target_after = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        orphan_after = os.stat(
            pending_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )

        def identity(metadata: os.stat_result) -> tuple[int, ...]:
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )

        if (
            identity(target_after) != identity(target)
            or identity(orphan_after) != identity(orphan)
            or (target_after.st_dev, target_after.st_ino)
            != (orphan_after.st_dev, orphan_after.st_ino)
        ):
            _fail(FAIL_STATE_IO, "state hardlink changed during recovery")
        try:
            os.unlink(pending_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except OSError as exc:
            _fail(
                FAIL_STATE_IO,
                f"state hardlink recovery failed: {type(exc).__name__}",
            )
        if read_regular_exact(path.name) != payload:
            _fail(FAIL_STATE_IO, "recovered state differs")

    try:
        directory_fd = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        directory_metadata = os.fstat(directory_fd)
        parent_metadata = parent.lstat()
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(directory_metadata.st_mode) != 0o700
            or (directory_metadata.st_dev, directory_metadata.st_ino)
            != (parent_metadata.st_dev, parent_metadata.st_ino)
        ):
            _fail(FAIL_STATE_IO, "state directory identity differs")
        recover_published_state_hardlink()
        if read_regular_exact(path.name, missing_ok=True) is not None:
            status = existing_status()
            _write_start_publication_receipt_exact_once_v1(
                path,
                document,
                local_admission,
                directory_descriptor=directory_fd,
            )
            return status
        try:
            descriptor = os.open(
                pending_name, flags, 0o600, dir_fd=directory_fd
            )
            pending_present = True
            os.fchmod(descriptor, 0o600)
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    _fail(FAIL_STATE_IO, "state write made no progress")
                written += count
            os.fsync(descriptor)
        except FileExistsError:
            pending_present = True
            if read_regular_exact(pending_name) != payload:
                _fail(FAIL_STATE_IO, "stale pending state differs")
        finally:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
        try:
            os.link(
                pending_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            status = "STARTED_NEW"
        except FileExistsError:
            status = existing_status()
            os.unlink(pending_name, dir_fd=directory_fd)
            pending_present = False
            os.fsync(directory_fd)
            _write_start_publication_receipt_exact_once_v1(
                path,
                document,
                local_admission,
                directory_descriptor=directory_fd,
            )
            return status
        pending_metadata = os.stat(
            pending_name, dir_fd=directory_fd, follow_symlinks=False
        )
        target_metadata = os.stat(
            path.name, dir_fd=directory_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(target_metadata.st_mode)
            or (pending_metadata.st_dev, pending_metadata.st_ino)
            != (target_metadata.st_dev, target_metadata.st_ino)
            or read_regular_exact(path.name, expected_nlink=2) != payload
        ):
            _fail(FAIL_STATE_IO, "published state inode or bytes differ")
        os.fsync(directory_fd)
        os.unlink(pending_name, dir_fd=directory_fd)
        pending_present = False
        os.fsync(directory_fd)
        if read_regular_exact(path.name) != payload:
            _fail(FAIL_STATE_IO, "published state changed after final link cleanup")
        _write_start_publication_receipt_exact_once_v1(
            path,
            document,
            local_admission,
            directory_descriptor=directory_fd,
        )
        return status
    except M3StartError:
        raise
    except OSError as exc:
        _fail(FAIL_STATE_IO, f"atomic state publication failed: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            if pending_present:
                try:
                    os.unlink(pending_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                except FileNotFoundError:
                    pass
            os.close(directory_fd)


__all__ = [
    "ACTION_ID",
    "EVIDENCE_REPOSITORY_PATH",
    "FAIL_ALREADY_STARTED",
    "FAIL_GATE_REPLAY",
    "FAIL_IDENTITY",
    "FAIL_JSON",
    "FAIL_PUBLICATION",
    "FAIL_STATE_IO",
    "FAIL_STATE_RECORD",
    "FORMAL_RUN_ID_HEX",
    "M3StartError",
    "PROMOTION_REPOSITORY_PATH",
    "PUBLICATION_COMMIT_B",
    "PreparedM3StartV1",
    "ReplayedGatePublicationV1",
    "STATE_SCHEMA",
    "canonical_run_root_v1",
    "canonical_start_publication_receipt_path_v1",
    "canonical_start_state_path_v1",
    "canonical_terminal_outcome_path_v1",
    "build_start_publication_receipt_v1",
    "ensure_canonical_run_root_v1",
    "load_publication_blobs_v1",
    "prepare_m3_start_v1",
    "prepare_authoritative_m3_start_v1",
    "read_start_publication_receipt_v1",
    "read_state_file_v1",
    "replay_gate_publication_v1",
    "require_canonical_start_state_path_v1",
    "strict_json_loads_v1",
    "validate_start_publication_receipt_v1",
    "validate_state_document_v1",
    "verify_m3_start_v1",
    "write_state_exact_once_v1",
]
