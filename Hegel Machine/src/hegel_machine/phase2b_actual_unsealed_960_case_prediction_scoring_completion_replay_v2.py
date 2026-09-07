"""Producer-free completion wire for the Phase-2B-C1 actual-960 replay v2.

This module owns only deterministic state records and the sealed public
completion artifact.  It never imports or invokes the recognizer, baselines,
scorer, target producer, or cohort generator.  A successful terminal records
that the bounded replay mechanics completed; it is explicitly neither a
formal-gate result nor C1 PASS.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Final, Mapping, NoReturn

from . import phase2b_actual_unsealed_960_prediction_scoring_replay_core_v2 as _core
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode, content_hash


ACTION_ID_V2: Final = (
    "PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2"
)
PREREGISTRATION_SCHEMA_V2: Final = (
    "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-"
    "preregistration/2"
)
CONTRACT_SCHEMA_V2: Final = (
    "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-contract/2"
)
CAPSULE_SCHEMA_V2: Final = (
    "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-capsule/2"
)
AUTHORIZATION_SCHEMA_V2: Final = (
    "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-"
    "authorization/2"
)
CONSUMPTION_SCHEMA_V2: Final = (
    "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-"
    "consumption/2"
)
PENDING_SCHEMA_V2: Final = (
    "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-pending/2"
)
TERMINAL_SCHEMA_V2: Final = (
    "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-terminal/2"
)
SEALED_ARTIFACT_SCHEMA_V2: Final = (
    b"hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-"
    b"sealed-public-artifact/2"
)

CAPSULE_READY_V2: Final = _core.READY_STATUS_V2
AUTHORIZATION_RECORDED_V2: Final = (
    "PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_"
    "AUTHORIZATION_RECORDED_NOT_CONSUMED"
)
AUTHORIZATION_CONSUMED_V2: Final = (
    "PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_"
    "AUTHORIZATION_CONSUMED"
)
PRECONSUMPTION_PENDING_V2: Final = (
    "PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_"
    "PRECONSUMPTION_PENDING_NOT_CONSUMED"
)
INVALIDATED_TERMINAL_NO_RESUME_V2: Final = (
    "PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_"
    "INVALIDATED_TERMINAL_NO_RESUME"
)
AUTHORIZED_SCOPE_V2: Final = (
    "PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_ONCE"
)

SEALED_ARTIFACT_TAG_V2: Final = 0x2BC1
SEALED_ARTIFACT_ROOT_DOMAIN_V2: Final = (
    "HEGEL/PHASE2B/C1/ACTUAL960/CASE_PREDICTION_SCORING_REPLAY/SEALED_ARTIFACT/V2"
)
MAXIMUM_SEALED_ARTIFACT_BYTES_V2: Final = 64 * 1024
MAXIMUM_JSON_BYTES_V2: Final = 32 * 1024 * 1024

DOMAINS_V2: Final = {
    "source_closure": (
        b"HEGEL/PHASE2B/C1/ACTUAL960/CASE_PREDICTION_SCORING_REPLAY/"
        b"SOURCE_CLOSURE/V2\x00"
    ),
    "capsule": (
        b"HEGEL/PHASE2B/C1/ACTUAL960/CASE_PREDICTION_SCORING_REPLAY/CAPSULE/V2\x00"
    ),
    "authorization": (
        b"HEGEL/PHASE2B/C1/ACTUAL960/CASE_PREDICTION_SCORING_REPLAY/"
        b"AUTHORIZATION/V2\x00"
    ),
    "consumption": (
        b"HEGEL/PHASE2B/C1/ACTUAL960/CASE_PREDICTION_SCORING_REPLAY/"
        b"CONSUMPTION/V2\x00"
    ),
    "pending": (
        b"HEGEL/PHASE2B/C1/ACTUAL960/CASE_PREDICTION_SCORING_REPLAY/PENDING/V2\x00"
    ),
    "terminal": (
        b"HEGEL/PHASE2B/C1/ACTUAL960/CASE_PREDICTION_SCORING_REPLAY/TERMINAL/V2\x00"
    ),
}

SOURCE_PATHS_V2: Final = (
    "src/hegel_machine/phase2b_actual_unsealed_960_prediction_scoring_replay_core_v2.py",
    "src/hegel_machine/phase2b_actual_unsealed_960_case_prediction_scoring_completion_replay_v2.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "tools/phase2b_actual_unsealed_960_case_prediction_scoring_replay_v2.py",
)

EXECUTION_POLICY_V2: Final = {
    "logical_campaign_limit": 1,
    "authorization_consumption_limit": 1,
    "campaign_id_allocation_limit": 0,
    "attempt_id_allocation_limit": 0,
    "actual_cohort_generation_limit": 0,
    "actual_cohort_producer_read_limit": 0,
    "recognizer_execution_limit": 0,
    "recognizer_result_admission_limit": 1,
    "baseline_result_admission_limit": 3,
    "prediction_archive_admission_limit": 1,
    "answer_manifest_admission_limit": 1,
    "scorer_replay_limit": 1,
    "formal_gate_evaluation_limit": 0,
    "c1_pass_publication_limit": 0,
    "preconsumption_pending_same_capsule_digest": True,
    "preconsumption_pending_consumes_authorization": False,
    "postconsumption_pending_allowed": False,
    "partial_finalization_artifact_to_terminal_only": True,
    "network_allowed": False,
}

AUTHORITY_BOUNDARY_V2: Final = {
    "claim_level": _core.CLAIM_LEVEL_V2,
    "mechanics_complete_is_formal_gate": False,
    "mechanics_complete_is_c1_pass": False,
    "formal_gate_evaluation_performed": False,
    "formal_c1_report_verified": False,
    "effect_evidence": False,
    "external_authority": False,
    "q6_replay_allowed": False,
    "q6_resignature_allowed": False,
    "q6_reissuance_allowed": False,
    "synthesis_allowed": False,
    "mdl_allowed": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "active_transition_allowed": False,
    "production_allowed": False,
    "current_pointer_update_allowed": False,
}

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")


class ActualUnsealed960CompletionReplayErrorV2(ValueError):
    """A sealed artifact or one of its immutable state bindings differs."""


def _fail(detail: str) -> NoReturn:
    raise ActualUnsealed960CompletionReplayErrorV2(detail)


@dataclass(frozen=True, slots=True)
class LoadedJsonV2:
    path: Path
    raw: bytes
    raw_sha256: str
    value: dict[str, Any]

    @property
    def raw_digest(self) -> str:
        return self.raw_sha256


@dataclass(frozen=True, slots=True)
class SealedCompletionArtifactReplayV2:
    projection: _core.ActualUnsealed960ReplayCompletionProjectionV2
    artifact_root: bytes
    canonical_object: tuple[object, ...]
    canonical_bytes: bytes

    @property
    def artifact_root_id(self) -> str:
        return "sha256:" + self.artifact_root.hex()


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON token {value!r}")


def _finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("non-finite JSON number")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail("JSON object keys must be strings")
        for child in value.values():
            _finite(child)
    elif type(value) in (list, tuple):
        for child in value:
            _finite(child)


def canonical_json_bytes(value: object) -> bytes:
    _finite(value)
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
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        _fail(f"canonical JSON rejected: {error}")


def raw_sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail("raw SHA-256 input must be exact bytes")
    return "sha256:" + sha256(payload).hexdigest()


def _digest(domain: bytes, value: object) -> str:
    return "sha256:" + sha256(domain + canonical_json_bytes(value)).hexdigest()


def _tagged(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be one tagged SHA-256")
    return value


def _identity(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        _fail(f"{label} must be one nonempty exact identity")
    return value


def _digest_bytes(value: object, label: str) -> bytes:
    return bytes.fromhex(_tagged(value, label)[7:])


def _text_bytes(value: object, label: str) -> bytes:
    return _identity(value, label).encode("utf-8")


def _text(value: object, label: str) -> str:
    if type(value) is not bytes or not value:
        _fail(f"{label} must be nonempty UTF-8 bytes")
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{label} is not UTF-8: {error}")
    if decoded.encode("utf-8") != value:
        _fail(f"{label} did not replay exactly")
    return _identity(decoded, label)


def _parse_json(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not 1 <= len(payload) <= MAXIMUM_JSON_BYTES_V2:
        _fail(f"{label} exceeds its byte bound")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not strict JSON: {error}")
    if type(value) is not dict:
        _fail(f"{label} must be one JSON object")
    _finite(value)
    return value


def load_pretty_json(path: Path, label: str) -> LoadedJsonV2:
    raw = path.read_bytes()
    return LoadedJsonV2(path, raw, raw_sha256(raw), _parse_json(raw, label))


def load_preregistration(path: Path) -> LoadedJsonV2:
    loaded = load_pretty_json(path, "C1 replay preregistration")
    if (
        loaded.value.get("schema_version") != PREREGISTRATION_SCHEMA_V2
        or loaded.value.get("action_id") != ACTION_ID_V2
    ):
        _fail("C1 replay preregistration identity differs")
    return loaded


def load_contract(path: Path) -> LoadedJsonV2:
    loaded = load_pretty_json(path, "C1 replay actual contract")
    if (
        loaded.value.get("schema_version") != CONTRACT_SCHEMA_V2
        or loaded.value.get("action_id") != ACTION_ID_V2
    ):
        _fail("C1 replay actual contract identity differs")
    _tagged(loaded.value.get("preregistration_raw_sha256"), "preregistration raw SHA-256")
    return loaded


def validate_source_closure_rows(rows: object) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(SOURCE_PATHS_V2):
        _fail("C1 replay source-closure row count differs")
    checked: list[dict[str, Any]] = []
    for expected_path, row in zip(SOURCE_PATHS_V2, rows):
        if (
            type(row) is not dict
            or set(row) != {"relative_path", "byte_count", "raw_sha256"}
            or row.get("relative_path") != expected_path
            or type(row.get("byte_count")) is not int
            or row["byte_count"] < 1
        ):
            _fail("C1 replay source-closure row differs")
        _tagged(row.get("raw_sha256"), f"source row {expected_path} raw SHA-256")
        checked.append(deepcopy(row))
    return checked


def source_closure_root(rows: object) -> str:
    return _digest(DOMAINS_V2["source_closure"], validate_source_closure_rows(rows))


def validate_static_binding_rows(value: object) -> list[dict[str, str]]:
    if type(value) is not list or len(value) != len(_core.REQUIRED_STATIC_BINDING_NAMES_V2):
        _fail("C1 replay static-binding row count differs")
    result: list[dict[str, str]] = []
    for expected_name, row in zip(_core.REQUIRED_STATIC_BINDING_NAMES_V2, value):
        if (
            type(row) is not dict
            or set(row) != {"name", "identity"}
            or row.get("name") != expected_name
        ):
            _fail("C1 replay static-binding row differs")
        result.append(
            {"name": expected_name, "identity": _identity(row.get("identity"), expected_name)}
        )
    return result


def _validate_contract_pair(contract: LoadedJsonV2, preregistration: LoadedJsonV2) -> None:
    if type(contract) is not LoadedJsonV2 or type(preregistration) is not LoadedJsonV2:
        _fail("C1 replay contract inputs differ")
    if contract.value.get("preregistration_raw_sha256") != preregistration.raw_sha256:
        _fail("C1 replay contract/preregistration binding differs")


def build_capsule(
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    *,
    source_commit: str,
    source_closure_rows: list[dict[str, Any]],
    static_binding_rows: list[dict[str, str]],
    dependency_receipt_root: str,
    run_id: str,
    created_at_unix_seconds: int,
) -> dict[str, Any]:
    _validate_contract_pair(contract, preregistration)
    if type(source_commit) is not str or _SOURCE_COMMIT.fullmatch(source_commit) is None:
        _fail("C1 replay source commit differs")
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        _fail("C1 replay run ID differs")
    if type(created_at_unix_seconds) is not int or created_at_unix_seconds < 0:
        _fail("C1 replay capsule creation time differs")
    rows = validate_source_closure_rows(source_closure_rows)
    bindings = validate_static_binding_rows(static_binding_rows)
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA_V2,
        "status": CAPSULE_READY_V2,
        "action_id": ACTION_ID_V2,
        "contract_raw_sha256": contract.raw_sha256,
        "preregistration_raw_sha256": preregistration.raw_sha256,
        "source_commit": source_commit,
        "source_closure_rows": rows,
        "source_closure_root": source_closure_root(rows),
        "static_binding_rows": bindings,
        "dependency_receipt_root": _tagged(
            dependency_receipt_root, "dependency receipt root"
        ),
        "run_id": run_id,
        "created_at_unix_seconds": created_at_unix_seconds,
        "execution_policy": deepcopy(EXECUTION_POLICY_V2),
        "authority_boundary": deepcopy(AUTHORITY_BOUNDARY_V2),
    }
    body["capsule_digest"] = _digest(DOMAINS_V2["capsule"], body)
    return body


def validate_capsule(
    value: object,
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
) -> dict[str, Any]:
    _validate_contract_pair(contract, preregistration)
    if type(value) is not dict:
        _fail("C1 replay capsule differs")
    expected_fields = {
        "schema_version", "status", "action_id", "contract_raw_sha256",
        "preregistration_raw_sha256", "source_commit", "source_closure_rows",
        "source_closure_root", "static_binding_rows", "dependency_receipt_root",
        "run_id", "created_at_unix_seconds", "execution_policy",
        "authority_boundary", "capsule_digest",
    }
    supplied_digest = value.get("capsule_digest")
    unsigned = deepcopy(value)
    unsigned.pop("capsule_digest", None)
    if (
        set(value) != expected_fields
        or value.get("schema_version") != CAPSULE_SCHEMA_V2
        or value.get("status") != CAPSULE_READY_V2
        or value.get("action_id") != ACTION_ID_V2
        or value.get("contract_raw_sha256") != contract.raw_sha256
        or value.get("preregistration_raw_sha256") != preregistration.raw_sha256
        or value.get("execution_policy") != EXECUTION_POLICY_V2
        or value.get("authority_boundary") != AUTHORITY_BOUNDARY_V2
        or supplied_digest != _digest(DOMAINS_V2["capsule"], unsigned)
    ):
        _fail("C1 replay capsule binding differs")
    if (
        type(value.get("source_commit")) is not str
        or _SOURCE_COMMIT.fullmatch(value["source_commit"]) is None
        or type(value.get("run_id")) is not str
        or _RUN_ID.fullmatch(value["run_id"]) is None
        or type(value.get("created_at_unix_seconds")) is not int
        or value["created_at_unix_seconds"] < 0
        or value.get("source_closure_root")
        != source_closure_root(value.get("source_closure_rows"))
    ):
        _fail("C1 replay capsule fixed input differs")
    validate_static_binding_rows(value.get("static_binding_rows"))
    _tagged(value.get("dependency_receipt_root"), "dependency receipt root")
    return deepcopy(value)


def static_bindings_from_capsule(
    capsule: Mapping[str, Any],
) -> _core.ActualUnsealed960ReplayStaticBindingsV2:
    rows = validate_static_binding_rows(capsule.get("static_binding_rows"))
    return _core.ActualUnsealed960ReplayStaticBindingsV2(
        capsule_digest=_tagged(capsule.get("capsule_digest"), "capsule digest"),
        source_closure_root=_tagged(
            capsule.get("source_closure_root"), "source closure root"
        ),
        binding_ids=tuple((row["name"], row["identity"]) for row in rows),
    )


def build_authorization(
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_capsule(capsule, contract, preregistration)
    body: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_V2,
        "status": AUTHORIZATION_RECORDED_V2,
        "action_id": ACTION_ID_V2,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked["capsule_digest"],
        "authorized_scope": AUTHORIZED_SCOPE_V2,
        "execution_policy": deepcopy(EXECUTION_POLICY_V2),
        "authority_boundary": deepcopy(AUTHORITY_BOUNDARY_V2),
    }
    body["authorization_digest"] = _digest(DOMAINS_V2["authorization"], body)
    return body


def validate_authorization(
    value: object,
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_authorization(contract, preregistration, capsule)
    if value != expected:
        _fail("C1 replay authorization differs")
    return deepcopy(expected)


def _validate_admission(
    admission: object,
) -> _core.ActualUnsealed960PreconsumptionAdmissionV2:
    if type(admission) is not _core.ActualUnsealed960PreconsumptionAdmissionV2:
        _fail("C1 replay pre-consumption admission type differs")
    return admission


def build_preconsumption_pending(
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    admission: _core.ActualUnsealed960PreconsumptionAdmissionV2,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    admission = _validate_admission(admission)
    if admission.admitted:
        _fail("C1 replay admitted input cannot produce PENDING")
    body: dict[str, Any] = {
        "schema_version": PENDING_SCHEMA_V2,
        "status": PRECONSUMPTION_PENDING_V2,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "pending_reasons": list(admission.pending_reasons),
        "resume_capsule_digest": checked_capsule["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "campaign_id_allocation_count": 0,
        "attempt_id_allocation_count": 0,
    }
    body["pending_digest"] = _digest(DOMAINS_V2["pending"], body)
    return body


def validate_preconsumption_pending(
    value: object,
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    admission: _core.ActualUnsealed960PreconsumptionAdmissionV2,
) -> dict[str, Any]:
    expected = build_preconsumption_pending(
        contract, preregistration, capsule, authorization, admission
    )
    if value != expected:
        _fail("C1 replay pre-consumption PENDING differs")
    return deepcopy(expected)


def build_consumption(
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    admission: _core.ActualUnsealed960PreconsumptionAdmissionV2,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    admission = _validate_admission(admission)
    if not admission.admitted:
        _fail("C1 replay authorization cannot be consumed before admission")
    body: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA_V2,
        "status": AUTHORIZATION_CONSUMED_V2,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "campaign_id_allocation_count": 0,
        "attempt_id_allocation_count": 0,
        "completed_prefix": [],
        "formal_gate_evaluation_count": 0,
        "c1_pass_publication_count": 0,
    }
    body["record_digest"] = _digest(DOMAINS_V2["consumption"], body)
    return body


def validate_consumption(
    value: object,
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    admission: _core.ActualUnsealed960PreconsumptionAdmissionV2,
) -> dict[str, Any]:
    expected = build_consumption(
        contract, preregistration, capsule, authorization, admission
    )
    if value != expected:
        _fail("C1 replay consumption differs")
    return deepcopy(expected)


def _projection_wire(
    projection: _core.ActualUnsealed960ReplayCompletionProjectionV2,
) -> tuple[object, ...]:
    if type(projection) is not _core.ActualUnsealed960ReplayCompletionProjectionV2:
        _fail("C1 replay completion projection type differs")
    if (
        projection.schema_version != _core.CORE_SCHEMA_VERSION_V2
        or projection.status != _core.MECHANICS_COMPLETE_STATUS_V2
        or projection.claim_level != _core.CLAIM_LEVEL_V2
        or projection.mechanics_complete is not True
        or projection.formal_gate_evaluation_performed is not False
        or projection.formal_c1_report_verified is not False
        or projection.c1_pass is not False
        or projection.effect_evidence is not False
        or projection.external_authority is not False
        or any(
            type(value) is not int
            for value in (
                projection.total_predictions,
                projection.main_scored_rows,
                projection.semantic_conflict_excluded,
                projection.metric_results,
                projection.available_overall_gate_mechanics,
                projection.available_slice_gate_mechanics,
                projection.unavailable_formal_overall_gates,
            )
        )
        or (
            projection.total_predictions,
            projection.main_scored_rows,
            projection.semantic_conflict_excluded,
            projection.metric_results,
            projection.available_overall_gate_mechanics,
            projection.available_slice_gate_mechanics,
            projection.unavailable_formal_overall_gates,
        ) != (960, 720, 240, 9, 10, 24, 2)
    ):
        _fail("C1 replay completion claim boundary differs")
    if type(projection.baseline_result_ids) is not tuple or len(projection.baseline_result_ids) != 3:
        _fail("C1 replay baseline result ordering differs")
    return (
        2,
        SEALED_ARTIFACT_TAG_V2,
        SEALED_ARTIFACT_SCHEMA_V2,
        _text_bytes(projection.schema_version, "core schema version"),
        _digest_bytes(projection.capsule_digest, "capsule digest"),
        _digest_bytes(projection.source_closure_root, "source closure root"),
        _digest_bytes(projection.authorization_digest, "authorization digest"),
        _digest_bytes(projection.consumption_record_digest, "consumption record digest"),
        _text_bytes(projection.authoritative_r16_bundle_id, "R16 bundle ID"),
        _text_bytes(projection.campaign_id, "campaign ID"),
        _text_bytes(projection.attempt_id, "attempt ID"),
        _text_bytes(projection.recognizer_result_id, "recognizer result ID"),
        tuple(_text_bytes(value, "baseline result ID") for value in projection.baseline_result_ids),
        _text_bytes(projection.prediction_archive_id, "prediction archive ID"),
        _text_bytes(projection.answer_manifest_id, "answer manifest ID"),
        _text_bytes(projection.scorer_bundle_id, "scorer bundle ID"),
        _text_bytes(projection.score_report_id, "score report ID"),
        _text_bytes(projection.available_gate_report_id, "available gate report ID"),
        _digest_bytes(
            projection.upstream_admission_manifest_raw_sha256,
            "upstream admission manifest raw SHA-256",
        ),
        _digest_bytes(projection.scorer_bundle_raw_sha256, "scorer bundle raw SHA-256"),
        projection.total_predictions,
        projection.main_scored_rows,
        projection.semantic_conflict_excluded,
        projection.metric_results,
        projection.available_overall_gate_mechanics,
        projection.available_slice_gate_mechanics,
        projection.unavailable_formal_overall_gates,
        _text_bytes(projection.status, "completion status"),
        _text_bytes(projection.claim_level, "claim level"),
        True,
        False,
        False,
        False,
        False,
        False,
    )


def build_sealed_completion_artifact_v2(
    snapshot: _core.ActualUnsealed960ReplaySnapshotV2,
    evidence: _core.ActualUnsealed960ReplaySealedEvidenceV2,
) -> SealedCompletionArtifactReplayV2:
    projection = _core.build_actual_unsealed_960_replay_completion_projection_v2(
        snapshot, evidence
    )
    wire = _projection_wire(projection)
    payload = canonical_cbor_encode(wire)
    return SealedCompletionArtifactReplayV2(
        projection=projection,
        artifact_root=content_hash(SEALED_ARTIFACT_ROOT_DOMAIN_V2, wire),
        canonical_object=wire,
        canonical_bytes=payload,
    )


def encode_sealed_completion_artifact_v2(
    snapshot: _core.ActualUnsealed960ReplaySnapshotV2,
    evidence: _core.ActualUnsealed960ReplaySealedEvidenceV2,
) -> bytes:
    return build_sealed_completion_artifact_v2(snapshot, evidence).canonical_bytes


def decode_sealed_completion_artifact_v2(
    payload: bytes,
) -> SealedCompletionArtifactReplayV2:
    if type(payload) is not bytes or not 1 <= len(payload) <= MAXIMUM_SEALED_ARTIFACT_BYTES_V2:
        _fail("C1 replay sealed artifact exceeds its byte bound")
    try:
        wire = canonical_cbor_decode(payload)
    except (TypeError, ValueError) as error:
        _fail(f"C1 replay sealed artifact is not canonical: {error}")
    if (
        type(wire) is not tuple
        or len(wire) != 35
        or wire[:3] != (2, SEALED_ARTIFACT_TAG_V2, SEALED_ARTIFACT_SCHEMA_V2)
        or type(wire[12]) is not tuple
        or len(wire[12]) != 3
        or wire[29:] != (True, False, False, False, False, False)
    ):
        _fail("C1 replay sealed artifact wire differs")
    for index, label in (
        (4, "capsule digest"),
        (5, "source closure root"),
        (6, "authorization digest"),
        (7, "consumption record digest"),
        (18, "upstream admission manifest raw SHA-256"),
        (19, "scorer bundle raw SHA-256"),
    ):
        if type(wire[index]) is not bytes or len(wire[index]) != 32:
            _fail(f"C1 replay sealed artifact {label} differs")
    projection = _core.ActualUnsealed960ReplayCompletionProjectionV2(
        schema_version=_text(wire[3], "core schema version"),
        capsule_digest="sha256:" + wire[4].hex(),
        source_closure_root="sha256:" + wire[5].hex(),
        authorization_digest="sha256:" + wire[6].hex(),
        consumption_record_digest="sha256:" + wire[7].hex(),
        authoritative_r16_bundle_id=_text(wire[8], "R16 bundle ID"),
        campaign_id=_text(wire[9], "campaign ID"),
        attempt_id=_text(wire[10], "attempt ID"),
        recognizer_result_id=_text(wire[11], "recognizer result ID"),
        baseline_result_ids=tuple(_text(value, "baseline result ID") for value in wire[12]),
        prediction_archive_id=_text(wire[13], "prediction archive ID"),
        answer_manifest_id=_text(wire[14], "answer manifest ID"),
        scorer_bundle_id=_text(wire[15], "scorer bundle ID"),
        score_report_id=_text(wire[16], "score report ID"),
        available_gate_report_id=_text(wire[17], "available gate report ID"),
        upstream_admission_manifest_raw_sha256="sha256:" + wire[18].hex(),
        scorer_bundle_raw_sha256="sha256:" + wire[19].hex(),
        total_predictions=wire[20],
        main_scored_rows=wire[21],
        semantic_conflict_excluded=wire[22],
        metric_results=wire[23],
        available_overall_gate_mechanics=wire[24],
        available_slice_gate_mechanics=wire[25],
        unavailable_formal_overall_gates=wire[26],
        status=_text(wire[27], "completion status"),
        claim_level=_text(wire[28], "claim level"),
        mechanics_complete=wire[29],
        formal_gate_evaluation_performed=wire[30],
        formal_c1_report_verified=wire[31],
        c1_pass=wire[32],
        effect_evidence=wire[33],
        external_authority=wire[34],
    )
    if canonical_cbor_encode(_projection_wire(projection)) != payload:
        _fail("C1 replay sealed artifact exact replay differs")
    return SealedCompletionArtifactReplayV2(
        projection=projection,
        artifact_root=content_hash(SEALED_ARTIFACT_ROOT_DOMAIN_V2, wire),
        canonical_object=wire,
        canonical_bytes=payload,
    )


def validate_artifact_binding(
    payload: bytes,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> SealedCompletionArtifactReplayV2:
    replay = decode_sealed_completion_artifact_v2(payload)
    projection = replay.projection
    if (
        projection.capsule_digest != capsule.get("capsule_digest")
        or projection.source_closure_root != capsule.get("source_closure_root")
        or projection.authorization_digest != authorization.get("authorization_digest")
        or projection.consumption_record_digest != consumption.get("record_digest")
    ):
        _fail("C1 replay sealed artifact state binding differs")
    return replay


def build_complete_terminal(
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    admission: _core.ActualUnsealed960PreconsumptionAdmissionV2,
    artifact_payload: bytes,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption,
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
        admission,
    )
    replay = validate_artifact_binding(
        artifact_payload, checked_capsule, checked_authorization, checked_consumption
    )
    projection = replay.projection
    outcome = {
        "status": _core.MECHANICS_COMPLETE_STATUS_V2,
        "claim_level": _core.CLAIM_LEVEL_V2,
        "artifact_raw_sha256": raw_sha256(artifact_payload),
        "artifact_root": replay.artifact_root_id,
        "authoritative_r16_bundle_id": projection.authoritative_r16_bundle_id,
        "campaign_id": projection.campaign_id,
        "attempt_id": projection.attempt_id,
        "recognizer_result_id": projection.recognizer_result_id,
        "baseline_result_ids": list(projection.baseline_result_ids),
        "prediction_archive_id": projection.prediction_archive_id,
        "answer_manifest_id": projection.answer_manifest_id,
        "scorer_bundle_id": projection.scorer_bundle_id,
        "score_report_id": projection.score_report_id,
        "available_gate_report_id": projection.available_gate_report_id,
        "upstream_admission_manifest_raw_sha256": (
            projection.upstream_admission_manifest_raw_sha256
        ),
        "scorer_bundle_raw_sha256": projection.scorer_bundle_raw_sha256,
        "total_predictions": projection.total_predictions,
        "main_scored_rows": projection.main_scored_rows,
        "semantic_conflict_excluded": projection.semantic_conflict_excluded,
        "metric_results": projection.metric_results,
        "available_overall_gate_mechanics": (
            projection.available_overall_gate_mechanics
        ),
        "available_slice_gate_mechanics": projection.available_slice_gate_mechanics,
        "unavailable_formal_overall_gates": (
            projection.unavailable_formal_overall_gates
        ),
        "producer_free_canonical_replay_completed": True,
        "mechanics_complete": True,
        "formal_gate_evaluation_performed": False,
        "formal_c1_report_verified": False,
        "c1_pass": False,
        "effect_evidence": False,
        "external_authority": False,
        "terminal_no_resume": False,
        "campaign_complete": True,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA_V2,
        "status": _core.MECHANICS_COMPLETE_STATUS_V2,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "outcome": outcome,
    }
    body["record_digest"] = _digest(DOMAINS_V2["terminal"], body)
    return body


def build_invalidated_terminal(
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    admission: _core.ActualUnsealed960PreconsumptionAdmissionV2,
    *,
    failure_phase: str,
    failure_code: str,
    failure_detail: str,
) -> dict[str, Any]:
    checked_capsule = validate_capsule(capsule, contract, preregistration)
    checked_authorization = validate_authorization(
        authorization, contract, preregistration, checked_capsule
    )
    checked_consumption = validate_consumption(
        consumption,
        contract,
        preregistration,
        checked_capsule,
        checked_authorization,
        admission,
    )
    for value, label in (
        (failure_phase, "failure phase"),
        (failure_code, "failure code"),
        (failure_detail, "failure detail"),
    ):
        _identity(value, label)
    outcome = {
        "status": INVALIDATED_TERMINAL_NO_RESUME_V2,
        "artifact_raw_sha256": None,
        "artifact_root": None,
        "formal_gate_evaluation_performed": False,
        "formal_c1_report_verified": False,
        "c1_pass": False,
        "effect_evidence": False,
        "external_authority": False,
        "failure_phase": failure_phase,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "terminal_no_resume": True,
        "campaign_complete": False,
    }
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA_V2,
        "status": INVALIDATED_TERMINAL_NO_RESUME_V2,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": checked_capsule["capsule_digest"],
        "authorization_digest": checked_authorization["authorization_digest"],
        "consumption_record_digest": checked_consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "outcome": outcome,
    }
    body["record_digest"] = _digest(DOMAINS_V2["terminal"], body)
    return body


def validate_terminal(
    value: object,
    contract: LoadedJsonV2,
    preregistration: LoadedJsonV2,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    admission: _core.ActualUnsealed960PreconsumptionAdmissionV2,
    *,
    artifact_payload: bytes | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("outcome")) is not dict:
        _fail("C1 replay terminal must contain one outcome")
    if value.get("status") == _core.MECHANICS_COMPLETE_STATUS_V2:
        if artifact_payload is None:
            _fail("C1 replay complete terminal requires the sealed public artifact")
        expected = build_complete_terminal(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            admission,
            artifact_payload,
        )
    elif value.get("status") == INVALIDATED_TERMINAL_NO_RESUME_V2:
        if artifact_payload is not None:
            _fail("C1 replay invalidated terminal forbids a valid artifact")
        outcome = value["outcome"]
        expected = build_invalidated_terminal(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            admission,
            failure_phase=outcome.get("failure_phase"),
            failure_code=outcome.get("failure_code"),
            failure_detail=outcome.get("failure_detail"),
        )
    else:
        _fail("C1 replay terminal status differs")
    if value != expected:
        _fail("C1 replay terminal projection differs")
    return deepcopy(expected)


__all__ = (
    "ACTION_ID_V2",
    "PREREGISTRATION_SCHEMA_V2",
    "CONTRACT_SCHEMA_V2",
    "CAPSULE_SCHEMA_V2",
    "AUTHORIZATION_SCHEMA_V2",
    "CONSUMPTION_SCHEMA_V2",
    "PENDING_SCHEMA_V2",
    "TERMINAL_SCHEMA_V2",
    "CAPSULE_READY_V2",
    "AUTHORIZATION_RECORDED_V2",
    "AUTHORIZATION_CONSUMED_V2",
    "PRECONSUMPTION_PENDING_V2",
    "INVALIDATED_TERMINAL_NO_RESUME_V2",
    "AUTHORIZED_SCOPE_V2",
    "SEALED_ARTIFACT_SCHEMA_V2",
    "SEALED_ARTIFACT_ROOT_DOMAIN_V2",
    "SOURCE_PATHS_V2",
    "EXECUTION_POLICY_V2",
    "AUTHORITY_BOUNDARY_V2",
    "ActualUnsealed960CompletionReplayErrorV2",
    "LoadedJsonV2",
    "SealedCompletionArtifactReplayV2",
    "canonical_json_bytes",
    "raw_sha256",
    "load_pretty_json",
    "load_preregistration",
    "load_contract",
    "validate_source_closure_rows",
    "source_closure_root",
    "validate_static_binding_rows",
    "build_capsule",
    "validate_capsule",
    "static_bindings_from_capsule",
    "build_authorization",
    "validate_authorization",
    "build_preconsumption_pending",
    "validate_preconsumption_pending",
    "build_consumption",
    "validate_consumption",
    "build_sealed_completion_artifact_v2",
    "encode_sealed_completion_artifact_v2",
    "decode_sealed_completion_artifact_v2",
    "validate_artifact_binding",
    "build_complete_terminal",
    "build_invalidated_terminal",
    "validate_terminal",
)
