"""Static Phase-2B C1 container-actor profile V2 mechanics.

V2 is a strict conservative extension of the exact V1 profile.  Its sole
policy expansion gives ``SCORER_EVALUATOR`` four additional post-reveal,
read-only logical inputs inside one canonical scorer bundle namespace.  This
module only validates, content-addresses, and deeply freezes that policy.  It
does not invoke Docker, generate keys or signatures, mutate a ledger, execute
the recognizer or scorer, evaluate a gate, or establish effect/C1 evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Final, Mapping, NoReturn

from . import phase2b_c1_container_actor_profile_v1 as _v1


PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA: Final = (
    "hegel-phase2b-c1-container-actor-profile/2"
)
PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID: Final = (
    "hegel-owner-accepted-phase2b-c1-container-technical-actors-v2"
)
# The authority/threat-model class is intentionally unchanged.  V2 changes
# scorer dataflow, not the owner-accepted authority semantics.
PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2: Final = (
    "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"
)
PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/CONTAINER_ACTOR_PROFILE/V2\x00"
)
PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_PREFIX: Final = (
    "phase2b_c1_container_actor_profile_v2_"
)
PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID: Final = (
    "phase2b_c1_container_actor_profile_v2_"
    "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
)
PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256: Final = (
    "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
PROFILE_PATH: Final = PROJECT_ROOT / "config/phase2b_c1_container_actor_profile_v2.json"
_V1_PROFILE_RELATIVE_PATH: Final = "config/phase2b_c1_container_actor_profile_v1.json"
_SCORING_SOURCE_RELATIVE_PATH: Final = (
    "src/hegel_machine/phase2b_unsealed_960_prediction_scoring_mechanics_v2.py"
)
_MAXIMUM_PROFILE_BYTES: Final = 256 * 1024

FAIL_PROFILE_IO: Final = "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_IO"
FAIL_PROFILE_JSON: Final = "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_JSON"
FAIL_PROFILE_SCHEMA: Final = "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA"
FAIL_PROFILE_BINDING: Final = "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_BINDING"
FAIL_PROFILE_CLAIM_BOUNDARY: Final = (
    "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CLAIM_BOUNDARY"
)


class Phase2BC1ContainerActorProfileErrorV2(ValueError):
    """Stable fail-closed error for the static V2 profile boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BC1ContainerActorProfileErrorV2(code, detail)


_SUPERSESSION_POLICY: Final = {
    "supersedes_schema_version": _v1.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA,
    "supersedes_profile_id": _v1.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID,
    "supersedes_source_path": _V1_PROFILE_RELATIVE_PATH,
    "supersedes_source_sha256": (
        "22908060372947bdedee30dd4c2920daec4a998b3dc9e0f50f4f3bb5f2e586ad"
    ),
    "supersession_kind": (
        "STRICT_CONSERVATIVE_EXTENSION_SCORER_POST_REVEAL_REPLAY_INPUTS_ONLY"
    ),
    "owner_accepted_authority_requirement_reused_exactly": True,
    "v1_profile_remains_historical_record": True,
    "new_attempts_must_bind_v2_profile_content_id": True,
}

_ADDED_SCORER_INPUTS: Final = [
    "COMMITTED_BLINDED_INPUT_ARCHIVE",
    "PUBLIC_PARTITION_SPEC",
    "EXECUTION_FREEZE_MANIFEST",
    "SCORER_RUN_MANIFEST",
]
_LOGICAL_SCORER_BUNDLE_SECTIONS: Final = [
    "EXECUTION_FREEZE",
    "RAW_INPUT",
    "COMMITTED_PREDICTION",
    "PUBLIC_PARTITION_720_240",
    "ANSWER_ROWS_720_AND_OPENING",
    "GATE_ROWS_720_AND_OPENING",
    "PREDICTION_IF_ABSENT_RECEIPT_ID",
    "SIGNED_REVEAL_TOKEN_ID",
    "SCORER_RUN_MANIFEST",
]
_SCORING_SOURCE_BINDING: Final = {
    "path": _SCORING_SOURCE_RELATIVE_PATH,
    "sha256": "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11",
    "version": "hegel-machine-phase2b-unsealed-960-prediction-scoring-mechanics/2",
    "claim_level": "NON_AUTHORITATIVE_SCORING_MECHANICS_ONLY",
}
_SCORER_DATAFLOW_EXTENSION: Final = {
    "target_role": "SCORER_EVALUATOR",
    "availability_stage": "POST_REVEAL_ONLY",
    "added_read_only_input_classes_in_order": _ADDED_SCORER_INPUTS,
    "single_bundle_namespace": True,
    "canonical_input_bundle_filename": "phase2b_scorer_bundle_v1.bin",
    "logical_sections_in_order": _LOGICAL_SCORER_BUNDLE_SECTIONS,
    "canonical_output_report_filenames_in_order": [
        "score_mechanics_report_v1.json",
        "available_gate_mechanics_report_v1.json",
    ],
    "runtime_receipt_transport": "STDOUT_ONLY_NO_THIRD_FILE",
    "scoring_mechanics_source_binding": _SCORING_SOURCE_BINDING,
    "independent_input_prediction_structure_replay_required": True,
    "independent_partition_replay_required": True,
    "nine_independent_input_files_allowed": False,
    "profile_loader_executes_scorer": False,
    "actual_scoring_claimed": False,
    "formal_gate_evaluation_claimed": False,
    "effect_evidence_claimed": False,
    "c1_exit_evidence_claimed": False,
}
_SCORING_BASE_CONTROL_BINDING: Final = {
    **_SCORING_SOURCE_BINDING,
    "reuse_kind": "EXACT_NON_AUTHORITATIVE_POST_REVEAL_SCORING_MECHANICS_SOURCE",
}
_SOURCE_CONFIG_FIELDS: Final = frozenset(
    {
        "schema_version",
        "profile_id",
        "authority_class",
        "supersession_policy",
        "scorer_dataflow_extension",
    }
)


def _exact_json_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _exact_json_equal(value[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_json_equal(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def _require_exact(value: object, expected: object, field: str) -> None:
    if not _exact_json_equal(value, expected):
        _fail(FAIL_PROFILE_SCHEMA, f"{field} differs from the frozen V2 profile")


def _translate_v1_error(exc: _v1.Phase2BC1ContainerActorProfileErrorV1) -> NoReturn:
    if exc.code == _v1.FAIL_PROFILE_IO:
        code = FAIL_PROFILE_IO
    elif exc.code == _v1.FAIL_PROFILE_JSON:
        code = FAIL_PROFILE_JSON
    elif exc.code == _v1.FAIL_PROFILE_BINDING:
        code = FAIL_PROFILE_BINDING
    elif exc.code == _v1.FAIL_PROFILE_CLAIM_BOUNDARY:
        code = FAIL_PROFILE_CLAIM_BOUNDARY
    else:
        code = FAIL_PROFILE_SCHEMA
    _fail(code, f"V1 conservative base rejected: {exc.detail}")


def _validate_source_config(value: object) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != _SOURCE_CONFIG_FIELDS:
        _fail(FAIL_PROFILE_SCHEMA, "source config field set differs")
    _require_exact(value["schema_version"], PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA, "schema_version")
    _require_exact(value["profile_id"], PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID, "profile_id")
    _require_exact(value["authority_class"], PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2, "authority_class")
    _require_exact(value["supersession_policy"], _SUPERSESSION_POLICY, "supersession_policy")
    _require_exact(value["scorer_dataflow_extension"], _SCORER_DATAFLOW_EXTENSION, "scorer_dataflow_extension")
    claims = value["scorer_dataflow_extension"]
    for name in (
        "nine_independent_input_files_allowed",
        "profile_loader_executes_scorer",
        "actual_scoring_claimed",
        "formal_gate_evaluation_claimed",
        "effect_evidence_claimed",
        "c1_exit_evidence_claimed",
    ):
        if type(claims[name]) is not bool or claims[name]:
            _fail(FAIL_PROFILE_CLAIM_BOUNDARY, f"{name} must remain exact false")
    return _v1._deep_freeze(_v1._detached_json_copy(value))


def _project_effective_profile_to_v1(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _fail(FAIL_PROFILE_SCHEMA, "effective profile root must be an exact object")
    projected = _v1._detached_json_copy(value)
    assert type(projected) is dict
    if set(projected) != set(_v1._TOP_LEVEL_FIELDS) | {
        "supersession_policy",
        "scorer_dataflow_extension",
    }:
        _fail(FAIL_PROFILE_SCHEMA, "effective profile field set differs")
    _require_exact(projected.pop("supersession_policy"), _SUPERSESSION_POLICY, "supersession_policy")
    _require_exact(projected.pop("scorer_dataflow_extension"), _SCORER_DATAFLOW_EXTENSION, "scorer_dataflow_extension")
    projected["schema_version"] = _v1.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA
    projected["profile_id"] = _v1.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID
    bindings = projected["base_control_bindings"]
    if type(bindings) is not dict:
        _fail(FAIL_PROFILE_SCHEMA, "base_control_bindings must be an exact object")
    _require_exact(
        bindings.pop("unsealed_960_scoring_mechanics_source", None),
        _SCORING_BASE_CONTROL_BINDING,
        "base_control_bindings.unsealed_960_scoring_mechanics_source",
    )
    roles = projected["actor_roles"]
    if type(roles) is not list or len(roles) != 6 or type(roles[5]) is not dict:
        _fail(FAIL_PROFILE_SCHEMA, "six-role layout differs")
    scorer = roles[5]
    if scorer.get("role") != "SCORER_EVALUATOR":
        _fail(FAIL_PROFILE_SCHEMA, "purpose six is not SCORER_EVALUATOR")
    inputs = scorer.get("read_only_input_classes")
    if type(inputs) is not list or inputs[-len(_ADDED_SCORER_INPUTS):] != _ADDED_SCORER_INPUTS:
        _fail(FAIL_PROFILE_SCHEMA, "V2 scorer input extension differs")
    del inputs[-len(_ADDED_SCORER_INPUTS):]
    return projected


def validate_phase2b_c1_container_actor_profile_v2(
    value: object,
) -> Mapping[str, object]:
    """Validate an effective V2 profile as an exact conservative V1 extension."""

    projected = _project_effective_profile_to_v1(value)
    try:
        _v1.validate_phase2b_c1_container_actor_profile_v1(projected)
    except _v1.Phase2BC1ContainerActorProfileErrorV1 as exc:
        _translate_v1_error(exc)
    assert type(value) is dict
    _require_exact(value["schema_version"], PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA, "schema_version")
    _require_exact(value["profile_id"], PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID, "profile_id")
    _require_exact(value["authority_class"], PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2, "authority_class")
    return _v1._deep_freeze(_v1._detached_json_copy(value))


def _materialize_effective_profile(
    base_profile: Mapping[str, object],
) -> Mapping[str, object]:
    value = _v1._deep_thaw(base_profile)
    if type(value) is not dict:
        _fail(FAIL_PROFILE_SCHEMA, "V1 snapshot did not contain an exact profile")
    value["schema_version"] = PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA
    value["profile_id"] = PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID
    bindings = value["base_control_bindings"]
    roles = value["actor_roles"]
    assert type(bindings) is dict and type(roles) is list and type(roles[5]) is dict
    bindings["unsealed_960_scoring_mechanics_source"] = _v1._detached_json_copy(
        _SCORING_BASE_CONTROL_BINDING
    )
    inputs = roles[5]["read_only_input_classes"]
    assert type(inputs) is list
    inputs.extend(_ADDED_SCORER_INPUTS)
    value["supersession_policy"] = _v1._detached_json_copy(_SUPERSESSION_POLICY)
    value["scorer_dataflow_extension"] = _v1._detached_json_copy(
        _SCORER_DATAFLOW_EXTENSION
    )
    return validate_phase2b_c1_container_actor_profile_v2(value)


def _is_lower_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


_PROFILE_SNAPSHOT_ISSUANCE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1ContainerActorProfileSnapshotV2:
    def __new__(cls, token: object = None) -> "Phase2BC1ContainerActorProfileSnapshotV2":
        if token is not _PROFILE_SNAPSHOT_ISSUANCE_TOKEN:
            raise TypeError("V2 profile snapshots are privately issued")
        return object.__new__(cls)

    schema_version: str
    profile_id: str
    authority_class: str
    profile_content_id: str
    canonical_profile_sha256: str
    source_profile_sha256: str
    source_path: str
    bound_source_sha256_verified: bool
    profile: Mapping[str, object]
    docker_invoked: bool = False
    keys_generated: bool = False
    signatures_generated: bool = False
    recognizer_executed: bool = False
    scoring_performed: bool = False
    c1_exit_evidence: bool = False

    def __post_init__(self) -> None:
        plain = _v1._deep_thaw(self.profile)
        validated = validate_phase2b_c1_container_actor_profile_v2(plain)
        canonical = _v1._canonical_profile_bytes(_v1._deep_thaw(validated))
        expected_content_id = PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_PREFIX + hashlib.sha256(
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_DOMAIN + canonical
        ).hexdigest()
        if (
            self.schema_version != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA
            or self.profile_id != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID
            or self.authority_class != PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2
            or expected_content_id
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
            or self.profile_content_id != expected_content_id
            or not _is_lower_sha256(self.canonical_profile_sha256)
            or self.canonical_profile_sha256 != hashlib.sha256(canonical).hexdigest()
            or self.canonical_profile_sha256
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
            or not _is_lower_sha256(self.source_profile_sha256)
            or type(self.source_path) is not str
            or not self.source_path.startswith("/")
            or self.bound_source_sha256_verified is not True
        ):
            _fail(FAIL_PROFILE_SCHEMA, "snapshot identity differs")
        for name in (
            "docker_invoked",
            "keys_generated",
            "signatures_generated",
            "recognizer_executed",
            "scoring_performed",
            "c1_exit_evidence",
        ):
            if type(getattr(self, name)) is not bool or getattr(self, name):
                _fail(FAIL_PROFILE_CLAIM_BOUNDARY, f"snapshot {name} must be exact false")


def _issue_snapshot(
    *,
    profile_content_id: str,
    canonical_profile_sha256: str,
    source_profile_sha256: str,
    source_path: str,
    profile: Mapping[str, object],
) -> Phase2BC1ContainerActorProfileSnapshotV2:
    value = Phase2BC1ContainerActorProfileSnapshotV2(_PROFILE_SNAPSHOT_ISSUANCE_TOKEN)
    for name, field_value in (
        ("schema_version", PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA),
        ("profile_id", PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID),
        ("authority_class", PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2),
        ("profile_content_id", profile_content_id),
        ("canonical_profile_sha256", canonical_profile_sha256),
        ("source_profile_sha256", source_profile_sha256),
        ("source_path", source_path),
        ("bound_source_sha256_verified", True),
        ("profile", profile),
        ("docker_invoked", False),
        ("keys_generated", False),
        ("signatures_generated", False),
        ("recognizer_executed", False),
        ("scoring_performed", False),
        ("c1_exit_evidence", False),
    ):
        object.__setattr__(value, name, field_value)
    value.__post_init__()
    return value


def load_phase2b_c1_container_actor_profile_v2(
    profile_path: Path | None = None,
    *,
    repository_root: Path | None = None,
) -> Phase2BC1ContainerActorProfileSnapshotV2:
    """Load V2 and return a fresh immutable effective-profile snapshot."""

    selected = PROFILE_PATH if profile_path is None else profile_path
    root = PROJECT_ROOT if repository_root is None else repository_root
    if not isinstance(selected, Path) or not isinstance(root, Path):
        _fail(FAIL_PROFILE_IO, "profile path and repository root must be exact Path values")
    try:
        payload, resolved = _v1._read_absolute_regular_file(
            selected,
            maximum_bytes=_MAXIMUM_PROFILE_BYTES,
            code=_v1.FAIL_PROFILE_IO,
            label="V2 profile source config",
        )
        decoded = _v1._decode_profile(payload)
    except _v1.Phase2BC1ContainerActorProfileErrorV1 as exc:
        _translate_v1_error(exc)
    _validate_source_config(decoded)
    try:
        _v1._verify_bound_sources(
            root,
            {
                "superseded_v1_profile": {
                    "path": _V1_PROFILE_RELATIVE_PATH,
                    "sha256": _SUPERSESSION_POLICY["supersedes_source_sha256"],
                },
                "unsealed_960_scoring_mechanics_source": _SCORING_SOURCE_BINDING,
            },
        )
        base = _v1.load_phase2b_c1_container_actor_profile_v1(
            root / _V1_PROFILE_RELATIVE_PATH,
            repository_root=root,
        )
    except _v1.Phase2BC1ContainerActorProfileErrorV1 as exc:
        _translate_v1_error(exc)
    profile = _materialize_effective_profile(base.profile)
    canonical = _v1._canonical_profile_bytes(_v1._deep_thaw(profile))
    digest = hashlib.sha256(
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_DOMAIN + canonical
    ).hexdigest()
    content_id = PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_PREFIX + digest
    canonical_sha256 = hashlib.sha256(canonical).hexdigest()
    if (
        content_id != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
        or canonical_sha256
        != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
    ):
        _fail(FAIL_PROFILE_BINDING, "effective V2 profile static identity differs")
    return _issue_snapshot(
        profile_content_id=content_id,
        canonical_profile_sha256=canonical_sha256,
        source_profile_sha256=hashlib.sha256(payload).hexdigest(),
        source_path=resolved.as_posix(),
        profile=profile,
    )


__all__ = [
    "FAIL_PROFILE_BINDING",
    "FAIL_PROFILE_CLAIM_BOUNDARY",
    "FAIL_PROFILE_IO",
    "FAIL_PROFILE_JSON",
    "FAIL_PROFILE_SCHEMA",
    "PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_DOMAIN",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_PREFIX",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA",
    "PROFILE_PATH",
    "Phase2BC1ContainerActorProfileErrorV2",
    "Phase2BC1ContainerActorProfileSnapshotV2",
    "load_phase2b_c1_container_actor_profile_v2",
    "validate_phase2b_c1_container_actor_profile_v2",
]
