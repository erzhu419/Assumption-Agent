"""Pure state core for the Phase-2B-C1 actual-960 replay v2.

The module deliberately owns no files, processes, containers, targets, or
cohort rows.  It validates the static capsule surface, reduces explicit
controller events, and projects already sealed evidence into a mechanics-only
completion record.  A successful projection is not a formal gate evaluation
and cannot be used as C1 PASS evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Final, Mapping, NoReturn


CORE_SCHEMA_VERSION_V2: Final = (
    "hegel-machine-phase2b-c1-actual-unsealed-960-case-prediction-scoring-"
    "replay-core/2"
)
READY_STATUS_V2: Final = (
    "PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_"
    "CAPSULE_READY_NOT_AUTHORIZED"
)
MECHANICS_COMPLETE_STATUS_V2: Final = (
    "PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_"
    "MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1"
)
CLAIM_LEVEL_V2: Final = "LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"

REQUIRED_STATIC_BINDING_NAMES_V2: Final = (
    "owner_actual_unsealed_960_authorization",
    "actual_unsealed_960_readiness_dag",
    "actual_unsealed_960_authoritative_evidence_schema_catalog",
    "container_actor_profile",
    "logical_stage_authorization",
    "three_baseline_registration_profile",
    "recognizer_runtime_image_build_receipt",
    "scorer_runtime_image_build_receipt",
    "scorer_bundle_codec",
    "actual_unsealed_960_replay_input_contract",
    "formal_unsealed_prediction_scoring_contract",
    "unsealed_960_prediction_scoring_mechanics",
    "unsealed_960_available_gate_mechanics",
    "actual_unsealed_960_replay_admission_gap_inventory",
    "actual_unsealed_960_signed_attempt_chain",
    "actual_unsealed_960_local_attempt_ledger",
    "actual_unsealed_960_external_custodian_checkpoint",
    "recognizer_host_executor",
    "scorer_host_executor",
    "c1_statistical_validation",
)

_COMPLETED_PREFIXES_V2: Final = (
    (),
    ("recognizer",),
    ("recognizer", "baselines"),
    ("recognizer", "baselines", "prediction_archive"),
    ("recognizer", "baselines", "prediction_archive", "answer_manifest"),
    (
        "recognizer",
        "baselines",
        "prediction_archive",
        "answer_manifest",
        "scorer_replay",
    ),
)


class ActualUnsealed960ReplayCoreErrorV2(ValueError):
    """An exact core binding, event, or completion invariant differs."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise ActualUnsealed960ReplayCoreErrorV2(code, detail)


def _require_exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail("REJECT_REPLAY_BOOLEAN", f"{label} must be exact bool")
    return value


def _require_sha256_id(value: object, label: str) -> str:
    if type(value) is not str or not value.startswith("sha256:"):
        _fail("REJECT_REPLAY_DIGEST", f"{label} must be sha256:<64 lowercase hex>")
    raw = value[7:]
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        _fail("REJECT_REPLAY_DIGEST", f"{label} must be sha256:<64 lowercase hex>")
    return value


def _require_identity(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        _fail("REJECT_REPLAY_IDENTITY", f"{label} must be a nonempty exact string")
    return value


class ActualUnsealed960ReplayStateV2(str, Enum):
    FROZEN_NOT_AUTHORIZED = "FROZEN_NOT_AUTHORIZED"
    AUTHORIZED_NOT_CONSUMED = "AUTHORIZED_NOT_CONSUMED"
    PRECONSUMPTION_PENDING = "PRECONSUMPTION_PENDING"
    CONSUMED = "CONSUMED"
    RECOGNIZER_RESULT_SEALED = "RECOGNIZER_RESULT_SEALED"
    BASELINE_RESULTS_ADMITTED = "BASELINE_RESULTS_ADMITTED"
    PREDICTION_ARCHIVE_COMMITTED = "PREDICTION_ARCHIVE_COMMITTED"
    ANSWER_REVEAL_ADMITTED = "ANSWER_REVEAL_ADMITTED"
    SCORER_REPLAY_SEALED = "SCORER_REPLAY_SEALED"
    MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1 = (
        "MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1"
    )
    COMPLETE_FAIL = "COMPLETE_FAIL"
    INVALIDATED_TERMINAL_NO_RESUME = "INVALIDATED_TERMINAL_NO_RESUME"


class ActualUnsealed960ReplayEventV2(str, Enum):
    RECORD_AUTHORIZATION = "RECORD_AUTHORIZATION"
    REPLAY_PRECONSUMPTION_ADMISSION = "REPLAY_PRECONSUMPTION_ADMISSION"
    CONSUME_AUTHORIZATION = "CONSUME_AUTHORIZATION"
    SEAL_RECOGNIZER_RESULT = "SEAL_RECOGNIZER_RESULT"
    ADMIT_BASELINE_RESULTS = "ADMIT_BASELINE_RESULTS"
    ADMIT_PREDICTION_ARCHIVE = "ADMIT_PREDICTION_ARCHIVE"
    ADMIT_ANSWER_MANIFEST = "ADMIT_ANSWER_MANIFEST"
    SEAL_SCORER_REPLAY = "SEAL_SCORER_REPLAY"
    COMPLETE_MECHANICS = "COMPLETE_MECHANICS"
    COMPLETE_FAIL = "COMPLETE_FAIL"
    INVALIDATE = "INVALIDATE"


class ActualUnsealed960PendingReasonV2(str, Enum):
    STATIC_CONTRACT_CLOSURE_UNVERIFIED = "STATIC_CONTRACT_CLOSURE_UNVERIFIED"
    AUTHORITATIVE_SCHEMA_VERIFIER_UNAVAILABLE = (
        "AUTHORITATIVE_SCHEMA_VERIFIER_UNAVAILABLE"
    )
    AUTHORITATIVE_R01_R16_BUNDLE_UNVERIFIED = (
        "AUTHORITATIVE_R01_R16_BUNDLE_UNVERIFIED"
    )
    EXTERNAL_CUSTODIAN_CHECKPOINT_UNVERIFIED = (
        "EXTERNAL_CUSTODIAN_CHECKPOINT_UNVERIFIED"
    )
    THREE_BASELINE_OUTPUTS_UNVERIFIED = "THREE_BASELINE_OUTPUTS_UNVERIFIED"
    EXACT_SCALE_BRIDGE_UNVERIFIED = "EXACT_SCALE_BRIDGE_UNVERIFIED"
    RECOGNIZER_EXECUTOR_BINDING_UNVERIFIED = (
        "RECOGNIZER_EXECUTOR_BINDING_UNVERIFIED"
    )
    SCORER_EXECUTOR_BINDING_UNVERIFIED = "SCORER_EXECUTOR_BINDING_UNVERIFIED"


@dataclass(frozen=True, slots=True)
class ActualUnsealed960ReplayStaticBindingsV2:
    capsule_digest: str
    source_closure_root: str
    binding_ids: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_sha256_id(self.capsule_digest, "capsule digest")
        _require_sha256_id(self.source_closure_root, "source closure root")
        if type(self.binding_ids) is not tuple:
            _fail("REJECT_REPLAY_BINDINGS", "binding IDs must be an exact tuple")
        names: list[str] = []
        for index, row in enumerate(self.binding_ids):
            if type(row) is not tuple or len(row) != 2:
                _fail("REJECT_REPLAY_BINDINGS", f"binding row {index} differs")
            name, identity = row
            names.append(_require_identity(name, f"binding name {index}"))
            _require_identity(identity, f"binding identity {name}")
        if tuple(names) != REQUIRED_STATIC_BINDING_NAMES_V2:
            _fail("REJECT_REPLAY_BINDINGS", "binding names or order differ")

    @property
    def binding_map(self) -> Mapping[str, str]:
        return dict(self.binding_ids)


@dataclass(frozen=True, slots=True)
class ActualUnsealed960PreconsumptionAdmissionV2:
    static_contract_closure_verified: bool
    authoritative_schema_verifier_available: bool
    authoritative_r01_r16_bundle_verified: bool
    external_custodian_checkpoint_verified: bool
    three_baseline_outputs_verified: bool
    exact_scale_bridge_verified: bool
    recognizer_executor_binding_verified: bool
    scorer_executor_binding_verified: bool

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_exact_bool(getattr(self, name), name)

    @property
    def pending_reasons(self) -> tuple[str, ...]:
        checks = (
            (
                self.static_contract_closure_verified,
                ActualUnsealed960PendingReasonV2.STATIC_CONTRACT_CLOSURE_UNVERIFIED,
            ),
            (
                self.authoritative_schema_verifier_available,
                ActualUnsealed960PendingReasonV2.AUTHORITATIVE_SCHEMA_VERIFIER_UNAVAILABLE,
            ),
            (
                self.authoritative_r01_r16_bundle_verified,
                ActualUnsealed960PendingReasonV2.AUTHORITATIVE_R01_R16_BUNDLE_UNVERIFIED,
            ),
            (
                self.external_custodian_checkpoint_verified,
                ActualUnsealed960PendingReasonV2.EXTERNAL_CUSTODIAN_CHECKPOINT_UNVERIFIED,
            ),
            (
                self.three_baseline_outputs_verified,
                ActualUnsealed960PendingReasonV2.THREE_BASELINE_OUTPUTS_UNVERIFIED,
            ),
            (
                self.exact_scale_bridge_verified,
                ActualUnsealed960PendingReasonV2.EXACT_SCALE_BRIDGE_UNVERIFIED,
            ),
            (
                self.recognizer_executor_binding_verified,
                ActualUnsealed960PendingReasonV2.RECOGNIZER_EXECUTOR_BINDING_UNVERIFIED,
            ),
            (
                self.scorer_executor_binding_verified,
                ActualUnsealed960PendingReasonV2.SCORER_EXECUTOR_BINDING_UNVERIFIED,
            ),
        )
        return tuple(reason.value for passed, reason in checks if not passed)

    @property
    def admitted(self) -> bool:
        return not self.pending_reasons


@dataclass(frozen=True, slots=True)
class ActualUnsealed960ReplayActionAuditV2:
    logical_campaigns: int = 0
    authorization_consumptions: int = 0
    campaign_attempt_id_allocations: int = 0
    readiness_admissions: int = 0
    actual_cohort_generations: int = 0
    recognizer_result_admissions: int = 0
    baseline_output_admissions: int = 0
    prediction_archive_admissions: int = 0
    answer_manifest_admissions: int = 0
    scorer_replays: int = 0
    formal_gate_evaluations: int = 0
    c1_pass_publications: int = 0
    q6_replays: int = 0
    q6_resignatures: int = 0
    q6_reissuances: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail("REJECT_REPLAY_AUDIT", f"{name} must be a nonnegative exact int")
        singletons = (
            "logical_campaigns",
            "authorization_consumptions",
            "readiness_admissions",
            "recognizer_result_admissions",
            "prediction_archive_admissions",
            "answer_manifest_admissions",
            "scorer_replays",
        )
        if any(getattr(self, name) > 1 for name in singletons):
            _fail("REJECT_REPLAY_AUDIT", "a one-shot action count exceeds one")
        if self.baseline_output_admissions not in (0, 3):
            _fail("REJECT_REPLAY_AUDIT", "baseline admission count must be zero or three")
        forbidden = (
            "campaign_attempt_id_allocations",
            "actual_cohort_generations",
            "formal_gate_evaluations",
            "c1_pass_publications",
            "q6_replays",
            "q6_resignatures",
            "q6_reissuances",
        )
        if any(getattr(self, name) != 0 for name in forbidden):
            _fail("REJECT_REPLAY_BOUNDARY", "a forbidden action count is nonzero")


ZERO_ACTION_AUDIT_V2: Final = ActualUnsealed960ReplayActionAuditV2()


@dataclass(frozen=True, slots=True)
class ActualUnsealed960ReplaySnapshotV2:
    bindings: ActualUnsealed960ReplayStaticBindingsV2
    state: ActualUnsealed960ReplayStateV2
    admission_satisfied: bool
    pending_reasons: tuple[str, ...]
    completed_prefix: tuple[str, ...]
    audit: ActualUnsealed960ReplayActionAuditV2

    def __post_init__(self) -> None:
        if type(self.bindings) is not ActualUnsealed960ReplayStaticBindingsV2:
            _fail("REJECT_REPLAY_SNAPSHOT", "bindings type differs")
        if type(self.state) is not ActualUnsealed960ReplayStateV2:
            _fail("REJECT_REPLAY_SNAPSHOT", "state type differs")
        _require_exact_bool(self.admission_satisfied, "admission satisfied")
        if (
            type(self.pending_reasons) is not tuple
            or any(type(reason) is not str or not reason for reason in self.pending_reasons)
        ):
            _fail("REJECT_REPLAY_SNAPSHOT", "pending reasons differ")
        if self.state is ActualUnsealed960ReplayStateV2.PRECONSUMPTION_PENDING:
            if self.admission_satisfied or not self.pending_reasons:
                _fail("REJECT_REPLAY_SNAPSHOT", "pending state lacks exact blockers")
        elif self.pending_reasons:
            _fail("REJECT_REPLAY_SNAPSHOT", "non-pending state retains blockers")
        if self.completed_prefix not in _COMPLETED_PREFIXES_V2:
            _fail("REJECT_REPLAY_SNAPSHOT", "completed prefix differs")
        if type(self.audit) is not ActualUnsealed960ReplayActionAuditV2:
            _fail("REJECT_REPLAY_SNAPSHOT", "action audit type differs")
        if self.state is ActualUnsealed960ReplayStateV2.FROZEN_NOT_AUTHORIZED:
            if self.admission_satisfied or self.audit != ZERO_ACTION_AUDIT_V2:
                _fail("REJECT_REPLAY_SNAPSHOT", "frozen state is not action-free")

    @property
    def terminal(self) -> bool:
        return self.state in (
            ActualUnsealed960ReplayStateV2.MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1,
            ActualUnsealed960ReplayStateV2.COMPLETE_FAIL,
            ActualUnsealed960ReplayStateV2.INVALIDATED_TERMINAL_NO_RESUME,
        )

    @property
    def formal_gate_evaluation_performed(self) -> bool:
        return False

    @property
    def formal_c1_report_verified(self) -> bool:
        return False

    @property
    def c1_pass(self) -> bool:
        return False

    @property
    def effect_evidence(self) -> bool:
        return False


def initial_actual_unsealed_960_replay_snapshot_v2(
    bindings: ActualUnsealed960ReplayStaticBindingsV2,
) -> ActualUnsealed960ReplaySnapshotV2:
    """Issue the action-free engineering-freeze snapshot."""

    if type(bindings) is not ActualUnsealed960ReplayStaticBindingsV2:
        _fail("REJECT_REPLAY_BINDINGS", "bindings type differs")
    return ActualUnsealed960ReplaySnapshotV2(
        bindings=bindings,
        state=ActualUnsealed960ReplayStateV2.FROZEN_NOT_AUTHORIZED,
        admission_satisfied=False,
        pending_reasons=(),
        completed_prefix=(),
        audit=ZERO_ACTION_AUDIT_V2,
    )


def _increment_audit(
    audit: ActualUnsealed960ReplayActionAuditV2,
    **increments: int,
) -> ActualUnsealed960ReplayActionAuditV2:
    values = {
        name: getattr(audit, name) + increments.get(name, 0)
        for name in audit.__dataclass_fields__
    }
    unknown = set(increments).difference(values)
    if unknown:
        _fail("REJECT_REPLAY_AUDIT", "unknown action audit increment")
    return ActualUnsealed960ReplayActionAuditV2(**values)


def reduce_actual_unsealed_960_prediction_scoring_replay_v2(
    snapshot: ActualUnsealed960ReplaySnapshotV2,
    event: ActualUnsealed960ReplayEventV2,
    *,
    admission: ActualUnsealed960PreconsumptionAdmissionV2 | None = None,
) -> ActualUnsealed960ReplaySnapshotV2:
    """Apply one exact event without performing the event's external action."""

    if type(snapshot) is not ActualUnsealed960ReplaySnapshotV2:
        _fail("REJECT_REPLAY_SNAPSHOT", "snapshot type differs")
    if type(event) is not ActualUnsealed960ReplayEventV2:
        _fail("REJECT_REPLAY_EVENT", "event type differs")
    if snapshot.terminal:
        _fail("REJECT_REPLAY_TERMINAL", "terminal replay cannot resume")

    state = snapshot.state
    if event is ActualUnsealed960ReplayEventV2.RECORD_AUTHORIZATION:
        if state is not ActualUnsealed960ReplayStateV2.FROZEN_NOT_AUTHORIZED:
            _fail("REJECT_REPLAY_TRANSITION", "authorization is not recordable here")
        return replace(
            snapshot,
            state=ActualUnsealed960ReplayStateV2.AUTHORIZED_NOT_CONSUMED,
        )

    if event is ActualUnsealed960ReplayEventV2.REPLAY_PRECONSUMPTION_ADMISSION:
        if state not in (
            ActualUnsealed960ReplayStateV2.AUTHORIZED_NOT_CONSUMED,
            ActualUnsealed960ReplayStateV2.PRECONSUMPTION_PENDING,
        ):
            _fail("REJECT_REPLAY_TRANSITION", "pre-consumption admission is closed")
        if type(admission) is not ActualUnsealed960PreconsumptionAdmissionV2:
            _fail("REJECT_REPLAY_ADMISSION", "exact admission result is required")
        reasons = admission.pending_reasons
        if reasons:
            return replace(
                snapshot,
                state=ActualUnsealed960ReplayStateV2.PRECONSUMPTION_PENDING,
                admission_satisfied=False,
                pending_reasons=reasons,
            )
        audit = snapshot.audit
        if snapshot.audit.readiness_admissions == 0:
            audit = _increment_audit(audit, readiness_admissions=1)
        return replace(
            snapshot,
            state=ActualUnsealed960ReplayStateV2.AUTHORIZED_NOT_CONSUMED,
            admission_satisfied=True,
            pending_reasons=(),
            audit=audit,
        )

    if admission is not None:
        _fail("REJECT_REPLAY_ADMISSION", "admission payload is event-inapplicable")

    if event is ActualUnsealed960ReplayEventV2.CONSUME_AUTHORIZATION:
        if (
            state is not ActualUnsealed960ReplayStateV2.AUTHORIZED_NOT_CONSUMED
            or not snapshot.admission_satisfied
        ):
            _fail("REJECT_REPLAY_TRANSITION", "authorization is not consumable")
        return replace(
            snapshot,
            state=ActualUnsealed960ReplayStateV2.CONSUMED,
            audit=_increment_audit(
                snapshot.audit,
                logical_campaigns=1,
                authorization_consumptions=1,
            ),
        )

    transitions = {
        ActualUnsealed960ReplayEventV2.SEAL_RECOGNIZER_RESULT: (
            ActualUnsealed960ReplayStateV2.CONSUMED,
            ActualUnsealed960ReplayStateV2.RECOGNIZER_RESULT_SEALED,
            ("recognizer",),
            {"recognizer_result_admissions": 1},
        ),
        ActualUnsealed960ReplayEventV2.ADMIT_BASELINE_RESULTS: (
            ActualUnsealed960ReplayStateV2.RECOGNIZER_RESULT_SEALED,
            ActualUnsealed960ReplayStateV2.BASELINE_RESULTS_ADMITTED,
            ("recognizer", "baselines"),
            {"baseline_output_admissions": 3},
        ),
        ActualUnsealed960ReplayEventV2.ADMIT_PREDICTION_ARCHIVE: (
            ActualUnsealed960ReplayStateV2.BASELINE_RESULTS_ADMITTED,
            ActualUnsealed960ReplayStateV2.PREDICTION_ARCHIVE_COMMITTED,
            ("recognizer", "baselines", "prediction_archive"),
            {"prediction_archive_admissions": 1},
        ),
        ActualUnsealed960ReplayEventV2.ADMIT_ANSWER_MANIFEST: (
            ActualUnsealed960ReplayStateV2.PREDICTION_ARCHIVE_COMMITTED,
            ActualUnsealed960ReplayStateV2.ANSWER_REVEAL_ADMITTED,
            ("recognizer", "baselines", "prediction_archive", "answer_manifest"),
            {"answer_manifest_admissions": 1},
        ),
        ActualUnsealed960ReplayEventV2.SEAL_SCORER_REPLAY: (
            ActualUnsealed960ReplayStateV2.ANSWER_REVEAL_ADMITTED,
            ActualUnsealed960ReplayStateV2.SCORER_REPLAY_SEALED,
            (
                "recognizer",
                "baselines",
                "prediction_archive",
                "answer_manifest",
                "scorer_replay",
            ),
            {"scorer_replays": 1},
        ),
    }
    if event in transitions:
        required, target, prefix, increments = transitions[event]
        if state is not required:
            _fail("REJECT_REPLAY_TRANSITION", f"{event.value} is out of order")
        return replace(
            snapshot,
            state=target,
            completed_prefix=prefix,
            audit=_increment_audit(snapshot.audit, **increments),
        )

    if event is ActualUnsealed960ReplayEventV2.COMPLETE_MECHANICS:
        if state is not ActualUnsealed960ReplayStateV2.SCORER_REPLAY_SEALED:
            _fail("REJECT_REPLAY_TRANSITION", "mechanics completion is premature")
        return replace(
            snapshot,
            state=(
                ActualUnsealed960ReplayStateV2.
                MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1
            ),
        )

    if event in (
        ActualUnsealed960ReplayEventV2.COMPLETE_FAIL,
        ActualUnsealed960ReplayEventV2.INVALIDATE,
    ):
        consumed_states = set(ActualUnsealed960ReplayStateV2).difference(
            {
                ActualUnsealed960ReplayStateV2.FROZEN_NOT_AUTHORIZED,
                ActualUnsealed960ReplayStateV2.AUTHORIZED_NOT_CONSUMED,
                ActualUnsealed960ReplayStateV2.PRECONSUMPTION_PENDING,
                ActualUnsealed960ReplayStateV2.MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1,
                ActualUnsealed960ReplayStateV2.COMPLETE_FAIL,
                ActualUnsealed960ReplayStateV2.INVALIDATED_TERMINAL_NO_RESUME,
            }
        )
        if state not in consumed_states:
            _fail("REJECT_REPLAY_TRANSITION", "terminal failure is not post-consumption")
        target = (
            ActualUnsealed960ReplayStateV2.COMPLETE_FAIL
            if event is ActualUnsealed960ReplayEventV2.COMPLETE_FAIL
            else ActualUnsealed960ReplayStateV2.INVALIDATED_TERMINAL_NO_RESUME
        )
        return replace(snapshot, state=target)

    _fail("REJECT_REPLAY_EVENT", "event is unsupported")


@dataclass(frozen=True, slots=True)
class ActualUnsealed960ReplaySealedEvidenceV2:
    authorization_digest: str
    consumption_record_digest: str
    authoritative_r16_bundle_id: str
    campaign_id: str
    attempt_id: str
    recognizer_result_id: str
    baseline_result_ids: tuple[str, str, str]
    prediction_archive_id: str
    answer_manifest_id: str
    scorer_bundle_id: str
    score_report_id: str
    available_gate_report_id: str
    upstream_admission_manifest_raw_sha256: str
    scorer_bundle_raw_sha256: str
    total_predictions: int
    main_scored_rows: int
    semantic_conflict_excluded: int
    metric_results: int
    available_overall_gate_mechanics: int
    available_slice_gate_mechanics: int
    unavailable_formal_overall_gates: int

    def __post_init__(self) -> None:
        _require_sha256_id(self.authorization_digest, "authorization digest")
        _require_sha256_id(self.consumption_record_digest, "consumption record digest")
        _require_sha256_id(
            self.upstream_admission_manifest_raw_sha256,
            "upstream admission manifest raw SHA-256",
        )
        _require_sha256_id(
            self.scorer_bundle_raw_sha256, "scorer bundle raw SHA-256"
        )
        for name in (
            "authoritative_r16_bundle_id",
            "campaign_id",
            "attempt_id",
            "recognizer_result_id",
            "prediction_archive_id",
            "answer_manifest_id",
            "scorer_bundle_id",
            "score_report_id",
            "available_gate_report_id",
        ):
            _require_identity(getattr(self, name), name)
        if (
            type(self.baseline_result_ids) is not tuple
            or len(self.baseline_result_ids) != 3
            or any(type(value) is not str for value in self.baseline_result_ids)
        ):
            _fail("REJECT_REPLAY_EVIDENCE", "baseline result IDs differ")
        for index, value in enumerate(self.baseline_result_ids):
            _require_identity(value, f"baseline result ID {index}")
        if len(set(self.baseline_result_ids)) != 3:
            _fail("REJECT_REPLAY_EVIDENCE", "baseline result IDs are not distinct")
        exact_counts = {
            "total_predictions": 960,
            "main_scored_rows": 720,
            "semantic_conflict_excluded": 240,
            "metric_results": 9,
            "available_overall_gate_mechanics": 10,
            "available_slice_gate_mechanics": 24,
            "unavailable_formal_overall_gates": 2,
        }
        if any(
            type(getattr(self, name)) is not int
            or getattr(self, name) != expected
            for name, expected in exact_counts.items()
        ):
            _fail("REJECT_REPLAY_EVIDENCE", "bounded replay counts differ")


@dataclass(frozen=True, slots=True)
class ActualUnsealed960ReplayCompletionProjectionV2:
    schema_version: str
    capsule_digest: str
    source_closure_root: str
    authorization_digest: str
    consumption_record_digest: str
    authoritative_r16_bundle_id: str
    campaign_id: str
    attempt_id: str
    recognizer_result_id: str
    baseline_result_ids: tuple[str, str, str]
    prediction_archive_id: str
    answer_manifest_id: str
    scorer_bundle_id: str
    score_report_id: str
    available_gate_report_id: str
    upstream_admission_manifest_raw_sha256: str
    scorer_bundle_raw_sha256: str
    total_predictions: int
    main_scored_rows: int
    semantic_conflict_excluded: int
    metric_results: int
    available_overall_gate_mechanics: int
    available_slice_gate_mechanics: int
    unavailable_formal_overall_gates: int
    status: str
    claim_level: str
    mechanics_complete: bool
    formal_gate_evaluation_performed: bool
    formal_c1_report_verified: bool
    c1_pass: bool
    effect_evidence: bool
    external_authority: bool


def build_actual_unsealed_960_replay_completion_projection_v2(
    snapshot: ActualUnsealed960ReplaySnapshotV2,
    evidence: ActualUnsealed960ReplaySealedEvidenceV2,
) -> ActualUnsealed960ReplayCompletionProjectionV2:
    """Project sealed public identities without rerunning any producer."""

    if type(snapshot) is not ActualUnsealed960ReplaySnapshotV2:
        _fail("REJECT_REPLAY_SNAPSHOT", "snapshot type differs")
    if type(evidence) is not ActualUnsealed960ReplaySealedEvidenceV2:
        _fail("REJECT_REPLAY_EVIDENCE", "sealed evidence type differs")
    if (
        snapshot.state
        is not ActualUnsealed960ReplayStateV2.MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1
    ):
        _fail("REJECT_REPLAY_COMPLETION", "snapshot is not mechanics-complete")
    expected_audit = ActualUnsealed960ReplayActionAuditV2(
        logical_campaigns=1,
        authorization_consumptions=1,
        readiness_admissions=1,
        recognizer_result_admissions=1,
        baseline_output_admissions=3,
        prediction_archive_admissions=1,
        answer_manifest_admissions=1,
        scorer_replays=1,
    )
    if snapshot.audit != expected_audit:
        _fail("REJECT_REPLAY_COMPLETION", "successful action audit differs")
    return ActualUnsealed960ReplayCompletionProjectionV2(
        schema_version=CORE_SCHEMA_VERSION_V2,
        capsule_digest=snapshot.bindings.capsule_digest,
        source_closure_root=snapshot.bindings.source_closure_root,
        authorization_digest=evidence.authorization_digest,
        consumption_record_digest=evidence.consumption_record_digest,
        authoritative_r16_bundle_id=evidence.authoritative_r16_bundle_id,
        campaign_id=evidence.campaign_id,
        attempt_id=evidence.attempt_id,
        recognizer_result_id=evidence.recognizer_result_id,
        baseline_result_ids=evidence.baseline_result_ids,
        prediction_archive_id=evidence.prediction_archive_id,
        answer_manifest_id=evidence.answer_manifest_id,
        scorer_bundle_id=evidence.scorer_bundle_id,
        score_report_id=evidence.score_report_id,
        available_gate_report_id=evidence.available_gate_report_id,
        upstream_admission_manifest_raw_sha256=(
            evidence.upstream_admission_manifest_raw_sha256
        ),
        scorer_bundle_raw_sha256=evidence.scorer_bundle_raw_sha256,
        total_predictions=evidence.total_predictions,
        main_scored_rows=evidence.main_scored_rows,
        semantic_conflict_excluded=evidence.semantic_conflict_excluded,
        metric_results=evidence.metric_results,
        available_overall_gate_mechanics=(
            evidence.available_overall_gate_mechanics
        ),
        available_slice_gate_mechanics=evidence.available_slice_gate_mechanics,
        unavailable_formal_overall_gates=evidence.unavailable_formal_overall_gates,
        status=MECHANICS_COMPLETE_STATUS_V2,
        claim_level=CLAIM_LEVEL_V2,
        mechanics_complete=True,
        formal_gate_evaluation_performed=False,
        formal_c1_report_verified=False,
        c1_pass=False,
        effect_evidence=False,
        external_authority=False,
    )


__all__ = (
    "CORE_SCHEMA_VERSION_V2",
    "READY_STATUS_V2",
    "MECHANICS_COMPLETE_STATUS_V2",
    "CLAIM_LEVEL_V2",
    "REQUIRED_STATIC_BINDING_NAMES_V2",
    "ActualUnsealed960ReplayCoreErrorV2",
    "ActualUnsealed960ReplayStateV2",
    "ActualUnsealed960ReplayEventV2",
    "ActualUnsealed960PendingReasonV2",
    "ActualUnsealed960ReplayStaticBindingsV2",
    "ActualUnsealed960PreconsumptionAdmissionV2",
    "ActualUnsealed960ReplayActionAuditV2",
    "ZERO_ACTION_AUDIT_V2",
    "ActualUnsealed960ReplaySnapshotV2",
    "ActualUnsealed960ReplaySealedEvidenceV2",
    "ActualUnsealed960ReplayCompletionProjectionV2",
    "initial_actual_unsealed_960_replay_snapshot_v2",
    "reduce_actual_unsealed_960_prediction_scoring_replay_v2",
    "build_actual_unsealed_960_replay_completion_projection_v2",
)
