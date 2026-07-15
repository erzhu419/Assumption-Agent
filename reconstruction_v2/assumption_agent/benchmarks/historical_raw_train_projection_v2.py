from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..models import SplitName, stable_hash
from .skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
    skilllearn_program_set_treatment_hash,
)
from .skilllearn_lifecycle import (
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
    _extract_train_action_trace_profile,
)
from .train_outcome_ranker_v2 import FrozenRawTrainBaselineSetV2


HISTORICAL_RAW_TRAIN_PROJECTION_VERSION = (
    "historical_raw_train_observation_projection_v2"
)
HISTORICAL_OBSERVATION_SCHEMA_VERSION = (
    "skilllearn_trial_observation_git_7da8a3d0_v1"
)
HISTORICAL_SOURCE_COMMIT = "7da8a3d0a40653013ea8e253d81982d00e5d3c37"

_CURRENT_ADDITIVE_OBSERVATION_FIELDS = (
    "runtime_profile_prompt_delivery_policy",
    "runtime_profile_prompt_injection_receipt_hash",
    "runtime_profile_effective_prompt_sha256",
)
_HISTORICAL_OBSERVATION_FIELDS = frozenset(
    {
        "request",
        "success",
        "score",
        "metrics",
        "total_tokens",
        "steps",
        "duration_seconds",
        "provider_fingerprint",
        "fairness_fingerprint",
        "error_type",
        "upstream_result_hash",
        "raw_trial_artifacts_persisted",
        "prebuilt_image_key",
        "prebuilt_image_id",
        "prebuilt_cache_reused",
        "agent_runtime_key",
        "agent_runtime_version",
        "offline_verifier_profile_id",
        "offline_verifier_runtime_key",
        "step_budget_policy",
        "step_budget_unit",
        "step_budget_limit",
        "step_budget_truncated",
        "step_budget_token_usage_complete",
        "step_budget_receipt_hash",
        "installed_skill_source_receipt_hash",
        "secret_value_persisted",
    }
)
_OPTIONAL_HISTORICAL_OBSERVATION_FIELDS = frozenset(
    {"proposal_action_trace_hash"}
)
_EVENT_ENVELOPE_FIELDS = frozenset(
    {
        "event",
        "stage",
        "trace_id",
        "payload",
        "payload_hash",
        "event_id",
        "raw_content_persisted",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class HistoricalRawTrainProjectionError(PermissionError):
    """Historical RAW evidence could not be projected without ambiguity."""


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise HistoricalRawTrainProjectionError(
            f"{label} is not a sha256 digest"
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise HistoricalRawTrainProjectionError(
            "historical artifact path is not relative"
        )
    candidate = root.joinpath(*Path(relative_path).parts)
    if candidate.is_symlink():
        raise HistoricalRawTrainProjectionError(
            "historical artifact file is a symlink"
        )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise HistoricalRawTrainProjectionError(
            "historical artifact escaped its frozen root"
        ) from exc
    if not resolved.is_file():
        raise HistoricalRawTrainProjectionError(
            "historical artifact is not a regular file"
        )
    return resolved


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoricalRawTrainProjectionError(
            f"{label} is not readable canonical JSON"
        ) from exc
    if not isinstance(value, dict):
        raise HistoricalRawTrainProjectionError(f"{label} is not an object")
    return value


def _load_verified_events(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise HistoricalRawTrainProjectionError(
            "historical event ledger is unreadable"
        ) from exc
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise HistoricalRawTrainProjectionError(
                f"historical event is malformed at line {line_number}"
            ) from exc
        if not isinstance(row, dict) or set(row) != _EVENT_ENVELOPE_FIELDS:
            raise HistoricalRawTrainProjectionError(
                f"historical event envelope drifted at line {line_number}"
            )
        payload = row.get("payload")
        event = row.get("event")
        stage = row.get("stage")
        trace_id = row.get("trace_id")
        if (
            not isinstance(payload, dict)
            or not isinstance(event, str)
            or not event
            or not isinstance(stage, str)
            or not stage
            or not isinstance(trace_id, str)
            or not trace_id
            or row.get("payload_hash") != stable_hash(payload)
            or row.get("event_id")
            != stable_hash(
                {
                    "event": event,
                    "stage": stage,
                    "trace_id": trace_id,
                    "payload": payload,
                }
            )[:24]
            or row.get("raw_content_persisted") is not False
        ):
            raise HistoricalRawTrainProjectionError(
                f"historical event envelope failed at line {line_number}"
            )
        rows.append(row)
    if not rows:
        raise HistoricalRawTrainProjectionError(
            "historical event ledger is empty"
        )
    return tuple(rows)


def _historical_observation_payload(
    observation: SkillLearnTrialObservation,
) -> dict[str, Any]:
    payload = observation.to_dict()
    allowed = (
        _HISTORICAL_OBSERVATION_FIELDS
        | _OPTIONAL_HISTORICAL_OBSERVATION_FIELDS
        | frozenset(_CURRENT_ADDITIVE_OBSERVATION_FIELDS)
    )
    if set(payload) - allowed:
        raise HistoricalRawTrainProjectionError(
            "current observation schema has unreviewed additive fields"
        )
    if not _HISTORICAL_OBSERVATION_FIELDS.issubset(payload):
        raise HistoricalRawTrainProjectionError(
            "current observation schema lost historical fields"
        )
    for field_name in _CURRENT_ADDITIVE_OBSERVATION_FIELDS:
        if payload.get(field_name) != "":
            raise HistoricalRawTrainProjectionError(
                "historical projection cannot drop a nonempty current field"
            )
        payload.pop(field_name, None)
    if not set(payload).issubset(
        _HISTORICAL_OBSERVATION_FIELDS
        | _OPTIONAL_HISTORICAL_OBSERVATION_FIELDS
    ):
        raise HistoricalRawTrainProjectionError(
            "historical observation projection is not closed"
        )
    return payload


def historical_observation_hash_v2(
    observation: SkillLearnTrialObservation,
) -> str:
    return stable_hash(_historical_observation_payload(observation))


@dataclass(frozen=True)
class HistoricalRawTrainSourceReceiptV2:
    manifest_hash: str
    source_commit: str
    source_trace_id_hash: str
    evaluator_epoch_hash: str
    protocol_lock_file_sha256: str
    event_ledger_file_sha256: str
    manifest_file_sha256: str
    raw_artifact_set_hash: str
    source_observation_set_hash: str
    projected_observation_set_hash: str
    projection_mapping_set_hash: str
    row_count: int
    success_count: int

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "projection_policy": HISTORICAL_RAW_TRAIN_PROJECTION_VERSION,
            "historical_observation_schema": (
                HISTORICAL_OBSERVATION_SCHEMA_VERSION
            ),
            "manifest_hash": self.manifest_hash,
            "source_commit": self.source_commit,
            "source_trace_id_hash": self.source_trace_id_hash,
            "evaluator_epoch_hash": self.evaluator_epoch_hash,
            "protocol_lock_file_sha256": self.protocol_lock_file_sha256,
            "event_ledger_file_sha256": self.event_ledger_file_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "raw_artifact_set_hash": self.raw_artifact_set_hash,
            "source_observation_set_hash": self.source_observation_set_hash,
            "projected_observation_set_hash": (
                self.projected_observation_set_hash
            ),
            "projection_mapping_set_hash": self.projection_mapping_set_hash,
            "row_count": self.row_count,
            "success_count": self.success_count,
            "source_raw_trial_artifact_row_count": self.row_count,
            "historical_observations_verified": True,
            "projected_observations_verified": True,
            "behavioral_values_preserved": True,
            "model_calls": 0,
            "evaluator_calls": 0,
            "network_calls": 0,
            "validation_accessed": False,
            "test_accessed": False,
            "source_raw_artifact_content_embedded": False,
            "source_raw_artifact_locator_embedded": False,
            "secret_value_persisted": False,
        }

    def verify(self) -> None:
        for label, value in (
            ("manifest hash", self.manifest_hash),
            ("source trace hash", self.source_trace_id_hash),
            ("evaluator epoch hash", self.evaluator_epoch_hash),
            ("protocol lock file hash", self.protocol_lock_file_sha256),
            ("event ledger file hash", self.event_ledger_file_sha256),
            ("manifest file hash", self.manifest_file_sha256),
            ("raw artifact set hash", self.raw_artifact_set_hash),
            ("source observation set hash", self.source_observation_set_hash),
            (
                "projected observation set hash",
                self.projected_observation_set_hash,
            ),
            ("projection mapping set hash", self.projection_mapping_set_hash),
        ):
            _require_sha256(value, label)
        if (
            not _GIT_COMMIT.fullmatch(self.source_commit)
            or self.row_count <= 0
            or self.success_count < 0
            or self.success_count > self.row_count
        ):
            raise HistoricalRawTrainProjectionError(
                "historical source receipt is invalid"
            )


@dataclass(frozen=True)
class HistoricalRawTrainProjectionV2:
    receipt: HistoricalRawTrainSourceReceiptV2
    baseline_set: FrozenRawTrainBaselineSetV2
    source_observation_hashes: tuple[str, ...]
    projected_observation_hashes: tuple[str, ...]

    def verify(self) -> None:
        self.receipt.verify()
        self.baseline_set.verify()
        if (
            len(self.source_observation_hashes) != self.receipt.row_count
            or len(self.projected_observation_hashes) != self.receipt.row_count
            or stable_hash({"hashes": list(self.source_observation_hashes)})
            != self.receipt.source_observation_set_hash
            or stable_hash({"hashes": list(self.projected_observation_hashes)})
            != self.receipt.projected_observation_set_hash
            or len(set(self.source_observation_hashes))
            != self.receipt.row_count
            or len(set(self.projected_observation_hashes))
            != self.receipt.row_count
            or tuple(sorted(
                row.observation.observation_hash
                for row in self.baseline_set.rows
            ))
            != tuple(sorted(self.projected_observation_hashes))
            or self.baseline_set.source_train_receipt_hash
            != self.receipt.receipt_hash
            or self.baseline_set.source_raw_trial_artifact_row_count
            != self.receipt.row_count
            or sum(row.success for row in self.baseline_set.rows)
            != self.receipt.success_count
        ):
            raise HistoricalRawTrainProjectionError(
                "historical RAW TRAIN projection drifted"
            )


def _completed_event_by_request(
    events: Sequence[Mapping[str, Any]],
    *,
    source_trace_id: str,
) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    prefix = f"{source_trace_id}:"
    for event in events:
        if (
            event.get("event") != "skilllearn_trial_completed"
            or not str(event.get("trace_id") or "").startswith(prefix)
        ):
            continue
        payload = event["payload"]
        request_hash = payload.get("request_hash")
        _require_sha256(request_hash, "historical request hash")
        grouped.setdefault(str(request_hash), []).append(event)
    selected: dict[str, Mapping[str, Any]] = {}
    for request_hash, rows in grouped.items():
        valid_rows = [row for row in rows if row["payload"].get("valid") is True]
        if (
            len(valid_rows) != 1
            or rows[-1] is not valid_rows[0]
            or valid_rows[0]["payload"].get("error_type") is not None
        ):
            raise HistoricalRawTrainProjectionError(
                "historical TRAIN retry history is ambiguous"
            )
        selected[request_hash] = valid_rows[0]
    return selected


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HistoricalRawTrainProjectionError(
            f"historical {label} is not a nonnegative integer"
        )
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HistoricalRawTrainProjectionError(
            f"historical {label} is not numeric"
        )
    number = float(value)
    if not math.isfinite(number):
        raise HistoricalRawTrainProjectionError(
            f"historical {label} is not finite"
        )
    return number


def load_historical_raw_train_projection_v2(
    *,
    source_root: Path,
    manifest_path: Path,
    source_trace_id: str,
    evaluator_epoch: str,
    expected_source_observation_set_hash: str,
    expected_source_commit: str = HISTORICAL_SOURCE_COMMIT,
    expected_protocol_lock_file_sha256: str | None = None,
    expected_event_ledger_file_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> HistoricalRawTrainProjectionV2:
    """Project exact historical policy-off TRAIN evidence into today's schema.

    This adapter performs no model or evaluator calls.  It authenticates the old
    event observations first, then maps the three later-added empty prompt fields
    into the current observation schema.  Raw source files are read only to verify
    their content hashes, result reward, and bounded action-trace profile.
    """

    if not isinstance(source_trace_id, str) or not source_trace_id:
        raise HistoricalRawTrainProjectionError("source trace ID is empty")
    if not isinstance(evaluator_epoch, str) or not evaluator_epoch:
        raise HistoricalRawTrainProjectionError("evaluator epoch is empty")
    _require_sha256(
        expected_source_observation_set_hash,
        "expected source observation set hash",
    )
    if not _GIT_COMMIT.fullmatch(expected_source_commit):
        raise HistoricalRawTrainProjectionError(
            "expected source commit is not a git digest"
        )

    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise HistoricalRawTrainProjectionError(
            "historical source root is not a directory"
        )
    protocol_path = _verified_file(root, "protocol_lock.json")
    events_path = _verified_file(root, "development_recursive.events.jsonl")
    trials_root = (
        root / "development_recursive" / "upstream_trials"
    ).resolve(strict=True)
    try:
        trials_root.relative_to(root)
    except ValueError as exc:
        raise HistoricalRawTrainProjectionError(
            "historical trial root escaped the source root"
        ) from exc
    if not trials_root.is_dir():
        raise HistoricalRawTrainProjectionError(
            "historical trial root is absent"
        )

    manifest_file = manifest_path.resolve(strict=True)
    if not manifest_file.is_file() or manifest_file.is_symlink():
        raise HistoricalRawTrainProjectionError(
            "TRAIN manifest is not a regular frozen file"
        )
    protocol_sha256 = _file_sha256(protocol_path)
    events_sha256 = _file_sha256(events_path)
    manifest_sha256 = _file_sha256(manifest_file)
    for expected, actual, label in (
        (
            expected_protocol_lock_file_sha256,
            protocol_sha256,
            "protocol lock file",
        ),
        (
            expected_event_ledger_file_sha256,
            events_sha256,
            "event ledger file",
        ),
        (expected_manifest_file_sha256, manifest_sha256, "manifest file"),
    ):
        if expected is not None:
            _require_sha256(expected, f"expected {label} hash")
            if expected != actual:
                raise HistoricalRawTrainProjectionError(
                    f"historical {label} hash drifted"
                )

    protocol = _json_object(protocol_path, "historical protocol lock")
    lock_hash = protocol.get("lock_hash")
    protocol_without_hash = dict(protocol)
    protocol_without_hash.pop("lock_hash", None)
    manifest = _json_object(manifest_file, "TRAIN manifest")
    manifest_hash = manifest.get("manifest_hash")
    _require_sha256(manifest_hash, "TRAIN manifest hash")
    if (
        lock_hash != stable_hash(protocol_without_hash)
        or protocol.get("git", {}).get("commit") != expected_source_commit
        or protocol.get("git", {}).get("scoped_dirty") is not False
        or protocol.get("primary_manifest_hash") != manifest_hash
        or protocol.get("model") != "gpt-5.4-mini"
        or protocol.get("max_steps") != 100
        or protocol.get("claim_eligible") is not True
        or manifest.get("raw_content_persisted") is not False
        or manifest.get("sealed_test") is not True
    ):
        raise HistoricalRawTrainProjectionError(
            "historical protocol/manifest authority drifted"
        )
    family_by_id = manifest.get("family_by_id")
    train_ids = manifest.get("train_ids")
    if (
        not isinstance(family_by_id, dict)
        or not isinstance(train_ids, list)
        or not train_ids
        or len(set(train_ids)) != len(train_ids)
        or any(
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(family_by_id.get(item_id), str)
            or not family_by_id[item_id]
            for item_id in train_ids
        )
    ):
        raise HistoricalRawTrainProjectionError(
            "historical TRAIN manifest inventory is invalid"
        )

    events = _load_verified_events(events_path)
    evidence_rows = [
        row
        for row in events
        if row.get("event") == "training_evidence_recorded"
        and row.get("trace_id") == source_trace_id
    ]
    if len(evidence_rows) != 1:
        raise HistoricalRawTrainProjectionError(
            "historical TRAIN evidence receipt is not unique"
        )
    evidence_payload = evidence_rows[0]["payload"]
    if (
        evidence_payload.get("source_trace_id") != source_trace_id
        or evidence_payload.get("observation_count") != len(train_ids)
        or evidence_payload.get("new_training_executions") != len(train_ids)
        or evidence_payload.get("observation_set_hash")
        != expected_source_observation_set_hash
        or evidence_payload.get("sealed_test_accessed") is not False
        or evidence_payload.get("raw_content_persisted") is not False
    ):
        raise HistoricalRawTrainProjectionError(
            "historical TRAIN evidence receipt drifted"
        )
    completed_by_request = _completed_event_by_request(
        events,
        source_trace_id=source_trace_id,
    )

    program_set_hash = skilllearn_program_set_treatment_hash(())
    codex_policy_hash = protocol.get(
        "resolved_codex_agent_execution_policy_hash"
    )
    _require_sha256(codex_policy_hash, "historical Codex policy hash")
    observations: list[SkillLearnTrialObservation] = []
    source_hashes: list[str] = []
    projected_hashes: list[str] = []
    mapping_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []

    for item_id in train_ids:
        family = family_by_id[item_id]
        pair_id = stable_hash(
            {
                "trace_id": source_trace_id,
                "item_id": item_id,
                "stage": "training_baseline",
                "program_set_hash": program_set_hash,
                "treatment_hash": NO_SKILL_TREATMENT_HASH,
            }
        )[:20]
        request = SkillLearnTrialRequest(
            item_id=item_id,
            family=family,
            split=SplitName.TRAIN,
            variant=TrialVariant.POLICY_OFF,
            evaluator_epoch=evaluator_epoch,
            pair_id=pair_id,
            repeat=1,
            agent_id="codex",
            model="gpt-5.4-mini",
            max_steps=100,
            manifest_hash=manifest_hash,
            codex_agent_execution_policy_hash=codex_policy_hash,
            program_set_hash=program_set_hash,
            treatment_hash=NO_SKILL_TREATMENT_HASH,
        )
        event = completed_by_request.get(request.request_hash)
        if event is None:
            raise HistoricalRawTrainProjectionError(
                "historical TRAIN request coverage is incomplete"
            )
        payload = event["payload"]
        if (
            payload.get("variant") != TrialVariant.POLICY_OFF.value
            or payload.get("valid") is not True
            or payload.get("error_type") is not None
            or payload.get("raw_trial_artifacts_persisted") is not True
            or not isinstance(payload.get("metrics"), dict)
            or payload["metrics"].get("evaluation_valid") != 1.0
        ):
            raise HistoricalRawTrainProjectionError(
                "historical TRAIN completion is not valid RAW evidence"
            )
        trial_relative = (
            Path("development_recursive")
            / "upstream_trials"
            / "no_skill"
            / family
            / item_id
            / request.trial_id
        )
        result_path = _verified_file(
            root,
            str(trial_relative / "result.json"),
        )
        trace_path = _verified_file(
            root,
            str(trial_relative / "agent" / "codex.txt"),
        )
        result = _json_object(result_path, "historical trial result")
        score = _finite_number(result.get("reward"), "trial reward")
        if (
            result.get("task_id") != f"{family}/{item_id}"
            or result.get("trial_id") != request.trial_id
            or result.get("model") != request.model
            or result.get("skill_config") != "no_skill"
            or result.get("passed") is not payload.get("success")
            or score < 0.0
            or score > 1.0
        ):
            raise HistoricalRawTrainProjectionError(
                "historical trial result identity drifted"
            )
        action_trace = _extract_train_action_trace_profile(
            trace_path,
            containment_root=trials_root,
        )
        if not action_trace:
            raise HistoricalRawTrainProjectionError(
                "historical TRAIN action trace profile is absent"
            )
        observation = SkillLearnTrialObservation(
            request=request,
            success=bool(payload["success"]),
            score=score,
            metrics=MappingProxyType(dict(payload["metrics"])),
            total_tokens=_nonnegative_int(
                payload.get("total_tokens"),
                "total token count",
            ),
            steps=_nonnegative_int(payload.get("steps"), "step count"),
            duration_seconds=_finite_number(
                payload.get("duration_seconds"),
                "duration",
            ),
            provider_fingerprint=str(payload.get("provider_fingerprint") or ""),
            fairness_fingerprint=str(payload.get("fairness_fingerprint") or ""),
            error_type=None,
            upstream_result_hash=_require_sha256(
                payload.get("upstream_result_hash"),
                "historical upstream result hash",
            ),
            raw_trial_artifacts_persisted=True,
            prebuilt_image_key=str(payload.get("prebuilt_image_key") or ""),
            prebuilt_image_id=str(payload.get("prebuilt_image_id") or ""),
            prebuilt_cache_reused=bool(payload.get("prebuilt_cache_reused")),
            agent_runtime_key=str(payload.get("agent_runtime_key") or ""),
            agent_runtime_version=str(payload.get("agent_runtime_version") or ""),
            offline_verifier_profile_id=str(
                payload.get("offline_verifier_profile_id") or ""
            ),
            offline_verifier_runtime_key=str(
                payload.get("offline_verifier_runtime_key") or ""
            ),
            step_budget_policy=str(payload.get("step_budget_policy") or ""),
            step_budget_unit=str(payload.get("step_budget_unit") or ""),
            step_budget_limit=_nonnegative_int(
                payload.get("step_budget_limit"),
                "step budget limit",
            ),
            step_budget_truncated=bool(payload.get("step_budget_truncated")),
            step_budget_token_usage_complete=bool(
                payload.get("step_budget_token_usage_complete")
            ),
            step_budget_receipt_hash=_require_sha256(
                payload.get("step_budget_receipt_hash"),
                "historical step budget receipt hash",
            ),
            installed_skill_source_receipt_hash=str(
                payload.get("installed_skill_source_receipt_hash") or ""
            ),
            proposal_action_trace=MappingProxyType(dict(action_trace)),
        )
        source_observation_hash = historical_observation_hash_v2(observation)
        if source_observation_hash != payload.get("observation_hash"):
            raise HistoricalRawTrainProjectionError(
                "historical TRAIN observation hash did not reconstruct"
            )
        projected_observation_hash = observation.observation_hash
        observations.append(observation)
        source_hashes.append(source_observation_hash)
        projected_hashes.append(projected_observation_hash)
        mapping_rows.append(
            {
                "item_id_hash": stable_hash({"item_id": item_id}),
                "request_hash": request.request_hash,
                "source_observation_hash": source_observation_hash,
                "projected_observation_hash": projected_observation_hash,
                "success": observation.success,
                "score_units": int(round(score * 1_000_000)),
            }
        )
        artifact_rows.append(
            {
                "item_id_hash": stable_hash({"item_id": item_id}),
                "result_file_sha256": _file_sha256(result_path),
                "codex_trace_file_sha256": _file_sha256(trace_path),
            }
        )

    if set(completed_by_request) != {
        observation.request.request_hash for observation in observations
    }:
        raise HistoricalRawTrainProjectionError(
            "historical TRAIN completion set has unexpected requests"
        )
    source_observation_set_hash = stable_hash({"hashes": source_hashes})
    if source_observation_set_hash != expected_source_observation_set_hash:
        raise HistoricalRawTrainProjectionError(
            "historical TRAIN observation set hash drifted"
        )
    projected_observation_set_hash = stable_hash({"hashes": projected_hashes})
    receipt = HistoricalRawTrainSourceReceiptV2(
        manifest_hash=manifest_hash,
        source_commit=expected_source_commit,
        source_trace_id_hash=stable_hash({"trace_id": source_trace_id}),
        evaluator_epoch_hash=stable_hash(
            {"evaluator_epoch": evaluator_epoch}
        ),
        protocol_lock_file_sha256=protocol_sha256,
        event_ledger_file_sha256=events_sha256,
        manifest_file_sha256=manifest_sha256,
        raw_artifact_set_hash=stable_hash({"artifacts": artifact_rows}),
        source_observation_set_hash=source_observation_set_hash,
        projected_observation_set_hash=projected_observation_set_hash,
        projection_mapping_set_hash=stable_hash(
            {"mappings": mapping_rows}
        ),
        row_count=len(observations),
        success_count=sum(row.success for row in observations),
    )
    receipt.verify()
    baseline_set = FrozenRawTrainBaselineSetV2.from_observations(
        observations,
        manifest_hash=manifest_hash,
        evaluator_epoch=evaluator_epoch,
        source_train_receipt_hash=receipt.receipt_hash,
        expected_item_ids=train_ids,
    )
    result = HistoricalRawTrainProjectionV2(
        receipt=receipt,
        baseline_set=baseline_set,
        source_observation_hashes=tuple(source_hashes),
        projected_observation_hashes=tuple(projected_hashes),
    )
    result.verify()
    return result
