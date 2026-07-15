from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .models import stable_hash


TYPED_ASSIGNMENT_CONTRACT_VERSION = (
    "public_destination_content_evidence_bijection_v3"
)
PUBLIC_DESTINATION_PARSE_POLICY_V3 = (
    "closed_organize_subject_destinations_from_public_instruction_v3"
)
EVIDENCE_PROFILE_POLICY_V3 = (
    "bounded_offline_content_evidence_profile_v3"
)
TYPED_ASSIGNMENT_PLAN_POLICY_V3 = (
    "agent_authored_typed_bijection_harness_applied_v3"
)
PRE_AGENT_ASSIGNMENT_RECEIPT_POLICY_V3 = (
    "pre_agent_source_snapshot_and_evidence_receipt_v3"
)
ASSIGNMENT_RECONCILIATION_POLICY_V3 = (
    "post_agent_harness_apply_and_reopen_reconciliation_v3"
)

ORGANIZE_PUBLIC_DESTINATIONS_V3 = (
    "LLM",
    "trapped_ion_and_qc",
    "black_hole",
    "DNA",
    "music_history",
)
ORGANIZE_PUBLIC_DEFAULT_DESTINATION_V3 = "music_history"

POSITIVE_CONTENT_EVIDENCE_BASIS = "positive_content_evidence"
PUBLIC_DEFAULT_BASIS = "public_default"
TYPED_ASSIGNMENT_BASES_V3 = frozenset(
    {POSITIVE_CONTENT_EVIDENCE_BASIS, PUBLIC_DEFAULT_BASIS}
)

SUPPORTED_EVIDENCE_MEDIA_KINDS_V3 = frozenset({"pdf", "docx", "pptx"})
SUPPORTED_EVIDENCE_KINDS_V3 = frozenset(
    {"title", "first_pages", "document_text", "slide_text", "metadata"}
)
SUPPORTED_EXTRACTION_STATUSES_V3 = frozenset(
    {"extracted", "empty", "failed"}
)

MAX_SOURCE_NAME_BYTES_V3 = 1024
MAX_EVIDENCE_FRAGMENT_BYTES_V3 = 16 * 1024
MAX_EVIDENCE_FRAGMENTS_PER_FILE_V3 = 16
MAX_ASSIGNMENT_FILES_V3 = 512

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DESTINATION_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_PLAN_KEYS = frozenset({"contract_hash", "evidence_set_hash", "assignments"})
_ASSIGNMENT_KEYS = frozenset(
    {"file_id", "destination", "basis", "evidence_ids"}
)

TYPED_ASSIGNMENT_PLAN_SCHEMA_V3: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["contract_hash", "evidence_set_hash", "assignments"],
    "properties": {
        "contract_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "evidence_set_hash": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "assignments": {
            "type": "array",
            "maxItems": MAX_ASSIGNMENT_FILES_V3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "file_id",
                    "destination",
                    "basis",
                    "evidence_ids",
                ],
                "properties": {
                    "file_id": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                    "destination": {"type": "string"},
                    "basis": {
                        "enum": [
                            POSITIVE_CONTENT_EVIDENCE_BASIS,
                            PUBLIC_DEFAULT_BASIS,
                        ]
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                    },
                },
            },
        },
    },
}
TYPED_ASSIGNMENT_PLAN_SCHEMA_HASH_V3 = stable_hash(
    dict(TYPED_ASSIGNMENT_PLAN_SCHEMA_V3)
)


class TypedAssignmentContractError(ValueError):
    """A public destination, evidence profile, or assignment is invalid."""


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TypedAssignmentContractError(f"{label} is not a sha256 digest")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypedAssignmentContractError(
            f"{label} must be a nonnegative integer"
        )
    return value


def _require_exact_keys(
    value: object,
    *,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TypedAssignmentContractError(
            f"{label} must contain exactly {sorted(expected)}"
        )
    if any(not isinstance(key, str) for key in value):
        raise TypedAssignmentContractError(f"{label} keys must be strings")
    return value


def public_instruction_hash_v3(public_instruction: str) -> str:
    if not isinstance(public_instruction, str) or not public_instruction.strip():
        raise TypedAssignmentContractError("public instruction is empty")
    return stable_hash({"public_instruction": public_instruction})


def _contains_public_token(text: str, token: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _public_instruction_declares_default(public_instruction: str) -> bool:
    normalized = " ".join(public_instruction.lower().split())
    patterns = (
        r"(?:doesn['’]t|does not|cannot|can['’]t).{0,180}"
        r"first\s+(?:4|four).{0,180}(?:last one|music_history)",
        r"(?:doesn['’]t|does not|cannot|can['’]t).{0,180}"
        r"(?:other|remaining)\s+(?:4|four).{0,180}"
        r"(?:last one|music_history)",
        r"(?:default(?:ed)?|fallback|catch[- ]all).{0,120}music_history",
        r"music_history.{0,120}(?:default|fallback|catch[- ]all)",
        r"(?:unrelated|ambiguous|cannot be confidently categorized)"
        r".{0,180}music_history",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


@dataclass(frozen=True)
class PublicDestinationSpec:
    public_instruction_hash: str
    destinations: tuple[str, ...]
    default_destination: str | None

    @classmethod
    def from_public_instruction(
        cls,
        public_instruction: str,
    ) -> "PublicDestinationSpec":
        instruction_hash = public_instruction_hash_v3(public_instruction)
        missing = [
            destination
            for destination in ORGANIZE_PUBLIC_DESTINATIONS_V3
            if not _contains_public_token(public_instruction, destination)
        ]
        if missing:
            raise TypedAssignmentContractError(
                "public instruction does not declare the closed destination "
                f"set; missing {missing}"
            )
        default_destination = (
            ORGANIZE_PUBLIC_DEFAULT_DESTINATION_V3
            if _public_instruction_declares_default(public_instruction)
            else None
        )
        spec = cls(
            public_instruction_hash=instruction_hash,
            destinations=ORGANIZE_PUBLIC_DESTINATIONS_V3,
            default_destination=default_destination,
        )
        spec.verify()
        return spec

    @property
    def destination_spec_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "parse_policy": PUBLIC_DESTINATION_PARSE_POLICY_V3,
            "public_instruction_hash": self.public_instruction_hash,
            "destinations": list(self.destinations),
            "default_destination": self.default_destination,
            "destination_count": len(self.destinations),
            "destination_names_from_public_instruction": True,
            "hidden_evaluator_literals_used": False,
            "raw_public_instruction_persisted": False,
        }

    def verify(self, *, public_instruction: str | None = None) -> None:
        _require_sha256(
            self.public_instruction_hash,
            "public destination instruction hash",
        )
        if self.destinations != ORGANIZE_PUBLIC_DESTINATIONS_V3:
            raise TypedAssignmentContractError(
                "destination set is not the closed public organize set"
            )
        if any(not _DESTINATION_TOKEN.fullmatch(row) for row in self.destinations):
            raise TypedAssignmentContractError("destination token is invalid")
        if self.default_destination not in {
            None,
            ORGANIZE_PUBLIC_DEFAULT_DESTINATION_V3,
        }:
            raise TypedAssignmentContractError(
                "default destination is not public and closed"
            )
        if public_instruction is not None:
            if public_instruction_hash_v3(public_instruction) != (
                self.public_instruction_hash
            ):
                raise TypedAssignmentContractError(
                    "public instruction hash drifted"
                )
            if any(
                not _contains_public_token(public_instruction, destination)
                for destination in self.destinations
            ):
                raise TypedAssignmentContractError(
                    "public destination declaration drifted"
                )
            expected_default = (
                ORGANIZE_PUBLIC_DEFAULT_DESTINATION_V3
                if _public_instruction_declares_default(public_instruction)
                else None
            )
            if self.default_destination != expected_default:
                raise TypedAssignmentContractError(
                    "public destination specification drifted"
                )


@dataclass(frozen=True)
class ContentEvidence:
    ordinal: int
    kind: str
    text: str = field(repr=False)

    @property
    def text_sha256(self) -> str:
        return stable_hash({"evidence_text": self.text})

    def evidence_id(self, *, file_id: str) -> str:
        _require_sha256(file_id, "evidence file id")
        return stable_hash(
            {
                "policy": EVIDENCE_PROFILE_POLICY_V3,
                "file_id": file_id,
                "ordinal": self.ordinal,
                "kind": self.kind,
                "text_sha256": self.text_sha256,
            }
        )

    def safe_payload(self, *, file_id: str) -> dict[str, Any]:
        text_bytes = self.text.encode("utf-8")
        return {
            "evidence_id": self.evidence_id(file_id=file_id),
            "ordinal": self.ordinal,
            "kind": self.kind,
            "text_sha256": self.text_sha256,
            "text_size": len(text_bytes),
            "raw_text_persisted": False,
        }

    def agent_payload(self, *, file_id: str) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id(file_id=file_id),
            "kind": self.kind,
            "text": self.text,
        }

    def verify(self) -> None:
        _require_nonnegative_int(self.ordinal, "evidence ordinal")
        if self.kind not in SUPPORTED_EVIDENCE_KINDS_V3:
            raise TypedAssignmentContractError("evidence kind is unsupported")
        if not isinstance(self.text, str) or not self.text.strip():
            raise TypedAssignmentContractError("evidence text is empty")
        if len(self.text.encode("utf-8")) > MAX_EVIDENCE_FRAGMENT_BYTES_V3:
            raise TypedAssignmentContractError("evidence text exceeds byte limit")


def content_file_id(*, source_name: str, source_sha256: str) -> str:
    _require_sha256(source_sha256, "source sha256")
    if (
        not isinstance(source_name, str)
        or not source_name
        or len(source_name.encode("utf-8")) > MAX_SOURCE_NAME_BYTES_V3
        or PurePosixPath(source_name).name != source_name
        or source_name in {".", ".."}
    ):
        raise TypedAssignmentContractError(
            "source name must be one bounded basename"
        )
    return stable_hash(
        {
            "policy": EVIDENCE_PROFILE_POLICY_V3,
            "source_name": source_name,
            "source_sha256": source_sha256,
        }
    )


@dataclass(frozen=True)
class ContentEvidenceProfile:
    file_id: str
    source_name: str = field(compare=False, repr=False)
    source_sha256: str
    source_size: int
    media_kind: str
    extraction_status: str
    truncated: bool
    evidence: tuple[ContentEvidence, ...]

    @classmethod
    def from_extracted_text(
        cls,
        *,
        source_name: str,
        source_sha256: str,
        source_size: int,
        media_kind: str,
        evidence: Sequence[tuple[str, str]],
        extraction_status: str = "extracted",
        truncated: bool = False,
    ) -> "ContentEvidenceProfile":
        file_id = content_file_id(
            source_name=source_name,
            source_sha256=source_sha256,
        )
        profile = cls(
            file_id=file_id,
            source_name=source_name,
            source_sha256=source_sha256,
            source_size=source_size,
            media_kind=media_kind,
            extraction_status=extraction_status,
            truncated=truncated,
            evidence=tuple(
                ContentEvidence(ordinal=index, kind=kind, text=text)
                for index, (kind, text) in enumerate(evidence)
            ),
        )
        profile.verify()
        return profile

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(row.evidence_id(file_id=self.file_id) for row in self.evidence)

    @property
    def evidence_profile_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "profile_policy": EVIDENCE_PROFILE_POLICY_V3,
            "file_id": self.file_id,
            "source_name_hash": stable_hash({"source_name": self.source_name}),
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "media_kind": self.media_kind,
            "extraction_status": self.extraction_status,
            "truncated": self.truncated,
            "evidence_count": len(self.evidence),
            "evidence": [
                row.safe_payload(file_id=self.file_id) for row in self.evidence
            ],
            "raw_source_name_persisted": False,
            "raw_content_persisted": False,
            "network_accessed": False,
        }

    def agent_payload(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "source_name": self.source_name,
            "media_kind": self.media_kind,
            "extraction_status": self.extraction_status,
            "evidence": [
                row.agent_payload(file_id=self.file_id) for row in self.evidence
            ],
        }

    def verify(self) -> None:
        _require_sha256(self.file_id, "content evidence file id")
        _require_sha256(self.source_sha256, "content evidence source sha256")
        _require_nonnegative_int(self.source_size, "content evidence source size")
        expected_file_id = content_file_id(
            source_name=self.source_name,
            source_sha256=self.source_sha256,
        )
        if self.file_id != expected_file_id:
            raise TypedAssignmentContractError("content evidence file id drifted")
        if self.media_kind not in SUPPORTED_EVIDENCE_MEDIA_KINDS_V3:
            raise TypedAssignmentContractError("content evidence media kind invalid")
        if self.extraction_status not in SUPPORTED_EXTRACTION_STATUSES_V3:
            raise TypedAssignmentContractError("extraction status is invalid")
        if not isinstance(self.truncated, bool):
            raise TypedAssignmentContractError("truncated must be boolean")
        if len(self.evidence) > MAX_EVIDENCE_FRAGMENTS_PER_FILE_V3:
            raise TypedAssignmentContractError("too many evidence fragments")
        for expected_ordinal, row in enumerate(self.evidence):
            row.verify()
            if row.ordinal != expected_ordinal:
                raise TypedAssignmentContractError(
                    "evidence ordinals must be contiguous and canonical"
                )
        if self.extraction_status == "extracted" and not self.evidence:
            raise TypedAssignmentContractError(
                "extracted profile has no positive evidence"
            )
        if self.extraction_status != "extracted" and self.evidence:
            raise TypedAssignmentContractError(
                "non-extracted profile cannot claim positive evidence"
            )


def canonicalize_evidence_profiles(
    profiles: Sequence[ContentEvidenceProfile],
) -> tuple[ContentEvidenceProfile, ...]:
    if not profiles or len(profiles) > MAX_ASSIGNMENT_FILES_V3:
        raise TypedAssignmentContractError(
            "evidence profile count is outside the closed bound"
        )
    ordered = tuple(sorted(profiles, key=lambda row: row.file_id))
    if len({row.file_id for row in ordered}) != len(ordered):
        raise TypedAssignmentContractError("evidence profile file ids repeat")
    for row in ordered:
        row.verify()
    return ordered


def evidence_set_hash(profiles: Sequence[ContentEvidenceProfile]) -> str:
    ordered = canonicalize_evidence_profiles(profiles)
    return stable_hash(
        {
            "profile_policy": EVIDENCE_PROFILE_POLICY_V3,
            "evidence_profile_hashes": [
                row.evidence_profile_hash for row in ordered
            ],
        }
    )


def typed_assignment_contract_hash(
    *,
    destination_spec: PublicDestinationSpec,
    profiles: Sequence[ContentEvidenceProfile],
) -> str:
    destination_spec.verify()
    return stable_hash(
        {
            "contract_version": TYPED_ASSIGNMENT_CONTRACT_VERSION,
            "destination_spec_hash": destination_spec.destination_spec_hash,
            "evidence_set_hash": evidence_set_hash(profiles),
            "plan_schema_hash": TYPED_ASSIGNMENT_PLAN_SCHEMA_HASH_V3,
        }
    )


@dataclass(frozen=True)
class TypedFileAssignment:
    file_id: str
    destination: str
    basis: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(sorted(self.evidence_ids)))

    @classmethod
    def from_agent_payload(cls, payload: object) -> "TypedFileAssignment":
        row = _require_exact_keys(
            payload,
            expected=_ASSIGNMENT_KEYS,
            label="typed file assignment",
        )
        raw_evidence_ids = row["evidence_ids"]
        if not isinstance(raw_evidence_ids, list) or any(
            not isinstance(value, str) for value in raw_evidence_ids
        ):
            raise TypedAssignmentContractError(
                "assignment evidence_ids must be a JSON string array"
            )
        assignment = cls(
            file_id=row["file_id"] if isinstance(row["file_id"], str) else "",
            destination=(
                row["destination"]
                if isinstance(row["destination"], str)
                else ""
            ),
            basis=row["basis"] if isinstance(row["basis"], str) else "",
            evidence_ids=tuple(raw_evidence_ids),
        )
        assignment.verify_shape()
        return assignment

    def agent_payload(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "destination": self.destination,
            "basis": self.basis,
            "evidence_ids": list(self.evidence_ids),
        }

    def safe_payload(self) -> dict[str, Any]:
        return self.agent_payload()

    def verify_shape(self) -> None:
        _require_sha256(self.file_id, "assignment file id")
        if not _DESTINATION_TOKEN.fullmatch(self.destination):
            raise TypedAssignmentContractError("assignment destination invalid")
        if self.basis not in TYPED_ASSIGNMENT_BASES_V3:
            raise TypedAssignmentContractError("assignment basis invalid")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise TypedAssignmentContractError("assignment evidence ids repeat")
        for evidence_id in self.evidence_ids:
            _require_sha256(evidence_id, "assignment evidence id")


@dataclass(frozen=True)
class TypedAssignmentPlan:
    contract_hash: str
    evidence_set_hash: str
    assignments: tuple[TypedFileAssignment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assignments",
            tuple(sorted(self.assignments, key=lambda row: row.file_id)),
        )

    @classmethod
    def from_agent_payload(
        cls,
        payload: object,
        *,
        destination_spec: PublicDestinationSpec,
        profiles: Sequence[ContentEvidenceProfile],
    ) -> "TypedAssignmentPlan":
        plan_payload = _require_exact_keys(
            payload,
            expected=_PLAN_KEYS,
            label="typed assignment plan",
        )
        raw_assignments = plan_payload["assignments"]
        if not isinstance(raw_assignments, list):
            raise TypedAssignmentContractError(
                "typed assignment plan assignments must be a JSON array"
            )
        plan = cls(
            contract_hash=(
                plan_payload["contract_hash"]
                if isinstance(plan_payload["contract_hash"], str)
                else ""
            ),
            evidence_set_hash=(
                plan_payload["evidence_set_hash"]
                if isinstance(plan_payload["evidence_set_hash"], str)
                else ""
            ),
            assignments=tuple(
                TypedFileAssignment.from_agent_payload(row)
                for row in raw_assignments
            ),
        )
        plan.verify(destination_spec=destination_spec, profiles=profiles)
        return plan

    @property
    def plan_hash(self) -> str:
        return stable_hash(
            {
                "plan_policy": TYPED_ASSIGNMENT_PLAN_POLICY_V3,
                **self.agent_payload(),
            }
        )

    @property
    def assignment_map_hash(self) -> str:
        return stable_hash(
            {
                "assignments": [
                    {
                        "file_id": row.file_id,
                        "destination": row.destination,
                    }
                    for row in self.assignments
                ]
            }
        )

    def agent_payload(self) -> dict[str, Any]:
        return {
            "contract_hash": self.contract_hash,
            "evidence_set_hash": self.evidence_set_hash,
            "assignments": [row.agent_payload() for row in self.assignments],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.agent_payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"

    def safe_payload(self) -> dict[str, Any]:
        return {
            "plan_policy": TYPED_ASSIGNMENT_PLAN_POLICY_V3,
            **self.agent_payload(),
            "assignment_count": len(self.assignments),
            "plan_schema_hash": TYPED_ASSIGNMENT_PLAN_SCHEMA_HASH_V3,
            "plan_hash": self.plan_hash,
            "raw_file_names_or_content_persisted": False,
        }

    def verify(
        self,
        *,
        destination_spec: PublicDestinationSpec,
        profiles: Sequence[ContentEvidenceProfile],
    ) -> None:
        destination_spec.verify()
        ordered_profiles = canonicalize_evidence_profiles(profiles)
        _require_sha256(self.contract_hash, "typed assignment contract hash")
        _require_sha256(self.evidence_set_hash, "typed evidence set hash")
        if self.contract_hash != typed_assignment_contract_hash(
            destination_spec=destination_spec,
            profiles=ordered_profiles,
        ):
            raise TypedAssignmentContractError(
                "typed assignment contract hash drifted"
            )
        if self.evidence_set_hash != evidence_set_hash(ordered_profiles):
            raise TypedAssignmentContractError("evidence set hash drifted")
        if len(self.assignments) != len(ordered_profiles):
            raise TypedAssignmentContractError(
                "typed assignment plan does not cover every source file"
            )
        assignment_ids = [row.file_id for row in self.assignments]
        profile_ids = [row.file_id for row in ordered_profiles]
        if len(set(assignment_ids)) != len(assignment_ids):
            raise TypedAssignmentContractError("source file assigned more than once")
        if assignment_ids != profile_ids:
            raise TypedAssignmentContractError(
                "typed assignment coverage differs from evidence files"
            )
        profiles_by_id = {row.file_id: row for row in ordered_profiles}
        for assignment in self.assignments:
            assignment.verify_shape()
            profile = profiles_by_id[assignment.file_id]
            if assignment.destination not in destination_spec.destinations:
                raise TypedAssignmentContractError(
                    "assignment destination is outside the public set"
                )
            profile_evidence_ids = set(profile.evidence_ids)
            if assignment.basis == POSITIVE_CONTENT_EVIDENCE_BASIS:
                if not assignment.evidence_ids:
                    raise TypedAssignmentContractError(
                        "positive assignment has no evidence"
                    )
                if not set(assignment.evidence_ids).issubset(
                    profile_evidence_ids
                ):
                    raise TypedAssignmentContractError(
                        "assignment cites evidence from another file"
                    )
            elif (
                destination_spec.default_destination is None
                or assignment.destination
                != destination_spec.default_destination
                or assignment.evidence_ids
            ):
                raise TypedAssignmentContractError(
                    "public default assignment is not authorized by the task"
                )


def parse_typed_assignment_plan(
    payload: object,
    *,
    destination_spec: PublicDestinationSpec,
    profiles: Sequence[ContentEvidenceProfile],
) -> TypedAssignmentPlan:
    return TypedAssignmentPlan.from_agent_payload(
        payload,
        destination_spec=destination_spec,
        profiles=profiles,
    )


@dataclass(frozen=True)
class PreAgentAssignmentReceipt:
    request_hash: str
    public_instruction_hash: str
    destination_spec_hash: str
    assignment_contract_hash: str
    evidence_set_hash: str
    evidence_profile_hashes: tuple[str, ...]
    source_tree_hash_before: str
    source_tree_hash_after_preparation: str
    source_file_count: int
    evidence_artifact_sha256: str
    evidence_artifact_size: int
    evidence_artifact_locator_hash: str
    plan_artifact_locator_hash: str

    @classmethod
    def create(
        cls,
        *,
        request_hash: str,
        destination_spec: PublicDestinationSpec,
        profiles: Sequence[ContentEvidenceProfile],
        source_tree_hash_before: str,
        source_tree_hash_after_preparation: str,
        evidence_artifact_sha256: str,
        evidence_artifact_size: int,
        evidence_artifact_locator_hash: str,
        plan_artifact_locator_hash: str,
    ) -> "PreAgentAssignmentReceipt":
        ordered = canonicalize_evidence_profiles(profiles)
        receipt = cls(
            request_hash=request_hash,
            public_instruction_hash=destination_spec.public_instruction_hash,
            destination_spec_hash=destination_spec.destination_spec_hash,
            assignment_contract_hash=typed_assignment_contract_hash(
                destination_spec=destination_spec,
                profiles=ordered,
            ),
            evidence_set_hash=evidence_set_hash(ordered),
            evidence_profile_hashes=tuple(
                row.evidence_profile_hash for row in ordered
            ),
            source_tree_hash_before=source_tree_hash_before,
            source_tree_hash_after_preparation=source_tree_hash_after_preparation,
            source_file_count=len(ordered),
            evidence_artifact_sha256=evidence_artifact_sha256,
            evidence_artifact_size=evidence_artifact_size,
            evidence_artifact_locator_hash=evidence_artifact_locator_hash,
            plan_artifact_locator_hash=plan_artifact_locator_hash,
        )
        receipt.verify(destination_spec=destination_spec, profiles=ordered)
        return receipt

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "receipt_policy": PRE_AGENT_ASSIGNMENT_RECEIPT_POLICY_V3,
            "request_hash": self.request_hash,
            "public_instruction_hash": self.public_instruction_hash,
            "destination_spec_hash": self.destination_spec_hash,
            "assignment_contract_hash": self.assignment_contract_hash,
            "evidence_set_hash": self.evidence_set_hash,
            "evidence_profile_hashes": list(self.evidence_profile_hashes),
            "source_tree_hash_before": self.source_tree_hash_before,
            "source_tree_hash_after_preparation": (
                self.source_tree_hash_after_preparation
            ),
            "source_file_count": self.source_file_count,
            "evidence_artifact_sha256": self.evidence_artifact_sha256,
            "evidence_artifact_size": self.evidence_artifact_size,
            "evidence_artifact_locator_hash": (
                self.evidence_artifact_locator_hash
            ),
            "plan_artifact_locator_hash": self.plan_artifact_locator_hash,
            "plan_schema_hash": TYPED_ASSIGNMENT_PLAN_SCHEMA_HASH_V3,
            "source_tree_unchanged_during_preparation": (
                self.source_tree_hash_before
                == self.source_tree_hash_after_preparation
            ),
            "content_extraction_completed_before_agent_start": True,
            "task_input_mutated": False,
            "network_accessed": False,
            "tests_solution_or_verifier_accessed": False,
            "raw_file_names_or_content_persisted": False,
        }

    def verify(
        self,
        *,
        destination_spec: PublicDestinationSpec | None = None,
        profiles: Sequence[ContentEvidenceProfile] | None = None,
    ) -> None:
        for label, value in (
            ("request hash", self.request_hash),
            ("public instruction hash", self.public_instruction_hash),
            ("destination spec hash", self.destination_spec_hash),
            ("assignment contract hash", self.assignment_contract_hash),
            ("evidence set hash", self.evidence_set_hash),
            ("source tree hash before", self.source_tree_hash_before),
            (
                "source tree hash after preparation",
                self.source_tree_hash_after_preparation,
            ),
            ("evidence artifact sha256", self.evidence_artifact_sha256),
            (
                "evidence artifact locator hash",
                self.evidence_artifact_locator_hash,
            ),
            ("plan artifact locator hash", self.plan_artifact_locator_hash),
        ):
            _require_sha256(value, label)
        _require_nonnegative_int(self.source_file_count, "source file count")
        _require_nonnegative_int(
            self.evidence_artifact_size,
            "evidence artifact size",
        )
        if self.source_file_count == 0:
            raise TypedAssignmentContractError("pre-agent source set is empty")
        if self.evidence_artifact_size == 0:
            raise TypedAssignmentContractError("evidence artifact is empty")
        if (
            len(set(self.evidence_profile_hashes))
            != len(self.evidence_profile_hashes)
            or len(self.evidence_profile_hashes) != self.source_file_count
        ):
            raise TypedAssignmentContractError(
                "pre-agent evidence profile hash set is not canonical"
            )
        for value in self.evidence_profile_hashes:
            _require_sha256(value, "pre-agent evidence profile hash")
        if self.source_tree_hash_before != self.source_tree_hash_after_preparation:
            raise TypedAssignmentContractError(
                "source tree changed during evidence preparation"
            )
        if destination_spec is not None:
            destination_spec.verify()
            if (
                self.public_instruction_hash
                != destination_spec.public_instruction_hash
                or self.destination_spec_hash
                != destination_spec.destination_spec_hash
            ):
                raise TypedAssignmentContractError(
                    "pre-agent destination binding drifted"
                )
        if profiles is not None:
            if destination_spec is None:
                raise TypedAssignmentContractError(
                    "destination spec required when verifying profiles"
                )
            ordered = canonicalize_evidence_profiles(profiles)
            if self.evidence_profile_hashes != tuple(
                row.evidence_profile_hash for row in ordered
            ):
                raise TypedAssignmentContractError(
                    "pre-agent evidence profile hashes drifted"
                )
            if self.evidence_set_hash != evidence_set_hash(ordered):
                raise TypedAssignmentContractError(
                    "pre-agent evidence set hash drifted"
                )
            if self.assignment_contract_hash != typed_assignment_contract_hash(
                destination_spec=destination_spec,
                profiles=ordered,
            ):
                raise TypedAssignmentContractError(
                    "pre-agent assignment contract hash drifted"
                )


@dataclass(frozen=True)
class AssignmentReconciliationReceipt:
    request_hash: str
    assignment_contract_hash: str
    evidence_set_hash: str
    plan_hash: str
    expected_assignment_map_hash: str
    reopened_assignment_map_hash: str
    source_tree_hash_before_agent: str
    source_tree_hash_before_apply: str
    source_tree_hash_after_apply: str
    expected_file_count: int
    applied_file_count: int
    reopened_file_count: int
    source_file_count_after_apply: int

    @classmethod
    def create(
        cls,
        *,
        pre_agent_receipt: PreAgentAssignmentReceipt,
        plan: TypedAssignmentPlan,
        reopened_assignments: Mapping[str, str],
        source_tree_hash_before_apply: str,
        source_tree_hash_after_apply: str,
        applied_file_count: int,
        source_file_count_after_apply: int,
    ) -> "AssignmentReconciliationReceipt":
        if not isinstance(reopened_assignments, Mapping) or any(
            not isinstance(file_id, str)
            or not isinstance(destination, str)
            for file_id, destination in reopened_assignments.items()
        ):
            raise TypedAssignmentContractError(
                "reopened assignments must map file ids to destinations"
            )
        reopened_rows = sorted(reopened_assignments.items())
        reopened_hash = stable_hash(
            {
                "assignments": [
                    {"file_id": file_id, "destination": destination}
                    for file_id, destination in reopened_rows
                ]
            }
        )
        receipt = cls(
            request_hash=pre_agent_receipt.request_hash,
            assignment_contract_hash=plan.contract_hash,
            evidence_set_hash=plan.evidence_set_hash,
            plan_hash=plan.plan_hash,
            expected_assignment_map_hash=plan.assignment_map_hash,
            reopened_assignment_map_hash=reopened_hash,
            source_tree_hash_before_agent=(
                pre_agent_receipt.source_tree_hash_before
            ),
            source_tree_hash_before_apply=source_tree_hash_before_apply,
            source_tree_hash_after_apply=source_tree_hash_after_apply,
            expected_file_count=len(plan.assignments),
            applied_file_count=applied_file_count,
            reopened_file_count=len(reopened_rows),
            source_file_count_after_apply=source_file_count_after_apply,
        )
        receipt.verify(pre_agent_receipt=pre_agent_receipt, plan=plan)
        return receipt

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "receipt_policy": ASSIGNMENT_RECONCILIATION_POLICY_V3,
            "request_hash": self.request_hash,
            "assignment_contract_hash": self.assignment_contract_hash,
            "evidence_set_hash": self.evidence_set_hash,
            "plan_hash": self.plan_hash,
            "expected_assignment_map_hash": self.expected_assignment_map_hash,
            "reopened_assignment_map_hash": self.reopened_assignment_map_hash,
            "source_tree_hash_before_agent": self.source_tree_hash_before_agent,
            "source_tree_hash_before_apply": self.source_tree_hash_before_apply,
            "source_tree_hash_after_apply": self.source_tree_hash_after_apply,
            "expected_file_count": self.expected_file_count,
            "applied_file_count": self.applied_file_count,
            "reopened_file_count": self.reopened_file_count,
            "source_file_count_after_apply": (
                self.source_file_count_after_apply
            ),
            "source_tree_unchanged_before_harness_apply": (
                self.source_tree_hash_before_agent
                == self.source_tree_hash_before_apply
            ),
            "source_collection_empty_after_apply": (
                self.source_file_count_after_apply == 0
            ),
            "destination_layout_reopened": True,
            "exact_assignment_reconciliation": (
                self.expected_assignment_map_hash
                == self.reopened_assignment_map_hash
                and self.expected_file_count
                == self.applied_file_count
                == self.reopened_file_count
            ),
            "harness_authored_assignment": False,
            "tests_solution_or_verifier_accessed": False,
            "network_accessed": False,
            "raw_file_names_or_content_persisted": False,
        }

    def verify(
        self,
        *,
        pre_agent_receipt: PreAgentAssignmentReceipt | None = None,
        plan: TypedAssignmentPlan | None = None,
    ) -> None:
        for label, value in (
            ("request hash", self.request_hash),
            ("assignment contract hash", self.assignment_contract_hash),
            ("evidence set hash", self.evidence_set_hash),
            ("plan hash", self.plan_hash),
            ("expected assignment map hash", self.expected_assignment_map_hash),
            ("reopened assignment map hash", self.reopened_assignment_map_hash),
            ("source tree hash before agent", self.source_tree_hash_before_agent),
            ("source tree hash before apply", self.source_tree_hash_before_apply),
            ("source tree hash after apply", self.source_tree_hash_after_apply),
        ):
            _require_sha256(value, label)
        for label, value in (
            ("expected file count", self.expected_file_count),
            ("applied file count", self.applied_file_count),
            ("reopened file count", self.reopened_file_count),
            (
                "source file count after apply",
                self.source_file_count_after_apply,
            ),
        ):
            _require_nonnegative_int(value, label)
        if self.expected_file_count == 0:
            raise TypedAssignmentContractError(
                "reconciliation expected file set is empty"
            )
        if self.source_tree_hash_before_agent != self.source_tree_hash_before_apply:
            raise TypedAssignmentContractError(
                "agent mutated the source tree before harness apply"
            )
        if self.source_file_count_after_apply != 0:
            raise TypedAssignmentContractError(
                "source collection is not empty after harness apply"
            )
        if not (
            self.expected_file_count
            == self.applied_file_count
            == self.reopened_file_count
        ):
            raise TypedAssignmentContractError(
                "applied/reopened assignment counts differ"
            )
        if self.expected_assignment_map_hash != self.reopened_assignment_map_hash:
            raise TypedAssignmentContractError(
                "reopened assignment map differs from the typed plan"
            )
        if pre_agent_receipt is not None:
            pre_agent_receipt.verify()
            if (
                self.request_hash != pre_agent_receipt.request_hash
                or self.assignment_contract_hash
                != pre_agent_receipt.assignment_contract_hash
                or self.evidence_set_hash != pre_agent_receipt.evidence_set_hash
                or self.source_tree_hash_before_agent
                != pre_agent_receipt.source_tree_hash_before
            ):
                raise TypedAssignmentContractError(
                    "reconciliation pre-agent binding drifted"
                )
        if plan is not None:
            if (
                self.assignment_contract_hash != plan.contract_hash
                or self.evidence_set_hash != plan.evidence_set_hash
                or self.plan_hash != plan.plan_hash
                or self.expected_assignment_map_hash
                != plan.assignment_map_hash
                or self.expected_file_count != len(plan.assignments)
            ):
                raise TypedAssignmentContractError(
                    "reconciliation plan binding drifted"
                )


__all__ = [
    "ASSIGNMENT_RECONCILIATION_POLICY_V3",
    "AssignmentReconciliationReceipt",
    "ContentEvidence",
    "ContentEvidenceProfile",
    "EVIDENCE_PROFILE_POLICY_V3",
    "ORGANIZE_PUBLIC_DEFAULT_DESTINATION_V3",
    "ORGANIZE_PUBLIC_DESTINATIONS_V3",
    "POSITIVE_CONTENT_EVIDENCE_BASIS",
    "PRE_AGENT_ASSIGNMENT_RECEIPT_POLICY_V3",
    "PUBLIC_DEFAULT_BASIS",
    "PUBLIC_DESTINATION_PARSE_POLICY_V3",
    "PreAgentAssignmentReceipt",
    "PublicDestinationSpec",
    "TYPED_ASSIGNMENT_CONTRACT_VERSION",
    "TYPED_ASSIGNMENT_PLAN_POLICY_V3",
    "TYPED_ASSIGNMENT_PLAN_SCHEMA_HASH_V3",
    "TYPED_ASSIGNMENT_PLAN_SCHEMA_V3",
    "TypedAssignmentContractError",
    "TypedAssignmentPlan",
    "TypedFileAssignment",
    "canonicalize_evidence_profiles",
    "content_file_id",
    "evidence_set_hash",
    "parse_typed_assignment_plan",
    "public_instruction_hash_v3",
    "typed_assignment_contract_hash",
]
