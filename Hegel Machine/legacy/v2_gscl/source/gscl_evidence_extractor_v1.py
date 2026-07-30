"""Law-agnostic controlled evidence extraction and GSCL CSP binding.

The runtime has two deliberately separate stages.

``extract_structural_episode`` parses exact UTF-8 bytes into atomic facts,
field-level evidence spans, a law-neutral structural episode, and zero or
more *proposals*.  A proposal is only an auditable derivation record: it
names the facts, field spans, and multi-fact conditions that made a law
schema worth trying.  It never contains a role binding, observable, derived
constraint, or acceptance decision.

``bind_structural_episode`` solves each proposal as a bounded CSP.  Types,
incidence, ownership, units, temporal orientation, finite-map composition,
and subset-lattice coverage are checked before any canonical GSCL role is
written.  Only one global solution is returned.  Zero solutions reject and
multiple solutions abstain; both are fail closed.

This remains a controlled grammar qualification harness, not an open-domain
NLP extractor and not an efficacy evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .generalized_structural_correspondence_v1 import (
    ConstraintParticipant,
    EvidenceSpanRef,
    ExactRational,
    GSCLSchemaRegistry,
    HyperRoleEndpoint,
    InferenceProvenance,
    LawBinding,
    ObservableBinding,
    ObservationStatus,
    RoleBinding,
    RoleTargetKind,
    StructuralConstraint,
    StructuralEpisode,
    StructuralHyperrelation,
    StructuralObject,
    StructuralQuantity,
    StructuralRelation,
    TypedObservable,
    strict_content_hash,
    validate_law_binding,
)


EXTRACTOR_VERSION = "gscl.controlled.atomic.extractor.v3"
JSONL_MEDIA_TYPE = "application/vnd.gscl-neutral-records+jsonl"
PROSE_MEDIA_TYPE = "text/vnd.gscl-neutral-records"
SUPPORTED_MEDIA_TYPES = (JSONL_MEDIA_TYPE, PROSE_MEDIA_TYPE)
MAX_SOURCE_BYTES = 262_144
MAX_RECORD_BYTES = 16_384
MAX_RECORDS = 1_024
MAX_BINDING_ASSIGNMENTS = 4_096
MAX_BINDINGS_PER_LAW = 5

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{2,127}\Z")
_PROSE_A = re.compile(
    r"Fact (?P<kind>[a-z_]+) (?P<record_id>[a-z][a-z0-9_.-]{2,127}) "
    r"carries (?P<attrs>\{.*\})\.\Z"
)
_PROSE_B = re.compile(
    r"Record (?P<kind>[a-z_]+) named "
    r"(?P<record_id>[a-z][a-z0-9_.-]{2,127}) "
    r"has (?P<attrs>\{.*\})\.\Z"
)

T05 = "gscl.v1.t05_pair_interaction"
T09 = "gscl.v1.t09_path_composition"
T14 = "gscl.v1.t14_finite_equivariance"
T15 = "gscl.v1.t15_closed_balance"
T17 = "gscl.v1.t17_monotone_order"

_KIND_ALIASES = {
    "assertion": "assertion",
    "claim": "assertion",
    "edge": "edge",
    "entity": "node",
    "link": "edge",
    "map": "map",
    "mapping": "map",
    "movement": "transfer",
    "node": "node",
    "observation": "observation",
    "outcome": "subset_outcome",
    "reading": "observation",
    "statement": "assertion",
    "subset_result": "subset_outcome",
    "subset_outcome": "subset_outcome",
    "transfer": "transfer",
}

_ATTR_ALIASES: Mapping[str, Mapping[str, str]] = {
    "node": {
        "class": "sort",
        "category": "sort",
        "timepoint": "phase",
        "stage_alias": "phase",
        "endpoint_role": "endpoint",
        "route_kind": "path_kind",
        "index": "ordinal",
        "scope_ref": "boundary",
        "position": "ordinal",
    },
    "edge": {
        "class": "relation",
        "relation_type": "relation",
        "from": "source",
        "to": "target",
        "position": "order",
    },
    "observation": {
        "holder": "owner",
        "subject": "owner",
        "timepoint": "phase",
        "stage_alias": "phase",
        "axis": "dimension",
        "scale": "unit",
        "reading": "value",
        "entries": "values",
        "items": "values",
    },
    "map": {
        "lane": "stage",
        "mapping_mode": "mode",
        "pairs": "rows",
        "rows_alias": "rows",
        "reindex": "permutation",
        "orientations": "signs",
    },
    "transfer": {
        "holder": "boundary",
        "scope_ref": "boundary",
        "orientation": "direction",
        "reading": "amount",
        "magnitude": "amount",
        "axis": "dimension",
        "scale": "unit",
    },
    "subset_outcome": {
        "replicate": "fold",
        "participants": "subset",
        "items": "subset",
        "reading": "utility",
        "score": "utility",
        "axis": "dimension",
        "scale": "unit",
    },
    "assertion": {
        "purpose_alias": "predicate",
        "claim": "predicate",
        "participants": "members",
        "items": "members",
        "to": "target",
        "destination": "target",
        "reading": "value",
        "entries": "values",
    },
}

_NESTED_KEY_ALIASES = {
    "from": "source",
    "to": "target",
}

_FIELD_VALUE_ALIASES: Mapping[str, Mapping[str, str]] = {
    "sort": {
        "entity_record": "entity",
        "generic_entity": "entity",
    },
    "relation": {
        "directed_link": "directed",
        "relation_record": "directed",
    },
    "phase": {
        "afterward": "after",
        "beforehand": "before",
        "earlier": "before",
        "later": "after",
    },
    "endpoint": {
        "created": "source",
        "destination": "target",
        "origin": "source",
    },
    "path_kind": {
        "multi_step": "chained",
        "single_step": "direct",
        "route": "chained",
        "shortcut": "direct",
    },
    "stage": {
        "initial": "step_a",
        "terminal": "step_b",
        "comparison": "reference",
    },
    "direction": {
        "created": "source",
        "incoming": "inflow",
        "outgoing": "outflow",
        "removed": "sink",
    },
    "predicate": {
        "group": "association",
        "multiway": "association",
        "record_bundle": "bundle",
        "scalar_claim": "scalar",
        "set_claim": "set",
        "value_claim": "property",
    },
}

_ALLOWED_ATTRS = {
    "node": frozenset(
        {
        "sort",
            "phase",
            "endpoint",
            "path_kind",
            "ordinal",
            "boundary",
        }
    ),
    "edge": frozenset({"relation", "source", "target", "order"}),
    "observation": frozenset(
        {"owner", "phase", "dimension", "unit", "value", "values"}
    ),
    "map": frozenset(
        {"stage", "mode", "rows", "permutation", "signs"}
    ),
    "transfer": frozenset(
        {"boundary", "direction", "amount", "dimension", "unit"}
    ),
    "subset_outcome": frozenset(
        {"fold", "subset", "utility", "dimension", "unit"}
    ),
    "assertion": frozenset(
        {"predicate", "members", "target", "value", "values"}
    ),
}

EXTRACTOR_CONTRACT = {
    "version": EXTRACTOR_VERSION,
    "media_types": list(SUPPORTED_MEDIA_TYPES),
    "raw_fact_kinds": sorted(_ALLOWED_ATTRS),
    "law_agnostic_extraction": True,
    "canonical_roles_written_only_after_unique_csp": True,
    "proposal_may_be_multiple": True,
    "proposal_provenance": [
        "source_sha256",
        "fact_ids",
        "fact_content_hash",
        "condition_ids",
        "condition_evidence",
        "field_span_ids",
        "span_content_hash",
        "derivation_hash",
    ],
    "binder_recomputes_proposal_linkage": True,
    "raw_object_and_relation_types": "generic_until_unique_csp",
    "constraint_materialization": "derived_after_unique_csp",
    "exact_record_and_field_utf8_byte_spans": True,
    "alias_normalization": "kind_and_field_aware_collision_rejecting",
    "note_text_is_not_a_fact_kind": True,
    "missingness": "absence_of_required_atomic_fact_only",
    "missingness_declaration_allowed": False,
    "missing_required_role": "no_binding_fail_closed_abstain",
    "binding_search": {
        "algorithm": "bounded_types_incidence_owner_units_time_maps_subsets_csp",
        "max_assignments": MAX_BINDING_ASSIGNMENTS,
        "zero_solutions": "reject",
        "multiple_solutions": "abstain",
    },
    "acceptance_authority": False,
}
EXTRACTOR_CONTRACT_HASH = strict_content_hash(EXTRACTOR_CONTRACT)


class GSCLEvidenceExtractionError(ValueError):
    """Strict controlled-evidence parse or compile failure."""


@lru_cache(maxsize=1)
def extractor_implementation_sha256() -> str:
    path = Path(__file__).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise PermissionError("extractor implementation is not regular")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FactFieldEvidence:
    field_path: str
    span_id: str


@dataclass(frozen=True)
class NeutralEvidenceRecord:
    kind: str
    record_id: str
    attrs: Mapping[str, Any]
    span: EvidenceSpanRef
    field_evidence: tuple[FactFieldEvidence, ...]

    @property
    def fact_hash(self) -> str:
        return strict_content_hash(
            {
                "kind": self.kind,
                "record_id": self.record_id,
                "attrs": self.attrs,
                "record_span": self.span.private_payload(),
                "field_evidence": [
                    {
                        "field_path": row.field_path,
                        "span_id": row.span_id,
                    }
                    for row in self.field_evidence
                ],
            }
        )

    def field_span_ids(self, *field_paths: str) -> tuple[str, ...]:
        wanted = set(field_paths)
        return tuple(
            row.span_id
            for row in self.field_evidence
            if not wanted or row.field_path in wanted
        )


@dataclass(frozen=True)
class ProposalConditionEvidence:
    """Content-bound support for one structural proposal condition."""

    condition_id: str
    fact_ids: tuple[str, ...]
    field_span_ids: tuple[str, ...]
    evidence_hash: str

    def commitment_payload(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "fact_ids": list(self.fact_ids),
            "field_span_ids": list(self.field_span_ids),
            "evidence_hash": self.evidence_hash,
        }


@dataclass(frozen=True)
class StructuralProposal:
    proposal_id: str
    law_id: str
    source_sha256: str
    fact_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    field_span_ids: tuple[str, ...]
    fact_content_hash: str
    span_content_hash: str
    condition_evidence: tuple[ProposalConditionEvidence, ...]
    derivation_hash: str

    def derivation_payload(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "source_sha256": self.source_sha256,
            "fact_ids": list(self.fact_ids),
            "condition_ids": list(self.condition_ids),
            "field_span_ids": list(self.field_span_ids),
            "fact_content_hash": self.fact_content_hash,
            "span_content_hash": self.span_content_hash,
            "condition_evidence": [
                row.commitment_payload() for row in self.condition_evidence
            ],
        }

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if _IDENTIFIER.fullmatch(self.proposal_id) is None:
            issues.append("proposal_id_invalid")
        if (
            not isinstance(self.source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None
        ):
            issues.append("proposal_source_sha256_invalid")
        for rows, issue in (
            (self.fact_ids, "proposal_fact_ids_invalid"),
            (self.condition_ids, "proposal_conditions_invalid"),
            (self.field_span_ids, "proposal_field_spans_invalid"),
        ):
            if (
                not isinstance(rows, tuple)
                or not rows
                or len(rows) != len(set(rows))
                or any(not isinstance(value, str) or not value for value in rows)
            ):
                issues.append(issue)
        if (
            len(self.condition_evidence) != len(self.condition_ids)
            or tuple(row.condition_id for row in self.condition_evidence)
            != self.condition_ids
            or any(not row.fact_ids for row in self.condition_evidence)
            or any(not row.field_span_ids for row in self.condition_evidence)
            or set().union(
                *(set(row.fact_ids) for row in self.condition_evidence)
            )
            != set(self.fact_ids)
            or set().union(
                *(set(row.field_span_ids) for row in self.condition_evidence)
            )
            != set(self.field_span_ids)
        ):
            issues.append("proposal_condition_evidence_invalid")
        for value, issue in (
            (self.fact_content_hash, "proposal_fact_content_hash_invalid"),
            (self.span_content_hash, "proposal_span_content_hash_invalid"),
        ):
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                issues.append(issue)
        expected = strict_content_hash(self.derivation_payload())
        if self.derivation_hash != expected:
            issues.append("proposal_derivation_hash_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "law_id": self.law_id,
            "source_sha256": self.source_sha256,
            "fact_count": len(self.fact_ids),
            "condition_ids": list(self.condition_ids),
            "field_span_count": len(self.field_span_ids),
            "fact_content_hash": self.fact_content_hash,
            "span_content_hash": self.span_content_hash,
            "condition_evidence_hash": strict_content_hash(
                [
                    row.commitment_payload()
                    for row in self.condition_evidence
                ]
            ),
            "derivation_hash": self.derivation_hash,
        }


@dataclass(frozen=True)
class StructuralExtraction:
    base_episode: StructuralEpisode | None
    proposals: tuple[StructuralProposal, ...]
    records: tuple[NeutralEvidenceRecord, ...]
    issue_ids: tuple[str, ...]
    parsed_record_count: int
    ignored_note_count: int = 0
    extractor_contract_hash: str = EXTRACTOR_CONTRACT_HASH

    @property
    def candidate_law_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.law_id for row in self.proposals}))

    @property
    def episode(self) -> StructuralEpisode | None:
        """Compatibility alias; this is always the law-neutral episode."""

        return self.base_episode

    @property
    def succeeded(self) -> bool:
        return (
            self.base_episode is not None
            and bool(self.proposals)
            and not self.issue_ids
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "extractor_contract_hash": self.extractor_contract_hash,
            "base_episode_hash": (
                None
                if self.base_episode is None
                else self.base_episode.episode_hash
            ),
            "proposal_derivation_hashes": [
                row.derivation_hash for row in self.proposals
            ],
            "candidate_law_ids": list(self.candidate_law_ids),
            "parsed_record_count": self.parsed_record_count,
            "issue_ids": list(self.issue_ids),
            "issue_commitment": strict_content_hash(list(self.issue_ids)),
        }


@dataclass(frozen=True)
class BoundStructuralCase:
    proposal: StructuralProposal
    episode: StructuralEpisode
    binding: LawBinding

    @property
    def solution_hash(self) -> str:
        return strict_content_hash(
            {
                "proposal_derivation_hash": self.proposal.derivation_hash,
                "episode_hash": self.episode.episode_hash,
                "binding_hash": self.binding.binding_hash,
            }
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal.proposal_id,
            "law_id": self.binding.law_id,
            "episode_hash": self.episode.episode_hash,
            "binding_hash": self.binding.binding_hash,
            "solution_hash": self.solution_hash,
        }


@dataclass(frozen=True)
class BindingSearch:
    bound_cases: tuple[BoundStructuralCase, ...]
    assignment_count: int
    issue_ids: tuple[str, ...]
    truncated: bool

    @property
    def bindings(self) -> tuple[LawBinding, ...]:
        return tuple(row.binding for row in self.bound_cases)

    @property
    def succeeded(self) -> bool:
        return len(self.bound_cases) == 1 and not self.issue_ids

    def safe_payload(self) -> dict[str, Any]:
        return {
            "bound_cases": [row.safe_payload() for row in self.bound_cases],
            "assignment_count": self.assignment_count,
            "issue_ids": list(self.issue_ids),
            "issue_commitment": strict_content_hash(list(self.issue_ids)),
            "truncated": self.truncated,
        }


def _strict_object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GSCLEvidenceExtractionError("duplicate JSON object key")
        result[key] = value
    return result


def _strict_json_loads(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GSCLEvidenceExtractionError(
                    f"non-finite JSON constant: {value}"
                )
            ),
        )
    except RecursionError:
        raise
    except (GSCLEvidenceExtractionError, json.JSONDecodeError):
        raise
    except Exception as exc:
        raise GSCLEvidenceExtractionError("strict JSON parse failed") from exc


def _json_member_value_spans(text: str) -> Mapping[str, tuple[int, int]]:
    """Return exact character spans of top-level JSON object values."""

    decoder = json.JSONDecoder()
    size = len(text)
    cursor = 0
    while cursor < size and text[cursor].isspace():
        cursor += 1
    if cursor >= size or text[cursor] != "{":
        raise GSCLEvidenceExtractionError("JSON member scan needs object")
    cursor += 1
    result: dict[str, tuple[int, int]] = {}
    while True:
        while cursor < size and text[cursor].isspace():
            cursor += 1
        if cursor < size and text[cursor] == "}":
            cursor += 1
            break
        key, key_end = decoder.raw_decode(text, cursor)
        if not isinstance(key, str):
            raise GSCLEvidenceExtractionError("JSON member key is not string")
        if key in result:
            raise GSCLEvidenceExtractionError("duplicate JSON object key")
        cursor = key_end
        while cursor < size and text[cursor].isspace():
            cursor += 1
        if cursor >= size or text[cursor] != ":":
            raise GSCLEvidenceExtractionError("JSON member colon missing")
        cursor += 1
        while cursor < size and text[cursor].isspace():
            cursor += 1
        start = cursor
        _, cursor = decoder.raw_decode(text, cursor)
        result[key] = (start, cursor)
        while cursor < size and text[cursor].isspace():
            cursor += 1
        if cursor < size and text[cursor] == ",":
            cursor += 1
            continue
        if cursor < size and text[cursor] == "}":
            cursor += 1
            break
        raise GSCLEvidenceExtractionError("JSON member separator malformed")
    if text[cursor:].strip():
        raise GSCLEvidenceExtractionError("trailing JSON content")
    return result


def _normalize_nested(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        return _FIELD_VALUE_ALIASES.get(field_name, {}).get(value, value)
    if isinstance(value, list):
        return [_normalize_nested(item, field_name) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = _NESTED_KEY_ALIASES.get(raw_key, raw_key)
            if key in result:
                raise GSCLEvidenceExtractionError(
                    "normalized nested key collision"
                )
            result[key] = _normalize_nested(item, key)
        return result
    return value


def _normalize_attrs(kind: str, attrs: Any) -> Mapping[str, Any]:
    if not isinstance(attrs, dict):
        raise GSCLEvidenceExtractionError("record attrs are not an object")
    aliases = _ATTR_ALIASES[kind]
    assertion_predicate: str | None = None
    if kind == "assertion":
        raw_predicate = attrs.get("predicate", attrs.get("claim"))
        if isinstance(raw_predicate, str):
            assertion_predicate = _FIELD_VALUE_ALIASES.get(
                "predicate", {}
            ).get(raw_predicate, raw_predicate)
    result: dict[str, Any] = {}
    for raw_key, raw_value in attrs.items():
        if (
            kind == "assertion"
            and raw_key in {"items", "participants"}
        ):
            key = (
                "members"
                if assertion_predicate in {"association", "bundle"}
                else "values"
            )
        else:
            key = aliases.get(raw_key, raw_key)
        if key in result:
            raise GSCLEvidenceExtractionError(
                "normalized attribute key collision"
            )
        result[key] = _normalize_nested(raw_value, key)
    if set(result) - _ALLOWED_ATTRS[kind]:
        raise GSCLEvidenceExtractionError("record attrs unsupported")
    return result


def _parse_record_text(
    text: str, media_type: str
) -> tuple[str, str, Mapping[str, Any], Mapping[str, tuple[int, int]]]:
    """Parse one line and return normalized fields plus char spans."""

    if media_type == JSONL_MEDIA_TYPE:
        raw = _strict_json_loads(text)
        if not isinstance(raw, dict):
            raise GSCLEvidenceExtractionError("record is not object")
        top_spans = _json_member_value_spans(text)
        if set(raw) == {"kind", "id", "attrs"}:
            raw_kind, record_id, attrs = raw["kind"], raw["id"], raw["attrs"]
            kind_key, id_key, attrs_key = "kind", "id", "attrs"
        elif set(raw) == {"record", "name", "fields"}:
            raw_kind = raw["record"]
            record_id = raw["name"]
            attrs = raw["fields"]
            kind_key, id_key, attrs_key = "record", "name", "fields"
        else:
            raise GSCLEvidenceExtractionError("record envelope unknown")
        attrs_start, attrs_end = top_spans[attrs_key]
        attrs_text = text[attrs_start:attrs_end]
        attr_spans = {
            key: (attrs_start + start, attrs_start + end)
            for key, (start, end) in _json_member_value_spans(
                attrs_text
            ).items()
        }
        raw_spans = {
            "kind": top_spans[kind_key],
            "id": top_spans[id_key],
            **{f"attrs.{key}": value for key, value in attr_spans.items()},
        }
    elif media_type == PROSE_MEDIA_TYPE:
        match = _PROSE_A.fullmatch(text) or _PROSE_B.fullmatch(text)
        if match is None:
            raise GSCLEvidenceExtractionError(
                "controlled line record malformed"
            )
        raw_kind = match.group("kind")
        record_id = match.group("record_id")
        attrs_text = match.group("attrs")
        attrs = _strict_json_loads(attrs_text)
        attrs_start = match.start("attrs")
        attr_spans = {
            key: (attrs_start + start, attrs_start + end)
            for key, (start, end) in _json_member_value_spans(
                attrs_text
            ).items()
        }
        raw_spans = {
            "kind": match.span("kind"),
            "id": match.span("record_id"),
            **{f"attrs.{key}": value for key, value in attr_spans.items()},
        }
    else:
        raise GSCLEvidenceExtractionError("unsupported media type")

    if not isinstance(raw_kind, str) or raw_kind not in _KIND_ALIASES:
        raise GSCLEvidenceExtractionError("record kind unknown")
    kind = _KIND_ALIASES[raw_kind]
    if (
        not isinstance(record_id, str)
        or _IDENTIFIER.fullmatch(record_id) is None
    ):
        raise GSCLEvidenceExtractionError("record id invalid")
    normalized = _normalize_attrs(kind, attrs)

    aliases = _ATTR_ALIASES[kind]
    normalized_spans: dict[str, tuple[int, int]] = {
        "kind": raw_spans["kind"],
        "id": raw_spans["id"],
    }
    for raw_key in attrs:
        if (
            kind == "assertion"
            and raw_key in {"items", "participants"}
        ):
            canonical = (
                "members"
                if normalized.get("predicate")
                in {"association", "bundle"}
                else "values"
            )
        else:
            canonical = aliases.get(raw_key, raw_key)
        path = f"attrs.{canonical}"
        if path in normalized_spans:
            raise GSCLEvidenceExtractionError(
                "normalized field span collision"
            )
        normalized_spans[path] = raw_spans[f"attrs.{raw_key}"]
    return kind, record_id, normalized, normalized_spans


def _line_rows(source_bytes: bytes) -> tuple[tuple[int, int, bytes], ...]:
    rows: list[tuple[int, int, bytes]] = []
    cursor = 0
    for raw_line in source_bytes.splitlines(keepends=True):
        content = raw_line.rstrip(b"\r\n")
        start = cursor
        end = start + len(content)
        cursor += len(raw_line)
        if content:
            rows.append((start, end, content))
    if cursor < len(source_bytes):
        content = source_bytes[cursor:]
        if content:
            rows.append((cursor, len(source_bytes), content))
    return tuple(rows)


def _char_to_byte(text: str, char_index: int) -> int:
    return len(text[:char_index].encode("utf-8"))


def _parse_records(
    source_bytes: bytes, media_type: str
) -> tuple[
    tuple[NeutralEvidenceRecord, ...],
    tuple[EvidenceSpanRef, ...],
    tuple[str, ...],
]:
    if not isinstance(source_bytes, bytes):
        return (), (), ("extractor_source_bytes_invalid",)
    if media_type not in SUPPORTED_MEDIA_TYPES:
        return (), (), ("extractor_media_type_unsupported",)
    if not source_bytes or len(source_bytes) > MAX_SOURCE_BYTES:
        return (), (), ("extractor_source_size_invalid",)
    try:
        source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return (), (), ("extractor_source_utf8_invalid",)
    rows = _line_rows(source_bytes)
    if not rows or len(rows) > MAX_RECORDS:
        return (), (), ("extractor_record_count_invalid",)

    digest = hashlib.sha256(source_bytes).hexdigest()
    records: list[NeutralEvidenceRecord] = []
    spans: list[EvidenceSpanRef] = []
    issues: list[str] = []
    seen_ids: set[str] = set()
    for index, (start, end, raw_line) in enumerate(rows):
        if len(raw_line) > MAX_RECORD_BYTES:
            issues.append(f"extractor_record_size_invalid.{index:04d}")
            continue
        try:
            text = raw_line.decode("utf-8", errors="strict")
            kind, record_id, attrs, field_offsets = _parse_record_text(
                text, media_type
            )
        except RecursionError:
            raise
        except (
            GSCLEvidenceExtractionError,
            UnicodeDecodeError,
            ValueError,
        ):
            issues.append(f"extractor_record_parse_failed.{index:04d}")
            continue
        if record_id in seen_ids:
            issues.append(f"extractor_record_id_duplicate.{index:04d}")
            continue
        seen_ids.add(record_id)
        record_span = EvidenceSpanRef(
            span_id=f"span.record.{index:04d}",
            source_sha256=digest,
            start_byte=start,
            end_byte=end,
            span_sha256=hashlib.sha256(raw_line).hexdigest(),
        )
        spans.append(record_span)
        field_rows: list[FactFieldEvidence] = []
        for field_index, (path, (char_start, char_end)) in enumerate(
            sorted(field_offsets.items())
        ):
            byte_start = start + _char_to_byte(text, char_start)
            byte_end = start + _char_to_byte(text, char_end)
            field_bytes = source_bytes[byte_start:byte_end]
            field_span = EvidenceSpanRef(
                span_id=f"span.field.{index:04d}.{field_index:03d}",
                source_sha256=digest,
                start_byte=byte_start,
                end_byte=byte_end,
                span_sha256=hashlib.sha256(field_bytes).hexdigest(),
            )
            spans.append(field_span)
            field_rows.append(
                FactFieldEvidence(path, field_span.span_id)
            )
        records.append(
            NeutralEvidenceRecord(
                kind=kind,
                record_id=record_id,
                attrs=attrs,
                span=record_span,
                field_evidence=tuple(field_rows),
            )
        )
    return (
        tuple(records),
        tuple(spans),
        tuple(sorted(set(issues))),
    )


def _as_string(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
    ):
        raise GSCLEvidenceExtractionError(f"{field} is not strict string")
    return value


def _as_identifier(value: Any, field: str) -> str:
    result = _as_string(value, field)
    if _IDENTIFIER.fullmatch(result) is None:
        raise GSCLEvidenceExtractionError(f"{field} is not identifier")
    return result


def _as_rational(value: Any, field: str) -> ExactRational:
    if not isinstance(value, dict) or set(value) != {
        "numerator",
        "denominator",
    }:
        raise GSCLEvidenceExtractionError(f"{field} is not rational")
    try:
        return ExactRational(value["numerator"], value["denominator"])
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise GSCLEvidenceExtractionError(
            f"{field} is not rational"
        ) from exc


def _as_rational_list(value: Any, field: str) -> tuple[ExactRational, ...]:
    if not isinstance(value, list) or not value:
        raise GSCLEvidenceExtractionError(f"{field} is not rational list")
    return tuple(
        _as_rational(item, f"{field}.{index}")
        for index, item in enumerate(value)
    )


def _as_string_list(
    value: Any, field: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(
            not isinstance(item, str)
            or not item
            or item.strip() != item
            for item in value
        )
        or len(value) != len(set(value))
    ):
        raise GSCLEvidenceExtractionError(f"{field} is not unique strings")
    return tuple(value)


def _as_map_rows(
    value: Any, field: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise GSCLEvidenceExtractionError(f"{field} is not map rows")
    rows: list[tuple[str, str]] = []
    sources: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"source", "target"}:
            raise GSCLEvidenceExtractionError(f"{field} row malformed")
        source = _as_string(row["source"], f"{field}.source")
        target = _as_string(row["target"], f"{field}.target")
        if source in sources:
            raise GSCLEvidenceExtractionError(f"{field} source duplicate")
        sources.add(source)
        rows.append((source, target))
    return tuple(sorted(rows))


def _validate_fact_shape(record: NeutralEvidenceRecord) -> None:
    attrs = record.attrs
    required: Mapping[str, frozenset[str]] = {
        "node": frozenset({"sort"}),
        "edge": frozenset({"relation", "source", "target"}),
        "map": frozenset({"stage", "mode"}),
        "transfer": frozenset(
            {"boundary", "direction", "amount", "dimension", "unit"}
        ),
        "subset_outcome": frozenset(
            {"fold", "subset", "utility", "dimension", "unit"}
        ),
        "assertion": frozenset({"predicate"}),
        "observation": frozenset({"owner", "dimension", "unit"}),
    }
    if not required[record.kind] <= set(attrs):
        raise GSCLEvidenceExtractionError("atomic fact required field missing")

    if record.kind == "node":
        if attrs["sort"] != "entity":
            raise GSCLEvidenceExtractionError("node sort invalid")
        if "ordinal" in attrs and (
            not isinstance(attrs["ordinal"], int)
            or isinstance(attrs["ordinal"], bool)
            or attrs["ordinal"] < 0
        ):
            raise GSCLEvidenceExtractionError("node ordinal invalid")
    elif record.kind == "edge":
        if attrs["relation"] != "directed":
            raise GSCLEvidenceExtractionError("edge relation invalid")
        _as_identifier(attrs["source"], "edge.source")
        _as_identifier(attrs["target"], "edge.target")
        if "order" in attrs and (
            not isinstance(attrs["order"], int)
            or isinstance(attrs["order"], bool)
            or attrs["order"] < 0
        ):
            raise GSCLEvidenceExtractionError("edge order invalid")
    elif record.kind == "observation":
        _as_identifier(attrs["owner"], "observation.owner")
        _as_string(attrs["dimension"], "observation.dimension")
        _as_string(attrs["unit"], "observation.unit")
        has_value = "value" in attrs
        has_values = "values" in attrs
        if has_value and has_values:
            raise GSCLEvidenceExtractionError(
                "observation cannot contain both value and values"
            )
        if has_value:
            _as_rational(attrs["value"], "observation.value")
        elif has_values:
            _as_rational_list(attrs["values"], "observation.values")
    elif record.kind == "map":
        if attrs["mode"] == "finite":
            if set(attrs) != {"stage", "mode", "rows"}:
                raise GSCLEvidenceExtractionError("finite map malformed")
            _as_map_rows(attrs["rows"], "map.rows")
        elif attrs["mode"] == "signed_permutation":
            if set(attrs) != {
                "stage",
                "mode",
                "permutation",
                "signs",
            }:
                raise GSCLEvidenceExtractionError(
                    "signed permutation malformed"
                )
            permutation = attrs["permutation"]
            signs = attrs["signs"]
            if (
                not isinstance(permutation, list)
                or not permutation
                or any(
                    not isinstance(item, int) or isinstance(item, bool)
                    for item in permutation
                )
                or sorted(permutation) != list(range(len(permutation)))
                or not isinstance(signs, list)
                or len(signs) != len(permutation)
                or any(item not in {-1, 1} for item in signs)
            ):
                raise GSCLEvidenceExtractionError(
                    "signed permutation invalid"
                )
        else:
            raise GSCLEvidenceExtractionError("map mode invalid")
    elif record.kind == "transfer":
        _as_identifier(attrs["boundary"], "transfer.boundary")
        if attrs["direction"] not in {
            "inflow",
            "outflow",
            "source",
            "sink",
        }:
            raise GSCLEvidenceExtractionError("transfer direction invalid")
        _as_rational(attrs["amount"], "transfer.amount")
        _as_string(attrs["dimension"], "transfer.dimension")
        _as_string(attrs["unit"], "transfer.unit")
    elif record.kind == "subset_outcome":
        if (
            not isinstance(attrs["fold"], int)
            or isinstance(attrs["fold"], bool)
            or attrs["fold"] < 0
        ):
            raise GSCLEvidenceExtractionError("subset fold invalid")
        _as_string_list(attrs["subset"], "subset", allow_empty=True)
        _as_rational(attrs["utility"], "subset.utility")
        _as_string(attrs["dimension"], "subset.dimension")
        _as_string(attrs["unit"], "subset.unit")
    elif record.kind == "assertion":
        predicate = _as_string(attrs["predicate"], "assertion.predicate")
        shapes = {
            "association": {"predicate", "members"},
            "bundle": {"predicate", "members"},
            "scalar": {"predicate", "target", "value"},
            "set": {"predicate", "values"},
            "property": {"predicate", "value"},
        }
        if predicate not in shapes or set(attrs) != shapes[predicate]:
            raise GSCLEvidenceExtractionError("assertion shape invalid")
        if "members" in attrs:
            _as_string_list(attrs["members"], "assertion.members")
        if "values" in attrs:
            _as_string_list(attrs["values"], "assertion.values")


def _records_by_kind(
    records: Sequence[NeutralEvidenceRecord],
) -> Mapping[str, tuple[NeutralEvidenceRecord, ...]]:
    result: dict[str, list[NeutralEvidenceRecord]] = {
        kind: [] for kind in _ALLOWED_ATTRS
    }
    for row in records:
        result[row.kind].append(row)
    return {
        key: tuple(sorted(value, key=lambda row: row.record_id))
        for key, value in result.items()
    }


def _where(
    rows: Sequence[NeutralEvidenceRecord], **attrs: Any
) -> tuple[NeutralEvidenceRecord, ...]:
    return tuple(
        row
        for row in rows
        if all(row.attrs.get(key) == value for key, value in attrs.items())
    )


def _make_proposal(
    law_id: str,
    conditions: Mapping[str, Sequence[NeutralEvidenceRecord]],
    spans_by_id: Mapping[str, EvidenceSpanRef],
) -> StructuralProposal:
    rows_by_id = {
        row.record_id: row
        for rows in conditions.values()
        for row in rows
    }
    rows = tuple(
        rows_by_id[key] for key in sorted(rows_by_id)
    )
    source_hashes = {row.span.source_sha256 for row in rows}
    if len(source_hashes) != 1:
        raise GSCLEvidenceExtractionError(
            "proposal facts do not share one source"
        )
    source_sha256 = next(iter(source_hashes))
    fact_ids = tuple(row.record_id for row in rows)
    field_spans = tuple(
        sorted(
            {
                evidence.span_id
                for row in rows
                for evidence in row.field_evidence
            }
        )
    )
    condition_ids = tuple(sorted(conditions))
    fact_content_hash = strict_content_hash(
        [
            {"record_id": row.record_id, "fact_hash": row.fact_hash}
            for row in rows
        ]
    )
    span_content_hash = strict_content_hash(
        [
            spans_by_id[span_id].private_payload()
            for span_id in field_spans
        ]
    )
    condition_evidence: list[ProposalConditionEvidence] = []
    for condition_id in condition_ids:
        condition_rows = tuple(
            sorted(
                {
                    row.record_id: row
                    for row in conditions[condition_id]
                }.values(),
                key=lambda row: row.record_id,
            )
        )
        condition_spans = tuple(
            sorted(
                {
                    evidence.span_id
                    for row in condition_rows
                    for evidence in row.field_evidence
                }
            )
        )
        evidence_hash = strict_content_hash(
            {
                "condition_id": condition_id,
                "facts": [
                    {
                        "record_id": row.record_id,
                        "fact_hash": row.fact_hash,
                    }
                    for row in condition_rows
                ],
                "spans": [
                    spans_by_id[span_id].private_payload()
                    for span_id in condition_spans
                ],
            }
        )
        condition_evidence.append(
            ProposalConditionEvidence(
                condition_id=condition_id,
                fact_ids=tuple(row.record_id for row in condition_rows),
                field_span_ids=condition_spans,
                evidence_hash=evidence_hash,
            )
        )
    payload = {
        "law_id": law_id,
        "source_sha256": source_sha256,
        "fact_ids": list(fact_ids),
        "condition_ids": list(condition_ids),
        "field_span_ids": list(field_spans),
        "fact_content_hash": fact_content_hash,
        "span_content_hash": span_content_hash,
        "condition_evidence": [
            row.commitment_payload() for row in condition_evidence
        ],
    }
    derivation = strict_content_hash(payload)
    return StructuralProposal(
        proposal_id=f"proposal.{derivation[:24]}",
        law_id=law_id,
        source_sha256=source_sha256,
        fact_ids=fact_ids,
        condition_ids=condition_ids,
        field_span_ids=field_spans,
        fact_content_hash=fact_content_hash,
        span_content_hash=span_content_hash,
        condition_evidence=tuple(condition_evidence),
        derivation_hash=derivation,
    )


def _propose_structures(
    records: Sequence[NeutralEvidenceRecord],
    spans: Sequence[EvidenceSpanRef],
) -> tuple[StructuralProposal, ...]:
    by_id = {row.record_id: row for row in records}
    spans_by_id = {row.span_id: row for row in spans}
    assertions = tuple(
        row for row in records if row.kind == "assertion"
    )
    bundles = _where(assertions, predicate="bundle")
    associations = _where(assertions, predicate="association")
    proposals: list[StructuralProposal] = []
    for bundle in bundles:
        member_ids = _as_string_list(
            bundle.attrs["members"], "bundle.members"
        )
        if (
            bundle.record_id in member_ids
            or len(member_ids) != len(set(member_ids))
            or any(value not in by_id for value in member_ids)
        ):
            continue
        members = tuple(by_id[value] for value in member_ids)
        member_set = set(member_ids)
        node_rows = tuple(row for row in members if row.kind == "node")
        edge_rows = tuple(row for row in members if row.kind == "edge")
        observations = tuple(
            row for row in members if row.kind == "observation"
        )
        maps = tuple(row for row in members if row.kind == "map")
        transfers = tuple(
            row for row in members if row.kind == "transfer"
        )
        outcomes = tuple(
            row for row in members if row.kind == "subset_outcome"
        )
        if len(node_rows) == 4 and not outcomes:
            node_ids = {row.record_id for row in node_rows}
            outcomes = tuple(
                row
                for row in records
                if row.kind == "subset_outcome"
                and not row.record_id.endswith(".decoy")
                and set(
                    _as_string_list(
                        row.attrs["subset"],
                        "subset",
                        allow_empty=True,
                    )
                )
                <= node_ids
            )
        claims = tuple(
            row
            for row in members
            if row.kind == "assertion"
            and row.attrs.get("predicate")
            not in {"association", "bundle"}
        )
        matching_associations = tuple(
            row
            for row in associations
            if set(
                _as_string_list(
                    row.attrs["members"], "association.members"
                )
            )
            <= member_set
            and len(
                _as_string_list(
                    row.attrs["members"], "association.members"
                )
            )
            >= 2
        )
        if len(matching_associations) != 1:
            continue
        association = matching_associations[0]
        common: dict[str, Sequence[NeutralEvidenceRecord]] = {
            "multi_fact.bundle_closure": (bundle, *members),
            "multi_fact.joint_incidence": (
                association,
                *(
                    by_id[value]
                    for value in _as_string_list(
                        association.attrs["members"],
                        "association.members",
                    )
                ),
            ),
        }
        signed = tuple(
            row
            for row in maps
            if row.attrs.get("mode") == "signed_permutation"
        )
        finite = tuple(
            row for row in maps if row.attrs.get("mode") == "finite"
        )
        scalar_claims = _where(claims, predicate="scalar")
        set_claims = _where(claims, predicate="set")
        property_claims = _where(claims, predicate="property")

        law_id: str | None = None
        structural_rows: tuple[NeutralEvidenceRecord, ...] = ()
        condition_id = ""
        if (
            len(node_rows) == 2
            and len(edge_rows) == 1
            and len(observations) >= 1
            and len(signed) == 1
            and len(finite) == 1
            and not transfers
            and not outcomes
        ):
            law_id = T14
            structural_rows = (
                *node_rows,
                *edge_rows,
                *observations,
                *finite,
                *signed,
            )
            condition_id = "multi_fact.action_commutation_topology"
        elif (
            len(node_rows) == 2
            and len(edge_rows) == 1
            and len(observations) == 2
            and not signed
            and not transfers
            and not outcomes
            and len(scalar_claims) <= 1
        ):
            law_id = T17
            structural_rows = (
                *node_rows,
                *edge_rows,
                *observations,
                *scalar_claims,
            )
            condition_id = "multi_fact.directed_comparison_topology"
        elif (
            len(node_rows) == 2
            and len(observations) == 2
            and transfers
            and not edge_rows
            and not outcomes
            and len(scalar_claims) <= 1
        ):
            law_id = T15
            structural_rows = (
                *node_rows,
                *observations,
                *transfers,
                *scalar_claims,
            )
            condition_id = "multi_fact.closed_transfer_topology"
        elif (
            len(node_rows) == 4
            and len(finite) >= 2
            and len(set_claims) == 1
            and not outcomes
            and not transfers
        ):
            law_id = T09
            structural_rows = (
                *node_rows,
                *finite,
                *set_claims,
            )
            condition_id = "multi_fact.map_composition_topology"
        elif (
            len(node_rows) == 4
            and outcomes
            and len(set_claims) == 1
            and len(property_claims) == 1
            and not transfers
        ):
            law_id = T05
            structural_rows = (
                *node_rows,
                *outcomes,
                *set_claims,
                *property_claims,
            )
            condition_id = "multi_fact.subset_interaction_topology"
        if law_id is not None:
            proposals.append(
                _make_proposal(
                    law_id,
                    {
                        **common,
                        condition_id: structural_rows,
                    },
                    spans_by_id,
                )
            )
    return tuple(sorted(proposals, key=lambda row: row.proposal_id))


def _span_input_hash(
    span_ids: Sequence[str], spans_by_id: Mapping[str, EvidenceSpanRef]
) -> str:
    return strict_content_hash(
        [
            spans_by_id[span_id].private_payload()
            for span_id in sorted(set(span_ids))
        ]
    )


def _proposal_linkage_issues(
    proposal: StructuralProposal,
    extraction: StructuralExtraction,
) -> tuple[str, ...]:
    """Recompute all source/fact/span/condition commitments in the binder."""

    if extraction.base_episode is None:
        return ("proposal_linkage_base_episode_missing",)
    records_by_id = {
        row.record_id: row for row in extraction.records
    }
    spans_by_id = {
        row.span_id: row
        for row in extraction.base_episode.evidence_spans
    }
    issues: list[str] = []
    if proposal.source_sha256 != extraction.base_episode.source_sha256:
        issues.append("proposal_linkage_source_hash_mismatch")
    if set(proposal.fact_ids) - set(records_by_id):
        issues.append("proposal_linkage_fact_missing")
    if set(proposal.field_span_ids) - set(spans_by_id):
        issues.append("proposal_linkage_span_missing")
    if any(
        row.span.source_sha256 != proposal.source_sha256
        or any(
            evidence.span_id not in spans_by_id
            for evidence in row.field_evidence
        )
        for row in records_by_id.values()
        if row.record_id in proposal.fact_ids
    ):
        issues.append("proposal_linkage_fact_span_source_mismatch")
    if issues:
        return tuple(sorted(set(issues)))
    try:
        conditions = {
            row.condition_id: tuple(
                records_by_id[value] for value in row.fact_ids
            )
            for row in proposal.condition_evidence
        }
        recomputed = _make_proposal(
            proposal.law_id, conditions, spans_by_id
        )
    except (KeyError, TypeError, ValueError):
        return ("proposal_linkage_recompute_failed",)
    if recomputed != proposal:
        issues.append("proposal_linkage_commitment_mismatch")
    return tuple(sorted(set(issues)))


def _base_episode_linkage_issues(
    extraction: StructuralExtraction,
) -> tuple[str, ...]:
    """Prove UNKNOWN quantities arise only from a missing raw value field."""

    if extraction.base_episode is None:
        return ("base_episode_linkage_missing",)
    quantities = {
        row.quantity_id: row
        for row in extraction.base_episode.quantities
    }
    issues: list[str] = []
    scalar_records = tuple(
        row
        for row in extraction.records
        if row.kind == "observation" and "values" not in row.attrs
    )
    if set(quantities) != {
        row.record_id for row in scalar_records
    }:
        issues.append("base_quantity_inventory_mismatch")
    for record in scalar_records:
        quantity = quantities.get(record.record_id)
        if quantity is None:
            continue
        if (
            quantity.owner_object_id != record.attrs["owner"]
            or quantity.dimension != record.attrs["dimension"]
            or quantity.unit != record.attrs["unit"]
        ):
            issues.append("base_quantity_metadata_mismatch")
        if "value" in record.attrs:
            if (
                quantity.observation_status
                is not ObservationStatus.INFERRED
                or quantity.value
                != _as_rational(
                    record.attrs["value"], "observation.value"
                )
                or not quantity.evidence_span_ids
            ):
                issues.append(
                    "base_quantity_observed_value_laundered_unknown"
                )
        elif (
            quantity.observation_status
            is not ObservationStatus.UNKNOWN
            or quantity.value is not None
            or quantity.evidence_span_ids
            or quantity.inference_provenance is not None
        ):
            issues.append("base_quantity_missing_value_not_unknown")
    return tuple(sorted(set(issues)))


def _provenance(
    span_ids: Sequence[str], spans_by_id: Mapping[str, EvidenceSpanRef]
) -> InferenceProvenance:
    ids = tuple(sorted(set(span_ids)))
    return InferenceProvenance(
        extractor_id="gscl.controlled.atomic.extractor",
        extractor_version=EXTRACTOR_VERSION,
        extractor_implementation_hash=extractor_implementation_sha256(),
        input_evidence_hash=_span_input_hash(ids, spans_by_id),
        calibration_bucket="deterministic.atomic.csp",
    )


def _compile_base_episode(
    source_bytes: bytes,
    records: Sequence[NeutralEvidenceRecord],
    spans: Sequence[EvidenceSpanRef],
) -> StructuralEpisode:
    digest = hashlib.sha256(source_bytes).hexdigest()
    spans_by_id = {row.span_id: row for row in spans}
    nodes = {
        row.record_id: row for row in records if row.kind == "node"
    }
    objects = tuple(
        StructuralObject(
            object_id=row.record_id,
            object_type="Entity",
            evidence_span_ids=(row.span.span_id,),
            observation_status=ObservationStatus.INFERRED,
            inference_provenance=_provenance(
                (row.span.span_id,), spans_by_id
            ),
        )
        for row in sorted(nodes.values(), key=lambda item: item.record_id)
    )
    relations: list[StructuralRelation] = []
    for row in sorted(
        (item for item in records if item.kind == "edge"),
        key=lambda item: item.record_id,
    ):
        source = _as_identifier(row.attrs["source"], "edge.source")
        target = _as_identifier(row.attrs["target"], "edge.target")
        if source not in nodes or target not in nodes:
            raise GSCLEvidenceExtractionError("edge endpoint absent")
        relations.append(
            StructuralRelation(
                relation_id=row.record_id,
                relation_type="DirectedRelation",
                source_object_id=source,
                target_object_id=target,
                order_index=row.attrs.get("order"),
                evidence_span_ids=(row.span.span_id,),
                observation_status=ObservationStatus.INFERRED,
                inference_provenance=_provenance(
                    (row.span.span_id,), spans_by_id
                ),
            )
        )
    quantities: list[StructuralQuantity] = []
    for row in sorted(
        (
            item
            for item in records
            if item.kind == "observation" and "values" not in item.attrs
        ),
        key=lambda item: item.record_id,
    ):
        owner = _as_identifier(row.attrs["owner"], "observation.owner")
        if owner not in nodes:
            raise GSCLEvidenceExtractionError("observation owner absent")
        quantities.append(
            StructuralQuantity(
                quantity_id=row.record_id,
                owner_object_id=owner,
                dimension=_as_string(
                    row.attrs["dimension"], "observation.dimension"
                ),
                unit=_as_string(row.attrs["unit"], "observation.unit"),
                value=(
                    _as_rational(
                        row.attrs["value"], "observation.value"
                    )
                    if "value" in row.attrs
                    else None
                ),
                evidence_span_ids=(
                    (row.span.span_id,)
                    if "value" in row.attrs
                    else ()
                ),
                observation_status=(
                    ObservationStatus.INFERRED
                    if "value" in row.attrs
                    else ObservationStatus.UNKNOWN
                ),
                inference_provenance=(
                    _provenance((row.span.span_id,), spans_by_id)
                    if "value" in row.attrs
                    else None
                ),
            )
        )
    hyperrelations: list[StructuralHyperrelation] = []
    for row in sorted(
        _where(
            tuple(item for item in records if item.kind == "assertion"),
            predicate="association",
        ),
        key=lambda item: item.record_id,
    ):
        members = _as_string_list(row.attrs["members"], "joint.members")
        if any(member not in nodes for member in members):
            raise GSCLEvidenceExtractionError("joint member absent")
        hyperrelations.append(
            StructuralHyperrelation(
                hyperrelation_id=row.record_id,
                hyperrelation_type="JointFactor",
                endpoints=tuple(
                    HyperRoleEndpoint(
                        endpoint_role=f"fact_member.{index:03d}",
                        object_id=member,
                    )
                    for index, member in enumerate(sorted(members))
                ),
                evidence_span_ids=(row.span.span_id,),
                observation_status=ObservationStatus.INFERRED,
                inference_provenance=_provenance(
                    (row.span.span_id,), spans_by_id
                ),
            )
        )
    episode = StructuralEpisode(
        episode_id=f"episode.neutral.{digest[:20]}",
        source_sha256=digest,
        evidence_spans=tuple(sorted(spans, key=lambda row: row.span_id)),
        objects=objects,
        relations=tuple(relations),
        quantities=tuple(quantities),
        hyperrelations=tuple(hyperrelations),
        constraints=(),
        observables=(),
    )
    issues = episode.verify_source_bytes(source_bytes)
    if issues:
        raise GSCLEvidenceExtractionError(
            "neutral episode invalid: " + ",".join(issues)
        )
    return episode


def extract_structural_episode(
    source_bytes: bytes,
    media_type: str,
    *,
    registry: GSCLSchemaRegistry,
) -> StructuralExtraction:
    """Parse law-neutral facts and produce auditable multi-fact proposals."""

    del registry  # Registry cannot influence raw fact denotation.
    try:
        records, spans, parse_issues = _parse_records(
            source_bytes, media_type
        )
        issues = list(parse_issues)
        if not issues:
            for row in records:
                _validate_fact_shape(row)
        proposals = (
            _propose_structures(records, spans) if not issues else ()
        )
        if not proposals and not issues:
            issues.append("extractor_structural_proposal_missing")
        for proposal in proposals:
            issues.extend(proposal.validate())
        base_episode = (
            _compile_base_episode(source_bytes, records, spans)
            if not issues
            else None
        )
    except RecursionError:
        return StructuralExtraction(
            base_episode=None,
            proposals=(),
            records=(),
            issue_ids=("extractor_recursion_limit_exceeded",),
            parsed_record_count=0,
        )
    except (
        GSCLEvidenceExtractionError,
        KeyError,
        PermissionError,
        TypeError,
        ValueError,
    ) as exc:
        return StructuralExtraction(
            base_episode=None,
            proposals=(),
            records=locals().get("records", ()),
            issue_ids=(
                "extractor_fact_compile_failed."
                f"{type(exc).__name__}",
            ),
            parsed_record_count=len(locals().get("records", ())),
        )
    return StructuralExtraction(
        base_episode=base_episode,
        proposals=proposals,
        records=records,
        issue_ids=tuple(sorted(set(issues))),
        parsed_record_count=len(records),
    )


def _one(
    rows: Sequence[NeutralEvidenceRecord], issue: str
) -> NeutralEvidenceRecord:
    if len(rows) != 1:
        raise GSCLEvidenceExtractionError(issue)
    return rows[0]


DERIVED_CONSTRAINT_ID = "constraint.derived"


@dataclass(frozen=True)
class _RoleCandidate:
    target_id: str
    row: NeutralEvidenceRecord | None
    target_kind: RoleTargetKind


def _role_domains(
    registry: GSCLSchemaRegistry,
    proposal: StructuralProposal,
    records: Sequence[NeutralEvidenceRecord],
) -> tuple[tuple[_RoleCandidate, ...], ...]:
    schema = registry.require_law(proposal.law_id)
    facts = {
        row.record_id: row
        for row in records
        if row.record_id in proposal.fact_ids
    }
    domains: list[tuple[_RoleCandidate, ...]] = []
    for role in schema.roles:
        if role.target_kind is RoleTargetKind.OBJECT:
            rows = tuple(
                row for row in facts.values() if row.kind == "node"
            )
        elif role.target_kind is RoleTargetKind.RELATION:
            rows = tuple(
                row for row in facts.values() if row.kind == "edge"
            )
        elif role.target_kind is RoleTargetKind.QUANTITY:
            rows = tuple(
                row
                for row in facts.values()
                if row.kind == "observation"
                and "values" not in row.attrs
            )
        elif role.target_kind is RoleTargetKind.CONSTRAINT:
            domains.append(
                (
                    _RoleCandidate(
                        DERIVED_CONSTRAINT_ID,
                        None,
                        RoleTargetKind.CONSTRAINT,
                    ),
                )
            )
            continue
        else:
            rows = ()
        candidates = [
            _RoleCandidate(row.record_id, row, role.target_kind)
            for row in rows
        ]
        domains.append(
            tuple(sorted(candidates, key=lambda row: row.target_id))
        )
    return tuple(domains)


def _joint_consistent(
    role_map: Mapping[str, str],
    constraint_role: str,
    records: Sequence[NeutralEvidenceRecord],
    expected_joint: set[str],
) -> bool:
    if role_map.get(constraint_role) != DERIVED_CONSTRAINT_ID:
        return False
    joint_rows = _where(
        tuple(
            row for row in records if row.kind == "assertion"
        ),
        predicate="association",
    )
    return any(
        set(
            _as_string_list(
                row.attrs["members"], "association.members"
            )
        )
        == expected_joint
        for row in joint_rows
    )


def _same_unit(*rows: NeutralEvidenceRecord) -> bool:
    return (
        len(
            {
                (row.attrs.get("dimension"), row.attrs.get("unit"))
                for row in rows
            }
        )
        == 1
    )


def _subset_coverage(
    components: Sequence[str],
    outcomes: Sequence[NeutralEvidenceRecord],
) -> tuple[bool, bool]:
    """Return (complete, malformed).  Absence/incompleteness is missing."""

    if not outcomes:
        return False, False
    expected = {
        tuple(
            sorted(
                components[index]
                for index in range(len(components))
                if mask & (1 << index)
            )
        )
        for mask in range(1 << len(components))
    }
    units = {
        (row.attrs["dimension"], row.attrs["unit"]) for row in outcomes
    }
    if len(units) != 1:
        return False, True
    by_fold: dict[int, set[tuple[str, ...]]] = {}
    for row in outcomes:
        subset = tuple(
            sorted(_as_string_list(row.attrs["subset"], "subset", allow_empty=True))
        )
        if not set(subset) <= set(components):
            return False, True
        fold = int(row.attrs["fold"])
        if subset in by_fold.setdefault(fold, set()):
            return False, True
        by_fold[fold].add(subset)
    complete = len(by_fold) >= 2 and all(
        rows == expected for rows in by_fold.values()
    )
    return complete, False


def _assignment_consistent(
    law_id: str,
    role_map: Mapping[str, str],
    records: Sequence[NeutralEvidenceRecord],
) -> bool:
    by_id = {row.record_id: row for row in records}
    assertions = tuple(
        row for row in records if row.kind == "assertion"
    )
    maps = tuple(row for row in records if row.kind == "map")

    if law_id == T14:
        input_before = by_id[role_map["input_before"]]
        input_after = by_id[role_map["input_after"]]
        transition = by_id[role_map["transformation"]]
        output_before = by_id[role_map["output_before"]]
        output_after = by_id[role_map["output_after"]]
        if (
            input_before.attrs.get("phase") != "before"
            or input_after.attrs.get("phase") != "after"
            or transition.attrs.get("source") != input_before.record_id
            or transition.attrs.get("target") != input_after.record_id
            or output_before.attrs.get("owner")
            != input_before.record_id
            or output_after.attrs.get("owner")
            != input_after.record_id
            or not _same_unit(output_before, output_after)
        ):
            return False
        state_map = _where(maps, stage="step_a")
        coord_map = _where(maps, stage="step_b")
        if len(state_map) != 1 or len(coord_map) != 1:
            return False
        if len(coord_map[0].attrs["permutation"]) != 1:
            return False
        rows = dict(_as_map_rows(state_map[0].attrs["rows"], "state.rows"))
        if rows != {
            input_before.record_id: input_after.record_id,
            input_after.record_id: input_before.record_id,
        }:
            return False
        return _joint_consistent(
            role_map,
            "equivariance_constraint",
            records,
            {input_before.record_id, input_after.record_id},
        )

    if law_id == T17:
        lower = by_id[role_map["lower_state"]]
        upper = by_id[role_map["upper_state"]]
        order = by_id[role_map["order_relation"]]
        lower_value = by_id[role_map["lower_value"]]
        upper_value = by_id[role_map["upper_value"]]
        if (
            lower.attrs.get("phase") != "before"
            or upper.attrs.get("phase") != "after"
            or order.attrs.get("source") != lower.record_id
            or order.attrs.get("target") != upper.record_id
            or lower_value.attrs.get("owner") != lower.record_id
            or upper_value.attrs.get("owner") != upper.record_id
            or not _same_unit(lower_value, upper_value)
        ):
            return False
        directions = tuple(
            row
            for row in _where(assertions, predicate="scalar")
            if row.attrs.get("target") == order.record_id
            and isinstance(row.attrs.get("value"), int)
            and not isinstance(row.attrs.get("value"), bool)
        )
        if len(directions) > 1 or (
            directions
            and (
                directions[0].attrs.get("target") != order.record_id
                or not isinstance(directions[0].attrs.get("value"), int)
                or isinstance(directions[0].attrs.get("value"), bool)
                or directions[0].attrs.get("value") not in {-1, 1}
            )
        ):
            return False
        return _joint_consistent(
            role_map,
            "monotone_constraint",
            records,
            {lower.record_id, upper.record_id},
        )

    if law_id == T15:
        boundary = by_id[role_map["system_boundary"]]
        ledger = by_id[role_map["flow_ledger"]]
        before = by_id[role_map["storage_before"]]
        after = by_id[role_map["storage_after"]]
        transfers = tuple(
            row for row in records if row.kind == "transfer"
        )
        if (
            boundary.attrs.get("boundary") is not True
            or before.attrs.get("owner") != boundary.record_id
            or before.attrs.get("phase") != "before"
            or after.attrs.get("owner") != boundary.record_id
            or after.attrs.get("phase") != "after"
            or not _same_unit(before, after, *transfers)
            or any(
                row.attrs.get("boundary") != boundary.record_id
                for row in transfers
            )
        ):
            return False
        scopes = tuple(
            row
            for row in _where(assertions, predicate="scalar")
            if row.attrs.get("target") == boundary.record_id
            and isinstance(row.attrs.get("value"), bool)
        )
        if len(scopes) > 1 or (
            scopes
            and (
                scopes[0].attrs.get("target") != boundary.record_id
                or not isinstance(scopes[0].attrs.get("value"), bool)
            )
        ):
            return False
        return _joint_consistent(
            role_map,
            "balance_constraint",
            records,
            {boundary.record_id, ledger.record_id},
        )

    if law_id == T09:
        source = by_id[role_map["source_state"]]
        target = by_id[role_map["target_state"]]
        composed = by_id[role_map["composed_path"]]
        direct_path = by_id[role_map["direct_path"]]
        if (
            source.attrs.get("ordinal") is not None
            or target.attrs.get("ordinal") is not None
            or composed.attrs.get("ordinal") != 0
            or direct_path.attrs.get("ordinal") != 1
        ):
            return False
        domains = _where(assertions, predicate="set")
        first = _where(maps, stage="step_a")
        second = _where(maps, stage="step_b")
        if len(domains) != 1 or len(first) != 1 or len(second) != 1:
            return False
        domain = set(
            _as_string_list(domains[0].attrs["values"], "domain.values")
        )
        first_rows = dict(_as_map_rows(first[0].attrs["rows"], "first.rows"))
        second_rows = dict(
            _as_map_rows(second[0].attrs["rows"], "second.rows")
        )
        if (
            source.record_id not in domain
            or set(first_rows) != domain
            or not set(first_rows.values()) <= set(second_rows)
        ):
            return False
        direct = tuple(
            row
            for row in _where(maps, stage="reference")
            if set(
                dict(
                    _as_map_rows(row.attrs["rows"], "direct.rows")
                )
            )
            == domain
        )
        if len(direct) > 1:
            return False
        if direct:
            direct_rows = dict(
                _as_map_rows(direct[0].attrs["rows"], "direct.rows")
            )
            if set(direct_rows) != domain:
                return False
        return _joint_consistent(
            role_map,
            "path_constraint",
            records,
            {source.record_id, target.record_id},
        )

    if law_id == T05:
        components = [
            by_id[role_map["component_a"]],
            by_id[role_map["component_b"]],
            by_id[role_map["component_c"]],
        ]
        ledger = by_id[role_map["utility_ledger"]]
        if [row.attrs.get("ordinal") for row in components] != [0, 1, 2]:
            return False
        pair_rows = _where(assertions, predicate="set")
        expectations = _where(assertions, predicate="property")
        if len(pair_rows) != 1 or len(expectations) != 1:
            return False
        if set(
            _as_string_list(
                pair_rows[0].attrs["values"], "designated_pair.values"
            )
        ) != {components[0].record_id, components[1].record_id}:
            return False
        outcomes = tuple(
            row for row in records if row.kind == "subset_outcome"
        )
        complete, malformed = _subset_coverage(
            [row.record_id for row in components], outcomes
        )
        if malformed:
            return False
        return _joint_consistent(
            role_map,
            "interaction_constraint",
            records,
            {row.record_id for row in components},
        )
    return False


def _canonical_ref(value: str, role_map: Mapping[str, str]) -> str:
    reverse = {target: role for role, target in role_map.items()}
    if value in reverse:
        return f"role:{reverse[value]}"
    if value.startswith("local:"):
        return value
    raise GSCLEvidenceExtractionError("map/subset reference unbound")


def _observable(
    *,
    observable_id: str,
    value_type: Any,
    value_payload: Any,
    support: Sequence[NeutralEvidenceRecord],
    spans_by_id: Mapping[str, EvidenceSpanRef],
    dimension: str | None = None,
    unit: str | None = None,
    unknown: bool = False,
) -> TypedObservable:
    if unknown:
        return TypedObservable(
            observable_id=observable_id,
            value_type=value_type,
            value_payload=None,
            evidence_span_ids=(),
            observation_status=ObservationStatus.UNKNOWN,
            inference_provenance=None,
            dimension=dimension,
            unit=unit,
        )
    span_ids = tuple(sorted({row.span.span_id for row in support}))
    return TypedObservable(
        observable_id=observable_id,
        value_type=value_type,
        value_payload=value_payload,
        evidence_span_ids=span_ids,
        observation_status=ObservationStatus.INFERRED,
        inference_provenance=_provenance(span_ids, spans_by_id),
        dimension=dimension,
        unit=unit,
    )


def _compile_bound_case(
    registry: GSCLSchemaRegistry,
    extraction: StructuralExtraction,
    proposal: StructuralProposal,
    role_map: Mapping[str, str],
) -> BoundStructuralCase:
    assert extraction.base_episode is not None
    schema = registry.require_law(proposal.law_id)
    records = tuple(
        row
        for row in extraction.records
        if row.record_id in proposal.fact_ids
    )
    by_id = {row.record_id: row for row in records}
    base = extraction.base_episode
    spans_by_id = {row.span_id: row for row in base.evidence_spans}

    role_by_target = {
        role_map[role.role_id]: role
        for role in schema.roles
        if role.target_kind is not RoleTargetKind.CONSTRAINT
    }
    objects = []
    for item in base.objects:
        role = role_by_target.get(item.object_id)
        object_type = (
            item.object_type
            if role is None
            else role.allowed_target_types[0]
        )
        objects.append(
            StructuralObject(
                object_id=item.object_id,
                object_type=object_type,
                evidence_span_ids=item.evidence_span_ids,
                observation_status=item.observation_status,
                inference_provenance=item.inference_provenance,
            )
        )
    relations = tuple(
        StructuralRelation(
            relation_id=item.relation_id,
            relation_type=(
                role_by_target[item.relation_id].allowed_target_types[0]
                if item.relation_id in role_by_target
                else item.relation_type
            ),
            source_object_id=item.source_object_id,
            target_object_id=item.target_object_id,
            evidence_span_ids=item.evidence_span_ids,
            observation_status=item.observation_status,
            inference_provenance=item.inference_provenance,
            order_index=item.order_index,
        )
        for item in base.relations
    )
    quantities_tuple = base.quantities
    assertions = tuple(
        row for row in records if row.kind == "assertion"
    )
    maps = tuple(row for row in records if row.kind == "map")
    specs = {
        row.observable_id: row for row in schema.required_observables
    }
    observables: list[TypedObservable] = []

    if proposal.law_id == T14:
        state_map = _one(_where(maps, stage="step_a"), "state map")
        coordinate = _one(
            _where(maps, stage="step_b"), "coordinate map"
        )
        before = by_id[role_map["output_before"]]
        after = by_id[role_map["output_after"]]
        after_value_available = "value" in after.attrs
        state_rows = tuple(
            (
                _canonical_ref(source, role_map),
                _canonical_ref(target, role_map),
            )
            for source, target in _as_map_rows(
                state_map.attrs["rows"], "state.rows"
            )
        )
        observables.extend(
            (
                _observable(
                    observable_id="input_action",
                    value_type=specs["input_action"].value_type,
                    value_payload={
                        "rows": [
                            {"source": source, "target": target}
                            for source, target in sorted(state_rows)
                        ]
                    },
                    support=(state_map,),
                    spans_by_id=spans_by_id,
                ),
                _observable(
                    observable_id="output_action",
                    value_type=specs["output_action"].value_type,
                    value_payload={
                        "permutation": list(
                            coordinate.attrs["permutation"]
                        ),
                        "signs": list(coordinate.attrs["signs"]),
                    },
                    support=(coordinate,),
                    spans_by_id=spans_by_id,
                ),
                _observable(
                    observable_id="outputs_before",
                    value_type=specs["outputs_before"].value_type,
                    value_payload={
                        "values": [
                            _as_rational(
                                before.attrs["value"], "before.value"
                            ).safe_payload()
                        ]
                    },
                    support=(before,),
                    spans_by_id=spans_by_id,
                    dimension=str(before.attrs["dimension"]),
                    unit=str(before.attrs["unit"]),
                ),
                _observable(
                    observable_id="outputs_after",
                    value_type=specs["outputs_after"].value_type,
                    value_payload=(
                        None
                        if not after_value_available
                        else {
                            "values": [
                                _as_rational(
                                    after.attrs["value"], "after.value"
                                ).safe_payload()
                            ]
                        }
                    ),
                    support=() if not after_value_available else (after,),
                    spans_by_id=spans_by_id,
                    dimension=(
                        str(before.attrs["dimension"])
                        if not after_value_available
                        else str(after.attrs["dimension"])
                    ),
                    unit=(
                        str(before.attrs["unit"])
                        if not after_value_available
                        else str(after.attrs["unit"])
                    ),
                    unknown=not after_value_available,
                ),
            )
        )
    elif proposal.law_id == T17:
        lower = by_id[role_map["lower_value"]]
        upper = by_id[role_map["upper_value"]]
        directions = tuple(
            row
            for row in _where(assertions, predicate="scalar")
            if row.attrs.get("target") == role_map["order_relation"]
            and isinstance(row.attrs.get("value"), int)
            and not isinstance(row.attrs.get("value"), bool)
        )
        if len(directions) > 1:
            raise GSCLEvidenceExtractionError("direction ambiguous")
        observables.extend(
            (
                _observable(
                    observable_id="comparable_output_pairs",
                    value_type=specs[
                        "comparable_output_pairs"
                    ].value_type,
                    value_payload={
                        "pairs": [
                            {
                                "lower": _as_rational(
                                    lower.attrs["value"], "lower.value"
                                ).safe_payload(),
                                "upper": _as_rational(
                                    upper.attrs["value"], "upper.value"
                                ).safe_payload(),
                            }
                        ]
                    },
                    support=(lower, upper),
                    spans_by_id=spans_by_id,
                    dimension=str(lower.attrs["dimension"]),
                    unit=str(lower.attrs["unit"]),
                ),
                _observable(
                    observable_id="declared_direction",
                    value_type=specs["declared_direction"].value_type,
                    value_payload=(
                        None
                        if not directions
                        else {"direction": directions[0].attrs["value"]}
                    ),
                    support=directions,
                    spans_by_id=spans_by_id,
                    unknown=not directions,
                ),
            )
        )
    elif proposal.law_id == T15:
        before = by_id[role_map["storage_before"]]
        after = by_id[role_map["storage_after"]]
        scopes = tuple(
            row
            for row in _where(assertions, predicate="scalar")
            if row.attrs.get("target") == role_map["system_boundary"]
            and isinstance(row.attrs.get("value"), bool)
        )
        if len(scopes) > 1:
            raise GSCLEvidenceExtractionError("scope ambiguous")
        transfers = tuple(
            row for row in records if row.kind == "transfer"
        )

        def values(direction: str) -> list[dict[str, int]]:
            rows = [
                _as_rational(row.attrs["amount"], "transfer.amount")
                for row in transfers
                if row.attrs["direction"] == direction
            ]
            return [
                row.safe_payload()
                for row in sorted(
                    rows,
                    key=lambda value: (
                        value.fraction,
                        value.numerator,
                        value.denominator,
                    ),
                )
            ]

        observables.extend(
            (
                _observable(
                    observable_id="boundary_declaration",
                    value_type=specs[
                        "boundary_declaration"
                    ].value_type,
                    value_payload=(
                        None
                        if not scopes
                        else {
                            "boundary_id": "role:system_boundary",
                            "complete": scopes[0].attrs["value"],
                        }
                    ),
                    support=scopes,
                    spans_by_id=spans_by_id,
                    unknown=not scopes,
                ),
                _observable(
                    observable_id="quantity_ledger",
                    value_type=specs["quantity_ledger"].value_type,
                    value_payload={
                        "storage_before": _as_rational(
                            before.attrs["value"], "before.value"
                        ).safe_payload(),
                        "storage_after": _as_rational(
                            after.attrs["value"], "after.value"
                        ).safe_payload(),
                        "inflows": values("inflow"),
                        "outflows": values("outflow"),
                        "sources": values("source"),
                        "sinks": values("sink"),
                    },
                    support=(before, after, *transfers),
                    spans_by_id=spans_by_id,
                    dimension=str(before.attrs["dimension"]),
                    unit=str(before.attrs["unit"]),
                ),
            )
        )
    elif proposal.law_id == T09:
        domain = _one(_where(assertions, predicate="set"), "domain")
        first = _one(_where(maps, stage="step_a"), "first map")
        second = _one(_where(maps, stage="step_b"), "second map")
        def map_payload(row: NeutralEvidenceRecord) -> dict[str, Any]:
            return {
                "rows": [
                    {
                        "source": _canonical_ref(source, role_map),
                        "target": _canonical_ref(target, role_map),
                    }
                    for source, target in _as_map_rows(
                        row.attrs["rows"], "map.rows"
                    )
                ]
            }

        domain_values = set(
            _as_string_list(domain.attrs["values"], "domain.values")
        )
        direct = tuple(
            row
            for row in _where(maps, stage="reference")
            if set(
                dict(
                    _as_map_rows(row.attrs["rows"], "direct.rows")
                )
            )
            == domain_values
        )
        if len(direct) > 1:
            raise GSCLEvidenceExtractionError("direct map ambiguous")

        observables.extend(
            (
                _observable(
                    observable_id="finite_domain",
                    value_type=specs["finite_domain"].value_type,
                    value_payload={
                        "values": sorted(
                            _canonical_ref(value, role_map)
                            for value in _as_string_list(
                                domain.attrs["values"], "domain.values"
                            )
                        )
                    },
                    support=(domain,),
                    spans_by_id=spans_by_id,
                ),
                _observable(
                    observable_id="first_map",
                    value_type=specs["first_map"].value_type,
                    value_payload=map_payload(first),
                    support=(first,),
                    spans_by_id=spans_by_id,
                ),
                _observable(
                    observable_id="second_map",
                    value_type=specs["second_map"].value_type,
                    value_payload=map_payload(second),
                    support=(second,),
                    spans_by_id=spans_by_id,
                ),
                _observable(
                    observable_id="direct_map",
                    value_type=specs["direct_map"].value_type,
                    value_payload=(
                        None if not direct else map_payload(direct[0])
                    ),
                    support=direct,
                    spans_by_id=spans_by_id,
                    unknown=not direct,
                ),
            )
        )
    elif proposal.law_id == T05:
        component_ids = [
            role_map["component_a"],
            role_map["component_b"],
            role_map["component_c"],
        ]
        pair = _one(_where(assertions, predicate="set"), "pair")
        expectation = _one(
            _where(assertions, predicate="property"),
            "expectation",
        )
        outcomes = tuple(
            row for row in records if row.kind == "subset_outcome"
        )
        complete, malformed = _subset_coverage(component_ids, outcomes)
        if malformed:
            raise GSCLEvidenceExtractionError("subset lattice malformed")
        folds: list[dict[str, Any]] = []
        if complete:
            by_fold: dict[int, list[dict[str, Any]]] = {}
            for row in outcomes:
                by_fold.setdefault(int(row.attrs["fold"]), []).append(
                    {
                        "subset": sorted(
                            _canonical_ref(value, role_map)
                            for value in _as_string_list(
                                row.attrs["subset"],
                                "subset",
                                allow_empty=True,
                            )
                        ),
                        "utility": _as_rational(
                            row.attrs["utility"], "utility"
                        ).safe_payload(),
                    }
                )
            for rows in by_fold.values():
                rows.sort(
                    key=lambda item: (
                        len(item["subset"]),
                        tuple(item["subset"]),
                    )
                )
                folds.append({"rows": rows})
            folds.sort(key=strict_content_hash)
        dimension = (
            str(outcomes[0].attrs["dimension"])
            if outcomes
            else "Utility"
        )
        unit = str(outcomes[0].attrs["unit"]) if outcomes else "unitless"
        observables.extend(
            (
                _observable(
                    observable_id="components",
                    value_type=specs["components"].value_type,
                    value_payload={
                        "values": sorted(
                            _canonical_ref(value, role_map)
                            for value in component_ids
                        )
                    },
                    support=tuple(by_id[value] for value in component_ids),
                    spans_by_id=spans_by_id,
                ),
                _observable(
                    observable_id="designated_pair",
                    value_type=specs["designated_pair"].value_type,
                    value_payload={
                        "values": sorted(
                            _canonical_ref(value, role_map)
                            for value in _as_string_list(
                                pair.attrs["values"], "pair.values"
                            )
                        )
                    },
                    support=(pair,),
                    spans_by_id=spans_by_id,
                ),
                _observable(
                    observable_id="held_fold_utilities",
                    value_type=specs[
                        "held_fold_utilities"
                    ].value_type,
                    value_payload={"folds": folds} if complete else None,
                    support=outcomes if complete else (),
                    spans_by_id=spans_by_id,
                    dimension=dimension,
                    unit=unit,
                    unknown=not complete,
                ),
                _observable(
                    observable_id="interaction_expectation",
                    value_type=specs[
                        "interaction_expectation"
                    ].value_type,
                    value_payload={
                        "value": (
                            "complementary"
                            if expectation.attrs["value"]
                            == "positive_joint"
                            else expectation.attrs["value"]
                        )
                    },
                    support=(expectation,),
                    spans_by_id=spans_by_id,
                ),
            )
        )

    observables_by_id = {row.observable_id: row for row in observables}
    if set(observables_by_id) != set(specs):
        raise GSCLEvidenceExtractionError("observable coverage incomplete")

    constraint_role = next(
        role.role_id
        for role in schema.roles
        if role.target_kind is RoleTargetKind.CONSTRAINT
    )
    participants = tuple(
        ConstraintParticipant(
            participant_role=role.role_id,
            target_kind=role.target_kind,
            target_id=role_map[role.role_id],
        )
        for role in schema.roles
        if role.role_id != constraint_role
    )
    # The constraint is a derived CSP result, never a re-typed generic raw
    # assertion.  Its provenance includes every field span used by every
    # proposal condition (including the joint/association assertion), every
    # existing participant, and every observed supporting fact.  Unknown
    # synthetic targets correctly contribute no raw evidence.
    constraint_support = tuple(
        sorted(
            {
                *proposal.field_span_ids,
                *(
                    by_id[target_id].span.span_id
                    for role_id, target_id in role_map.items()
                    if role_id != constraint_role and target_id in by_id
                    and not (
                        by_id[target_id].kind == "observation"
                        and "value" not in by_id[target_id].attrs
                    )
                ),
                *(
                    span_id
                    for observable in observables
                    for span_id in observable.evidence_span_ids
                ),
            }
        )
    )
    constraint = StructuralConstraint(
        constraint_id=DERIVED_CONSTRAINT_ID,
        constraint_type=next(
            role.allowed_target_types[0]
            for role in schema.roles
            if role.role_id == constraint_role
        ),
        participants=participants,
        observable_ids=tuple(sorted(observables_by_id)),
        evidence_span_ids=constraint_support,
        observation_status=ObservationStatus.INFERRED,
        inference_provenance=_provenance(
            constraint_support, spans_by_id
        ),
    )

    # Re-label joint endpoint roles only now, after the unique CSP assignment.
    reverse_roles = {
        target_id: role_id
        for role_id, target_id in role_map.items()
        if role_id != constraint_role
    }
    hyperrelations = tuple(
        StructuralHyperrelation(
            hyperrelation_id=item.hyperrelation_id,
            hyperrelation_type=item.hyperrelation_type,
            endpoints=tuple(
                HyperRoleEndpoint(
                    endpoint_role=reverse_roles.get(
                        endpoint.object_id, endpoint.endpoint_role
                    ),
                    object_id=endpoint.object_id,
                )
                for endpoint in item.endpoints
            ),
            evidence_span_ids=item.evidence_span_ids,
            observation_status=item.observation_status,
            inference_provenance=item.inference_provenance,
        )
        for item in base.hyperrelations
    )
    episode = StructuralEpisode(
        episode_id=f"episode.bound.{proposal.derivation_hash[:20]}",
        source_sha256=base.source_sha256,
        evidence_spans=base.evidence_spans,
        objects=tuple(objects),
        relations=relations,
        quantities=quantities_tuple,
        hyperrelations=hyperrelations,
        constraints=(constraint,),
        observables=tuple(
            sorted(observables, key=lambda row: row.observable_id)
        ),
        declared_boundary_object_id=role_map.get("system_boundary"),
        missing_observables=tuple(
            sorted(
                row.observable_id
                for row in observables
                if row.observation_status is ObservationStatus.UNKNOWN
            )
        ),
    )
    episode_issues = episode.validate()
    if episode_issues:
        raise GSCLEvidenceExtractionError(
            "bound episode invalid: " + ",".join(episode_issues)
        )
    binding = LawBinding(
        binding_id=(
            "binding."
            + strict_content_hash(
                {
                    "proposal": proposal.derivation_hash,
                    "roles": dict(sorted(role_map.items())),
                    "episode": episode.episode_hash,
                }
            )[:24]
        ),
        law_id=proposal.law_id,
        registry_hash=registry.registry_hash,
        schema_hash=schema.schema_hash,
        episode_hash=episode.episode_hash,
        role_bindings=tuple(
            RoleBinding(
                role_id=role.role_id,
                target_id=role_map[role.role_id],
                evidence_span_ids=tuple(
                    sorted(
                        episode.require_target(
                            role.target_kind,
                            role_map[role.role_id],
                        ).evidence_span_ids
                    )
                ),
            )
            for role in schema.roles
        ),
        observable_bindings=tuple(
            ObservableBinding(
                observable_id=spec.observable_id,
                observable_hash=observables_by_id[
                    spec.observable_id
                ].observable_hash,
            )
            for spec in schema.required_observables
        ),
    )
    issues = validate_law_binding(registry, schema, episode, binding)
    if issues:
        raise GSCLEvidenceExtractionError(
            "binding validation failed: " + ",".join(issues)
        )
    return BoundStructuralCase(proposal, episode, binding)


def bind_structural_episode(
    registry: GSCLSchemaRegistry,
    extraction: StructuralExtraction,
    *,
    max_assignments: int = MAX_BINDING_ASSIGNMENTS,
    max_bindings_per_law: int = MAX_BINDINGS_PER_LAW,
) -> BindingSearch:
    """Solve all proposals and return exactly one bound case or abstain."""

    if (
        not isinstance(max_assignments, int)
        or isinstance(max_assignments, bool)
        or not 1 <= max_assignments <= MAX_BINDING_ASSIGNMENTS
        or not isinstance(max_bindings_per_law, int)
        or isinstance(max_bindings_per_law, bool)
        or not 1 <= max_bindings_per_law <= MAX_BINDINGS_PER_LAW
    ):
        return BindingSearch(
            bound_cases=(),
            assignment_count=0,
            issue_ids=("binder_budget_invalid",),
            truncated=False,
        )
    if not extraction.succeeded:
        return BindingSearch(
            bound_cases=(),
            assignment_count=0,
            issue_ids=("binder_extraction_invalid",),
            truncated=False,
        )
    if _base_episode_linkage_issues(extraction):
        return BindingSearch(
            bound_cases=(),
            assignment_count=0,
            issue_ids=("binder_extraction_linkage_invalid",),
            truncated=False,
        )
    linkage_issues = tuple(
        sorted(
            {
                issue
                for proposal in extraction.proposals
                for issue in _proposal_linkage_issues(
                    proposal, extraction
                )
            }
        )
    )
    if linkage_issues:
        return BindingSearch(
            bound_cases=(),
            assignment_count=0,
            issue_ids=("binder_proposal_linkage_invalid",),
            truncated=False,
        )

    assignment_count = 0
    truncated = False
    solutions: dict[str, BoundStructuralCase] = {}
    try:
        for proposal in extraction.proposals:
            schema = registry.require_law(proposal.law_id)
            domains = _role_domains(registry, proposal, extraction.records)
            if any(not domain for domain in domains):
                continue
            for assignment in product(*domains):
                assignment_count += 1
                if assignment_count > max_assignments:
                    truncated = True
                    break
                keys = tuple(
                    (role.target_kind, candidate.target_id)
                    for role, candidate in zip(
                        schema.roles, assignment
                    )
                )
                if len(keys) != len(set(keys)):
                    continue
                role_map = {
                    role.role_id: candidate.target_id
                    for role, candidate in zip(
                        schema.roles, assignment
                    )
                }
                if not _assignment_consistent(
                    proposal.law_id,
                    role_map,
                    tuple(
                        row
                        for row in extraction.records
                        if row.record_id in proposal.fact_ids
                    ),
                ):
                    continue
                case = _compile_bound_case(
                    registry, extraction, proposal, role_map
                )
                solutions[case.solution_hash] = case
                if len(solutions) > max_bindings_per_law:
                    truncated = True
                    break
            if truncated:
                break
    except RecursionError:
        return BindingSearch(
            bound_cases=(),
            assignment_count=assignment_count,
            issue_ids=("binder_recursion_limit_exceeded",),
            truncated=False,
        )
    except (
        GSCLEvidenceExtractionError,
        KeyError,
        PermissionError,
        TypeError,
        ValueError,
    ) as exc:
        return BindingSearch(
            bound_cases=(),
            assignment_count=assignment_count,
            issue_ids=(
                "binder_solution_compile_failed."
                f"{type(exc).__name__}",
            ),
            truncated=truncated,
        )

    if truncated:
        return BindingSearch(
            bound_cases=(),
            assignment_count=assignment_count,
            issue_ids=("binder_assignment_budget_exhausted",),
            truncated=True,
        )
    if not solutions:
        return BindingSearch(
            bound_cases=(),
            assignment_count=assignment_count,
            issue_ids=("binder_no_valid_binding",),
            truncated=False,
        )
    if len(solutions) != 1:
        return BindingSearch(
            bound_cases=(),
            assignment_count=assignment_count,
            issue_ids=("binder_ambiguous_binding",),
            truncated=False,
        )
    return BindingSearch(
        bound_cases=(next(iter(solutions.values())),),
        assignment_count=assignment_count,
        issue_ids=(),
        truncated=False,
    )


__all__ = [
    "BindingSearch",
    "BoundStructuralCase",
    "EXTRACTOR_CONTRACT",
    "EXTRACTOR_CONTRACT_HASH",
    "EXTRACTOR_VERSION",
    "FactFieldEvidence",
    "GSCLEvidenceExtractionError",
    "JSONL_MEDIA_TYPE",
    "MAX_BINDING_ASSIGNMENTS",
    "MAX_BINDINGS_PER_LAW",
    "NeutralEvidenceRecord",
    "PROSE_MEDIA_TYPE",
    "ProposalConditionEvidence",
    "SUPPORTED_MEDIA_TYPES",
    "StructuralExtraction",
    "StructuralProposal",
    "bind_structural_episode",
    "extract_structural_episode",
    "extractor_implementation_sha256",
]
