"""Pure algebra for the same-study GSCL/SCAR repair development.

This module deliberately has no filesystem, network, source-archive, model,
label-pack, or scorer I/O.  It accepts only already archived, caller-supplied
values and implements the frozen repair-development wire: strict commitments,
three-valued evidence, grouped folds, sixteen fixed features, a weighted
lambda-one ridge, conservative thresholding, pair utility, and the final
development-only verdict.

Nothing in this module grants confirmatory authority or changes the immutable
negative result of ``GSCL_SCAR_CSSM_INTRINSIC_FORMAL_V1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import hmac
import json
import math
import random
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "gscl_scar_cssm_repair_contract_v2"
FEATURE_WIDTH = 16
FOLD_COUNT = 5
COSINE_QUANTIZATION_SCALE = 1_000_000
MAXIMUM_ARITY = 14
MAXIMUM_DIAGNOSTIC_COUNT = 4_096
MAXIMUM_SCORE_ABS = 1_000_000_000
RIDGE_ALPHA = 1.0
RIDGE_SVD_RCOND = 1.0e-12
INNER_THRESHOLD_PRESERVATION_FLOOR = 0.99
OLD_SUCCESS_PRESERVATION_FLOOR = 0.98
# Compatibility name for the final, externally reported safety guardrail.
PRESERVATION_FLOOR = OLD_SUCCESS_PRESERVATION_FLOOR
UTILITY_MID = 0.01
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_ALPHA = 0.05

THRESHOLD_GRID = (
    0.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    math.inf,
)

FEATURE_ORDER = (
    "arity_fraction_of_14",
    "semantic_score_per_slot_quantum",
    "semantic_score_delta_per_slot_quantum",
    "flat_structural_score_per_slot",
    "typed_incidence_match_fraction",
    "typed_incidence_total_per_slot",
    "typed_incidence_zero_indicator",
    "semantic_origin_indicator",
    "structural_origin_indicator",
    "operator_orientation_inverted_indicator",
    "operator_polarity_inverted_indicator",
    "operator_slots_reversed_indicator",
    "mean_retained_edges_per_slot",
    "mean_dropped_edge_fraction",
    "mean_unbound_endpoint_fraction",
    "zero_degree_fraction_across_both_sides",
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_OPERATOR = re.compile(
    r"ori_(?P<orientation>keep|inv)\."
    r"pol_(?P<polarity>keep|inv)\."
    r"slots_(?P<slots>identity|reverse)\Z"
)

_ARCHIVED_ROW_FIELDS = frozenset(
    {"arity", "baseline_semantic_score", "proposal", "left_binder", "right_binder"}
)
_PROPOSAL_FIELDS = frozenset(
    {
        "selected_operator",
        "semantic_origin_count",
        "structural_origin_count",
        "incidence_match_count",
        "incidence_total_count",
        "length2_path_count",
        "length2_path_total_count",
        "typed_incidence_verified",
        "length2_composition_verified",
        "proposal_hash",
        "semantic_score",
        "flat_structural_score",
    }
)
_BINDER_FIELDS = frozenset(
    {
        "coverage_disposition",
        "unbound_count",
        "dropped_edge_count",
        "retained_edge_count",
        "zero_degree_count",
        "endpoint_count",
        "self_loop_count",
    }
)
_BINDER_COVERAGE_DISPOSITIONS = frozenset(
    {"COMPLETE_SELECTED_SET", "PARTIAL_SELECTED_SET", "EMPTY_ABSTENTION"}
)


class ScarRepairContractError(ValueError):
    """Stable, content-free failure for the pure repair contract."""

    _KNOWN = frozenset(
        {
            "SCAR_REPAIR_ARCHIVED_FEATURE_INVALID",
            "SCAR_REPAIR_BOOTSTRAP_INVALID",
            "SCAR_REPAIR_CANONICAL_JSON_INVALID",
            "SCAR_REPAIR_EVIDENCE_INVALID",
            "SCAR_REPAIR_FOLD_ASSIGNMENT_INVALID",
            "SCAR_REPAIR_PAIR_SET_INVALID",
            "SCAR_REPAIR_RIDGE_INPUT_INVALID",
            "SCAR_REPAIR_RIDGE_MODEL_INVALID",
            "SCAR_REPAIR_RIDGE_SOLUTION_INVALID",
            "SCAR_REPAIR_SELECTION_INVALID",
            "SCAR_REPAIR_SELF_SEAL_INVALID",
            "SCAR_REPAIR_THRESHOLD_INVALID",
            "SCAR_REPAIR_VERDICT_INPUT_INVALID",
        }
    )

    def __init__(self, issue_id: str) -> None:
        if issue_id not in self._KNOWN:
            raise ValueError("scar_repair_contract_issue_unknown")
        self.issue_id = issue_id
        super().__init__(issue_id)


def _strict_json(value: Any, *, path: str = "$") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        # The strict commitment wire never relies on implementation-dependent
        # JSON float spelling.  Numeric model values are committed as hex text.
        raise ScarRepairContractError("SCAR_REPAIR_CANONICAL_JSON_INVALID")
    if type(value) is list:
        for index, row in enumerate(value):
            _strict_json(row, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ScarRepairContractError("SCAR_REPAIR_CANONICAL_JSON_INVALID")
        for key, row in value.items():
            _strict_json(row, path=f"{path}.{key}")
        return
    raise ScarRepairContractError("SCAR_REPAIR_CANONICAL_JSON_INVALID")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole strict JSON serialization accepted by this contract."""

    _strict_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ScarRepairContractError(
            "SCAR_REPAIR_CANONICAL_JSON_INVALID"
        ) from exc


def canonical_bytes(value: Any) -> bytes:
    """Compatibility spelling for the contract's sole canonical wire."""

    return canonical_json_bytes(value)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def domain_hash(domain: str, value: Any) -> str:
    """Hash a canonical payload under an unambiguous caller-supplied domain."""

    if type(domain) is not str or not domain:
        raise ScarRepairContractError("SCAR_REPAIR_CANONICAL_JSON_INVALID")
    try:
        domain_wire = domain.encode("ascii")
    except UnicodeError as exc:
        raise ScarRepairContractError(
            "SCAR_REPAIR_CANONICAL_JSON_INVALID"
        ) from exc
    wire = (
        len(domain_wire).to_bytes(8, byteorder="big", signed=False)
        + domain_wire
        + canonical_json_bytes(value)
    )
    return hashlib.sha256(wire).hexdigest()


def seal_payload(
    domain_or_payload: str | Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy and self-seal one strict mapping without mutating the caller.

    The two-argument form is the public, domain-separated wire.  The legacy
    one-argument form remains available for already-bound internal callers.
    """

    domain: str | None
    if payload is None:
        domain = None
        candidate = domain_or_payload
    else:
        if type(domain_or_payload) is not str:
            raise ScarRepairContractError("SCAR_REPAIR_SELF_SEAL_INVALID")
        domain = domain_or_payload
        candidate = payload

    if type(candidate) is not dict or "self_sha256" in candidate:
        raise ScarRepairContractError("SCAR_REPAIR_SELF_SEAL_INVALID")
    body = dict(candidate)
    _strict_json(body)
    digest = content_hash(body) if domain is None else domain_hash(domain, body)
    return {**body, "self_sha256": digest}


def validate_self_seal(
    domain_or_payload: str | Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
    *,
    expected_schema: str | None = None,
) -> Mapping[str, Any] | bool:
    """Validate a self seal.

    Domain-separated calls return ``True``.  The legacy one-argument form
    returns an immutable validated view, preserving its earlier contract.
    """

    domain: str | None
    if payload is None:
        domain = None
        candidate = domain_or_payload
    else:
        if type(domain_or_payload) is not str:
            raise ScarRepairContractError("SCAR_REPAIR_SELF_SEAL_INVALID")
        domain = domain_or_payload
        candidate = payload

    if type(candidate) is not dict or type(candidate.get("self_sha256")) is not str:
        raise ScarRepairContractError("SCAR_REPAIR_SELF_SEAL_INVALID")
    if expected_schema is not None and (
        type(expected_schema) is not str
        or not expected_schema
        or candidate.get("schema") != expected_schema
    ):
        raise ScarRepairContractError("SCAR_REPAIR_SELF_SEAL_INVALID")
    body = dict(candidate)
    claimed = body.pop("self_sha256")
    if _HEX64.fullmatch(claimed) is None:
        raise ScarRepairContractError("SCAR_REPAIR_SELF_SEAL_INVALID")
    try:
        actual = content_hash(body) if domain is None else domain_hash(domain, body)
    except ScarRepairContractError as exc:
        raise ScarRepairContractError("SCAR_REPAIR_SELF_SEAL_INVALID") from exc
    if not hmac.compare_digest(claimed, actual):
        raise ScarRepairContractError("SCAR_REPAIR_SELF_SEAL_INVALID")
    if domain is not None:
        return True
    return MappingProxyType(dict(candidate))


def parse_canonical_json_bytes(
    raw: bytes, *, expected_schema: str | None = None, require_self_seal: bool = False
) -> Mapping[str, Any]:
    """Parse bytes only when they are byte-exact strict canonical JSON."""

    if type(raw) is not bytes:
        raise ScarRepairContractError("SCAR_REPAIR_CANONICAL_JSON_INVALID")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ScarRepairContractError(
            "SCAR_REPAIR_CANONICAL_JSON_INVALID"
        ) from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ScarRepairContractError("SCAR_REPAIR_CANONICAL_JSON_INVALID")
    if require_self_seal:
        return validate_self_seal(value, expected_schema=expected_schema)
    if expected_schema is not None and value.get("schema") != expected_schema:
        raise ScarRepairContractError("SCAR_REPAIR_CANONICAL_JSON_INVALID")
    return MappingProxyType(value)


class EvidenceValue(str, Enum):
    """The three values used by the open-world evidence algebra."""

    M = "M"
    V = "V"
    U = "U"
    MATCH = "M"
    VIOLATION = "V"
    UNKNOWN = "U"


def _evidence_values(values: object) -> tuple[EvidenceValue, ...]:
    if isinstance(values, EvidenceValue):
        return (values,)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ScarRepairContractError("SCAR_REPAIR_EVIDENCE_INVALID")
    rows = tuple(values)
    if any(not isinstance(row, EvidenceValue) for row in rows):
        raise ScarRepairContractError("SCAR_REPAIR_EVIDENCE_INVALID")
    return rows  # type: ignore[return-value]


def strong_kleene_and(values: Sequence[EvidenceValue]) -> EvidenceValue:
    """Non-vacuous strong-Kleene conjunction (V dominates, then U)."""

    rows = _evidence_values(values)
    if not rows:
        return EvidenceValue.U
    if EvidenceValue.V in rows:
        return EvidenceValue.V
    if EvidenceValue.U in rows:
        return EvidenceValue.U
    return EvidenceValue.M


def exhaustive_or(
    values: Sequence[EvidenceValue], *, exhaustive: bool
) -> EvidenceValue:
    """Disjoin evidence; all-V is V only for an explicitly exhaustive set."""

    if type(exhaustive) is not bool:
        raise ScarRepairContractError("SCAR_REPAIR_EVIDENCE_INVALID")
    rows = _evidence_values(values)
    if not rows:
        return EvidenceValue.U
    if EvidenceValue.M in rows:
        return EvidenceValue.M
    if exhaustive and all(row is EvidenceValue.V for row in rows):
        return EvidenceValue.V
    return EvidenceValue.U


def _parse_evidence_token(value: object) -> EvidenceValue:
    if type(value) is not str or value not in {"M", "V", "U"}:
        raise ScarRepairContractError("SCAR_REPAIR_EVIDENCE_INVALID")
    return EvidenceValue(value)


def conjoin_evidence(left: str, right: str) -> str:
    """Binary strong-Kleene conjunction on canonical wire tokens."""

    return strong_kleene_and(
        (_parse_evidence_token(left), _parse_evidence_token(right))
    ).value


def exhaustive_or_evidence(
    states: Sequence[str], *, domain_complete: bool
) -> str:
    """Open-world disjunction on canonical wire tokens.

    Empty evidence is rejected at this public boundary: a caller must supply
    an explicit ``U`` rather than rely on vacuity.
    """

    if (
        isinstance(states, (str, bytes))
        or not isinstance(states, Sequence)
        or not states
        or type(domain_complete) is not bool
    ):
        raise ScarRepairContractError("SCAR_REPAIR_EVIDENCE_INVALID")
    parsed = tuple(_parse_evidence_token(row) for row in states)
    return exhaustive_or(parsed, exhaustive=domain_complete).value


def combine_three_valued_evidence(
    match_count: int, violation_count: int, unknown_count: int
) -> EvidenceValue:
    """Aggregate counts with the frozen non-vacuous strong-AND rule."""

    counts = (match_count, violation_count, unknown_count)
    if any(type(row) is not int or row < 0 for row in counts):
        raise ScarRepairContractError("SCAR_REPAIR_EVIDENCE_INVALID")
    if violation_count:
        return EvidenceValue.V
    if unknown_count:
        return EvidenceValue.U
    if match_count:
        return EvidenceValue.M
    return EvidenceValue.U


def assign_grouped_folds(
    group_ids: Sequence[str],
    fold_count: int = FOLD_COUNT,
    binding_root: str = "0" * 64,
    *,
    n_folds: int | None = None,
) -> tuple[int, ...]:
    """Assign duplicate group IDs together using hash-sort then round-robin."""

    if (
        isinstance(group_ids, (str, bytes))
        or not isinstance(group_ids, Sequence)
        or not group_ids
        or (n_folds is not None and fold_count != FOLD_COUNT)
        or type(n_folds if n_folds is not None else fold_count) is not int
        or not 2 <= (n_folds if n_folds is not None else fold_count) <= 64
        or type(binding_root) is not str
        or _HEX64.fullmatch(binding_root) is None
        or any(type(row) is not str or not row for row in group_ids)
    ):
        raise ScarRepairContractError("SCAR_REPAIR_FOLD_ASSIGNMENT_INVALID")
    resolved_fold_count = n_folds if n_folds is not None else fold_count
    assert resolved_fold_count is not None
    unique = set(group_ids)
    ordered = sorted(
        unique,
        key=lambda group_id: (
            content_hash(
                {
                    "binding_root": binding_root,
                    "group_id": group_id,
                    "version": VERSION,
                }
            ),
            group_id,
        ),
    )
    fold_by_group = {
        group_id: ordinal % resolved_fold_count
        for ordinal, group_id in enumerate(ordered)
    }
    return tuple(fold_by_group[row] for row in group_ids)


def _exact_int(value: object, *, minimum: int = 0, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    return value


def _validate_binder(value: object, *, arity: int) -> dict[str, int | str]:
    if type(value) is not dict or set(value) != _BINDER_FIELDS:
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    if value["coverage_disposition"] not in _BINDER_COVERAGE_DISPOSITIONS:
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    counts = {
        key: _exact_int(value[key], maximum=MAXIMUM_DIAGNOSTIC_COUNT)
        for key in (
            "unbound_count",
            "dropped_edge_count",
            "retained_edge_count",
            "zero_degree_count",
            "endpoint_count",
            "self_loop_count",
        )
    }
    if (
        counts["unbound_count"] > counts["endpoint_count"]
        or counts["self_loop_count"] > counts["retained_edge_count"]
        or counts["zero_degree_count"] > arity
    ):
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    return {"coverage_disposition": value["coverage_disposition"], **counts}


def _validate_proposal(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _PROPOSAL_FIELDS:
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    operator = value["selected_operator"]
    proposal_hash = value["proposal_hash"]
    if (
        type(operator) is not str
        or _OPERATOR.fullmatch(operator) is None
        or type(proposal_hash) is not str
        or _HEX64.fullmatch(proposal_hash) is None
    ):
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    semantic_origin = _exact_int(value["semantic_origin_count"], maximum=1)
    structural_origin = _exact_int(value["structural_origin_count"], maximum=1)
    if not (semantic_origin or structural_origin):
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    incidence_matched = _exact_int(
        value["incidence_match_count"], maximum=MAXIMUM_DIAGNOSTIC_COUNT
    )
    incidence_total = _exact_int(
        value["incidence_total_count"], maximum=MAXIMUM_DIAGNOSTIC_COUNT
    )
    path_matched = _exact_int(
        value["length2_path_count"], maximum=MAXIMUM_DIAGNOSTIC_COUNT
    )
    path_total = _exact_int(
        value["length2_path_total_count"], maximum=MAXIMUM_DIAGNOSTIC_COUNT
    )
    if (
        incidence_matched > incidence_total
        or path_matched > path_total
        or type(value["typed_incidence_verified"]) is not bool
        or type(value["length2_composition_verified"]) is not bool
    ):
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    if value["typed_incidence_verified"] and (
        incidence_total == 0 or incidence_matched != incidence_total
    ):
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    if value["length2_composition_verified"] and (
        not value["typed_incidence_verified"]
        or path_total == 0
        or path_matched != path_total
    ):
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    for key in ("semantic_score", "flat_structural_score"):
        if (
            type(value[key]) is not int
            or abs(value[key]) > MAXIMUM_SCORE_ABS
        ):
            raise ScarRepairContractError(
                "SCAR_REPAIR_ARCHIVED_FEATURE_INVALID"
            )
    return dict(value)


def extract_archived_features(value: Mapping[str, object]) -> tuple[float, ...]:
    """Extract the exact frozen 16-vector from one archived proposal row."""

    if type(value) is not dict or set(value) != _ARCHIVED_ROW_FIELDS:
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    arity = _exact_int(value["arity"], minimum=1, maximum=MAXIMUM_ARITY)
    baseline = value["baseline_semantic_score"]
    if type(baseline) is not int or abs(baseline) > MAXIMUM_SCORE_ABS:
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    proposal = _validate_proposal(value["proposal"])
    left = _validate_binder(value["left_binder"], arity=arity)
    right = _validate_binder(value["right_binder"], arity=arity)
    operator = _OPERATOR.fullmatch(str(proposal["selected_operator"]))
    assert operator is not None

    semantic = int(proposal["semantic_score"])
    flat = int(proposal["flat_structural_score"])
    incidence_matched = int(proposal["incidence_match_count"])
    incidence_total = int(proposal["incidence_total_count"])
    retained_mean_per_slot = (
        int(left["retained_edge_count"]) + int(right["retained_edge_count"])
    ) / (2.0 * arity)

    drop_fractions = []
    unbound_fractions = []
    for binder in (left, right):
        retained = int(binder["retained_edge_count"])
        dropped = int(binder["dropped_edge_count"])
        endpoints = int(binder["endpoint_count"])
        unbound = int(binder["unbound_count"])
        edge_denominator = retained + dropped
        drop_fractions.append(
            dropped / edge_denominator if edge_denominator else 0.0
        )
        unbound_fractions.append(unbound / max(endpoints, 1))

    features = (
        arity / float(MAXIMUM_ARITY),
        semantic / float(arity * COSINE_QUANTIZATION_SCALE),
        (semantic - baseline) / float(arity * COSINE_QUANTIZATION_SCALE),
        flat / float(arity),
        incidence_matched / float(max(incidence_total, 1)),
        incidence_total / float(arity),
        float(incidence_total == 0),
        float(int(proposal["semantic_origin_count"])),
        float(int(proposal["structural_origin_count"])),
        float(operator.group("orientation") == "inv"),
        float(operator.group("polarity") == "inv"),
        float(operator.group("slots") == "reverse"),
        retained_mean_per_slot,
        math.fsum(drop_fractions) / 2.0,
        math.fsum(unbound_fractions) / 2.0,
        (
            int(left["zero_degree_count"]) + int(right["zero_degree_count"])
        )
        / float(2 * arity),
    )
    if len(features) != FEATURE_WIDTH or not all(
        math.isfinite(row) for row in features
    ):
        raise ScarRepairContractError("SCAR_REPAIR_ARCHIVED_FEATURE_INVALID")
    return features


# A short name for callers that already bind the archived schema externally.
extract_fixed_features = extract_archived_features


def _finite_matrix(value: object, *, expected_width: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_INPUT_INVALID") from exc
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[1] < 1
        or (expected_width is not None and result.shape[1] != expected_width)
        or not np.isfinite(result).all()
    ):
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_INPUT_INVALID")
    return result


def _finite_vector(value: object, *, length: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_INPUT_INVALID") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_INPUT_INVALID")
    return result


@dataclass(frozen=True, slots=True)
class StandardizedRidgeModel:
    """Weighted-population-standardized, unpenalized-intercept ridge."""

    population_mean: tuple[float, ...]
    population_std: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    alpha: float = RIDGE_ALPHA
    solver: str = "numpy_float64_svd_rcond_1e-12_v1"

    def __post_init__(self) -> None:
        width = len(self.population_mean)
        if (
            width < 1
            or len(self.population_std) != width
            or len(self.coefficients) != width
            or self.alpha != RIDGE_ALPHA
            or self.solver != "numpy_float64_svd_rcond_1e-12_v1"
            or any(row < 0.0 for row in self.population_std)
            or not all(
                math.isfinite(row)
                for row in (
                    *self.population_mean,
                    *self.population_std,
                    self.intercept,
                    *self.coefficients,
                )
            )
        ):
            raise ScarRepairContractError("SCAR_REPAIR_RIDGE_MODEL_INVALID")

    @property
    def feature_means(self) -> tuple[float, ...]:
        return self.population_mean

    @property
    def feature_scales(self) -> tuple[float, ...]:
        # Expose the conventional unit scale for an inert zero-variance
        # column, while retaining the exact population SD=0 commitment.
        return tuple(row if row != 0.0 else 1.0 for row in self.population_std)

    def standardize(self, features: Sequence[float]) -> tuple[float, ...]:
        width = len(self.population_mean)
        values = _finite_vector(features, length=width)
        standardized = np.divide(
            values - np.asarray(self.population_mean, dtype=np.float64),
            np.asarray(self.population_std, dtype=np.float64),
            out=np.zeros(width, dtype=np.float64),
            where=np.asarray(self.population_std, dtype=np.float64) != 0.0,
        )
        return tuple(float(row) for row in standardized)

    def _predict_one(self, features: Sequence[float]) -> float:
        standardized = self.standardize(features)
        prediction = self.intercept + math.fsum(
            coefficient * feature
            for coefficient, feature in zip(self.coefficients, standardized)
        )
        if not math.isfinite(prediction):
            raise ScarRepairContractError("SCAR_REPAIR_RIDGE_MODEL_INVALID")
        return max(-1.0, min(1.0, prediction))

    def predict(
        self, features: Sequence[float] | Sequence[Sequence[float]] | np.ndarray
    ) -> float | tuple[float, ...]:
        """Predict one row or a non-empty matrix using the same fitted wire."""

        try:
            array = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ScarRepairContractError("SCAR_REPAIR_RIDGE_MODEL_INVALID") from exc
        if array.ndim == 1:
            return self._predict_one(array)
        if array.ndim == 2:
            matrix = _finite_matrix(array, expected_width=len(self.population_mean))
            return tuple(self._predict_one(row) for row in matrix)
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_MODEL_INVALID")

    def payload(self) -> dict[str, object]:
        return {
            "alpha_float64_hex": self.alpha.hex(),
            "coefficient_float64_hex": [row.hex() for row in self.coefficients],
            "feature_order": (
                list(FEATURE_ORDER)
                if len(self.population_mean) == FEATURE_WIDTH
                else [f"feature_{index:02d}" for index in range(len(self.population_mean))]
            ),
            "intercept_float64_hex": self.intercept.hex(),
            "intercept_penalized": False,
            "population_mean_float64_hex": [
                row.hex() for row in self.population_mean
            ],
            "population_std_float64_hex": [
                row.hex() for row in self.population_std
            ],
            "scaler": "training_weighted_population_mean_and_std_v1",
            "prediction_clip": ["-1", "1"],
            "solver": self.solver,
            "version": VERSION,
            "zero_variance_maps_to_zero": True,
        }

    @property
    def commitment(self) -> str:
        return content_hash(self.payload())


def fit_standardized_ridge(
    X: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    l2: float = RIDGE_ALPHA,
    *,
    sample_weights: Sequence[float] | np.ndarray | None = None,
) -> StandardizedRidgeModel:
    """Fit the frozen float64 alpha-one ridge through one SVD code path."""

    if type(l2) not in {int, float} or isinstance(l2, bool) or float(l2) != RIDGE_ALPHA:
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_INPUT_INVALID")
    matrix = _finite_matrix(X)
    width = matrix.shape[1]
    target = _finite_vector(y, length=matrix.shape[0])
    weights = (
        np.ones(matrix.shape[0], dtype=np.float64)
        if sample_weights is None
        else _finite_vector(sample_weights, length=matrix.shape[0])
    )
    if np.any(weights <= 0.0) or not math.isfinite(float(np.sum(weights))):
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_INPUT_INVALID")
    weight_sum = float(np.sum(weights, dtype=np.float64))
    means = np.sum(matrix * weights[:, None], axis=0, dtype=np.float64) / weight_sum
    centered = matrix - means
    variance = (
        np.sum(centered * centered * weights[:, None], axis=0, dtype=np.float64)
        / weight_sum
    )
    variance = np.maximum(variance, 0.0)
    stds = np.sqrt(variance, dtype=np.float64)
    standardized = np.divide(
        centered,
        stds,
        out=np.zeros_like(centered),
        where=stds != 0.0,
    )
    design = np.column_stack(
        (np.ones(matrix.shape[0], dtype=np.float64), standardized)
    )
    root_weights = np.sqrt(weights, dtype=np.float64)
    weighted_design = design * root_weights[:, None]
    weighted_target = target * root_weights
    penalty = np.zeros((width, width + 1), dtype=np.float64)
    penalty[:, 1:] = np.eye(width, dtype=np.float64) * math.sqrt(
        RIDGE_ALPHA
    )
    augmented_design = np.vstack((weighted_design, penalty))
    augmented_target = np.concatenate(
        (weighted_target, np.zeros(width, dtype=np.float64))
    )
    try:
        left, singular, right_t = np.linalg.svd(
            augmented_design, full_matrices=False
        )
    except np.linalg.LinAlgError as exc:
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_SOLUTION_INVALID") from exc
    if singular.size == 0 or not np.isfinite(singular).all():
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_SOLUTION_INVALID")
    cutoff = float(singular[0]) * RIDGE_SVD_RCOND
    inverse = np.divide(
        1.0,
        singular,
        out=np.zeros_like(singular),
        where=singular > cutoff,
    )
    fitted = right_t.T @ (inverse * (left.T @ augmented_target))
    if fitted.shape != (width + 1,) or not np.isfinite(fitted).all():
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_SOLUTION_INVALID")
    fitted[1:][stds == 0.0] = 0.0
    return StandardizedRidgeModel(
        population_mean=tuple(float(row) for row in means),
        population_std=tuple(float(row) for row in stds),
        intercept=float(fitted[0]),
        coefficients=tuple(float(row) for row in fitted[1:]),
    )


def predict_standardized_ridge(
    model: StandardizedRidgeModel, rows: Sequence[Sequence[float]] | np.ndarray
) -> tuple[float, ...]:
    if not isinstance(model, StandardizedRidgeModel):
        raise ScarRepairContractError("SCAR_REPAIR_RIDGE_MODEL_INVALID")
    matrix = _finite_matrix(rows, expected_width=len(model.population_mean))
    return tuple(model._predict_one(row) for row in matrix)


def _finite_rows(value: object, *, issue_id: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScarRepairContractError(issue_id)
    rows: list[float] = []
    for row in value:
        if type(row) not in {int, float} or isinstance(row, bool):
            raise ScarRepairContractError(issue_id)
        parsed = float(row)
        if not math.isfinite(parsed):
            raise ScarRepairContractError(issue_id)
        rows.append(parsed)
    return tuple(rows)


def _validated_grid(value: Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
    rows: list[float] = []
    for row in value:
        if type(row) not in {int, float} or isinstance(row, bool):
            raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
        parsed = float(row)
        if math.isnan(parsed) or parsed < 0.0:
            raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
        rows.append(parsed)
    if not rows or len(set(rows)) != len(rows) or math.inf not in rows:
        raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
    return tuple(sorted(rows))


def _select_override_threshold_arrays(
    scores: Sequence[float],
    deltas: Sequence[float],
    preservation: Sequence[float | None],
    grid: Sequence[float] = THRESHOLD_GRID,
    *,
    preservation_floor: float = INNER_THRESHOLD_PRESERVATION_FLOOR,
) -> float:
    """Choose max summed utility, then the higher threshold, under safety."""

    normalized_scores = _finite_rows(
        scores, issue_id="SCAR_REPAIR_THRESHOLD_INVALID"
    )
    normalized_deltas = _finite_rows(
        deltas, issue_id="SCAR_REPAIR_THRESHOLD_INVALID"
    )
    if isinstance(preservation, (str, bytes)) or not isinstance(
        preservation, Sequence
    ):
        raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
    normalized_preservation: tuple[float | None, ...] = tuple(
        None
        if row is None
        else (
            float(row)
            if type(row) in {int, float} and not isinstance(row, bool)
            else math.nan
        )
        for row in preservation
    )
    if (
        not normalized_scores
        or len(normalized_scores) != len(normalized_deltas)
        or len(normalized_scores) != len(normalized_preservation)
        or any(
            row is not None and (not math.isfinite(row) or not 0.0 <= row <= 1.0)
            for row in normalized_preservation
        )
        or type(preservation_floor) not in {int, float}
        or isinstance(preservation_floor, bool)
        or not 0.0 <= float(preservation_floor) <= 1.0
    ):
        raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
    thresholds = _validated_grid(grid)
    feasible: list[tuple[float, float]] = []
    perfect_indices = tuple(
        index for index, row in enumerate(normalized_preservation) if row is not None
    )
    for threshold in thresholds:
        selected = tuple(
            index
            for index, score in enumerate(normalized_scores)
            if math.isfinite(threshold) and score > threshold
        )
        utility = math.fsum(normalized_deltas[index] for index in selected)
        preservation_rate = (
            math.fsum(
                float(normalized_preservation[index])
                if index in selected
                else 1.0
                for index in perfect_indices
            )
            / len(perfect_indices)
            if perfect_indices
            else 1.0
        )
        if preservation_rate >= float(preservation_floor):
            feasible.append((utility, threshold))
    if not feasible:
        # The +inf no-op must always be feasible for a valid preservation floor.
        raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
    best_utility, best_threshold = max(feasible, key=lambda row: (row[0], row[1]))
    return best_threshold if best_utility > 0.0 else math.inf


@dataclass(frozen=True, slots=True)
class ThresholdExample:
    """One fixed inner-OOF item supplied to threshold selection."""

    selector_score: int | float | Fraction
    utility_delta: int | float | Fraction
    old_success_count: int = 0
    override_preserved_count: int = 0


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: int | float | Fraction
    net_utility_delta: Fraction
    override_count: int
    preservation: Fraction


def _finite_fraction(value: object, *, issue_id: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ScarRepairContractError(issue_id)
    if isinstance(value, float) and not math.isfinite(value):
        raise ScarRepairContractError(issue_id)
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ScarRepairContractError(issue_id) from exc
    return result


def _select_override_threshold_examples(
    examples: Sequence[ThresholdExample],
    *,
    thresholds: Sequence[int | float | Fraction],
    minimum_preservation: int | float | Fraction,
) -> ThresholdSelection:
    issue_id = "SCAR_REPAIR_THRESHOLD_INVALID"
    if (
        isinstance(examples, (str, bytes))
        or not isinstance(examples, Sequence)
        or not examples
        or isinstance(thresholds, (str, bytes))
        or not isinstance(thresholds, Sequence)
        or not thresholds
    ):
        raise ScarRepairContractError(issue_id)
    minimum = _finite_fraction(minimum_preservation, issue_id=issue_id)
    if not 0 <= minimum <= 1:
        raise ScarRepairContractError(issue_id)

    normalized_examples: list[tuple[Fraction, Fraction, int, int]] = []
    for row in examples:
        if not isinstance(row, ThresholdExample):
            raise ScarRepairContractError(issue_id)
        score = _finite_fraction(row.selector_score, issue_id=issue_id)
        delta = _finite_fraction(row.utility_delta, issue_id=issue_id)
        if (
            type(row.old_success_count) is not int
            or row.old_success_count < 0
            or type(row.override_preserved_count) is not int
            or row.override_preserved_count < 0
            or row.override_preserved_count > row.old_success_count
        ):
            raise ScarRepairContractError(issue_id)
        normalized_examples.append(
            (score, delta, row.old_success_count, row.override_preserved_count)
        )

    normalized_thresholds: list[tuple[int | float | Fraction, Fraction | None]] = []
    seen: set[tuple[str, object]] = set()
    for threshold in thresholds:
        if isinstance(threshold, float) and math.isinf(threshold):
            if threshold < 0:
                raise ScarRepairContractError(issue_id)
            normalized = None
            identity = ("inf", 1)
        else:
            normalized = _finite_fraction(threshold, issue_id=issue_id)
            if normalized < 0:
                raise ScarRepairContractError(issue_id)
            identity = ("finite", normalized)
        if identity in seen:
            raise ScarRepairContractError(issue_id)
        seen.add(identity)
        normalized_thresholds.append((threshold, normalized))

    old_success_total = sum(row[2] for row in normalized_examples)
    feasible: list[ThresholdSelection] = []
    for original_threshold, threshold in normalized_thresholds:
        selected_indices = tuple(
            index
            for index, row in enumerate(normalized_examples)
            if threshold is not None and row[0] > threshold
        )
        selected_set = frozenset(selected_indices)
        utility = sum(
            (normalized_examples[index][1] for index in selected_indices),
            Fraction(0),
        )
        preserved = sum(
            row[3] if index in selected_set else row[2]
            for index, row in enumerate(normalized_examples)
        )
        preservation_rate = (
            Fraction(preserved, old_success_total)
            if old_success_total
            else Fraction(1)
        )
        if preservation_rate >= minimum:
            feasible.append(
                ThresholdSelection(
                    threshold=original_threshold,
                    net_utility_delta=utility,
                    override_count=len(selected_indices),
                    preservation=preservation_rate,
                )
            )
    if not feasible:
        raise ScarRepairContractError(issue_id)

    def rank(row: ThresholdSelection) -> tuple[Fraction, float, int]:
        numeric_threshold = (
            math.inf
            if isinstance(row.threshold, float) and math.isinf(row.threshold)
            else float(Fraction(row.threshold))
        )
        return (row.net_utility_delta, numeric_threshold, -row.override_count)

    best = max(feasible, key=rank)
    if best.net_utility_delta < 0:
        no_ops = [row for row in feasible if row.override_count == 0]
        if not no_ops:
            raise ScarRepairContractError(issue_id)
        best = max(no_ops, key=rank)
    return best


def select_override_threshold(
    examples_or_scores: Sequence[ThresholdExample] | Sequence[float],
    deltas: Sequence[float] | None = None,
    preservation: Sequence[float | None] | None = None,
    grid: Sequence[float] = THRESHOLD_GRID,
    *,
    thresholds: Sequence[int | float | Fraction] | None = None,
    minimum_preservation: int | float | Fraction | None = None,
    preservation_floor: float = INNER_THRESHOLD_PRESERVATION_FLOOR,
) -> ThresholdSelection | float:
    """Select the conservative threshold on examples or legacy parallel rows."""

    if thresholds is not None or minimum_preservation is not None:
        if deltas is not None or preservation is not None:
            raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
        resolved_thresholds = grid if thresholds is None else thresholds
        resolved_minimum = (
            preservation_floor
            if minimum_preservation is None
            else minimum_preservation
        )
        return _select_override_threshold_examples(
            examples_or_scores,  # type: ignore[arg-type]
            thresholds=resolved_thresholds,
            minimum_preservation=resolved_minimum,
        )
    if deltas is None or preservation is None:
        raise ScarRepairContractError("SCAR_REPAIR_THRESHOLD_INVALID")
    return _select_override_threshold_arrays(
        examples_or_scores,  # type: ignore[arg-type]
        deltas,
        preservation,
        grid,
        preservation_floor=preservation_floor,
    )


def select_action(
    baseline: Any,
    override: Any,
    eligible: bool,
    score: float,
    threshold: float,
) -> Any:
    """Return the exact baseline object unless every override condition holds."""

    if (
        type(eligible) is not bool
        or type(score) not in {int, float}
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or type(threshold) not in {int, float}
        or isinstance(threshold, bool)
        or math.isnan(float(threshold))
        or float(threshold) < 0.0
    ):
        raise ScarRepairContractError("SCAR_REPAIR_SELECTION_INVALID")
    try:
        baseline_bytes = canonical_json_bytes(baseline)
        override_bytes = canonical_json_bytes(override)
    except ScarRepairContractError as exc:
        raise ScarRepairContractError("SCAR_REPAIR_SELECTION_INVALID") from exc
    if (
        not eligible
        or math.isinf(float(threshold))
        or float(score) <= float(threshold)
        or baseline_bytes == override_bytes
    ):
        return baseline
    return override


def select_action_output(
    baseline_output: bytes,
    override_output: bytes,
    *,
    structurally_eligible: bool,
    selector_score: int | float | Fraction,
    threshold: int | float | Fraction,
) -> bytes:
    """Select serialized action bytes while preserving an exact S0 no-op."""

    if (
        type(baseline_output) is not bytes
        or type(override_output) is not bytes
        or type(structurally_eligible) is not bool
    ):
        raise ScarRepairContractError("SCAR_REPAIR_SELECTION_INVALID")
    score = _finite_fraction(
        selector_score, issue_id="SCAR_REPAIR_SELECTION_INVALID"
    )
    if isinstance(threshold, float) and math.isinf(threshold):
        if threshold < 0:
            raise ScarRepairContractError("SCAR_REPAIR_SELECTION_INVALID")
        passes = False
    else:
        parsed_threshold = _finite_fraction(
            threshold, issue_id="SCAR_REPAIR_SELECTION_INVALID"
        )
        if parsed_threshold < 0:
            raise ScarRepairContractError("SCAR_REPAIR_SELECTION_INVALID")
        passes = score > parsed_threshold
    if not baseline_output or not override_output:
        raise ScarRepairContractError("SCAR_REPAIR_SELECTION_INVALID")
    return (
        override_output
        if structurally_eligible and passes and override_output != baseline_output
        else baseline_output
    )


def _pair_set(value: object) -> frozenset[tuple[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScarRepairContractError("SCAR_REPAIR_PAIR_SET_INVALID")
    rows: list[tuple[str, str]] = []
    for row in value:
        if (
            isinstance(row, (str, bytes))
            or not isinstance(row, Sequence)
            or len(row) != 2
            or any(type(cell) is not str or not cell for cell in row)
        ):
            raise ScarRepairContractError("SCAR_REPAIR_PAIR_SET_INVALID")
        rows.append((row[0], row[1]))
    if len(rows) != len(set(rows)):
        raise ScarRepairContractError("SCAR_REPAIR_PAIR_SET_INVALID")
    return frozenset(rows)


def pair_f1(
    predicted_pairs: Sequence[Sequence[str]],
    gold_pairs: Sequence[Sequence[str]],
) -> Fraction:
    predicted = _pair_set(predicted_pairs)
    gold = _pair_set(gold_pairs)
    correct = len(predicted & gold)
    denominator = 2 * correct + len(predicted - gold) + len(gold - predicted)
    return Fraction(2 * correct, denominator) if denominator else Fraction(0)


@dataclass(frozen=True, slots=True)
class OldSuccessPreservation:
    old_success_count: int
    preserved_count: int
    fraction: Fraction


def _pair_collections(value: object) -> tuple[frozenset[tuple[str, str]], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScarRepairContractError("SCAR_REPAIR_PAIR_SET_INVALID")
    rows = tuple(value)
    if not rows:
        return (_pair_set(()),)
    first = rows[0]
    is_single_pair_collection = (
        isinstance(first, Sequence)
        and not isinstance(first, (str, bytes))
        and len(first) == 2
        and all(type(cell) is str for cell in first)
    )
    if is_single_pair_collection:
        return (_pair_set(rows),)
    return tuple(_pair_set(row) for row in rows)


def old_success_preservation(
    baseline_pairs: Sequence[Any],
    override_pairs: Sequence[Any],
    gold_pairs: Sequence[Any],
) -> OldSuccessPreservation:
    """Pair-level recall of gold pairs already found by S0, across items."""

    baseline_rows = _pair_collections(baseline_pairs)
    override_rows = _pair_collections(override_pairs)
    gold_rows = _pair_collections(gold_pairs)
    if not (
        len(baseline_rows) == len(override_rows) == len(gold_rows)
    ):
        raise ScarRepairContractError("SCAR_REPAIR_PAIR_SET_INVALID")
    old_success_count = 0
    preserved_count = 0
    for baseline, override, gold in zip(
        baseline_rows, override_rows, gold_rows, strict=True
    ):
        old_success = baseline & gold
        old_success_count += len(old_success)
        preserved_count += len(old_success & override)
    return OldSuccessPreservation(
        old_success_count=old_success_count,
        preserved_count=preserved_count,
        fraction=(
            Fraction(preserved_count, old_success_count)
            if old_success_count
            else Fraction(1)
        ),
    )


def paired_bootstrap_delta(
    deltas: Sequence[float],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, object]:
    """Item-cluster bootstrap of fixed OOF deltas; no model is refit."""

    rows = _finite_rows(deltas, issue_id="SCAR_REPAIR_BOOTSTRAP_INVALID")
    if (
        not rows
        or type(seed) is not int
        or isinstance(seed, bool)
        or type(replicates) is not int
        or isinstance(replicates, bool)
        or replicates < 1
        or type(alpha) not in {int, float}
        or isinstance(alpha, bool)
        or not 0.0 < float(alpha) < 0.5
    ):
        raise ScarRepairContractError("SCAR_REPAIR_BOOTSTRAP_INVALID")
    count = len(rows)
    rng = random.Random(seed)
    samples = [
        sum(rows[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    ]
    samples.sort()
    lower_index = math.floor(float(alpha) * (replicates - 1))
    return {
        "alpha": float(alpha),
        "mean_delta": sum(rows) / count,
        "one_sided_lower_bound": samples[lower_index],
        "lower_quantile_zero_based_index": lower_index,
        "paired_item_count": count,
        "replicates": replicates,
        "resampling": "python_random_MT19937_item_cluster_with_replacement_no_refit_v1",
        "seed": seed,
    }


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    observed_mean_delta: Fraction
    bootstrap_mean_deltas: tuple[Fraction, ...]
    one_sided_lower_bound: Fraction
    lower_quantile_zero_based_index: int
    replicate_count: int
    seed: int


def paired_bootstrap_mean_delta(
    successor_values: Sequence[int | float | Fraction],
    baseline_values: Sequence[int | float | Fraction],
    *,
    seed: int,
    replicate_count: int = BOOTSTRAP_REPLICATES,
    alpha: int | float | Fraction = Fraction(1, 20),
) -> PairedBootstrapResult:
    """MT19937 paired item bootstrap over fixed, caller-supplied outcomes."""

    issue_id = "SCAR_REPAIR_BOOTSTRAP_INVALID"
    if (
        isinstance(successor_values, (str, bytes))
        or not isinstance(successor_values, Sequence)
        or isinstance(baseline_values, (str, bytes))
        or not isinstance(baseline_values, Sequence)
        or not successor_values
        or len(successor_values) != len(baseline_values)
        or type(seed) is not int
        or type(replicate_count) is not int
        or replicate_count < 1
    ):
        raise ScarRepairContractError(issue_id)
    alpha_fraction = _finite_fraction(alpha, issue_id=issue_id)
    if not 0 < alpha_fraction < Fraction(1, 2):
        raise ScarRepairContractError(issue_id)
    successor = tuple(
        _finite_fraction(row, issue_id=issue_id) for row in successor_values
    )
    baseline = tuple(
        _finite_fraction(row, issue_id=issue_id) for row in baseline_values
    )
    deltas = tuple(
        successor_row - baseline_row
        for successor_row, baseline_row in zip(successor, baseline, strict=True)
    )
    item_count = len(deltas)
    rng = random.Random(seed)
    bootstrap = tuple(
        sum(
            (deltas[rng.randrange(item_count)] for _ in range(item_count)),
            Fraction(0),
        )
        / item_count
        for _ in range(replicate_count)
    )
    lower_index = math.floor(float(alpha_fraction) * (replicate_count - 1))
    ordered = sorted(bootstrap)
    return PairedBootstrapResult(
        observed_mean_delta=sum(deltas, Fraction(0)) / item_count,
        bootstrap_mean_deltas=bootstrap,
        one_sided_lower_bound=ordered[lower_index],
        lower_quantile_zero_based_index=lower_index,
        replicate_count=replicate_count,
        seed=seed,
    )


class DevelopmentVerdict(str, Enum):
    IMPLEMENTATION_INVALID = "REPAIR_DEVELOPMENT_IMPLEMENTATION_INVALID"
    UNSAFE_OLD_SUCCESS_REGRESSION = (
        "REPAIR_DEVELOPMENT_UNSAFE_OLD_SUCCESS_REGRESSION"
    )
    NO_PRACTICALLY_IMPORTANT_GAIN = (
        "REPAIR_DEVELOPMENT_NO_PRACTICALLY_IMPORTANT_GAIN"
    )
    QUALIFIED = "POSTHOC_REPAIR_DEVELOPMENT_QUALIFIED"


def decide_development_verdict(
    implementation_valid: bool,
    preservation: float,
    bootstrap_lower_bound: float,
    *,
    error_count: int = 0,
    preservation_floor: float = OLD_SUCCESS_PRESERVATION_FLOOR,
    mid: float = UTILITY_MID,
) -> DevelopmentVerdict:
    """Apply the frozen priority: invalid, unsafe, no gain, qualified."""

    numeric = (preservation, bootstrap_lower_bound, preservation_floor, mid)
    if (
        type(implementation_valid) is not bool
        or type(error_count) is not int
        or isinstance(error_count, bool)
        or error_count < 0
        or any(type(row) not in {int, float} or isinstance(row, bool) for row in numeric)
        or not all(math.isfinite(float(row)) for row in numeric)
        or not 0.0 <= float(preservation) <= 1.0
        or not 0.0 <= float(preservation_floor) <= 1.0
        or float(mid) < 0.0
    ):
        raise ScarRepairContractError("SCAR_REPAIR_VERDICT_INPUT_INVALID")
    if not implementation_valid or error_count:
        return DevelopmentVerdict.IMPLEMENTATION_INVALID
    if float(preservation) < float(preservation_floor):
        return DevelopmentVerdict.UNSAFE_OLD_SUCCESS_REGRESSION
    if float(bootstrap_lower_bound) <= float(mid):
        return DevelopmentVerdict.NO_PRACTICALLY_IMPORTANT_GAIN
    return DevelopmentVerdict.QUALIFIED


def decide_repair_development_verdict(
    *,
    implementation_valid: bool,
    old_success_preservation: int | float | Fraction,
    minimum_old_success_preservation: int | float | Fraction,
    primary_ci_lower_bound: int | float | Fraction,
    minimum_practically_important_gain: int | float | Fraction,
) -> str:
    """Apply the frozen verdict priority using exact comparison values."""

    issue_id = "SCAR_REPAIR_VERDICT_INPUT_INVALID"
    if type(implementation_valid) is not bool:
        raise ScarRepairContractError(issue_id)
    preservation = _finite_fraction(
        old_success_preservation, issue_id=issue_id
    )
    preservation_floor = _finite_fraction(
        minimum_old_success_preservation, issue_id=issue_id
    )
    lower_bound = _finite_fraction(primary_ci_lower_bound, issue_id=issue_id)
    important_gain = _finite_fraction(
        minimum_practically_important_gain, issue_id=issue_id
    )
    if (
        not 0 <= preservation <= 1
        or not 0 <= preservation_floor <= 1
        or important_gain < 0
    ):
        raise ScarRepairContractError(issue_id)
    if not implementation_valid:
        return DevelopmentVerdict.IMPLEMENTATION_INVALID.value
    if preservation < preservation_floor:
        return DevelopmentVerdict.UNSAFE_OLD_SUCCESS_REGRESSION.value
    if lower_bound <= important_gain:
        return DevelopmentVerdict.NO_PRACTICALLY_IMPORTANT_GAIN.value
    return DevelopmentVerdict.QUALIFIED.value


__all__ = [
    "BOOTSTRAP_ALPHA",
    "BOOTSTRAP_REPLICATES",
    "COSINE_QUANTIZATION_SCALE",
    "DevelopmentVerdict",
    "EvidenceValue",
    "FEATURE_ORDER",
    "FEATURE_WIDTH",
    "FOLD_COUNT",
    "INNER_THRESHOLD_PRESERVATION_FLOOR",
    "OLD_SUCCESS_PRESERVATION_FLOOR",
    "OldSuccessPreservation",
    "PairedBootstrapResult",
    "PRESERVATION_FLOOR",
    "RIDGE_ALPHA",
    "ScarRepairContractError",
    "StandardizedRidgeModel",
    "ThresholdExample",
    "ThresholdSelection",
    "THRESHOLD_GRID",
    "UTILITY_MID",
    "VERSION",
    "assign_grouped_folds",
    "canonical_bytes",
    "canonical_json_bytes",
    "combine_three_valued_evidence",
    "content_hash",
    "conjoin_evidence",
    "decide_repair_development_verdict",
    "decide_development_verdict",
    "domain_hash",
    "exhaustive_or",
    "exhaustive_or_evidence",
    "extract_archived_features",
    "extract_fixed_features",
    "fit_standardized_ridge",
    "old_success_preservation",
    "pair_f1",
    "paired_bootstrap_delta",
    "paired_bootstrap_mean_delta",
    "parse_canonical_json_bytes",
    "predict_standardized_ridge",
    "seal_payload",
    "select_action",
    "select_action_output",
    "select_override_threshold",
    "strong_kleene_and",
    "validate_self_seal",
]
