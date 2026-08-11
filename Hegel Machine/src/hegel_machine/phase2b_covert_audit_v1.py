"""Frozen, non-authoritative Phase-2B envelope covert-audit mechanics.

This module accepts only fixed-size public envelopes and auditor-side metadata.
It deliberately cannot issue formal audit evidence: generator answers, caller
thresholds, caller seeds, and caller-provided content roots are absent from the
authoritative API.

The frozen unique-ID feature family is applied to the envelope prefix and
suffix 128-bit blocks.  This is not a parser or field-level audit of every UUID
namespace in a ``PublicEvidenceBundle``.  A trusted RFC-8785 wire parser and
namespace-aware ID extractor remain prerequisites for the complete covert-
channel audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from inspect import signature
import math
from typing import Final

from .hashing import stable_hash
from .phase2b_freeze_v1 import frozen_phase2b_exact_freeze


SCHEMA_VERSION: Final = "hegel-machine-phase2b-covert-audit-mechanics/1"
SEMANTICS_VERSION: Final = "hegel-machine-phase2b-covert-audit-statistics/1"
NON_AUTHORITATIVE_CLAIM_LEVEL: Final = "NON_AUTHORITATIVE_MECHANICS_ONLY"
PERMUTATION_DOMAIN: Final = b"hegel-phase2b-covert-audit-stratified-permutation-v1"
TARGETS: Final = (
    "family",
    "binding",
    "scale",
    "answerable_vs_abstain",
    "joint_decision_class",
)
FROZEN_UNIQUE_ID_FEATURE_FAMILY: Final = (
    "128_individual_bits",
    "first_8_16_32_bits",
    "last_8_16_32_bits",
    "hamming_weight",
    "integer_mod_3_5_7_11_13",
    "hex_character_histogram",
)
BYTE_FEATURE_GROUP_SEMANTICS: Final = (
    "prefix_128=exact_envelope_bytes_0_through_15",
    "suffix_128=exact_envelope_bytes_65520_through_65535",
    "individual_bit_order=most_significant_bit_first",
    "multi_byte_integer_order=big_endian",
    "hex_histogram=counts_of_16_nibble_values_over_each_128_bit_group",
    "whole_envelope_aggregates=xor_hamming_weight_byte_sum_moduli_hamming_moduli",
)
SEMANTICS_DESCRIPTOR: Final = (
    "features=recomputed_internally_from_exact_65536_byte_envelope",
    "feature_groups=" + "|".join(BYTE_FEATURE_GROUP_SEMANTICS),
    "unique_id_feature_family=" + "|".join(FROZEN_UNIQUE_ID_FEATURE_FAMILY),
    "nmi=mutual_information_divided_by_sqrt_feature_entropy_times_label_entropy_base2",
    "constant_feature_nmi=0",
    "cv=leave_one_out_single_categorical_feature_mode",
    "cv_unseen_value_fallback=global_training_mode",
    "cv_tie_break=utf8_lexicographically_smallest_label",
    "balanced_accuracy=unweighted_mean_per_label_recall",
    "balanced_accuracy_advantage=balanced_accuracy_minus_one_over_label_count",
    "permutation=sha256_stream_fisher_yates_within_frozen_target_strata",
    "permutation_p=(one_plus_ge_observed_count)/(one_plus_10000)",
    "permutation_stat=max(nmi/0.02,max(0,ba_advantage)/0.05)",
    "multiple_testing=single_holm_bonferroni_family_over_all_target_feature_pairs",
    "holm_gate=adjusted_p_greater_than_or_equal_to_0.01",
    "insufficient_class_strata_or_cv=ABSTAIN",
    "statistics_resource_cap=64_rows_and_1036800000_declared_work_units",
    "floating_runtime=python_binary64_math_log2_mechanics_not_cross_runtime_formal",
)
SEMANTICS_ID: Final = stable_hash(
    {"version": SEMANTICS_VERSION, "descriptor": SEMANTICS_DESCRIPTOR},
    prefix="phase2b_covert_audit_semantics_",
)


def _require_exact_str(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be an exact non-empty string")
    if len(value.encode("utf-8")) > 256:
        raise ValueError(f"{name} exceeds the frozen UTF-8 budget")
    return value


def _require_exact_str_tuple(value: object, name: str, length: int | None = None) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} has the wrong frozen width")
    for item in value:
        _require_exact_str(item, name)
    return value


def _require_content_id(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise TypeError(f"{name} must use the frozen content-id prefix")
    suffix = value[len(prefix) :]
    if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
        raise ValueError(f"{name} must end in an exact lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class CovertAuditPolicy:
    schema_version: str = SCHEMA_VERSION
    semantics_version: str = SEMANTICS_VERSION
    semantics_id: str = SEMANTICS_ID
    unique_id_feature_family: tuple[str, ...] = FROZEN_UNIQUE_ID_FEATURE_FAMILY
    byte_feature_group_semantics: tuple[str, ...] = BYTE_FEATURE_GROUP_SEMANTICS
    envelope_bytes: int = 65_536
    label_permutations: int = 10_000
    family_wise_alpha: float = 0.01
    maximum_nmi: float = 0.02
    maximum_balanced_accuracy_advantage: float = 0.05
    global_renamings: int = 32
    global_case_order_permutations: int = 32
    observation_order_permutations: int = 16
    maximum_rows: int = 1_024
    maximum_statistical_rows: int = 64
    frozen_feature_count_per_row: int = 324
    maximum_statistical_work_units: int = 1_036_800_000
    maximum_total_envelope_bytes: int = 67_108_864
    maximum_residual_vector_bytes_per_row: int = 1_048_576
    maximum_total_residual_vector_bytes: int = 67_108_864
    maximum_total_invariance_receipt_rows: int = 82_944
    maximum_total_invariance_residual_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        frozen = frozen_phase2b_exact_freeze().covert_channel_audit
        if type(self) is not CovertAuditPolicy:
            raise TypeError("covert audit policy must use the exact frozen type")
        if (
            self.schema_version != SCHEMA_VERSION
            or self.semantics_version != SEMANTICS_VERSION
            or self.semantics_id != SEMANTICS_ID
            or self.unique_id_feature_family != frozen.unique_id_feature_family
            or self.byte_feature_group_semantics != BYTE_FEATURE_GROUP_SEMANTICS
            or self.envelope_bytes != frozen.envelope_bytes
            or self.label_permutations != frozen.label_permutations
            or self.family_wise_alpha != frozen.family_wise_alpha
            or self.maximum_nmi != frozen.maximum_normalized_mutual_information
            or self.maximum_balanced_accuracy_advantage
            != frozen.maximum_balanced_accuracy_advantage
            or self.global_renamings != frozen.global_consistent_renamings
            or self.global_case_order_permutations
            != frozen.global_case_order_permutations
            or self.observation_order_permutations
            != frozen.within_case_observation_order_permutations
        ):
            raise ValueError("covert audit policy drift")
        if (
            type(self.maximum_rows) is not int
            or not 1 <= self.maximum_rows <= 1_024
            or self.maximum_total_envelope_bytes
            != self.maximum_rows * self.envelope_bytes
            or type(self.maximum_residual_vector_bytes_per_row) is not int
            or self.maximum_residual_vector_bytes_per_row <= 0
            or self.maximum_total_residual_vector_bytes
            != self.maximum_rows * 65_536
            or self.maximum_statistical_rows != 64
            or self.frozen_feature_count_per_row != 324
            or self.maximum_statistical_work_units
            != (
                self.maximum_statistical_rows
                * self.frozen_feature_count_per_row
                * len(TARGETS)
                * self.label_permutations
            )
            or self.maximum_total_invariance_receipt_rows
            != (
                1
                + self.global_renamings
                + self.global_case_order_permutations
                + self.observation_order_permutations
            )
            * self.maximum_rows
            or self.maximum_total_invariance_residual_bytes != 536_870_912
        ):
            raise ValueError("covert audit resource budget drift")

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2b_covert_audit_policy_")


DEFAULT_COVERT_AUDIT_POLICY: Final = CovertAuditPolicy()


class AnswerabilityLabel(str, Enum):
    ANSWERABLE = "answerable"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class AuditorLabels:
    family: str
    binding: str
    scale: str
    answerable: AnswerabilityLabel
    joint_class: str
    case_type: str

    def __post_init__(self) -> None:
        if type(self) is not AuditorLabels:
            raise TypeError("auditor labels must use the exact frozen type")
        for name in ("family", "binding", "scale", "joint_class", "case_type"):
            _require_exact_str(getattr(self, name), f"auditor label {name}")
        if type(self.answerable) is not AnswerabilityLabel:
            raise TypeError("answerability must use the exact frozen binary enum")


@dataclass(frozen=True, slots=True)
class AuditorPermutationStrata:
    family: tuple[str, str]
    scale: tuple[str, str]
    binding: tuple[str, str, str]
    answerable: tuple[str, str]
    joint_class: tuple[str]

    def __post_init__(self) -> None:
        if type(self) is not AuditorPermutationStrata:
            raise TypeError("auditor strata must use the exact frozen type")
        _require_exact_str_tuple(self.family, "family strata", 2)
        _require_exact_str_tuple(self.scale, "scale strata", 2)
        _require_exact_str_tuple(self.binding, "binding strata", 3)
        _require_exact_str_tuple(self.answerable, "answerable strata", 2)
        _require_exact_str_tuple(self.joint_class, "joint-class strata", 1)


@dataclass(frozen=True, slots=True)
class EnvelopeAuditRow:
    auditor_row_id: int
    envelope: bytes
    labels: AuditorLabels
    strata: AuditorPermutationStrata
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not EnvelopeAuditRow:
            raise TypeError("audit row must use the exact frozen type")
        if (
            type(self.auditor_row_id) is not int
            or not 0 <= self.auditor_row_id < DEFAULT_COVERT_AUDIT_POLICY.maximum_rows
        ):
            raise TypeError("auditor row id must be an exact non-negative integer")
        if type(self.envelope) is not bytes:
            raise TypeError("audit envelope must use exact bytes")
        if len(self.envelope) != DEFAULT_COVERT_AUDIT_POLICY.envelope_bytes:
            raise ValueError("audit envelope must contain exactly 65,536 bytes")
        if type(self.labels) is not AuditorLabels or type(self.strata) is not AuditorPermutationStrata:
            raise TypeError("audit row contains non-exact labels or strata")
        case_type = self.labels.case_type
        expected = (
            (case_type, self.labels.scale),
            (case_type, self.labels.family),
            (case_type, self.labels.family, self.labels.scale),
            (self.labels.family, self.labels.scale),
            (case_type,),
        )
        if (
            self.strata.family,
            self.strata.scale,
            self.strata.binding,
            self.strata.answerable,
            self.strata.joint_class,
        ) != expected:
            raise ValueError("auditor strata do not match the frozen target table")
        object.__setattr__(
            self,
            "content_id",
            self.recompute_content_id(),
        )

    def recompute_content_id(self) -> str:
        return stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "auditor_row_id": self.auditor_row_id,
                "envelope_sha256": hashlib.sha256(self.envelope).hexdigest(),
                "envelope_length": len(self.envelope),
                "labels": self.labels,
                "strata": self.strata,
            },
            prefix="phase2b_covert_audit_input_row_",
        )


@dataclass(frozen=True, slots=True)
class ByteFeatureVector:
    auditor_row_id: int
    values: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if type(self) is not ByteFeatureVector or type(self.values) is not tuple:
            raise TypeError("byte feature vector must use exact frozen types")
        if type(self.auditor_row_id) is not int or not 0 <= self.auditor_row_id < DEFAULT_COVERT_AUDIT_POLICY.maximum_rows:
            raise TypeError("byte feature row id must be exact and bounded")
        if len(self.values) != DEFAULT_COVERT_AUDIT_POLICY.frozen_feature_count_per_row:
            raise ValueError("byte feature count drift")
        if any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not int
            for item in self.values
        ):
            raise TypeError("byte features must be exact name/integer pairs")
        names = tuple(name for name, _ in self.values)
        if len(names) != len(set(names)):
            raise ValueError("byte feature identifiers must be unique")


def _append_128_bit_feature_family(
    values: list[tuple[str, int]],
    group_name: str,
    block: bytes,
) -> None:
    if type(block) is not bytes or len(block) != 16:
        raise ValueError("frozen 128-bit feature group must contain 16 bytes")
    for bit_index in range(128):
        byte_value = block[bit_index // 8]
        bit_value = (byte_value >> (7 - (bit_index % 8))) & 1
        values.append((f"{group_name}_individual_bit_{bit_index:03d}", bit_value))
    for width in (8, 16, 32):
        byte_width = width // 8
        values.append(
            (
                f"{group_name}_first_{width}_bits",
                int.from_bytes(block[:byte_width], "big"),
            )
        )
        values.append(
            (
                f"{group_name}_last_{width}_bits",
                int.from_bytes(block[-byte_width:], "big"),
            )
        )
    hamming_weight = sum(value.bit_count() for value in block)
    values.append((f"{group_name}_hamming_weight", hamming_weight))
    integer_value = int.from_bytes(block, "big")
    for modulus in (3, 5, 7, 11, 13):
        values.append((f"{group_name}_integer_mod_{modulus}", integer_value % modulus))
    histogram = [0] * 16
    for byte_value in block:
        histogram[byte_value >> 4] += 1
        histogram[byte_value & 0x0F] += 1
    for nibble, count in enumerate(histogram):
        values.append((f"{group_name}_hex_histogram_{nibble:x}", count))


def extract_frozen_byte_features(row: EnvelopeAuditRow) -> ByteFeatureVector:
    """Recompute the frozen prefix/suffix and whole-envelope feature family."""

    if type(row) is not EnvelopeAuditRow:
        raise TypeError("byte feature extraction requires an exact audit row")
    data = row.envelope
    byte_sum = sum(data)
    xor_value = 0
    hamming_weight = 0
    for value in data:
        xor_value ^= value
        hamming_weight += value.bit_count()
    values: list[tuple[str, int]] = []
    _append_128_bit_feature_family(values, "prefix_128", data[:16])
    _append_128_bit_feature_family(values, "suffix_128", data[-16:])
    values.extend(
        (
            ("envelope_xor_u8", xor_value),
            ("envelope_hamming_weight", hamming_weight),
        )
    )
    for modulus in (3, 5, 7, 11, 13):
        values.append((f"envelope_byte_sum_mod_{modulus}", byte_sum % modulus))
        values.append((f"envelope_hamming_weight_mod_{modulus}", hamming_weight % modulus))
    return ByteFeatureVector(row.auditor_row_id, tuple(values))


def _frozen_byte_feature_ids() -> tuple[str, ...]:
    values: list[tuple[str, int]] = []
    _append_128_bit_feature_family(values, "prefix_128", bytes(16))
    _append_128_bit_feature_family(values, "suffix_128", bytes(16))
    values.extend(
        (
            ("envelope_xor_u8", 0),
            ("envelope_hamming_weight", 0),
        )
    )
    for modulus in (3, 5, 7, 11, 13):
        values.append((f"envelope_byte_sum_mod_{modulus}", 0))
        values.append((f"envelope_hamming_weight_mod_{modulus}", 0))
    feature_ids = tuple(name for name, _ in values)
    if (
        len(feature_ids)
        != DEFAULT_COVERT_AUDIT_POLICY.frozen_feature_count_per_row
        or len(feature_ids) != len(set(feature_ids))
    ):
        raise RuntimeError("frozen byte feature identity drift")
    return feature_ids


FROZEN_BYTE_FEATURE_IDS: Final = _frozen_byte_feature_ids()


class InvarianceKind(str, Enum):
    BASELINE = "baseline"
    GLOBAL_RENAMING = "global_renaming"
    CASE_ORDER = "case_order"
    OBSERVATION_ORDER = "observation_order"


@dataclass(frozen=True, slots=True)
class InvariantDecisionRow:
    auditor_row_id: int
    decision: str
    family: str
    binding: tuple[tuple[str, str], ...]
    scale_set: tuple[str, ...]
    candidate_residual_vector: bytes

    def __post_init__(self) -> None:
        if type(self) is not InvariantDecisionRow:
            raise TypeError("invariant decision row must use the exact frozen type")
        if (
            type(self.auditor_row_id) is not int
            or not 0 <= self.auditor_row_id < DEFAULT_COVERT_AUDIT_POLICY.maximum_rows
        ):
            raise TypeError("invariant auditor row id must be exact")
        _require_exact_str(self.decision, "decision")
        _require_exact_str(self.family, "family")
        if type(self.binding) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(item) is not str or not item for item in pair)
            for pair in self.binding
        ):
            raise TypeError("binding must be an exact tuple of string pairs")
        for role, entity in self.binding:
            _require_exact_str(role, "binding role")
            _require_exact_str(entity, "binding entity")
        _require_exact_str_tuple(self.scale_set, "scale set")
        if len(self.binding) > 512 or len(self.scale_set) > 64:
            raise ValueError("invariant decision row exceeds the frozen width budget")
        if type(self.candidate_residual_vector) is not bytes:
            raise TypeError("candidate residual vector must use exact bytes")
        if len(self.candidate_residual_vector) > DEFAULT_COVERT_AUDIT_POLICY.maximum_residual_vector_bytes_per_row:
            raise ValueError("candidate residual vector exceeds the frozen budget")

    def as_hash_payload(self) -> dict[str, object]:
        return {
            "auditor_row_id": self.auditor_row_id,
            "decision": self.decision,
            "family": self.family,
            "binding": self.binding,
            "scale_set": self.scale_set,
            "candidate_residual_vector_length": len(self.candidate_residual_vector),
            "candidate_residual_vector_sha256": hashlib.sha256(
                self.candidate_residual_vector
            ).hexdigest(),
        }


@dataclass(frozen=True, slots=True)
class RecognizerInvariantReceipt:
    kind: InvarianceKind
    trial_index: int
    rows: tuple[InvariantDecisionRow, ...]
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not RecognizerInvariantReceipt:
            raise TypeError("recognizer receipt must use the exact frozen type")
        if type(self.kind) is not InvarianceKind:
            raise TypeError("recognizer receipt kind must use InvarianceKind")
        trial_limits = {
            InvarianceKind.BASELINE: 1,
            InvarianceKind.GLOBAL_RENAMING: DEFAULT_COVERT_AUDIT_POLICY.global_renamings,
            InvarianceKind.CASE_ORDER: DEFAULT_COVERT_AUDIT_POLICY.global_case_order_permutations,
            InvarianceKind.OBSERVATION_ORDER: DEFAULT_COVERT_AUDIT_POLICY.observation_order_permutations,
        }
        if (
            type(self.trial_index) is not int
            or not 0 <= self.trial_index < trial_limits[self.kind]
        ):
            raise TypeError("receipt trial index must be an exact non-negative integer")
        if type(self.rows) is not tuple:
            raise TypeError("recognizer receipt rows must be an exact tuple")
        if len(self.rows) > DEFAULT_COVERT_AUDIT_POLICY.maximum_rows:
            raise ValueError("recognizer receipt row count exceeds the frozen budget")
        if any(type(row) is not InvariantDecisionRow for row in self.rows):
            raise TypeError("recognizer receipt rows must contain exact decision rows")
        if (
            sum(len(row.candidate_residual_vector) for row in self.rows)
            > DEFAULT_COVERT_AUDIT_POLICY.maximum_total_residual_vector_bytes
        ):
            raise ValueError("recognizer receipt residual vectors exceed the frozen budget")
        object.__setattr__(self, "content_id", self.recompute_content_id())

    def recompute_content_id(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind.value,
            "trial_index": self.trial_index,
            "rows": tuple(row.as_hash_payload() for row in self.rows),
        }
        return stable_hash(payload, prefix="phase2b_invariance_receipt_")


@dataclass(frozen=True, slots=True)
class InvarianceAuditBatch:
    baseline: RecognizerInvariantReceipt
    global_renamings: tuple[RecognizerInvariantReceipt, ...]
    case_order_permutations: tuple[RecognizerInvariantReceipt, ...]
    observation_order_permutations: tuple[RecognizerInvariantReceipt, ...]


@dataclass(frozen=True, slots=True)
class CovertAuditStructuralReceipt:
    schema_version: str
    semantics_version: str
    semantics_id: str
    policy_id: str
    audit_input_content_id: str
    auditor_row_content_ids: tuple[str, ...]
    row_count: int
    feature_count_per_row: int
    invariant_receipt_content_ids: tuple[str, ...]
    exact_invariant_comparisons: int
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    formal_audit_evidence: bool = False
    sealed_holdout_eligible: bool = False
    statistics_executed: bool = False
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not CovertAuditStructuralReceipt:
            raise TypeError("structural receipt must use the exact frozen type")
        if (
            self.schema_version != SCHEMA_VERSION
            or self.semantics_version != SEMANTICS_VERSION
            or self.semantics_id != SEMANTICS_ID
            or self.policy_id != DEFAULT_COVERT_AUDIT_POLICY.policy_id
        ):
            raise ValueError("structural receipt identity or row coverage drift")
        _require_content_id(
            self.policy_id,
            "phase2b_covert_audit_policy_",
            "structural policy id",
        )
        _require_content_id(
            self.audit_input_content_id,
            "phase2b_covert_audit_input_",
            "structural input id",
        )
        if type(self.row_count) is not int or not 1 <= self.row_count <= DEFAULT_COVERT_AUDIT_POLICY.maximum_rows:
            raise TypeError("structural row count must be exact and bounded")
        if self.feature_count_per_row != len(FROZEN_BYTE_FEATURE_IDS):
            raise ValueError("structural feature count drift")
        if type(self.auditor_row_content_ids) is not tuple:
            raise TypeError("structural row content ids must use an exact tuple")
        if len(self.auditor_row_content_ids) != self.row_count:
            raise ValueError("structural row content-id coverage drift")
        if type(self.invariant_receipt_content_ids) is not tuple:
            raise TypeError("structural invariance content ids must use an exact tuple")
        expected_invariance_receipts = (
            1
            + DEFAULT_COVERT_AUDIT_POLICY.global_renamings
            + DEFAULT_COVERT_AUDIT_POLICY.global_case_order_permutations
            + DEFAULT_COVERT_AUDIT_POLICY.observation_order_permutations
        )
        if len(self.invariant_receipt_content_ids) != expected_invariance_receipts:
            raise ValueError("structural invariance content-id coverage drift")
        for content_id in self.auditor_row_content_ids:
            _require_content_id(
                content_id,
                "phase2b_covert_audit_input_row_",
                "structural row content id",
            )
        for content_id in self.invariant_receipt_content_ids:
            _require_content_id(
                content_id,
                "phase2b_invariance_receipt_",
                "structural invariance content id",
            )
        expected_input_content_id = stable_hash(
            {"row_content_ids": self.auditor_row_content_ids},
            prefix="phase2b_covert_audit_input_",
        )
        if self.audit_input_content_id != expected_input_content_id:
            raise ValueError("structural input content id is not self-consistent")
        expected_comparisons = (
            self.row_count
            * (
                DEFAULT_COVERT_AUDIT_POLICY.global_renamings
                + DEFAULT_COVERT_AUDIT_POLICY.global_case_order_permutations
                + DEFAULT_COVERT_AUDIT_POLICY.observation_order_permutations
            )
            * 5
        )
        if (
            type(self.exact_invariant_comparisons) is not int
            or self.exact_invariant_comparisons != expected_comparisons
        ):
            raise ValueError("structural invariant comparison count drift")
        if any(
            type(value) is not bool
            for value in (
                self.formal_audit_evidence,
                self.sealed_holdout_eligible,
                self.statistics_executed,
            )
        ):
            raise TypeError("structural evidence flags must be exact booleans")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL or any(
            (self.formal_audit_evidence, self.sealed_holdout_eligible, self.statistics_executed)
        ):
            raise ValueError("structural mechanics cannot issue audit evidence")
        object.__setattr__(
            self,
            "content_id",
            stable_hash(self, prefix="phase2b_covert_audit_structural_receipt_"),
        )


class CovertAuditDisposition(str, Enum):
    STATISTICS_COMPLETE = "STATISTICS_COMPLETE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True)
class TargetSufficiencyReceipt:
    target: str
    label_count: int
    minimum_label_count: int
    stratum_count: int
    minimum_stratum_size: int
    mutable_stratum_count: int
    sufficient: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self) is not TargetSufficiencyReceipt:
            raise TypeError("sufficiency receipt must use the exact type")
        _require_exact_str(self.target, "sufficiency target")
        _require_exact_str(self.reason, "sufficiency reason")
        for name in (
            "label_count",
            "minimum_label_count",
            "stratum_count",
            "minimum_stratum_size",
            "mutable_stratum_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact non-negative integer")
        if type(self.sufficient) is not bool:
            raise TypeError("sufficiency disposition must be an exact boolean")


@dataclass(frozen=True, slots=True)
class TargetFeatureAuditResult:
    target: str
    feature_id: str
    normalized_mutual_information: float
    balanced_accuracy_advantage: float
    permutation_exceedance_count: int
    raw_permutation_p: float
    holm_adjusted_p: float
    nmi_within_limit: bool
    balanced_accuracy_advantage_within_limit: bool
    adjusted_p_within_limit: bool

    def __post_init__(self) -> None:
        if type(self) is not TargetFeatureAuditResult:
            raise TypeError("target-feature result must use the exact type")
        _require_exact_str(self.target, "target-feature target")
        _require_exact_str(self.feature_id, "target-feature id")
        if self.target not in TARGETS or self.feature_id not in FROZEN_BYTE_FEATURE_IDS:
            raise ValueError("target-feature identity is outside the frozen coverage")
        for name in (
            "normalized_mutual_information",
            "balanced_accuracy_advantage",
            "raw_permutation_p",
            "holm_adjusted_p",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be an exact finite float")
        if (
            type(self.permutation_exceedance_count) is not int
            or not 0
            <= self.permutation_exceedance_count
            <= DEFAULT_COVERT_AUDIT_POLICY.label_permutations
        ):
            raise TypeError("permutation exceedance count must be exact")
        if not 0.0 <= self.normalized_mutual_information <= 1.0:
            raise ValueError("NMI is outside [0,1]")
        if not 0.0 <= self.raw_permutation_p <= 1.0 or not 0.0 <= self.holm_adjusted_p <= 1.0:
            raise ValueError("permutation probability is outside [0,1]")
        if any(
            type(getattr(self, name)) is not bool
            for name in (
                "nmi_within_limit",
                "balanced_accuracy_advantage_within_limit",
                "adjusted_p_within_limit",
            )
        ):
            raise TypeError("target-feature gates must be exact booleans")
        expected_raw_p = (
            self.permutation_exceedance_count + 1
        ) / (DEFAULT_COVERT_AUDIT_POLICY.label_permutations + 1)
        if self.raw_permutation_p != expected_raw_p:
            raise ValueError("raw permutation p-value does not match its exact count")
        if (
            self.nmi_within_limit
            != (
                self.normalized_mutual_information
                <= DEFAULT_COVERT_AUDIT_POLICY.maximum_nmi
            )
            or self.balanced_accuracy_advantage_within_limit
            != (
                self.balanced_accuracy_advantage
                <= DEFAULT_COVERT_AUDIT_POLICY.maximum_balanced_accuracy_advantage
            )
            or self.adjusted_p_within_limit
            != (
                self.holm_adjusted_p
                >= DEFAULT_COVERT_AUDIT_POLICY.family_wise_alpha
            )
        ):
            raise ValueError("target-feature gate booleans are not self-consistent")

    @property
    def envelope_feature_gate_acceptable(self) -> bool:
        return (
            self.nmi_within_limit
            and self.balanced_accuracy_advantage_within_limit
            and self.adjusted_p_within_limit
        )


@dataclass(frozen=True, slots=True)
class CovertAuditMechanicsReceipt:
    schema_version: str
    semantics_version: str
    semantics_id: str
    policy_id: str
    structural_receipt_content_id: str
    disposition: CovertAuditDisposition
    sufficiency: tuple[TargetSufficiencyReceipt, ...]
    results: tuple[TargetFeatureAuditResult, ...]
    permutation_schedule_domain_root: str
    permutation_schedule_root: str | None
    permutations_requested: int
    permutations_executed_per_target: int
    holm_hypothesis_count: int
    audited_row_count: int
    estimated_statistical_work_units: int
    maximum_statistical_work_units: int
    envelope_feature_gate_acceptable: bool
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    formal_audit_evidence: bool = False
    sealed_holdout_eligible: bool = False
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not CovertAuditMechanicsReceipt:
            raise TypeError("covert audit receipt must use the exact frozen type")
        if (
            self.schema_version != SCHEMA_VERSION
            or self.semantics_version != SEMANTICS_VERSION
            or self.semantics_id != SEMANTICS_ID
            or self.policy_id != DEFAULT_COVERT_AUDIT_POLICY.policy_id
            or self.maximum_statistical_work_units
            != DEFAULT_COVERT_AUDIT_POLICY.maximum_statistical_work_units
        ):
            raise ValueError("covert audit receipt frozen identity drift")
        _require_content_id(
            self.structural_receipt_content_id,
            "phase2b_covert_audit_structural_receipt_",
            "mechanics structural receipt id",
        )
        expected_domain_root = stable_hash(
            {
                "domain_hex": PERMUTATION_DOMAIN.hex(),
                "policy_id": DEFAULT_COVERT_AUDIT_POLICY.policy_id,
                "semantics_id": SEMANTICS_ID,
            },
            prefix="phase2b_covert_permutation_domain_",
        )
        _require_content_id(
            self.permutation_schedule_domain_root,
            "phase2b_covert_permutation_domain_",
            "permutation schedule domain root",
        )
        if self.permutation_schedule_domain_root != expected_domain_root:
            raise ValueError("permutation schedule domain root drift")
        for name in (
            "permutations_requested",
            "permutations_executed_per_target",
            "holm_hypothesis_count",
            "audited_row_count",
            "estimated_statistical_work_units",
            "maximum_statistical_work_units",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if (
            self.permutations_requested
            != DEFAULT_COVERT_AUDIT_POLICY.label_permutations
            or not 1
            <= self.audited_row_count
            <= DEFAULT_COVERT_AUDIT_POLICY.maximum_rows
        ):
            raise ValueError("mechanics count is outside the frozen budget")
        expected_work_units = (
            self.audited_row_count
            * len(FROZEN_BYTE_FEATURE_IDS)
            * len(TARGETS)
            * self.permutations_requested
        )
        if self.estimated_statistical_work_units != expected_work_units:
            raise ValueError("mechanics statistical work formula drift")
        if any(
            type(value) is not bool
            for value in (
                self.envelope_feature_gate_acceptable,
                self.formal_audit_evidence,
                self.sealed_holdout_eligible,
            )
        ):
            raise TypeError("mechanics evidence and gate flags must be exact booleans")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL or any(
            (self.formal_audit_evidence, self.sealed_holdout_eligible)
        ):
            raise ValueError("covert audit mechanics cannot issue formal evidence")
        if type(self.sufficiency) is not tuple or type(self.results) is not tuple:
            raise TypeError("covert audit receipt arrays must use exact tuples")
        if len(self.sufficiency) != len(TARGETS):
            raise ValueError("mechanics sufficiency coverage drift")
        maximum_result_count = len(TARGETS) * len(FROZEN_BYTE_FEATURE_IDS)
        if len(self.results) > maximum_result_count:
            raise ValueError("mechanics result tuple exceeds the frozen cap")
        if any(type(item) is not TargetSufficiencyReceipt for item in self.sufficiency):
            raise TypeError("mechanics sufficiency rows must use exact types")
        if any(type(item) is not TargetFeatureAuditResult for item in self.results):
            raise TypeError("mechanics result rows must use exact types")
        for item in self.sufficiency:
            item.__post_init__()
        for item in self.results:
            item.__post_init__()
        if tuple(item.target for item in self.sufficiency) != TARGETS:
            raise ValueError("mechanics sufficiency target order drift")
        if self.disposition is CovertAuditDisposition.ABSTAIN:
            if (
                self.results
                or self.permutations_executed_per_target != 0
                or self.permutation_schedule_root is not None
                or self.holm_hypothesis_count != 0
                or all(item.sufficient for item in self.sufficiency)
            ):
                raise ValueError("abstaining audit cannot claim statistical execution")
            if self.envelope_feature_gate_acceptable:
                raise ValueError("abstaining audit cannot accept the channel gate")
        elif self.disposition is CovertAuditDisposition.STATISTICS_COMPLETE:
            _require_content_id(
                self.permutation_schedule_root,
                "phase2b_covert_permutation_schedule_",
                "complete permutation schedule root",
            )
            expected_coverage = tuple(
                (target, feature_id)
                for target in TARGETS
                for feature_id in FROZEN_BYTE_FEATURE_IDS
            )
            actual_coverage = tuple(
                (item.target, item.feature_id) for item in self.results
            )
            if (
                self.permutations_executed_per_target != self.permutations_requested
                or self.audited_row_count
                > DEFAULT_COVERT_AUDIT_POLICY.maximum_statistical_rows
                or len(self.results) != maximum_result_count
                or self.holm_hypothesis_count != maximum_result_count
                or actual_coverage != expected_coverage
                or not all(item.sufficient for item in self.sufficiency)
            ):
                raise ValueError("complete audit receipt is missing frozen statistics")
            expected_adjusted_p = _holm_adjust(
                tuple(item.raw_permutation_p for item in self.results)
            )
            if tuple(item.holm_adjusted_p for item in self.results) != expected_adjusted_p:
                raise ValueError("complete audit receipt Holm adjustment drift")
            expected_gate = all(
                item.envelope_feature_gate_acceptable for item in self.results
            )
            if self.envelope_feature_gate_acceptable is not expected_gate:
                raise ValueError("complete audit aggregate gate drift")
        else:
            raise TypeError("covert audit disposition must use the exact enum")
        object.__setattr__(
            self,
            "content_id",
            stable_hash(self, prefix="phase2b_covert_audit_mechanics_receipt_"),
        )


def _validate_run(
    receipt: RecognizerInvariantReceipt,
    expected_kind: InvarianceKind,
    expected_index: int,
    expected_ids: tuple[int, ...],
    baseline_by_id: dict[int, InvariantDecisionRow],
) -> int:
    if type(receipt) is not RecognizerInvariantReceipt:
        raise TypeError("invariance batch contains a non-exact receipt")
    if receipt.content_id != receipt.recompute_content_id():
        raise ValueError("invariance receipt cached content id drift")
    if receipt.kind is not expected_kind or receipt.trial_index != expected_index:
        raise ValueError("invariance receipt kind or trial index drift")
    ids = tuple(row.auditor_row_id for row in receipt.rows)
    if len(ids) != len(set(ids)) or tuple(sorted(ids)) != expected_ids:
        raise ValueError("invariance receipt does not cover every auditor row exactly once")
    if (
        sum(len(row.candidate_residual_vector) for row in receipt.rows)
        > DEFAULT_COVERT_AUDIT_POLICY.maximum_total_residual_vector_bytes
    ):
        raise ValueError("invariance receipt residual vectors exceed the frozen budget")
    comparisons = 0
    for row in receipt.rows:
        baseline = baseline_by_id[row.auditor_row_id]
        if (
            row.decision != baseline.decision
            or row.family != baseline.family
            or row.binding != baseline.binding
            or row.scale_set != baseline.scale_set
            or row.candidate_residual_vector != baseline.candidate_residual_vector
        ):
            raise ValueError("invariance receipt differs from the canonical baseline")
        comparisons += 5
    return comparisons


def validate_frozen_covert_audit_structure(
    rows: tuple[EnvelopeAuditRow, ...],
    invariance: InvarianceAuditBatch,
) -> CovertAuditStructuralReceipt:
    """Validate structural mechanics without issuing statistical/formal evidence."""

    policy = DEFAULT_COVERT_AUDIT_POLICY
    if tuple(signature(validate_frozen_covert_audit_structure).parameters) != ("rows", "invariance"):
        raise RuntimeError("authoritative covert-audit API surface drift")
    if type(rows) is not tuple:
        raise TypeError("audit rows must be an exact tuple")
    if not rows or len(rows) > policy.maximum_rows:
        raise ValueError("audit row count exceeds the frozen resource budget")
    if any(type(row) is not EnvelopeAuditRow for row in rows):
        raise TypeError("audit rows must contain exact audit rows")
    if any(row.content_id != row.recompute_content_id() for row in rows):
        raise ValueError("audit row cached content id drift")
    row_ids = tuple(row.auditor_row_id for row in rows)
    if row_ids != tuple(range(len(rows))):
        raise ValueError("audit rows must provide contiguous, ordered, all-row coverage")
    if len(rows) * policy.envelope_bytes > policy.maximum_total_envelope_bytes:
        raise ValueError("audit envelopes exceed the frozen total-byte budget")
    if type(invariance) is not InvarianceAuditBatch:
        raise TypeError("invariance input must use the exact batch type")
    groups = (
        (invariance.global_renamings, InvarianceKind.GLOBAL_RENAMING, policy.global_renamings),
        (invariance.case_order_permutations, InvarianceKind.CASE_ORDER, policy.global_case_order_permutations),
        (invariance.observation_order_permutations, InvarianceKind.OBSERVATION_ORDER, policy.observation_order_permutations),
    )
    baseline = invariance.baseline
    if type(baseline) is not RecognizerInvariantReceipt or baseline.kind is not InvarianceKind.BASELINE or baseline.trial_index != 0:
        raise ValueError("invariance baseline receipt drift")
    for receipts, _, count in groups:
        if type(receipts) is not tuple or len(receipts) != count:
            raise ValueError("invariance receipt count does not match the frozen audit")
    all_receipts = (baseline,) + tuple(
        receipt for receipts, _, _ in groups for receipt in receipts
    )
    if any(type(receipt) is not RecognizerInvariantReceipt for receipt in all_receipts):
        raise TypeError("invariance batch contains a non-exact receipt")
    if any(
        type(receipt.rows) is not tuple or len(receipt.rows) > policy.maximum_rows
        for receipt in all_receipts
    ):
        raise ValueError("invariance batch contains an oversized receipt")
    total_receipt_rows = sum(len(receipt.rows) for receipt in all_receipts)
    if total_receipt_rows > policy.maximum_total_invariance_receipt_rows:
        raise ValueError("invariance batch row work exceeds the frozen budget")
    if any(
        any(type(row) is not InvariantDecisionRow for row in receipt.rows)
        for receipt in all_receipts
    ):
        raise TypeError("invariance batch contains a non-exact decision row")
    total_invariance_residual_bytes = sum(
        len(row.candidate_residual_vector)
        for receipt in all_receipts
        for row in receipt.rows
    )
    if total_invariance_residual_bytes > policy.maximum_total_invariance_residual_bytes:
        raise ValueError("invariance batch residual bytes exceed the frozen budget")
    if baseline.content_id != baseline.recompute_content_id():
        raise ValueError("baseline cached content id drift")
    baseline_ids = tuple(row.auditor_row_id for row in baseline.rows)
    if baseline_ids != row_ids:
        raise ValueError("baseline receipt must cover rows in canonical order")
    baseline_by_id = {row.auditor_row_id: row for row in baseline.rows}
    total_residual_bytes = sum(len(row.candidate_residual_vector) for row in baseline.rows)
    if total_residual_bytes > policy.maximum_total_residual_vector_bytes:
        raise ValueError("baseline residual vectors exceed the frozen total budget")
    content_ids = [baseline.content_id]
    comparisons = 0
    for receipts, kind, count in groups:
        for index, receipt in enumerate(receipts):
            comparisons += _validate_run(receipt, kind, index, row_ids, baseline_by_id)
            content_ids.append(receipt.content_id)
    features = tuple(extract_frozen_byte_features(row) for row in rows)
    row_content_ids = tuple(row.content_id for row in rows)
    return CovertAuditStructuralReceipt(
        schema_version=SCHEMA_VERSION,
        semantics_version=SEMANTICS_VERSION,
        semantics_id=SEMANTICS_ID,
        policy_id=policy.policy_id,
        audit_input_content_id=stable_hash(
            {"row_content_ids": row_content_ids},
            prefix="phase2b_covert_audit_input_",
        ),
        auditor_row_content_ids=row_content_ids,
        row_count=len(rows),
        feature_count_per_row=len(features[0].values),
        invariant_receipt_content_ids=tuple(content_ids),
        exact_invariant_comparisons=comparisons,
    )


def _target_label(row: EnvelopeAuditRow, target: str) -> str:
    return {
        "family": row.labels.family,
        "binding": row.labels.binding,
        "scale": row.labels.scale,
        "answerable_vs_abstain": row.labels.answerable.value,
        "joint_decision_class": row.labels.joint_class,
    }[target]


def _target_stratum(row: EnvelopeAuditRow, target: str) -> tuple[str, ...]:
    return {
        "family": row.strata.family,
        "binding": row.strata.binding,
        "scale": row.strata.scale,
        "answerable_vs_abstain": row.strata.answerable,
        "joint_decision_class": row.strata.joint_class,
    }[target]


def _sufficiency_for_target(
    rows: tuple[EnvelopeAuditRow, ...],
    target: str,
) -> TargetSufficiencyReceipt:
    labels = tuple(_target_label(row, target) for row in rows)
    label_counts = Counter(labels)
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row, label in zip(rows, labels):
        groups[_target_stratum(row, target)].append(label)
    minimum_label_count = min(label_counts.values()) if label_counts else 0
    minimum_stratum_size = min(map(len, groups.values())) if groups else 0
    mutable_strata = sum(len(set(values)) >= 2 for values in groups.values())
    if len(rows) > DEFAULT_COVERT_AUDIT_POLICY.maximum_statistical_rows:
        reason = "STATISTICAL_WORK_BUDGET_EXCEEDED"
    elif len(label_counts) < 2:
        reason = "TARGET_HAS_FEWER_THAN_TWO_CLASSES"
    elif minimum_label_count < 2:
        reason = "LOO_CV_CLASS_HAS_FEWER_THAN_TWO_ROWS"
    elif minimum_stratum_size < 2:
        reason = "PERMUTATION_STRATUM_HAS_FEWER_THAN_TWO_ROWS"
    elif mutable_strata != len(groups):
        reason = "PERMUTATION_STRATUM_HAS_FEWER_THAN_TWO_TARGET_CLASSES"
    elif len(label_counts) > 256 or len(groups) > DEFAULT_COVERT_AUDIT_POLICY.maximum_rows:
        reason = "CLASS_OR_STRATUM_RESOURCE_BUDGET_EXCEEDED"
    else:
        reason = "SUFFICIENT"
    return TargetSufficiencyReceipt(
        target=target,
        label_count=len(label_counts),
        minimum_label_count=minimum_label_count,
        stratum_count=len(groups),
        minimum_stratum_size=minimum_stratum_size,
        mutable_stratum_count=mutable_strata,
        sufficient=reason == "SUFFICIENT",
        reason=reason,
    )


def _normalized_mutual_information(
    feature_values: tuple[int, ...],
    labels: tuple[str, ...],
) -> float:
    if type(feature_values) is not tuple or type(labels) is not tuple or len(feature_values) != len(labels) or not labels:
        raise ValueError("NMI inputs must be non-empty aligned exact tuples")
    total = len(labels)
    feature_counts = Counter(feature_values)
    label_counts = Counter(labels)
    if len(feature_counts) < 2 or len(label_counts) < 2:
        return 0.0
    joint = Counter(zip(feature_values, labels))
    mutual_information = math.fsum(
        (count / total)
        * math.log2(
            count * total / (feature_counts[feature] * label_counts[label])
        )
        for (feature, label), count in sorted(
            joint.items(),
            key=lambda item: (item[0][0], _utf8_label_order(item[0][1])),
        )
    )
    feature_entropy = -math.fsum(
        (feature_counts[value] / total)
        * math.log2(feature_counts[value] / total)
        for value in sorted(feature_counts)
    )
    label_entropy = -math.fsum(
        (label_counts[label] / total)
        * math.log2(label_counts[label] / total)
        for label in sorted(label_counts, key=_utf8_label_order)
    )
    denominator = math.sqrt(feature_entropy * label_entropy)
    if denominator == 0.0:
        return 0.0
    return min(1.0, max(0.0, mutual_information / denominator))


def _utf8_label_order(label: str) -> bytes:
    return label.encode("utf-8")


def _leave_one_out_balanced_accuracy_advantage(
    feature_values: tuple[int, ...],
    labels: tuple[str, ...],
) -> float:
    if type(feature_values) is not tuple or type(labels) is not tuple or len(feature_values) != len(labels) or not labels:
        raise ValueError("CV inputs must be non-empty aligned exact tuples")
    label_order = tuple(sorted(set(labels), key=_utf8_label_order))
    totals = Counter(labels)
    if len(label_order) < 2 or min(totals.values()) < 2:
        raise ValueError("leave-one-out CV is insufficient")
    by_value: dict[int, Counter[str]] = defaultdict(Counter)
    for value, label in zip(feature_values, labels):
        by_value[value][label] += 1
    correct = Counter({label: 0 for label in label_order})
    for value, true_label in zip(feature_values, labels):
        local = by_value[value]
        local_training_size = sum(local.values()) - 1
        source = local if local_training_size > 0 else totals
        best_label = min(
            label_order,
            key=lambda label: (
                -(source[label] - (1 if label == true_label else 0)),
                _utf8_label_order(label),
            ),
        )
        if best_label == true_label:
            correct[true_label] += 1
    balanced_accuracy = math.fsum(
        correct[label] / totals[label] for label in label_order
    ) / len(label_order)
    return balanced_accuracy - (1.0 / len(label_order))


def _combined_leak_statistic(nmi: float, advantage: float) -> float:
    policy = DEFAULT_COVERT_AUDIT_POLICY
    return max(
        nmi / policy.maximum_nmi,
        max(0.0, advantage) / policy.maximum_balanced_accuracy_advantage,
    )


class _Sha256Stream:
    __slots__ = ("_seed", "_counter", "_buffer")

    def __init__(self, seed: bytes) -> None:
        self._seed = seed
        self._counter = 0
        self._buffer = b""

    def _uint64(self) -> int:
        if len(self._buffer) < 8:
            self._buffer += hashlib.sha256(
                self._seed + self._counter.to_bytes(8, "big")
            ).digest()
            self._counter += 1
        value = int.from_bytes(self._buffer[:8], "big")
        self._buffer = self._buffer[8:]
        return value

    def randbelow(self, bound: int) -> int:
        if type(bound) is not int or bound <= 0:
            raise ValueError("SHA-256 stream bound must be positive")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self._uint64()
            if value < limit:
                return value % bound


def _encode_stratum(stratum: tuple[str, ...]) -> bytes:
    encoded = bytearray()
    for item in stratum:
        raw = item.encode("utf-8")
        encoded.extend(len(raw).to_bytes(4, "big"))
        encoded.extend(raw)
    return bytes(encoded)


def _stratified_permutation(
    labels: tuple[str, ...],
    strata: tuple[tuple[str, ...], ...],
    target: str,
    permutation_index: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if len(labels) != len(strata):
        raise ValueError("permutation labels and strata are not aligned")
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row_index, stratum in enumerate(strata):
        groups[stratum].append(row_index)
    permuted = list(labels)
    donors_by_position = list(range(len(labels)))
    for stratum in sorted(groups, key=lambda value: tuple(item.encode("utf-8") for item in value)):
        positions = groups[stratum]
        donors = list(positions)
        seed = hashlib.sha256(
            PERMUTATION_DOMAIN
            + len(target).to_bytes(2, "big")
            + target.encode("ascii")
            + permutation_index.to_bytes(8, "big")
            + _encode_stratum(stratum)
        ).digest()
        stream = _Sha256Stream(seed)
        for index in range(len(donors) - 1, 0, -1):
            swap = stream.randbelow(index + 1)
            donors[index], donors[swap] = donors[swap], donors[index]
        for position, donor in zip(positions, donors):
            permuted[position] = labels[donor]
            donors_by_position[position] = donor
    return tuple(permuted), tuple(donors_by_position)


def _holm_adjust(raw_p_values: tuple[float, ...]) -> tuple[float, ...]:
    if type(raw_p_values) is not tuple or any(
        type(value) is not float or not 0.0 <= value <= 1.0
        for value in raw_p_values
    ):
        raise TypeError("Holm input must be an exact tuple of probabilities")
    count = len(raw_p_values)
    order = sorted(range(count), key=lambda index: (raw_p_values[index], index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * raw_p_values[index]))
        adjusted[index] = running
    return tuple(adjusted)


def run_frozen_envelope_covert_audit_mechanics(
    rows: tuple[EnvelopeAuditRow, ...],
    invariance: InvarianceAuditBatch,
) -> CovertAuditMechanicsReceipt:
    """Run envelope-level mechanics; never issue a complete/formal audit claim."""

    if tuple(signature(run_frozen_envelope_covert_audit_mechanics).parameters) != ("rows", "invariance"):
        raise RuntimeError("authoritative statistical API surface drift")
    policy = DEFAULT_COVERT_AUDIT_POLICY
    structural = validate_frozen_covert_audit_structure(rows, invariance)
    estimated_work_units = (
        len(rows)
        * policy.frozen_feature_count_per_row
        * len(TARGETS)
        * policy.label_permutations
    )
    sufficiency = tuple(_sufficiency_for_target(rows, target) for target in TARGETS)
    schedule_domain_root = stable_hash(
        {
            "domain_hex": PERMUTATION_DOMAIN.hex(),
            "policy_id": policy.policy_id,
            "semantics_id": SEMANTICS_ID,
        },
        prefix="phase2b_covert_permutation_domain_",
    )
    if not all(item.sufficient for item in sufficiency):
        return CovertAuditMechanicsReceipt(
            schema_version=SCHEMA_VERSION,
            semantics_version=SEMANTICS_VERSION,
            semantics_id=SEMANTICS_ID,
            policy_id=policy.policy_id,
            structural_receipt_content_id=structural.content_id,
            disposition=CovertAuditDisposition.ABSTAIN,
            sufficiency=sufficiency,
            results=(),
            permutation_schedule_domain_root=schedule_domain_root,
            permutation_schedule_root=None,
            permutations_requested=policy.label_permutations,
            permutations_executed_per_target=0,
            holm_hypothesis_count=0,
            audited_row_count=len(rows),
            estimated_statistical_work_units=estimated_work_units,
            maximum_statistical_work_units=policy.maximum_statistical_work_units,
            envelope_feature_gate_acceptable=False,
        )

    feature_vectors = tuple(extract_frozen_byte_features(row) for row in rows)
    feature_ids = tuple(name for name, _ in feature_vectors[0].values)
    if feature_ids != FROZEN_BYTE_FEATURE_IDS or any(
        tuple(name for name, _ in vector.values) != FROZEN_BYTE_FEATURE_IDS
        for vector in feature_vectors
    ):
        raise RuntimeError("frozen byte feature identity drift")
    columns = tuple(
        tuple(vector.values[index][1] for vector in feature_vectors)
        for index in range(len(feature_ids))
    )
    observed: list[tuple[str, str, float, float, int]] = []
    exceedance_counts: list[int] = []
    schedule_hasher = hashlib.sha256()
    schedule_hasher.update(PERMUTATION_DOMAIN)
    schedule_hasher.update(bytes.fromhex(SEMANTICS_ID.rsplit("_", 1)[-1]))
    schedule_hasher.update(len(rows).to_bytes(4, "big"))

    for target in TARGETS:
        labels = tuple(_target_label(row, target) for row in rows)
        strata = tuple(_target_stratum(row, target) for row in rows)
        target_observed: list[tuple[float, float, float]] = []
        for feature_id, column in zip(feature_ids, columns):
            nmi = _normalized_mutual_information(column, labels)
            advantage = _leave_one_out_balanced_accuracy_advantage(column, labels)
            target_observed.append((nmi, advantage, _combined_leak_statistic(nmi, advantage)))
            observed.append((target, feature_id, nmi, advantage, 0))
            exceedance_counts.append(0)
        target_offset = len(observed) - len(feature_ids)
        metric_cache: dict[tuple[str, ...], tuple[float, ...]] = {}
        for permutation_index in range(policy.label_permutations):
            permuted_labels, donors = _stratified_permutation(
                labels,
                strata,
                target,
                permutation_index,
            )
            schedule_hasher.update(len(target).to_bytes(2, "big"))
            schedule_hasher.update(target.encode("ascii"))
            schedule_hasher.update(permutation_index.to_bytes(8, "big"))
            for donor in donors:
                schedule_hasher.update(donor.to_bytes(4, "big"))
            statistics = metric_cache.get(permuted_labels)
            if statistics is None:
                values: list[float] = []
                for column in columns:
                    nmi = _normalized_mutual_information(column, permuted_labels)
                    advantage = _leave_one_out_balanced_accuracy_advantage(
                        column,
                        permuted_labels,
                    )
                    values.append(_combined_leak_statistic(nmi, advantage))
                statistics = tuple(values)
                metric_cache[permuted_labels] = statistics
            for feature_index, statistic in enumerate(statistics):
                if statistic >= target_observed[feature_index][2]:
                    exceedance_counts[target_offset + feature_index] += 1

    raw_p_values = tuple(
        (count + 1) / (policy.label_permutations + 1)
        for count in exceedance_counts
    )
    adjusted_p_values = _holm_adjust(raw_p_values)
    results = tuple(
        TargetFeatureAuditResult(
            target=target,
            feature_id=feature_id,
            normalized_mutual_information=nmi,
            balanced_accuracy_advantage=advantage,
            permutation_exceedance_count=exceedance,
            raw_permutation_p=raw_p,
            holm_adjusted_p=adjusted_p,
            nmi_within_limit=nmi <= policy.maximum_nmi,
            balanced_accuracy_advantage_within_limit=(
                advantage <= policy.maximum_balanced_accuracy_advantage
            ),
            adjusted_p_within_limit=adjusted_p >= policy.family_wise_alpha,
        )
        for (
            (target, feature_id, nmi, advantage, _),
            exceedance,
            raw_p,
            adjusted_p,
        ) in zip(observed, exceedance_counts, raw_p_values, adjusted_p_values)
    )
    schedule_root = (
        "phase2b_covert_permutation_schedule_" + schedule_hasher.hexdigest()
    )
    return CovertAuditMechanicsReceipt(
        schema_version=SCHEMA_VERSION,
        semantics_version=SEMANTICS_VERSION,
        semantics_id=SEMANTICS_ID,
        policy_id=policy.policy_id,
        structural_receipt_content_id=structural.content_id,
        disposition=CovertAuditDisposition.STATISTICS_COMPLETE,
        sufficiency=sufficiency,
        results=results,
        permutation_schedule_domain_root=schedule_domain_root,
        permutation_schedule_root=schedule_root,
        permutations_requested=policy.label_permutations,
        permutations_executed_per_target=policy.label_permutations,
        holm_hypothesis_count=len(results),
        audited_row_count=len(rows),
        estimated_statistical_work_units=estimated_work_units,
        maximum_statistical_work_units=policy.maximum_statistical_work_units,
        envelope_feature_gate_acceptable=all(
            result.envelope_feature_gate_acceptable for result in results
        ),
    )


__all__ = [
    "AnswerabilityLabel",
    "AuditorLabels",
    "AuditorPermutationStrata",
    "ByteFeatureVector",
    "CovertAuditDisposition",
    "CovertAuditMechanicsReceipt",
    "CovertAuditPolicy",
    "CovertAuditStructuralReceipt",
    "DEFAULT_COVERT_AUDIT_POLICY",
    "EnvelopeAuditRow",
    "InvariantDecisionRow",
    "InvarianceAuditBatch",
    "InvarianceKind",
    "NON_AUTHORITATIVE_CLAIM_LEVEL",
    "RecognizerInvariantReceipt",
    "SEMANTICS_ID",
    "TargetFeatureAuditResult",
    "TargetSufficiencyReceipt",
    "extract_frozen_byte_features",
    "run_frozen_envelope_covert_audit_mechanics",
    "validate_frozen_covert_audit_structure",
]
