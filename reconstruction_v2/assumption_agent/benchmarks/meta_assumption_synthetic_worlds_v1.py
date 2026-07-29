"""Mechanism-computing, source-free development qualification.

Every fixture has the same numeric schema.  Five independent registered probe
rules compute statistics from that schema; no family label is available to the
selector.  The selected claim is then compiled and executed by
``PolicyRuntime`` against a closed synthetic operator lane.  This remains a
development qualification rather than efficacy evidence on a reality source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Callable, Mapping, Sequence

from assumption_agent.meta_assumption import (
    CompilerTrustAnchor,
    CompilationReceipt,
    CompiledTreatment,
    CompilerTarget,
    DiagnosticProbePlan,
    HypothesisClaim,
    HypothesisSpaceCompilerRegistry,
    ProbeDisposition,
    ProbeEvidenceBundle,
    ProbeObservationStatistic,
    ProbeReceipt,
    ProbeTrustAnchor,
    ProbeVerificationResult,
    ProbeVerifierRegistry,
    RecipeActionBinding,
    TreatmentDisposition,
    UniversalAssumptionOntology,
    action_node_semantics_hash,
    verify_compilation_receipt,
)
from assumption_agent.models import (
    ActionNode,
    ExpectedEffect,
    FeaturePredicate,
    HypothesisKind,
    HypothesisProgram,
    HypothesisStatus,
    LaneResult,
    SplitName,
    TaskInput,
    TriggerSpec,
    VerifierContract,
    stable_hash,
)
from assumption_agent.runtime import LaneRegistry, PolicyRuntime
from assumption_agent.universal_assumption_ontology_v1 import (
    build_universal_assumption_ontology_v1,
)


VERSION = "meta_assumption_synthetic_worlds_v1"
RECEIPT_SCHEMA = f"{VERSION}_development_receipt"
STATUS = "passed_nonformal_source_free_development_qualification"
PRIMARY_METRIC = "synthetic_structural_contract"
EVALUATOR_EPOCH = "source_free_synthetic_v1"
SELECTION_POLICY_ID = "unique_supported_nonfalsified_minimum_description_v1"
TREATMENT_COMPILER_ID = "synthetic_closed_treatment_compiler_v1"
TREATMENT_COMPILER_VERSION = "synthetic_closed_treatment_compiler_v1"
PROBE_VERIFIER_VERSION = "synthetic_numeric_probe_verifier_v1"
VARIANTS = ("a", "b")
FAMILIES = (
    "sparse",
    "set_interaction",
    "local",
    "contamination",
    "no_op",
)
TEMPLATE_ID_BY_FAMILY = {
    "sparse": "uao.v1.t02_sparsity",
    "set_interaction": "uao.v1.t05_low_order_interaction",
    "local": "uao.v1.t08_locality_markov_blanket",
    "contamination": "uao.v1.t18_sparse_contamination",
    "no_op": "uao.v1.t19_minimum_commitment",
}
OPERATOR_BY_FAMILY = {
    "sparse": "SPARSE_SUPPORT_SELECT",
    "set_interaction": "SET_INTERACTION_SCORE",
    "local": "LOCAL_NEIGHBORHOOD",
    "contamination": "ROBUST_TRIM_OR_MIXTURE",
    "no_op": "PRESERVE_BASELINE",
}
ACTIVE_OPERATORS = tuple(
    OPERATOR_BY_FAMILY[family]
    for family in FAMILIES
    if family != "no_op"
)
BASELINE_LANE = "synthetic.baseline"
OPERATOR_LANE = "synthetic.closed_operator"
RECIPE_PARAMETER = "meta_assumption.synthetic_operator"
ACTION_COUNT = 6
COMPONENT_COUNT = 6
NODE_COUNT = 6
OBSERVATION_COUNT = 8
DECISION_ACTION_COUNT = 4


class SyntheticQualificationError(RuntimeError):
    """The deterministic development qualification failed closed."""


def canonical_bytes(value: object) -> bytes:
    """Return strict canonical ASCII JSON bytes with one trailing newline."""

    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SyntheticQualificationError(
            "qualification receipt is not canonical JSON"
        ) from exc


def semantic_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _rotate(values: Sequence[str], amount: int) -> tuple[str, ...]:
    if not values:
        return ()
    offset = amount % len(values)
    return tuple(values[offset:]) + tuple(values[:offset])


@dataclass(frozen=True)
class NumericWorldPayload:
    """One fixed-shape numeric payload shared by every mechanism family."""

    action_fold_utilities: tuple[tuple[int, ...], ...]
    subset_utility_folds: tuple[tuple[int, ...], ...]
    adjacency: tuple[tuple[int, ...], ...]
    focal_node: int
    node_effect_folds: tuple[tuple[int, ...], ...]
    observation_folds: tuple[tuple[int, ...], ...]
    decision_payoffs: tuple[tuple[int, ...], ...]

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if (
            len(self.action_fold_utilities) != 4
            or any(
                len(row) != ACTION_COUNT
                for row in self.action_fold_utilities
            )
        ):
            issues.append("numeric_world_action_shape_invalid")
        if (
            len(self.subset_utility_folds) != 2
            or any(
                len(row) != 2**COMPONENT_COUNT
                for row in self.subset_utility_folds
            )
        ):
            issues.append("numeric_world_subset_shape_invalid")
        if (
            len(self.adjacency) != NODE_COUNT
            or any(len(row) != NODE_COUNT for row in self.adjacency)
            or any(
                type(value) is not int or value not in (0, 1)
                for row in self.adjacency
                for value in row
            )
            or any(
                self.adjacency[left][right]
                != self.adjacency[right][left]
                for left in range(NODE_COUNT)
                for right in range(NODE_COUNT)
            )
        ):
            issues.append("numeric_world_adjacency_invalid")
        if (
            type(self.focal_node) is not int
            or not 0 <= self.focal_node < NODE_COUNT
        ):
            issues.append("numeric_world_focal_node_invalid")
        if (
            len(self.node_effect_folds) != 4
            or any(
                len(row) != NODE_COUNT for row in self.node_effect_folds
            )
        ):
            issues.append("numeric_world_node_effect_shape_invalid")
        if (
            len(self.observation_folds) != 4
            or any(
                len(row) != OBSERVATION_COUNT
                for row in self.observation_folds
            )
        ):
            issues.append("numeric_world_observation_shape_invalid")
        if (
            len(self.decision_payoffs) != 4
            or any(
                len(row) != DECISION_ACTION_COUNT
                for row in self.decision_payoffs
            )
        ):
            issues.append("numeric_world_decision_shape_invalid")
        if any(
            type(value) is not int
            for panel in (
                self.action_fold_utilities,
                self.subset_utility_folds,
                self.node_effect_folds,
                self.observation_folds,
                self.decision_payoffs,
            )
            for row in panel
            for value in row
        ):
            issues.append("numeric_world_noninteger_value")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, object]:
        return {
            "action_fold_utilities": [
                list(row) for row in self.action_fold_utilities
            ],
            "subset_utility_folds": [
                list(row) for row in self.subset_utility_folds
            ],
            "adjacency": [list(row) for row in self.adjacency],
            "focal_node": self.focal_node,
            "node_effect_folds": [
                list(row) for row in self.node_effect_folds
            ],
            "observation_folds": [
                list(row) for row in self.observation_folds
            ],
            "decision_payoffs": [
                list(row) for row in self.decision_payoffs
            ],
        }


@dataclass(frozen=True)
class NumericProbeBundle:
    """The complete numeric selector input; no family label is present."""

    world_id: str
    numeric_payload: NumericWorldPayload

    def safe_payload(self) -> dict[str, object]:
        return {
            "world_id": self.world_id,
            "numeric_payload": self.numeric_payload.safe_payload(),
        }


@dataclass(frozen=True)
class SyntheticWorld:
    """Private fixture wrapper; only ``selector_input`` reaches selection."""

    selector_input: NumericProbeBundle
    expected_family: str
    variant: str
    claim_order: tuple[str, ...]


@dataclass(frozen=True)
class CompilationBinding:
    """Expected immutable chain for independent receipt verification."""

    ontology: UniversalAssumptionOntology
    claim: HypothesisClaim
    probes: tuple[ProbeReceipt, ...]
    probe_evidence_bundles: tuple[ProbeEvidenceBundle, ...]
    probe_verifier_registry: ProbeVerifierRegistry
    treatment: CompiledTreatment
    receipt_id: str
    ontology_hash: str
    template_hashes: tuple[str, ...]
    claim_hash: str
    probe_receipt_hashes: tuple[str, ...]
    compiler_id: str
    compiler_version: str
    compiler_implementation_hash: str
    primary_metric: str
    compiler_trust_anchor_hash: str
    compiler_target: CompilerTarget
    treatment_disposition: TreatmentDisposition
    recipe_ids: tuple[str, ...]
    recipe_action_binding_hashes: tuple[str, ...]
    treatment_behavior_hash: str
    selection_policy_hash: str
    compiler_trust_anchor: CompilerTrustAnchor


@dataclass(frozen=True)
class ProbeComputation:
    receipt: ProbeReceipt
    evidence_bundle: ProbeEvidenceBundle
    statistic_values: tuple[tuple[str, int], ...]
    statistic_hash: str


@dataclass(frozen=True)
class RuntimeEvidence:
    baseline_plan_hash: str
    candidate_plan_hash: str
    baseline_utility: int
    candidate_utility: int
    selected_lane: str
    active_runtime_differential: bool
    noop_runtime_equivalent: bool
    wrong_operator_trial_count: int
    wrong_operator_harm_count: int
    semantic_commitment: str


@dataclass(frozen=True)
class TamperOutcome:
    case_id: str
    rejected: bool
    expected_issue_ids: tuple[str, ...]
    observed_issue_ids: tuple[str, ...]
    cause_type: str


@dataclass(frozen=True)
class WorldQualification:
    world: SyntheticWorld
    selected_family: str
    selected_claim: HypothesisClaim
    probe_receipts: tuple[ProbeReceipt, ...]
    probe_computations: tuple[ProbeComputation, ...]
    treatment: CompiledTreatment
    compilation_receipt: CompilationReceipt
    binding: CompilationBinding
    runtime_evidence: RuntimeEvidence


@dataclass(frozen=True)
class QualificationArtifacts:
    ontology: UniversalAssumptionOntology
    probe_verifier_registry: ProbeVerifierRegistry
    claims_by_family: Mapping[str, HypothesisClaim]
    probe_plans_by_family: Mapping[str, DiagnosticProbePlan]
    worlds: tuple[SyntheticWorld, ...]
    world_qualifications: tuple[WorldQualification, ...]


def _adjacency(edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    rows = [[0 for _ in range(NODE_COUNT)] for _ in range(NODE_COUNT)]
    for left, right in edges:
        rows[left][right] = 1
        rows[right][left] = 1
    return tuple(tuple(row) for row in rows)


def _subset_table(
    *,
    pair_effects: Sequence[tuple[int, int, int]] = (),
    triple: tuple[int, int, int] | None = None,
) -> tuple[int, ...]:
    values: list[int] = []
    pair_masks = tuple(
        ((1 << left) | (1 << right), strength)
        for left, right, strength in pair_effects
    )
    triple_mask = (
        (1 << triple[0]) | (1 << triple[1]) | (1 << triple[2])
        if triple is not None
        else 0
    )
    for mask in range(2**COMPONENT_COUNT):
        value = 0
        value += sum(
            strength
            for pair_mask, strength in pair_masks
            if mask & pair_mask == pair_mask
        )
        if triple is not None and mask & triple_mask == triple_mask:
            value += 20
        values.append(value)
    return tuple(values)


def _control_payload() -> NumericWorldPayload:
    payload = NumericWorldPayload(
        action_fold_utilities=tuple(
            tuple(5 for _ in range(ACTION_COUNT))
            for _ in range(4)
        ),
        subset_utility_folds=(
            _subset_table(triple=(0, 1, 2)),
            _subset_table(triple=(0, 1, 2)),
        ),
        adjacency=_adjacency(
            tuple((index, index + 1) for index in range(NODE_COUNT - 1))
        ),
        focal_node=0,
        node_effect_folds=tuple(
            tuple(-2 for _ in range(NODE_COUNT))
            for _ in range(4)
        ),
        observation_folds=(
            (-3, -2, -1, 0, 0, 1, 2, 3),
            (-2, -1, 0, 1, -1, 0, 1, 2),
            (-4, -2, -1, 0, 0, 1, 2, 4),
            (-3, -1, 0, 1, -1, 0, 1, 3),
        ),
        decision_payoffs=tuple(
            (15, 0, -5, -10) for _ in range(4)
        ),
    )
    if payload.validate():
        raise SyntheticQualificationError("control numeric payload is invalid")
    return payload


def _payload_for(family: str, variant: str) -> NumericWorldPayload:
    base = _control_payload()
    if family == "sparse":
        action_folds = (
            (
                (30, -2, 0, -1, 25, -2),
                (28, -1, -1, 0, 27, -2),
                (31, -2, 0, -1, 24, -1),
                (29, -1, -1, -2, 26, 0),
            )
            if variant == "a"
            else (
                (-1, 25, -2, 22, -1, 20),
                (0, 26, -1, 21, -1, 19),
                (-2, 24, 0, 23, -2, 21),
                (-1, 25, -2, 22, -1, 20),
            )
        )
        payload = replace(base, action_fold_utilities=action_folds)
    elif family == "set_interaction":
        pair_graph = (
            ((0, 1, 20), (1, 2, 20), (2, 3, 20))
            if variant == "a"
            else ((0, 1, 20), (0, 2, 20), (0, 3, 20))
        )
        held_pair_graph = tuple(
            (left, right, strength + 1)
            for left, right, strength in pair_graph
        )
        payload = replace(
            base,
            subset_utility_folds=(
                _subset_table(pair_effects=pair_graph),
                _subset_table(pair_effects=held_pair_graph),
            ),
        )
    elif family == "local":
        if variant == "a":
            adjacency = _adjacency(
                tuple(
                    (index, index + 1)
                    for index in range(NODE_COUNT - 1)
                )
            )
            focal = 2
            rows = (
                (2, 20, 30, 21, 2, 1),
                (2, 19, 31, 20, 2, 1),
                (1, 21, 29, 20, 2, 1),
                (2, 20, 30, 19, 1, 1),
            )
        else:
            adjacency = _adjacency(
                ((0, 1), (1, 2), (1, 3), (3, 4), (3, 5))
            )
            focal = 3
            rows = (
                (1, 20, 2, 32, 21, 19),
                (2, 19, 1, 30, 20, 21),
                (1, 21, 2, 31, 19, 20),
                (2, 20, 1, 30, 21, 19),
            )
        payload = replace(
            base,
            adjacency=adjacency,
            focal_node=focal,
            node_effect_folds=rows,
        )
    elif family == "contamination":
        rows = (
            (
                (10, 200, 9, 11, 10, 8, 12, 10),
                (11, 190, 10, 9, 12, 10, 8, 10),
                (9, 180, 11, 10, 8, 12, 10, 10),
                (10, 210, 8, 12, 11, 9, 10, 10),
            )
            if variant == "a"
            else (
                (-200, -11, -9, -10, -12, -8, -10, -180),
                (-180, -9, -11, -10, -8, -12, -10, -210),
                (-210, -10, -8, -12, -11, -9, -10, -190),
                (-190, -12, -10, -8, -9, -11, -10, -200),
            )
        )
        payload = replace(base, observation_folds=rows)
    elif family == "no_op":
        payoffs = (
            (
                (10, -20, 5, -10),
                (-20, 10, -10, 5),
                (10, -20, 5, -10),
                (-20, 10, -10, 5),
            )
            if variant == "a"
            else (
                (20, -15, -15, -15),
                (-15, 20, -15, -15),
                (-15, -15, 20, -15),
                (-15, -15, -15, 20),
            )
        )
        payload = replace(base, decision_payoffs=payoffs)
    else:
        raise SyntheticQualificationError("unknown synthetic family")
    issues = payload.validate()
    if issues:
        raise SyntheticQualificationError(
            f"numeric payload is invalid: {list(issues)}"
        )
    return payload


def _build_worlds() -> tuple[SyntheticWorld, ...]:
    worlds: list[SyntheticWorld] = []
    family_claim_ids = tuple(
        f"synthetic.claim.{family}" for family in FAMILIES
    )
    for family_index, family in enumerate(FAMILIES):
        for variant_index, variant in enumerate(VARIANTS):
            payload = _payload_for(family, variant)
            world_id = "omega." + semantic_hash(
                {
                    "numeric_payload": payload.safe_payload(),
                    "variant_ordinal": variant_index,
                    "fixture_version": VERSION,
                }
            )[:20]
            worlds.append(
                SyntheticWorld(
                    selector_input=NumericProbeBundle(
                        world_id=world_id,
                        numeric_payload=payload,
                    ),
                    expected_family=family,
                    variant=variant,
                    claim_order=_rotate(
                        family_claim_ids,
                        family_index * 3 + variant_index + 1,
                    ),
                )
            )
    return tuple(worlds)


def _variant_structure_descriptor(
    world: SyntheticWorld,
) -> dict[str, object]:
    payload = world.selector_input.numeric_payload
    if world.expected_family == "sparse":
        statistics, _support, _counter = _sparse_probe(payload)
        selected = tuple(
            index
            for index, value in enumerate(
                _column_sums(payload.action_fold_utilities)
            )
            if value > 40
        )
        return {
            "active_set_size": statistics["active_set_size"],
            "active_indices": list(selected),
        }
    if world.expected_family == "set_interaction":
        _unary, pairs, _residual = _subset_decomposition(
            payload.subset_utility_folds[0]
        )
        edges = tuple(
            pair for pair, effect in pairs.items() if effect > 0
        )
        degrees = [0] * COMPONENT_COUNT
        for left, right in edges:
            degrees[left] += 1
            degrees[right] += 1
        return {
            "interaction_edges": [list(edge) for edge in edges],
            "degree_sequence": sorted(degrees),
        }
    if world.expected_family == "local":
        edges = tuple(
            (left, right)
            for left in range(NODE_COUNT)
            for right in range(left + 1, NODE_COUNT)
            if payload.adjacency[left][right]
        )
        return {
            "graph_edges": [list(edge) for edge in edges],
            "degree_sequence": sorted(
                sum(row) for row in payload.adjacency
            ),
            "focal_node": payload.focal_node,
        }
    if world.expected_family == "contamination":
        outlier_positions = tuple(
            tuple(
                index
                for index, value in enumerate(row)
                if abs(2 * value - _median_twice(row)) >= 100
            )
            for row in payload.observation_folds
        )
        return {
            "outlier_positions": [
                list(values) for values in outlier_positions
            ],
            "outlier_count_per_fold": [
                len(values) for values in outlier_positions
            ],
        }
    if world.expected_family == "no_op":
        return {
            "positive_action_incidence": [
                [int(value > 0) for value in row]
                for row in payload.decision_payoffs
            ],
            "positive_action_count_by_context": [
                sum(value > 0 for value in row)
                for row in payload.decision_payoffs
            ],
            "positive_context_count_by_action": [
                sum(row[index] > 0 for row in payload.decision_payoffs)
                for index in range(DECISION_ACTION_COUNT)
            ],
        }
    raise SyntheticQualificationError("unknown structural variant family")


def _structural_variant_commitments(
    worlds: Sequence[SyntheticWorld],
) -> list[dict[str, object]]:
    commitments: list[dict[str, object]] = []
    for family in FAMILIES:
        pair = tuple(
            world for world in worlds if world.expected_family == family
        )
        if len(pair) != 2 or {world.variant for world in pair} != set(
            VARIANTS
        ):
            raise SyntheticQualificationError(
                "structural variant registry is incomplete"
            )
        ordered = tuple(sorted(pair, key=lambda world: world.variant))
        descriptors = tuple(
            _variant_structure_descriptor(world) for world in ordered
        )
        hashes = tuple(semantic_hash(value) for value in descriptors)
        if hashes[0] == hashes[1]:
            raise SyntheticQualificationError(
                "synthetic variants are structurally identical"
            )
        if family == "sparse" and {
            descriptor["active_set_size"] for descriptor in descriptors
        } != {2, 3}:
            raise SyntheticQualificationError(
                "sparse variants are not 2-active versus 3-active"
            )
        if family == "set_interaction" and (
            descriptors[0]["degree_sequence"]
            == descriptors[1]["degree_sequence"]
        ):
            raise SyntheticQualificationError(
                "interaction variants are graph-isomorphic by degree"
            )
        if family == "local" and (
            descriptors[0]["degree_sequence"]
            == descriptors[1]["degree_sequence"]
        ):
            raise SyntheticQualificationError(
                "local variants are graph-isomorphic by degree"
            )
        if family == "contamination" and {
            tuple(descriptor["outlier_count_per_fold"])
            for descriptor in descriptors
        } != {(1, 1, 1, 1), (2, 2, 2, 2)}:
            raise SyntheticQualificationError(
                "contamination variants are not singleton versus pair"
            )
        if family == "no_op" and (
            descriptors[0]["positive_action_incidence"]
            == descriptors[1]["positive_action_incidence"]
            or (
                sorted(
                    descriptors[0][
                        "positive_action_count_by_context"
                    ]
                ),
                sorted(
                    descriptors[0][
                        "positive_context_count_by_action"
                    ]
                ),
            )
            == (
                sorted(
                    descriptors[1][
                        "positive_action_count_by_context"
                    ]
                ),
                sorted(
                    descriptors[1][
                        "positive_context_count_by_action"
                    ]
                ),
            )
        ):
            raise SyntheticQualificationError(
                "no-op variants have identical bipartite incidence degrees"
            )
        commitments.append(
            {
                "family": family,
                "variant_a_structure_hash": hashes[0],
                "variant_b_structure_hash": hashes[1],
            }
        )
    return commitments


def _claim_id(family: str) -> str:
    return f"synthetic.claim.{family}"


def _build_claims(
    ontology: UniversalAssumptionOntology,
) -> dict[str, HypothesisClaim]:
    formation_evidence_hash = stable_hash(
        {
            "fixture_version": VERSION,
            "split": SplitName.TRAIN.value,
            "world_ids": [
                world.selector_input.world_id for world in _build_worlds()
            ],
        }
    )
    claims: dict[str, HypothesisClaim] = {}
    for family_index, family in enumerate(FAMILIES):
        template_id = TEMPLATE_ID_BY_FAMILY[family]
        template = ontology.require_template(template_id)
        competing = tuple(
            _claim_id(other)
            for other in FAMILIES
            if other != family
        )
        claims[family] = HypothesisClaim(
            claim_id=_claim_id(family),
            ontology_hash=ontology.ontology_hash,
            template_ids=(template_id,),
            mechanism_statement=(
                (
                    "Only after all four active TRAIN-only mechanism probes "
                    "are falsified, the registered minimum-commitment probe "
                    "should preserve baseline when no fixed action has "
                    "stable positive utility."
                )
                if family == "no_op"
                else (
                    "The registered TRAIN-only numeric probe for "
                    f"{template_id} should satisfy its support signatures "
                    "and reject its declared counter signatures."
                )
            ),
            scope_hash=stable_hash(
                {
                    "scope": "source_free_known_mechanism",
                    "version": VERSION,
                }
            ),
            bound_variable_types=(template.admissible_variable_types[0],),
            observable_predictions=template.support_signatures,
            counter_predictions=template.counter_signatures,
            competing_claim_ids=competing,
            description_length_bits=100 + family_index,
            evidence_receipt_hashes=(formation_evidence_hash,),
            formation_split=SplitName.TRAIN,
            lineage_claim_ids=(),
        )
        issues = claims[family].validate(ontology)
        if issues:
            raise SyntheticQualificationError(
                f"synthetic claim is invalid: {list(issues)}"
            )
    return claims


def _build_probe_plans(
    ontology: UniversalAssumptionOntology,
    claims_by_family: Mapping[str, HypothesisClaim],
) -> dict[str, DiagnosticProbePlan]:
    return {
        family: ontology.require_template(
            claim.template_ids[0]
        ).probe_plan
        for family, claim in claims_by_family.items()
    }


def _column_sums(rows: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(
        sum(row[index] for row in rows)
        for index in range(len(rows[0]))
    )


def _ratio_basis_points(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (10_000 * numerator + denominator // 2) // denominator


def _median_twice(values: Sequence[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return 2 * ordered[middle]
    return ordered[middle - 1] + ordered[middle]


def _sparse_probe(
    payload: NumericWorldPayload,
) -> tuple[dict[str, int], tuple[bool, bool], tuple[bool, bool]]:
    column_sums = _column_sums(payload.action_fold_utilities)
    positive_total = sum(max(value, 0) for value in column_sums)
    fold_capture_sets: list[frozenset[int]] = []
    fold_capture_numerators: list[int] = []
    for row in payload.action_fold_utilities:
        positive = tuple(max(value, 0) for value in row)
        total = sum(positive)
        ranked = sorted(
            range(ACTION_COUNT),
            key=lambda index: (-positive[index], index),
        )
        selected: list[int] = []
        captured = 0
        for index in ranked:
            if positive[index] <= 0:
                break
            selected.append(index)
            captured += positive[index]
            if 5 * captured >= 4 * total:
                break
        fold_capture_sets.append(frozenset(selected))
        fold_capture_numerators.append(captured)
    union = frozenset().union(*fold_capture_sets)
    intersection = set(fold_capture_sets[0]).intersection(
        *fold_capture_sets[1:]
    )
    active_size = max(len(values) for values in fold_capture_sets)
    captured_numerator = sum(fold_capture_numerators)
    captured_denominator = sum(
        sum(max(value, 0) for value in row)
        for row in payload.action_fold_utilities
    )
    support_small = (
        0 < active_size <= 3
        and all(
            5 * captured >= 4 * sum(max(value, 0) for value in row)
            for captured, row in zip(
                fold_capture_numerators,
                payload.action_fold_utilities,
            )
        )
        and positive_total >= 80
    )
    support_stable = (
        support_small
        and 4 * len(intersection) >= 3 * (len(union) or 1)
    )
    counter_diffuse = (
        active_size >= 4
    )
    counter_unstable = (
        2 * len(intersection) <= (len(union) or 1)
    )
    return (
        {
            "active_set_size": active_size,
            "captured_utility_fraction": _ratio_basis_points(
                captured_numerator, captured_denominator
            ),
            "fold_selection_frequency": _ratio_basis_points(
                len(intersection), len(union)
            ),
            "rule_capture_margin": min(
                5 * captured
                - 4 * sum(max(value, 0) for value in row)
                for captured, row in zip(
                    fold_capture_numerators,
                    payload.action_fold_utilities,
                )
            ),
            "rule_positive_total": positive_total,
            "rule_stability_support_margin": (
                4 * len(intersection) - 3 * (len(union) or 1)
            ),
            "rule_stability_counter_margin": (
                2 * len(intersection) - (len(union) or 1)
            ),
        },
        (support_small, support_stable),
        (counter_diffuse, counter_unstable),
    )


def _subset_decomposition(
    values: Sequence[int],
) -> tuple[tuple[int, ...], dict[tuple[int, int], int], int]:
    base = values[0]
    unary = tuple(values[1 << index] - base for index in range(COMPONENT_COUNT))
    pairs: dict[tuple[int, int], int] = {}
    for left in range(COMPONENT_COUNT):
        for right in range(left + 1, COMPONENT_COUNT):
            mask = (1 << left) | (1 << right)
            pairs[(left, right)] = (
                values[mask] - base - unary[left] - unary[right]
            )
    residual = 0
    for mask, observed in enumerate(values):
        predicted = base + sum(
            unary[index]
            for index in range(COMPONENT_COUNT)
            if mask & (1 << index)
        )
        predicted += sum(
            effect
            for (left, right), effect in pairs.items()
            if mask & (1 << left) and mask & (1 << right)
        )
        residual = max(residual, abs(observed - predicted))
    return unary, pairs, residual


def _set_interaction_probe(
    payload: NumericWorldPayload,
) -> tuple[dict[str, int], tuple[bool, bool], tuple[bool, bool]]:
    decompositions = tuple(
        _subset_decomposition(values)
        for values in payload.subset_utility_folds
    )
    unary_effect = max(
        abs(value)
        for unary, _pairs, _residual in decompositions
        for value in unary
    )
    pair_effect = min(
        max(pairs.values())
        for _unary, pairs, _residual in decompositions
    )
    higher_order_residual = max(
        residual for _unary, _pairs, residual in decompositions
    )
    best_pairs = tuple(
        min(
            pairs,
            key=lambda pair: (-pairs[pair], pair),
        )
        for _unary, pairs, _residual in decompositions
    )
    held_error = max(
        abs(
            payload.subset_utility_folds[0][mask]
            - payload.subset_utility_folds[1][mask]
        )
        for mask in range(2**COMPONENT_COUNT)
    )
    low_order_explains = (
        unary_effect <= 1
        and pair_effect >= 10
        and higher_order_residual <= 1
    )
    transfers = (
        low_order_explains
        and len(set(best_pairs)) == 1
        and held_error <= 3
    )
    irreducible = higher_order_residual >= 8
    held_failure = pair_effect < 2 or held_error >= 8
    return (
        {
            "unary_effect": unary_effect,
            "pair_effect": pair_effect,
            "higher_order_residual": higher_order_residual,
            "held_combination_error": held_error,
        },
        (low_order_explains, transfers),
        (irreducible, held_failure),
    )


def _graph_distances(
    adjacency: Sequence[Sequence[int]], focal_node: int
) -> tuple[int, ...]:
    distances = [-1] * len(adjacency)
    distances[focal_node] = 0
    frontier = [focal_node]
    while frontier:
        node = frontier.pop(0)
        for neighbor, connected in enumerate(adjacency[node]):
            if connected and distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                frontier.append(neighbor)
    if any(distance < 0 for distance in distances):
        raise SyntheticQualificationError("synthetic graph is disconnected")
    return tuple(distances)


def _local_probe(
    payload: NumericWorldPayload,
) -> tuple[dict[str, int], tuple[bool, bool], tuple[bool, bool]]:
    distances = _graph_distances(payload.adjacency, payload.focal_node)
    effects = tuple(
        abs(value) for value in _column_sums(payload.node_effect_folds)
    )
    near = tuple(
        effects[index] for index, distance in enumerate(distances)
        if distance <= 1
    )
    remote = tuple(
        effects[index] for index, distance in enumerate(distances)
        if distance > 1
    )
    near_sum = sum(near)
    remote_sum = sum(remote)
    decays = (
        near_sum * len(remote)
        >= 3 * remote_sum * len(near)
    )
    sufficient = 4 * max(remote) <= max(near)
    distant_equal = (
        5 * remote_sum * len(near)
        >= 4 * near_sum * len(remote)
    )
    omitted_parent = 5 * max(remote) >= 3 * max(near)
    return (
        {
            "distance_bin": _ratio_basis_points(
                near_sum * len(remote),
                remote_sum * len(near),
            ),
            "ablated_effect": (
                near_sum * len(remote)
                - remote_sum * len(near)
            ),
            "conditional_increment": _ratio_basis_points(
                max(remote), max(near)
            ),
            "rule_decay_margin": (
                near_sum * len(remote)
                - 3 * remote_sum * len(near)
            ),
            "rule_sufficiency_margin": max(near) - 4 * max(remote),
            "rule_distant_equal_margin": (
                5 * remote_sum * len(near)
                - 4 * near_sum * len(remote)
            ),
            "rule_omitted_parent_margin": (
                5 * max(remote) - 3 * max(near)
            ),
        },
        (decays, sufficient),
        (distant_equal, omitted_parent),
    )


def _contamination_probe(
    payload: NumericWorldPayload,
) -> tuple[dict[str, int], tuple[bool, bool], tuple[bool, bool]]:
    fold_medians_twice = tuple(
        _median_twice(row) for row in payload.observation_folds
    )
    gap_numerators = tuple(
        abs(
            2 * sum(row)
            - len(row) * fold_medians_twice[fold_index]
        )
        for fold_index, row in enumerate(payload.observation_folds)
    )
    influences = tuple(
        sum(
            abs(2 * row[index] - fold_medians_twice[fold_index])
            for fold_index, row in enumerate(payload.observation_folds)
        )
        for index in range(OBSERVATION_COUNT)
    )
    total_influence = sum(influences)
    top_two_influence = sum(sorted(influences, reverse=True)[:2])
    robust_spread_twice = (
        max(fold_medians_twice) - min(fold_medians_twice)
    )
    gap_numerator = min(gap_numerators)
    gap_denominator = 2 * OBSERVATION_COUNT
    bounded_minority = (
        4 * top_two_influence >= 3 * total_influence
        and gap_numerator >= 10 * gap_denominator
    )
    robust_stable = (
        gap_numerator >= 10 * gap_denominator
        and robust_spread_twice <= 6
    )
    diffuse = (
        gap_numerator <= 3 * gap_denominator
        or 20 * top_two_influence < 11 * total_influence
    )
    no_improvement = (
        gap_numerator <= 3 * gap_denominator
        or robust_spread_twice > 16
    )
    return (
        {
            "unit_influence": _ratio_basis_points(
                top_two_influence, total_influence
            ),
            "bounded_leave_set_out_delta": (
                1_000 * gap_numerator // gap_denominator
            ),
            "robust_fold_stability": robust_spread_twice,
            "rule_minority_margin": (
                4 * top_two_influence - 3 * total_influence
            ),
            "rule_gap_support_margin": (
                gap_numerator - 10 * gap_denominator
            ),
            "rule_gap_diffuse_margin": (
                3 * gap_denominator - gap_numerator
            ),
            "rule_influence_diffuse_margin": (
                11 * total_influence - 20 * top_two_influence
            ),
        },
        (bounded_minority, robust_stable),
        (diffuse, no_improvement),
    )


def _noop_probe(
    payload: NumericWorldPayload,
) -> tuple[dict[str, int], tuple[bool, bool], tuple[bool, bool]]:
    payoff_sums = _column_sums(payload.decision_payoffs)
    ranked_sums = sorted(payoff_sums, reverse=True)
    support_margin = ranked_sums[0] - ranked_sums[1]
    context_best = tuple(
        max(0, max(row)) for row in payload.decision_payoffs
    )
    noop_regret = max(context_best)
    active_regrets = tuple(
        max(
            context_best[context] - payload.decision_payoffs[context][action]
            for context in range(len(payload.decision_payoffs))
        )
        for action in range(DECISION_ACTION_COUNT)
    )
    best_sum = ranked_sums[0]
    indistinguishable = best_sum <= 0 and support_margin <= 5
    conservative = noop_regret + 2 < min(active_regrets)
    decisive = best_sum >= 20 and support_margin >= 12
    noop_harmful = noop_regret > min(active_regrets) + 2
    return (
        {
            "claim_support_margin": support_margin,
            "calibration_error": max(best_sum, 0),
            "noop_regret": noop_regret - min(active_regrets),
        },
        (indistinguishable, conservative),
        (decisive, noop_harmful),
    )


def _probe_statistics(
    *,
    template_id: str,
    payload: NumericWorldPayload,
    contextual_statistics: Mapping[str, int] | None = None,
) -> tuple[dict[str, int], tuple[bool, bool], tuple[bool, bool]]:
    rules = {
        TEMPLATE_ID_BY_FAMILY["sparse"]: _sparse_probe,
        TEMPLATE_ID_BY_FAMILY["set_interaction"]: _set_interaction_probe,
        TEMPLATE_ID_BY_FAMILY["local"]: _local_probe,
        TEMPLATE_ID_BY_FAMILY["contamination"]: _contamination_probe,
        TEMPLATE_ID_BY_FAMILY["no_op"]: _noop_probe,
    }
    try:
        statistics, measured_support, measured_counter = rules[
            template_id
        ](payload)
    except KeyError as exc:
        raise SyntheticQualificationError(
            "no frozen numeric probe rule for template"
        ) from exc
    statistics = dict(statistics)
    if contextual_statistics:
        if set(statistics).intersection(contextual_statistics):
            raise SyntheticQualificationError(
                "contextual probe statistics overwrite measurements"
            )
        statistics.update(contextual_statistics)
    if template_id == TEMPLATE_ID_BY_FAMILY["no_op"]:
        statistics.setdefault(
            "rule_active_claim_falsified_count",
            -1,
        )
    recomputed_support, recomputed_counter = (
        _signature_flags_from_statistics(
            template_id=template_id,
            statistics=statistics,
        )
    )
    if (
        template_id != TEMPLATE_ID_BY_FAMILY["no_op"]
        and (
            measured_support != recomputed_support
            or measured_counter != recomputed_counter
        )
    ):
        raise SyntheticQualificationError(
            "numeric measurement and registered rule disagree"
        )
    return statistics, recomputed_support, recomputed_counter


def _signature_flags_from_statistics(
    *,
    template_id: str,
    statistics: Mapping[str, int],
) -> tuple[tuple[bool, bool], tuple[bool, bool]]:
    """Apply the frozen integer rule to committed measurements.

    Every ratio-sensitive decision uses an already committed integer
    cross-product margin.  Rounded basis-point values are descriptive only.
    """

    def value(key: str) -> int:
        observed = statistics.get(key)
        if type(observed) is not int:
            raise PermissionError(
                f"probe statistic {key!r} is missing or noninteger"
            )
        return observed

    if template_id == TEMPLATE_ID_BY_FAMILY["sparse"]:
        active_size = value("active_set_size")
        capture_margin = value("rule_capture_margin")
        positive_total = value("rule_positive_total")
        stable_support_margin = value(
            "rule_stability_support_margin"
        )
        stable_counter_margin = value(
            "rule_stability_counter_margin"
        )
        small = (
            0 < active_size <= 3
            and capture_margin >= 0
            and positive_total >= 80
        )
        return (
            (small, small and stable_support_margin >= 0),
            (active_size >= 4, stable_counter_margin <= 0),
        )
    if template_id == TEMPLATE_ID_BY_FAMILY["set_interaction"]:
        unary = value("unary_effect")
        pair = value("pair_effect")
        residual = value("higher_order_residual")
        held_error = value("held_combination_error")
        explained = unary <= 1 and pair >= 10 and residual <= 1
        return (
            (explained, explained and held_error <= 3),
            (residual >= 8, pair < 2 or held_error >= 8),
        )
    if template_id == TEMPLATE_ID_BY_FAMILY["local"]:
        decays = value("rule_decay_margin") >= 0
        sufficient = value("rule_sufficiency_margin") >= 0
        distant_equal = value("rule_distant_equal_margin") >= 0
        omitted_parent = value("rule_omitted_parent_margin") >= 0
        return (
            (decays, sufficient),
            (distant_equal, omitted_parent),
        )
    if template_id == TEMPLATE_ID_BY_FAMILY["contamination"]:
        minority = value("rule_minority_margin") >= 0
        large_gap = value("rule_gap_support_margin") >= 0
        diffuse = (
            value("rule_gap_diffuse_margin") >= 0
            or value("rule_influence_diffuse_margin") > 0
        )
        no_improvement = (
            value("rule_gap_diffuse_margin") >= 0
            or value("robust_fold_stability") > 16
        )
        return (
            (
                minority and large_gap,
                large_gap and value("robust_fold_stability") <= 6,
            ),
            (diffuse, no_improvement),
        )
    if template_id == TEMPLATE_ID_BY_FAMILY["no_op"]:
        best_sum = value("calibration_error")
        support_margin = value("claim_support_margin")
        noop_regret = value("noop_regret")
        active_falsified = value(
            "rule_active_claim_falsified_count"
        )
        all_active_falsified = active_falsified == len(ACTIVE_OPERATORS)
        return (
            (
                (
                    all_active_falsified
                    and best_sum <= 0
                    and support_margin <= 5
                ),
                all_active_falsified and noop_regret <= -3,
            ),
            (
                best_sum >= 20 and support_margin >= 12,
                noop_regret > 2,
            ),
        )
    raise PermissionError("no frozen numeric signature rule for template")


PROBE_RULE_SPEC_BY_TEMPLATE: Mapping[str, Mapping[str, object]] = {
    TEMPLATE_ID_BY_FAMILY["sparse"]: {
        "statistic_ids": (
            "active_set_size",
            "captured_utility_fraction",
            "fold_selection_frequency",
            "rule_capture_margin",
            "rule_positive_total",
            "rule_stability_counter_margin",
            "rule_stability_support_margin",
        ),
        "support": (
            "0<active_set_size<=3 and rule_capture_margin>=0 "
            "and rule_positive_total>=80",
            "support_0 and rule_stability_support_margin>=0",
        ),
        "counter": (
            "active_set_size>=4",
            "rule_stability_counter_margin<=0",
        ),
    },
    TEMPLATE_ID_BY_FAMILY["set_interaction"]: {
        "statistic_ids": (
            "held_combination_error",
            "higher_order_residual",
            "pair_effect",
            "unary_effect",
        ),
        "support": (
            "unary_effect<=1 and pair_effect>=10 "
            "and higher_order_residual<=1",
            "support_0 and held_combination_error<=3",
        ),
        "counter": (
            "higher_order_residual>=8",
            "pair_effect<2 or held_combination_error>=8",
        ),
    },
    TEMPLATE_ID_BY_FAMILY["local"]: {
        "statistic_ids": (
            "ablated_effect",
            "conditional_increment",
            "distance_bin",
            "rule_decay_margin",
            "rule_distant_equal_margin",
            "rule_omitted_parent_margin",
            "rule_sufficiency_margin",
        ),
        "support": (
            "rule_decay_margin>=0",
            "rule_sufficiency_margin>=0",
        ),
        "counter": (
            "rule_distant_equal_margin>=0",
            "rule_omitted_parent_margin>=0",
        ),
    },
    TEMPLATE_ID_BY_FAMILY["contamination"]: {
        "statistic_ids": (
            "bounded_leave_set_out_delta",
            "robust_fold_stability",
            "rule_gap_diffuse_margin",
            "rule_gap_support_margin",
            "rule_influence_diffuse_margin",
            "rule_minority_margin",
            "unit_influence",
        ),
        "support": (
            "rule_minority_margin>=0 and rule_gap_support_margin>=0",
            "rule_gap_support_margin>=0 and robust_fold_stability<=6",
        ),
        "counter": (
            "rule_gap_diffuse_margin>=0 "
            "or rule_influence_diffuse_margin>0",
            "rule_gap_diffuse_margin>=0 "
            "or robust_fold_stability>16",
        ),
    },
    TEMPLATE_ID_BY_FAMILY["no_op"]: {
        "statistic_ids": (
            "calibration_error",
            "claim_support_margin",
            "noop_regret",
            "rule_active_claim_falsified_count",
        ),
        "support": (
            "active_claim_falsified_count==4 and calibration_error<=0 "
            "and claim_support_margin<=5",
            "active_claim_falsified_count==4 and noop_regret<=-3",
        ),
        "counter": (
            "calibration_error>=20 and claim_support_margin>=12",
            "noop_regret>2",
        ),
    },
}


def _probe_verifier_implementation_hash(template_id: str) -> str:
    try:
        rule_spec = PROBE_RULE_SPEC_BY_TEMPLATE[template_id]
    except KeyError as exc:
        raise SyntheticQualificationError(
            "no implementation identity for numeric probe verifier"
        ) from exc
    return stable_hash(
        {
            "verifier_version": PROBE_VERIFIER_VERSION,
            "template_id": template_id,
            "rule_spec": rule_spec,
            "arithmetic": "strict_integer_with_cross_product_margins",
            "receipt_policy": (
                "one_first_matching_signature_or_inconclusive"
            ),
        }
    )


# These harness-owned identities are deliberately distinct from verifier
# objects.  The literals are filled from the reviewed frozen rule spec.
EXPECTED_PROBE_IMPLEMENTATION_HASH_BY_TEMPLATE: Mapping[str, str] = {
    "uao.v1.t02_sparsity": (
        "cfb26d2121ac2960ecf1f04fe74257d7ef764eaff53446cd30adbdec16d84a60"
    ),
    "uao.v1.t05_low_order_interaction": (
        "819ab406c951a44ac2c423f036212e2a6edd40f9f13e3a01479c74d12a70cdeb"
    ),
    "uao.v1.t08_locality_markov_blanket": (
        "d0c72dd0e4c006b8e37c75ad7cd27509f029be41873532ff60cbb2dbe090ebe5"
    ),
    "uao.v1.t18_sparse_contamination": (
        "3fd18c903f81cc5217dd892bf2e8dc2d1c9e743a9bacf343d11ca25d09a66a5e"
    ),
    "uao.v1.t19_minimum_commitment": (
        "b5946e4ed37432badd5ed1eddadc0c60553dae538fbc46f1b92b66be31cf528e"
    ),
}


class _NumericProbeVerifier:
    """Closed verifier deriving signatures only from committed statistics."""

    def __init__(
        self,
        *,
        template_id: str,
        plan: DiagnosticProbePlan,
    ) -> None:
        self.template_id = template_id
        self.verifier_id = (
            "synthetic.numeric_probe_verifier."
            + template_id.rsplit(".", 1)[-1]
        )
        self.verifier_version = PROBE_VERIFIER_VERSION
        self.implementation_hash = _probe_verifier_implementation_hash(
            template_id
        )
        self.probe_id = plan.probe_id
        self.support_rule_id = plan.support_rule_id
        self.counter_rule_id = plan.counter_rule_id

    def match_signatures(
        self,
        *,
        template: object,
        claim: HypothesisClaim,
        evidence: ProbeEvidenceBundle,
    ) -> ProbeVerificationResult:
        del claim
        if (
            evidence.template_id != self.template_id
            or getattr(template, "template_id", None) != self.template_id
            or len(evidence.observation_statistics) != 1
        ):
            raise PermissionError(
                "numeric probe verifier received the wrong evidence scope"
            )
        row = evidence.observation_statistics[0]
        raw_values = dict(row.statistic_values)
        expected_ids = tuple(
            PROBE_RULE_SPEC_BY_TEMPLATE[self.template_id]["statistic_ids"]
        )
        if tuple(sorted(raw_values)) != tuple(sorted(expected_ids)):
            raise PermissionError(
                "numeric probe evidence statistic registry drifted"
            )
        statistics: dict[str, int] = {}
        for key, raw_value in raw_values.items():
            try:
                parsed = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise PermissionError(
                    "numeric probe evidence contains a noninteger statistic"
                ) from exc
            if str(parsed) != raw_value:
                raise PermissionError(
                    "numeric probe evidence integer is not canonical"
                )
            statistics[key] = parsed
        support_flags, counter_flags = _signature_flags_from_statistics(
            template_id=self.template_id,
            statistics=statistics,
        )
        matched_support = tuple(
            signature
            for signature, matched in zip(
                getattr(template, "support_signatures"),
                support_flags,
            )
            if matched
        )
        matched_counter = tuple(
            signature
            for signature, matched in zip(
                getattr(template, "counter_signatures"),
                counter_flags,
            )
            if matched
        )
        if matched_support and matched_counter:
            raise PermissionError(
                "numeric probe produced contradictory signature evidence"
            )
        return ProbeVerificationResult(
            observed_support_signature_ids=matched_support[:1],
            observed_counter_signature_ids=matched_counter[:1],
        )


def _build_probe_verifier_registry(
    ontology: UniversalAssumptionOntology,
) -> ProbeVerifierRegistry:
    registry = ProbeVerifierRegistry()
    for template_id in TEMPLATE_ID_BY_FAMILY.values():
        template = ontology.require_template(template_id)
        verifier = _NumericProbeVerifier(
            template_id=template_id,
            plan=template.probe_plan,
        )
        expected_hash = (
            EXPECTED_PROBE_IMPLEMENTATION_HASH_BY_TEMPLATE[template_id]
        )
        if verifier.implementation_hash != expected_hash:
            raise SyntheticQualificationError(
                "numeric probe verifier implementation identity drifted"
            )
        registry.register(
            verifier,
            trust_anchor=ProbeTrustAnchor(
                verifier_id=verifier.verifier_id,
                verifier_version=verifier.verifier_version,
                implementation_hash=expected_hash,
                probe_id=template.probe_plan.probe_id,
                support_rule_id=template.probe_plan.support_rule_id,
                counter_rule_id=template.probe_plan.counter_rule_id,
            ),
        )
    return registry


def _observe_claim(
    *,
    ontology: UniversalAssumptionOntology,
    probe_verifier_registry: ProbeVerifierRegistry,
    selector_input: NumericProbeBundle,
    claim: HypothesisClaim,
    plan: DiagnosticProbePlan,
    contextual_statistics: Mapping[str, int] | None = None,
) -> ProbeComputation:
    template = ontology.require_template(claim.template_ids[0])
    statistics, support_flags, counter_flags = _probe_statistics(
        template_id=template.template_id,
        payload=selector_input.numeric_payload,
        contextual_statistics=contextual_statistics,
    )
    if not set(plan.observable_ids).issubset(statistics):
        raise SyntheticQualificationError(
            "numeric probe statistic registry drifted from ontology plan"
        )
    if any(support_flags) and any(counter_flags):
        raise SyntheticQualificationError(
            "numeric probe produced contradictory support and counter signatures"
        )
    statistic_values = tuple(sorted(statistics.items()))
    observation = ProbeObservationStatistic(
        observation_hash=stable_hash(
            {
                "world_id": selector_input.world_id,
                "numeric_payload_hash": semantic_hash(
                    selector_input.numeric_payload.safe_payload()
                ),
                "claim_hash": claim.claim_hash,
                "probe_plan_hash": plan.plan_hash,
                "measurement_version": (
                    f"{VERSION}.integer_measurements.v3"
                ),
            }
        ),
        statistic_values=tuple(
            (key, str(value)) for key, value in statistic_values
        ),
    )
    observation_issues = observation.validate()
    if observation_issues:
        raise SyntheticQualificationError(
            "numeric probe observation is invalid: "
            f"{list(observation_issues)}"
        )
    evidence = ProbeEvidenceBundle(
        bundle_id="probe-evidence.pending",
        ontology_hash=ontology.ontology_hash,
        claim_hash=claim.claim_hash,
        template_id=claim.template_ids[0],
        probe_plan_hash=plan.plan_hash,
        train_split_hash=claim.evidence_receipt_hashes[0],
        observation_statistics=(observation,),
        formation_split=SplitName.TRAIN,
        source_payload_access_count=0,
        validation_or_test_access_count=0,
        online_or_api_evaluation_count=0,
    )
    evidence = replace(evidence, bundle_id=evidence.expected_bundle_id)
    evidence_issues = evidence.validate(
        ontology=ontology,
        claim=claim,
    )
    if evidence_issues:
        raise SyntheticQualificationError(
            "numeric probe evidence bundle is invalid: "
            f"{list(evidence_issues)}"
        )
    receipt = probe_verifier_registry.issue_receipt(
        ontology=ontology,
        claim=claim,
        evidence=evidence,
    )
    return ProbeComputation(
        receipt=receipt,
        evidence_bundle=evidence,
        statistic_values=statistic_values,
        statistic_hash=observation.statistic_commitment_hash,
    )


def _select_claim(
    *,
    ontology: UniversalAssumptionOntology,
    probe_verifier_registry: ProbeVerifierRegistry,
    selector_input: NumericProbeBundle,
    ordered_claims: Sequence[HypothesisClaim],
    plans_by_claim_hash: Mapping[str, DiagnosticProbePlan],
    probe_template_order: Sequence[str] | None = None,
) -> tuple[HypothesisClaim, tuple[ProbeComputation, ...]]:
    """Select from numeric probes without access to the expected family."""

    active_claims = tuple(
        claim
        for claim in ordered_claims
        if claim.template_ids
        != (TEMPLATE_ID_BY_FAMILY["no_op"],)
    )
    no_op_claims = tuple(
        claim
        for claim in ordered_claims
        if claim.template_ids
        == (TEMPLATE_ID_BY_FAMILY["no_op"],)
    )
    if len(active_claims) != len(ACTIVE_OPERATORS) or len(
        no_op_claims
    ) != 1:
        raise SyntheticQualificationError(
            "two-stage selector claim registry is incomplete"
        )
    if probe_template_order is not None:
        requested_order = tuple(probe_template_order)
        active_by_template = {
            claim.template_ids[0]: claim for claim in active_claims
        }
        if (
            len(requested_order) != len(active_claims)
            or len(set(requested_order)) != len(requested_order)
            or set(requested_order) != set(active_by_template)
        ):
            raise SyntheticQualificationError(
                "probe template order is not a closed permutation"
            )
        active_claims = tuple(
            active_by_template[template_id]
            for template_id in requested_order
        )
    active_computations = tuple(
        _observe_claim(
            ontology=ontology,
            probe_verifier_registry=probe_verifier_registry,
            selector_input=selector_input,
            claim=claim,
            plan=plans_by_claim_hash[claim.claim_hash],
        )
        for claim in active_claims
    )
    active_falsified_count = sum(
        computation.receipt.disposition
        is ProbeDisposition.FALSIFIED
        for computation in active_computations
    )
    no_op_computation = _observe_claim(
        ontology=ontology,
        probe_verifier_registry=probe_verifier_registry,
        selector_input=selector_input,
        claim=no_op_claims[0],
        plan=plans_by_claim_hash[no_op_claims[0].claim_hash],
        contextual_statistics={
            "rule_active_claim_falsified_count": (
                active_falsified_count
            )
        },
    )
    computation_by_claim_hash = {
        computation.receipt.claim_hash: computation
        for computation in (*active_computations, no_op_computation)
    }
    computations = tuple(
        computation_by_claim_hash[claim.claim_hash]
        for claim in ordered_claims
    )
    eligible = [
        (claim, computation.receipt)
        for claim, computation in zip(ordered_claims, computations)
        if computation.receipt.disposition is ProbeDisposition.SUPPORTED
        and bool(computation.receipt.observed_support_signature_ids)
        and not computation.receipt.observed_counter_signature_ids
    ]
    if len(eligible) != 1:
        raise SyntheticQualificationError(
            "selector did not find one uniquely supported claim"
        )
    for claim, computation in zip(ordered_claims, computations):
        if claim.claim_hash == eligible[0][0].claim_hash:
            continue
        receipt = computation.receipt
        if (
            receipt.disposition is not ProbeDisposition.FALSIFIED
            or receipt.observed_support_signature_ids
            or not receipt.observed_counter_signature_ids
        ):
            raise SyntheticQualificationError(
                "wrong claim lacked computed counter-signature evidence"
            )
    return eligible[0][0], computations


def _treatment_compiler_hash() -> str:
    return stable_hash(
        {
            "compiler_id": TREATMENT_COMPILER_ID,
            "operator_by_template": {
                TEMPLATE_ID_BY_FAMILY[family]: OPERATOR_BY_FAMILY[family]
                for family in FAMILIES
            },
            "no_op_contract": "program_none_preserve_baseline",
            "active_action_contract": (
                "enable_lane_then_bind_closed_recipe_then_prioritize_lane"
            ),
            "operator_lane": OPERATOR_LANE,
            "recipe_parameter": RECIPE_PARAMETER,
            "version": VERSION,
        }
    )


EXPECTED_COMPILER_IMPLEMENTATION_HASH = (
    "ce1bd610960961985a99afc207c5ad2dfbedbd130b290902627a5176dab9a5f6"
)


def _compiler_trust_anchor() -> CompilerTrustAnchor:
    """Return the harness-owned identity, separate from a compiler instance."""

    return CompilerTrustAnchor(
        compiler_id=TREATMENT_COMPILER_ID,
        compiler_version=TREATMENT_COMPILER_VERSION,
        implementation_hash=EXPECTED_COMPILER_IMPLEMENTATION_HASH,
        primary_metric=PRIMARY_METRIC,
    )


def _active_program(
    *,
    claim: HypothesisClaim,
    operator: str,
    probe_receipt_hash: str,
) -> HypothesisProgram:
    return HypothesisProgram(
        id=f"synthetic.program.{operator.lower()}.{claim.claim_hash[:12]}",
        kind=HypothesisKind.POLICY,
        statement=(
            "Synthetic-only lowering of a frozen meta-assumption claim."
        ),
        trigger=TriggerSpec(
            all_of=(
                FeaturePredicate(
                    key="synthetic_scope_hash",
                    op="eq",
                    value=claim.scope_hash,
                ),
            )
        ),
        anti_trigger=TriggerSpec(),
        action_graph=(
            ActionNode(
                id="enable_synthetic_operator_lane",
                operation="enable_lane",
                target=OPERATOR_LANE,
            ),
            ActionNode(
                id="bind_closed_synthetic_recipe",
                operation="set_parameter",
                target=RECIPE_PARAMETER,
                value=operator,
                depends_on=("enable_synthetic_operator_lane",),
            ),
            ActionNode(
                id="prioritize_synthetic_operator_lane",
                operation="prioritize_lane",
                target=OPERATOR_LANE,
                value=100,
                depends_on=("enable_synthetic_operator_lane",),
            ),
        ),
        expected_effect=ExpectedEffect(
            metric=PRIMARY_METRIC,
            minimum_delta=0.0,
            maximum_harm_rate=0.0,
            maximum_cost_ratio=1.0,
        ),
        verifier=VerifierContract(
            checks=("verify_compilation_binding",),
            required_evidence=(probe_receipt_hash,),
            anchor_id=claim.claim_hash,
            repair_on_failure=False,
            max_repair_depth=0,
        ),
        evaluator_epoch=EVALUATOR_EPOCH,
        fallback="preserve_baseline",
    )


class SyntheticClosedTreatmentCompiler:
    """Harness-owned lowering for the five qualification claim templates."""

    compiler_id = TREATMENT_COMPILER_ID
    compiler_version = TREATMENT_COMPILER_VERSION
    primary_metric = PRIMARY_METRIC
    implementation_hash = _treatment_compiler_hash()

    def compile(
        self,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        probes: Sequence[ProbeReceipt],
    ) -> CompiledTreatment:
        del ontology
        if len(probes) != 1:
            raise PermissionError(
                "synthetic compiler requires one bound probe receipt"
            )
        probe = probes[0]
        if probe.claim_hash != claim.claim_hash or probe.falsified:
            raise PermissionError(
                "synthetic compiler claim/probe binding failed"
            )
        family_by_template = {
            template_id: family
            for family, template_id in TEMPLATE_ID_BY_FAMILY.items()
        }
        if len(claim.template_ids) != 1:
            raise PermissionError(
                "synthetic compiler requires one mechanism template"
            )
        family = family_by_template.get(claim.template_ids[0])
        if family is None:
            raise PermissionError(
                "synthetic compiler received an unknown template"
            )
        operator = OPERATOR_BY_FAMILY[family]
        if family == "no_op":
            return CompiledTreatment(
                disposition=TreatmentDisposition.PRESERVE_BASELINE,
                program=None,
                recipe_ids=(),
                evaluator_artifact_hash="",
                recipe_action_bindings=(),
            )
        program = _active_program(
            claim=claim,
            operator=operator,
            probe_receipt_hash=probe.receipt_hash,
        )
        issues = program.validate()
        if issues:
            raise PermissionError(
                f"synthetic active program is invalid: {issues}"
            )
        recipe_action = next(
            action
            for action in program.action_graph
            if action.id == "bind_closed_synthetic_recipe"
        )
        return CompiledTreatment(
            disposition=TreatmentDisposition.ACTIVE_PROGRAM,
            program=program,
            recipe_ids=(operator,),
            evaluator_artifact_hash="",
            recipe_action_bindings=(
                RecipeActionBinding(
                    recipe_id=operator,
                    action_id=recipe_action.id,
                    action_semantics_hash=action_node_semantics_hash(
                        recipe_action
                    ),
                ),
            ),
        )


def _operator_from_treatment(treatment: CompiledTreatment) -> str:
    if treatment.program is None:
        return "PRESERVE_BASELINE"
    if len(treatment.recipe_ids) != 1:
        raise SyntheticQualificationError(
            "synthetic treatment recipe binding drifted"
        )
    if len(treatment.program.action_graph) != 3:
        raise SyntheticQualificationError(
            "synthetic treatment action graph drifted"
        )
    actions = {
        action.id: action for action in treatment.program.action_graph
    }
    if (
        actions["enable_synthetic_operator_lane"].operation != "enable_lane"
        or actions["enable_synthetic_operator_lane"].target != OPERATOR_LANE
        or actions["prioritize_synthetic_operator_lane"].operation
        != "prioritize_lane"
        or actions["prioritize_synthetic_operator_lane"].target
        != OPERATOR_LANE
        or actions["prioritize_synthetic_operator_lane"].value != 100
    ):
        raise SyntheticQualificationError(
            "synthetic runtime lane actions drifted"
        )
    recipe_action = actions["bind_closed_synthetic_recipe"]
    value = recipe_action.value
    if not isinstance(value, str):
        raise SyntheticQualificationError(
            "synthetic treatment operator is not closed"
        )
    if treatment.recipe_ids[0] != value:
        raise SyntheticQualificationError(
            "synthetic recipe/program operator mismatch"
        )
    if (
        len(treatment.recipe_action_bindings) != 1
        or treatment.recipe_action_bindings[0].recipe_id != value
        or treatment.recipe_action_bindings[0].action_id
        != recipe_action.id
        or treatment.recipe_action_bindings[0].action_semantics_hash
        != action_node_semantics_hash(recipe_action)
    ):
        raise SyntheticQualificationError(
            "synthetic recipe/action semantics binding drifted"
        )
    return value


def _baseline_decision() -> dict[str, object]:
    return {
        "recipe": "PRESERVE_BASELINE",
        "decision_kind": "no_op",
        "indices": [],
        "estimate_twice": 0,
    }


def _closed_recipe_decision(
    payload: NumericWorldPayload, recipe: str
) -> dict[str, object]:
    if recipe == OPERATOR_BY_FAMILY["sparse"]:
        column_sums = _column_sums(payload.action_fold_utilities)
        indices = [
            index for index, value in enumerate(column_sums) if value > 40
        ]
        return {
            "recipe": recipe,
            "decision_kind": "action_subset",
            "indices": indices,
            "estimate_twice": 0,
        }
    if recipe == OPERATOR_BY_FAMILY["set_interaction"]:
        _unary, pairs, _residual = _subset_decomposition(
            payload.subset_utility_folds[0]
        )
        pair = min(pairs, key=lambda value: (-pairs[value], value))
        return {
            "recipe": recipe,
            "decision_kind": "component_pair",
            "indices": list(pair),
            "estimate_twice": 0,
        }
    if recipe == OPERATOR_BY_FAMILY["local"]:
        distances = _graph_distances(payload.adjacency, payload.focal_node)
        return {
            "recipe": recipe,
            "decision_kind": "graph_neighborhood",
            "indices": [
                index
                for index, distance in enumerate(distances)
                if distance <= 1
            ],
            "estimate_twice": 0,
        }
    if recipe == OPERATOR_BY_FAMILY["contamination"]:
        estimate_twice = _median_twice(
            tuple(
                value for row in payload.observation_folds for value in row
            )
        )
        return {
            "recipe": recipe,
            "decision_kind": "robust_estimate",
            "indices": [],
            "estimate_twice": estimate_twice,
        }
    raise SyntheticQualificationError("operator lane received unknown recipe")


@dataclass(frozen=True)
class _SyntheticBaselineLane:
    name: str = BASELINE_LANE

    def run(
        self, task: TaskInput, parameters: Mapping[str, object]
    ) -> LaneResult:
        del parameters
        if not isinstance(task.payload, NumericWorldPayload):
            raise SyntheticQualificationError(
                "baseline lane received nonnumeric payload"
            )
        return LaneResult(
            lane=self.name,
            answer=_baseline_decision(),
            confidence=0.75,
            cost=0.0,
            metadata={
                "numeric_payload_hash": semantic_hash(
                    task.payload.safe_payload()
                ),
                "recipe_consumed": "PRESERVE_BASELINE",
            },
        )


@dataclass(frozen=True)
class _SyntheticOperatorLane:
    name: str = OPERATOR_LANE

    def run(
        self, task: TaskInput, parameters: Mapping[str, object]
    ) -> LaneResult:
        if not isinstance(task.payload, NumericWorldPayload):
            raise SyntheticQualificationError(
                "operator lane received nonnumeric payload"
            )
        recipe = parameters.get(RECIPE_PARAMETER)
        if not isinstance(recipe, str) or recipe not in ACTIVE_OPERATORS:
            raise SyntheticQualificationError(
                "operator lane did not receive one closed recipe"
            )
        return LaneResult(
            lane=self.name,
            answer=_closed_recipe_decision(task.payload, recipe),
            confidence=0.99,
            cost=0.0,
            metadata={
                "numeric_payload_hash": semantic_hash(
                    task.payload.safe_payload()
                ),
                "recipe_consumed": recipe,
            },
        )


def _synthetic_runtime() -> PolicyRuntime:
    return PolicyRuntime(
        registry=LaneRegistry(
            (_SyntheticBaselineLane(), _SyntheticOperatorLane())
        ),
        baseline_lane=BASELINE_LANE,
        runtime_version=f"{VERSION}.policy_runtime.v2",
    )


def _execution_semantic_payload(execution: object) -> dict[str, object]:
    selected = execution.selected_result
    return {
        "task_id": execution.task_id,
        "selected_lane": selected.lane,
        "selected_answer": selected.answer,
        "selected_confidence": selected.confidence,
        "lane_results": [
            {
                "lane": result.lane,
                "answer": result.answer,
                "confidence": result.confidence,
                "cost": result.cost,
                "metadata": dict(result.metadata),
            }
            for result in execution.lane_results
        ],
        "activated_hypothesis_ids": list(
            execution.activated_hypothesis_ids
        ),
        "plan_hash": execution.plan_hash,
        "action_activated": execution.action_activated,
        "baseline_preserved": execution.baseline_preserved,
    }


def _oracle_utility(
    world: SyntheticWorld, answer: object
) -> int:
    """Score every typed decision from the same numeric payload.

    No expected-family label participates.  Each closed recipe has a numeric
    utility functional over its corresponding panel, including a fixed
    intervention cost.  Non-target panels are genuine counterexamples, so a
    wrong recipe is harmful by computation rather than by label lookup.
    """

    if not isinstance(answer, Mapping):
        return -100
    recipe = answer.get("recipe")
    if recipe == "PRESERVE_BASELINE":
        return 0
    payload = world.selector_input.numeric_payload
    indices_value = answer.get("indices")
    indices = (
        tuple(int(index) for index in indices_value)
        if isinstance(indices_value, list)
        else ()
    )
    if recipe == OPERATOR_BY_FAMILY["sparse"]:
        return (
            sum(
                row[index]
                for row in payload.action_fold_utilities
                for index in indices
            )
            - 8 * len(indices)
            - 1
        )
    if recipe == OPERATOR_BY_FAMILY["set_interaction"]:
        mask = sum(1 << index for index in indices)
        return sum(
            fold[mask] for fold in payload.subset_utility_folds
        ) - 1
    if recipe == OPERATOR_BY_FAMILY["local"]:
        return (
            sum(
                row[index]
                for row in payload.node_effect_folds
                for index in indices
            )
            - 4 * len(indices)
            - 1
        )
    if recipe == OPERATOR_BY_FAMILY["contamination"]:
        estimate_twice = answer.get("estimate_twice")
        if type(estimate_twice) is not int:
            return -100
        flattened = tuple(
            value
            for row in payload.observation_folds
            for value in row
        )
        center_fourfold = _median_twice(
            tuple(
                _median_twice(row)
                for row in payload.observation_folds
            )
        )
        count = len(flattened)
        mean_deviation_numerator = abs(
            4 * sum(flattened) - count * center_fourfold
        )
        estimate_deviation_numerator = count * abs(
            2 * estimate_twice - center_fourfold
        )
        return (
            mean_deviation_numerator
            - estimate_deviation_numerator
            - count
        )
    return -100


def _runtime_evidence(
    *,
    world: SyntheticWorld,
    claim: HypothesisClaim,
    selected_probe: ProbeReceipt,
    treatment: CompiledTreatment,
) -> RuntimeEvidence:
    runtime = _synthetic_runtime()
    task = TaskInput(
        id=f"runtime.{world.selector_input.world_id}",
        family="synthetic_numeric_mechanism",
        features={"synthetic_scope_hash": claim.scope_hash},
        payload=world.selector_input.numeric_payload,
    )
    allowed = {HypothesisStatus.CANDIDATE}
    baseline = runtime.execute(
        task, (), allowed_statuses=allowed, trace_id="baseline"
    )
    programs = (
        (treatment.program,) if treatment.program is not None else ()
    )
    candidate = runtime.execute(
        task, programs, allowed_statuses=allowed, trace_id="candidate"
    )
    baseline_semantic = _execution_semantic_payload(baseline)
    candidate_semantic = _execution_semantic_payload(candidate)
    baseline_utility = _oracle_utility(
        world, baseline.selected_result.answer
    )
    candidate_utility = _oracle_utility(
        world, candidate.selected_result.answer
    )
    active_differential = (
        treatment.program is not None
        and candidate.action_activated
        and candidate.selected_result.lane == OPERATOR_LANE
        and candidate.selected_result.answer
        != baseline.selected_result.answer
        and candidate.plan_hash != baseline.plan_hash
        and candidate_utility > baseline_utility
    )
    noop_equivalent = (
        treatment.program is None
        and candidate.plan_hash == baseline.plan_hash
        and canonical_bytes(candidate_semantic)
        == canonical_bytes(baseline_semantic)
    )
    correct_operator = _operator_from_treatment(treatment)
    wrong_operators = tuple(
        operator
        for operator in ACTIVE_OPERATORS
        if operator != correct_operator
    )
    wrong_harms = 0
    for operator in wrong_operators:
        wrong_program = _active_program(
            claim=claim,
            operator=operator,
            probe_receipt_hash=selected_probe.receipt_hash,
        )
        wrong_execution = runtime.execute(
            task,
            (wrong_program,),
            allowed_statuses=allowed,
            trace_id=f"wrong.{operator.lower()}",
        )
        wrong_utility = _oracle_utility(
            world, wrong_execution.selected_result.answer
        )
        if wrong_utility < baseline_utility:
            wrong_harms += 1
    return RuntimeEvidence(
        baseline_plan_hash=baseline.plan_hash,
        candidate_plan_hash=candidate.plan_hash,
        baseline_utility=baseline_utility,
        candidate_utility=candidate_utility,
        selected_lane=candidate.selected_result.lane,
        active_runtime_differential=active_differential,
        noop_runtime_equivalent=noop_equivalent,
        wrong_operator_trial_count=len(wrong_operators),
        wrong_operator_harm_count=wrong_harms,
        semantic_commitment=stable_hash(
            {
                "baseline": baseline_semantic,
                "candidate": candidate_semantic,
                "wrong_operator_trial_count": len(wrong_operators),
                "wrong_operator_harm_count": wrong_harms,
            }
        ),
    )


def _selection_policy_hash() -> str:
    return stable_hash(
        {
            "selection_policy_id": SELECTION_POLICY_ID,
            "stages": (
                "probe_four_active_claims_then_condition_minimum_commitment"
            ),
            "support": "computed_signature_nonempty",
            "counter": "computed_signature_empty",
            "falsified": False,
            "minimum_commitment_condition": (
                "all_four_active_claims_falsified"
            ),
            "tie_break": "forbidden_unique_support_required",
        }
    )


def _binding_for(
    receipt: CompilationReceipt,
    *,
    ontology: UniversalAssumptionOntology,
    claim: HypothesisClaim,
    probes: tuple[ProbeReceipt, ...],
    probe_evidence_bundles: tuple[ProbeEvidenceBundle, ...],
    probe_verifier_registry: ProbeVerifierRegistry,
    treatment: CompiledTreatment,
) -> CompilationBinding:
    trust_anchor = _compiler_trust_anchor()
    expected_target = (
        CompilerTarget.NO_DIRECT_TREATMENT
        if treatment.program is None
        else CompilerTarget.POLICY_PROGRAM
    )
    return CompilationBinding(
        ontology=ontology,
        claim=claim,
        probes=probes,
        probe_evidence_bundles=probe_evidence_bundles,
        probe_verifier_registry=probe_verifier_registry,
        treatment=treatment,
        receipt_id=receipt.expected_receipt_id,
        ontology_hash=ontology.ontology_hash,
        template_hashes=tuple(
            sorted(
                ontology.require_template(template_id).template_hash
                for template_id in claim.template_ids
            )
        ),
        claim_hash=claim.claim_hash,
        probe_receipt_hashes=tuple(
            sorted(probe.receipt_hash for probe in probes)
        ),
        compiler_id=trust_anchor.compiler_id,
        compiler_version=trust_anchor.compiler_version,
        compiler_implementation_hash=trust_anchor.implementation_hash,
        primary_metric=trust_anchor.primary_metric,
        compiler_trust_anchor_hash=trust_anchor.anchor_hash,
        compiler_target=expected_target,
        treatment_disposition=treatment.disposition,
        recipe_ids=treatment.recipe_ids,
        recipe_action_binding_hashes=(
            treatment.recipe_action_binding_hashes
        ),
        treatment_behavior_hash=treatment.treatment_behavior_hash,
        selection_policy_hash=_selection_policy_hash(),
        compiler_trust_anchor=trust_anchor,
    )


def validate_compilation_binding(
    receipt: CompilationReceipt,
    binding: CompilationBinding,
    *,
    ontology_override: UniversalAssumptionOntology | None = None,
    claim_override: HypothesisClaim | None = None,
    probes_override: Sequence[ProbeReceipt] | None = None,
    probe_evidence_bundles_override: (
        Sequence[ProbeEvidenceBundle] | None
    ) = None,
    probe_verifier_registry_override: ProbeVerifierRegistry | None = None,
    treatment_override: CompiledTreatment | None = None,
) -> None:
    """Validate every claim→probe→compiler→treatment edge."""

    ontology = ontology_override or binding.ontology
    claim = claim_override or binding.claim
    probes = (
        tuple(probes_override)
        if probes_override is not None
        else binding.probes
    )
    probe_evidence_bundles = (
        tuple(probe_evidence_bundles_override)
        if probe_evidence_bundles_override is not None
        else binding.probe_evidence_bundles
    )
    probe_verifier_registry = (
        probe_verifier_registry_override
        or binding.probe_verifier_registry
    )
    treatment = treatment_override or binding.treatment
    try:
        verify_compilation_receipt(
            receipt,
            ontology=ontology,
            claim=claim,
            probes=probes,
            probe_evidence_bundles=probe_evidence_bundles,
            probe_verifier_registry=probe_verifier_registry,
            treatment=treatment,
            trust_anchor=binding.compiler_trust_anchor,
        )
    except PermissionError as exc:
        raise SyntheticQualificationError(
            f"compilation receipt is invalid: {exc}"
        ) from exc
    expected_fields = {
        "receipt_id": binding.receipt_id,
        "ontology_hash": binding.ontology_hash,
        "template_hashes": binding.template_hashes,
        "claim_hash": binding.claim_hash,
        "probe_receipt_hashes": binding.probe_receipt_hashes,
        "compiler_id": binding.compiler_id,
        "compiler_version": binding.compiler_version,
        "compiler_implementation_hash": (
            binding.compiler_implementation_hash
        ),
        "primary_metric": binding.primary_metric,
        "compiler_trust_anchor_hash": (
            binding.compiler_trust_anchor_hash
        ),
        "compiler_target": binding.compiler_target,
        "treatment_disposition": binding.treatment_disposition,
        "recipe_ids": binding.recipe_ids,
        "recipe_action_binding_hashes": (
            binding.recipe_action_binding_hashes
        ),
        "treatment_behavior_hash": binding.treatment_behavior_hash,
    }
    for field, expected in expected_fields.items():
        if getattr(receipt, field) != expected:
            raise SyntheticQualificationError(
                f"compilation_binding_{field}_mismatch"
            )
    if _selection_policy_hash() != binding.selection_policy_hash:
        raise SyntheticQualificationError(
            "compilation_binding_selection_policy_hash_mismatch"
        )
    if treatment.program is None:
        if (
            treatment.disposition
            is not TreatmentDisposition.PRESERVE_BASELINE
        ):
            raise SyntheticQualificationError(
                "no-op treatment disposition drifted"
            )
    else:
        if (
            treatment.disposition
            is not TreatmentDisposition.ACTIVE_PROGRAM
        ):
            raise SyntheticQualificationError(
                "active treatment disposition drifted"
            )
        if treatment.program.expected_effect.metric != receipt.primary_metric:
            raise SyntheticQualificationError(
                "active program metric binding drifted"
            )
        if treatment.program.evaluator_epoch != EVALUATOR_EPOCH:
            raise SyntheticQualificationError(
                "active program evaluator epoch binding drifted"
            )


def build_qualification_artifacts() -> QualificationArtifacts:
    ontology = build_universal_assumption_ontology_v1()
    ontology_issues = ontology.validate()
    if ontology_issues:
        raise SyntheticQualificationError(
            f"ontology is invalid: {list(ontology_issues)}"
        )
    claims_by_family = _build_claims(ontology)
    probe_plans_by_family = _build_probe_plans(
        ontology, claims_by_family
    )
    claims_by_id = {
        claim.claim_id: claim for claim in claims_by_family.values()
    }
    plans_by_claim_hash = {
        claims_by_family[family].claim_hash: plan
        for family, plan in probe_plans_by_family.items()
    }
    probe_verifier_registry = _build_probe_verifier_registry(ontology)
    registry = HypothesisSpaceCompilerRegistry(
        probe_verifier_registry=probe_verifier_registry
    )
    registry.register(
        SyntheticClosedTreatmentCompiler(),
        trust_anchor=_compiler_trust_anchor(),
    )
    qualifications: list[WorldQualification] = []
    for world in _build_worlds():
        ordered_claims = tuple(
            claims_by_id[claim_id] for claim_id in world.claim_order
        )
        selected, probe_computations = _select_claim(
            ontology=ontology,
            probe_verifier_registry=probe_verifier_registry,
            selector_input=world.selector_input,
            ordered_claims=ordered_claims,
            plans_by_claim_hash=plans_by_claim_hash,
        )
        selected_family = next(
            family
            for family, claim in claims_by_family.items()
            if claim.claim_hash == selected.claim_hash
        )
        selected_computation = next(
            computation
            for computation in probe_computations
            if computation.receipt.claim_hash == selected.claim_hash
        )
        selected_probe = selected_computation.receipt
        selected_evidence = selected_computation.evidence_bundle
        treatment, compilation = registry.compile(
            compiler_id=TREATMENT_COMPILER_ID,
            compiler_version=TREATMENT_COMPILER_VERSION,
            ontology=ontology,
            claim=selected,
            probes=(selected_probe,),
            probe_evidence_bundles=(selected_evidence,),
        )
        binding = _binding_for(
            compilation,
            ontology=ontology,
            claim=selected,
            probes=(selected_probe,),
            probe_evidence_bundles=(selected_evidence,),
            probe_verifier_registry=probe_verifier_registry,
            treatment=treatment,
        )
        validate_compilation_binding(compilation, binding)
        runtime_evidence = _runtime_evidence(
            world=world,
            claim=selected,
            selected_probe=selected_probe,
            treatment=treatment,
        )
        qualifications.append(
            WorldQualification(
                world=world,
                selected_family=selected_family,
                selected_claim=selected,
                probe_receipts=tuple(
                    computation.receipt
                    for computation in probe_computations
                ),
                probe_computations=probe_computations,
                treatment=treatment,
                compilation_receipt=compilation,
                binding=binding,
                runtime_evidence=runtime_evidence,
            )
        )
    return QualificationArtifacts(
        ontology=ontology,
        probe_verifier_registry=probe_verifier_registry,
        claims_by_family=claims_by_family,
        probe_plans_by_family=probe_plans_by_family,
        worlds=_build_worlds(),
        world_qualifications=tuple(qualifications),
    )


def _prediction_signatures_distinct(
    claims_by_family: Mapping[str, HypothesisClaim],
) -> bool:
    signatures = {
        (
            claim.observable_predictions,
            claim.counter_predictions,
        )
        for claim in claims_by_family.values()
    }
    return len(signatures) == len(FAMILIES)


def _fixed_tamper_matrix(
    artifacts: QualificationArtifacts,
) -> tuple[TamperOutcome, ...]:
    """Require each tamper to fail for its exact preregistered issue IDs."""

    active = next(
        result
        for result in artifacts.world_qualifications
        if result.selected_family == "sparse"
    )
    no_op = next(
        result
        for result in artifacts.world_qualifications
        if result.selected_family == "no_op"
    )

    def receipt_case(
        case_id: str,
        expected_issue_ids: tuple[str, ...],
        *,
        preserve_bad_receipt_id: bool = False,
        **changes: object,
    ) -> tuple[str, tuple[str, ...], Callable[[], None]]:
        tampered = replace(active.compilation_receipt, **changes)
        if not preserve_bad_receipt_id:
            tampered = replace(
                tampered, receipt_id=tampered.expected_receipt_id
            )
        return (
            case_id,
            expected_issue_ids,
            lambda: validate_compilation_binding(
                tampered,
                active.binding,
            ),
        )

    tampered_probe = replace(
        active.binding.probes[0],
        observation_hashes=("6" * 64,),
    )
    tampered_probe_anchor = replace(
        active.binding.probes[0],
        probe_trust_anchor_hash="9" * 64,
    )
    tampered_probe_anchor = replace(
        tampered_probe_anchor,
        receipt_id=tampered_probe_anchor.expected_receipt_id,
    )
    original_evidence = active.binding.probe_evidence_bundles[0]
    original_observation = original_evidence.observation_statistics[0]

    def rebind_evidence(
        observation: ProbeObservationStatistic,
    ) -> ProbeEvidenceBundle:
        changed = replace(
            original_evidence,
            bundle_id="probe-evidence.pending",
            observation_statistics=(observation,),
        )
        return replace(changed, bundle_id=changed.expected_bundle_id)

    changed_observation_evidence = rebind_evidence(
        replace(
            original_observation,
            observation_hash="a" * 64,
        )
    )
    changed_observation_probe = (
        artifacts.probe_verifier_registry.issue_receipt(
            ontology=artifacts.ontology,
            claim=active.selected_claim,
            evidence=changed_observation_evidence,
        )
    )
    statistic_values = dict(original_observation.statistic_values)
    statistic_values["captured_utility_fraction"] = str(
        int(statistic_values["captured_utility_fraction"]) + 1
    )
    changed_statistic_evidence = rebind_evidence(
        replace(
            original_observation,
            statistic_values=tuple(sorted(statistic_values.items())),
        )
    )
    changed_statistic_probe = (
        artifacts.probe_verifier_registry.issue_receipt(
            ontology=artifacts.ontology,
            claim=active.selected_claim,
            evidence=changed_statistic_evidence,
        )
    )
    no_op_disposition = replace(
        no_op.compilation_receipt,
        treatment_disposition=TreatmentDisposition.ACTIVE_PROGRAM,
    )
    no_op_disposition = replace(
        no_op_disposition,
        receipt_id=no_op_disposition.expected_receipt_id,
    )
    cases = (
        receipt_case(
            "receipt_id",
            ("compilation_receipt_id_binding_mismatch",),
            preserve_bad_receipt_id=True,
            receipt_id="compilation.forged_but_valid",
        ),
        receipt_case(
            "ontology",
            ("compilation_receipt_ontology_mismatch",),
            ontology_hash="0" * 64,
        ),
        receipt_case(
            "template",
            ("compilation_receipt_template_hashes_mismatch",),
            template_hashes=("1" * 64,),
        ),
        receipt_case(
            "claim",
            ("compilation_receipt_claim_mismatch",),
            claim_hash="2" * 64,
        ),
        (
            "probe",
            (
                "compilation_receipt_probe_hashes_mismatch",
                "probe_receipt_id_binding_mismatch",
                "probe_receipt_trusted_recomputation_mismatch",
            ),
            lambda: validate_compilation_binding(
                active.compilation_receipt,
                active.binding,
                probes_override=(tampered_probe,),
            ),
        ),
        (
            "probe_trust_anchor",
            (
                "compilation_receipt_probe_hashes_mismatch",
                "probe_receipt_trusted_recomputation_mismatch",
            ),
            lambda: validate_compilation_binding(
                active.compilation_receipt,
                active.binding,
                probes_override=(tampered_probe_anchor,),
            ),
        ),
        (
            "probe_evidence_bundle",
            ("compilation_receipt_probe_hashes_mismatch",),
            lambda: validate_compilation_binding(
                active.compilation_receipt,
                active.binding,
                probes_override=(changed_observation_probe,),
                probe_evidence_bundles_override=(
                    changed_observation_evidence,
                ),
            ),
        ),
        (
            "probe_statistic_commitment",
            ("compilation_receipt_probe_hashes_mismatch",),
            lambda: validate_compilation_binding(
                active.compilation_receipt,
                active.binding,
                probes_override=(changed_statistic_probe,),
                probe_evidence_bundles_override=(
                    changed_statistic_evidence,
                ),
            ),
        ),
        receipt_case(
            "compiler_id",
            ("compilation_receipt_compiler_id_mismatch",),
            compiler_id="tampered.compiler",
        ),
        receipt_case(
            "compiler_version",
            ("compilation_receipt_compiler_version_mismatch",),
            compiler_version="tampered.compiler.version",
        ),
        receipt_case(
            "compiler_hash",
            ("compilation_receipt_compiler_hash_mismatch",),
            compiler_implementation_hash="3" * 64,
        ),
        receipt_case(
            "trust_anchor_hash",
            ("compilation_receipt_trust_anchor_hash_mismatch",),
            compiler_trust_anchor_hash="7" * 64,
        ),
        receipt_case(
            "primary_metric",
            (
                "compilation_receipt_primary_metric_mismatch",
                "compilation_receipt_trusted_metric_mismatch",
            ),
            primary_metric="tampered_metric",
        ),
        receipt_case(
            "compiler_target",
            ("compilation_receipt_target_mismatch",),
            compiler_target=CompilerTarget.EVALUATOR_ARTIFACT,
        ),
        receipt_case(
            "treatment_disposition",
            ("compilation_receipt_disposition_mismatch",),
            treatment_disposition=(
                TreatmentDisposition.PRESERVE_BASELINE
            ),
        ),
        receipt_case(
            "recipe",
            ("compilation_receipt_recipe_ids_mismatch",),
            recipe_ids=("TAMPERED_RECIPE",),
        ),
        receipt_case(
            "recipe_action_binding",
            (
                "compilation_receipt_recipe_action_bindings_mismatch",
            ),
            recipe_action_binding_hashes=("8" * 64,),
        ),
        receipt_case(
            "behavior",
            ("compilation_receipt_behavior_hash_mismatch",),
            treatment_behavior_hash="4" * 64,
        ),
        (
            "no_op",
            ("compilation_receipt_disposition_mismatch",),
            lambda: validate_compilation_binding(
                no_op_disposition,
                no_op.binding,
            ),
        ),
    )

    def issue_ids(error: BaseException) -> tuple[str, ...]:
        messages: list[str] = []
        cursor: BaseException | None = error
        while cursor is not None:
            messages.append(str(cursor))
            cursor = cursor.__cause__
        return tuple(
            sorted(
                set(
                    re.findall(
                        r"(?:compilation_receipt|probe_receipt|"
                        r"compiled_treatment|compiled_noop|"
                        r"recipe_action_binding)_"
                        r"[a-z0-9_]+",
                        " ".join(messages),
                    )
                )
            )
        )

    outcomes: list[TamperOutcome] = []
    for case_id, expected_issue_ids, exercise in cases:
        try:
            exercise()
        except SyntheticQualificationError as exc:
            observed = issue_ids(exc)
            cause_type = (
                type(exc.__cause__).__name__
                if exc.__cause__ is not None
                else ""
            )
            outcomes.append(
                TamperOutcome(
                    case_id=case_id,
                    rejected=(
                        observed == tuple(sorted(expected_issue_ids))
                        and cause_type == "PermissionError"
                    ),
                    expected_issue_ids=tuple(sorted(expected_issue_ids)),
                    observed_issue_ids=observed,
                    cause_type=cause_type,
                )
            )
        else:
            outcomes.append(
                TamperOutcome(
                    case_id=case_id,
                    rejected=False,
                    expected_issue_ids=tuple(sorted(expected_issue_ids)),
                    observed_issue_ids=(),
                    cause_type="",
                )
            )
    return tuple(outcomes)


def _probe_signature_summary(
    computations: Sequence[ProbeComputation],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                computation.receipt.template_id,
                computation.receipt.disposition.value,
                computation.receipt.observed_support_signature_ids,
                computation.receipt.observed_counter_signature_ids,
            )
            for computation in computations
        )
    )


def _run_metamorphic_trials(
    artifacts: QualificationArtifacts,
) -> dict[str, dict[str, object]]:
    """Execute fixed selector/runtime metamorphisms and commit safe outcomes."""

    claims_by_id = {
        claim.claim_id: claim
        for claim in artifacts.claims_by_family.values()
    }
    plans_by_claim_hash = {
        artifacts.claims_by_family[family].claim_hash: plan
        for family, plan in artifacts.probe_plans_by_family.items()
    }
    reversed_probe_order = tuple(
        reversed(
            tuple(
                TEMPLATE_ID_BY_FAMILY[family]
                for family in FAMILIES
                if family != "no_op"
            )
        )
    )
    records: dict[str, list[dict[str, object]]] = {
        "claim_order": [],
        "probe_rule_order": [],
        "world_id": [],
        "expected_label": [],
    }
    qualification_by_world_id = {
        result.world.selector_input.world_id: result
        for result in artifacts.world_qualifications
    }
    for world in artifacts.worlds:
        baseline = qualification_by_world_id[
            world.selector_input.world_id
        ]
        baseline_summary = _probe_signature_summary(
            baseline.probe_computations
        )
        original_claims = tuple(
            claims_by_id[claim_id] for claim_id in world.claim_order
        )

        reversed_claims = tuple(reversed(original_claims))
        claim_selected, claim_computations = _select_claim(
            ontology=artifacts.ontology,
            probe_verifier_registry=artifacts.probe_verifier_registry,
            selector_input=world.selector_input,
            ordered_claims=reversed_claims,
            plans_by_claim_hash=plans_by_claim_hash,
        )
        records["claim_order"].append(
            {
                "world_id": world.selector_input.world_id,
                "perturbation_hash": stable_hash(
                    tuple(claim.claim_hash for claim in reversed_claims)
                ),
                "selected_claim_hash": claim_selected.claim_hash,
                "signature_summary_hash": stable_hash(
                    _probe_signature_summary(claim_computations)
                ),
                "invariant": (
                    claim_selected.claim_hash
                    == baseline.selected_claim.claim_hash
                    and _probe_signature_summary(claim_computations)
                    == baseline_summary
                ),
            }
        )

        probe_selected, probe_computations = _select_claim(
            ontology=artifacts.ontology,
            probe_verifier_registry=artifacts.probe_verifier_registry,
            selector_input=world.selector_input,
            ordered_claims=original_claims,
            plans_by_claim_hash=plans_by_claim_hash,
            probe_template_order=reversed_probe_order,
        )
        records["probe_rule_order"].append(
            {
                "world_id": world.selector_input.world_id,
                "perturbation_hash": stable_hash(
                    reversed_probe_order
                ),
                "selected_claim_hash": probe_selected.claim_hash,
                "signature_summary_hash": stable_hash(
                    _probe_signature_summary(probe_computations)
                ),
                "invariant": (
                    probe_selected.claim_hash
                    == baseline.selected_claim.claim_hash
                    and _probe_signature_summary(probe_computations)
                    == baseline_summary
                ),
            }
        )

        renamed_input = replace(
            world.selector_input,
            world_id=(
                "omega.metamorphic."
                + stable_hash(
                    {
                        "world_id": world.selector_input.world_id,
                        "trial": "world_id_rename_v1",
                    }
                )[:20]
            ),
        )
        world_id_selected, world_id_computations = _select_claim(
            ontology=artifacts.ontology,
            probe_verifier_registry=artifacts.probe_verifier_registry,
            selector_input=renamed_input,
            ordered_claims=original_claims,
            plans_by_claim_hash=plans_by_claim_hash,
        )
        records["world_id"].append(
            {
                "world_id": world.selector_input.world_id,
                "renamed_world_id": renamed_input.world_id,
                "selected_claim_hash": world_id_selected.claim_hash,
                "signature_summary_hash": stable_hash(
                    _probe_signature_summary(world_id_computations)
                ),
                "invariant": (
                    world_id_selected.claim_hash
                    == baseline.selected_claim.claim_hash
                    and _probe_signature_summary(world_id_computations)
                    == baseline_summary
                ),
            }
        )

        family_index = FAMILIES.index(world.expected_family)
        relabeled = replace(
            world,
            expected_family=FAMILIES[
                (family_index + 1) % len(FAMILIES)
            ],
        )
        relabel_selected, relabel_computations = _select_claim(
            ontology=artifacts.ontology,
            probe_verifier_registry=artifacts.probe_verifier_registry,
            selector_input=relabeled.selector_input,
            ordered_claims=original_claims,
            plans_by_claim_hash=plans_by_claim_hash,
        )
        selected_probe = next(
            computation.receipt
            for computation in relabel_computations
            if computation.receipt.claim_hash
            == relabel_selected.claim_hash
        )
        relabel_runtime = _runtime_evidence(
            world=relabeled,
            claim=relabel_selected,
            selected_probe=selected_probe,
            treatment=baseline.treatment,
        )
        records["expected_label"].append(
            {
                "world_id": world.selector_input.world_id,
                "relabel_hash": stable_hash(
                    {
                        "from": world.expected_family,
                        "to": relabeled.expected_family,
                    }
                ),
                "selected_claim_hash": relabel_selected.claim_hash,
                "signature_summary_hash": stable_hash(
                    _probe_signature_summary(relabel_computations)
                ),
                "runtime_semantic_commitment": (
                    relabel_runtime.semantic_commitment
                ),
                "invariant": (
                    relabel_selected.claim_hash
                    == baseline.selected_claim.claim_hash
                    and _probe_signature_summary(relabel_computations)
                    == baseline_summary
                    and relabel_runtime
                    == baseline.runtime_evidence
                ),
            }
        )

    result: dict[str, dict[str, object]] = {}
    for trial_id, rows in records.items():
        trial_count = len(rows)
        invariant_count = sum(row["invariant"] is True for row in rows)
        result[trial_id] = {
            "trial_count": trial_count,
            "invariant_count": invariant_count,
            "all_invariant": invariant_count == trial_count,
            "content_commitment": stable_hash(rows),
        }
    return result


def recompute_safe_qualification_counts(
    *,
    world_rows: Sequence[Mapping[str, object]],
    probe_matrix_rows: Sequence[Mapping[str, object]],
    tamper_rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Purely recompute qualification totals from disclosure-safe rows."""

    expected_by_world: dict[str, str] = {}
    for row in world_rows:
        world_id = row.get("world_id")
        expected_template = row.get("expected_template_id")
        if (
            not isinstance(world_id, str)
            or not isinstance(expected_template, str)
            or world_id in expected_by_world
        ):
            raise SyntheticQualificationError(
                "safe world rows are malformed or duplicated"
            )
        expected_by_world[world_id] = expected_template

    correct = sum(
        row.get("selected_template_id")
        == row.get("expected_template_id")
        for row in world_rows
    )
    wrong_claim_count = 0
    wrong_counterevidenced = 0
    matrix_keys: set[tuple[str, str]] = set()
    for row in probe_matrix_rows:
        world_id = row.get("world_id")
        template_id = row.get("template_id")
        if (
            not isinstance(world_id, str)
            or not isinstance(template_id, str)
            or world_id not in expected_by_world
            or (world_id, template_id) in matrix_keys
        ):
            raise SyntheticQualificationError(
                "safe probe-matrix rows are malformed or duplicated"
            )
        matrix_keys.add((world_id, template_id))
        if template_id == expected_by_world[world_id]:
            continue
        wrong_claim_count += 1
        if (
            row.get("disposition")
            == ProbeDisposition.FALSIFIED.value
            and row.get("observed_support_signature_ids") == []
            and isinstance(
                row.get("observed_counter_signature_ids"), list
            )
            and bool(row.get("observed_counter_signature_ids"))
        ):
            wrong_counterevidenced += 1

    active_rows = tuple(
        row
        for row in world_rows
        if row.get("compiled_operator") != "PRESERVE_BASELINE"
    )
    noop_rows = tuple(
        row
        for row in world_rows
        if row.get("compiled_operator") == "PRESERVE_BASELINE"
    )

    def strict_count(row: Mapping[str, object], key: str) -> int:
        value = row.get(key)
        if type(value) is not int or value < 0:
            raise SyntheticQualificationError(
                f"safe row count {key!r} is invalid"
            )
        return value

    wrong_trials = sum(
        strict_count(row, "wrong_operator_trial_count")
        for row in world_rows
    )
    wrong_harms = sum(
        strict_count(row, "wrong_operator_harm_count")
        for row in world_rows
    )
    tamper_rejected = sum(
        row.get("rejected") is True
        and row.get("cause_type") == "PermissionError"
        and row.get("expected_issue_ids")
        == row.get("observed_issue_ids")
        for row in tamper_rows
    )
    return {
        "world_count": len(world_rows),
        "correct_identification_count": correct,
        "wrong_claim_count": wrong_claim_count,
        "wrong_claims_with_counterevidence_count": (
            wrong_counterevidenced
        ),
        "runtime_active_trial_count": len(active_rows),
        "runtime_active_differential_count": sum(
            row.get("active_runtime_differential") is True
            for row in active_rows
        ),
        "runtime_noop_trial_count": len(noop_rows),
        "runtime_noop_semantic_equivalence_count": sum(
            row.get("noop_runtime_equivalent") is True
            for row in noop_rows
        ),
        "wrong_operator_trial_count": wrong_trials,
        "wrong_operator_harm_count": wrong_harms,
        "wrong_operator_harm_world_count": sum(
            strict_count(row, "wrong_operator_harm_count") > 0
            for row in world_rows
        ),
        "tamper_case_count": len(tamper_rows),
        "tamper_rejected_count": tamper_rejected,
    }


def qualify() -> dict[str, object]:
    artifacts = build_qualification_artifacts()
    qualifications = artifacts.world_qualifications
    structural_variant_commitments = _structural_variant_commitments(
        artifacts.worlds
    )
    claim_by_hash = {
        claim.claim_hash: claim
        for claim in artifacts.claims_by_family.values()
    }
    operator_rows: list[dict[str, object]] = []
    probe_matrix_rows: list[dict[str, object]] = []
    for result in qualifications:
        selected_computation = next(
            computation
            for computation in result.probe_computations
            if computation.receipt.claim_hash
            == result.selected_claim.claim_hash
        )
        evidence = result.runtime_evidence
        expected_template_id = TEMPLATE_ID_BY_FAMILY[
            result.world.expected_family
        ]
        operator_rows.append({
            "world_id": result.world.selector_input.world_id,
            "structural_variant": result.world.variant,
            "numeric_payload_hash": semantic_hash(
                result.world.selector_input.numeric_payload.safe_payload()
            ),
            "expected_template_id": expected_template_id,
            "selected_template_id": (
                result.selected_claim.template_ids[0]
            ),
            "selected_probe_receipt_hash": (
                selected_computation.receipt.receipt_hash
            ),
            "probe_evidence_bundle_hash": (
                selected_computation.evidence_bundle
                .evidence_bundle_hash
            ),
            "probe_trust_anchor_hash": (
                selected_computation.receipt
                .probe_trust_anchor_hash
            ),
            "probe_statistic_commitment": (
                selected_computation.statistic_hash
            ),
            "observed_support_signature_ids": list(
                selected_computation.receipt
                .observed_support_signature_ids
            ),
            "observed_counter_signature_ids": list(
                selected_computation.receipt
                .observed_counter_signature_ids
            ),
            "compiled_operator": _operator_from_treatment(
                result.treatment
            ),
            "treatment_disposition": (
                result.treatment.disposition.value
            ),
            "compilation_receipt_hash": (
                result.compilation_receipt.receipt_hash
            ),
            "baseline_plan_hash": evidence.baseline_plan_hash,
            "candidate_plan_hash": evidence.candidate_plan_hash,
            "baseline_oracle_utility": evidence.baseline_utility,
            "candidate_oracle_utility": evidence.candidate_utility,
            "runtime_selected_lane": evidence.selected_lane,
            "active_runtime_differential": (
                evidence.active_runtime_differential
            ),
            "noop_runtime_equivalent": evidence.noop_runtime_equivalent,
            "wrong_operator_trial_count": (
                evidence.wrong_operator_trial_count
            ),
            "wrong_operator_harm_count": (
                evidence.wrong_operator_harm_count
            ),
            "runtime_semantic_commitment": (
                evidence.semantic_commitment
            ),
        })
        for computation in result.probe_computations:
            receipt = computation.receipt
            claim = claim_by_hash[receipt.claim_hash]
            probe_matrix_rows.append(
                {
                    "world_id": result.world.selector_input.world_id,
                    "template_id": receipt.template_id,
                    "claim_id": claim.claim_id,
                    "claim_hash": claim.claim_hash,
                    "disposition": receipt.disposition.value,
                    "observed_support_signature_ids": list(
                        receipt.observed_support_signature_ids
                    ),
                    "observed_counter_signature_ids": list(
                        receipt.observed_counter_signature_ids
                    ),
                    "probe_trust_anchor_hash": (
                        receipt.probe_trust_anchor_hash
                    ),
                    "statistic_commitment_hash": (
                        computation.statistic_hash
                    ),
                    "evidence_bundle_hash": (
                        computation.evidence_bundle
                        .evidence_bundle_hash
                    ),
                    "probe_receipt_hash": receipt.receipt_hash,
                }
            )
    probe_matrix_rows.sort(
        key=lambda row: (
            str(row["world_id"]),
            str(row["template_id"]),
        )
    )
    if not _prediction_signatures_distinct(artifacts.claims_by_family):
        raise SyntheticQualificationError(
            "claim prediction signatures are not distinct"
        )
    if any(
        row["compiled_operator"]
        != OPERATOR_BY_FAMILY[
            next(
                family
                for family, template_id in TEMPLATE_ID_BY_FAMILY.items()
                if template_id == row["selected_template_id"]
            )
        ]
        for row in operator_rows
    ):
        raise SyntheticQualificationError(
            "compiled operator did not match the selected claim"
        )
    tamper_outcomes = _fixed_tamper_matrix(artifacts)
    tamper_rows = [
        {
            "case_id": outcome.case_id,
            "expected_issue_ids": list(outcome.expected_issue_ids),
            "observed_issue_ids": list(outcome.observed_issue_ids),
            "cause_type": outcome.cause_type,
            "rejected": outcome.rejected,
        }
        for outcome in tamper_outcomes
    ]
    recomputed = recompute_safe_qualification_counts(
        world_rows=operator_rows,
        probe_matrix_rows=probe_matrix_rows,
        tamper_rows=tamper_rows,
    )
    expected_recomputed = {
        "world_count": 10,
        "correct_identification_count": 10,
        "wrong_claim_count": 40,
        "wrong_claims_with_counterevidence_count": 40,
        "runtime_active_trial_count": 8,
        "runtime_active_differential_count": 8,
        "runtime_noop_trial_count": 2,
        "runtime_noop_semantic_equivalence_count": 2,
        "wrong_operator_trial_count": 32,
        "wrong_operator_harm_count": 32,
        "wrong_operator_harm_world_count": 10,
        "tamper_case_count": 19,
        "tamper_rejected_count": 19,
    }
    if recomputed != expected_recomputed:
        raise SyntheticQualificationError(
            "safe-row recomputation did not reproduce frozen totals"
        )
    metamorphic_trials = _run_metamorphic_trials(artifacts)
    if (
        set(metamorphic_trials)
        != {
            "claim_order",
            "probe_rule_order",
            "world_id",
            "expected_label",
        }
        or any(
            row["trial_count"] != 10
            or row["invariant_count"] != 10
            or row["all_invariant"] is not True
            for row in metamorphic_trials.values()
        )
    ):
        raise SyntheticQualificationError(
            "fixed metamorphic qualification trial failed"
        )
    body: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "version": VERSION,
        "status": STATUS,
        "formal_result": False,
        "efficacy_evidence": False,
        "fixture_provenance": (
            "hand_authored_fixed_schema_numeric_mechanisms_only"
        ),
        "ontology_hash": artifacts.ontology.ontology_hash,
        "mechanism_families": list(FAMILIES),
        "structural_variants": list(VARIANTS),
        "structural_variant_commitments": (
            structural_variant_commitments
        ),
        "structural_variant_nonisomorphism_verified": True,
        "world_count": recomputed["world_count"],
        "correct_identification_count": (
            recomputed["correct_identification_count"]
        ),
        "all_known_mechanisms_identified": (
            recomputed["correct_identification_count"]
            == recomputed["world_count"]
        ),
        "prediction_signatures_distinct": True,
        "wrong_claim_count": recomputed["wrong_claim_count"],
        "wrong_claims_with_counterevidence_count": (
            recomputed["wrong_claims_with_counterevidence_count"]
        ),
        "all_wrong_claims_counterevidenced": (
            recomputed["wrong_claim_count"]
            == recomputed["wrong_claims_with_counterevidence_count"]
        ),
        "claim_order_fixed_perturbation": (
            metamorphic_trials["claim_order"]["all_invariant"]
        ),
        "probe_rule_order_invariant": (
            metamorphic_trials["probe_rule_order"]["all_invariant"]
        ),
        "world_id_invariant": (
            metamorphic_trials["world_id"]["all_invariant"]
        ),
        "expected_label_invariant": (
            metamorphic_trials["expected_label"]["all_invariant"]
        ),
        "metamorphic_trials": metamorphic_trials,
        "selector_input_contract": (
            "opaque_world_id_and_fixed_schema_numeric_payload_only"
        ),
        "numeric_payload_contract": (
            "same_shape_integer_panels_without_family_or_expected_label"
        ),
        "numeric_payload_shape": {
            "action_fold_utilities": [4, 6],
            "subset_utility_folds": [2, 64],
            "adjacency": [6, 6],
            "node_effect_folds": [4, 6],
            "observation_folds": [4, 8],
            "decision_payoffs": [4, 4],
        },
        "integer_ratio_decision_contract": (
            "all_threshold_decisions_use_committed_cross_product_margins"
        ),
        "oracle_utility_contract": (
            "recipe_typed_numeric_panel_without_expected_family_lookup"
        ),
        "minimum_commitment_two_stage": True,
        "minimum_commitment_stage_order": (
            "four_active_probes_then_context_bound_noop_probe"
        ),
        "probe_verifier_trust_anchors": [
            artifacts.probe_verifier_registry.require_trust_anchor(
                artifacts.ontology.require_template(template_id)
            ).safe_payload()
            for template_id in TEMPLATE_ID_BY_FAMILY.values()
        ],
        "probe_evidence_bundle_count": len(probe_matrix_rows),
        "all_probe_evidence_commitment": stable_hash(
            tuple(
                sorted(
                    str(row["evidence_bundle_hash"])
                    for row in probe_matrix_rows
                )
            )
        ),
        "compiler_id": TREATMENT_COMPILER_ID,
        "compiler_hash": _treatment_compiler_hash(),
        "compiler_trust_anchor_hash": (
            _compiler_trust_anchor().anchor_hash
        ),
        "selection_policy_hash": _selection_policy_hash(),
        "world_compilations": operator_rows,
        "probe_matrix_rows": probe_matrix_rows,
        "safe_recomputed_counts": recomputed,
        "probe_statistic_commitments": [
            {
                "world_id": row["world_id"],
                "selected_template_id": row["selected_template_id"],
                "probe_statistic_commitment": (
                    row["probe_statistic_commitment"]
                ),
                "probe_evidence_bundle_hash": (
                    row["probe_evidence_bundle_hash"]
                ),
                "probe_trust_anchor_hash": (
                    row["probe_trust_anchor_hash"]
                ),
                "selected_probe_receipt_hash": (
                    row["selected_probe_receipt_hash"]
                ),
            }
            for row in operator_rows
        ],
        "all_probe_receipts_trusted_recomputed": True,
        "all_compilation_receipts_valid": True,
        "tamper_case_ids": [
            outcome.case_id for outcome in tamper_outcomes
        ],
        "tamper_case_count": recomputed["tamper_case_count"],
        "tamper_rejected_count": (
            recomputed["tamper_rejected_count"]
        ),
        "all_tampers_rejected": (
            recomputed["tamper_case_count"]
            == recomputed["tamper_rejected_count"]
        ),
        "tamper_rejections": tamper_rows,
        "no_op_disposition": "preserve_baseline_program_none",
        "runtime_active_trial_count": (
            recomputed["runtime_active_trial_count"]
        ),
        "runtime_active_differential_count": (
            recomputed["runtime_active_differential_count"]
        ),
        "runtime_noop_trial_count": (
            recomputed["runtime_noop_trial_count"]
        ),
        "runtime_noop_semantic_equivalence_count": (
            recomputed["runtime_noop_semantic_equivalence_count"]
        ),
        "wrong_operator_trial_count": (
            recomputed["wrong_operator_trial_count"]
        ),
        "wrong_operator_harm_count": (
            recomputed["wrong_operator_harm_count"]
        ),
        "wrong_operator_harm_world_count": (
            recomputed["wrong_operator_harm_world_count"]
        ),
        "all_wrong_operators_harmful": (
            recomputed["wrong_operator_harm_count"]
            == recomputed["wrong_operator_trial_count"]
        ),
        "formal_source_access_count": 0,
        "source_payload_access_count": 0,
        "network_call_count": 0,
        "model_asset_access_count": 0,
        "api_call_count": 0,
        "online_evaluator_call_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
    }
    return {**body, "self_sha256": semantic_hash(body)}
