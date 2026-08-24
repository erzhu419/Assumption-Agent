"""Bounded, public-synthetic relation-invention mechanics.

This module exercises the missing connection between an ontology-inadequacy
report and an executable :class:`RelationLawCandidate`.  It is deliberately a
shadow-only mechanics qualification.  The fixture is public and synthetic,
the holdout is process-local rather than independently custodied, C1 has not
passed, and the language compiler remains closed.  Consequently no result
from this module is C2 evidence or an active theory mutation.

The synthesizer searches a frozen Boolean program grammar.  It receives only
discovery and validation labels.  A separate prediction commitment is created
from holdout inputs before the process-local vault releases holdout labels.
The accepted candidate is a reusable stateful threshold relation, not a row
lookup: its executable program contains only frozen grammar operators and no
example identifier or outcome literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Iterable, NoReturn

from .bootstrap import initial_theory
from .diagnostics import (
    InadequacyVerdict,
    OntologyInadequacyReport,
    ResidualProfile,
    diagnose_ontology_inadequacy,
)
from .governance import compile_patch
from .hashing import canonical_json, stable_hash
from .schema import (
    AuthorityAssignment,
    AuthorityRole,
    EvidenceSplit,
    IdentifiabilityCertificate,
    PatchCoordinate,
    PreregisteredPrediction,
    ReductionMap,
    RelationLawCandidate,
    TheoryPatch,
    ViolationFunctionalSpec,
    freeze_pairs,
)


SCHEMA_VERSION: Final = "hegel-machine-bounded-relation-invention-v1/1"
GRAMMAR_VERSION: Final = "hegel-bounded-boolean-program-grammar-v1"
FIXTURE_VERSION: Final = "public-synthetic-stateful-threshold-fixture-v1"
STATUS: Final = (
    "SHADOW_RELATION_CANDIDATE_VERIFIED_PUBLIC_SYNTHETIC_C2_LOCKED"
)
CLAIM_LEVEL: Final = (
    "LOCAL_PUBLIC_SYNTHETIC_MECHANICS_ONLY_NOT_SEALED_NOT_INDEPENDENT_"
    "NOT_C1_NOT_C2_NOT_OPEN_WORLD_NOT_ACTIVE_THEORY"
)
SYNTHESIS_MAX_NODES: Final = 5
OLD_COMPARISON_MAX_NODES: Final = 3
OLD_COMPARISON_EXPRESSION: Final = "drive>=0"
THRESHOLD_GRID: Final = (-2, -1, 0, 1, 2, 3)
PROGRAM_NODE_BITS: Final = 2
RESIDUAL_ERROR_BITS: Final = 4
SCOPE: Final = "public_synthetic_stateful_threshold_v1"
SYMBOL: Final = "memory_conditioned_activation_v1"

NONCLAIMS: Final = (
    "public_synthetic_fixture_is_not_independently_generated_hidden_evidence",
    "process_local_holdout_is_not_external_custody_or_non_access_proof",
    "bounded_program_search_is_not_unrestricted_open_world_discovery",
    "finite_fixture_non_equivalence_is_not_universal_non_expressibility",
    "frozen_synthetic_diagnostic_profile_is_not_executed_repair_evidence",
    "discovery_residual_probe_is_not_external_preregistration",
    "distinct_process_local_role_ids_are_not_independent_authorities",
    "declared_reduction_map_is_not_executed_reduction_evidence",
    "fixed_bit_description_gain_is_not_formal_mdl_or_new_symbol_cost",
    "candidate_branch_is_not_c1_or_c2_pass",
    "candidate_branch_is_not_active_theory_or_promotion_authority",
    "shadow_byte_preservation_is_not_multigeneration_retention_evidence",
    "no_network_container_external_actor_or_real_scientific_outcome_used",
)


class RelationInventionError(ValueError):
    """Fail-closed error for invalid or inconclusive mechanics."""


@dataclass(frozen=True, slots=True)
class RelationInput:
    input_id: str
    drive: float
    prior_on: bool
    seed: int

    def __post_init__(self) -> None:
        if not self.input_id:
            raise RelationInventionError("relation input id is required")
        if isinstance(self.drive, bool) or not isinstance(self.drive, (int, float)):
            raise RelationInventionError("drive must be a finite real number")
        if not isfinite(float(self.drive)):
            raise RelationInventionError("drive must be finite")
        if type(self.prior_on) is not bool or type(self.seed) is not int:
            raise RelationInventionError("relation input types differ")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="relation_input_")


@dataclass(frozen=True, slots=True)
class LabeledRelationExample:
    relation_input: RelationInput
    split: EvidenceSplit
    next_on: bool

    def __post_init__(self) -> None:
        if type(self.next_on) is not bool:
            raise RelationInventionError("relation outcome must be Boolean")

    @property
    def observation_id(self) -> str:
        return stable_hash(self, prefix="observation_")


@dataclass(frozen=True, slots=True)
class Program:
    op: str
    value: int | bool | None = None
    children: tuple["Program", ...] = ()

    def __post_init__(self) -> None:
        if type(self.children) is not tuple:
            raise RelationInventionError("program children must be an exact tuple")
        if self.op == "const":
            if type(self.value) is not bool or self.children:
                raise RelationInventionError("constant program differs")
        elif self.op == "prior_on":
            if self.value is not None or self.children:
                raise RelationInventionError("prior-state program differs")
        elif self.op == "drive_ge":
            if type(self.value) is not int or self.value not in THRESHOLD_GRID:
                raise RelationInventionError("threshold program differs")
            if self.children:
                raise RelationInventionError("threshold atom has children")
        elif self.op == "not":
            if (
                self.value is not None
                or len(self.children) != 1
                or self.children[0].node_count != 1
            ):
                raise RelationInventionError("not program differs")
        elif self.op in {"and", "or"}:
            if self.value is not None or len(self.children) != 2:
                raise RelationInventionError("binary program differs")
            if self.children[0].expression > self.children[1].expression:
                raise RelationInventionError("commutative program is not canonical")
        else:
            raise RelationInventionError("program operator is outside the grammar")

    @property
    def node_count(self) -> int:
        return 1 + sum(child.node_count for child in self.children)

    @property
    def expression(self) -> str:
        if self.op == "const":
            return "true" if self.value else "false"
        if self.op == "prior_on":
            return "prior_on"
        if self.op == "drive_ge":
            return f"drive>={self.value}"
        if self.op == "not":
            return f"not({self.children[0].expression})"
        return (
            f"{self.op}({self.children[0].expression},"
            f"{self.children[1].expression})"
        )

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="relation_program_")

    def evaluate(self, relation_input: RelationInput) -> bool:
        if self.op == "const":
            return bool(self.value)
        if self.op == "prior_on":
            return relation_input.prior_on
        if self.op == "drive_ge":
            return relation_input.drive >= int(self.value)
        if self.op == "not":
            return not self.children[0].evaluate(relation_input)
        left = self.children[0].evaluate(relation_input)
        right = self.children[1].evaluate(relation_input)
        return left and right if self.op == "and" else left or right


def _binary(op: str, left: Program, right: Program) -> Program:
    children = tuple(sorted((left, right), key=lambda item: item.expression))
    return Program(op, children=children)


def enumerate_programs(maximum_nodes: int = SYNTHESIS_MAX_NODES) -> tuple[Program, ...]:
    """Enumerate the complete frozen grammar up to ``maximum_nodes``."""

    if type(maximum_nodes) is not int or not 1 <= maximum_nodes <= SYNTHESIS_MAX_NODES:
        raise RelationInventionError("program budget is outside the frozen range")
    atoms = {
        Program("const", False),
        Program("const", True),
        Program("prior_on"),
        *(Program("drive_ge", threshold) for threshold in THRESHOLD_GRID),
    }
    by_size: dict[int, set[Program]] = {1: atoms}
    for size in range(2, maximum_nodes + 1):
        current: set[Program] = set()
        if size == 2:
            current.update(Program("not", children=(atom,)) for atom in atoms)
        for left_size in range(1, size - 1):
            right_size = size - 1 - left_size
            for left in by_size.get(left_size, set()):
                for right in by_size.get(right_size, set()):
                    current.add(_binary("and", left, right))
                    current.add(_binary("or", left, right))
        by_size[size] = current
    programs = tuple(
        sorted(
            (program for size in by_size for program in by_size[size]),
            key=lambda item: (item.node_count, item.expression),
        )
    )
    if len({program.expression for program in programs}) != len(programs):
        raise AssertionError("program enumeration emitted duplicate syntax")
    return programs


def _accuracy(program: Program, examples: tuple[LabeledRelationExample, ...]) -> float:
    if not examples:
        raise RelationInventionError("accuracy requires examples")
    return sum(
        program.evaluate(example.relation_input) is example.next_on
        for example in examples
    ) / len(examples)


@dataclass(frozen=True, slots=True)
class SynthesisCommitment:
    grammar_version: str
    fixture_version: str
    ontology_report_id: str
    selected_program: Program
    discovery_root: str
    validation_root: str
    discovery_accuracy: float
    validation_accuracy: float
    old_comparison_accuracy: float
    old_comparison_program: Program
    exact_behavior_class_count: int
    minimum_exact_node_count: int

    @property
    def commitment_id(self) -> str:
        return stable_hash(self, prefix="relation_synthesis_commitment_")


def _select_old_program(
    discovery: tuple[LabeledRelationExample, ...],
) -> Program:
    """Return the predeclared comparator without reading validation labels."""

    if type(discovery) is not tuple:
        raise RelationInventionError("comparison discovery must be an exact tuple")
    if not discovery:
        raise RelationInventionError("comparison discovery cannot be empty")
    old_programs = enumerate_programs(OLD_COMPARISON_MAX_NODES)
    selected = next(
        (
            program
            for program in old_programs
            if program.expression == OLD_COMPARISON_EXPRESSION
        ),
        None,
    )
    if selected is None:
        raise AssertionError("frozen old comparator left the old grammar")
    best_discovery_accuracy = max(
        _accuracy(program, discovery) for program in old_programs
    )
    if _accuracy(selected, discovery) != best_discovery_accuracy:
        raise RelationInventionError(
            "frozen old comparator is no longer discovery-optimal"
        )
    return selected


def _require_no_exact_old_program(
    discovery: tuple[LabeledRelationExample, ...],
    validation: tuple[LabeledRelationExample, ...],
) -> None:
    """Use validation only to reject an already-expressible target."""

    if any(
        _accuracy(program, discovery) == 1.0
        and _accuracy(program, validation) == 1.0
        for program in enumerate_programs(OLD_COMPARISON_MAX_NODES)
    ):
        raise RelationInventionError("old low-order language already fits exactly")


def derive_ontology_report(
    discovery: tuple[LabeledRelationExample, ...],
    validation: tuple[LabeledRelationExample, ...],
) -> tuple[ResidualProfile, OntologyInadequacyReport, Program]:
    """Replay a frozen synthetic profile before any language synthesis.

    The residual probe is frozen from discovery errors only: a predeclared old
    threshold program fails in the interval between adjacent frozen thresholds
    when the prior state disagrees.  Its ability to predict the validation
    error mask is then measured before opening the synthesis grammar or the
    holdout vault.  The cheaper-repair gains are public-fixture assertions, not
    outputs of independent repair evaluators.
    """

    if type(discovery) is not tuple or type(validation) is not tuple:
        raise RelationInventionError("comparison partitions must be exact tuples")
    if not discovery or not validation:
        raise RelationInventionError("comparison partitions cannot be empty")
    _require_no_exact_old_program(discovery, validation)
    old_program = _select_old_program(discovery)
    discovery_errors = tuple(
        item
        for item in discovery
        if old_program.evaluate(item.relation_input) is not item.next_on
    )
    validation_errors = tuple(
        item
        for item in validation
        if old_program.evaluate(item.relation_input) is not item.next_on
    )
    if old_program.expression != OLD_COMPARISON_EXPRESSION:
        raise RelationInventionError("frozen old comparator changed")

    def predicted_residual(item: LabeledRelationExample) -> bool:
        relation_input = item.relation_input
        return (
            0 <= relation_input.drive < 2
            and relation_input.prior_on is False
        )

    if not discovery_errors or not validation_errors:
        raise RelationInventionError("residual probe lacks repeated errors")
    discovery_mask_accuracy = sum(
        predicted_residual(item)
        is (old_program.evaluate(item.relation_input) is not item.next_on)
        for item in discovery
    ) / len(discovery)
    if discovery_mask_accuracy != 1.0:
        raise RelationInventionError(
            "frozen discovery residual mask has false positives or negatives"
        )
    validation_mask_accuracy = sum(
        predicted_residual(item)
        is (old_program.evaluate(item.relation_input) is not item.next_on)
        for item in validation
    ) / len(validation)
    if validation_mask_accuracy != 1.0:
        raise RelationInventionError("frozen discovery residual mask did not transfer")

    naive_error_bits = (
        len(discovery_errors + validation_errors) * RESIDUAL_ERROR_BITS
    )
    residual_signature_bits = 3 * PROGRAM_NODE_BITS
    compression_gain = (
        naive_error_bits - residual_signature_bits
    ) / naive_error_bits
    profile = ResidualProfile(
        profile_id=stable_hash(
            (
                old_program.content_id,
                tuple(item.observation_id for item in discovery),
                tuple(item.observation_id for item in validation),
                "residual_if_0_le_drive_lt_2_and_prior_off",
            ),
            prefix="bounded_residual_profile_",
        ),
        refit_gain=0.0,
        uncertainty_coverage=0.0,
        scope_repair_gain=0.0,
        mixture_gain=0.0,
        low_order_composition_gain=0.0,
        idealization_gain=0.0,
        robustification_tail_gain=0.0,
        added_probe_gain=0.0,
        cross_seed_stability=1.0,
        structural_coherence=1.0,
        uncertainty_excess=1.0,
        compression_gain=compression_gain,
        preregistered_prediction_gain=(
            len(validation_errors) / len(validation)
        ),
        case_count=len(discovery + validation),
        outlier_fraction=0.0,
    )
    report = diagnose_ontology_inadequacy(profile)
    if (
        report.verdict is not InadequacyVerdict.ONTOLOGY_DEFECT
        or not report.language_extension_allowed
    ):
        raise RelationInventionError(
            "synthetic diagnostic profile replay did not open language"
        )
    return profile, report, old_program


def synthesize_relation(
    discovery: tuple[LabeledRelationExample, ...],
    validation: tuple[LabeledRelationExample, ...],
    ontology_report: OntologyInadequacyReport,
) -> SynthesisCommitment:
    """Select one exact behavior class after the synthetic profile gate."""

    if (
        type(ontology_report) is not OntologyInadequacyReport
        or ontology_report.verdict is not InadequacyVerdict.ONTOLOGY_DEFECT
        or not ontology_report.language_extension_allowed
    ):
        raise RelationInventionError("synthesis requires an ontology-defect report")
    if type(discovery) is not tuple or type(validation) is not tuple:
        raise RelationInventionError("synthesis partitions must be exact tuples")
    if not discovery or not validation:
        raise RelationInventionError("synthesis partitions cannot be empty")
    _, expected_report, _ = derive_ontology_report(discovery, validation)
    if ontology_report.report_id != expected_report.report_id:
        raise RelationInventionError(
            "ontology report does not bind the synthesis partitions"
        )
    if any(example.split is not EvidenceSplit.TRAIN for example in discovery):
        raise RelationInventionError("discovery split differs")
    if any(example.split is not EvidenceSplit.VALIDATION for example in validation):
        raise RelationInventionError("validation split differs")
    identifiers = tuple(
        example.observation_id for example in discovery + validation
    )
    if len(set(identifiers)) != len(identifiers):
        raise RelationInventionError("synthesis partitions repeat evidence")

    examples = discovery + validation
    target_vector = tuple(example.next_on for example in examples)
    quotient: dict[tuple[bool, ...], Program] = {}
    for program in enumerate_programs():
        vector = tuple(program.evaluate(example.relation_input) for example in examples)
        incumbent = quotient.get(vector)
        if incumbent is None or (
            program.node_count,
            program.expression,
        ) < (incumbent.node_count, incumbent.expression):
            quotient[vector] = program
    selected = quotient.get(target_vector)
    if selected is None:
        raise RelationInventionError("no exact relation in the frozen synthesis grammar")

    old_program = _select_old_program(discovery)
    if selected.node_count <= OLD_COMPARISON_MAX_NODES:
        raise RelationInventionError("old low-order language already fits exactly")
    if _accuracy(selected, discovery) != 1.0 or _accuracy(selected, validation) != 1.0:
        raise AssertionError("selected exact behavior does not replay exactly")
    return SynthesisCommitment(
        grammar_version=GRAMMAR_VERSION,
        fixture_version=FIXTURE_VERSION,
        ontology_report_id=ontology_report.report_id,
        selected_program=selected,
        discovery_root=stable_hash(discovery, prefix="discovery_partition_"),
        validation_root=stable_hash(validation, prefix="validation_partition_"),
        discovery_accuracy=1.0,
        validation_accuracy=1.0,
        old_comparison_accuracy=min(
            _accuracy(old_program, discovery), _accuracy(old_program, validation)
        ),
        old_comparison_program=old_program,
        exact_behavior_class_count=len(quotient),
        minimum_exact_node_count=selected.node_count,
    )


@dataclass(frozen=True, slots=True)
class PredictionCommitment:
    synthesis_commitment_id: str
    holdout_input_root: str
    input_ids_in_order: tuple[str, ...]
    predicted_outcomes_in_order: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.synthesis_commitment_id or not self.holdout_input_root:
            raise RelationInventionError("prediction commitment identity is required")
        if (
            type(self.input_ids_in_order) is not tuple
            or type(self.predicted_outcomes_in_order) is not tuple
        ):
            raise RelationInventionError("prediction commitment vectors must be tuples")
        if (
            not self.input_ids_in_order
            or len(self.input_ids_in_order) != len(self.predicted_outcomes_in_order)
            or len(set(self.input_ids_in_order)) != len(self.input_ids_in_order)
        ):
            raise RelationInventionError("prediction commitment cardinality differs")
        if any(type(outcome) is not bool for outcome in self.predicted_outcomes_in_order):
            raise RelationInventionError("prediction commitment outcomes must be Boolean")

    @property
    def commitment_id(self) -> str:
        return stable_hash(self, prefix="relation_prediction_commitment_")


def preregister_holdout_predictions(
    synthesis: SynthesisCommitment,
    holdout_inputs: tuple[RelationInput, ...],
) -> PredictionCommitment:
    if type(holdout_inputs) is not tuple or not holdout_inputs:
        raise RelationInventionError("holdout inputs must be a nonempty tuple")
    input_ids = tuple(item.content_id for item in holdout_inputs)
    if len(set(input_ids)) != len(input_ids):
        raise RelationInventionError("holdout inputs repeat")
    return PredictionCommitment(
        synthesis_commitment_id=synthesis.commitment_id,
        holdout_input_root=stable_hash(holdout_inputs, prefix="holdout_input_partition_"),
        input_ids_in_order=input_ids,
        predicted_outcomes_in_order=tuple(
            synthesis.selected_program.evaluate(item) for item in holdout_inputs
        ),
    )


class ProcessLocalHoldoutVault:
    """A chronology fixture, explicitly not an independent custodian."""

    def __init__(
        self,
        inputs: tuple[RelationInput, ...],
        outcomes: tuple[bool, ...],
    ) -> None:
        if type(inputs) is not tuple or type(outcomes) is not tuple:
            raise RelationInventionError("holdout vault inputs must be tuples")
        if not inputs or len(inputs) != len(outcomes):
            raise RelationInventionError("holdout vault cardinality differs")
        if any(type(outcome) is not bool for outcome in outcomes):
            raise RelationInventionError("holdout vault outcomes differ")
        self._inputs = inputs
        self._outcomes = outcomes
        self._opened = False

    @property
    def inputs(self) -> tuple[RelationInput, ...]:
        return self._inputs

    @property
    def opened(self) -> bool:
        return self._opened

    def open(
        self,
        prediction: PredictionCommitment,
        *,
        expected_synthesis_commitment_id: str,
    ) -> tuple[LabeledRelationExample, ...]:
        if self._opened:
            raise RelationInventionError("process-local holdout already opened")
        if (
            not expected_synthesis_commitment_id
            or prediction.synthesis_commitment_id
            != expected_synthesis_commitment_id
        ):
            raise RelationInventionError(
                "prediction does not bind the expected synthesis commitment"
            )
        expected_root = stable_hash(self._inputs, prefix="holdout_input_partition_")
        expected_ids = tuple(item.content_id for item in self._inputs)
        if (
            prediction.holdout_input_root != expected_root
            or prediction.input_ids_in_order != expected_ids
        ):
            raise RelationInventionError("prediction does not bind the holdout inputs")
        self._opened = True
        return tuple(
            LabeledRelationExample(item, EvidenceSplit.HOLDOUT, outcome)
            for item, outcome in zip(self._inputs, self._outcomes, strict=True)
        )


def _target(drive: float, prior_on: bool) -> bool:
    return drive >= 2 or (prior_on and drive >= 0)


def _input(label: str, drive: float, prior_on: bool, seed: int) -> RelationInput:
    return RelationInput(
        input_id=f"{FIXTURE_VERSION}:{label}:d={drive}:m={int(prior_on)}:s={seed}",
        drive=drive,
        prior_on=prior_on,
        seed=seed,
    )


def build_public_synthetic_fixture() -> tuple[
    tuple[LabeledRelationExample, ...],
    tuple[LabeledRelationExample, ...],
    tuple[LabeledRelationExample, ...],
    ProcessLocalHoldoutVault,
]:
    discovery = tuple(
        LabeledRelationExample(
            relation_input := _input("discovery", drive, prior, seed),
            EvidenceSplit.TRAIN,
            _target(relation_input.drive, relation_input.prior_on),
        )
        for seed in (0, 1, 2)
        for drive in (-2.0, 0.0, 2.0)
        for prior in (False, True)
    )
    validation = tuple(
        LabeledRelationExample(
            relation_input := _input("validation", drive, prior, seed),
            EvidenceSplit.VALIDATION,
            _target(relation_input.drive, relation_input.prior_on),
        )
        for seed in (3, 4)
        for drive in (-1.0, 1.0, 3.0)
        for prior in (False, True)
    )
    hard_negative_inputs = tuple(
        _input("hard-negative", drive, prior, 7)
        for drive, prior in ((-2.5, True), (0.5, False), (1.5, False))
    )
    hard_negatives = tuple(
        LabeledRelationExample(item, EvidenceSplit.HARD_NEGATIVE, False)
        for item in hard_negative_inputs
    )
    holdout_inputs = tuple(
        _input("holdout", drive, prior, seed)
        for seed, drive in ((5, -2.5), (6, 0.5), (7, 1.5), (8, 2.5))
        for prior in (False, True)
    )
    holdout_outcomes = tuple(
        _target(item.drive, item.prior_on) for item in holdout_inputs
    )
    return (
        discovery,
        validation,
        hard_negatives,
        ProcessLocalHoldoutVault(holdout_inputs, holdout_outcomes),
    )


def _compression_gain(
    synthesis: SynthesisCommitment,
    examples: tuple[LabeledRelationExample, ...],
) -> float:
    old_errors = sum(
        synthesis.old_comparison_program.evaluate(example.relation_input)
        is not example.next_on
        for example in examples
    )
    old_bits = (
        synthesis.old_comparison_program.node_count * PROGRAM_NODE_BITS
        + old_errors * RESIDUAL_ERROR_BITS
    )
    candidate_bits = synthesis.selected_program.node_count * PROGRAM_NODE_BITS
    if not 0 < candidate_bits < old_bits:
        raise RelationInventionError("candidate lacks frozen-code compression gain")
    return (old_bits - candidate_bits) / old_bits


def _preregistered_prediction(
    prediction: PredictionCommitment,
) -> PreregisteredPrediction:
    return PreregisteredPrediction(
        prediction_id=prediction.commitment_id,
        input_condition=(
            "the exact process-local holdout input root "
            + prediction.holdout_input_root
        ),
        outcome_name="exact committed Boolean outcome-vector accuracy",
        expected_direction="equal_to_preopened_prediction_vector",
        expected_range=(1.0, 1.0),
        failure_criterion="any holdout outcome differs from the committed vector",
        registered_at_cutoff="2026-08-24T00:00:00+08:00",
    )


def _candidate_and_patch(
    synthesis: SynthesisCommitment,
    prediction: PredictionCommitment,
    report: OntologyInadequacyReport,
    discovery: tuple[LabeledRelationExample, ...],
    validation: tuple[LabeledRelationExample, ...],
    hard_negatives: tuple[LabeledRelationExample, ...],
) -> tuple[RelationLawCandidate, TheoryPatch]:
    parent = initial_theory()
    preregistered = _preregistered_prediction(prediction)
    reduction = ReductionMap(
        reduction_id="reduction_shadow_stateful_threshold_to_parent_v1",
        parent_version_id=parent.version_id,
        child_candidate_id="candidate_memory_conditioned_activation_v1",
        old_scope=parent.scope,
        mapping_description=(
            "outside the bounded synthetic scope the shadow candidate is not "
            "dispatched and the parent theory bytes are returned unchanged"
        ),
        executable_check_id="check_shadow_parent_byte_identity_v1",
        maximum_error=0.0,
    )
    functional = ViolationFunctionalSpec(
        functional_id="vf_memory_conditioned_activation_candidate_v1",
        law_kind=None,
        required_observables=("drive", "prior_on", "next_on"),
        output_semantics=(
            "zero iff the executable candidate prediction equals observed next_on; "
            "one otherwise"
        ),
        tolerance=0.0,
    )
    positives = tuple(
        item.observation_id
        for item in discovery + validation
        if item.next_on
    )
    candidate = RelationLawCandidate(
        candidate_id="candidate_memory_conditioned_activation_v1",
        symbol=SYMBOL,
        kind=None,
        arity=2,
        roles=("drive_signal", "prior_state"),
        executable_definition=synthesis.selected_program.expression,
        violation_functional=functional,
        scopes=(SCOPE,),
        scale_ids=("phase2_default",),
        positive_observation_ids=positives,
        hard_negative_ids=tuple(item.observation_id for item in hard_negatives),
        predictions=(preregistered,),
        reduction_map=reduction,
        description_length=float(
            synthesis.selected_program.node_count * PROGRAM_NODE_BITS
        ),
        identifiability=IdentifiabilityCertificate(
            certificate_id="identifiability_frozen_fixture_truth_vector_v1",
            identifiable_up_to=("exact Boolean behavior on bound fixture inputs",),
            required_interventions=(
                "vary drive across both activation thresholds",
                "toggle prior state at fixed drive",
            ),
            equivalent_representations=(
                "canonical AST rewrites with the same bound truth vector",
            ),
            remaining_gauge_freedom=(
                "behavior outside the finite bound input universe",
            ),
            verified=True,
        ),
    )
    patch = TheoryPatch(
        patch_id="patch_shadow_memory_conditioned_activation_v1",
        candidate_id=candidate.candidate_id,
        parent_version_id=parent.version_id,
        coordinate=PatchCoordinate.LANGUAGE,
        claim=(
            "add a shadow-only memory-conditioned activation relation on the "
            "bounded public synthetic scope"
        ),
        scope=(SCOPE,),
        failure_boundary=(
            "outside_bound_input_universe",
            "non_boolean_prior_state",
            "holdout_prediction_mismatch",
        ),
        predictions=(preregistered,),
        hard_negative_ids=candidate.hard_negative_ids,
        reduction_map_id=reduction.reduction_id,
        conditional_description_length=candidate.description_length,
        payload=freeze_pairs(
            {
                "operation": "add_shadow_relation_candidate",
                "symbol": candidate.symbol,
                "program": synthesis.selected_program.expression,
                "program_content_id": synthesis.selected_program.content_id,
                "scope": SCOPE,
            }
        ),
        authority_assignments=tuple(
            AuthorityAssignment(role, f"bounded_{role.value}_actor_v1")
            for role in AuthorityRole
        ),
        ontology_report_id=report.report_id,
    )
    return candidate, patch


def _require_exact_prediction_replay(
    prediction: PredictionCommitment,
    holdout: tuple[LabeledRelationExample, ...],
) -> float:
    if prediction.input_ids_in_order != tuple(
        item.relation_input.content_id for item in holdout
    ):
        raise RelationInventionError("opened holdout input order differs")
    matches = tuple(
        expected is item.next_on
        for expected, item in zip(
            prediction.predicted_outcomes_in_order, holdout, strict=True
        )
    )
    return sum(matches) / len(matches)


def _require_candidate_not_lookup(
    candidate: RelationLawCandidate,
    examples: Iterable[LabeledRelationExample],
) -> None:
    forbidden = {
        item.observation_id for item in examples
    } | {
        item.relation_input.input_id for item in examples
    }
    if any(token in candidate.executable_definition for token in forbidden):
        raise RelationInventionError("candidate executable contains fixture identity")
    if candidate.executable_definition.count("next_on"):
        raise RelationInventionError("candidate executable contains an outcome lookup")


def _active_compiler_must_remain_closed(parent, patch: TheoryPatch) -> str:
    try:
        compile_patch(parent, patch)
    except ValueError as exc:
        if "no active compiler for coordinate: language" not in str(exc):
            raise
        return "LANGUAGE_COMPILER_CLOSED_CANDIDATE_BRANCH_ONLY"
    raise AssertionError("shadow relation unexpectedly compiled into active theory")


def run_bounded_relation_invention_mechanics_v1() -> dict[str, object]:
    """Run the deterministic public fixture and return a canonical report."""

    chronology: list[str] = []
    parent = initial_theory()
    parent_before = canonical_json(parent)
    discovery, validation, hard_negatives, vault = build_public_synthetic_fixture()
    profile, ontology_report, diagnostic_old_program = derive_ontology_report(
        discovery, validation
    )
    chronology.append("synthetic_profile_diagnostic_ladder_replayed")
    synthesis = synthesize_relation(discovery, validation, ontology_report)
    if synthesis.old_comparison_program != diagnostic_old_program:
        raise AssertionError("diagnostic and synthesis old comparators differ")
    chronology.append("candidate_synthesis_committed")
    prediction = preregister_holdout_predictions(synthesis, vault.inputs)
    chronology.append("holdout_prediction_preregistered")
    candidate, patch = _candidate_and_patch(
        synthesis,
        prediction,
        ontology_report,
        discovery,
        validation,
        hard_negatives,
    )
    _require_candidate_not_lookup(
        candidate, discovery + validation + hard_negatives
    )
    holdout = vault.open(
        prediction,
        expected_synthesis_commitment_id=synthesis.commitment_id,
    )
    chronology.append("process_local_holdout_opened")
    holdout_accuracy = _require_exact_prediction_replay(prediction, holdout)
    chronology.append("holdout_scored_without_candidate_change")
    hard_negative_accuracy = _accuracy(synthesis.selected_program, hard_negatives)
    if holdout_accuracy != 1.0:
        raise RelationInventionError("preregistered holdout prediction was falsified")
    if hard_negative_accuracy != 1.0:
        raise RelationInventionError("candidate failed a hard negative")

    compiler_status = _active_compiler_must_remain_closed(parent, patch)
    parent_after = canonical_json(parent)
    if parent_after != parent_before:
        raise AssertionError("shadow mechanics mutated the parent theory")
    if patch.ontology_report_id != ontology_report.report_id:
        raise AssertionError("language patch lost its diagnostic binding")
    if candidate.reduction_map.parent_version_id != parent.version_id:
        raise AssertionError("candidate reduction map lost its parent binding")

    all_fit_examples = discovery + validation
    compression_gain = _compression_gain(synthesis, all_fit_examples)
    old_holdout_accuracy = _accuracy(synthesis.old_comparison_program, holdout)
    candidate_id = stable_hash(candidate, prefix="relation_candidate_")
    measured_synthetic_metrics = {
        "synthetic_discovery_accuracy": synthesis.discovery_accuracy,
        "synthetic_validation_accuracy": synthesis.validation_accuracy,
        "process_local_holdout_accuracy": holdout_accuracy,
        "public_hard_negative_accuracy": hard_negative_accuracy,
        "synthetic_expressivity_gain": 1.0 - synthesis.old_comparison_accuracy,
        "synthetic_description_compression_gain": compression_gain,
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "claim_level": CLAIM_LEVEL,
        "fixture_version": FIXTURE_VERSION,
        "grammar_version": GRAMMAR_VERSION,
        "parent_version_id": parent.version_id,
        "parent_theory_bytes_preserved": parent_before == parent_after,
        "active_law_ids_preserved": [law.law_id for law in parent.relation_laws],
        "active_theory_mutated": False,
        "compiler_status": compiler_status,
        "ontology_verdict": ontology_report.verdict.value,
        "ontology_report_id": ontology_report.report_id,
        "residual_profile_id": profile.content_id,
        "diagnostic_steps": [item.value for item in ontology_report.checked_steps],
        "diagnostic_profile_source": (
            "frozen_public_synthetic_fixture_assertions_not_executed_"
            "repair_evaluators"
        ),
        "cheaper_repair_evaluators_executed": False,
        "synthesis_commitment_id": synthesis.commitment_id,
        "prediction_commitment_id": prediction.commitment_id,
        "candidate_id": candidate.candidate_id,
        "candidate_content_id": candidate_id,
        "candidate_kind": None,
        "candidate_symbol": candidate.symbol,
        "candidate_program": synthesis.selected_program.expression,
        "candidate_program_content_id": synthesis.selected_program.content_id,
        "candidate_program_node_count": synthesis.selected_program.node_count,
        "old_comparison_max_node_count": OLD_COMPARISON_MAX_NODES,
        "old_comparison_program": synthesis.old_comparison_program.expression,
        "old_comparison_accuracy": synthesis.old_comparison_accuracy,
        "old_comparison_holdout_accuracy": old_holdout_accuracy,
        "quotient_behavior_class_count": synthesis.exact_behavior_class_count,
        "discovery_count": len(discovery),
        "validation_count": len(validation),
        "hard_negative_count": len(hard_negatives),
        "holdout_count": len(holdout),
        "measured_synthetic_metrics": measured_synthetic_metrics,
        "unmeasured_gate_metrics": [
            "old_success_preservation",
            "limiting_case_reduction",
            "regression_cost",
        ],
        "reduction_map_declared_not_executed": True,
        "patch_id": patch.patch_id,
        "patch_content_id": patch.content_id,
        "reduction_map_id": candidate.reduction_map.reduction_id,
        "five_distinct_role_actor_ids": len(
            {item.actor_id for item in patch.authority_assignments}
        )
        == len(AuthorityRole),
        "role_actor_ids_external": False,
        "chronology": chronology,
        "holdout_labels_passed_to_synthesis_api": False,
        "sealed_holdout": False,
        "external_trust_root": False,
        "c1_pass": False,
        "c2_pass": False,
        "promotion_allowed": False,
        "nonclaims": list(NONCLAIMS),
    }
    report["report_content_id"] = stable_hash(
        report, prefix="bounded_relation_invention_report_"
    )
    return report


def require_candidate_mechanics_report(report: dict[str, object]) -> None:
    """Fail closed unless a report retains every shadow-only boundary."""

    if type(report) is not dict:
        raise RelationInventionError("mechanics report must be an exact dict")
    if report.get("status") != STATUS or report.get("claim_level") != CLAIM_LEVEL:
        raise RelationInventionError("mechanics report status differs")
    required_false = (
        "active_theory_mutated",
        "cheaper_repair_evaluators_executed",
        "holdout_labels_passed_to_synthesis_api",
        "role_actor_ids_external",
        "sealed_holdout",
        "external_trust_root",
        "c1_pass",
        "c2_pass",
        "promotion_allowed",
    )
    if any(report.get(name) is not False for name in required_false):
        raise RelationInventionError("mechanics report overclaims authority")
    if report.get("parent_theory_bytes_preserved") is not True:
        raise RelationInventionError("mechanics report lacks parent preservation")
    if report.get("nonclaims") != list(NONCLAIMS):
        raise RelationInventionError("mechanics report nonclaims differ")
    supplied_id = report.get("report_content_id")
    body = dict(report)
    body.pop("report_content_id", None)
    if supplied_id != stable_hash(
        body, prefix="bounded_relation_invention_report_"
    ):
        raise RelationInventionError("mechanics report content id differs")
    expected = run_bounded_relation_invention_mechanics_v1()
    if canonical_json(report) != canonical_json(expected):
        raise RelationInventionError("mechanics report differs from frozen replay")


def _main() -> NoReturn:
    report = run_bounded_relation_invention_mechanics_v1()
    require_candidate_mechanics_report(report)
    print(canonical_json(report))
    raise SystemExit(0)


if __name__ == "__main__":
    _main()


__all__ = [
    "CLAIM_LEVEL",
    "FIXTURE_VERSION",
    "GRAMMAR_VERSION",
    "NONCLAIMS",
    "OLD_COMPARISON_EXPRESSION",
    "OLD_COMPARISON_MAX_NODES",
    "PROGRAM_NODE_BITS",
    "ProcessLocalHoldoutVault",
    "Program",
    "PredictionCommitment",
    "RelationInput",
    "RelationInventionError",
    "SCHEMA_VERSION",
    "STATUS",
    "SYNTHESIS_MAX_NODES",
    "SynthesisCommitment",
    "build_public_synthetic_fixture",
    "derive_ontology_report",
    "enumerate_programs",
    "preregister_holdout_predictions",
    "require_candidate_mechanics_report",
    "run_bounded_relation_invention_mechanics_v1",
    "synthesize_relation",
]
