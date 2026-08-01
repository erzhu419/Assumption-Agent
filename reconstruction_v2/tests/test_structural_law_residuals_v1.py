from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import inspect

import pytest

from assumption_agent.generalized_structural_correspondence_v1 import (
    ConstraintParticipant,
    CorrespondenceDisposition,
    EvidenceSpanRef,
    ExactRational,
    GSCLSchemaRegistry,
    HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES,
    LawBinding,
    ObservableBinding,
    ObservableValueType,
    ObservationStatus,
    ResidualComponent,
    ResidualDisposition,
    RoleBinding,
    RoleTargetKind,
    StructuralConstraint,
    StructuralEpisode,
    StructuralObject,
    StructuralQuantity,
    StructuralRelation,
    TypedObservable,
    build_gscl_schema_registry_v1,
    compare_structural_bindings,
)
from assumption_agent.structural_law_residuals_v1 import (
    InteractionExpectation,
    MeasuredQuantity,
    ResidualPolicy,
    build_law_residual_receipt,
    evaluate_bound_law,
    evaluate_closed_balance,
    evaluate_equivariance,
    evaluate_low_order_interaction,
    evaluate_monotone_order,
    evaluate_path_composition,
    evaluate_transformed_law,
    verify_law_residual_receipt_trusted,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    build_universal_assumption_ontology_v1,
)


def _assert_disposition(
    evaluation: object,
    expected: ResidualDisposition,
) -> None:
    assert evaluation.disposition is expected
    assert evaluation.validate() == ()


def _quantity(value: int | None) -> MeasuredQuantity:
    return MeasuredQuantity(
        value=None if value is None else ExactRational(value),
        dimension="Mass",
        unit="kg",
    )


def _interaction_folds(
    pair_value: int,
) -> tuple[dict[frozenset[str], int], ...]:
    fold = {
        frozenset(): 0,
        frozenset({"a"}): 1,
        frozenset({"b"}): 1,
        frozenset({"a", "b"}): pair_value,
    }
    return (dict(fold), dict(fold))


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        (
            evaluate_equivariance(
                (1, 2), (2, 1), (1, 0), (1, 1)
            ),
            ResidualDisposition.SATISFIED,
        ),
        (
            evaluate_equivariance(
                (1, 2), (2, 2), (1, 0), (1, 1)
            ),
            ResidualDisposition.VIOLATED,
        ),
        (
            evaluate_equivariance(
                (1, 2), (2,), (1, 0), (1, 1)
            ),
            ResidualDisposition.NOT_APPLICABLE,
        ),
        (
            evaluate_equivariance(
                None, (2, 1), (1, 0), (1, 1)
            ),
            ResidualDisposition.INCONCLUSIVE,
        ),
    ],
)
def test_equivariance_all_four_dispositions(
    evaluation: object,
    expected: ResidualDisposition,
) -> None:
    _assert_disposition(evaluation, expected)


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        (
            evaluate_monotone_order(((1, 2),), direction=1),
            ResidualDisposition.SATISFIED,
        ),
        (
            evaluate_monotone_order(((2, 1),), direction=1),
            ResidualDisposition.VIOLATED,
        ),
        (
            evaluate_monotone_order(((1, 2),), direction=0),
            ResidualDisposition.NOT_APPLICABLE,
        ),
        (
            evaluate_monotone_order(None, direction=1),
            ResidualDisposition.INCONCLUSIVE,
        ),
    ],
)
def test_monotone_order_all_four_dispositions(
    evaluation: object,
    expected: ResidualDisposition,
) -> None:
    _assert_disposition(evaluation, expected)


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        (
            evaluate_closed_balance(
                _quantity(10),
                _quantity(12),
                (_quantity(3),),
                (_quantity(1),),
                (),
                (),
                boundary_id="system.boundary",
                boundary_complete=True,
            ),
            ResidualDisposition.SATISFIED,
        ),
        (
            evaluate_closed_balance(
                _quantity(10),
                _quantity(13),
                (_quantity(3),),
                (_quantity(1),),
                (),
                (),
                boundary_id="system.boundary",
                boundary_complete=True,
            ),
            ResidualDisposition.VIOLATED,
        ),
        (
            evaluate_closed_balance(
                _quantity(10),
                _quantity(12),
                (_quantity(3),),
                (_quantity(1),),
                (),
                (),
                boundary_id=None,
                boundary_complete=True,
            ),
            ResidualDisposition.NOT_APPLICABLE,
        ),
        (
            evaluate_closed_balance(
                None,
                _quantity(12),
                (_quantity(3),),
                (_quantity(1),),
                (),
                (),
                boundary_id="system.boundary",
                boundary_complete=True,
            ),
            ResidualDisposition.INCONCLUSIVE,
        ),
    ],
)
def test_closed_balance_all_four_dispositions(
    evaluation: object,
    expected: ResidualDisposition,
) -> None:
    _assert_disposition(evaluation, expected)


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        (
            evaluate_path_composition(
                ("a", "b"),
                {"a": "x", "b": "y"},
                {"x": "u", "y": "v"},
                {"a": "u", "b": "v"},
            ),
            ResidualDisposition.SATISFIED,
        ),
        (
            evaluate_path_composition(
                ("a", "b"),
                {"a": "x", "b": "y"},
                {"x": "u", "y": "v"},
                {"a": "u", "b": "u"},
            ),
            ResidualDisposition.VIOLATED,
        ),
        (
            evaluate_path_composition(
                ("a", "a"),
                {"a": "x"},
                {"x": "u"},
                {"a": "u"},
            ),
            ResidualDisposition.NOT_APPLICABLE,
        ),
        (
            evaluate_path_composition(
                ("a",),
                None,
                {"x": "u"},
                {"a": "u"},
            ),
            ResidualDisposition.INCONCLUSIVE,
        ),
    ],
)
def test_path_composition_all_four_dispositions(
    evaluation: object,
    expected: ResidualDisposition,
) -> None:
    _assert_disposition(evaluation, expected)


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        (
            evaluate_low_order_interaction(
                _interaction_folds(3),
                ("a", "b"),
                ("a", "b"),
                expected_relation=InteractionExpectation.COMPLEMENTARY,
                relation_threshold=Fraction(1, 2),
            ),
            ResidualDisposition.SATISFIED,
        ),
        (
            evaluate_low_order_interaction(
                _interaction_folds(2),
                ("a", "b"),
                ("a", "b"),
                expected_relation=InteractionExpectation.COMPLEMENTARY,
                relation_threshold=Fraction(1, 2),
            ),
            ResidualDisposition.VIOLATED,
        ),
        (
            evaluate_low_order_interaction(
                _interaction_folds(3),
                ("a", "b"),
                ("a", "b"),
                expected_relation=InteractionExpectation.COMPLEMENTARY,
                relation_threshold=Fraction(1, 2),
                common_utility_scale=False,
            ),
            ResidualDisposition.NOT_APPLICABLE,
        ),
        (
            evaluate_low_order_interaction(
                (_interaction_folds(3)[0],),
                ("a", "b"),
                ("a", "b"),
                expected_relation=InteractionExpectation.COMPLEMENTARY,
                relation_threshold=Fraction(1, 2),
            ),
            ResidualDisposition.INCONCLUSIVE,
        ),
    ],
)
def test_low_order_interaction_all_four_dispositions(
    evaluation: object,
    expected: ResidualDisposition,
) -> None:
    _assert_disposition(evaluation, expected)


FIXTURE_SOURCE = b"monotone receipt evidence"
FIXTURE_SOURCE_HASH = hashlib.sha256(FIXTURE_SOURCE).hexdigest()


def _receipt_fixture(
    *,
    lower: int = 1,
    upper: int = 2,
    unknown_direction: bool = False,
) -> tuple[object, object, StructuralEpisode, LawBinding, ResidualPolicy]:
    ontology = build_universal_assumption_ontology_v1()
    registry = build_gscl_schema_registry_v1(ontology)
    schema = registry.require_law("gscl.v1.t17_monotone_order")
    span = EvidenceSpanRef(
        span_id="span.receipt",
        source_sha256=FIXTURE_SOURCE_HASH,
        start_byte=0,
        end_byte=len(FIXTURE_SOURCE),
        span_sha256=FIXTURE_SOURCE_HASH,
    )
    evidence = (span.span_id,)
    lower_id = "receipt.lower.state"
    upper_id = "receipt.upper.state"
    relation_id = "receipt.partial.order"
    lower_quantity_id = "receipt.lower.quantity"
    upper_quantity_id = "receipt.upper.quantity"
    constraint_id = "receipt.monotone.constraint"
    objects = (
        StructuralObject(lower_id, "State", evidence),
        StructuralObject(upper_id, "State", evidence),
    )
    relation = StructuralRelation(
        relation_id,
        "PartialOrder",
        lower_id,
        upper_id,
        evidence,
        order_index=0,
    )
    quantities = (
        StructuralQuantity(
            lower_quantity_id,
            lower_id,
            "Score",
            "point",
            ExactRational(lower),
            evidence,
        ),
        StructuralQuantity(
            upper_quantity_id,
            upper_id,
            "Score",
            "point",
            ExactRational(upper),
            evidence,
        ),
    )
    pairs = TypedObservable(
        "comparable_output_pairs",
        ObservableValueType.COMPARABLE_PAIRS,
        {
            "pairs": [
                {
                    "lower": {
                        "numerator": lower,
                        "denominator": 1,
                    },
                    "upper": {
                        "numerator": upper,
                        "denominator": 1,
                    },
                }
            ]
        },
        evidence,
        dimension="Score",
        unit="point",
    )
    direction = TypedObservable(
        "declared_direction",
        ObservableValueType.DIRECTION,
        None if unknown_direction else {"direction": 1},
        () if unknown_direction else evidence,
        observation_status=(
            ObservationStatus.UNKNOWN
            if unknown_direction
            else ObservationStatus.OBSERVED
        ),
    )
    constraint = StructuralConstraint(
        constraint_id,
        "Monotonicity",
        (
            ConstraintParticipant(
                "lower_state",
                RoleTargetKind.OBJECT,
                lower_id,
            ),
            ConstraintParticipant(
                "upper_state",
                RoleTargetKind.OBJECT,
                upper_id,
            ),
            ConstraintParticipant(
                "order_relation",
                RoleTargetKind.RELATION,
                relation_id,
            ),
            ConstraintParticipant(
                "lower_value",
                RoleTargetKind.QUANTITY,
                lower_quantity_id,
            ),
            ConstraintParticipant(
                "upper_value",
                RoleTargetKind.QUANTITY,
                upper_quantity_id,
            ),
        ),
        ("comparable_output_pairs", "declared_direction"),
        evidence,
    )
    episode = StructuralEpisode(
        episode_id="receipt.monotone.episode",
        source_sha256=FIXTURE_SOURCE_HASH,
        evidence_spans=(span,),
        objects=objects,
        relations=(relation,),
        quantities=quantities,
        hyperrelations=(),
        constraints=(constraint,),
        observables=(pairs, direction),
        missing_observables=(
            ("declared_direction",) if unknown_direction else ()
        ),
    )
    targets = {
        "lower_state": lower_id,
        "upper_state": upper_id,
        "order_relation": relation_id,
        "lower_value": lower_quantity_id,
        "upper_value": upper_quantity_id,
        "monotone_constraint": constraint_id,
    }
    binding = LawBinding(
        binding_id="receipt.monotone.binding",
        law_id=schema.law_id,
        registry_hash=registry.registry_hash,
        schema_hash=schema.schema_hash,
        episode_hash=episode.episode_hash,
        role_bindings=tuple(
            RoleBinding(role_id, target_id, evidence)
            for role_id, target_id in targets.items()
        ),
        observable_bindings=tuple(
            ObservableBinding(
                observable.observable_id,
                observable.observable_hash,
            )
            for observable in episode.observables
        ),
    )
    policy = ResidualPolicy(law_id=schema.law_id)
    assert episode.validate() == ()
    return registry, schema, episode, binding, policy


def test_forged_verifier_and_false_satisfied_receipts_are_rejected() -> None:
    registry, schema, episode, binding, policy = _receipt_fixture(
        lower=2,
        upper=1,
    )
    evaluation = evaluate_bound_law(
        registry, schema, episode, binding, policy
    )
    assert evaluation.disposition is ResidualDisposition.VIOLATED
    receipt = build_law_residual_receipt(
        registry,
        schema,
        episode,
        binding,
        policy,
        receipt_id="receipt.monotone.result",
        evidence_span_ids=("span.receipt",),
    )
    assert verify_law_residual_receipt_trusted(
        receipt, registry, schema, episode, binding, policy
    ) == ()

    forged_verifier = replace(
        receipt, verifier_contract_hash="f" * 64
    )
    assert (
        "law_residual_verifier_hash_mismatch"
        in verify_law_residual_receipt_trusted(
            forged_verifier,
            registry,
            schema,
            episode,
            binding,
            policy,
        )
    )

    false_satisfied = replace(
        receipt,
        disposition=ResidualDisposition.SATISFIED,
        components=tuple(
            ResidualComponent(
                component_id=component.component_id,
                value=ExactRational(0),
                tolerance=component.tolerance,
            )
            for component in receipt.components
        ),
    )
    assert false_satisfied.validate(
        registry,
        schema,
        episode,
        binding,
        expected_policy_hash=policy.policy_hash,
    ) == ()
    assert (
        "trusted_primary_recomputation_mismatch"
        in verify_law_residual_receipt_trusted(
            false_satisfied,
            registry,
            schema,
            episode,
            binding,
            policy,
        )
    )
    forged_correspondence = compare_structural_bindings(
        registry,
        schema,
        episode,
        binding,
        false_satisfied,
        episode,
        binding,
        false_satisfied,
        correspondence_id="correspondence.forged.receipt",
        source_policy=policy,
        target_policy=policy,
    )
    assert (
        forged_correspondence.disposition
        is CorrespondenceDisposition.INCONCLUSIVE
    )
    assert (
        "source_receipt_trusted_recomputation_failed"
        in forged_correspondence.unresolved_constraints
    )


def test_abstention_correspondence_is_inconclusive_not_rejected() -> None:
    registry, schema, episode, binding, policy = _receipt_fixture(
        unknown_direction=True
    )
    evaluation = evaluate_bound_law(
        registry, schema, episode, binding, policy
    )
    assert evaluation.disposition is ResidualDisposition.INCONCLUSIVE
    receipt = build_law_residual_receipt(
        registry,
        schema,
        episode,
        binding,
        policy,
        receipt_id="receipt.abstention.result",
        evidence_span_ids=("span.receipt",),
    )

    correspondence = compare_structural_bindings(
        registry,
        schema,
        episode,
        binding,
        receipt,
        episode,
        binding,
        receipt,
        correspondence_id="correspondence.abstention",
        source_policy=policy,
        target_policy=policy,
    )

    assert (
        correspondence.disposition
        is CorrespondenceDisposition.INCONCLUSIVE
    )
    assert correspondence.disposition is not CorrespondenceDisposition.REJECTED
    assert "law_residual_abstention" in correspondence.unresolved_constraints
    assert "law_residual_satisfaction" not in correspondence.broken_constraints


def _rebind_episode(
    binding: LawBinding,
    episode: StructuralEpisode,
) -> LawBinding:
    return replace(
        binding,
        episode_hash=episode.episode_hash,
        observable_bindings=tuple(
            ObservableBinding(
                observable.observable_id,
                observable.observable_hash,
            )
            for observable in episode.observables
        ),
    )


def test_monotone_graph_and_observable_ledger_must_agree() -> None:
    registry, schema, episode, binding, policy = _receipt_fixture()

    contradicted_quantities = replace(
        episode,
        quantities=(
            replace(episode.quantities[0], value=ExactRational(100)),
            replace(episode.quantities[1], value=ExactRational(-100)),
        ),
    )
    contradicted_binding = _rebind_episode(
        binding, contradicted_quantities
    )
    with pytest.raises(
        PermissionError,
        match="semantic_monotone_.*quantity_mismatch",
    ):
        evaluate_bound_law(
            registry,
            schema,
            contradicted_quantities,
            contradicted_binding,
            policy,
        )

    reversed_relation = replace(
        episode,
        relations=(
            replace(
                episode.relations[0],
                source_object_id=episode.relations[0].target_object_id,
                target_object_id=episode.relations[0].source_object_id,
            ),
        ),
    )
    reversed_binding = _rebind_episode(binding, reversed_relation)
    with pytest.raises(
        PermissionError,
        match="semantic_monotone_relation_direction_mismatch",
    ):
        evaluate_bound_law(
            registry,
            schema,
            reversed_relation,
            reversed_binding,
            policy,
        )


def test_role_swap_changes_evaluation_input_and_fails_semantics() -> None:
    registry, schema, episode, binding, policy = _receipt_fixture()
    targets = {
        row.role_id: row.target_id for row in binding.role_bindings
    }
    swapped = replace(
        binding,
        role_bindings=tuple(
            replace(
                row,
                target_id=(
                    targets["upper_state"]
                    if row.role_id == "lower_state"
                    else targets["lower_state"]
                    if row.role_id == "upper_state"
                    else row.target_id
                ),
            )
            for row in binding.role_bindings
        ),
    )

    assert swapped.binding_hash != binding.binding_hash
    assert swapped.evaluation_input_hash != binding.evaluation_input_hash
    with pytest.raises(PermissionError, match="semantic_"):
        evaluate_bound_law(
            registry, schema, episode, swapped, policy
        )


def test_all_evaluation_entrypoints_reject_nonfrozen_registry() -> None:
    registry, schema, episode, binding, policy = _receipt_fixture()
    remapped_schema = replace(
        schema, residual_function_id="evil.residual.fn"
    )
    remapped_registry = GSCLSchemaRegistry(
        ontology_hash=registry.ontology_hash,
        schemas=tuple(
            remapped_schema if row.law_id == schema.law_id else row
            for row in registry.schemas
        ),
    )
    remapped_binding = replace(
        binding,
        registry_hash=remapped_registry.registry_hash,
        schema_hash=remapped_schema.schema_hash,
    )
    calls = (
        lambda: evaluate_bound_law(
            remapped_registry,
            remapped_schema,
            episode,
            remapped_binding,
            policy,
        ),
        lambda: evaluate_transformed_law(
            remapped_registry,
            remapped_schema,
            episode,
            remapped_binding,
            policy,
            transformation_id="direction_flip",
        ),
        lambda: build_law_residual_receipt(
            remapped_registry,
            remapped_schema,
            episode,
            remapped_binding,
            policy,
            receipt_id="receipt.nonfrozen.registry",
            evidence_span_ids=("span.receipt",),
        ),
    )
    for call in calls:
        with pytest.raises(PermissionError):
            call()

    wrong_ontology_registry = replace(
        registry, ontology_hash="f" * 64
    )
    wrong_ontology_binding = replace(
        binding,
        registry_hash=wrong_ontology_registry.registry_hash,
    )
    with pytest.raises(
        PermissionError,
        match="gscl_registry_frozen_ontology_mismatch",
    ):
        evaluate_bound_law(
            wrong_ontology_registry,
            schema,
            episode,
            wrong_ontology_binding,
            policy,
        )


def test_receipt_builder_accepts_only_recomputation_inputs() -> None:
    signature = inspect.signature(build_law_residual_receipt)
    transformed_signature = inspect.signature(evaluate_transformed_law)
    assert "evaluation" not in signature.parameters
    assert "contrastive_evaluations" not in signature.parameters
    assert "transformation_overrides" not in signature.parameters
    assert (
        "observable_payload_overrides"
        not in transformed_signature.parameters
    )

    registry, schema, episode, binding, policy = _receipt_fixture()
    receipt = build_law_residual_receipt(
        registry,
        schema,
        episode,
        binding,
        policy,
        receipt_id="receipt.internal.recomputation",
        evidence_span_ids=("span.receipt",),
    )
    assert verify_law_residual_receipt_trusted(
        receipt,
        registry,
        schema,
        episode,
        binding,
        policy,
    ) == ()
    assert {
        row.transformation_id
        for row in receipt.contrastive_residuals
    } == set(schema.hard_negative_transformations)
    assert {
        row.operator_contract_hash
        for row in receipt.contrastive_residuals
    } == {
        HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES[row]
        for row in schema.hard_negative_transformations
    }
    assert len(
        {
            row.transformed_input_hash
            for row in receipt.contrastive_residuals
        }
    ) == len(schema.hard_negative_transformations)

    first = receipt.contrastive_residuals[0]
    forged = replace(
        receipt,
        contrastive_residuals=(
            replace(first, operator_contract_hash="f" * 64),
            *receipt.contrastive_residuals[1:],
        ),
    )
    assert (
        "law_residual_contrastive_operator_contract_mismatch"
        in forged.validate(
            registry,
            schema,
            episode,
            binding,
            expected_policy_hash=policy.policy_hash,
        )
    )
