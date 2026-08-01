"""Exact, deterministic and offline law residuals for GSCL Phase 0."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from itertools import combinations
from typing import Any, Mapping, Sequence

from .generalized_structural_correspondence_v1 import (
    ContrastiveResidual,
    ExactRational,
    ExecutableLawSchema,
    GSCLSchemaRegistry,
    HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES,
    LawBinding,
    LawKind,
    LawResidualReceipt,
    ObservationStatus,
    ObservableValueType,
    ResidualComponent,
    ResidualDisposition,
    RoleTargetKind,
    StructuralConstraint,
    StructuralEpisode,
    StructuralQuantity,
    StructuralRelation,
    TypedObservable,
    strict_content_hash,
    validate_law_binding,
)


ExactLike = ExactRational | Fraction | int


class InteractionExpectation(str, Enum):
    COMPLEMENTARY = "complementary"
    REDUNDANT = "redundant"
    ADDITIVE = "additive"


@dataclass(frozen=True)
class ResidualPolicy:
    """Study-bound exact policy; no tolerance is accepted from a receipt."""

    law_id: str
    tolerance: ExactRational = ExactRational(0)
    require_involution: bool = True
    relation_threshold: ExactRational = ExactRational(0)
    high_order_tolerance: ExactRational = ExactRational(0)
    common_utility_scale: bool = True
    policy_version: str = "gscl.residual.policy.v1"

    @property
    def policy_hash(self) -> str:
        return strict_content_hash(self.safe_payload())

    def validate(self, schema: ExecutableLawSchema) -> tuple[str, ...]:
        issues: list[str] = []
        if self.law_id != schema.law_id:
            issues.append("residual_policy_law_mismatch")
        if (
            not isinstance(self.tolerance, ExactRational)
            or self.tolerance.fraction < 0
        ):
            issues.append("residual_policy_tolerance_invalid")
        if (
            not isinstance(self.relation_threshold, ExactRational)
            or self.relation_threshold.fraction < 0
        ):
            issues.append("residual_policy_relation_threshold_invalid")
        if (
            not isinstance(self.high_order_tolerance, ExactRational)
            or self.high_order_tolerance.fraction < 0
        ):
            issues.append("residual_policy_high_order_tolerance_invalid")
        if not isinstance(self.require_involution, bool):
            issues.append("residual_policy_involution_invalid")
        if not isinstance(self.common_utility_scale, bool):
            issues.append("residual_policy_utility_scale_invalid")
        if (
            not isinstance(self.policy_version, str)
            or not self.policy_version.startswith("gscl.")
        ):
            issues.append("residual_policy_version_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "tolerance": self.tolerance.safe_payload(),
            "require_involution": self.require_involution,
            "relation_threshold": self.relation_threshold.safe_payload(),
            "high_order_tolerance": (
                self.high_order_tolerance.safe_payload()
            ),
            "common_utility_scale": self.common_utility_scale,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class LawEvaluation:
    disposition: ResidualDisposition
    components: tuple[ResidualComponent, ...] = ()
    missing_observables: tuple[str, ...] = ()
    applicability_failures: tuple[str, ...] = ()
    evaluation_input_hash: str | None = None
    policy_hash: str | None = None

    def validate(self, *, require_bound: bool = False) -> tuple[str, ...]:
        issues: list[str] = []
        if not isinstance(self.disposition, ResidualDisposition):
            issues.append("law_evaluation_disposition_invalid")
        if not isinstance(self.components, tuple):
            issues.append("law_evaluation_components_invalid")
            component_ids: tuple[str, ...] = ()
        else:
            component_ids = tuple(
                component.component_id for component in self.components
            )
            if len(component_ids) != len(set(component_ids)):
                issues.append("law_evaluation_components_duplicate")
            issues.extend(
                issue
                for component in self.components
                for issue in component.validate()
            )
        for values, issue in (
            (
                self.missing_observables,
                "law_evaluation_missing_observables_invalid",
            ),
            (
                self.applicability_failures,
                "law_evaluation_applicability_failures_invalid",
            ),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != len(set(values))
                or any(
                    not isinstance(value, str)
                    or not value
                    or value.strip() != value
                    for value in values
                )
            ):
                issues.append(issue)
        if (
            self.disposition is ResidualDisposition.INCONCLUSIVE
            and (
                not self.missing_observables
                or self.applicability_failures
                or self.components
            )
        ):
            issues.append("law_evaluation_inconclusive_contract_invalid")
        if (
            self.disposition is ResidualDisposition.NOT_APPLICABLE
            and (
                not self.applicability_failures
                or self.missing_observables
                or self.components
            )
        ):
            issues.append("law_evaluation_not_applicable_contract_invalid")
        if (
            self.disposition
            in {
                ResidualDisposition.SATISFIED,
                ResidualDisposition.VIOLATED,
            }
            and (
                not component_ids
                or self.missing_observables
                or self.applicability_failures
                or (
                    all(
                        component.value.fraction
                        <= component.tolerance.fraction
                        for component in self.components
                    )
                    != (
                        self.disposition
                        is ResidualDisposition.SATISFIED
                    )
                )
            )
        ):
            issues.append("law_evaluation_decided_contract_invalid")
        hashes = (self.evaluation_input_hash, self.policy_hash)
        if require_bound and any(
            not isinstance(value, str) or len(value) != 64
            for value in hashes
        ):
            issues.append("law_evaluation_binding_missing")
        if not require_bound and any(value is not None for value in hashes):
            if any(
                not isinstance(value, str) or len(value) != 64
                for value in hashes
            ):
                issues.append("law_evaluation_binding_invalid")
        return tuple(sorted(set(issues)))


def _fraction(value: ExactLike) -> Fraction:
    return ExactRational.from_value(value).fraction


def _nonnegative_fraction(value: ExactLike, *, name: str) -> Fraction:
    result = _fraction(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _component(
    component_id: str,
    value: ExactLike,
    tolerance: ExactLike = 0,
) -> ResidualComponent:
    return ResidualComponent(
        component_id=component_id,
        value=ExactRational.from_value(value),
        tolerance=ExactRational.from_value(tolerance),
    )


def _decided(
    components: Sequence[ResidualComponent],
) -> LawEvaluation:
    evaluation = LawEvaluation(
        disposition=(
            ResidualDisposition.SATISFIED
            if all(
                component.value.fraction
                <= component.tolerance.fraction
                for component in components
            )
            else ResidualDisposition.VIOLATED
        ),
        components=tuple(components),
    )
    assert not evaluation.validate()
    return evaluation


def _inconclusive(*missing: str) -> LawEvaluation:
    evaluation = LawEvaluation(
        disposition=ResidualDisposition.INCONCLUSIVE,
        missing_observables=tuple(sorted(set(missing))),
    )
    assert not evaluation.validate()
    return evaluation


def _not_applicable(*failures: str) -> LawEvaluation:
    evaluation = LawEvaluation(
        disposition=ResidualDisposition.NOT_APPLICABLE,
        applicability_failures=tuple(sorted(set(failures))),
    )
    assert not evaluation.validate()
    return evaluation


def evaluate_equivariance(
    outputs_before: Sequence[ExactLike] | None,
    outputs_after: Sequence[ExactLike] | None,
    output_permutation: Sequence[int] | None,
    output_signs: Sequence[int] | None,
    *,
    tolerance: ExactLike = 0,
    require_involution: bool = True,
) -> LawEvaluation:
    tolerance_value = _nonnegative_fraction(
        tolerance, name="tolerance"
    )
    if outputs_before is None or outputs_after is None:
        return _inconclusive("paired_outputs")
    if output_permutation is None or output_signs is None:
        return _inconclusive("declared_output_action")
    coordinate_count = len(outputs_before)
    if coordinate_count == 0:
        return _inconclusive("paired_outputs")
    if (
        len(outputs_after) != coordinate_count
        or len(output_permutation) != coordinate_count
        or len(output_signs) != coordinate_count
    ):
        return _not_applicable("output_coordinates_are_comparable")
    if (
        any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in output_permutation
        )
        or set(output_permutation) != set(range(coordinate_count))
        or any(
            not isinstance(sign, int)
            or isinstance(sign, bool)
            or sign not in {-1, 1}
            for sign in output_signs
        )
    ):
        return _not_applicable("finite_output_action_is_valid")
    before = tuple(_fraction(value) for value in outputs_before)
    after = tuple(_fraction(value) for value in outputs_after)
    residual = max(
        abs(
            after[index]
            - output_signs[index]
            * before[output_permutation[index]]
        )
        for index in range(coordinate_count)
    )
    involution_failures = 0
    if require_involution:
        for index in range(coordinate_count):
            first = output_permutation[index]
            if (
                output_permutation[first] != index
                or output_signs[index] * output_signs[first] != 1
            ):
                involution_failures += 1
    return _decided(
        (
            _component(
                "equivariance_max_abs_residual",
                residual,
                tolerance_value,
            ),
            _component(
                "equivariance_involution_failure_count",
                involution_failures,
                0,
            ),
        )
    )


def evaluate_monotone_order(
    comparable_output_pairs: Sequence[
        tuple[ExactLike, ExactLike]
    ]
    | None,
    *,
    direction: int | None,
    tolerance: ExactLike = 0,
) -> LawEvaluation:
    tolerance_value = _nonnegative_fraction(
        tolerance, name="tolerance"
    )
    if comparable_output_pairs is None or not comparable_output_pairs:
        return _inconclusive("comparable_output_pairs")
    if direction is None:
        return _inconclusive("declared_direction")
    if (
        not isinstance(direction, int)
        or isinstance(direction, bool)
        or direction not in {-1, 1}
    ):
        return _not_applicable("declared_direction_is_valid")
    residuals = tuple(
        max(
            Fraction(0),
            -direction * (_fraction(upper) - _fraction(lower)),
        )
        for lower, upper in comparable_output_pairs
    )
    violations = sum(
        residual > tolerance_value for residual in residuals
    )
    return _decided(
        (
            _component(
                "monotone_max_order_residual",
                max(residuals),
                tolerance_value,
            ),
            _component("monotone_violation_count", violations, 0),
        )
    )


@dataclass(frozen=True)
class MeasuredQuantity:
    value: ExactRational | None
    dimension: str
    unit: str

    @classmethod
    def measured(
        cls,
        value: ExactLike,
        *,
        dimension: str,
        unit: str,
    ) -> "MeasuredQuantity":
        return cls(
            value=ExactRational.from_value(value),
            dimension=dimension,
            unit=unit,
        )


def evaluate_closed_balance(
    storage_before: MeasuredQuantity | None,
    storage_after: MeasuredQuantity | None,
    inflows: Sequence[MeasuredQuantity] | None,
    outflows: Sequence[MeasuredQuantity] | None,
    sources: Sequence[MeasuredQuantity] | None,
    sinks: Sequence[MeasuredQuantity] | None,
    *,
    boundary_id: str | None,
    boundary_complete: bool,
    tolerance: ExactLike = 0,
) -> LawEvaluation:
    tolerance_value = _nonnegative_fraction(
        tolerance, name="tolerance"
    )
    if boundary_id is None or not isinstance(boundary_id, str):
        return _not_applicable("system_boundary_is_explicit")
    if not boundary_complete:
        return _not_applicable("unobserved_boundary_flow_is_absent")
    if storage_before is None or storage_after is None:
        return _inconclusive("storage_delta")
    ledgers = (inflows, outflows, sources, sinks)
    if any(ledger is None for ledger in ledgers):
        return _inconclusive(
            "complete_inflow_outflow_source_sink_ledger"
        )
    assert inflows is not None
    assert outflows is not None
    assert sources is not None
    assert sinks is not None
    quantities = (
        storage_before,
        storage_after,
        *inflows,
        *outflows,
        *sources,
        *sinks,
    )
    if any(
        not isinstance(quantity.value, ExactRational)
        for quantity in quantities
    ):
        return _inconclusive("complete_quantity_values")
    if len({quantity.dimension for quantity in quantities}) != 1 or len(
        {quantity.unit for quantity in quantities}
    ) != 1:
        return _not_applicable("compatible_dimensions_and_units")

    def total(rows: Sequence[MeasuredQuantity]) -> Fraction:
        return sum(
            (
                row.value.fraction
                for row in rows
                if isinstance(row.value, ExactRational)
            ),
            Fraction(0),
        )

    assert isinstance(storage_before.value, ExactRational)
    assert isinstance(storage_after.value, ExactRational)
    observed_delta = (
        storage_after.value.fraction
        - storage_before.value.fraction
    )
    ledger_delta = (
        total(inflows)
        - total(outflows)
        + total(sources)
        - total(sinks)
    )
    return _decided(
        (
            _component(
                "closed_balance_abs_residual",
                abs(observed_delta - ledger_delta),
                tolerance_value,
            ),
        )
    )


def evaluate_path_composition(
    domain: Sequence[str] | None,
    first_map: Mapping[str, str] | None,
    second_map: Mapping[str, str] | None,
    direct_map: Mapping[str, str] | None,
) -> LawEvaluation:
    if domain is None or not domain:
        return _inconclusive("finite_domain")
    if first_map is None or second_map is None:
        return _inconclusive("typed_composable_maps")
    if direct_map is None:
        return _inconclusive("direct_map")
    if (
        len(domain) != len(set(domain))
        or any(
            not isinstance(value, str)
            or not value
            or value.strip() != value
            for value in domain
        )
    ):
        return _not_applicable("finite_domain_is_valid")
    if any(item not in first_map for item in domain):
        return _inconclusive("first_map_domain_coverage")
    first_outputs = tuple(first_map[item] for item in domain)
    if any(item not in second_map for item in first_outputs):
        return _inconclusive("second_map_domain_coverage")
    if any(item not in direct_map for item in domain):
        return _inconclusive("direct_map_domain_coverage")
    mismatch_count = sum(
        second_map[first_map[item]] != direct_map[item]
        for item in domain
    )
    return _decided(
        (
            _component(
                "path_composition_mismatch_rate",
                Fraction(mismatch_count, len(domain)),
                0,
            ),
        )
    )


def _all_subsets(
    components: Sequence[str],
) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(subset)
        for size in range(len(components) + 1)
        for subset in combinations(components, size)
    )


def mobius_coefficients(
    utilities: Mapping[frozenset[str], ExactLike],
    components: Sequence[str],
) -> dict[frozenset[str], Fraction]:
    component_tuple = tuple(components)
    expected = frozenset(_all_subsets(component_tuple))
    normalized = {
        frozenset(key): _fraction(value)
        for key, value in utilities.items()
    }
    if frozenset(normalized) != expected:
        missing = expected - frozenset(normalized)
        extra = frozenset(normalized) - expected
        raise ValueError(
            "subset lattice must be complete: "
            f"missing={len(missing)},extra={len(extra)}"
        )
    coefficients: dict[frozenset[str], Fraction] = {}
    for subset in _all_subsets(component_tuple):
        coefficient = Fraction(0)
        for nested in _all_subsets(tuple(sorted(subset))):
            sign = -1 if (len(subset) - len(nested)) % 2 else 1
            coefficient += sign * normalized[nested]
        coefficients[subset] = coefficient
    return coefficients


def evaluate_low_order_interaction(
    held_fold_utilities: Sequence[
        Mapping[frozenset[str], ExactLike]
    ]
    | None,
    components: Sequence[str],
    designated_pair: tuple[str, str],
    *,
    expected_relation: InteractionExpectation,
    relation_threshold: ExactLike = 0,
    high_order_tolerance: ExactLike = 0,
    common_utility_scale: bool = True,
) -> LawEvaluation:
    relation_threshold_value = _nonnegative_fraction(
        relation_threshold, name="relation_threshold"
    )
    high_order_tolerance_value = _nonnegative_fraction(
        high_order_tolerance, name="high_order_tolerance"
    )
    component_tuple = tuple(components)
    if (
        len(component_tuple) < 2
        or len(component_tuple) != len(set(component_tuple))
        or any(
            not isinstance(component, str)
            or not component
            or component.strip() != component
            for component in component_tuple
        )
    ):
        return _not_applicable("at_least_two_distinct_components")
    if (
        len(designated_pair) != 2
        or designated_pair[0] == designated_pair[1]
        or any(item not in component_tuple for item in designated_pair)
    ):
        return _not_applicable("designated_pair_is_valid")
    if not isinstance(expected_relation, InteractionExpectation):
        return _not_applicable("expected_relation_is_declared")
    if not common_utility_scale:
        return _not_applicable("common_utility_scale")
    if held_fold_utilities is None or len(held_fold_utilities) < 2:
        return _inconclusive("held_fold_measurements")
    try:
        fold_coefficients = [
            mobius_coefficients(utilities, component_tuple)
            for utilities in held_fold_utilities
        ]
    except (TypeError, ValueError):
        return _inconclusive("complete_subset_utilities")
    pair_key = frozenset(designated_pair)
    pair_coefficients = tuple(
        coefficients[pair_key]
        for coefficients in fold_coefficients
    )
    if expected_relation is InteractionExpectation.COMPLEMENTARY:
        pair_relation_residual = max(
            max(Fraction(0), relation_threshold_value - value)
            for value in pair_coefficients
        )
    elif expected_relation is InteractionExpectation.REDUNDANT:
        pair_relation_residual = max(
            max(Fraction(0), relation_threshold_value + value)
            for value in pair_coefficients
        )
    else:
        pair_relation_residual = max(
            max(Fraction(0), abs(value) - relation_threshold_value)
            for value in pair_coefficients
        )
    high_order_magnitudes = tuple(
        abs(value)
        for coefficients in fold_coefficients
        for subset, value in coefficients.items()
        if len(subset) > 2
    )
    high_order_max = (
        max(high_order_magnitudes)
        if high_order_magnitudes
        else Fraction(0)
    )
    high_order_excess = max(
        Fraction(0),
        high_order_max - high_order_tolerance_value,
    )
    return _decided(
        (
            _component(
                "interaction_pair_relation_residual",
                pair_relation_residual,
                0,
            ),
            _component(
                "interaction_high_order_excess",
                high_order_excess,
                0,
            ),
        )
    )


def _decode_rational(payload: Any) -> ExactRational:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"denominator", "numerator"}
    ):
        raise TypeError("rational payload keys invalid")
    return ExactRational(
        numerator=payload["numerator"],
        denominator=payload["denominator"],
    )


def _decode_vector(observable: TypedObservable) -> tuple[ExactRational, ...]:
    payload = observable.value_payload
    if not isinstance(payload, dict) or set(payload) != {"values"}:
        raise TypeError("vector payload invalid")
    values = payload["values"]
    if not isinstance(values, list):
        raise TypeError("vector values invalid")
    return tuple(_decode_rational(value) for value in values)


def _decode_map(observable: TypedObservable) -> dict[str, str]:
    payload = observable.value_payload
    if not isinstance(payload, dict) or set(payload) != {"rows"}:
        raise TypeError("map payload invalid")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise TypeError("map rows invalid")
    result: dict[str, str] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"source", "target"}
            or not isinstance(row["source"], str)
            or not isinstance(row["target"], str)
            or row["source"] in result
        ):
            raise TypeError("map row invalid")
        result[row["source"]] = row["target"]
    return result


def _decode_string_values(observable: TypedObservable) -> tuple[str, ...]:
    payload = observable.value_payload
    if not isinstance(payload, dict) or set(payload) != {"values"}:
        raise TypeError("string-list payload invalid")
    values = payload["values"]
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
    ):
        raise TypeError("string-list values invalid")
    return tuple(values)


def _decode_quantity(
    payload: Any, *, dimension: str, unit: str
) -> MeasuredQuantity:
    return MeasuredQuantity(
        value=_decode_rational(payload),
        dimension=dimension,
        unit=unit,
    )


def _decode_fold_utilities(
    observable: TypedObservable,
) -> tuple[dict[frozenset[str], ExactRational], ...]:
    payload = observable.value_payload
    if not isinstance(payload, dict) or set(payload) != {"folds"}:
        raise TypeError("fold payload invalid")
    folds = payload["folds"]
    if not isinstance(folds, list):
        raise TypeError("fold list invalid")
    decoded: list[dict[frozenset[str], ExactRational]] = []
    for fold in folds:
        if not isinstance(fold, dict) or set(fold) != {"rows"}:
            raise TypeError("fold row container invalid")
        rows = fold["rows"]
        if not isinstance(rows, list):
            raise TypeError("fold rows invalid")
        values: dict[frozenset[str], ExactRational] = {}
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"subset", "utility"}
                or not isinstance(row["subset"], list)
                or any(
                    not isinstance(value, str)
                    for value in row["subset"]
                )
            ):
                raise TypeError("utility row invalid")
            subset = frozenset(row["subset"])
            if len(subset) != len(row["subset"]) or subset in values:
                raise TypeError("utility subset invalid")
            values[subset] = _decode_rational(row["utility"])
        decoded.append(values)
    return tuple(decoded)


def _require_value_type(
    observable: TypedObservable,
    expected: ObservableValueType,
) -> None:
    if observable.value_type is not expected:
        raise TypeError(
            f"expected {expected.value}, got {observable.value_type.value}"
        )


def _observables_for_binding(
    episode: StructuralEpisode,
    binding: LawBinding,
) -> dict[str, TypedObservable]:
    return {
        row.observable_id: episode.require_observable(
            row.observable_id
        )
        for row in binding.observable_bindings
    }


def _map_payload(values: Mapping[str, str]) -> dict[str, Any]:
    return {
        "rows": [
            {"source": source, "target": target}
            for source, target in sorted(values.items())
        ]
    }


def _fold_payload(
    folds: Sequence[Mapping[frozenset[str], ExactRational]],
) -> dict[str, Any]:
    return {
        "folds": [
            {
                "rows": [
                    {
                        "subset": list(sorted(subset)),
                        "utility": value.safe_payload(),
                    }
                    for subset, value in sorted(
                        fold.items(),
                        key=lambda row: (
                            len(row[0]),
                            tuple(sorted(row[0])),
                        ),
                    )
                ]
            }
            for fold in folds
        ]
    }


def _canonical_hard_negative_overrides(
    schema: ExecutableLawSchema,
    observables: Mapping[str, TypedObservable],
    transformation_id: str,
) -> dict[str, Any]:
    """Derive one frozen operator from the committed base observables."""

    if transformation_id not in schema.hard_negative_transformations:
        raise PermissionError("hard-negative transformation is not frozen")
    if transformation_id not in HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES:
        raise PermissionError("hard-negative operator contract is unknown")

    if transformation_id == "output_sign_flip":
        payload = observables["output_action"].value_payload
        if (
            not isinstance(payload, dict)
            or set(payload) != {"permutation", "signs"}
            or not isinstance(payload["permutation"], list)
            or not isinstance(payload["signs"], list)
            or not payload["signs"]
        ):
            raise PermissionError("hard-negative base action is invalid")
        signs = list(payload["signs"])
        signs[0] = -signs[0]
        return {
            "output_action": {
                "permutation": list(payload["permutation"]),
                "signs": signs,
            }
        }
    if transformation_id == "role_swap_input_output":
        return {
            "outputs_after": observables[
                "outputs_before"
            ].value_payload
        }
    if transformation_id == "direction_flip":
        payload = observables["declared_direction"].value_payload
        if (
            not isinstance(payload, dict)
            or set(payload) != {"direction"}
            or payload["direction"] not in {-1, 1}
        ):
            raise PermissionError(
                "hard-negative base direction is invalid"
            )
        return {
            "declared_direction": {
                "direction": -payload["direction"]
            }
        }
    if transformation_id == "lower_upper_role_swap":
        pairs = _decode_comparable_pairs(
            observables["comparable_output_pairs"]
        )
        return {
            "comparable_output_pairs": {
                "pairs": [
                    {
                        "lower": upper.safe_payload(),
                        "upper": lower.safe_payload(),
                    }
                    for lower, upper in pairs
                ]
            }
        }
    if transformation_id in {
        "delete_boundary_flow",
        "flow_sign_flip",
    }:
        payload = observables["quantity_ledger"].value_payload
        if not isinstance(payload, dict):
            raise PermissionError(
                "hard-negative base quantity ledger is invalid"
            )
        ledger = {
            key: (
                list(value) if isinstance(value, list) else value
            )
            for key, value in payload.items()
        }
        if transformation_id == "delete_boundary_flow":
            for key in ("inflows", "sources"):
                values = ledger.get(key)
                if isinstance(values, list) and values:
                    ledger[key] = values[1:]
                    break
        else:
            ledger["inflows"], ledger["outflows"] = (
                ledger["outflows"],
                ledger["inflows"],
            )
            ledger["sources"], ledger["sinks"] = (
                ledger["sinks"],
                ledger["sources"],
            )
        return {"quantity_ledger": ledger}
    if transformation_id in {
        "intermediate_map_substitution",
        "path_order_reversal",
    }:
        domain = _decode_string_values(observables["finite_domain"])
        first = _decode_map(observables["first_map"])
        if transformation_id == "path_order_reversal":
            if len(domain) < 2:
                raise PermissionError(
                    "path reversal requires two domain rows"
                )
            left, right = domain[0], domain[1]
            first[left], first[right] = first[right], first[left]
            return {"first_map": _map_payload(first)}
        second = _decode_map(observables["second_map"])
        source_ref = _role_ref("source_state")
        nonanchor = [value for value in domain if value != source_ref]
        if not nonanchor:
            raise PermissionError(
                "intermediate substitution requires an auxiliary row"
            )
        intermediate = first[nonanchor[0]]
        second[intermediate] = "local:hard_negative_substitute"
        return {"second_map": _map_payload(second)}
    if transformation_id in {
        "interaction_sign_flip",
        "unmodeled_third_order_term",
    }:
        components = tuple(
            _decode_string_values(observables["components"])
        )
        pair = tuple(
            _decode_string_values(observables["designated_pair"])
        )
        if len(pair) != 2:
            raise PermissionError(
                "hard-negative designated pair is invalid"
            )
        decoded = _decode_fold_utilities(
            observables["held_fold_utilities"]
        )
        transformed_folds: list[
            dict[frozenset[str], ExactRational]
        ] = []
        pair_set = frozenset(pair)
        full_set = frozenset(components)
        for fold in decoded:
            transformed = dict(fold)
            if transformation_id == "interaction_sign_flip":
                coefficient = (
                    fold[pair_set].fraction
                    - fold[frozenset({pair[0]})].fraction
                    - fold[frozenset({pair[1]})].fraction
                    + fold[frozenset()].fraction
                )
                for subset, value in fold.items():
                    if pair_set <= subset:
                        transformed[subset] = ExactRational.from_value(
                            value.fraction - 2 * coefficient
                        )
            else:
                transformed[full_set] = ExactRational.from_value(
                    transformed[full_set].fraction + 1
                )
            transformed_folds.append(transformed)
        return {
            "held_fold_utilities": _fold_payload(transformed_folds)
        }
    raise PermissionError("hard-negative operator is not implemented")


_CONSTRAINT_ROLE_CONTRACT: Mapping[LawKind, tuple[str, tuple[str, ...]]] = {
    LawKind.EQUIVARIANCE: (
        "equivariance_constraint",
        (
            "input_before",
            "input_after",
            "transformation",
            "output_before",
            "output_after",
        ),
    ),
    LawKind.MONOTONE_ORDER: (
        "monotone_constraint",
        (
            "lower_state",
            "upper_state",
            "order_relation",
            "lower_value",
            "upper_value",
        ),
    ),
    LawKind.CLOSED_BALANCE: (
        "balance_constraint",
        (
            "system_boundary",
            "storage_before",
            "storage_after",
            "flow_ledger",
        ),
    ),
    LawKind.PATH_COMPOSITION: (
        "path_constraint",
        (
            "source_state",
            "target_state",
            "composed_path",
            "direct_path",
        ),
    ),
    LawKind.LOW_ORDER_INTERACTION: (
        "interaction_constraint",
        (
            "component_a",
            "component_b",
            "component_c",
            "utility_ledger",
        ),
    ),
}


def _role_ref(role_id: str) -> str:
    return f"role:{role_id}"


def _role_target_ids(binding: LawBinding) -> dict[str, str]:
    return {
        row.role_id: row.target_id for row in binding.role_bindings
    }


def _observed(observable: TypedObservable) -> bool:
    return observable.observation_status is not ObservationStatus.UNKNOWN


def _quantity_matches_observable(
    quantity: StructuralQuantity,
    value: ExactRational,
    observable: TypedObservable,
) -> bool:
    return (
        quantity.value == value
        and quantity.dimension == observable.dimension
        and quantity.unit == observable.unit
    )


def _decode_comparable_pairs(
    observable: TypedObservable,
) -> tuple[tuple[ExactRational, ExactRational], ...]:
    payload = observable.value_payload
    if (
        not isinstance(payload, dict)
        or set(payload) != {"pairs"}
        or not isinstance(payload["pairs"], list)
    ):
        raise TypeError("comparable-pairs payload invalid")
    pairs: list[tuple[ExactRational, ExactRational]] = []
    for row in payload["pairs"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"lower", "upper"}
        ):
            raise TypeError("comparable pair invalid")
        pairs.append(
            (
                _decode_rational(row["lower"]),
                _decode_rational(row["upper"]),
            )
        )
    return tuple(pairs)


def _validate_constraint_bridge(
    schema: ExecutableLawSchema,
    episode: StructuralEpisode,
    binding: LawBinding,
    role_targets: Mapping[str, str],
) -> tuple[str, ...]:
    issues: list[str] = []
    contract = _CONSTRAINT_ROLE_CONTRACT.get(schema.law_kind)
    if contract is None:
        return ("semantic_constraint_contract_unknown",)
    constraint_role, participant_roles = contract
    schema_roles = {row.role_id: row for row in schema.roles}
    if (
        constraint_role not in role_targets
        or any(role_id not in role_targets for role_id in participant_roles)
    ):
        return ("semantic_constraint_role_binding_missing",)
    try:
        target = episode.require_target(
            RoleTargetKind.CONSTRAINT,
            role_targets[constraint_role],
        )
    except KeyError:
        return ("semantic_constraint_target_missing",)
    if not isinstance(target, StructuralConstraint):
        return ("semantic_constraint_target_kind_mismatch",)
    participants = {
        row.participant_role: row for row in target.participants
    }
    if set(participants) != set(participant_roles):
        issues.append("semantic_constraint_participant_coverage_mismatch")
    for role_id in participant_roles:
        participant = participants.get(role_id)
        role_spec = schema_roles.get(role_id)
        if participant is None or role_spec is None:
            continue
        if (
            participant.target_kind is not role_spec.target_kind
            or participant.target_id != role_targets[role_id]
        ):
            issues.append(
                f"semantic_constraint_participant_binding_mismatch.{role_id}"
            )
    required_observable_ids = {
        row.observable_id for row in schema.required_observables
    }
    if set(target.observable_ids) != required_observable_ids:
        issues.append("semantic_constraint_observable_coverage_mismatch")
    return tuple(sorted(set(issues)))


def _validate_observable_payload_contract(
    schema: ExecutableLawSchema,
    observables: Mapping[str, TypedObservable],
) -> tuple[str, ...]:
    """Reject malformed typed data; applicability failures stay elsewhere."""

    try:
        if schema.law_kind is LawKind.EQUIVARIANCE:
            for observable_id in ("outputs_before", "outputs_after"):
                observable = observables[observable_id]
                if _observed(observable):
                    _decode_vector(observable)
            input_action = observables["input_action"]
            if _observed(input_action):
                _decode_map(input_action)
            output_action = observables["output_action"]
            if _observed(output_action):
                payload = output_action.value_payload
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"permutation", "signs"}
                    or not isinstance(payload["permutation"], list)
                    or not isinstance(payload["signs"], list)
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        for value in (
                            *payload["permutation"],
                            *payload["signs"],
                        )
                    )
                    or payload["permutation"] != [0]
                    or len(payload["signs"]) != 1
                    or payload["signs"][0] not in {-1, 1}
                ):
                    raise TypeError("signed permutation payload invalid")
        elif schema.law_kind is LawKind.MONOTONE_ORDER:
            pairs = observables["comparable_output_pairs"]
            if _observed(pairs):
                _decode_comparable_pairs(pairs)
            direction = observables["declared_direction"]
            if _observed(direction):
                payload = direction.value_payload
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"direction"}
                    or not isinstance(payload["direction"], int)
                    or isinstance(payload["direction"], bool)
                    or payload["direction"] not in {-1, 0, 1}
                ):
                    raise TypeError("direction payload invalid")
        elif schema.law_kind is LawKind.CLOSED_BALANCE:
            boundary = observables["boundary_declaration"]
            if _observed(boundary):
                payload = boundary.value_payload
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"boundary_id", "complete"}
                    or not isinstance(payload["boundary_id"], str)
                    or not isinstance(payload["complete"], bool)
                ):
                    raise TypeError("boundary payload invalid")
            ledger = observables["quantity_ledger"]
            if _observed(ledger):
                payload = ledger.value_payload
                keys = {
                    "inflows",
                    "outflows",
                    "sinks",
                    "sources",
                    "storage_after",
                    "storage_before",
                }
                if not isinstance(payload, dict) or set(payload) != keys:
                    raise TypeError("quantity ledger payload invalid")
                _decode_rational(payload["storage_before"])
                _decode_rational(payload["storage_after"])
                for key in ("inflows", "outflows", "sources", "sinks"):
                    if not isinstance(payload[key], list):
                        raise TypeError("quantity ledger rows invalid")
                    for value in payload[key]:
                        _decode_rational(value)
        elif schema.law_kind is LawKind.PATH_COMPOSITION:
            domain = observables["finite_domain"]
            if _observed(domain):
                _decode_string_values(domain)
            for observable_id in ("first_map", "second_map", "direct_map"):
                observable = observables[observable_id]
                if _observed(observable):
                    _decode_map(observable)
        elif schema.law_kind is LawKind.LOW_ORDER_INTERACTION:
            for observable_id in ("components", "designated_pair"):
                observable = observables[observable_id]
                if _observed(observable):
                    _decode_string_values(observable)
            folds = observables["held_fold_utilities"]
            if _observed(folds):
                _decode_fold_utilities(folds)
            expectation = observables["interaction_expectation"]
            if _observed(expectation):
                payload = expectation.value_payload
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"value"}
                    or payload["value"]
                    not in {value.value for value in InteractionExpectation}
                ):
                    raise TypeError(
                        "interaction expectation payload invalid"
                    )
    except (KeyError, TypeError, ValueError):
        return ("semantic_typed_observable_payload_invalid",)
    return ()


def validate_bound_law_semantics(
    registry: GSCLSchemaRegistry,
    schema: ExecutableLawSchema,
    episode: StructuralEpisode,
    binding: LawBinding,
    policy: ResidualPolicy,
) -> tuple[str, ...]:
    """Check the Phase-0 residual-critical role/observable bridge.

    This bounded validator checks the frozen incidence/value subset represented
    on both sides; opaque carrier contents remain an extractor qualification
    problem.  It deliberately does not check whether a law is satisfied; that
    remains the residual kernel's job.  ``role:*`` references survive entity
    renaming, while ``local:*`` references are scoped to one observable bundle.
    """

    issues: list[str] = []
    binding_issues = validate_law_binding(
        registry, schema, episode, binding
    )
    if binding_issues:
        return tuple(
            sorted(
                {
                    "semantic_binding_invalid",
                    *binding_issues,
                }
            )
        )
    policy_issues = policy.validate(schema)
    if policy_issues:
        return tuple(
            sorted({"semantic_policy_invalid", *policy_issues})
        )
    role_targets = _role_target_ids(binding)
    observables = _observables_for_binding(episode, binding)
    payload_issues = _validate_observable_payload_contract(
        schema, observables
    )
    if payload_issues:
        return payload_issues
    issues.extend(
        _validate_constraint_bridge(
            schema, episode, binding, role_targets
        )
    )

    try:
        if schema.law_kind is LawKind.EQUIVARIANCE:
            transformation = episode.require_target(
                RoleTargetKind.RELATION,
                role_targets["transformation"],
            )
            before_quantity = episode.require_target(
                RoleTargetKind.QUANTITY,
                role_targets["output_before"],
            )
            after_quantity = episode.require_target(
                RoleTargetKind.QUANTITY,
                role_targets["output_after"],
            )
            assert isinstance(transformation, StructuralRelation)
            assert isinstance(before_quantity, StructuralQuantity)
            assert isinstance(after_quantity, StructuralQuantity)
            if (
                transformation.source_object_id
                != role_targets["input_before"]
                or transformation.target_object_id
                != role_targets["input_after"]
            ):
                issues.append("semantic_equivariance_relation_direction_mismatch")
            if (
                before_quantity.owner_object_id
                != role_targets["input_before"]
                or after_quantity.owner_object_id
                != role_targets["input_after"]
            ):
                issues.append("semantic_equivariance_quantity_owner_mismatch")

            before = observables["outputs_before"]
            after = observables["outputs_after"]
            if _observed(before) and _observed(after):
                before_values = _decode_vector(before)
                after_values = _decode_vector(after)
                if len(before_values) != 1 or len(after_values) != 1:
                    issues.append(
                        "semantic_equivariance_phase0_scalar_arity_mismatch"
                    )
                else:
                    if not _quantity_matches_observable(
                        before_quantity, before_values[0], before
                    ):
                        issues.append(
                            "semantic_equivariance_before_quantity_mismatch"
                        )
                    if not _quantity_matches_observable(
                        after_quantity, after_values[0], after
                    ):
                        issues.append(
                            "semantic_equivariance_after_quantity_mismatch"
                        )
            input_action = observables["input_action"]
            if _observed(input_action):
                action = _decode_map(input_action)
                expected_action = {
                    _role_ref("input_before"): _role_ref("input_after")
                }
                if policy.require_involution:
                    expected_action[
                        _role_ref("input_after")
                    ] = _role_ref("input_before")
                if action != expected_action:
                    issues.append(
                        "semantic_equivariance_input_action_binding_mismatch"
                    )

        elif schema.law_kind is LawKind.MONOTONE_ORDER:
            relation = episode.require_target(
                RoleTargetKind.RELATION,
                role_targets["order_relation"],
            )
            lower_quantity = episode.require_target(
                RoleTargetKind.QUANTITY,
                role_targets["lower_value"],
            )
            upper_quantity = episode.require_target(
                RoleTargetKind.QUANTITY,
                role_targets["upper_value"],
            )
            assert isinstance(relation, StructuralRelation)
            assert isinstance(lower_quantity, StructuralQuantity)
            assert isinstance(upper_quantity, StructuralQuantity)
            if (
                relation.source_object_id != role_targets["lower_state"]
                or relation.target_object_id
                != role_targets["upper_state"]
            ):
                issues.append("semantic_monotone_relation_direction_mismatch")
            if (
                lower_quantity.owner_object_id
                != role_targets["lower_state"]
                or upper_quantity.owner_object_id
                != role_targets["upper_state"]
            ):
                issues.append("semantic_monotone_quantity_owner_mismatch")
            pairs = observables["comparable_output_pairs"]
            if _observed(pairs):
                decoded_pairs = _decode_comparable_pairs(pairs)
                if len(decoded_pairs) != 1:
                    issues.append(
                        "semantic_monotone_phase0_pair_arity_mismatch"
                    )
                else:
                    lower, upper = decoded_pairs[0]
                    if not _quantity_matches_observable(
                        lower_quantity, lower, pairs
                    ):
                        issues.append(
                            "semantic_monotone_lower_quantity_mismatch"
                        )
                    if not _quantity_matches_observable(
                        upper_quantity, upper, pairs
                    ):
                        issues.append(
                            "semantic_monotone_upper_quantity_mismatch"
                        )

        elif schema.law_kind is LawKind.CLOSED_BALANCE:
            boundary_id = role_targets["system_boundary"]
            before_quantity = episode.require_target(
                RoleTargetKind.QUANTITY,
                role_targets["storage_before"],
            )
            after_quantity = episode.require_target(
                RoleTargetKind.QUANTITY,
                role_targets["storage_after"],
            )
            assert isinstance(before_quantity, StructuralQuantity)
            assert isinstance(after_quantity, StructuralQuantity)
            if episode.declared_boundary_object_id != boundary_id:
                issues.append("semantic_balance_declared_boundary_mismatch")
            if (
                before_quantity.owner_object_id != boundary_id
                or after_quantity.owner_object_id != boundary_id
            ):
                issues.append("semantic_balance_storage_owner_mismatch")
            boundary = observables["boundary_declaration"]
            if _observed(boundary):
                payload = boundary.value_payload
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"boundary_id", "complete"}
                    or not isinstance(payload["complete"], bool)
                ):
                    raise TypeError("boundary payload invalid")
                if payload["boundary_id"] != _role_ref(
                    "system_boundary"
                ):
                    issues.append("semantic_balance_boundary_reference_mismatch")
            ledger = observables["quantity_ledger"]
            if _observed(ledger):
                payload = ledger.value_payload
                required_keys = {
                    "inflows",
                    "outflows",
                    "sinks",
                    "sources",
                    "storage_after",
                    "storage_before",
                }
                if not isinstance(payload, dict) or set(payload) != required_keys:
                    raise TypeError("quantity ledger payload invalid")
                before = _decode_rational(payload["storage_before"])
                after = _decode_rational(payload["storage_after"])
                if not _quantity_matches_observable(
                    before_quantity, before, ledger
                ):
                    issues.append("semantic_balance_storage_before_mismatch")
                if not _quantity_matches_observable(
                    after_quantity, after, ledger
                ):
                    issues.append("semantic_balance_storage_after_mismatch")

        elif schema.law_kind is LawKind.PATH_COMPOSITION:
            required = (
                "finite_domain",
                "first_map",
                "second_map",
                "direct_map",
            )
            if all(_observed(observables[row]) for row in required):
                domain = _decode_string_values(
                    observables["finite_domain"]
                )
                first = _decode_map(observables["first_map"])
                second = _decode_map(observables["second_map"])
                direct = _decode_map(observables["direct_map"])
                source_ref = _role_ref("source_state")
                target_ref = _role_ref("target_state")

                def valid_domain_ref(value: str) -> bool:
                    return value == source_ref or value.startswith(
                        "local:"
                    )

                def valid_output_ref(value: str) -> bool:
                    return value == target_ref or value.startswith(
                        "local:"
                    )

                if (
                    len(domain) != len(set(domain))
                    or len(domain) < 2
                    or domain.count(source_ref) != 1
                    or any(not valid_domain_ref(value) for value in domain)
                ):
                    issues.append("semantic_path_domain_reference_mismatch")
                if set(first) != set(domain) or any(
                    not value.startswith("local:")
                    for value in first.values()
                ):
                    issues.append("semantic_path_first_map_binding_mismatch")
                if set(second) != set(first.values()) or any(
                    not valid_output_ref(value)
                    for value in second.values()
                ):
                    issues.append("semantic_path_second_map_binding_mismatch")
                if set(direct) != set(domain) or any(
                    not valid_output_ref(value)
                    for value in direct.values()
                ):
                    issues.append("semantic_path_direct_map_binding_mismatch")
                if (
                    source_ref not in first
                    or first.get(source_ref) not in second
                    or second.get(first.get(source_ref)) != target_ref
                    or direct.get(source_ref) != target_ref
                ):
                    issues.append("semantic_path_anchor_endpoint_mismatch")

        elif schema.law_kind is LawKind.LOW_ORDER_INTERACTION:
            component_refs = {
                _role_ref("component_a"),
                _role_ref("component_b"),
                _role_ref("component_c"),
            }
            components = observables["components"]
            if _observed(components):
                component_values = _decode_string_values(components)
                if (
                    len(component_values) != 3
                    or len(set(component_values)) != 3
                    or set(component_values) != component_refs
                ):
                    issues.append("semantic_interaction_component_binding_mismatch")
            pair = observables["designated_pair"]
            if _observed(pair):
                pair_values = _decode_string_values(pair)
                if (
                    len(pair_values) != 2
                    or len(set(pair_values)) != 2
                    or set(pair_values)
                    != {
                        _role_ref("component_a"),
                        _role_ref("component_b"),
                    }
                ):
                    issues.append("semantic_interaction_pair_binding_mismatch")
            folds = observables["held_fold_utilities"]
            if _observed(folds):
                expected_subsets = {
                    frozenset(subset)
                    for size in range(len(component_refs) + 1)
                    for subset in combinations(
                        sorted(component_refs), size
                    )
                }
                decoded_folds = _decode_fold_utilities(folds)
                if not decoded_folds or any(
                    set(fold) != expected_subsets
                    for fold in decoded_folds
                ):
                    issues.append(
                        "semantic_interaction_subset_lattice_mismatch"
                    )
    except (AssertionError, KeyError, TypeError, ValueError):
        issues.append("semantic_observable_role_payload_contract_invalid")
    return tuple(sorted(set(issues)))


def _evaluate_observable_mapping(
    schema: ExecutableLawSchema,
    observables: Mapping[str, TypedObservable],
    policy: ResidualPolicy,
) -> LawEvaluation:
    missing = tuple(
        sorted(
            observable_id
            for observable_id, observable in observables.items()
            if observable.observation_status.value == "unknown"
        )
    )
    if missing:
        return _inconclusive(*missing)
    try:
        if schema.law_kind is LawKind.EQUIVARIANCE:
            before = observables["outputs_before"]
            after = observables["outputs_after"]
            action = observables["output_action"]
            input_action = observables["input_action"]
            _require_value_type(
                before, ObservableValueType.EXACT_VECTOR
            )
            _require_value_type(
                after, ObservableValueType.EXACT_VECTOR
            )
            _require_value_type(
                action, ObservableValueType.SIGNED_PERMUTATION
            )
            _require_value_type(
                input_action, ObservableValueType.FINITE_MAP
            )
            if (
                before.dimension != after.dimension
                or before.unit != after.unit
            ):
                return _not_applicable(
                    "output_coordinates_are_comparable"
                )
            if not _decode_map(input_action):
                return _not_applicable("finite_action_is_declared")
            action_payload = action.value_payload
            if (
                not isinstance(action_payload, dict)
                or set(action_payload) != {"permutation", "signs"}
                or not isinstance(
                    action_payload["permutation"], list
                )
                or not isinstance(action_payload["signs"], list)
            ):
                raise TypeError("signed permutation payload invalid")
            return evaluate_equivariance(
                _decode_vector(before),
                _decode_vector(after),
                tuple(action_payload["permutation"]),
                tuple(action_payload["signs"]),
                tolerance=policy.tolerance,
                require_involution=policy.require_involution,
            )
        if schema.law_kind is LawKind.MONOTONE_ORDER:
            pairs = observables["comparable_output_pairs"]
            direction = observables["declared_direction"]
            _require_value_type(
                pairs, ObservableValueType.COMPARABLE_PAIRS
            )
            _require_value_type(
                direction, ObservableValueType.DIRECTION
            )
            pair_payload = pairs.value_payload
            if (
                not isinstance(pair_payload, dict)
                or set(pair_payload) != {"pairs"}
                or not isinstance(pair_payload["pairs"], list)
            ):
                raise TypeError("comparable-pairs payload invalid")
            decoded_pairs = []
            for row in pair_payload["pairs"]:
                if (
                    not isinstance(row, dict)
                    or set(row) != {"lower", "upper"}
                ):
                    raise TypeError("comparable pair invalid")
                decoded_pairs.append(
                    (
                        _decode_rational(row["lower"]),
                        _decode_rational(row["upper"]),
                    )
                )
            direction_payload = direction.value_payload
            if (
                not isinstance(direction_payload, dict)
                or set(direction_payload) != {"direction"}
            ):
                raise TypeError("direction payload invalid")
            return evaluate_monotone_order(
                tuple(decoded_pairs),
                direction=direction_payload["direction"],
                tolerance=policy.tolerance,
            )
        if schema.law_kind is LawKind.CLOSED_BALANCE:
            boundary = observables["boundary_declaration"]
            ledger = observables["quantity_ledger"]
            _require_value_type(
                boundary, ObservableValueType.BOUNDARY_DECLARATION
            )
            _require_value_type(
                ledger, ObservableValueType.QUANTITY_LEDGER
            )
            boundary_payload = boundary.value_payload
            if (
                not isinstance(boundary_payload, dict)
                or set(boundary_payload)
                != {"boundary_id", "complete"}
                or not isinstance(
                    boundary_payload["complete"], bool
                )
            ):
                raise TypeError("boundary payload invalid")
            ledger_payload = ledger.value_payload
            ledger_keys = {
                "inflows",
                "outflows",
                "sinks",
                "sources",
                "storage_after",
                "storage_before",
            }
            if (
                not isinstance(ledger_payload, dict)
                or set(ledger_payload) != ledger_keys
                or not all(
                    isinstance(ledger_payload[key], list)
                    for key in ("inflows", "outflows", "sources", "sinks")
                )
                or ledger.dimension is None
                or ledger.unit is None
            ):
                raise TypeError("quantity ledger payload invalid")
            kwargs = {
                key: tuple(
                    _decode_quantity(
                        value,
                        dimension=ledger.dimension,
                        unit=ledger.unit,
                    )
                    for value in ledger_payload[key]
                )
                for key in ("inflows", "outflows", "sources", "sinks")
            }
            return evaluate_closed_balance(
                _decode_quantity(
                    ledger_payload["storage_before"],
                    dimension=ledger.dimension,
                    unit=ledger.unit,
                ),
                _decode_quantity(
                    ledger_payload["storage_after"],
                    dimension=ledger.dimension,
                    unit=ledger.unit,
                ),
                kwargs["inflows"],
                kwargs["outflows"],
                kwargs["sources"],
                kwargs["sinks"],
                boundary_id=boundary_payload["boundary_id"],
                boundary_complete=boundary_payload["complete"],
                tolerance=policy.tolerance,
            )
        if schema.law_kind is LawKind.PATH_COMPOSITION:
            domain = observables["finite_domain"]
            first = observables["first_map"]
            second = observables["second_map"]
            direct = observables["direct_map"]
            _require_value_type(
                domain, ObservableValueType.FINITE_DOMAIN
            )
            for observable in (first, second, direct):
                _require_value_type(
                    observable, ObservableValueType.FINITE_MAP
                )
            return evaluate_path_composition(
                _decode_string_values(domain),
                _decode_map(first),
                _decode_map(second),
                _decode_map(direct),
            )
        if schema.law_kind is LawKind.LOW_ORDER_INTERACTION:
            components = observables["components"]
            pair = observables["designated_pair"]
            expectation = observables["interaction_expectation"]
            folds = observables["held_fold_utilities"]
            _require_value_type(
                components, ObservableValueType.COMPONENT_SET
            )
            _require_value_type(
                pair, ObservableValueType.DESIGNATED_PAIR
            )
            _require_value_type(
                expectation,
                ObservableValueType.INTERACTION_EXPECTATION,
            )
            _require_value_type(
                folds, ObservableValueType.SUBSET_UTILITY_FOLDS
            )
            pair_values = _decode_string_values(pair)
            if len(pair_values) != 2:
                raise TypeError("designated pair payload invalid")
            expectation_payload = expectation.value_payload
            if (
                not isinstance(expectation_payload, dict)
                or set(expectation_payload) != {"value"}
            ):
                raise TypeError("interaction expectation payload invalid")
            return evaluate_low_order_interaction(
                _decode_fold_utilities(folds),
                _decode_string_values(components),
                (pair_values[0], pair_values[1]),
                expected_relation=InteractionExpectation(
                    expectation_payload["value"]
                ),
                relation_threshold=policy.relation_threshold,
                high_order_tolerance=policy.high_order_tolerance,
                common_utility_scale=policy.common_utility_scale,
            )
    except (KeyError, TypeError, ValueError):
        return _not_applicable("typed_observable_payload_contract")
    return _not_applicable("unsupported_law_kind")


def evaluate_bound_law(
    registry: GSCLSchemaRegistry,
    schema: ExecutableLawSchema,
    episode: StructuralEpisode,
    binding: LawBinding,
    policy: ResidualPolicy,
) -> LawEvaluation:
    binding_issues = validate_law_binding(
        registry, schema, episode, binding
    )
    if binding_issues:
        raise PermissionError(
            "cannot evaluate invalid law binding: "
            + ",".join(binding_issues)
        )
    policy_issues = policy.validate(schema)
    if policy_issues:
        raise PermissionError(
            "cannot evaluate invalid residual policy: "
            + ",".join(policy_issues)
        )
    semantic_issues = validate_bound_law_semantics(
        registry, schema, episode, binding, policy
    )
    if semantic_issues:
        raise PermissionError(
            "cannot evaluate semantically inconsistent law binding: "
            + ",".join(semantic_issues)
        )
    evaluation = _evaluate_observable_mapping(
        schema, _observables_for_binding(episode, binding), policy
    )
    bound = replace(
        evaluation,
        evaluation_input_hash=binding.evaluation_input_hash,
        policy_hash=policy.policy_hash,
    )
    issues = bound.validate(require_bound=True)
    if issues:
        raise PermissionError(
            "bound law evaluation is invalid: " + ",".join(issues)
        )
    return bound


def evaluate_transformed_law(
    registry: GSCLSchemaRegistry,
    schema: ExecutableLawSchema,
    episode: StructuralEpisode,
    binding: LawBinding,
    policy: ResidualPolicy,
    *,
    transformation_id: str,
) -> LawEvaluation:
    binding_issues = validate_law_binding(
        registry, schema, episode, binding
    )
    if binding_issues:
        raise PermissionError(
            "cannot transform an invalid law binding: "
            + ",".join(binding_issues)
        )
    policy_issues = policy.validate(schema)
    if policy_issues:
        raise PermissionError(
            "cannot transform with an invalid residual policy: "
            + ",".join(policy_issues)
        )
    semantic_issues = validate_bound_law_semantics(
        registry, schema, episode, binding, policy
    )
    if semantic_issues:
        raise PermissionError(
            "cannot transform a semantically inconsistent base binding: "
            + ",".join(semantic_issues)
        )
    base = _observables_for_binding(episode, binding)
    observable_payload_overrides = _canonical_hard_negative_overrides(
        schema, base, transformation_id
    )
    transformed = {
        observable_id: (
            replace(
                observable,
                value_payload=observable_payload_overrides[observable_id],
            )
            if observable_id in observable_payload_overrides
            else observable
        )
        for observable_id, observable in base.items()
    }
    evaluation = _evaluate_observable_mapping(
        schema, transformed, policy
    )
    transformed_input_hash = strict_content_hash(
        {
            "base_evaluation_input_hash": binding.evaluation_input_hash,
            "transformation_id": transformation_id,
            "operator_contract_hash": (
                HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES[
                    transformation_id
                ]
            ),
            "observable_payload_overrides": dict(
                sorted(observable_payload_overrides.items())
            ),
        }
    )
    bound = replace(
        evaluation,
        evaluation_input_hash=transformed_input_hash,
        policy_hash=policy.policy_hash,
    )
    issues = bound.validate(require_bound=True)
    if issues:
        raise PermissionError(
            "transformed law evaluation is invalid: "
            + ",".join(issues)
        )
    return bound


def _evaluation_private_payload(
    evaluation: LawEvaluation,
) -> dict[str, Any]:
    return {
        "disposition": evaluation.disposition.value,
        "components": [
            component.private_payload()
            for component in sorted(
                evaluation.components, key=lambda row: row.component_id
            )
        ],
        "missing_observables": list(
            sorted(evaluation.missing_observables)
        ),
        "applicability_failures": list(
            sorted(evaluation.applicability_failures)
        ),
        "evaluation_input_hash": evaluation.evaluation_input_hash,
        "policy_hash": evaluation.policy_hash,
    }


def build_law_residual_receipt(
    registry: GSCLSchemaRegistry,
    schema: ExecutableLawSchema,
    episode: StructuralEpisode,
    binding: LawBinding,
    policy: ResidualPolicy,
    *,
    receipt_id: str,
    evidence_span_ids: Sequence[str],
) -> LawResidualReceipt:
    binding_issues = validate_law_binding(
        registry, schema, episode, binding
    )
    if binding_issues:
        raise PermissionError(
            "cannot receipt an invalid law binding: "
            + ",".join(binding_issues)
        )
    policy_issues = policy.validate(schema)
    if policy_issues:
        raise PermissionError(
            "cannot receipt an invalid residual policy: "
            + ",".join(policy_issues)
        )
    evaluation = evaluate_bound_law(
        registry, schema, episode, binding, policy
    )
    contrastives: list[ContrastiveResidual] = []
    transformation_ids = (
        tuple(sorted(schema.hard_negative_transformations))
        if evaluation.disposition
        in {
            ResidualDisposition.SATISFIED,
            ResidualDisposition.VIOLATED,
        }
        else ()
    )
    for transformation_id in transformation_ids:
        transformed = evaluate_transformed_law(
            registry,
            schema,
            episode,
            binding,
            policy,
            transformation_id=transformation_id,
        )
        assert transformed.evaluation_input_hash is not None
        contrastives.append(
            ContrastiveResidual(
                transformation_id=transformation_id,
                operator_contract_hash=(
                    HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES[
                        transformation_id
                    ]
                ),
                transformed_input_hash=(
                    transformed.evaluation_input_hash
                ),
                policy_hash=policy.policy_hash,
                disposition=transformed.disposition,
                components=transformed.components,
            )
        )
    receipt = LawResidualReceipt(
        receipt_id=receipt_id,
        law_id=schema.law_id,
        registry_hash=registry.registry_hash,
        schema_hash=schema.schema_hash,
        episode_hash=episode.episode_hash,
        binding_hash=binding.binding_hash,
        evaluation_input_hash=binding.evaluation_input_hash,
        policy_hash=policy.policy_hash,
        verifier_id=schema.residual_function_id,
        verifier_version=schema.verifier_version,
        verifier_contract_hash=schema.verifier_contract_hash,
        disposition=evaluation.disposition,
        components=evaluation.components,
        missing_observables=evaluation.missing_observables,
        applicability_failures=evaluation.applicability_failures,
        contrastive_residuals=tuple(contrastives),
        evidence_span_ids=tuple(evidence_span_ids),
    )
    receipt_issues = receipt.validate(
        registry,
        schema,
        episode,
        binding,
        expected_policy_hash=policy.policy_hash,
    )
    if receipt_issues:
        raise PermissionError(
            "invalid law residual receipt: "
            + ",".join(receipt_issues)
        )
    return receipt


def verify_law_residual_receipt_trusted(
    receipt: LawResidualReceipt,
    registry: GSCLSchemaRegistry,
    schema: ExecutableLawSchema,
    episode: StructuralEpisode,
    binding: LawBinding,
    policy: ResidualPolicy,
) -> tuple[str, ...]:
    """Recompute primary and contrastive residuals from committed observables."""

    issues = list(
        receipt.validate(
            registry,
            schema,
            episode,
            binding,
            expected_policy_hash=policy.policy_hash,
        )
    )
    if issues:
        return tuple(sorted(set(issues)))
    expected_primary = evaluate_bound_law(
        registry, schema, episode, binding, policy
    )
    actual_primary = LawEvaluation(
        disposition=receipt.disposition,
        components=receipt.components,
        missing_observables=receipt.missing_observables,
        applicability_failures=receipt.applicability_failures,
        evaluation_input_hash=receipt.evaluation_input_hash,
        policy_hash=receipt.policy_hash,
    )
    if _evaluation_private_payload(actual_primary) != (
        _evaluation_private_payload(expected_primary)
    ):
        issues.append("trusted_primary_recomputation_mismatch")
    receipt_by_id = {
        row.transformation_id: row
        for row in receipt.contrastive_residuals
    }
    expected_ids = (
        set(schema.hard_negative_transformations)
        if receipt.disposition
        in {
            ResidualDisposition.SATISFIED,
            ResidualDisposition.VIOLATED,
        }
        else set()
    )
    if set(receipt_by_id) != expected_ids:
        issues.append("trusted_contrastive_coverage_mismatch")
    for transformation_id in sorted(set(receipt_by_id) & expected_ids):
        expected = evaluate_transformed_law(
            registry,
            schema,
            episode,
            binding,
            policy,
            transformation_id=transformation_id,
        )
        actual = receipt_by_id[transformation_id]
        actual_evaluation = LawEvaluation(
            disposition=actual.disposition,
            components=actual.components,
            evaluation_input_hash=actual.transformed_input_hash,
            policy_hash=actual.policy_hash,
        )
        if _evaluation_private_payload(actual_evaluation) != (
            _evaluation_private_payload(expected)
        ):
            issues.append(
                f"trusted_contrastive_recomputation_mismatch.{transformation_id}"
            )
    return tuple(sorted(set(issues)))


__all__ = [
    "InteractionExpectation",
    "LawEvaluation",
    "MeasuredQuantity",
    "ResidualPolicy",
    "build_law_residual_receipt",
    "evaluate_bound_law",
    "evaluate_closed_balance",
    "evaluate_equivariance",
    "evaluate_low_order_interaction",
    "evaluate_monotone_order",
    "evaluate_path_composition",
    "evaluate_transformed_law",
    "mobius_coefficients",
    "validate_bound_law_semantics",
    "verify_law_residual_receipt_trusted",
]
