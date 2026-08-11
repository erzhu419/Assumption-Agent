import ast
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from pathlib import Path

import pytest

from hegel_machine.phase2b_uncertainty_compiler import (
    BundleUncertaintyDisposition,
    DEFAULT_EXACT_UNCERTAINTY_POLICY,
    FROZEN_PHASE2B_EXACT_FREEZE_ID,
    FROZEN_RATIONAL_GRID_ID,
    ExactUncertaintyCompilerPolicy,
    ObservationValueKind,
    compile_bundle_uncertainty,
)
from hegel_machine.phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from hegel_machine.phase2b_wire import (
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
    PublicEvidenceBundle,
)
from hegel_machine.phase3_dsl_v1 import OLD_DSL_V1, RATIONAL_VALUE_GRID


def uid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def observation(
    index: int,
    *,
    value: dict[str, object] | None,
    uncertainty: dict[str, object],
    missingness: str = "observed",
) -> dict[str, object]:
    return {
        "observation_id": uid(index),
        "source_channel_id": uid(20),
        "entity_ids": [uid(3)],
        "role_candidate_ids": [uid(5)],
        "quantity_id": uid(7),
        "value": value,
        "unit_dimension": {"si_exponents": [0, 0, 0, 0, 0, 0, 0]},
        "temporal_support": None,
        "spatial_support": None,
        "uncertainty": uncertainty,
        "provenance_sha256": f"{index % 16:x}" * 64,
        "missingness": missingness,
    }


def bundle_mapping() -> dict[str, object]:
    return {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "bundle_id": uid(1),
        "entity_candidates": [
            {"entity_id": uid(3), "role_candidate_ids": [uid(5)]}
        ],
        "role_ids": [uid(5)],
        "quantity_ids": [uid(7)],
        "observations": [
            observation(
                100,
                value={"kind": "numeric", "values": [1.0, -0.5]},
                uncertainty={"model": "absolute_bound", "radius": [0.1, 0.0]},
            ),
            observation(
                101,
                value={"kind": "interval", "lower": [0.25], "upper": [0.5]},
                uncertainty={"model": "absolute_bound", "radius": [0.0]},
            ),
            observation(
                102,
                value={"kind": "boolean", "value": True},
                uncertainty={"model": "not_applicable", "radius": []},
            ),
            observation(
                103,
                value=None,
                uncertainty={"model": "not_applicable", "radius": []},
                missingness="missing",
            ),
        ],
        "task_target": {
            "task_id": uid(8),
            "entity_ids": [uid(3)],
            "quantity_ids": [uid(7)],
        },
        "aggregation_graph": {
            "scale_ids": [uid(9)],
            "root_scale_ids": [uid(9)],
            "edges": [],
        },
        "transform_catalog": [
            {
                "transform_id": uid(11),
                "operation": "identity",
                "parameters": [],
            }
        ],
        "missingness_mask": [uid(103)],
    }


def test_policy_binds_exact_freeze_and_663_point_rational_grid():
    policy = DEFAULT_EXACT_UNCERTAINTY_POLICY
    assert policy.rational_grid_id == OLD_DSL_V1.rational_grid_id == (
        "rational_grid_"
        "94131eb37f198c4e42c14266c8c4cacd7eb2a6758997fe5381a2758b6f37277f"
    )
    assert policy.rational_grid_id == FROZEN_RATIONAL_GRID_ID
    assert policy.phase2b_exact_freeze_id == (
        frozen_phase2b_exact_freeze().freeze_id
    )
    assert policy.phase2b_exact_freeze_id == FROZEN_PHASE2B_EXACT_FREEZE_ID
    assert policy.rational_grid_cardinality == len(RATIONAL_VALUE_GRID) == 663
    assert policy.endpoint_rounding == "outward_to_frozen_RationalValue_grid"
    assert policy.standard_error_status == "STANDARD_ERROR_UNSUPPORTED"
    with pytest.raises(ValueError, match="cardinality drift"):
        replace(policy, rational_grid_cardinality=662)


def test_absolute_bounds_are_exact_grid_atoms_and_boolean_missing_are_legal():
    bundle = PublicEvidenceBundle.from_mapping(bundle_mapping())
    result = compile_bundle_uncertainty(bundle)
    assert result.disposition is BundleUncertaintyDisposition.COMPLETE
    assert result.reason == "complete_exact_uncertainty_bundle"
    assert result.failures == ()
    assert len(result.observations) == 4
    compiled = {item.observation_id: item for item in result.observations}

    numeric = compiled[uid(100)]
    assert numeric.value_kind is ObservationValueKind.NUMERIC_INTERVAL
    first, second = numeric.numeric_bounds
    exact_lower = Fraction.from_float(1.0) - Fraction.from_float(0.1)
    exact_upper = Fraction.from_float(1.0) + Fraction.from_float(0.1)
    assert first.lower_fraction <= exact_lower
    assert first.upper_fraction >= exact_upper
    assert (first.lower_fraction, first.upper_fraction) == (
        Fraction(7, 8),
        Fraction(9, 8),
    )
    assert (second.lower_fraction, second.upper_fraction) == (
        Fraction(-1, 2),
        Fraction(-1, 2),
    )
    assert all(
        endpoint in RATIONAL_VALUE_GRID
        for bound in numeric.numeric_bounds
        for endpoint in (bound.lower, bound.upper)
    )

    interval = compiled[uid(101)]
    assert interval.numeric_bounds[0].lower_fraction == Fraction(1, 4)
    assert interval.numeric_bounds[0].upper_fraction == Fraction(1, 2)
    boolean = compiled[uid(102)]
    assert boolean.value_kind is ObservationValueKind.BOOLEAN
    assert boolean.boolean_value is True
    missing = compiled[uid(103)]
    assert missing.value_kind is ObservationValueKind.MISSING
    assert missing.numeric_bounds == ()
    assert missing.boolean_value is None
    with pytest.raises(FrozenInstanceError):
        first.lower = RATIONAL_VALUE_GRID[0]


def test_binary64_value_is_recovered_with_fraction_from_float_before_rounding():
    mapping = bundle_mapping()
    numeric = mapping["observations"][0]
    numeric["value"] = {"kind": "numeric", "values": [1.0]}
    numeric["uncertainty"] = {"model": "absolute_bound", "radius": [0.1]}
    result = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    bound = next(
        item for item in result.observations if item.observation_id == uid(100)
    ).numeric_bounds[0]
    represented_radius = Fraction.from_float(0.1)
    assert represented_radius != Fraction(1, 10)
    assert bound.lower_fraction <= Fraction(1) - represented_radius
    assert bound.upper_fraction >= Fraction(1) + represented_radius

    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hegel_machine"
        / "phase2b_uncertainty_compiler.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = {
        (node.func.value.id, node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }
    assert ("Fraction", "from_float") in calls


def test_binary64_one_third_is_not_reinterpreted_as_decimal_or_exact_third():
    mapping = bundle_mapping()
    numeric = mapping["observations"][0]
    numeric["value"] = {"kind": "numeric", "values": [1.0 / 3.0]}
    numeric["uncertainty"] = {"model": "absolute_bound", "radius": [0.0]}
    result = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    bound = next(
        item for item in result.observations if item.observation_id == uid(100)
    ).numeric_bounds[0]
    binary64_third = Fraction.from_float(1.0 / 3.0)
    assert binary64_third < Fraction(1, 3)
    assert (bound.lower_fraction, bound.upper_fraction) == (
        Fraction(2, 7),
        Fraction(1, 3),
    )
    assert bound.lower_fraction <= binary64_third <= bound.upper_fraction


def test_numeric_interval_radius_and_negative_rounding_expand_outward():
    mapping = bundle_mapping()
    interval = mapping["observations"][1]
    interval["uncertainty"] = {"model": "absolute_bound", "radius": [0.125]}
    numeric = mapping["observations"][0]
    numeric["value"] = {"kind": "numeric", "values": [-1.0]}
    numeric["uncertainty"] = {"model": "absolute_bound", "radius": [0.1]}
    result = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    compiled = {item.observation_id: item for item in result.observations}
    interval_bound = compiled[uid(101)].numeric_bounds[0]
    assert (interval_bound.lower_fraction, interval_bound.upper_fraction) == (
        Fraction(1, 8),
        Fraction(5, 8),
    )
    negative_bound = compiled[uid(100)].numeric_bounds[0]
    assert (negative_bound.lower_fraction, negative_bound.upper_fraction) == (
        Fraction(-9, 8),
        Fraction(-7, 8),
    )


def test_standard_error_rejects_whole_bundle_without_partial_values():
    mapping = bundle_mapping()
    numeric = mapping["observations"][1]
    numeric["uncertainty"] = {"model": "standard_error", "radius": [0.2]}
    result = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    assert result.disposition is BundleUncertaintyDisposition.ABSTAIN
    assert result.reason == (
        "bundle_uncertainty_preflight:STANDARD_ERROR_UNSUPPORTED"
    )
    assert result.observations == ()
    assert len(result.failures) == 1
    assert result.failures[0].observation_id == uid(101)
    assert result.failures[0].error_code == "STANDARD_ERROR_UNSUPPORTED"


@pytest.mark.parametrize("value", [64.0, -64.0])
def test_out_of_grid_absolute_bound_rejects_bundle_and_never_clamps(value: float):
    mapping = bundle_mapping()
    numeric = mapping["observations"][0]
    numeric["value"] = {"kind": "numeric", "values": [value]}
    numeric["uncertainty"] = {"model": "absolute_bound", "radius": [0.125]}
    result = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    assert result.disposition is BundleUncertaintyDisposition.ABSTAIN
    assert result.reason == (
        "bundle_uncertainty_preflight:RATIONAL_VALUE_GRID_OUT_OF_RANGE"
    )
    assert result.observations == ()
    assert tuple(item.error_code for item in result.failures) == (
        "RATIONAL_VALUE_GRID_OUT_OF_RANGE",
    )


@pytest.mark.parametrize("value", [64.0, -64.0])
def test_exact_grid_extrema_with_zero_radius_remain_legal(value: float):
    mapping = bundle_mapping()
    numeric = mapping["observations"][0]
    numeric["value"] = {"kind": "numeric", "values": [value]}
    numeric["uncertainty"] = {"model": "absolute_bound", "radius": [0.0]}
    result = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    assert result.disposition is BundleUncertaintyDisposition.COMPLETE
    bound = next(
        item for item in result.observations if item.observation_id == uid(100)
    ).numeric_bounds[0]
    assert bound.lower_fraction == bound.upper_fraction == Fraction.from_float(value)


def test_all_observation_failures_are_reported_but_no_success_is_returned():
    mapping = bundle_mapping()
    first = mapping["observations"][0]
    first["value"] = {"kind": "numeric", "values": [64.0]}
    first["uncertainty"] = {"model": "absolute_bound", "radius": [1.0]}
    second = mapping["observations"][1]
    second["uncertainty"] = {"model": "standard_error", "radius": [0.1]}
    result = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    assert result.disposition is BundleUncertaintyDisposition.ABSTAIN
    assert result.reason == "bundle_uncertainty_preflight:MULTIPLE_FAILURES"
    assert result.observations == ()
    assert tuple(item.observation_id for item in result.failures) == (
        uid(100),
        uid(101),
    )
    assert {item.error_code for item in result.failures} == {
        "RATIONAL_VALUE_GRID_OUT_OF_RANGE",
        "STANDARD_ERROR_UNSUPPORTED",
    }


def test_bundle_compilation_is_canonical_order_invariant():
    mapping = bundle_mapping()
    first = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    reordered = deepcopy(mapping)
    reordered["observations"] = list(reversed(reordered["observations"]))
    second = compile_bundle_uncertainty(
        PublicEvidenceBundle.from_mapping(reordered)
    )
    assert second == first
    assert second.result_id == first.result_id


@pytest.mark.parametrize("changed_field", ["provenance", "support", "transform"])
def test_result_identity_binds_entire_public_bundle(changed_field: str):
    mapping = bundle_mapping()
    baseline = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(mapping))
    changed = deepcopy(mapping)
    if changed_field == "provenance":
        changed["observations"][0]["provenance_sha256"] = "f" * 64
    elif changed_field == "support":
        changed["observations"][0]["temporal_support"] = {
            "clock_id": uid(21),
            "start": 0.0,
            "end": 1.0,
        }
    else:
        changed["transform_catalog"][0]["operation"] = "unit_conversion"
        changed["transform_catalog"][0]["parameters"] = [1.0]
    compiled = compile_bundle_uncertainty(PublicEvidenceBundle.from_mapping(changed))
    assert compiled.bundle_content_id != baseline.bundle_content_id
    assert compiled.result_id != baseline.result_id


def test_compiled_result_contains_no_binary_float_and_rows_bind_policy():
    result = compile_bundle_uncertainty(
        PublicEvidenceBundle.from_mapping(bundle_mapping())
    )

    def visit(value: object) -> None:
        assert not isinstance(value, float)
        if is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                visit(item)
        elif isinstance(value, Enum):
            return

    visit(result)
    assert all(
        item.compiler_policy_id == result.compiler_policy_id
        and item.phase2b_exact_freeze_id == result.phase2b_exact_freeze_id
        and item.rational_grid_id == result.rational_grid_id
        for item in result.observations
    )


def test_compiler_does_not_import_projection_selector_or_fixture_modules():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hegel_machine"
        / "phase2b_uncertainty_compiler.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = ("projection", "selector", "phase2_exit", "generator", "evaluator")
    assert not any(
        token in module
        for module in imported
        for token in banned
    )


def test_bundle_and_policy_subclasses_cannot_override_commitment_properties():
    bundle = PublicEvidenceBundle.from_mapping(bundle_mapping())

    class SpoofBundle(PublicEvidenceBundle):
        @property
        def content_id(self) -> str:
            return "phase2b_evidence_" + "0" * 64

    class SpoofPolicy(ExactUncertaintyCompilerPolicy):
        @property
        def policy_id(self) -> str:
            return "phase2b_exact_uncertainty_policy_" + "0" * 64

    spoof_bundle = SpoofBundle(
        **{field.name: getattr(bundle, field.name) for field in fields(bundle)}
    )
    with pytest.raises(TypeError, match="requires PublicEvidenceBundle"):
        compile_bundle_uncertainty(spoof_bundle)
    with pytest.raises(TypeError, match="policy has the wrong type"):
        compile_bundle_uncertainty(bundle, policy=SpoofPolicy())
