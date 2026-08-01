import ast
from dataclasses import replace
from pathlib import Path

import pytest

from hegel_machine.bootstrap import initial_theory
from hegel_machine.phase2b_adapter import (
    AdapterDisposition,
    Phase2BAdapterRegistry,
    enumerate_candidate_hypotheses,
)
from hegel_machine.phase2b_wire import (
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
    PublicEvidenceBundle,
)
from hegel_machine.phase2b_selector import (
    CandidateEvaluation,
    ClosedInterval,
    TypedSelectionDisposition,
    select_typed_candidate_evaluations,
)
from hegel_machine.schema import LawKind


def uid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def adapter_inputs():
    theory = initial_theory()
    law_roles = tuple(
        (law.law_id, role)
        for law in theory.relation_laws
        for role in law.roles
    )
    role_ids = {
        key: uid(100 + index) for index, key in enumerate(law_roles)
    }
    all_role_ids = tuple(sorted(role_ids.values()))
    observables = tuple(
        sorted(
            {
                observable
                for law in theory.relation_laws
                for observable in law.required_observables
            }
        )
    )
    quantity_ids = {
        observable: uid(200 + index)
        for index, observable in enumerate(observables)
    }
    family_ids = {
        kind: uid(700 + index) for index, kind in enumerate(LawKind)
    }
    entity_ids = (uid(10), uid(11), uid(12))
    observations = []
    for index, quantity_id in enumerate(quantity_ids.values()):
        observations.append(
            {
                "observation_id": uid(300 + index),
                "source_channel_id": uid(950),
                "entity_ids": list(entity_ids),
                "role_candidate_ids": list(all_role_ids),
                "quantity_id": quantity_id,
                "value": {"kind": "numeric", "values": [float(index + 1)]},
                "unit_dimension": {"si_exponents": [0, 0, 0, 0, 0, 0, 0]},
                "temporal_support": {
                    "clock_id": uid(951),
                    "start": 0.0,
                    "end": 1.0,
                },
                "spatial_support": None,
                "uncertainty": {"model": "absolute_bound", "radius": [0.1]},
                "provenance_sha256": f"{index % 16:x}" * 64,
                "missingness": "observed",
            }
        )
    mapping = {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "bundle_id": uid(1),
        "entity_candidates": [
            {
                "entity_id": entity_id,
                "role_candidate_ids": list(all_role_ids),
            }
            for entity_id in entity_ids
        ],
        "role_ids": list(all_role_ids),
        "quantity_ids": list(quantity_ids.values()),
        "observations": observations,
        "task_target": {
            "task_id": uid(800),
            "entity_ids": list(entity_ids),
            "quantity_ids": list(quantity_ids.values()),
        },
        "aggregation_graph": {
            "scale_ids": [uid(900), uid(901)],
            "root_scale_ids": [uid(900)],
            "edges": [
                {
                    "source_scale_id": uid(900),
                    "target_scale_id": uid(901),
                    "transform_id": uid(902),
                }
            ],
        },
        "transform_catalog": [
            {
                "transform_id": uid(902),
                "operation": "coarse_graining",
                "parameters": [2.0],
            }
        ],
        "missingness_mask": [],
    }
    bundle = PublicEvidenceBundle.from_mapping(mapping)
    registry = Phase2BAdapterRegistry.from_theory(
        theory,
        family_ids=family_ids,
        role_ids=role_ids,
        quantity_ids=quantity_ids,
    )
    return theory, mapping, bundle, registry


def test_adapter_derives_complete_role_and_scale_grid_from_public_evidence():
    _, _, bundle, registry = adapter_inputs()
    result = enumerate_candidate_hypotheses(bundle, registry)
    assert result.disposition is AdapterDisposition.COMPLETE
    assert result.reason == "complete_internal_candidate_grid"
    assert len(result.hypotheses) == 72
    assert {item.law_kind for item in result.hypotheses} == set(LawKind)
    assert {item.scale_hypothesis_id for item in result.hypotheses} == {
        uid(900),
        uid(901),
    }
    assert all(item.source_observation_ids for item in result.hypotheses)
    assert all(
        item.registry_id == registry.registry_id for item in result.hypotheses
    )
    assert all(
        item.transform_path_ids == ()
        if item.scale_hypothesis_id == uid(900)
        else item.transform_path_ids == (uid(902),)
        for item in result.hypotheses
    )
    commitment = result.candidate_grid_commitment
    assert commitment.adapter_result_id == result.result_id
    assert commitment.expected_candidate_ids == tuple(
        item.candidate_id for item in result.hypotheses
    )


def test_public_selector_reenumerates_bundle_and_rejects_self_signed_partial_grid():
    _, _, bundle, registry = adapter_inputs()
    result = enumerate_candidate_hypotheses(bundle, registry)
    selected = result.hypotheses[0]
    evaluations = tuple(
        CandidateEvaluation(
            candidate_id=item.candidate_id,
            law_kind=item.law_kind,
            role_binding=item.role_binding,
            scale_hypothesis_id=item.scale_hypothesis_id,
            residual=(
                ClosedInterval(0.1, 0.2)
                if item.candidate_id == selected.candidate_id
                else ClosedInterval(3.0, 4.0)
            ),
            tolerance=ClosedInterval(1.0, 1.0),
            completed=True,
            footprint_id=item.footprint_id,
        )
        for item in result.hypotheses
    )
    decision = select_typed_candidate_evaluations(
        evaluations,
        evidence_bundle=bundle,
        adapter_registry=registry,
    )
    assert decision.disposition is TypedSelectionDisposition.UNIQUE_IDENTIFICATION

    keys_by_family = {}
    for item in result.hypotheses:
        key = (item.law_kind, item.role_binding)
        keys_by_family.setdefault(item.law_kind, []).append(key)
    retained_keys = set()
    for law_kind, keys in keys_by_family.items():
        unique_keys = tuple(dict.fromkeys(keys))
        retained_keys.update(
            unique_keys[:2] if law_kind is selected.law_kind else unique_keys[:1]
        )
    truncated = tuple(
        evaluation
        for evaluation in evaluations
        if (evaluation.law_kind, evaluation.role_binding) in retained_keys
    )
    assert len(truncated) == 14
    fail_closed = select_typed_candidate_evaluations(
        truncated,
        evidence_bundle=bundle,
        adapter_registry=registry,
    )
    assert fail_closed.disposition is TypedSelectionDisposition.ABSTAIN
    assert fail_closed.reason == "incomplete_or_drifted_candidate_grid"

    with pytest.raises(TypeError, match="grid_commitment"):
        select_typed_candidate_evaluations(
            evaluations,
            evidence_bundle=bundle,
            adapter_registry=registry,
            grid_commitment=result.candidate_grid_commitment,  # type: ignore[call-arg]
        )


def test_adapter_budget_overflow_returns_no_partial_candidate_grid():
    _, _, bundle, registry = adapter_inputs()
    constrained = replace(registry, maximum_candidate_count=71)
    result = enumerate_candidate_hypotheses(bundle, constrained)
    assert result.disposition is AdapterDisposition.ABSTAIN
    assert result.reason == "candidate_budget_exceeded"
    assert result.hypotheses == ()
    with pytest.raises(ValueError, match="no candidate-grid"):
        _ = result.candidate_grid_commitment


def test_registry_or_public_channel_mismatch_fails_closed():
    _, _, bundle, registry = adapter_inputs()
    first = registry.law_bindings[0]
    changed_roles = tuple(
        (role, uid(999) if index == 0 else wire_id)
        for index, (role, wire_id) in enumerate(first.role_ids)
    )
    changed = replace(
        registry,
        law_bindings=(replace(first, role_ids=changed_roles),)
        + registry.law_bindings[1:],
    )
    result = enumerate_candidate_hypotheses(bundle, changed)
    assert result.disposition is AdapterDisposition.ABSTAIN
    assert result.reason == "registry_role_absent_from_bundle"


def test_nonunique_transform_path_fails_without_partial_hypotheses():
    _, mapping, _, registry = adapter_inputs()
    mapping["aggregation_graph"] = {
        "scale_ids": [uid(899), uid(900), uid(901)],
        "root_scale_ids": [uid(899), uid(900)],
        "edges": [
            {
                "source_scale_id": uid(899),
                "target_scale_id": uid(901),
                "transform_id": uid(898),
            },
            {
                "source_scale_id": uid(900),
                "target_scale_id": uid(901),
                "transform_id": uid(902),
            },
        ],
    }
    mapping["transform_catalog"].append(  # type: ignore[union-attr]
        {
            "transform_id": uid(898),
            "operation": "temporal_aggregation",
            "parameters": [2.0],
        }
    )
    bundle = PublicEvidenceBundle.from_mapping(mapping)
    result = enumerate_candidate_hypotheses(bundle, registry)
    assert result.disposition is AdapterDisposition.ABSTAIN
    assert result.reason == "nonunique_transform_path"
    assert result.hypotheses == ()


def test_adapter_enumeration_is_input_order_invariant():
    _, mapping, bundle, registry = adapter_inputs()
    mapping["entity_candidates"].reverse()  # type: ignore[union-attr]
    mapping["observations"].reverse()  # type: ignore[union-attr]
    mapping["role_ids"].reverse()  # type: ignore[union-attr]
    reordered = PublicEvidenceBundle.from_mapping(mapping)
    assert reordered == bundle
    assert enumerate_candidate_hypotheses(reordered, registry) == (
        enumerate_candidate_hypotheses(bundle, registry)
    )


def test_adapter_source_has_no_phase2a_fixture_or_answer_dependency():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hegel_machine"
        / "phase2b_adapter.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert all("phase2_exit" not in name for name in imported)
    assert all("answer" not in name for name in imported)


def test_registry_requires_exact_theory_observable_coverage():
    theory, _, _, _ = adapter_inputs()
    family_ids = {kind: uid(700 + index) for index, kind in enumerate(LawKind)}
    role_ids = {
        (law.law_id, role): uid(100 + index)
        for index, (law, role) in enumerate(
            (item for law in theory.relation_laws for item in ((law, role) for role in law.roles))
        )
    }
    with pytest.raises(ValueError, match="exactly cover"):
        Phase2BAdapterRegistry.from_theory(
            theory,
            family_ids=family_ids,
            role_ids=role_ids,
            quantity_ids={},
        )
