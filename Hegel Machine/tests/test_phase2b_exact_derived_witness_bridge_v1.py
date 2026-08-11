import inspect
from collections import Counter
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from itertools import product

import pytest

import hegel_machine.phase2b_exact_bridge_v1 as legacy_bridge
import hegel_machine.phase2b_exact_derived_witness_bridge_v1 as bridge
import hegel_machine.phase2b_exact_transform_semantics_v1 as tx
from hegel_machine.bootstrap import initial_theory
from hegel_machine.hashing import stable_hash
from hegel_machine.phase2_exit import HARD_NEGATIVE_OBSERVABLES, PASS_OBSERVABLES
from hegel_machine.phase2b_adapter import Phase2BAdapterRegistry
from hegel_machine.phase2b_uncertainty_compiler import compile_bundle_uncertainty
from hegel_machine.phase2b_wire import (
    BooleanValue,
    NumericValue,
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
    PublicEvidenceBundle,
    TransformOperation,
)
from hegel_machine.schema import LawKind


def uid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


ROOT_SCALE = uid(100)
TARGET_SCALE = uid(101)
OTHER_ROOT_SCALE = uid(102)
SAMPLING_ROOT_SCALE = uid(103)
SAMPLING_TARGET_SCALE = uid(104)
IDENTITY_TRANSFORM_ID = uid(200)
SAMPLING_TRANSFORM_ID = uid(201)
SOURCE_CHANNEL_ID = uid(202)
UNIT_ID = uid(203)
CLOCK_ID = uid(204)


def _theory_registry_and_ids():
    theory = initial_theory()
    role_keys = tuple(
        (law.law_id, role)
        for law in theory.relation_laws
        for role in law.roles
    )
    role_ids = {key: uid(1_000 + index) for index, key in enumerate(role_keys)}
    entity_ids = {
        (key, variant): uid(2_000 + 2 * index + variant)
        for index, key in enumerate(role_keys)
        for variant in (0, 1)
    }
    observable_names = tuple(
        sorted(
            {
                observable
                for law in theory.relation_laws
                for observable in law.required_observables
            }
        )
    )
    quantity_ids = {
        observable: uid(3_000 + index)
        for index, observable in enumerate(observable_names)
    }
    family_ids = {kind: uid(4_000 + index) for index, kind in enumerate(LawKind)}
    registry = Phase2BAdapterRegistry.from_theory(
        theory,
        family_ids=family_ids,
        role_ids=role_ids,
        quantity_ids=quantity_ids,
    )
    return theory, registry, {
        "role_ids": role_ids,
        "entity_ids": entity_ids,
        "quantity_ids": quantity_ids,
        "family_ids": family_ids,
    }


def _value_mapping(raw_value: object) -> tuple[dict[str, object], dict[str, object]]:
    if type(raw_value) is bool:
        return (
            {"kind": "boolean", "value": raw_value},
            {"model": "not_applicable", "radius": []},
        )
    values = (
        list(raw_value)
        if isinstance(raw_value, (tuple, list))
        else [raw_value]
    )
    if not values:
        values = [0.0]
    return (
        {"kind": "numeric", "values": values},
        {"model": "absolute_bound", "radius": [0.0 for _ in values]},
    )


def _full_observation_mappings(
    theory,
    ids,
    *,
    supports: tuple[tuple[float, float] | None, ...],
    pass_kinds_by_slice: tuple[frozenset[LawKind], ...],
    omit_observable: str | None = None,
    duplicate_observable: str | None = None,
    all_entity_variants: bool = False,
    pass_all_bindings: bool = False,
) -> list[dict[str, object]]:
    assert len(supports) == len(pass_kinds_by_slice)
    observations: list[dict[str, object]] = []
    observation_index = 10_000
    for support, pass_kinds in zip(
        supports,
        pass_kinds_by_slice,
        strict=True,
    ):
        for law in theory.relation_laws:
            for observable_name in law.required_observables:
                if observable_name == omit_observable:
                    continue
                witness_roles = tuple(
                    role
                    for role, names in law.role_observable_requirements
                    if observable_name in names
                )
                if not witness_roles:
                    witness_roles = law.roles
                variant_rows = (
                    tuple(product((0, 1), repeat=len(witness_roles)))
                    if all_entity_variants
                    else ((0,) * len(witness_roles),)
                )
                for variants in variant_rows:
                    use_pass = law.kind in pass_kinds and (
                        pass_all_bindings or all(variant == 0 for variant in variants)
                    )
                    payload = (
                        PASS_OBSERVABLES[law.kind]
                        if use_pass
                        else HARD_NEGATIVE_OBSERVABLES[law.kind]
                    )
                    entity_ids = sorted(
                        ids["entity_ids"][((law.law_id, role), variant)]
                        for role, variant in zip(
                            witness_roles,
                            variants,
                            strict=True,
                        )
                    )
                    role_ids = sorted(
                        ids["role_ids"][(law.law_id, role)]
                        for role in witness_roles
                    )
                    value, uncertainty = _value_mapping(payload[observable_name])
                    observation = {
                        "observation_id": uid(observation_index),
                        "source_channel_id": SOURCE_CHANNEL_ID,
                        "entity_ids": entity_ids,
                        "role_candidate_ids": role_ids,
                        "quantity_id": ids["quantity_ids"][observable_name],
                        "value": value,
                        "unit_dimension": {
                            "si_exponents": [0, 0, 0, 0, 0, 0, 0]
                        },
                        "temporal_support": (
                            None
                            if support is None
                            else {
                                "clock_id": CLOCK_ID,
                                "start": support[0],
                                "end": support[1],
                            }
                        ),
                        "spatial_support": None,
                        "uncertainty": uncertainty,
                        "provenance_sha256": f"{observation_index % 16:x}" * 64,
                        "missingness": "observed",
                    }
                    observations.append(observation)
                    observation_index += 1
                    if observable_name == duplicate_observable:
                        duplicate = dict(observation)
                        duplicate["observation_id"] = uid(observation_index)
                        duplicate["provenance_sha256"] = (
                            f"{observation_index % 16:x}" * 64
                        )
                        observations.append(duplicate)
                        observation_index += 1
    return observations


def _base_mapping(
    theory,
    ids,
    observations: list[dict[str, object]],
    *,
    scale_ids: list[str],
    root_scale_ids: list[str],
    edges: list[dict[str, object]],
    catalog: list[dict[str, object]],
    task_entity_ids: list[str] | None = None,
    task_quantity_ids: list[str] | None = None,
    entity_variants: tuple[int, ...] = (0,),
) -> dict[str, object]:
    all_entities = [
        ids["entity_ids"][(key, variant)]
        for key in (
            (law.law_id, role)
            for law in theory.relation_laws
            for role in law.roles
        )
        for variant in entity_variants
    ]
    all_quantities = list(ids["quantity_ids"].values())
    return {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "bundle_id": uid(1),
        "entity_candidates": [
            {
                "entity_id": ids["entity_ids"][(key, variant)],
                "role_candidate_ids": [ids["role_ids"][key]],
            }
            for key in (
                (law.law_id, role)
                for law in theory.relation_laws
                for role in law.roles
            )
            for variant in entity_variants
        ],
        "role_ids": list(ids["role_ids"].values()),
        "quantity_ids": all_quantities,
        "observations": observations,
        "task_target": {
            "task_id": uid(2),
            "entity_ids": all_entities if task_entity_ids is None else task_entity_ids,
            "quantity_ids": (
                all_quantities if task_quantity_ids is None else task_quantity_ids
            ),
        },
        "aggregation_graph": {
            "scale_ids": scale_ids,
            "root_scale_ids": root_scale_ids,
            "edges": edges,
        },
        "transform_catalog": catalog,
        "missingness_mask": [],
    }


def _value_kind(observation) -> tx.ComponentValueKind:
    if type(observation.value) is BooleanValue:
        return tx.ComponentValueKind.BOOLEAN
    return tx.ComponentValueKind.NUMERIC_INTERVAL


def _exact_temporal(observation):
    return (
        None
        if observation.temporal_support is None
        else tx.ExactTemporalSupport.from_wire(observation.temporal_support)
    )


def _metadata(
    base: PublicEvidenceBundle,
    scale_by_observation_id: dict[str, str],
) -> tuple[tx.ObservationComponentMetadata, ...]:
    result = []
    for index, observation in enumerate(base.observations):
        width = (
            len(observation.value.values)
            if type(observation.value) is NumericValue
            else 1
        )
        boolean = type(observation.value) is BooleanValue
        result.append(
            tx.ObservationComponentMetadata(
                observation_id=observation.observation_id,
                scale_id=scale_by_observation_id[observation.observation_id],
                component_ids=tuple(
                    uid(30_000 + 256 * index + ordinal)
                    for ordinal in range(width)
                ),
                axis=(tx.ComponentAxis.CONTROL if boolean else tx.ComponentAxis.SCALAR),
                value_role=(
                    tx.ComponentValueRole.BOOLEAN_CONTROL
                    if boolean
                    else tx.ComponentValueRole.INTENSIVE
                ),
                unit_id=None if boolean else UNIT_ID,
            )
        )
    return tuple(sorted(result, key=lambda item: item.observation_id))


def _component_descriptors(
    base: PublicEvidenceBundle,
    metadata: tuple[tx.ObservationComponentMetadata, ...],
) -> dict[str, tuple[tx.ComponentDescriptor, ...]]:
    metadata_by_id = {item.observation_id: item for item in metadata}
    result = {}
    for observation in base.observations:
        item = metadata_by_id[observation.observation_id]
        result[observation.observation_id] = tuple(
            tx.ComponentDescriptor(
                ref=tx.ComponentRef(
                    item.scale_id,
                    observation.observation_id,
                    ordinal,
                    component_id,
                ),
                axis=item.axis,
                value_role=item.value_role,
                unit_id=item.unit_id,
                si_exponents=observation.unit_dimension.si_exponents,
                coordinate_frame_id=item.coordinate_frame_id,
                temporal_support=_exact_temporal(observation),
                spatial_support=None,
            )
            for ordinal, component_id in enumerate(item.component_ids)
        )
    return result


def _root_observation_descriptor(
    observation,
    descriptors: tuple[tx.ComponentDescriptor, ...],
) -> tx.DerivedObservationDescriptor:
    return tx.DerivedObservationDescriptor(
        scale_id=descriptors[0].ref.scale_id,
        observation_id=observation.observation_id,
        source_channel_id=observation.source_channel_id,
        entity_ids=observation.entity_ids,
        role_candidate_ids=observation.role_candidate_ids,
        quantity_id=observation.quantity_id,
        unit_id=descriptors[0].unit_id,
        si_exponents=observation.unit_dimension.si_exponents,
        temporal_support=_exact_temporal(observation),
        spatial_support=None,
        provenance_sha256=observation.provenance_sha256,
        source_observation_ids=(observation.observation_id,),
        value_kind=_value_kind(observation),
        component_refs=tuple(item.ref for item in descriptors),
    )


def _descriptor_without_provenance(descriptor) -> dict[str, object]:
    return {
        field.name: getattr(descriptor, field.name)
        for field in fields(descriptor)
        if field.name != "provenance_sha256"
    }


def _sign_contract_outputs(
    base: PublicEvidenceBundle,
    metadata: tuple[tx.ObservationComponentMetadata, ...],
    contract: tx.ExactTransformContract,
) -> tx.ExactTransformContract:
    compiled = compile_bundle_uncertainty(base)
    assert compiled.disposition.value == "complete"
    compiled_by_id = {item.observation_id: item for item in compiled.observations}
    base_by_id = {item.observation_id: item for item in base.observations}
    roots = _component_descriptors(base, metadata)
    root_observations = {
        observation_id: _root_observation_descriptor(
            base_by_id[observation_id],
            descriptors,
        )
        for observation_id, descriptors in roots.items()
    }
    input_by_ref = {
        descriptor.ref: observation_id
        for observation_id, descriptors in roots.items()
        for descriptor in descriptors
    }
    inputs_by_output = {
        row.output_ref: tuple(term.input_ref for term in row.terms)
        for row in contract.kernel_rows
    }
    inputs_by_output.update(
        {
            mapping.output_ref: (mapping.input_ref,)
            for mapping in contract.discrete_mappings
        }
    )
    semantics_id = contract.semantics_id
    signed = []
    for output in contract.output_observations:
        source_ids = tuple(
            sorted(
                {
                    input_by_ref[input_ref]
                    for output_ref in output.component_refs
                    for input_ref in inputs_by_output[output_ref]
                }
            )
        )
        provenance = stable_hash(
            {
                "input_observation_descriptors": tuple(
                    (
                        root_observations[source_id].descriptor_id,
                        root_observations[source_id].provenance_sha256,
                        root_observations[source_id].source_observation_ids,
                    )
                    for source_id in source_ids
                ),
                "input_uncertainty_compilation_ids": tuple(
                    sorted(
                        {
                            compiled_by_id[source_id].compilation_id
                            for source_id in source_ids
                        }
                    )
                ),
                "contract_semantics_without_provenance_id": semantics_id,
                "ordered_transform_path_ids": (contract.transform_id,),
                "ordered_contract_semantics_ids": (semantics_id,),
                "output_observation": _descriptor_without_provenance(output),
            },
            prefix="",
        )
        signed.append(replace(output, provenance_sha256=provenance))
    result = replace(contract, output_observations=tuple(signed))
    assert result.semantics_id == semantics_id
    return result


def _identity_authority(
    *,
    supports: tuple[tuple[float, float] | None, ...] = ((0.0, 0.0),),
    pass_kinds_by_slice: tuple[frozenset[LawKind], ...] = (frozenset(),),
    duplicate_observable: str | None = None,
    task_quantity_ids: list[str] | None = None,
    all_entity_variants: bool = False,
    pass_all_bindings: bool = False,
):
    theory, registry, ids = _theory_registry_and_ids()
    observations = _full_observation_mappings(
        theory,
        ids,
        supports=supports,
        pass_kinds_by_slice=pass_kinds_by_slice,
        duplicate_observable=duplicate_observable,
        all_entity_variants=all_entity_variants,
        pass_all_bindings=pass_all_bindings,
    )
    base = PublicEvidenceBundle.from_mapping(
        _base_mapping(
            theory,
            ids,
            observations,
            scale_ids=[ROOT_SCALE, TARGET_SCALE],
            root_scale_ids=[ROOT_SCALE],
            edges=[
                {
                    "source_scale_id": ROOT_SCALE,
                    "target_scale_id": TARGET_SCALE,
                    "transform_id": IDENTITY_TRANSFORM_ID,
                }
            ],
            catalog=[
                {
                    "transform_id": IDENTITY_TRANSFORM_ID,
                    "operation": TransformOperation.IDENTITY.value,
                    "parameters": [],
                }
            ],
            task_quantity_ids=task_quantity_ids,
            entity_variants=(0, 1) if all_entity_variants else (0,),
        )
    )
    metadata = _metadata(
        base,
        {item.observation_id: ROOT_SCALE for item in base.observations},
    )
    roots = _component_descriptors(base, metadata)
    output_descriptors = []
    output_observations = []
    rows = []
    discrete = []
    for index, observation in enumerate(base.observations):
        source = roots[observation.observation_id]
        output_observation_id = uid(50_000 + index)
        outputs = tuple(
            replace(
                descriptor,
                ref=tx.ComponentRef(
                    TARGET_SCALE,
                    output_observation_id,
                    ordinal,
                    uid(60_000 + 256 * index + ordinal),
                ),
            )
            for ordinal, descriptor in enumerate(source)
        )
        output_descriptors.extend(outputs)
        if _value_kind(observation) is tx.ComponentValueKind.NUMERIC_INTERVAL:
            rows.extend(
                tx.ExactSparseAffineRow(
                    output.ref,
                    (tx.ExactSparseTerm(input_.ref, tx.ExactTransformAtom(1)),),
                )
                for input_, output in zip(source, outputs, strict=True)
            )
        else:
            discrete.extend(
                tx.ExactDiscreteMapping(input_.ref, output.ref)
                for input_, output in zip(source, outputs, strict=True)
            )
        output_observations.append(
            tx.DerivedObservationDescriptor(
                scale_id=TARGET_SCALE,
                observation_id=output_observation_id,
                source_channel_id=observation.source_channel_id,
                entity_ids=observation.entity_ids,
                role_candidate_ids=observation.role_candidate_ids,
                quantity_id=observation.quantity_id,
                unit_id=outputs[0].unit_id,
                si_exponents=observation.unit_dimension.si_exponents,
                temporal_support=_exact_temporal(observation),
                spatial_support=None,
                provenance_sha256="0" * 64,
                source_observation_ids=(observation.observation_id,),
                value_kind=_value_kind(observation),
                component_refs=tuple(item.ref for item in outputs),
            )
        )
    contract = tx.ExactTransformContract(
        transform_id=IDENTITY_TRANSFORM_ID,
        operation=TransformOperation.IDENTITY,
        source_scale_id=ROOT_SCALE,
        target_scale_id=TARGET_SCALE,
        input_components=tuple(
            sorted(item.ref for group in roots.values() for item in group)
        ),
        output_components=tuple(sorted(output_descriptors, key=lambda item: item.ref)),
        output_observations=tuple(
            sorted(output_observations, key=lambda item: item.observation_id)
        ),
        kernel_rows=tuple(sorted(rows, key=lambda item: item.output_ref)),
        discrete_mappings=tuple(
            sorted(discrete, key=lambda item: (item.output_ref, item.input_ref))
        ),
        certificate=tx.IdentityTransformCertificate(),
    )
    contract = _sign_contract_outputs(base, metadata, contract)
    authority = tx.PublicTransformEvidenceBundleV2(
        tx.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
        base,
        metadata,
        (contract,),
    )
    return theory, registry, ids, authority


def _sampling_authority(*, duplicate_other_observable: str | None = None):
    """Build a full-registry forest with one genuine sampling edge.

    Non-sampled quantities stay on an independent root.  The sampled quantity
    occupies its own root so the frozen single-series sampling contract remains
    valid while the global inventory still exactly covers the registry.
    """

    theory, registry, ids = _theory_registry_and_ids()
    observations = _full_observation_mappings(
        theory,
        ids,
        supports=((0.0, 0.0),),
        pass_kinds_by_slice=(frozenset(),),
        duplicate_observable=duplicate_other_observable,
    )
    sampled_quantity = ids["quantity_ids"]["x_low"]
    sampled_template = next(
        item for item in observations if item["quantity_id"] == sampled_quantity
    )
    observations = [
        item for item in observations if item["quantity_id"] != sampled_quantity
    ]
    for index, point in enumerate((0.0, 1.0, 2.0)):
        item = dict(sampled_template)
        item["observation_id"] = uid(70_000 + index)
        item["temporal_support"] = {
            "clock_id": CLOCK_ID,
            "start": point,
            "end": point,
        }
        item["provenance_sha256"] = f"{(index + 10) % 16:x}" * 64
        observations.append(item)
    base = PublicEvidenceBundle.from_mapping(
        _base_mapping(
            theory,
            ids,
            observations,
            scale_ids=[
                OTHER_ROOT_SCALE,
                SAMPLING_ROOT_SCALE,
                SAMPLING_TARGET_SCALE,
            ],
            root_scale_ids=[OTHER_ROOT_SCALE, SAMPLING_ROOT_SCALE],
            edges=[
                {
                    "source_scale_id": SAMPLING_ROOT_SCALE,
                    "target_scale_id": SAMPLING_TARGET_SCALE,
                    "transform_id": SAMPLING_TRANSFORM_ID,
                }
            ],
            catalog=[
                {
                    "transform_id": SAMPLING_TRANSFORM_ID,
                    "operation": TransformOperation.SAMPLING_RESOLUTION.value,
                    "parameters": [],
                }
            ],
        )
    )
    scale_by_observation = {
        observation.observation_id: (
            SAMPLING_ROOT_SCALE
            if observation.quantity_id == sampled_quantity
            else OTHER_ROOT_SCALE
        )
        for observation in base.observations
    }
    metadata = _metadata(base, scale_by_observation)
    observations_by_id = {
        observation.observation_id: observation for observation in base.observations
    }
    metadata = tuple(
        replace(item, axis=tx.ComponentAxis.TEMPORAL)
        if observations_by_id[item.observation_id].quantity_id
        == sampled_quantity
        else item
        for item in metadata
    )
    roots = _component_descriptors(base, metadata)
    sampled_observations = tuple(
        sorted(
            (
                observation
                for observation in base.observations
                if observation.quantity_id == sampled_quantity
            ),
            key=lambda item: item.temporal_support.start,
        )
    )
    sampled_sources = tuple(
        roots[observation.observation_id][0]
        for observation in sampled_observations
    )
    selected_sources = sampled_sources[:2]
    output_descriptors = tuple(
        replace(
            source,
            ref=tx.ComponentRef(
                SAMPLING_TARGET_SCALE,
                uid(80_000 + index),
                0,
                uid(81_000 + index),
            ),
        )
        for index, source in enumerate(selected_sources)
    )
    rows = tuple(
        tx.ExactSparseAffineRow(
            output.ref,
            (tx.ExactSparseTerm(source.ref, tx.ExactTransformAtom(1)),),
        )
        for source, output in zip(
            selected_sources,
            output_descriptors,
            strict=True,
        )
    )
    output_observations = tuple(
        tx.DerivedObservationDescriptor(
            scale_id=SAMPLING_TARGET_SCALE,
            observation_id=output.ref.observation_id,
            source_channel_id=source_observation.source_channel_id,
            entity_ids=source_observation.entity_ids,
            role_candidate_ids=source_observation.role_candidate_ids,
            quantity_id=source_observation.quantity_id,
            unit_id=output.unit_id,
            si_exponents=source_observation.unit_dimension.si_exponents,
            temporal_support=output.temporal_support,
            spatial_support=None,
            provenance_sha256="0" * 64,
            source_observation_ids=(source_observation.observation_id,),
            value_kind=tx.ComponentValueKind.NUMERIC_INTERVAL,
            component_refs=(output.ref,),
        )
        for source_observation, output in zip(
            sampled_observations[:2],
            output_descriptors,
            strict=True,
        )
    )
    contract = tx.ExactTransformContract(
        transform_id=SAMPLING_TRANSFORM_ID,
        operation=TransformOperation.SAMPLING_RESOLUTION,
        source_scale_id=SAMPLING_ROOT_SCALE,
        target_scale_id=SAMPLING_TARGET_SCALE,
        input_components=tuple(item.ref for item in sampled_sources),
        output_components=output_descriptors,
        output_observations=output_observations,
        kernel_rows=rows,
        discrete_mappings=(),
        certificate=tx.SamplingResolutionCertificate(
            axis=tx.ComponentAxis.TEMPORAL,
            selected_inputs=tuple(item.ref for item in selected_sources),
            discarded_inputs=(sampled_sources[2].ref,),
            grid_points=((tx.ExactTransformAtom(0),), (tx.ExactTransformAtom(1),)),
            grid_dimension=1,
            grid_frame_id=None,
        ),
    )
    contract = _sign_contract_outputs(base, metadata, contract)
    authority = tx.PublicTransformEvidenceBundleV2(
        tx.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
        base,
        metadata,
        (contract,),
    )
    return theory, registry, ids, authority


def _visit_without_float(value: object) -> None:
    assert type(value) is not float
    if is_dataclass(value):
        for field in fields(value):
            _visit_without_float(getattr(value, field.name))
    elif type(value) is tuple:
        for item in value:
            _visit_without_float(item)
    elif isinstance(value, Enum):
        return


def _run(authority, theory, registry):
    return bridge.run_exact_derived_witness_bridge(
        authority=authority,
        theory=theory,
        registry=registry,
    )


def _assert_atomic_bridge_abstention(run) -> None:
    assert run.disposition is bridge.ExactBridgeDisposition.ABSTAIN
    assert run.inventory is None
    assert run.compilation.candidate_grid is None
    assert run.compilation.candidate_grid_commitment_id is None
    assert run.compilation.scale_aggregate_commitment_id is None
    assert run.compilation.evaluations == ()
    assert run.compilation.scale_aggregates == ()
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN
    assert run.decision.candidate_grid_commitment_id is None
    assert run.decision.scale_aggregate_commitment_id is None
    assert run.decision.evaluated_candidate_ids == ()
    assert run.decision.consumed_scale_aggregate_ids == ()


def _replace_completed_normalized(evaluation, lower: Fraction, upper: Fraction):
    assert evaluation.completed and evaluation.tolerance is not None
    assert evaluation.tolerance.is_point
    tolerance = evaluation.tolerance.lower_fraction
    return replace(
        evaluation,
        residual=bridge.ExactInterval.from_fractions(
            tolerance * lower,
            tolerance * upper,
        ),
        normalized=bridge.ExactInterval.from_fractions(lower, upper),
    )


def _run_with_consistent_selector_fault(monkeypatch, mode: str):
    theory, registry, _, authority = _identity_authority(
        pass_kinds_by_slice=(frozenset({LawKind.SYMMETRY}),),
        all_entity_variants=True,
    )
    original_compile = bridge._compile_grid

    def injected_compile(
        grid,
        *,
        theory,
        inventory,
        transform_result,
    ):
        evaluations, aggregates, error = original_compile(
            grid,
            theory=theory,
            inventory=inventory,
            transform_result=transform_result,
        )
        assert evaluations is not None and aggregates is not None and error is None
        selected = tuple(
            item
            for item in evaluations
            if item.status is bridge.ExactCandidateStatus.PASS
        )
        assert len(selected) == 2
        selected_key = (selected[0].law_kind, selected[0].role_binding)
        competitor = next(
            item
            for item in evaluations
            if item.status is bridge.ExactCandidateStatus.FAIL
            and (item.law_kind, item.role_binding) != selected_key
        )
        changed = []
        for item in evaluations:
            if mode == "selected_inconclusive" and item is selected[0]:
                item = _replace_completed_normalized(
                    item,
                    Fraction(1, 2),
                    Fraction(3, 2),
                )
            elif mode == "competitor_inconclusive" and item is competitor:
                item = _replace_completed_normalized(
                    item,
                    Fraction(1, 2),
                    Fraction(3, 2),
                )
            elif mode == "insufficient_margin" and item in selected:
                item = _replace_completed_normalized(
                    item,
                    Fraction(3, 4),
                    Fraction(3, 4),
                )
            elif mode == "insufficient_margin" and item is competitor:
                item = _replace_completed_normalized(
                    item,
                    Fraction(3, 2),
                    Fraction(3, 2),
                )
            changed.append(item)
        changed_evaluations = tuple(
            sorted(changed, key=lambda item: item.candidate_id)
        )
        changed_aggregates = bridge._scale_aggregates(
            changed_evaluations,
            grid,
            grid.candidate_grid_commitment_id,
        )
        return changed_evaluations, changed_aggregates, None

    monkeypatch.setattr(bridge, "_compile_grid", injected_compile)
    return _run(authority, theory, registry)


def test_authoritative_api_is_keyword_only_has_no_caller_receipt_and_no_float():
    theory, registry, _, authority = _identity_authority()
    signature = inspect.signature(bridge.run_exact_derived_witness_bridge)
    assert tuple(signature.parameters) == ("authority", "theory", "registry")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    run = _run(authority, theory, registry)
    assert type(run) is bridge.ExactDerivedBridgeRun
    _visit_without_float(run)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        bridge.run_exact_derived_witness_bridge(
            authority=authority,
            theory=theory,
            registry=registry,
            transform_receipt=run.transform_result,
        )


def test_authoritative_path_never_calls_legacy_candidate_enumeration(monkeypatch):
    theory, registry, _, authority = _identity_authority()

    def forbidden_legacy_enumeration(*_args, **_kwargs):
        raise AssertionError("legacy base-footprint enumeration was called")

    monkeypatch.setattr(
        legacy_bridge,
        "enumerate_candidate_hypotheses",
        forbidden_legacy_enumeration,
    )
    run = _run(authority, theory, registry)
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE


def test_identity_fixture_commits_the_complete_six_law_slice_grid_and_lineage():
    theory, registry, _, authority = _identity_authority()
    run = _run(authority, theory, registry)

    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.transform_result.disposition is tx.TransformCompilationDisposition.COMPLETE
    assert run.inventory is not None
    grid = run.compilation.candidate_grid
    assert grid is not None
    assert run.compilation.candidate_grid_commitment_id == (
        grid.candidate_grid_commitment_id
    )
    assert len(run.inventory.observations) == 70
    assert len(grid.candidates) == 12
    assert len(run.compilation.evaluations) == len(grid.candidates)
    assert len(run.compilation.scale_aggregates) == 12
    assert {item.law_kind for item in grid.candidates} == set(LawKind)
    assert {
        item.law_kind: sum(
            candidate.law_kind is item.law_kind for candidate in grid.candidates
        )
        for item in grid.candidates
    } == {kind: 2 for kind in LawKind}
    assert all(
        len(slot.matches) == 1
        for candidate in grid.candidates
        for slot in candidate.slots
    )
    assert tuple(item.candidate_id for item in grid.candidates) == tuple(
        item.candidate_id for item in run.compilation.evaluations
    )
    assert all(item.completed for item in run.compilation.evaluations)

    root_inventory = tuple(
        item
        for item in run.inventory.observations
        if item.descriptor.scale_id == ROOT_SCALE
    )
    target_inventory = tuple(
        item
        for item in run.inventory.observations
        if item.descriptor.scale_id == TARGET_SCALE
    )
    assert len(root_inventory) == len(target_inventory) == 35
    assert all(item.ordered_transform_path_ids == () for item in root_inventory)
    assert all(
        item.ordered_transform_path_ids == (IDENTITY_TRANSFORM_ID,)
        and len(item.ordered_contract_semantics_ids) == 1
        for item in target_inventory
    )
    assert all(
        item.wrapper_content_id == authority.content_id
        and item.transform_result_id == run.transform_result_id
        and item.base_bundle_content_id == authority.base_bundle.content_id
        and item.contract_commitment_id
        == run.transform_result.contract_commitment_id
        and item.graph_commitment_id == run.transform_result.graph_commitment_id
        for item in run.inventory.observations
    )
    matched_inventory_ids = {
        match.inventory_observation_id
        for candidate in grid.candidates
        for slot in candidate.slots
        for match in slot.matches
    }
    assert matched_inventory_ids == {
        item.inventory_observation_id for item in run.inventory.observations
    }
    aggregate_ids = tuple(
        sorted(item.scale_aggregate_id for item in run.compilation.scale_aggregates)
    )
    assert run.decision.consumed_scale_aggregate_ids == aggregate_ids


def test_genuine_sampling_points_are_distinct_exact_slices_not_ambiguous():
    theory, registry, ids, authority = _sampling_authority()
    run = _run(authority, theory, registry)

    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.inventory is not None
    grid = run.compilation.candidate_grid
    assert grid is not None
    sampled_quantity = ids["quantity_ids"]["x_low"]
    sampled_inventory = tuple(
        item
        for item in run.inventory.observations
        if item.descriptor.quantity_id == sampled_quantity
    )
    assert len(sampled_inventory) == 5
    assert sum(
        item.descriptor.scale_id == SAMPLING_ROOT_SCALE
        for item in sampled_inventory
    ) == 3
    assert sum(
        item.descriptor.scale_id == SAMPLING_TARGET_SCALE
        for item in sampled_inventory
    ) == 2
    sampled_slices = {
        item.support_slice.support_slice_id for item in sampled_inventory
    }
    assert len(sampled_slices) == 5
    for item in sampled_inventory:
        assert item.support_slice.temporal_support == item.descriptor.temporal_support
        assert item.support_slice.spatial_support == item.descriptor.spatial_support

    monotonicity = tuple(
        candidate
        for candidate in grid.candidates
        if candidate.law_kind is LawKind.MONOTONICITY
        and candidate.support_slice.support_slice_id in sampled_slices
    )
    assert len(monotonicity) == 5
    for candidate in monotonicity:
        slot = next(item for item in candidate.slots if item.observable_id == "x_low")
        assert len(slot.matches) == 1
        assert slot.support_slice_id == candidate.support_slice.support_slice_id
    other_root_candidate = next(
        candidate
        for candidate in grid.candidates
        if candidate.law_kind is LawKind.MONOTONICITY
        and candidate.support_slice.scale_id == OTHER_ROOT_SCALE
    )
    missing_slot = next(
        item for item in other_root_candidate.slots if item.observable_id == "x_low"
    )
    assert missing_slot.matches == ()
    assert all(
        len(slot.matches) <= 1
        for candidate in grid.candidates
        for slot in candidate.slots
    )


def test_pass_and_fail_slices_on_one_scale_aggregate_to_inconclusive():
    theory, registry, _, authority = _identity_authority(
        supports=((0.0, 0.0), (1.0, 1.0)),
        pass_kinds_by_slice=(
            frozenset({LawKind.SYMMETRY}),
            frozenset(),
        ),
    )
    run = _run(authority, theory, registry)
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    symmetry_evaluations = tuple(
        item
        for item in run.compilation.evaluations
        if item.law_kind is LawKind.SYMMETRY
    )
    assert {item.status for item in symmetry_evaluations} == {
        bridge.ExactCandidateStatus.PASS,
        bridge.ExactCandidateStatus.FAIL,
    }
    symmetry_aggregates = tuple(
        item
        for item in run.compilation.scale_aggregates
        if item.law_kind is LawKind.SYMMETRY
    )
    assert len(symmetry_aggregates) == 2
    assert all(
        item.status is bridge.ExactCandidateStatus.INCONCLUSIVE
        and len(item.slice_evaluation_ids) == 2
        for item in symmetry_aggregates
    )
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN


def test_same_slice_many_match_is_unconsumed_and_abstains_atomically():
    theory, registry, _, authority = _sampling_authority(
        duplicate_other_observable="forward"
    )
    run = _run(authority, theory, registry)
    assert run.reason == "unused_or_ambiguously_consumed_derived_observation"
    assert run.transform_result.disposition is tx.TransformCompilationDisposition.COMPLETE
    _assert_atomic_bridge_abstention(run)


def test_strict_task_scope_rejects_before_transform_receipt(monkeypatch):
    theory, registry, _, authority = _identity_authority()
    base = authority.base_bundle
    changed_base = replace(
        base,
        task_target=replace(
            base.task_target,
            quantity_ids=base.task_target.quantity_ids[:-1],
        ),
    )
    changed = replace(authority, base_bundle=changed_base)

    def forbidden_transform(_authority):
        raise AssertionError("transform ran before strict task-scope preflight")

    monkeypatch.setattr(bridge, "run_exact_transform_semantics", forbidden_transform)
    rejected = _run(changed, theory, registry)
    assert type(rejected) is bridge.ExactDerivedBridgePreflightRejection
    assert rejected.reason == "strict_task_quantity_scope_mismatch"
    assert rejected.transform_result is None
    assert rejected.inventory is None
    assert rejected.compilation is None
    assert rejected.decision is None


def test_invalid_transform_provenance_is_atomic_and_never_builds_inventory():
    theory, registry, _, authority = _identity_authority()
    contract = authority.transform_contracts[0]
    changed_contract = replace(
        contract,
        output_observations=(
            replace(contract.output_observations[0], provenance_sha256="f" * 64),
            *contract.output_observations[1:],
        ),
    )
    changed = replace(authority, transform_contracts=(changed_contract,))
    run = _run(changed, theory, registry)
    assert run.transform_result.disposition is tx.TransformCompilationDisposition.ABSTAIN
    assert run.transform_result.observations == ()
    assert run.transform_result.components == ()
    _assert_atomic_bridge_abstention(run)


def test_missing_temporal_and_spatial_support_is_not_collapsed_into_a_slice():
    theory, registry, _, authority = _identity_authority(supports=(None,))
    run = _run(authority, theory, registry)
    assert run.transform_result.disposition is tx.TransformCompilationDisposition.COMPLETE
    assert run.reason == "explicit_support_required"
    _assert_atomic_bridge_abstention(run)


def test_authoritative_entry_rejects_outer_and_nested_external_types():
    theory, registry, _, authority = _identity_authority()

    class ExternalAuthority(tx.PublicTransformEvidenceBundleV2):
        pass

    class ExternalTheory(type(theory)):
        pass

    class ExternalRegistry(Phase2BAdapterRegistry):
        pass

    class ExternalBundle(PublicEvidenceBundle):
        pass

    external_authority = ExternalAuthority(
        **{field.name: getattr(authority, field.name) for field in fields(authority)}
    )
    external_theory = ExternalTheory(
        **{field.name: getattr(theory, field.name) for field in fields(theory)}
    )
    external_registry = ExternalRegistry(
        **{field.name: getattr(registry, field.name) for field in fields(registry)}
    )
    external_base = ExternalBundle(
        **{
            field.name: getattr(authority.base_bundle, field.name)
            for field in fields(authority.base_bundle)
        }
    )
    nested_external = replace(authority, base_bundle=external_base)
    for changed_authority, changed_theory, changed_registry in (
        (external_authority, theory, registry),
        (authority, external_theory, registry),
        (authority, theory, external_registry),
        (nested_external, theory, registry),
    ):
        with pytest.raises(TypeError):
            _run(changed_authority, changed_theory, changed_registry)


def test_selector_identifies_one_binding_with_two_exact_admissible_scales():
    theory, registry, _, authority = _identity_authority(
        pass_kinds_by_slice=(frozenset({LawKind.SYMMETRY}),),
        all_entity_variants=True,
    )
    run = _run(authority, theory, registry)
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.decision.disposition is (
        bridge.ExactSelectionDisposition.ADMISSIBLE_SCALE_SET
    )
    assert run.decision.selected_law_kind is LawKind.SYMMETRY
    assert run.decision.admissible_scale_ids == tuple(
        sorted((ROOT_SCALE, TARGET_SCALE))
    )
    assert run.decision.normalized_structural_margin is not None
    assert run.decision.normalized_structural_margin.as_fraction() >= Fraction(1)
    passing = tuple(
        item
        for item in run.compilation.scale_aggregates
        if item.status is bridge.ExactCandidateStatus.PASS
    )
    assert len(passing) == 2
    assert {item.law_kind for item in passing} == {LawKind.SYMMETRY}
    assert {item.role_binding for item in passing} == {
        run.decision.selected_role_binding
    }
    symmetry_bindings = {
        item.role_binding
        for item in run.compilation.scale_aggregates
        if item.law_kind is LawKind.SYMMETRY
    }
    assert len(symmetry_bindings) == 4
    assert run.decision.bridge_result_id == run.compilation.result_id
    assert run.decision.candidate_grid_commitment_id == (
        run.compilation.candidate_grid_commitment_id
    )
    assert run.decision.scale_aggregate_commitment_id == (
        run.compilation.scale_aggregate_commitment_id
    )


def test_selector_abstains_without_a_binding_competitor():
    theory, registry, _, authority = _identity_authority(
        pass_kinds_by_slice=(frozenset({LawKind.SYMMETRY}),),
    )
    run = _run(authority, theory, registry)
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN
    assert run.decision.reason == "missing_binding_competitor"


def test_selector_abstains_for_multiple_passing_bindings():
    theory, registry, _, authority = _identity_authority(
        pass_kinds_by_slice=(frozenset({LawKind.SYMMETRY}),),
        all_entity_variants=True,
        pass_all_bindings=True,
    )
    run = _run(authority, theory, registry)
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN
    assert run.decision.reason == "multiple_passing_structures"
    assert sum(
        item.status is bridge.ExactCandidateStatus.PASS
        for item in run.compilation.scale_aggregates
    ) > 2


def test_selector_abstains_when_selected_structure_has_inconclusive_scale(
    monkeypatch,
):
    run = _run_with_consistent_selector_fault(
        monkeypatch,
        "selected_inconclusive",
    )
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN
    assert run.decision.reason == "selected_structure_has_inconclusive_scale"


def test_selector_abstains_for_inconclusive_structural_competitor(monkeypatch):
    run = _run_with_consistent_selector_fault(
        monkeypatch,
        "competitor_inconclusive",
    )
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN
    assert run.decision.reason == "inconclusive_structural_competitor"


def test_selector_abstains_when_exact_structural_margin_is_below_one(monkeypatch):
    run = _run_with_consistent_selector_fault(monkeypatch, "insufficient_margin")
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN
    assert run.decision.reason == "insufficient_structural_margin"


def test_any_candidate_error_forces_selector_abstention():
    theory, registry, _, authority = _sampling_authority()
    run = _run(authority, theory, registry)
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert any(
        item.status is bridge.ExactCandidateStatus.ERROR
        for item in run.compilation.scale_aggregates
    )
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN
    assert run.decision.reason == "candidate_evaluation_error"


def test_large_content_roots_are_cached_outside_candidate_inner_loops(monkeypatch):
    theory, registry, _, authority = _identity_authority(
        pass_kinds_by_slice=(frozenset({LawKind.SYMMETRY}),),
        all_entity_variants=True,
    )
    counts: Counter[str] = Counter()
    original_bridge_hash = bridge.stable_hash
    original_transform_hash = tx.stable_hash

    def counted_bridge_hash(value, *, prefix=""):
        counts[prefix] += 1
        return original_bridge_hash(value, prefix=prefix)

    def counted_transform_hash(value, *, prefix=""):
        counts[prefix] += 1
        return original_transform_hash(value, prefix=prefix)

    monkeypatch.setattr(bridge, "stable_hash", counted_bridge_hash)
    monkeypatch.setattr(tx, "stable_hash", counted_transform_hash)
    run = _run(authority, theory, registry)
    assert run.disposition is bridge.ExactBridgeDisposition.COMPLETE
    assert run.inventory is not None
    grid = run.compilation.candidate_grid
    assert grid is not None
    observation_count = len(run.inventory.observations)
    candidate_count = len(grid.candidates)
    slot_count = sum(len(item.slots) for item in grid.candidates)

    assert counts["phase2b_exact_transform_result_"] <= 2
    assert counts["phase2b_exact_derived_inventory_"] == 1
    assert counts["phase2b_derived_inventory_observation_"] <= observation_count
    assert counts["phase2b_exact_derived_grid_"] == 1
    assert counts["phase2b_exact_derived_candidate_"] <= 5 * candidate_count
    assert counts["phase2b_exact_derived_footprint_"] <= candidate_count
    assert counts["phase2b_exact_derived_slot_"] <= slot_count


@pytest.mark.parametrize(
    ("field_name", "reason"),
    (
        ("observations", "RESOURCE_LIMIT:inventory_observation_count"),
        ("components", "RESOURCE_LIMIT:inventory_component_count"),
    ),
)
def test_transform_output_length_cap_precedes_result_hash(
    monkeypatch,
    field_name,
    reason,
):
    theory, registry, _, authority = _identity_authority()
    transform_result = tx.run_exact_transform_semantics(authority)
    assert type(transform_result) is tx.ExactTransformCompilation
    item = getattr(transform_result, field_name)[0]
    oversized = replace(
        transform_result,
        **{field_name: (item,) * 262_145},
    )

    monkeypatch.setattr(
        bridge,
        "run_exact_transform_semantics",
        lambda _authority: oversized,
    )

    def forbidden_result_id(_self):
        raise AssertionError("oversized transform output was hashed before length cap")

    monkeypatch.setattr(
        tx.ExactTransformCompilation,
        "result_id",
        property(forbidden_result_id),
    )
    rejected = _run(authority, theory, registry)
    assert type(rejected) is bridge.ExactDerivedBridgePreflightRejection
    assert rejected.reason == reason
    assert rejected.transform_result is None


def test_registry_candidate_budget_is_frozen_not_caller_selectable(monkeypatch):
    theory, registry, _, authority = _identity_authority()
    changed_registry = replace(registry, maximum_candidate_count=49_999)

    def forbidden_transform(_authority):
        raise AssertionError("transform ran before registry budget drift rejection")

    monkeypatch.setattr(bridge, "run_exact_transform_semantics", forbidden_transform)
    rejected = _run(authority, theory, changed_registry)
    assert type(rejected) is bridge.ExactDerivedBridgePreflightRejection
    assert rejected.reason == "registry_candidate_budget_drift"
    assert rejected.transform_result is None


@pytest.mark.parametrize(
    "mutation",
    ("wrapper_root", "path", "uncertainty_lineage"),
)
def test_forged_transform_lineage_path_and_roots_fail_closed(
    monkeypatch,
    mutation,
):
    theory, registry, _, authority = _identity_authority()
    result = tx.run_exact_transform_semantics(authority)
    assert type(result) is tx.ExactTransformCompilation
    assert result.disposition is tx.TransformCompilationDisposition.COMPLETE
    counts = Counter(item.observation_descriptor_id for item in result.components)
    target_index = next(
        index
        for index, item in enumerate(result.components)
        if item.descriptor.ref.scale_id == TARGET_SCALE
        and counts[item.observation_descriptor_id] == 1
    )
    if mutation == "wrapper_root":
        forged_root = "forged_wrapper_root"
        forged = replace(
            result,
            wrapper_content_id=forged_root,
            components=tuple(
                replace(item, wrapper_content_id=forged_root)
                for item in result.components
            ),
        )
    else:
        target = result.components[target_index]
        changed_target = (
            replace(target, ordered_transform_path_ids=(uid(99_000),))
            if mutation == "path"
            else replace(
                target,
                uncertainty_compilation_ids=("forged_compilation_lineage",),
            )
        )
        forged = replace(
            result,
            components=tuple(
                changed_target if index == target_index else item
                for index, item in enumerate(result.components)
            ),
        )
    monkeypatch.setattr(
        bridge,
        "run_exact_transform_semantics",
        lambda _authority: forged,
    )
    run = _run(authority, theory, registry)
    assert run.transform_result is forged
    assert run.transform_result.disposition is tx.TransformCompilationDisposition.COMPLETE
    if mutation == "wrapper_root":
        assert run.reason in {
            "transform_authority_root_drift",
            "transform_wrapper_content_root_drift",
        }
    elif mutation == "path":
        assert "path" in run.reason
    else:
        assert "lineage" in run.reason
    _assert_atomic_bridge_abstention(run)


def test_selector_recomputes_aggregate_hull_from_slice_evaluations(monkeypatch):
    theory, registry, _, authority = _identity_authority(
        pass_kinds_by_slice=(frozenset({LawKind.SYMMETRY}),),
        all_entity_variants=True,
    )
    original_compile = bridge._compile_grid

    def forged_compile(*args, **kwargs):
        evaluations, aggregates, error = original_compile(*args, **kwargs)
        assert evaluations is not None and aggregates is not None and error is None
        target_index = next(
            index
            for index, item in enumerate(aggregates)
            if item.status is bridge.ExactCandidateStatus.FAIL
        )
        target = aggregates[target_index]
        forged_target = replace(
            target,
            normalized_hull=bridge.ExactInterval.from_fractions(
                Fraction(0),
                Fraction(0),
            ),
        )
        return (
            evaluations,
            tuple(
                forged_target if index == target_index else item
                for index, item in enumerate(aggregates)
            ),
            None,
        )

    monkeypatch.setattr(bridge, "_compile_grid", forged_compile)
    run = _run(authority, theory, registry)
    assert run.decision.disposition is bridge.ExactSelectionDisposition.ABSTAIN
    assert "aggregate" in run.decision.reason
    assert "drift" in run.decision.reason


def test_uncertainty_lineage_is_bound_per_source_observation_not_only_by_union(
    monkeypatch,
):
    theory, registry, _, authority = _identity_authority()
    result = tx.run_exact_transform_semantics(authority)
    assert type(result) is tx.ExactTransformCompilation
    counts = Counter(item.observation_descriptor_id for item in result.components)
    scalar_root_indices = tuple(
        index
        for index, item in enumerate(result.components)
        if item.descriptor.ref.scale_id == ROOT_SCALE
        and counts[item.observation_descriptor_id] == 1
    )
    left_index, right_index = scalar_root_indices[:2]
    left = result.components[left_index]
    right = result.components[right_index]
    assert left.uncertainty_compilation_ids != right.uncertainty_compilation_ids
    forged_components = list(result.components)
    forged_components[left_index] = replace(
        left,
        uncertainty_compilation_ids=right.uncertainty_compilation_ids,
    )
    forged_components[right_index] = replace(
        right,
        uncertainty_compilation_ids=left.uncertainty_compilation_ids,
    )
    forged = replace(result, components=tuple(forged_components))
    assert {
        item
        for component in forged.components
        for item in component.uncertainty_compilation_ids
    } == {
        item
        for component in result.components
        for item in component.uncertainty_compilation_ids
    }
    monkeypatch.setattr(
        bridge,
        "run_exact_transform_semantics",
        lambda _authority: forged,
    )
    run = _run(authority, theory, registry)
    assert "uncertainty" in run.reason
    assert "lineage" in run.reason
    _assert_atomic_bridge_abstention(run)


def test_only_the_authoritative_entrypoint_is_exported():
    assert bridge.__all__ == ("run_exact_derived_witness_bridge",)


def test_aggregate_replay_index_has_two_bounded_scans_and_canonical_sort():
    theory, registry, _, authority = _identity_authority(
        pass_kinds_by_slice=(frozenset({LawKind.SYMMETRY}),),
        all_entity_variants=True,
    )
    run = _run(authority, theory, registry)
    evaluations = run.compilation.evaluations
    indexed, payloads, work = bridge._linear_aggregate_replay_index(evaluations)

    assert len(indexed) == len(evaluations)
    assert len(payloads) == len(run.compilation.scale_aggregates)
    assert work == 2 * len(evaluations)
    assert work <= bridge._DEFAULT_POLICY.maximum_aggregate_replay_work
