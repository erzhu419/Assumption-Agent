from __future__ import annotations

from dataclasses import replace
import inspect
import json
from typing import Any

import assumption_agent.gscl_evidence_extractor_v1 as extractor_module
from assumption_agent.benchmarks.gscl_controlled_evidence_corpus_v1 import (
    ROOT_COUNT,
    VIEW_COUNT,
    T05,
    T09,
    build_controlled_roots,
    controlled_view_gold_linkage,
    render_controlled_views,
    validate_no_runtime_answer_leak,
)
from assumption_agent.generalized_structural_correspondence_v1 import (
    ExactRational,
    ObservationStatus,
    build_gscl_schema_registry_v1,
    strict_content_hash,
)
from assumption_agent.gscl_evidence_extractor_v1 import (
    JSONL_MEDIA_TYPE,
    MAX_BINDING_ASSIGNMENTS,
    bind_structural_episode,
    extract_structural_episode,
)
from assumption_agent.structural_law_residuals_v1 import (
    ResidualPolicy,
    evaluate_bound_law,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    build_universal_assumption_ontology_v1,
)


def _registry():
    return build_gscl_schema_registry_v1(
        build_universal_assumption_ontology_v1()
    )


def _json_lines(records: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in records
        )
        + "\n"
    ).encode()


def _view_for(
    roots,
    root_id: str,
    view_kind: str,
):
    views = render_controlled_views(roots)
    links = {
        (row.root_id, row.view_kind): row.item_id
        for row in controlled_view_gold_linkage(roots)
    }
    item_id = links[(root_id, view_kind)]
    return next(row for row in views if row.item_id == item_id)


def _renamed_records(
    records: tuple[dict[str, Any], ...],
    prefix: str,
) -> list[dict[str, Any]]:
    ids = {str(row["id"]) for row in records}

    def rename(value: Any) -> Any:
        if isinstance(value, str) and value in ids:
            return f"{prefix}.{value}"
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    return [
        {
            "kind": row["kind"],
            "id": f"{prefix}.{row['id']}",
            "attrs": rename(row["attrs"]),
        }
        for row in records
    ]


def test_atomic_corpus_is_blind_paired_and_has_four_views() -> None:
    roots = build_controlled_roots()
    views = render_controlled_views(roots)

    assert len(roots) == ROOT_COUNT == 25
    assert len(views) == VIEW_COUNT == 100
    assert validate_no_runtime_answer_leak(views) == ()
    links = controlled_view_gold_linkage(roots)
    assert {row.view_kind for row in links} == {
        "json_canonical",
        "json_alias",
        "line_canonical",
        "line_alias",
    }
    assert all(
        sum(row.root_id == root.root_id for row in links) == 4
        for root in roots
    )


def test_all_atomic_views_propose_bind_and_reach_gold_outcome() -> None:
    registry = _registry()
    roots = build_controlled_roots()
    roots_by_id = {root.root_id: root for root in roots}
    links_by_item = {
        row.item_id: row
        for row in controlled_view_gold_linkage(roots)
    }

    for view in render_controlled_views(roots):
        root = roots_by_id[links_by_item[view.item_id].root_id]
        extraction = extract_structural_episode(
            view.source_bytes, view.media_type, registry=registry
        )
        assert extraction.issue_ids == ()
        assert extraction.candidate_law_ids == (root.law_id,)
        assert len(extraction.proposals) == 1
        assert extraction.proposals[0].validate() == ()
        assert extraction.base_episode is not None
        assert extraction.base_episode.constraints == ()
        assert extraction.base_episode.observables == ()
        assert extraction.base_episode.verify_source_bytes(
            view.source_bytes
        ) == ()

        search = bind_structural_episode(registry, extraction)
        assert search.issue_ids == ()
        assert search.truncated is False
        assert search.assignment_count <= MAX_BINDING_ASSIGNMENTS
        assert len(search.bound_cases) == 1
        case = search.bound_cases[0]
        assert {
            row.role_id: row.target_id
            for row in case.binding.role_bindings
        } == dict(root.role_targets)

        policy = ResidualPolicy(
            law_id=root.law_id,
            relation_threshold=(
                ExactRational(1)
                if root.law_id == T05
                else ExactRational(0)
            ),
        )
        disposition = evaluate_bound_law(
            registry,
            registry.require_law(root.law_id),
            case.episode,
            case.binding,
            policy,
        ).disposition.value
        assert disposition == root.expected_disposition


def test_proposals_are_multifact_field_grounded_and_law_neutral() -> None:
    registry = _registry()
    roots = build_controlled_roots()
    view = _view_for(
        roots, "root.t14.primary_a", "json_alias"
    )
    extraction = extract_structural_episode(
        view.source_bytes, view.media_type, registry=registry
    )
    assert extraction.base_episode is not None
    proposal = extraction.proposals[0]
    assert len(proposal.fact_ids) >= 6
    assert len(proposal.condition_ids) >= 3
    assert proposal.field_span_ids
    expected_derivation = strict_content_hash(
        proposal.derivation_payload()
    )
    assert proposal.derivation_hash == expected_derivation
    span_ids = {
        span.span_id
        for span in extraction.base_episode.evidence_spans
    }
    assert set(proposal.field_span_ids) <= span_ids

    raw_neutral_payload = json.dumps(
        extraction.base_episode.private_payload(),
        sort_keys=True,
    )
    for canonical_role in (
        "input_before",
        "input_after",
        "transformation",
        "output_before",
        "output_after",
        "equivariance_constraint",
    ):
        assert canonical_role not in raw_neutral_payload


def test_bound_constraint_provenance_covers_every_observable_support() -> None:
    registry = _registry()
    roots = build_controlled_roots()
    view = _view_for(
        roots, "root.t15.primary_a", "line_alias"
    )
    extraction = extract_structural_episode(
        view.source_bytes, view.media_type, registry=registry
    )
    case = bind_structural_episode(registry, extraction).bound_cases[0]
    constraint = case.episode.constraints[0]
    observable_support = {
        span_id
        for observable in case.episode.observables
        for span_id in observable.evidence_span_ids
    }
    assert observable_support <= set(constraint.evidence_span_ids)
    proposal_condition_support = {
        span_id
        for condition in case.proposal.condition_evidence
        for span_id in condition.field_span_ids
    }
    assert proposal_condition_support <= set(
        constraint.evidence_span_ids
    )
    association = next(
        row
        for row in extraction.records
        if row.kind == "assertion"
        and row.attrs.get("predicate") == "association"
    )
    assert set(association.field_span_ids()) <= set(
        constraint.evidence_span_ids
    )
    assert constraint.inference_provenance is not None
    assert constraint.inference_provenance.validate() == ()

    for span in case.episode.evidence_spans:
        assert span.verify_against(view.source_bytes) == ()
    assert any(
        span.span_id.startswith("span.field.")
        and (span.end_byte - span.start_byte)
        < len(view.source_bytes)
        for span in case.episode.evidence_spans
    )


def test_missingness_is_caused_by_absent_facts_only() -> None:
    registry = _registry()
    roots = build_controlled_roots()
    controls = [
        root for root in roots if root.root_kind == "missingness_control"
    ]
    assert len(controls) == 5
    for root in controls:
        view = _view_for(
            (root,), root.root_id, "json_canonical"
        )
        extraction = extract_structural_episode(
            view.source_bytes, view.media_type, registry=registry
        )
        search = bind_structural_episode(registry, extraction)
        case = search.bound_cases[0]
        assert case.episode.missing_observables
        assert any(
            row.observation_status is ObservationStatus.UNKNOWN
            for row in case.episode.observables
        )
        assert all(
            row["attrs"].get("predicate") != "unavailable"
            for row in root.records
            if row["kind"] == "assertion"
        )
        assert all(
            not (
                row.observation_status
                is ObservationStatus.UNKNOWN
                and row.value is not None
            )
            for row in case.episode.quantities
        )
        assert all(
            not (
                row.kind == "observation"
                and "value" in row.attrs
                and any(
                    quantity.quantity_id == row.record_id
                    and quantity.observation_status
                    is ObservationStatus.UNKNOWN
                    for quantity in case.episode.quantities
                )
            )
            for row in extraction.records
        )


def test_multiple_structural_proposals_abstain_in_binder() -> None:
    registry = _registry()
    roots = build_controlled_roots()
    t14 = next(
        root
        for root in roots
        if root.root_id == "root.t14.primary_a"
    )
    t09 = next(
        root
        for root in roots
        if root.root_id == "root.t09.primary_a"
    )
    source = _json_lines(
        [
            *_renamed_records(t14.records, "left"),
            *_renamed_records(t09.records, "right"),
        ]
    )
    extraction = extract_structural_episode(
        source, JSONL_MEDIA_TYPE, registry=registry
    )
    assert extraction.issue_ids == ()
    assert set(extraction.candidate_law_ids) == {t14.law_id, T09}
    assert len(extraction.proposals) == 2
    search = bind_structural_episode(registry, extraction)
    assert search.bound_cases == ()
    assert search.issue_ids == ("binder_ambiguous_binding",)


def test_alias_normalization_collision_is_rejected() -> None:
    registry = _registry()
    source = (
        b'{"kind":"node","id":"node.alpha","attrs":'
        b'{"sort":"state","category":"configuration"}}\n'
    )
    result = extract_structural_episode(
        source, JSONL_MEDIA_TYPE, registry=registry
    )
    assert result.base_episode is None
    assert result.issue_ids == (
        "extractor_record_parse_failed.0000",
    )


def test_duplicate_json_keys_and_unavailable_claim_are_rejected() -> None:
    registry = _registry()
    duplicate = (
        b'{"kind":"node","kind":"edge","id":"node.alpha",'
        b'"attrs":{"sort":"state"}}\n'
    )
    result = extract_structural_episode(
        duplicate, JSONL_MEDIA_TYPE, registry=registry
    )
    assert result.base_episode is None
    assert result.issue_ids == (
        "extractor_record_parse_failed.0000",
    )

    unavailable = (
        b'{"kind":"assertion","id":"assertion.alpha","attrs":'
        b'{"predicate":"unavailable","target":"direct_map",'
        b'"value":true}}\n'
    )
    result = extract_structural_episode(
        unavailable, JSONL_MEDIA_TYPE, registry=registry
    )
    assert result.base_episode is None
    assert result.issue_ids[0].startswith(
        "extractor_fact_compile_failed"
    )


def test_recursion_error_fails_closed(monkeypatch) -> None:
    registry = _registry()

    def recurse(*_args, **_kwargs):
        raise RecursionError

    monkeypatch.setattr(extractor_module, "_parse_records", recurse)
    result = extract_structural_episode(
        b"not-used", JSONL_MEDIA_TYPE, registry=registry
    )
    assert result.base_episode is None
    assert result.proposals == ()
    assert result.issue_ids == ("extractor_recursion_limit_exceeded",)

    monkeypatch.undo()
    root = next(
        row
        for row in build_controlled_roots()
        if row.root_id == "root.t14.primary_a"
    )
    view = _view_for(
        (root,), root.root_id, "json_canonical"
    )
    extraction = extract_structural_episode(
        view.source_bytes, view.media_type, registry=registry
    )
    monkeypatch.setattr(
        extractor_module, "_assignment_consistent", recurse
    )
    search = bind_structural_episode(registry, extraction)
    assert search.bound_cases == ()
    assert search.issue_ids == ("binder_recursion_limit_exceeded",)


def test_runtime_surface_has_no_gold_or_item_metadata() -> None:
    signature = inspect.signature(extract_structural_episode)
    assert set(signature.parameters) == {
        "source_bytes",
        "media_type",
        "registry",
    }
    source = inspect.getsource(extract_structural_episode)
    assert "root_id" not in source
    assert "expected_disposition" not in source


def test_binder_recomputes_proposal_source_fact_span_and_condition_links() -> None:
    registry = _registry()
    roots = build_controlled_roots()
    view = _view_for(
        roots, "root.t14.primary_a", "json_canonical"
    )
    extraction = extract_structural_episode(
        view.source_bytes, view.media_type, registry=registry
    )
    proposal = extraction.proposals[0]
    condition = proposal.condition_evidence[-1]
    forged_condition = replace(
        condition, evidence_hash="0" * 64
    )
    forged = replace(
        proposal,
        condition_evidence=(
            *proposal.condition_evidence[:-1],
            forged_condition,
        ),
    )
    forged = replace(
        forged,
        derivation_hash=strict_content_hash(
            forged.derivation_payload()
        ),
    )
    assert forged.validate() == ()
    search = bind_structural_episode(
        registry,
        replace(extraction, proposals=(forged,)),
    )
    assert search.bound_cases == ()
    assert search.issue_ids == (
        "binder_proposal_linkage_invalid",
    )

    assert extraction.base_episode is not None
    observed = next(
        row
        for row in extraction.base_episode.quantities
        if row.quantity_id == "observation.beta"
    )
    forged_unknown = replace(
        observed,
        value=None,
        evidence_span_ids=(),
        observation_status=ObservationStatus.UNKNOWN,
        inference_provenance=None,
    )
    forged_episode = replace(
        extraction.base_episode,
        quantities=tuple(
            forged_unknown
            if row.quantity_id == forged_unknown.quantity_id
            else row
            for row in extraction.base_episode.quantities
        ),
    )
    search = bind_structural_episode(
        registry,
        replace(extraction, base_episode=forged_episode),
    )
    assert search.bound_cases == ()
    assert search.issue_ids == (
        "binder_extraction_linkage_invalid",
    )

    first = extraction.records[0]
    tampered_record = replace(
        first, attrs={**first.attrs, "phase": "tampered"}
    )
    tampered_records = (
        tampered_record,
        *extraction.records[1:],
    )
    search = bind_structural_episode(
        registry,
        replace(extraction, records=tampered_records),
    )
    assert search.bound_cases == ()
    assert search.issue_ids == (
        "binder_proposal_linkage_invalid",
    )


def test_generic_raw_types_are_not_retyped_before_unique_csp() -> None:
    registry = _registry()
    roots = build_controlled_roots()
    view = _view_for(
        roots, "root.t15.primary_a", "json_canonical"
    )
    extraction = extract_structural_episode(
        view.source_bytes, view.media_type, registry=registry
    )
    assert extraction.base_episode is not None
    assert {row.object_type for row in extraction.base_episode.objects} == {
        "Entity"
    }
    assert {
        row.relation_type for row in extraction.base_episode.relations
    } == {"DirectedRelation"}
    assert extraction.base_episode.constraints == ()
    assert "law_id" not in inspect.signature(
        extractor_module._role_domains
    ).parameters
