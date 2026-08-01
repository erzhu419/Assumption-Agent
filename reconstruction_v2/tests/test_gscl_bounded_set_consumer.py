"""Public source-free contract tests for the bounded set consumer."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from assumption_agent.generalized_structural_correspondence_v1 import (
    ObservationStatus,
    build_gscl_schema_registry_v1,
)
from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    build_universal_assumption_ontology_v1,
)
from replication_runtime.gscl_narrative_extractor_v1.closed_choice_worker import (
    PromptAnswer,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    bounded_set_consumer as consumer,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice as leaf_v2,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    document_envelope,
)
from replication_runtime.gscl_narrative_extractor_v2.contract import (
    ClosedChoiceV2Abstention,
    ClosedChoiceV2Error,
)


_RUNTIME_COMMITMENT = hashlib.sha256(
    b"gscl-bounded-set-consumer-public-fake-leaf"
).hexdigest()


class _Backend:
    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    def score_batch(
        self, pairs: tuple[PromptAnswer, ...]
    ) -> tuple[leaf_v2.TeacherForcedScore, ...]:
        rows: list[leaf_v2.TeacherForcedScore] = []
        for pair in pairs:
            preferred = int(
                pair.candidate_key.endswith(".plan.one_relation")
            )
            answer_tokens = max(1, len(pair.answer.split()))
            rows.append(
                leaf_v2.TeacherForcedScore(
                    total_logprob_microunits=(
                        preferred * 1_000_000 * answer_tokens
                    ),
                    answer_token_count=answer_tokens,
                    context_and_answer_token_count=answer_tokens + 80,
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int:
        return max(1, len(completion.encode("utf-8")) // 4)


def _parser(story: str, completion: str) -> NarrativeExtraction:
    return parse_untrusted_generator_completion(
        NarrativeSource(
            "bounded."
            + hashlib.sha256(story.encode("utf-8")).hexdigest()[:24],
            story,
        ),
        completion,
    )


class _LeafSelector:
    def __init__(
        self,
        *,
        no_relation_calls: frozenset[int] = frozenset(),
        typed_failure_calls: frozenset[int] = frozenset(),
    ) -> None:
        self.no_relation_calls = no_relation_calls
        self.typed_failure_calls = typed_failure_calls
        self.calls: list[str] = []

    def select_story(self, story_text: str) -> leaf_v2.ClosedChoiceV2Decision:
        call_index = len(self.calls)
        self.calls.append(story_text)
        if call_index in self.no_relation_calls:
            raise ClosedChoiceV2Abstention(
                "V2_PLAN_NO_RELATION_SELECTED",
                before_model_forward=False,
            )
        if call_index in self.typed_failure_calls:
            raise ClosedChoiceV2Error("V2_MODEL_FORWARD_FAILED")
        return leaf_v2.select_hierarchical_qualification_only(
            story_text,
            backend=_Backend(),
            narrative_parser=_parser,
        )


def _tokens(count: int, prefix: str = "Token") -> str:
    return " ".join(f"{prefix}{index:04d}" for index in range(count))


def _sentence(count: int, index: int, prefix: str = "Sentence") -> str:
    return _tokens(count, prefix=f"{prefix}{index:03d}") + "."


def _envelope(
    story: str, selector: _LeafSelector | None = None
) -> document_envelope.NarrativeDocumentEnvelopeV1:
    return document_envelope.select_document_qualification_only(
        story,
        leaf_selector=_LeafSelector() if selector is None else selector,
    )


def _all_strings(value: object) -> tuple[str, ...]:
    rows: list[str] = []
    if isinstance(value, str):
        rows.append(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_all_strings(key))
            rows.extend(_all_strings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            rows.extend(_all_strings(child))
    return tuple(rows)


def test_complete_document_maps_one_to_one_without_old_abi() -> None:
    upstream = _envelope(_tokens(176, "Complete"))
    result = consumer.consume_document_envelope(upstream)

    assert type(result) is consumer.BoundedNarrativeRelationSetV1
    assert not isinstance(result, NarrativeExtraction)
    assert result.disposition is (
        consumer.SetConsumerDisposition.COMPLETE_SELECTED_SET
    )
    assert len(result.coverage) == 2
    assert len(result.units) == len(upstream.relations) == 2
    episode = result.structural_episode
    assert episode is not None
    assert not episode.validate()
    assert not episode.verify_source_bytes(
        upstream.source_text.encode("utf-8")
    )
    assert len(episode.objects) == 2 * len(result.units)
    assert len(episode.relations) == len(result.units)
    assert len(episode.evidence_spans) == 3 * len(result.units)
    assert all(
        row.semantic_payload()["endpoint_roles"] == ["slot0", "slot1"]
        and row.relation_type.startswith("NarrativeOrderedSlots")
        and not hasattr(row, "source_object_id")
        and not hasattr(row, "target_object_id")
        for row in result.units
    )
    assert episode.quantities == ()
    assert episode.hyperrelations == ()
    assert episode.constraints == ()
    assert episode.observables == ()
    assert episode.missing_observables == ()
    assert all(
        row.observation_status is ObservationStatus.INFERRED
        and row.inference_provenance is not None
        and row.inference_provenance.calibration_bucket
        == "qualification.unscored"
        for row in (*episode.objects, *episode.relations)
    )
    receipt = result.receipt
    assert receipt["law_binding_count"] == 0
    assert receipt["numeric_observable_count"] == 0
    assert receipt["quantity_count"] == 0
    assert receipt["law_evaluation_eligible"] is False
    assert receipt["correspondence_acceptance_established"] is False
    assert receipt["directed_endpoint_semantics_established"] is False
    assert (
        receipt["structural_source_target_fields_are_positional_slots"]
        is True
    )
    assert receipt["relation_recall_total"] is False
    assert receipt["standalone_unit_authority_established"] is False
    assert receipt["upstream_downstream_eligible"] is False


def test_signature_is_rename_invariant_but_evidence_binding_is_not() -> None:
    first = consumer.consume_document_envelope(
        _envelope(_tokens(176, "Alpha"))
    )
    second = consumer.consume_document_envelope(
        _envelope(_tokens(176, "Beta"))
    )
    assert first.relation_set_signature_bytes == (
        second.relation_set_signature_bytes
    )
    assert first.relation_set_signature_sha256 == (
        second.relation_set_signature_sha256
    )
    assert first.evidence_binding_sha256 != second.evidence_binding_sha256
    signature = consumer.canonical_relation_set_signature(first)
    assert signature["unit_count"] == 2
    assert signature["unit_signature_multiset"] == [
        {
            "count": 2,
            "unit_signature_sha256": (
                first.units[0].semantic_signature_sha256
            ),
        }
    ]
    larger = consumer.consume_document_envelope(
        _envelope(_tokens(351, "Alpha"))
    )
    assert larger.relation_set_signature_sha256 != (
        first.relation_set_signature_sha256
    )


def test_repeated_quote_across_segments_never_merges_objects() -> None:
    first = ["Echo", "relates", "Alpha"] + [
        f"First{index:02d}" for index in range(14)
    ]
    second = ["Echo", "supports", "Beta"] + [
        f"Second{index:02d}" for index in range(14)
    ]
    story = " ".join(first) + ". " + " ".join(second) + "."
    result = consumer.consume_document_envelope(_envelope(story))
    episode = result.structural_episode
    assert episode is not None
    assert len(result.units) == 2
    assert len(episode.objects) == 4
    assert len({row.object_id for row in episode.objects}) == 4
    assert len(episode.evidence_spans) == 6
    assert len({row.span_id for row in episode.evidence_spans}) == 6


def test_context_and_no_relation_are_coverage_not_synthetic_units() -> None:
    story = (
        _sentence(17, 0)
        + " "
        + _sentence(17, 1)
        + " 尾部。"
    )
    selector = _LeafSelector(no_relation_calls=frozenset({1}))
    result = consumer.consume_document_envelope(
        _envelope(story, selector)
    )
    assert result.disposition is (
        consumer.SetConsumerDisposition.PARTIAL_SELECTED_SET
    )
    assert [row.disposition for row in result.coverage] == [
        "EXTRACTED",
        "NO_RELATION",
        "CONTEXT_ONLY_SHORT_SENTENCE",
    ]
    assert len(result.units) == 1
    assert result.structural_episode is not None
    assert len(result.structural_episode.objects) == 2
    counts = result.receipt["coverage_disposition_counts"]
    assert counts == {
        "CONTEXT_ONLY_SHORT_SENTENCE": 1,
        "EXTRACTED": 1,
        "NO_RELATION": 1,
        "TYPED_FAILURE": 0,
    }


@pytest.mark.parametrize(
    "story",
    ["。", _tokens(17, "NoRelation")],
)
def test_zero_extracted_units_abstain_without_dummy_episode(
    story: str,
) -> None:
    selector = (
        _LeafSelector(no_relation_calls=frozenset({0}))
        if "NoRelation" in story
        else _LeafSelector()
    )
    result = consumer.consume_document_envelope(
        _envelope(story, selector)
    )
    assert result.disposition is (
        consumer.SetConsumerDisposition.EMPTY_ABSTENTION
    )
    assert result.units == ()
    assert result.structural_episode is None
    assert result.relation_set_signature_bytes is None
    assert result.relation_set_signature_sha256 is None
    with pytest.raises(
        consumer.BoundedSetConsumerError,
        match="SET_CONSUMER_SIGNATURE_UNAVAILABLE",
    ):
        consumer.canonical_relation_set_signature(result)


def test_any_typed_failure_blocks_prior_partial_projection() -> None:
    selector = _LeafSelector(typed_failure_calls=frozenset({1}))
    upstream = _envelope(_tokens(351, "Failure"), selector)
    assert len(upstream.relations) == 1
    result = consumer.consume_document_envelope(upstream)
    assert result.disposition is (
        consumer.SetConsumerDisposition.TYPED_FAILURE_BLOCKED
    )
    assert result.units == ()
    assert result.structural_episode is None
    assert result.relation_set_signature_bytes is None
    assert result.receipt["selected_set_available"] is False
    assert result.receipt["coverage_disposition_counts"][
        "TYPED_FAILURE"
    ] == 2


def test_all_five_laws_remain_inconclusive_without_invented_values() -> None:
    registry = build_gscl_schema_registry_v1(
        build_universal_assumption_ontology_v1()
    )
    result = consumer.consume_document_envelope(
        _envelope(_tokens(176, "LawNeutral"))
    )
    rows = consumer.assess_law_readiness(result, registry)
    assert len(rows) == 5
    assert {row.law_id for row in rows} == {
        schema.law_id for schema in registry.schemas
    }
    by_id = {row.law_id: row for row in rows}
    for schema in registry.schemas:
        row = by_id[schema.law_id]
        assert row.disposition is (
            consumer.LawReadinessDisposition.INCONCLUSIVE_MISSING_EVIDENCE
        )
        assert set(row.missing_role_ids) == {
            role.role_id for role in schema.roles
        }
        assert set(row.missing_observable_ids) == {
            observable.observable_id
            for observable in schema.required_observables
        }
        assert "numeric_or_typed_values_were_not_invented" in row.reasons
        assert "no_law_binding_or_residual_was_constructed" in row.reasons

    partial = consumer.consume_document_envelope(
        _envelope(_sentence(17, 0) + " 尾部。")
    )
    assert all(
        row.disposition
        is consumer.LawReadinessDisposition.INCONCLUSIVE_PARTIAL_COVERAGE
        for row in consumer.assess_law_readiness(partial, registry)
    )

    class _ForgedRegistry(type(registry)):
        def validate_frozen_contract(self) -> tuple[str, ...]:
            return ()

    forged = _ForgedRegistry(
        ontology_hash=registry.ontology_hash,
        schemas=registry.schemas,
    )
    with pytest.raises(
        consumer.BoundedSetConsumerError,
        match="SET_CONSUMER_REGISTRY_INVALID",
    ):
        consumer.assess_law_readiness(result, forged)


def test_maximum_root_is_bounded_repeat_exact_and_content_free() -> None:
    story = _tokens(1_024, "MaximumSecret")
    upstream = _envelope(story)
    first = consumer.consume_document_envelope(upstream)
    second = consumer.consume_document_envelope(upstream)
    assert len(first.units) == 6
    assert first.structural_episode is not None
    assert len(first.structural_episode.objects) == 12
    assert len(first.structural_episode.evidence_spans) == 18
    assert len(first.units) <= consumer.MAXIMUM_RELATION_UNITS
    assert (
        len(first.structural_episode.objects)
        <= consumer.MAXIMUM_STRUCTURAL_OBJECTS
    )
    assert (
        len(first.structural_episode.evidence_spans)
        <= consumer.MAXIMUM_EVIDENCE_SPANS
    )
    assert first.receipt_bytes == second.receipt_bytes
    assert first.relation_set_signature_bytes == (
        second.relation_set_signature_bytes
    )
    nested = _all_strings(json.loads(first.receipt_bytes.decode("ascii")))
    assert story not in nested
    assert all("MaximumSecret" not in value for value in nested)


def test_old_abi_and_self_consistent_result_tamper_fail_closed() -> None:
    upstream = _envelope(_tokens(176, "Attack"))
    extraction = upstream.segments[0].leaf_decision.extraction
    with pytest.raises(
        consumer.BoundedSetConsumerError,
        match="SET_CONSUMER_UPSTREAM_INVALID",
    ):
        consumer.consume_document_envelope(extraction)  # type: ignore[arg-type]

    result = consumer.consume_document_envelope(upstream)
    with pytest.raises(
        consumer.BoundedSetConsumerError,
        match="SET_CONSUMER_OWNERSHIP_INVALID",
    ):
        replace(result, units=result.units[:-1])
    with pytest.raises(
        consumer.BoundedSetConsumerError,
        match="SET_CONSUMER_OWNERSHIP_INVALID",
    ):
        replace(
            result,
            disposition=consumer.SetConsumerDisposition.PARTIAL_SELECTED_SET,
        )
    with pytest.raises(
        consumer.BoundedSetConsumerError,
        match="SET_CONSUMER_AUTHORITY_INVALID",
    ):
        replace(result, _marker=object())


def test_api_has_no_content_law_binding_or_numeric_injection_surface() -> None:
    assert tuple(
        inspect.signature(consumer.consume_document_envelope).parameters
    ) == ("envelope",)
    assert tuple(
        inspect.signature(
            consumer.canonical_relation_set_signature
        ).parameters
    ) == ("result",)
    assert tuple(
        inspect.signature(consumer.assess_law_readiness).parameters
    ) == ("result", "registry")
    source = Path(consumer.__file__).read_text(encoding="utf-8")
    assert "from assumption_agent.gscl_narrative_correspondence_v1" not in source
    assert "NarrativeExtraction(" not in source
    assert "LawBinding(" not in source
    assert "TypedObservable(" not in source
    assert "StructuralQuantity(" not in source
    assert "ExactRational(" not in source
    assert "requests." not in source
    assert "urlopen(" not in source
    assert "http://" not in source
    assert "https://" not in source
