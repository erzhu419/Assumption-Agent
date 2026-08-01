"""Source-free contract tests for the within-side SCAR slot binder."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect

import numpy as np
import pytest

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from assumption_agent import gscl_slot_graph_binder_v1 as binder
from replication_runtime.gscl_narrative_extractor_v1.closed_choice_worker import (
    PromptAnswer,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    bounded_set_consumer,
    closed_choice,
    document_envelope,
)
from replication_runtime.gscl_narrative_extractor_v2.contract import (
    ClosedChoiceV2Error,
)


_RUNTIME = hashlib.sha256(b"binder-source-free-fake-runtime").hexdigest()
_ENCODER = hashlib.sha256(b"binder-source-free-fake-encoder").hexdigest()


class _Backend:
    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME

    def score_batch(
        self, pairs: tuple[PromptAnswer, ...]
    ) -> tuple[closed_choice.TeacherForcedScore, ...]:
        rows = []
        for pair in pairs:
            preferred = int(pair.candidate_key.endswith(".plan.one_relation"))
            tokens = max(1, len(pair.answer.split()))
            rows.append(
                closed_choice.TeacherForcedScore(
                    total_logprob_microunits=preferred * tokens * 1_000_000,
                    answer_token_count=tokens,
                    context_and_answer_token_count=tokens + 64,
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(self, completion: str) -> int:
        return max(1, len(completion.encode("utf-8")) // 4)


def _parser(story: str, completion: str) -> NarrativeExtraction:
    return parse_untrusted_generator_completion(
        NarrativeSource(
            "binder." + hashlib.sha256(story.encode()).hexdigest()[:24],
            story,
        ),
        completion,
    )


class _Selector:
    def select_story(self, story_text: str):
        return closed_choice.select_hierarchical_qualification_only(
            story_text,
            backend=_Backend(),
            narrative_parser=_parser,
        )


class _FailureSelector:
    def select_story(self, story_text: str):
        raise ClosedChoiceV2Error("V2_MODEL_FORWARD_FAILED")


def _relation_set():
    story = " ".join(f"UniqueToken{index:03d}" for index in range(44)) + "."
    envelope = document_envelope.select_document_qualification_only(
        story, leaf_selector=_Selector()
    )
    result = bounded_set_consumer.consume_document_envelope(envelope)
    assert len(result.units) == 1
    return result


def _endpoint_quotes(result) -> tuple[str, str]:
    raw = result.upstream_envelope.source_text.encode()
    spans = {
        row.span_id: row for row in result.structural_episode.evidence_spans
    }
    unit = result.units[0]
    return tuple(
        raw[spans[span_id].start_byte : spans[span_id].end_byte].decode()
        for span_id in (unit.slot0_span_id, unit.slot1_span_id)
    )


class _ExactTextEncoder:
    """Deterministic text identity vectors; no source or label capability."""

    def encode(self, texts):
        unique = {text: index for index, text in enumerate(sorted(set(texts)))}
        width = max(8, len(unique))
        matrix = np.zeros((len(texts), width), dtype=np.float32)
        for row, text in enumerate(texts):
            matrix[row, unique[text]] = 1.0
        return matrix


class _CollapsedEncoder:
    def encode(self, texts):
        return np.ones((len(texts), 8), dtype=np.float32)


def _labels(result):
    left, right = _endpoint_quotes(result)
    assert left != right
    return {"slot.a": left, "slot.b": right, "slot.zero": "UnusedSurface"}


def test_exact_mentions_bind_within_side_and_zero_degree_slot_survives() -> None:
    relation_set = _relation_set()
    result = binder.bind_relation_set_to_slots_v1(
        relation_set,
        slot_labels=_labels(relation_set),
        encoder=_ExactTextEncoder(),
        encoder_binding_sha256=_ENCODER,
    )
    assert len(result.graph.slots) == 3
    assert len(result.graph.relations) == 1
    assert {row.selected_slot_id for row in result.endpoint_bindings} == {
        "slot.a",
        "slot.b",
    }
    assert result.graph.coverage_complete is False
    assert result.receipt["zero_degree_slots_retained"] is True
    assert result.receipt["gold_mapping_access_count"] == 0
    assert result.receipt["cross_side_text_access_count"] == 0
    assert result.receipt["formal_law_binding_count"] == 0


def test_quantized_tie_abstains_and_drops_edges_without_threshold() -> None:
    relation_set = _relation_set()
    result = binder.bind_relation_set_to_slots_v1(
        relation_set,
        slot_labels=_labels(relation_set),
        encoder=_CollapsedEncoder(),
        encoder_binding_sha256=_ENCODER,
    )
    assert not result.graph.relations
    assert all(row.selected_slot_id is None for row in result.endpoint_bindings)
    assert result.receipt["unbound_endpoint_count"] == 2
    assert result.receipt["dropped_relation_count"] == 1
    assert result.receipt["threshold_applied"] is False


def test_mapping_order_is_not_an_input_channel() -> None:
    relation_set = _relation_set()
    labels = _labels(relation_set)
    reversed_labels = dict(reversed(tuple(labels.items())))
    first = binder.bind_relation_set_to_slots_v1(
        relation_set,
        slot_labels=labels,
        encoder=_ExactTextEncoder(),
        encoder_binding_sha256=_ENCODER,
    )
    second = binder.bind_relation_set_to_slots_v1(
        relation_set,
        slot_labels=reversed_labels,
        encoder=_ExactTextEncoder(),
        encoder_binding_sha256=_ENCODER,
    )
    assert first.graph.graph_evidence_binding_sha256 == (
        second.graph.graph_evidence_binding_sha256
    )
    assert first.receipt_bytes == second.receipt_bytes


def test_semantic_matrix_is_complete_quantized_and_surface_only() -> None:
    result = binder.semantic_slot_score_matrix_v1(
        source_slot_labels={"s.a": "alpha", "s.b": "beta"},
        target_slot_labels={"t.a": "alpha", "t.b": "gamma"},
        encoder=_ExactTextEncoder(),
        encoder_binding_sha256=_ENCODER,
    )
    assert len(result.matrix.rows) == 4
    values = {(a, b): score for a, b, score in result.matrix.rows}
    assert values[("s.a", "t.a")] == 1_000_000
    assert values[("s.a", "t.b")] == 0
    assert result.receipt["gold_mapping_access_count"] == 0
    assert result.receipt["cross_system_background_access_count"] == 0
    assert result.receipt["matrix_commitment"] == result.matrix.commitment


def test_tamper_and_bad_encoder_fail_closed() -> None:
    relation_set = _relation_set()
    good = binder.bind_relation_set_to_slots_v1(
        relation_set,
        slot_labels=_labels(relation_set),
        encoder=_ExactTextEncoder(),
        encoder_binding_sha256=_ENCODER,
    )
    with pytest.raises(binder.SlotGraphBinderError):
        replace(good, receipt_bytes=b"{}")

    class _Bad:
        def encode(self, texts):
            return np.full((len(texts), 2), np.nan, dtype=np.float32)

    with pytest.raises(
        binder.SlotGraphBinderError, match="SCAR_BINDER_ENCODER_INVALID"
    ):
        binder.bind_relation_set_to_slots_v1(
            relation_set,
            slot_labels=_labels(relation_set),
            encoder=_Bad(),
            encoder_binding_sha256=_ENCODER,
        )


def test_slot_validation_precedes_encoder_and_rejects_nfkc_collision() -> None:
    relation_set = _relation_set()

    class _Spy:
        called = False

        def encode(self, texts):
            self.called = True
            return np.ones((len(texts), 8), dtype=np.float32)

    for labels in (
        {"slot.a": "K", "slot.b": "\N{KELVIN SIGN}"},
        {"slot.a": "valid", "slot.b": "\ud800"},
        {"not a slot": "valid", "slot.b": "also valid"},
    ):
        encoder = _Spy()
        with pytest.raises(
            binder.SlotGraphBinderError, match="SCAR_BINDER_SLOT_INVALID"
        ):
            binder.bind_relation_set_to_slots_v1(
                relation_set,
                slot_labels=labels,
                encoder=encoder,
                encoder_binding_sha256=_ENCODER,
            )
        assert encoder.called is False


def test_empty_relation_set_still_supports_semantic_arm_graph() -> None:
    empty_story = "Too short."
    empty = bounded_set_consumer.consume_document_envelope(
        document_envelope.select_document_qualification_only(
            empty_story, leaf_selector=_Selector()
        )
    )
    result = binder.bind_relation_set_to_slots_v1(
        empty,
        slot_labels={"slot.a": "a", "slot.b": "b"},
        encoder=_ExactTextEncoder(),
        encoder_binding_sha256=_ENCODER,
    )
    assert len(result.graph.slots) == 2
    assert result.graph.relations == ()
    assert result.endpoint_bindings == ()
    assert result.receipt["relation_set_disposition"] == "EMPTY_ABSTENTION"


def test_upstream_typed_failure_blocks_graph_and_precedes_encoder() -> None:
    story = " ".join(f"FailureToken{index:03d}" for index in range(44)) + "."
    blocked = bounded_set_consumer.consume_document_envelope(
        document_envelope.select_document_qualification_only(
            story, leaf_selector=_FailureSelector()
        )
    )

    class _Spy:
        called = False

        def encode(self, texts):
            self.called = True
            return np.ones((len(texts), 8), dtype=np.float32)

    encoder = _Spy()
    with pytest.raises(
        binder.SlotGraphBinderError, match="SCAR_BINDER_INPUT_INVALID"
    ):
        binder.bind_relation_set_to_slots_v1(
            blocked,
            slot_labels={"slot.a": "a", "slot.b": "b"},
            encoder=encoder,
            encoder_binding_sha256=_ENCODER,
        )
    assert encoder.called is False


def test_module_has_no_source_scorer_network_or_filesystem_capability() -> None:
    source = inspect.getsource(binder)
    for forbidden in (
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "source_path",
        "gold_mapping=",
        "scorer=",
        "open(",
        "Path(",
    ):
        assert forbidden not in source
