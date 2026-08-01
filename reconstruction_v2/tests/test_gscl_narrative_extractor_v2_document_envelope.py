"""Source-free adversarial contract tests for the v2 document envelope.

All narratives in this file are public synthetic fixtures.  The fake leaf
uses the existing qualification-only closed-choice engine, so successful
outcomes contain real v2 parser provenance and resource summaries without
opening a benchmark, label, scorer, network, or private archive.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import inspect
import json
import re

import pytest

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from replication_runtime.gscl_narrative_extractor_v1.closed_choice_worker import (
    PromptAnswer,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice as leaf_v2,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    document_envelope as envelope,
)
from replication_runtime.gscl_narrative_extractor_v2.contract import (
    ClosedChoiceV2Abstention,
    ClosedChoiceV2Error,
)


_LEXICAL_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_RUNTIME_COMMITMENT = hashlib.sha256(
    b"gscl-document-envelope-public-fake-leaf"
).hexdigest()


class _Backend:
    """Tie-stable finite scorer that always selects ONE_RELATION."""

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
            "document."
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


def _tokens(count: int, *, prefix: str = "Token") -> str:
    return " ".join(f"{prefix}{index:04d}" for index in range(count))


def _sentence(count: int, index: int, *, terminator: str = ".") -> str:
    return _tokens(count, prefix=f"S{index:03d}T") + terminator


def _plan_counts(story: str) -> tuple[int, ...]:
    return tuple(
        row.lexical_token_count
        for row in envelope.plan_document_segments(story)
    )


def _assert_exact_byte_partition(
    story: str, plans: tuple[envelope.SegmentPlan, ...]
) -> None:
    raw = story.encode("utf-8")
    cursor = 0
    for index, plan in enumerate(plans):
        assert plan.segment_id == f"seg{index:03d}"
        assert plan.core_start_byte == cursor
        assert 0 <= plan.parent_start_byte <= plan.core_start_byte
        assert plan.core_start_byte < plan.core_end_byte
        assert plan.core_end_byte <= plan.parent_end_byte <= len(raw)
        raw[plan.core_start_byte : plan.core_end_byte].decode(
            "utf-8", errors="strict"
        )
        cursor = plan.core_end_byte
    assert cursor == len(raw)


@pytest.mark.parametrize(
    ("token_count", "eligible"),
    [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (16, False),
        (17, True),
        (175, True),
    ],
)
def test_zero_through_leaf_boundaries_have_fixed_disposition(
    token_count: int, eligible: bool
) -> None:
    story = " \t。" if token_count == 0 else _tokens(token_count)
    plans = envelope.plan_document_segments(story)
    assert len(plans) == 1
    assert plans[0].lexical_token_count == token_count
    assert plans[0].leaf_eligible is eligible
    _assert_exact_byte_partition(story, plans)

    selector = _LeafSelector()
    result = envelope.select_document_qualification_only(
        story, leaf_selector=selector
    )
    assert len(selector.calls) == int(eligible)
    assert result.segments[0].disposition is (
        envelope.SegmentDisposition.EXTRACTED
        if eligible
        else envelope.SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE
    )
    assert result.receipt["byte_outcome_coverage_complete"] is True
    assert result.receipt["downstream_eligible"] is False
    assert result.partial_projection_available is eligible


def test_empty_root_is_invalid_but_nonempty_zero_lexical_is_context_only() -> None:
    with pytest.raises(envelope.DocumentEnvelopeError) as observed:
        envelope.plan_document_segments("")
    assert observed.value.issue_id == "DOCUMENT_ROOT_INVALID"


@pytest.mark.parametrize(
    ("token_count", "expected"),
    [
        (176, (88, 88)),
        (351, (117, 117, 117)),
        (1_024, (171, 171, 171, 171, 170, 170)),
    ],
)
def test_long_single_sentence_is_balanced_without_a_short_tail(
    token_count: int, expected: tuple[int, ...]
) -> None:
    story = _tokens(token_count)
    plans = envelope.plan_document_segments(story)
    assert tuple(row.lexical_token_count for row in plans) == expected
    assert all(row.leaf_eligible for row in plans)
    assert tuple(row.chunk_index for row in plans) == tuple(
        range(len(plans))
    )
    assert {row.chunk_count for row in plans} == {len(plans)}
    assert {row.parent_sentence_id for row in plans} == {"sent000"}
    _assert_exact_byte_partition(story, plans)


def test_1025_tokens_abstain_before_any_leaf_access() -> None:
    selector = _LeafSelector()
    with pytest.raises(envelope.DocumentEnvelopeError) as observed:
        envelope.select_document_qualification_only(
            _tokens(1_025), leaf_selector=selector
        )
    assert observed.value.issue_id == (
        "DOCUMENT_TOKEN_CAPACITY_UNSUPPORTED"
    )
    assert selector.calls == []


def test_more_than_twenty_one_sentences_are_document_envelope_valid() -> None:
    story = " ".join(_sentence(17, index) for index in range(22))
    plans = envelope.plan_document_segments(story)
    assert len(plans) == 22
    assert all(row.leaf_eligible for row in plans)
    assert len({row.parent_sentence_id for row in plans}) == 22
    _assert_exact_byte_partition(story, plans)

    selector = _LeafSelector()
    result = envelope.select_document_qualification_only(
        story, leaf_selector=selector
    )
    assert len(selector.calls) == 22
    assert len(result.relations) == 22
    assert len(result.mentions) == 66
    assert result.downstream_eligible is False
    assert result.partial_projection_available is True


def test_thirty_three_eligible_sentences_abstain_before_leaf_access() -> None:
    story = " ".join(_sentence(17, index) for index in range(33))
    selector = _LeafSelector()
    with pytest.raises(envelope.DocumentEnvelopeError) as observed:
        envelope.select_document_qualification_only(
            story, leaf_selector=selector
        )
    assert observed.value.issue_id == (
        "DOCUMENT_EXTRACTABLE_SEGMENT_CAPACITY_UNSUPPORTED"
    )
    assert selector.calls == []


def test_one_hundred_short_segments_are_total_context_without_leaf_calls() -> None:
    story = " ".join(f"短段{index:03d}。" for index in range(101))
    plans = envelope.plan_document_segments(story)
    assert len(plans) == 101
    assert all(row.lexical_token_count == 1 for row in plans)
    _assert_exact_byte_partition(story, plans)

    selector = _LeafSelector()
    result = envelope.select_document_qualification_only(
        story, leaf_selector=selector
    )
    assert selector.calls == []
    assert all(
        row.disposition
        is envelope.SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE
        for row in result.segments
    )
    assert result.receipt["byte_outcome_coverage_complete"] is True
    assert (
        result.receipt["semantic_short_segment_coverage_complete"]
        is False
    )
    assert result.downstream_eligible is False


def test_extractable_sentence_plus_short_tail_is_not_silently_dropped() -> None:
    story = _sentence(17, 0) + " 尾巴。"
    selector = _LeafSelector()
    result = envelope.select_document_qualification_only(
        story, leaf_selector=selector
    )
    assert [row.plan.lexical_token_count for row in result.segments] == [
        17,
        1,
    ]
    assert [row.disposition for row in result.segments] == [
        envelope.SegmentDisposition.EXTRACTED,
        envelope.SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE,
    ]
    assert len(selector.calls) == 1
    assert result.receipt["byte_outcome_coverage_complete"] is True
    assert result.receipt["relation_recall_total"] is False


def _unicode_story() -> str:
    first = [
        "Echo",
        "relates",
        "Alpha",
        "Élodie",
        "東京",
        "e\u0301quipe",
    ] + [f"甲{index:02d}" for index in range(14)]
    second = [
        "Echo",
        "supports",
        "Beta",
        "München",
        "η",
        "e\u0301lan",
    ] + [f"乙{index:02d}" for index in range(14)]
    return " ".join(first) + "。" + " ".join(second) + "！"


def test_ascii_cjk_unicode_nfd_and_repeated_quote_rebase_globally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = _unicode_story()
    calls: Counter[str] = Counter()
    original = envelope._quote_byte_positions

    def counted(
        root: str,
        quote: str,
        character_to_byte: tuple[int, ...],
    ) -> tuple[int, ...]:
        calls[quote] += 1
        return original(root, quote, character_to_byte)

    monkeypatch.setattr(envelope, "_quote_byte_positions", counted)
    selector = _LeafSelector()
    result = envelope.select_document_qualification_only(
        story, leaf_selector=selector
    )
    assert len(selector.calls) == 2
    _assert_exact_byte_partition(
        story, tuple(row.plan for row in result.segments)
    )
    echo = tuple(row for row in result.mentions if row.quote == "Echo")
    assert tuple(row.occurrence for row in echo) == (0, 1)
    # Projection and the independently constructed envelope each scan once;
    # neither phase rescans once per repeated mention.
    assert calls["Echo"] == 2
    raw = story.encode("utf-8")
    for mention in result.mentions:
        assert raw[mention.start_byte : mention.end_byte].decode(
            "utf-8"
        ) == mention.quote
        assert mention.quote_sha256 == hashlib.sha256(
            raw[mention.start_byte : mention.end_byte]
        ).hexdigest()
    assert result.receipt["root_source_sha256"] == hashlib.sha256(
        raw
    ).hexdigest()
    assert "e\u0301quipe" in story
    assert "équipe" not in story.casefold()


def test_no_relation_is_local_and_does_not_abort_later_segments() -> None:
    story = _tokens(351, prefix="Local")
    selector = _LeafSelector(no_relation_calls=frozenset({1}))
    result = envelope.select_document_qualification_only(
        story, leaf_selector=selector
    )
    assert len(selector.calls) == 3
    assert [row.disposition for row in result.segments] == [
        envelope.SegmentDisposition.EXTRACTED,
        envelope.SegmentDisposition.NO_RELATION,
        envelope.SegmentDisposition.EXTRACTED,
    ]
    assert result.receipt["typed_failure_count"] == 0
    assert result.receipt["disposition_counts"]["NO_RELATION"] == 1
    assert result.downstream_eligible is False
    assert result.partial_projection_available is True


def test_typed_leaf_failure_blocks_later_leaf_and_downstream() -> None:
    story = _tokens(351, prefix="Failure")
    selector = _LeafSelector(typed_failure_calls=frozenset({1}))
    result = envelope.select_document_qualification_only(
        story, leaf_selector=selector
    )
    assert len(selector.calls) == 2
    assert [row.disposition for row in result.segments] == [
        envelope.SegmentDisposition.EXTRACTED,
        envelope.SegmentDisposition.TYPED_FAILURE,
        envelope.SegmentDisposition.TYPED_FAILURE,
    ]
    assert result.segments[1].error_code == "V2_MODEL_FORWARD_FAILED"
    assert result.segments[2].error_code == (
        "DOCUMENT_ABORTED_AFTER_TYPED_FAILURE"
    )
    assert result.segments[2].leaf_called is False
    assert result.receipt["typed_failure_count"] == 2
    assert result.downstream_eligible is False
    assert result.partial_projection_available is False


def _rebuild_with_segments(
    result: envelope.NarrativeDocumentEnvelopeV1,
    segments: tuple[envelope.SegmentOutcome, ...],
) -> envelope.NarrativeDocumentEnvelopeV1:
    receipt = envelope._receipt_bytes(
        source_text=result.source_text,
        outcomes=segments,
        mentions=result.mentions,
        relations=result.relations,
    )
    return envelope.NarrativeDocumentEnvelopeV1(
        source_text=result.source_text,
        segments=segments,
        mentions=result.mentions,
        relations=result.relations,
        receipt_bytes=receipt,
    )


def test_self_consistent_noncanonical_plan_is_rejected() -> None:
    result = envelope.select_document_qualification_only(
        _tokens(17), leaf_selector=_LeafSelector()
    )
    forged = replace(
        result.segments[0],
        plan=replace(
            result.segments[0].plan, lexical_token_count=18
        ),
    )
    with pytest.raises(envelope.DocumentEnvelopeError) as observed:
        _rebuild_with_segments(result, (forged,))
    assert observed.value.issue_id == (
        "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"
    )


def test_outcome_must_account_for_every_segment_projection() -> None:
    result = envelope.select_document_qualification_only(
        _tokens(17), leaf_selector=_LeafSelector()
    )
    forged = replace(
        result.segments[0],
        mention_ids=result.segments[0].mention_ids[:-1],
    )
    with pytest.raises(envelope.DocumentEnvelopeError) as observed:
        _rebuild_with_segments(result, (forged,))
    assert observed.value.issue_id == "DOCUMENT_OWNERSHIP_INVALID"


def test_leaf_commitments_recompute_from_private_decision() -> None:
    result = envelope.select_document_qualification_only(
        _tokens(17), leaf_selector=_LeafSelector()
    )
    forged = replace(
        result.segments[0], leaf_decision_sha256="0" * 64
    )
    with pytest.raises(envelope.DocumentEnvelopeError) as observed:
        _rebuild_with_segments(result, (forged,))
    assert observed.value.issue_id == "DOCUMENT_LEAF_DECISION_INVALID"


@pytest.mark.parametrize(
    "tamper",
    ["reorder", "drop", "gap", "overlap", "duplicate"],
)
def test_segment_topology_tamper_fails_closed(tamper: str) -> None:
    story = _tokens(176, prefix="Topology")
    raw = story.encode("utf-8")
    plans = list(envelope.plan_document_segments(story))
    if tamper == "reorder":
        plans.reverse()
    elif tamper == "drop":
        plans.pop()
    elif tamper == "gap":
        plans[1] = replace(
            plans[1], core_start_byte=plans[1].core_start_byte + 1
        )
    elif tamper == "overlap":
        plans[1] = replace(
            plans[1], core_start_byte=plans[1].core_start_byte - 1
        )
    else:
        plans[1] = plans[0]
    with pytest.raises(envelope.DocumentEnvelopeError) as observed:
        envelope._validate_plan(raw, tuple(plans))
    assert observed.value.issue_id == "DOCUMENT_SEGMENT_TOPOLOGY_INVALID"


def test_cross_core_projected_mentions_fail_ownership_validation() -> None:
    story = " ".join("Echo" for _ in range(176))
    result = envelope.select_document_qualification_only(
        story, leaf_selector=_LeafSelector()
    )
    assert len(result.segments) == 2
    first_ids = result.segments[0].mention_ids
    lexical = tuple(_LEXICAL_TOKEN.finditer(story))
    targets = lexical[98:101]
    moved = {
        identifier: target
        for identifier, target in zip(first_ids, targets, strict=True)
    }
    mentions = tuple(
        sorted(
            (
                replace(
                    row,
                    occurrence=int(moved[row.mention_id].start() / 5),
                    start_byte=moved[row.mention_id].start(),
                    end_byte=moved[row.mention_id].end(),
                )
                if row.mention_id in moved
                else row
                for row in result.mentions
            ),
            key=lambda row: (row.start_byte, row.end_byte, row.mention_id),
        )
    )
    mention_by_id = {row.mention_id: row for row in mentions}
    relations = tuple(
        sorted(
            result.relations,
            key=lambda row: (
                mention_by_id[row.anchor_mention_id].start_byte,
                row.relation_id,
            ),
        )
    )
    receipt = envelope._receipt_bytes(
        source_text=story,
        outcomes=result.segments,
        mentions=mentions,
        relations=relations,
    )
    with pytest.raises(envelope.DocumentEnvelopeError) as observed:
        envelope.NarrativeDocumentEnvelopeV1(
            source_text=story,
            segments=result.segments,
            mentions=mentions,
            relations=relations,
            receipt_bytes=receipt,
        )
    assert observed.value.issue_id == "DOCUMENT_OWNERSHIP_INVALID"


def test_two_runs_are_byte_exact_and_resources_obey_global_bounds() -> None:
    story = _tokens(1_024, prefix="Maximum")
    first = envelope.select_document_qualification_only(
        story, leaf_selector=_LeafSelector()
    )
    second = envelope.select_document_qualification_only(
        story, leaf_selector=_LeafSelector()
    )
    assert first.segments == second.segments
    assert first.mentions == second.mentions
    assert first.relations == second.relations
    assert first.receipt_bytes == second.receipt_bytes
    resource = first.receipt["resource_summary"]
    assert resource["root_lexical_token_count"] == 1_024
    assert resource["segment_count"] == 6
    assert resource["leaf_call_count"] == 6
    assert resource["projected_mention_count"] == 18
    assert resource["projected_relation_count"] == 6
    assert (
        resource["reported_success_candidate_count"]
        <= resource["declared_candidate_bound"]
        <= envelope.MAXIMUM_DECLARED_CANDIDATES
    )
    assert (
        resource["reported_success_forward_batch_count"]
        <= resource["declared_forward_batch_call_bound"]
        <= envelope.MAXIMUM_DECLARED_FORWARD_BATCH_CALLS
    )


def test_public_surface_has_no_old_arn_private_or_evaluator_access() -> None:
    assert tuple(
        inspect.signature(
            envelope.select_document_qualification_only
        ).parameters
    ) == ("story_text", "leaf_selector")
    source = inspect.getsource(envelope).casefold()
    forbidden = (
        "predictor_pack",
        "label_access",
        "online_evaluator",
        "scorer_access",
        "arn_input",
        ".read_bytes(",
        "urlopen(",
        "requests.",
    )
    assert all(fragment not in source for fragment in forbidden)

    result = envelope.select_document_qualification_only(
        _tokens(17), leaf_selector=_LeafSelector()
    )
    receipt = result.receipt
    assert receipt["claim_scope"] == (
        "caller_bound_document_orchestration_consistency_only"
    )
    assert receipt["formal_leaf_authority_established"] is False
    assert (
        receipt["private_leaf_evidence_required_for_validation"]
        is True
    )
    assert receipt["downstream_eligible"] is False
