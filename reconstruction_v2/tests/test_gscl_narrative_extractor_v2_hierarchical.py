from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
import re
from typing import Callable

import pytest

from assumption_agent.gscl_arn_intrinsic_arms_v1 import (
    IntrinsicArm,
    evaluate_intrinsic_item,
)
from assumption_agent.gscl_narrative_correspondence_v1 import (
    MappingSearchConfig,
    NarrativeExtraction,
    NarrativeSource,
    SemanticScoreTable,
    parse_untrusted_generator_completion,
)
from replication_runtime.gscl_narrative_extractor_v1.closed_choice_worker import (
    PromptAnswer,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice as v2,
)
from replication_runtime.gscl_narrative_extractor_v2.contract import (
    ERROR_TAXONOMY,
    ClosedChoiceV2Abstention,
    ClosedChoiceV2Error,
    ErrorCategory,
    non_content_failure_record,
)

TeacherForcedScore = v2.TeacherForcedScore


_RUNTIME_COMMITMENT = hashlib.sha256(
    b"gscl-v2-source-free-regression-backend"
).hexdigest()
_PLAN_SUFFIX = {
    "none": "no_relation",
    "one": "one_relation",
    "two": "two_relations",
}


class FakeBackend:
    """Deterministic finite-choice backend; it never authors content."""

    def __init__(
        self,
        *,
        plan: str = "one",
        plan_by_sentence: dict[str, str] | None = None,
        preferred_width_by_role: dict[str, int] | None = None,
        favour_later_roles: tuple[str, ...] = (),
        malformed: bool = False,
        raised: ClosedChoiceV2Error | None = None,
        wire_token_count: int | None = None,
    ) -> None:
        if plan not in _PLAN_SUFFIX:
            raise ValueError("fake_plan_invalid")
        self.plan = plan
        self.plan_by_sentence = dict(plan_by_sentence or {})
        self.preferred_width_by_role = dict(
            preferred_width_by_role or {}
        )
        self.favour_later_roles = favour_later_roles
        self.malformed = malformed
        self.raised = raised
        self.wire_token_count = wire_token_count
        self.calls: list[tuple[PromptAnswer, ...]] = []

    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    @staticmethod
    def _is_width_choice(
        pair: PromptAnswer, width: int
    ) -> bool:
        blob = pair.candidate_key + "\n" + pair.answer
        patterns = (
            rf"(?:lexical_width|span_width|boundary_width)"
            rf"[^0-9]*0*{width}(?:[^0-9]|$)",
            rf"\.w0*{width}(?:[^0-9]|$)",
            rf"\.(?:width|boundary)\.0*{width}"
            rf"(?:[^0-9]|$)",
        )
        return any(re.search(pattern, blob) for pattern in patterns)

    def _score(self, pair: PromptAnswer) -> int:
        key = pair.candidate_key
        plan_match = re.match(r"^(s[0-9]{2})\.plan\.", key)
        if plan_match is not None:
            sentence_id = plan_match.group(1)
            desired = self.plan_by_sentence.get(
                sentence_id, self.plan
            )
            return (
                1_000_000
                if key.endswith(_PLAN_SUFFIX[desired])
                else 0
            )
        for role, width in self.preferred_width_by_role.items():
            if key.startswith(role + ".") and self._is_width_choice(
                pair, width
            ):
                return 900_000
        if any(
            key.startswith(role + ".")
            for role in self.favour_later_roles
        ):
            # Numeric candidate components are program-owned enumeration
            # indices.  Preferring larger ones deterministically perturbs
            # only the named later relation.
            numbers = tuple(int(row) for row in re.findall(r"[0-9]+", key))
            weighted = sum(
                (index + 1) * value
                for index, value in enumerate(numbers)
            )
            return 100_000 + weighted
        return 0

    def score_batch(
        self, pairs: tuple[PromptAnswer, ...]
    ) -> tuple[TeacherForcedScore, ...]:
        self.calls.append(pairs)
        if self.raised is not None:
            raise self.raised
        if self.malformed:
            return ()
        rows: list[TeacherForcedScore] = []
        for pair in pairs:
            answer_tokens = max(1, len(pair.answer.split()))
            score = self._score(pair)
            rows.append(
                TeacherForcedScore(
                    total_logprob_microunits=(
                        score * answer_tokens
                    ),
                    answer_token_count=answer_tokens,
                    context_and_answer_token_count=(
                        answer_tokens + 80
                    ),
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int:
        if self.wire_token_count is not None:
            return self.wire_token_count
        return max(1, len(completion.encode("utf-8")) // 4)

    @property
    def candidate_keys(self) -> tuple[str, ...]:
        return tuple(
            pair.candidate_key
            for batch in self.calls
            for pair in batch
        )


def _parser(story: str, completion: str) -> NarrativeExtraction:
    return parse_untrusted_generator_completion(
        NarrativeSource(
            "qualification."
            + hashlib.sha256(story.encode("utf-8")).hexdigest()[:24],
            story,
        ),
        completion,
    )


def _story(word_count: int, *, prefix: str = "Token") -> str:
    return " ".join(
        f"{prefix}{index:03d}" for index in range(word_count)
    )


def _multi_sentence(total: int, sentence_count: int) -> str:
    base, remainder = divmod(total, sentence_count)
    if base < 3:
        raise ValueError("test_sentence_too_short")
    rows: list[str] = []
    cursor = 0
    for sentence in range(sentence_count):
        size = base + (1 if sentence < remainder else 0)
        words = [
            f"S{sentence}Word{cursor + offset}"
            for offset in range(size)
        ]
        cursor += size
        rows.append(" ".join(words) + ".")
    return " ".join(rows)


def _run(
    story: str, backend: FakeBackend | None = None
) -> tuple[v2.ClosedChoiceV2Decision, FakeBackend]:
    actual = backend or FakeBackend()
    return (
        v2.select_hierarchical_qualification_only(
            story,
            backend=actual,
            narrative_parser=_parser,
        ),
        actual,
    )


def _wire_relations(
    decision: v2.ClosedChoiceV2Decision,
) -> list[dict[str, object]]:
    wire = json.loads(decision.wire_completion)
    return [
        relation
        for episode in wire["episodes"]
        for relation in episode["relations"]
    ]


def _canonical_wire(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@pytest.mark.parametrize(
    "word_count", [17, 31, 32, 33, 64, 128, 175]
)
def test_supported_length_boundaries_parse_and_stay_bounded(
    word_count: int,
) -> None:
    decision, backend = _run(_story(word_count))
    resource = decision.receipt["resource_summary"]
    assert isinstance(decision.extraction, NarrativeExtraction)
    assert resource["relation_count"] == 1
    assert resource["candidate_count"] <= v2.MAXIMUM_TOTAL_CANDIDATES
    assert (
        resource["forward_batch_count"]
        <= v2.MAXIMUM_FORWARD_BATCH_CALLS
    )
    assert resource["episode_count"] <= v2.MAXIMUM_EPISODES
    assert all(
        1 <= len(batch) <= v2.SCORING_BATCH_SIZE
        for batch in backend.calls
    )


@pytest.mark.parametrize("sentence_count", [1, 2, 5])
def test_one_two_and_five_parser_aligned_sentences(
    sentence_count: int,
) -> None:
    decision, _ = _run(_multi_sentence(40, sentence_count))
    resource = decision.receipt["resource_summary"]
    assert resource["sentence_count"] == sentence_count
    assert resource["relation_count"] == sentence_count
    assert len(decision.extraction.generators) == sentence_count
    assert len(decision.extraction.mentions) == 3 * sentence_count


def test_plan_none_one_two_on_same_seventeen_token_story() -> None:
    story = _story(17)
    none_backend = FakeBackend(plan="none")
    with pytest.raises(ClosedChoiceV2Abstention) as abstention:
        _run(story, none_backend)
    assert abstention.value.issue_id == "V2_PLAN_NO_RELATION_SELECTED"
    assert abstention.value.before_model_forward is False
    assert none_backend.candidate_keys
    assert all(".plan." in key for key in none_backend.candidate_keys)

    one, _ = _run(story, FakeBackend(plan="one"))
    two, _ = _run(story, FakeBackend(plan="two"))
    assert len(one.extraction.generators) == 1
    assert len(one.extraction.mentions) == 3
    assert len(two.extraction.generators) == 2
    assert len(two.extraction.mentions) == 6


def test_five_token_episode_does_not_enumerate_two_but_six_does() -> None:
    def plan_keys(first_sentence_tokens: int) -> tuple[str, ...]:
        story = (
            _story(first_sentence_tokens, prefix="First")
            + ". "
            + _story(
                17 - first_sentence_tokens, prefix="Second"
            )
            + "."
        )
        episodes = tuple(
            episode
            for episode in v2.build_hierarchical_episodes(story)
            if episode.sentence_id == "s00"
        )
        backend = FakeBackend(plan="one")
        v2._select_sentence_plan(
            sentence_id="s00",
            episodes=episodes,
            backend=backend,
        )
        return tuple(
            key
            for key in backend.candidate_keys
            if ".plan." in key
        )

    assert not any(
        key.endswith("two_relations") for key in plan_keys(5)
    )
    assert any(
        key.endswith("two_relations") for key in plan_keys(6)
    )


def test_none_plan_has_no_episode_span_or_enum_forward() -> None:
    backend = FakeBackend(plan="none")
    with pytest.raises(ClosedChoiceV2Abstention):
        _run(_story(17), backend)
    assert backend.candidate_keys
    assert all(
        re.fullmatch(
            r"s[0-9]{2}\.plan\."
            r"(?:no_relation|one_relation|two_relations)",
            key,
        )
        for key in backend.candidate_keys
    )


def test_three_independent_endpoint_rankings_per_generator() -> None:
    decision, backend = _run(_story(17), FakeBackend(plan="two"))
    assert len(decision.extraction.generators) == 2
    keys = backend.candidate_keys
    commitments = decision.receipt[
        "endpoint_selection_receipt_commitments"
    ]
    assert set(commitments) == {"r00", "r01"}
    for relation_index in range(2):
        relation = f"r{relation_index:02d}"
        assert set(commitments[relation]) == {
            "anchor",
            "object0",
            "object1",
        }
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in commitments[relation].values()
        )
        assert len(set(commitments[relation].values())) == 3
        for endpoint in ("anchor", "object0", "object1"):
            role = f"{relation}.{endpoint}"
            role_keys = tuple(
                key for key in keys if key.startswith(role + ".")
            )
            assert role_keys
            assert any(".group." in key for key in role_keys)
            assert any(".leaf." in key for key in role_keys)
            assert any(
                marker in key
                for key in role_keys
                for marker in (".width.", ".boundary.", ".w")
            )
    assert decision.receipt["exclusive_endpoint_ownership"] is True
    assert "independently_model_ranked" in (
        decision.receipt["slot_binding_semantics"]
    )


def test_changing_later_generator_never_changes_earlier_slots() -> None:
    story = _story(17)
    baseline, _ = _run(story, FakeBackend(plan="two"))
    perturbed, _ = _run(
        story,
        FakeBackend(
            plan="two",
            favour_later_roles=("r01",),
        ),
    )
    baseline_relations = _wire_relations(baseline)
    perturbed_relations = _wire_relations(perturbed)
    assert baseline_relations[0] == perturbed_relations[0]
    assert baseline_relations[1] != perturbed_relations[1]


def test_hierarchical_boundary_selection_supports_widths_one_to_four() -> None:
    story = _multi_sentence(24, 4)
    backend = FakeBackend(
        preferred_width_by_role={
            "r00.anchor": 1,
            "r01.anchor": 2,
            "r02.anchor": 3,
            "r03.anchor": 4,
        }
    )
    decision, backend = _run(story, backend)
    mentions = {
        mention.mention_id: mention
        for mention in decision.extraction.mentions
    }
    widths = [
        len(
            re.findall(
                r"[^\W_]+",
                mentions[generator.anchor_mention_id].quote,
                re.UNICODE,
            )
        )
        for generator in decision.extraction.generators
    ]
    assert widths == [1, 2, 3, 4]
    for role in backend.preferred_width_by_role:
        assert any(
            key.startswith(role + ".")
            and FakeBackend._is_width_choice(pair, widths[int(role[1:3])])
            for batch in backend.calls
            for pair in batch
            for key in (pair.candidate_key,)
        )


def test_repeated_phrases_bind_exact_disjoint_occurrences() -> None:
    story = (
        "Aster guides Birch while Aster supports Cedar. "
        "Aster guides Birch while Aster supports Cedar. "
        "Aster follows Dahlia while Aster greets Elm."
    )
    episodes = v2.build_hierarchical_episodes(story)
    aster = [
        atom
        for episode in episodes
        for atom in episode.atoms
        if atom.quote == "Aster"
    ]
    assert [atom.occurrence for atom in aster] == list(
        range(len(aster))
    )
    decision, _ = _run(story)
    intervals = [
        (mention.start_byte, mention.end_byte)
        for mention in decision.extraction.mentions
    ]
    assert len(intervals) == len(set(intervals))
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            assert left[1] <= right[0] or right[1] <= left[0]


def test_no_terminal_punctuation_semicolon_quotes_and_unicode() -> None:
    story = (
        '甲方 observes "Beta"; Gamma supports Delta while Élodie guides '
        "Zeta and η links Theta before Iota follows Kappa then Lambda "
        "helps Mu near Nu"
    )
    decision, _ = _run(story)
    assert isinstance(decision.extraction, NarrativeExtraction)
    assert decision.receipt["resource_summary"]["sentence_count"] == 1


def test_object_degree_is_one_and_shared_slot_bonus_is_structurally_zero() -> None:
    decision, _ = _run(_story(17), FakeBackend(plan="two"))
    slot_degrees = Counter(
        slot
        for generator in decision.extraction.generators
        for slot in generator.slot_mention_ids
    )
    assert set(slot_degrees.values()) == {1}
    assert set(slot_degrees) == set(
        decision.extraction.hypergraph.object_mention_ids
    )
    assert all(
        len(generator.slot_mention_ids) == 2
        for generator in decision.extraction.generators
    )
    # A shared-slot term has no support when every object degree is one.
    assert sum(value - 1 for value in slot_degrees.values()) == 0


def test_twenty_one_relations_parse_to_sixty_three_mentions() -> None:
    story = _multi_sentence(63, 21)
    decision, _ = _run(story)
    assert len(decision.extraction.generators) == 21
    assert len(decision.extraction.mentions) == 63
    assert (
        decision.receipt["resource_summary"]["relation_count"]
        == v2.MAXIMUM_RELATION_UNITS
    )


def test_twenty_two_planned_relations_typed_reject_before_spans() -> None:
    story = _multi_sentence(66, 11)
    backend = FakeBackend(plan="two")
    with pytest.raises(ClosedChoiceV2Abstention) as error:
        _run(story, backend)
    assert error.value.issue_id == (
        "V2_PLAN_RELATION_CAPACITY_EXCEEDED"
    )
    assert error.value.before_model_forward is False
    assert not any(
        re.match(r"^r[0-9]{2}\.", key)
        for key in backend.candidate_keys
    )


def test_ties_are_deterministic_for_scoring_batches_one_and_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = _multi_sentence(35, 2)
    by_batch: dict[
        int, tuple[v2.ClosedChoiceV2Decision, FakeBackend]
    ] = {}
    for batch_size in (1, 4):
        monkeypatch.setattr(v2, "SCORING_BATCH_SIZE", batch_size)
        first, first_backend = _run(story)
        second, second_backend = _run(story)
        assert first.wire_completion == second.wire_completion
        assert first.canonical_completion == second.canonical_completion
        assert first.receipt_bytes == second.receipt_bytes
        assert max(map(len, first_backend.calls)) <= batch_size
        assert max(map(len, second_backend.calls)) <= batch_size
        by_batch[batch_size] = (first, first_backend)
    assert (
        by_batch[1][0].wire_completion
        == by_batch[4][0].wire_completion
    )
    assert (
        by_batch[1][0].canonical_completion
        == by_batch[4][0].canonical_completion
    )


def _install_wire_tamper(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object], tuple[object, ...]], None],
) -> None:
    original = v2._build_wire

    def tampered(relations: object) -> str:
        relation_tuple = tuple(relations)
        value = json.loads(original(relation_tuple))
        mutate(value, relation_tuple)
        return _canonical_wire(value)

    monkeypatch.setattr(v2, "_build_wire", tampered)


def _width_one_id(atom_id: str, exemplar: str) -> str:
    match = re.fullmatch(
        r"(e[0-9]{2}\.t[0-9]{2})(\.w(0*)1)", exemplar
    )
    if match is None:
        return atom_id
    return atom_id + ".w" + match.group(3) + "1"


def test_unscored_but_catalogued_endpoint_is_rejected_specifically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = _story(17)
    episodes = {
        episode.episode_id: episode
        for episode in v2.build_hierarchical_episodes(story)
    }

    def mutate(
        wire: dict[str, object], relations: tuple[object, ...]
    ) -> None:
        selected = {
            span.span_id
            for relation in relations
            for span in (
                relation.anchor,
                relation.object0,
                relation.object1,
            )
        }
        relation = relations[0]
        for atom in episodes[relation.episode_id].atoms:
            candidate = _width_one_id(
                atom.span_id, relation.object1.span_id
            )
            if candidate not in selected:
                wire["episodes"][0]["relations"][0][
                    "object1_span_id"
                ] = candidate
                return
        raise AssertionError("test_unselected_endpoint_missing")

    _install_wire_tamper(monkeypatch, mutate)
    with pytest.raises(ClosedChoiceV2Error) as error:
        _run(story)
    assert error.value.issue_id == (
        "V2_WIRE_ENDPOINT_SELECTION_MISSING"
    )


def test_cross_episode_endpoint_is_rejected_specifically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(
        wire: dict[str, object], _relations: tuple[object, ...]
    ) -> None:
        first = wire["episodes"][0]["relations"][0]
        second = wire["episodes"][1]["relations"][0]
        first["object1_span_id"] = second["object1_span_id"]

    _install_wire_tamper(monkeypatch, mutate)
    with pytest.raises(ClosedChoiceV2Error) as error:
        _run(_multi_sentence(20, 2))
    assert error.value.issue_id == "V2_WIRE_ENDPOINT_REF_INVALID"


def test_overlapping_endpoint_is_rejected_specifically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(
        wire: dict[str, object], _relations: tuple[object, ...]
    ) -> None:
        relation = wire["episodes"][0]["relations"][0]
        relation["object1_span_id"] = relation["anchor_span_id"]

    _install_wire_tamper(monkeypatch, mutate)
    with pytest.raises(ClosedChoiceV2Error) as error:
        _run(_story(17))
    assert error.value.issue_id == "V2_WIRE_ENDPOINT_OVERLAP"


def test_program_owns_wire_fields_ids_and_enum_primitives() -> None:
    decision, _ = _run(_story(17), FakeBackend(plan="two"))
    wire = json.loads(decision.wire_completion)
    assert set(wire) == {"episodes", "schema_version"}
    for episode in wire["episodes"]:
        assert set(episode) == {
            "episode_id",
            "relations",
            "sentence_id",
        }
        for relation in episode["relations"]:
            assert set(relation) == {
                "anchor_span_id",
                "causal_orientation",
                "generator_kind",
                "object0_span_id",
                "object1_span_id",
                "polarity",
                "relation_id",
                "temporal_orientation",
            }
            assert relation["generator_kind"] in v2.GENERATOR_KINDS
            assert relation["polarity"] in v2.POLARITIES
            assert relation["temporal_orientation"] in v2.ORIENTATIONS
            assert relation["causal_orientation"] in v2.ORIENTATIONS


def test_atomic_catalog_and_resource_bounds_are_linear() -> None:
    story = _story(175)
    episodes = v2.build_hierarchical_episodes(story)
    atoms = [atom for episode in episodes for atom in episode.atoms]
    assert len(atoms) == 175
    assert all(len(atom.quote.split()) == 1 for atom in atoms)
    assert max(len(episode.atoms) for episode in episodes) <= 24
    decision, backend = _run(story)
    resource = decision.receipt["resource_summary"]
    assert resource["candidate_count"] <= v2.MAXIMUM_TOTAL_CANDIDATES
    assert (
        resource["forward_batch_count"]
        <= v2.MAXIMUM_FORWARD_BATCH_CALLS
    )
    assert max(len(batch) for batch in backend.calls) <= 4
    source = inspect.getsource(v2.build_hierarchical_episodes)
    assert "itertools.product" not in source
    assert "itertools.combinations" not in source
    assert "MAXIMUM_SPAN_WORDS" not in source


def test_source_has_no_fixed_relation_threshold_or_cyclic_endpoint() -> None:
    source = inspect.getsource(v2)
    forbidden = (
        "singleton_support_object_span_id",
        "slot1_is_not_a_model_extracted_endpoint",
        "object_pool[(index + 1) % len(object_pool)]",
        "_relation_count_for_token_count",
        "relation_count = 1 if token_count",
    )
    assert all(fragment not in source for fragment in forbidden)
    assert ".generate(" not in source
    assert "free_form_generation_count" in source


def test_error_taxonomy_preserves_distinct_failure_categories() -> None:
    story = _story(17)
    for issue, category in (
        ("V2_CONTEXT_TOKEN_LIMIT_EXCEEDED", ErrorCategory.CONTEXT),
        ("V2_CUDA_RUNTIME_UNAVAILABLE", ErrorCategory.CUDA),
        ("V2_MODEL_FORWARD_FAILED", ErrorCategory.MODEL),
    ):
        with pytest.raises(ClosedChoiceV2Error) as error:
            _run(
                story,
                FakeBackend(raised=ClosedChoiceV2Error(issue)),
            )
        assert error.value.issue_id == issue
        assert error.value.category is category

    with pytest.raises(ClosedChoiceV2Error) as score_error:
        _run(story, FakeBackend(malformed=True))
    assert score_error.value.issue_id == (
        "V2_MODEL_SCORE_BATCH_INVALID"
    )

    with pytest.raises(ClosedChoiceV2Error) as parser_error:
        v2.select_hierarchical_qualification_only(
            story,
            backend=FakeBackend(),
            narrative_parser=lambda _story, _completion: (
                (_ for _ in ()).throw(ValueError("private"))
            ),
        )
    assert parser_error.value.issue_id == "V2_PARSER_REJECTED"

    with pytest.raises(ClosedChoiceV2Error) as token_error:
        _run(story, FakeBackend(wire_token_count=0))
    assert token_error.value.issue_id == "V2_TOKEN_BOUNDARY_INVALID"

    with pytest.raises(ClosedChoiceV2Abstention) as catalog_error:
        _run(_story(16))
    assert catalog_error.value.category is ErrorCategory.CATALOG

    with pytest.raises(ClosedChoiceV2Abstention) as plan_error:
        _run(story, FakeBackend(plan="none"))
    assert plan_error.value.category.value == "selection"

    assert ERROR_TAXONOMY["V2_WIRE_ENDPOINT_REF_INVALID"] is (
        ErrorCategory.VERIFIER
    )
    assert {
        ErrorCategory.CATALOG.value,
        ErrorCategory.CONTEXT.value,
        ErrorCategory.CUDA.value,
        ErrorCategory.MODEL.value,
        ErrorCategory.PARSER.value,
        ErrorCategory.TOKEN_BOUNDARY.value,
        ErrorCategory.VERIFIER.value,
        plan_error.value.category.value,
    } == {
        "catalog",
        "context",
        "cuda",
        "model",
        "parser",
        "token_boundary",
        "verifier",
        "selection",
    }

    record = non_content_failure_record(plan_error.value)
    assert record == {
        "error_category": "selection",
        "error_code": "V2_PLAN_NO_RELATION_SELECTED",
        "generation_valid": False,
        "pre_model_abstention": False,
    }


def _structural_scores(
    query: NarrativeExtraction, candidate: NarrativeExtraction
) -> SemanticScoreTable:
    return SemanticScoreTable.from_mappings(
        object_scores={
            (left, right): 100
            for left in query.hypergraph.object_mention_ids
            for right in candidate.hypergraph.object_mention_ids
        },
        generator_scores={
            (left.generator_id, right.generator_id): 100
            for left in query.generators
            for right in candidate.generators
        },
    )


def test_extraction_binds_semantic_legacy_flat_and_full_consumers() -> None:
    query, _ = _run(_story(17, prefix="Query"))
    first, _ = _run(_story(17, prefix="First"))
    second, _ = _run(_story(17, prefix="Second"))
    result = evaluate_intrinsic_item(
        opaque_item_id=hashlib.sha256(b"v2 item").hexdigest(),
        query=query.extraction,
        candidates=(first.extraction, second.extraction),
        raw_text_scorer=lambda left, right: len(
            set(re.findall(rb"[A-Za-z]+", left))
            & set(re.findall(rb"[A-Za-z]+", right))
        ),
        legacy_vectorizer=lambda extraction, features: (
            len(extraction.mentions),
        ),
        legacy_feature_ids=("mention_count",),
        structural_scorer=_structural_scores,
        mapping_config=MappingSearchConfig(),
        raw_text_scorer_commitment=hashlib.sha256(
            b"semantic"
        ).hexdigest(),
        legacy_vectorizer_commitment=hashlib.sha256(
            b"legacy"
        ).hexdigest(),
        structural_scorer_commitment=hashlib.sha256(
            b"structural"
        ).hexdigest(),
    )
    assert {prediction.arm for prediction in result.predictions} == set(
        IntrinsicArm
    )
    binding = query.receipt["consumer_binding"]
    assert set(binding) == {
        "flat_label_no_verifier",
        "full",
        "legacy_keyword",
        "semantic_only",
    }
    assert len(set(binding.values())) == 1


def test_authority_surface_fails_closed() -> None:
    with pytest.raises(ClosedChoiceV2Error) as error:
        v2._HierarchicalEngine(object())
    assert error.value.issue_id == "V2_AUTHORITY_INVALID"
