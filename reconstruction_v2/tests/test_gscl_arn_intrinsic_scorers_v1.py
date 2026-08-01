from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from assumption_agent import gscl_arn_intrinsic_scorers_v1 as scorer_module
from assumption_agent.gscl_arn_intrinsic_arms_v1 import (
    IntrinsicContractError,
    evaluate_frozen_intrinsic_item,
)
from assumption_agent.gscl_arn_intrinsic_scorers_v1 import (
    FrozenNarrativeScorers,
    IntrinsicScorerError,
    LEGACY_FEATURE_IDS,
    lossless_token_chunks,
)
from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from replication_runtime.qasper_minilm_v1.binding import (
    EMBEDDING_DIMENSION,
)
from replication_runtime.gscl_minilm_portable_v1.binding import (
    GSCLPortableOfflineMiniLMEncoder,
)


def _completion(story: str, left: str, verb: str, right: str) -> str:
    assert left in story and verb in story and right in story
    return json.dumps(
        {
            "generators": [
                {
                    "anchor_mention_id": "a0",
                    "causal_orientation": "none",
                    "generator_id": "g0",
                    "generator_kind": "relation",
                    "polarity": "positive",
                    "slot_mention_ids": ["m0", "m1"],
                    "temporal_orientation": "forward",
                }
            ],
            "mentions": [
                {
                    "kind": "object",
                    "mention_id": "m0",
                    "occurrence": 0,
                    "quote": left,
                },
                {
                    "kind": "object",
                    "mention_id": "m1",
                    "occurrence": 0,
                    "quote": right,
                },
                {
                    "kind": "generator",
                    "mention_id": "a0",
                    "occurrence": 0,
                    "quote": verb,
                },
            ],
            "schema_version": "gscl.narrative.extraction.v1",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _extraction(source_id: str, story: str, left: str, verb: str, right: str):
    return parse_untrusted_generator_completion(
        NarrativeSource(source_id, story),
        _completion(story, left, verb, right),
    )


class _Tokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        # Deliberately make the count depend on Unicode characters so that the
        # test forces multiple lossless chunks without model truncation.
        return {"input_ids": list(range(len(text) + 2))}


class _Model:
    tokenizer = _Tokenizer()


class _Encoder:
    _model = _Model()
    runtime_receipt = {"runtime": "synthetic"}
    canary_receipt = {"canary": "synthetic"}

    def encode(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = np.asarray(
                [
                    ((digest[index % len(digest)] + index) % 251) + 1
                    for index in range(EMBEDDING_DIMENSION)
                ],
                dtype=np.float32,
            )
            rows.append(values / np.linalg.norm(values))
        return np.vstack(rows).astype(np.float32)


def test_lossless_chunking_has_complete_utf8_coverage() -> None:
    text = ("Alpha βeta gamma. " * 200) + "终"
    chunks = lossless_token_chunks(text, _Tokenizer())
    assert len(chunks) > 1
    assert "".join(chunk for chunk, _ in chunks) == text
    assert b"".join(chunk.encode() for chunk, _ in chunks) == text.encode()
    assert max(count for _, count in chunks) <= 256


def test_build_replays_complete_batch_and_binds_private_inputs() -> None:
    first = _extraction(
        "synthetic.first",
        "Aster guides Birch.",
        "Aster",
        "guides",
        "Birch",
    )
    second = _extraction(
        "synthetic.second",
        "Cedar follows Dune.",
        "Cedar",
        "follows",
        "Dune",
    )
    scorers = FrozenNarrativeScorers.build(
        (first, second), encoder=_Encoder()
    )
    assert scorers.receipt["actual_batch_replay_exact"] is True
    assert scorers.receipt["benchmark_source_accessed"] is False
    assert scorers.receipt["labels_accessed"] is False
    assert scorers.raw_text_scorer(
        first.source.utf8_bytes, second.source.utf8_bytes
    ) == scorers.raw_text_scorer(
        first.source.utf8_bytes, second.source.utf8_bytes
    )


def test_legacy_registry_and_vector_are_fixed_and_replayed() -> None:
    extraction = _extraction(
        "synthetic.legacy",
        "A conserved quantity balances before and after transformation.",
        "quantity",
        "balances",
        "transformation",
    )
    scorers = FrozenNarrativeScorers.build(
        (extraction,), encoder=_Encoder()
    )
    first = scorers.legacy_vectorizer(extraction, LEGACY_FEATURE_IDS)
    second = scorers.legacy_vectorizer(extraction, LEGACY_FEATURE_IDS)
    assert first == second
    assert len(first) == 10
    assert all(isinstance(value, int) and value >= 0 for value in first)
    with pytest.raises(IntrinsicScorerError) as error:
        scorers.legacy_vectorizer(extraction, tuple(reversed(LEGACY_FEATURE_IDS)))
    assert error.value.issue_id == "legacy_feature_registry_drifted"


def test_structural_scores_cover_exact_object_and_generator_cross_products() -> None:
    source = _extraction(
        "synthetic.source",
        "Aster guides Birch.",
        "Aster",
        "guides",
        "Birch",
    )
    target = _extraction(
        "synthetic.target",
        "Cedar guides Dune.",
        "Cedar",
        "guides",
        "Dune",
    )
    scorers = FrozenNarrativeScorers.build(
        (source, target), encoder=_Encoder()
    )
    table = scorers.structural_scorer(source, target)
    assert len(table.object_scores) == 4
    assert len(table.generator_scores) == 1
    assert (
        table.safe_payload()
        == scorers.structural_scorer(source, target).safe_payload()
    )


def test_unprimed_input_fails_closed() -> None:
    primed = _extraction(
        "synthetic.primed",
        "Aster guides Birch.",
        "Aster",
        "guides",
        "Birch",
    )
    other = _extraction(
        "synthetic.other",
        "Cedar follows Dune.",
        "Cedar",
        "follows",
        "Dune",
    )
    scorers = FrozenNarrativeScorers.build((primed,), encoder=_Encoder())
    with pytest.raises(IntrinsicScorerError) as error:
        scorers.structural_scorer(primed, other)
    assert error.value.issue_id == "extraction_not_primed"


def test_authoritative_factory_accepts_no_injected_lane_or_commitment() -> None:
    query = _extraction(
        "synthetic.query",
        "Aster guides Birch.",
        "Aster",
        "guides",
        "Birch",
    )
    first = _extraction(
        "synthetic.first.choice",
        "Cedar guides Dune.",
        "Cedar",
        "guides",
        "Dune",
    )
    second = _extraction(
        "synthetic.second.choice",
        "Elm opposes Fir.",
        "Elm",
        "opposes",
        "Fir",
    )
    scorers = FrozenNarrativeScorers.build(
        (query, first, second), encoder=_Encoder()
    )
    # An injected test encoder remains qualification-only; it cannot cross
    # the authoritative frozen-lane boundary.
    with pytest.raises(IntrinsicContractError) as error:
        evaluate_frozen_intrinsic_item(
            opaque_item_id=hashlib.sha256(
                b"synthetic item"
            ).hexdigest(),
            query=query,
            candidates=(first, second),
            scorers=scorers,
        )
    assert error.value.issue_id == "frozen_scorer_not_formal"


def test_uninitialized_exact_formal_encoder_type_cannot_set_formal_domain() -> None:
    extraction = _extraction(
        "synthetic.forged.formal.encoder",
        "Aster guides Birch.",
        "Aster",
        "guides",
        "Birch",
    )
    forged = object.__new__(GSCLPortableOfflineMiniLMEncoder)
    with pytest.raises(
        IntrinsicScorerError, match="formal_encoder_binding_invalid"
    ):
        FrozenNarrativeScorers.build((extraction,), encoder=forged)


def test_scorer_state_is_exact_type_bound_and_deeply_immutable() -> None:
    extraction = _extraction(
        "synthetic.sealed",
        "Aster guides Birch.",
        "Aster",
        "guides",
        "Birch",
    )
    scorers = FrozenNarrativeScorers.build(
        (extraction,), encoder=_Encoder()
    )
    vector = next(iter(scorers.source_vectors.values()))
    assert vector.flags.writeable is False
    with pytest.raises(TypeError):
        scorers.source_vectors["0" * 64] = vector  # type: ignore[index]
    with pytest.raises(TypeError):
        scorers.receipt["self_hash"] = "0" * 64  # type: ignore[index]

    class InjectedScorers(FrozenNarrativeScorers):
        pass

    with pytest.raises(IntrinsicScorerError) as error:
        InjectedScorers(
            source_vectors=scorers.source_vectors,
            mention_vectors=scorers.mention_vectors,
            primed_extraction_hashes=scorers.primed_extraction_hashes,
            receipt=scorers.receipt,
            _construction_token=object(),
        )
    assert error.value.issue_id == "scorer_subclass_forbidden"

    tampered = dict(scorers.receipt)
    tampered["source_count"] = 99
    with pytest.raises(IntrinsicScorerError) as error:
        FrozenNarrativeScorers(
            source_vectors=scorers.source_vectors,
            mention_vectors=scorers.mention_vectors,
            primed_extraction_hashes=scorers.primed_extraction_hashes,
            receipt=tampered,
            _construction_token=object(),
        )
    assert error.value.issue_id == "scorer_construction_not_authorized"


def test_validate_internal_is_zero_copy_and_does_not_rehash_or_reseal_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _extraction(
        "synthetic.zero.copy",
        "Aster guides Birch.",
        "Aster",
        "guides",
        "Birch",
    )
    scorers = FrozenNarrativeScorers.build(
        (extraction,), encoder=_Encoder()
    )
    source_mapping_id = id(scorers.source_vectors)
    mention_mapping_id = id(scorers.mention_vectors)
    receipt_id = id(scorers.receipt)
    vector_ids = tuple(
        id(value)
        for value in (
            *scorers.source_vectors.values(),
            *scorers.mention_vectors.values(),
        )
    )
    monkeypatch.setattr(
        scorer_module,
        "_vector_mapping_commitment",
        lambda value: (_ for _ in ()).throw(
            AssertionError("immutable vectors must not be re-hashed")
        ),
    )
    scorers.validate_internal()
    scorers.validate_internal()
    assert id(scorers.source_vectors) == source_mapping_id
    assert id(scorers.mention_vectors) == mention_mapping_id
    assert id(scorers.receipt) == receipt_id
    assert tuple(
        id(value)
        for value in (
            *scorers.source_vectors.values(),
            *scorers.mention_vectors.values(),
        )
    ) == vector_ids


def test_validate_internal_rejects_replaced_mapping_identity() -> None:
    extraction = _extraction(
        "synthetic.replaced.mapping",
        "Aster guides Birch.",
        "Aster",
        "guides",
        "Birch",
    )
    scorers = FrozenNarrativeScorers.build(
        (extraction,), encoder=_Encoder()
    )
    object.__setattr__(
        scorers,
        "source_vectors",
        type(scorers.source_vectors)(dict(scorers.source_vectors)),
    )
    with pytest.raises(
        IntrinsicScorerError, match="scorer_construction_not_authorized"
    ):
        scorers.validate_internal()
