from __future__ import annotations

from dataclasses import replace
import inspect
import math
from typing import Mapping, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import feverous_atomic_corpus_v1 as atomic
from assumption_agent.benchmarks import feverous_offline_semantic_tensor_v1 as subject
from assumption_agent.benchmarks import feverous_p6_query_anchored_operator_v1 as operator


class FakeMiniLM:
    binding = subject.make_synthetic_backend_binding("MiniLM")

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = tuple(texts)
        self.calls.append(rows)
        matrix = np.zeros((len(rows), 384), dtype=np.float32)
        corpus_call = len(rows) == subject.CORPUS_SIZE
        for index, _text in enumerate(rows):
            if corpus_call:
                cosine = np.float32(
                    (subject.CORPUS_SIZE - index) / 9000.0
                )
                matrix[index, 0] = cosine
                matrix[index, 1] = np.float32(
                    math.sqrt(max(0.0, 1.0 - float(cosine) ** 2))
                )
            else:
                matrix[index, 0] = np.float32(1.0)
        return matrix


class FakeNER:
    binding = subject.make_synthetic_backend_binding("NER")

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    @property
    def call_sizes(self) -> list[int]:
        return [len(row) for row in self.calls]

    def extract_texts(
        self, texts: Sequence[str]
    ) -> tuple[tuple[subject.DetectedEntity, ...], ...]:
        rows = tuple(texts)
        self.calls.append(rows)
        result: list[tuple[subject.DetectedEntity, ...]] = []
        for text in rows:
            start = text.find("Alice")
            result.append(
                ()
                if start < 0
                else (
                    subject.DetectedEntity(
                        entity_type="PER",
                        start=start,
                        end=start + len("Alice"),
                        text="Alice",
                    ),
                )
            )
        return tuple(result)


class FakeNLI:
    binding = subject.make_synthetic_backend_binding("NLI")

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, str], ...]] = []

    def score_pairs(
        self, pairs: Sequence[Mapping[str, str]]
    ) -> tuple[int, ...]:
        normalized = tuple(dict(pair) for pair in pairs)
        self.calls.append(normalized)
        return tuple(
            750_000
            if pair["premise"].startswith("TARGET: positive NLI target\n")
            else -250_000
            for pair in normalized
        )


def _sentence_text(*, target: str, title: str, section: tuple[str, ...]) -> str:
    section_text = " > ".join(section) or "<ROOT>"
    return (
        f"TARGET: {target}\n"
        f"TITLE: {title}\n"
        f"SECTION_PATH: {section_text}\n"
        "TYPE: sentence"
    )


def _corpus() -> tuple[subject.SemanticCorpusUnit, ...]:
    rows: list[subject.SemanticCorpusUnit] = []
    for ordinal in range(subject.CORPUS_SIZE):
        target = "generic payload"
        title = f"Page {ordinal}"
        section: tuple[str, ...] = ()
        page_key = f"page-{ordinal}"
        official_order = 0
        if ordinal == 0:
            section = ("Section A",)
            page_key = "shared-page"
        elif ordinal == 1:
            section = ("Section B",)
            page_key = "shared-page"
            official_order = 1
        elif ordinal == 5:
            target = "positive NLI target"
        elif ordinal == 100:
            target = "Alice target"
        elif ordinal == 200:
            target = "year 2020 target"
        elif ordinal == 300:
            target = "number 3 target"
        elif ordinal == 400:
            title = "Alice"
        elif ordinal == 500:
            section = ("2020",)
        rows.append(
            subject.SemanticCorpusUnit(
                corpus_ordinal=ordinal,
                linearized_text=_sentence_text(
                    target=target,
                    title=title,
                    section=section,
                ),
                unit_type="sentence",
                page_key=page_key,
                official_order=official_order,
                section_path=section,
            )
        )
    return tuple(rows)


@pytest.fixture(scope="module")
def built_twice():
    minilm = FakeMiniLM()
    ner = FakeNER()
    nli = FakeNLI()
    prepared = subject.prepare_semantic_corpus(
        corpus_units=_corpus(),
        minilm_backend=minilm,
        ner_backend=ner,
        allow_synthetic_backends=True,
    )
    first = subject.build_prepared_offline_semantic_tensor(
        claim_text="Alice won 3 awards in 2020.",
        prepared_corpus=prepared,
        minilm_backend=minilm,
        ner_backend=ner,
        nli_backend=nli,
        allow_synthetic_backends=True,
    )
    second = subject.build_prepared_offline_semantic_tensor(
        claim_text="A generic second claim.",
        prepared_corpus=prepared,
        minilm_backend=minilm,
        ner_backend=ner,
        nli_backend=nli,
        allow_synthetic_backends=True,
    )
    return prepared, first, second, minilm, ner, nli


def test_real_atomic_multiline_target_first_text_is_accepted() -> None:
    page = {
        "title": "Synthetic_Page",
        "order": ["section_0", "sentence_0"],
        "section_0": {"value": "Main", "level": 1},
        "sentence_0": "An atomic sentence.",
    }
    compiled = atomic.compile_official_page("Synthetic_Page", page)
    source = compiled.units[0]
    unit = subject.SemanticCorpusUnit(
        corpus_ordinal=0,
        linearized_text=source.text,
        unit_type=source.sidecar.unit_type,
        page_key=source.sidecar.page,
        official_order=source.sidecar.official_ordinal,
        section_path=source.sidecar.section_path,
    )
    assert unit.target_text == source.target == "An atomic sentence."
    assert unit.linearized_text.startswith(
        "TARGET: An atomic sentence.\nTITLE: Synthetic_Page\n"
    )


def test_prepare_once_reuses_corpus_ner_embedding_and_graph_for_two_queries(
    built_twice,
) -> None:
    prepared, first, second, minilm, ner, _nli = built_twice
    assert [len(call) for call in minilm.calls].count(subject.CORPUS_SIZE) == 1
    assert len(minilm.calls) == 3
    assert ner.call_sizes == [4096, 4096, 1, 1]
    assert all("\nTITLE:" not in text for call in ner.calls[:2] for text in call)
    assert first.atomic_units is prepared.atomic_units
    assert second.atomic_units is prepared.atomic_units
    assert first.graph is prepared.graph is second.graph
    assert first.receipt["preparation_receipt_sha256"] == (
        prepared.preparation_receipt_sha256
    )
    assert first.receipt["preparation_mode"] == "precomputed_formal_path"
    assert first.receipt["corpus_MiniLM_calls_in_query"] == 0
    assert first.receipt["corpus_NER_calls_in_query"] == 0
    assert first.receipt["MiniLM_call_count"] == 1
    assert first.receipt["MiniLM_encoded_text_count"] == (
        1 + len(first.tensor.facets)
    )
    assert first.receipt["MiniLM_similarity_count"] == (
        (1 + len(first.tensor.facets)) * subject.CORPUS_SIZE
    )
    assert first.receipt["MiniLM_quantization"] == (
        "Qasper_binary64_products_math_fsum_Python_round_"
        "ties_to_even_scale_1000000"
    )


def test_target_only_ner_numeric_and_section_scoped_graph(built_twice) -> None:
    prepared, first, _second, _minilm, _ner, _nli = built_twice
    assert prepared.atomic_units[100].entities == (
        subject.make_entity_key("PER", "Alice"),
    )
    assert prepared.atomic_units[400].entities == ()
    assert "2020" in prepared.numeric_keys[200]
    assert "2020" not in prepared.numeric_keys[500]
    assert prepared.atomic_units[0].section_path == ("Section A",)
    assert prepared.atomic_units[1].section_path == ("Section B",)
    assert not any(
        edge.family == operator.SAME_PAGE_ADJACENT_OFFICIAL_ORDER
        and (edge.left_ordinal, edge.right_ordinal) == (0, 1)
        for edge in prepared.graph.edges
    )
    entity_row = first.tensor.rows[0]
    numeric_2020_row = first.tensor.rows[2]
    assert entity_row.direct_anchor_strength_ints[100] == 1_000_000
    assert entity_row.direct_anchor_strength_ints[400] == 0
    assert numeric_2020_row.direct_anchor_strength_ints[200] == 1_000_000
    assert numeric_2020_row.direct_anchor_strength_ints[500] == 0


def test_claim_facets_are_exact_atomic_compiler_projection() -> None:
    claim = "Alice was born on January 2, 2001; she earned $1,000."
    entity = subject.DetectedEntity("PER", 0, 5, "Alice")
    atomic_facets = atomic.compile_claim_facets(claim, ((0, 5),)).facets
    semantic_facets = subject.extract_claim_facets(
        claim_text=claim,
        claim_entities=(entity,),
    )
    assert [
        (row.facet.facet_type, row.facet.normalized_text)
        for row in semantic_facets
    ] == [
        (facet.kind, operator.normalize_key(facet.text))
        for facet in atomic_facets
    ]
    assert semantic_facets[0].exact_entity_key == subject.make_entity_key(
        "PER", "Alice"
    )
    assert semantic_facets[1].exact_numeric_key == operator.normalize_key(
        "January 2, 2001"
    )


def test_full_scan_shortlist_and_semantic_combination(built_twice) -> None:
    prepared, result, _second, _minilm, _ner, nli = built_twice
    assert tuple(facet.facet_type for facet in result.tensor.facets) == (
        "entity",
        "numeric_or_date",
        "numeric_or_date",
        "relation_clause",
    )
    assert len(result.tensor.dense_relevance_ints) == subject.CORPUS_SIZE
    assert all(
        len(row.semantic_coverage_ints) == subject.CORPUS_SIZE
        and len(row.direct_anchor_strength_ints) == subject.CORPUS_SIZE
        for row in result.tensor.rows
    )
    first_query_calls = nli.calls[: len(result.tensor.facets)]
    expected_exact = ({100}, {300}, {200}, set())
    for pairs, exact in zip(first_query_calls, expected_exact):
        premises = {pair["premise"] for pair in pairs}
        assert all(
            prepared.corpus_units[ordinal].linearized_text in premises
            for ordinal in range(32)
        )
        assert all(
            prepared.corpus_units[ordinal].linearized_text in premises
            for ordinal in exact
        )
        assert len(pairs) == 32 + len(exact)
    entity_row = result.tensor.rows[0]
    assert entity_row.semantic_coverage_ints[100] == 1_000_000
    assert entity_row.direct_anchor_strength_ints[100] == 1_000_000
    assert entity_row.direct_anchor_strength_ints[5] == 750_000
    assert entity_row.direct_anchor_strength_ints[40] == 0
    assert entity_row.semantic_coverage_ints[40] > 0
    assert result.receipt["semantic_combination"] == subject.SEMANTIC_COMBINATION


def test_preparation_and_query_receipts_reject_rehashed_forgery(built_twice) -> None:
    prepared, result, _second, minilm, ner, nli = built_twice
    forged_preparation = dict(prepared.receipt)
    forged_preparation["design_sha256"] = "0" * 64
    forged_body = dict(forged_preparation)
    forged_body.pop("preparation_receipt_sha256")
    forged_preparation["preparation_receipt_sha256"] = subject.stable_hash(
        forged_body
    )
    forged_prepared = replace(prepared, receipt=forged_preparation)
    with pytest.raises(subject.FeverousSemanticTensorError, match="receipt drifted"):
        subject.build_prepared_offline_semantic_tensor(
            claim_text="Alice claim.",
            prepared_corpus=forged_prepared,
            minilm_backend=minilm,
            ner_backend=ner,
            nli_backend=nli,
            allow_synthetic_backends=True,
        )

    forged_query = dict(result.receipt)
    forged_query["design_sha256"] = "0" * 64
    forged_body = dict(forged_query)
    forged_body.pop("semantic_receipt_sha256")
    forged_query["semantic_receipt_sha256"] = subject.stable_hash(forged_body)
    with pytest.raises(subject.FeverousSemanticTensorError, match="receipt drifted"):
        subject.verify_semantic_receipt(forged_query)


def test_receipts_bind_assets_and_self_hashes(built_twice) -> None:
    prepared, result, _second, _minilm, _ner, _nli = built_twice
    assert subject.verify_prepared_semantic_corpus(prepared) == (
        prepared.preparation_receipt_sha256
    )
    assert subject.verify_semantic_receipt(result.receipt) == result.receipt[
        "semantic_receipt_sha256"
    ]
    assert prepared.receipt["MiniLM_asset_sha256"] == (
        "921d9b1945581130e03c53f448092c3de3b30714431c6cac9b3b32c2ec10abad"
    )
    assert prepared.receipt["NER_asset_sha256"] == (
        "b70ab3da9d01f0bc61650ddd8f81d27fdf01e434a1d67a0b378e226bd6b3b5c5"
    )
    assert result.receipt["NLI_asset_sha256"] == (
        "d64f4403e7603ea71e622e7e7124eae466cbf67bf4c758979b54c4ccf9bb5fe8"
    )


def test_multiline_schema_and_section_sidecar_fail_closed() -> None:
    with pytest.raises(subject.FeverousSemanticTensorError, match="multiline schema"):
        subject.SemanticCorpusUnit(
            corpus_ordinal=0,
            linearized_text="TARGET: one line only",
            unit_type="sentence",
            page_key="p",
            official_order=0,
        )
    with pytest.raises(subject.FeverousSemanticTensorError, match="SECTION_PATH"):
        subject.SemanticCorpusUnit(
            corpus_ordinal=0,
            linearized_text=_sentence_text(
                target="target", title="Page", section=("Section",)
            ),
            unit_type="sentence",
            page_key="p",
            official_order=0,
            section_path=(),
        )


def test_synthetic_backend_is_rejected_without_explicit_test_scope() -> None:
    with pytest.raises(subject.FeverousSemanticTensorError, match="synthetic backend"):
        subject.prepare_semantic_corpus(
            corpus_units=_corpus(),
            minilm_backend=FakeMiniLM(),
            ner_backend=FakeNER(),
        )


def test_public_formal_api_cannot_represent_outcomes_or_comparator_inputs() -> None:
    parameters = set(
        inspect.signature(
            subject.build_prepared_offline_semantic_tensor
        ).parameters
    )
    assert parameters == {
        "claim_text",
        "prepared_corpus",
        "minilm_backend",
        "ner_backend",
        "nli_backend",
        "allow_synthetic_backends",
    }
    forbidden = ("label", "family", "gold", "evidence", "hippo", "raw")
    assert not any(
        token in name.casefold() for token in forbidden for name in parameters
    )


def test_minilm_quantizer_reuses_python_ties_to_even_scale_one_million() -> None:
    left = np.zeros(384, dtype=np.float32)
    left[0] = 1.0
    half_even_down = np.zeros(384, dtype=np.float32)
    half_even_down[0] = np.float32(0.0000005)
    half_even_down[1] = np.float32(
        math.sqrt(1.0 - float(half_even_down[0]) ** 2)
    )
    half_even_up = np.zeros(384, dtype=np.float32)
    half_even_up[0] = np.float32(0.0000015)
    half_even_up[1] = np.float32(
        math.sqrt(1.0 - float(half_even_up[0]) ** 2)
    )
    assert subject.quantized_minilm_similarity(left, half_even_down) == 0
    assert subject.quantized_minilm_similarity(left, half_even_up) == 2


def test_vectorized_products_are_bit_exact_to_frozen_scalar_quantizer() -> None:
    rng = np.random.default_rng(20260719)
    query = rng.normal(size=384).astype(np.float32)
    query /= np.linalg.norm(query)
    corpus = rng.normal(size=(97, 384)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    expected = tuple(
        subject.quantized_minilm_similarity(query, row) for row in corpus
    )
    assert subject._quantized_vector(query, corpus) == expected


@pytest.mark.parametrize("seed", (7, 20260719, 2**31 - 1))
def test_batched_randomized_quantization_is_integer_exact_for_every_cell(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    queries = rng.normal(size=(7, 384)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    corpus = rng.normal(size=(257, 384)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    expected = tuple(
        tuple(
            subject.quantized_minilm_similarity(query, row)
            for row in corpus
        )
        for query in queries
    )
    assert subject._quantized_matrix(queries, corpus) == expected


def test_adversarial_boundaries_cancellation_and_extremes_fall_back_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = np.zeros((4, 384), dtype=np.float32)
    queries[0, 0] = np.float32(1.0)
    queries[1, :4] = np.asarray((1.0, 1.0, -1.0, -1.0), dtype=np.float32)
    queries[2, 0] = np.finfo(np.float32).max
    queries[3, 0] = np.nextafter(
        np.float32(0.0), np.float32(1.0), dtype=np.float32
    )

    rows: list[np.ndarray] = []
    for integer in range(-4, 5):
        boundary = np.float32((integer + 0.5) / 1_000_000)
        for direction in (
            np.float32(-np.inf),
            np.float32(np.inf),
        ):
            row = np.zeros(384, dtype=np.float32)
            row[0] = np.nextafter(boundary, direction, dtype=np.float32)
            rows.append(row)
    cancellation = np.zeros(384, dtype=np.float32)
    cancellation[:4] = np.asarray(
        (1.0, -1.0, 2.0**-23, -(2.0**-23)), dtype=np.float32
    )
    rows.append(cancellation)
    maximum = np.zeros(384, dtype=np.float32)
    maximum[0] = np.finfo(np.float32).max
    rows.append(maximum)
    signed_zero = np.zeros(384, dtype=np.float32)
    signed_zero[::2] = np.float32(-0.0)
    rows.append(signed_zero)
    corpus = np.stack(rows)

    expected = tuple(
        tuple(
            subject.quantized_minilm_similarity(query, row)
            for row in corpus
        )
        for query in queries
    )
    fallback_count = 0
    original = subject._exact_quantized_similarity

    def counted(left: np.ndarray, right: np.ndarray) -> int:
        nonlocal fallback_count
        fallback_count += 1
        return original(left, right)

    monkeypatch.setattr(subject, "_exact_quantized_similarity", counted)
    observed = subject._quantized_matrix(queries, corpus)

    assert fallback_count > 0
    assert observed == expected


def test_quantization_chunk_boundaries_preserve_every_scalar_integer() -> None:
    rng = np.random.default_rng(1024)
    queries = rng.normal(size=(2, 384)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    corpus = rng.normal(size=(2051, 384)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    for ordinal in (1023, 1024, 2047, 2048):
        corpus[ordinal] = np.float32(0.0)
        corpus[ordinal, 0] = np.float32((ordinal % 3 + 0.5) / 1_000_000)
    expected = tuple(
        tuple(
            subject.quantized_minilm_similarity(query, row)
            for row in corpus
        )
        for query in queries
    )
    assert subject._quantized_matrix(queries, corpus) == expected
