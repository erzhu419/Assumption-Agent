from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from replication_runtime.quac_p1_official_v1 import contract
from replication_runtime.quac_p1_official_v1 import worker


def _opaque(prefix: int) -> str:
    return f"{prefix:064x}"


def _input(
    *,
    corpus_count: int = 7,
    query_count: int = 3,
    duplicate_text: bool = False,
) -> dict[str, object]:
    units = [
        {
            "unit_id": _opaque(index + 100),
            "text": (
                "identical raw evidence text"
                if duplicate_text
                else f"synthetic evidence window {index}"
            ),
        }
        for index in range(corpus_count)
    ]
    queries = [
        {
            "query_id": _opaque(index + 1_000),
            "text": f"[CURRENT] synthetic question {index}",
        }
        for index in range(query_count)
    ]
    return contract.build_input(
        block_id=_opaque(9_999),
        units=units,
        queries=queries,
    )


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    @staticmethod
    def vcount() -> int:
        return 29

    @staticmethod
    def ecount() -> int:
        return 43


class _Core:
    def __init__(self) -> None:
        self.index_calls: list[list[str]] = []
        self.retrieve_calls: list[tuple[list[str], int]] = []
        self.graph = _Graph()

    def index(self, documents) -> None:
        self.index_calls.append(list(documents))

    def retrieve(self, queries, num_to_retrieve):
        eager_queries = list(queries)
        self.retrieve_calls.append((eager_queries, num_to_retrieve))
        documents = self.index_calls[0]
        # Every result is a complete permutation.  All scores tie, forcing
        # the adapter to use opaque unit ID rather than official return order.
        return [
            _Solution(
                docs=list(reversed(documents)),
                doc_scores=[0.25] * len(documents),
            )
            for _query in eager_queries
        ]


@pytest.mark.parametrize("corpus_count", [5, 7, 19])
def test_variable_block_corpus_builds_once_and_retrieves_all_queries_eagerly(
    corpus_count: int,
) -> None:
    private_input = _input(corpus_count=corpus_count, query_count=4)
    core = _Core()
    result = worker.retrieve_block_with_core(
        core=core,
        private_input=private_input,
    )

    block = contract.validate_input(private_input)
    assert len(core.index_calls) == 1
    assert len(core.index_calls[0]) == corpus_count
    assert core.retrieve_calls == [
        ([row.text for row in block.queries], corpus_count)
    ]
    assert result["runtime"]["index_call_count"] == 1
    assert result["runtime"]["retrieve_call_count"] == 1
    assert result["runtime"]["complete_ranking_count"] == 4
    assert result["runtime"]["graph_node_count"] == 29
    assert result["runtime"]["graph_edge_count"] == 43
    expected_top5 = sorted(row.unit_id for row in block.units)[:5]
    assert all(
        row["top5_unit_ids"] == expected_top5
        for row in result["rows"]
    )
    assert (
        contract.validate_output(result, expected_input=private_input)
        == result
    )


def test_duplicate_raw_text_has_unique_canonical_ascii_document_addressing() -> None:
    private_input = _input(
        corpus_count=6,
        query_count=2,
        duplicate_text=True,
    )
    block = contract.validate_input(private_input)
    documents = contract.serialize_corpus(block.units)
    assert len(set(documents)) == 6
    assert all(document.isascii() for document in documents)
    assert all("QUAC_EVIDENCE_UNIT_" in document for document in documents)

    core = _Core()
    result = worker.retrieve_block_with_core(
        core=core,
        private_input=private_input,
    )
    assert len(result["rows"]) == 2


def test_complete_permutation_and_score_then_unit_id_tie_break() -> None:
    private_input = _input(corpus_count=6, query_count=1)
    block = contract.validate_input(private_input)
    documents = contract.serialize_corpus(block.units)
    mapping = {
        document: row.unit_id
        for document, row in zip(documents, block.units)
    }
    scores = [0.0, 3.0, 3.0, -1.0, 2.0, 1.0]
    ranking = contract.stable_complete_ranking(
        retrieved_documents=list(reversed(documents)),
        retrieved_scores=list(reversed(scores)),
        document_to_unit_id=mapping,
    )
    expected = tuple(
        unit_id
        for _score, unit_id in sorted(
            zip(scores, (row.unit_id for row in block.units)),
            key=lambda row: (-row[0], row[1]),
        )
    )
    assert ranking == expected

    with pytest.raises(
        contract.QuacP1OfficialHippoRAGError,
        match="complete corpus permutation",
    ):
        contract.stable_complete_ranking(
            retrieved_documents=documents[:-1],
            retrieved_scores=[0.0] * (len(documents) - 1),
            document_to_unit_id=mapping,
        )
    invalid_scores = [0.0] * len(documents)
    invalid_scores[0] = float("nan")
    with pytest.raises(
        contract.QuacP1OfficialHippoRAGError,
        match="nonfinite",
    ):
        contract.stable_complete_ranking(
            retrieved_documents=documents,
            retrieved_scores=invalid_scores,
            document_to_unit_id=mapping,
        )


@pytest.mark.parametrize(
    ("scope", "forbidden"),
    [
        ("root", "family"),
        ("root", "split"),
        ("root", "qrel"),
        ("unit", "answer"),
        ("unit", "score"),
        ("query", "family"),
        ("query", "answer"),
        ("query", "candidate_ids"),
    ],
)
def test_strict_input_rejects_every_forbidden_or_candidate_field(
    scope: str,
    forbidden: str,
) -> None:
    private_input = _input()
    tampered = deepcopy(private_input)
    if scope == "root":
        tampered[forbidden] = "forbidden"
    elif scope == "unit":
        tampered["corpus"][0][forbidden] = "forbidden"
    else:
        tampered["queries"][0][forbidden] = "forbidden"
    with pytest.raises(
        contract.QuacP1OfficialHippoRAGError,
        match="drifted",
    ):
        contract.validate_input(tampered)


def test_input_and_output_commitments_fail_closed_on_tamper() -> None:
    private_input = _input()
    core = _Core()
    result = worker.retrieve_block_with_core(
        core=core,
        private_input=private_input,
    )

    input_tamper = deepcopy(private_input)
    input_tamper["corpus"][0]["text"] += " changed"
    with pytest.raises(
        contract.QuacP1OfficialHippoRAGError,
        match="commitment",
    ):
        contract.validate_input(input_tamper)

    output_tamper = deepcopy(result)
    output_tamper["rows"][0]["top5_unit_ids"].reverse()
    with pytest.raises(
        contract.QuacP1OfficialHippoRAGError,
        match="commitment",
    ):
        contract.validate_output(
            output_tamper,
            expected_input=private_input,
        )


def test_real_core_constructor_reuses_qualified_maud_compatibility_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    llm = tmp_path / "llm"
    embedding = tmp_path / "embedding"
    llm.mkdir()
    embedding.mkdir()
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_build_core(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker.qualified_maud, "_build_core", fake_build_core)
    result = worker.build_official_core(
        index_root=tmp_path / "index",
        llm_model_alias="llm",
        embedding_model_alias="embedding",
        corpus_count=11,
    )
    assert result is sentinel
    assert calls == [
        {
            "save_dir": tmp_path / "index",
            "llm_alias": "llm",
            "embedding_alias": "embedding",
            "passage_count": 11,
        }
    ]


def test_query_text_is_passed_exactly_once_without_answer_or_graph_surface() -> None:
    private_input = _input(corpus_count=8, query_count=5)
    block = contract.validate_input(private_input)
    core = _Core()
    worker.retrieve_block_with_core(
        core=core,
        private_input=private_input,
    )
    assert core.retrieve_calls == [
        ([row.text for row in block.queries], len(block.units))
    ]
    assert set(private_input) == {
        "block_id",
        "corpus",
        "corpus_sha256",
        "queries",
        "queries_sha256",
        "schema",
    }
    assert all(set(row) == {"unit_id", "text"} for row in private_input["corpus"])
    assert all(set(row) == {"query_id", "text"} for row in private_input["queries"])
