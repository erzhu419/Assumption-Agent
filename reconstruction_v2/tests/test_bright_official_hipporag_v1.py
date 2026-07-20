from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from replication_runtime.bright_official_hipporag_v1 import contract
from replication_runtime.bright_official_hipporag_v1 import worker


def _documents():
    return [
        {"ordinal": index, "content": f"synthetic document {index} unique"}
        for index in range(contract.CANDIDATE_COUNT)
    ]


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    def vcount(self):
        return 41

    def ecount(self):
        return 17


class _Core:
    def __init__(self):
        self.documents = []
        self.graph = _Graph()

    def index(self, documents):
        self.documents = list(documents)

    def retrieve(self, queries, num_to_retrieve):
        assert len(queries) == 1
        assert num_to_retrieve == contract.CANDIDATE_COUNT
        return [_Solution(list(reversed(self.documents)), [float(i) for i in range(32)])]


def test_exact_candidate_contract_and_canonical_output() -> None:
    query, rows = contract.validate_input("synthetic query", _documents())
    assert query == "synthetic query"
    payload = worker.retrieve_with_core(core=_Core(), query=query, documents=rows)
    assert payload["top_ordinals"] == list(range(10))
    assert payload["graph_node_count"] == 41
    assert payload["graph_edge_count"] == 17
    assert contract.parse_output(contract.canonical_json_bytes(payload)) == payload


def test_candidate_shape_and_invalid_content_fail_closed() -> None:
    with pytest.raises(contract.BrightOfficialHippoRAGError, match="count"):
        contract.validate_input("query", _documents()[:-1])
    rows = _documents()
    rows[1]["ordinal"] = 0
    with pytest.raises(contract.BrightOfficialHippoRAGError, match="ordinals"):
        contract.validate_input("query", rows)
    rows = _documents()
    rows[1]["content"] = "contains\x00nul"
    with pytest.raises(contract.BrightOfficialHippoRAGError, match="content"):
        contract.validate_input("query", rows)


def test_stable_top_k_rejects_partial_or_nonfinite_official_results() -> None:
    _, rows = contract.validate_input("query", _documents())
    serialized = contract.serialize_documents(rows)
    mapping = {text: row.ordinal for text, row in zip(serialized, rows)}
    with pytest.raises(contract.BrightOfficialHippoRAGError, match="all candidates"):
        contract.stable_top_k(
            retrieved_documents=serialized[:-1],
            retrieved_scores=[0.0] * 31,
            document_to_ordinal=mapping,
        )
    scores = [0.0] * 32
    scores[0] = float("nan")
    with pytest.raises(contract.BrightOfficialHippoRAGError, match="finite"):
        contract.stable_top_k(
            retrieved_documents=serialized,
            retrieved_scores=scores,
            document_to_ordinal=mapping,
        )


def test_compatibility_backend_returns_completion_only_and_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        eos_token_id = 7

        @staticmethod
        def decode(tokens, skip_special_tokens):
            assert skip_special_tokens is True
            assert tokens.tolist() == [8, 9]
            return '{"named_entities":["Alice Carter"]}'

    class _Model:
        device = torch.device("cpu")

        @staticmethod
        def generate(**kwargs):
            assert kwargs["input_ids"].tolist() == [[1, 2]]
            assert kwargs["attention_mask"].tolist() == [[1, 1]]
            assert kwargs["max_new_tokens"] == worker.OPENIE_MAX_NEW_TOKENS
            assert kwargs["do_sample"] is False
            assert kwargs["pad_token_id"] == 7
            return torch.tensor([[1, 2, 8, 9]])

    backend_type = type("TransformersLLM", (), {})
    backend = backend_type()
    backend.model = _Model()
    backend.tokenizer = _Tokenizer()
    backend.llm_config = SimpleNamespace(generate_params={"temperature": 0})
    core = SimpleNamespace(llm_model=backend)

    package = ModuleType("hipporag")
    package.__path__ = []  # type: ignore[attr-defined]
    llm_package = ModuleType("hipporag.llm")
    llm_package.__path__ = []  # type: ignore[attr-defined]
    backend_module = ModuleType("hipporag.llm.transformers_llm")
    backend_module.convert_text_chat_messages_to_input_ids = (
        lambda messages, tokenizer: torch.tensor([[1, 2]])
    )
    monkeypatch.setitem(sys.modules, "hipporag", package)
    monkeypatch.setitem(sys.modules, "hipporag.llm", llm_package)
    monkeypatch.setitem(
        sys.modules, "hipporag.llm.transformers_llm", backend_module
    )

    worker._install_completion_only_backend(core)
    response, metadata, cache_hit = backend.infer(
        [{"role": "user", "content": "Alice works at Orion Laboratory."}]
    )
    assert response == '{"named_entities":["Alice Carter"]}'
    assert metadata == {
        "completion_tokens": 2,
        "finish_reason": "stop",
        "prompt_tokens": 2,
    }
    assert cache_hit is False
    with pytest.raises(contract.BrightOfficialHippoRAGError, match="budget"):
        backend.infer([], max_tokens=95)
