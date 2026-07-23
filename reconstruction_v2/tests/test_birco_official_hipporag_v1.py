from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from replication_runtime.birco_official_hipporag_v1 import contract
from replication_runtime.birco_official_hipporag_v1 import worker


OBJECTIVE = "frozen task objective"
QUERY = "original private query"


def _documents(count: int) -> list[dict[str, object]]:
    return [
        {
            "ordinal": ordinal,
            "text": f"private synthetic candidate {ordinal} sentinel-{ordinal}",
        }
        for ordinal in range(count)
    ]


def _projection_hash(
    documents: list[dict[str, object]],
    *,
    objective: str = OBJECTIVE,
    query: str = QUERY,
) -> str:
    return contract.common_projection_sha256(
        objective=objective, query=query, documents=documents
    )


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    def vcount(self) -> int:
        return 41

    def ecount(self) -> int:
        return 17


class _Core:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.graph = _Graph()
        self.index_call_count = 0
        self.retrieve_call_count = 0

    def index(self, documents: list[str]) -> None:
        self.index_call_count += 1
        self.documents = list(documents)

    def retrieve(
        self, queries: list[str], num_to_retrieve: int
    ) -> list[_Solution]:
        self.retrieve_call_count += 1
        assert queries == [
            contract.core_query_text(objective=OBJECTIVE, query=QUERY)
        ]
        assert num_to_retrieve == len(self.documents)
        return [
            _Solution(
                list(reversed(self.documents)),
                [float(index) for index in range(len(self.documents))],
            )
        ]

    def rag_qa(self, _queries: object) -> None:
        raise AssertionError("the retrieve-only adapter must not invoke QA")


@pytest.mark.parametrize("candidate_count", [10, 37, 256])
def test_dynamic_pool_returns_complete_canonical_permutation(
    candidate_count: int,
) -> None:
    documents = _documents(candidate_count)
    projection_hash = _projection_hash(documents)
    work_id, objective, query, rows, validated_hash = contract.validate_input(
        "opaque-work-001", OBJECTIVE, QUERY, documents, projection_hash
    )
    core = _Core()
    payload = worker.retrieve_with_core(
        core=core,
        work_id=work_id,
        objective=objective,
        query=query,
        documents=rows,
        common_projection_sha256=validated_hash,
    )

    assert core.index_call_count == 1
    assert core.retrieve_call_count == 1
    assert payload == {
        "candidate_count": candidate_count,
        "common_projection_sha256": projection_hash,
        "graph_edge_count": 17,
        "graph_node_count": 41,
        "rank_ordinals": list(range(candidate_count)),
        "schema": contract.OUTPUT_SCHEMA,
        "work_id": "opaque-work-001",
    }
    raw = contract.canonical_json_bytes(payload)
    assert contract.parse_output(raw) == payload
    assert b"original private query" not in raw
    assert b"private synthetic candidate" not in raw


def test_score_ties_are_broken_by_source_ordinal() -> None:
    documents = _documents(contract.MIN_CANDIDATE_COUNT)
    _, _, _, rows, _ = contract.validate_input(
        "opaque",
        OBJECTIVE,
        QUERY,
        documents,
        _projection_hash(documents),
    )
    serialized = contract.serialize_documents(rows)
    mapping = {
        document: row.ordinal for document, row in zip(serialized, rows)
    }
    assert contract.stable_permutation(
        retrieved_documents=reversed(serialized),
        retrieved_scores=[1.0] * len(serialized),
        document_to_ordinal=mapping,
    ) == tuple(range(len(serialized)))


def test_malformed_duplicate_or_forbidden_input_fails_closed() -> None:
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="bounds"):
        contract.validate_input("work", OBJECTIVE, QUERY, _documents(9), "0" * 64)
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="bounds"):
        contract.validate_input("work", OBJECTIVE, QUERY, _documents(257), "0" * 64)
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="work_id"):
        contract.validate_input(" ", OBJECTIVE, QUERY, _documents(10), "0" * 64)
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="query"):
        contract.validate_input(
            "work", OBJECTIVE, "contains\x00nul", _documents(10), "0" * 64
        )

    rows = _documents(10)
    rows[1] = dict(rows[0])
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="ordinals"):
        contract.validate_input("work", OBJECTIVE, QUERY, rows, "0" * 64)

    rows = _documents(10)
    rows[0]["qrel"] = 2
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="shape"):
        contract.validate_input("work", OBJECTIVE, QUERY, rows, "0" * 64)
    rows = _documents(10)
    rows[0]["candidate_id"] = "source-id"
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="shape"):
        contract.validate_input("work", OBJECTIVE, QUERY, rows, "0" * 64)

    duplicate = contract.CandidateDocument(ordinal=0, text="duplicate")
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="duplicated"):
        contract.serialize_documents((duplicate, duplicate))


def test_common_projection_hash_binds_objective_query_and_documents() -> None:
    documents = _documents(10)
    projection_hash = _projection_hash(documents)
    assert projection_hash == contract.common_projection_sha256(
        objective=OBJECTIVE, query=QUERY, documents=documents
    )
    assert contract.core_query_text(objective=OBJECTIVE, query=QUERY) == (
        '{"objective":"frozen task objective","query":"original private query"}'
    )
    changed = _documents(10)
    changed[0]["text"] = "changed common projection"
    changed_values = (
        ("changed objective", QUERY, documents),
        (OBJECTIVE, "changed query", documents),
        (OBJECTIVE, QUERY, changed),
    )
    for objective, query, candidate_documents in changed_values:
        with pytest.raises(
            contract.BircoOfficialHippoRAGError, match="mismatched"
        ):
            contract.validate_input(
                "work",
                objective,
                query,
                candidate_documents,
                projection_hash,
            )


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -float("inf")])
def test_official_result_omissions_duplicates_and_nonfinite_fail_closed(
    invalid_score: float,
) -> None:
    documents = _documents(10)
    _, _, _, rows, _ = contract.validate_input(
        "work", OBJECTIVE, QUERY, documents, _projection_hash(documents)
    )
    serialized = contract.serialize_documents(rows)
    mapping = {
        document: row.ordinal for document, row in zip(serialized, rows)
    }

    with pytest.raises(contract.BircoOfficialHippoRAGError, match="all candidates"):
        contract.stable_permutation(
            retrieved_documents=serialized[:-1],
            retrieved_scores=[0.0] * 9,
            document_to_ordinal=mapping,
        )

    duplicated_documents = list(serialized)
    duplicated_documents[-1] = duplicated_documents[0]
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="duplicate"):
        contract.stable_permutation(
            retrieved_documents=duplicated_documents,
            retrieved_scores=[0.0] * 10,
            document_to_ordinal=mapping,
        )

    scores = [0.0] * 10
    scores[3] = invalid_score
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="finite"):
        contract.stable_permutation(
            retrieved_documents=serialized,
            retrieved_scores=scores,
            document_to_ordinal=mapping,
        )


def test_input_envelope_excludes_qrels_ids_family_and_block(tmp_path) -> None:
    documents = _documents(10)
    projection_hash = _projection_hash(
        documents, objective=OBJECTIVE, query="query"
    )
    payload = {
        "common_projection_sha256": projection_hash,
        "documents": documents,
        "objective": OBJECTIVE,
        "query": "query",
        "schema": contract.INPUT_SCHEMA,
        "work_id": "opaque-work",
    }
    path = tmp_path / "input.json"
    path.write_bytes(contract.canonical_json_bytes(payload))
    work_id, objective, query, rows, validated_hash = worker._load_input(path)
    assert (work_id, objective, query, len(rows), validated_hash) == (
        "opaque-work",
        OBJECTIVE,
        "query",
        10,
        projection_hash,
    )

    for forbidden in ("qrel", "query_id", "family", "block"):
        forbidden_payload = {**payload, forbidden: "forbidden"}
        path.write_bytes(contract.canonical_json_bytes(forbidden_payload))
        with pytest.raises(
            contract.BircoOfficialHippoRAGError, match="envelope"
        ):
            worker._load_input(path)


def test_output_parser_requires_exact_canonical_complete_output() -> None:
    projection_hash = "a" * 64
    payload = contract.output_payload(
        work_id="opaque-work",
        common_projection_sha256=projection_hash,
        candidate_count=10,
        rank_ordinals=range(10),
        graph_nodes=3,
        graph_edges=2,
    )
    raw = contract.canonical_json_bytes(payload)
    assert contract.parse_output(raw) == payload

    with pytest.raises(contract.BircoOfficialHippoRAGError, match="canonical"):
        contract.parse_output(raw.rstrip(b"\n"))
    extra = {**payload, "query": "must-not-leak"}
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="shape"):
        contract.parse_output(contract.canonical_json_bytes(extra))
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="permutation"):
        contract.output_payload(
            work_id="opaque-work",
            common_projection_sha256=projection_hash,
            candidate_count=10,
            rank_ordinals=[0] * 10,
            graph_nodes=3,
            graph_edges=2,
        )


def test_model_alias_validation_accepts_short_directory_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    llm_target = tmp_path / "llm-target"
    embedding_target = tmp_path / "embedding-target"
    llm_target.mkdir()
    embedding_target.mkdir()
    (tmp_path / "smollm2").symlink_to(llm_target, target_is_directory=True)
    (tmp_path / "minilm").symlink_to(
        embedding_target, target_is_directory=True
    )
    monkeypatch.chdir(tmp_path)

    assert worker._validate_model_alias(
        "smollm2", label="LLM model"
    ) == "smollm2"
    assert worker._validate_model_alias(
        "minilm", label="embedding model"
    ) == "minilm"


def test_model_alias_validation_rejects_paths_traversal_and_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    nested = working_directory / "nested"
    nested.mkdir()
    (nested / "model").mkdir()
    (working_directory / "nested\\model").mkdir()
    (working_directory / "model..alias").mkdir()
    overlong = "a" * (worker.MAX_MODEL_ALIAS_CHARACTERS + 1)
    (working_directory / overlong).mkdir()
    (working_directory / "not-a-directory").write_text(
        "not a model directory", encoding="ascii"
    )
    monkeypatch.chdir(working_directory)

    invalid_aliases = (
        str(target),
        "..",
        "../target",
        "nested/model",
        "nested\\model",
        "model..alias",
        overlong,
        "missing",
        "not-a-directory",
    )
    for alias in invalid_aliases:
        with pytest.raises(
            contract.BircoOfficialHippoRAGError, match="alias"
        ) as exc_info:
            worker._validate_model_alias(alias, label="LLM model")
        assert alias not in str(exc_info.value)


def test_main_validates_aliases_before_creating_index_or_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    input_path = tmp_path / "input.json"
    documents = _documents(10)
    input_path.write_bytes(
        contract.canonical_json_bytes(
            {
                "common_projection_sha256": _projection_hash(documents),
                "documents": documents,
                "objective": OBJECTIVE,
                "query": QUERY,
                "schema": contract.INPUT_SCHEMA,
                "work_id": "opaque-alias-check",
            }
        )
    )
    target = tmp_path / "model-target"
    target.mkdir()
    (tmp_path / "smollm2").symlink_to(target, target_is_directory=True)
    (tmp_path / "minilm").symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    def _must_not_build_core(**_kwargs: object) -> object:
        raise AssertionError("core must not load for an invalid model alias")

    monkeypatch.setattr(worker, "_build_core", _must_not_build_core)
    invocations = (
        (str(target), "minilm", tmp_path / "absolute-path-index"),
        ("smollm2", "missing", tmp_path / "missing-alias-index"),
    )
    for llm_model, embedding_model, index_root in invocations:
        with pytest.raises(
            contract.BircoOfficialHippoRAGError, match="alias"
        ) as exc_info:
            worker.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(tmp_path / "unused-output.json"),
                    "--index-root",
                    str(index_root),
                    "--llm-model",
                    llm_model,
                    "--embedding-model",
                    embedding_model,
                ]
            )
        assert QUERY not in str(exc_info.value)
        assert not index_root.exists()


def test_main_requires_fresh_index_root_and_writes_no_source_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    index_root = tmp_path / "one-query-index"
    documents = _documents(10)
    projection_hash = _projection_hash(documents)
    input_path.write_bytes(
        contract.canonical_json_bytes(
            {
                "common_projection_sha256": projection_hash,
                "documents": documents,
                "objective": OBJECTIVE,
                "query": QUERY,
                "schema": contract.INPUT_SCHEMA,
                "work_id": "opaque-main-work",
            }
        )
    )
    core = _Core()
    llm_target = tmp_path / "llm-target"
    embedding_target = tmp_path / "embedding-target"
    llm_target.mkdir()
    embedding_target.mkdir()
    (tmp_path / "smollm2").symlink_to(llm_target, target_is_directory=True)
    (tmp_path / "minilm").symlink_to(
        embedding_target, target_is_directory=True
    )
    monkeypatch.chdir(tmp_path)

    def _mock_build_core(**kwargs: object) -> _Core:
        assert kwargs["save_dir"] == index_root
        assert kwargs["candidate_count"] == 10
        assert kwargs["llm_model"] == "smollm2"
        assert kwargs["embedding_model"] == "minilm"
        return core

    monkeypatch.setattr(worker, "_build_core", _mock_build_core)
    assert worker.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            "smollm2",
            "--embedding-model",
            "minilm",
        ]
    ) == 0
    result = output_path.read_bytes()
    parsed = contract.parse_output(result)
    assert parsed["rank_ordinals"] == list(range(10))
    assert parsed["common_projection_sha256"] == projection_hash
    assert b"original private query" not in result
    assert b"private synthetic candidate" not in result
    assert "original private query" not in capsys.readouterr().out

    with pytest.raises(contract.BircoOfficialHippoRAGError, match="already exists"):
        worker.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "second-output.json"),
                "--index-root",
                str(index_root),
                "--llm-model",
                "smollm2",
                "--embedding-model",
                "minilm",
            ]
        )


def test_compatibility_backend_returns_completion_only_and_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        eos_token_id = 7

        @staticmethod
        def decode(tokens: torch.Tensor, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            assert tokens.tolist() == [8, 9]
            return '{"named_entities":["Alice Carter"]}'

    class _Model:
        device = torch.device("cpu")

        @staticmethod
        def generate(**kwargs: object) -> torch.Tensor:
            assert kwargs["input_ids"].tolist() == [[1, 2]]  # type: ignore[union-attr]
            assert kwargs["attention_mask"].tolist() == [[1, 1]]  # type: ignore[union-attr]
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
    with pytest.raises(contract.BircoOfficialHippoRAGError, match="budget"):
        backend.infer([], max_tokens=95)


def test_frozen_contract_has_no_source_label_or_identity_fields() -> None:
    assert contract.DOCUMENT_KEYS == frozenset({"ordinal", "text"})
    assert contract.INPUT_KEYS == frozenset(
        {
            "common_projection_sha256",
            "documents",
            "objective",
            "query",
            "schema",
            "work_id",
        }
    )
    assert not {
        "qrel",
        "qrel_value",
        "candidate_id",
        "query_id",
        "family",
        "block",
    } & (contract.DOCUMENT_KEYS | contract.INPUT_KEYS | contract.OUTPUT_KEYS)
    assert contract.OFFICIAL_HIPPORAG_COMMIT == (
        "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
    )
