from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from replication_runtime.ebmnlp_p1_official_v1 import contract
from replication_runtime.ebmnlp_p1_official_v1 import worker


def _documents(count: int = 7) -> list[dict[str, object]]:
    return [
        {
            "ordinal": ordinal,
            "text": f"Evidence window {ordinal} with unique trial content.",
            "window_id": f"W:{ordinal * 24:08d}:{ordinal * 24 + 48:08d}",
        }
        for ordinal in range(count)
    ]


def _queries() -> list[dict[str, object]]:
    return [
        {
            "ordinal": ordinal,
            "role": role,
            "text": contract.ROLE_QUERIES[role],
            "work_id": hashlib.sha256(role.encode("ascii")).hexdigest(),
        }
        for ordinal, role in enumerate(contract.ROLE_ORDER)
    ]


def _payload(count: int = 7) -> dict[str, object]:
    return contract.input_payload(
        abstract_work_id="opaque-" + "a" * 64,
        documents=_documents(count),
        queries=_queries(),
    )


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    def vcount(self) -> int:
        return 11

    def ecount(self) -> int:
        return 17


class _Core:
    def __init__(self) -> None:
        self.graph = _Graph()
        self.indexed: list[str] = []
        self.queries: list[str] = []

    def index(self, documents: list[str]) -> None:
        self.indexed = list(documents)

    def retrieve(
        self, queries: list[str], *, num_to_retrieve: int
    ) -> list[_Solution]:
        self.queries = list(queries)
        assert num_to_retrieve == len(self.indexed)
        return [
            _Solution(
                docs=list(reversed(self.indexed)),
                doc_scores=[
                    float(index)
                    for index in range(len(self.indexed))
                ],
            )
            for _query in queries
        ]


def test_complete_three_query_contract_round_trip() -> None:
    payload = _payload()
    abstract_id, corpus_hash, documents, queries = contract.validate_input(
        payload
    )
    assert abstract_id.startswith("opaque-")
    assert corpus_hash == payload["corpus_sha256"]
    assert len(documents) == 7
    assert tuple(row.role for row in queries) == contract.ROLE_ORDER

    core = _Core()
    result = contract.retrieve_abstract_with_core(
        core=core, payload=payload
    )
    assert len(core.indexed) == 7
    assert core.queries == [
        contract.ROLE_QUERIES[role] for role in contract.ROLE_ORDER
    ]
    assert result["graph_node_count"] == 11
    assert result["graph_edge_count"] == 17
    for row in result["rows"]:
        assert sorted(row["rank_window_ordinals"]) == list(range(7))
    raw = contract.canonical_json_bytes(result)
    assert contract.parse_output(raw) == result


def test_one_document_is_total_and_not_padded() -> None:
    result = contract.retrieve_abstract_with_core(
        core=_Core(), payload=_payload(1)
    )
    assert all(
        row["rank_window_ordinals"] == [0] for row in result["rows"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["queries"][0].update(
            {"text": "participants keywords"}
        ),
        lambda value: value["documents"][0].update({"ordinal": 3}),
        lambda value: value.update({"corpus_sha256": "0" * 64}),
        lambda value: value["queries"].pop(),
    ],
)
def test_contract_rejects_drift(mutation) -> None:
    value = json.loads(json.dumps(_payload()))
    mutation(value)
    with pytest.raises(contract.EBMNLPOfficialHippoRAGError):
        contract.validate_input(value)


def test_worker_load_requires_canonical_private_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(contract.canonical_json_bytes(_payload()))
    assert worker._load_input(path) == _payload()
    path.write_text(json.dumps(_payload(), indent=2), encoding="ascii")
    with pytest.raises(contract.EBMNLPOfficialHippoRAGError):
        worker._load_input(path)


def test_model_alias_is_single_local_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "minilm").mkdir()
    assert worker._validate_model_alias("minilm", "model") == "minilm"
    for value in ("../minilm", "/tmp/minilm", "a/b", "missing"):
        with pytest.raises(contract.EBMNLPOfficialHippoRAGError):
            worker._validate_model_alias(value, "model")


class _Device:
    def __init__(self, kind: str, index: int | None) -> None:
        self.type = kind
        self.index = index


class _Parameter:
    def __init__(self, kind: str = "cuda", index: int = 0) -> None:
        self.device = _Device(kind, index)
        self.dtype = "torch.float32"

    def numel(self) -> int:
        return 16


class _Module:
    def __init__(
        self, *, kind: str = "cuda", device_map: object = None
    ) -> None:
        self._parameters = (
            ("weight", _Parameter(kind)),
            ("bias", _Parameter(kind)),
        )
        if device_map is not None:
            self.hf_device_map = device_map

    def named_parameters(self):
        return iter(self._parameters)


class _Sentinel:
    def __init__(self) -> None:
        self.value = 1.0
        self.device = _Device("cuda", 0)

    def add_(self, value: float):
        self.value += value
        return self

    def item(self) -> float:
        return self.value


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def synchronize(_device: int) -> None:
        return None

    @staticmethod
    def get_device_name(_device: int) -> str:
        return "synthetic CUDA"

    @staticmethod
    def memory_allocated(_device: int) -> int:
        return 4096


class _Torch:
    cuda = _Cuda()
    float32 = "torch.float32"

    @staticmethod
    def ones(_shape, *, dtype, device):
        assert dtype == "torch.float32"
        assert device == "cuda:0"
        return _Sentinel()


def test_worker_cuda_attestation_requires_both_models_on_single_cuda0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    core = SimpleNamespace(
        llm_model=SimpleNamespace(
            model=_Module(device_map={"": 0})
        ),
        embedding_model=SimpleNamespace(model=_Module()),
    )
    receipt = worker._attest_cuda_residency(
        core, torch_module=_Torch
    )
    assert receipt["physical_visible_gpu_binding"] == "1"
    assert receipt["LLM"]["parameter_device"] == "cuda:0"
    assert receipt["embedding"]["parameter_device"] == "cuda:0"

    core.embedding_model.model = _Module(kind="cpu")
    with pytest.raises(
        contract.EBMNLPOfficialHippoRAGError,
        match="cuda:0",
    ):
        worker._attest_cuda_residency(core, torch_module=_Torch)


@pytest.mark.parametrize("offload", ("cpu", "disk", "cuda:1"))
def test_worker_cuda_attestation_rejects_llm_offload(
    offload: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    core = SimpleNamespace(
        llm_model=SimpleNamespace(
            model=_Module(device_map={"layer": offload})
        ),
        embedding_model=SimpleNamespace(model=_Module()),
    )
    with pytest.raises(
        contract.EBMNLPOfficialHippoRAGError,
        match="offload",
    ):
        worker._attest_cuda_residency(core, torch_module=_Torch)
