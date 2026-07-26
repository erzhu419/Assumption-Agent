from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from replication_runtime.averitec_p1_official_v1.worker import (
    CANARY_BLOCK,
    input_payload,
    retrieve_with_core,
    validate_output,
)


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    def vcount(self):
        return 6

    def ecount(self):
        return 5


class _Core:
    graph = _Graph()

    def __init__(self):
        self.documents = []
        self.index_calls = 0
        self.retrieve_calls = 0

    def index(self, documents):
        self.index_calls += 1
        self.documents = list(documents)

    def retrieve(self, queries, num_to_retrieve):
        self.retrieve_calls += 1
        assert num_to_retrieve == 6
        return [
            _Solution(
                docs=list(self.documents),
                doc_scores=[float(6 - index) for index in range(6)],
            )
            for _query in queries
        ]


def _cuda_phase() -> dict[str, object]:
    return {
        "cuda_allocation_and_synchronize_succeeded": True,
        "logical_cuda_current_device": 0,
        "physical_visible_gpu_binding": "1",
        "torch_cuda_is_available": True,
        "visible_cuda_device_count": 1,
    }


def test_dynamic_official_worker_builds_once_and_returns_only_ordinals(
    tmp_path: Path,
) -> None:
    payload = input_payload(
        block=CANARY_BLOCK,
        articles=[
            {"idx": index, "title": f"title {index}", "body": f"body {index}"}
            for index in range(6)
        ],
        queries=[("a" * 64, "query one"), ("b" * 64, "query two")],
    )
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "frozen.index").write_bytes(b"fixture")
    core = _Core()
    output = retrieve_with_core(
        core=core,
        private_input=payload,
        index_root=index_root,
        cuda_receipt={
            "post_inference": _cuda_phase(),
            "pre_inference": _cuda_phase(),
        },
        observed_process_thread_peak=3,
    )
    assert validate_output(output, expected_input=payload) == output
    assert core.index_calls == 1
    assert core.retrieve_calls == 1
    assert [row["item_id"] for row in output["rows"]] == [
        "a" * 64,
        "b" * 64,
    ]
    assert all(
        row["top5_document_ordinals"] == [0, 1, 2, 3, 4]
        for row in output["rows"]
    )
    rendered = str(output)
    assert "title 0" not in rendered
    assert "query one" not in rendered
