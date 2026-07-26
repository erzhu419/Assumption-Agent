from __future__ import annotations

import json

import numpy as np

from assumption_agent.benchmarks.averitec_p1_coordinate_worker_v1 import (
    coordinate_output,
    private_input_payload,
    validate_output,
)
from assumption_agent.benchmarks.averitec_p1_typed_core_v1 import (
    QUERY_VARIANT_IDS,
)


def test_coordinate_worker_emits_opaque_quantized_rows_without_text() -> None:
    documents = [f"PRIVATE DOCUMENT {index}" for index in range(6)]
    queries = [
        ("a" * 64, "PRIVATE QUERY A"),
        ("b" * 64, "PRIVATE QUERY B"),
    ]
    private_input = private_input_payload(
        documents=documents,
        queries=queries,
    )

    def encode(texts):
        rows = []
        for index, _text in enumerate(texts):
            vector = np.zeros(4, dtype=np.float64)
            vector[index % 4] = 1.0
            rows.append(vector)
        return np.stack(rows)

    output = coordinate_output(
        private_input=private_input,
        encode=encode,
        runtime_receipt={
            "cuda_allocate_and_synchronize": True,
            "cuda_device_count": 1,
            "deterministic_algorithms_enabled": True,
            "cuda_logical_device": 0,
            "minilm_all_parameters_cuda0": True,
            "minilm_parameter_count": 1,
            "native_and_torch_thread_count": 1,
            "torch_manual_seed": 0,
        },
    )
    assert validate_output(output, expected_input=private_input) == output
    assert output["document_count"] == 6
    assert output["query_count"] == 2
    assert len(output["rows"]) == 2
    for row in output["rows"]:
        assert tuple(row["variant_scores"]) == QUERY_VARIANT_IDS
        assert all(
            len(values) == 6
            and all(0 <= value <= 1_000_000 for value in values)
            for values in row["variant_scores"].values()
        )
    rendered = json.dumps(output, sort_keys=True)
    assert "PRIVATE DOCUMENT" not in rendered
    assert "PRIVATE QUERY" not in rendered
