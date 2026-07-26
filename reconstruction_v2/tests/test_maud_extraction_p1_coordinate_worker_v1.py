from __future__ import annotations

import hashlib
import math

import pytest

from assumption_agent.benchmarks import maud_extraction_p1_coordinate_worker_v1 as subject
from assumption_agent.benchmarks import maud_extraction_p1_typed_core_v1 as typed_core


def _contracts():
    passages = [
        {"ordinal": i, "start": i * 10, "end": i * 10 + 9, "text": f"passage {i}"}
        for i in range(5)
    ]
    queries = []
    families = (
        "definition_reference",
        "condition_obligation",
        "protection_exception_remedy",
    )
    for i in range(22):
        queries.append(
            {
                "work_id": f"{i + 1:064x}",
                "family": families[i % 3],
                "question": f"question {i}",
            }
        )
    return [{"contract_id": "f" * 64, "passages": passages, "queries": queries}]


def test_private_contract_requires_all_22_queries():
    value = subject.private_input_payload(_contracts())
    assert len(subject.validate_private_input(value)[0]["queries"]) == 22
    value["contracts"][0]["queries"].pop()
    with pytest.raises(subject.MaudCoordinateError):
        subject.validate_private_input(value)


def test_private_contract_accepts_worst_case_canonical_passage_expansion():
    raw = "\U0001f600" * typed_core.HARD_MAXIMUM_CODE_POINTS
    raw_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    passage = typed_core.Passage(
        ordinal=0,
        context_sha256=raw_sha256,
        start=0,
        end=len(raw),
        text=raw,
        exact_substring_sha256=raw_sha256,
    )
    contracts = _contracts()
    contracts[0]["passages"][0]["text"] = passage.serialized_bytes().decode("ascii")
    assert len(contracts[0]["passages"][0]["text"]) > 1_500
    subject.private_input_payload(contracts)


def test_coordinate_native_and_torch_threads_are_hard_bounded(monkeypatch):
    for key in subject.NATIVE_THREAD_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, "1")
    subject._require_native_thread_environment()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(subject.MaudCoordinateError, match="BLAS/OpenMP"):
        subject._require_native_thread_environment()

    class FakeTorch:
        intra = 0
        inter = 0

        @classmethod
        def set_num_threads(cls, value):
            cls.intra = value

        @classmethod
        def set_num_interop_threads(cls, value):
            cls.inter = value

        @classmethod
        def get_num_threads(cls):
            return cls.intra

        @classmethod
        def get_num_interop_threads(cls):
            return cls.inter

    subject._configure_torch_threads(FakeTorch)
    assert (FakeTorch.intra, FakeTorch.inter) == (1, 1)


def test_minilm_encodes_passages_once_and_quantizes_cosine():
    calls = []

    def encoder(texts):
        calls.append(tuple(texts))
        return [[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i in range(len(texts))]

    output = subject.compute_minilm(_contracts(), encoder)
    rows = output["rows"]
    assert len(calls) == 1
    assert len(calls[0]) == 5 + 22
    assert len(rows) == 22
    assert rows[0]["scores"] == [500_000, 1_000_000, 500_000, 1_000_000, 500_000]
    assert output["contract_pairwise"][0]["pairwise_scores"][0] == [
        1_000_000,
        500_000,
        1_000_000,
        500_000,
        1_000_000,
    ]


def test_cross_encoder_batches_once_per_pair_and_sigmoids():
    sizes = []

    def scorer(pairs):
        sizes.append(len(pairs))
        return [0.0] * len(pairs)

    rows = subject.compute_cross_encoder(_contracts(), scorer, batch_size=3)
    assert len(rows) == 22
    assert all(row["scores"] == [500_000] * 5 for row in rows)
    assert sum(sizes) == 22 * 5
    assert max(sizes) == 3


def test_coordinate_output_rejects_noninteger_or_duplicate_rows():
    with pytest.raises(subject.MaudCoordinateError):
        subject.coordinate_output(
            role=subject.ROLE_MINILM,
            rows=[
                {"work_id": "a" * 64, "scores": [1, 2, 3, 4, 5]},
                {"work_id": "a" * 64, "scores": [1, 2, 3, 4, 5]},
            ],
            input_sha256="b" * 64,
            model_tree_sha256="c" * 64,
            contract_pairwise=[
                {
                    "contract_id": "d" * 64,
                    "pairwise_scores": [
                        [1_000_000 if i == j else 500_000 for j in range(5)]
                        for i in range(5)
                    ],
                }
            ],
        )
    with pytest.raises(subject.MaudCoordinateError):
        subject.coordinate_output(
            role=subject.ROLE_CROSS_ENCODER,
            rows=[{"work_id": "a" * 64, "scores": [math.nan] * 5}],
            input_sha256="b" * 64,
            model_tree_sha256="c" * 64,
        )
