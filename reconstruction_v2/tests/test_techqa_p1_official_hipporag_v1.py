from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile

import pytest

from assumption_agent.benchmarks import (
    techqa_p1_official_hipporag_v1 as adapter,
)
from replication_runtime.averitec_p1_official_v1 import worker as inner
from replication_runtime.morehopqa_official_hipporag_v1 import (
    contract as inner_contract,
)


@pytest.fixture
def secure_tmp_path() -> Path:
    """Use a Linux filesystem because the contract audits POSIX modes."""

    root = Path(tempfile.mkdtemp(prefix="techqa-hippo-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _queries() -> list[dict[str, object]]:
    return [
        {
            "ordinal": 0,
            "question_text": "The service fails with error ZXQ-19.",
            "question_title": "Why does the service fail?",
        },
        {
            "ordinal": 1,
            "question_text": "I need the exact supported procedure.",
            "question_title": "How can I configure the service?",
        },
    ]


def _documents(count: int = 7) -> list[dict[str, object]]:
    return [
        {
            "ordinal": ordinal,
            "text": (
                f"Synthetic public Technote body {ordinal}. "
                f"Unique token TECH-{ordinal}."
            ),
            "title": f"Synthetic Technote {ordinal}",
        }
        for ordinal in range(count)
    ]


def _payload(
    *,
    stage: str = "A_hold",
    cluster_ordinal: int = 0,
) -> dict[str, object]:
    return adapter.input_payload(
        stage=stage,
        cluster_ordinal=cluster_ordinal,
        queries=_queries(),
        documents=_documents(),
    )


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    @staticmethod
    def vcount() -> int:
        return 9

    @staticmethod
    def ecount() -> int:
        return 11


class _Core:
    def __init__(self, index_root: Path) -> None:
        self.graph = _Graph()
        self.index_root = index_root
        self.documents: list[str] = []
        self.index_calls = 0
        self.retrieve_calls = 0

    def index(self, documents: list[str]) -> None:
        self.index_calls += 1
        self.documents = list(documents)

    def retrieve(
        self,
        queries: list[str],
        *,
        num_to_retrieve: int,
    ) -> list[_Solution]:
        self.retrieve_calls += 1
        assert num_to_retrieve == len(self.documents)
        (self.index_root / "query.cache").write_bytes(b"synthetic cache")
        return [
            _Solution(
                docs=list(self.documents),
                doc_scores=[
                    float(len(self.documents) - ordinal)
                    for ordinal in range(len(self.documents))
                ],
            )
            for _query in queries
        ]


def _cuda_phase() -> dict[str, object]:
    return {
        "cuda_allocation_and_synchronize_succeeded": True,
        "logical_cuda_current_device": 0,
        "physical_visible_gpu_binding": "0",
        "torch_cuda_is_available": True,
        "visible_cuda_device_count": 1,
    }


class _FakeInner:
    def __init__(self, mutation=None) -> None:
        self.calls = 0
        self.inputs: list[dict[str, object]] = []
        self.index_roots: list[Path] = []
        self.mutation = mutation

    def __call__(
        self,
        *,
        private_input: dict[str, object],
        index_root: Path,
    ) -> dict[str, object]:
        self.calls += 1
        self.inputs.append(copy.deepcopy(private_input))
        self.index_roots.append(index_root)
        assert not index_root.exists()
        assert not index_root.is_symlink()
        index_root.mkdir(mode=0o700)
        (index_root / "frozen.index").write_bytes(b"synthetic index")
        core = _Core(index_root)
        result = inner.retrieve_with_core(
            core=core,
            private_input=private_input,
            index_root=index_root,
            cuda_receipt={
                "post_inference": _cuda_phase(),
                "pre_inference": _cuda_phase(),
            },
            observed_process_thread_peak=2,
        )
        assert core.index_calls == 1
        assert core.retrieve_calls == 1
        if self.mutation is not None:
            self.mutation(result)
        return result


def _rehash_outer_input(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("self_sha256", None)
    value["self_sha256"] = adapter.stable_hash(body)
    return value


def _rehash_inner_output(value: dict[str, object]) -> None:
    body = dict(value)
    body.pop("self_sha256", None)
    value["self_sha256"] = inner.stable_hash(body)


def _all_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        result.update(str(key) for key in value)
        for nested in value.values():
            result.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            result.update(_all_keys(nested))
    return result


def _all_strings(value: object) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        result.append(value)
    elif isinstance(value, dict):
        for key, nested in value.items():
            result.append(str(key))
            result.extend(_all_strings(nested))
    elif isinstance(value, list):
        for nested in value:
            result.extend(_all_strings(nested))
    return result


def test_public_contract_round_trip_and_inner_surface_is_label_free() -> None:
    payload = _payload()
    cluster = adapter.validate_input(payload)
    assert cluster.stage == "A_hold"
    assert cluster.cluster_ordinal == 0
    assert cluster.query_serialized_sha256 == (
        payload["query_serialized_sha256"]
    )
    assert cluster.document_serialized_sha256 == (
        payload["document_serialized_sha256"]
    )

    inner_payload = adapter.inner_payload(cluster)
    assert inner_payload["block"] == "A_hold"
    assert set(inner_payload) == {
        "articles",
        "block",
        "queries",
        "schema",
        "study_id",
    }
    assert [row["text"] for row in inner_payload["queries"]] == [
        query["question_title"] + "\n" + query["question_text"]
        for query in _queries()
    ]
    assert [row["idx"] for row in inner_payload["articles"]] == list(
        range(len(_documents()))
    )
    assert inner_contract.serialize_corpus(
        inner_contract.validate_corpus(inner_payload["articles"])
    ) == tuple(
        adapter.serialize_document(row) for row in cluster.documents
    )
    assert adapter.DOCUMENT_SERIALIZATION == (
        "title_utf8_then_two_lf_then_text_utf8_v1"
    )
    assert adapter.INNER_SERIALIZATION == (
        "title_utf8_then_two_lf_then_body_utf8_v1"
    )
    keys = {key.casefold() for key in _all_keys(inner_payload)}
    assert not keys.intersection(adapter.FORBIDDEN_INPUT_KEYS)
    assert "stage" not in keys
    assert "cluster_ordinal" not in keys


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["queries"][0].update({"family": "PROCEDURE"}),
        lambda value: value["documents"][0].update({"doc_id": "secret"}),
        lambda value: value.update({"gold_document": 2}),
        lambda value: value.update({"qrels": [2]}),
        lambda value: value.update({"answer": "private answer"}),
        lambda value: value.update({"start_offset": 1}),
    ],
)
def test_forbidden_label_gold_source_and_span_fields_are_rejected(
    mutation,
) -> None:
    value = copy.deepcopy(_payload())
    mutation(value)
    with pytest.raises(
        adapter.TechqaP1OfficialHippoRAGError,
        match="forbidden",
    ):
        adapter.validate_input(value)


@pytest.mark.parametrize("target", ["query", "document"])
def test_supplied_serialized_hash_mismatch_fails_closed(target: str) -> None:
    value = copy.deepcopy(_payload())
    if target == "query":
        value["queries"][0]["question_text"] += " changed"
    else:
        value["documents"][0]["text"] += " changed"
    _rehash_outer_input(value)
    with pytest.raises(
        adapter.TechqaP1OfficialHippoRAGError,
        match=f"{target} serialized SHA-256 mismatch",
    ):
        adapter.validate_input(value)


def test_outer_stage_and_cluster_are_audit_only_and_inner_is_invariant(
    secure_tmp_path: Path,
) -> None:
    ahold = _payload(stage="A_hold", cluster_ordinal=0)
    msearch = _payload(stage="M_search", cluster_ordinal=3)
    assert adapter.inner_payload(adapter.validate_input(ahold)) == (
        adapter.inner_payload(adapter.validate_input(msearch))
    )

    first = _FakeInner()
    second = _FakeInner()
    output_a = adapter.execute_cluster_once(
        ahold,
        work_root=secure_tmp_path / "A",
        inner_runner=first,
    )
    output_m = adapter.execute_cluster_once(
        msearch,
        work_root=secure_tmp_path / "M",
        inner_runner=second,
    )
    assert first.inputs == second.inputs
    assert output_a["rows"] == output_m["rows"]
    assert output_a["inner_input_sha256"] == output_m[
        "inner_input_sha256"
    ]
    assert output_a["inner_output_sha256"] == output_m[
        "inner_output_sha256"
    ]
    assert output_a["query_serialized_sha256"] == output_m[
        "query_serialized_sha256"
    ]
    assert output_a["document_serialized_sha256"] == output_m[
        "document_serialized_sha256"
    ]
    assert output_a["outer_binding_sha256"] != output_m[
        "outer_binding_sha256"
    ]
    assert output_a["self_sha256"] != output_m["self_sha256"]
    assert output_a["stage"] == "A_hold"
    assert output_m["stage"] == "M_search"


def test_fresh_index_attempt_marker_and_no_retry(
    secure_tmp_path: Path,
) -> None:
    runner = _FakeInner()
    work_root = secure_tmp_path / "cluster-work"
    output = adapter.execute_cluster_once(
        _payload(),
        work_root=work_root,
        inner_runner=runner,
    )
    assert runner.calls == 1
    assert runner.index_roots == [work_root / "fresh_index"]
    assert output["fresh_index_create_count"] == 1
    assert output["inner_build_index_call_count"] == 1
    assert output["inner_retrieval_index_call_count"] == 0
    assert output["retry_replay_resample_count"] == 0
    assert output["online_or_API_evaluator_call_count"] == 0
    marker = work_root / "attempt.json"
    assert marker.is_file()
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == (
        output["attempt_marker_file_sha256"]
    )

    with pytest.raises(
        adapter.TechqaP1OfficialHippoRAGError,
        match="exactly once",
    ):
        adapter.execute_cluster_once(
            _payload(),
            work_root=work_root,
            inner_runner=runner,
        )
    assert runner.calls == 1


def test_safe_output_has_only_ordinals_hashes_and_audit_receipts(
    secure_tmp_path: Path,
) -> None:
    payload = _payload()
    output = adapter.execute_cluster_once(
        payload,
        work_root=secure_tmp_path / "safe-output",
        inner_runner=_FakeInner(),
    )
    assert adapter.validate_output(output, expected_input=payload) == output
    assert all(
        row["top5_document_ordinals"] == [0, 1, 2, 3, 4]
        for row in output["rows"]
    )
    for row in output["rows"]:
        assert set(row) == {
            "query_ordinal",
            "top5_document_ordinals",
        }
        assert len(row["top5_document_ordinals"]) == adapter.TOP_K
        assert len(set(row["top5_document_ordinals"])) == adapter.TOP_K

    rendered = "\n".join(_all_strings(output))
    for query in _queries():
        assert query["question_title"] not in rendered
        assert query["question_text"] not in rendered
    for document in _documents():
        assert document["title"] not in rendered
        assert document["text"] not in rendered
    assert not {
        "qrel",
        "family",
        "gold",
        "answer",
        "source_id",
        "doc_id",
        "start_offset",
        "end_offset",
    }.intersection(key.casefold() for key in _all_keys(output))


@pytest.mark.parametrize("mutation_kind", ["duplicate", "short"])
def test_invalid_inner_top_five_fails_closed(
    secure_tmp_path: Path,
    mutation_kind: str,
) -> None:
    def mutate(value: dict[str, object]) -> None:
        row = value["rows"][0]
        if mutation_kind == "duplicate":
            row["top5_document_ordinals"] = [0, 0, 1, 2, 3]
        else:
            row["top5_document_ordinals"] = [0, 1, 2, 3]
        _rehash_inner_output(value)

    with pytest.raises(
        adapter.TechqaP1OfficialHippoRAGError,
        match="inner official output validation failed",
    ):
        adapter.execute_cluster_once(
            _payload(),
            work_root=secure_tmp_path / mutation_kind,
            inner_runner=_FakeInner(mutate),
        )
    assert (secure_tmp_path / mutation_kind / "attempt.json").exists()


def test_canonical_file_entrypoint_writes_safe_output_exclusively(
    secure_tmp_path: Path,
) -> None:
    payload = _payload()
    input_path = secure_tmp_path / "input.json"
    input_path.write_bytes(adapter.canonical_bytes(payload, newline=True))
    input_path.chmod(0o600)
    output_path = secure_tmp_path / "output.json"
    result = adapter.run_from_files(
        input_path=input_path,
        output_path=output_path,
        work_root=secure_tmp_path / "work",
        inner_runner=_FakeInner(),
    )
    assert output_path.read_bytes() == adapter.canonical_bytes(
        result, newline=True
    )
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    assert adapter.validate_output(
        json.loads(output_path.read_text(encoding="ascii")),
        expected_input=payload,
    ) == result

    with pytest.raises(
        adapter.TechqaP1OfficialHippoRAGError,
        match="already exists",
    ):
        adapter.run_from_files(
            input_path=input_path,
            output_path=output_path,
            work_root=secure_tmp_path / "unused-work",
            inner_runner=_FakeInner(),
        )
    assert not (secure_tmp_path / "unused-work").exists()


def test_noncanonical_or_wrong_mode_input_does_not_consume_attempt(
    secure_tmp_path: Path,
) -> None:
    payload = _payload()
    input_path = secure_tmp_path / "input.json"
    input_path.write_text(
        json.dumps(payload, indent=2),
        encoding="ascii",
    )
    input_path.chmod(0o600)
    with pytest.raises(
        adapter.TechqaP1OfficialHippoRAGError,
        match="metadata drifted",
    ):
        adapter.run_from_files(
            input_path=input_path,
            output_path=secure_tmp_path / "output.json",
            work_root=secure_tmp_path / "work",
            inner_runner=_FakeInner(),
        )
    assert not (secure_tmp_path / "work").exists()

    input_path.write_bytes(adapter.canonical_bytes(payload, newline=True))
    input_path.chmod(0o644)
    with pytest.raises(
        adapter.TechqaP1OfficialHippoRAGError,
        match="metadata drifted",
    ):
        adapter.run_from_files(
            input_path=input_path,
            output_path=secure_tmp_path / "output.json",
            work_root=secure_tmp_path / "work",
            inner_runner=_FakeInner(),
        )
    assert not (secure_tmp_path / "work").exists()
