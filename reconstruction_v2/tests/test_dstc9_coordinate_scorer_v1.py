from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from assumption_agent.benchmarks import dstc9_p1_typed_core_v1 as typed_core
import replication_runtime.dstc9_coordinate_scorer_v1.adapter as adapter
import replication_runtime.dstc9_coordinate_scorer_v1.contract as contract
import replication_runtime.dstc9_coordinate_scorer_v1.worker as worker


MODEL_BINDING_SHA256 = "b" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _snippets(
    count: int = contract.CORPUS_SIZE,
) -> list[dict[str, object]]:
    return [
        {
            "body": f"Synthetic source-free body {ordinal}.",
            "entity_name": (
                None if ordinal == 0 else f"Synthetic entity {ordinal % 13}"
            ),
            "ordinal": ordinal,
            "title": f"Synthetic title {ordinal}.",
        }
        for ordinal in range(count)
    ]


def _histories(count: int = 1) -> list[dict[str, object]]:
    return [
        {
            "turns": [
                {
                    "speaker": "U",
                    "text": f"Synthetic opening question {ordinal}.",
                },
                {
                    "speaker": "S",
                    "text": f"Synthetic system answer {ordinal}.",
                },
                {
                    "speaker": "U",
                    "text": f"Synthetic final question {ordinal}.",
                },
            ],
            "work_id": f"opaque-work-{ordinal:04d}",
        }
        for ordinal in range(count)
    ]


def _input(query_count: int = 1) -> dict[str, object]:
    return contract.input_payload(
        snippets=_snippets(),
        histories=_histories(query_count),
    )


def _recompute_self_hash(payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("self_sha256", None)
    payload["self_sha256"] = contract.stable_hash(body)


def _score_rows(query_count: int, value: int = 0) -> list[dict[str, list[int]]]:
    return [
        {
            name: [value] * contract.CORPUS_SIZE
            for name in contract.SCORE_NAMES
        }
        for _ in range(query_count)
    ]


class _FakeMiniLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: object) -> np.ndarray:
        rows = tuple(texts)  # type: ignore[arg-type]
        self.calls.append(rows)
        matrix = np.zeros(
            (len(rows), 384),
            dtype=np.float32,
        )
        matrix[:, 0] = 1.0
        return matrix


class _FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def __call__(
        self, pairs: object
    ) -> list[float]:
        checked = tuple(pairs)  # type: ignore[arg-type]
        self.calls.append(checked)
        return [
            ordinal / contract.SCORE_SCALE
            for ordinal in range(len(checked))
        ]


def test_exact_source_free_input_delegates_typed_core_hashes() -> None:
    payload = _input(2)
    scorer_input = contract.validate_input(payload)

    assert set(payload) == contract.INPUT_KEYS
    assert len(scorer_input.snippets) == 2900
    assert len(scorer_input.histories) == 2
    assert payload["study_id"] == typed_core.STUDY_ID == contract.STUDY_ID
    assert contract.TYPED_CORE_SHA256 == (
        "a8290586595922e074e0a1aff52fd0d3eee396d0f1d366ccfc8407a5db65aa32"
    )
    assert contract.stable_hash(payload["snippets"]) == typed_core.stable_hash(
        payload["snippets"]
    )
    assert contract.input_projection(scorer_input) == payload
    assert contract.serialize_model_queries(scorer_input.histories)[0] == (
        typed_core.serialize_model_query(scorer_input.histories[0].turns)
    )
    assert contract.serialize_passages(scorer_input.snippets)[0] == (
        typed_core.serialize_passage(scorer_input.snippets[0])
    )
    assert contract.serialize_entity_fields(scorer_input.snippets)[0] == (
        contract.ENTITY_NONE_SERIALIZATION
    )
    assert contract.verify_typed_core_binding(PROJECT_ROOT).name == (
        "dstc9_p1_typed_core_v1.py"
    )


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("snippet", "domain"),
        ("snippet", "family"),
        ("snippet", "entity_id"),
        ("snippet", "doc_id"),
        ("history", "qrel"),
        ("history", "label"),
        ("history", "response"),
        ("turn", "score"),
    ],
)
def test_forbidden_source_label_response_and_score_fields_fail_closed(
    location: str,
    field: str,
) -> None:
    payload = _input()
    if location == "snippet":
        target = payload["snippets"][0]  # type: ignore[index]
    elif location == "history":
        target = payload["histories"][0]  # type: ignore[index]
    else:
        target = payload["histories"][0]["turns"][0]  # type: ignore[index]
    target[field] = "forbidden"  # type: ignore[index]
    _recompute_self_hash(payload)
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="forbidden",
    ):
        contract.validate_input(payload)


def test_exact_corpus_history_and_public_field_contracts() -> None:
    for count in (2899, 2901):
        with pytest.raises(
            contract.Dstc9CoordinateScorerError,
            match="exactly 2900",
        ):
            contract.input_payload(
                snippets=_snippets(count),
                histories=_histories(),
            )

    bad_ordinal = _snippets()
    bad_ordinal[7]["ordinal"] = 8
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="contiguous",
    ):
        contract.input_payload(
            snippets=bad_ordinal,
            histories=_histories(),
        )

    duplicate_work = _input(2)
    histories = duplicate_work["histories"]
    histories[1]["work_id"] = histories[0]["work_id"]  # type: ignore[index]
    duplicate_work["history_projection_sha256"] = contract.stable_hash(
        histories
    )
    parsed = [
        contract.HistoryItem(
            work_id=row["work_id"],
            turns=tuple(
                typed_core.turn_from_public_fields(turn)
                for turn in row["turns"]
            ),
        )
        for row in histories  # type: ignore[union-attr]
    ]
    duplicate_work["model_query_sha256"] = contract.stable_hash(
        list(contract.serialize_model_queries(parsed))
    )
    _recompute_self_hash(duplicate_work)
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="unique",
    ):
        contract.validate_input(duplicate_work)

    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="1..256",
    ):
        contract.input_payload(snippets=_snippets(), histories=[])
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="1..256",
    ):
        contract.input_payload(
            snippets=_snippets(),
            histories=_histories(257),
        )

    wrong_speaker = _histories()
    wrong_speaker[0]["turns"][-1]["speaker"] = "S"  # type: ignore[index]
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="history validation",
    ):
        contract.input_payload(
            snippets=_snippets(),
            histories=wrong_speaker,
        )


def test_dependency_injected_worker_computes_six_exact_vectors_once() -> None:
    payload = _input(2)
    minilm = _FakeMiniLM()
    cross = _FakeCrossEncoder()

    output = worker.score_with_dependencies(
        payload,
        minilm_encoder=minilm,
        cross_encoder=cross,
        model_binding_sha256=MODEL_BINDING_SHA256,
    )
    validated = contract.validate_output(
        output,
        expected_input=contract.validate_input(payload),
        expected_model_binding_sha256=MODEL_BINDING_SHA256,
    )

    assert validated == output
    assert [len(call) for call in minilm.calls] == [
        2,
        2900,
        2900,
        2900,
        2900,
    ]
    assert len(cross.calls) == 4
    assert all(len(call) == 2900 for call in cross.calls)
    queries = contract.serialize_model_queries(
        contract.validate_input(payload).histories
    )
    passages = contract.serialize_passages(
        contract.validate_input(payload).snippets
    )
    assert cross.calls[0][0] == (queries[0], passages[0])
    assert cross.calls[1][0] == (
        "Synthetic final question 0.",
        passages[0],
    )
    for row in output["rows"]:  # type: ignore[union-attr]
        vectors = row["vectors"]
        assert tuple(vectors) == contract.SCORE_NAMES
        assert set(vectors) == set(contract.SCORE_NAMES)
        assert all(len(vector) == 2900 for vector in vectors.values())
        assert vectors["global_ce"] == list(range(2900))
        assert vectors["last_turn_ce"] == list(range(2900))
        for name in ("minilm", "entity", "title", "body"):
            assert vectors[name] == [contract.SCORE_SCALE] * 2900
    receipt = output["receipt"]
    assert set(receipt) == contract.RECEIPT_KEYS
    receipt_body = dict(receipt)
    receipt_sha256 = receipt_body.pop("receipt_sha256")
    assert receipt_sha256 == contract.stable_hash(receipt_body)
    assert receipt["minilm_model_load_count"] == 1
    assert receipt["minilm_encode_call_count"] == 5
    assert receipt["minilm_text_count"] == 2 + 4 * 2900
    assert receipt["cross_encoder_model_load_count"] == 1
    assert receipt["cross_encoder_call_count"] == 4
    assert receipt["cross_encoder_pair_count"] == 4 * 2900
    assert receipt["retry_count"] == receipt["dynamic_resize_count"] == 0
    assert receipt["network_access"] == "denied"


def test_private_output_is_canonical_self_hashed_and_content_free() -> None:
    payload = _input()
    scorer_input = contract.validate_input(payload)
    output = contract.make_output(
        scorer_input=scorer_input,
        score_rows=_score_rows(1, value=17),
        model_binding_sha256=MODEL_BINDING_SHA256,
    )
    raw = contract.canonical_bytes(output)

    assert b"Synthetic source-free body" not in raw
    assert b"Synthetic final question" not in raw
    assert contract.parse_output_bytes(
        raw,
        expected_input=scorer_input,
        expected_model_binding_sha256=MODEL_BINDING_SHA256,
    ) == output

    tampered = copy.deepcopy(output)
    tampered["rows"][0]["vectors"]["body"][0] = 17.0
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="bounded integer",
    ):
        contract.validate_output(
            tampered,
            expected_input=scorer_input,
            expected_model_binding_sha256=MODEL_BINDING_SHA256,
        )

    extra = copy.deepcopy(output)
    extra["rows"][0]["label"] = 1
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="row binding",
    ):
        contract.validate_output(
            extra,
            expected_input=scorer_input,
            expected_model_binding_sha256=MODEL_BINDING_SHA256,
        )

    noncanonical = json.dumps(output).encode("utf-8")
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="canonical",
    ):
        contract.parse_output_bytes(
            noncanonical,
            expected_input=scorer_input,
            expected_model_binding_sha256=MODEL_BINDING_SHA256,
        )


def test_gpu1_clean_environment_and_logical_cuda0_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = adapter._worker_environment(
        runtime_python=Path(sys.executable),
        project_root=PROJECT_ROOT,
        writable_root=tmp_path,
    )
    assert set(environment) == contract.WORKER_ENVIRONMENT_KEYS
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert not any(
        key.casefold().endswith(("api_key", "token"))
        for key in environment
    )
    worker._validate_effective_environment(environment)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    class _FakeCuda:
        def is_available(self) -> bool:
            return True

        def device_count(self) -> int:
            return 1

        def set_device(self, device: int) -> None:
            assert device == 0

        def current_device(self) -> int:
            return 0

    fake_torch = SimpleNamespace(
        cuda=_FakeCuda(),
        empty=lambda _size, *, device: SimpleNamespace(device=device),
    )
    worker._validate_logical_cuda0(fake_torch)

    bad_environment = dict(environment)
    bad_environment["HTTP_PROXY"] = "forbidden"
    with pytest.raises(
        contract.Dstc9CoordinateScorerError,
        match="environment",
    ):
        worker._validate_effective_environment(bad_environment)


def test_adapter_launches_one_network_denied_offline_worker(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    tmp_path = Path(tempfile.mkdtemp(prefix="dstc9-scorer-", dir="/tmp"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path))
    payload = _input()
    project_root = PROJECT_ROOT
    runtime_python = tmp_path / "frozen-python"
    runtime_python.write_bytes(b"synthetic executable")
    runtime_python.chmod(0o700)
    manifest = tmp_path / "minilm.manifest.json"
    manifest.write_bytes(b"{}\n")
    minilm_root = tmp_path / "minilm"
    cross_root = tmp_path / "cross"
    minilm_root.mkdir()
    cross_root.mkdir()
    work_root = tmp_path / "private-work"
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(adapter, "_preflight_systemd_transport", lambda: None)

    def _run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        scorer_input = contract.validate_input(
            json.loads(input_path.read_text(encoding="utf-8"))
        )
        output = contract.make_output(
            scorer_input=scorer_input,
            score_rows=_score_rows(1),
            model_binding_sha256=MODEL_BINDING_SHA256,
        )
        descriptor = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contract.canonical_bytes(output))
        terminal = {
            "corpus_count": 2900,
            "model_binding_sha256": MODEL_BINDING_SHA256,
            "output_self_sha256": output["self_sha256"],
            "query_count": 1,
            "stage": "coordinate_score",
            "status": "passed",
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(terminal).encode("utf-8") + b"\n",
            stderr=b"",
        )

    monkeypatch.setattr(adapter.subprocess, "run", _run)
    output = adapter.run_dstc9_coordinate_scorer_v1(
        input_value=payload,
        runtime_python=runtime_python,
        project_root=project_root,
        minilm_asset_manifest=manifest,
        minilm_model_root=minilm_root,
        cross_encoder_model_root=cross_root,
        work_root=work_root,
        timeout_seconds=111,
    )

    assert len(calls) == 1
    command, keyword_arguments = calls[0]
    assert "--ignore-environment" in command
    assert "IPAddressDeny=any" in command
    assert "RestrictAddressFamilies=AF_UNIX" in command
    assert (
        "replication_runtime.dstc9_coordinate_scorer_v1.worker" in command
    )
    assert "CUDA_VISIBLE_DEVICES=1" in command
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert keyword_arguments["timeout"] == 111
    assert set(keyword_arguments["env"]) <= {
        "DBUS_SESSION_BUS_ADDRESS",
        "HOME",
        "LANG",
        "PATH",
        "XDG_RUNTIME_DIR",
    }
    assert (work_root.stat().st_mode & 0o777) == 0o700
    assert (work_root / "private_input.json").stat().st_mode & 0o777 == 0o600
    assert (
        (work_root / "private_coordinate_scores.json").stat().st_mode & 0o777
        == 0o600
    )
    assert output["self_sha256"] == json.loads(
        (work_root / "private_coordinate_scores.json").read_text(
            encoding="utf-8"
        )
    )["self_sha256"]


def test_adapter_transport_command_denies_network_and_clears_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    systemd_run = tmp_path / "systemd-run"
    env_executable = tmp_path / "env"
    systemd_run.write_bytes(b"synthetic")
    env_executable.write_bytes(b"synthetic")
    monkeypatch.setattr(adapter, "SYSTEMD_RUN", systemd_run)
    monkeypatch.setattr(adapter, "ENV_EXECUTABLE", env_executable)
    command = adapter._systemd_command_prefix()
    for property_value in contract.SYSTEMD_NETWORK_PROPERTIES:
        assert command.count("--property") == 2
        assert property_value in command
    clean = adapter._clean_environment_exec_prefix(
        {"LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "1"}
    )
    assert clean[:3] == [str(env_executable), "--ignore-environment", "--"]
    assert clean[3:] == [
        "CUDA_VISIBLE_DEVICES=1",
        "LANG=C.UTF-8",
    ]
