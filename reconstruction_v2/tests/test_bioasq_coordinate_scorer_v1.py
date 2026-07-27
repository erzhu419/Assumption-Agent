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

from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as typed_core
import replication_runtime.bioasq_coordinate_scorer_v1.adapter as adapter
import replication_runtime.bioasq_coordinate_scorer_v1.contract as contract
import replication_runtime.bioasq_coordinate_scorer_v1.worker as worker


MODEL_BINDING_SHA256 = "b" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _passages(
    count: int = contract.CORPUS_SIZE,
) -> list[dict[str, object]]:
    return [
        {
            "ordinal": ordinal,
            "text": f"Synthetic source-free evidence passage {ordinal}.",
        }
        for ordinal in range(count)
    ]


def _queries(count: int = 1) -> list[dict[str, object]]:
    return [
        {
            "text": (
                f"Which synthetic intervention supports outcome {ordinal}?"
            ),
        }
        for ordinal in range(count)
    ]


def _input(query_count: int = 1) -> dict[str, object]:
    return contract.input_payload(
        passages=_passages(),
        queries=_queries(query_count),
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
        matrix = np.zeros((len(rows), 384), dtype=np.float32)
        matrix[:, 0] = 1.0
        return matrix


class _FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def __call__(self, pairs: object) -> list[float]:
        checked = tuple(pairs)  # type: ignore[arg-type]
        self.calls.append(checked)
        return [
            ordinal / contract.SCORE_SCALE
            for ordinal in range(len(checked))
        ]


def test_exact_source_free_input_delegates_typed_core_serializers() -> None:
    payload = _input(2)
    scorer_input = contract.validate_input(payload)

    assert set(payload) == contract.INPUT_KEYS
    assert len(scorer_input.passages) == 2900
    assert len(scorer_input.queries) == 2
    assert payload["study_id"] == typed_core.STUDY_ID == contract.STUDY_ID
    assert tuple(typed_core.SCORE_NAMES) == contract.SCORE_NAMES
    assert contract.stable_hash(payload["passages"]) == typed_core.stable_hash(
        payload["passages"]
    )
    assert contract.input_projection(scorer_input) == payload
    assert contract.serialize_passages(scorer_input.passages)[0] == (
        typed_core.serialize_passage(scorer_input.passages[0])
    )
    bundle = typed_core.serialize_score_queries(scorer_input.queries[0].text)
    variants = contract.serialize_query_variants(scorer_input.queries)[0]
    assert variants == {
        name: getattr(bundle, name) for name in contract.SCORE_NAMES
    }
    assert contract.verify_typed_core_binding(PROJECT_ROOT).name == (
        "bioasq_p1_typed_core_v1.py"
    )


def test_passage_serialization_digest_matches_typed_action_slate() -> None:
    payload = _input()
    scorer_input = contract.validate_input(payload)
    zeros = [0] * contract.CORPUS_SIZE
    slate = typed_core.build_action_slate(
        scorer_input.queries[0].text,
        scorer_input.passages,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
    )
    assert (
        scorer_input.passage_serialization_sha256
        == slate.passage_serialization_sha256
    )


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("passage", "document_id"),
        ("passage", "pmid"),
        ("passage", "family"),
        ("passage", "qrels"),
        ("query", "question_id"),
        ("query", "question_type"),
        ("query", "label"),
        ("query", "score"),
    ],
)
def test_forbidden_family_qrel_id_label_and_score_fields_fail_closed(
    location: str,
    field: str,
) -> None:
    payload = _input()
    target = (
        payload["passages"][0]  # type: ignore[index]
        if location == "passage"
        else payload["queries"][0]  # type: ignore[index]
    )
    target[field] = "forbidden"  # type: ignore[index]
    _recompute_self_hash(payload)
    with pytest.raises(
        contract.BioasqCoordinateScorerError,
        match="forbidden",
    ):
        contract.validate_input(payload)


def test_exact_corpus_query_and_public_field_contracts() -> None:
    for count in (2899, 2901):
        with pytest.raises(
            contract.BioasqCoordinateScorerError,
            match="exactly 2900",
        ):
            contract.input_payload(
                passages=_passages(count),
                queries=_queries(),
            )

    bad_ordinal = _passages()
    bad_ordinal[7]["ordinal"] = 8
    with pytest.raises(
        contract.BioasqCoordinateScorerError,
        match="contiguous",
    ):
        contract.input_payload(
            passages=bad_ordinal,
            queries=_queries(),
        )

    with pytest.raises(
        contract.BioasqCoordinateScorerError,
        match="1..256",
    ):
        contract.input_payload(passages=_passages(), queries=[])
    with pytest.raises(
        contract.BioasqCoordinateScorerError,
        match="1..256",
    ):
        contract.input_payload(
            passages=_passages(),
            queries=_queries(257),
        )

    extra = _queries()
    extra[0]["answer"] = "forbidden"
    with pytest.raises(
        contract.BioasqCoordinateScorerError,
        match="only question text",
    ):
        contract.input_payload(
            passages=_passages(),
            queries=extra,
        )


def test_dependency_worker_batches_exactly_six_coordinates_once() -> None:
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
    assert [len(call) for call in minilm.calls] == [2900 + 2 * 4]
    assert len(cross.calls) == 2
    assert all(len(call) == 2 * 2900 for call in cross.calls)
    scorer_input = contract.validate_input(payload)
    bundles = contract.serialize_query_variants(scorer_input.queries)
    passages = contract.serialize_passages(scorer_input.passages)
    assert cross.calls[0][0] == (bundles[0]["raw_ce"], passages[0])
    assert cross.calls[0][2900] == (bundles[1]["raw_ce"], passages[0])
    assert cross.calls[1][0] == (bundles[0]["focus_ce"], passages[0])
    for query_ordinal, row in enumerate(output["rows"]):  # type: ignore[index]
        vectors = row["vectors"]
        assert tuple(vectors) == contract.SCORE_NAMES
        assert all(len(vector) == 2900 for vector in vectors.values())
        offset = query_ordinal * 2900
        assert vectors["raw_ce"] == list(range(offset, offset + 2900))
        assert vectors["focus_ce"] == list(range(offset, offset + 2900))
        for name in contract.DENSE_SCORE_NAMES:
            assert vectors[name] == [contract.SCORE_SCALE] * 2900
    receipt = output["receipt"]
    assert set(receipt) == contract.RECEIPT_KEYS
    receipt_body = dict(receipt)
    receipt_sha256 = receipt_body.pop("receipt_sha256")
    assert receipt_sha256 == contract.stable_hash(receipt_body)
    assert receipt["minilm_model_load_count"] == 1
    assert receipt["minilm_constructor_canary_encode_call_count"] == 2
    assert receipt["minilm_formal_batch_encode_call_count"] == 1
    assert receipt["minilm_total_encode_call_count"] == 3
    assert receipt["minilm_text_count"] == 2900 + 2 * 4
    assert receipt["minilm_query_variant_count"] == 2 * 4
    assert receipt["cross_encoder_model_load_count"] == 1
    assert receipt["cross_encoder_call_count"] == 2
    assert receipt["cross_encoder_pair_count"] == 2 * 2 * 2900
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

    assert b"Synthetic source-free evidence" not in raw
    assert b"synthetic intervention" not in raw
    assert contract.parse_output_bytes(
        raw,
        expected_input=scorer_input,
        expected_model_binding_sha256=MODEL_BINDING_SHA256,
    ) == output

    tampered = copy.deepcopy(output)
    tampered["rows"][0]["vectors"]["dense_coverage"][0] = 17.0
    with pytest.raises(
        contract.BioasqCoordinateScorerError,
        match="bounded integer",
    ):
        contract.validate_output(
            tampered,
            expected_input=scorer_input,
            expected_model_binding_sha256=MODEL_BINDING_SHA256,
        )

    extra = copy.deepcopy(output)
    extra["rows"][0]["question_id"] = "forbidden"
    with pytest.raises(
        contract.BioasqCoordinateScorerError,
        match="row binding",
    ):
        contract.validate_output(
            extra,
            expected_input=scorer_input,
            expected_model_binding_sha256=MODEL_BINDING_SHA256,
        )

    noncanonical = json.dumps(output).encode("utf-8")
    with pytest.raises(
        contract.BioasqCoordinateScorerError,
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
        contract.BioasqCoordinateScorerError,
        match="environment",
    ):
        worker._validate_effective_environment(bad_environment)


def test_production_dependency_binds_two_constructor_canary_encodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from assumption_agent.benchmarks import hitab_p1_runtime_v1
    from replication_runtime.bright_minilm_v1 import encoder as minilm_module

    monkeypatch.setattr(worker, "_validate_logical_cuda0", lambda: None)

    class _ProductionMiniLM:
        repeat_count = 2

        def __init__(self, **_kwargs: object) -> None:
            self.runtime_receipt = {"binding": "synthetic"}
            self.canary_receipt = {
                "repeat_count": self.repeat_count,
                "repeat_exact": True,
            }

    class _ProductionCross:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(
        minilm_module,
        "BrightMiniLMEncoder",
        _ProductionMiniLM,
    )
    monkeypatch.setattr(
        hitab_p1_runtime_v1,
        "BrightCrossEncoderProductionScorer",
        _ProductionCross,
    )
    manifest = tmp_path / "asset.json"
    manifest.write_bytes(b"{}\n")
    _minilm, _cross, binding = worker._build_production_dependencies(
        minilm_asset_manifest=manifest,
        minilm_model_root=tmp_path,
        cross_encoder_model_root=tmp_path,
    )
    assert len(binding) == 64

    _ProductionMiniLM.repeat_count = 1
    with pytest.raises(
        contract.BioasqCoordinateScorerError,
        match="constructor canary count",
    ):
        worker._build_production_dependencies(
            minilm_asset_manifest=manifest,
            minilm_model_root=tmp_path,
            cross_encoder_model_root=tmp_path,
        )


def test_adapter_launches_one_network_denied_offline_worker(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    tmp_path = Path(tempfile.mkdtemp(prefix="bioasq-scorer-", dir="/tmp"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path))
    payload = _input()
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
    output = adapter.run_bioasq_coordinate_scorer_v1(
        input_value=payload,
        runtime_python=runtime_python,
        project_root=PROJECT_ROOT,
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
        "replication_runtime.bioasq_coordinate_scorer_v1.worker" in command
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
