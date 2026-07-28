from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import threading

import pytest

from assumption_agent.benchmarks import quac_p1_action_adapter_v1 as action
from assumption_agent.benchmarks import quac_p1_runtime_v1 as runtime
from replication_runtime import quac_p1_source_free_canary_v1 as subject
from replication_runtime.quac_p1_official_v1 import contract as official_contract
from replication_runtime.quac_p1_official_v1 import worker as official_worker


def _write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def _bindings(root: Path) -> runtime.RuntimeBindings:
    python0 = root / "runtime0" / "bin" / "python"
    python1 = root / "runtime1" / "bin" / "python"
    _write(python0, b"synthetic python zero\n", 0o755)
    _write(python1, b"synthetic python one\n", 0o755)
    site0 = root / "runtime0" / "site"
    site1 = root / "runtime1" / "site"
    overlay1 = root / "runtime1" / "overlay"
    base1 = root / "runtime1" / "base"
    _write(site0 / "typed.dist-info" / "METADATA", b"typed runtime\n")
    _write(site1 / "hippo.dist-info" / "METADATA", b"official runtime\n")
    _write(overlay1 / "overlay.dist-info" / "METADATA", b"overlay\n")
    _write(base1 / "base.dist-info" / "METADATA", b"base\n")
    minilm = root / "assets" / "minilm"
    llm = root / "assets" / "llm"
    hippo = root / "assets" / "hipporag"
    _write(minilm / "model.safetensors", b"synthetic minilm")
    _write(llm / "model.safetensors", b"synthetic llm")
    _write(
        hippo / "src" / "hipporag" / "__init__.py",
        b"# synthetic\n",
    )
    return runtime.RuntimeBindings(
        gpu0_python=runtime.PythonRuntimeBinding.capture(
            executable=python0,
            import_tree=site0,
        ),
        gpu1_python=runtime.PythonRuntimeBinding.capture(
            executable=python1,
            import_tree=site1,
        ),
        gpu1_overlay_import_tree=runtime.FrozenTreeBinding.capture(
            overlay1
        ),
        gpu1_base_import_tree=runtime.FrozenTreeBinding.capture(base1),
        minilm_asset=runtime.FrozenTreeBinding.capture(minilm),
        llm_asset=runtime.FrozenTreeBinding.capture(llm),
        hipporag_source=runtime.FrozenTreeBinding.capture(hippo),
    )


def _config(
    tmp_path: Path,
    bindings: runtime.RuntimeBindings,
    *,
    name: str = "canary",
) -> tuple[subject.SourceFreeCanaryConfig, Path]:
    asset_freeze_path = tmp_path / f"{name}.asset_freeze.json"
    _write(
        asset_freeze_path,
        subject.canonical_bytes(
            subject.build_asset_freeze_payload(bindings)
        ),
        0o400,
    )
    payload = subject.build_config_payload(
        work_root=tmp_path / name,
        bindings=bindings,
        asset_freeze_path=asset_freeze_path,
    )
    path = tmp_path / f"{name}.config.json"
    _write(path, subject.canonical_bytes(payload), 0o400)
    return subject.load_config(path), path


class _FakeEncoder:
    def __init__(
        self,
        attempt_path: Path,
        *,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.attempt_path = attempt_path
        self.barrier = barrier
        self.calls: list[tuple[str, ...]] = []

    def encode(
        self,
        texts,
        *,
        batch_size,
        device,
        normalize_embeddings,
        dtype,
    ):
        assert self.attempt_path.is_file()
        assert stat.S_IMODE(self.attempt_path.stat().st_mode) == 0o400
        assert batch_size == runtime.MINILM_BATCH_SIZE
        assert device == runtime.MINILM_DEVICE
        assert normalize_embeddings is True
        assert dtype == "float32"
        rows = tuple(texts)
        assert len(rows) == subject.SYNTHETIC_UNIQUE_EMBEDDING_COUNT
        assert len(set(rows)) == len(rows)
        self.calls.append(rows)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        result = []
        for text in rows:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [0.0] * action.MINILM_EMBEDDING_DIMENSION
            vector[int.from_bytes(digest[:2], "big") % len(vector)] = 1.0
            vector[int.from_bytes(digest[2:4], "big") % len(vector)] += 0.25
            result.append(vector)
        return result


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    @staticmethod
    def vcount() -> int:
        return 17

    @staticmethod
    def ecount() -> int:
        return 23


class _Core:
    def __init__(self) -> None:
        self.index_call_count = 0
        self.retrieve_call_count = 0
        self.documents: list[str] = []
        self.graph = _Graph()

    def index(self, documents) -> None:
        self.index_call_count += 1
        self.documents = list(documents)

    def retrieve(self, queries, num_to_retrieve):
        self.retrieve_call_count += 1
        assert num_to_retrieve == subject.SYNTHETIC_DOCUMENT_COUNT
        return [
            _Solution(
                docs=list(reversed(self.documents)),
                doc_scores=[0.0] * len(self.documents),
            )
            for _query in queries
        ]


class _FakeOfficialLane:
    def __init__(
        self,
        *,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.barrier = barrier
        self.calls: list[runtime.OfficialLaunchRequest] = []
        self.cores: list[_Core] = []

    def __call__(self, request: runtime.OfficialLaunchRequest):
        assert request.attempt_path.is_file()
        assert request.environment["CUDA_VISIBLE_DEVICES"] == runtime.GPU1
        assert request.environment["HF_HUB_OFFLINE"] == "1"
        assert request.environment["TRANSFORMERS_OFFLINE"] == "1"
        assert not any(
            "API_KEY" in key or "RUOLI" in key
            for key in request.environment
        )
        self.calls.append(request)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        request.index_root.mkdir(mode=0o700)
        _write(request.index_root / "index.bin", b"synthetic index")
        core = _Core()
        self.cores.append(core)
        result = official_worker.retrieve_block_with_core(
            core=core,
            private_input=request.private_input,
        )
        _write(
            request.output_path,
            official_contract.canonical_bytes(result),
            0o600,
        )
        return result


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_all_keys(item) for item in value.values()),
        )
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_config_is_canonical_binding_only_and_has_no_dataset_channel(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path / "bindings")
    config, config_path = _config(tmp_path, bindings)
    payload = config.payload()
    assert set(payload) == {
        "asset_freeze_binding",
        "design_binding",
        "project_binding",
        "runtime_bindings",
        "schema",
        "self_sha256",
        "work_root",
    }
    assert not (
        _all_keys(payload)
        & {
            "answer",
            "dataset",
            "dev",
            "family",
            "label",
            "loader",
            "qrel",
            "quac_source_path",
            "split",
            "train",
        }
    )
    assert config_path.read_bytes() == subject.canonical_bytes(payload)
    assert subject.load_config(config_path) == config

    extra = {**payload, "quac_source_path": "/forbidden"}
    extra_path = tmp_path / "extra.json"
    _write(extra_path, subject.canonical_bytes(extra), 0o400)
    with pytest.raises(
        subject.QuacP1SourceFreeCanaryError,
        match="shape",
    ):
        subject.load_config(extra_path)

    pretty_path = tmp_path / "pretty.json"
    _write(
        pretty_path,
        json.dumps(payload, indent=2).encode("ascii"),
        0o400,
    )
    with pytest.raises(
        subject.QuacP1SourceFreeCanaryError,
        match="canonical",
    ):
        subject.load_config(pretty_path)


def test_project_binding_covers_and_rejects_each_import_closure_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = {
        "assumption_agent/__init__.py",
        "assumption_agent/models.py",
        "assumption_agent/benchmarks/__init__.py",
        "replication_runtime/__init__.py",
        "replication_runtime/quac_p1_official_v1/__init__.py",
        "replication_runtime/maud_extraction_p2_official_v1/__init__.py",
        "replication_runtime/maud_extraction_p2_official_v1/worker.py",
    }
    assert required.issubset(set(subject._PROJECT_FILES.values()))
    original_root = subject._PROJECT_ROOT
    copied_root = tmp_path / "project"
    originals = {}
    for relative in subject._PROJECT_FILES.values():
        raw = (original_root / relative).read_bytes()
        originals[relative] = raw
        _write(copied_root / relative, raw, 0o400)
    monkeypatch.setattr(subject, "_PROJECT_ROOT", copied_root)
    binding = subject.ProjectBinding.capture()
    binding.verify()
    for relative, raw in originals.items():
        path = copied_root / relative
        path.chmod(0o600)
        path.write_bytes(raw + b"\n# tamper\n")
        path.chmod(0o400)
        with pytest.raises(
            subject.QuacP1SourceFreeCanaryError,
            match="mismatched",
        ):
            binding.verify()
        path.chmod(0o600)
        path.write_bytes(raw)
        path.chmod(0o400)
    binding.verify()


def test_asset_freeze_is_independent_runtime_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings(tmp_path / "bindings")
    normative = subject.normative_hipporag_content_receipt(
        Path(bindings.hipporag_source.path)
    )
    monkeypatch.setattr(
        subject,
        "EXPECTED_HIPPORAG_NORMATIVE_CONTENT_SHA256",
        normative["tree_sha256"],
    )
    monkeypatch.setattr(
        subject,
        "EXPECTED_HIPPORAG_NORMATIVE_FILE_COUNT",
        normative["file_count"],
    )
    monkeypatch.setattr(
        subject,
        "EXPECTED_HIPPORAG_NORMATIVE_SIZE_BYTES",
        normative["size_bytes"],
    )
    path = tmp_path / "asset_freeze.json"
    _write(
        path,
        subject.canonical_bytes(
            subject.build_asset_freeze_payload(bindings)
        ),
        0o400,
    )
    authority = subject.AssetFreezeBinding.capture(path)
    authority.verify(bindings)

    other = _bindings(tmp_path / "other_bindings")
    with pytest.raises(
        subject.QuacP1SourceFreeCanaryError,
        match="asset freeze",
    ):
        authority.verify(other)

    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o400)
    with pytest.raises(
        subject.QuacP1SourceFreeCanaryError,
        match="asset freeze",
    ):
        authority.verify(bindings)


def test_canary_runs_real_block_orchestrator_two_lanes_and_one_index(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path / "bindings")
    config, _path = _config(tmp_path, bindings)
    barrier = threading.Barrier(2)
    inner_attempt = (
        config.work_root
        / subject.INNER_RUNTIME_DIRECTORY
        / "private"
        / "attempt.private.json"
    )
    encoder = _FakeEncoder(inner_attempt, barrier=barrier)
    official = _FakeOfficialLane(barrier=barrier)

    terminal = subject.run_source_free_canary_once(
        config,
        encoder=encoder,
        official_lane=official,
        asset_authority_verifier=lambda _config: None,
    )

    assert terminal["status"] == (
        "passed_source_free_two_lane_single_index_canary"
    )
    assert terminal["formal_source_access_count"] == 0
    assert terminal["source_path_loader_label_qrel_answer_input_count"] == 0
    assert terminal["minilm_encode_call_count"] == 1
    assert terminal["official_index_call_count"] == 1
    assert terminal["official_retrieve_call_count"] == 1
    assert terminal["max_concurrent_physical_model_lanes"] == 2
    assert terminal["parallel_submission_barrier_passed"] is True
    assert len(encoder.calls) == 1
    assert len(official.calls) == 1
    assert len(official.cores) == 1
    assert official.cores[0].index_call_count == 1
    assert official.cores[0].retrieve_call_count == 1
    assert not official.calls[0].index_root.exists()
    index_archive = (
        config.work_root
        / subject.INNER_RUNTIME_DIRECTORY
        / "private"
        / "official_index.private"
    )
    assert index_archive.is_dir() and not index_archive.is_symlink()
    assert stat.S_IMODE(index_archive.stat().st_mode) == 0o500
    assert stat.S_IMODE(
        (index_archive / "index.bin").stat().st_mode
    ) == 0o400
    safe_path = config.work_root / subject.TERMINAL_FILENAME
    assert stat.S_IMODE(safe_path.stat().st_mode) == 0o400
    assert safe_path.read_bytes() == subject.canonical_bytes(terminal)
    safe_text = safe_path.read_text("ascii")
    for forbidden in (
        "Archive Delta",
        "question_id",
        "top5_unit_ids",
        "ranking",
        "embedding",
    ):
        assert forbidden not in safe_text


def test_failure_is_aggregate_and_consumes_the_only_attempt(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path / "bindings")
    config, _path = _config(tmp_path, bindings)

    def fail_verification(_bindings, *, source_access_count):
        assert source_access_count == 0
        raise RuntimeError("synthetic private verification detail")

    with pytest.raises(
        subject.QuacP1SourceFreeCanaryError,
        match="failed closed",
    ):
        subject.run_source_free_canary_once(
            config,
            verify_bindings_once=fail_verification,
            asset_authority_verifier=lambda _config: None,
        )
    failure_path = config.work_root / subject.TERMINAL_FILENAME
    first = failure_path.read_bytes()
    failure = json.loads(first.decode("ascii"))
    assert failure["schema"] == subject.SAFE_FAILURE_SCHEMA
    assert failure["status"] == "failed_source_free_canary_no_retry"
    assert failure["formal_source_access_count"] == 0
    assert failure["retry_replay_resample_or_fallback_count"] == 0
    assert "synthetic private verification detail" not in first.decode(
        "ascii"
    )

    with pytest.raises(
        subject.QuacP1SourceFreeCanaryError,
        match="retry is forbidden",
    ):
        subject.run_source_free_canary_once(config)
    assert failure_path.read_bytes() == first


class _FakeToken:
    def __init__(self, bindings: runtime.RuntimeBindings) -> None:
        binding_sha256 = runtime.stable_hash(
            bindings.semantic_payload()
        )
        runtime_receipt = {"binding_sha256": binding_sha256}
        body = {
            "binding_sha256": binding_sha256,
            "full_tree_verification_count": 1,
            "runtime_receipt": runtime_receipt,
            "schema": runtime.VERIFIED_BINDINGS_SCHEMA,
            "source_access_count_at_verification": 0,
        }
        self.receipt = {
            **body,
            "self_sha256": runtime.stable_hash(body),
        }
        self.token_sha256 = "9" * 64
        self.require_calls = 0

    def require(self, _bindings):
        self.require_calls += 1
        return self.receipt


def _fake_runtime_safe(
    *,
    binding_sha256: str,
    token_sha256: str,
) -> dict[str, object]:
    body = {
        "API_or_online_evaluation_call_count": 0,
        "action_count": 1,
        "action_pack_file_sha256": "1" * 64,
        "asset_binding_sha256": binding_sha256,
        "attempt_count": 1,
        "attempt_file_sha256": "2" * 64,
        "binding_verification_token_sha256": token_sha256,
        "block_input_file_sha256": "3" * 64,
        "block_role": "A_hold",
        "corpus_count": 5,
        "index_cleanup": {
            "cleanup_verified": True,
            "file_count": 1,
            "total_bytes": 1,
            "tree_sha256": "4" * 64,
        },
        "label_family_qrel_or_answer_input_count": 0,
        "logical_action_query_count": 1,
        "max_concurrent_physical_model_lanes": 2,
        "minilm_encode_call_count": 1,
        "minilm_receipt_file_sha256": "5" * 64,
        "official_full_rankings_sha256": "6" * 64,
        "official_index_call_count": 1,
        "official_output_file_sha256": "7" * 64,
        "official_required": True,
        "official_retrieve_call_count": 1,
        "parallel_submission_barrier_passed": True,
        "query_count": 1,
        "retry_replay_resample_or_fallback_count": 0,
        "schema": runtime.SAFE_RESULT_SCHEMA,
        "status": "passed_label_free_block_runtime",
        "unique_embedding_count": 8,
    }
    return {**body, "self_sha256": runtime.stable_hash(body)}


def test_core_accepts_injected_encoder_lane_and_verified_token(
    tmp_path: Path,
) -> None:
    bindings = _bindings(tmp_path / "bindings")
    config, _path = _config(tmp_path, bindings)
    token = _FakeToken(bindings)
    fake_encoder = object()
    fake_lane = object()
    calls = []

    def fake_run_block(**kwargs):
        calls.append(kwargs)
        assert kwargs["verified_bindings"] is token
        assert kwargs["encoder"] is fake_encoder
        assert kwargs["official_lane"] is fake_lane
        block = kwargs["block"]
        query_id = block.queries[0].query_id
        top5 = tuple(row.unit_id for row in block.documents)
        safe = _fake_runtime_safe(
            binding_sha256=token.receipt["binding_sha256"],
            token_sha256=token.token_sha256,
        )
        return runtime.BlockRuntimeResult(
            actions={query_id: object()},
            official_top5={query_id: top5},
            safe_receipt=safe,
        )

    terminal = subject.run_source_free_canary_once(
        config,
        encoder=fake_encoder,
        official_lane=fake_lane,
        verified_bindings_token=token,
        verify_bindings_once=lambda *_args, **_kwargs: pytest.fail(
            "injected token must suppress a second full-tree verification"
        ),
        run_block_once=fake_run_block,
        asset_authority_verifier=lambda _config: None,
    )
    assert terminal["status"].startswith("passed_source_free")
    assert token.require_calls == 1
    assert len(calls) == 1


def test_production_cli_has_only_the_strict_config_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings(tmp_path / "bindings")
    _config_value, path = _config(tmp_path, bindings)
    observed = []
    monkeypatch.setattr(
        subject,
        "run_source_free_canary_once",
        lambda config: observed.append(config) or {},
    )
    assert subject.main(["--config", str(path)]) == 0
    assert len(observed) == 1
    with pytest.raises(SystemExit) as exc:
        subject.main(
            [
                "--config",
                str(path),
                "--quac-source",
                "/forbidden",
            ]
        )
    assert exc.value.code == 2
