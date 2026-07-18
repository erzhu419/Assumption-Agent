from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from assumption_agent.benchmarks import feverous_local_runtime_v1 as runtime
from assumption_agent.benchmarks import feverous_offline_semantic_tensor_v1 as semantic
from replication_runtime.feverous_official_hipporag_v1.contract import RetrievalBatch
from replication_runtime.multihoprag_ner_v1.contract import (
    decode_request,
    encode_response,
)
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


def _units() -> list[dict[str, object]]:
    return [
        {
            "idx": index,
            "text": (
                f"TARGET: Synthetic FEVEROUS unit {index}\n"
                f"TITLE: Synthetic page {index // 16}\n"
                "SECTION_PATH: Root\nTYPE: sentence"
            ),
        }
        for index in range(8192)
    ]


def _runtime_receipt(role: str) -> dict[str, object]:
    if role == "MiniLM":
        return {
            "asset_sha256": minilm_binding.ASSET_SELF_SHA256,
            "model_tree_sha256": minilm_binding.MODEL_TREE_SHA256,
            "status": "verified_offline_immutable_qasper_minilm_runtime",
            "weights_sha256": minilm_binding.WEIGHTS_SHA256,
        }
    if role == "NER":
        return {
            "asset_sha256": runtime.ner_binding.ASSET_SELF_SHA256,
            "canary_output_sha256": runtime.ner_binding.CANARY_OUTPUT_SHA256,
            "model_tree_sha256": runtime.ner_binding.MODEL_TREE_SHA256,
            "status": "verified_exact_six_file_offline_ner_runtime",
            "weights_sha256": runtime.ner_binding.WEIGHTS_SHA256,
        }
    if role == "NLI":
        return {
            "asset_sha256": runtime.nli_binding.ASSET_SELF_SHA256,
            "model_tree_sha256": runtime.nli_binding.MODEL_TREE_SHA256,
            "status": "verified_offline_immutable_runtime",
            "weights_sha256": runtime.nli_binding.WEIGHTS_SHA256,
        }
    raise AssertionError(role)


def test_default_config_and_preflight_are_canonical_and_inference_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def verifier(name: str, receipt: Mapping[str, object]):
        def call(**kwargs: object) -> Mapping[str, object]:
            calls.append((name, dict(kwargs)))
            return dict(receipt)

        return call

    monkeypatch.setattr(
        runtime,
        "verify_minilm_runtime_binding",
        verifier("minilm", {"status": "verified_minilm"}),
    )
    monkeypatch.setattr(
        runtime,
        "verify_ner_runtime_binding",
        verifier("ner", {"status": "verified_ner"}),
    )
    monkeypatch.setattr(
        runtime,
        "verify_feverous_design",
        lambda project: calls.append(("nli_design", {"project": project}))
        or {"status": "verified_nli_design"},
    )
    monkeypatch.setattr(
        runtime.nli_binding,
        "verify_runtime_binding",
        verifier("nli", {"status": "verified_nli"}),
    )
    monkeypatch.setattr(
        runtime,
        "verify_formal_runtime_attestation_v3",
        verifier("hippo", {"status": "verified_hippo"}),
    )
    monkeypatch.setattr(
        runtime,
        "verify_hippo_transport",
        lambda: calls.append(("transport", {})),
    )
    monkeypatch.setattr(
        runtime,
        "OfflineMiniLMEncoder",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight performed inference")
        ),
    )

    config = runtime.default_formal_runtime_config(tmp_path)
    assert config.project == tmp_path.resolve()
    assert config.local_runtime_python.is_absolute()
    assert config.nli_model_root == (
        tmp_path.resolve() / "artifacts/qasc_nli_runtime_v3/model"
    )
    assert config.local_item_worker_cap == 64
    assert config.hippo_query_batch_cap == 8
    assert config.ner_process_count == 1
    assert config.nli_worker_count == 8
    receipt = runtime.preflight_formal_runtime_config(config)
    assert receipt["model_inference_calls"] == 0
    assert receipt["external_network_calls"] == 0
    assert receipt["benchmark_source_or_private_pack_reads"] == 0
    assert receipt["hipporag_transport"] == {
        "IPAddressDeny": "any",
        "RestrictAddressFamilies": "AF_UNIX",
        "status": "verified_systemd_network_isolation_capability",
    }
    assert [name for name, _ in calls] == [
        "minilm",
        "ner",
        "nli_design",
        "nli",
        "hippo",
        "transport",
    ]
    assert calls[0][1] == {
        "asset_manifest_path": config.minilm_asset_manifest,
        "model_root": config.minilm_model_root,
    }
    assert calls[1][1] == {
        "asset_manifest_path": config.ner_asset_manifest,
        "model_root": config.ner_model_root,
    }
    assert calls[3][1] == {
        "asset_manifest_path": config.nli_asset_manifest,
        "model_root": config.nli_model_root,
    }

    calls.clear()
    with pytest.raises(runtime.FeverousLocalRuntimeError, match="not canonical"):
        runtime.preflight_formal_runtime_config(
            replace(config, nli_worker_count=7)
        )
    assert calls == []


def test_hippo_gateway_preserves_exact_8192_text_and_fixed_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    gateway = runtime.DEFAULT_RUNTIME_FACTORY.create_hippo(config)
    observed: dict[str, dict[str, object]] = {}

    def build(**kwargs: object) -> Mapping[str, object]:
        observed["build"] = dict(kwargs)
        return {"status": "synthetic_build"}

    expected = RetrievalBatch(
        indices=((0, 1, 2, 3, 4),),
        receipt={"status": "synthetic_retrieve"},
    )

    def retrieve(**kwargs: object) -> RetrievalBatch:
        observed["retrieve"] = dict(kwargs)
        return expected

    monkeypatch.setattr(
        runtime,
        "build_feverous_official_hipporag_global_index_v1",
        build,
    )
    monkeypatch.setattr(
        runtime,
        "retrieve_feverous_official_hipporag_global_index_v1",
        retrieve,
    )
    units = _units()
    assert gateway.build(units) == {"status": "synthetic_build"}
    forwarded = observed["build"]["units"]
    assert isinstance(forwarded, tuple) and len(forwarded) == 8192
    assert tuple(row["text"] for row in forwarded) == tuple(
        row["text"] for row in units
    )
    assert all(set(row) == {"idx", "text"} for row in forwarded)
    assert gateway.retrieve(block="F_search", queries=("Synthetic claim",)) is expected
    assert observed["retrieve"]["work_root"] == config.hippo_work_root / "F_search"
    with pytest.raises(runtime.FeverousLocalRuntimeError, match="block is invalid"):
        gateway.retrieve(block="legacy", queries=("Synthetic claim",))


class _CoupledProcess:
    class _Input:
        def __init__(self, owner: "_CoupledProcess") -> None:
            self.owner = owner
            self.writes: list[bytes] = []
            self.flush_count = 0
            self.closed = False

        def write(self, raw: bytes) -> int:
            with self.owner.state_lock:
                if self.owner.pending_count is not None:
                    raise AssertionError("unlocked concurrent pipe write")
                self.owner.pending_count = len(decode_request(raw))
                self.writes.append(raw)
            return len(raw)

        def flush(self) -> None:
            self.flush_count += 1

        def close(self) -> None:
            self.closed = True

    class _Output:
        def __init__(self, owner: "_CoupledProcess") -> None:
            self.owner = owner

        def readline(self, _maximum: int = -1) -> bytes:
            time.sleep(0.002)
            with self.owner.state_lock:
                count = self.owner.pending_count
                if count is None:
                    return b""
                self.owner.pending_count = None
            return encode_response(tuple(() for _ in range(count)))

    def __init__(self) -> None:
        self.state_lock = threading.Lock()
        self.pending_count: int | None = None
        self.stdin = self._Input(self)
        self.stdout = self._Output(self)
        self.stderr = io.BytesIO()
        self.returncode: int | None = None
        self.wait_timeouts: list[int] = []
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: int) -> int:
        self.wait_timeouts.append(timeout)
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_ner_persistent_worker_canary_twice_locked_pipe_and_scrubbed_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    expected_hash = runtime._canonical_hash([[] for _ in range(16)])
    monkeypatch.setattr(runtime.ner_binding, "CANARY_OUTPUT_SHA256", expected_hash)
    binding = _runtime_receipt("NER")
    binding["canary_output_sha256"] = expected_hash
    monkeypatch.setattr(
        runtime,
        "verify_ner_runtime_binding",
        lambda **_kwargs: binding,
    )
    process = _CoupledProcess()
    launch: dict[str, object] = {}

    def popen(command: Sequence[str], **kwargs: object) -> _CoupledProcess:
        launch["command"] = list(command)
        launch.update(kwargs)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    client = runtime.OfflineNERJSONLClient(
        project_root=config.project,
        runtime_python=config.local_runtime_python,
        asset_manifest_path=config.ner_asset_manifest,
        model_root=config.ner_model_root,
        pycache_root=config.ner_pycache_root,
    )
    assert len(process.stdin.writes) == 2
    assert process.stdin.writes[0] == process.stdin.writes[1]
    assert client.canary_receipt["repeat_count"] == 2
    assert client.canary_receipt["repeat_exact"] is True
    assert client.canary_receipt["output_sha256"] == expected_hash
    environment = launch["env"]
    assert isinstance(environment, dict)
    assert set(environment) == {
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "HOME",
        "LANG",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
    }
    for forbidden in (
        "ALL_PROXY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "OPENAI_API_KEY",
    ):
        assert forbidden not in environment
    query = ({"kind": "query", "query": "Synthetic person in Test City?"},)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = tuple(executor.submit(client.extract_inputs, query) for _ in range(8))
        assert all(future.result() == ((),) for future in futures)
    assert len(process.stdin.writes) == 10
    client.close()
    assert process.stdin.closed is True
    assert process.wait_timeouts == [30]
    assert process.killed is False


def test_ner_wrong_startup_hash_fails_and_closes_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    monkeypatch.setattr(runtime.ner_binding, "CANARY_OUTPUT_SHA256", "f" * 64)
    binding = _runtime_receipt("NER")
    binding["canary_output_sha256"] = "f" * 64
    monkeypatch.setattr(
        runtime,
        "verify_ner_runtime_binding",
        lambda **_kwargs: binding,
    )
    process = _CoupledProcess()
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    with pytest.raises(runtime.FeverousLocalRuntimeError, match="not repeat-exact"):
        runtime.OfflineNERJSONLClient(
            project_root=config.project,
            runtime_python=config.local_runtime_python,
            asset_manifest_path=config.ner_asset_manifest,
            model_root=config.ner_model_root,
            pycache_root=config.ner_pycache_root,
        )
    assert process.stdin.closed is True
    assert process.wait_timeouts == [30]


def test_ner_rejects_symlinked_private_runtime_chain_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    formal_root = tmp_path / runtime.FORMAL_ROOT_RELATIVE
    formal_root.parent.mkdir(parents=True)
    formal_root.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe NER path reached process launch")
        ),
    )
    with pytest.raises(runtime.FeverousLocalRuntimeError, match="symlink component"):
        runtime.OfflineNERJSONLClient(
            project_root=config.project,
            runtime_python=config.local_runtime_python,
            asset_manifest_path=config.ner_asset_manifest,
            model_root=config.ner_model_root,
            pycache_root=config.ner_pycache_root,
        )


class _FakeMiniLM:
    def __init__(self, **_kwargs: object) -> None:
        self.runtime_receipt = _runtime_receipt("MiniLM")
        self.canary_receipt = {"repeat_exact": True, "status": "passed"}

    def encode(self, texts: Sequence[str]) -> tuple[str, ...]:
        return tuple(texts)


class _FakeNER:
    instances: list["_FakeNER"] = []

    def __init__(self, **_kwargs: object) -> None:
        self.runtime_binding = _runtime_receipt("NER")
        self.canary_receipt = {"repeat_exact": True, "status": "passed"}
        self.closed = False
        self.instances.append(self)

    def extract_inputs(self, values: Sequence[Mapping[str, object]]) -> tuple[tuple, ...]:
        return tuple(() for _ in values)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_FakeNER":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _FakeNLI:
    instances: list["_FakeNLI"] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.closed = False
        self.receipt = runtime.FeverousNLIPoolReceipt(
            design={"status": "synthetic"},
            runtime=_runtime_receipt("NLI"),
            canary={
                "per_worker_startup_repeat_exact": True,
                "status": "passed",
            },
            receipt_sha256="a" * 64,
        )
        self.instances.append(self)

    def score_pairs(self, pairs: Sequence[Mapping[str, str]]) -> tuple[int, ...]:
        return tuple(1 for _ in pairs)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_FakeNLI":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _mock_bundle_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeNER.instances.clear()
    _FakeNLI.instances.clear()
    monkeypatch.setattr(
        runtime,
        "preflight_formal_runtime_config",
        lambda _config: {"status": "synthetic_preflight"},
    )
    monkeypatch.setattr(runtime, "OfflineMiniLMEncoder", _FakeMiniLM)
    monkeypatch.setattr(runtime, "OfflineNERJSONLClient", _FakeNER)
    monkeypatch.setattr(runtime, "FeverousNLIWorkerPool", _FakeNLI)
    monkeypatch.setattr(runtime, "verify_pool_receipt", lambda _receipt: "a" * 64)


def test_context_bundle_produces_verified_bound_backends_and_closes_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_bundle_dependencies(monkeypatch)
    config = runtime.default_formal_runtime_config(tmp_path)
    context = runtime.DEFAULT_RUNTIME_FACTORY.create_semantic_runtime_bundle(config)
    with context as bundle:
        assert isinstance(bundle.minilm, semantic.BoundMiniLMBackend)
        assert isinstance(bundle.ner, semantic.BoundNERBackend)
        assert isinstance(bundle.nli, semantic.BoundNLIBackend)
        assert bundle.minilm.encode(("a", "b")) == ("a", "b")
        assert bundle.ner.extract_texts(("Synthetic text",)) == ((),)
        assert bundle.nli.score_pairs(({"premise": "p", "hypothesis": "h"},)) == (1,)
        receipt = bundle.receipt()
        assert receipt["schema"] == runtime.BUNDLE_SCHEMA
        assert receipt["ner_process_count"] == 1
        assert receipt["nli_worker_count"] == 8
        assert receipt["minilm_binding"]["backend_kind"] == "verified_local_runtime"
        assert bundle.hippo is not None
    assert _FakeNER.instances[0].closed is True
    assert _FakeNLI.instances[0].closed is True
    with pytest.raises(runtime.FeverousLocalRuntimeError, match="unavailable"):
        context.receipt()


def test_bundle_startup_exception_closes_every_started_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_bundle_dependencies(monkeypatch)
    original = runtime.make_verified_backend_binding

    def fail_nli(**kwargs: object):
        if kwargs.get("role") == "NLI":
            raise RuntimeError("synthetic binding failure")
        return original(**kwargs)

    monkeypatch.setattr(runtime, "make_verified_backend_binding", fail_nli)
    context = runtime.SemanticRuntimeBundle(
        runtime.default_formal_runtime_config(tmp_path)
    )
    with pytest.raises(RuntimeError, match="synthetic binding failure"):
        context.__enter__()
    assert _FakeNER.instances[0].closed is True
    assert _FakeNLI.instances[0].closed is True


def test_factory_rejects_noncanonical_config_before_runtime_creation(
    tmp_path: Path,
) -> None:
    config = runtime.default_formal_runtime_config(tmp_path)
    drifted = replace(config, ner_batch_size=config.ner_batch_size + 1)
    with pytest.raises(runtime.FeverousLocalRuntimeError, match="not canonical"):
        runtime.DEFAULT_RUNTIME_FACTORY.create_hippo(drifted)
    with pytest.raises(runtime.FeverousLocalRuntimeError, match="not canonical"):
        runtime.DEFAULT_RUNTIME_FACTORY.create_semantic_runtime_bundle(drifted)


def test_runtime_source_has_no_controller_acquisition_or_inherited_environment() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "feverous_p6_e2_acquisition_v1" not in source
    assert "formal_controller" not in source
    assert "dict(os.environ)" not in source
    assert "HTTP_PROXY" not in source
    assert "OPENAI_API_KEY" not in source


def test_runtime_private_roots_are_exclusively_in_successor_v2_epoch() -> None:
    assert runtime.FORMAL_ROOT_RELATIVE == Path("artifacts/feverous_p6_e2_formal_v2")
    for relative in (
        runtime.HIPPORAG_STAGE_RELATIVE,
        runtime.HIPPORAG_WORK_RELATIVE,
        runtime.NER_PRIVATE_RELATIVE,
        runtime.NER_PYCACHE_RELATIVE,
    ):
        assert relative.is_relative_to(runtime.FORMAL_ROOT_RELATIVE)
