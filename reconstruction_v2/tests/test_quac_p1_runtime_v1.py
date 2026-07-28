from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import threading
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import quac_p1_action_adapter_v1 as action
from assumption_agent.benchmarks import quac_p1_runtime_v1 as runtime
from replication_runtime.quac_p1_official_v1 import contract as official_contract
from replication_runtime.quac_p1_official_v1 import worker as official_worker


def _opaque(value: int) -> str:
    return f"{value:064x}"


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
    overlay1 = root / "runtime1" / "p16-site"
    base1 = root / "runtime1" / "base-site"
    _write(site0 / "typed.dist-info" / "METADATA", b"typed runtime\n")
    _write(site1 / "hippo.dist-info" / "METADATA", b"official runtime\n")
    _write(overlay1 / "torch" / "__init__.py", b"# overlay\n")
    _write(base1 / "distro.py", b"# base\n")
    minilm = root / "assets" / "minilm"
    llm = root / "assets" / "llm"
    hippo = root / "assets" / "hipporag"
    _write(minilm / "model.safetensors", b"synthetic minilm")
    _write(llm / "model.safetensors", b"synthetic llm")
    _write(hippo / "src" / "hipporag" / "__init__.py", b"# synthetic\n")
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


def _verified(
    bindings: runtime.RuntimeBindings,
) -> runtime.VerifiedRuntimeBindings:
    return runtime.verify_runtime_bindings_once(
        bindings,
        source_access_count=0,
    )


def _block(query_count: int = 3) -> runtime.RuntimeBlock:
    documents = []
    for ordinal in range(8):
        documents.append(
            action.BlockDocument(
                unit_id=_opaque(100 + ordinal),
                context_id=_opaque(500 + ordinal // 4),
                title=f"Synthetic title {ordinal // 4}",
                section_title="Synthetic section",
                context_window_ordinal=ordinal % 4,
                text=(
                    f"Alice Example visits Synthetic Place {ordinal}. "
                    f"Window evidence number {ordinal}."
                ),
            )
        )
    queries = tuple(
        runtime.RuntimeQuery(
            query_id=_opaque(1_000 + ordinal),
            question_turns=(
                action.QuestionTurn(
                    f"Where did Alice Example visit in question {ordinal}?"
                ),
                action.QuestionTurn(
                    f"Who was discussed before question {ordinal}?"
                ),
            ),
        )
        for ordinal in range(query_count)
    )
    return runtime.RuntimeBlock(
        block_id=_opaque(9_999),
        documents=tuple(documents),
        queries=queries,
    )


class _FakeEncoder:
    def __init__(
        self,
        *,
        attempt_path: Path,
        lane_barrier: threading.Barrier | None = None,
        fail: bool = False,
    ) -> None:
        self.attempt_path = attempt_path
        self.lane_barrier = lane_barrier
        self.fail = fail
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
        self.calls.append(rows)
        assert len(set(rows)) == len(rows)
        if self.lane_barrier is not None:
            self.lane_barrier.wait(timeout=5)
        if self.fail:
            raise RuntimeError("synthetic encoder failure")
        output = []
        for text in rows:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [0.0] * action.MINILM_EMBEDDING_DIMENSION
            vector[int.from_bytes(digest[:2], "big") % len(vector)] = 1.0
            vector[int.from_bytes(digest[2:4], "big") % len(vector)] += 0.25
            output.append(vector)
        return output


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    @staticmethod
    def vcount() -> int:
        return 31

    @staticmethod
    def ecount() -> int:
        return 47


class _MockCore:
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
        assert num_to_retrieve == len(self.documents)
        rows = list(queries)
        return [
            _Solution(
                docs=list(reversed(self.documents)),
                doc_scores=[0.0] * len(self.documents),
            )
            for _query in rows
        ]


class _FakeOfficialLane:
    def __init__(
        self,
        *,
        lane_barrier: threading.Barrier | None = None,
        mutate: str | None = None,
    ) -> None:
        self.lane_barrier = lane_barrier
        self.mutate = mutate
        self.calls: list[runtime.OfficialLaunchRequest] = []
        self.cores: list[_MockCore] = []

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
        if self.lane_barrier is not None:
            self.lane_barrier.wait(timeout=5)
        request.index_root.mkdir(mode=0o700)
        _write(request.index_root / "index.bin", b"synthetic index")
        core = _MockCore()
        self.cores.append(core)
        result = official_worker.retrieve_block_with_core(
            core=core,
            private_input=request.private_input,
        )
        if self.mutate is not None:
            result = deepcopy(result)
            if self.mutate == "missing":
                result["rows"].pop()
            elif self.mutate == "extra":
                result["rows"].append(deepcopy(result["rows"][0]))
            else:
                raise AssertionError(self.mutate)
            body = dict(result)
            body.pop("self_sha256")
            result["self_sha256"] = official_contract.stable_hash(body)
        raw = official_contract.canonical_bytes(result)
        _write(request.output_path, raw, 0o600)
        return result


def test_unified_agent_and_official_serializations_are_byte_identical() -> None:
    block = _block(query_count=3)
    private_input = runtime._official_input(block)
    parsed = official_contract.validate_input(private_input)
    assert official_contract.serialize_corpus(parsed.units) == tuple(
        action.serialize_evidence_unit(document)
        for document in block.documents
    )
    assert tuple(row.text for row in parsed.queries) == tuple(
        action.serialize_full_query(query.question_turns)
        for query in block.queries
    )


def test_two_lane_eager_bulk_barrier_no_duplicate_embedding_or_index(
    tmp_path: Path,
) -> None:
    block = _block(query_count=4)
    root = tmp_path / "run"
    attempt_path = root / "private" / "attempt.private.json"
    lane_barrier = threading.Barrier(2)
    encoder = _FakeEncoder(
        attempt_path=attempt_path,
        lane_barrier=lane_barrier,
    )
    official = _FakeOfficialLane(lane_barrier=lane_barrier)
    bindings = _bindings(tmp_path / "binding")
    verified = _verified(bindings)

    result = runtime.run_block(
        block_role="A_hold",
        block=block,
        work_root=root,
        bindings=bindings,
        verified_bindings=verified,
        encoder=encoder,
        official_lane=official,
    )

    assert len(encoder.calls) == 1
    assert len(set(encoder.calls[0])) == len(encoder.calls[0])
    assert len(official.calls) == 1
    assert len(official.cores) == 1
    assert official.cores[0].index_call_count == 1
    assert official.cores[0].retrieve_call_count == 1
    assert set(result.actions) == {
        query.query_id for query in block.queries
    }
    assert set(result.official_top5 or {}) == set(result.actions)
    safe = result.safe_receipt
    assert safe["parallel_submission_barrier_passed"] is True
    assert safe["max_concurrent_physical_model_lanes"] == 2
    assert safe["minilm_encode_call_count"] == 1
    assert safe["official_index_call_count"] == 1
    assert safe["official_retrieve_call_count"] == 1
    assert safe["retry_replay_resample_or_fallback_count"] == 0
    assert safe["index_cleanup"]["cleanup_verified"] is True
    assert safe["index_cleanup"]["file_count"] == 1
    assert set(safe["index_cleanup"]) == {
        "cleanup_verified",
        "file_count",
        "total_bytes",
        "tree_sha256",
    }
    assert not official.calls[0].index_root.exists()
    index_archive = root / "private" / "official_index.private"
    assert index_archive.is_dir() and not index_archive.is_symlink()
    assert stat.S_IMODE(index_archive.stat().st_mode) == 0o500
    archived_file = index_archive / "index.bin"
    assert archived_file.read_bytes() == b"synthetic index"
    assert stat.S_IMODE(archived_file.stat().st_mode) == 0o400
    assert runtime._snapshot_tree(index_archive) == (
        safe["index_cleanup"]["tree_sha256"],
        safe["index_cleanup"]["file_count"],
        safe["index_cleanup"]["total_bytes"],
    )
    assert (root / "private" / "actions.private.json").is_file()
    assert (root / "private" / "official_output.private.json").is_file()
    assert stat.S_IMODE((root / "runtime.safe.json").stat().st_mode) == 0o400


def test_production_launcher_binds_one_process_argv_env_aliases_and_timeout(
    tmp_path: Path,
) -> None:
    block = _block(query_count=2)
    root = tmp_path / "run"
    bindings = _bindings(tmp_path / "binding")
    verified = _verified(bindings)
    encoder = _FakeEncoder(
        attempt_path=root / "private" / "attempt.private.json"
    )
    calls = []

    def process_runner(command, **kwargs):
        calls.append((tuple(command), dict(kwargs)))
        assert command[:5] == [
            bindings.gpu1_python.executable.path,
            "-S",
            "-B",
            "-m",
            "replication_runtime.quac_p1_official_v1.worker",
        ]
        assert kwargs["check"] is False
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["timeout"] == runtime.OFFICIAL_WORKER_TIMEOUT_SECONDS
        environment = kwargs["env"]
        assert environment["CUDA_VISIBLE_DEVICES"] == runtime.GPU1
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TRANSFORMERS_OFFLINE"] == "1"
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert environment["PYTHONPATH"].split(os.pathsep) == [
            str(runtime._PROJECT_IMPORT_ROOT),
            bindings.gpu1_python.import_tree.path,
            bindings.hipporag_source.path,
            bindings.gpu1_overlay_import_tree.path,
            bindings.gpu1_base_import_tree.path,
        ]
        alias_root = kwargs["cwd"]
        assert alias_root.is_dir() and not alias_root.is_symlink()
        assert os.path.samefile(
            alias_root / bindings.minilm_alias,
            bindings.minilm_asset.path,
        )
        assert os.path.samefile(
            alias_root / bindings.llm_alias,
            bindings.llm_asset.path,
        )
        assert bindings.minilm_asset.path not in command
        assert bindings.llm_asset.path not in command
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        index_root = Path(command[command.index("--index-root") + 1])
        private_input = json.loads(input_path.read_text("ascii"))
        index_root.mkdir(mode=0o700)
        _write(index_root / "index.bin", b"production-launcher mock")
        result = official_worker.retrieve_block_with_core(
            core=_MockCore(),
            private_input=private_input,
        )
        _write(
            output_path,
            official_contract.canonical_bytes(result),
            0o600,
        )
        kwargs["stdout"].write(b'{"status":"passed"}\n')
        kwargs["stderr"].write(b"")
        return SimpleNamespace(returncode=0)

    result = runtime.run_block(
        block_role="A_hold",
        block=block,
        work_root=root,
        bindings=bindings,
        verified_bindings=verified,
        encoder=encoder,
        official_lane=runtime.LocalOfficialGpu1Lane(
            process_runner=process_runner,
        ),
    )
    assert len(calls) == 1
    assert result.safe_receipt["official_index_call_count"] == 1
    assert (
        root / "private" / "official_index.private"
    ).is_dir()
    assert stat.S_IMODE(
        (root / "private" / "official.stdout.private.bin").stat().st_mode
    ) == 0o400
    assert stat.S_IMODE(
        (root / "private" / "official.stderr.private.bin").stat().st_mode
    ) == 0o400


def test_nonofficial_block_uses_only_one_bulk_minilm_lane(
    tmp_path: Path,
) -> None:
    block = _block(query_count=2)
    root = tmp_path / "run"
    encoder = _FakeEncoder(
        attempt_path=root / "private" / "attempt.private.json"
    )
    bindings = _bindings(tmp_path / "binding")
    verified = _verified(bindings)
    result = runtime.run_block(
        block_role="A_form",
        block=block,
        work_root=root,
        bindings=bindings,
        verified_bindings=verified,
        encoder=encoder,
        official_lane=None,
    )
    assert len(encoder.calls) == 1
    assert result.official_top5 is None
    assert result.safe_receipt["max_concurrent_physical_model_lanes"] == 1
    assert result.safe_receipt["official_index_call_count"] == 0
    assert result.safe_receipt["parallel_submission_barrier_passed"] is None
    assert not (
        root / "private" / "official_index.private"
    ).exists()


@pytest.mark.parametrize("entry_kind", ["symlink", "fifo"])
def test_private_index_archive_rejects_links_and_special_files(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    source = tmp_path / "scratch" / "official_index"
    source.mkdir(parents=True, mode=0o700)
    _write(source / "index.bin", b"synthetic index")
    if entry_kind == "symlink":
        (source / "forbidden").symlink_to(source / "index.bin")
    else:
        try:
            os.mkfifo(source / "forbidden", mode=0o600)
        except OSError as exc:
            pytest.skip(f"test filesystem does not support FIFO: {exc}")
    archive = private_root / "official_index.private"

    with pytest.raises(
        runtime.QuacP1RuntimeError,
        match="symbolic link|special file",
    ):
        runtime._seal_private_tree_once(source, archive)

    assert source.is_dir()
    assert not archive.exists()


@pytest.mark.parametrize(
    ("block_role", "official_present"),
    [
        ("A_form", True),
        ("A_hold", False),
        ("M_search", False),
        ("not_a_block", False),
    ],
)
def test_block_role_strictly_controls_official_lifecycle(
    tmp_path: Path,
    block_role: str,
    official_present: bool,
) -> None:
    root = tmp_path / "run"
    encoder = _FakeEncoder(
        attempt_path=root / "private" / "attempt.private.json"
    )
    official = _FakeOfficialLane() if official_present else None
    bindings = _bindings(tmp_path / "binding")
    verified = _verified(bindings)
    with pytest.raises(
        runtime.QuacP1RuntimeError,
        match="block role|block lifecycle",
    ):
        runtime.run_block(
            block_role=block_role,
            block=_block(query_count=1),
            work_root=root,
            bindings=bindings,
            verified_bindings=verified,
            encoder=encoder,
            official_lane=official,
        )
    assert not root.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_official_query_ids_fail_closed_without_retry(
    tmp_path: Path,
    mutation: str,
) -> None:
    block = _block(query_count=3)
    root = tmp_path / "run"
    encoder = _FakeEncoder(
        attempt_path=root / "private" / "attempt.private.json"
    )
    official = _FakeOfficialLane(mutate=mutation)
    bindings = _bindings(tmp_path / "binding")
    verified = _verified(bindings)
    with pytest.raises(
        (
            runtime.QuacP1RuntimeError,
            official_contract.QuacP1OfficialHippoRAGError,
        )
    ):
            runtime.run_block(
                block_role="A_hold",
                block=block,
                work_root=root,
                bindings=bindings,
                verified_bindings=verified,
                encoder=encoder,
                official_lane=official,
            )
    assert len(official.calls) == 1
    assert len(encoder.calls) == 1
    failure = json.loads((root / "runtime.safe.json").read_text("ascii"))
    assert failure["status"] == "implementation_or_infrastructure_invalid"
    assert failure["retry_replay_resample_or_fallback_authorized"] is False


def test_attempt_precedes_worker_and_failed_attempt_cannot_retry(
    tmp_path: Path,
) -> None:
    block = _block(query_count=1)
    root = tmp_path / "run"
    encoder = _FakeEncoder(
        attempt_path=root / "private" / "attempt.private.json",
        fail=True,
    )
    bindings = _bindings(tmp_path / "binding")
    verified = _verified(bindings)
    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        runtime.run_block(
            block_role="A_form",
            block=block,
            work_root=root,
            bindings=bindings,
            verified_bindings=verified,
            encoder=encoder,
            official_lane=None,
        )
    assert len(encoder.calls) == 1
    assert (root / "private" / "attempt.private.json").is_file()
    with pytest.raises(
        runtime.QuacP1RuntimeError,
        match="retry is forbidden",
    ):
        runtime.run_block(
            block_role="A_form",
            block=block,
            work_root=root,
            bindings=bindings,
            verified_bindings=verified,
            encoder=encoder,
            official_lane=None,
        )
    assert len(encoder.calls) == 1


def test_production_launcher_timeout_consumes_attempt_without_retry(
    tmp_path: Path,
) -> None:
    block = _block(query_count=1)
    root = tmp_path / "run"
    calls = []

    def timeout_runner(command, **kwargs):
        calls.append(tuple(command))
        raise subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
        )

    bindings = _bindings(tmp_path / "binding")
    verified = _verified(bindings)
    with pytest.raises(
        runtime.QuacP1RuntimeError,
        match="timed out; no retry",
    ):
        runtime.run_block(
            block_role="A_hold",
            block=block,
            work_root=root,
            bindings=bindings,
            verified_bindings=verified,
            encoder=_FakeEncoder(
                attempt_path=(
                    root / "private" / "attempt.private.json"
                )
            ),
            official_lane=runtime.LocalOfficialGpu1Lane(
                process_runner=timeout_runner,
            ),
        )
    assert len(calls) == 1
    assert (root / "private" / "attempt.private.json").is_file()
    with pytest.raises(
        runtime.QuacP1RuntimeError,
        match="retry is forbidden",
    ):
        runtime.run_block(
            block_role="A_hold",
            block=block,
            work_root=root,
            bindings=bindings,
            verified_bindings=verified,
            encoder=_FakeEncoder(
                attempt_path=(
                    root / "private" / "attempt.private.json"
                )
            ),
            official_lane=runtime.LocalOfficialGpu1Lane(
                process_runner=timeout_runner,
            ),
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("root", "family"),
        ("root", "split"),
        ("root", "qrel"),
        ("document", "answer"),
        ("document", "score"),
        ("query", "label"),
        ("query", "native_context_id"),
        ("turn", "answer"),
    ],
)
def test_private_block_rejects_label_or_query_context_surfaces(
    scope: str,
    field: str,
) -> None:
    payload = runtime.block_payload(_block(query_count=1))
    tampered = deepcopy(payload)
    if scope == "root":
        tampered[field] = "forbidden"
    elif scope == "document":
        tampered["documents"][0][field] = "forbidden"
    elif scope == "query":
        tampered["queries"][0][field] = "forbidden"
    else:
        tampered["queries"][0]["question_turns"][0][field] = "forbidden"
    with pytest.raises(runtime.QuacP1RuntimeError, match="drifted"):
        runtime.validate_block_payload(tampered)


def test_asset_and_independent_runtime_binding_detects_tree_tamper(
    tmp_path: Path,
) -> None:
    binding_root = tmp_path / "binding"
    bindings = _bindings(binding_root)
    receipt = bindings.verify()
    assert receipt["gpu0_python"]["identity_sha256"] != receipt[
        "gpu1_python"
    ]["identity_sha256"]
    _write(
        Path(bindings.minilm_asset.path) / "unexpected.bin",
        b"tamper",
    )
    with pytest.raises(
        runtime.QuacP1RuntimeError,
        match="tree binding mismatched",
    ):
        bindings.verify()


def test_one_pre_source_full_verification_is_reused_without_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings(tmp_path / "binding")
    observed_calls = 0
    original = runtime.RuntimeBindings.verify

    def counted(instance):
        nonlocal observed_calls
        observed_calls += 1
        return original(instance)

    monkeypatch.setattr(runtime.RuntimeBindings, "verify", counted)
    verified = runtime.verify_runtime_bindings_once(
        bindings,
        source_access_count=0,
    )
    receipt = json.loads(verified.canonical_receipt.decode("ascii"))
    assert receipt["full_tree_verification_count"] == 1
    assert receipt["source_access_count_at_verification"] == 0
    for ordinal in range(2):
        root = tmp_path / f"run-{ordinal}"
        runtime.run_block(
            block_role="A_form",
            block=_block(query_count=1),
            work_root=root,
            bindings=bindings,
            verified_bindings=verified,
            encoder=_FakeEncoder(
                attempt_path=(
                    root / "private" / "attempt.private.json"
                )
            ),
            official_lane=None,
        )
    assert observed_calls == 1


def test_forged_or_wrong_verified_binding_is_rejected_before_attempt(
    tmp_path: Path,
) -> None:
    bound = _bindings(tmp_path / "bound")
    wrong = _bindings(tmp_path / "wrong")
    verified = _verified(bound)
    with pytest.raises(
        runtime.QuacP1RuntimeError,
        match="verifier factory",
    ):
        runtime.VerifiedRuntimeBindings()
    forged = object.__new__(runtime.VerifiedRuntimeBindings)
    for ordinal, (bindings, token) in enumerate(
        ((wrong, verified), (bound, forged))
    ):
        root = tmp_path / f"run-{ordinal}"
        with pytest.raises(
            runtime.QuacP1RuntimeError,
            match="token",
        ):
            runtime.run_block(
                block_role="A_form",
                block=_block(query_count=1),
                work_root=root,
                bindings=bindings,
                verified_bindings=token,
                encoder=_FakeEncoder(
                    attempt_path=(
                        root / "private" / "attempt.private.json"
                    )
                ),
                official_lane=None,
            )
        assert not root.exists()


def test_runtime_verification_rejects_nonzero_source_access_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runtime.QuacP1RuntimeError,
        match="precede every source access",
    ):
        runtime.verify_runtime_bindings_once(
            _bindings(tmp_path / "binding"),
            source_access_count=1,
        )
