from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import maud_extraction_p1_runtime_v1 as runtime
from assumption_agent.benchmarks import (
    maud_extraction_p1_typed_core_v1 as typed_core,
)
from replication_runtime.maud_extraction_p1_official_v1 import (
    worker,
)


def _payload() -> dict[str, object]:
    documents = [
        {
            "ordinal": ordinal,
            "text": worker.canonical_passage_document(
                ordinal=ordinal,
                text=f"Synthetic clause {ordinal} refers to Section {ordinal + 1}.",
            ),
        }
        for ordinal in range(8)
    ]
    queries = [
        {
            "ordinal": ordinal,
            "text": f"Synthetic query {ordinal}",
            "work_id": f"opaque-query-{ordinal:02d}",
        }
        for ordinal in range(worker.QUERY_COUNT)
    ]
    return worker.input_payload(
        contract_work_id="opaque-contract",
        documents=documents,
        queries=queries,
    )


def test_openie_executor_and_native_threads_are_hard_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = worker._SingleWorkerOpenIEExecutor()
    try:
        assert executor._max_workers == 1
    finally:
        executor.shutdown(wait=True)
    with pytest.raises(
        worker.MaudOfficialHippoRAGError, match="unbounded worker pool"
    ):
        worker._SingleWorkerOpenIEExecutor(max_workers=2)
    for key in worker.NATIVE_THREAD_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, "1")
    worker._require_native_thread_environment()
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    with pytest.raises(
        worker.MaudOfficialHippoRAGError, match="BLAS/OpenMP"
    ):
        worker._require_native_thread_environment()


def test_process_monitor_observes_real_extra_os_threads() -> None:
    monitor = worker._ProcessThreadPeakMonitor.start(os.getpid())
    release = threading.Event()
    ready = [threading.Event() for _ in range(3)]

    def hold(started: threading.Event) -> None:
        started.set()
        release.wait(timeout=5)

    threads = [
        threading.Thread(target=hold, args=(started,))
        for started in ready
    ]
    for thread in threads:
        thread.start()
    try:
        assert all(started.wait(timeout=1) for started in ready)
        peak = monitor.stop()
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=1)
    assert peak >= 4


class _Graph:
    def vcount(self) -> int:
        return 17

    def ecount(self) -> int:
        return 29


class _Solution:
    def __init__(self, documents: list[str], offset: int) -> None:
        self.docs = list(reversed(documents))
        self.doc_scores = [
            float((position + offset) % len(documents))
            for position in range(len(documents))
        ]


class _Core:
    def __init__(self) -> None:
        self.graph = _Graph()
        self.index_calls: list[list[str]] = []
        self.retrieve_calls: list[tuple[list[str], int]] = []

    def index(self, documents: list[str]) -> None:
        self.index_calls.append(list(documents))

    def retrieve(
        self, queries: list[str], *, num_to_retrieve: int
    ) -> list[_Solution]:
        self.retrieve_calls.append((list(queries), num_to_retrieve))
        documents = self.index_calls[0]
        return [
            _Solution(documents, offset)
            for offset in range(len(queries))
        ]


def _output_for_payload(payload: dict[str, object]) -> dict[str, object]:
    return worker.retrieve_contract_with_core(
        core=_Core(), payload=payload
    )


def _paths(tmp_path: Path) -> runtime.RuntimePaths:
    names = {
        "deployed_project_root": "project",
        "official_python": "official/bin/python",
        "official_pyvenv_cfg": "official/pyvenv.cfg",
        "overlay_root": "official/site",
        "hipporag_source_root": "hippo/src",
        "p16_site_root": "p16",
        "official_base_site_root": "official/base-site",
        "smollm_model_root": "models/smollm2",
        "minilm_model_root": "models/minilm",
        "typed_python": "typed/bin/python",
        "typed_pyvenv_cfg": "typed/pyvenv.cfg",
        "typed_site_root": "typed/site",
        "cross_encoder_model_root": "models/cross",
    }
    resolved = {}
    for field, relative in names.items():
        path = tmp_path / relative
        if field.endswith("_python"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"python")
        elif field.endswith("_cfg"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cfg")
        else:
            path.mkdir(parents=True, exist_ok=True)
        resolved[field] = str(path)
    return runtime.RuntimePaths(**resolved)


def _safe_terminal() -> dict[str, object]:
    value = {
        "schema": runtime.SAFE_TERMINAL_SCHEMA,
        "safe_phase": "worker_completed",
        "returncode": 0,
        "stdout": {"bytes": 0, "mode": "0600", "sha256": "a" * 64},
        "stderr": {"bytes": 0, "mode": "0600", "sha256": "b" * 64},
        "output": {"bytes": 1, "sha256": "c" * 64},
        "private_content_exposed": False,
    }
    return {**value, "terminal_sha256": runtime.semantic_sha256(value)}


def test_passage_and_input_contract_are_exact_and_label_free() -> None:
    passage = worker.canonical_passage_document(
        ordinal=7,
        text="Exact clause \u03b2.",
    )
    assert passage == (
        '{"text":"Exact clause \\u03b2.","title":"MAUD passage 000007"}\n'
    )
    exact_text = "Exact clause \u03b2."
    exact_hash = hashlib.sha256(exact_text.encode("utf-8")).hexdigest()
    typed_passage = typed_core.Passage(
        ordinal=7,
        context_sha256="a" * 64,
        start=0,
        end=len(exact_text),
        text=exact_text,
        exact_substring_sha256=exact_hash,
    )
    assert passage.encode("ascii") == typed_passage.serialized_bytes()
    payload = _payload()
    contract_id, corpus_hash, documents, queries = worker.validate_input(
        payload
    )
    assert contract_id == "opaque-contract"
    assert corpus_hash == worker.corpus_sha256(payload["documents"])
    assert len(documents) == 8
    assert len(queries) == 22
    serialized = worker.canonical_json_bytes(payload).decode("ascii")
    for forbidden in ("answer", "gold", "qrel", "score", "family"):
        assert f'"{forbidden}"' not in serialized

    drifted = dict(payload)
    drifted["gold"] = []
    with pytest.raises(worker.MaudOfficialHippoRAGError):
        worker.validate_input(drifted)

    short = json.loads(json.dumps(payload))
    short["queries"].pop()
    with pytest.raises(
        worker.MaudOfficialHippoRAGError, match="query count"
    ):
        worker.validate_input(short)


def test_one_core_index_serves_exactly_one_22_query_batch() -> None:
    payload = _payload()
    core = _Core()
    output = worker.retrieve_contract_with_core(
        core=core, payload=payload
    )
    assert len(core.index_calls) == 1
    assert len(core.retrieve_calls) == 1
    assert len(core.retrieve_calls[0][0]) == 22
    assert core.retrieve_calls[0][1] == 8
    assert len(output["rows"]) == 22
    assert all(
        len(row["top5_passage_ordinals"]) == 5
        for row in output["rows"]
    )
    assert output["graph_node_count"] == 17
    assert output["graph_edge_count"] == 29


def test_worker_output_is_canonical_and_fails_closed_on_tamper() -> None:
    output = _output_for_payload(_payload())
    raw = worker.canonical_json_bytes(output)
    assert worker.parse_output(raw) == output

    noncanonical = json.dumps(output).encode("ascii")
    with pytest.raises(worker.MaudOfficialHippoRAGError):
        worker.parse_output(noncanonical)

    tampered = json.loads(raw)
    tampered["rows"][0]["top5_passage_ordinals"][0] = 99
    with pytest.raises(
        worker.MaudOfficialHippoRAGError, match="top-five"
    ):
        worker.parse_output(worker.canonical_json_bytes(tampered))

    escaped = dict(output)
    escaped["contract_work_id"] = "different-contract"
    with pytest.raises(
        runtime.MaudExtractionP1RuntimeError, match="input binding"
    ):
        runtime._validate_output_binding(_payload(), escaped)


def test_stable_top5_uses_score_then_ordinal_and_rejects_nonfinite() -> None:
    mapping = {f"d{index}": index for index in range(8)}
    assert worker._stable_top5(
        retrieved_documents=list(mapping),
        retrieved_scores=[1.0] * 8,
        document_to_ordinal=mapping,
    ) == (0, 1, 2, 3, 4)
    scores = [1.0] * 8
    scores[2] = float("nan")
    with pytest.raises(
        worker.MaudOfficialHippoRAGError, match="finite"
    ):
        worker._stable_top5(
            retrieved_documents=list(mapping),
            retrieved_scores=scores,
            document_to_ordinal=mapping,
        )


def test_short_model_alias_validator_rejects_absolute_and_parent_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "minilm").mkdir()
    assert worker._validate_model_alias("minilm", "embedding") == "minilm"
    for invalid in (
        str(tmp_path / "minilm"),
        "../minilm",
        "a/b",
        "a\\b",
        "x" * 65,
    ):
        with pytest.raises(worker.MaudOfficialHippoRAGError):
            worker._validate_model_alias(invalid, "embedding")


def test_public_synthetic_fixture_is_fixed_22_query_source_free() -> None:
    payload = runtime.synthetic_contract_payload()
    assert runtime.semantic_sha256(payload) == (
        runtime.PUBLIC_SYNTHETIC_FIXTURE_SHA256
    )
    _contract, _corpus, documents, queries = worker.validate_input(payload)
    assert len(documents) == 8
    assert len(queries) == 22
    raw = runtime.canonical_json_bytes(payload).decode("ascii")
    assert "public-synthetic-contract" in raw
    assert "answer_start" not in raw
    assert "maud_squad" not in raw


def test_runtime_paths_freeze_project_first_import_order(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    assert paths.pythonpath().split(os.pathsep) == [
        paths.deployed_project_root,
        paths.overlay_root,
        paths.hipporag_source_root,
        paths.p16_site_root,
        paths.official_base_site_root,
    ]
    assert paths.typed_pythonpath().split(os.pathsep) == [
        paths.deployed_project_root,
        paths.typed_site_root,
    ]
    assert set(paths.path_commitments()) == {
        f"{field}_sha256" for field in paths.__dataclass_fields__
    }
    with pytest.raises(runtime.MaudExtractionP1RuntimeError):
        runtime.RuntimePaths(
            **{
                **{
                    field: getattr(paths, field)
                    for field in paths.__dataclass_fields__
                },
                "deployed_project_root": "relative",
            }
        )


def test_source_tree_receipt_excludes_bytecode_without_mutating(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"variable")
    receipt = runtime._tree_receipt(source, exclude_bytecode=True)
    assert receipt["file_count"] == 1
    assert receipt["size_bytes"] == len(b"VALUE = 1\n")
    assert (cache / "module.pyc").exists()


def test_runtime_import_inspector_binds_transitive_origins_and_flags(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    roots = {
        "deployed_project_root": Path(paths.deployed_project_root),
        "overlay_root": Path(paths.overlay_root),
        "hipporag_source_root": Path(paths.hipporag_source_root),
        "p16_site_root": Path(paths.p16_site_root),
        "official_base_site_root": Path(paths.official_base_site_root),
    }
    origins = {}
    for index, (module, label) in enumerate(
        runtime.EXPECTED_MODULE_IMPORT_ROOTS.items()
    ):
        path = roots[label] / f"module_{index}.py"
        path.write_text(f"# {module}\n", encoding="utf-8")
        origins[module] = str(path)

    observed = {}

    def fake_runner(command: list[str], **kwargs: object) -> object:
        observed["command"] = command
        observed["env"] = kwargs["env"]
        value = {
            "module_origins": origins,
            "package_versions": runtime.EXPECTED_PACKAGE_VERSIONS,
            "python_version": runtime.EXPECTED_PYTHON_VERSION,
            "pythondontwritebytecode": "1",
            "pythonpath": paths.pythonpath(),
            "sys_path": [
                "",
                paths.deployed_project_root,
                paths.overlay_root,
                paths.hipporag_source_root,
                paths.p16_site_root,
                paths.official_base_site_root,
            ],
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode(),
            stderr=b"",
        )

    receipt = runtime.production_runtime_inspector(
        paths, runner=fake_runner
    )
    assert observed["command"][1:4] == ["-S", "-B", "-c"]
    assert observed["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert list(receipt["module_origins"]) == sorted(
        runtime.EXPECTED_MODULE_IMPORT_ROOTS
    )
    for module, row in receipt["module_origins"].items():
        assert row["import_root"] == (
            runtime.EXPECTED_MODULE_IMPORT_ROOTS[module]
        )
        assert len(row["file_sha256"]) == 64


def test_typed_import_inspector_excludes_official_sites(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    origins = {}
    for index, module in enumerate(runtime.EXPECTED_TYPED_MODULES):
        root = (
            Path(paths.deployed_project_root)
            if index == 0
            else Path(paths.typed_site_root)
        )
        path = root / f"typed_module_{index}.py"
        path.write_text(f"# {module}\n", encoding="utf-8")
        origins[module] = str(path)

    observed = {}

    def fake_runner(command: list[str], **kwargs: object) -> object:
        observed["command"] = command
        observed["env"] = kwargs["env"]
        value = {
            "module_origins": origins,
            "package_versions": runtime.EXPECTED_TYPED_PACKAGE_VERSIONS,
            "python_version": runtime.EXPECTED_PYTHON_VERSION,
            "pythonpath": paths.typed_pythonpath(),
            "sys_path": [
                "",
                paths.deployed_project_root,
                paths.typed_site_root,
            ],
            "torch_cuda_version": "12.8",
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode(),
            stderr=b"",
        )

    receipt = runtime.production_typed_runtime_inspector(
        paths, runner=fake_runner
    )
    assert observed["command"][1:4] == ["-S", "-B", "-c"]
    assert observed["env"]["PYTHONPATH"] == paths.typed_pythonpath()
    assert paths.overlay_root not in observed["env"]["PYTHONPATH"]
    assert receipt["package_versions"] == (
        runtime.EXPECTED_TYPED_PACKAGE_VERSIONS
    )


def test_runtime_fingerprint_is_path_free_and_self_bound(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    filesystem = {"assets": "verified"}
    imports = {"imports": "verified"}
    typed_imports = {"typed_imports": "verified"}
    hardware = {
        "gpu_count": 2,
        "kernel": runtime.EXPECTED_KERNEL,
        "nvidia_driver": runtime.EXPECTED_NVIDIA_DRIVER,
    }
    receipt = runtime.build_source_free_runtime_fingerprint(
        paths,
        nvidia_smi="/usr/bin/nvidia-smi",
        filesystem_inspector=lambda _paths: filesystem,
        runtime_inspector=lambda _paths: imports,
        typed_runtime_inspector=lambda _paths: typed_imports,
        hardware_inspector=lambda _path: hardware,
    )
    body = dict(receipt)
    declared = body.pop("self_sha256")
    assert declared == runtime.semantic_sha256(body)
    assert receipt["execution_limits"] == {
        "absolute_model_argv_count": 0,
        "cpu_thread_cap": 4,
        "gpu_lane_cap": 2,
        "hipporag_processes_per_gpu": 1,
        "queries_per_contract_index": 22,
    }
    serialized = runtime.canonical_json_bytes(receipt).decode("ascii")
    assert str(tmp_path) not in serialized
    assert '"formal_MAUD_file_or_row_access_count":0' in serialized


def test_gpu_parser_binds_post_reboot_driver_and_two_cards() -> None:
    raw = (
        b"0, GPU-32d6e292-70cd-50a0-405b-e344d2da8d39, "
        b"NVIDIA GeForce RTX 2080, 8192, 595.84\n"
        b"1, GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb, "
        b"NVIDIA GeForce RTX 2080, 8192, 595.84\n"
    )
    assert runtime._parse_gpu_rows(raw) == runtime.EXPECTED_GPU_ROWS
    with pytest.raises(
        runtime.MaudExtractionP1RuntimeError, match="GPU identity"
    ):
        runtime._parse_gpu_rows(raw.replace(b"595.84", b"595.85"))


def test_worker_failure_preserves_private_logs_outside_destroyed_scratch(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    scratch = tmp_path / "scratch"
    custody = tmp_path / "custody"
    target_paths = {
        paths.smollm_model_root,
        paths.minilm_model_root,
    }

    def failed_runner(command: list[str], **kwargs: object) -> object:
        assert command[1:4] == ["-S", "-B", "-m"]
        assert not target_paths.intersection(command)
        assert command[-3:] == [
            "--llm-model",
            runtime.LLM_ALIAS,
            "--embedding-model",
        ] or command[-1] == runtime.EMBEDDING_ALIAS
        kwargs["stdout"].write(b"private query fragment")
        kwargs["stderr"].write(b"private failure detail")
        return SimpleNamespace(returncode=7)

    with pytest.raises(
        runtime.MaudExtractionP1RuntimeError
    ) as captured:
        runtime.production_contract_launcher(
            payload=_payload(),
            runtime_paths=paths,
            scratch_root=scratch,
            private_custody_root=custody,
            physical_gpu="0",
            runner=failed_runner,
        )
    terminal = captured.value.safe_terminal
    assert terminal["safe_phase"] == "worker_failed"
    assert terminal["returncode"] == 7
    assert terminal["stdout"]["mode"] == "0600"
    assert terminal["stderr"]["mode"] == "0600"
    assert "query fragment" not in json.dumps(terminal)
    assert not scratch.exists()
    for name in (
        "worker.stdout.private.bin",
        "worker.stderr.private.bin",
    ):
        path = custody / name
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_successful_launcher_uses_aliases_and_returns_ordinal_only_output(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    payload = _payload()
    output = _output_for_payload(payload)
    observed = {}

    def successful_runner(command: list[str], **kwargs: object) -> object:
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_bytes(worker.canonical_json_bytes(output))
        output_path.chmod(0o600)
        kwargs["stdout"].write(b'{"status":"passed"}\n')
        return SimpleNamespace(returncode=0)

    result = runtime.production_contract_launcher(
        payload=payload,
        runtime_paths=paths,
        scratch_root=tmp_path / "scratch",
        private_custody_root=tmp_path / "custody",
        physical_gpu="1",
        runner=successful_runner,
    )
    assert result.output == output
    assert result.safe_terminal["safe_phase"] == "worker_completed"
    command = observed["command"]
    assert command[1:4] == ["-S", "-B", "-m"]
    assert command[command.index("--llm-model") + 1] == "smollm2"
    assert command[command.index("--embedding-model") + 1] == "minilm"
    assert not (tmp_path / "scratch").exists()


def test_source_free_canary_receipt_exposes_no_ranks_or_text(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    payload = runtime.synthetic_contract_payload()
    output = _output_for_payload(payload)

    def fake_launcher(**kwargs: object) -> runtime.WorkerRun:
        assert kwargs["physical_gpu"] == "0"
        return runtime.WorkerRun(
            output=output, safe_terminal=_safe_terminal()
        )

    receipt = runtime.run_source_free_synthetic_canary(
        runtime_paths=paths,
        runtime_fingerprint_sha256="d" * 64,
        scratch_root=tmp_path / "scratch",
        private_custody_root=tmp_path / "custody",
        launcher=fake_launcher,
    )
    assert receipt["shape"] == {
        "contract_index_count": 1,
        "passage_count": 8,
        "query_count": 22,
        "top_k": 5,
    }
    body = dict(receipt)
    declared = body.pop("self_sha256")
    assert declared == runtime.semantic_sha256(body)
    serialized = runtime.canonical_json_bytes(receipt).decode("ascii")
    assert "top5_passage_ordinals" not in serialized
    assert "Synthetic clause" not in serialized


def test_batch_is_bounded_to_two_gpu_lanes_and_preserves_order(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    payload = _payload()
    output = _output_for_payload(payload)
    active = 0
    maximum = 0
    lock = threading.Lock()

    def fake_launcher(**kwargs: object) -> runtime.WorkerRun:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        result = dict(output)
        result["contract_work_id"] = str(kwargs["scratch_root"])
        return runtime.WorkerRun(
            output=result, safe_terminal=_safe_terminal()
        )

    jobs = [
        runtime.ContractLaunchJob(
            payload=payload,
            scratch_root=str(tmp_path / f"scratch-{index}"),
            private_custody_root=str(tmp_path / f"custody-{index}"),
            physical_gpu=str(index % 2),
        )
        for index in range(5)
    ]
    results = runtime.run_contract_batch(
        jobs,
        runtime_paths=paths,
        launcher=fake_launcher,
        executor_factory=ThreadPoolExecutor,
    )
    assert maximum == 2
    assert [result.output["contract_work_id"] for result in results] == [
        job.scratch_root for job in jobs
    ]
    assert runtime.MAX_CPU_WORKERS == (
        runtime.MAX_GPU_LANES
        * runtime.CPU_THREADS_PER_HIPPORAG_PROCESS
    )
