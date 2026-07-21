"""Execute all P15 action-producing work on one frozen remote runtime."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_direct_c_confirm_v1 as p14,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p15_all_remote_c_confirm_v1 as contract,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p15_extension_acquisition_v1 as acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_c_confirm_runtime_v1 as p11_runtime,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_bridge_safe_candidate_v1 as candidate,
)
from reconstruction_v2.replication_runtime.bright_minilm_v1.encoder import (
    BrightMiniLMEncoder,
)


INTENT_SCHEMA = "bright_p15_all_remote_action_intents_v1"
LAUNCH_SCHEMA = "bright_p15_all_remote_hipporag_launch_v1"
COMPLETION_SCHEMA = "bright_p15_all_remote_hipporag_completion_v1"
CONTROLLER_SCHEMA = "bright_p15_all_remote_controller_launch_v1"
CONCURRENCY = 12
TIMEOUT_SECONDS = 1800
DENIED_NETWORK_SYSCALLS = ("connect", "sendto", "sendmsg", "sendmmsg")


class P15RemoteRuntimeError(RuntimeError):
    """The frozen P15 remote action runtime failed closed."""


class _Counter:
    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current -= 1


def _canonical_json_bytes(value: Any) -> bytes:
    return acquisition.p14_acquisition.utilities.canonical_json_bytes(value)


def _stable_hash(value: Any) -> str:
    return acquisition.p14_acquisition.utilities.stable_hash(value)


def _file_sha256(path: Path) -> str:
    return acquisition.p14_acquisition.utilities.file_sha256(path)


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    return acquisition.p14_acquisition.utilities.self_hashed(value, field="pack_sha256")


def _write_json(path: Path, value: Mapping[str, Any], mode: int = 0o600) -> None:
    acquisition.p14_acquisition.utilities._write_exclusive(
        path, _canonical_json_bytes(value), mode=mode
    )


def _read_canonical(path: Path, name: str) -> Mapping[str, Any]:
    value = acquisition._read_json(path, name)
    if _canonical_json_bytes(value) != path.read_bytes():
        raise P15RemoteRuntimeError(f"{name} is not canonical")
    return value


def _verify_self(value: Mapping[str, Any], schema: str) -> str:
    body = dict(value)
    declared = body.pop("pack_sha256", None)
    if value.get("schema") != schema or declared != _stable_hash(body):
        raise P15RemoteRuntimeError(f"{schema} self hash drifted")
    return declared


def _network_audit(root: Path, prefix: str) -> Mapping[str, Any]:
    paths = sorted(root.glob(prefix + "*"), key=lambda path: path.name)
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise P15RemoteRuntimeError("network trace set is unavailable")
    attempted = 0
    external = 0
    rows = []
    for path in paths:
        text = path.read_text(encoding="ascii")
        for line in text.splitlines():
            if not any(token + "(" in line for token in DENIED_NETWORK_SYSCALLS):
                continue
            attempted += 1
            if not any(
                marker in line
                for marker in (
                    "AF_UNIX",
                    'inet_addr("127.',
                    'inet_pton(AF_INET6, "::1"',
                )
            ):
                external += 1
        rows.append(
            {
                "path": path.name,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "denied_external_network_syscall_count": external,
        "denied_local_network_syscall_count": attempted - external,
        "denied_network_syscall_count": attempted,
        "external_network_call_count": 0,
        "trace_file_count": len(rows),
        "trace_set_sha256": _stable_hash(rows),
    }


def _minimal_environment(
    *, root: Path, visible_gpu: str, omp_threads: str = "2"
) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": visible_gpu,
        "HF_HOME": str(root / "hf"),
        "HF_HUB_OFFLINE": "1",
        "HOME": str(root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MPLCONFIGDIR": str(root / "tmp" / "mpl"),
        "OMP_NUM_THREADS": omp_threads,
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TMPDIR": str(root / "tmp"),
        "TRANSFORMERS_OFFLINE": "1",
    }


def _strace_command(command: Sequence[str], trace_prefix: Path) -> list[str]:
    return [
        "/usr/bin/strace",
        "--seccomp-bpf",
        "-ff",
        "-e",
        "trace=network",
        "-e",
        "inject=connect,sendto,sendmsg,sendmmsg:error=EPERM",
        "-o",
        str(trace_prefix),
        *command,
    ]


def _verify_plan(path: Path) -> tuple[Mapping[str, Any], Path, tuple[acquisition.RuntimeItem, ...]]:
    plan = _read_canonical(path, "P15 remote plan")
    _verify_self(plan, contract.PLAN_SCHEMA)
    if (
        plan.get("remote_hostname") != socket.gethostname()
        or plan.get("remote_hostname") != contract.REMOTE_HOSTNAME
        or plan.get("attempt_count") != contract.ATTEMPT_COUNT
        or plan.get("target_terminal_count_per_family") != contract.TARGET_PER_FAMILY
        or plan.get("study_design_self_sha256") != acquisition.DESIGN_SELF_SHA256
    ):
        raise P15RemoteRuntimeError("P15 remote plan policy drifted")
    base = Path(str(plan.get("remote_base"))).resolve(strict=True)
    if base != contract.REMOTE_BASE:
        raise P15RemoteRuntimeError("P15 remote base drifted")
    acquisition_result, items = contract.load_acquisition(base)
    binding = plan.get("acquisition_result")
    if (
        not isinstance(binding, Mapping)
        or binding.get("self_sha256") != acquisition_result.get("self_sha256")
        or binding.get("file_sha256")
        != _file_sha256(base / acquisition.RESULT_RELATIVE)
        or dict(plan.get("view_binding", {}))
        != acquisition_result.get("pack_bindings", {}).get("C_confirm_view")
    ):
        raise P15RemoteRuntimeError("P15 acquisition plan binding drifted")
    policy = plan.get("execution_policy")
    if not isinstance(policy, Mapping) or dict(policy) != {
        "cross_encoder_visible_GPU": "1",
        "external_network_call_count_allowed": 0,
        "HippoRAG_concurrency": 12,
        "HippoRAG_visible_GPU": "",
        "local_P14_output_reuse_count": 0,
        "MiniLM_and_Qwen_visible_GPU": "0",
        "online_evaluator_call_count": 0,
        "retry_replay_or_resample_count": 0,
    }:
        raise P15RemoteRuntimeError("P15 execution policy drifted")
    forbidden = (
        base / acquisition.SELECTION_SECRET_RELATIVE,
        base / acquisition.source.SOURCE_ROOT_RELATIVE / "examples",
        base / "artifacts/bright_p14_direct_c_confirm_v1/hipporag",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise P15RemoteRuntimeError("forbidden local, secret, or label-bearing input is staged")
    return plan, base, items


def _verify_assets(base: Path, plan: Mapping[str, Any]) -> tuple[Path, Path]:
    for family in contract.FAMILIES:
        slug = acquisition.source.SLUGS[family]
        relative = f"documents/{slug}-00000-of-00001.parquet"
        path = base / acquisition.source.SOURCE_ROOT_RELATIVE / relative
        binding = plan["source_document_bindings"][family]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != binding["size_bytes"]
            or _file_sha256(path) != binding["sha256"]
        ):
            raise P15RemoteRuntimeError("P15 remote source document drifted")
    bright = p11_runtime.train.bright_runtime
    directories = (
        base / bright.QWEN_MODEL_RELATIVE,
        base / bright.MINILM_MODEL_RELATIVE,
        base / bright.HIPPORAG_LLM_RELATIVE,
        base / p11_runtime.train.CROSS_MODEL_RELATIVE,
    )
    if any(path.is_symlink() or not path.is_dir() for path in directories):
        raise P15RemoteRuntimeError("P15 remote model asset is unavailable")
    python = base / bright.HIPPORAG_PYTHON_RELATIVE
    if not python.is_file() or not os.access(python, os.X_OK):
        raise P15RemoteRuntimeError("P15 remote Python is unavailable")
    baseline = (
        base
        / p11_runtime.hardening_qualification.BASELINE_REPO_RELATIVE
        / p11_runtime.hardening_qualification.BASELINE_SOURCE_WITHIN_REPO
    )
    if (
        baseline.is_symlink()
        or not baseline.is_file()
        or _file_sha256(baseline) != p11_runtime.backport.PATCHED_SOURCE_SHA256
    ):
        raise P15RemoteRuntimeError("P15 hardened HippoRAG source drifted")
    return python, baseline


def _run_qwen(
    *, base: Path, root: Path, python: Path, items: Sequence[acquisition.RuntimeItem]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = {
        "items": [{"ordinal": item.ordinal, "query": item.query} for item in items],
        "schema": p11_runtime.train.qwen_contract.INPUT_SCHEMA,
    }
    input_path = root / "qwen.input.json"
    output_path = root / "qwen.output.json"
    p11_runtime.train.bright_runtime._write_exclusive(
        input_path,
        p11_runtime.train.qwen_contract.canonical_json_bytes(payload),
        mode=0o600,
    )
    for name in ("home", "hf", "tmp"):
        (root / name).mkdir(mode=0o700)
    command = _strace_command(
        [
            str(python),
            "-I",
            "-B",
            "-m",
            "replication_runtime.bright_query_generator_v1.worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(base / p11_runtime.train.bright_runtime.QWEN_MODEL_RELATIVE),
            "--batch-size",
            str(p11_runtime.train.bright_runtime.QWEN_BATCH_SIZE),
        ],
        root / "qwen.network.trace",
    )
    completed = subprocess.run(
        command,
        cwd=base,
        env=_minimal_environment(root=root, visible_gpu="0"),
        check=False,
        capture_output=True,
        timeout=TIMEOUT_SECONDS,
    )
    p11_runtime.train.bright_runtime._write_exclusive(
        root / "qwen.stdout.log", completed.stdout, mode=0o600
    )
    p11_runtime.train.bright_runtime._write_exclusive(
        root / "qwen.stderr.log", completed.stderr, mode=0o600
    )
    audit = _network_audit(root, "qwen.network.trace")
    if completed.returncode != 0 or audit["denied_external_network_syscall_count"] != 0:
        raise P15RemoteRuntimeError(
            "P15 Qwen worker failed: " + hashlib.sha256(completed.stderr).hexdigest()
        )
    raw = p11_runtime.train.qwen_contract.parse_output(output_path.read_bytes())
    projected, projection = candidate.totalize_and_project_qwen_output(
        raw,
        tuple(
            p11_runtime.train.bright_runtime.ViewItem(
                ordinal=item.ordinal,
                family=item.family,
                commitment=item.item_key,
                query=item.query,
                excluded_ids=item.excluded_ids,
            )
            for item in items
        ),
    )
    projection_path = root / "qwen.totalized_projected.json"
    p11_runtime.train.bright_runtime._write_json(projection_path, projection)
    receipt = {
        "input_file_sha256": _file_sha256(input_path),
        "network_audit": audit,
        "output_file_sha256": _file_sha256(output_path),
        "projected_file_sha256": _file_sha256(projection_path),
        "projected_pack_sha256": projection["pack_sha256"],
        "source_valid_generation_count": projection["source_valid_generation_count"],
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "totalized_generation_count": projection["totalized_generation_count"],
    }
    return projected, receipt


def _new_minilm(base: Path, expected_canary: Mapping[str, Any]) -> BrightMiniLMEncoder:
    bright = p11_runtime.train.bright_runtime
    encoder = BrightMiniLMEncoder(
        asset_manifest=base / bright.MINILM_MANIFEST_RELATIVE,
        model_root=base / bright.MINILM_MODEL_RELATIVE,
    )
    if dict(encoder.canary_receipt) != dict(expected_canary):
        raise P15RemoteRuntimeError("P15 remote MiniLM canary drifted")
    return encoder


def _run_cross(
    *, base: Path, root: Path, python: Path, input_path: Path, output_path: Path
) -> Mapping[str, Any]:
    cross_root = root / "cross_worker"
    cross_root.mkdir(mode=0o700)
    for name in ("home", "hf", "tmp"):
        (cross_root / name).mkdir(mode=0o700)
    command = _strace_command(
        [
            str(python),
            "-I",
            "-B",
            "-m",
            "replication_runtime.bridge_expanded_cross_encoder_v1.worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(base / p11_runtime.train.CROSS_MODEL_RELATIVE),
        ],
        cross_root / "network.trace",
    )
    completed = subprocess.run(
        command,
        cwd=base,
        env=_minimal_environment(root=cross_root, visible_gpu="1"),
        check=False,
        capture_output=True,
        timeout=TIMEOUT_SECONDS,
    )
    p11_runtime.train.bright_runtime._write_exclusive(
        cross_root / "stdout.log", completed.stdout, mode=0o600
    )
    p11_runtime.train.bright_runtime._write_exclusive(
        cross_root / "stderr.log", completed.stderr, mode=0o600
    )
    audit = _network_audit(cross_root, "network.trace")
    if completed.returncode != 0 or audit["denied_external_network_syscall_count"] != 0:
        raise P15RemoteRuntimeError(
            "P15 cross worker failed: " + hashlib.sha256(completed.stderr).hexdigest()
        )
    output = p11_runtime.train.cross_contract.parse_output(output_path.read_bytes())
    return {
        "network_audit": audit,
        "output_file_sha256": _file_sha256(output_path),
        "output_item_count": len(output["items"]),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def _cleanup_item(item_root: Path) -> int:
    removed = 0
    for name in ("index", "home", "hf", "tmp"):
        path = item_root / name
        if path.is_symlink():
            raise P15RemoteRuntimeError("P15 HippoRAG work path became a symlink")
        if path.exists():
            shutil.rmtree(path)
            removed += 1
    return removed


def _run_hipporag(
    *,
    base: Path,
    item_root: Path,
    candidate_rows: Sequence[int],
    python: Path,
    baseline: Path,
    ordinal: int,
    counter: _Counter,
) -> Mapping[str, Any]:
    launch_path = item_root / "launch.json"
    completion_path = item_root / "completion.json"
    input_path = item_root / "input.json"
    if launch_path.exists() or completion_path.exists():
        raise P15RemoteRuntimeError("P15 HippoRAG attempt was already consumed")
    launch = _self_hashed(
        {
            "input_file_sha256": _file_sha256(input_path),
            "ordinal": ordinal,
            "remote_hostname": socket.gethostname(),
            "schema": LAUNCH_SCHEMA,
        }
    )
    _write_json(launch_path, launch)
    command = _strace_command(
        [
            str(python),
            "-I",
            "-B",
            "-m",
            "replication_runtime.bright_official_hipporag_v1.worker",
            "--input",
            str(input_path),
            "--output",
            str(item_root / "output.json"),
            "--index-root",
            str(item_root / "index"),
            "--llm-model",
            str(base / p11_runtime.train.bright_runtime.HIPPORAG_LLM_RELATIVE),
            "--embedding-model",
            str(base / p11_runtime.train.bright_runtime.MINILM_MODEL_RELATIVE),
        ],
        item_root / "network.trace",
    )
    counter.enter()
    timed_out = False
    try:
        process = subprocess.Popen(
            command,
            cwd=base,
            env=_minimal_environment(root=item_root, visible_gpu=""),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            returncode = None
    finally:
        counter.leave()
    p11_runtime.train.bright_runtime._write_exclusive(
        item_root / "stdout.log", stdout, mode=0o600
    )
    p11_runtime.train.bright_runtime._write_exclusive(
        item_root / "stderr.log", stderr, mode=0o600
    )
    audit = _network_audit(item_root, "network.trace")
    status = "failed"
    top_rows: list[int] | None = None
    output_sha256 = None
    failure_sha256 = None
    if (
        not timed_out
        and returncode == 0
        and audit["denied_external_network_syscall_count"] == 0
    ):
        try:
            output_path = item_root / "output.json"
            payload = p11_runtime.train.hippo_contract.parse_output(output_path.read_bytes())
            if payload["graph_node_count"] <= 32 or payload["graph_edge_count"] <= 0:
                raise P15RemoteRuntimeError("P15 HippoRAG graph is nonterminal")
            top_rows = [candidate_rows[position] for position in payload["top_ordinals"]]
            status = "terminal"
            output_sha256 = _file_sha256(output_path)
            graph_nodes = payload["graph_node_count"]
            graph_edges = payload["graph_edge_count"]
        except BaseException as exc:
            failure_sha256 = hashlib.sha256(
                (type(exc).__name__ + "\n" + str(exc)).encode()
            ).hexdigest()
    else:
        failure_sha256 = hashlib.sha256(
            f"timeout={timed_out};returncode={returncode};stderr={hashlib.sha256(stderr).hexdigest()}".encode()
        ).hexdigest()
    removed = _cleanup_item(item_root)
    completion = _self_hashed(
        {
            "failure_sha256": failure_sha256,
            "graph_edge_count": graph_edges if status == "terminal" else None,
            "graph_node_count": graph_nodes if status == "terminal" else None,
            "launch_pack_sha256": launch["pack_sha256"],
            "network_audit": audit,
            "ordinal": ordinal,
            "output_file_sha256": output_sha256,
            "remote_hostname": socket.gethostname(),
            "returncode": returncode,
            "schema": COMPLETION_SCHEMA,
            "status": status,
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "timed_out": timed_out,
            "top_rows": top_rows,
            "working_directory_count_removed": removed,
        }
    )
    _write_json(completion_path, completion)
    return completion


def run(plan_path: Path) -> Mapping[str, Any]:
    plan, base, items = _verify_plan(plan_path.resolve(strict=True))
    root = Path(str(plan["remote_work_root"]))
    if root.exists() or root.is_symlink():
        raise P15RemoteRuntimeError("P15 remote work root already exists")
    if root != contract.REMOTE_WORK_ROOT:
        raise P15RemoteRuntimeError("P15 remote work root drifted")
    python, baseline = _verify_assets(base, plan)
    fingerprint = contract.load_fingerprint(base)
    if fingerprint["self_sha256"] != plan["remote_runtime_fingerprint_self_sha256"]:
        raise P15RemoteRuntimeError("P15 remote fingerprint binding drifted")
    root.mkdir(mode=0o700, parents=True)
    controller = _self_hashed(
        {
            "plan_pack_sha256": plan["pack_sha256"],
            "remote_hostname": socket.gethostname(),
            "schema": CONTROLLER_SCHEMA,
        }
    )
    _write_json(root / "controller.launch.json", controller)
    corpora = p14.load_corpora(base)
    qwen_output, qwen_receipt = _run_qwen(
        base=base, root=root, python=python, items=items
    )
    qwen_rows = qwen_output.get("items")
    if not isinstance(qwen_rows, list) or len(qwen_rows) != contract.ATTEMPT_COUNT:
        raise P15RemoteRuntimeError("P15 Qwen row count drifted")
    encoder = _new_minilm(base, fingerprint["minilm_canary_receipt"])
    corpus_embeddings = {
        family: p11_runtime.train.bright_runtime._encode_chunks(
            encoder, corpora[family].contents
        )
        for family in contract.FAMILIES
    }
    tensor_root = root / "corpus_tensors"
    tensor_root.mkdir(mode=0o700)
    tensor_bindings = {}
    for family in contract.FAMILIES:
        matrix = np.asarray(corpus_embeddings[family], dtype=np.float32)
        if matrix.shape != (len(corpora[family].ids), 384) or not np.isfinite(matrix).all():
            raise P15RemoteRuntimeError("P15 corpus tensor drifted")
        path = tensor_root / f"{family}.embeddings.npy"
        p11_runtime.train.bright_runtime._save_npy_exclusive(path, matrix)
        tensor_bindings[family] = {
            "document_count": len(corpora[family].ids),
            "file_sha256": _file_sha256(path),
            "float32_sha256": p11_runtime.train.float32_matrix_sha256(matrix),
            "shape": list(matrix.shape),
        }
    flattened: list[str] = []
    slices: list[tuple[int, int]] = []
    for item, row in zip(items, qwen_rows):
        expansions = row.get("expansions")
        if not isinstance(expansions, list) or len(expansions) != 4:
            raise P15RemoteRuntimeError("P15 typed query row drifted")
        start = len(flattened)
        flattened.extend([item.query, *expansions])
        slices.append((start, len(flattened)))
    query_embeddings = p11_runtime.train.bright_runtime._encode_chunks(encoder, flattened)
    local_plans = []
    for item, row, (start, end) in zip(items, qwen_rows, slices):
        scores = [
            p11_runtime.train.quantized_scores(
                corpus_embeddings[item.family], query_embeddings[index]
            )
            for index in range(start, end)
        ]
        view = p11_runtime.train.ViewItem(
            ordinal=item.ordinal,
            item_key=item.item_key,
            query=item.query,
            excluded_ids=item.excluded_ids,
        )
        local_plans.append(
            p11_runtime.train.build_local_plan(
                item=view,
                document_ids=corpora[item.family].ids,
                document_contents=corpora[item.family].contents,
                query_score_vectors=scores,
                expansions=row["expansions"],
            )
        )
    bridge_queries = [query.text for plan_row in local_plans for query in plan_row.bridge_queries]
    bridge_embeddings = (
        p11_runtime.train.bright_runtime._encode_chunks(encoder, bridge_queries)
        if bridge_queries else np.empty((0, 384), dtype=np.float32)
    )
    expanded_plans = []
    offset = 0
    for item, local in zip(items, local_plans):
        count = len(local.bridge_queries)
        vectors = [
            p11_runtime.train.quantized_scores(
                corpus_embeddings[item.family], bridge_embeddings[index]
            )
            for index in range(offset, offset + count)
        ]
        expanded_plans.append(p11_runtime.train.expand_plan(local, vectors))
        offset += count
    if offset != len(bridge_queries):
        raise P15RemoteRuntimeError("P15 bridge accounting drifted")
    del encoder, query_embeddings, bridge_embeddings, corpus_embeddings
    p11_runtime._release_cuda()
    cross_payload = p11_runtime._prepare_cross_input(
        plans=expanded_plans, items=items, corpora=corpora
    )
    cross_input = root / "cross_encoder.input.json"
    cross_output = root / "cross_encoder.output.json"
    p11_runtime.train.bright_runtime._write_exclusive(
        cross_input,
        p11_runtime.train.cross_contract.canonical_json_bytes(cross_payload),
        mode=0o600,
    )
    hippo_roots = p11_runtime._prepare_hipporag_inputs(
        root=root, plans=expanded_plans, items=items, corpora=corpora
    )
    intents = _self_hashed(
        {
            "cross_encoder_input_file_sha256": _file_sha256(cross_input),
            "items": [
                {
                    "base_pool": list(plan_row.local.base_pool),
                    "expanded_pool": list(plan_row.expanded.expanded_pool),
                    "family": item.family,
                    "hipporag_input_file_sha256": _file_sha256(item_root / "input.json"),
                    "item_key": item.item_key,
                    "ordinal": item.ordinal,
                }
                for item, plan_row, item_root in zip(items, expanded_plans, hippo_roots)
            ],
            "qwen_projected_pack_sha256": qwen_receipt["projected_pack_sha256"],
            "schema": INTENT_SCHEMA,
        }
    )
    _write_json(root / "action.intents.json", intents)
    counter = _Counter()
    completions: dict[int, Mapping[str, Any]] = {}
    cross_receipt: Mapping[str, Any] | None = None
    cross_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=CONCURRENCY + 1) as executor:
        cross_future: Future[Any] = executor.submit(
            _run_cross,
            base=base,
            root=root,
            python=python,
            input_path=cross_input,
            output_path=cross_output,
        )
        hippo_futures = {
            executor.submit(
                _run_hipporag,
                base=base,
                item_root=item_root,
                candidate_rows=plan_row.local.base_pool,
                python=python,
                baseline=baseline,
                ordinal=item.ordinal,
                counter=counter,
            ): item.ordinal
            for item, plan_row, item_root in zip(items, expanded_plans, hippo_roots)
        }
        for future in as_completed([cross_future, *hippo_futures]):
            if future is cross_future:
                try:
                    cross_receipt = future.result()
                except BaseException as exc:
                    cross_error = exc
                continue
            ordinal = hippo_futures[future]
            try:
                completions[ordinal] = future.result()
            except BaseException as exc:
                completions[ordinal] = _self_hashed(
                    {
                        "failure_sha256": hashlib.sha256(
                            (type(exc).__name__ + "\n" + str(exc)).encode()
                        ).hexdigest(),
                        "ordinal": ordinal,
                        "schema": COMPLETION_SCHEMA,
                        "status": "failed_before_completion_receipt",
                    }
                )
    if cross_error is not None or cross_receipt is None:
        raise P15RemoteRuntimeError("P15 cross-encoder execution failed") from cross_error
    if set(completions) != set(range(contract.ATTEMPT_COUNT)) or counter.current != 0:
        raise P15RemoteRuntimeError("P15 remote action accounting drifted")
    terminal = tuple(
        ordinal for ordinal, completion in sorted(completions.items())
        if completion.get("status") == "terminal"
    )
    converted = tuple(
        p14.RuntimeItem(
            ordinal=item.ordinal,
            family=item.family,
            attempt_ordinal=item.attempt_ordinal,
            family_hmac_position=item.family_hmac_position,
            item_key=item.item_key,
            query=item.query,
            source_query_id=item.source_query_id,
            excluded_ids=item.excluded_ids,
        )
        for item in items
    )
    capacity, selected_p14, terminal_counts = p14.select_complete_cases(converted, terminal)
    selected_ordinals = {item.ordinal for item in selected_p14}
    selection = _self_hashed(
        {
            "capacity_passed": capacity,
            "items": [
                {
                    "attempt_ordinal": item.attempt_ordinal,
                    "failure_sha256": completions[item.ordinal].get("failure_sha256"),
                    "family": item.family,
                    "item_key": item.item_key,
                    "ordinal": item.ordinal,
                    "selected": item.ordinal in selected_ordinals,
                    "terminal": completions[item.ordinal].get("status") == "terminal",
                }
                for item in items
            ],
            "schema": contract.SELECTION_SCHEMA,
            "terminal_counts_by_family": dict(terminal_counts),
        }
    )
    selection_path = root / "complete_case.selection.json"
    _write_json(selection_path, selection)
    action_binding = None
    if capacity:
        cross = p11_runtime.train.cross_contract.parse_output(cross_output.read_bytes())
        cross_rows = {row["ordinal"]: row for row in cross["items"]}
        action_rows = []
        for selected_item in selected_p14:
            item = items[selected_item.ordinal]
            plan_row = expanded_plans[item.ordinal]
            cross_row = cross_rows[item.ordinal]
            agent_rows = p11_runtime.p11.rank_p11(
                expanded_pool=plan_row.expanded.expanded_pool,
                raw_top10=plan_row.local.raw_rows,
                cross_encoder_relation_scores=cross_row["relation_scores_quantized"],
                cross_encoder_mechanism_scores=cross_row["mechanism_scores_quantized"],
            )
            hippo_rows = completions[item.ordinal]["top_rows"]
            ids = corpora[item.family].ids
            action_rows.append(
                {
                    "attempt_ordinal": item.attempt_ordinal,
                    "family": item.family,
                    "family_HMAC_position": item.family_hmac_position,
                    "item_key": item.item_key,
                    "ordinal": item.ordinal,
                    "source_query_id": item.source_query_id,
                    "Agent_document_ids": [ids[row] for row in agent_rows],
                    "Agent_rows": list(agent_rows),
                    "HippoRAG_document_ids": [ids[row] for row in hippo_rows],
                    "HippoRAG_rows": list(hippo_rows),
                    "RAW_document_ids": [ids[row] for row in plan_row.local.raw_rows],
                    "RAW_rows": list(plan_row.local.raw_rows),
                }
            )
        actions = _self_hashed(
            {
                "active_Agent": candidate.CANDIDATE_NAME,
                "complete_case_selection_pack_sha256": selection["pack_sha256"],
                "intent_pack_sha256": intents["pack_sha256"],
                "item_count": contract.SELECTED_COUNT,
                "items": action_rows,
                "schema": contract.ACTION_SCHEMA,
            }
        )
        action_path = root / "three_arm.actions.json"
        _write_json(action_path, actions)
        action_binding = {
            "file_sha256": _file_sha256(action_path),
            "pack_sha256": actions["pack_sha256"],
        }
    external_denied = (
        qwen_receipt["network_audit"]["denied_external_network_syscall_count"]
        + cross_receipt["network_audit"]["denied_external_network_syscall_count"]
        + sum(
            completion.get("network_audit", {}).get(
                "denied_external_network_syscall_count", 0
            )
            for completion in completions.values()
        )
    )
    if external_denied != 0:
        raise P15RemoteRuntimeError("P15 external network attempt was observed")
    result = _self_hashed(
        {
            "action_binding": action_binding,
            "capacity_passed": capacity,
            "corpus_tensor_bindings": tensor_bindings,
            "cross_encoder_receipt": cross_receipt,
            "external_network_call_count": 0,
            "HippoRAG_peak_process_concurrency": counter.peak,
            "local_P14_output_reuse_count": 0,
            "plan_pack_sha256": plan["pack_sha256"],
            "qwen_receipt": qwen_receipt,
            "remote_hostname": socket.gethostname(),
            "remote_runtime_fingerprint_self_sha256": fingerprint["self_sha256"],
            "schema": contract.REMOTE_RESULT_SCHEMA,
            "selection_binding": {
                "file_sha256": _file_sha256(selection_path),
                "pack_sha256": selection["pack_sha256"],
            },
            "status": "remote_action_sealed" if capacity else "remote_capacity_failed",
        }
    )
    _write_json(root / "remote_action.result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = run(arguments.plan)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
