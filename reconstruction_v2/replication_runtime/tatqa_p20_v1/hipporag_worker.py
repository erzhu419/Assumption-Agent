"""Private offline official-HippoRAG worker for one TAT-QA item."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import threading
import time
from types import MethodType
from typing import Any, Mapping, Sequence

from .hipporag_contract import (
    TatqaP20OfficialHippoRAGError,
    canonical_json_bytes,
    input_binding_sha256,
    output_payload,
    parse_input,
    serialize_units,
    stable_top_k,
)


OPENIE_MAX_NEW_TOKENS = 96
MAXIMUM_PROCESS_THREAD_COUNT = 2
TORCH_INTRAOP_THREAD_COUNT = 1
TORCH_INTEROP_THREAD_COUNT = 1
NATIVE_THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
# Compatibility alias for callers that used the former single-dimensional
# setting.  It now means the actually configured intra-op count, not an
# unmeasured maximum.
TORCH_THREAD_COUNT = TORCH_INTRAOP_THREAD_COUNT


class _SingleWorkerOpenIEExecutor(_ThreadPoolExecutor):
    """Preserve upstream OpenIE semantics while bounding its worker pool."""

    def __init__(self, max_workers=None, *args, **kwargs):
        if max_workers not in (None, 1):
            raise TatqaP20OfficialHippoRAGError(
                "upstream OpenIE requested an unbounded worker pool"
            )
        super().__init__(max_workers=1, *args, **kwargs)


def _install_single_worker_openie_executor() -> None:
    # Official online OpenIE constructs ``ThreadPoolExecutor()`` twice with the
    # platform default (up to 32 threads).  That would violate the frozen
    # two-thread process maximum even though inference is lock-serialized.
    # Rebinding only that imported executor changes scheduling, not retrieval,
    # prompts, model calls, graph construction, or ranking.
    from hipporag.information_extraction import openie_openai

    observed = getattr(openie_openai, "ThreadPoolExecutor", None)
    if observed not in (_ThreadPoolExecutor, _SingleWorkerOpenIEExecutor):
        raise TatqaP20OfficialHippoRAGError("official OpenIE executor drifted")
    openie_openai.ThreadPoolExecutor = _SingleWorkerOpenIEExecutor


def _require_native_thread_environment() -> None:
    if any(os.environ.get(key) != "1" for key in NATIVE_THREAD_ENVIRONMENT_KEYS):
        raise TatqaP20OfficialHippoRAGError(
            "native BLAS/OpenMP thread environment drifted"
        )


def _process_thread_count(pid: int) -> int:
    """Read the live Linux OS-thread count for ``pid`` from procfs."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise TatqaP20OfficialHippoRAGError("thread-monitor PID drifted")
    task_root = Path("/proc") / str(pid) / "task"
    try:
        count = sum(1 for row in os.scandir(task_root) if row.name.isdecimal())
    except OSError as exc:
        raise TatqaP20OfficialHippoRAGError("worker thread count is unavailable") from exc
    if count <= 0:
        raise TatqaP20OfficialHippoRAGError("worker thread count drifted")
    return count


class _ProcessThreadPeakMonitor:
    """Measure worker OS threads from a separate process.

    A monitoring thread would itself increase the value being audited.  This
    monitor therefore forks before torch/HippoRAG are imported and samples the
    worker's ``/proc/<pid>/task`` directory from the child process.  The parent
    does not proceed until the child has completed its first real sample.
    """

    def __init__(self, *, pid: int, monitor_pid: int, stop_fd: int, result_fd: int):
        self.pid = pid
        self.monitor_pid = monitor_pid
        self.stop_fd = stop_fd
        self.result_fd = result_fd
        self._stopped = False

    @classmethod
    def start(cls, pid: int) -> "_ProcessThreadPeakMonitor":
        if not hasattr(os, "fork"):
            raise TatqaP20OfficialHippoRAGError("process thread monitor is unavailable")
        stop_read, stop_write = os.pipe()
        result_read, result_write = os.pipe()
        ready_read, ready_write = os.pipe()
        try:
            monitor_pid = os.fork()
        except OSError as exc:
            for descriptor in (
                stop_read,
                stop_write,
                result_read,
                result_write,
                ready_read,
                ready_write,
            ):
                os.close(descriptor)
            raise TatqaP20OfficialHippoRAGError(
                "process thread monitor could not start"
            ) from exc
        if monitor_pid == 0:  # pragma: no branch - exercised in the forked child
            os.close(stop_write)
            os.close(result_read)
            os.close(ready_read)
            try:
                peak = _process_thread_count(pid)
                os.write(ready_write, b"1")
                os.close(ready_write)
                while True:
                    readable, _, _ = select.select([stop_read], [], [], 0.002)
                    peak = max(peak, _process_thread_count(pid))
                    if readable:
                        if os.read(stop_read, 1) != b"1":
                            raise RuntimeError("thread-monitor stop signal drifted")
                        # One final sample closes the signal-to-observation race.
                        peak = max(peak, _process_thread_count(pid))
                        os.write(result_write, str(peak).encode("ascii"))
                        os._exit(0)
            except BaseException:
                try:
                    os.write(ready_write, b"0")
                except OSError:
                    pass
                try:
                    os.write(result_write, b"ERROR")
                except OSError:
                    pass
                os._exit(91)
        os.close(stop_read)
        os.close(result_write)
        os.close(ready_write)
        try:
            ready = os.read(ready_read, 1)
        finally:
            os.close(ready_read)
        if ready != b"1":
            os.close(stop_write)
            os.close(result_read)
            os.waitpid(monitor_pid, 0)
            raise TatqaP20OfficialHippoRAGError(
                "process thread monitor failed its first sample"
            )
        return cls(
            pid=pid,
            monitor_pid=monitor_pid,
            stop_fd=stop_write,
            result_fd=result_read,
        )

    def stop(self) -> int:
        if self._stopped:
            raise TatqaP20OfficialHippoRAGError("process thread monitor replayed")
        self._stopped = True
        try:
            os.write(self.stop_fd, b"1")
        finally:
            os.close(self.stop_fd)
        raw = b""
        try:
            while True:
                block = os.read(self.result_fd, 64)
                if not block:
                    break
                raw += block
        finally:
            os.close(self.result_fd)
        waited_pid, wait_status = os.waitpid(self.monitor_pid, 0)
        if waited_pid != self.monitor_pid or wait_status != 0:
            raise TatqaP20OfficialHippoRAGError("process thread monitor failed")
        try:
            peak = int(raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise TatqaP20OfficialHippoRAGError(
                "process thread peak receipt drifted"
            ) from exc
        if peak <= 0:
            raise TatqaP20OfficialHippoRAGError("process thread peak receipt drifted")
        return peak


def _install_completion_only_backend(core: object) -> None:
    """Apply the already-attested local-backend compatibility boundary."""

    llm = getattr(core, "llm_model", None)
    if llm is None or llm.__class__.__name__ != "TransformersLLM":
        raise TatqaP20OfficialHippoRAGError("official local LLM backend drifted")
    if any(getattr(llm, name, None) is None for name in ("model", "tokenizer", "llm_config")):
        raise TatqaP20OfficialHippoRAGError("official local LLM state is incomplete")
    lock = threading.Lock()

    def compatible_infer(self: object, messages: Sequence[Mapping[str, str]], **kwargs: Any):
        import torch
        from hipporag.llm.transformers_llm import (
            convert_text_chat_messages_to_input_ids,
        )

        params = deepcopy(getattr(self, "llm_config").generate_params)
        params.update(kwargs)
        maximum = params.get("max_tokens", OPENIE_MAX_NEW_TOKENS)
        if maximum != OPENIE_MAX_NEW_TOKENS:
            raise TatqaP20OfficialHippoRAGError("OpenIE token budget drifted")
        tokenizer = getattr(self, "tokenizer")
        model = getattr(self, "model")
        input_ids = convert_text_chat_messages_to_input_ids(
            list(messages), tokenizer
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        with lock, torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=maximum,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0, input_ids.shape[1] :]
        response = tokenizer.decode(completion_ids, skip_special_tokens=True)
        return (
            response,
            {
                "completion_tokens": int(completion_ids.numel()),
                "finish_reason": (
                    "length" if completion_ids.numel() >= maximum else "stop"
                ),
                "prompt_tokens": int(input_ids.shape[1]),
            },
            False,
        )

    llm.llm_config.generate_params["max_tokens"] = OPENIE_MAX_NEW_TOKENS
    llm.infer = MethodType(compatible_infer, llm)


def _build_core(
    *, save_dir: Path, llm_model: Path, embedding_model: Path, unit_count: int
) -> object:
    import torch
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    _install_single_worker_openie_executor()
    torch.set_num_threads(TORCH_INTRAOP_THREAD_COUNT)
    torch.set_num_interop_threads(TORCH_INTEROP_THREAD_COUNT)
    if (
        torch.get_num_threads() != TORCH_INTRAOP_THREAD_COUNT
        or torch.get_num_interop_threads() != TORCH_INTEROP_THREAD_COUNT
    ):
        raise TatqaP20OfficialHippoRAGError("torch thread configuration drifted")
    torch.manual_seed(0)
    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name="Transformers/" + str(llm_model),
        embedding_model_name="Transformers/" + str(embedding_model),
        openie_mode="online",
        max_new_tokens=OPENIE_MAX_NEW_TOKENS,
        retrieval_top_k=unit_count,
        qa_top_k=5,
        force_index_from_scratch=True,
        save_openie=True,
        temperature=0,
        seed=0,
    )
    core = HippoRAG(global_config=config)
    _install_completion_only_backend(core)
    return core


def retrieve_with_core(*, core: object, query: str, units: Sequence[Any]) -> dict[str, Any]:
    serialized = serialize_units(units)
    mapping = {text: row for text, row in zip(serialized, units)}
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise TatqaP20OfficialHippoRAGError("official core methods unavailable")
    index(list(serialized))
    rows = retrieve([query], num_to_retrieve=len(units))
    if not isinstance(rows, list) or len(rows) != 1:
        raise TatqaP20OfficialHippoRAGError("official query result drifted")
    solution = rows[0]
    top = stable_top_k(
        retrieved_documents=getattr(solution, "docs", None),
        retrieved_scores=getattr(solution, "doc_scores", None),
        document_to_unit=mapping,
    )
    graph = getattr(core, "graph", None)
    vcount = getattr(graph, "vcount", None)
    ecount = getattr(graph, "ecount", None)
    if not callable(vcount) or not callable(ecount):
        raise TatqaP20OfficialHippoRAGError("official graph counters unavailable")
    return output_payload(
        top_unit_ids=top,
        graph_nodes=int(vcount()),
        graph_edges=int(ecount()),
        unit_count=len(units),
        input_sha256=input_binding_sha256(query, units),
    )


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.input.is_symlink() or not arguments.input.is_file():
        raise TatqaP20OfficialHippoRAGError("worker input is unavailable")
    if arguments.index_root.exists() or arguments.index_root.is_symlink():
        raise TatqaP20OfficialHippoRAGError("index root already exists")
    _require_native_thread_environment()
    arguments.index_root.mkdir(mode=0o700)
    query, units = parse_input(arguments.input.read_bytes())
    monitor = _ProcessThreadPeakMonitor.start(os.getpid())
    model_execution_started_monotonic_ns = time.monotonic_ns()
    try:
        core = _build_core(
            save_dir=arguments.index_root,
            llm_model=arguments.llm_model,
            embedding_model=arguments.embedding_model,
            unit_count=len(units),
        )
        payload = retrieve_with_core(core=core, query=query, units=units)
        model_execution_finished_monotonic_ns = time.monotonic_ns()
    finally:
        observed_process_thread_peak = monitor.stop()
    if (
        isinstance(model_execution_started_monotonic_ns, bool)
        or isinstance(model_execution_finished_monotonic_ns, bool)
        or not isinstance(model_execution_started_monotonic_ns, int)
        or not isinstance(model_execution_finished_monotonic_ns, int)
        or model_execution_started_monotonic_ns < 0
        or model_execution_finished_monotonic_ns
        <= model_execution_started_monotonic_ns
    ):
        raise TatqaP20OfficialHippoRAGError("model execution interval drifted")
    if observed_process_thread_peak > MAXIMUM_PROCESS_THREAD_COUNT:
        raise TatqaP20OfficialHippoRAGError("worker OS thread peak exceeded the maximum")
    _write_exclusive(arguments.output, payload)
    print(
        json.dumps(
            {
                "configured_torch_interop_threads": TORCH_INTEROP_THREAD_COUNT,
                "configured_torch_intraop_threads": TORCH_INTRAOP_THREAD_COUNT,
                "graph_edge_count": payload["graph_edge_count"],
                "graph_node_count": payload["graph_node_count"],
                "model_execution_finished_monotonic_ns": (
                    model_execution_finished_monotonic_ns
                ),
                "model_execution_started_monotonic_ns": (
                    model_execution_started_monotonic_ns
                ),
                "observed_process_thread_peak": observed_process_thread_peak,
                "status": "passed",
                "unit_count": len(units),
                "worker_pid": os.getpid(),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAXIMUM_PROCESS_THREAD_COUNT",
    "NATIVE_THREAD_ENVIRONMENT_KEYS",
    "OPENIE_MAX_NEW_TOKENS",
    "TORCH_INTEROP_THREAD_COUNT",
    "TORCH_INTRAOP_THREAD_COUNT",
    "TORCH_THREAD_COUNT",
    "retrieve_with_core",
]
