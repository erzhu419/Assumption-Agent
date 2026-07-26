"""One-contract, 22-query official HippoRAG worker for MAUD extraction P1.

The worker has no benchmark loader, answer field, score, evaluator, API, or
network surface.  It receives one opaque contract work ID, one ordered corpus
of already-canonical MAUD passage documents, and exactly 22 opaque queries.
It creates one fresh official HippoRAG index, indexes the corpus once, and
retrieves all queries in one call.  The output contains only passage ordinals
and content-free graph counts.

Model arguments must be short single-component aliases available in the
process working directory.  Absolute model paths are deliberately
unrepresentable on the command line, preventing HippoRAG from constructing a
working-directory basename longer than Linux ``NAME_MAX``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import select
import threading
from types import MethodType
from typing import Any, Mapping, Sequence


VERSION = "maud_extraction_p1_official_hipporag_contract_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
QUERY_COUNT = 22
TOP_K = 5
MIN_PASSAGE_COUNT = TOP_K
MAX_PASSAGE_COUNT = 8_192
MAX_PASSAGE_CHARACTERS = 20_000
MAX_QUERY_CHARACTERS = 250_000
MAX_OPAQUE_ID_CHARACTERS = 1_024
OPENIE_MAX_NEW_TOKENS = 96
MAXIMUM_PROCESS_THREAD_COUNT = 2
TORCH_INTRAOP_THREAD_COUNT = 1
TORCH_INTEROP_THREAD_COUNT = 1
TORCH_THREAD_COUNT = TORCH_INTRAOP_THREAD_COUNT
NATIVE_THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
MAX_MODEL_ALIAS_CHARACTERS = 64

INPUT_KEYS = frozenset(
    {
        "contract_work_id",
        "corpus_sha256",
        "documents",
        "queries",
        "schema",
    }
)
DOCUMENT_KEYS = frozenset({"ordinal", "text"})
QUERY_KEYS = frozenset({"ordinal", "text", "work_id"})
OUTPUT_KEYS = frozenset(
    {
        "contract_work_id",
        "corpus_sha256",
        "graph_edge_count",
        "graph_node_count",
        "passage_count",
        "rows",
        "schema",
    }
)
OUTPUT_ROW_KEYS = frozenset(
    {"query_ordinal", "top5_passage_ordinals", "work_id"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_ALIAS = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_MODEL_ALIAS_CHARACTERS - 1}}}\Z"
)


class MaudOfficialHippoRAGError(RuntimeError):
    """The isolated official-core contract failed closed."""


class _SingleWorkerOpenIEExecutor(_ThreadPoolExecutor):
    """Preserve official OpenIE semantics with one bounded worker."""

    def __init__(self, max_workers=None, *args, **kwargs):  # noqa: ANN001
        if max_workers not in (None, 1):
            raise MaudOfficialHippoRAGError(
                "official OpenIE requested an unbounded worker pool"
            )
        super().__init__(max_workers=1, *args, **kwargs)


def _install_single_worker_openie_executor() -> None:
    from hipporag.information_extraction import openie_openai

    observed = getattr(openie_openai, "ThreadPoolExecutor", None)
    if observed not in (_ThreadPoolExecutor, _SingleWorkerOpenIEExecutor):
        raise MaudOfficialHippoRAGError("official OpenIE executor drifted")
    openie_openai.ThreadPoolExecutor = _SingleWorkerOpenIEExecutor


def _require_native_thread_environment() -> None:
    if any(
        os.environ.get(key) != "1"
        for key in NATIVE_THREAD_ENVIRONMENT_KEYS
    ):
        raise MaudOfficialHippoRAGError(
            "native BLAS/OpenMP thread environment drifted"
        )


def _process_thread_count(pid: int) -> int:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise MaudOfficialHippoRAGError("thread-monitor PID drifted")
    try:
        count = sum(
            1
            for row in os.scandir(Path("/proc") / str(pid) / "task")
            if row.name.isdecimal()
        )
    except OSError as exc:
        raise MaudOfficialHippoRAGError(
            "worker thread count is unavailable"
        ) from exc
    if count <= 0:
        raise MaudOfficialHippoRAGError("worker thread count drifted")
    return count


class _ProcessThreadPeakMonitor:
    """Measure the worker's OS-thread peak from a separate process."""

    def __init__(
        self, *, monitor_pid: int, stop_fd: int, result_fd: int
    ) -> None:
        self.monitor_pid = monitor_pid
        self.stop_fd = stop_fd
        self.result_fd = result_fd
        self._stopped = False

    @classmethod
    def start(cls, pid: int) -> "_ProcessThreadPeakMonitor":
        if not hasattr(os, "fork"):
            raise MaudOfficialHippoRAGError(
                "process thread monitor is unavailable"
            )
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
            raise MaudOfficialHippoRAGError(
                "process thread monitor could not start"
            ) from exc
        if monitor_pid == 0:  # pragma: no branch - forked monitor
            os.close(stop_write)
            os.close(result_read)
            os.close(ready_read)
            try:
                peak = _process_thread_count(pid)
                os.write(ready_write, b"1")
                os.close(ready_write)
                while True:
                    readable, _, _ = select.select(
                        [stop_read], [], [], 0.002
                    )
                    peak = max(peak, _process_thread_count(pid))
                    if readable:
                        if os.read(stop_read, 1) != b"1":
                            raise RuntimeError(
                                "thread-monitor stop signal drifted"
                            )
                        peak = max(peak, _process_thread_count(pid))
                        os.write(
                            result_write, str(peak).encode("ascii")
                        )
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
            raise MaudOfficialHippoRAGError(
                "process thread monitor failed its first sample"
            )
        return cls(
            monitor_pid=monitor_pid,
            stop_fd=stop_write,
            result_fd=result_read,
        )

    def stop(self) -> int:
        if self._stopped:
            raise MaudOfficialHippoRAGError(
                "process thread monitor replayed"
            )
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
            raise MaudOfficialHippoRAGError(
                "process thread monitor failed"
            )
        try:
            peak = int(raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise MaudOfficialHippoRAGError(
                "process thread peak receipt drifted"
            ) from exc
        if peak <= 0:
            raise MaudOfficialHippoRAGError(
                "process thread peak receipt drifted"
            )
        return peak


@dataclass(frozen=True)
class PassageDocument:
    ordinal: int
    text: str


@dataclass(frozen=True)
class QueryDocument:
    ordinal: int
    work_id: str
    text: str


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    """Return the exact ASCII JSON envelope used for private IPC."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MaudOfficialHippoRAGError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def _bounded_text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise MaudOfficialHippoRAGError(f"{field} is invalid")
    return value


def _opaque_id(value: object, field: str) -> str:
    return _bounded_text(value, field, MAX_OPAQUE_ID_CHARACTERS)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MaudOfficialHippoRAGError(f"{field} is invalid")
    return value


def canonical_passage_document(*, ordinal: object, text: object) -> str:
    """Serialize one gold-independent passage identically for all arms."""

    passage_text = _bounded_text(
        text, "passage text", MAX_PASSAGE_CHARACTERS
    )
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 0 <= ordinal < MAX_PASSAGE_COUNT
    ):
        raise MaudOfficialHippoRAGError("passage ordinal is invalid")
    return canonical_json_bytes(
        {
            "text": passage_text,
            "title": f"MAUD passage {ordinal:06d}",
        },
    ).decode("ascii")


def _validate_documents(value: object) -> tuple[PassageDocument, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MaudOfficialHippoRAGError("documents are not a sequence")
    if not MIN_PASSAGE_COUNT <= len(value) <= MAX_PASSAGE_COUNT:
        raise MaudOfficialHippoRAGError(
            "passage count is outside frozen bounds"
        )
    rows: list[PassageDocument] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != DOCUMENT_KEYS:
            raise MaudOfficialHippoRAGError("document shape drifted")
        ordinal = raw.get("ordinal")
        if type(ordinal) is not int or ordinal != position:
            raise MaudOfficialHippoRAGError(
                "document ordinals are not canonical"
            )
        rows.append(
            PassageDocument(
                ordinal=ordinal,
                text=_bounded_text(
                    raw.get("text"),
                    f"documents[{position}].text",
                    MAX_PASSAGE_CHARACTERS,
                ),
            )
        )
    documents = tuple(rows)
    if len({row.text for row in documents}) != len(documents):
        raise MaudOfficialHippoRAGError(
            "serialized passage documents are not unique"
        )
    return documents


def _validate_queries(value: object) -> tuple[QueryDocument, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MaudOfficialHippoRAGError("queries are not a sequence")
    if len(value) != QUERY_COUNT:
        raise MaudOfficialHippoRAGError("query count drifted")
    rows: list[QueryDocument] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != QUERY_KEYS:
            raise MaudOfficialHippoRAGError("query shape drifted")
        ordinal = raw.get("ordinal")
        if type(ordinal) is not int or ordinal != position:
            raise MaudOfficialHippoRAGError(
                "query ordinals are not canonical"
            )
        rows.append(
            QueryDocument(
                ordinal=ordinal,
                work_id=_opaque_id(
                    raw.get("work_id"), f"queries[{position}].work_id"
                ),
                text=_bounded_text(
                    raw.get("text"),
                    f"queries[{position}].text",
                    MAX_QUERY_CHARACTERS,
                ),
            )
        )
    queries = tuple(rows)
    if len({row.work_id for row in queries}) != QUERY_COUNT:
        raise MaudOfficialHippoRAGError("query work IDs are not unique")
    return queries


def _corpus_projection(
    documents: Sequence[PassageDocument],
) -> list[dict[str, object]]:
    return [
        {"ordinal": row.ordinal, "text": row.text} for row in documents
    ]


def corpus_sha256(documents: object) -> str:
    """Hash the complete ordered passage corpus, independent of queries."""

    rows = _validate_documents(documents)
    return hashlib.sha256(
        canonical_json_bytes(_corpus_projection(rows), newline=False)
    ).hexdigest()


def validate_input(
    value: object,
) -> tuple[
    str,
    str,
    tuple[PassageDocument, ...],
    tuple[QueryDocument, ...],
]:
    if (
        not isinstance(value, Mapping)
        or set(value) != INPUT_KEYS
        or value.get("schema") != INPUT_SCHEMA
    ):
        raise MaudOfficialHippoRAGError("input envelope drifted")
    contract_work_id = _opaque_id(
        value.get("contract_work_id"), "contract work ID"
    )
    documents = _validate_documents(value.get("documents"))
    queries = _validate_queries(value.get("queries"))
    claimed = _sha256(value.get("corpus_sha256"), "corpus SHA-256")
    expected = hashlib.sha256(
        canonical_json_bytes(
            _corpus_projection(documents), newline=False
        )
    ).hexdigest()
    if claimed != expected:
        raise MaudOfficialHippoRAGError("corpus SHA-256 mismatched")
    return contract_work_id, claimed, documents, queries


def input_payload(
    *,
    contract_work_id: object,
    documents: object,
    queries: object,
) -> dict[str, object]:
    """Build one exact label-free worker envelope."""

    document_rows = _validate_documents(documents)
    query_rows = _validate_queries(queries)
    value = {
        "contract_work_id": _opaque_id(
            contract_work_id, "contract work ID"
        ),
        "corpus_sha256": hashlib.sha256(
            canonical_json_bytes(
                _corpus_projection(document_rows), newline=False
            )
        ).hexdigest(),
        "documents": _corpus_projection(document_rows),
        "queries": [
            {
                "ordinal": row.ordinal,
                "text": row.text,
                "work_id": row.work_id,
            }
            for row in query_rows
        ],
        "schema": INPUT_SCHEMA,
    }
    validate_input(value)
    return value


def _stable_top5(
    *,
    retrieved_documents: object,
    retrieved_scores: object,
    document_to_ordinal: Mapping[str, int],
) -> tuple[int, ...]:
    if isinstance(retrieved_documents, (str, bytes)) or isinstance(
        retrieved_scores, (str, bytes)
    ):
        raise MaudOfficialHippoRAGError("official result is malformed")
    try:
        documents = list(retrieved_documents)  # type: ignore[arg-type]
        scores = list(retrieved_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise MaudOfficialHippoRAGError(
            "official result is not iterable"
        ) from exc
    count = len(document_to_ordinal)
    if len(documents) != count or len(scores) != count:
        raise MaudOfficialHippoRAGError(
            "official result did not return the complete corpus"
        )
    ranked: list[tuple[float, int]] = []
    seen: set[str] = set()
    for document, score in zip(documents, scores):
        if (
            not isinstance(document, str)
            or document not in document_to_ordinal
            or document in seen
        ):
            raise MaudOfficialHippoRAGError(
                "official result document set drifted"
            )
        if isinstance(score, bool) or not isinstance(score, Real):
            raise MaudOfficialHippoRAGError(
                "official result score is not numeric"
            )
        numeric = float(score)
        if not math.isfinite(numeric):
            raise MaudOfficialHippoRAGError(
                "official result score is not finite"
            )
        seen.add(document)
        ranked.append((numeric, document_to_ordinal[document]))
    if seen != set(document_to_ordinal):
        raise MaudOfficialHippoRAGError(
            "official result omitted a passage"
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return tuple(ordinal for _score, ordinal in ranked[:TOP_K])


def _output_payload(
    *,
    contract_work_id: str,
    corpus_hash: str,
    passage_count: int,
    queries: Sequence[QueryDocument],
    top5_rows: Sequence[Sequence[int]],
    graph_nodes: object,
    graph_edges: object,
) -> dict[str, object]:
    if (
        type(passage_count) is not int
        or not MIN_PASSAGE_COUNT <= passage_count <= MAX_PASSAGE_COUNT
    ):
        raise MaudOfficialHippoRAGError("output passage count drifted")
    if len(queries) != QUERY_COUNT or len(top5_rows) != QUERY_COUNT:
        raise MaudOfficialHippoRAGError("output query count drifted")
    rows = []
    for query, raw_top5 in zip(queries, top5_rows):
        top5 = tuple(raw_top5)
        if (
            len(top5) != TOP_K
            or len(set(top5)) != TOP_K
            or any(
                type(value) is not int
                or not 0 <= value < passage_count
                for value in top5
            )
        ):
            raise MaudOfficialHippoRAGError(
                "top-five passage ordinals drifted"
            )
        rows.append(
            {
                "query_ordinal": query.ordinal,
                "top5_passage_ordinals": list(top5),
                "work_id": query.work_id,
            }
        )
    for value, field in (
        (graph_nodes, "graph nodes"),
        (graph_edges, "graph edges"),
    ):
        if type(value) is not int or value < 0:
            raise MaudOfficialHippoRAGError(f"{field} is invalid")
    return {
        "contract_work_id": _opaque_id(
            contract_work_id, "contract work ID"
        ),
        "corpus_sha256": _sha256(corpus_hash, "corpus SHA-256"),
        "graph_edge_count": graph_edges,
        "graph_node_count": graph_nodes,
        "passage_count": passage_count,
        "rows": rows,
        "schema": OUTPUT_SCHEMA,
    }


def parse_output(raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise MaudOfficialHippoRAGError("worker output is not bytes")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaudOfficialHippoRAGError("worker output is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != OUTPUT_KEYS
        or value.get("schema") != OUTPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise MaudOfficialHippoRAGError(
            "worker output envelope drifted"
        )
    rows = value.get("rows")
    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or len(rows) != QUERY_COUNT
    ):
        raise MaudOfficialHippoRAGError("worker output rows drifted")
    query_rows: list[QueryDocument] = []
    top5_rows: list[Sequence[int]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != OUTPUT_ROW_KEYS:
            raise MaudOfficialHippoRAGError(
                "worker output row shape drifted"
            )
        if row.get("query_ordinal") != position:
            raise MaudOfficialHippoRAGError(
                "worker output query ordinals drifted"
            )
        query_rows.append(
            QueryDocument(
                ordinal=position,
                work_id=_opaque_id(row.get("work_id"), "query work ID"),
                text="private-output-placeholder",
            )
        )
        top5 = row.get("top5_passage_ordinals")
        if isinstance(top5, (str, bytes)) or not isinstance(top5, Sequence):
            raise MaudOfficialHippoRAGError(
                "worker output top-five drifted"
            )
        top5_rows.append(top5)  # type: ignore[arg-type]
    return _output_payload(
        contract_work_id=str(value.get("contract_work_id")),
        corpus_hash=str(value.get("corpus_sha256")),
        passage_count=value.get("passage_count"),  # type: ignore[arg-type]
        queries=query_rows,
        top5_rows=top5_rows,
        graph_nodes=value.get("graph_node_count"),
        graph_edges=value.get("graph_edge_count"),
    )


def retrieve_contract_with_core(
    *, core: object, payload: Mapping[str, object]
) -> dict[str, object]:
    """Index one contract once and retrieve its complete 22-query batch."""

    contract_work_id, corpus_hash, documents, queries = validate_input(payload)
    mapping = {row.text: row.ordinal for row in documents}
    if len(mapping) != len(documents):
        raise MaudOfficialHippoRAGError(
            "passage content addressing collided"
        )
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise MaudOfficialHippoRAGError(
            "official core methods are unavailable"
        )
    serialized = [row.text for row in documents]
    index(serialized)
    raw_solutions = retrieve(
        [row.text for row in queries],
        num_to_retrieve=len(documents),
    )
    if not isinstance(raw_solutions, list) or len(raw_solutions) != QUERY_COUNT:
        raise MaudOfficialHippoRAGError(
            "official query batch result drifted"
        )
    top5_rows = [
        _stable_top5(
            retrieved_documents=getattr(solution, "docs", None),
            retrieved_scores=getattr(solution, "doc_scores", None),
            document_to_ordinal=mapping,
        )
        for solution in raw_solutions
    ]
    graph = getattr(core, "graph", None)
    vcount = getattr(graph, "vcount", None)
    ecount = getattr(graph, "ecount", None)
    if not callable(vcount) or not callable(ecount):
        raise MaudOfficialHippoRAGError(
            "official graph counters are unavailable"
        )
    return _output_payload(
        contract_work_id=contract_work_id,
        corpus_hash=corpus_hash,
        passage_count=len(documents),
        queries=queries,
        top5_rows=top5_rows,
        graph_nodes=vcount(),
        graph_edges=ecount(),
    )


def _validate_model_alias(value: object, label: str) -> str:
    if not isinstance(value, str) or _MODEL_ALIAS.fullmatch(value) is None:
        raise MaudOfficialHippoRAGError(f"{label} alias is invalid")
    if (
        "/" in value
        or "\\" in value
        or ".." in value
        or Path(value).is_absolute()
    ):
        raise MaudOfficialHippoRAGError(f"{label} alias is invalid")
    try:
        available = Path(value).is_dir()
    except OSError:
        available = False
    if not available:
        raise MaudOfficialHippoRAGError(f"{label} alias is unavailable")
    return value


def _install_completion_only_backend(core: object) -> None:
    llm = getattr(core, "llm_model", None)
    if llm is None or llm.__class__.__name__ != "TransformersLLM":
        raise MaudOfficialHippoRAGError(
            "official local LLM backend drifted"
        )
    if any(
        getattr(llm, field, None) is None
        for field in ("model", "tokenizer", "llm_config")
    ):
        raise MaudOfficialHippoRAGError(
            "official local LLM state is incomplete"
        )
    lock = threading.Lock()

    def compatible_infer(
        self: object,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ):
        import torch
        from hipporag.llm.transformers_llm import (
            convert_text_chat_messages_to_input_ids,
        )

        params = deepcopy(getattr(self, "llm_config").generate_params)
        params.update(kwargs)
        maximum = params.get("max_tokens", OPENIE_MAX_NEW_TOKENS)
        if type(maximum) is not int or maximum != OPENIE_MAX_NEW_TOKENS:
            raise MaudOfficialHippoRAGError(
                "OpenIE token budget drifted"
            )
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
        response = tokenizer.decode(
            completion_ids, skip_special_tokens=True
        )
        metadata = {
            "completion_tokens": int(completion_ids.numel()),
            "finish_reason": (
                "length"
                if completion_ids.numel() >= maximum
                else "stop"
            ),
            "prompt_tokens": int(input_ids.shape[1]),
        }
        return response, metadata, False

    llm.llm_config.generate_params["max_tokens"] = OPENIE_MAX_NEW_TOKENS
    llm.infer = MethodType(compatible_infer, llm)


def _build_core(
    *,
    save_dir: Path,
    llm_alias: str,
    embedding_alias: str,
    passage_count: int,
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
        raise MaudOfficialHippoRAGError(
            "torch thread configuration drifted"
        )
    torch.manual_seed(0)
    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name="Transformers/" + llm_alias,
        embedding_model_name="Transformers/" + embedding_alias,
        openie_mode="online",
        max_new_tokens=OPENIE_MAX_NEW_TOKENS,
        retrieval_top_k=passage_count,
        qa_top_k=TOP_K,
        force_index_from_scratch=True,
        save_openie=True,
        temperature=0,
        seed=0,
    )
    core = HippoRAG(global_config=config)
    _install_completion_only_backend(core)
    return core


def _read_input(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise MaudOfficialHippoRAGError("worker input is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaudOfficialHippoRAGError("worker input is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or canonical_json_bytes(value) != raw
    ):
        raise MaudOfficialHippoRAGError(
            "worker input is not canonical"
        )
    validate_input(value)
    return dict(value)


def _write_output(path: Path, payload: Mapping[str, object]) -> None:
    raw = canonical_json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    arguments = parser.parse_args(argv)
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise MaudOfficialHippoRAGError(
            "PYTHONDONTWRITEBYTECODE is not frozen"
        )
    _require_native_thread_environment()
    llm_alias = _validate_model_alias(arguments.llm_model, "LLM model")
    embedding_alias = _validate_model_alias(
        arguments.embedding_model, "embedding model"
    )
    if arguments.index_root.exists() or arguments.index_root.is_symlink():
        raise MaudOfficialHippoRAGError("index root already exists")
    arguments.index_root.mkdir(mode=0o700)
    payload = _read_input(arguments.input)
    _contract_id, _corpus_hash, documents, _queries = validate_input(payload)
    monitor = _ProcessThreadPeakMonitor.start(os.getpid())
    try:
        core = _build_core(
            save_dir=arguments.index_root,
            llm_alias=llm_alias,
            embedding_alias=embedding_alias,
            passage_count=len(documents),
        )
        output = retrieve_contract_with_core(core=core, payload=payload)
    finally:
        observed_process_thread_peak = monitor.stop()
    if observed_process_thread_peak > MAXIMUM_PROCESS_THREAD_COUNT:
        raise MaudOfficialHippoRAGError(
            "worker OS thread peak exceeded the maximum"
        )
    _write_output(arguments.output, output)
    print(
        json.dumps(
            {
                "graph_edge_count": output["graph_edge_count"],
                "graph_node_count": output["graph_node_count"],
                "configured_torch_interop_threads": (
                    TORCH_INTEROP_THREAD_COUNT
                ),
                "configured_torch_intraop_threads": (
                    TORCH_INTRAOP_THREAD_COUNT
                ),
                "observed_process_thread_peak": (
                    observed_process_thread_peak
                ),
                "passage_count": output["passage_count"],
                "query_count": QUERY_COUNT,
                "status": "passed",
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
