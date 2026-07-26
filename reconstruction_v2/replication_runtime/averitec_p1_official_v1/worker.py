"""One-process, one-GPU official HippoRAG comparator for AVeriTeC P1.

The worker receives only a claim-query batch and the exact QA-pair corpus
shared with Agent and RAW.  It constructs one fresh official HippoRAG index,
retrieves every query in batches of at most eight, and emits only five corpus
ordinals per opaque item plus content-free execution receipts.

The production core is the already-qualified patched local HippoRAG source.
MAUD P2 supplies the bounded single-worker OpenIE and completion-only backend;
EBM-NLP v4 supplies pre/post CUDA-residency attestation.  This module adds the
dynamic AVeriTeC corpus/query contract and a single-process build/retrieve
lifecycle.  It has no source, family, qrel, label, score, evaluator, API, or
online transport surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

from replication_runtime.ebmnlp_p1_official_v1 import worker as ebm_worker
from replication_runtime.maud_extraction_p2_official_v1 import (
    worker as qualified_base,
)
from replication_runtime.morehopqa_official_hipporag_v1 import (
    contract as global_contract,
)


VERSION = "averitec_p1_official_hipporag_worker_v1"
STUDY_ID = "AVERITEC_P1_TYPED_QA_SET_EVALUATOR_V1"
INPUT_SCHEMA = f"{VERSION}_private_input_v1"
OUTPUT_SCHEMA = f"{VERSION}_private_output_v1"
RUNTIME_SCHEMA = f"{VERSION}_private_runtime_receipt_v1"
FORMAL_BLOCK = "A_hold"
CANARY_BLOCK = "source_free_synthetic_canary"
ALLOWED_BLOCKS = frozenset({FORMAL_BLOCK, CANARY_BLOCK})
MAX_QUERY_COUNT = 512
MAX_QUERY_CHARACTERS = 4_000
MAX_MODEL_ALIAS_CHARACTERS = 64
OPENIE_MAX_NEW_TOKENS = 96
QUERY_BATCH_SIZE = 8
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
NATIVE_THREAD_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_ALIAS = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_MODEL_ALIAS_CHARACTERS - 1}}}\Z"
)


class AveritecP1OfficialError(RuntimeError):
    """The isolated official-core contract failed closed."""


@dataclass(frozen=True)
class QueryRow:
    item_id: str
    text: str


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AveritecP1OfficialError(
            "official worker value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    value = dict(body)
    if "self_sha256" in value:
        raise AveritecP1OfficialError("self hash was supplied twice")
    value["self_sha256"] = stable_hash(value)
    return value


def _opaque(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise AveritecP1OfficialError(f"{field} is not an opaque SHA-256 ID")
    return value


def _query_text(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > MAX_QUERY_CHARACTERS
    ):
        raise AveritecP1OfficialError("official query text is invalid")
    return value


def validate_input(
    value: object,
) -> tuple[str, tuple[global_contract.CorpusArticle, ...], tuple[QueryRow, ...]]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "articles",
            "block",
            "queries",
            "schema",
            "study_id",
        }
        or value.get("schema") != INPUT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("block") not in ALLOWED_BLOCKS
    ):
        raise AveritecP1OfficialError("official input envelope drifted")
    raw_articles = value.get("articles")
    raw_queries = value.get("queries")
    if not isinstance(raw_articles, list):
        raise AveritecP1OfficialError("official input articles drifted")
    try:
        articles = global_contract.validate_corpus(raw_articles)
    except global_contract.MoreHopQAOfficialHippoRAGError as exc:
        raise AveritecP1OfficialError("official corpus contract drifted") from exc
    if (
        not isinstance(raw_queries, list)
        or not 1 <= len(raw_queries) <= MAX_QUERY_COUNT
    ):
        raise AveritecP1OfficialError("official query count drifted")
    queries: list[QueryRow] = []
    seen: set[str] = set()
    for ordinal, row in enumerate(raw_queries):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"item_id", "ordinal", "text"}
            or row.get("ordinal") != ordinal
        ):
            raise AveritecP1OfficialError("official query row drifted")
        item_id = _opaque(row.get("item_id"), "item_id")
        if item_id in seen:
            raise AveritecP1OfficialError("official item ID is duplicated")
        seen.add(item_id)
        queries.append(QueryRow(item_id, _query_text(row.get("text"))))
    try:
        global_contract.validate_queries([row.text for row in queries])
    except global_contract.MoreHopQAOfficialHippoRAGError as exc:
        raise AveritecP1OfficialError(
            "official query contract drifted"
        ) from exc
    return str(value["block"]), articles, tuple(queries)


def input_payload(
    *,
    block: str,
    articles: Sequence[Mapping[str, object]],
    queries: Sequence[tuple[str, str]],
) -> dict[str, object]:
    payload = {
        "articles": [dict(row) for row in articles],
        "block": block,
        "queries": [
            {"item_id": item_id, "ordinal": ordinal, "text": text}
            for ordinal, (item_id, text) in enumerate(queries)
        ],
        "schema": INPUT_SCHEMA,
        "study_id": STUDY_ID,
    }
    validate_input(payload)
    return payload


def _graph_count(core: object, name: str) -> int:
    graph = getattr(core, "graph", None)
    method = getattr(graph, name, None)
    if not callable(method):
        return 0
    try:
        value = int(method())
    except BaseException as exc:
        raise AveritecP1OfficialError(
            "official graph count is unavailable"
        ) from exc
    if value < 0:
        raise AveritecP1OfficialError("official graph count drifted")
    return value


def retrieve_with_core(
    *,
    core: object,
    private_input: Mapping[str, object],
    index_root: Path,
    cuda_receipt: Mapping[str, object],
    observed_process_thread_peak: int,
) -> dict[str, object]:
    block, articles, queries = validate_input(private_input)
    documents = global_contract.serialize_corpus(articles)
    document_to_idx = {
        document: article.idx
        for document, article in zip(documents, articles)
    }
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise AveritecP1OfficialError("official core surface drifted")
    index(list(documents))
    build_snapshot = global_contract.snapshot_index_tree(index_root)
    rows: list[dict[str, object]] = []
    batch_sizes: list[int] = []
    for offset in range(0, len(queries), QUERY_BATCH_SIZE):
        batch_rows = queries[offset : offset + QUERY_BATCH_SIZE]
        batch = [row.text for row in batch_rows]
        try:
            result = retrieve(batch, num_to_retrieve=len(documents))
        except BaseException as exc:
            raise AveritecP1OfficialError(
                "official retrieve call failed"
            ) from exc
        if not isinstance(result, list) or len(result) != len(batch_rows):
            raise AveritecP1OfficialError(
                "official retrieve batch shape drifted"
            )
        for query, solution in zip(batch_rows, result):
            try:
                top5 = global_contract.stable_top_five_from_official_result(
                    retrieved_documents=getattr(solution, "docs", None),
                    retrieved_scores=getattr(solution, "doc_scores", None),
                    document_to_idx=document_to_idx,
                )
            except global_contract.MoreHopQAOfficialHippoRAGError as exc:
                raise AveritecP1OfficialError(
                    "official retrieve result drifted"
                ) from exc
            rows.append(
                {
                    "item_id": query.item_id,
                    "top5_document_ordinals": list(top5),
                }
            )
        batch_sizes.append(len(batch_rows))
    post_snapshot = global_contract.snapshot_index_tree(index_root)
    index_changed = post_snapshot != build_snapshot
    receipt = _self_hashed(
        {
            "batch_sizes": batch_sizes,
            "build_force_index_from_scratch": True,
            "build_index_call_count": 1,
            "corpus_count": len(articles),
            "corpus_sha256": global_contract.corpus_sha256(documents),
            "cuda_receipt": dict(cuda_receipt),
            "graph_edge_count": _graph_count(core, "ecount"),
            "graph_node_count": _graph_count(core, "vcount"),
            "index_changed_during_retrieve": index_changed,
            "index_file_count": build_snapshot.file_count,
            "index_post_file_count": post_snapshot.file_count,
            "index_post_total_bytes": post_snapshot.total_bytes,
            "index_post_tree_sha256": post_snapshot.tree_sha256,
            "index_total_bytes": build_snapshot.total_bytes,
            "index_tree_sha256": build_snapshot.tree_sha256,
            "observed_process_thread_peak": observed_process_thread_peak,
            "official_hipporag_commit": (
                global_contract.OFFICIAL_HIPPORAG_COMMIT
            ),
            "openie_max_new_tokens": OPENIE_MAX_NEW_TOKENS,
            "query_batch_upper_bound": QUERY_BATCH_SIZE,
            "query_count": len(queries),
            "retrieval_call_count": len(batch_sizes),
            "retrieval_index_call_count": 0,
            "schema": RUNTIME_SCHEMA,
            "single_worker_openie": True,
            "study_id": STUDY_ID,
            "torch_and_native_compute_thread_count": 1,
        }
    )
    output = _self_hashed(
        {
            "block": block,
            "input_sha256": stable_hash(private_input),
            "receipt": receipt,
            "rows": rows,
            "schema": OUTPUT_SCHEMA,
            "study_id": STUDY_ID,
        }
    )
    return output


def validate_output(
    value: object,
    *,
    expected_input: Mapping[str, object],
) -> dict[str, object]:
    block, articles, queries = validate_input(expected_input)
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "block",
            "input_sha256",
            "receipt",
            "rows",
            "schema",
            "self_sha256",
            "study_id",
        }
    ):
        raise AveritecP1OfficialError("official output envelope drifted")
    normalized = dict(value)
    output_hash = normalized.pop("self_sha256", None)
    if (
        not isinstance(output_hash, str)
        or _HEX64.fullmatch(output_hash) is None
        or output_hash != stable_hash(normalized)
        or normalized.get("block") != block
        or normalized.get("input_sha256") != stable_hash(expected_input)
        or normalized.get("schema") != OUTPUT_SCHEMA
        or normalized.get("study_id") != STUDY_ID
    ):
        raise AveritecP1OfficialError("official output binding drifted")
    receipt = normalized.get("receipt")
    if not isinstance(receipt, Mapping):
        raise AveritecP1OfficialError("official runtime receipt disappeared")
    receipt_hash = _verify_self(receipt)
    if (
        receipt.get("schema") != RUNTIME_SCHEMA
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("corpus_count") != len(articles)
        or receipt.get("query_count") != len(queries)
        or receipt.get("build_index_call_count") != 1
        or receipt.get("retrieval_index_call_count") != 0
        or receipt.get("query_batch_upper_bound") != QUERY_BATCH_SIZE
        or receipt.get("openie_max_new_tokens") != OPENIE_MAX_NEW_TOKENS
        or receipt.get("single_worker_openie") is not True
        or receipt.get("torch_and_native_compute_thread_count") != 1
    ):
        raise AveritecP1OfficialError("official runtime receipt drifted")
    changed = receipt.get("index_changed_during_retrieve")
    pre_tree = receipt.get("index_tree_sha256")
    post_tree = receipt.get("index_post_tree_sha256")
    pre_count = receipt.get("index_file_count")
    post_count = receipt.get("index_post_file_count")
    pre_bytes = receipt.get("index_total_bytes")
    post_bytes = receipt.get("index_post_total_bytes")
    if (
        type(changed) is not bool
        or not isinstance(pre_tree, str)
        or _HEX64.fullmatch(pre_tree) is None
        or not isinstance(post_tree, str)
        or _HEX64.fullmatch(post_tree) is None
        or type(pre_count) is not int
        or pre_count < 1
        or type(post_count) is not int
        or post_count < 1
        or type(pre_bytes) is not int
        or pre_bytes < 0
        or type(post_bytes) is not int
        or post_bytes < 0
        or changed
        != ((pre_tree, pre_count, pre_bytes) != (post_tree, post_count, post_bytes))
    ):
        raise AveritecP1OfficialError(
            "official ephemeral index mutation receipt drifted"
        )
    cuda = receipt.get("cuda_receipt")
    if (
        not isinstance(cuda, Mapping)
        or set(cuda) != {"post_inference", "pre_inference"}
    ):
        raise AveritecP1OfficialError("official CUDA receipt drifted")
    for phase in ("pre_inference", "post_inference"):
        row = cuda[phase]
        if (
            not isinstance(row, Mapping)
            or row.get("torch_cuda_is_available") is not True
            or row.get("visible_cuda_device_count") != 1
            or row.get("logical_cuda_current_device") != 0
            or row.get("physical_visible_gpu_binding") not in {"0", "1"}
            or row.get("cuda_allocation_and_synchronize_succeeded") is not True
        ):
            raise AveritecP1OfficialError(
                "official CUDA residency receipt drifted"
            )
    rows = normalized.get("rows")
    if not isinstance(rows, list) or len(rows) != len(queries):
        raise AveritecP1OfficialError("official result row count drifted")
    for query, row in zip(queries, rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"item_id", "top5_document_ordinals"}
            or row.get("item_id") != query.item_id
        ):
            raise AveritecP1OfficialError("official result item drifted")
        top5 = row.get("top5_document_ordinals")
        if (
            not isinstance(top5, list)
            or len(top5) != 5
            or len(set(top5)) != 5
            or any(
                type(ordinal) is not int
                or not 0 <= ordinal < len(articles)
                for ordinal in top5
            )
        ):
            raise AveritecP1OfficialError("official top5 drifted")
    normalized["receipt"] = dict(receipt)
    normalized["receipt"]["self_sha256"] = receipt_hash
    normalized["self_sha256"] = output_hash
    return normalized


def _verify_self(value: Mapping[str, object]) -> str:
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or stable_hash(body) != claimed
    ):
        raise AveritecP1OfficialError("official receipt self hash drifted")
    return claimed


def _read_private(path: Path) -> dict[str, object]:
    try:
        info = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AveritecP1OfficialError("official input is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or not isinstance(value, dict)
        or raw != canonical_bytes(value)
    ):
        raise AveritecP1OfficialError("official input metadata drifted")
    return value


def _write_private(path: Path, value: Mapping[str, object]) -> None:
    raw = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise AveritecP1OfficialError("official output mode drifted")


def _model_alias(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or _MODEL_ALIAS.fullmatch(value) is None
        or "/" in value
        or "\\" in value
        or ".." in value
        or Path(value).is_absolute()
        or not Path(value).is_dir()
    ):
        raise AveritecP1OfficialError(f"{field} alias drifted")
    return value


def _require_environment() -> None:
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != CUBLAS_WORKSPACE_CONFIG
        or any(os.environ.get(key) != "1" for key in NATIVE_THREAD_KEYS)
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise AveritecP1OfficialError(
            "official offline/thread environment drifted"
        )


def _require_project_origins(project_root: Path) -> None:
    try:
        root = project_root.resolve(strict=True)
    except OSError as exc:
        raise AveritecP1OfficialError("project root is unavailable") from exc
    expected = {
        __name__: "replication_runtime/averitec_p1_official_v1/worker.py",
        "replication_runtime": "replication_runtime/__init__.py",
        "replication_runtime.averitec_p1_official_v1": (
            "replication_runtime/averitec_p1_official_v1/__init__.py"
        ),
        qualified_base.__name__: (
            "replication_runtime/maud_extraction_p2_official_v1/worker.py"
        ),
        ebm_worker.__name__: (
            "replication_runtime/ebmnlp_p1_official_v1/worker.py"
        ),
        global_contract.__name__: (
            "replication_runtime/morehopqa_official_hipporag_v1/contract.py"
        ),
    }
    for module_name, relative in expected.items():
        module = sys.modules.get(module_name)
        origin = getattr(module, "__file__", None)
        try:
            valid = (
                isinstance(origin, str)
                and Path(origin).resolve(strict=True)
                == (root / relative).resolve(strict=True)
            )
        except OSError:
            valid = False
        if not valid:
            raise AveritecP1OfficialError(
                "official project import origin drifted"
            )


def run_once(
    *,
    private_input: Mapping[str, object],
    output_path: Path,
    index_root: Path,
    llm_model: str,
    embedding_model: str,
    hipporag_source_root: Path,
) -> dict[str, object]:
    _block, articles, _queries = validate_input(private_input)
    if index_root.exists() or index_root.is_symlink():
        raise AveritecP1OfficialError("official index root is not fresh")
    index_root.mkdir(mode=0o700)
    try:
        core = ebm_worker._build_core(
            save_dir=index_root,
            llm_model=llm_model,
            embedding_model=embedding_model,
            document_count=len(articles),
            hipporag_source_root=hipporag_source_root,
        )
    except BaseException as exc:
        raise AveritecP1OfficialError("official core build failed") from exc
    monitor = qualified_base._ProcessThreadPeakMonitor.start(os.getpid())
    try:
        pre = ebm_worker._attest_cuda_residency(core)
        # The result function indexes and retrieves while the same models stay
        # resident.  The post receipt is taken immediately after it returns.
        temporary_cuda = {"pre_inference": pre, "post_inference": pre}
        result = retrieve_with_core(
            core=core,
            private_input=private_input,
            index_root=index_root,
            cuda_receipt=temporary_cuda,
            observed_process_thread_peak=1,
        )
        post = ebm_worker._attest_cuda_residency(core)
    finally:
        peak = monitor.stop()
    # Replace the provisional values and recompute both nested self hashes.
    receipt = dict(result["receipt"])  # type: ignore[arg-type]
    receipt.pop("self_sha256")
    receipt["cuda_receipt"] = {
        "post_inference": post,
        "pre_inference": pre,
    }
    receipt["observed_process_thread_peak"] = peak
    receipt = _self_hashed(receipt)
    body = dict(result)
    body.pop("self_sha256")
    body["receipt"] = receipt
    result = _self_hashed(body)
    validate_output(result, expected_input=private_input)
    _write_private(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--hipporag-source-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    _require_environment()
    _require_project_origins(arguments.project_root)
    private_input = _read_private(arguments.input)
    validate_input(private_input)
    result = run_once(
        private_input=private_input,
        output_path=arguments.output,
        index_root=arguments.index_root,
        llm_model=_model_alias(arguments.llm_model, "LLM model"),
        embedding_model=_model_alias(
            arguments.embedding_model, "embedding model"
        ),
        hipporag_source_root=arguments.hipporag_source_root,
    )
    print(
        json.dumps(
            {
                "corpus_count": result["receipt"]["corpus_count"],  # type: ignore[index]
                "query_count": result["receipt"]["query_count"],  # type: ignore[index]
                "status": "passed",
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
