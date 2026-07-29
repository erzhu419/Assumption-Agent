"""Sequential item-local official-HippoRAG worker for WikiSQL UAO.

There is no WikiSQL loader, SQL parser, label reader, evaluator, API client,
network client, retry loop, replay path, or shared index in this module.  The
production CLI consumes the frozen private contract, creates a fresh pinned
official core for each item, and executes one index call followed by one
complete-corpus retrieve call on a single visible GPU lane.  It publishes the
shared common action pack and a separate detail-free aggregate receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from replication_runtime.tatqa_p20_v1 import (
    hipporag_worker as qualified_tatqa,
)

from . import contract


VERSION = "wikisql_uao_official_hipporag_worker_v1"
NATIVE_THREAD_ENVIRONMENT_KEYS = (
    qualified_tatqa.NATIVE_THREAD_ENVIRONMENT_KEYS
)
TORCH_INTRAOP_THREAD_COUNT = 1
TORCH_INTEROP_THREAD_COUNT = 1
_VISIBLE_GPU = re.compile(r"(?:0|[1-9][0-9]*)\Z")

CoreFactory = Callable[..., object]


@dataclass(frozen=True, slots=True)
class OfficialRunArtifacts:
    """The public action and content-free audit receipt of one native run."""

    action_pack: Mapping[str, object]
    safe_receipt: Mapping[str, object]


def _graph_count(core: object, method_name: str) -> int:
    graph = getattr(core, "graph", None)
    method = getattr(graph, method_name, None)
    if not callable(method):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "official graph counter is unavailable"
        )
    try:
        value = int(method())
    except BaseException as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "official graph counter failed"
        ) from exc
    if value < 0:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "official graph counter drifted"
        )
    return value


def retrieve_item_with_core(
    *, core: object, item: contract.WikiSQLItem
) -> tuple[tuple[int, ...], int, int]:
    """Execute exactly one item-local index and one retrieval call."""

    documents = contract.serialize_rows(item)
    document_to_ordinal = {
        document: ordinal
        for ordinal, document in enumerate(documents)
    }
    if len(document_to_ordinal) != len(item.rows):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "item-local row addressing collided"
        )
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "official core surface drifted"
        )
    try:
        index(list(documents))
        solutions = retrieve(
            [item.question],
            num_to_retrieve=len(documents),
        )
    except BaseException as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "official item execution failed without retry"
        ) from exc
    if not isinstance(solutions, list) or len(solutions) != 1:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "official item query result drifted"
        )
    solution = solutions[0]
    top5 = contract.stable_top_k(
        retrieved_documents=getattr(solution, "docs", None),
        retrieved_scores=getattr(solution, "doc_scores", None),
        document_to_ordinal=document_to_ordinal,
    )
    return (
        top5,
        _graph_count(core, "vcount"),
        _graph_count(core, "ecount"),
    )


def _hash_regular_file(path: Path) -> tuple[str, int]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "index file metadata is unavailable"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "index tree contains a non-regular file"
        )
    digest = hashlib.sha256()
    observed_size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                observed_size += len(block)
        after = path.lstat()
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "index file could not be committed"
        ) from exc
    if (
        not stat.S_ISREG(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or observed_size != after.st_size
    ):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "index file changed while being committed"
        )
    return digest.hexdigest(), observed_size


def snapshot_index_tree(index_root: Path) -> tuple[str, int, int]:
    """Hash every safe entry in one completed item-local index."""

    try:
        root_metadata = index_root.lstat()
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "item-local index root is unavailable"
        ) from exc
    if (
        index_root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
    ):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "item-local index root drifted"
        )
    try:
        paths = sorted(
            index_root.rglob("*"),
            key=lambda path: path.relative_to(index_root).as_posix(),
        )
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "item-local index tree could not be scanned"
        ) from exc
    entries: list[dict[str, object]] = []
    file_count = 0
    byte_count = 0
    for path in paths:
        relative = path.relative_to(index_root).as_posix()
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise contract.WikiSQLUAOOfficialHippoRAGError(
                "index entry metadata is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise contract.WikiSQLUAOOfficialHippoRAGError(
                "index tree contains a symbolic link"
            )
        if stat.S_ISDIR(metadata.st_mode):
            entries.append(
                {
                    "kind": "directory",
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "path": relative,
                }
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise contract.WikiSQLUAOOfficialHippoRAGError(
                "index tree contains a special file"
            )
        file_sha256, file_size = _hash_regular_file(path)
        entries.append(
            {
                "kind": "file",
                "mode": stat.S_IMODE(metadata.st_mode),
                "path": relative,
                "sha256": file_sha256,
                "size": file_size,
            }
        )
        file_count += 1
        byte_count += file_size
    return contract.semantic_sha256(entries), file_count, byte_count


def _fresh_item_index_root(
    *, index_parent: Path, item_ordinal: int, item_id: str
) -> Path:
    leaf = index_parent / (
        f"item-{item_ordinal:03d}-{item_id[:16]}"
    )
    if leaf.exists() or leaf.is_symlink():
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "item-local index root is not fresh"
        )
    try:
        leaf.mkdir(mode=0o700)
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "item-local index root could not be created"
        ) from exc
    return leaf


def run_with_core_factory(
    *,
    private_input: Mapping[str, object],
    index_parent: Path,
    core_factory: CoreFactory,
) -> OfficialRunArtifacts:
    """Run all items sequentially, constructing one fresh core per item.

    ``core_factory`` is the sole test seam.  It is called once per item with
    keyword arguments ``index_root``, ``item``, ``item_ordinal``, and
    ``row_count``.  An exception immediately aborts the batch; there is no
    retry, replay, fallback, or continuation to later items.
    """

    items = contract.validate_input(private_input)
    if not callable(core_factory):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "official core factory is unavailable"
        )
    if index_parent.exists() or index_parent.is_symlink():
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "index parent is not fresh"
        )
    try:
        index_parent.mkdir(mode=0o700)
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "index parent could not be created"
        ) from exc

    rankings: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for item_ordinal, item in enumerate(items):
        item_index_root = _fresh_item_index_root(
            index_parent=index_parent,
            item_ordinal=item_ordinal,
            item_id=item.item_id,
        )
        try:
            core = core_factory(
                index_root=item_index_root,
                item=item,
                item_ordinal=item_ordinal,
                row_count=len(item.rows),
            )
        except BaseException as exc:
            raise contract.WikiSQLUAOOfficialHippoRAGError(
                "official core construction failed without retry"
            ) from exc
        try:
            top5, graph_nodes, graph_edges = retrieve_item_with_core(
                core=core,
                item=item,
            )
            tree_sha256, file_count, byte_count = snapshot_index_tree(
                item_index_root
            )
        finally:
            del core
            gc.collect()
        root_binding = contract.semantic_sha256(
            {
                "index_leaf": item_index_root.name,
                "index_tree_sha256": tree_sha256,
                "item_id": item.item_id,
                "item_ordinal": item_ordinal,
                "row_corpus_sha256": item.row_corpus_sha256,
            }
        )
        rankings.append(
            contract.make_ranking_row(
                item=item,
                item_ordinal=item_ordinal,
                top5_row_ordinals=top5,
            )
        )
        receipts.append(
            contract.make_index_receipt(
                item=item,
                item_ordinal=item_ordinal,
                index_tree_sha256=tree_sha256,
                index_root_binding_sha256=root_binding,
                file_count=file_count,
                byte_count=byte_count,
                graph_node_count=graph_nodes,
                graph_edge_count=graph_edges,
            )
        )
    native_output = contract.build_output(
        input_value=private_input,
        rankings=rankings,
        index_receipts=receipts,
    )
    validated_native = contract.validate_output(
        native_output, expected_input=private_input
    )
    action_pack = contract.build_common_action_pack(
        expected_input=private_input,
        native_output=validated_native,
    )
    safe_receipt = contract.build_safe_receipt(
        expected_input=private_input,
        native_output=validated_native,
        action_pack=action_pack,
    )
    return OfficialRunArtifacts(
        action_pack=action_pack,
        safe_receipt=contract.validate_safe_receipt(
            safe_receipt,
            expected_input=private_input,
            native_output=validated_native,
            action_pack=action_pack,
        ),
    )


def _require_offline_single_gpu_environment() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("TOKENIZERS_PARALLELISM") != "false"
        or not isinstance(visible, str)
        or _VISIBLE_GPU.fullmatch(visible) is None
        or any(
            os.environ.get(key) != "1"
            for key in NATIVE_THREAD_ENVIRONMENT_KEYS
        )
    ):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "offline/single-GPU/single-thread environment drifted"
        )


def _prepare_official_runtime() -> None:
    """Prepare the already-qualified local official-core runtime once."""

    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise contract.WikiSQLUAOOfficialHippoRAGError(
                "worker must see exactly one available GPU"
            )
        qualified_tatqa._install_single_worker_openie_executor()
        torch.set_num_threads(TORCH_INTRAOP_THREAD_COUNT)
        torch.set_num_interop_threads(TORCH_INTEROP_THREAD_COUNT)
        if (
            torch.get_num_threads() != TORCH_INTRAOP_THREAD_COUNT
            or torch.get_num_interop_threads()
            != TORCH_INTEROP_THREAD_COUNT
        ):
            raise contract.WikiSQLUAOOfficialHippoRAGError(
                "torch thread configuration drifted"
            )
        torch.manual_seed(0)
    except contract.WikiSQLUAOOfficialHippoRAGError:
        raise
    except BaseException as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "qualified official runtime preparation failed"
        ) from exc


def build_official_core(
    *,
    index_root: Path,
    llm_model: Path,
    embedding_model: Path,
    row_count: int,
) -> object:
    """Construct a new pinned official core for exactly one table."""

    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or not contract.MIN_ROW_COUNT
        <= row_count
        <= contract.MAX_ROW_COUNT
    ):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "official row count drifted"
        )
    try:
        import torch
        from hipporag import HippoRAG
        from hipporag.utils.config_utils import BaseConfig

        # Re-seed each independently constructed item core.  Inference is
        # greedy, but this also prevents one table's construction lifecycle
        # from becoming an implicit random-state input to the next table.
        torch.manual_seed(0)
        config = BaseConfig(
            save_dir=str(index_root),
            llm_name="Transformers/" + str(llm_model),
            embedding_model_name=(
                "Transformers/" + str(embedding_model)
            ),
            openie_mode=contract.FROZEN_CORE_CONFIG["openie_mode"],
            max_new_tokens=contract.FROZEN_CORE_CONFIG[
                "max_new_tokens"
            ],
            retrieval_top_k=row_count,
            qa_top_k=contract.TOP_K,
            force_index_from_scratch=True,
            save_openie=True,
            temperature=0,
            seed=0,
        )
        core = HippoRAG(global_config=config)
        qualified_tatqa._install_completion_only_backend(core)
        return core
    except BaseException as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "pinned official core construction failed"
        ) from exc


def _read_private(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "private worker input is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "private worker input metadata drifted"
        )
    value, _items = contract.parse_input(raw)
    return value


def _write_private(path: Path, value: Mapping[str, object]) -> None:
    raw = contract.canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "private worker output could not be created"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise contract.WikiSQLUAOOfficialHippoRAGError(
                    "private worker output write failed"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        metadata = path.lstat()
        observed = path.read_bytes()
    except OSError as exc:
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "private worker output verification failed"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or observed != raw
    ):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "private worker output verification failed"
        )


def run_once(
    *,
    private_input: Mapping[str, object],
    action_output_path: Path,
    safe_receipt_output_path: Path,
    index_parent: Path,
    core_factory: CoreFactory,
) -> OfficialRunArtifacts:
    """Publish both exact outputs only after the no-retry batch succeeds."""

    output_paths = (action_output_path, safe_receipt_output_path)
    if (
        action_output_path == safe_receipt_output_path
        or any(path.exists() or path.is_symlink() for path in output_paths)
    ):
        raise contract.WikiSQLUAOOfficialHippoRAGError(
            "worker outputs are not fresh and distinct"
        )
    artifacts = run_with_core_factory(
        private_input=private_input,
        index_parent=index_parent,
        core_factory=core_factory,
    )
    _write_private(action_output_path, artifacts.action_pack)
    _write_private(
        safe_receipt_output_path,
        artifacts.safe_receipt,
    )
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--action-output", required=True, type=Path)
    parser.add_argument(
        "--safe-receipt-output",
        required=True,
        type=Path,
    )
    parser.add_argument("--index-parent", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    private_input = _read_private(arguments.input)
    _require_offline_single_gpu_environment()
    _prepare_official_runtime()

    def production_core_factory(
        *,
        index_root: Path,
        item: contract.WikiSQLItem,
        item_ordinal: int,
        row_count: int,
    ) -> object:
        del item, item_ordinal
        return build_official_core(
            index_root=index_root,
            llm_model=arguments.llm_model,
            embedding_model=arguments.embedding_model,
            row_count=row_count,
        )

    artifacts = run_once(
        private_input=private_input,
        action_output_path=arguments.action_output,
        safe_receipt_output_path=arguments.safe_receipt_output,
        index_parent=arguments.index_parent,
        core_factory=production_core_factory,
    )
    print(
        json.dumps(
            {
                "action_pack_sha256": artifacts.action_pack[
                    "self_sha256"
                ],
                "index_call_count": artifacts.safe_receipt[
                    "runtime"
                ][  # type: ignore[index]
                    "index_call_count"
                ],
                "item_count": artifacts.action_pack["item_count"],
                "official_hipporag_commit": (
                    contract.OFFICIAL_HIPPORAG_COMMIT
                ),
                "receipt_sha256": artifacts.safe_receipt[
                    "self_sha256"
                ],
                "retrieve_call_count": artifacts.safe_receipt[
                    "runtime"
                ][  # type: ignore[index]
                    "retrieve_call_count"
                ],
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


__all__ = [
    "NATIVE_THREAD_ENVIRONMENT_KEYS",
    "OfficialRunArtifacts",
    "TORCH_INTEROP_THREAD_COUNT",
    "TORCH_INTRAOP_THREAD_COUNT",
    "VERSION",
    "build_official_core",
    "retrieve_item_with_core",
    "run_once",
    "run_with_core_factory",
    "snapshot_index_tree",
]
