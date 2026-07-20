"""One-shot offline NanoBEIR C_confirm runtime for P11, RAW, and HippoRAG."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_runtime_v1 as train,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    hipporag_upstream_hardening_qualification_v1 as hardening_qualification,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_acquisition_v1 as acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import p11_raw_ce_rrf_v1 as p11
from reconstruction_v2.replication_runtime.hipporag_upstream_hardening_v1 import (
    backport,
)


SCHEMA = "nanobeir_p11_c_confirm_runtime_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p11_c_confirm_runtime_attempt_v1"
INTENT_SCHEMA = "nanobeir_p11_c_confirm_runtime_intents_v1"
ACTION_SCHEMA = "nanobeir_p11_c_confirm_runtime_actions_v1"
FREEZE_SCHEMA = "nanobeir_p11_c_confirm_runtime_implementation_freeze_v1"
ITEM_COUNT = 36
ITEMS_PER_FAMILY = 12
HIPPORAG_CONCURRENCY = 12
EXTERNAL_PROCESS_CONCURRENCY = 13

RUN_ROOT_RELATIVE = Path("artifacts/nanobeir_p11_c_confirm_runtime_v1")
RESULT_RELATIVE = Path("manifests/nanobeir_p11_c_confirm_runtime_result_v1.json")
FREEZE_RELATIVE = Path(
    "manifests/nanobeir_p11_c_confirm_runtime_implementation_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p11_c_confirm_runtime_v1.py"
)
TEST_RELATIVE = Path("tests/test_nanobeir_p11_c_confirm_runtime_v1.py")
ACQUISITION_RESULT_RELATIVE = Path(
    "manifests/nanobeir_p11_acquisition_result_v1.json"
)
ACQUISITION_RESULT_FILE_SHA256 = (
    "c0e64dcf585cb3ab81e56f86696aabe8ad1850cad918fa11b044d26f09ce4a05"
)
ACQUISITION_RESULT_SELF_SHA256 = (
    "2b4012edeec86ac05c07fe6538b7d6f93bd7592fd246d78cc18e56a9321fbdbe"
)
CANDIDATE_FREEZE_SELF_SHA256 = (
    "aa49d5c0c194bd600486d64b5e94b29576d746cf4d23eff74d17653771293791"
)
HARDENING_RESULT_SELF_SHA256 = (
    "2c9571cf7437d47d6e0dad3317841f77ebc0b782132b945185514f106c0ed8a3"
)
STUDY_DESIGN_SELF_SHA256 = (
    "992f817ff35aa9da0cde0c8c70b659338cce5ab0f4399fc87b041e16ac6ce17f"
)


class NanoBEIRCConfirmError(RuntimeError):
    """The frozen prospective C_confirm runtime failed closed."""


class OneShotRefusal(NanoBEIRCConfirmError):
    """The formal runtime root or result is already consumed."""


@dataclass(frozen=True)
class RuntimeItem:
    ordinal: int
    family: str
    family_ordinal: int
    item_key: str
    query: str
    source_query_id: str


@dataclass(frozen=True)
class FamilyCorpus:
    ids: tuple[str, ...]
    contents: tuple[str, ...]


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NanoBEIRCConfirmError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NanoBEIRCConfirmError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise NanoBEIRCConfirmError(f"{name} is not an object")
    return value


def _verify_self(value: Mapping[str, Any], expected: str) -> None:
    try:
        acquisition.verify_self_hash(value, expected)
    except acquisition.NanoBEIRAcquisitionError as exc:
        raise NanoBEIRCConfirmError(str(exc)) from exc


def _load_acquisition(base: Path) -> Mapping[str, Any]:
    path = base / ACQUISITION_RESULT_RELATIVE
    if acquisition.file_sha256(path) != ACQUISITION_RESULT_FILE_SHA256:
        raise NanoBEIRCConfirmError("acquisition result file drifted")
    value = _read_json(path, "acquisition result")
    _verify_self(value, ACQUISITION_RESULT_SELF_SHA256)
    if (
        value.get("schema") != acquisition.SCHEMA
        or value.get("status")
        != "passed_138_item_private_acquisition_ready_for_C_confirm_runtime"
        or value.get("study_design_self_sha256") != STUDY_DESIGN_SELF_SHA256
    ):
        raise NanoBEIRCConfirmError("acquisition completion drifted")
    return value


def _verify_freeze(base: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "C_confirm freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise NanoBEIRCConfirmError("C_confirm freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise NanoBEIRCConfirmError("C_confirm freeze hash is absent")
    _verify_self(value, declared)
    if (
        value.get("acquisition_result_self_sha256")
        != ACQUISITION_RESULT_SELF_SHA256
        or value.get("candidate_freeze_self_sha256")
        != CANDIDATE_FREEZE_SELF_SHA256
        or value.get("hardening_result_self_sha256")
        != HARDENING_RESULT_SELF_SHA256
    ):
        raise NanoBEIRCConfirmError("C_confirm freeze prerequisite drifted")
    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise NanoBEIRCConfirmError("C_confirm freeze bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_paths = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != expected_paths:
        raise NanoBEIRCConfirmError("C_confirm implementation set drifted")
    for relative, expected in observed.items():
        if not isinstance(expected, str) or acquisition.file_sha256(base / str(relative)) != expected:
            raise NanoBEIRCConfirmError("C_confirm implementation drifted")
    return value


def _pack_binding(
    acquisition_result: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    bindings = acquisition_result.get("pack_bindings")
    if not isinstance(bindings, Mapping):
        raise NanoBEIRCConfirmError("acquisition pack registry drifted")
    binding = bindings.get(name)
    if not isinstance(binding, Mapping):
        raise NanoBEIRCConfirmError(f"{name} binding is absent")
    return binding


def _load_pack(base: Path, binding: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    relative = binding.get("relative_path")
    if not isinstance(relative, str):
        raise NanoBEIRCConfirmError(f"{name} path drifted")
    path = base / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != binding.get("size_bytes")
        or acquisition.file_sha256(path) != binding.get("file_sha256")
    ):
        raise NanoBEIRCConfirmError(f"{name} file drifted")
    value = _read_json(path, name)
    declared = binding.get("pack_sha256")
    body = dict(value)
    observed = body.pop("pack_sha256", None)
    if not isinstance(declared, str) or observed != declared or acquisition.stable_hash(body) != declared:
        raise NanoBEIRCConfirmError(f"{name} pack hash drifted")
    return value


def load_views(
    base: Path, acquisition_result: Mapping[str, Any]
) -> tuple[RuntimeItem, ...]:
    binding = _pack_binding(acquisition_result, "C_confirm_view")
    if binding.get("item_count") != ITEM_COUNT:
        raise NanoBEIRCConfirmError("C_confirm view count drifted")
    pack = _load_pack(base, binding, "C_confirm view")
    if pack.get("schema") != "nanobeir_p11_private_view_v1" or pack.get("block") != "C_confirm":
        raise NanoBEIRCConfirmError("C_confirm view envelope drifted")
    rows = pack.get("items")
    if not isinstance(rows, list) or len(rows) != ITEM_COUNT:
        raise NanoBEIRCConfirmError("C_confirm view rows drifted")
    items: list[RuntimeItem] = []
    family_counts = {family: 0 for family in acquisition.FAMILIES}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "family",
            "family_ordinal",
            "item_key",
            "query",
            "source_query_id",
        }:
            raise NanoBEIRCConfirmError("C_confirm view row shape drifted")
        family = row.get("family")
        family_ordinal = row.get("family_ordinal")
        if (
            family not in acquisition.FAMILIES
            or isinstance(family_ordinal, bool)
            or not isinstance(family_ordinal, int)
            or family_ordinal != family_counts[family]
        ):
            raise NanoBEIRCConfirmError("C_confirm family order drifted")
        family_counts[family] += 1
        item_key = row.get("item_key")
        query = row.get("query")
        query_id = row.get("source_query_id")
        if not all(isinstance(value, str) and value for value in (item_key, query, query_id)):
            raise NanoBEIRCConfirmError("C_confirm view text drifted")
        items.append(
            RuntimeItem(
                ordinal=ordinal,
                family=family,
                family_ordinal=family_ordinal,
                item_key=item_key,
                query=query,
                source_query_id=query_id,
            )
        )
    if any(count != ITEMS_PER_FAMILY for count in family_counts.values()):
        raise NanoBEIRCConfirmError("C_confirm family count drifted")
    if len({item.item_key for item in items}) != ITEM_COUNT:
        raise NanoBEIRCConfirmError("C_confirm item keys are duplicated")
    return tuple(items)


def load_labels_after_action_seal(
    *,
    base: Path,
    acquisition_result: Mapping[str, Any],
    items: Sequence[RuntimeItem],
    action_path: Path,
    expected_action_sha256: str,
) -> Mapping[str, tuple[str, ...]]:
    if (
        action_path.is_symlink()
        or not action_path.is_file()
        or acquisition.file_sha256(action_path) != expected_action_sha256
    ):
        raise NanoBEIRCConfirmError("action seal is absent before labels")
    binding = _pack_binding(acquisition_result, "C_confirm_labels")
    if binding.get("item_count") != ITEM_COUNT:
        raise NanoBEIRCConfirmError("C_confirm label count drifted")
    pack = _load_pack(base, binding, "C_confirm labels")
    if pack.get("schema") != "nanobeir_p11_private_labels_v1" or pack.get("block") != "C_confirm":
        raise NanoBEIRCConfirmError("C_confirm label envelope drifted")
    rows = pack.get("items")
    if not isinstance(rows, list) or len(rows) != ITEM_COUNT:
        raise NanoBEIRCConfirmError("C_confirm label rows drifted")
    labels: dict[str, tuple[str, ...]] = {}
    for item, row in zip(items, rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"family", "gold_document_ids", "item_key"}
            or row.get("family") != item.family
            or row.get("item_key") != item.item_key
        ):
            raise NanoBEIRCConfirmError("C_confirm label identity drifted")
        gold = row.get("gold_document_ids")
        if (
            not isinstance(gold, list)
            or not gold
            or any(not isinstance(value, str) or not value for value in gold)
            or len(set(gold)) != len(gold)
        ):
            raise NanoBEIRCConfirmError("C_confirm gold list drifted")
        labels[item.item_key] = tuple(gold)
    return labels


def load_corpora(base: Path) -> Mapping[str, FamilyCorpus]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NanoBEIRCConfirmError("pyarrow is unavailable") from exc
    output: dict[str, FamilyCorpus] = {}
    for family in acquisition.FAMILIES:
        path = (
            base
            / acquisition.SOURCE_ROOT_RELATIVE
            / "corpus"
            / f"{family}-00000-of-00001.parquet"
        )
        expected = acquisition.SOURCE_FILES[
            f"corpus/{family}-00000-of-00001.parquet"
        ]
        if acquisition.file_sha256(path) != expected:
            raise NanoBEIRCConfirmError("corpus source drifted")
        table = pq.read_table(path)
        if table.column_names != ["_id", "text"]:
            raise NanoBEIRCConfirmError("corpus schema drifted")
        ids: list[str] = []
        contents: list[str] = []
        for row in table.to_pylist():
            identifier = row.get("_id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise NanoBEIRCConfirmError("corpus ID drifted")
            try:
                content = acquisition.project_document(row.get("text"))
            except acquisition.NanoBEIRAcquisitionError as exc:
                raise NanoBEIRCConfirmError(str(exc)) from exc
            ids.append(identifier)
            contents.append(content)
        if len(ids) != len(set(ids)) or len(ids) < 32:
            raise NanoBEIRCConfirmError("corpus identity drifted")
        output[family] = FamilyCorpus(tuple(ids), tuple(contents))
    return output


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _materialize_hardened_source(base: Path, root: Path) -> Path:
    baseline = (
        base
        / hardening_qualification.BASELINE_REPO_RELATIVE
        / hardening_qualification.BASELINE_SOURCE_WITHIN_REPO
    )
    patched = backport.apply_fixed_backport(baseline.read_bytes())
    patched_root = root / "patched_source"
    patched_root.mkdir(mode=0o700)
    path = patched_root / "HippoRAG.py"
    train.bright_runtime._write_exclusive(path, patched, mode=0o600)
    if acquisition.file_sha256(path) != backport.PATCHED_SOURCE_SHA256:
        raise NanoBEIRCConfirmError("materialized hardened source drifted")
    return path


def _prepare_cross_input(
    *,
    plans: Sequence[train.ExpandedPlan],
    items: Sequence[RuntimeItem],
    corpora: Mapping[str, FamilyCorpus],
) -> Mapping[str, Any]:
    rows: list[dict[str, Any]] = []
    for plan, item in zip(plans, items):
        if plan.local.item.ordinal != item.ordinal:
            raise NanoBEIRCConfirmError("cross input ordinal drifted")
        contents = corpora[item.family].contents
        rows.append(
            {
                "documents": [
                    {"content": contents[row], "ordinal": position}
                    for position, row in enumerate(plan.expanded.expanded_pool)
                ],
                "mechanism_query": plan.local.mechanism_query,
                "ordinal": item.ordinal,
                "relation_query": plan.local.relation_query,
            }
        )
    return train.cross_contract.input_payload(rows)


def _prepare_hipporag_inputs(
    *,
    root: Path,
    plans: Sequence[train.ExpandedPlan],
    items: Sequence[RuntimeItem],
    corpora: Mapping[str, FamilyCorpus],
) -> tuple[Path, ...]:
    hippo_root = root / "hipporag"
    hippo_root.mkdir(mode=0o700)
    result: list[Path] = []
    for plan, item in zip(plans, items):
        item_root = hippo_root / f"item_{item.ordinal:03d}"
        item_root.mkdir(mode=0o700)
        for name in ("home", "hf", "tmp"):
            (item_root / name).mkdir(mode=0o700)
        contents = corpora[item.family].contents
        payload = {
            "documents": [
                {"content": contents[row], "ordinal": position}
                for position, row in enumerate(plan.local.base_pool)
            ],
            "query": item.query,
            "schema": train.hippo_contract.INPUT_SCHEMA,
        }
        train.hippo_contract.validate_input(payload["query"], payload["documents"])
        train.bright_runtime._write_json(item_root / "input.json", payload)
        result.append(item_root)
    return tuple(result)


def _run_hardened_hipporag_item(
    *,
    base: Path,
    item_root: Path,
    candidate_rows: Sequence[int],
    patched_source: Path,
    semaphore: threading.Semaphore,
    counter: Any,
) -> Mapping[str, Any]:
    with semaphore:
        counter.enter()
        try:
            baseline_source = (
                base
                / hardening_qualification.BASELINE_REPO_RELATIVE
                / hardening_qualification.BASELINE_SOURCE_WITHIN_REPO
            ).resolve(strict=True)
            command = [
                "/usr/bin/bwrap",
                "--die-with-parent",
                "--unshare-all",
                "--new-session",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/models",
                "--ro-bind",
                str(base / train.bright_runtime.HIPPORAG_LLM_RELATIVE),
                "/tmp/models/llm",
                "--ro-bind",
                str(base / train.bright_runtime.MINILM_MODEL_RELATIVE),
                "/tmp/models/embed",
                "--ro-bind",
                str(patched_source),
                str(baseline_source),
                "--bind",
                str(item_root),
                str(item_root),
                "--chdir",
                str(base),
                "--setenv",
                "CUDA_VISIBLE_DEVICES",
                "",
                "--setenv",
                "HF_HOME",
                str(item_root / "hf"),
                "--setenv",
                "HF_HUB_OFFLINE",
                "1",
                "--setenv",
                "HOME",
                str(item_root / "home"),
                "--setenv",
                "MPLCONFIGDIR",
                str(item_root / "tmp" / "mpl"),
                "--setenv",
                "OMP_NUM_THREADS",
                "2",
                "--setenv",
                "TOKENIZERS_PARALLELISM",
                "false",
                "--setenv",
                "TMPDIR",
                str(item_root / "tmp"),
                "--setenv",
                "TRANSFORMERS_OFFLINE",
                "1",
                str(base / train.bright_runtime.HIPPORAG_PYTHON_RELATIVE),
                "-I",
                "-B",
                "-m",
                "replication_runtime.bright_official_hipporag_v1.worker",
                "--input",
                str(item_root / "input.json"),
                "--output",
                str(item_root / "output.json"),
                "--index-root",
                str(item_root / "index"),
                "--llm-model",
                "/tmp/models/llm",
                "--embedding-model",
                "/tmp/models/embed",
            ]
            try:
                completed = subprocess.run(
                    command, cwd=base, check=False, capture_output=True, timeout=1800
                )
            except subprocess.TimeoutExpired as exc:
                raise NanoBEIRCConfirmError("HippoRAG item timed out") from exc
            train.bright_runtime._write_exclusive(
                item_root / "stdout.log", completed.stdout, mode=0o600
            )
            train.bright_runtime._write_exclusive(
                item_root / "stderr.log", completed.stderr, mode=0o600
            )
            if completed.returncode != 0:
                raise NanoBEIRCConfirmError(
                    "HippoRAG item failed: "
                    + hashlib.sha256(completed.stderr).hexdigest()
                )
            output_path = item_root / "output.json"
            payload = train.hippo_contract.parse_output(output_path.read_bytes())
            if payload["graph_node_count"] <= 32 or payload["graph_edge_count"] <= 0:
                raise NanoBEIRCConfirmError("HippoRAG item did not build a graph")
            top_rows = tuple(
                candidate_rows[position] for position in payload["top_ordinals"]
            )
            return {
                "graph_edge_count": payload["graph_edge_count"],
                "graph_node_count": payload["graph_node_count"],
                "output_file_sha256": acquisition.file_sha256(output_path),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "top_rows": list(top_rows),
            }
        finally:
            counter.leave()


def _paired(left: Sequence[int], right: Sequence[int]) -> Mapping[str, int]:
    if len(left) != len(right):
        raise NanoBEIRCConfirmError("paired score shape drifted")
    deltas = [int(a) - int(b) for a, b in zip(left, right)]
    return {
        "gain": sum(value > 0 for value in deltas),
        "harm": sum(value < 0 for value in deltas),
        "net_integer_ndcg": sum(deltas),
        "tie": sum(value == 0 for value in deltas),
    }


def primary_decision(
    *, items: Sequence[RuntimeItem], arm_scores: Mapping[str, Sequence[int]]
) -> tuple[bool, Mapping[str, Any]]:
    if set(arm_scores) != {"P11", "RAW", "HippoRAG"}:
        raise NanoBEIRCConfirmError("arm registry drifted")
    if any(len(values) != ITEM_COUNT for values in arm_scores.values()):
        raise NanoBEIRCConfirmError("arm score vector drifted")
    comparisons: dict[str, Any] = {}
    for baseline in ("RAW", "HippoRAG"):
        paired = _paired(arm_scores["P11"], arm_scores[baseline])
        family_nets = {
            family: sum(
                arm_scores["P11"][index] - arm_scores[baseline][index]
                for index, item in enumerate(items)
                if item.family == family
            )
            for family in acquisition.FAMILIES
        }
        comparisons[f"P11_minus_{baseline}"] = {
            **paired,
            "family_net_integer_ndcg": family_nets,
        }
    passed = all(
        row["net_integer_ndcg"] > 0
        and all(value > 0 for value in row["family_net_integer_ndcg"].values())
        for row in comparisons.values()
    )
    return passed, comparisons


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("C_confirm root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("C_confirm result already exists")
    acquisition_result = _load_acquisition(base)
    freeze = _verify_freeze(base)
    items = load_views(base, acquisition_result)
    corpora = load_corpora(base)
    root.mkdir(mode=0o700)
    marker = {
        "acquisition_result_self_sha256": ACQUISITION_RESULT_SELF_SHA256,
        "candidate_freeze_self_sha256": CANDIDATE_FREEZE_SELF_SHA256,
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
    }
    marker_path = root / "attempt.marker"
    train.bright_runtime._write_json(marker_path, marker)
    patched_source = _materialize_hardened_source(base, root)

    bright_items = tuple(
        train.bright_runtime.ViewItem(
            ordinal=item.ordinal,
            family=item.family,
            commitment=item.item_key,
            query=item.query,
            excluded_ids=(),
        )
        for item in items
    )
    try:
        qwen_output, qwen_receipt = train.bright_runtime._run_qwen(
            base, root, bright_items
        )
    except train.bright_runtime.BrightStudyError as exc:
        raise NanoBEIRCConfirmError(str(exc)) from exc
    qwen_rows = qwen_output.get("items")
    if (
        not isinstance(qwen_rows, list)
        or len(qwen_rows) != ITEM_COUNT
        or not all(row.get("generation_valid") is True for row in qwen_rows)
    ):
        raise NanoBEIRCConfirmError("typed query generation was not fully valid")

    try:
        encoder = train.bright_runtime._new_minilm(base)
        corpus_embeddings = {
            family: train.bright_runtime._encode_chunks(
                encoder, corpora[family].contents
            )
            for family in acquisition.FAMILIES
        }
    except train.bright_runtime.BrightStudyError as exc:
        raise NanoBEIRCConfirmError(str(exc)) from exc
    tensor_root = root / "corpus_tensors"
    tensor_root.mkdir(mode=0o700)
    corpus_tensor_bindings: dict[str, Any] = {}
    for family in acquisition.FAMILIES:
        matrix = np.asarray(corpus_embeddings[family], dtype=np.float32)
        if matrix.shape != (len(corpora[family].ids), 384) or not np.isfinite(matrix).all():
            raise NanoBEIRCConfirmError("corpus embedding tensor drifted")
        path = tensor_root / f"{family}.embeddings.npy"
        train.bright_runtime._save_npy_exclusive(path, matrix)
        corpus_tensor_bindings[family] = {
            "document_count": len(corpora[family].ids),
            "file_sha256": acquisition.file_sha256(path),
            "float32_sha256": train.float32_matrix_sha256(matrix),
        }

    flattened_queries: list[str] = []
    query_slices: list[tuple[int, int]] = []
    for item, row in zip(items, qwen_rows):
        expansions = row.get("expansions")
        if not isinstance(expansions, list) or len(expansions) != 4:
            raise NanoBEIRCConfirmError("typed query expansion row drifted")
        start = len(flattened_queries)
        flattened_queries.extend([item.query, *expansions])
        query_slices.append((start, len(flattened_queries)))
    query_embeddings = train.bright_runtime._encode_chunks(encoder, flattened_queries)
    local_plans: list[train.LocalPlan] = []
    for item, row, (start, end) in zip(items, qwen_rows, query_slices):
        scores = [
            train.quantized_scores(corpus_embeddings[item.family], query_embeddings[index])
            for index in range(start, end)
        ]
        view = train.ViewItem(
            ordinal=item.ordinal,
            item_key=item.item_key,
            query=item.query,
            excluded_ids=(),
        )
        try:
            local_plans.append(
                train.build_local_plan(
                    item=view,
                    document_ids=corpora[item.family].ids,
                    document_contents=corpora[item.family].contents,
                    query_score_vectors=scores,
                    expansions=row["expansions"],
                )
            )
        except train.FiqaTrainRuntimeError as exc:
            raise NanoBEIRCConfirmError(str(exc)) from exc
    bridge_queries = [query.text for plan in local_plans for query in plan.bridge_queries]
    bridge_embeddings = (
        train.bright_runtime._encode_chunks(encoder, bridge_queries)
        if bridge_queries
        else np.empty((0, 384), dtype=np.float32)
    )
    expanded_plans: list[train.ExpandedPlan] = []
    offset = 0
    for item, plan in zip(items, local_plans):
        count = len(plan.bridge_queries)
        vectors = [
            train.quantized_scores(corpus_embeddings[item.family], bridge_embeddings[index])
            for index in range(offset, offset + count)
        ]
        try:
            expanded_plans.append(train.expand_plan(plan, vectors))
        except train.FiqaTrainRuntimeError as exc:
            raise NanoBEIRCConfirmError(str(exc)) from exc
        offset += count
    if offset != len(bridge_queries):
        raise NanoBEIRCConfirmError("bridge embedding accounting drifted")
    query_embedding_path = root / "typed_query.embeddings.npy"
    bridge_embedding_path = root / "bridge_query.embeddings.npy"
    train.bright_runtime._save_npy_exclusive(query_embedding_path, query_embeddings)
    train.bright_runtime._save_npy_exclusive(bridge_embedding_path, bridge_embeddings)
    del encoder, query_embeddings, bridge_embeddings, corpus_embeddings
    _release_cuda()

    cross_payload = _prepare_cross_input(
        plans=expanded_plans, items=items, corpora=corpora
    )
    cross_input_path = root / "cross_encoder.input.json"
    cross_output_path = root / "cross_encoder.output.json"
    train.bright_runtime._write_exclusive(
        cross_input_path,
        train.cross_contract.canonical_json_bytes(cross_payload),
        mode=0o600,
    )
    hippo_roots = _prepare_hipporag_inputs(
        root=root, plans=expanded_plans, items=items, corpora=corpora
    )
    intents = acquisition.self_hashed(
        {
            "cross_encoder_input_file_sha256": acquisition.file_sha256(
                cross_input_path
            ),
            "items": [
                {
                    "base_pool": list(plan.local.base_pool),
                    "expanded_pool": list(plan.expanded.expanded_pool),
                    "family": item.family,
                    "item_key": item.item_key,
                    "ordinal": item.ordinal,
                }
                for item, plan in zip(items, expanded_plans)
            ],
            "schema": INTENT_SCHEMA,
        },
        field="pack_sha256",
    )
    intent_path = root / "action.intents.json"
    train.bright_runtime._write_json(intent_path, intents)

    semaphore = threading.Semaphore(HIPPORAG_CONCURRENCY)
    counter = train.bright_runtime._ConcurrencyCounter()
    completed_hippo: dict[int, Mapping[str, Any]] = {}
    environment_updates = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous_environment = {key: os.environ.get(key) for key in environment_updates}
    os.environ.update(environment_updates)
    try:
        with ThreadPoolExecutor(max_workers=EXTERNAL_PROCESS_CONCURRENCY) as executor:
            cross_future: Future[Any] = executor.submit(
                train.cross_worker.run,
                input_path=cross_input_path,
                output_path=cross_output_path,
                model_root=base / train.CROSS_MODEL_RELATIVE,
            )
            hippo_futures = {
                executor.submit(
                    _run_hardened_hipporag_item,
                    base=base,
                    item_root=item_root,
                    candidate_rows=plan.local.base_pool,
                    patched_source=patched_source,
                    semaphore=semaphore,
                    counter=counter,
                ): item.ordinal
                for item, plan, item_root in zip(items, expanded_plans, hippo_roots)
            }
            for future in as_completed([cross_future, *hippo_futures]):
                if future is cross_future:
                    future.result()
                else:
                    completed_hippo[hippo_futures[future]] = future.result()
    except Exception as exc:
        if isinstance(exc, NanoBEIRCConfirmError):
            raise
        raise NanoBEIRCConfirmError("external action execution failed") from exc
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if (
        counter.current != 0
        or counter.peak > HIPPORAG_CONCURRENCY
        or set(completed_hippo) != set(range(ITEM_COUNT))
        or not cross_output_path.is_file()
    ):
        raise NanoBEIRCConfirmError("external action completion drifted")
    try:
        cross_output = train.cross_contract.parse_output(cross_output_path.read_bytes())
    except Exception as exc:
        raise NanoBEIRCConfirmError("cross-encoder output drifted") from exc
    cross_rows = cross_output.get("items")
    if not isinstance(cross_rows, list) or len(cross_rows) != ITEM_COUNT:
        raise NanoBEIRCConfirmError("cross-encoder item count drifted")

    action_rows: list[dict[str, Any]] = []
    for item, plan, cross_row in zip(items, expanded_plans, cross_rows):
        if (
            cross_row.get("ordinal") != item.ordinal
            or cross_row.get("document_count") != len(plan.expanded.expanded_pool)
        ):
            raise NanoBEIRCConfirmError("cross-encoder row identity drifted")
        p11_rows = p11.rank_p11(
            expanded_pool=plan.expanded.expanded_pool,
            raw_top10=plan.local.raw_rows,
            cross_encoder_relation_scores=cross_row["relation_scores_quantized"],
            cross_encoder_mechanism_scores=cross_row["mechanism_scores_quantized"],
        )
        hippo = dict(completed_hippo[item.ordinal])
        ids = corpora[item.family].ids
        action_rows.append(
            {
                "bridge_anchor_count": len(plan.local.anchors),
                "bridge_query_count": len(plan.local.bridge_queries),
                "candidate_expansion": {
                    "expanded_pool_size": len(plan.expanded.expanded_pool),
                    "P11_top10_documents_outside_base_pool": len(
                        set(p11_rows) - set(plan.local.base_pool)
                    ),
                    "unique_bridge_candidates_outside_base_pool": len(
                        set(plan.expanded.expanded_pool) - set(plan.local.base_pool)
                    ),
                },
                "family": item.family,
                "HippoRAG": {
                    **hippo,
                    "document_ids": [ids[row] for row in hippo["top_rows"]],
                },
                "item_key": item.item_key,
                "ordinal": item.ordinal,
                "P11_document_ids": [ids[row] for row in p11_rows],
                "P11_rows": list(p11_rows),
                "RAW_document_ids": [ids[row] for row in plan.local.raw_rows],
                "RAW_rows": list(plan.local.raw_rows),
            }
        )
    actions = acquisition.self_hashed(
        {
            "active_Agent": p11.CANDIDATE_NAME,
            "hardened_HippoRAG_source_sha256": backport.PATCHED_SOURCE_SHA256,
            "item_count": ITEM_COUNT,
            "items": action_rows,
            "schema": ACTION_SCHEMA,
            "intent_pack_sha256": intents["pack_sha256"],
        },
        field="pack_sha256",
    )
    action_path = root / "three_arm.actions.json"
    train.bright_runtime._write_json(action_path, actions)
    action_file_sha256 = acquisition.file_sha256(action_path)

    labels = load_labels_after_action_seal(
        base=base,
        acquisition_result=acquisition_result,
        items=items,
        action_path=action_path,
        expected_action_sha256=action_file_sha256,
    )
    arm_scores: dict[str, list[int]] = {"P11": [], "RAW": [], "HippoRAG": []}
    recovered_gold_count = 0
    for item, plan, action in zip(items, expanded_plans, action_rows):
        id_to_row = {
            identifier: row for row, identifier in enumerate(corpora[item.family].ids)
        }
        gold_ids = labels[item.item_key]
        if not set(gold_ids) <= set(id_to_row):
            raise NanoBEIRCConfirmError("gold document is absent from corpus")
        gold_rows = tuple(id_to_row[value] for value in gold_ids)
        arm_scores["P11"].append(
            train.bridge.integer_ndcg_at_10(action["P11_rows"], gold_rows)
        )
        arm_scores["RAW"].append(
            train.bridge.integer_ndcg_at_10(action["RAW_rows"], gold_rows)
        )
        arm_scores["HippoRAG"].append(
            train.bridge.integer_ndcg_at_10(
                action["HippoRAG"]["top_rows"], gold_rows
            )
        )
        recovered_gold_count += len(
            (set(gold_rows) - set(plan.local.base_pool)).intersection(
                action["P11_rows"]
            )
        )
    primary_passed, comparisons = primary_decision(
        items=items, arm_scores=arm_scores
    )
    family_aggregates = {
        family: {
            arm: sum(
                arm_scores[arm][index]
                for index, item in enumerate(items)
                if item.family == family
            )
            for arm in arm_scores
        }
        for family in acquisition.FAMILIES
    }
    aggregates = {
        arm: {
            "mean_ndcg_at_10": sum(values) / (ITEM_COUNT * 1_000_000_000),
            "sum_integer_ndcg": sum(values),
        }
        for arm, values in arm_scores.items()
    }
    status = (
        "NanoBEIR_C_confirm_primary_passed_A_form_authorized"
        if primary_passed
        else "NanoBEIR_C_confirm_primary_failed_same_source_P11_stops"
    )
    result = acquisition.self_hashed(
        {
            "aggregates": aggregates,
            "candidate_expansion": {
                "gold_absent_from_base_recovered_by_P11_top10": recovered_gold_count,
                "P11_top10_outside_base_count": sum(
                    row["candidate_expansion"]["P11_top10_documents_outside_base_pool"]
                    for row in action_rows
                ),
                "unique_bridge_candidates_outside_base_count_sum": sum(
                    row["candidate_expansion"]["unique_bridge_candidates_outside_base_pool"]
                    for row in action_rows
                ),
            },
            "claim_boundary": {
                "A_form_label_open_count": 0,
                "C_confirm_label_pack_open_count": 1,
                "external_network_call_count": 0,
                "labels_opened_after_all_action_seal": True,
                "M_search_label_open_count": 0,
                "online_evaluator_call_count": 0,
                "population_inference": False,
            },
            "comparisons": comparisons,
            "execution": {
                "corpus_tensor_bindings": corpus_tensor_bindings,
                "cross_encoder_document_count_sum": sum(
                    len(plan.expanded.expanded_pool) for plan in expanded_plans
                ),
                "HippoRAG_peak_process_concurrency": counter.peak,
                "HippoRAG_terminal_count": len(completed_hippo),
                "qwen_network_audit": qwen_receipt["network_audit"],
                "valid_generation_count": qwen_receipt["valid_generation_count"],
            },
            "family_aggregates": family_aggregates,
            "formal_binding": {
                "acquisition_result_self_sha256": ACQUISITION_RESULT_SELF_SHA256,
                "action_file_sha256": action_file_sha256,
                "action_pack_sha256": actions["pack_sha256"],
                "attempt_marker_sha256": acquisition.file_sha256(marker_path),
                "candidate_freeze_self_sha256": CANDIDATE_FREEZE_SELF_SHA256,
                "formal_execution_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "intent_pack_sha256": intents["pack_sha256"],
            },
            "item_count": ITEM_COUNT,
            "primary_passed": primary_passed,
            "primary_rule": "P11_minus_RAW_and_P11_minus_HippoRAG_strictly_positive_in_aggregate_and_each_of_three_families",
            "recorded_date": "2026-07-20",
            "schema": SCHEMA,
            "status": status,
        }
    )
    train.bright_runtime._write_json(result_path, result, mode=0o644)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.formal:
        raise SystemExit("--formal is required")
    result = run_formal(arguments.project_root)
    print(
        json.dumps(
            {
                "primary_passed": result["primary_passed"],
                "self_sha256": result["self_sha256"],
                "status": result["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
