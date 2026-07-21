"""Frozen direct complete-case P14 C_confirm on three fresh BRIGHT families."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_acquisition_v1 as acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_source_qualification_v1 as source,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_c_confirm_runtime_v1 as p11_runtime,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_bridge_safe_candidate_v1 as candidate,
)


SCHEMA = "bright_p14_direct_c_confirm_result_v1"
ATTEMPT_SCHEMA = "bright_p14_direct_c_confirm_attempt_v1"
INTENT_SCHEMA = "bright_p14_direct_c_confirm_intents_v1"
SELECTION_SCHEMA = "bright_p14_direct_c_confirm_selection_v1"
ACTION_SCHEMA = "bright_p14_direct_c_confirm_actions_v1"
FREEZE_SCHEMA = "bright_p14_direct_c_confirm_freeze_v1"
CANDIDATE_NAME = candidate.CANDIDATE_NAME
FAMILIES = acquisition.FAMILIES
ATTEMPTS_PER_FAMILY = 20
TARGET_PER_FAMILY = 10
ATTEMPT_COUNT = 60
SELECTED_COUNT = 30
HIPPORAG_CONCURRENCY = 12
EXTERNAL_PROCESS_CONCURRENCY = 13

RUN_ROOT_RELATIVE = Path("artifacts/bright_p14_direct_c_confirm_v1")
RESULT_RELATIVE = Path(
    "manifests/bright_p14_direct_c_confirm_result_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/bright_p14_direct_c_confirm_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/bright_p14_direct_c_confirm_v1.py"
)
TEST_RELATIVE = Path("tests/test_bright_p14_direct_c_confirm_v1.py")
ACQUISITION_RESULT_RELATIVE = acquisition.RESULT_RELATIVE
ACQUISITION_RESULT_FILE_SHA256 = (
    "b9528887a80d5fb93b0b2840555b038f6b83907804f9e15f216613d68b5465d7"
)
ACQUISITION_RESULT_SELF_SHA256 = (
    "062b1a2636ec7e756acb798264623375661d2e701c10a8fbac04df7b7f82b9e7"
)
CANDIDATE_FREEZE_SELF_SHA256 = (
    "17f9865483cd3c4846db8a63c1047f8af6bdaa24b78ece09245f3e568e0457f0"
)
STUDY_DESIGN_SELF_SHA256 = (
    "8a4492ec320adb308174c9a26d6b380105e298d8dfccd2a495cfb0fadb9c33c5"
)
HARDENING_RESULT_SELF_SHA256 = (
    "2c9571cf7437d47d6e0dad3317841f77ebc0b782132b945185514f106c0ed8a3"
)
HARDENING_RESULT_FILE_SHA256 = (
    "55f8295539d2e6b1d6c776cf8c7c7e2b7ac6bccbda89aebf5d447f55de854da5"
)
EXPECTED_DOCUMENT_COUNTS = {
    "EARTH_SCIENCE": 121249,
    "PSYCHOLOGY": 52835,
    "SUSTAINABLE_LIVING": 60792,
}
DEPENDENCY_RELATIVES = (
    acquisition.IMPLEMENTATION_RELATIVE,
    source.IMPLEMENTATION_RELATIVE,
    Path("assumption_agent/benchmarks/nanobeir_p11_c_confirm_runtime_v1.py"),
    Path("assumption_agent/benchmarks/nanobeir_p12_c_confirm_runtime_v1.py"),
    Path("assumption_agent/benchmarks/nanobeir_p13_bridge_safe_candidate_v1.py"),
    Path("assumption_agent/benchmarks/p11_raw_ce_rrf_v1.py"),
    Path("assumption_agent/benchmarks/fiqa_bridge_expansion_train_runtime_v1.py"),
    Path("replication_runtime/bright_official_hipporag_v1/contract.py"),
    Path("replication_runtime/bright_official_hipporag_v1/worker.py"),
    Path("replication_runtime/hipporag_upstream_hardening_v1/backport.py"),
)


class P14DirectCConfirmError(RuntimeError):
    """The frozen P14 direct C_confirm runtime failed closed."""


class OneShotRefusal(P14DirectCConfirmError):
    """The formal P14 direct C_confirm root or result is consumed."""


@dataclass(frozen=True)
class RuntimeItem:
    ordinal: int
    family: str
    attempt_ordinal: int
    family_hmac_position: int
    item_key: str
    query: str
    source_query_id: str
    excluded_ids: tuple[str, ...]


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise P14DirectCConfirmError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P14DirectCConfirmError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise P14DirectCConfirmError(f"{name} is not an object")
    return value


def _verify_self(value: Mapping[str, Any], expected: str, name: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != expected or acquisition.utilities.stable_hash(body) != expected:
        raise P14DirectCConfirmError(f"{name} self hash drifted")


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _runtime_freeze(base: Path, project_root: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "P14 C_confirm freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise P14DirectCConfirmError("P14 C_confirm freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise P14DirectCConfirmError("P14 C_confirm freeze hash is absent")
    _verify_self(value, declared, "P14 C_confirm freeze")
    commit = value.get("formal_implementation_commit")
    if (
        not isinstance(commit, str)
        or not acquisition._git_is_ancestor(commit, project_root)
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("candidate_freeze_self_sha256")
        != CANDIDATE_FREEZE_SELF_SHA256
        or value.get("hardening_result_self_sha256")
        != HARDENING_RESULT_SELF_SHA256
    ):
        raise P14DirectCConfirmError("P14 C_confirm prerequisite drifted")

    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise P14DirectCConfirmError("implementation bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    required = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != required:
        raise P14DirectCConfirmError("implementation set drifted")
    for relative, expected in observed.items():
        if (
            not isinstance(expected, str)
            or acquisition.utilities.file_sha256(base / str(relative))
            != expected
        ):
            raise P14DirectCConfirmError("implementation file drifted")

    rows = value.get("dependency_bindings")
    if not isinstance(rows, list):
        raise P14DirectCConfirmError("dependency bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    required = {path.as_posix() for path in DEPENDENCY_RELATIVES}
    if set(observed) != required:
        raise P14DirectCConfirmError("dependency set drifted")
    for relative, expected in observed.items():
        if (
            not isinstance(expected, str)
            or acquisition.utilities.file_sha256(base / str(relative))
            != expected
        ):
            raise P14DirectCConfirmError("dependency file drifted")

    binding = value.get("acquisition_result_binding")
    if not isinstance(binding, Mapping) or dict(binding) != {
        "file_sha256": ACQUISITION_RESULT_FILE_SHA256,
        "self_sha256": ACQUISITION_RESULT_SELF_SHA256,
    }:
        raise P14DirectCConfirmError("acquisition result binding drifted")
    return value


def _verify_hardening_result(base: Path) -> None:
    path = base / p11_runtime.hardening_qualification.RESULT_RELATIVE
    if acquisition.utilities.file_sha256(path) != HARDENING_RESULT_FILE_SHA256:
        raise P14DirectCConfirmError("HippoRAG hardening result file drifted")
    value = _read_json(path, "HippoRAG hardening result")
    _verify_self(value, HARDENING_RESULT_SELF_SHA256, "HippoRAG hardening result")
    if value.get("status") != (
        "passed_upstream_fixed_comparator_qualified_for_future_new_studies_only"
    ):
        raise P14DirectCConfirmError("HippoRAG hardening is not qualified")


def _load_acquisition(
    base: Path, project_root: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    freeze = _runtime_freeze(base, project_root)
    path = base / ACQUISITION_RESULT_RELATIVE
    if acquisition.utilities.file_sha256(path) != (
        ACQUISITION_RESULT_FILE_SHA256
    ):
        raise P14DirectCConfirmError("P14 acquisition result file drifted")
    value = _read_json(path, "P14 acquisition result")
    _verify_self(value, ACQUISITION_RESULT_SELF_SHA256, "P14 acquisition result")
    if (
        value.get("schema") != acquisition.SCHEMA
        or value.get("status")
        != "passed_private_acquisition_ready_for_P14_direct_C_confirm"
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("candidate_name") != CANDIDATE_NAME
    ):
        raise P14DirectCConfirmError("P14 acquisition completion drifted")
    _verify_hardening_result(base)
    return value, freeze


def _pack_binding(result: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    bindings = result.get("pack_bindings")
    if not isinstance(bindings, Mapping) or not isinstance(
        bindings.get(name), Mapping
    ):
        raise P14DirectCConfirmError(f"{name} binding is absent")
    return bindings[name]


def _load_pack(
    base: Path, result: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    binding = _pack_binding(result, name)
    relative = binding.get("relative_path")
    if not isinstance(relative, str):
        raise P14DirectCConfirmError(f"{name} path drifted")
    path = base / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != binding.get("size_bytes")
        or acquisition.utilities.file_sha256(path)
        != binding.get("file_sha256")
    ):
        raise P14DirectCConfirmError(f"{name} file drifted")
    value = _read_json(path, name)
    body = dict(value)
    observed = body.pop("pack_sha256", None)
    if (
        observed != binding.get("pack_sha256")
        or acquisition.utilities.stable_hash(body) != observed
    ):
        raise P14DirectCConfirmError(f"{name} pack hash drifted")
    return value


def load_views(
    base: Path, acquisition_result: Mapping[str, Any]
) -> tuple[RuntimeItem, ...]:
    binding = _pack_binding(acquisition_result, "C_confirm_view")
    if binding.get("item_count") != ATTEMPT_COUNT:
        raise P14DirectCConfirmError("C_confirm view count drifted")
    pack = _load_pack(base, acquisition_result, "C_confirm_view")
    if (
        pack.get("schema") != acquisition.VIEW_SCHEMA
        or pack.get("block") != "C_confirm"
    ):
        raise P14DirectCConfirmError("C_confirm view envelope drifted")
    rows = pack.get("items")
    if not isinstance(rows, list) or len(rows) != ATTEMPT_COUNT:
        raise P14DirectCConfirmError("C_confirm view rows drifted")
    items: list[RuntimeItem] = []
    family_counts = {family: 0 for family in FAMILIES}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "attempt_ordinal",
            "excluded_document_ids",
            "family",
            "family_HMAC_position",
            "item_key",
            "query",
            "source_query_id",
        }:
            raise P14DirectCConfirmError("C_confirm view row shape drifted")
        family = row.get("family")
        attempt_ordinal = row.get("attempt_ordinal")
        hmac_position = row.get("family_HMAC_position")
        if (
            family not in FAMILIES
            or isinstance(attempt_ordinal, bool)
            or not isinstance(attempt_ordinal, int)
            or attempt_ordinal != family_counts[family]
            or hmac_position != attempt_ordinal
        ):
            raise P14DirectCConfirmError("C_confirm family order drifted")
        family_counts[family] += 1
        excluded = row.get("excluded_document_ids")
        if (
            not isinstance(excluded, list)
            or any(not isinstance(value, str) or not value for value in excluded)
            or len(excluded) != len(set(excluded))
        ):
            raise P14DirectCConfirmError("C_confirm exclusions drifted")
        texts = (
            row.get("item_key"),
            row.get("query"),
            row.get("source_query_id"),
        )
        if not all(isinstance(value, str) and value for value in texts):
            raise P14DirectCConfirmError("C_confirm view text drifted")
        items.append(
            RuntimeItem(
                ordinal=ordinal,
                family=family,
                attempt_ordinal=attempt_ordinal,
                family_hmac_position=hmac_position,
                item_key=texts[0],
                query=texts[1],
                source_query_id=texts[2],
                excluded_ids=tuple(excluded),
            )
        )
    if any(count != ATTEMPTS_PER_FAMILY for count in family_counts.values()):
        raise P14DirectCConfirmError("C_confirm family count drifted")
    if len({item.item_key for item in items}) != ATTEMPT_COUNT:
        raise P14DirectCConfirmError("C_confirm item keys are duplicated")
    return tuple(items)


def load_corpora(base: Path) -> Mapping[str, p11_runtime.FamilyCorpus]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise P14DirectCConfirmError("pyarrow is unavailable") from exc
    output: dict[str, p11_runtime.FamilyCorpus] = {}
    for family in FAMILIES:
        slug = source.SLUGS[family]
        relative = f"documents/{slug}-00000-of-00001.parquet"
        path = base / source.SOURCE_ROOT_RELATIVE / relative
        binding = source.SOURCE_FILES[relative]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != binding["size_bytes"]
            or acquisition.utilities.file_sha256(path) != binding["sha256"]
        ):
            raise P14DirectCConfirmError("P14 document source drifted")
        table = pq.read_table(path)
        if table.column_names != ["id", "content"]:
            raise P14DirectCConfirmError("P14 document schema drifted")
        ids: list[str] = []
        contents: list[str] = []
        seen: set[str] = set()
        for row in table.to_pylist():
            identifier = row.get("id")
            content = row.get("content")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in seen
                or not isinstance(content, str)
                or not content.strip()
                or "\x00" in content
            ):
                raise P14DirectCConfirmError("P14 document row drifted")
            seen.add(identifier)
            ids.append(identifier)
            contents.append(content[: source.DOCUMENT_CHARACTER_CAP])
        if len(ids) != EXPECTED_DOCUMENT_COUNTS[family]:
            raise P14DirectCConfirmError("P14 document count drifted")
        output[family] = p11_runtime.FamilyCorpus(
            tuple(ids), tuple(contents)
        )
    return output


def _verify_runtime_assets(base: Path) -> None:
    bright = p11_runtime.train.bright_runtime
    directories = (
        base / bright.QWEN_MODEL_RELATIVE,
        base / bright.MINILM_MODEL_RELATIVE,
        base / bright.HIPPORAG_LLM_RELATIVE,
        base / p11_runtime.train.CROSS_MODEL_RELATIVE,
    )
    runtime_python = base / bright.HIPPORAG_PYTHON_RELATIVE
    baseline = (
        base
        / p11_runtime.hardening_qualification.BASELINE_REPO_RELATIVE
        / p11_runtime.hardening_qualification.BASELINE_SOURCE_WITHIN_REPO
    )
    if any(path.is_symlink() or not path.is_dir() for path in directories):
        raise P14DirectCConfirmError("offline model asset is unavailable")
    if baseline.is_symlink() or not baseline.is_file():
        raise P14DirectCConfirmError("offline runtime asset is unavailable")
    if (
        not runtime_python.is_file()
        or not os.access(runtime_python, os.X_OK)
        or runtime_python.resolve() != Path("/usr/bin/python3.10")
    ):
        raise P14DirectCConfirmError("offline Python binding drifted")
    if shutil.which("bwrap") != "/usr/bin/bwrap" or shutil.which("strace") is None:
        raise P14DirectCConfirmError("offline isolation executable is unavailable")
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != (
        p11_runtime.backport.BASELINE_SOURCE_SHA256
    ):
        raise P14DirectCConfirmError("HippoRAG baseline source drifted")
    try:
        import torch
    except ImportError as exc:
        raise P14DirectCConfirmError("CUDA runtime is unavailable") from exc
    if not torch.cuda.is_available():
        raise P14DirectCConfirmError("the frozen CUDA device is unavailable")
    if p11_runtime.train.qwen_contract.MAXIMUM_ITEM_COUNT < ATTEMPT_COUNT:
        raise P14DirectCConfirmError("Qwen item capacity drifted")


def select_complete_cases(
    items: Sequence[RuntimeItem], terminal_ordinals: Sequence[int]
) -> tuple[bool, tuple[RuntimeItem, ...], Mapping[str, int]]:
    terminal = set(terminal_ordinals)
    if len(terminal) != len(tuple(terminal_ordinals)) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in terminal
    ):
        raise P14DirectCConfirmError("terminal ordinal set drifted")
    item_ordinals = {item.ordinal for item in items}
    if not terminal <= item_ordinals:
        raise P14DirectCConfirmError("terminal ordinal is outside C_confirm")
    selected: list[RuntimeItem] = []
    counts: dict[str, int] = {}
    for family in FAMILIES:
        candidates = sorted(
            (
                item
                for item in items
                if item.family == family and item.ordinal in terminal
            ),
            key=lambda item: item.attempt_ordinal,
        )
        counts[family] = len(candidates)
        selected.extend(candidates[:TARGET_PER_FAMILY])
    capacity = all(count >= TARGET_PER_FAMILY for count in counts.values())
    if not capacity:
        return False, (), counts
    if len(selected) != SELECTED_COUNT:
        raise P14DirectCConfirmError("complete-case selection count drifted")
    return True, tuple(selected), counts


def _paired(left: Sequence[int], right: Sequence[int]) -> Mapping[str, int]:
    if len(left) != len(right):
        raise P14DirectCConfirmError("paired score shape drifted")
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
    if set(arm_scores) != {"Agent", "RAW", "HippoRAG"}:
        raise P14DirectCConfirmError("arm registry drifted")
    if len(items) != SELECTED_COUNT or any(
        len(values) != SELECTED_COUNT for values in arm_scores.values()
    ):
        raise P14DirectCConfirmError("arm score vector drifted")
    comparisons: dict[str, Any] = {}
    for baseline in ("RAW", "HippoRAG"):
        paired = _paired(arm_scores["Agent"], arm_scores[baseline])
        family_nets = {
            family: sum(
                arm_scores["Agent"][index]
                - arm_scores[baseline][index]
                for index, item in enumerate(items)
                if item.family == family
            )
            for family in FAMILIES
        }
        comparisons[f"Agent_minus_{baseline}"] = {
            **paired,
            "family_net_integer_ndcg": family_nets,
        }
    passed = all(
        row["net_integer_ndcg"] > 0
        and all(value > 0 for value in row["family_net_integer_ndcg"].values())
        for row in comparisons.values()
    )
    return passed, comparisons


def _failure_receipt(exc: BaseException) -> Mapping[str, str]:
    name = type(exc).__name__
    digest = hashlib.sha256(
        (name + "\n" + str(exc)).encode("utf-8", errors="replace")
    ).hexdigest()
    return {"exception_type": name, "failure_sha256": digest}


def _valid_terminal_receipt(
    receipt: Mapping[str, Any], base_pool: Sequence[int]
) -> bool:
    rows = receipt.get("top_rows")
    return (
        isinstance(rows, list)
        and len(rows) == 10
        and len(set(rows)) == 10
        and all(
            isinstance(row, int) and not isinstance(row, bool) and row >= 0
            for row in rows
        )
        and set(rows) <= set(base_pool)
        and isinstance(receipt.get("graph_node_count"), int)
        and receipt["graph_node_count"] > 32
        and isinstance(receipt.get("graph_edge_count"), int)
        and receipt["graph_edge_count"] > 0
    )


def _cleanup_hippo_workdirs(roots: Sequence[Path]) -> int:
    removed = 0
    for root in roots:
        for name in ("index", "home", "hf", "tmp"):
            path = root / name
            if path.is_symlink():
                raise P14DirectCConfirmError("HippoRAG work path became a symlink")
            if path.exists():
                shutil.rmtree(path)
                removed += 1
    return removed


def _load_labels_after_action_seal(
    *,
    base: Path,
    acquisition_result: Mapping[str, Any],
    all_items: Sequence[RuntimeItem],
    action_path: Path,
    expected_action_sha256: str,
) -> Mapping[str, tuple[str, ...]]:
    if (
        action_path.is_symlink()
        or not action_path.is_file()
        or acquisition.utilities.file_sha256(action_path)
        != expected_action_sha256
    ):
        raise P14DirectCConfirmError("action seal is absent before labels")
    binding = _pack_binding(acquisition_result, "C_confirm_labels")
    if binding.get("item_count") != ATTEMPT_COUNT:
        raise P14DirectCConfirmError("C_confirm label count drifted")
    pack = _load_pack(base, acquisition_result, "C_confirm_labels")
    if (
        pack.get("schema") != acquisition.LABEL_SCHEMA
        or pack.get("block") != "C_confirm"
    ):
        raise P14DirectCConfirmError("C_confirm label envelope drifted")
    rows = pack.get("items")
    if not isinstance(rows, list) or len(rows) != ATTEMPT_COUNT:
        raise P14DirectCConfirmError("C_confirm label rows drifted")
    labels: dict[str, tuple[str, ...]] = {}
    for item, row in zip(all_items, rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"family", "gold_document_ids", "item_key"}
            or row.get("family") != item.family
            or row.get("item_key") != item.item_key
        ):
            raise P14DirectCConfirmError("C_confirm label identity drifted")
        gold = row.get("gold_document_ids")
        if (
            not isinstance(gold, list)
            or not gold
            or any(not isinstance(value, str) or not value for value in gold)
            or len(gold) != len(set(gold))
        ):
            raise P14DirectCConfirmError("C_confirm gold list drifted")
        labels[item.item_key] = tuple(gold)
    return labels


def _write_capacity_result(
    *,
    project_root: Path,
    base: Path,
    root: Path,
    result_path: Path,
    freeze: Mapping[str, Any],
    marker_path: Path,
    intents: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_path: Path,
    terminal_counts: Mapping[str, int],
    failure_counts: Mapping[str, int],
    counter: Any,
    qwen_receipt: Mapping[str, Any],
    corpus_tensor_bindings: Mapping[str, Any],
    cross_output_path: Path,
    removed_work_directory_count: int,
) -> Mapping[str, Any]:
    result = acquisition.utilities.self_hashed(
        {
            "attempt_count": ATTEMPT_COUNT,
            "capacity_passed": False,
            "claim_boundary": {
                "action_seal_count": 0,
                "C_confirm_label_pack_open_count": 0,
                "external_network_call_count": 0,
                "performance_score_count": 0,
                "population_inference": False,
            },
            "execution": {
                "corpus_tensor_bindings": corpus_tensor_bindings,
                "cross_encoder_output_file_sha256": (
                    acquisition.utilities.file_sha256(cross_output_path)
                ),
                "HippoRAG_attempt_count": ATTEMPT_COUNT,
                "HippoRAG_failure_counts_by_family": dict(failure_counts),
                "HippoRAG_peak_process_concurrency": counter.peak,
                "HippoRAG_terminal_counts_by_family": dict(terminal_counts),
                "qwen_network_audit": qwen_receipt["network_audit"],
                "source_valid_generation_count": qwen_receipt[
                    "source_valid_generation_count"
                ],
                "totalized_generation_count": qwen_receipt[
                    "totalized_generation_count"
                ],
                "working_directory_count_removed": removed_work_directory_count,
            },
            "formal_binding": {
                "acquisition_result_self_sha256": (
                    ACQUISITION_RESULT_SELF_SHA256
                ),
                "attempt_marker_sha256": acquisition.utilities.file_sha256(
                    marker_path
                ),
                "candidate_freeze_self_sha256": (
                    CANDIDATE_FREEZE_SELF_SHA256
                ),
                "complete_case_selection_file_sha256": (
                    acquisition.utilities.file_sha256(selection_path)
                ),
                "complete_case_selection_pack_sha256": selection[
                    "pack_sha256"
                ],
                "formal_execution_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "intent_pack_sha256": intents["pack_sha256"],
            },
            "primary_evaluated": False,
            "primary_passed": None,
            "recorded_date": "2026-07-21",
            "schema": SCHEMA,
            "status": (
                "P14_direct_C_confirm_capacity_failed_same_source_stops"
            ),
            "target_terminal_count_per_family": TARGET_PER_FAMILY,
        }
    )
    p11_runtime.train.bright_runtime._write_json(
        result_path, result, mode=0o644
    )
    return result


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("P14 direct C_confirm root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("P14 direct C_confirm result already exists")

    acquisition_result, freeze = _load_acquisition(base, project_root)
    items = load_views(base, acquisition_result)
    corpora = load_corpora(base)
    _verify_runtime_assets(base)

    root.mkdir(mode=0o700)
    marker = {
        "acquisition_result_self_sha256": ACQUISITION_RESULT_SELF_SHA256,
        "candidate_freeze_self_sha256": CANDIDATE_FREEZE_SELF_SHA256,
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
    }
    marker_path = root / "attempt.marker"
    p11_runtime.train.bright_runtime._write_json(marker_path, marker)
    patched_source = p11_runtime._materialize_hardened_source(base, root)

    bright_items = tuple(
        p11_runtime.train.bright_runtime.ViewItem(
            ordinal=item.ordinal,
            family=item.family,
            commitment=item.item_key,
            query=item.query,
            excluded_ids=item.excluded_ids,
        )
        for item in items
    )
    try:
        raw_qwen_output, raw_qwen_receipt = (
            p11_runtime.train.bright_runtime._run_qwen(
                base, root, bright_items
            )
        )
        qwen_output, qwen_audit = (
            candidate.totalize_and_project_qwen_output(
                raw_qwen_output, bright_items
            )
        )
    except Exception as exc:
        raise P14DirectCConfirmError("typed query execution failed") from exc
    qwen_audit_path = root / "qwen.totalized_projected.json"
    p11_runtime.train.bright_runtime._write_json(qwen_audit_path, qwen_audit)
    qwen_receipt = {
        **raw_qwen_receipt,
        "projected_audit_file_sha256": acquisition.utilities.file_sha256(
            qwen_audit_path
        ),
        "projected_audit_pack_sha256": qwen_audit["pack_sha256"],
        "source_valid_generation_count": qwen_audit[
            "source_valid_generation_count"
        ],
        "totalized_generation_count": qwen_audit[
            "totalized_generation_count"
        ],
        "valid_generation_count": ATTEMPT_COUNT,
    }
    qwen_rows = qwen_output.get("items")
    if (
        not isinstance(qwen_rows, list)
        or len(qwen_rows) != ATTEMPT_COUNT
        or not all(row.get("generation_valid") is True for row in qwen_rows)
    ):
        raise P14DirectCConfirmError("typed query totalization drifted")

    try:
        encoder = p11_runtime.train.bright_runtime._new_minilm(base)
        corpus_embeddings = {
            family: p11_runtime.train.bright_runtime._encode_chunks(
                encoder, corpora[family].contents
            )
            for family in FAMILIES
        }
    except Exception as exc:
        raise P14DirectCConfirmError("P14 corpus embedding failed") from exc
    tensor_root = root / "corpus_tensors"
    tensor_root.mkdir(mode=0o700)
    corpus_tensor_bindings: dict[str, Any] = {}
    for family in FAMILIES:
        matrix = np.asarray(corpus_embeddings[family], dtype=np.float32)
        if (
            matrix.shape != (len(corpora[family].ids), 384)
            or not np.isfinite(matrix).all()
        ):
            raise P14DirectCConfirmError("corpus embedding tensor drifted")
        path = tensor_root / f"{family}.embeddings.npy"
        p11_runtime.train.bright_runtime._save_npy_exclusive(path, matrix)
        corpus_tensor_bindings[family] = {
            "document_count": len(corpora[family].ids),
            "file_sha256": acquisition.utilities.file_sha256(path),
            "float32_sha256": p11_runtime.train.float32_matrix_sha256(matrix),
            "shape": [len(corpora[family].ids), 384],
        }

    flattened_queries: list[str] = []
    query_slices: list[tuple[int, int]] = []
    for item, row in zip(items, qwen_rows):
        expansions = row.get("expansions")
        if not isinstance(expansions, list) or len(expansions) != 4:
            raise P14DirectCConfirmError("typed query expansion row drifted")
        start = len(flattened_queries)
        flattened_queries.extend([item.query, *expansions])
        query_slices.append((start, len(flattened_queries)))
    query_embeddings = p11_runtime.train.bright_runtime._encode_chunks(
        encoder, flattened_queries
    )
    local_plans: list[p11_runtime.train.LocalPlan] = []
    for item, row, (start, end) in zip(items, qwen_rows, query_slices):
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
        try:
            local_plans.append(
                p11_runtime.train.build_local_plan(
                    item=view,
                    document_ids=corpora[item.family].ids,
                    document_contents=corpora[item.family].contents,
                    query_score_vectors=scores,
                    expansions=row["expansions"],
                )
            )
        except Exception as exc:
            raise P14DirectCConfirmError("candidate pool formation failed") from exc

    bridge_queries = [
        query.text for plan in local_plans for query in plan.bridge_queries
    ]
    bridge_embeddings = (
        p11_runtime.train.bright_runtime._encode_chunks(encoder, bridge_queries)
        if bridge_queries
        else np.empty((0, 384), dtype=np.float32)
    )
    expanded_plans: list[p11_runtime.train.ExpandedPlan] = []
    offset = 0
    for item, plan in zip(items, local_plans):
        count = len(plan.bridge_queries)
        vectors = [
            p11_runtime.train.quantized_scores(
                corpus_embeddings[item.family], bridge_embeddings[index]
            )
            for index in range(offset, offset + count)
        ]
        try:
            expanded_plans.append(p11_runtime.train.expand_plan(plan, vectors))
        except Exception as exc:
            raise P14DirectCConfirmError("candidate expansion failed") from exc
        offset += count
    if offset != len(bridge_queries):
        raise P14DirectCConfirmError("bridge embedding accounting drifted")
    query_embedding_path = root / "typed_query.embeddings.npy"
    bridge_embedding_path = root / "bridge_query.embeddings.npy"
    p11_runtime.train.bright_runtime._save_npy_exclusive(
        query_embedding_path, query_embeddings
    )
    p11_runtime.train.bright_runtime._save_npy_exclusive(
        bridge_embedding_path, bridge_embeddings
    )
    del encoder, query_embeddings, bridge_embeddings, corpus_embeddings
    p11_runtime._release_cuda()

    cross_payload = p11_runtime._prepare_cross_input(
        plans=expanded_plans, items=items, corpora=corpora
    )
    cross_input_path = root / "cross_encoder.input.json"
    cross_output_path = root / "cross_encoder.output.json"
    p11_runtime.train.bright_runtime._write_exclusive(
        cross_input_path,
        p11_runtime.train.cross_contract.canonical_json_bytes(cross_payload),
        mode=0o600,
    )
    hippo_roots = p11_runtime._prepare_hipporag_inputs(
        root=root, plans=expanded_plans, items=items, corpora=corpora
    )
    intents = acquisition.utilities.self_hashed(
        {
            "cross_encoder_input_file_sha256": (
                acquisition.utilities.file_sha256(cross_input_path)
            ),
            "items": [
                {
                    "base_pool": list(plan.local.base_pool),
                    "expanded_pool": list(plan.expanded.expanded_pool),
                    "family": item.family,
                    "hipporag_input_file_sha256": (
                        acquisition.utilities.file_sha256(
                            item_root / "input.json"
                        )
                    ),
                    "item_key": item.item_key,
                    "ordinal": item.ordinal,
                }
                for item, plan, item_root in zip(
                    items, expanded_plans, hippo_roots
                )
            ],
            "schema": INTENT_SCHEMA,
            "typed_query_projected_pack_sha256": qwen_audit["pack_sha256"],
        },
        field="pack_sha256",
    )
    intent_path = root / "action.intents.json"
    p11_runtime.train.bright_runtime._write_json(intent_path, intents)

    semaphore = threading.Semaphore(HIPPORAG_CONCURRENCY)
    counter = p11_runtime.train.bright_runtime._ConcurrencyCounter()
    completed_hippo: dict[int, Mapping[str, Any]] = {}
    hippo_failures: dict[int, Mapping[str, str]] = {}
    cross_error: BaseException | None = None
    environment_updates = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous_environment = {
        key: os.environ.get(key) for key in environment_updates
    }
    os.environ.update(environment_updates)
    try:
        with ThreadPoolExecutor(
            max_workers=EXTERNAL_PROCESS_CONCURRENCY
        ) as executor:
            cross_future: Future[Any] = executor.submit(
                p11_runtime.train.cross_worker.run,
                input_path=cross_input_path,
                output_path=cross_output_path,
                model_root=base / p11_runtime.train.CROSS_MODEL_RELATIVE,
            )
            hippo_futures = {
                executor.submit(
                    p11_runtime._run_hardened_hipporag_item,
                    base=base,
                    item_root=item_root,
                    candidate_rows=plan.local.base_pool,
                    patched_source=patched_source,
                    semaphore=semaphore,
                    counter=counter,
                ): item.ordinal
                for item, plan, item_root in zip(
                    items, expanded_plans, hippo_roots
                )
            }
            for future in as_completed([cross_future, *hippo_futures]):
                if future is cross_future:
                    try:
                        future.result()
                    except BaseException as exc:
                        cross_error = exc
                    continue
                ordinal = hippo_futures[future]
                try:
                    receipt = future.result()
                    plan = expanded_plans[ordinal]
                    if not _valid_terminal_receipt(
                        receipt, plan.local.base_pool
                    ):
                        raise P14DirectCConfirmError(
                            "HippoRAG terminal receipt drifted"
                        )
                    completed_hippo[ordinal] = receipt
                except BaseException as exc:
                    hippo_failures[ordinal] = _failure_receipt(exc)
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if cross_error is not None:
        raise P14DirectCConfirmError("cross-encoder execution failed") from cross_error
    if (
        counter.current != 0
        or counter.peak > HIPPORAG_CONCURRENCY
        or set(completed_hippo).intersection(hippo_failures)
        or set(completed_hippo).union(hippo_failures)
        != set(range(ATTEMPT_COUNT))
        or not cross_output_path.is_file()
    ):
        raise P14DirectCConfirmError("external action accounting drifted")
    removed_work_directory_count = _cleanup_hippo_workdirs(hippo_roots)

    capacity_passed, selected_items, terminal_counts = select_complete_cases(
        items, tuple(completed_hippo)
    )
    selected_ordinals = {item.ordinal for item in selected_items}
    selection = acquisition.utilities.self_hashed(
        {
            "capacity_passed": capacity_passed,
            "HippoRAG_launch_count": ATTEMPT_COUNT,
            "items": [
                {
                    "attempt_ordinal": item.attempt_ordinal,
                    "failure_sha256": (
                        hippo_failures[item.ordinal]["failure_sha256"]
                        if item.ordinal in hippo_failures
                        else None
                    ),
                    "family": item.family,
                    "item_key": item.item_key,
                    "ordinal": item.ordinal,
                    "selected": item.ordinal in selected_ordinals,
                    "terminal": item.ordinal in completed_hippo,
                }
                for item in items
            ],
            "schema": SELECTION_SCHEMA,
            "selection_rule": (
                "first_10_terminal_attempts_per_family_in_frozen_HMAC_order"
            ),
            "target_terminal_count_per_family": TARGET_PER_FAMILY,
        },
        field="pack_sha256",
    )
    selection_path = root / "complete_case.selection.json"
    p11_runtime.train.bright_runtime._write_json(selection_path, selection)
    failure_counts = {
        family: sum(
            item.family == family and item.ordinal in hippo_failures
            for item in items
        )
        for family in FAMILIES
    }
    if not capacity_passed:
        return _write_capacity_result(
            project_root=project_root,
            base=base,
            root=root,
            result_path=result_path,
            freeze=freeze,
            marker_path=marker_path,
            intents=intents,
            selection=selection,
            selection_path=selection_path,
            terminal_counts=terminal_counts,
            failure_counts=failure_counts,
            counter=counter,
            qwen_receipt=qwen_receipt,
            corpus_tensor_bindings=corpus_tensor_bindings,
            cross_output_path=cross_output_path,
            removed_work_directory_count=removed_work_directory_count,
        )

    try:
        cross_output = p11_runtime.train.cross_contract.parse_output(
            cross_output_path.read_bytes()
        )
    except Exception as exc:
        raise P14DirectCConfirmError("cross-encoder output drifted") from exc
    cross_rows = cross_output.get("items")
    if not isinstance(cross_rows, list) or len(cross_rows) != ATTEMPT_COUNT:
        raise P14DirectCConfirmError("cross-encoder item count drifted")
    cross_by_ordinal: dict[int, Mapping[str, Any]] = {}
    for ordinal, row in enumerate(cross_rows):
        if not isinstance(row, Mapping) or row.get("ordinal") != ordinal:
            raise P14DirectCConfirmError("cross-encoder row identity drifted")
        cross_by_ordinal[ordinal] = row

    action_rows: list[dict[str, Any]] = []
    selected_plans: list[p11_runtime.train.ExpandedPlan] = []
    for item in selected_items:
        plan = expanded_plans[item.ordinal]
        cross_row = cross_by_ordinal[item.ordinal]
        if cross_row.get("document_count") != len(
            plan.expanded.expanded_pool
        ):
            raise P14DirectCConfirmError("cross-encoder document count drifted")
        try:
            agent_rows = p11_runtime.p11.rank_p11(
                expanded_pool=plan.expanded.expanded_pool,
                raw_top10=plan.local.raw_rows,
                cross_encoder_relation_scores=cross_row[
                    "relation_scores_quantized"
                ],
                cross_encoder_mechanism_scores=cross_row[
                    "mechanism_scores_quantized"
                ],
            )
        except Exception as exc:
            raise P14DirectCConfirmError("Agent rank formation failed") from exc
        hippo = dict(completed_hippo[item.ordinal])
        ids = corpora[item.family].ids
        action_rows.append(
            {
                "attempt_ordinal": item.attempt_ordinal,
                "bridge_anchor_count": len(plan.local.anchors),
                "bridge_query_count": len(plan.local.bridge_queries),
                "candidate_expansion": {
                    "expanded_pool_size": len(plan.expanded.expanded_pool),
                    "Agent_top10_documents_outside_base_pool": len(
                        set(agent_rows) - set(plan.local.base_pool)
                    ),
                    "unique_bridge_candidates_outside_base_pool": len(
                        set(plan.expanded.expanded_pool)
                        - set(plan.local.base_pool)
                    ),
                },
                "family": item.family,
                "HippoRAG": {
                    **hippo,
                    "document_ids": [ids[row] for row in hippo["top_rows"]],
                },
                "item_key": item.item_key,
                "ordinal": item.ordinal,
                "Agent_document_ids": [ids[row] for row in agent_rows],
                "Agent_rows": list(agent_rows),
                "RAW_document_ids": [
                    ids[row] for row in plan.local.raw_rows
                ],
                "RAW_rows": list(plan.local.raw_rows),
            }
        )
        selected_plans.append(plan)
    actions = acquisition.utilities.self_hashed(
        {
            "active_Agent": CANDIDATE_NAME,
            "complete_case_selection_pack_sha256": selection["pack_sha256"],
            "hardened_HippoRAG_source_sha256": (
                p11_runtime.backport.PATCHED_SOURCE_SHA256
            ),
            "item_count": SELECTED_COUNT,
            "items": action_rows,
            "schema": ACTION_SCHEMA,
            "intent_pack_sha256": intents["pack_sha256"],
        },
        field="pack_sha256",
    )
    action_path = root / "three_arm.actions.json"
    p11_runtime.train.bright_runtime._write_json(action_path, actions)
    action_file_sha256 = acquisition.utilities.file_sha256(action_path)

    labels = _load_labels_after_action_seal(
        base=base,
        acquisition_result=acquisition_result,
        all_items=items,
        action_path=action_path,
        expected_action_sha256=action_file_sha256,
    )
    id_to_row = {
        family: {
            identifier: row
            for row, identifier in enumerate(corpora[family].ids)
        }
        for family in FAMILIES
    }
    arm_scores: dict[str, list[int]] = {
        "Agent": [],
        "RAW": [],
        "HippoRAG": [],
    }
    recovered_gold_count = 0
    for item, plan, action in zip(
        selected_items, selected_plans, action_rows
    ):
        gold_ids = labels[item.item_key]
        if not set(gold_ids) <= set(id_to_row[item.family]):
            raise P14DirectCConfirmError("gold document is absent from corpus")
        gold_rows = tuple(id_to_row[item.family][value] for value in gold_ids)
        arm_scores["Agent"].append(
            p11_runtime.train.bridge.integer_ndcg_at_10(
                action["Agent_rows"], gold_rows
            )
        )
        arm_scores["RAW"].append(
            p11_runtime.train.bridge.integer_ndcg_at_10(
                action["RAW_rows"], gold_rows
            )
        )
        arm_scores["HippoRAG"].append(
            p11_runtime.train.bridge.integer_ndcg_at_10(
                action["HippoRAG"]["top_rows"], gold_rows
            )
        )
        recovered_gold_count += len(
            (set(gold_rows) - set(plan.local.base_pool)).intersection(
                action["Agent_rows"]
            )
        )
    primary_passed, comparisons = primary_decision(
        items=selected_items, arm_scores=arm_scores
    )
    family_aggregates = {
        family: {
            arm: sum(
                arm_scores[arm][index]
                for index, item in enumerate(selected_items)
                if item.family == family
            )
            for arm in arm_scores
        }
        for family in FAMILIES
    }
    aggregates = {
        arm: {
            "mean_ndcg_at_10": sum(values)
            / (SELECTED_COUNT * 1_000_000_000),
            "sum_integer_ndcg": sum(values),
        }
        for arm, values in arm_scores.items()
    }
    status = (
        "P14_direct_C_confirm_primary_passed_A_form_authorized"
        if primary_passed
        else "P14_direct_C_confirm_primary_failed_same_source_stops"
    )
    result = acquisition.utilities.self_hashed(
        {
            "aggregates": aggregates,
            "attempt_count": ATTEMPT_COUNT,
            "candidate_expansion": {
                "Agent_top10_outside_base_count": sum(
                    row["candidate_expansion"][
                        "Agent_top10_documents_outside_base_pool"
                    ]
                    for row in action_rows
                ),
                "gold_absent_from_base_recovered_by_Agent_top10": (
                    recovered_gold_count
                ),
                "unique_bridge_candidates_outside_base_count_sum": sum(
                    row["candidate_expansion"][
                        "unique_bridge_candidates_outside_base_pool"
                    ]
                    for row in action_rows
                ),
            },
            "capacity_passed": True,
            "claim_boundary": {
                "A_form_label_open_count": 0,
                "C_confirm_label_pack_open_count": 1,
                "C_confirm_label_rows_read_after_selection": ATTEMPT_COUNT,
                "C_confirm_selected_label_score_count": SELECTED_COUNT,
                "external_network_call_count": 0,
                "labels_opened_after_selected_action_seal": True,
                "M_search_label_open_count": 0,
                "online_evaluator_call_count": 0,
                "population_inference": False,
            },
            "comparisons": comparisons,
            "execution": {
                "corpus_tensor_bindings": corpus_tensor_bindings,
                "cross_encoder_document_count_sum": sum(
                    len(plan.expanded.expanded_pool)
                    for plan in expanded_plans
                ),
                "HippoRAG_attempt_count": ATTEMPT_COUNT,
                "HippoRAG_failure_counts_by_family": failure_counts,
                "HippoRAG_peak_process_concurrency": counter.peak,
                "HippoRAG_terminal_counts_by_family": dict(terminal_counts),
                "qwen_network_audit": qwen_receipt["network_audit"],
                "source_valid_generation_count": qwen_receipt[
                    "source_valid_generation_count"
                ],
                "totalized_generation_count": qwen_receipt[
                    "totalized_generation_count"
                ],
                "working_directory_count_removed": (
                    removed_work_directory_count
                ),
            },
            "family_aggregates": family_aggregates,
            "formal_binding": {
                "acquisition_result_self_sha256": (
                    ACQUISITION_RESULT_SELF_SHA256
                ),
                "action_file_sha256": action_file_sha256,
                "action_pack_sha256": actions["pack_sha256"],
                "attempt_marker_sha256": acquisition.utilities.file_sha256(
                    marker_path
                ),
                "candidate_freeze_self_sha256": (
                    CANDIDATE_FREEZE_SELF_SHA256
                ),
                "complete_case_selection_file_sha256": (
                    acquisition.utilities.file_sha256(selection_path)
                ),
                "complete_case_selection_pack_sha256": selection[
                    "pack_sha256"
                ],
                "formal_execution_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "intent_pack_sha256": intents["pack_sha256"],
            },
            "item_count": SELECTED_COUNT,
            "primary_evaluated": True,
            "primary_passed": primary_passed,
            "primary_rule": (
                "Agent_minus_RAW_and_Agent_minus_HippoRAG_strictly_positive_"
                "in_aggregate_and_each_of_three_families"
            ),
            "recorded_date": "2026-07-21",
            "schema": SCHEMA,
            "status": status,
        }
    )
    p11_runtime.train.bright_runtime._write_json(
        result_path, result, mode=0o644
    )
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
