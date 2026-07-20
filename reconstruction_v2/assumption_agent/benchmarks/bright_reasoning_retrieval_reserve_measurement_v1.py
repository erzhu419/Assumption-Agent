"""Crash-resilient, late-label BRIGHT reserve measurement."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Iterable, Mapping, Sequence

from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_acquisition_v1 as acquisition
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as base
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v3 as v3
from replication_runtime.bright_official_hipporag_v1.contract import (
    parse_output as parse_hipporag_output,
)


VERSION = "bright_reasoning_retrieval_reserve_measurement_v1"
PREPARE_SCHEMA = f"{VERSION}_prepare_result"
ACTION_RESULT_SCHEMA = f"{VERSION}_action_result"
FINAL_SCHEMA = f"{VERSION}_final_result"
INTENT_SCHEMA = f"{VERSION}_action_intent"
THREE_ARM_SCHEMA = f"{VERSION}_three_arm_action"
SCORED_SCHEMA = f"{VERSION}_scored"
MARKER_SCHEMA = f"{VERSION}_attempt"

ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_reserve_measurement_v1")
PREPARE_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_prepare_result_v1.json"
)
ACTION_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_actions_result_v1.json"
)
FINAL_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_final_result_v1.json"
)
ACQUISITION_RESULT_RELATIVE = acquisition.RESULT_RELATIVE
ACQUISITION_PRIVATE_RELATIVE = acquisition.PRIVATE_RELATIVE

A_FORM_RESULT_FILE_SHA256 = (
    "b23a001a24102a7869cd55532d7025d8dc1335a37a79a135cbb27c2f5af69a30"
)
F_RESULT_FILE_SHA256 = (
    "45ec7ddcae3198981819fb37540e3c221a88a2a1f627396f2f566582b5a7e087"
)
F_RESULT_SHA256 = "641f5cc794820f1c95b40edcd861714445f8d827c8b9dc91d8f7b17cc30ced09"
A_HOLD_RESULT_FILE_SHA256 = (
    "9c86739061a38506295fcfcc6a1d7ed80ccf423b2073a59a22389be72d77b26e"
)
A_HOLD_RESULT_SHA256 = (
    "75e2ba838f2b66fa1dedd095d3834359dccc8b721632acb30a368ad1459cb4a4"
)
P_BASE = "P6_RELATION_MECHANISM_RRF"
HIPPORAG_CONCURRENCY = 12
CONTROLLER_LAUNCH_BUDGET = 2


class BrightReserveMeasurementError(RuntimeError):
    """Reserve measurement failed closed."""


def _read_canonical(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BrightReserveMeasurementError(f"{field} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightReserveMeasurementError(f"{field} is invalid") from exc
    if not isinstance(value, dict) or base.canonical_json_bytes(value) != raw:
        raise BrightReserveMeasurementError(f"{field} is not canonical")
    return value


def _verify_result(
    path: Path, *, file_hash: str, result_hash: str, stage: str
) -> dict[str, Any]:
    if base.file_sha256(path) != file_hash:
        raise BrightReserveMeasurementError(f"{stage} file binding drifted")
    value = _read_canonical(path, stage)
    if base.verify_self_hash(value, "result_sha256") != result_hash:
        raise BrightReserveMeasurementError(f"{stage} result binding drifted")
    return value


def _verify_preconditions(project_root: Path) -> dict[str, Any]:
    freeze = acquisition._verify_freeze(project_root)
    acquisition_result = _read_canonical(
        project_root / ACQUISITION_RESULT_RELATIVE, "reserve acquisition result"
    )
    if (
        acquisition_result.get("schema") != acquisition.RESULT_SCHEMA
        or acquisition_result.get("status")
        != "fresh_RESERVE_R_search_acquired_labels_sealed"
    ):
        raise BrightReserveMeasurementError("reserve acquisition did not complete")
    base.verify_self_hash(acquisition_result, "result_sha256")
    if (
        acquisition_result.get("formal_binding", {}).get("design_self_sha256")
        != acquisition.DESIGN_SELF_SHA256
        or acquisition_result.get("formal_binding", {}).get(
            "implementation_freeze_self_sha256"
        )
        != freeze["self_sha256"]
    ):
        raise BrightReserveMeasurementError("reserve acquisition binding drifted")
    a_form = _verify_result(
        project_root / v3.A_FORM_RESULT_RELATIVE,
        file_hash=A_FORM_RESULT_FILE_SHA256,
        result_hash="11a23a0a2c95c3351e5c49147ded59734688349aa48a46aaa5660a73d9cbf60e",
        stage="A_form",
    )
    f_result = _verify_result(
        project_root / v3.F_RESULT_RELATIVE,
        file_hash=F_RESULT_FILE_SHA256,
        result_hash=F_RESULT_SHA256,
        stage="F_search",
    )
    a_hold = _verify_result(
        project_root / v3.A_HOLD_RESULT_RELATIVE,
        file_hash=A_HOLD_RESULT_FILE_SHA256,
        result_hash=A_HOLD_RESULT_SHA256,
        stage="A_hold",
    )
    if (
        f_result.get("P_base") != P_BASE
        or a_hold.get("promoted") is not False
        or a_hold.get("active_evaluator") != "E0_GLOBAL_P_BASE"
    ):
        raise BrightReserveMeasurementError("retained policy state drifted")
    return {
        "A_form": a_form,
        "A_hold": a_hold,
        "acquisition": acquisition_result,
        "F_search": f_result,
        "freeze": freeze,
    }


def _cohort_binding(preconditions: Mapping[str, Any]) -> Mapping[str, Any]:
    cohort = preconditions["acquisition"].get("cohort")
    if not isinstance(cohort, Mapping) or cohort.get("item_count") != acquisition.ITEM_COUNT:
        raise BrightReserveMeasurementError("cohort binding drifted")
    return cohort


def _load_view(
    project_root: Path, preconditions: Mapping[str, Any]
) -> tuple[base.ViewItem, ...]:
    binding = _cohort_binding(preconditions)
    path = project_root / ACQUISITION_PRIVATE_RELATIVE / f"{acquisition.BLOCK}.view.json"
    if base.file_sha256(path) != binding.get("view_pack_file_sha256"):
        raise BrightReserveMeasurementError("reserve view file drifted")
    value = _read_canonical(path, "reserve view")
    if (
        value.get("schema") != acquisition.VIEW_SCHEMA
        or value.get("block") != acquisition.BLOCK
        or value.get("item_count") != acquisition.ITEM_COUNT
        or base.verify_self_hash(value, "pack_sha256")
        != binding.get("view_pack_sha256")
    ):
        raise BrightReserveMeasurementError("reserve view contract drifted")
    rows = value.get("items")
    if not isinstance(rows, list) or len(rows) != acquisition.ITEM_COUNT:
        raise BrightReserveMeasurementError("reserve view rows drifted")
    items: list[base.ViewItem] = []
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "excluded_ids",
            "family",
            "item_commitment_sha256",
            "ordinal",
            "query",
        }:
            raise BrightReserveMeasurementError("reserve view row shape drifted")
        excluded = raw.get("excluded_ids")
        if isinstance(excluded, (str, bytes)) or not isinstance(excluded, list):
            raise BrightReserveMeasurementError("reserve excluded IDs drifted")
        items.append(
            base.ViewItem(
                ordinal=position,
                family=str(raw.get("family")),
                commitment=str(raw.get("item_commitment_sha256")),
                query=base._required_text(raw.get("query"), "query"),
                excluded_ids=tuple(
                    base._required_text(value, "excluded ID") for value in excluded
                ),
            )
        )
        if raw.get("ordinal") != position or items[-1].family not in base.core.FAMILY_ORDER:
            raise BrightReserveMeasurementError("reserve view row identity drifted")
    if Counter(item.family for item in items) != Counter(
        {family: acquisition.COUNT_PER_FAMILY for family in base.core.FAMILY_ORDER}
    ):
        raise BrightReserveMeasurementError("reserve view family balance drifted")
    return tuple(items)


def _load_labels(
    project_root: Path,
    preconditions: Mapping[str, Any],
    items: Sequence[base.ViewItem],
) -> tuple[tuple[str, ...], ...]:
    binding = _cohort_binding(preconditions)
    path = project_root / ACQUISITION_PRIVATE_RELATIVE / f"{acquisition.BLOCK}.labels.json"
    if base.file_sha256(path) != binding.get("label_pack_file_sha256"):
        raise BrightReserveMeasurementError("reserve label file drifted")
    value = _read_canonical(path, "reserve labels")
    if (
        value.get("schema") != acquisition.LABEL_SCHEMA
        or value.get("block") != acquisition.BLOCK
        or value.get("item_count") != len(items)
        or base.verify_self_hash(value, "pack_sha256")
        != binding.get("label_pack_sha256")
    ):
        raise BrightReserveMeasurementError("reserve label contract drifted")
    rows = value.get("items")
    if not isinstance(rows, list) or len(rows) != len(items):
        raise BrightReserveMeasurementError("reserve label rows drifted")
    output: list[tuple[str, ...]] = []
    for position, (raw, item) in enumerate(zip(rows, items)):
        if not isinstance(raw, Mapping) or set(raw) != {
            "gold_ids",
            "item_commitment_sha256",
            "ordinal",
        }:
            raise BrightReserveMeasurementError("reserve label row shape drifted")
        gold = raw.get("gold_ids")
        if (
            raw.get("ordinal") != position
            or raw.get("item_commitment_sha256") != item.commitment
            or isinstance(gold, (str, bytes))
            or not isinstance(gold, list)
        ):
            raise BrightReserveMeasurementError("reserve label row identity drifted")
        values = tuple(base._required_text(value, "gold ID") for value in gold)
        if not values or len(set(values)) != len(values):
            raise BrightReserveMeasurementError("reserve gold IDs drifted")
        output.append(values)
    return tuple(output)


def _start_prepare(project_root: Path, preconditions: Mapping[str, Any]) -> Path:
    root = project_root / ROOT_RELATIVE
    if (
        root.exists()
        or (project_root / PREPARE_RESULT_RELATIVE).exists()
        or (project_root / ACTION_RESULT_RELATIVE).exists()
        or (project_root / FINAL_RESULT_RELATIVE).exists()
    ):
        raise BrightReserveMeasurementError("reserve prepare is one-shot")
    root.mkdir(mode=0o700)
    marker = base.self_hashed(
        {
            "acquisition_result_sha256": preconditions["acquisition"]["result_sha256"],
            "design_self_sha256": acquisition.DESIGN_SELF_SHA256,
            "implementation_freeze_self_sha256": preconditions["freeze"]["self_sha256"],
            "schema": MARKER_SCHEMA,
        },
        "attempt_sha256",
    )
    base._write_json(root / "attempt.marker", marker)
    return root


def prepare(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    v3._activate_v3()
    preconditions = _verify_preconditions(project_root)
    root = _start_prepare(project_root, preconditions)
    items = _load_view(project_root, preconditions)
    qwen_output, qwen_receipt = v3._run_qwen_v3(project_root, root, items)
    corpus = v3._load_corpus_v3(project_root)
    action_pack, embeddings = base._local_actions(
        project_root=project_root,
        stage_root=root,
        items=items,
        qwen_output=qwen_output,
        corpus=corpus,
    )
    action_rows = base._validate_action_pack(action_pack, items)
    item_roots = base._prepare_hipporag_inputs(
        project_root=project_root,
        root=root,
        items=items,
        action_pack=action_pack,
        corpus=corpus,
    )
    f_result = preconditions["F_search"]
    spec = base._evaluator_spec_from_result(f_result)
    a_families, a_utilities, _a_raw, a_embeddings = base._load_scored_stage(
        project_root, "A_form"
    )
    intent_rows: list[dict[str, Any]] = []
    for item, embedding, action, item_root in zip(
        items, embeddings, action_rows, item_roots
    ):
        challenger = base.core.route_with_evaluator(
            target_family=item.family,
            target_embedding=embedding,
            training_families=a_families,
            training_embeddings=a_embeddings,
            training_utility_rows=a_utilities,
            portfolio=tuple(f_result["recipe_portfolio"]),
            spec=spec,
        )
        intent_rows.append(
            {
                "Agent_document_ids": list(action["recipe_document_ids"][P_BASE]),
                "Agent_rows": list(action["recipe_rows"][P_BASE]),
                "E1_counterfactual_document_ids": list(
                    action["recipe_document_ids"][challenger]
                ),
                "E1_counterfactual_recipe": challenger,
                "RAW_document_ids": list(action["raw_document_ids"]),
                "RAW_rows": list(action["raw_rows"]),
                "candidate_rows": list(action["candidate_rows"]),
                "family": item.family,
                "HippoRAG_input_file_sha256": base.file_sha256(
                    item_root / "input.json"
                ),
                "item_commitment_sha256": item.commitment,
                "ordinal": item.ordinal,
            }
        )
    intent = base.self_hashed(
        {
            "active_Agent_recipe": P_BASE,
            "E1_counterfactual_evaluator": spec.evaluator_id,
            "item_count": len(items),
            "items": intent_rows,
            "logical_action_intent_count": len(items) * 3,
            "schema": INTENT_SCHEMA,
        },
        "pack_sha256",
    )
    intent_path = root / "action.intent.json"
    base._write_json(intent_path, intent)
    result = base.self_hashed(
        {
            "claim_boundary": {
                "action_intent_count": len(items) * 3,
                "HippoRAG_execution_count": 0,
                "label_open_count": 0,
                "online_model_or_evaluator_count": 0,
                "score_count": 0,
            },
            "formal_binding": {
                "acquisition_result_sha256": preconditions["acquisition"]["result_sha256"],
                "attempt_marker_file_sha256": base.file_sha256(root / "attempt.marker"),
                "formal_implementation_commit": base._git_head(project_root),
                "implementation_freeze_self_sha256": preconditions["freeze"]["self_sha256"],
            },
            "item_count": len(items),
            "private_bindings": {
                "action_intent_file_sha256": base.file_sha256(intent_path),
                "action_intent_pack_sha256": intent["pack_sha256"],
                "local_action_file_sha256": base.file_sha256(root / "local.action.json"),
                "local_action_pack_sha256": action_pack["pack_sha256"],
                "original_query_embeddings_file_sha256": base.file_sha256(
                    root / "original_query_embeddings.npy"
                ),
                "qwen": qwen_receipt,
            },
            "schema": PREPARE_SCHEMA,
            "status": "reserve_action_intents_prepared_labels_sealed",
            "valid_generation_count": sum(
                row["generation_valid"] for row in action_pack["items"]
            ),
        },
        "result_sha256",
    )
    base._write_json(project_root / PREPARE_RESULT_RELATIVE, result, mode=0o644)
    return result


def _load_prepare(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _read_canonical(
        project_root / PREPARE_RESULT_RELATIVE, "reserve prepare result"
    )
    if (
        result.get("schema") != PREPARE_SCHEMA
        or result.get("status") != "reserve_action_intents_prepared_labels_sealed"
    ):
        raise BrightReserveMeasurementError("reserve prepare did not complete")
    base.verify_self_hash(result, "result_sha256")
    root = project_root / ROOT_RELATIVE
    intent = _read_canonical(root / "action.intent.json", "action intent")
    if (
        intent.get("schema") != INTENT_SCHEMA
        or base.verify_self_hash(intent, "pack_sha256")
        != result.get("private_bindings", {}).get("action_intent_pack_sha256")
        or base.file_sha256(root / "action.intent.json")
        != result.get("private_bindings", {}).get("action_intent_file_sha256")
    ):
        raise BrightReserveMeasurementError("action intent binding drifted")
    return result, intent


def _existing_hipporag(
    item_root: Path, candidate_rows: Sequence[int]
) -> dict[str, Any]:
    output_path = item_root / "output.json"
    stdout_path = item_root / "stdout.log"
    stderr_path = item_root / "stderr.log"
    if not output_path.is_file() or not stdout_path.is_file() or not stderr_path.is_file():
        raise BrightReserveMeasurementError("terminal HippoRAG artifact set is incomplete")
    payload = parse_hipporag_output(output_path.read_bytes())
    if payload["graph_node_count"] <= base.core.POOL_SIZE or payload["graph_edge_count"] <= 0:
        raise BrightReserveMeasurementError("existing HippoRAG graph is invalid")
    top_rows = [candidate_rows[index] for index in payload["top_ordinals"]]
    if len(top_rows) != base.core.TOP_K or len(set(top_rows)) != base.core.TOP_K:
        raise BrightReserveMeasurementError("existing HippoRAG top-k drifted")
    return {
        "graph_edge_count": payload["graph_edge_count"],
        "graph_node_count": payload["graph_node_count"],
        "output_file_sha256": base.file_sha256(output_path),
        "stderr_sha256": base.file_sha256(stderr_path),
        "stdout_sha256": base.file_sha256(stdout_path),
        "top_rows": top_rows,
    }


def execute_actions(project_root: Path, *, resume: bool) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    v3._activate_v3()
    _preconditions = _verify_preconditions(project_root)
    prepare_result, intent = _load_prepare(project_root)
    if (project_root / ACTION_RESULT_RELATIVE).exists():
        raise BrightReserveMeasurementError("reserve action result already exists")
    root = project_root / ROOT_RELATIVE
    if (root / "three_arm.action.json").exists():
        raise BrightReserveMeasurementError("three-arm action seal already exists")
    launches_root = root / "launches"
    launches_root.mkdir(mode=0o700, exist_ok=True)
    launches = sorted(launches_root.glob("launch_*.marker"))
    if len(launches) >= CONTROLLER_LAUNCH_BUDGET:
        raise BrightReserveMeasurementError("controller launch budget is exhausted")
    if (resume and len(launches) != 1) or (not resume and launches):
        raise BrightReserveMeasurementError("controller launch mode drifted")
    items = _load_view(project_root, _preconditions)
    action_pack = _read_canonical(root / "local.action.json", "local action pack")
    action_rows = base._validate_action_pack(action_pack, items)
    intent_rows = intent.get("items")
    if not isinstance(intent_rows, list) or len(intent_rows) != len(items):
        raise BrightReserveMeasurementError("action intent rows drifted")
    existing: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for index, action in enumerate(action_rows):
        item_root = root / "hipporag" / f"item_{index:03d}"
        output_path = item_root / "output.json"
        if output_path.exists():
            existing[index] = _existing_hipporag(item_root, action["candidate_rows"])
        else:
            missing.append(index)
            if not resume and any(
                (item_root / name).exists()
                for name in ("index", "stdout.log", "stderr.log")
            ):
                raise BrightReserveMeasurementError("initial action root is not clean")
            if resume:
                for name in ("index", "stdout.log", "stderr.log"):
                    path = item_root / name
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    elif path.exists() or path.is_symlink():
                        path.unlink()
    if resume and not existing:
        raise BrightReserveMeasurementError("resume has no terminal output to preserve")
    launch = base.self_hashed(
        {
            "action_intent_pack_sha256": intent["pack_sha256"],
            "existing_terminal_ordinals": sorted(existing),
            "launch_index": len(launches) + 1,
            "missing_ordinals_submitted_before_join": missing,
            "schema": f"{VERSION}_controller_launch",
        },
        "launch_sha256",
    )
    base._write_json(
        launches_root / f"launch_{len(launches) + 1:03d}.marker", launch
    )
    semaphore = threading.Semaphore(HIPPORAG_CONCURRENCY)
    counter = base._ConcurrencyCounter()
    futures: dict[Future[Any], int] = {}
    completed_rows = dict(existing)
    with ThreadPoolExecutor(max_workers=len(missing) or 1) as executor:
        for index in missing:
            futures[
                executor.submit(
                    base._run_hipporag_item,
                    project_root=project_root,
                    item_root=root / "hipporag" / f"item_{index:03d}",
                    candidate_rows=action_rows[index]["candidate_rows"],
                    semaphore=semaphore,
                    counter=counter,
                )
            ] = index
        if len(futures) != len(missing):
            raise BrightReserveMeasurementError("HippoRAG submission drifted")
        for future in as_completed(futures):
            completed_rows[futures[future]] = future.result()
    if (
        counter.current != 0
        or counter.peak > HIPPORAG_CONCURRENCY
        or set(completed_rows) != set(range(len(items)))
    ):
        raise BrightReserveMeasurementError("HippoRAG completion set drifted")
    corpus = v3._load_corpus_v3(project_root)
    sealed_rows: list[dict[str, Any]] = []
    for index, (item, raw_intent) in enumerate(zip(items, intent_rows)):
        if (
            not isinstance(raw_intent, Mapping)
            or raw_intent.get("ordinal") != index
            or raw_intent.get("item_commitment_sha256") != item.commitment
        ):
            raise BrightReserveMeasurementError("sealed intent row drifted")
        hippo = dict(completed_rows[index])
        hippo["document_ids"] = [
            corpus[item.family].ids[row] for row in hippo["top_rows"]
        ]
        sealed_rows.append(
            {
                "Agent_document_ids": list(raw_intent["Agent_document_ids"]),
                "E1_counterfactual_document_ids": list(
                    raw_intent["E1_counterfactual_document_ids"]
                ),
                "E1_counterfactual_recipe": raw_intent["E1_counterfactual_recipe"],
                "family": item.family,
                "HippoRAG": hippo,
                "item_commitment_sha256": item.commitment,
                "ordinal": index,
                "RAW_document_ids": list(raw_intent["RAW_document_ids"]),
            }
        )
    action_seal = base.self_hashed(
        {
            "active_Agent_recipe": P_BASE,
            "item_count": len(items),
            "items": sealed_rows,
            "logical_action_intent_count": len(items) * 3,
            "schema": THREE_ARM_SCHEMA,
        },
        "pack_sha256",
    )
    action_path = root / "three_arm.action.json"
    base._write_json(action_path, action_seal)
    result = base.self_hashed(
        {
            "claim_boundary": {
                "all_action_intents_preceded_first_join": True,
                "label_open_count": 0,
                "online_model_or_evaluator_count": 0,
                "score_count": 0,
            },
            "controller": {
                "concurrency_cap": HIPPORAG_CONCURRENCY,
                "current_launch_peak_process_concurrency": counter.peak,
                "launch_count": len(launches) + 1,
                "preserved_terminal_output_count": len(existing),
                "submitted_missing_output_count": len(missing),
            },
            "formal_binding": {
                "action_intent_pack_sha256": intent["pack_sha256"],
                "formal_implementation_commit": base._git_head(project_root),
                "prepare_result_sha256": prepare_result["result_sha256"],
            },
            "HippoRAG": {
                "graph_edge_count_min": min(
                    row["HippoRAG"]["graph_edge_count"] for row in sealed_rows
                ),
                "graph_node_count_min": min(
                    row["HippoRAG"]["graph_node_count"] for row in sealed_rows
                ),
                "terminal_action_count": len(sealed_rows),
            },
            "item_count": len(items),
            "private_bindings": {
                "three_arm_action_file_sha256": base.file_sha256(action_path),
                "three_arm_action_pack_sha256": action_seal["pack_sha256"],
            },
            "schema": ACTION_RESULT_SCHEMA,
            "status": "reserve_three_arm_actions_complete_labels_sealed",
        },
        "result_sha256",
    )
    base._write_json(project_root / ACTION_RESULT_RELATIVE, result, mode=0o644)
    return result


def _paired(left: Sequence[int], right: Sequence[int]) -> dict[str, int]:
    return {
        "gain": sum(a > b for a, b in zip(left, right)),
        "harm": sum(a < b for a, b in zip(left, right)),
        "tie": sum(a == b for a, b in zip(left, right)),
    }


def score(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    v3._activate_v3()
    preconditions = _verify_preconditions(project_root)
    action_result = _read_canonical(
        project_root / ACTION_RESULT_RELATIVE, "reserve action result"
    )
    if (
        action_result.get("schema") != ACTION_RESULT_SCHEMA
        or action_result.get("status")
        != "reserve_three_arm_actions_complete_labels_sealed"
    ):
        raise BrightReserveMeasurementError("reserve actions did not complete")
    base.verify_self_hash(action_result, "result_sha256")
    final_path = project_root / FINAL_RESULT_RELATIVE
    root = project_root / ROOT_RELATIVE
    score_marker = root / "score.marker"
    if final_path.exists() or score_marker.exists():
        raise BrightReserveMeasurementError("reserve scoring is one-shot")
    marker = base.self_hashed(
        {
            "action_result_sha256": action_result["result_sha256"],
            "schema": f"{VERSION}_score_attempt",
        },
        "attempt_sha256",
    )
    base._write_json(score_marker, marker)
    items = _load_view(project_root, preconditions)
    action_path = root / "three_arm.action.json"
    action_pack = _read_canonical(action_path, "three-arm action pack")
    if (
        action_pack.get("schema") != THREE_ARM_SCHEMA
        or base.verify_self_hash(action_pack, "pack_sha256")
        != action_result.get("private_bindings", {}).get(
            "three_arm_action_pack_sha256"
        )
        or base.file_sha256(action_path)
        != action_result.get("private_bindings", {}).get(
            "three_arm_action_file_sha256"
        )
    ):
        raise BrightReserveMeasurementError("three-arm action seal drifted")
    action_rows = action_pack.get("items")
    if not isinstance(action_rows, list) or len(action_rows) != len(items):
        raise BrightReserveMeasurementError("three-arm action rows drifted")
    labels = _load_labels(project_root, preconditions, items)
    arm_scores: dict[str, list[int]] = {
        "Agent": [],
        "E1_counterfactual": [],
        "HippoRAG": [],
        "RAW": [],
    }
    score_rows: list[dict[str, Any]] = []
    for item, action, gold in zip(items, action_rows, labels):
        if (
            not isinstance(action, Mapping)
            or action.get("item_commitment_sha256") != item.commitment
            or action.get("family") != item.family
        ):
            raise BrightReserveMeasurementError("score action identity drifted")
        values = {
            "Agent": base.core.integer_ndcg_at_10(action["Agent_document_ids"], gold),
            "E1_counterfactual": base.core.integer_ndcg_at_10(
                action["E1_counterfactual_document_ids"], gold
            ),
            "HippoRAG": base.core.integer_ndcg_at_10(
                action["HippoRAG"]["document_ids"], gold
            ),
            "RAW": base.core.integer_ndcg_at_10(action["RAW_document_ids"], gold),
        }
        for arm, value in values.items():
            arm_scores[arm].append(value)
        score_rows.append(
            {
                "arm_scores": values,
                "family": item.family,
                "item_commitment_sha256": item.commitment,
                "ordinal": item.ordinal,
            }
        )
    scored = base.self_hashed(
        {"item_count": len(items), "items": score_rows, "schema": SCORED_SCHEMA},
        "pack_sha256",
    )
    scored_path = root / "scored.json"
    base._write_json(scored_path, scored)
    aggregates = base._family_arm_aggregates(items, arm_scores)
    family_agent_minus_hippo = {
        family: aggregates["Agent"]["family_sum_integer_ndcg"][family]
        - aggregates["HippoRAG"]["family_sum_integer_ndcg"][family]
        for family in base.core.FAMILY_ORDER
    }
    result = base.self_hashed(
        {
            "active_Agent_recipe": P_BASE,
            "arm_aggregates": aggregates,
            "claim_boundary": {
                "answer_generation_count": 0,
                "external_network_call_count": 0,
                "label_open_count_after_action_seal": 1,
                "L5_claim": false,
                "online_evaluator_count": 0,
                "replay_or_resample_count": 0,
            },
            "descriptive_evidence": {
                "Agent_minus_HippoRAG_family_sum_integer_ndcg": family_agent_minus_hippo,
                "Agent_minus_HippoRAG_positive_in_all_three_families": all(
                    value > 0 for value in family_agent_minus_hippo.values()
                ),
                "Agent_minus_HippoRAG_sum_integer_ndcg": sum(arm_scores["Agent"])
                - sum(arm_scores["HippoRAG"]),
                "Agent_minus_RAW_sum_integer_ndcg": sum(arm_scores["Agent"])
                - sum(arm_scores["RAW"]),
                "E1_counterfactual_minus_retained_P_sum_integer_ndcg": sum(
                    arm_scores["E1_counterfactual"]
                )
                - sum(arm_scores["Agent"]),
                "E1_was_not_promoted_on_A_hold": True,
                "L5_supported": False,
            },
            "formal_binding": {
                "action_result_sha256": action_result["result_sha256"],
                "formal_implementation_commit": base._git_head(project_root),
                "score_attempt_marker_file_sha256": base.file_sha256(score_marker),
            },
            "item_count": len(items),
            "paired": {
                "Agent_minus_HippoRAG": _paired(
                    arm_scores["Agent"], arm_scores["HippoRAG"]
                ),
                "Agent_minus_RAW": _paired(arm_scores["Agent"], arm_scores["RAW"]),
                "E1_counterfactual_minus_Agent": _paired(
                    arm_scores["E1_counterfactual"], arm_scores["Agent"]
                ),
            },
            "private_bindings": {
                "scored_file_sha256": base.file_sha256(scored_path),
                "scored_pack_sha256": scored["pack_sha256"],
            },
            "schema": FINAL_SCHEMA,
            "status": "fresh_RESERVE_measurement_complete",
        },
        "result_sha256",
    )
    base._write_json(final_path, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "execute-actions", "resume-actions", "score")
    )
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare":
        result = prepare(arguments.project_root)
    elif arguments.command == "execute-actions":
        result = execute_actions(arguments.project_root, resume=False)
    elif arguments.command == "resume-actions":
        result = execute_actions(arguments.project_root, resume=True)
    else:
        result = score(arguments.project_root)
    print(
        json.dumps(
            {"result_sha256": result["result_sha256"], "status": result["status"]},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
