"""Prospective same-source BRIGHT confirmation of the frozen P9 retriever."""

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

from assumption_agent.benchmarks import bright_reasoning_retrieval_acquisition_v1 as source
from assumption_agent.benchmarks import bright_reasoning_retrieval_cross_encoder_formation_v1 as formation
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_measurement_v1 as reserve
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as base
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v3 as v3
from replication_runtime.bright_cross_encoder_v1 import contract as cross_contract
from replication_runtime.bright_cross_encoder_v1 import worker as cross_worker
from replication_runtime.bright_official_hipporag_v1 import contract as hippo_contract


VERSION = "bright_reasoning_retrieval_p9_confirmation_v1"
BLOCK = "C_confirm"
P9 = formation.P9
P6 = "P6_RELATION_MECHANISM_RRF"
COUNT_PER_FAMILY = 11
CONSUMED_PER_FAMILY = 15
ITEM_COUNT = 33
HIPPORAG_CONCURRENCY = 12
EXTERNAL_PROCESS_CONCURRENCY = 13
CONTROLLER_LAUNCH_BUDGET = 2

DESIGN_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_p9_confirmation_design_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_p9_confirmation_implementation_freeze_v1.json"
)
ACQUISITION_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_p9_confirmation_acquisition_result_v1.json"
)
PREPARE_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_p9_confirmation_prepare_result_v1.json"
)
ACTION_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_p9_confirmation_actions_result_v1.json"
)
FINAL_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_p9_confirmation_final_result_v1.json"
)
ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_p9_confirmation_v1")
PRIVATE_RELATIVE = ROOT_RELATIVE / "private"

DESIGN_FILE_SHA256 = (
    "b4758ad04fa5418dde4d91957cd9925a5dd8b3a17cf78653c14aef28463820d0"
)
DESIGN_SELF_SHA256 = (
    "8e2ba07c8617d99a9a4e6f5c9a8c1d3a3edea3f8a64766ee09ecd8de07e85c0c"
)
ORIGINAL_RESULT_FILE_SHA256 = (
    "f637c369015b0e8d991a1d43373360e0292cf8362ba801347006bd297e7a8e1b"
)
ORIGINAL_RESULT_SHA256 = (
    "5736847df8a9a57f674ee02dc1fbc1fdf08120faa358631358d7d80498092ce7"
)
ORIGINAL_RESERVE_VIEW_FILE_SHA256 = (
    "6019129c48302f5f26440ad285efb4c684c1ca938bd75518f4679142fcc435e8"
)
ORIGINAL_RESERVE_VIEW_SHA256 = (
    "9a8ce1db47632c042ce2308c1153477a8ecc504dd774a52ab889bf48a0aabba1"
)
ORIGINAL_SELECTION_SECRET_SHA256 = (
    "00f64a57001fea3d2922db0e807d92920cc880dd1d5cf214d79e364e7ec8d046"
)
CONSUMED_ACQUISITION_RESULT_FILE_SHA256 = (
    "7e25eb23cbe1741d64d7f367d7b1922fbdb1f6bde682e7159b251c7d6f6e151a"
)
CONSUMED_ACQUISITION_RESULT_SHA256 = (
    "5ea204347db1dbf675639952f327ebf421711f9fe0eeae7696286e0fb604eea2"
)
P9_FORMATION_RESULT_FILE_SHA256 = (
    "1f4cfd144930c81aa1c353786f3e66be16a22c98b9bb52752931f2258fbb16e6"
)
P9_FORMATION_RESULT_SHA256 = (
    "79a7ca357d178b05692cfaf0c297af9273e50a6e090624b592c575426ea643b6"
)

VIEW_SCHEMA = f"{VERSION}_block_view"
LABEL_SCHEMA = f"{VERSION}_block_labels"
ACQUISITION_SCHEMA = f"{VERSION}_acquisition_result"
PREPARE_SCHEMA = f"{VERSION}_prepare_result"
INTENT_SCHEMA = f"{VERSION}_action_intent"
ACTION_SCHEMA = f"{VERSION}_action"
ACTION_RESULT_SCHEMA = f"{VERSION}_action_result"
SCORED_SCHEMA = f"{VERSION}_scored"
FINAL_SCHEMA = f"{VERSION}_final_result"


class BrightP9ConfirmationError(RuntimeError):
    """The frozen P9 confirmation contract failed closed."""


def _read_json(path: Path, field: str, *, canonical: bool) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BrightP9ConfirmationError(f"{field} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightP9ConfirmationError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise BrightP9ConfirmationError(f"{field} root drifted")
    if canonical and base.canonical_json_bytes(value) != raw:
        raise BrightP9ConfirmationError(f"{field} is not canonical")
    return value


def _verify_self(value: Mapping[str, Any], field: str, expected: str | None = None) -> str:
    try:
        observed = base.verify_self_hash(value, field)
    except Exception as exc:
        raise BrightP9ConfirmationError(f"{field} drifted") from exc
    if expected is not None and observed != expected:
        raise BrightP9ConfirmationError(f"{field} binding drifted")
    return observed


def _verify_result(
    path: Path,
    *,
    file_sha256: str | None,
    schema: str,
    status: str | None,
    expected_self: str | None = None,
) -> dict[str, Any]:
    if file_sha256 is not None and base.file_sha256(path) != file_sha256:
        raise BrightP9ConfirmationError(f"{path.name} file binding drifted")
    value = _read_json(path, path.name, canonical=True)
    if value.get("schema") != schema or (
        status is not None and value.get("status") != status
    ):
        raise BrightP9ConfirmationError(f"{path.name} identity drifted")
    _verify_self(value, "result_sha256", expected_self)
    return value


def _verify_freeze(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    design_path = project_root / DESIGN_RELATIVE
    if base.file_sha256(design_path) != DESIGN_FILE_SHA256:
        raise BrightP9ConfirmationError("confirmation design file drifted")
    design = _read_json(design_path, "confirmation design", canonical=False)
    _verify_self(design, "self_sha256", DESIGN_SELF_SHA256)
    freeze = _read_json(
        project_root / FREEZE_RELATIVE,
        "confirmation implementation freeze",
        canonical=False,
    )
    if (
        freeze.get("schema")
        != "bright_reasoning_retrieval_p9_confirmation_implementation_freeze_v1"
        or freeze.get("design_self_sha256") != DESIGN_SELF_SHA256
    ):
        raise BrightP9ConfirmationError("confirmation freeze identity drifted")
    _verify_self(freeze, "self_sha256")
    bindings = freeze.get("implementation_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise BrightP9ConfirmationError("confirmation implementation bindings drifted")
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {"relative_path", "sha256"}:
            raise BrightP9ConfirmationError("implementation binding shape drifted")
        if base.file_sha256(project_root / str(binding["relative_path"])) != binding["sha256"]:
            raise BrightP9ConfirmationError("confirmation implementation file drifted")
    return design, freeze


def _verify_prior_bindings(project_root: Path) -> dict[str, Any]:
    original = _verify_result(
        project_root / source.RESULT_RELATIVE,
        file_sha256=ORIGINAL_RESULT_FILE_SHA256,
        schema=source.RESULT_SCHEMA,
        status="acquired_gold_separated_blocks_G_only_authorized",
        expected_self=ORIGINAL_RESULT_SHA256,
    )
    consumed = _verify_result(
        project_root / reserve.ACQUISITION_RESULT_RELATIVE,
        file_sha256=CONSUMED_ACQUISITION_RESULT_FILE_SHA256,
        schema=reserve.acquisition.RESULT_SCHEMA,
        status="fresh_RESERVE_R_search_acquired_labels_sealed",
        expected_self=CONSUMED_ACQUISITION_RESULT_SHA256,
    )
    p9 = _verify_result(
        project_root / formation.RESULT_RELATIVE,
        file_sha256=P9_FORMATION_RESULT_FILE_SHA256,
        schema=f"{formation.VERSION}_result",
        status="consumed_TRAIN45_postterminal_candidate_formation_complete",
        expected_self=P9_FORMATION_RESULT_SHA256,
    )
    if p9.get("candidate") != P9:
        raise BrightP9ConfirmationError("P9 candidate identity drifted")
    formation._verify_runtime(project_root.parent)
    return {"consumed": consumed, "original": original, "p9": p9}


def select_confirmation_rows(
    reserve_rows: Sequence[source.SourceItem],
) -> tuple[source.SourceItem, ...]:
    selected: list[source.SourceItem] = []
    for family in source.FAMILY_ORDER:
        family_rows = [row for row in reserve_rows if row.family == family]
        end = CONSUMED_PER_FAMILY + COUNT_PER_FAMILY
        if len(family_rows) < end:
            raise BrightP9ConfirmationError("confirmation family capacity drifted")
        selected.extend(family_rows[CONSUMED_PER_FAMILY:end])
    if (
        len(selected) != ITEM_COUNT
        or len({row.commitment_sha256 for row in selected}) != ITEM_COUNT
        or Counter(row.family for row in selected)
        != Counter({family: COUNT_PER_FAMILY for family in source.FAMILY_ORDER})
    ):
        raise BrightP9ConfirmationError("confirmation selection drifted")
    return tuple(selected)


def _view(rows: Sequence[source.SourceItem]) -> dict[str, Any]:
    return base.self_hashed(
        {
            "block": BLOCK,
            "excluded_fields": [
                "source_example_id",
                "reasoning",
                "gold_ids_long",
                "gold_ids",
                "gold_answer",
            ],
            "item_count": len(rows),
            "items": [
                {
                    "excluded_ids": list(item.excluded_ids),
                    "family": item.family,
                    "item_commitment_sha256": item.commitment_sha256,
                    "ordinal": ordinal,
                    "query": item.query,
                }
                for ordinal, item in enumerate(rows)
            ],
            "schema": VIEW_SCHEMA,
        },
        "pack_sha256",
    )


def _labels(rows: Sequence[source.SourceItem]) -> dict[str, Any]:
    return base.self_hashed(
        {
            "block": BLOCK,
            "item_count": len(rows),
            "items": [
                {
                    "gold_ids": list(item.gold_ids),
                    "item_commitment_sha256": item.commitment_sha256,
                    "ordinal": ordinal,
                }
                for ordinal, item in enumerate(rows)
            ],
            "schema": LABEL_SCHEMA,
        },
        "pack_sha256",
    )


def acquire(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    design, freeze = _verify_freeze(project_root)
    prior = _verify_prior_bindings(project_root)
    root = project_root / ROOT_RELATIVE
    result_path = project_root / ACQUISITION_RESULT_RELATIVE
    if root.exists() or result_path.exists():
        raise BrightP9ConfirmationError("confirmation acquisition is one-shot")
    root.mkdir(mode=0o700)
    private = root / "private"
    private.mkdir(mode=0o700)
    marker = base.self_hashed(
        {
            "design_self_sha256": design["self_sha256"],
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "schema": f"{VERSION}_acquisition_attempt",
        },
        "attempt_sha256",
    )
    marker_path = root / "acquisition.attempt.marker"
    base._write_json(marker_path, marker)

    secret_path = project_root / reserve.acquisition.ORIGINAL_PRIVATE_RELATIVE / "selection.secret.bin"
    secret = secret_path.read_bytes()
    if hashlib.sha256(secret).hexdigest() != ORIGINAL_SELECTION_SECRET_SHA256:
        raise BrightP9ConfirmationError("original selection secret drifted")
    items_by_family = source._source_items(project_root)
    assignments = source.assign_blocks(items_by_family, secret)
    reserve_rows = assignments["RESERVE"]
    reconstructed = source.block_view("RESERVE", reserve_rows)
    original_view_path = (
        project_root
        / reserve.acquisition.ORIGINAL_PRIVATE_RELATIVE
        / "RESERVE.view.json"
    )
    if (
        source.file_sha256(original_view_path) != ORIGINAL_RESERVE_VIEW_FILE_SHA256
        or reconstructed["pack_sha256"] != ORIGINAL_RESERVE_VIEW_SHA256
        or original_view_path.read_bytes()
        != source.canonical_json_bytes(reconstructed) + b"\n"
    ):
        raise BrightP9ConfirmationError("original RESERVE reconstruction drifted")

    consumed_path = (
        project_root
        / reserve.ACQUISITION_PRIVATE_RELATIVE
        / f"{reserve.acquisition.BLOCK}.view.json"
    )
    consumed_binding = prior["consumed"]["cohort"]
    if base.file_sha256(consumed_path) != consumed_binding["view_pack_file_sha256"]:
        raise BrightP9ConfirmationError("consumed cohort view file drifted")
    consumed = _read_json(consumed_path, "consumed cohort view", canonical=True)
    if _verify_self(consumed, "pack_sha256") != consumed_binding["view_pack_sha256"]:
        raise BrightP9ConfirmationError("consumed cohort view pack drifted")
    consumed_commitments = [row["item_commitment_sha256"] for row in consumed["items"]]
    expected_consumed = [
        row.commitment_sha256
        for family in source.FAMILY_ORDER
        for row in [item for item in reserve_rows if item.family == family][
            :CONSUMED_PER_FAMILY
        ]
    ]
    if consumed_commitments != expected_consumed:
        raise BrightP9ConfirmationError("consumed cohort prefix drifted")

    selected = select_confirmation_rows(reserve_rows)
    if set(consumed_commitments).intersection(
        row.commitment_sha256 for row in selected
    ):
        raise BrightP9ConfirmationError("confirmation overlaps consumed cohort")
    view = _view(selected)
    labels = _labels(selected)
    view_path = private / f"{BLOCK}.view.json"
    labels_path = private / f"{BLOCK}.labels.json"
    base._write_json(view_path, view)
    base._write_json(labels_path, labels)
    result = base.self_hashed(
        {
            "claim_boundary": {
                "document_content_column_read": False,
                "gold_answer_column_read": False,
                "model_retrieval_or_score_count": 0,
                "network_call_count": 0,
                "reasoning_column_read": False,
                "selection_used_content_gold_or_outcomes": False,
            },
            "cohort": {
                "family_counts": dict(
                    sorted(Counter(row.family for row in selected).items())
                ),
                "item_count": len(selected),
                "label_pack_file_sha256": base.file_sha256(labels_path),
                "label_pack_sha256": labels["pack_sha256"],
                "remaining_untouched_count": 4,
                "view_pack_file_sha256": base.file_sha256(view_path),
                "view_pack_sha256": view["pack_sha256"],
            },
            "formal_binding": {
                "attempt_marker_file_sha256": base.file_sha256(marker_path),
                "consumed_acquisition_result_sha256": prior["consumed"][
                    "result_sha256"
                ],
                "design_self_sha256": design["self_sha256"],
                "formal_implementation_commit": base._git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "original_acquisition_result_sha256": prior["original"][
                    "result_sha256"
                ],
                "original_RESERVE_view_pack_sha256": ORIGINAL_RESERVE_VIEW_SHA256,
                "selection_secret_sha256": ORIGINAL_SELECTION_SECRET_SHA256,
            },
            "schema": ACQUISITION_SCHEMA,
            "status": "prospective_C_confirm_acquired_labels_sealed",
        },
        "result_sha256",
    )
    base._write_json(result_path, result, mode=0o644)
    return result


def _load_acquisition(
    project_root: Path,
) -> tuple[dict[str, Any], tuple[base.ViewItem, ...]]:
    result = _verify_result(
        project_root / ACQUISITION_RESULT_RELATIVE,
        file_sha256=None,
        schema=ACQUISITION_SCHEMA,
        status="prospective_C_confirm_acquired_labels_sealed",
    )
    binding = result.get("cohort")
    if not isinstance(binding, Mapping) or binding.get("item_count") != ITEM_COUNT:
        raise BrightP9ConfirmationError("confirmation cohort binding drifted")
    view_path = project_root / PRIVATE_RELATIVE / f"{BLOCK}.view.json"
    if base.file_sha256(view_path) != binding.get("view_pack_file_sha256"):
        raise BrightP9ConfirmationError("confirmation view file drifted")
    view = _read_json(view_path, "confirmation view", canonical=True)
    if (
        view.get("schema") != VIEW_SCHEMA
        or view.get("block") != BLOCK
        or view.get("item_count") != ITEM_COUNT
        or _verify_self(view, "pack_sha256") != binding.get("view_pack_sha256")
    ):
        raise BrightP9ConfirmationError("confirmation view pack drifted")
    rows = view.get("items")
    if not isinstance(rows, list) or len(rows) != ITEM_COUNT:
        raise BrightP9ConfirmationError("confirmation view rows drifted")
    items: list[base.ViewItem] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "excluded_ids",
            "family",
            "item_commitment_sha256",
            "ordinal",
            "query",
        }:
            raise BrightP9ConfirmationError("confirmation view row shape drifted")
        excluded = row.get("excluded_ids")
        if isinstance(excluded, (str, bytes)) or not isinstance(excluded, list):
            raise BrightP9ConfirmationError("confirmation excluded IDs drifted")
        item = base.ViewItem(
            ordinal=position,
            family=str(row.get("family")),
            commitment=str(row.get("item_commitment_sha256")),
            query=base._required_text(row.get("query"), "query"),
            excluded_ids=tuple(
                base._required_text(value, "excluded ID") for value in excluded
            ),
        )
        if row.get("ordinal") != position or item.family not in base.core.FAMILY_ORDER:
            raise BrightP9ConfirmationError("confirmation view identity drifted")
        items.append(item)
    if Counter(item.family for item in items) != Counter(
        {family: COUNT_PER_FAMILY for family in base.core.FAMILY_ORDER}
    ):
        raise BrightP9ConfirmationError("confirmation view balance drifted")
    return result, tuple(items)


def _load_labels(
    project_root: Path,
    acquisition_result: Mapping[str, Any],
    items: Sequence[base.ViewItem],
) -> tuple[tuple[str, ...], ...]:
    binding = acquisition_result["cohort"]
    path = project_root / PRIVATE_RELATIVE / f"{BLOCK}.labels.json"
    if base.file_sha256(path) != binding.get("label_pack_file_sha256"):
        raise BrightP9ConfirmationError("confirmation label file drifted")
    labels = _read_json(path, "confirmation labels", canonical=True)
    if (
        labels.get("schema") != LABEL_SCHEMA
        or labels.get("block") != BLOCK
        or labels.get("item_count") != ITEM_COUNT
        or _verify_self(labels, "pack_sha256") != binding.get("label_pack_sha256")
    ):
        raise BrightP9ConfirmationError("confirmation label pack drifted")
    rows = labels.get("items")
    if not isinstance(rows, list) or len(rows) != len(items):
        raise BrightP9ConfirmationError("confirmation label rows drifted")
    output: list[tuple[str, ...]] = []
    for position, (row, item) in enumerate(zip(rows, items)):
        if not isinstance(row, Mapping) or set(row) != {
            "gold_ids",
            "item_commitment_sha256",
            "ordinal",
        }:
            raise BrightP9ConfirmationError("confirmation label row shape drifted")
        gold = row.get("gold_ids")
        if (
            row.get("ordinal") != position
            or row.get("item_commitment_sha256") != item.commitment
            or isinstance(gold, (str, bytes))
            or not isinstance(gold, list)
        ):
            raise BrightP9ConfirmationError("confirmation label identity drifted")
        values = tuple(base._required_text(value, "gold ID") for value in gold)
        if not values or len(set(values)) != len(values):
            raise BrightP9ConfirmationError("confirmation gold IDs drifted")
        output.append(values)
    return tuple(output)


def _hippo_input(path: Path) -> tuple[str, tuple[hippo_contract.CandidateDocument, ...]]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightP9ConfirmationError("HippoRAG input is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"documents", "query", "schema"}
        or value.get("schema") != hippo_contract.INPUT_SCHEMA
        or hippo_contract.canonical_json_bytes(value) != raw
    ):
        raise BrightP9ConfirmationError("HippoRAG input envelope drifted")
    return hippo_contract.validate_input(value.get("query"), value.get("documents"))


def prepare(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    v3._activate_v3()
    _design, freeze = _verify_freeze(project_root)
    prior = _verify_prior_bindings(project_root)
    acquisition_result, items = _load_acquisition(project_root)
    if any(
        (project_root / path).exists()
        for path in (PREPARE_RESULT_RELATIVE, ACTION_RESULT_RELATIVE, FINAL_RESULT_RELATIVE)
    ):
        raise BrightP9ConfirmationError("confirmation prepare is one-shot")
    root = project_root / ROOT_RELATIVE
    for name in (
        "prepare.attempt.marker",
        "qwen.input.json",
        "qwen.output.json",
        "local.action.json",
        "original_query_embeddings.npy",
        "action.intent.json",
        "cross_encoder.input.json",
        "hipporag",
    ):
        if (root / name).exists():
            raise BrightP9ConfirmationError("confirmation prepare root is not clean")
    marker = base.self_hashed(
        {
            "acquisition_result_sha256": acquisition_result["result_sha256"],
            "candidate": P9,
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "schema": f"{VERSION}_prepare_attempt",
        },
        "attempt_sha256",
    )
    marker_path = root / "prepare.attempt.marker"
    base._write_json(marker_path, marker)

    qwen, qwen_receipt = v3._run_qwen_v3(project_root, root, items)
    corpus = v3._load_corpus_v3(project_root)
    local, _embeddings = base._local_actions(
        project_root=project_root,
        stage_root=root,
        items=items,
        qwen_output=qwen,
        corpus=corpus,
    )
    local_rows = base._validate_action_pack(local, items)
    item_roots = base._prepare_hipporag_inputs(
        project_root=project_root,
        root=root,
        items=items,
        action_pack=local,
        corpus=corpus,
    )
    cross_items: list[dict[str, Any]] = []
    intent_rows: list[dict[str, Any]] = []
    for position, (item, qwen_row, local_row, item_root) in enumerate(
        zip(items, qwen["items"], local_rows, item_roots)
    ):
        if (
            qwen_row.get("ordinal") != position
            or qwen_row.get("generation_valid") is not True
            or len(qwen_row.get("expansions", ())) != 4
        ):
            raise BrightP9ConfirmationError("confirmation Qwen output drifted")
        hippo_path = item_root / "input.json"
        query, documents = _hippo_input(hippo_path)
        if query != item.query:
            raise BrightP9ConfirmationError("confirmation HippoRAG query drifted")
        expansions = qwen_row["expansions"]
        cross_items.append(
            {
                "documents": [
                    {"content": document.content, "ordinal": document.ordinal}
                    for document in documents
                ],
                "mechanism_query": expansions[2],
                "ordinal": position,
                "relation_query": expansions[1],
            }
        )
        intent_rows.append(
            {
                "candidate_document_ids": list(local_row["candidate_document_ids"]),
                "candidate_rows": list(local_row["candidate_rows"]),
                "family": item.family,
                "HippoRAG_input_file_sha256": base.file_sha256(hippo_path),
                "item_commitment_sha256": item.commitment,
                "ordinal": position,
                "P6_document_ids": list(local_row["recipe_document_ids"][P6]),
                "P6_rows": list(local_row["recipe_rows"][P6]),
                "RAW_document_ids": list(local_row["raw_document_ids"]),
                "RAW_rows": list(local_row["raw_rows"]),
            }
        )
    cross_input = cross_contract.input_payload(cross_items)
    cross_input_path = root / "cross_encoder.input.json"
    base._write_json(cross_input_path, cross_input)
    intent = base.self_hashed(
        {
            "candidate": P9,
            "cross_encoder_input_file_sha256": base.file_sha256(cross_input_path),
            "external_model_intent_count": len(items) * 2,
            "item_count": len(items),
            "items": intent_rows,
            "logical_frozen_arm_count": len(items) * 5,
            "schema": INTENT_SCHEMA,
        },
        "pack_sha256",
    )
    intent_path = root / "action.intent.json"
    base._write_json(intent_path, intent)
    result = base.self_hashed(
        {
            "claim_boundary": {
                "action_intents_persisted": len(items) * 2,
                "cross_encoder_execution_count": 0,
                "HippoRAG_execution_count": 0,
                "label_open_count": 0,
                "network_call_count": 0,
                "score_count": 0,
            },
            "formal_binding": {
                "acquisition_result_sha256": acquisition_result["result_sha256"],
                "attempt_marker_file_sha256": base.file_sha256(marker_path),
                "formal_implementation_commit": base._git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "P9_formation_result_sha256": prior["p9"]["result_sha256"],
            },
            "item_count": len(items),
            "private_bindings": {
                "action_intent_file_sha256": base.file_sha256(intent_path),
                "action_intent_pack_sha256": intent["pack_sha256"],
                "cross_encoder_input_file_sha256": base.file_sha256(cross_input_path),
                "local_action_file_sha256": base.file_sha256(root / "local.action.json"),
                "local_action_pack_sha256": local["pack_sha256"],
                "original_query_embeddings_file_sha256": base.file_sha256(
                    root / "original_query_embeddings.npy"
                ),
                "qwen": qwen_receipt,
            },
            "schema": PREPARE_SCHEMA,
            "status": "prospective_C_confirm_actions_prepared_labels_sealed",
            "valid_generation_count": sum(
                row["generation_valid"] for row in qwen["items"]
            ),
        },
        "result_sha256",
    )
    base._write_json(project_root / PREPARE_RESULT_RELATIVE, result, mode=0o644)
    return result


def _load_prepare(
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _verify_result(
        project_root / PREPARE_RESULT_RELATIVE,
        file_sha256=None,
        schema=PREPARE_SCHEMA,
        status="prospective_C_confirm_actions_prepared_labels_sealed",
    )
    root = project_root / ROOT_RELATIVE
    intent_path = root / "action.intent.json"
    if base.file_sha256(intent_path) != result["private_bindings"][
        "action_intent_file_sha256"
    ]:
        raise BrightP9ConfirmationError("confirmation intent file drifted")
    intent = _read_json(intent_path, "confirmation action intent", canonical=True)
    if (
        intent.get("schema") != INTENT_SCHEMA
        or _verify_self(intent, "pack_sha256")
        != result["private_bindings"]["action_intent_pack_sha256"]
    ):
        raise BrightP9ConfirmationError("confirmation intent pack drifted")
    return result, intent


def execute_actions(project_root: Path, *, resume: bool) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    v3._activate_v3()
    _verify_freeze(project_root)
    _verify_prior_bindings(project_root)
    acquisition_result, items = _load_acquisition(project_root)
    prepare_result, intent = _load_prepare(project_root)
    if (project_root / ACTION_RESULT_RELATIVE).exists():
        raise BrightP9ConfirmationError("confirmation action result already exists")
    root = project_root / ROOT_RELATIVE
    action_path = root / "five_arm.action.json"
    if action_path.exists():
        raise BrightP9ConfirmationError("confirmation action seal already exists")
    local_path = root / "local.action.json"
    if base.file_sha256(local_path) != prepare_result["private_bindings"][
        "local_action_file_sha256"
    ]:
        raise BrightP9ConfirmationError("confirmation local action file drifted")
    local = _read_json(local_path, "confirmation local action", canonical=True)
    local_rows = base._validate_action_pack(local, items)
    intent_rows = intent.get("items")
    if not isinstance(intent_rows, list) or len(intent_rows) != len(items):
        raise BrightP9ConfirmationError("confirmation intent rows drifted")
    cross_input_path = root / "cross_encoder.input.json"
    if (
        base.file_sha256(cross_input_path)
        != intent.get("cross_encoder_input_file_sha256")
        or base.file_sha256(cross_input_path)
        != prepare_result["private_bindings"]["cross_encoder_input_file_sha256"]
    ):
        raise BrightP9ConfirmationError("confirmation cross-encoder input drifted")
    cross_contract.parse_input(cross_input_path.read_bytes())

    launches_root = root / "launches"
    launches_root.mkdir(mode=0o700, exist_ok=True)
    launches = sorted(launches_root.glob("launch_*.marker"))
    if len(launches) >= CONTROLLER_LAUNCH_BUDGET:
        raise BrightP9ConfirmationError("confirmation launch budget exhausted")
    if (resume and len(launches) != 1) or (not resume and launches):
        raise BrightP9ConfirmationError("confirmation launch mode drifted")

    existing_hippo: dict[int, dict[str, Any]] = {}
    missing_hippo: list[int] = []
    for index, local_row in enumerate(local_rows):
        item_root = root / "hipporag" / f"item_{index:03d}"
        if (item_root / "output.json").exists():
            existing_hippo[index] = reserve._existing_hipporag(
                item_root, local_row["candidate_rows"]
            )
        else:
            missing_hippo.append(index)
            if not resume and any(
                (item_root / name).exists()
                for name in ("index", "stdout.log", "stderr.log")
            ):
                raise BrightP9ConfirmationError("initial HippoRAG root is not clean")
            if resume:
                for name in ("index", "stdout.log", "stderr.log"):
                    path = item_root / name
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    elif path.exists() or path.is_symlink():
                        path.unlink()
    cross_output_path = root / "cross_encoder.output.json"
    existing_cross = cross_output_path.exists()
    if existing_cross:
        cross_contract.parse_output(cross_output_path.read_bytes())
    launch = base.self_hashed(
        {
            "action_intent_pack_sha256": intent["pack_sha256"],
            "existing_cross_encoder_output": existing_cross,
            "existing_HippoRAG_ordinals": sorted(existing_hippo),
            "launch_index": len(launches) + 1,
            "missing_cross_encoder_submitted": not existing_cross,
            "missing_HippoRAG_ordinals_submitted_before_join": missing_hippo,
            "schema": f"{VERSION}_controller_launch",
        },
        "launch_sha256",
    )
    launch_path = launches_root / f"launch_{len(launches) + 1:03d}.marker"
    base._write_json(launch_path, launch)

    semaphore = threading.Semaphore(HIPPORAG_CONCURRENCY)
    counter = base._ConcurrencyCounter()
    hippo_futures: dict[Future[Any], int] = {}
    cross_future: Future[Any] | None = None
    completed_hippo = dict(existing_hippo)
    with ThreadPoolExecutor(max_workers=EXTERNAL_PROCESS_CONCURRENCY) as executor:
        if not existing_cross:
            cross_future = executor.submit(
                cross_worker.run,
                input_path=cross_input_path,
                output_path=cross_output_path,
                model_root=project_root / formation.MODEL_RELATIVE,
            )
        for index in missing_hippo:
            future = executor.submit(
                base._run_hipporag_item,
                project_root=project_root,
                item_root=root / "hipporag" / f"item_{index:03d}",
                candidate_rows=local_rows[index]["candidate_rows"],
                semaphore=semaphore,
                counter=counter,
            )
            hippo_futures[future] = index
        for future in as_completed([*hippo_futures, *([cross_future] if cross_future else [])]):
            if cross_future is not None and future is cross_future:
                future.result()
            else:
                completed_hippo[hippo_futures[future]] = future.result()
    if (
        counter.current != 0
        or counter.peak > HIPPORAG_CONCURRENCY
        or set(completed_hippo) != set(range(len(items)))
        or not cross_output_path.is_file()
    ):
        raise BrightP9ConfirmationError("confirmation action completion drifted")
    cross_output = cross_contract.parse_output(cross_output_path.read_bytes())

    sealed_rows: list[dict[str, Any]] = []
    for position, (item, local_row, intent_row, cross_row) in enumerate(
        zip(items, local_rows, intent_rows, cross_output["items"])
    ):
        if (
            not isinstance(intent_row, Mapping)
            or intent_row.get("ordinal") != position
            or intent_row.get("item_commitment_sha256") != item.commitment
            or list(intent_row.get("candidate_rows", ()))
            != list(local_row["candidate_rows"])
        ):
            raise BrightP9ConfirmationError("confirmation sealed intent drifted")
        candidate_rows = tuple(local_row["candidate_rows"])
        candidate_ids = tuple(local_row["candidate_document_ids"])
        hippo = dict(completed_hippo[position])
        p9 = formation.p9_rows(
            candidate_rows=candidate_rows,
            cross_encoder_ranked_ordinals=cross_row["ranked_ordinals"],
            raw_rows=tuple(local_row["raw_rows"]),
            hipporag_rows=tuple(hippo["top_rows"]),
        )
        row_to_id = dict(zip(candidate_rows, candidate_ids))
        hippo["document_ids"] = [row_to_id[row] for row in hippo["top_rows"]]
        sealed_rows.append(
            {
                "CrossEncoder_RM_document_ids": [
                    candidate_ids[index]
                    for index in cross_row["ranked_ordinals"][: base.core.TOP_K]
                ],
                "cross_encoder_mean_logit_quantized": list(
                    cross_row["mean_logit_quantized"]
                ),
                "cross_encoder_ranked_ordinals": list(cross_row["ranked_ordinals"]),
                "family": item.family,
                "HippoRAG": hippo,
                "item_commitment_sha256": item.commitment,
                "ordinal": position,
                "P6_document_ids": list(intent_row["P6_document_ids"]),
                "P9_document_ids": [row_to_id[row] for row in p9],
                "P9_rows": list(p9),
                "RAW_document_ids": list(intent_row["RAW_document_ids"]),
            }
        )
    actions = base.self_hashed(
        {
            "active_Agent": P9,
            "item_count": len(items),
            "items": sealed_rows,
            "schema": ACTION_SCHEMA,
        },
        "pack_sha256",
    )
    base._write_json(action_path, actions)
    result = base.self_hashed(
        {
            "claim_boundary": {
                "all_action_intents_preceded_first_join": True,
                "label_open_count": 0,
                "network_call_count": 0,
                "score_count": 0,
            },
            "controller": {
                "cross_encoder_submitted_this_launch": not existing_cross,
                "external_process_concurrency_cap": EXTERNAL_PROCESS_CONCURRENCY,
                "HippoRAG_concurrency_cap": HIPPORAG_CONCURRENCY,
                "HippoRAG_peak_process_concurrency": counter.peak,
                "launch_count": len(launches) + 1,
                "preserved_terminal_HippoRAG_count": len(existing_hippo),
                "submitted_missing_HippoRAG_count": len(missing_hippo),
            },
            "formal_binding": {
                "acquisition_result_sha256": acquisition_result["result_sha256"],
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
                "action_file_sha256": base.file_sha256(action_path),
                "action_pack_sha256": actions["pack_sha256"],
                "cross_encoder_output_file_sha256": base.file_sha256(
                    cross_output_path
                ),
                "launch_marker_file_sha256": base.file_sha256(launch_path),
            },
            "schema": ACTION_RESULT_SCHEMA,
            "status": "prospective_C_confirm_five_arm_actions_complete_labels_sealed",
        },
        "result_sha256",
    )
    base._write_json(project_root / ACTION_RESULT_RELATIVE, result, mode=0o644)
    return result


def confirmation_passed(
    aggregates: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, dict[str, int], dict[str, int]]:
    try:
        p9 = aggregates["P9"]
        raw = aggregates["RAW"]
        hippo = aggregates["HippoRAG"]
        raw_delta = {
            family: int(p9["family_sum_integer_ndcg"][family])
            - int(raw["family_sum_integer_ndcg"][family])
            for family in base.core.FAMILY_ORDER
        }
        hippo_delta = {
            family: int(p9["family_sum_integer_ndcg"][family])
            - int(hippo["family_sum_integer_ndcg"][family])
            for family in base.core.FAMILY_ORDER
        }
        aggregate_positive = (
            int(p9["sum_integer_ndcg"]) > int(raw["sum_integer_ndcg"])
            and int(p9["sum_integer_ndcg"]) > int(hippo["sum_integer_ndcg"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BrightP9ConfirmationError("confirmation aggregate shape drifted") from exc
    passed = aggregate_positive and all(value > 0 for value in raw_delta.values()) and all(
        value > 0 for value in hippo_delta.values()
    )
    return passed, raw_delta, hippo_delta


def score(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    v3._activate_v3()
    _verify_freeze(project_root)
    _verify_prior_bindings(project_root)
    acquisition_result, items = _load_acquisition(project_root)
    action_result = _verify_result(
        project_root / ACTION_RESULT_RELATIVE,
        file_sha256=None,
        schema=ACTION_RESULT_SCHEMA,
        status="prospective_C_confirm_five_arm_actions_complete_labels_sealed",
    )
    root = project_root / ROOT_RELATIVE
    final_path = project_root / FINAL_RESULT_RELATIVE
    marker_path = root / "score.attempt.marker"
    if final_path.exists() or marker_path.exists():
        raise BrightP9ConfirmationError("confirmation scoring is one-shot")
    marker = base.self_hashed(
        {
            "action_result_sha256": action_result["result_sha256"],
            "schema": f"{VERSION}_score_attempt",
        },
        "attempt_sha256",
    )
    base._write_json(marker_path, marker)
    action_path = root / "five_arm.action.json"
    if base.file_sha256(action_path) != action_result["private_bindings"][
        "action_file_sha256"
    ]:
        raise BrightP9ConfirmationError("confirmation action file drifted")
    actions = _read_json(action_path, "confirmation action seal", canonical=True)
    if (
        actions.get("schema") != ACTION_SCHEMA
        or _verify_self(actions, "pack_sha256")
        != action_result["private_bindings"]["action_pack_sha256"]
    ):
        raise BrightP9ConfirmationError("confirmation action seal drifted")
    action_rows = actions.get("items")
    if not isinstance(action_rows, list) or len(action_rows) != len(items):
        raise BrightP9ConfirmationError("confirmation action rows drifted")
    labels = _load_labels(project_root, acquisition_result, items)
    arm_scores: dict[str, list[int]] = {
        "P9": [],
        "CrossEncoder_RM": [],
        "P6": [],
        "HippoRAG": [],
        "RAW": [],
    }
    score_rows: list[dict[str, Any]] = []
    for item, action, gold in zip(items, action_rows, labels):
        if (
            not isinstance(action, Mapping)
            or action.get("ordinal") != item.ordinal
            or action.get("family") != item.family
            or action.get("item_commitment_sha256") != item.commitment
        ):
            raise BrightP9ConfirmationError("confirmation score identity drifted")
        values = {
            "P9": base.core.integer_ndcg_at_10(action["P9_document_ids"], gold),
            "CrossEncoder_RM": base.core.integer_ndcg_at_10(
                action["CrossEncoder_RM_document_ids"], gold
            ),
            "P6": base.core.integer_ndcg_at_10(action["P6_document_ids"], gold),
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
    passed, raw_delta, hippo_delta = confirmation_passed(aggregates)
    result = base.self_hashed(
        {
            "arm_aggregates": aggregates,
            "candidate": P9,
            "claim_boundary": {
                "family_out_or_new_domain_claim": False,
                "L5_claim": False,
                "label_open_count_after_action_seal": 1,
                "network_call_count": 0,
                "population_inference": False,
                "same_source_prospective_confirmation": True,
            },
            "decision": {
                "primary_passed": passed,
                "rule": "aggregate_and_each_family_P9_minus_RAW_and_P9_minus_HippoRAG_strictly_positive",
            },
            "descriptive_evidence": {
                "P9_minus_HippoRAG_family_sum_integer_ndcg": hippo_delta,
                "P9_minus_HippoRAG_sum_integer_ndcg": sum(arm_scores["P9"])
                - sum(arm_scores["HippoRAG"]),
                "P9_minus_RAW_family_sum_integer_ndcg": raw_delta,
                "P9_minus_RAW_sum_integer_ndcg": sum(arm_scores["P9"])
                - sum(arm_scores["RAW"]),
            },
            "formal_binding": {
                "acquisition_result_sha256": acquisition_result["result_sha256"],
                "action_result_sha256": action_result["result_sha256"],
                "formal_implementation_commit": base._git_head(project_root),
                "score_attempt_marker_file_sha256": base.file_sha256(marker_path),
            },
            "item_count": len(items),
            "paired": {
                "P9_minus_HippoRAG": reserve._paired(
                    arm_scores["P9"], arm_scores["HippoRAG"]
                ),
                "P9_minus_RAW": reserve._paired(
                    arm_scores["P9"], arm_scores["RAW"]
                ),
            },
            "private_bindings": {
                "scored_file_sha256": base.file_sha256(scored_path),
                "scored_pack_sha256": scored["pack_sha256"],
            },
            "schema": FINAL_SCHEMA,
            "status": (
                "prospective_C_confirm_complete_primary_passed"
                if passed
                else "prospective_C_confirm_complete_primary_failed"
            ),
        },
        "result_sha256",
    )
    base._write_json(final_path, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("acquire", "prepare", "execute-actions", "resume-actions", "score")
    )
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "acquire":
        result = acquire(arguments.project_root)
    elif arguments.command == "prepare":
        result = prepare(arguments.project_root)
    elif arguments.command == "execute-actions":
        result = execute_actions(arguments.project_root, resume=False)
    elif arguments.command == "resume-actions":
        result = execute_actions(arguments.project_root, resume=True)
    else:
        result = score(arguments.project_root)
    print(
        json.dumps(
            {
                "result_sha256": result["result_sha256"],
                "status": result["status"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

