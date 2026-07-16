"""Create the launchable v2 pre-run freeze for MuSiQue three-arm development.

The planner emits the formal runner's sole worker/item/private-index schema.
The runner private index is copied as opaque bytes and remains unopened until
all eighteen generator terminals join.  A separate controller receipt audits
the exact 6 x 3 pre-launch grid and request-hash transition.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - subprocess tested
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "assumption_agent.benchmarks"

from ..models import stable_hash
from . import musique_three_arm_formal_runner_v1 as formal_runner
from .musique_development_custody_v1 import (
    DEVELOPMENT_ITEM_COUNT,
    GENERATION_DIRECTORY,
    HIPPORAG_OFFICIAL_COMMIT,
    PUBLISHED_ANCHORS,
    RUNNER_PRIVATE_INDEX_NAME,
    MuSiQueDevelopmentCustodyError,
    _absolute_lexical,
    _reject_symlink_components,
    _require_ignored_untracked_if_in_repository,
    _secure_json,
    _secure_read_bytes,
    _sha256_bytes,
    current_development_implementation_binding,
    generation_view_set_sha256,
    load_generation_item,
    load_public_custody_receipt,
    load_public_private_index_binding,
    verify_formal_anchor_bundle,
)
from replication_runtime.musique_official_hipporag_v1.binding import (
    validate_binding_receipt,
)


FREEZE_VERSION = "musique_formal_development_pre_run_freeze_v2"
PUBLIC_FREEZE_SCHEMA = "musique_formal_development_public_pre_run_freeze_v2"
FORMAL_PUBLIC_FREEZE_RELATIVE = (
    "manifests/musique_formal_development_pre_run_freeze_v2.json"
)
CONTROLLER_PLAN_VERSION = "musique_formal_development_controller_plan_v2"
REGISTERED_PROVIDER_ORIGIN = "https://ruoli.dev"
ARM_IDS = formal_runner.ARM_IDS
WORK_UNIT_COUNT = formal_runner.WORK_UNIT_COUNT
TOP_K = formal_runner.TOP_K
MODEL_ID = formal_runner.MODEL_ID
REQUEST_BODY_BYTE_BUDGET = formal_runner.MODEL_REQUEST_BODY_BYTE_BUDGET
MAXIMUM_MODEL_CONCURRENCY = formal_runner.MAXIMUM_MODEL_CONCURRENCY
WORKER_PLAN_NAME = formal_runner.WORKER_PLAN_FILENAME
PRIVATE_INDEX_NAME = formal_runner.PRIVATE_INDEX_FILENAME
CONTROLLER_PLAN_NAME = "controller_plan.private.json"
TYPED_DIRECTORY = "typed"
INPUT_DIRECTORY = "inputs"
EXECUTION_ROOT_RELATIVE_PATH = "formal_execution"
CONSUMPTION_MARKER_RELATIVE_PATH = "execution.authorization.consumed.json"


class MuSiQueDevelopmentFreezeError(RuntimeError):
    """The launchable v2 protocol could not be frozen exactly."""


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MuSiQueDevelopmentFreezeError(f"{field} must be lowercase SHA-256")
    return value


def _write_bytes_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if mode == 0o600 else 0o755)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], *, mode: int) -> None:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _write_bytes_exclusive(path, raw, mode=mode)


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    normalized = dict(body)
    return {**normalized, field: stable_hash(normalized)}


def _provider_identity(
    *, plus_channel_id: str, pro_channel_id: str
) -> dict[str, Any]:
    try:
        binding = formal_runner.provider_identity_binding(
            plus_channel_id=plus_channel_id,
            pro_channel_id=pro_channel_id,
        )
    except Exception as exc:
        raise MuSiQueDevelopmentFreezeError("provider identity is invalid") from exc
    if binding.get("api_origin") != REGISTERED_PROVIDER_ORIGIN or binding.get("model") != MODEL_ID:
        raise MuSiQueDevelopmentFreezeError("provider identity drifted")
    return binding


def _shared_contract() -> dict[str, Any]:
    return {
        "context_serialization_version": formal_runner.CONTEXT_SERIALIZATION_VERSION,
        "generator_prompt_version": formal_runner.PROMPT_VERSION,
        "maximum_model_concurrency": MAXIMUM_MODEL_CONCURRENCY,
        "model": MODEL_ID,
        "model_output_token_budget": formal_runner.MODEL_OUTPUT_TOKEN_BUDGET,
        "model_request_body_byte_budget": REQUEST_BODY_BYTE_BUDGET,
        "overflow_policy": "fail_closed_no_truncation",
        "online_evaluator_calls": 0,
        "replays": 0,
        "resamples": 0,
        "retries": 0,
        "temperature": 0,
        "top_k": TOP_K,
        "work_unit_count": WORK_UNIT_COUNT,
    }


def _published_bindings(
    *,
    preregistration_path: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    qualification_path: str | Path,
    official_adapter_binding_path: str | Path,
    formal: bool,
) -> dict[str, Any]:
    if formal:
        return verify_formal_anchor_bundle(
            preregistration_path=preregistration_path,
            acquisition_receipt_path=acquisition_receipt_path,
            formation_receipt_path=formation_receipt_path,
            frozen_program_path=frozen_program_path,
            qualification_path=qualification_path,
            official_adapter_binding_path=official_adapter_binding_path,
        )
    payloads = {}
    hashes = {}
    for name, path in (
        ("preregistration", preregistration_path),
        ("acquisition", acquisition_receipt_path),
        ("formation", formation_receipt_path),
        ("program", frozen_program_path),
        ("qualification", qualification_path),
        ("official_adapter", official_adapter_binding_path),
    ):
        payload, raw = _secure_json(path, field=f"synthetic {name}")
        payloads[name] = payload
        hashes[name] = _sha256_bytes(raw)
    return {"payloads": payloads, "file_hashes": hashes}


def _official_binding(
    *, payload: Mapping[str, Any], raw: bytes, formal: bool
) -> dict[str, Any]:
    project = Path(__file__).resolve(strict=True).parents[2]
    try:
        validated = validate_binding_receipt(
            payload,
            project_root=project,
            verify_implementation=True,
        )
    except Exception as exc:
        raise MuSiQueDevelopmentFreezeError("official adapter binding is invalid") from exc
    if (
        validated.get("official_source_binding", {}).get("commit")
        != HIPPORAG_OFFICIAL_COMMIT
        or validated.get("decision") != "frozen_official_core_retrieve_only_adapter"
    ):
        raise MuSiQueDevelopmentFreezeError("official adapter is not launchable")
    if formal and _sha256_bytes(raw) != PUBLISHED_ANCHORS["official_adapter"]["file_sha256"]:
        raise MuSiQueDevelopmentFreezeError("official adapter published file drifted")
    return {
        "adapter_id": formal_runner.OFFICIAL_ADAPTER_ID,
        "binding_receipt_relative_path": formal_runner.OFFICIAL_BINDING_RECEIPT_RELATIVE,
        "binding_receipt_file_sha256": _sha256_bytes(raw),
        "binding_receipt_sha256": validated["receipt_sha256"],
        "official_commit": HIPPORAG_OFFICIAL_COMMIT,
        "implementation_set_sha256": validated["implementation_binding"]["set_sha256"],
        "qualification_file_sha256": validated["qualification_binding"]["file_sha256"],
        "qualification_sha256": validated["qualification_binding"]["qualification_sha256"],
    }


def _stage_gold_free_items(
    *, source_root: Path, development_root: Path
) -> list[dict[str, Any]]:
    items = []
    for ordinal in range(DEVELOPMENT_ITEM_COUNT):
        value = load_generation_item(source_root, ordinal)
        source = source_root / GENERATION_DIRECTORY / f"development_item_{ordinal:02d}.json"
        raw = _secure_read_bytes(source, field="gold-free source item")
        relative = f"{INPUT_DIRECTORY}/development_item_{ordinal:02d}.json"
        destination = development_root / relative
        _write_bytes_exclusive(destination, raw, mode=0o600)
        copied = _secure_read_bytes(destination, field="staged gold-free item")
        if copied != raw:
            raise MuSiQueDevelopmentFreezeError("staged item differs from custody source")
        items.append(
            {
                "ordinal": ordinal,
                "anonymous_item_id": value["anonymous_item_id"],
                "input_relative_path": relative,
                "input_sha256": _sha256_bytes(raw),
                "corpus_document_count": len(value["corpus"]),
            }
        )
    return items


def _stage_typed_files(
    *,
    formation_path: Path,
    program_path: Path,
    formation_payload: Mapping[str, Any],
    program_payload: Mapping[str, Any],
    development_root: Path,
) -> dict[str, Any]:
    formation_raw = _secure_read_bytes(formation_path, field="formation receipt")
    program_raw = _secure_read_bytes(program_path, field="frozen program")
    formation_relative = f"{TYPED_DIRECTORY}/formation.receipt.json"
    program_relative = f"{TYPED_DIRECTORY}/frozen_program.json"
    _write_bytes_exclusive(development_root / formation_relative, formation_raw, mode=0o600)
    _write_bytes_exclusive(development_root / program_relative, program_raw, mode=0o600)
    return {
        "formation_receipt_relative_path": formation_relative,
        "formation_receipt_file_sha256": _sha256_bytes(formation_raw),
        "formation_receipt_hash": _require_sha256(
            formation_payload.get("receipt_hash"), "formation receipt"
        ),
        "frozen_program_relative_path": program_relative,
        "frozen_program_file_sha256": _sha256_bytes(program_raw),
        "frozen_program_hash": _require_sha256(program_payload.get("program_hash"), "program"),
    }


def _opaque_stage_private_index(
    *,
    source_root: Path,
    public_binding_path: Path,
    custody_receipt_sha256: str,
    development_root: Path,
) -> dict[str, Any]:
    binding = load_public_private_index_binding(public_binding_path)
    binding_raw = _secure_read_bytes(public_binding_path, field="private-index public binding")
    if binding.get("custody_receipt_sha256") != custody_receipt_sha256:
        raise MuSiQueDevelopmentFreezeError("private-index/custody binding mismatch")
    source = source_root / RUNNER_PRIVATE_INDEX_NAME
    opaque = _secure_read_bytes(source, field="opaque runner private index")
    if _sha256_bytes(opaque) != binding.get("private_index_file_sha256"):
        raise MuSiQueDevelopmentFreezeError("opaque private-index file hash drifted")
    destination = development_root / PRIVATE_INDEX_NAME
    _write_bytes_exclusive(destination, opaque, mode=0o600)
    copied = _secure_read_bytes(destination, field="staged opaque private index")
    if copied != opaque:
        raise MuSiQueDevelopmentFreezeError("opaque private-index copy changed")
    return {
        "private_index_relative_path": PRIVATE_INDEX_NAME,
        "private_index_file_sha256": binding["private_index_file_sha256"],
        "private_index_hash": binding["private_index_hash"],
        "custody_receipt_sha256": custody_receipt_sha256,
        "binding_sidecar_file_sha256": _sha256_bytes(binding_raw),
        "binding_sha256": binding["binding_sha256"],
    }


def _controller_work_units(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for arm in ARM_IDS:
            rows.append(
                {
                    "work_unit_id": f"{item['anonymous_item_id']}:{arm}",
                    "anonymous_item_id": item["anonymous_item_id"],
                    "arm_id": arm,
                    "input_relative_path": item["input_relative_path"],
                    "input_sha256": item["input_sha256"],
                    "top_k": TOP_K,
                    "attempt_budget": 1,
                    "generator_request_sha256": None,
                    "request_hash_state": "runner_launch_precommit_after_all_retrieval_before_any_generator_call",
                }
            )
    return rows


def _worker_expected_keys() -> set[str]:
    return {
        "worker_plan_version", "study_id", "authorization_id",
        "development_root_commitment", "execution_root_relative_path",
        "execution_root_commitment", "consumption_marker_relative_path",
        "item_count", "arm_ids", "items", "shared_contract", "provider_identity",
        "typed_binding", "formal_official_binding", "runner_implementation",
        "custody_binding", "private_index_binding", "worker_plan_hash",
    }


def verify_worker_plan(
    payload: Mapping[str, Any], *, development_root: str | Path
) -> dict[str, Any]:
    worker = dict(payload)
    if set(worker) != _worker_expected_keys():
        raise MuSiQueDevelopmentFreezeError("worker plan exact schema mismatch")
    declared = worker.get("worker_plan_hash")
    body = dict(worker)
    body.pop("worker_plan_hash", None)
    if stable_hash(body) != declared:
        raise MuSiQueDevelopmentFreezeError("worker plan self-hash mismatch")
    if (
        worker.get("worker_plan_version") != formal_runner.WORKER_PLAN_VERSION
        or worker.get("study_id") != "musique_homologous_three_arm_development_v1"
        or worker.get("item_count") != DEVELOPMENT_ITEM_COUNT
        or worker.get("arm_ids") != list(ARM_IDS)
        or worker.get("execution_root_relative_path") != EXECUTION_ROOT_RELATIVE_PATH
        or worker.get("consumption_marker_relative_path") != CONSUMPTION_MARKER_RELATIVE_PATH
        or worker.get("shared_contract") != _shared_contract()
        or worker.get("runner_implementation")
        != formal_runner.current_runner_implementation_binding()
    ):
        raise MuSiQueDevelopmentFreezeError("worker plan identity drifted")
    root = _absolute_lexical(development_root)
    items = worker.get("items")
    if not isinstance(items, list) or len(items) != 6:
        raise MuSiQueDevelopmentFreezeError("worker item count mismatch")
    item_fields = {
        "ordinal", "anonymous_item_id", "input_relative_path", "input_sha256",
        "corpus_document_count"
    }
    for ordinal, item in enumerate(items):
        expected_id = f"development_item_{ordinal:02d}"
        expected_relative = f"{INPUT_DIRECTORY}/{expected_id}.json"
        if (
            not isinstance(item, dict)
            or set(item) != item_fields
            or item.get("ordinal") != ordinal
            or item.get("anonymous_item_id") != expected_id
            or item.get("input_relative_path") != expected_relative
            or type(item.get("corpus_document_count")) is not int
            or item["corpus_document_count"] < TOP_K
        ):
            raise MuSiQueDevelopmentFreezeError("worker item exact binding mismatch")
        raw = _secure_read_bytes(root / expected_relative, field="worker gold-free item")
        if _sha256_bytes(raw) != item.get("input_sha256"):
            raise MuSiQueDevelopmentFreezeError("worker item file hash mismatch")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise MuSiQueDevelopmentFreezeError("worker item JSON is invalid") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "anonymous_item_id", "question", "corpus"}
            or value.get("schema") != formal_runner.ITEM_INPUT_VERSION
            or value.get("anonymous_item_id") != expected_id
            or not isinstance(value.get("question"), str)
            or not value["question"].strip()
            or not isinstance(value.get("corpus"), list)
            or len(value["corpus"]) != item["corpus_document_count"]
        ):
            raise MuSiQueDevelopmentFreezeError("worker item content schema drifted")
        for index, paragraph in enumerate(value["corpus"]):
            if (
                not isinstance(paragraph, dict)
                or set(paragraph) != {"idx", "title", "paragraph_text"}
                or paragraph.get("idx") != index
                or not isinstance(paragraph.get("title"), str)
                or not paragraph["title"].strip()
                or not isinstance(paragraph.get("paragraph_text"), str)
                or not paragraph["paragraph_text"].strip()
            ):
                raise MuSiQueDevelopmentFreezeError("worker paragraph schema drifted")
    private = worker.get("private_index_binding")
    if not isinstance(private, dict) or set(private) != {
        "private_index_relative_path", "private_index_file_sha256", "private_index_hash",
        "custody_receipt_sha256", "binding_sidecar_file_sha256", "binding_sha256"
    } or private.get("private_index_relative_path") != PRIVATE_INDEX_NAME:
        raise MuSiQueDevelopmentFreezeError("worker private-index binding mismatch")
    private_raw = _secure_read_bytes(root / PRIVATE_INDEX_NAME, field="opaque worker private index")
    if _sha256_bytes(private_raw) != private.get("private_index_file_sha256"):
        raise MuSiQueDevelopmentFreezeError("worker private-index file drifted")
    typed = worker.get("typed_binding")
    if not isinstance(typed, dict) or set(typed) != {
        "formation_receipt_relative_path", "formation_receipt_file_sha256",
        "formation_receipt_hash", "frozen_program_relative_path",
        "frozen_program_file_sha256", "frozen_program_hash"
    }:
        raise MuSiQueDevelopmentFreezeError("worker typed binding schema mismatch")
    for prefix in ("formation_receipt", "frozen_program"):
        relative = typed[f"{prefix}_relative_path"]
        raw = _secure_read_bytes(root / relative, field=f"worker {prefix}")
        if _sha256_bytes(raw) != typed[f"{prefix}_file_sha256"]:
            raise MuSiQueDevelopmentFreezeError("worker typed file drifted")
    official = worker.get("formal_official_binding")
    if not isinstance(official, dict) or set(official) != {
        "adapter_id", "binding_receipt_relative_path", "binding_receipt_file_sha256",
        "binding_receipt_sha256", "official_commit", "implementation_set_sha256",
        "qualification_file_sha256", "qualification_sha256"
    }:
        raise MuSiQueDevelopmentFreezeError("worker official binding schema mismatch")
    custody = worker.get("custody_binding")
    if not isinstance(custody, dict) or set(custody) != {
        "custody_receipt_file_sha256", "custody_receipt_sha256",
        "acquisition_receipt_file_sha256", "acquisition_sha256",
        "generation_view_set_sha256"
    }:
        raise MuSiQueDevelopmentFreezeError("worker custody binding schema mismatch")
    provider = worker.get("provider_identity")
    if not isinstance(provider, dict) or provider.get("api_origin") != REGISTERED_PROVIDER_ORIGIN or provider.get("model") != MODEL_ID:
        raise MuSiQueDevelopmentFreezeError("worker provider identity mismatch")
    expected_commitment = stable_hash(
        {
            "authorization_id": worker["authorization_id"],
            "development_root_commitment": worker["development_root_commitment"],
            "execution_root_relative_path": EXECUTION_ROOT_RELATIVE_PATH,
        }
    )
    if worker.get("execution_root_commitment") != expected_commitment:
        raise MuSiQueDevelopmentFreezeError("execution root commitment mismatch")
    return worker


def verify_controller_plan(
    payload: Mapping[str, Any], *, worker_plan: Mapping[str, Any]
) -> dict[str, Any]:
    controller = dict(payload)
    expected = {
        "schema", "status", "authorization_id", "worker_plan_hash",
        "work_units", "terminal_contract", "offline_evaluation_contract",
        "controller_plan_hash"
    }
    if set(controller) != expected or controller.get("schema") != CONTROLLER_PLAN_VERSION:
        raise MuSiQueDevelopmentFreezeError("controller schema mismatch")
    declared = controller.get("controller_plan_hash")
    body = dict(controller)
    body.pop("controller_plan_hash", None)
    if stable_hash(body) != declared:
        raise MuSiQueDevelopmentFreezeError("controller self-hash mismatch")
    if (
        controller.get("status") != "launchable_pre_run_not_consumed"
        or controller.get("authorization_id") != worker_plan.get("authorization_id")
        or controller.get("worker_plan_hash") != worker_plan.get("worker_plan_hash")
    ):
        raise MuSiQueDevelopmentFreezeError("controller identity mismatch")
    units = controller.get("work_units")
    if not isinstance(units, list) or len(units) != WORK_UNIT_COUNT:
        raise MuSiQueDevelopmentFreezeError("controller work grid count mismatch")
    unit_fields = {
        "work_unit_id", "anonymous_item_id", "arm_id", "input_relative_path",
        "input_sha256", "top_k", "attempt_budget", "generator_request_sha256",
        "request_hash_state"
    }
    item_by_id = {item["anonymous_item_id"]: item for item in worker_plan["items"]}
    expected_pairs = {
        (item_id, arm) for item_id in item_by_id for arm in ARM_IDS
    }
    observed_pairs = set()
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != unit_fields:
            raise MuSiQueDevelopmentFreezeError("controller work-unit keys drifted")
        item = item_by_id.get(unit.get("anonymous_item_id"))
        arm = unit.get("arm_id")
        expected_id = f"{unit.get('anonymous_item_id')}:{arm}"
        if (
            item is None or arm not in ARM_IDS
            or unit.get("work_unit_id") != expected_id
            or unit.get("input_relative_path") != item["input_relative_path"]
            or unit.get("input_sha256") != item["input_sha256"]
            or unit.get("top_k") != TOP_K
            or unit.get("attempt_budget") != 1
            or unit.get("generator_request_sha256") is not None
            or unit.get("request_hash_state")
            != "runner_launch_precommit_after_all_retrieval_before_any_generator_call"
        ):
            raise MuSiQueDevelopmentFreezeError("controller work-unit binding drifted")
        observed_pairs.add((unit["anonymous_item_id"], arm))
    if observed_pairs != expected_pairs or len({unit["work_unit_id"] for unit in units}) != WORK_UNIT_COUNT:
        raise MuSiQueDevelopmentFreezeError("controller grid is not exact 6 x 3")
    if controller.get("terminal_contract") != {
        "expected_terminal_count": 18,
        "all_claims_durable_before_work_start": True,
        "all_terminals_join_before_gold_release": True,
        "invalid_or_transport_failure_is_intention_to_treat_incorrect": True,
        "retry_replay_resample": 0,
    }:
        raise MuSiQueDevelopmentFreezeError("terminal contract mismatch")
    if controller.get("offline_evaluation_contract") != {
        "answer_oracles": ["primary", "secondary"],
        "support_oracles": ["primary", "secondary"],
        "dual_oracle_consensus_required": True,
        "online_evaluator_calls": 0,
        "paired_item_denominator_per_contrast": 6,
    }:
        raise MuSiQueDevelopmentFreezeError("offline evaluation contract mismatch")
    return controller


def verify_public_pre_run_freeze(payload: Mapping[str, Any]) -> dict[str, Any]:
    freeze = dict(payload)
    expected = {
        "schema", "status", "authorization", "binding_hashes", "counts",
        "protocol_amendment", "execution_contract", "gold_release_contract",
        "call_counts", "freeze_sha256"
    }
    if set(freeze) != expected or freeze.get("schema") != PUBLIC_FREEZE_SCHEMA:
        raise MuSiQueDevelopmentFreezeError("public freeze exact schema mismatch")
    declared = freeze.get("freeze_sha256")
    body = dict(freeze)
    body.pop("freeze_sha256", None)
    if stable_hash(body) != declared:
        raise MuSiQueDevelopmentFreezeError("public freeze self-hash mismatch")
    if freeze.get("status") != "launchable_v2_pre_run_not_consumed":
        raise MuSiQueDevelopmentFreezeError("public freeze status mismatch")
    authorization = freeze.get("authorization")
    if not isinstance(authorization, dict) or set(authorization) != {
        "authorization_id", "development_root_commitment", "execution_root_relative_path",
        "execution_root_commitment", "consumption_marker_relative_path", "launch_authorized"
    } or authorization.get("launch_authorized") is not True:
        raise MuSiQueDevelopmentFreezeError("public authorization mismatch")
    hashes = freeze.get("binding_hashes")
    expected_hashes = {
        "preregistration_file_sha256", "preregistration_sha256",
        "acquisition_receipt_file_sha256", "acquisition_sha256",
        "custody_receipt_file_sha256", "custody_receipt_sha256",
        "formation_receipt_file_sha256", "formation_receipt_hash",
        "frozen_program_file_sha256", "frozen_program_hash",
        "qualification_file_sha256", "qualification_sha256",
        "official_adapter_binding_file_sha256", "official_adapter_binding_receipt_sha256",
        "official_adapter_implementation_set_sha256",
        "private_index_binding_sidecar_file_sha256", "private_index_binding_sha256",
        "private_index_file_sha256", "private_index_hash",
        "worker_plan_file_sha256", "worker_plan_hash",
        "controller_plan_file_sha256", "controller_plan_hash",
        "development_implementation_set_sha256", "runner_implementation_set_sha256",
        "provider_identity_sha256", "shared_contract_sha256",
    }
    if not isinstance(hashes, dict) or set(hashes) != expected_hashes:
        raise MuSiQueDevelopmentFreezeError("public binding hash set mismatch")
    for field, value in hashes.items():
        _require_sha256(value, field)
    if freeze.get("counts") != {
        "arms": 3, "development_items": 6, "generator_work_units": 18,
        "private_index_items": 6, "total_work_units": 18
    }:
        raise MuSiQueDevelopmentFreezeError("public freeze counts mismatch")
    if freeze.get("protocol_amendment") != {
        "registered_before_development_access_or_outcome": True,
        "source_preregistration_sha256": PUBLISHED_ANCHORS["preregistration"]["self_sha256"],
        "superseded_draft_field": "maximum_context_tokens=8192",
        "supersession_reason": "draft_had_no_executable_tokenizer_or_counting_semantics",
        "formal_budget_scope": "complete_canonical_model_request_body_utf8_bytes",
        "formal_budget_bytes": REQUEST_BODY_BYTE_BUDGET,
        "overflow_policy": "fail_closed_no_truncation",
        "token_budget_claimed_by_formal_protocol": False,
        "performance_outcome_used": False,
    }:
        raise MuSiQueDevelopmentFreezeError("protocol amendment mismatch")
    if freeze.get("execution_contract") != {
        "attempts_per_work_unit": 1,
        "claims_durable_before_work_start": True,
        "generator_request_hashes_precommitted_before_any_generator_call": True,
        "invalid_or_transport_failure_intention_to_treat_incorrect": True,
        "maximum_model_concurrency": 18,
        "model": MODEL_ID,
        "request_hashes_precommitted_at_freeze": False,
        "request_hash_precommit_phase": "runner_after_all_retrieval_before_any_generator_call",
        "retrieval_top_k": 5,
        "replays": 0, "resamples": 0, "retries": 0,
        "terminal_grid_required": 18,
    }:
        raise MuSiQueDevelopmentFreezeError("public execution contract mismatch")
    if freeze.get("gold_release_contract") != {
        "private_index_copied_as_opaque_bytes": True,
        "private_index_parsed_by_freeze": False,
        "private_index_release_after_terminal_count": 18,
        "dual_answer_oracles": True,
        "dual_support_oracles": True,
        "online_evaluator_calls": 0,
        "paired_item_denominator_per_contrast": 6,
        "sealed_runtime_accessed": False,
    }:
        raise MuSiQueDevelopmentFreezeError("gold release contract mismatch")
    if freeze.get("call_counts") != {
        "generator": 0, "network": 0, "online_evaluator": 0,
        "retriever": 0, "scoring": 0
    }:
        raise MuSiQueDevelopmentFreezeError("public freeze call ledger mismatch")
    serialized = json.dumps(freeze, sort_keys=True)
    if any(token in serialized for token in ('"question"', '"corpus"', '"answers"', '"support_indices"', "development_item_", "/artifacts/", "sk-")):
        raise MuSiQueDevelopmentFreezeError("public freeze leaks private content")
    return freeze


def _prepare(
    *,
    source_view_root: str | Path,
    custody_receipt_path: str | Path,
    private_index_binding_path: str | Path,
    preregistration_path: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    qualification_path: str | Path,
    official_adapter_binding_path: str | Path,
    development_root: str | Path,
    public_freeze_path: str | Path,
    plus_channel_id: str,
    pro_channel_id: str,
    formal: bool,
) -> dict[str, Any]:
    source_root = _reject_symlink_components(source_view_root, "source-view root")
    output_root = _reject_symlink_components(development_root, "development root")
    public_path = _reject_symlink_components(public_freeze_path, "public freeze")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    if public_path.exists() or public_path.is_symlink():
        raise FileExistsError(public_path)
    _require_ignored_untracked_if_in_repository(output_root, "development root", directory=True)
    if formal:
        registered_public = (
            Path(__file__).resolve(strict=True).parents[2]
            / FORMAL_PUBLIC_FREEZE_RELATIVE
        )
        if public_path != registered_public:
            raise MuSiQueDevelopmentFreezeError(
                "formal public freeze must use its registered manifest path"
            )

    # Anchor every public root before reading any source-view development item.
    anchors = _published_bindings(
        preregistration_path=preregistration_path,
        acquisition_receipt_path=acquisition_receipt_path,
        formation_receipt_path=formation_receipt_path,
        frozen_program_path=frozen_program_path,
        qualification_path=qualification_path,
        official_adapter_binding_path=official_adapter_binding_path,
        formal=formal,
    )
    custody = load_public_custody_receipt(custody_receipt_path)
    custody_raw = _secure_read_bytes(custody_receipt_path, field="custody receipt")
    development_implementation = current_development_implementation_binding()
    if (
        custody["hashes"]["development_implementation_set_sha256"]
        != development_implementation["set_sha256"]
    ):
        raise MuSiQueDevelopmentFreezeError(
            "custody development implementation binding drifted"
        )
    source_set = generation_view_set_sha256(source_root)
    if custody["hashes"]["generation_view_set_sha256"] != source_set:
        raise MuSiQueDevelopmentFreezeError("source view drifted from custody")
    acquisition = anchors["payloads"]["acquisition"]
    formation = anchors["payloads"]["formation"]
    program = anchors["payloads"]["program"]
    if (
        custody["hashes"]["acquisition_receipt_file_sha256"] != anchors["file_hashes"]["acquisition"]
        or custody["hashes"]["acquisition_sha256"] != acquisition.get("acquisition_sha256")
        or custody["hashes"]["formation_receipt_file_sha256"] != anchors["file_hashes"]["formation"]
        or custody["hashes"]["formation_receipt_hash"] != formation.get("receipt_hash")
        or custody["hashes"]["frozen_program_file_sha256"] != anchors["file_hashes"]["program"]
        or custody["hashes"]["frozen_program_hash"] != program.get("program_hash")
    ):
        raise MuSiQueDevelopmentFreezeError("custody/public anchor chain drifted")
    official_raw = _secure_read_bytes(official_adapter_binding_path, field="official adapter binding")
    official = _official_binding(
        payload=anchors["payloads"]["official_adapter"],
        raw=official_raw,
        formal=formal,
    )
    provider = _provider_identity(
        plus_channel_id=plus_channel_id, pro_channel_id=pro_channel_id
    )
    runner_implementation = formal_runner.current_runner_implementation_binding()
    shared = _shared_contract()

    try:
        output_root.mkdir(parents=True, mode=0o700)
        os.chmod(output_root, 0o700)
        items = _stage_gold_free_items(source_root=source_root, development_root=output_root)
        typed = _stage_typed_files(
            formation_path=_absolute_lexical(formation_receipt_path),
            program_path=_absolute_lexical(frozen_program_path),
            formation_payload=formation,
            program_payload=program,
            development_root=output_root,
        )
        private = _opaque_stage_private_index(
            source_root=source_root,
            public_binding_path=_absolute_lexical(private_index_binding_path),
            custody_receipt_sha256=custody["receipt_sha256"],
            development_root=output_root,
        )
        staged_set = stable_hash(
            [{"anonymous_item_id": item["anonymous_item_id"], "input_sha256": item["input_sha256"]} for item in items]
        )
        authorization_id = stable_hash(
            {
                "authorization_version": "musique_formal_development_authorization_v2",
                "custody_receipt_sha256": custody["receipt_sha256"],
                "private_index_file_sha256": private["private_index_file_sha256"],
                "formation_receipt_hash": typed["formation_receipt_hash"],
                "frozen_program_hash": typed["frozen_program_hash"],
                "official_adapter_binding_receipt_sha256": official["binding_receipt_sha256"],
                "provider_identity_sha256": provider["identity_sha256"],
                "runner_implementation_set_sha256": runner_implementation["set_sha256"],
                "staged_generation_set_sha256": staged_set,
            }
        )
        development_root_commitment = stable_hash(
            {
                "authorization_id": authorization_id,
                "private_index_file_sha256": private["private_index_file_sha256"],
                "staged_generation_set_sha256": staged_set,
                "typed_binding_sha256": stable_hash(typed),
            }
        )
        execution_root_commitment = stable_hash(
            {
                "authorization_id": authorization_id,
                "development_root_commitment": development_root_commitment,
                "execution_root_relative_path": EXECUTION_ROOT_RELATIVE_PATH,
            }
        )
        custody_binding = {
            "custody_receipt_file_sha256": _sha256_bytes(custody_raw),
            "custody_receipt_sha256": custody["receipt_sha256"],
            "acquisition_receipt_file_sha256": anchors["file_hashes"]["acquisition"],
            "acquisition_sha256": acquisition["acquisition_sha256"],
            "generation_view_set_sha256": source_set,
        }
        worker_body = {
            "worker_plan_version": formal_runner.WORKER_PLAN_VERSION,
            "study_id": "musique_homologous_three_arm_development_v1",
            "authorization_id": authorization_id,
            "development_root_commitment": development_root_commitment,
            "execution_root_relative_path": EXECUTION_ROOT_RELATIVE_PATH,
            "execution_root_commitment": execution_root_commitment,
            "consumption_marker_relative_path": CONSUMPTION_MARKER_RELATIVE_PATH,
            "item_count": 6,
            "arm_ids": list(ARM_IDS),
            "items": items,
            "shared_contract": shared,
            "provider_identity": provider,
            "typed_binding": typed,
            "formal_official_binding": official,
            "runner_implementation": runner_implementation,
            "custody_binding": custody_binding,
            "private_index_binding": private,
        }
        worker = _self_hashed(worker_body, "worker_plan_hash")
        verify_worker_plan(worker, development_root=output_root)
        worker_path = output_root / WORKER_PLAN_NAME
        _write_json_exclusive(worker_path, worker, mode=0o600)
        controller_body = {
            "schema": CONTROLLER_PLAN_VERSION,
            "status": "launchable_pre_run_not_consumed",
            "authorization_id": authorization_id,
            "worker_plan_hash": worker["worker_plan_hash"],
            "work_units": _controller_work_units(items),
            "terminal_contract": {
                "expected_terminal_count": 18,
                "all_claims_durable_before_work_start": True,
                "all_terminals_join_before_gold_release": True,
                "invalid_or_transport_failure_is_intention_to_treat_incorrect": True,
                "retry_replay_resample": 0,
            },
            "offline_evaluation_contract": {
                "answer_oracles": ["primary", "secondary"],
                "support_oracles": ["primary", "secondary"],
                "dual_oracle_consensus_required": True,
                "online_evaluator_calls": 0,
                "paired_item_denominator_per_contrast": 6,
            },
        }
        controller = _self_hashed(controller_body, "controller_plan_hash")
        verify_controller_plan(controller, worker_plan=worker)
        controller_path = output_root / CONTROLLER_PLAN_NAME
        _write_json_exclusive(controller_path, controller, mode=0o600)
        private_sidecar_raw = _secure_read_bytes(
            private_index_binding_path, field="private-index public binding"
        )
        public_body = {
            "schema": PUBLIC_FREEZE_SCHEMA,
            "status": "launchable_v2_pre_run_not_consumed",
            "authorization": {
                "authorization_id": authorization_id,
                "development_root_commitment": development_root_commitment,
                "execution_root_relative_path": EXECUTION_ROOT_RELATIVE_PATH,
                "execution_root_commitment": execution_root_commitment,
                "consumption_marker_relative_path": CONSUMPTION_MARKER_RELATIVE_PATH,
                "launch_authorized": True,
            },
            "binding_hashes": {
                "preregistration_file_sha256": anchors["file_hashes"]["preregistration"],
                "preregistration_sha256": anchors["payloads"]["preregistration"].get("preregistration_sha256", "0" * 64),
                "acquisition_receipt_file_sha256": anchors["file_hashes"]["acquisition"],
                "acquisition_sha256": acquisition["acquisition_sha256"],
                "custody_receipt_file_sha256": _sha256_bytes(custody_raw),
                "custody_receipt_sha256": custody["receipt_sha256"],
                "formation_receipt_file_sha256": typed["formation_receipt_file_sha256"],
                "formation_receipt_hash": typed["formation_receipt_hash"],
                "frozen_program_file_sha256": typed["frozen_program_file_sha256"],
                "frozen_program_hash": typed["frozen_program_hash"],
                "qualification_file_sha256": official["qualification_file_sha256"],
                "qualification_sha256": official["qualification_sha256"],
                "official_adapter_binding_file_sha256": official["binding_receipt_file_sha256"],
                "official_adapter_binding_receipt_sha256": official["binding_receipt_sha256"],
                "official_adapter_implementation_set_sha256": official["implementation_set_sha256"],
                "private_index_binding_sidecar_file_sha256": _sha256_bytes(private_sidecar_raw),
                "private_index_binding_sha256": private["binding_sha256"],
                "private_index_file_sha256": private["private_index_file_sha256"],
                "private_index_hash": private["private_index_hash"],
                "worker_plan_file_sha256": _sha256_bytes(
                    _secure_read_bytes(worker_path, field="worker plan")
                ),
                "worker_plan_hash": worker["worker_plan_hash"],
                "controller_plan_file_sha256": _sha256_bytes(
                    _secure_read_bytes(controller_path, field="controller plan")
                ),
                "controller_plan_hash": controller["controller_plan_hash"],
                "development_implementation_set_sha256": development_implementation["set_sha256"],
                "runner_implementation_set_sha256": runner_implementation["set_sha256"],
                "provider_identity_sha256": provider["identity_sha256"],
                "shared_contract_sha256": stable_hash(shared),
            },
            "counts": {
                "arms": 3,
                "development_items": 6,
                "generator_work_units": 18,
                "private_index_items": 6,
                "total_work_units": 18,
            },
            "protocol_amendment": {
                "registered_before_development_access_or_outcome": True,
                "source_preregistration_sha256": PUBLISHED_ANCHORS["preregistration"]["self_sha256"],
                "superseded_draft_field": "maximum_context_tokens=8192",
                "supersession_reason": "draft_had_no_executable_tokenizer_or_counting_semantics",
                "formal_budget_scope": "complete_canonical_model_request_body_utf8_bytes",
                "formal_budget_bytes": REQUEST_BODY_BYTE_BUDGET,
                "overflow_policy": "fail_closed_no_truncation",
                "token_budget_claimed_by_formal_protocol": False,
                "performance_outcome_used": False,
            },
            "execution_contract": {
                "attempts_per_work_unit": 1,
                "claims_durable_before_work_start": True,
                "generator_request_hashes_precommitted_before_any_generator_call": True,
                "invalid_or_transport_failure_intention_to_treat_incorrect": True,
                "maximum_model_concurrency": 18,
                "model": MODEL_ID,
                "request_hashes_precommitted_at_freeze": False,
                "request_hash_precommit_phase": "runner_after_all_retrieval_before_any_generator_call",
                "retrieval_top_k": 5,
                "replays": 0,
                "resamples": 0,
                "retries": 0,
                "terminal_grid_required": 18,
            },
            "gold_release_contract": {
                "private_index_copied_as_opaque_bytes": True,
                "private_index_parsed_by_freeze": False,
                "private_index_release_after_terminal_count": 18,
                "dual_answer_oracles": True,
                "dual_support_oracles": True,
                "online_evaluator_calls": 0,
                "paired_item_denominator_per_contrast": 6,
                "sealed_runtime_accessed": False,
            },
            "call_counts": {
                "generator": 0,
                "network": 0,
                "online_evaluator": 0,
                "retriever": 0,
                "scoring": 0,
            },
        }
        public = _self_hashed(public_body, "freeze_sha256")
        verify_public_pre_run_freeze(public)
        _write_json_exclusive(public_path, public, mode=0o644)
    except BaseException:
        if output_root.exists() and output_root.is_dir() and not output_root.is_symlink():
            shutil.rmtree(output_root)
        if public_path.exists() and public_path.is_file() and not public_path.is_symlink():
            public_path.unlink()
        raise
    return public


def prepare_development_pre_run_freeze(
    *,
    source_view_root: str | Path,
    custody_receipt_path: str | Path,
    private_index_binding_path: str | Path,
    preregistration_path: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    qualification_path: str | Path,
    official_adapter_binding_path: str | Path,
    development_root: str | Path,
    public_freeze_path: str | Path,
    plus_channel_id: str,
    pro_channel_id: str,
) -> dict[str, Any]:
    return _prepare(
        source_view_root=source_view_root,
        custody_receipt_path=custody_receipt_path,
        private_index_binding_path=private_index_binding_path,
        preregistration_path=preregistration_path,
        acquisition_receipt_path=acquisition_receipt_path,
        formation_receipt_path=formation_receipt_path,
        frozen_program_path=frozen_program_path,
        qualification_path=qualification_path,
        official_adapter_binding_path=official_adapter_binding_path,
        development_root=development_root,
        public_freeze_path=public_freeze_path,
        plus_channel_id=plus_channel_id,
        pro_channel_id=pro_channel_id,
        formal=True,
    )


def prepare_synthetic_development_pre_run_freeze_for_tests(**kwargs: Any) -> dict[str, Any]:
    """Explicit non-formal planner entry for generated fixture tests."""

    return _prepare(**kwargs, formal=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view-root", type=Path, required=True)
    parser.add_argument("--custody-receipt", type=Path, required=True)
    parser.add_argument("--private-index-binding", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--formation-receipt", type=Path, required=True)
    parser.add_argument("--frozen-program", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--official-adapter-binding", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--public-freeze", type=Path, required=True)
    parser.add_argument("--plus-channel-id", required=True)
    parser.add_argument("--pro-channel-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    freeze = prepare_development_pre_run_freeze(
        source_view_root=arguments.source_view_root,
        custody_receipt_path=arguments.custody_receipt,
        private_index_binding_path=arguments.private_index_binding,
        preregistration_path=arguments.preregistration,
        acquisition_receipt_path=arguments.acquisition_receipt,
        formation_receipt_path=arguments.formation_receipt,
        frozen_program_path=arguments.frozen_program,
        qualification_path=arguments.qualification,
        official_adapter_binding_path=arguments.official_adapter_binding,
        development_root=arguments.development_root,
        public_freeze_path=arguments.public_freeze,
        plus_channel_id=arguments.plus_channel_id,
        pro_channel_id=arguments.pro_channel_id,
    )
    print(json.dumps({"authorization": freeze["authorization"], "freeze_sha256": freeze["freeze_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
